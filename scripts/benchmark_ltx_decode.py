"""Benchmark the LTX DiffVAE decode (one 121-frame window per call) on this GPU.

Decodes a random latent through ``decode_latent`` exactly as the compose pass does, for
each ``--variants`` entry in turn, and reports seconds per window, peak VRAM and the
pixel difference against the first variant (same seed and tile layout, so any difference
is kernel numerics). Needs an otherwise idle GPU.

A variant is ``+``-joined options applied to the shipped path (jasna's Triton ``na3d``,
stage-5 attention slab and MLP chunk compiled):
    shipped   the shipped path unchanged
    kitchen   comfy-kitchen's fused ``na3d`` in place of the Triton one
    natten    NATTEN's fused ``na3d`` in place of the Triton one
    eager     the stage-5 slab, MLP chunk and context inject uncompiled
    autotune  them compiled with ``max-autotune-no-cudagraphs``

``--budget-gib`` sets the tile budget: the default is this GPU's segment budget
(``restore.SEGMENT_DECODE_BUDGETS``); 24 is roughly a whole-video run on a 32 GB card.
``--split`` adds a synchronized per-stage time split (slower, for the first variant).
The default weights are the trial's random ones; ``--model-dir`` loads the real files,
which the pixel difference needs to be meaningful.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_ltx_decode.py \\
        [--canvas 512|768] [--budget-gib N] [--variants shipped,natten,eager] \\
        [--repeats 3] [--split] [--model-dir model_weights/ltx-restore] [--csv out.csv]
"""

import argparse
import collections
import contextlib
import csv
import datetime
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jasna._suppress_noise import install as install_noise_filters
from jasna.engine_paths import configure_torch_compile_cache
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.plan import LATENT_FRAMES
from jasna.ltx.restore import segment_decode_budget
from jasna.models.ltx_vae import decode_latent, load_video_decoder
from jasna.models.ltx_vae import diffusion_video_decoder as D
from jasna.models.ltx_vae import transformer as T
from jasna.tensor_utils import compile_static, half_dtype

REPO_ROOT = Path(__file__).resolve().parent.parent
LATENT_CHANNELS = 128
LATENT_SPATIAL_SCALE = 32
OPTIONS = ("shipped", "kitchen", "natten", "eager", "autotune")


def full_width_queries(q: torch.Tensor, k: torch.Tensor, query_w_start: int) -> torch.Tensor:
    """``q`` zero-padded to the K/V width (each query's output depends only on itself)."""
    if q.shape[3] == k.shape[3]:
        return q
    full = q.new_zeros(k.shape)
    full[:, :, :, query_w_start : query_w_start + q.shape[3]] = q
    return full


def kitchen_na3d(q, k, v, kernel_size, query_w_start):
    """comfy-kitchen's fused ``na3d`` with the decoder's core-column query interface."""
    import comfy_kitchen

    full = comfy_kitchen.na3d(full_width_queries(q, k, query_w_start), k, v, list(kernel_size), scale=1.0)
    return full[:, :, :, query_w_start : query_w_start + q.shape[3]]


def natten_na3d(q, k, v, kernel_size, query_w_start):
    import natten

    full = natten.na3d(full_width_queries(q, k, query_w_start), k, v, kernel_size=tuple(kernel_size), scale=1.0)
    return full[:, :, :, query_w_start : query_w_start + q.shape[3]]


@contextlib.contextmanager
def patched(target, name: str, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def variant_patches(variant: str) -> contextlib.ExitStack:
    options = variant.split("+")
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise ValueError(f"unknown variant options {sorted(unknown)}; choose from {OPTIONS}")
    stack = contextlib.ExitStack()
    if "kitchen" in options:
        stack.enter_context(patched(T, "_fused_na3d", kitchen_na3d))
    if "natten" in options:
        stack.enter_context(patched(T, "_fused_na3d", natten_na3d))
    if "eager" in options:
        stack.enter_context(patched(T, "_slab_attention", T._attn_on_w_slab))
        stack.enter_context(patched(T, "_mlp_chunk", T.modulated_swiglu))
        stack.enter_context(patched(T, "_add_context", T.add_folded_context))
    if "autotune" in options:
        mode, device = "max-autotune-no-cudagraphs", torch.device("cuda")
        for name, fn in (("_slab_attention", T._attn_on_w_slab), ("_mlp_chunk", T.modulated_swiglu),
                         ("_add_context", T.add_folded_context)):
            stack.enter_context(patched(T, name, compile_static(fn, mode=mode, device=device, dtype=half_dtype(device))))
    return stack


class StageTimer:
    """Synchronized wall time per decoder stage (adds a sync per call)."""

    STAGES = (
        (D.DiffusionVideoDecoder, "forward_stages_1_to_3", "stages 1-3"),
        (D.DiffusionVideoDecoder, "forward_stage_4", "stage 4"),
        (T, "inject_deferred_context", "s5 context upsample+proj"),
        (T, "w_chunked_attention_residual", "s5 attention (incl. NA)"),
        (T, "neighborhood_attention", "  NA kernel (s4 + s5)"),
        (T, "residual_modulating_mlp", "s5 MLP"),
    )

    def __init__(self) -> None:
        self.seconds: dict[str, float] = collections.defaultdict(float)

    def _wrap(self, label: str, fn):
        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            self.seconds[label] += time.perf_counter() - start
            return out

        return timed

    def install(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        for target, name, label in self.STAGES:
            stack.enter_context(patched(target, name, self._wrap(label, getattr(target, name))))
        return stack


def decode(decoder, latent: torch.Tensor, budget: int, seed: int) -> tuple[torch.Tensor, float]:
    generator = torch.Generator(device=latent.device).manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        video = decode_latent(decoder, latent, free_bytes=budget, generator=generator)
    torch.cuda.synchronize()
    return video, time.perf_counter() - start


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR of two ``[-1, 1]`` decodes, clamped as the compose pass clamps them."""
    mse = F.mse_loss(a.float().clamp(-1, 1), b.float().clamp(-1, 1)).item()
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--variants", default="shipped")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    install_noise_filters()
    configure_torch_compile_cache()
    device = torch.device("cuda")
    files = (
        LtxModelFiles.from_dir(args.model_dir, "distilled", fast=False)
        if args.model_dir is not None
        else LtxModelFiles.placeholder("distilled", fast=False)
    )
    decoder = load_video_decoder(files.vae, files.tuned_decoder, device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    budget = int(args.budget_gib * 2**30) if args.budget_gib else segment_decode_budget(total_bytes)(args.canvas)
    side = args.canvas // LATENT_SPATIAL_SCALE
    latent = torch.randn(
        1, LATENT_CHANNELS, LATENT_FRAMES, side, side, device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(0),
    )
    scale = decoder.video_downscale_factors
    tiling = decoder.recommended_tiling_config(
        height=args.canvas, width=args.canvas, num_frames=(LATENT_FRAMES - 1) * scale.time + 1, free_bytes=budget
    )
    print(
        f"{torch.cuda.get_device_name(device)} | canvas {args.canvas} | budget {budget / 2**30:.1f} GiB | "
        f"tile {tiling.frames.tile_size}f x {tiling.height.tile_size} x {tiling.width.tile_size} px | "
        f"{'real' if args.model_dir else 'trial'} weights"
    )

    reference: torch.Tensor | None = None
    for index, variant in enumerate(args.variants.split(",")):
        with variant_patches(variant):
            _video, first_s = decode(decoder, latent, budget, seed=1)
            torch.cuda.reset_peak_memory_stats(device)
            runs = []
            for _ in range(args.repeats):
                video, seconds = decode(decoder, latent, budget, seed=1)
                runs.append(seconds)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            if args.split and index == 0:
                timer = StageTimer()
                with timer.install():
                    _video, split_total = decode(decoder, latent, budget, seed=1)
        if reference is None:
            reference = video
        row = {
            "date": datetime.date.today().isoformat(),
            "commit": git_commit(),
            "label": args.label,
            "gpu": torch.cuda.get_device_name(device),
            "canvas": args.canvas,
            "budget_gib": round(budget / 2**30, 1),
            "tile": f"{tiling.frames.tile_size}x{tiling.height.tile_size}x{tiling.width.tile_size}",
            "weights": "real" if args.model_dir else "trial",
            "variant": variant,
            "first_s": round(first_s, 2),
            "s_per_window": round(statistics.median(runs), 3),
            "peak_vram_gib": round(peak, 2),
            "psnr_vs_first_db": round(psnr(video, reference), 2),
            "max_abs_vs_first": round((video.float() - reference.float()).abs().max().item(), 4),
            "runs_s": " ".join(f"{t:.3f}" for t in runs),
        }
        print(
            f"  {variant:<22} {row['s_per_window']:7.3f} s/window  first {row['first_s']:6.1f} s  "
            f"peak {row['peak_vram_gib']:5.1f} GiB  PSNR vs first {row['psnr_vs_first_db']} dB  "
            f"max|d| {row['max_abs_vs_first']}"
        )
        if args.split and index == 0:
            print(f"    split (synchronized, total {split_total:.3f} s):")
            for label, seconds in timer.seconds.items():
                print(f"      {label:<28} {seconds:7.3f} s  {100 * seconds / split_total:5.1f} %")
        if args.csv is not None:
            new = not args.csv.exists()
            with args.csv.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    main()

"""Sweep the launch configs of the LTX decoder's fused GEMM kernels (``diffusion_triton``) on this NVIDIA GPU.

Records the kernel calls of one real 121-frame decode at the canvas (shapes and counts), then times every config
of each kernel at those shapes, with fp32 accumulation (bf16 operands, the shipped kernels) and with fp16
accumulation, and prints both ``NvidiaLaunch`` entries for ``diffusion_triton.NVIDIA_LAUNCH`` with the summed
kernel time per window of each. ``--verify`` then decodes the ``--latent`` windows with this GPU's current entry and
each swept one, alternating, and reports seconds per window and the distance of the pixels from an fp32 decoder
drawing the same noise (PSNR on [-1, 1], mean absolute difference of the uint8 frames). Needs an otherwise idle GPU.

Usage:
    python scripts/benchmark_ltx_decoder_kernels.py [--canvas 512] [--model-dir model_weights/ltx-restore]
        [--decoder full] [--latent window.pt ...] [--modes bf16,fp16] [--verify 7] [--json out.json]
"""

import argparse
import collections
import contextlib
import copy
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

from jasna._suppress_noise import install as install_noise_filters
from jasna.engine_paths import configure_torch_compile_cache
from jasna.ltx.model_files import VAE_DECODERS, VAE_ENCODERS, LtxModelFiles
from jasna.ltx.plan import LATENT_FRAMES
from jasna.models.ltx_vae import decode_latent, diffusion_triton, load_video_decoder
from jasna.models.ltx_vae import diffusion_video_decoder as D
from jasna.models.ltx_vae import transformer as T

LATENT_CHANNELS = 128
LATENT_SPATIAL_SCALE = 32
GEMM_CONFIGS = [config for config in itertools.product((64, 128), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4))
                if config[0] * config[1] <= 128 * 128 or config[2] == 32]
"""(rows, columns, K step, warps, stages) tried for the GEMM kernels."""
QKV_CONFIGS = list(itertools.product((32, 64, 128), (4, 8)))
"""(rows, warps) tried for the slab Q/K/V."""


class Call:
    """One recorded kernel call: enough to rebuild its operands (random values; the time does not depend on them)."""

    def __init__(self, kernel: str, key: tuple, build):
        self.kernel, self.key, self.build = kernel, key, build


def record_calls(decoder, latent: torch.Tensor, budget: int) -> collections.Counter:
    """Calls of the fused GEMM kernels in one decode, counted by shape."""
    shipped = next(m.kernels for m in decoder.modules() if isinstance(m, T.NeighborhoodAttention3D))
    calls: collections.Counter = collections.Counter()

    def swiglu(x, norm, scale, shift, w_gate, w_up, w_down):
        calls[("swiglu", x.shape[0], x.shape[1])] += 1
        return shipped.residual_swiglu(x, norm, scale, shift, w_gate, w_up, w_down)

    def proj(x, out, weight, bias, core_start):
        calls[("proj_residual", tuple(x.shape), out.shape[3], core_start)] += 1
        return shipped.proj_residual(x, out, weight, bias, core_start)

    def context(x, w_start, w_len, feat, weight, bias, stride, drop):
        full_w = feat.stride(2) // feat.stride(3)
        calls[("context_add", tuple(x.shape), w_start, w_len, tuple(feat.shape), full_w, tuple(stride), drop)] += 1
        return shipped.context_add(x, w_start, w_len, feat, weight, bias, stride, drop)

    def qkv(x, left_halo, columns, modulation, weights, inv_freqs, w_pos, rope_pairs, core_len, halo, q_scale):
        calls[("qkv", tuple(x.shape), columns.numel(), bool(modulation), core_len, halo)] += 1
        return shipped.slab_qkv(x, left_halo, columns, modulation, weights, inv_freqs, w_pos, rope_pairs, core_len,
                                halo, q_scale)

    T.use_kernels(decoder, shipped._replace(residual_swiglu=swiglu, slab_qkv=qkv, proj_residual=proj,
                                            context_add=context))
    try:
        with patched_slabs():
            decode_latent(decoder, latent, free_bytes=budget, generator=torch.Generator(latent.device).manual_seed(0))
    finally:
        T.use_kernels(decoder, shipped)
    return calls


@contextlib.contextmanager
def patched_slabs():
    """The stage-5 slab uncompiled, so the recording wrappers run in Python."""
    original = T._slab_attention, T._mlp_chunk, T._add_context
    T._slab_attention, T._mlp_chunk, T._add_context = T._attn_on_w_slab, T.modulated_swiglu, T.add_folded_context
    try:
        yield
    finally:
        T._slab_attention, T._mlp_chunk, T._add_context = original


def operands(key: tuple, device: torch.device, shared: dict):
    """Random bf16 operands of a recorded call, as the launcher's arguments. Tensors of one name and shape are made
    once in ``shared`` and reused by every call of the kind (the calls of a block write into the same activation)."""

    def randn(name: str, *shape, scale=1.0):
        if (name, shape) not in shared:
            generator = torch.Generator(device=device).manual_seed(len(shared))
            shared[name, shape] = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator).mul_(scale)
        return shared[name, shape]

    kind = key[0]
    if kind == "swiglu":
        _, rows, dim = key
        hidden = T._swiglu_hidden_dim(dim)
        return (randn("x", rows, dim), randn("norm", dim), randn("scale", dim, scale=0.1), randn("shift", dim, scale=0.1),
                randn("gate", hidden, dim, scale=dim**-0.5), randn("up", hidden, dim, scale=dim**-0.5),
                randn("down", dim, hidden, scale=hidden**-0.5))
    if kind == "proj_residual":
        _, x_shape, core_len, core_start = key
        dim = x_shape[4]
        return (randn("x", *x_shape), randn("out", *x_shape[:3], core_len, dim),
                randn("weight", dim, dim, scale=dim**-0.5), randn("bias", dim), core_start)
    if kind == "context_add":
        _, x_shape, w_start, w_len, feat_shape, full_w, stride, drop = key
        dim, in_dim = x_shape[4], feat_shape[4]
        feat = randn("feat", *feat_shape[:3], full_w, in_dim)[:, :, :, : feat_shape[3]]
        positions = stride[0] * stride[1] * stride[2]
        return (randn("x", *x_shape), w_start, w_len, feat, randn("weight", positions * dim, in_dim, scale=in_dim**-0.5),
                randn("bias", positions * dim), stride, drop)
    _, x_shape, columns, modulate, core_len, halo = key
    dim = x_shape[4]
    head_dim = T.DIFFUSION_HEAD_DIM
    weights = [randn(f"w{index}", dim, dim, scale=dim**-0.5) if index % 2 == 0 else randn(f"b{index}", dim)
               for index in range(6)] + [randn("q_norm", head_dim), randn("k_norm", head_dim)]
    modulation = [randn("norm", dim), randn("scale", dim, scale=0.1), randn("shift", dim, scale=0.1)] if modulate else []
    return (randn("x", *x_shape), randn("halo", *x_shape[:3], halo, dim), torch.arange(columns, device=device) % x_shape[3],
            modulation, weights, torch.rand(head_dim // 2, device=device),
            torch.arange(halo + columns, device=device, dtype=torch.float32), list(T.DIFFUSION_ROPE_PAIRS[:2]), core_len,
            halo, head_dim**-0.5)


LAUNCHERS = {
    "swiglu": diffusion_triton.residual_swiglu_,
    "proj_residual": diffusion_triton.proj_residual_,
    "context_add": diffusion_triton.context_add_,
    "qkv": diffusion_triton.slab_qkv,
}


def milliseconds(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@contextlib.contextmanager
def launch_entry(entry: diffusion_triton.NvidiaLaunch):
    original = diffusion_triton.nvidia_launch
    diffusion_triton.nvidia_launch = lambda device: entry
    try:
        yield
    finally:
        diffusion_triton.nvidia_launch = original


def window_ms(calls, kind: str, dim: int | None, entry, device, repeats: int, cache: dict) -> float:
    """Summed time per window of the recorded ``kind`` calls (of width ``dim``) under ``entry``; ``cache`` holds the
    kind's operands."""
    total = 0.0
    with launch_entry(entry):
        for key, count in calls.items():
            if key[0] != kind or (dim is not None and width(key) != dim):
                continue
            if key not in cache:
                cache[key] = operands(key, device, cache.setdefault("shared", {}))
            args = cache[key]
            total += count * milliseconds(lambda: LAUNCHERS[kind](*args), repeats)
    return total


def width(key: tuple) -> int:
    return key[2] if key[0] == "swiglu" else key[1][4]


def swiglu_entry(entry, dim: int, gate_up=None, down=None):
    current_gate_up, current_down = entry.swiglu[dim]
    return entry._replace(swiglu={**entry.swiglu, dim: (gate_up or current_gate_up, down or current_down)})


def sweep(calls, start: diffusion_triton.NvidiaLaunch, device, repeats: int, kinds: set[str], log) -> tuple:
    """The fastest config per kernel of ``kinds``, one kernel at a time from ``start``; returns the entry and, per
    kernel, its config and the window time of all calls of its kind (the MLP's gate/up and down share their calls).
    Only one kind's operands are held at a time."""
    cache: dict = {}
    cached_kind = None
    entry = start
    report = {}
    dims = sorted({width(key) for key in calls if key[0] == "swiglu"})
    plans = [(f"gate/up {dim}", "swiglu", dim, GEMM_CONFIGS, lambda e, c, d=dim: swiglu_entry(e, d, gate_up=c))
             for dim in dims]
    plans += [(f"down {dim}", "swiglu", dim, GEMM_CONFIGS, lambda e, c, d=dim: swiglu_entry(e, d, down=c))
              for dim in dims]
    plans += [("proj_residual", "proj_residual", None, GEMM_CONFIGS, lambda e, c: e._replace(proj_residual=c)),
              ("context_add", "context_add", None, GEMM_CONFIGS, lambda e, c: e._replace(context_add=c))]
    plans += [(f"qkv {dim}", "qkv", dim, QKV_CONFIGS, lambda e, c, d=dim: e._replace(qkv={**e.qkv, d: c}))
              for dim in sorted({width(key) for key in calls if key[0] == "qkv"})]
    for label, kind, dim, configs, apply in plans:
        if kind not in kinds:
            continue
        if kind != cached_kind:
            cache.clear()
            torch.cuda.empty_cache()
            cached_kind = kind
        best = None
        for config in configs:
            try:
                ms = window_ms(calls, kind, dim, apply(entry, config), device, repeats, cache)
            except Exception as error:  # noqa: BLE001 - a config that does not fit this GPU is skipped
                log(f"  {label} {config}: skipped ({type(error).__name__})")
                torch.cuda.empty_cache()
                continue
            if best is None or ms < best[0]:
                best = (ms, config)
        if best is None:
            raise RuntimeError(f"no {label} launch config could be measured on this GPU (see the skipped lines)")
        entry = apply(entry, best[1])
        report[label] = (best[1], round(best[0], 3))
        log(f"  {label:<14} {best[1]} {best[0]:8.3f} ms/window")
    return entry, report


@contextlib.contextmanager
def bf16_noise():
    """Decoder noise drawn in bf16 whatever the decoder's dtype, so an fp32 decoder sees the bf16 decoder's noise."""
    randn = torch.randn

    def bf16_randn(*args, dtype=None, **kwargs):
        return randn(*args, dtype=torch.bfloat16, **kwargs).to(dtype or torch.float32)

    D.torch.randn = bf16_randn
    try:
        yield
    finally:
        D.torch.randn = randn


def decode(decoder, latent: torch.Tensor, budget: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=latent.device).manual_seed(seed)
    with torch.inference_mode():
        return decode_latent(decoder, latent, free_bytes=budget, generator=generator)[0].float().clamp(-1, 1)


def distance(video: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """PSNR on [-1, 1] and the mean absolute difference of the uint8 frames the compose pass makes."""
    mse = torch.nn.functional.mse_loss(video, reference).item()
    as_uint8 = [((v + 1) / 2 * 255).round() for v in (video, reference)]
    return (math.inf if mse == 0 else 10 * math.log10(4 / mse)), (as_uint8[0] - as_uint8[1]).abs().mean().item()


def verify(decoder, latents: list[torch.Tensor], budget: int, entries: dict, rounds: int, log) -> dict:
    """Distance of each entry's decode from an fp32 decoder's, then alternating timed decodes. The slab runs
    uncompiled for every entry, since an entry decides the dtype of the slab's Q/K/V."""
    kernels = next(m.kernels for m in decoder.modules() if isinstance(m, T.NeighborhoodAttention3D))
    T.use_kernels(decoder, T.DecoderKernels())
    fp32 = copy.deepcopy(decoder).float()
    T.use_kernels(decoder, kernels)
    fp32.recommended_tiling_config = decoder.recommended_tiling_config
    report = {name: {"psnr": [], "mad": [], "seconds": []} for name in entries}
    with patched_slabs(), bf16_noise():
        for index, latent in enumerate(latents):
            reference = decode(fp32, latent, budget, index)
            for name, entry in entries.items():
                with launch_entry(entry):
                    psnr, mad = distance(decode(decoder, latent, budget, index), reference)
                report[name]["psnr"].append(psnr)
                report[name]["mad"].append(mad)
            del reference
        del fp32
        torch.cuda.empty_cache()
        for name, entry in entries.items():
            with launch_entry(entry):
                decode(decoder, latents[0], budget, 0)
        for round_index in range(rounds):
            order = list(entries.items()) if round_index % 2 == 0 else list(entries.items())[::-1]
            for name, entry in order:
                with launch_entry(entry):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    decode(decoder, latents[round_index % len(latents)], budget, round_index)
                    torch.cuda.synchronize()
                    report[name]["seconds"].append(time.perf_counter() - start)
    for name, values in report.items():
        log(f"  {name:<10} {statistics.median(values['seconds']):.4f} s/window (median of {rounds}), vs fp32: "
            f"PSNR {statistics.mean(values['psnr']):.2f} dB, mean |d| {statistics.mean(values['mad']):.4f} of 255")
    return report


def entry_source(entry: diffusion_triton.NvidiaLaunch) -> str:
    swiglu = ", ".join(f"{dim}: ({gate_up}, {down})" for dim, (gate_up, down) in sorted(entry.swiglu.items()))
    return (f"NvidiaLaunch(swiglu={{{swiglu}}}, proj_residual={entry.proj_residual}, "
            f"context_add={entry.context_add}, qkv={entry.qkv}, fp16_accumulate={entry.fp16_accumulate})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--budget-gib", type=float, default=24.0)
    parser.add_argument("--model-dir", type=Path, help="holds vae.safetensors and the decoder file")
    parser.add_argument("--decoder", choices=tuple(VAE_DECODERS), default="full")
    parser.add_argument("--latent", type=Path, nargs="*", default=[],
                        help="saved (1, 128, 16, S/32, S/32) latent windows; the first is the one swept")
    parser.add_argument("--modes", default="bf16,fp16")
    parser.add_argument("--kernels", default=",".join(LAUNCHERS), help=f"kernels to sweep, of {', '.join(LAUNCHERS)}")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--verify", type=int, default=0, help="alternating timed decodes per entry (0: no verify)")
    parser.add_argument("--memory-cap-gib", type=float,
                        help="limit this process's GPU memory, to check that a run fits a smaller card")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    install_noise_filters()
    configure_torch_compile_cache()
    device = torch.device("cuda")
    if args.memory_cap_gib:
        torch.cuda.set_per_process_memory_fraction(
            args.memory_cap_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory, torch.cuda.current_device())
    if args.model_dir is not None:
        decoder = load_video_decoder(args.model_dir / VAE_ENCODERS["full"], args.model_dir / VAE_DECODERS[args.decoder],
                                     device)
    else:
        files = LtxModelFiles.placeholder("distilled2", fast=False, decoder=args.decoder)
        decoder = load_video_decoder(files.vae, files.tuned_decoder, device)
    side = args.canvas // LATENT_SPATIAL_SCALE
    latents = [torch.load(path, weights_only=True).to(device) for path in args.latent] or [
        torch.randn(1, LATENT_CHANNELS, LATENT_FRAMES, side, side, device=device, dtype=torch.bfloat16)]
    budget = int(args.budget_gib * 2**30)
    calls = record_calls(decoder, latents[0], budget)
    print(f"{torch.cuda.get_device_name(device)} {torch.cuda.get_device_capability(device)} | canvas {args.canvas} | "
          f"{sum(calls.values())} fused GEMM calls, {len(calls)} shapes", flush=True)
    shipped = diffusion_triton.nvidia_launch(device)
    results = {}
    for mode in args.modes.split(","):
        print(f"{mode} accumulation:", flush=True)
        entry, report = sweep(calls, shipped._replace(fp16_accumulate=mode == "fp16"), device, args.repeats,
                              set(args.kernels.split(",")), lambda line: print(line, flush=True))
        print(f"  entry: {entry_source(entry)}", flush=True)
        results[mode] = {"entry": entry, "kernels": report}
    entries = {"current": shipped, **{f"swept {mode}": result["entry"] for mode, result in results.items()}}
    summary = {"gpu": torch.cuda.get_device_name(device), "canvas": args.canvas, "budget_gib": args.budget_gib,
               "results": {mode: {**result, "entry": result["entry"]._asdict()} for mode, result in results.items()}}

    def save() -> None:
        if args.json is not None:
            args.json.write_text(json.dumps(summary, indent=1))

    summary["sweep_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"sweep peak {summary['sweep_peak_gib']:.2f} GiB", flush=True)
    save()
    if args.verify:
        print("verify:", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        summary["verify"] = verify(decoder, latents, budget, entries, args.verify, lambda line: print(line, flush=True))
        summary["verify_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        print(f"verify peak {summary['verify_peak_gib']:.2f} GiB", flush=True)
        save()


if __name__ == "__main__":
    main()

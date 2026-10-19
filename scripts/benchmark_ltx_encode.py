"""Benchmark the LTX VAE encode (one 121-frame window per call) on this GPU.

Encodes the first 121 frames of a clip, center-cropped to the canvas, through
``load_video_encoder`` exactly as the encode pass does, for each ``--variants`` entry in
turn, and reports seconds per window, peak VRAM and the latent difference against the
first variant. Needs an otherwise idle GPU.

A variant is ``+``-joined options:
    shipped   the shipped encoder (``channels_last_3d`` weights, compiled layers)
    eager     contiguous weights, layers uncompiled (the base the options below change)
    cl3d      weights and input in ``channels_last_3d``
    compile   the layers compiled with static shapes
    benchmark cuDNN autotuning (``cudnn.benchmark``) around the call
    fp32      weights and input in float32 (a numerics reference: list it first)

``--profile`` prints a torch.profiler kernel table for the first variant.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_ltx_encode.py \\
        [--canvas 512|768] [--variants shipped,eager] [--repeats 5] [--profile] \\
        [--vae model_weights/ltx-restore/vae.safetensors] [--clip assets/test_clip1_1080p.mp4] [--csv out.csv]
"""

import argparse
import contextlib
import csv
import datetime
import functools
import math
import statistics
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from jasna._suppress_noise import install as install_noise_filters
from jasna.engine_paths import configure_torch_compile_cache
from jasna.ltx.plan import WINDOW_FRAMES
from jasna.models.ltx_vae import load_video_encoder
from jasna.models.ltx_vae.video_encoder import ENCODER_COMPILE_MODE, VideoEncoder, encode_cropped
from jasna.tensor_utils import compile_static

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIONS = ("shipped", "eager", "cl3d", "benchmark", "compile", "fp32")


def clip_pixels(clip: Path, canvas: int, device: torch.device) -> torch.Tensor:
    """``(1, 3, 121, canvas, canvas)`` bf16 pixels in [-1, 1], center-cropped like a window canvas."""
    frames = []
    with av.open(str(clip)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == WINDOW_FRAMES:
                break
    video = torch.from_numpy(np.stack(frames)).to(device)
    height, width = video.shape[1:3]
    top, left = (height - canvas) // 2, (width - canvas) // 2
    video = video[:, top : top + canvas, left : left + canvas].permute(3, 0, 1, 2).unsqueeze(0)
    return video.float().div(127.5).sub(1.0).to(torch.bfloat16).contiguous()


def variant_encoder(variant: str, encoder: VideoEncoder) -> tuple[callable, contextlib.ExitStack]:
    options = variant.split("+")
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise ValueError(f"unknown variant options {sorted(unknown)}; choose from {OPTIONS}")
    stack = contextlib.ExitStack()
    shipped = "shipped" in options
    memory_format = torch.channels_last_3d if shipped or "cl3d" in options else torch.contiguous_format
    dtype = torch.float32 if "fp32" in options else torch.bfloat16
    encoder.to(dtype=dtype, memory_format=memory_format)
    fn = encoder if shipped else functools.partial(encode_cropped, encoder)
    if "compile" in options:
        fn = compile_static(fn, mode=ENCODER_COMPILE_MODE, device=torch.device("cuda"))
    if "benchmark" in options:
        stack.enter_context(torch.backends.cudnn.flags(enabled=True, benchmark=True))

    def run(pixels: torch.Tensor) -> torch.Tensor:
        return fn(pixels.to(dtype).contiguous(memory_format=torch.channels_last_3d) if "cl3d" in options else pixels.to(dtype))

    return run, stack


def encode(fn, pixels: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        latent = fn(pixels)
    torch.cuda.synchronize()
    return latent, time.perf_counter() - start


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR of two latents over the reference latent's peak-to-peak range."""
    mse = F.mse_loss(a.float(), b.float()).item()
    peak = (b.float().max() - b.float().min()).item()
    return math.inf if mse == 0 else 10 * math.log10(peak**2 / mse)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def print_profile(fn, pixels: torch.Tensor) -> None:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof, torch.inference_mode():
        fn(pixels)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--variants", default="shipped")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--vae", type=Path, default=REPO_ROOT / "model_weights/ltx-restore/vae.safetensors")
    parser.add_argument("--clip", type=Path, default=REPO_ROOT / "assets/test_clip1_1080p.mp4")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    install_noise_filters()
    configure_torch_compile_cache()
    device = torch.device("cuda")
    encoder = load_video_encoder(args.vae, device)
    pixels = clip_pixels(args.clip, args.canvas, device)
    print(f"{torch.cuda.get_device_name(device)} | canvas {args.canvas} | {args.clip.name}")

    reference: torch.Tensor | None = None
    for index, variant in enumerate(args.variants.split(",")):
        fn, stack = variant_encoder(variant, encoder)
        with stack:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _latent, first_s = encode(fn, pixels)
            first_peak = torch.cuda.max_memory_allocated(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
            runs = []
            for _ in range(args.repeats):
                latent, seconds = encode(fn, pixels)
                runs.append(seconds)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            if args.profile and index == 0:
                print_profile(fn, pixels)
        if reference is None:
            reference = latent
        row = {
            "date": datetime.date.today().isoformat(),
            "commit": git_commit(),
            "label": args.label,
            "gpu": torch.cuda.get_device_name(device),
            "canvas": args.canvas,
            "variant": variant,
            "first_s": round(first_s, 2),
            "s_per_window": round(statistics.median(runs), 4),
            "peak_vram_gib": round(peak, 2),
            "first_peak_vram_gib": round(first_peak, 2),
            "psnr_vs_first_db": round(psnr(latent, reference), 2),
            "max_abs_vs_first": round((latent.float() - reference.float()).abs().max().item(), 4),
            "mean_abs_vs_first": round((latent.float() - reference.float()).abs().mean().item(), 5),
            "runs_s": " ".join(f"{t:.4f}" for t in runs),
        }
        print(
            f"  {variant:<26} {row['s_per_window']:7.4f} s/window  first {row['first_s']:6.2f} s  "
            f"peak {row['peak_vram_gib']:5.2f} GiB (first {row['first_peak_vram_gib']:5.2f})  "
            f"PSNR vs first {row['psnr_vs_first_db']} dB  max|d| {row['max_abs_vs_first']}  "
            f"mean|d| {row['mean_abs_vs_first']}"
        )
        if args.csv is not None:
            new = not args.csv.exists()
            with args.csv.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    main()

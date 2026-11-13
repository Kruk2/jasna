"""Benchmark the LTX VAE encode (one 121-frame window per call) on this GPU.

Encodes the first 121 frames of a clip, center-cropped to the canvas, through
``load_video_encoder`` exactly as the encode pass does, for each ``--variants`` entry in
turn, and reports seconds per window, peak VRAM (with and without the weights) and the latent
difference against the first variant. Needs an otherwise idle GPU.

A variant is ``+``-joined options, each on a freshly loaded encoder:
    shipped   the encoder as ``load_video_encoder`` returns it
    eager     bf16 contiguous weights, layers uncompiled, torch convs (the base the options below change)
    cl3d      weights and input in ``channels_last_3d``
    compile   the layers compiled with static shapes
    benchmark cuDNN autotuning (``cudnn.benchmark``) around the call
    fp32      weights and input in float32 (a numerics reference: list it first)
    fp16tail  fp16 weights with an fp32 tail (``to_fp16_with_float32_tail``), fp16 input
    fp16headN fp16 ``conv_in`` and first N down blocks (``to_fp16_head``), fp16 input
    kitchenN  fp16headN with those convs on comfy-kitchen's fp16-accumulate ``fp16_conv3d``

``eager+cl3d+compile`` is the bf16 cuDNN encoder; ``shipped`` is that or the GPU's fp16 head (kitchen and
Triton convs) per ``video_encoder.ENCODER_HEADS``. ``--peaks`` first prints the GPU's bf16 and fp16 GEMM
rates (fp16 with fp32 and with fp16 accumulation), ``--profile`` a torch.profiler kernel table for
the first variant. To measure a GPU for ``ENCODER_HEADS`` (then ``benchmark_ltx_encoder_convs.py``):
    --peaks --variants fp32,eager+cl3d+compile,kitchen1+cl3d+compile,kitchen3+cl3d+compile,kitchen6+cl3d+compile

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_ltx_encode.py \\
        [--canvas 512|768] [--variants shipped,eager] [--repeats 5] [--peaks] [--profile] \\
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
from jasna.models.ltx_vae.video_encoder import ENCODER_COMPILE_MODE, CausalConv3d, encode_cropped
from jasna.tensor_utils import compile_static, half_dtype

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIONS = ("shipped", "eager", "cl3d", "benchmark", "compile", "fp32", "fp16tail")
HEAD_OPTIONS = ("fp16head", "kitchen")


def head_blocks(options: list[str], prefix: str) -> int | None:
    """N of a ``<prefix>N`` option, else None."""
    counts = (option.removeprefix(prefix) for option in options if option.startswith(prefix))
    return next((int(count) for count in counts if count.isdigit()), None)


def clip_pixels(clip: Path, canvas: int, device: torch.device) -> torch.Tensor:
    """``(1, 3, 121, canvas, canvas)`` uint8 pixels, center-cropped like a window canvas."""
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
    return video.contiguous()


def variant_encoder(variant: str, vae: Path, device: torch.device) -> tuple[callable, contextlib.ExitStack]:
    options = variant.split("+")
    unknown = {o for o in options if o not in OPTIONS and all(head_blocks([o], p) is None for p in HEAD_OPTIONS)}
    if unknown:
        raise ValueError(f"unknown variant options {sorted(unknown)}; choose from {OPTIONS} or {HEAD_OPTIONS} + N")
    stack = contextlib.ExitStack()
    encoder = load_video_encoder(vae, device)
    fp16_head = head_blocks(options, "fp16head")
    kitchen_head = head_blocks(options, "kitchen")
    if "shipped" in options:
        fn = encoder
    else:
        memory_format = torch.channels_last_3d if "cl3d" in options else torch.contiguous_format
        if "fp16tail" in options:
            encoder.to_fp16_with_float32_tail()
        else:
            encoder.to(torch.float32 if "fp32" in options else torch.bfloat16)
        if fp16_head is not None or kitchen_head is not None:
            encoder.to_fp16_head(fp16_head if fp16_head is not None else kitchen_head)
        for conv in encoder.modules():
            if isinstance(conv, CausalConv3d):
                conv.kitchen = False
                conv.triton_launch = None
        if kitchen_head is not None:
            encoder.run_fp16_convs_on_kitchen()
        encoder.to(memory_format=memory_format)
        fn = functools.partial(encode_cropped, encoder)
        if "compile" in options:
            fn = compile_static(fn, mode=ENCODER_COMPILE_MODE, device=device, dtype=encoder.conv_in.conv.weight.dtype)
    if "benchmark" in options:
        stack.enter_context(torch.backends.cudnn.flags(enabled=True, benchmark=True))
    # shipped takes the pixels as the encode pass builds them (_vae_pixels), the others in their weights' dtype
    input_dtype = half_dtype(device) if "shipped" in options else encoder.conv_in.conv.weight.dtype
    input_format = torch.channels_last_3d if "cl3d" in options else torch.contiguous_format

    def run(pixels: torch.Tensor) -> torch.Tensor:
        return fn(pixels.float().div_(127.5).sub_(1.0).to(input_dtype).contiguous(memory_format=input_format))

    return run, stack


def encode(fn, pixels: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        latent = fn(pixels)
    torch.cuda.synchronize()
    return latent, time.perf_counter() - start


def gemm_tflops(device: torch.device, size: int = 8192) -> dict[str, float]:
    """TFLOPS of a ``size``-cubed GEMM in bf16 and fp16, fp16 with fp32 and with fp16 accumulation."""
    a, b = (torch.randn(size, size, device=device, dtype=torch.float16) for _ in range(2))
    a_bf16, b_bf16 = a.bfloat16(), b.bfloat16()
    rates = {}
    for name, x, y, fp16_accumulation in (
        ("bf16", a_bf16, b_bf16, False), ("fp16_fp32acc", a, b, False), ("fp16_fp16acc", a, b, True)
    ):
        torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulation
        try:
            for _ in range(3):
                x @ y
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                x @ y
            torch.cuda.synchronize()
            rates[name] = round(2 * size**3 * 10 / (time.perf_counter() - start) / 1e12, 1)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
    return rates


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
    parser.add_argument("--peaks", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--vae", type=Path, default=REPO_ROOT / "model_weights/ltx-restore/vae.safetensors")
    parser.add_argument("--clip", type=Path, default=REPO_ROOT / "assets/test_clip1_1080p.mp4")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    install_noise_filters()
    configure_torch_compile_cache()
    device = torch.device("cuda")
    pixels = clip_pixels(args.clip, args.canvas, device)
    print(f"{torch.cuda.get_device_name(device)} | SM {torch.cuda.get_device_capability(device)} | "
          f"canvas {args.canvas} | {args.clip.name}")
    if args.peaks:
        print(f"  GEMM TFLOPS {gemm_tflops(device)}")

    reference: torch.Tensor | None = None
    for index, variant in enumerate(args.variants.split(",")):
        fn, stack = variant_encoder(variant, args.vae, device)
        with stack:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _latent, first_s = encode(fn, pixels)
            first_peak = torch.cuda.max_memory_allocated(device) / 2**30
            weights = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            runs = []
            for _ in range(args.repeats):
                latent, seconds = encode(fn, pixels)
                runs.append(seconds)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            if args.profile and index == 0:
                print_profile(fn, pixels)
        del fn, stack
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
            "activation_peak_gib": round(peak - weights / 2**30, 2),
            "psnr_vs_first_db": round(psnr(latent, reference), 2),
            "max_abs_vs_first": round((latent.float() - reference.float()).abs().max().item(), 4),
            "mean_abs_vs_first": round((latent.float() - reference.float()).abs().mean().item(), 5),
            "rel_l2_vs_first": float((latent.float() - reference.float()).norm() / reference.float().norm()),
            "runs_s": " ".join(f"{t:.4f}" for t in runs),
        }
        print(
            f"  {variant:<26} {row['s_per_window']:7.4f} s/window  first {row['first_s']:6.2f} s  "
            f"peak {row['peak_vram_gib']:5.2f} GiB (first {row['first_peak_vram_gib']:5.2f}, "
            f"activations {row['activation_peak_gib']:5.2f})  PSNR vs first {row['psnr_vs_first_db']} dB  "
            f"rel L2 {row['rel_l2_vs_first']:.2e}  max|d| {row['max_abs_vs_first']}  mean|d| {row['mean_abs_vs_first']}"
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

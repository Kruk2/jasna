"""Benchmark the LTX denoise pass (the ~80 % of an LTX run) on this GPU.

Runs ``LtxTransformer.denoise_chain`` on one track of overlapping windows with the
trial's random weights: same shapes, kernels and sampler steps as the real model, so
the speed matches, but nothing is restored. Needs an otherwise idle GPU.

The first call compiles (with autotuning on an empty compile cache) and is reported as
``compile_s``; the timed repeats run the full sampler after that.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_ltx_denoise.py \\
        [--model distilled|undistilled] [--fast] [--canvas 512|768] [--windows 3] \\
        [--repeats 3] [--fresh-cache] [--inductor force_fuse_int_mm_with_mul=True] \\
        [--profile] [--trace trace.json] [--csv out.csv] [--label text]

    --profile   instead of timing, breaks one sampler step down by kernel kind
                (GEMM / attention / fused Triton / other) and lists the top kernels
"""

import argparse
import ast
import csv
import datetime
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from jasna._suppress_noise import install as install_noise_filters
from jasna.engine_paths import configure_torch_compile_cache
from jasna.ltx import transformer as T
from jasna.ltx.model_files import LTX_MODELS, LtxModelFiles
from jasna.ltx.plan import LATENT_FRAMES, WINDOW_HOP
from jasna.ltx.sampler import LtxTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
LATENT_CHANNELS = 128
KERNEL_KINDS = (
    ("gemm", re.compile(r"^triton_tem|gemm|cutlass|nvjet|xmma|gemv|cublas", re.I)),
    ("fused triton", re.compile(r"^triton_")),
    ("attention", re.compile(r"attn|fmha|flash|attention|sdpa|QuantInt8Kernel|TransposePadPermute", re.I)),
)


def kernel_kind(name: str) -> str:
    return next((kind for kind, pattern in KERNEL_KINDS if pattern.search(name)), "other")


def set_inductor_flags(flags: list[str]) -> None:
    for flag in flags:
        key, value = flag.split("=", 1)
        target = torch._inductor.config
        *parents, leaf = key.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, leaf, ast.literal_eval(value))


def references(windows: int, canvas: int, device: torch.device) -> list[torch.Tensor]:
    side = canvas // T.LATENT_SPATIAL_SCALE
    generator = torch.Generator(device=device).manual_seed(0)
    return [
        torch.randn(1, LATENT_CHANNELS, LATENT_FRAMES, side, side, device=device, generator=generator, dtype=torch.bfloat16)
        for _ in range(windows)
    ]


def run_chain(transformer: LtxTransformer, refs: list[torch.Tensor], steps: int) -> float:
    conditions = transformer.conditions
    transformer.conditions = conditions[:steps]
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        transformer.denoise_chain(refs, list(range(len(refs))), WINDOW_HOP, lambda _n: None)
        torch.cuda.synchronize()
        return time.perf_counter() - start
    finally:
        transformer.conditions = conditions


def profile_step(transformer: LtxTransformer, refs: list[torch.Tensor], trace: Path | None) -> None:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_chain(transformer, refs, 1)
    if trace is not None:
        prof.export_chrome_trace(str(trace))
    kernels = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.self_device_time_total for e in kernels)
    per_kind: dict[str, float] = {}
    for event in kernels:
        kind = kernel_kind(event.key)
        per_kind[kind] = per_kind.get(kind, 0.0) + event.self_device_time_total
    windows = len(refs)
    print(f"\nGPU time per window-step: {total / 1e3 / windows:.1f} ms")
    for kind, micros in sorted(per_kind.items(), key=lambda item: -item[1]):
        print(f"  {kind:<13} {micros / 1e3 / windows:8.1f} ms  {100 * micros / total:5.1f} %")
    print("\nTop kernels (ms per window-step, calls):")
    for event in sorted(kernels, key=lambda e: -e.self_device_time_total)[:25]:
        name = event.key if len(event.key) <= 90 else event.key[:87] + "..."
        print(f"  {event.self_device_time_total / 1e3 / windows:8.2f}  {event.count:6d}  [{kernel_kind(event.key)}] {name}")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=LTX_MODELS, default="distilled")
    parser.add_argument("--fast", action="store_true", help="NVFP4 transformer (Blackwell)")
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fresh-cache", action="store_true", help="compile into an empty cache (measures autotuning)")
    parser.add_argument("--inductor", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if args.fresh_cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="ltx-bench-inductor-")
    install_noise_filters()
    configure_torch_compile_cache()
    set_inductor_flags(args.inductor)
    device = torch.device("cuda")
    files = LtxModelFiles.placeholder(args.model, fast=args.fast)
    transformer = LtxTransformer(files.transformer, device)
    refs = references(args.windows, args.canvas, device)
    steps = len(transformer.conditions)
    stg = bool(transformer.stg_scale)
    print(
        f"{torch.cuda.get_device_name(device)} | {args.model} {'nvfp4' if args.fast else 'int8'} | canvas {args.canvas} "
        f"| {args.windows} windows x {steps} steps | STG {stg} | {transformer.blocks.resident}/{T.NUM_BLOCKS} blocks resident"
        f" | {transformer.block_backend} blocks"
    )

    compile_s = run_chain(transformer, refs, 1)
    print(f"compile + first step: {compile_s:.1f} s")
    if args.profile:
        profile_step(transformer, refs, args.trace)
        return

    torch.cuda.reset_peak_memory_stats(device)
    times = [run_chain(transformer, refs, steps) for _ in range(args.repeats)]
    median = statistics.median(times)
    per_window = median / args.windows
    row = {
        "date": datetime.date.today().isoformat(),
        "commit": git_commit(),
        "label": args.label,
        "gpu": torch.cuda.get_device_name(device),
        "model": args.model,
        "quant": "nvfp4" if args.fast else "int8",
        "canvas": args.canvas,
        "windows": args.windows,
        "steps": steps,
        "stg": stg,
        "blocks": transformer.block_backend,
        "resident": transformer.blocks.resident,
        "inductor": " ".join(args.inductor),
        "compile_s": round(compile_s, 1),
        "s_per_window": round(per_window, 2),
        "ms_per_window_step": round(1e3 * per_window / steps, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "runs_s": " ".join(f"{t:.2f}" for t in times),
    }
    for key, value in row.items():
        print(f"  {key}: {value}")
    if args.csv is not None:
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()

"""Benchmark the Triton 3D neighborhood attention against comfy-kitchen and NATTEN.

Times the shipped Triton config for this GPU's vendor (``na_triton.config_for``), comfy-kitchen's and NATTEN's
``na3d`` (when installed) on the LTX DiffVAE decoder's NA shapes: stage 5 at 512 and 768 px
as the decoder calls it (W slab with 5 halo columns each side, queries on the core columns
only; 4 slab chunks or 1) and stage 4. Prints effective TFLOPS (useful FLOPs only:
4 x queries x window volume x channels) and the max abs difference vs kitchen.
``--sweep`` also times every candidate tile config. Needs an otherwise idle GPU.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_na3d.py [--sweep] [--shapes s5-512-w42 ...]
"""

import argparse
import itertools
import math

import torch
import triton
from triton.runtime.errors import OutOfResources

from benchmark_ltx_decode import full_width_queries, kitchen_na3d
from jasna.models.ltx_vae.na_triton import config_for, na3d_with_config, tile_config

# name: (K/V shape (B, T, H, W, heads, head_dim), kernel, first query column, query columns)
SHAPES = {
    "s5-512-w42": ((1, 121, 128, 42, 4, 64), (11, 11, 11), 5, 32),
    "s5-512-w138": ((1, 121, 128, 138, 4, 64), (11, 11, 11), 5, 128),
    "s5-768-w58": ((1, 121, 192, 58, 4, 64), (11, 11, 11), 5, 48),
    "s5-768-w202": ((1, 121, 192, 202, 4, 64), (11, 11, 11), 5, 192),
    "s4-512": ((1, 69, 64, 64, 8, 64), (3, 5, 5), 0, 64),
    "s4-768": ((1, 69, 96, 96, 8, 64), (3, 5, 5), 0, 96),
}
SWEEP = [
    tile_config(tile, block_n, warps, stages, False)
    for tile in ((4, 4, 4), (4, 4, 2), (2, 8, 4), (4, 2, 4), (2, 4, 4), (2, 4, 8))
    for block_n, warps, stages in itertools.product((32, 64), (2, 4), (1, 2, 3, 4))
] + [
    tile_config(tile, 16, warps, 1, True)
    for tile in ((4, 4, 4), (4, 2, 4), (2, 4, 4), (1, 4, 4), (2, 2, 4), (4, 4, 2))
    for warps in (1, 2, 4)
]


def config_name(config: triton.Config) -> str:
    kw = config.kwargs
    walk = " rows" if kw["BOX_ROWS"] else ""
    return f"{kw['TILE_T']}x{kw['TILE_H']}x{kw['TILE_W']} n{kw['BLOCK_N']} w{config.num_warps} s{config.num_stages}{walk}"


def time_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=50, rep=500)


def run_benchmark(names: list[str], sweep: bool) -> None:
    try:
        import natten
    except ImportError:
        natten = None
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device))
    for name in names:
        shape, kernel, w_start, w_len = SHAPES[name]
        batch, t, h, _w, heads, head_dim = shape
        useful_tflop = 4 * batch * t * h * w_len * math.prod(kernel) * heads * head_dim / 1e12
        generator = torch.Generator(device=device).manual_seed(0)
        k, v = (torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2))
        q_shape = (batch, t, h, w_len, heads, head_dim)
        q = torch.randn(q_shape, device=device, dtype=torch.bfloat16, generator=generator) * head_dim**-0.5
        print(f"\n{name} K/V {shape} queries W[{w_start}:{w_start + w_len}] kernel {kernel}")

        def report(label: str, fn) -> None:
            try:
                diff = (fn() - reference).abs().max().item()
            except OutOfResources as error:
                print(f"  {label:<24} out of resources: {error}")
                return
            ms = time_ms(fn)
            print(f"  {label:<24} {ms:9.2f} ms {useful_tflop / ms * 1e3:6.1f} TFLOPS  diff {diff:.4f}")

        reference = kitchen_na3d(q, k, v, list(kernel), w_start)
        report("comfy-kitchen", lambda: kitchen_na3d(q, k, v, list(kernel), w_start))
        if natten is not None:
            full_q = full_width_queries(q, k, w_start)
            report(
                "natten",
                lambda: natten.na3d(full_q, k, v, kernel_size=kernel, scale=1.0)[:, :, :, w_start : w_start + w_len],
            )
        shipped = config_for(kernel, amd=torch.version.hip is not None)
        report(f"triton {config_name(shipped)}", lambda: na3d_with_config(q, k, v, kernel, w_start, shipped))
        for config in SWEEP if sweep else []:
            report(f"  {config_name(config)}", lambda config=config: na3d_with_config(q, k, v, kernel, w_start, config))
        del q, k, v, reference
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="also time every candidate tile config")
    parser.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    args = parser.parse_args()
    run_benchmark(args.shapes, args.sweep)


if __name__ == "__main__":
    main()

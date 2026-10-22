"""Launch policy of the LTX Triton kernels (``kernels.py``) that needs no Triton: launch constants,
per-GPU tile tables and the block widths the kernels are specialized for. ``kernels.py`` and the
native block (``jasna/ltx/native``) both read it, so the native block also loads where Triton is
missing (Windows ROCm) and runs the kernels a release prebuilt."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from jasna.accelerator import amd_gpu_architecture

ROTATE_QUANTIZE_WARPS = 8
GROUP_M = 8
# Outputs narrower than SMALL_MM_MAX_N (the per-head gate logits) take SMALL_MM_TILE; inputs
# wider than WIDE_K (the FF down projection) take the second tile of INT8_MM_TILES.
SMALL_MM_MAX_N = 128
SMALL_MM_TILE = (32, 32, 256, 3, 2)
WIDE_K = 4096
# Stream-K only pays when the grid is a few waves long.
STREAM_K_MAX_WAVES = 8
# (BM, BN) of the cuBLAS-path epilogue: wide row strips, from a sweep over the LTX block shapes
# on an RTX 2080 Ti (8-17 % faster than 64x64 on every large shape).
CUBLAS_EPILOGUE_TILE = (8, 512)
CUBLAS_EPILOGUE_WARPS = 8
FP4_QUANTIZE_ROWS = 32
FP4_QUANTIZE_COLS = 256
FP4_QUANTIZE_WARPS = 4
# cuBLAS block-scale layout: 128 x 4 tiles of float8 scales, zero padded.
FP4_SCALE_TILE_ROWS = 128
FP4_SCALE_TILE_COLS = 4

# (BM, BN, BK, stages, warps) per GPU for LTX GEMMs of width up to 4096 and wider, from tile
# sweeps over the attention, FF-up and FF-down shapes.
INT8_MM_TILES = {
    (8, 6): ((128, 128, 64, 3, 4),) * 2,  # RTX 3090
    (8, 9): ((256, 128, 128, 3, 8),) * 2,  # RTX 4090
    (12, 0): ((128, 256, 128, 3, 8), (128, 128, 128, 3, 4)),  # RTX 5090
    (12, 1): ((128, 256, 128, 3, 8), (128, 128, 128, 3, 4)),
}
# The same per AMD architecture, keyed by name: ROCm reports gfx1201 as capability 12.0. Swept on
# a Radeon AI PRO R9700 (gfx1201); waves_per_eu, stages 1-3 and 4 vs 8 warps moved the
# plateau tiles by under 2 %. gfx1200 and RDNA3 (gfx110x) take the same tiles, unmeasured.
AMD_INT8_MM_TILES = dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"),
                                  ((128, 256, 128, 2, 8), (128, 128, 128, 1, 8)))
# GPUs on which Triton runs dots as scalar FMA (it has no tensor-core dot below SM 8.0): the
# int8 GEMM runs on cuBLAS (``torch._int_mm``) and only its epilogue in Triton.
CUBLAS_INT8_MM = {(7, 5)}  # RTX 20 series
FP4_ARCHS = ((12, 0), (12, 1))


@dataclass(frozen=True)
class BlockDims:
    """Widths of one transformer block: hidden, feed-forward and attention heads."""

    dim: int
    ff_dim: int
    heads: int

    @classmethod
    def from_specs(cls, specs: dict[str, tuple[object, tuple[int, ...]]]) -> BlockDims:
        """From a block's ``{name: (dtype, shape)}``; works for INT8 and NVFP4 files."""
        def out_features(name: str) -> int:
            return specs[f"{name}.qdata" if f"{name}.qdata" in specs else f"{name}.weight"][1][0]

        return cls(
            dim=specs["attn1.q_norm.weight"][1][0],
            ff_dim=out_features("ff.net.0.proj"),
            heads=out_features("attn1.to_gate_logits"),
        )


def is_amd(architecture: tuple[int, int] | str) -> bool:
    """Whether ``architecture`` (``gpu_architecture``) is an AMD gfx name."""
    return isinstance(architecture, str)


def profile_name(architecture: tuple[int, int] | str, *, fp4: bool) -> str:
    """Directory prefix of one architecture's ahead-of-time kernels: ``sm120-int8``, ``gfx1201-int8``."""
    gpu = architecture if is_amd(architecture) else f"sm{architecture[0]}{architecture[1]}"
    return f"{gpu}-{'nvfp4' if fp4 else 'int8'}"


def has_tma(architecture: tuple[int, int] | str) -> bool:
    return not is_amd(architecture) and architecture[0] >= 9


def gemm_tile(n: int, k: int, architecture: tuple[int, int] | str) -> tuple[int, int, int, int, int]:
    """(BM, BN, BK, stages, warps) of the Triton int8 GEMM for ``n`` outputs of width ``k``."""
    if n < SMALL_MM_MAX_N:
        return SMALL_MM_TILE
    tiles = AMD_INT8_MM_TILES if is_amd(architecture) else INT8_MM_TILES
    return tiles[architecture][k > WIDE_K]


def gpu_architecture(device: torch.device) -> tuple[int, int] | str:
    """Key of the tile tables: the CUDA capability on NVIDIA, the gfx name on AMD."""
    return amd_gpu_architecture(device) if torch.version.hip else torch.cuda.get_device_capability(device)


def supports(device: torch.device) -> bool:
    architecture = gpu_architecture(device)
    return architecture in INT8_MM_TILES or architecture in AMD_INT8_MM_TILES or architecture in CUBLAS_INT8_MM


@functools.cache
def programs(device: torch.device) -> int:
    """Persistent-kernel program count: one per SM."""
    return torch.cuda.get_device_properties(device).multi_processor_count

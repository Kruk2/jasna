"""Native LTX transformer block: ``transformer.block_forward`` as one C++/CUDA call.

``block.cpp`` runs a whole block on plain weight tensors: the Triton kernels of
``ltx/kernels.py`` compiled ahead of time for this GPU (``aot.py``), the same cuBLAS
NVFP4 GEMMs and prompt SDPA as the torch path, comfy-kitchen's INT8 self-attention and
the element-wise glue of ``glue.cu``. Python owns every model constant and launch policy
value and hands them over in ``Config``.

Source installs build it on first use into the compile cache (needs nvcc, a C++ compiler
and ninja); a release ships it prebuilt (``prebuild``). Without the toolchain, on a GPU it
has no kernels for, or with ``JASNA_LTX_NATIVE=0`` the sampler keeps the torch.compile path.
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from jasna.ltx import kernels
from jasna.ltx import transformer as T
from jasna.ltx.native.aot import FP4_ARCHS, BlockDims, build_kernels

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_SOURCES = ("block.cpp", "glue.cu")
PREBUILT = _HERE / "prebuilt"
EXTENSION_NAME = "jasna_ltx_native"
# block.cpp launches comfy-kitchen's INT8 attention the way this release's Python wrapper does
KITCHEN_VERSION = "0.2.36"


def block_quantization(block_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]]) -> str | None:
    """``int8`` or ``nvfp4`` for a block whose linears are quantized, else None."""
    if "attn1.to_v.qdata" in block_specs:
        return "int8"
    return "nvfp4" if block_specs["attn1.to_v.weight"][0] == torch.uint8 else None


def _ninja_available() -> bool:
    """Whether ninja runs; puts the ``ninja`` package's binary on PATH when only it has one."""
    if shutil.which("ninja"):
        return True
    if importlib.util.find_spec("ninja") is None:
        return False
    import ninja

    if not ninja.BIN_DIR:
        return False
    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    return True


def _prebuilt_extension() -> Path | None:
    return next(PREBUILT.glob(f"{EXTENSION_NAME}*.so"), None)


def unavailable_reason(device: torch.device, block_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]]) -> str | None:
    """Why the native block cannot run blocks shaped like ``block_specs`` here, or None when it can."""
    if os.environ.get("JASNA_LTX_NATIVE") == "0":
        return "turned off by JASNA_LTX_NATIVE=0"
    if device.type != "cuda" or torch.version.cuda is None:
        return "it needs an NVIDIA GPU"
    dims = BlockDims.from_specs(block_specs)
    quantization = block_quantization(block_specs)
    if quantization is None:
        return "the model's linears are not INT8 or NVFP4"
    major, minor = torch.cuda.get_device_capability(device)
    if not kernels.supports(device):
        return f"there are no tuned kernels for SM {major}.{minor}"
    if quantization == "nvfp4" and (major, minor) not in FP4_ARCHS:
        return f"NVFP4 blocks need SM {' / '.join(f'{a}.{b}' for a, b in FP4_ARCHS)}"
    if dims.dim // dims.heads not in (64, 128, 256) or dims.dim % dims.heads:
        return f"INT8 attention has no kernel for {dims.dim // dims.heads}-wide heads"
    if importlib.metadata.version("comfy-kitchen") != KITCHEN_VERSION:
        return f"it launches the INT8 attention of comfy-kitchen {KITCHEN_VERSION} only"
    if _prebuilt_extension() is not None:
        return None
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        return "no CUDA compiler (nvcc) found to build it"
    if shutil.which(os.environ.get("CXX", "c++")) is None:
        return "no C++ compiler found to build it"
    if not _ninja_available():
        return "ninja is not installed"
    return None


def _cache() -> Path:
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return Path(cache_dir()) / "jasna-ltx-native"


def _build_extension(build_directory: Path):
    from torch.utils import cpp_extension

    if not _ninja_available():
        raise RuntimeError("building the LTX native block needs ninja")
    build_directory.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name=EXTENSION_NAME,
        sources=[str(_HERE / source) for source in _SOURCES],
        build_directory=str(build_directory),
        extra_cflags=["-O3", "-fvisibility=hidden"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{Path(cpp_extension.CUDA_HOME) / 'lib64' / 'stubs'}", "-lcuda"],
    )


def _extension():
    prebuilt = _prebuilt_extension()
    if prebuilt is None:
        return _build_extension(_cache() / f"extension-torch{torch.__version__}")
    spec = importlib.util.spec_from_file_location(EXTENSION_NAME, prebuilt)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel_manifest(capability: tuple[int, int], dims: BlockDims, *, fp4: bool) -> Path:
    prebuilt = PREBUILT / "kernels"
    if prebuilt.is_dir():
        return build_kernels(capability, dims, fp4=fp4, cache=prebuilt)
    return build_kernels(capability, dims, fp4=fp4, cache=_cache() / "kernels")


def _linear_tensors(w: T.Weights, name: str) -> tuple[torch.Tensor | None, ...]:
    """(weight, scale, scale_2, bias) of one linear in ``block.cpp``'s slot order."""
    if f"{name}.qdata" in w:
        return w[f"{name}.qdata"], w[f"{name}.scales"], None, w.get(f"{name}.bias")
    return w[f"{name}.weight"], w.get(f"{name}.weight_scale"), w.get(f"{name}.weight_scale_2"), w.get(f"{name}.bias")


class NativeBlockForward:
    """``transformer.block_forward`` (same arguments, default heads) for one model's blocks."""

    def __init__(self, module, device: torch.device, dims: BlockDims, manifest: Path) -> None:
        import comfy_kitchen.backends.cuda as kitchen_cuda
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

        capability = torch.cuda.get_device_capability(device)
        config = module.Config()
        config.device = device.index if device.index is not None else torch.cuda.current_device()
        config.heads = dims.heads
        config.eps = T.EPS
        config.chunk_rows = T.CHUNK_ROWS
        config.cublas_int8 = capability in kernels.CUBLAS_INT8_MM
        tiles = kernels.INT8_MM_TILES.get(capability, (kernels.SMALL_MM_TILE,) * 2)
        config.narrow_tile, config.wide_tile = (list(tile) for tile in tiles)
        config.small_tile = list(kernels.SMALL_MM_TILE)
        config.small_max_n = kernels.SMALL_MM_MAX_N
        config.wide_k = kernels.WIDE_K
        config.group_m = kernels.GROUP_M
        config.stream_k_max_waves = kernels.STREAM_K_MAX_WAVES
        config.programs = kernels.programs(device)
        config.has_tma = kernels.has_tma(capability)
        config.epilogue_tile = list(kernels.CUBLAS_EPILOGUE_TILE)
        config.fp4_tensor_scale_divisor = T.FP4_MAX * T.FP8_E4M3_MAX
        config.fp4_block = T.FP4_BLOCK
        config.fp4_quantize_rows = kernels.FP4_QUANTIZE_ROWS
        config.fp4_quantize_cols = kernels.FP4_QUANTIZE_COLS
        config.fp4_scale_tile_rows = kernels.FP4_SCALE_TILE_ROWS
        config.fp4_scale_tile_cols = kernels.FP4_SCALE_TILE_COLS
        config.fp4_recipe = [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value]
        config.fp4_swizzle = [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value]
        config.kitchen_library = kitchen_cuda._C.__file__
        config.kitchen_bf16 = DTYPE_TO_CODE[torch.bfloat16]
        config.kernels = json.loads(manifest.read_text())
        config.kernel_directory = str(manifest.parent)
        self.block = module.Block(config)
        self.head_dim = dims.dim // dims.heads

    @functools.cache
    def cta_k(self, rows: int) -> int:
        """comfy-kitchen's INT8 attention key tile for ``rows`` tokens."""
        from comfy_kitchen.sage_attention import _select_cta_k

        return _select_cta_k(self.head_dim, rows, has_mask=False)

    def __call__(
        self,
        w: T.Weights,
        x: torch.Tensor,
        cond: T.StepConditioning,
        prompt: tuple[torch.Tensor, torch.Tensor],
        rope: T.Rope,
        *,
        skip_self_attention: bool = False,
    ) -> torch.Tensor:
        linears = [tensor for name in T.LINEAR_NAMES for tensor in _linear_tensors(w, name)]
        with sdpa_kernel(T.SDPA_PRIORITY, set_priority=True):
            return self.block.forward(
                linears, w["attn1.q_norm.weight"], w["attn1.k_norm.weight"], w["attn2.q_norm.weight"],
                w["scale_shift_table"], x, cond.block_rows, *prompt, rope.cos, rope.sin, self.cta_k(x.shape[1]),
                skip_self_attention,
            )


def load(device: torch.device, block_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]]) -> NativeBlockForward | None:
    """The native block forward for blocks shaped like ``block_specs``, or None (logged) when it
    cannot run here, so the caller keeps the torch.compile path."""
    reason = unavailable_reason(device, block_specs)
    if reason is not None:
        logger.info("LTX native blocks off (%s); using the torch.compile path", reason)
        return None
    dims = BlockDims.from_specs(block_specs)
    try:
        manifest = _kernel_manifest(
            torch.cuda.get_device_capability(device), dims, fp4=block_quantization(block_specs) == "nvfp4"
        )
        return NativeBlockForward(_extension(), device, dims, manifest)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        logger.warning("LTX native blocks failed to build or load; using the torch.compile path: %s", error)
        return None


def prebuild(directory: Path) -> None:
    """Release build: kernels for every architecture of the kernel tables and the extension for
    all of them, laid out as ``PREBUILT`` expects (copy ``directory`` there)."""
    from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors

    archs = sorted({*kernels.INT8_MM_TILES, *kernels.CUBLAS_INT8_MM})
    for quantization in ("int8", "nvfp4"):
        specs = PlaceholderHandle(PlaceholderTensors(quantization, "distilled")).tensor_specs()
        dims = BlockDims.from_specs({name.split(".", 2)[2]: spec for name, spec in specs.items() if name.startswith("blocks.0.")})
        for capability in archs:
            if quantization == "int8" or capability in FP4_ARCHS:
                build_kernels(capability, dims, fp4=quantization == "nvfp4", cache=directory / "kernels")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{major}.{minor}" for major, minor in archs)
    module = _build_extension(directory / "build")
    shutil.copy2(module.__file__, directory / Path(module.__file__).name)
    shutil.rmtree(directory / "build")

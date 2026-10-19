"""Native LTX transformer block: ``transformer.block_forward`` as one C++/CUDA call.

``block.cpp`` runs a whole block on plain weight tensors: the Triton kernels of
``ltx/kernels.py`` compiled ahead of time for this GPU (``aot.py``), the same cuBLAS
NVFP4 GEMMs and prompt SDPA as the torch path, the INT8 self-attention vendored from
comfy-kitchen (``sage_attention/``) and the element-wise glue of ``glue.cu``. Python owns
every model constant and launch policy value and hands them over in ``Config``.

Source installs build it on first use into the compile cache (needs nvcc, a C++ compiler
and ninja; on Windows MSVC's ``cl`` on PATH, from an x64 Visual Studio developer shell whose
MSVC this CUDA supports); a frozen release never builds it and loads it only when ``prebuild``
output is shipped in ``PREBUILT``. Without the toolchain, on a GPU it has no kernels for, or with
``JASNA_LTX_NATIVE=0`` the sampler keeps the torch.compile path.
"""

from __future__ import annotations

import hashlib
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

from jasna._frozen import is_frozen
from jasna.ltx import kernels
from jasna.ltx import transformer as T
from jasna.ltx.native.aot import FP4_ARCHS, BlockDims, build_kernels

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_SOURCES = ("module.cpp", "block.cpp", "glue.cu")
_ATTENTION = _HERE / "sage_attention"
_ATTENTION_SOURCES = ("quant_qk_int8.cu", "quant_v_int8.cu", "sage_attn_launcher.cu")
# comfy-kitchen 0.2.38's nvcc flags (its CMakeLists.txt): the vendored attention computes kitchen's
# bits only with fast math, which glue.cu must not get, so it compiles apart from the extension.
# NDEBUG keeps CUDA 13's fp4/fp8 header asserts out of device code, which the release build's
# glibc 2.35 host headers cannot compile.
_ATTENTION_NVCC_FLAGS = ("-std=c++17", "-O3", "-DNDEBUG", "--use_fast_math", "--expt-relaxed-constexpr")
PREBUILT = _HERE / "prebuilt"
_IS_WINDOWS = os.name == "nt"
EXTENSION_NAME = "jasna_ltx_native"


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
    if _prebuilt_extension() is not None:
        return None
    if is_frozen():
        return "a release ships no native block to load and cannot build one"
    from torch.utils.cpp_extension import CUDA_HOME, get_cxx_compiler

    if CUDA_HOME is None or not _nvcc(Path(CUDA_HOME)).is_file():
        return "no CUDA compiler (nvcc) found to build it"
    compiler = get_cxx_compiler()
    if shutil.which(compiler) is None:
        return f"no C++ compiler ({compiler}) found to build it"
    if not _ninja_available():
        return "ninja is not installed"
    return None


def _cache() -> Path:
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return Path(cache_dir()) / "jasna-ltx-native"


def _nvcc(cuda_home: Path) -> Path:
    return cuda_home / "bin" / ("nvcc.exe" if _IS_WINDOWS else "nvcc")


def _build_flags(cuda_home: Path) -> dict[str, list[str]]:
    """Compiler and linker flags to build the extension against the CUDA driver API, per platform."""
    if _IS_WINDOWS:
        return {
            "extra_cflags": ["/O2"],
            "extra_cuda_cflags": ["-O3"],
            "extra_ldflags": [str(cuda_home / "lib" / "x64" / "cuda.lib")],
        }
    return {
        "extra_cflags": ["-O3", "-fvisibility=hidden"],
        "extra_cuda_cflags": ["-O3"],
        "extra_ldflags": [f"-L{cuda_home / 'lib64' / 'stubs'}", "-lcuda"],
    }


def _attention_nvcc_command(cuda_home: Path) -> list[str]:
    """nvcc with every flag but the source and object, for the extension's host compiler and architectures."""
    from torch.utils import cpp_extension

    host = ("/MD", "/Zc:preprocessor") if _IS_WINDOWS else ("-fPIC", "-fvisibility=hidden")
    ccbin = ["-ccbin", os.environ["CC"]] if "CC" in os.environ else []
    return [str(_nvcc(cuda_home)), *ccbin, *cpp_extension._get_cuda_arch_flags(), *_ATTENTION_NVCC_FLAGS,
            *(f"-Xcompiler={flag}" for flag in host)]


def _attention_objects(build_directory: Path, cuda_home: Path) -> list[Path]:
    """The vendored INT8 attention compiled for the extension's link, in a directory named after its
    sources and command so a change relinks the extension."""
    command = _attention_nvcc_command(cuda_home)
    digest = hashlib.sha256("\0".join(command).encode())
    for path in sorted(_ATTENTION.iterdir()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    directory = build_directory / f"sage_attention-{digest.hexdigest()[:16]}"
    directory.mkdir(parents=True, exist_ok=True)
    objects = []
    for source in _ATTENTION_SOURCES:
        target = directory / Path(source).with_suffix(".obj" if _IS_WINDOWS else ".o").name
        if not target.is_file():
            partial = target.with_name(f"partial-{target.name}")
            subprocess.run([*command, "-c", str(_ATTENTION / source), "-o", str(partial)], check=True)
            partial.replace(target)
        objects.append(target)
    return objects


def _build_extension(build_directory: Path):
    from torch.utils import cpp_extension

    if not _ninja_available():
        raise RuntimeError("building the native LTX block needs ninja")
    build_directory.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("CC", "CXX")}
    if shutil.which("gcc-15") and shutil.which("g++-15"):
        os.environ["CC"], os.environ["CXX"] = "gcc-15", "g++-15"
    try:
        cuda_home = Path(cpp_extension.CUDA_HOME)
        flags = _build_flags(cuda_home)
        flags["extra_ldflags"] += [str(path) for path in _attention_objects(build_directory, cuda_home)]
        return cpp_extension.load(
            name=EXTENSION_NAME,
            sources=[str(_HERE / source) for source in _SOURCES],
            build_directory=str(build_directory),
            **flags,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extension():
    prebuilt = _prebuilt_extension()
    if prebuilt is None:
        checkout = hashlib.sha256(str(_HERE).encode()).hexdigest()[:12]
        return _build_extension(_cache() / f"extension-torch{torch.__version__}-{checkout}")
    spec = importlib.util.spec_from_file_location(EXTENSION_NAME, prebuilt)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel_manifest(capability: tuple[int, int], dims: BlockDims, *, fp4: bool) -> Path:
    prebuilt = PREBUILT / "kernels"
    if prebuilt.is_dir():
        return build_kernels(capability, dims, fp4=fp4, cache=prebuilt)
    return build_kernels(capability, dims, fp4=fp4, cache=_cache() / "kernels")


def attention_cta_k(head_dim: int, rows: int) -> int:
    """The INT8 attention key tile for ``rows`` tokens, as comfy-kitchen 0.2.38 picks it unmasked."""
    return 128 if head_dim >= 128 and rows > 1024 else 64


def _linear_tensors(w: T.Weights, name: str) -> tuple[torch.Tensor | None, ...]:
    """(weight, scale, scale_2, bias) of one linear in ``block.cpp``'s slot order."""
    if f"{name}.qdata" in w:
        return w[f"{name}.qdata"], w[f"{name}.scales"], None, w.get(f"{name}.bias")
    return w[f"{name}.weight"], w.get(f"{name}.weight_scale"), w.get(f"{name}.weight_scale_2"), w.get(f"{name}.bias")


class NativeBlockForward:
    """``transformer.block_forward`` (same arguments, default heads) for one model's blocks."""

    def __init__(self, module, device: torch.device, dims: BlockDims, manifest: Path) -> None:
        values = config_values(device, dims, manifest)
        config = module.Config()
        for name, value in values.items():
            setattr(config, name, value)
        self.block = module.Block(config)
        self.head_dim = dims.dim // dims.heads

    def __call__(
        self, w: T.Weights, x: torch.Tensor, cond: T.StepConditioning,
        prompt: tuple[torch.Tensor, torch.Tensor], rope: T.Rope, *, skip_self_attention: bool = False,
    ) -> torch.Tensor:
        linears = [tensor for name in T.LINEAR_NAMES for tensor in _linear_tensors(w, name)]
        with sdpa_kernel(T.SDPA_PRIORITY, set_priority=True):
            return self.block.forward(
                linears, w["attn1.q_norm.weight"], w["attn1.k_norm.weight"], w["attn2.q_norm.weight"],
                w["scale_shift_table"], x, cond.block_rows, *prompt, rope.cos, rope.sin,
                attention_cta_k(self.head_dim, x.shape[1]), skip_self_attention,
            )


def block_parameters(capability: tuple[int, int], heads: int) -> dict[str, object]:
    """Public launch policy from architecture only; no device or kernel queries."""
    tiles = kernels.INT8_MM_TILES.get(capability, (kernels.SMALL_MM_TILE,) * 2)
    return {
        "heads": heads, "eps": T.EPS, "chunk_rows": T.CHUNK_ROWS,
        "cublas_int8": capability in kernels.CUBLAS_INT8_MM,
        "narrow_tile": list(tiles[0]), "wide_tile": list(tiles[1]), "small_tile": list(kernels.SMALL_MM_TILE),
        "small_max_n": kernels.SMALL_MM_MAX_N, "wide_k": kernels.WIDE_K, "group_m": kernels.GROUP_M,
        "stream_k_max_waves": kernels.STREAM_K_MAX_WAVES,
        "has_tma": kernels.has_tma(capability), "epilogue_tile": list(kernels.CUBLAS_EPILOGUE_TILE),
        "fp4_tensor_scale_divisor": T.FP4_MAX * T.FP8_E4M3_MAX, "fp4_block": T.FP4_BLOCK,
        "fp4_quantize_rows": kernels.FP4_QUANTIZE_ROWS, "fp4_quantize_cols": kernels.FP4_QUANTIZE_COLS,
        "fp4_scale_tile_rows": kernels.FP4_SCALE_TILE_ROWS, "fp4_scale_tile_cols": kernels.FP4_SCALE_TILE_COLS,
        "fp4_recipe": [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value],
        "fp4_swizzle": [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value],
    }


def config_values(device: torch.device, dims: BlockDims, manifest: Path) -> dict[str, object]:
    """Typed native-block configuration shared by public and protected bindings."""
    capability = torch.cuda.get_device_capability(device)
    return {
        "device": device.index if device.index is not None else torch.cuda.current_device(),
        **block_parameters(capability, dims.heads),
        "programs": kernels.programs(device),
        "kernels": json.loads(manifest.read_text()), "kernel_directory": str(manifest.parent),
    }


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


def release_architectures() -> list[tuple[int, int]]:
    """Every SM a release build compiles the native block for: those with tuned INT8 kernels."""
    return sorted({*kernels.INT8_MM_TILES, *kernels.CUBLAS_INT8_MM})


def cuda_arch_list() -> str:
    """``TORCH_CUDA_ARCH_LIST`` for a release build of code that links the native block."""
    return ";".join(f"{major}.{minor}" for major, minor in release_architectures())


def prebuild_profiles() -> list[tuple[tuple[int, int], BlockDims, bool]]:
    """The public architecture and model-shape profiles compiled into a release."""
    from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors

    archs = release_architectures()
    profiles = []
    for quantization in ("int8", "nvfp4"):
        specs = PlaceholderHandle(PlaceholderTensors(quantization, "distilled")).tensor_specs()
        dims = BlockDims.from_specs({name.split(".", 2)[2]: spec for name, spec in specs.items() if name.startswith("blocks.0.")})
        for capability in archs:
            if quantization == "int8" or capability in FP4_ARCHS:
                profiles.append((capability, dims, quantization == "nvfp4"))
    return profiles


def prebuild_kernels(directory: Path) -> None:
    """Build release kernel profiles without building the native extension or querying a GPU."""
    for capability, dims, fp4 in prebuild_profiles():
        build_kernels(capability, dims, fp4=fp4, cache=directory / "kernels")


def prebuild(directory: Path) -> None:
    """Release build: kernels for every architecture of the kernel tables and the extension for
    all of them, laid out as ``PREBUILT`` expects (copy ``directory`` there)."""
    prebuild_kernels(directory)
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list()
    module = _build_extension(directory / "build")
    shutil.copy2(module.__file__, directory / Path(module.__file__).name)
    shutil.rmtree(directory / "build")

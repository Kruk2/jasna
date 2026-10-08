from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
from enum import StrEnum
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
#
# Expandable segments release memory by unmapping virtual address ranges rather
# than by a device-synchronizing free, so a VramOffloader empty_cache() could
# pull pages out from under kernels another thread still had in flight — issue
# #252 caught a restorer fp16 GEMM faulting with "Page not present". The env
# names cover the versions in the field; whichever one the build reads wins.
#
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL opts in to the AOTriton flash-attention
# kernels on Radeon parts that PyTorch ships as "not officially validated". It is
# what makes FLASH/EFFICIENT SDPA usable on the consumer cards this build targets;
# see aotriton_image_dir() below for the second half of that story.
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:
    environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


# --- AOTriton flash-attention availability -----------------------------------
#
# Windows ROCm wheels split the AOTriton payload across two device packages and
# neither one is sufficient on its own:
#
#   amd-torch-device-<exact arch>    -> torch/.kpack/torch_<arch>.kpack
#                                       (the torch kernels)
#   amd-torch-device-<arch family>   -> torch/lib/aotriton.images/<family>/...
#                                       (the SDPA kernel images)
#
# With only the exact-arch package installed the images are absent and the first
# FLASH/EFFICIENT SDPA launch submits a null kernel: hipModuleLaunchKernel is
# called with a null function pointer and the process dies with
# hipErrorInvalidImage (ROCm/TheRock#7315). Disabling those two backends keeps the
# run alive on the numerically identical MATH backend, but it is much slower — on
# RF-DETR's DINOv2 backbone, which is attention bound, MATH costs 38.3 ms/frame
# against 18.8 ms/frame with FLASH at 576x576 (RX 7900 XT, gfx1100). So the
# backends are only switched off when the images for *this* GPU are really
# missing.
_AOTRITON_IMAGES_DIRNAME = "aotriton.images"
_AOTRITON_IMAGE_GLOB = "*/flash/attn_fwd.zip"
_FORCE_MATH_ENV = "JASNA_FORCE_MATH_SDP"
_TRUTHY = {"1", "true", "yes", "on"}


def _image_sets(root: Path) -> list[Path]:
    """Image directories under ``root`` that carry a complete flash kernel set."""
    if not root.is_dir():
        return []
    return sorted({marker.parent.parent for marker in root.glob(_AOTRITON_IMAGE_GLOB)})


def _device_gcn_arch() -> str:
    """``gfxNNNN`` for device 0, or an empty string when it cannot be read.

    Only consulted when several image sets are installed, so the usual single-set
    case never initialises the device just to answer this question.
    """
    try:
        name = str(torch.cuda.get_device_properties(0).gcnArchName)
    except Exception:  # pragma: no cover - depends on the runtime
        return ""
    return name.split(":")[0].strip()


def aotriton_image_dir(lib_dir: Path | str | None = None, *, arch: str | None = None) -> Path | None:
    """AOTriton image directory usable by this GPU, or ``None`` when there is none.

    ``lib_dir`` overrides ``torch/lib`` (used by the tests); ``arch`` overrides the
    device query. Image sets are named after the architecture family, so a
    ``gfx1100`` device is served by ``amd-gfx110x`` and a ``gfx1201`` by
    ``amd-gfx120x``; an exact-architecture directory wins when it exists.
    """
    base = Path(lib_dir) if lib_dir is not None else Path(torch.__file__).resolve().parent / "lib"
    sets = _image_sets(base / _AOTRITON_IMAGES_DIRNAME)
    if not sets:
        return None
    if len(sets) == 1:
        return sets[0]

    # Several families installed (or a shared wheel): pick the one for this GPU.
    resolved_arch = arch if arch is not None else _device_gcn_arch()
    if not resolved_arch:
        logger.debug(
            "Several AOTriton image sets found but the GPU arch is unknown; keeping the MATH backend"
        )
        return None
    names = {resolved_arch, resolved_arch[:-1] + "x"} if len(resolved_arch) > 1 else {resolved_arch}
    for name in (resolved_arch, *sorted(names - {resolved_arch})):
        for candidate in sets:
            if candidate.name.rsplit("-", 1)[-1] == name:
                return candidate
    logger.debug(
        "AOTriton image sets %s do not cover %s; keeping the MATH backend",
        [c.name for c in sets],
        resolved_arch,
    )
    return None


def _math_sdp_forced() -> bool:
    return os.environ.get(_FORCE_MATH_ENV, "").strip().lower() in _TRUTHY


#: Device-visibility restrictions apply in the HIP runtime's own enumeration, while
#: HSA-level tools (``hipInfo``, ``rocm-smi``) still list every agent. A machine whose
#: iGPU is device 0 and whose card is device 1 therefore looks like a one-GPU machine to
#: PyTorch: the pipeline runs on the iGPU and the MIGraphX engine dies with
#: ``RUNTIME_EXCEPTION ... Failed to call function`` (gfx103x has no device code).
#: jasna needs the whole list to find the discrete Radeon, unless the user pins a device
#: on purpose.
_VISIBLE_DEVICES_ENV = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
_KEEP_VISIBLE_DEVICES_ENV = "JASNA_KEEP_HIP_VISIBLE_DEVICES"


def allow_all_devices(environ: MutableMapping[str, str] | None = None) -> None:
    """Drop a device-visibility variable so every GPU is enumerated again.

    Must happen before the first ``torch.cuda`` call: the HIP runtime reads
    ``HIP_VISIBLE_DEVICES`` when it initialises, so clearing it later changes nothing.
    Set ``JASNA_KEEP_HIP_VISIBLE_DEVICES=1`` (e.g. a workstation that pins one card per
    job) to keep the restriction and select with ``--device`` instead.
    """
    env = os.environ if environ is None else environ
    if str(env.get(_KEEP_VISIBLE_DEVICES_ENV, "")).strip().lower() in _TRUTHY:
        return
    for name in _VISIBLE_DEVICES_ENV:
        value = env.pop(name, None)
        if value is not None:
            logger.info(
                "%s=%s restricted this machine's visible GPUs; listing all of them so "
                "jasna can pick the discrete Radeon (set %s=1 to keep the restriction)",
                name, value, _KEEP_VISIBLE_DEVICES_ENV,
            )


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    allow_all_devices()
    if torch.version.hip:
        apply_rocm_env_defaults(os.environ)
        if _math_sdp_forced():
            logger.info("%s is set: using the MATH SDPA backend", _FORCE_MATH_ENV)
        elif aotriton_image_dir() is None:
            logger.warning(
                "No AOTriton SDPA images for this GPU were found, so FLASH/EFFICIENT "
                "attention cannot be launched (ROCm/TheRock#7315). Falling back to the "
                "MATH backend: correct but noticeably slower. Install the image package "
                "for this architecture, e.g. 'amd-torch-device-gfx110x' for gfx1100/1101/"
                "1102/1103, to get the fast path back."
            )
        else:
            return
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:  # pragma: no cover - depends on the torch build
            pass


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None and resolved.type == "cpu":
        return AcceleratorVendor.CPU
    if torch.version.hip:
        return AcceleratorVendor.AMD
    if torch.version.cuda:
        return AcceleratorVendor.NVIDIA
    return AcceleratorVendor.CPU


def is_nvidia_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.NVIDIA


def is_amd_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.AMD


#: RDNA 3 / 4 *discrete* cards. Integrated Radeons (gfx103x, gfx1103, gfx115x) share
#: system memory and carry no MIGraphX device code at all, so a desktop whose driver
#: order puts the iGPU first would otherwise run the whole pipeline - and the detection
#: engine - on the wrong device.
DISCRETE_RADEON_ARCHS = frozenset({
    "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201",
})

#: ``preferred_gpu_index`` reports the device list once per process (see below).
_device_choice_logged = False


def hip_device_archs() -> list[str]:
    """``gcnArchName`` of every CUDA/HIP device, in the runtime's device order."""
    try:
        count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - no GPU / no usable runtime
        return []
    archs: list[str] = []
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            arch = str(getattr(props, "gcnArchName", "") or "").split(":")[0].strip()
        except Exception:  # pragma: no cover - depends on the runtime
            arch = ""
        archs.append(arch)
    return archs


def preferred_gpu_index() -> int:
    """Device index jasna should run on: the first *discrete* Radeon, otherwise 0.

    Some driver builds enumerate a Ryzen iGPU as device 0 and the discrete card as
    device 1. Everything that hard-coded ``cuda:0`` then targeted the iGPU: the MIGraphX
    execution provider aborts on ``gfx103x`` (``RUNTIME_EXCEPTION ... Failed to call
    function``, there is no device code for it) and the rest of the pipeline runs on the
    slowest device in the machine. Selecting the discrete Radeon keeps the caller on the
    card the user actually bought. Machines without one - a Strix Halo APU, an NVIDIA
    card, plain CPU - keep device 0, so this is a no-op for them.
    """
    global _device_choice_logged
    if not torch.version.hip:
        return 0
    archs = hip_device_archs()
    choice = 0
    for index, arch in enumerate(archs):
        if arch in DISCRETE_RADEON_ARCHS:
            choice = index
            break
    if not _device_choice_logged:
        # Once per process: this is the line to look at when a machine behaves as if it
        # had one GPU, or as if the engine ran on the wrong one.
        _device_choice_logged = True
        logger.info("ROCm devices %s -> running on device %d", archs or "none", choice)
    return choice


def preferred_device() -> torch.device:
    """The device jasna should use: :func:`preferred_gpu_index` resolved to a device."""
    return torch.device(f"cuda:{preferred_gpu_index()}")


def device_module(device: torch.device | str):
    return torch.get_device_module(torch.device(device))


def device_context(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return nullcontext()
    return device_module(resolved).device(resolved)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    try:
        return device_module(stream.device).stream(stream)
    except (TypeError, ValueError):
        # Also supports lightweight stream doubles used by callers/tests.
        return torch.cuda.stream(stream)


def new_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).Stream(resolved)


def current_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).current_stream(resolved)


def new_event(device: torch.device | str):
    return device_module(torch.device(device)).Event()


def set_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).set_device(resolved)


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.accelerator.synchronize()
        return
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).synchronize(resolved)


def empty_cache(device: torch.device | str | None = None) -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.empty_cache()
        return
    if device is not None:
        module = device_module(torch.device(device))
        if hasattr(module, "empty_cache"):
            module.empty_cache()


def ipc_collect(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def reset_peak_memory_stats(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(torch.device(device))


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    module = device_module(torch.device(device))
    return module.mem_get_info(torch.device(device))


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))

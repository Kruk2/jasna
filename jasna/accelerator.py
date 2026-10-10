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


def _device_pin_file() -> str | None:
    """A stale ``sitecustomize.py`` that pins device visibility, if one is installed.

    ``sitecustomize`` is imported by ``site`` before the application runs, so a helper
    that writes ``HIP_VISIBLE_DEVICES`` from a cached index re-applies the pin inside
    every interpreter: clearing the variable at the entry point changes nothing, and the
    symptom looks like a driver problem.
    """
    import site

    directories = [*site.getsitepackages(), site.getusersitepackages()]
    for directory in directories:
        candidate = Path(directory) / "sitecustomize.py"
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "HIP_VISIBLE_DEVICES" in text:
            return str(candidate)
    return None


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
            pin = _device_pin_file()
            if pin:
                logger.warning(
                    "%s is pinned again at every interpreter start by %s; rename it to "
                    "sitecustomize.py.disabled (or delete it) to keep the pin gone",
                    name, pin,
                )


_VRAM_LIMIT_GB_ENV = "JASNA_VRAM_LIMIT_GB"
_VRAM_FRACTION_ENV = "JASNA_VRAM_FRACTION"
#: Small fixed headroom subtracted from the *discrete* card's VRAM when no
#: explicit budget is set: 500 MB covers the compositor's basic buffers without
#: giving away gigabytes of the card to the desktop.
_VRAM_DEFAULT_HEADROOM_BYTES = 500 * 1024 * 1024
#: Fraction of the budget above which the allocator's reserve counts as hoarded
#: cold blocks and a clip boundary triggers a defragmenting release
#: (JASNA_VRAM_DEFRAG_THRESHOLD overrides).
_VRAM_DEFRAG_THRESHOLD_DEFAULT = 0.75
_VRAM_DEFRAG_THRESHOLD_ENV = "JASNA_VRAM_DEFRAG_THRESHOLD"
#: Minimum seconds between two defragmenting releases. empty_cache forces the
#: next clips to re-allocate their workspaces, so back-to-back cleanups churn
#: (reported as "slower towards the end") without reclaiming anything new.
_VRAM_DEFRAG_COOLDOWN_S = 60.0
_last_defrag_monotonic = 0.0
#: The fraction apply_vram_budget last installed (None = no budget applied).
_vram_budget_fraction: float | None = None


def apply_vram_budget() -> None:
    """Cap the CUDA caching allocator so long jobs stop ballooning.

    The allocator caches every freed block and never returns VRAM on its own,
    so a long 4K job creeps toward the whole card; once Windows starts demoting
    pages to shared memory (PCIe slow path) the frame rate collapses.
    ``set_per_process_memory_fraction`` makes the allocator reclaim its own
    cached blocks at the cap instead of growing past it.

    The budget is derived from the discrete card that ``preferred_gpu_index``
    picks (the iGPU is skipped): that card's VRAM minus 500 MB. Tune with
    ``JASNA_VRAM_LIMIT_GB`` (absolute GB) or ``JASNA_VRAM_FRACTION`` (0-1].
    """
    global _vram_budget_fraction
    if not torch.cuda.is_available():
        return
    try:
        index = preferred_gpu_index()
        total = torch.cuda.get_device_properties(index).total_memory
        if not total:
            return
        limit = os.environ.get(_VRAM_LIMIT_GB_ENV, "").strip()
        frac_env = os.environ.get(_VRAM_FRACTION_ENV, "").strip()
        if limit:
            fraction = float(limit) * (1024 ** 3) / total
        elif frac_env:
            fraction = float(frac_env)
        else:
            fraction = (total - _VRAM_DEFAULT_HEADROOM_BYTES) / total
        if not 0.05 <= fraction <= 0.99:
            logger.warning(
                "VRAM budget %s resolves to fraction %.2f, outside 0.05-0.99; ignored",
                limit or frac_env or "default", fraction,
            )
            return
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:  # noqa: BLE001
            name = "GPU"
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        _vram_budget_fraction = fraction
        logger.info(
            "VRAM budget: device %d (%s) capped at %.1f GB of %.1f GB (%.0f%%); "
            "tune with %s / %s",
            index, name,
            total * fraction / (1024 ** 3), total / (1024 ** 3), fraction * 100,
            _VRAM_LIMIT_GB_ENV, _VRAM_FRACTION_ENV,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM budget not applied: %s", exc)


def release_vram_cache() -> None:
    """Return every cached-but-unused block to the driver (defragmentation).

    Live allocations - the resident detector and secondary restorer, the
    current clip's tensors - are untouched. Only the allocator's cold cache is
    released, which is what otherwise grows until Windows starts demoting the
    hot pages to shared memory and the frame rate collapses.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def maybe_release_vram_cache() -> bool:
    """Defragment when the reserve creeps toward the budget; cheap no-op otherwise.

    Called at clip boundaries. The check itself only reads allocator stats; the
    actual release runs once the reserve crosses the defrag threshold. The freed
    blocks are the allocator's cold cache - the part that does not need to sit
    in VRAM - which the driver may then hold in system memory; the
    speed-critical resident models stay untouched and, with the reserve back
    under the budget, un-demoted.
    """
    global _last_defrag_monotonic
    if not torch.cuda.is_available():
        return False
    try:
        threshold = float(
            os.environ.get(_VRAM_DEFRAG_THRESHOLD_ENV, "").strip()
            or _VRAM_DEFRAG_THRESHOLD_DEFAULT
        )
        if not 0.1 <= threshold <= 1.0:
            threshold = _VRAM_DEFRAG_THRESHOLD_DEFAULT
        index = preferred_gpu_index()
        total = torch.cuda.get_device_properties(index).total_memory
        allowed = int(total * _vram_budget_fraction) if _vram_budget_fraction \
            else int(total - _VRAM_DEFAULT_HEADROOM_BYTES)
        reserved = torch.cuda.memory_reserved(index)
        if allowed and reserved > allowed * threshold:
            import time as _time

            now = _time.monotonic()
            if now - _last_defrag_monotonic < _VRAM_DEFRAG_COOLDOWN_S:
                return False
            _last_defrag_monotonic = now
            logger.info(
                "VRAM defrag: %.2f GB reserved exceeds %.0f%% of the %.2f GB budget; "
                "returning the cold cache to the driver",
                reserved / (1024 ** 3), threshold * 100, allowed / (1024 ** 3),
            )
            release_vram_cache()
            logger.info(
                "VRAM defrag: reserve %.2f GB -> %.2f GB",
                reserved / (1024 ** 3), torch.cuda.memory_reserved(index) / (1024 ** 3),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM defrag check failed: %s", exc)
    return False


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    allow_all_devices()
    apply_vram_budget()
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

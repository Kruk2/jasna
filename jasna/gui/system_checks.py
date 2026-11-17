"""Dependency checks and the CUDA warm-up, run off the UI thread."""

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable

from jasna import os_utils, system_requirements
from jasna.accelerator import AcceleratorVendor
from jasna.gui.locales import t
from jasna.gui.workers import run_in_background

logger = logging.getLogger(__name__)

WARNING_ONLY_CHECKS = {"sysmem"}


def evaluate_check_results(
    results: dict[str, tuple[bool | None, str]], keys: Iterable[str]
) -> tuple[bool, bool]:
    """Return ``(all_passed, has_required_failure)`` for the displayed check keys.

    A key missing from ``results`` counts as a failure: a check that never ran (e.g. the
    check thread died early) must read as failed, never as passed — otherwise the wizard
    shows red rows while still reporting "ready to use" and enabling Get Started."""
    passed = {key: results.get(key, (False, ""))[0] for key in keys}
    all_passed = all(passed.values())
    has_required_failure = any(
        not ok and key not in WARNING_ONLY_CHECKS and not (key == "driver" and ok is None)
        for key, ok in passed.items()
    )
    return all_passed, has_required_failure


def run_system_checks(results: dict[str, tuple[bool | None, str]]) -> None:
    """Fill ``results`` one check at a time, so a caller that stops waiting still sees the finished ones."""
    results["ascii_path"] = system_requirements.check_ascii_install_path()
    results["ffprobe"] = check_ffprobe()
    results["driver"] = system_requirements.check_gpu_driver_version()
    if results["driver"][0] is False:
        results["gpu"] = results["cuda"] = results["driver"]
    else:
        results["gpu"] = check_gpu()
        results["cuda"] = check_cuda()
    if os.name == "nt":
        results["sysmem"] = system_requirements.check_windows_nvidia_sysmem_fallback_policy()
    from jasna.accelerator import is_amd_device

    if is_amd_device():
        results["amf"] = check_amf()


def start_system_checks(results: dict[str, tuple[bool | None, str]]) -> threading.Thread:
    """Run ``run_system_checks(results)`` on a daemon thread."""
    thread = threading.Thread(target=run_system_checks, args=(results,), daemon=True)
    thread.start()
    return thread


def warm_up_cuda() -> tuple[AcceleratorVendor, bool, bool]:
    """Import torch and create the CUDA context. Run off the UI thread after the window
    paints, so the window shows first and the first job skips cold torch/CUDA init.
    An unsupported GPU gets no kernel launch; the system check reports it instead.
    Returns (GPU vendor, Blackwell GPU, frame generation support) for the GPU-specific options;
    CPU without a supported GPU."""
    from jasna.accelerator import supports_frame_generation, supports_nvfp4, vendor_for_device

    if vendor_for_device() is AcceleratorVendor.AMD:
        driver_ok, _ = system_requirements.check_gpu_driver_version()
        if driver_ok is False:
            return AcceleratorVendor.CPU, False, False
    gpu_ok, _ = system_requirements.check_supported_gpu()
    if not gpu_ok:
        return AcceleratorVendor.CPU, False, False
    import torch

    torch.zeros(1, device="cuda")
    vendor = vendor_for_device()
    return vendor, supports_nvfp4("cuda:0"), supports_frame_generation("cuda:0")


def start_cuda_warmup(on_support: Callable[[AcceleratorVendor, bool, bool], None]) -> threading.Thread:
    """Run ``warm_up_cuda`` on the "cuda-warmup" thread, then ``on_support(vendor, blackwell, frame_generation)``
    on that thread; a failed warm-up reports CPU."""

    def warmed_up(support, error):
        if error is not None:
            logger.warning("CUDA warm-up failed", exc_info=error)
            support = AcceleratorVendor.CPU, False, False
        on_support(*support)

    return run_in_background(warm_up_cuda, warmed_up, name="cuda-warmup")


def check_amf() -> tuple[bool, str]:
    ok, info = os_utils.check_amd_amf_runtime()
    if ok:
        return True, info
    logger.warning("AMD AMF runtime did not load: %s", info)
    return False, t("wizard_amf_missing_windows" if sys.platform == "win32" else "wizard_amf_missing_linux")


def check_ffprobe() -> tuple[bool, str]:
    status = system_requirements.ffprobe_status()
    if status.path is None:
        return False, t("wizard_not_found")
    if not status.runs:
        return False, t("wizard_not_callable", path=status.path)
    if status.major is None:
        return False, t("wizard_found_no_major", path=status.path)
    if not status.ok:
        return False, t("wizard_found_bad_major", path=status.path, major=status.major)
    return True, t("wizard_found_major", path=status.path, major=status.major)


def check_gpu() -> tuple[bool, str]:
    try:
        ok, result = system_requirements.check_supported_gpu()
        if ok:
            return True, result
        if result == "no_cuda":
            return False, t("wizard_no_cuda")
        if result[0] == "arch_unsupported":
            return False, t("wizard_gpu_arch_unsupported", arch=result[1])
        if result[0] == "datacenter_unsupported":
            return False, t("wizard_gpu_datacenter_unsupported", gpu=result[1])
        _, major, minor = result
        return False, t("wizard_gpu_compute_too_low", major=major, minor=minor)
    except Exception as e:
        logger.warning("GPU check failed", exc_info=True)
        return False, str(e)


def check_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.hip or torch.version.cuda
            if torch.version.hip:
                return True, f"ROCm {version}"
            capability = torch.cuda.get_device_capability(0)
            return True, t(
                "wizard_cuda_version_compute",
                version=version,
                major=capability[0],
                minor=capability[1],
            )
        return False, t("wizard_not_available")
    except Exception as e:
        logger.warning("GPU runtime check failed", exc_info=True)
        return False, str(e)

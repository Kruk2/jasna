"""First-run dependency checks, run off the UI thread by the wizard."""

import logging
import os
import subprocess
from collections.abc import Iterable

from jasna import os_utils
from jasna.gui.locales import t

logger = logging.getLogger(__name__)

WARNING_ONLY_CHECKS = {"sysmem"}


def evaluate_check_results(
    results: dict[str, tuple[bool, str]], keys: Iterable[str]
) -> tuple[bool, bool]:
    """Return ``(all_passed, has_required_failure)`` for the displayed check keys.

    A key missing from ``results`` counts as a failure: a check that never ran (e.g. the
    check thread died early) must read as failed, never as passed — otherwise the wizard
    shows red rows while still reporting "ready to use" and enabling Get Started."""
    passed = {key: results.get(key, (False, ""))[0] for key in keys}
    all_passed = all(passed.values())
    has_required_failure = any(
        not ok and key not in WARNING_ONLY_CHECKS for key, ok in passed.items()
    )
    return all_passed, has_required_failure


def run_system_checks(results: dict[str, tuple[bool, str]]) -> None:
    """Fill ``results`` one check at a time, so a caller that stops waiting still sees the finished ones."""
    results["ascii_path"] = os_utils.check_ascii_install_path()
    results["ffprobe"] = check_ffprobe()
    results["gpu"] = check_gpu()
    results["cuda"] = check_cuda()
    results["driver"] = os_utils.check_gpu_driver_version()
    if os.name == "nt":
        results["sysmem"] = os_utils.check_windows_nvidia_sysmem_fallback_policy()


def check_ffprobe() -> tuple[bool, str]:
    path = os_utils.find_executable("ffprobe")
    if not path:
        return False, t("wizard_not_found")
    completed = subprocess.run(
        [path, "-version"],
        capture_output=True,
        text=True,
        check=False,
        **os_utils.subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        logger.error(
            "ffprobe failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )
        return False, t("wizard_not_callable", path=path)
    try:
        major = os_utils._parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
    except ValueError:
        return False, t("wizard_found_no_major", path=path)
    if major != 8:
        return False, t("wizard_found_bad_major", path=path, major=major)
    return True, t("wizard_found_major", path=path, major=major)


def check_gpu() -> tuple[bool, str]:
    try:
        ok, result = os_utils.check_supported_gpu()
        if ok:
            return True, result
        if result == "no_cuda":
            return False, t("wizard_no_cuda")
        _, major, minor = result
        return False, t("wizard_gpu_compute_too_low", major=major, minor=minor)
    except Exception as e:
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
        return False, str(e)

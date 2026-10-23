import sys
import types

import pytest

from jasna import startup_timing
from jasna.accelerator import AcceleratorVendor
from jasna.gui.system_checks import evaluate_check_results


def test_elapsed_ms_is_monotonic_nonnegative():
    first = startup_timing.elapsed_ms()
    second = startup_timing.elapsed_ms()
    assert first >= 0.0
    assert second >= first


def test_evaluate_all_passed():
    results = {"ascii_path": (True, ""), "gpu": (True, ""), "sysmem": (True, "")}
    all_passed, required_failure = evaluate_check_results(results, results.keys())
    assert all_passed is True
    assert required_failure is False


def test_evaluate_missing_check_counts_as_required_failure():
    # gpu/cuda never ran (e.g. check thread died) -> must read as failure, never "ready".
    results = {"ascii_path": (True, "")}
    all_passed, required_failure = evaluate_check_results(
        results, ["ascii_path", "gpu", "cuda"]
    )
    assert all_passed is False
    assert required_failure is True


def test_evaluate_sysmem_only_failure_is_warning_not_required():
    results = {"gpu": (True, ""), "sysmem": (False, "")}
    all_passed, required_failure = evaluate_check_results(results, results.keys())
    assert all_passed is False  # a warning still means "not all passed"
    assert required_failure is False  # but sysmem is warning-only, not blocking


def _warm_up_with_gpu_check(monkeypatch, gpu_check_result, *, vendor=AcceleratorVendor.NVIDIA, major=8) -> tuple[dict, tuple]:
    from jasna import accelerator, os_utils

    calls = {}
    fake_torch = types.SimpleNamespace(
        zeros=lambda *a, **k: calls.setdefault("device", k.get("device")),
        cuda=types.SimpleNamespace(get_device_capability=lambda _index: (major, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "torch", fake_torch)
    monkeypatch.setattr(os_utils, "check_supported_gpu", lambda: gpu_check_result)
    monkeypatch.setattr(accelerator, "vendor_for_device", lambda device=None: vendor)
    from jasna.gui.app import _warm_up_cuda

    return calls, _warm_up_cuda()


def test_warm_up_cuda_inits_context_when_gpu_supported(monkeypatch):
    calls, support = _warm_up_with_gpu_check(monkeypatch, (True, "RTX 4090"))
    assert calls["device"] == "cuda"
    assert support == (AcceleratorVendor.NVIDIA, False)


@pytest.mark.parametrize(
    ("vendor", "major", "support"),
    [
        (AcceleratorVendor.NVIDIA, 12, (AcceleratorVendor.NVIDIA, True)),
        (AcceleratorVendor.AMD, 12, (AcceleratorVendor.AMD, False)),
    ],
)
def test_warm_up_cuda_reports_ltx_gpu_support(monkeypatch, vendor, major, support):
    assert _warm_up_with_gpu_check(monkeypatch, (True, "GPU"), vendor=vendor, major=major)[1] == support


@pytest.mark.parametrize(
    "gpu_check_result",
    [(False, "no_cuda"), (False, ("arch_unsupported", "gfx1103"))],
)
def test_warm_up_cuda_launches_no_kernel_on_unsupported_gpu(monkeypatch, gpu_check_result):
    calls, support = _warm_up_with_gpu_check(monkeypatch, gpu_check_result)
    assert calls == {}
    assert support == (AcceleratorVendor.CPU, False)


@pytest.mark.parametrize("loaded", [True, False])
def test_startup_warns_only_when_windhawk_is_loaded(monkeypatch, loaded):
    from tkinter import messagebox

    from jasna import os_utils
    from jasna.gui import app as app_module
    from jasna.gui.locales import t

    monkeypatch.setattr(os_utils, "windows_module_loaded", lambda name: loaded and name == "windhawk.dll")
    shown = []
    monkeypatch.setattr(messagebox, "showwarning", lambda title, message, parent: shown.append((title, message)))
    logged = []
    fake_app = types.SimpleNamespace(_log_panel=types.SimpleNamespace(warning=logged.append))

    app_module.JasnaApp._warn_if_windhawk_loaded(fake_app)

    expected = [(t("windhawk_warning_title"), t("windhawk_warning_body"))] if loaded else []
    assert shown == expected
    assert logged == [message for _title, message in expected]


@pytest.mark.parametrize("hard_disk", [True, False])
def test_startup_warns_when_installed_on_a_hard_disk(monkeypatch, hard_disk):
    from jasna import os_utils
    from jasna.gui import app as app_module
    from jasna.gui.locales import t

    monkeypatch.setattr(os_utils, "install_on_hard_disk", lambda: hard_disk)
    logged = []
    fake_app = types.SimpleNamespace(_log_panel=types.SimpleNamespace(warning=logged.append))

    app_module.JasnaApp._warn_if_installed_on_hard_disk(fake_app)

    assert logged == ([t("hard_disk_warning")] if hard_disk else [])


def test_evaluate_amf_failure_blocks():
    assert evaluate_check_results({"gpu": (True, ""), "amf": (False, "")}, ["gpu", "amf"]) == (False, True)


@pytest.mark.parametrize("amd", [True, False])
def test_system_checks_check_amf_only_on_amd(monkeypatch, amd):
    from jasna import accelerator, os_utils
    from jasna.gui import system_checks

    for name in ("check_ascii_install_path", "check_gpu_driver_version", "check_windows_nvidia_sysmem_fallback_policy"):
        monkeypatch.setattr(os_utils, name, lambda: (True, ""))
    for name in ("check_ffprobe", "check_gpu", "check_cuda"):
        monkeypatch.setattr(system_checks, name, lambda: (True, ""))
    monkeypatch.setattr(os_utils, "check_amd_amf_runtime", lambda: (True, "libamfrt64.so.1"))
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: amd)

    results = {}
    system_checks.run_system_checks(results)

    assert ("amf" in results) is amd


@pytest.mark.parametrize(
    "platform,message",
    [
        ("linux", "Not installed. Install it with AMD's AMF installer."),
        ("win32", "Not found. Update your AMD graphics driver."),
    ],
)
def test_missing_amf_runtime_tells_the_user_what_to_do(monkeypatch, platform, message):
    from jasna import os_utils
    from jasna.gui import system_checks
    from jasna.gui.locales import get_locale

    monkeypatch.setattr(get_locale(), "_current_lang", "en")
    monkeypatch.setattr(system_checks.sys, "platform", platform)
    monkeypatch.setattr(os_utils, "check_amd_amf_runtime", lambda: (False, "cannot open"))

    assert system_checks.check_amf() == (False, message)

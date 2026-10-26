import sys
import threading
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


@pytest.mark.parametrize(
    ("results", "checks", "expected"),
    [
        ({"ascii_path": (True, ""), "gpu": (True, ""), "sysmem": (True, "")}, None, (True, False)),
        # A check that never ran (the check thread died) must read as a failure, never "ready".
        ({"ascii_path": (True, "")}, ["ascii_path", "gpu", "cuda"], (False, True)),
        ({"gpu": (True, ""), "sysmem": (False, "")}, None, (False, False)),
        ({"gpu": (True, ""), "amf": (False, "")}, None, (False, True)),
    ],
    ids=["all-passed", "missing-check", "sysmem-warning-only", "amf-blocks"],
)
def test_evaluate_check_results(results, checks, expected):
    assert evaluate_check_results(results, results.keys() if checks is None else checks) == expected


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


def _logged_warnings(app) -> list[str]:
    return [message for _, level, message in app._log_panel._entries if level == "WARNING"]


@pytest.mark.parametrize("loaded", [True, False])
def test_startup_warns_only_when_windhawk_is_loaded(jasna_app, monkeypatch, loaded):
    from tkinter import messagebox

    from jasna import os_utils
    from jasna.gui.locales import t

    monkeypatch.setattr(os_utils, "windows_module_loaded", lambda name: loaded and name == "windhawk.dll")
    shown = []
    monkeypatch.setattr(messagebox, "showwarning", lambda title, message, parent: shown.append((title, message)))

    jasna_app._warn_if_windhawk_loaded()

    expected = [(t("windhawk_warning_title"), t("windhawk_warning_body"))] if loaded else []
    assert shown == expected
    assert _logged_warnings(jasna_app) == [message for _title, message in expected]


@pytest.mark.parametrize("hard_disk", [True, False])
def test_startup_warns_when_installed_on_a_hard_disk(jasna_app, monkeypatch, hard_disk):
    from jasna import os_utils
    from jasna.gui.locales import t

    monkeypatch.setattr(os_utils, "install_on_hard_disk", lambda: hard_disk)

    jasna_app._warn_if_installed_on_hard_disk()

    assert _logged_warnings(jasna_app) == ([t("hard_disk_warning")] if hard_disk else [])


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
def test_missing_amf_runtime_tells_the_user_what_to_do(monkeypatch, gui_language, platform, message):
    from jasna import os_utils
    from jasna.gui import system_checks

    gui_language("en")
    monkeypatch.setattr(system_checks.sys, "platform", platform)
    monkeypatch.setattr(os_utils, "check_amd_amf_runtime", lambda: (False, "cannot open"))

    assert system_checks.check_amf() == (False, message)


@pytest.fixture
def wizard_factory(tk_root, monkeypatch):
    """Build the first-run wizard with system checks that run ``checks`` instead of probing the machine."""
    from jasna.gui import wizard as wizard_module

    def build(checks=lambda results: None, on_complete=None):
        monkeypatch.setattr(wizard_module, "run_system_checks", checks)
        return wizard_module.FirstRunWizard(tk_root, on_complete=on_complete)

    return build


def test_wizard_close_button_releases_the_user(wizard_factory) -> None:
    completed: list[tuple[bool, bool]] = []
    wizard = wizard_factory(on_complete=lambda can_continue, all_passed: completed.append((can_continue, all_passed)))

    wizard.tk.call(wizard.protocol("WM_DELETE_WINDOW"))

    assert not wizard.winfo_exists()
    assert completed == [(False, False)]


def test_wizard_stops_waiting_on_a_wedged_check_thread(wizard_factory, monkeypatch) -> None:
    from jasna.gui import wizard as wizard_module
    from jasna.gui.locales import t

    monkeypatch.setattr(wizard_module, "_CHECKS_TIMEOUT_SECONDS", 0.0)
    release = threading.Event()
    wizard = wizard_factory(checks=lambda results: release.wait())
    try:
        assert wizard._continue_btn.cget("state") == "disabled"

        wizard._start_checks_in_background()
        wizard._poll_checks_thread()

        assert wizard._continue_btn.cget("state") == "normal"
        assert wizard._continue_btn.cget("text") == t("btn_exit")
    finally:
        release.set()

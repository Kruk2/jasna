import threading

import pytest

from jasna import startup_timing
from jasna.accelerator import AcceleratorVendor
from jasna.gui.system_checks import evaluate_check_results


@pytest.mark.parametrize(("checked_at", "line"), [
    (104.0, "[timing] app startup: 10.0s to the first job (process start 1.5s, startup check 2.5s, until the job 6.0s)"),
    (None, "[timing] app startup: 10.0s to the first job (process start 1.5s, until the job 8.5s)"),
], ids=["release", "source"])
def test_app_startup_splits_the_time_to_the_first_job(checked_at, line):
    assert startup_timing.app_startup_line(100.0, 101.5, checked_at, 110.0) == line


@pytest.mark.parametrize(
    ("results", "checks", "expected"),
    [
        ({"ascii_path": (True, ""), "gpu": (True, ""), "sysmem": (True, "")}, None, (True, False)),
        # A check that never ran (the check thread died) must read as a failure, never "ready".
        ({"ascii_path": (True, "")}, ["ascii_path", "gpu", "cuda"], (False, True)),
        ({"gpu": (True, ""), "sysmem": (False, "")}, None, (False, False)),
        ({"gpu": (True, ""), "amf": (False, "")}, None, (False, True)),
        ({"gpu": (True, ""), "driver": (None, "unknown driver")}, None, (False, False)),
        ({"gpu": (True, ""), "driver": (False, "old driver")}, None, (False, True)),
    ],
    ids=["all-passed", "missing-check", "sysmem-warning-only", "amf-blocks", "unknown-driver-warns", "old-driver-blocks"],
)
def test_evaluate_check_results(results, checks, expected):
    assert evaluate_check_results(results, results.keys() if checks is None else checks) == expected


NVIDIA, AMD, CPU = AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD, AcceleratorVendor.CPU


@pytest.mark.parametrize(
    ("vendor", "major", "gpu_check", "driver_ok", "support", "launched"),
    [
        (NVIDIA, 8, (True, "RTX 3090"), True, (NVIDIA, False, False), ["cuda"]),
        (NVIDIA, 9, (True, "GPU"), True, (NVIDIA, False, True), ["cuda"]),
        (NVIDIA, 12, (True, "GPU"), True, (NVIDIA, True, True), ["cuda"]),
        (AMD, 12, (True, "GPU"), True, (AMD, False, False), ["cuda"]),
        (NVIDIA, 12, (False, "no_cuda"), True, (CPU, False, False), []),
        (NVIDIA, 12, (False, ("arch_unsupported", "gfx1103")), True, (CPU, False, False), []),
        (AMD, 12, (True, "AMD GPU"), False, (CPU, False, False), []),
    ],
    ids=["ampere", "hopper", "blackwell", "amd", "no-cuda", "unsupported-arch", "amd-old-driver"],
)
def test_warm_up_reports_gpu_support_and_launches_a_kernel_only_on_a_usable_gpu(
    request, monkeypatch, cuda_capability, vendor, major, gpu_check, driver_ok, support, launched
):
    import torch

    from jasna import system_requirements
    from jasna.gui.system_checks import warm_up_cuda

    cuda_capability((major, 0))
    if vendor is AMD:
        request.getfixturevalue("amd_gpu")()
    devices: list[str] = []
    monkeypatch.setattr(torch, "zeros", lambda *_a, device=None, **_k: devices.append(device))
    monkeypatch.setattr(system_requirements, "check_supported_gpu", lambda: gpu_check)
    monkeypatch.setattr(system_requirements, "check_gpu_driver_version", lambda: (driver_ok, "driver"))

    assert warm_up_cuda() == support
    assert devices == launched


def _logged_warnings(app) -> list[str]:
    return [message for _, level, message in app._log_panel._entries if level == "WARNING"]


@pytest.mark.parametrize("rerun", [False, True])
@pytest.mark.parametrize("all_passed", [False, True])
def test_continuing_with_warnings_does_not_record_a_pass(jasna_app, rerun, all_passed):
    from jasna import __version__
    from jasna.gui.locales import t

    previous = jasna_app._preset_manager.get_system_check_passed_version()
    jasna_app._on_system_check_complete(True, all_passed, quit_when_blocked=not rerun)

    expected = __version__ if all_passed else previous
    assert jasna_app._preset_manager.get_system_check_passed_version() == expected
    assert _logged_warnings(jasna_app) == ([] if all_passed else [t("wizard_warnings_only")])


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
    from jasna import system_requirements
    from jasna.gui.locales import t

    monkeypatch.setattr(system_requirements, "install_on_hard_disk", lambda: hard_disk)

    jasna_app._warn_if_installed_on_hard_disk()

    assert _logged_warnings(jasna_app) == ([t("hard_disk_warning")] if hard_disk else [])


@pytest.mark.parametrize("leftover", ["c10.dll", None])
def test_startup_warns_when_an_older_version_left_dlls_in_the_folder(jasna_app, monkeypatch, leftover):
    from tkinter import messagebox

    from jasna import system_requirements
    from jasna.gui.locales import t

    monkeypatch.setattr(system_requirements, "leftover_install_dll", lambda: leftover)
    shown = []
    monkeypatch.setattr(messagebox, "showwarning", lambda title, message, parent: shown.append((title, message)))

    jasna_app._warn_if_older_version_files_left()

    expected = [(t("old_files_warning_title"), t("old_files_warning_body", file=leftover))] if leftover else []
    assert shown == expected
    assert _logged_warnings(jasna_app) == [message for _title, message in expected]


@pytest.mark.parametrize("os_name", ["nt", "posix"])
@pytest.mark.parametrize("amd", [True, False])
def test_system_checks_run_exactly_the_shown_checks(monkeypatch, amd, os_name):
    import os

    from jasna import accelerator, os_utils, system_requirements
    from jasna.gui import system_checks

    monkeypatch.setattr(os, "name", os_name)

    for name in ("check_ascii_install_path", "check_gpu_driver_version", "check_windows_nvidia_sysmem_fallback_policy"):
        monkeypatch.setattr(system_requirements, name, lambda: (True, ""))
    for name in ("check_ffprobe", "check_gpu", "check_cuda"):
        monkeypatch.setattr(system_checks, name, lambda: (True, ""))
    monkeypatch.setattr(os_utils, "check_amd_amf_runtime", lambda: (True, "libamfrt64.so.1"))
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: amd)

    results = {}
    system_checks.run_system_checks(results)

    assert ("amf" in results) is amd
    assert ("sysmem" in results) is (os_name == "nt")
    assert sorted(results) == sorted(system_checks.system_check_keys())


def test_system_checks_do_not_probe_the_gpu_with_an_old_driver(monkeypatch):
    from jasna import accelerator, system_requirements
    from jasna.gui import system_checks

    monkeypatch.setattr(system_requirements, "check_ascii_install_path", lambda: (True, ""))
    monkeypatch.setattr(system_requirements, "check_windows_nvidia_sysmem_fallback_policy", lambda: (True, ""))
    monkeypatch.setattr(system_checks, "check_ffprobe", lambda: (True, ""))
    monkeypatch.setattr(system_requirements, "check_gpu_driver_version", lambda: (False, "old driver"))
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: False)
    def unavailable_probe():
        raise AssertionError("probed an incompatible driver")
    monkeypatch.setattr(system_checks, "check_gpu", unavailable_probe)
    monkeypatch.setattr(system_checks, "check_cuda", unavailable_probe)

    results = {}
    system_checks.run_system_checks(results)

    assert results["gpu"][0] is False
    assert results["cuda"][0] is False
    assert evaluate_check_results(results, results) == (False, True)


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
    from jasna.gui import system_checks, wizard as wizard_module

    def build(checks=lambda results: None, on_complete=None):
        monkeypatch.setattr(system_checks, "run_system_checks", checks)
        return wizard_module.FirstRunWizard(tk_root, on_complete=on_complete)

    return build


@pytest.mark.parametrize("driver_ok,can_continue", [(None, True), (False, False)])
def test_wizard_distinguishes_an_unverified_driver_from_an_old_one(wizard_factory, driver_ok, can_continue):
    from jasna.gui.locales import t
    from jasna.gui.theme import Colors

    def checks(results):
        results.update({key: (True, "ok") for key in ("ascii_path", "ffprobe", "gpu", "cuda", "sysmem", "amf")})
        results["driver"] = driver_ok, "AMD driver"

    completed = []
    wizard = wizard_factory(checks=checks, on_complete=lambda allowed, passed: completed.append((allowed, passed)))
    wizard._start_checks_in_background()
    wizard._checks_thread.join()
    wizard._poll_checks_thread()

    label = wizard._check_labels["driver"][0]
    assert label.cget("text") == ("⚠" if can_continue else "✕")
    assert label.cget("text_color") == (Colors.STATUS_WARNING if can_continue else Colors.STATUS_ERROR)
    assert wizard._continue_btn.cget("text") == t("btn_get_started" if can_continue else "btn_exit")
    wizard._continue_btn.invoke()
    assert completed == [(can_continue, False)]


def test_wizard_fits_a_long_check_result(wizard_factory, monkeypatch, show_window) -> None:
    from jasna import accelerator

    monkeypatch.setattr(accelerator, "is_amd_device", lambda: True)
    long_info = "ROCm 7.1.52802-26aae437f6; in-kernel amdgpu (Linux 6.17.0-1012-generic-64k-and-a-long-suffix)"

    def checks(results):
        results.update({key: (True, "ok") for key in ("ascii_path", "ffprobe", "gpu", "cuda", "sysmem", "amf")})
        results["driver"] = True, long_info

    wizard = wizard_factory(checks=checks)
    wizard._start_checks_in_background()
    wizard._checks_thread.join()
    wizard._poll_checks_thread()
    show_window(wizard)

    info_label = wizard._check_labels["driver"][1]
    assert info_label.winfo_width() >= info_label.winfo_reqwidth()
    amf_row = wizard._check_labels["amf"][0].master
    assert amf_row.winfo_rooty() + amf_row.winfo_height() <= wizard.winfo_rooty() + wizard.winfo_height()


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


@pytest.mark.parametrize(("info", "error", "key"), [
    ((True, "offline"), None, "license_chip_offline"),
    ((True, "online"), None, "license_chip_online"),
    ((True, None), None, "license_chip_active"),
    ((False, None), None, "license_chip_inactive"),
    (None, OSError("unreadable"), "license_chip_inactive"),
])
def test_license_chip_names_how_jasna_checks_the_license(jasna_app, info, error, key):
    from jasna.gui.locales import t

    jasna_app._header_bar._license_chip_ready(info, error)
    assert jasna_app._header_bar._license_chip.cget("text") == t(key)


@pytest.fixture
def notice_checks(monkeypatch):
    """Record each startup check for the unofficial-copy note instead of running it."""
    from jasna.gui import activation_worker

    started = []
    monkeypatch.setattr(activation_worker, "start_unofficial_notice_check", started.append)
    return started


def unofficial_notices(window):
    from jasna.gui.components import XianyuQrDialog

    return [child for child in window.winfo_children() if isinstance(child, XianyuQrDialog)]


@pytest.fixture
def ui_language(request, gui_language):
    """The GUI language (``request.param``) chosen before the window is built."""
    gui_language(request.param)


@pytest.mark.parametrize(("ui_language", "checks"), [("zh", 1), ("en", 0), ("ja", 0)], indirect=["ui_language"])
def test_only_the_chinese_ui_checks_for_the_unofficial_copy_note_at_startup(ui_language, notice_checks, jasna_app, checks):
    assert len(notice_checks) == checks


@pytest.mark.parametrize("ui_language", ["zh"], indirect=True)
@pytest.mark.parametrize(("due", "error", "shown"), [(True, None, 1), (False, None, 0), (None, OSError("unreadable"), 0)])
def test_the_startup_check_shows_the_note_only_when_it_is_due(ui_language, notice_checks, jasna_app, due, error, shown):
    notice_checks[0](due, error)
    jasna_app._main_thread._run_pending()
    assert len(unofficial_notices(jasna_app)) == shown


@pytest.fixture
def shown_errors(monkeypatch):
    from tkinter import messagebox

    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, message, **_kwargs: errors.append((title, message)))
    return errors


def test_a_job_refused_for_an_unofficial_activation_shows_the_note(jasna_app, shown_errors):
    from jasna.gui.models import JobStatus
    from jasna.gui.processor import ProgressUpdate

    jasna_app._handle_progress(ProgressUpdate(job_id=1, status=JobStatus.ERROR, message="", license_problem="expired"))
    assert unofficial_notices(jasna_app) == []
    jasna_app._handle_progress(ProgressUpdate(job_id=1, status=JobStatus.ERROR, message="", license_problem="unofficial_key"))
    assert len(unofficial_notices(jasna_app)) == 1


def test_a_failed_license_check_tells_the_user_why(jasna_app, shown_errors):
    from jasna.gui.locales import t
    from jasna.gui.models import JobStatus
    from jasna.gui.processor import ProgressUpdate

    jasna_app._handle_progress(ProgressUpdate(
        job_id=1, status=JobStatus.ERROR, message=t("license_problem_offline"), license_problem="offline",
    ))
    assert shown_errors == [(t("license_problem_title"), t("license_problem_offline"))]

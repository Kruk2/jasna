from __future__ import annotations

import customtkinter as ctk

from jasna.gui import wizard
from jasna.gui.locales import t


def _label_texts(widget) -> list[str]:
    texts = [widget.cget("text")] if isinstance(widget, ctk.CTkLabel) else []
    for child in widget.winfo_children():
        texts += _label_texts(child)
    return texts


def test_wizard_names_the_official_sellers(tk_root, monkeypatch):
    monkeypatch.setattr(wizard, "run_system_checks", lambda results: None)

    dialog = wizard.FirstRunWizard(tk_root)

    assert t("official_sellers_notice") in _label_texts(dialog)


def test_license_dialog_names_the_official_sellers(tk_root, settle_activation, user_config_dir):
    from jasna.gui.components import LicenseDialog

    dialog = LicenseDialog(tk_root, on_activated=lambda: None)
    settle_activation(dialog)

    assert t("license_official_sellers") in _label_texts(dialog)


def test_wizard_blocks_amd_without_amf_and_links_the_fix(tk_root, monkeypatch):
    from jasna import accelerator, os_utils
    from jasna.gui import system_checks

    for name in ("check_ascii_install_path", "check_gpu_driver_version", "check_windows_nvidia_sysmem_fallback_policy"):
        monkeypatch.setattr(os_utils, name, lambda: (True, ""))
    for name in ("check_ffprobe", "check_gpu", "check_cuda"):
        monkeypatch.setattr(system_checks, name, lambda: (True, ""))
    monkeypatch.setattr(system_checks, "check_amf", lambda: (False, t("wizard_amf_missing_linux")))
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: True)
    dialog = wizard.FirstRunWizard(tk_root)
    assert t("wizard_check_amf") in _label_texts(dialog)

    dialog._start_checks_in_background()
    dialog._checks_thread.join()
    dialog._poll_checks_thread()

    assert dialog._continue_btn.cget("text") == t("btn_exit")
    assert dialog._check_labels["amf"][2].winfo_manager() == "pack"


def test_chinese_license_dialog_has_the_xianyu_button_and_stays_modal_after_the_qr(tk_root, settle_activation, gui_language):
    from jasna.gui.components import LicenseDialog, XianyuButton, XianyuQrDialog

    gui_language("zh")
    dialog = LicenseDialog(tk_root, on_activated=lambda: None)
    settle_activation(dialog)
    [button] = [child for child in dialog.winfo_children()[0].winfo_children() if isinstance(child, XianyuButton)]

    button.invoke()
    [qr] = [child for child in dialog.winfo_children() if isinstance(child, XianyuQrDialog)]
    qr.destroy()

    assert dialog.grab_current() == dialog

from __future__ import annotations

import customtkinter as ctk
import pytest

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


@pytest.mark.parametrize("error_name, message_key", [
    ("ForgedLicenseError", "license_forged"),
    ("RetiredLicenseError", "license_retired"),
    ("LicenseError", "license_invalid"),
    ("MalformedLicenseError", "license_malformed"),
])
def test_license_dialog_explains_rejected_keys(tk_root, monkeypatch, settle_activation, error_name, message_key):
    license_store = pytest.importorskip("jasna.protection.license_store")
    from jasna import protection
    from jasna.gui.components import LicenseDialog

    def reject(key):
        raise getattr(protection, error_name)("english text")

    monkeypatch.setattr(license_store, "load_license", lambda: None)
    monkeypatch.setattr(license_store, "is_licensed", lambda: False)
    monkeypatch.setattr(license_store, "_decode", lambda key: (b"JM1-" + b"1" * 64, {}))
    monkeypatch.setattr(protection.signing, "license_identity", lambda proof: proof.decode())
    monkeypatch.setattr(license_store, "set_license", reject)
    dialog = LicenseDialog(tk_root, on_activated=lambda: None)
    assert t("license_official_sellers") in _label_texts(dialog)
    settle_activation(dialog)

    dialog._activate_button.invoke()
    settle_activation(dialog)

    expected = t(message_key) + ("\nenglish text" if error_name == "LicenseError" else "")
    assert dialog._status.cget("text") == expected


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

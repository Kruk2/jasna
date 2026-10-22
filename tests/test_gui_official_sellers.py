from __future__ import annotations

from tkinter import TclError
import time

import customtkinter as ctk
import pytest

from jasna.gui import wizard
from jasna.gui.locales import t


def _root() -> ctk.CTk:
    try:
        return ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")


def _label_texts(widget) -> list[str]:
    texts = [widget.cget("text")] if isinstance(widget, ctk.CTkLabel) else []
    for child in widget.winfo_children():
        texts += _label_texts(child)
    return texts


def test_wizard_names_the_official_sellers(monkeypatch):
    monkeypatch.setattr(wizard, "run_system_checks", lambda results: None)
    root = _root()
    try:
        dialog = wizard.FirstRunWizard(root)
        assert t("official_sellers_notice") in _label_texts(dialog)
    finally:
        root.destroy()


@pytest.mark.parametrize("error_name, message_key", [
    ("ForgedLicenseError", "license_forged"),
    ("RetiredLicenseError", "license_retired"),
    ("LicenseError", "license_invalid"),
    ("MalformedLicenseError", "license_malformed"),
])
def test_license_dialog_explains_rejected_keys(monkeypatch, error_name, message_key):
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
    root = _root()
    try:
        dialog = LicenseDialog(root, on_activated=lambda: None)
        assert t("license_official_sellers") in _label_texts(dialog)
        deadline = time.monotonic() + 3
        while dialog._activate_button.cget("state") != "normal" and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        dialog._activate_button.invoke()
        expected = t(message_key)
        if error_name == "LicenseError":
            expected += "\nenglish text"
        deadline = time.monotonic() + 3
        while dialog._status.cget("text") != expected and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert dialog._status.cget("text") == expected
    finally:
        root.destroy()


def test_wizard_blocks_amd_without_amf_and_links_the_fix(monkeypatch):
    from jasna import accelerator

    monkeypatch.setattr(wizard, "run_system_checks", lambda results: None)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: True)
    root = _root()
    try:
        dialog = wizard.FirstRunWizard(root)
        assert t("wizard_check_amf") in _label_texts(dialog)
        dialog._check_results.update({key: (True, "") for key in dialog._check_labels})
        dialog._check_results["amf"] = (False, t("wizard_amf_missing_linux"))
        dialog._apply_check_results_to_ui()
        assert dialog._has_required_failure
        assert dialog._check_labels["amf"][2].winfo_manager() == "pack"
    finally:
        root.destroy()

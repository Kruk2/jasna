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
    protection = pytest.importorskip("jasna.protection")
    from jasna.gui.components import LicenseDialog
    from jasna.protection import license_store

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

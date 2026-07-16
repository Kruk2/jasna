"""The two support entry points (Buy Me a Coffee + Unifans) point at the right pages."""

from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.components import (
    BMC_URL,
    UNIFANS_URL,
    BuyMeCoffeeButton,
    UnifansButton,
    _SupportButton,
)


def test_support_urls():
    assert BMC_URL == "https://buymeacoffee.com/Kruk2"
    assert UNIFANS_URL == "https://app.unifans.io/c/kruk2"


def test_both_buttons_share_support_base():
    assert issubclass(BuyMeCoffeeButton, _SupportButton)
    assert issubclass(UnifansButton, _SupportButton)


def test_support_buttons_do_not_depend_on_emoji_fonts():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        labels = {
            BuyMeCoffeeButton(root, compact=False).cget("text"),
            UnifansButton(root, compact=False).cget("text"),
        }
        assert all(not any(icon in label for icon in "☕🚀💜") for label in labels)
    finally:
        root.destroy()

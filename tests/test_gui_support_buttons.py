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
            BuyMeCoffeeButton(root, width=100, height=28).cget("text"),
            UnifansButton(root, width=110, height=28).cget("text"),
        }
        assert all(not any(icon in label for icon in "☕🚀💜") for label in labels)
    finally:
        root.destroy()


def test_support_button_hover_scales_from_its_own_size():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        button = BuyMeCoffeeButton(root, width=140, height=48)
        button._on_enter()
        assert (button.cget("width"), button.cget("height")) == (147, 50)
        button._on_leave()
        assert (button.cget("width"), button.cget("height")) == (140, 48)
    finally:
        root.destroy()

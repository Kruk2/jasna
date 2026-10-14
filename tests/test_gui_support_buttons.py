"""The support entry points (Buy Me a Coffee, Unifans, and Xianyu for Chinese) point at the right pages."""

from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.components import (
    BMC_URL,
    UNIFANS_URL,
    BuyMeCoffeeButton,
    UnifansButton,
    XianyuButton,
    XianyuQrDialog,
    _SupportButton,
    shows_xianyu_seller,
)
from jasna.gui.locales import TRANSLATIONS


def test_support_urls():
    assert BMC_URL == "https://buymeacoffee.com/Kruk2"
    assert UNIFANS_URL == "https://app.unifans.io/c/kruk2"


def test_xianyu_button_shows_the_qr_code():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        XianyuButton(root, width=80, height=28).invoke()
        dialogs = [w for w in root.winfo_children() if isinstance(w, XianyuQrDialog)]
        assert len(dialogs) == 1
        assert dialogs[0].qr_label.cget("image") is not None
    finally:
        root.destroy()


def test_xianyu_seller_shown_only_in_chinese():
    assert shows_xianyu_seller("zh")
    assert not any(shows_xianyu_seller(lang) for lang in ("en", "ja", "ko", "th"))


def test_official_seller_texts_name_xianyu_only_in_chinese_and_english():
    for lang, strings in TRANSLATIONS.items():
        names_xianyu = lang in ("zh", "en")
        assert ("闲鱼" in strings["license_official_sellers"]) is names_xianyu, lang
        assert ("闲鱼" in strings["license_forged"]) is names_xianyu, lang


def test_all_buttons_share_support_base():
    assert issubclass(BuyMeCoffeeButton, _SupportButton)
    assert issubclass(UnifansButton, _SupportButton)
    assert issubclass(XianyuButton, _SupportButton)


def test_support_buttons_do_not_depend_on_emoji_fonts():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        labels = {
            BuyMeCoffeeButton(root, width=100, height=28).cget("text"),
            UnifansButton(root, width=110, height=28).cget("text"),
            XianyuButton(root, width=80, height=28).cget("text"),
        }
        assert all(not any(icon in label for icon in "☕🚀💜") for label in labels)
    finally:
        root.destroy()


def test_support_button_runs_its_command_on_a_mouse_click(show_window):
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        clicks = []
        button = _SupportButton(root, "Support", lambda: clicks.append(True), "#ffdd00", "#000000", 140, 48)
        button.pack()
        show_window(root)
        button._canvas.event_generate("<Enter>")
        button._canvas.event_generate("<ButtonRelease-1>")
        root.update()
        assert clicks == [True]
    finally:
        root.destroy()


def test_support_button_hover_scales_from_its_own_size():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        button = BuyMeCoffeeButton(root, width=140, height=48)
        button._grow_on_hover()
        assert (button.cget("width"), button.cget("height")) == (147, 50)
        button._restore_size()
        assert (button.cget("width"), button.cget("height")) == (140, 48)
    finally:
        root.destroy()

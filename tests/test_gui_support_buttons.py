"""The support entry points (Buy Me a Coffee, Unifans, and Xianyu for Chinese) point at the right pages."""

from __future__ import annotations



from jasna.gui.components import (
    BuyMeCoffeeButton,
    UnifansButton,
    XianyuButton,
    XianyuQrDialog,
    _SupportButton,
    shows_xianyu_seller,
)
from jasna.gui.locales import TRANSLATIONS


def test_xianyu_button_shows_the_qr_code(tk_root):
    XianyuButton(tk_root, width=80, height=28).invoke()
    dialogs = [w for w in tk_root.winfo_children() if isinstance(w, XianyuQrDialog)]
    assert len(dialogs) == 1
    assert dialogs[0].qr_label.cget("image") is not None


def test_the_unofficial_copy_note_sits_above_the_same_qr_code(tk_root):
    from jasna.gui.locales import t

    def texts(dialog):
        return [child.cget("text") for child in dialog.winfo_children()[0].winfo_children()]

    plain, note = XianyuQrDialog(tk_root), XianyuQrDialog(tk_root, unofficial_notice=True)
    assert (plain.title(), note.title()) == (t("xianyu_dialog_title"), t("unofficial_notice_title"))
    assert texts(note)[0] == t("unofficial_notice_text") and t("unofficial_notice_text") not in texts(plain)
    assert note.qr_label.cget("image") is not None


def test_xianyu_seller_shown_only_in_chinese():
    assert shows_xianyu_seller("zh")
    assert not any(shows_xianyu_seller(lang) for lang in ("en", "ja", "ko", "th"))


def test_official_seller_text_names_xianyu_only_in_chinese_and_english():
    for lang, strings in TRANSLATIONS.items():
        names_xianyu = lang in ("zh", "en")
        assert ("闲鱼" in strings["license_official_sellers"]) is names_xianyu, lang


def test_support_buttons_do_not_depend_on_emoji_fonts(tk_root):
    labels = {
        BuyMeCoffeeButton(tk_root, width=100, height=28).cget("text"),
        UnifansButton(tk_root, width=110, height=28).cget("text"),
        XianyuButton(tk_root, width=80, height=28).cget("text"),
    }
    assert all(not any(icon in label for icon in "☕🚀💜") for label in labels)


def test_support_button_runs_its_command_on_a_mouse_click(tk_root, show_window):
    clicks = []
    button = _SupportButton(tk_root, "Support", lambda: clicks.append(True), "#ffdd00", "#000000", 140, 48)
    button.pack()
    show_window(tk_root)
    button._canvas.event_generate("<Enter>")
    button._canvas.event_generate("<ButtonRelease-1>")
    tk_root.update()
    assert clicks == [True]


def test_support_button_hover_scales_from_its_own_size(tk_root):
    button = BuyMeCoffeeButton(tk_root, width=140, height=48)
    button._grow_on_hover()
    assert (button.cget("width"), button.cget("height")) == (147, 50)
    button._restore_size()
    assert (button.cget("width"), button.cget("height")) == (140, 48)

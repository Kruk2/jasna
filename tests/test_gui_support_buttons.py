"""The two support entry points (Buy Me a Coffee + Unifans) point at the right pages."""

from __future__ import annotations

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

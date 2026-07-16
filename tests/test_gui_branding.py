from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

from PIL import Image

from jasna.gui import branding


def test_brand_assets_are_valid_and_high_resolution() -> None:
    svg_path = branding.brand_asset_path(branding.LOGO_SVG)
    png_path = branding.brand_asset_path(branding.LOGO_PNG)
    ico_path = branding.brand_asset_path(branding.LOGO_ICO)

    root = ET.parse(svg_path).getroot()
    assert root.attrib["viewBox"] == "0 0 512 512"
    assert root.find("{http://www.w3.org/2000/svg}title") is not None

    with Image.open(png_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (512, 512)

    with Image.open(ico_path) as image:
        assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= image.ico.sizes()


def test_brand_asset_path_uses_executable_directory_when_frozen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "jasna.dist" / "jasna"
    monkeypatch.setattr(branding, "is_frozen", lambda: True)
    monkeypatch.setattr(branding.sys, "executable", str(executable))

    assert branding.brand_asset_path(branding.LOGO_PNG) == (
        executable.parent / "assets" / branding.LOGO_PNG
    )


def test_header_logo_uses_high_resolution_source_at_display_size() -> None:
    logo = branding.create_header_logo()

    assert logo.cget("size") == branding.HEADER_LOGO_SIZE
    assert logo.cget("light_image").size == (512, 512)
    assert logo.cget("dark_image").size == (512, 512)


def test_install_window_icon_sets_default_icon(monkeypatch) -> None:
    window = MagicMock()
    icon = object()
    constructor = MagicMock(return_value=icon)
    monkeypatch.setattr(branding.tk, "PhotoImage", constructor)

    result = branding.install_window_icon(window)

    assert result is icon
    constructor.assert_called_once_with(
        master=window,
        file=str(branding.brand_asset_path(branding.LOGO_PNG)),
    )
    window.iconphoto.assert_called_once_with(True, icon)

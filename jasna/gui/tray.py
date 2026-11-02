"""System tray icon that stands in for the main window while it is hidden."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PIL import Image

from jasna.gui.branding import LOGO_PNG, brand_asset_path
from jasna.gui.locales import t


@dataclass(frozen=True)
class TraySupport:
    available: bool
    has_menu: bool


def tray_support() -> TraySupport:
    """Whether a hidden window can be brought back from a tray icon right now."""
    if sys.platform == "win32":
        return TraySupport(available=True, has_menu=True)
    if sys.platform.startswith("linux"):
        from jasna.gui import status_notifier

        return TraySupport(available=status_notifier.host_running(), has_menu=status_notifier.StatusNotifierItem.HAS_MENU)
    return TraySupport(available=False, has_menu=False)


def _create_icon(image: Image.Image, title: str, on_open, on_quit):
    if sys.platform == "win32":
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(t("tray_open"), lambda: on_open(), default=True),
            pystray.MenuItem(t("tray_quit"), lambda: on_quit()),
        )
        return pystray.Icon("jasna", image, title, menu)
    from jasna.gui.status_notifier import StatusNotifierItem

    return StatusNotifierItem(image, title, on_open)


class TrayIcon:
    """Shown only while the window is hidden. ``on_open``/``on_quit`` run on the tray thread."""

    def __init__(self, on_open, on_quit):
        self._on_open = on_open
        self._on_quit = on_quit
        self._title = "Jasna"
        self._icon = None

    @property
    def shown(self) -> bool:
        return self._icon is not None

    def show(self) -> None:
        if self._icon is not None:
            return
        with Image.open(brand_asset_path(LOGO_PNG)) as logo:
            image = logo.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
        self._icon = _create_icon(image, self._title, self._on_open, self._on_quit)
        self._icon.run_detached()

    def hide(self) -> None:
        if self._icon is None:
            return
        self._icon.stop()
        self._icon = None

    def set_title(self, title: str) -> None:
        self._title = title
        if self._icon is not None:
            self._icon.title = title

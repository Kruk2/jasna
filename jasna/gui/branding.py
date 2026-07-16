"""Application branding assets shared by source and frozen GUI builds."""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk

import customtkinter as ctk
from PIL import Image

from jasna._frozen import is_frozen


LOGO_SVG = "jasna-logo.svg"
LOGO_PNG = "jasna-logo.png"
LOGO_ICO = "jasna-logo.ico"
HEADER_LOGO_SIZE = (28, 28)


def brand_asset_path(filename: str) -> Path:
    """Resolve a brand asset from the repo or beside a frozen executable."""
    if is_frozen():
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "assets" / filename


def create_header_logo() -> ctk.CTkImage:
    """Return a HiDPI-aware header image at its intended display size."""
    with Image.open(brand_asset_path(LOGO_PNG)) as source:
        image = source.convert("RGBA")
    return ctk.CTkImage(
        light_image=image,
        dark_image=image,
        size=HEADER_LOGO_SIZE,
    )


def install_window_icon(window: tk.Tk) -> tk.PhotoImage:
    """Set the PNG as the default icon for this and future Tk windows."""
    icon = tk.PhotoImage(master=window, file=str(brand_asset_path(LOGO_PNG)))
    window.iconphoto(True, icon)
    return icon

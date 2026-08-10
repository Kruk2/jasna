from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from tkinter import messagebox

from jasna.gui.locales import t

logger = logging.getLogger(__name__)


def open_containing_folder(path: Path, *, parent, select_file: bool = False) -> None:
    folder = path.parent
    system = platform.system()
    if system == "Windows":
        command = ["explorer", "/select,", str(path)] if select_file else ["explorer", str(folder)]
    elif system == "Darwin":
        command = ["open", "-R", str(path)] if select_file else ["open", str(folder)]
    else:
        command = ["xdg-open", str(folder)]

    try:
        subprocess.Popen(command)
    except OSError as exc:
        logger.warning("Could not open containing folder %s", folder, exc_info=True)
        messagebox.showerror(
            t("open_containing_folder_failed_title"),
            t("open_containing_folder_failed", message=str(exc)),
            parent=parent,
        )


def open_file(path: Path, *, parent) -> None:
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        logger.warning("Could not open file %s", path, exc_info=True)
        messagebox.showerror(
            t("open_file_failed_title"),
            t("open_file_failed", message=str(exc)),
            parent=parent,
        )

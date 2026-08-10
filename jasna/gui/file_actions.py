from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from tkinter import messagebox

from jasna.gui.locales import t

logger = logging.getLogger(__name__)


def open_containing_folder(path: Path, *, parent) -> None:
    folder = path.parent
    system = platform.system()
    if system == "Windows":
        command = ["explorer", str(folder)]
    elif system == "Darwin":
        command = ["open", str(folder)]
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

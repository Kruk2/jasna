"""The DLSS 5 runtime download, asked for when a job needs it. UI glue: it asks with a message
box and runs the download on a worker thread, reporting back on the main thread."""

from collections.abc import Callable

from jasna.gui.download_worker import start_download
from jasna.gui.locales import t
from jasna.gui.queues import MainThreadCalls


class Dlss5Runtime:
    def __init__(self, main_thread: MainThreadCalls, on_change: Callable[[], None]) -> None:
        self._main_thread = main_thread
        self._on_change = on_change
        self.percent: int | None = None

    @property
    def downloading(self) -> bool:
        return self.percent is not None

    @staticmethod
    def installed() -> bool:
        from jasna.upscale.dlss5 import DLSS5_RUNTIME_FILES, dlss5_runtime_dir

        directory = dlss5_runtime_dir()
        return all((directory / name).is_file() for name in DLSS5_RUNTIME_FILES)

    def ensure(self, *, on_ready: Callable[[], None]) -> None:
        """Ask to download the missing runtime; ``on_ready`` runs after the download succeeds."""
        from tkinter import messagebox

        from jasna.upscale.dlss5 import DLSS5_RELEASE_PAGE, DLSS5_RELEASE_SIZE, download_dlss5_runtime, dlss5_runtime_dir

        if self.downloading:
            return
        size = f"{DLSS5_RELEASE_SIZE / 1e6:.0f} MB"
        if not messagebox.askyesno(t("dlss5_download_title"), t("dlss5_download_confirm", size=size, page=DLSS5_RELEASE_PAGE)):
            return
        self.percent = 0
        self._on_change()
        start_download(
            lambda progress: download_dlss5_runtime(dlss5_runtime_dir(), progress),
            lambda percent: self._main_thread.post(lambda: self._set_percent(percent)),
            lambda error: self._main_thread.post(lambda: self._finish(error, on_ready)),
        )

    def _set_percent(self, percent: int) -> None:
        self.percent = percent
        self._on_change()

    def _finish(self, error: str | None, on_ready: Callable[[], None]) -> None:
        from tkinter import messagebox

        self.percent = None
        self._on_change()
        if error:
            messagebox.showerror(t("dlss5_download_title"), t("dlss5_download_failed", message=error))
        else:
            on_ready()

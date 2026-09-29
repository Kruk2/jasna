"""Which LTX models the GUI can run, and the one download that fetches a missing model."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from jasna.gui.download_worker import start_download
from jasna.gui.locales import t
from jasna.gui.queues import MainThreadCalls
from jasna.ltx import model_files
from jasna.ltx.model_files import LTX_MODELS, DownloadableFile, download_size_text
from jasna.session_config import LtxModelName

Variant = tuple[LtxModelName, bool]


def model_key(restoration_model: str, ltx_model: str) -> str:
    """Locale stem naming a restoration choice: ``model_<key>`` and ``tip_model_<key>``."""
    if restoration_model != "ltx":
        return "basicvsrpp"
    return "ltx" if ltx_model == "distilled" else f"ltx_{ltx_model}"


@dataclass(frozen=True)
class LtxInstallState:
    """The files each (model, fast) variant still needs, and whether they can be downloaded."""

    missing: Mapping[Variant, tuple[DownloadableFile, ...]]
    downloadable: bool

    def installed(self, model: LtxModelName, fast: bool) -> bool:
        return not self.missing[(model, fast)]

    def usable(self, model: LtxModelName, fast: bool) -> bool:
        return self.downloadable or self.installed(model, fast)

    def download_size(self, model: LtxModelName, fast: bool) -> str:
        return download_size_text(list(self.missing[(model, fast)]))


def read_install_state(directory: Path) -> LtxInstallState:
    return LtxInstallState(
        {
            (model, fast): tuple(model_files.missing_downloads(directory, model, fast=fast))
            for model in LTX_MODELS
            for fast in (False, True)
        },
        model_files.LTX_DOWNLOAD_URL is not None,
    )


def card_unavailable_reason(*, nvidia: bool | None) -> str | None:
    """Locale key saying why the LTX card cannot be picked; ``nvidia`` is None until the GPU is known.
    Without model files LTX still runs as a trial."""
    return "model_ltx_needs_nvidia" if nvidia is False else None


def trial_only(state: LtxInstallState) -> bool:
    """No LTX model is installed or downloadable, so only a trial run is possible."""
    return not any(state.usable(model, fast) for model in LTX_MODELS for fast in (False, True))


def run_unavailable_reason(
    state: LtxInstallState, model: LtxModelName, fast: bool, *, nvidia: bool | None, trial: bool
) -> str | None:
    """Locale key saying why LTX cannot restore right now with this variant."""
    if nvidia is False:
        return "model_ltx_needs_nvidia"
    if trial or state.installed(model, fast):
        return None
    return "model_ltx_not_downloaded" if state.downloadable else "model_ltx_not_installed"


def license_missing(directory: Path, model: LtxModelName, fast: bool) -> bool:
    """Whether the installed ``model`` needs a license this PC lacks. Verifies the stored
    license only; reads no model data."""
    from jasna.protection import ProtectionError

    try:
        model_files.LtxModelFiles.from_dir(directory, model, fast=fast)
    except ProtectionError:
        return True
    return False


class LtxModels:
    """The install state plus the single running download. UI glue: it asks with a message
    box and runs the download on a worker thread, reporting back on the main thread."""

    def __init__(self, directory: Path, main_thread: MainThreadCalls, on_change: Callable[[], None]) -> None:
        self._directory = directory
        self._main_thread = main_thread
        self._on_change = on_change
        self.directory = directory
        self.state = read_install_state(directory)
        self.percent: int | None = None

    @property
    def downloading(self) -> bool:
        return self.percent is not None

    def ensure(self, model: LtxModelName, fast: bool, *, on_ready: Callable[[], None]) -> bool:
        """True when the variant is installed, or the user agreed to download it and the
        download started (``on_ready`` runs after it succeeds); False otherwise."""
        from tkinter import messagebox

        files = self.state.missing[(model, fast)]
        if not files:
            return True
        if self.downloading or not self.state.downloadable:
            return False
        name = t(f"model_{model_key('ltx', model)}")
        if not messagebox.askyesno(
            t("ltx_download_title"), t("ltx_download_confirm", model=name, size=download_size_text(list(files)))
        ):
            return False
        self.percent = 0
        self._on_change()
        start_download(
            lambda progress: model_files.download_files(self._directory, list(files), progress),
            lambda percent: self._main_thread.post(lambda: self._set_percent(percent)),
            lambda error: self._main_thread.post(lambda: self._finish(error, on_ready)),
        )
        return True

    def _set_percent(self, percent: int) -> None:
        self.percent = percent
        self._on_change()

    def _finish(self, error: str | None, on_ready: Callable[[], None]) -> None:
        from tkinter import messagebox

        self.percent = None
        self.state = read_install_state(self._directory)
        self._on_change()
        if error:
            messagebox.showerror(t("ltx_download_title"), t("ltx_download_failed", message=error))
        else:
            on_ready()

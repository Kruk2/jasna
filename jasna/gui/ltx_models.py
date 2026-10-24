"""Which LTX models the GUI can run, and the one download that fetches a missing model."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from jasna.accelerator import AcceleratorVendor, supports_licensed_ltx_on_amd
from jasna.gui.download_worker import start_download
from jasna.gui.activation_worker import read_license_info
from jasna.gui.locales import get_locale, t
from jasna.gui.queues import MainThreadCalls
from jasna.ltx import model_files
from jasna.ltx.model_files import LTX_MODELS, DownloadableFile, download_size_text
from jasna.license_api import protection_available
from jasna.session_config import LtxModelName

Variant = tuple[LtxModelName, bool]


def model_key(restoration_model: str, ltx_model: str) -> str:
    """Locale stem naming a restoration choice: ``model_<key>`` and ``tip_model_<key>``."""
    if restoration_model != "ltx":
        return "basicvsrpp"
    return f"ltx_{ltx_model}"


@dataclass(frozen=True)
class LtxInstallState:
    """The files each (model, fast) variant still needs, whether they can be downloaded, and
    which variants need a license (the protected files)."""

    missing: Mapping[Variant, tuple[DownloadableFile, ...]]
    downloadable: bool
    licensed: Mapping[Variant, bool]

    def installed(self, model: LtxModelName, fast: bool) -> bool:
        return not self.missing[(model, fast)]

    def usable(self, model: LtxModelName, fast: bool, vendor: AcceleratorVendor | None) -> bool:
        if vendor is AcceleratorVendor.AMD and not supports_licensed_ltx_on_amd():
            return self.installed(model, fast) and not self.licensed[(model, fast)]
        return self.downloadable or self.installed(model, fast)


def read_install_state(directory: Path) -> LtxInstallState:
    variants = [(model, fast) for model in LTX_MODELS for fast in (False, True)]
    return LtxInstallState(
        {(model, fast): tuple(model_files.missing_downloads(directory, model, fast=fast)) for model, fast in variants},
        protection_available(),
        {(model, fast): model_files.needs_license(directory, model, fast=fast) for model, fast in variants},
    )


def card_unavailable_reason(vendor: AcceleratorVendor | None) -> str | None:
    """Locale key saying why the LTX card cannot be picked; ``vendor`` is None until the GPU is known.
    Without model files LTX still runs as a trial."""
    return "model_ltx_needs_nvidia" if vendor is AcceleratorVendor.CPU else None


def trial_only(state: LtxInstallState, vendor: AcceleratorVendor | None) -> bool:
    """No LTX model can run here, so only a trial run is possible."""
    return not any(state.usable(model, fast, vendor) for model in LTX_MODELS for fast in (False, True))


def trial_only_reason(vendor: AcceleratorVendor | None) -> str:
    """Locale key saying why only a trial run is possible."""
    return (
        "model_ltx_licensed_needs_nvidia"
        if vendor is AcceleratorVendor.AMD and not supports_licensed_ltx_on_amd()
        else "model_ltx_trial_only"
    )


def run_unavailable_reason(
    state: LtxInstallState, model: LtxModelName, fast: bool, vendor: AcceleratorVendor | None, *, trial: bool
) -> str | None:
    """Locale key saying why LTX cannot restore right now with this variant."""
    if vendor is AcceleratorVendor.CPU:
        return "model_ltx_needs_nvidia"
    if trial:
        return None
    if vendor is AcceleratorVendor.AMD and not supports_licensed_ltx_on_amd() and not state.usable(model, fast, vendor):
        return "model_ltx_licensed_needs_nvidia"
    if state.installed(model, fast):
        return None
    return "model_ltx_not_downloaded" if state.downloadable else "model_ltx_not_installed"


def license_missing(directory: Path, model: LtxModelName, fast: bool) -> bool:
    """Whether the installed ``model`` needs a license this PC lacks. Verifies the stored
    license only; reads no model data."""
    if not protection_available():
        return not all((directory / name).is_file() for name in model_files.bundle_names(model, fast=fast))
    from jasna.protection import ProtectionError

    try:
        if model_files.needs_license(directory, model, fast=fast):
            if not read_license_info()[1]:
                return True
            from jasna.protection.protected_tensors import check_ltx_license

            check_ltx_license()
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
        confirm_key = (
            "ltx_update_confirm" if any((self._directory / file.name).is_file() for file in files)
            else "ltx_download_confirm"
        )
        if not messagebox.askyesno(
            t("ltx_download_title"), t(confirm_key, model=name, size=download_size_text(list(files)))
        ):
            return False
        self.percent = 0
        self._on_change()
        mirror_first = get_locale().current_language == "zh"
        start_download(
            lambda progress: model_files.download_files(self._directory, list(files), progress, mirror_first=mirror_first),
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

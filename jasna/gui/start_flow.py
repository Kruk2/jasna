"""What stands between pressing Start and a run."""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jasna.gui import ltx_models, validation
from jasna.gui.models import AppSettings, JobItem


class LtxLicense(enum.Enum):
    USABLE = enum.auto()
    LOCKED = enum.auto()
    MISSING = enum.auto()


def read_ltx_license(directory: Path, settings: AppSettings) -> LtxLicense:
    if ltx_models.license_blocked():
        return LtxLicense.LOCKED
    if ltx_models.license_missing(
        directory, settings.ltx_model, settings.ltx_fast, settings.ltx_vae_encoder, settings.ltx_vae_decoder
    ):
        return LtxLicense.MISSING
    return LtxLicense.USABLE


class StartBlock(enum.Enum):
    DOWNLOAD_LTX = enum.auto()
    DOWNLOAD_DLSS5 = enum.auto()
    LTX_LOCKED = enum.auto()
    OFFER_LTX_TRIAL = enum.auto()
    INVALID = enum.auto()


@dataclass(frozen=True)
class StartBlocked:
    block: StartBlock
    errors: tuple[str, ...] = ()


def start_blocker(
    settings: AppSettings,
    jobs: list[JobItem],
    *,
    ltx_ready: bool,
    ltx_runs_on_this_pc: bool,
    ltx_downloadable: bool,
    dlss5_installed: Callable[[], bool],
    ltx_license: Callable[[], LtxLicense],
) -> StartBlocked | None:
    """The first thing that stops ``jobs`` from starting, or None when they can start.

    ``ltx_ready``: the chosen LTX variant can restore now; ``ltx_runs_on_this_pc``: this GPU and
    system can run LTX at all. ``dlss5_installed`` is asked only when DLSS 5 is chosen and
    ``ltx_license`` only for LTX jobs that are not a trial."""
    uses_ltx = validation.any_job_uses_ltx(jobs, settings)
    if uses_ltx and not ltx_ready and ltx_runs_on_this_pc and ltx_downloadable:
        return StartBlocked(StartBlock.DOWNLOAD_LTX)
    if settings.upscale_model == "dlss5" and not dlss5_installed():
        return StartBlocked(StartBlock.DOWNLOAD_DLSS5)
    if uses_ltx and ltx_ready and not settings.ltx_trial:
        license_state = ltx_license()
        if license_state is LtxLicense.LOCKED:
            return StartBlocked(StartBlock.LTX_LOCKED)
        if license_state is LtxLicense.MISSING:
            return StartBlocked(StartBlock.OFFER_LTX_TRIAL)
    errors = validation.validate_gui_start(settings, jobs, ltx_available=ltx_ready)
    if errors:
        return StartBlocked(StartBlock.INVALID, tuple(errors))
    return None

from dataclasses import replace
from pathlib import Path

import pytest

from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.start_flow import LtxLicense, StartBlock, StartBlocked, start_blocker

BASIC = AppSettings()
LTX = replace(AppSettings(), restoration_model="ltx")
LTX_TRIAL = replace(LTX, ltx_trial=True)
LTX_DLSS5 = replace(LTX, upscale_model="dlss5")
DLSS5 = replace(AppSettings(), upscale_model="dlss5")
UNUSABLE_LTX = StartBlocked(StartBlock.INVALID, (t("error_ltx_ranges_unavailable", files="a.mp4"),))


def never_asked():
    pytest.fail("asked a probe the start did not need")


@pytest.mark.parametrize(
    ("settings", "ltx_ready", "ltx_runs_on_this_pc", "ltx_downloadable", "dlss5_installed", "ltx_license", "blocked"),
    [
        (BASIC, True, True, True, never_asked, never_asked, None),
        (LTX, True, True, True, never_asked, lambda: LtxLicense.USABLE, None),
        (LTX, False, True, True, never_asked, never_asked, StartBlocked(StartBlock.DOWNLOAD_LTX)),
        (LTX_DLSS5, False, True, True, never_asked, never_asked, StartBlocked(StartBlock.DOWNLOAD_LTX)),
        (LTX, False, False, True, never_asked, never_asked, UNUSABLE_LTX),
        (LTX, False, True, False, never_asked, never_asked, UNUSABLE_LTX),
        (DLSS5, True, True, True, lambda: False, never_asked, StartBlocked(StartBlock.DOWNLOAD_DLSS5)),
        (DLSS5, True, True, True, lambda: True, never_asked, None),
        (LTX, True, True, True, never_asked, lambda: LtxLicense.LOCKED, StartBlocked(StartBlock.LTX_LOCKED)),
        (LTX, True, True, True, never_asked, lambda: LtxLicense.MISSING, StartBlocked(StartBlock.OFFER_LTX_TRIAL)),
        (LTX_TRIAL, True, True, True, never_asked, never_asked, None),
    ],
)
def test_start_blocker_names_the_first_thing_that_stops_a_start(
    settings, ltx_ready, ltx_runs_on_this_pc, ltx_downloadable, dlss5_installed, ltx_license, blocked
) -> None:
    jobs = [JobItem(path=Path("a.mp4"))]

    assert start_blocker(
        settings,
        jobs,
        ltx_ready=ltx_ready,
        ltx_runs_on_this_pc=ltx_runs_on_this_pc,
        ltx_downloadable=ltx_downloadable,
        dlss5_installed=dlss5_installed,
        ltx_license=ltx_license,
    ) == blocked

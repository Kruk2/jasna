from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from factories import video_metadata
from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.media.video_decoder import VideoReader, _shown_frames

NTSC_60 = Fraction(60_000, 1_001)


@pytest.mark.parametrize(
    ("source", "measured", "enabled", "stride", "output_fps", "mismatch"),
    [
        (Fraction(60), 60.0, True, 2, Fraction(30), False),
        (NTSC_60, float(NTSC_60), True, 2, Fraction(30_000, 1_001), False),
        # ffprobe reports 19001/317 for 59.94 fps footage whose timestamps drift off the grid.
        (Fraction(19_001, 317), 59.94, True, 2, Fraction(19_001, 634), False),
        (NTSC_60, 59.93, True, 2, Fraction(30_000, 1_001), False),
        (Fraction(60), 0.0, True, 2, Fraction(30), False),
        (Fraction(24), 24.0, True, 1, Fraction(24), False),
        (Fraction(25), 25.0, True, 1, Fraction(25), False),
        (Fraction(30_000, 1_001), 29.97, True, 1, Fraction(30_000, 1_001), False),
        (Fraction(30), 30.0, True, 1, Fraction(30), False),
        (Fraction(50), 50.0, True, 1, Fraction(50), False),
        (Fraction(120), 120.0, True, 1, Fraction(120), False),
        (Fraction(60), 60.0, False, 1, Fraction(60), False),
        # NHDTB-634.mp4 from issue #248: r_frame_rate says 59.94, frames arrive at 29.98.
        (NTSC_60, 29.9803, True, 1, NTSC_60, True),
        (Fraction(24), 12.0, True, 1, Fraction(24), False),
    ],
    ids=[
        "60", "59.94", "drifting-59.94", "measured-close", "unmeasured", "24", "25", "29.97", "30", "50",
        "120", "disabled", "doubled-container-rate", "never-halved-rate",
    ],
)
def test_only_standard_high_frame_rates_are_halved(source, measured, enabled, stride, output_fps, mismatch):
    retarget = resolve_frame_rate_retarget(source, enabled=enabled, measured_fps=measured)

    assert (retarget.active, retarget.frame_stride) == (stride == 2, stride)
    assert retarget.output_fps == output_fps
    assert retarget.rate_mismatch is mismatch


@pytest.mark.parametrize(("source_count", "output_count"), [(0, 0), (1, 1), (4, 2), (5, 3)])
def test_output_frame_count_keeps_first_and_even_indexed_frames(source_count, output_count):
    retarget = resolve_frame_rate_retarget(Fraction(60, 1), enabled=True, measured_fps=60.0)
    assert retarget.output_frame_count(source_count) == output_count


@pytest.mark.parametrize(
    ("decoded", "shown"),
    [(range(7), [0, 2, 4, 6]), (range(31, 38), [31, 33, 35, 37])],
    ids=["from-start", "reanchored-after-seek"],
)
def test_reader_keeps_every_second_frame_from_the_first_one_it_decodes(decoded, shown):
    assert list(_shown_frames(iter(decoded), lambda pts: pts, 0, 2)) == shown


def test_reader_rejects_invalid_frame_stride():
    with pytest.raises(ValueError, match="frame_stride must be > 0"):
        VideoReader("unused.mp4", batch_size=4, device=torch.device("cpu"), metadata=video_metadata(), frame_stride=0)

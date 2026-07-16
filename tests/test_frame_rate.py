from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.media.video_decoder import NvidiaVideoReader


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (Fraction(60, 1), Fraction(30, 1)),
        (Fraction(60_000, 1_001), Fraction(30_000, 1_001)),
    ],
)
def test_standard_high_frame_rates_are_halved_exactly(source: Fraction, target: Fraction):
    retarget = resolve_frame_rate_retarget(source, enabled=True)

    assert retarget.active is True
    assert retarget.frame_stride == 2
    assert retarget.output_fps == target


@pytest.mark.parametrize(
    "source",
    [
        Fraction(24, 1),
        Fraction(25, 1),
        Fraction(30_000, 1_001),
        Fraction(30, 1),
        Fraction(50, 1),
    ],
)
def test_other_frame_rates_are_unchanged(source: Fraction):
    retarget = resolve_frame_rate_retarget(source, enabled=True)

    assert retarget.active is False
    assert retarget.frame_stride == 1
    assert retarget.output_fps == source


def test_disabled_retarget_keeps_60_fps():
    retarget = resolve_frame_rate_retarget(Fraction(60, 1), enabled=False)

    assert retarget.active is False
    assert retarget.output_fps == Fraction(60, 1)


@pytest.mark.parametrize(("source_count", "output_count"), [(0, 0), (1, 1), (4, 2), (5, 3)])
def test_output_frame_count_keeps_first_and_even_indexed_frames(source_count, output_count):
    retarget = resolve_frame_rate_retarget(Fraction(60, 1), enabled=True)
    assert retarget.output_frame_count(source_count) == output_count


def test_reader_selects_every_second_decoded_frame_before_batching():
    reader = NvidiaVideoReader(
        "unused.mp4",
        batch_size=4,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(),
        frame_stride=2,
    )
    frames = [SimpleNamespace(pts=pts) for pts in range(7)]

    selected = list(reader._selected_frames(iter(frames)))

    assert [frame.pts for frame in selected] == [0, 2, 4, 6]


def test_reader_rejects_invalid_frame_stride():
    with pytest.raises(ValueError, match="frame_stride must be > 0"):
        NvidiaVideoReader(
            "unused.mp4",
            batch_size=4,
            device=torch.device("cpu"),
            metadata=SimpleNamespace(),
            frame_stride=0,
        )


def test_reader_rejects_seek_with_strided_selection():
    reader = NvidiaVideoReader(
        "unused.mp4",
        batch_size=4,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(),
        frame_stride=2,
    )

    with pytest.raises(ValueError, match="anchored to the start of the file"):
        next(reader.frames(seek_ts=1.0))

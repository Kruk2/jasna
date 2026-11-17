from __future__ import annotations

import pytest
from PIL import Image

from jasna.gui.restoration_preview import RestoredClipFrame
from jasna.gui.segment_restore_preview import clip_finished, next_window_start, restored_clip_covers

FPS = 10.0


def _clip(*seconds: float) -> tuple[RestoredClipFrame, ...]:
    return tuple(RestoredClipFrame(value, Image.new("RGB", (2, 2))) for value in seconds)


@pytest.mark.parametrize(
    ("clip", "seconds", "covered"),
    [
        (_clip(1.0), 1.0, False),
        (_clip(1.0, 1.1, 1.2), 0.93, True),
        (_clip(1.0, 1.1, 1.2), 0.9, False),
        (_clip(1.0, 1.1, 1.2), 1.1, True),
        (_clip(1.0, 1.1, 1.2), 1.13, False),
    ],
    ids=["one frame", "just before", "a frame before", "inside", "at the last frame"],
)
def test_restored_clip_covers_playback_from_its_first_to_before_its_last_frame(clip, seconds, covered) -> None:
    assert restored_clip_covers(clip, seconds, FPS) is covered


def test_clip_finishes_half_a_frame_before_its_last_frame() -> None:
    clip = _clip(1.0, 1.1, 1.2)

    assert not clip_finished(clip, 1.14, FPS)
    assert clip_finished(clip, 1.16, FPS)


@pytest.mark.parametrize(("duration", "start"), [(10.0, pytest.approx(1.3)), (1.33, None)], ids=["more", "video end"])
def test_next_window_starts_a_frame_after_the_clip_unless_the_video_ends(duration, start) -> None:
    assert next_window_start(_clip(1.0, 1.1, 1.2), FPS, duration) == start

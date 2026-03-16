import time
from unittest.mock import MagicMock, patch

import pytest

from jasna.progressbar import Progressbar


class TestFormatDuration:
    def _fmt(self, s):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        return pb._format_duration(s)

    def test_zero(self):
        assert self._fmt(0) == "0:00"

    def test_negative(self):
        assert self._fmt(-5) == "0:00"

    def test_none(self):
        assert self._fmt(None) == "0:00"

    def test_seconds(self):
        assert self._fmt(45) == "0:45"

    def test_minutes(self):
        assert self._fmt(125) == "2:05"

    def test_hours(self):
        assert self._fmt(3723) == "1:02:03"


class TestProgressbarLifecycle:
    def test_init_sets_timer(self):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        assert pb.duration_start is None
        pb.init()
        assert pb.duration_start is not None

    def test_update_increments_frames(self):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        pb.init()
        pb.update(5)
        assert pb.frames_processed == 5
        pb.update(3)
        assert pb.frames_processed == 8

    def test_update_auto_inits(self):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        pb.update(1)
        assert pb.duration_start is not None

    def test_close_adjusts_total_on_mismatch(self):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        pb.init()
        pb.update(50)
        pb.close(ensure_completed_bar=True)
        # After close with ensure_completed_bar, tqdm.n == tqdm.total
        assert pb.tqdm.total == pb.tqdm.n

    def test_close_no_adjust_on_error(self):
        pb = Progressbar(total_frames=100, video_fps=30, disable=True)
        pb.init()
        pb.update(50)
        pb.error = True
        pb.close(ensure_completed_bar=True)
        assert pb.tqdm.total == 100

    def test_callback_invoked(self):
        cb = MagicMock()
        pb = Progressbar(total_frames=10, video_fps=1, disable=True, callback=cb)
        pb.init()
        for _ in range(10):
            pb.update(1)
        assert cb.call_count == 10
        last_call = cb.call_args
        assert last_call[0][0] == pytest.approx(100.0)
        assert last_call[0][3] == 10
        assert last_call[0][4] == 10

    def test_buffer_respects_max_len(self):
        pb = Progressbar(total_frames=200, video_fps=1, disable=True)
        pb.init()
        for _ in range(200):
            pb.update(1)
        assert len(pb.frame_processing_durations_buffer) <= pb.frame_processing_durations_buffer_max_len

    def test_speed_display_after_enough_data(self):
        pb = Progressbar(total_frames=100, video_fps=1, disable=True)
        pb.init()
        for _ in range(5):
            pb.update(1)
        assert "Speed:" in pb.tqdm.desc or "?" in pb.tqdm.desc

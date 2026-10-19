import time
from unittest.mock import MagicMock, patch

import pytest

from jasna.progressbar import Progressbar, StageProgress


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
        assert last_call[0][5] is None

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


class TestJobProgress:
    def test_time_left_is_the_running_part_plus_the_estimates_of_the_parts_after_it(self, monkeypatch):
        from jasna import progressbar

        clock = [100.0]
        monkeypatch.setattr(progressbar.time, "monotonic", lambda: clock[0])
        calls = []
        job = progressbar.JobProgress(lambda *args: calls.append(args), {"standard": 0.0, "ltx": 900.0})
        standard, ltx = job.part("standard"), job.part("ltx")

        clock[0] = 110.0
        standard(50.0, 30.0, 10.0, 60, 120, None)
        assert calls[-1] == (pytest.approx(100.0 * 10.0 / 920.0), 30.0, 910.0, 60, 120, None)

        clock[0] = 130.0
        scan = StageProgress("scan", 1, 4, 0.5, 0.0)
        ltx(10.0, 0.0, 0.0, 0, 0, scan)
        assert calls[-1] == (pytest.approx(100.0 * 30.0 / 840.0), 0.0, 810.0, 0, 0, scan)

        clock[0] = 200.0
        compose = StageProgress("compose", 4, 4, 0.5, 50.0)
        ltx(90.0, 0.0, 50.0, 0, 0, compose)
        assert calls[-1] == (pytest.approx(100.0 * 100.0 / 150.0), 0.0, 50.0, 0, 0, compose)

    def test_percent_never_drops_when_the_ltx_estimate_grows_but_time_left_stays_honest(self, monkeypatch):
        from jasna import progressbar

        clock = [0.0]
        monkeypatch.setattr(progressbar.time, "monotonic", lambda: clock[0])
        calls = []
        job = progressbar.JobProgress(lambda *args: calls.append(args), {"standard": 0.0, "ltx": 600.0})
        standard, ltx = job.part("standard"), job.part("ltx")

        clock[0] = 300.0
        standard(100.0, 30.0, 0.0, 120, 120, None)
        assert calls[-1][0] == pytest.approx(100.0 * 300.0 / 900.0)

        scan = StageProgress("scan", 1, 4, 1.0, 1500.0)
        ltx(5.0, 0.0, 1500.0, 0, 0, scan)
        assert calls[-1] == (pytest.approx(100.0 * 300.0 / 900.0), 0.0, 1500.0, 0, 0, scan)

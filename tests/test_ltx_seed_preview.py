from __future__ import annotations

import threading
from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from PIL import Image

from factories import video_metadata

from jasna.gui import ltx_seed_preview
from jasna.gui.ltx_seed_preview import (
    LtxSeedPreviewWorker,
    SeedFrame,
    SeedFrameWriter,
    SeedProgress,
    SeedReady,
    prepared_key,
)
from jasna.gui.models import AppSettings
from jasna.ltx.inference import Cancelled
from jasna.media.splice import KeyframeIndex
from jasna.progressbar import StageProgress
from jasna.segments import SegmentRange

METADATA = video_metadata(video_width=8, video_height=4, time_base=Fraction(1, 100))
KEYFRAMES = KeyframeIndex(pts=(0,), time_base=Fraction(1, 100), start_pts=0, end_pts=1000)


def test_seed_frame_writer_keeps_only_the_range_frames(tmp_path) -> None:
    writer = SeedFrameWriter(tmp_path, (100, 300), METADATA, None, "off")

    for pts in (0, 100, 200, 300):
        writer.write(torch.full((3, 4, 8), 200, dtype=torch.uint8), pts)

    assert writer.frames == [SeedFrame(1.0, tmp_path / "100.jpg"), SeedFrame(2.0, tmp_path / "200.jpg")]
    assert all(frame.path.is_file() for frame in writer.frames)


@pytest.mark.parametrize(("stereo", "size"), [("off", (8, 4)), ("sbs", (4, 4)), ("tb", (8, 2))])
def test_seed_frame_writer_saves_the_left_eye_of_vr(tmp_path, stereo, size) -> None:
    writer = SeedFrameWriter(tmp_path, (0, 100), METADATA, None, stereo)

    writer.write(torch.full((3, 4, 8), 200, dtype=torch.uint8), 0)

    with Image.open(writer.frames[0].path) as image:
        assert image.size == size


@pytest.mark.parametrize("segment, changes, changes_key", [
    (SegmentRange(1, 2), {"lut_path": "x.cube", "ltx_seed": 5}, False),
    (SegmentRange(1, 2), {"detection_score_threshold": 0.9}, True),
    (SegmentRange(1, 2), {"ltx_large_canvas": True}, True),
    (SegmentRange(1, 2), {"ltx_overlap": 33}, True),
    (SegmentRange(1, 2), {"vr_projection": "fisheye"}, True),
    (SegmentRange(1, 2), {"vr_mode": "off"}, True),
    (SegmentRange(1, 2), {"ltx_model": "undistilled"}, True),
    (SegmentRange(1, 3), {}, True),
])
def test_prepared_key_follows_range_detection_canvas_vr_and_model_but_not_the_look(segment, changes, changes_key) -> None:
    settings = replace(AppSettings(), restoration_model="ltx")

    assert (prepared_key(segment, replace(settings, **changes)) != prepared_key(SegmentRange(1, 2), settings)) == changes_key


class _Renderer:
    def __init__(self, tmp_path, *, block: bool = False) -> None:
        self.tmp_path = tmp_path
        self.block = block
        self.started = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def render(self, segment, seed, settings, *, writer_for, report, cancel):
        self.started.set()
        if self.block:
            cancel.wait(5)
            raise Cancelled()
        report(0.5, 12.0, StageProgress("denoise", 3, 4, 0.5, 6.0))
        writer = writer_for(self.tmp_path, (0, 1000), "off")
        writer.frames.append(SeedFrame(0.5, self.tmp_path / "50.jpg"))
        return writer

    def release(self):
        self.released.set()

    def close(self):
        self.closed = True


def _worker(monkeypatch, renderer) -> tuple[LtxSeedPreviewWorker, threading.Event]:
    stopped = threading.Event()
    worker = LtxSeedPreviewWorker(
        "video.mp4", METADATA, KEYFRAMES, renderer.tmp_path, on_stopped=stopped.set
    )
    monkeypatch.setattr(worker, "_make_renderer", lambda: renderer)
    worker.start()
    return worker, stopped


def test_worker_reports_progress_then_the_seed_frames(monkeypatch, tmp_path) -> None:
    renderer = _Renderer(tmp_path)
    worker, stopped = _worker(monkeypatch, renderer)

    generation = worker.try_seed(SegmentRange(1, 2), 42, AppSettings())

    assert worker.events.get(timeout=5) == SeedProgress(42, "denoise", 0.5, 12.0, generation)
    assert worker.events.get(timeout=5) == SeedReady(42, (SeedFrame(0.5, tmp_path / "50.jpg"),), generation)
    worker.release()
    assert renderer.released.wait(5)
    worker.close()
    worker.join(timeout=5)
    assert renderer.closed and stopped.is_set()


def test_cancel_stops_a_running_seed_without_a_result(monkeypatch, tmp_path) -> None:
    renderer = _Renderer(tmp_path, block=True)
    worker, stopped = _worker(monkeypatch, renderer)

    worker.try_seed(SegmentRange(1, 2), 42, AppSettings())
    assert renderer.started.wait(5)
    worker.cancel()
    worker.close()
    worker.join(timeout=5)

    assert worker.events.empty()
    assert renderer.closed and stopped.is_set()


def test_cancel_drops_a_seed_still_waiting_for_the_worker(tmp_path) -> None:
    worker = LtxSeedPreviewWorker("video.mp4", METADATA, KEYFRAMES, tmp_path, on_stopped=lambda: None)
    worker.try_seed(SegmentRange(1, 2), 42, AppSettings())

    worker.cancel()

    assert worker._commands.empty()


def test_seed_renderer_replaces_crashed_temp_dirs_and_deletes_its_own_on_close(tmp_path) -> None:
    (tmp_path / ".jasna-seeds-crashed").mkdir()
    renderer = ltx_seed_preview.SeedRenderer(tmp_path / "video.mp4", METADATA, KEYFRAMES, tmp_path)
    (temp,) = tmp_path.glob(".jasna-seeds-*")
    assert temp.name != ".jasna-seeds-crashed"

    renderer.close()

    assert not temp.exists()

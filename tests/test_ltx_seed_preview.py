from __future__ import annotations

import threading
import time
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import torch

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
from jasna.ltx.restore import Cancelled
from jasna.segments import SegmentRange

METADATA = SimpleNamespace(video_width=8, video_height=4, start_pts=0, time_base=Fraction(1, 100))


def test_seed_frame_writer_keeps_only_the_range_frames(tmp_path) -> None:
    writer = SeedFrameWriter(tmp_path, (100, 300), METADATA, None)

    for pts in (0, 100, 200, 300):
        writer.write(torch.full((3, 4, 8), 200, dtype=torch.uint8), pts, apply_lut=pts != 0)

    assert writer.frames == [SeedFrame(1.0, tmp_path / "100.jpg"), SeedFrame(2.0, tmp_path / "200.jpg")]
    assert all(frame.path.is_file() for frame in writer.frames)


def test_prepared_key_follows_detection_and_canvas_but_not_the_look() -> None:
    segment = SegmentRange(1, 2)
    settings = AppSettings()

    assert prepared_key(segment, replace(settings, lut_path="x.cube", ltx_seed=5)) == prepared_key(segment, settings)
    assert prepared_key(segment, replace(settings, detection_score_threshold=0.9)) != prepared_key(segment, settings)
    assert prepared_key(segment, replace(settings, ltx_large_canvas=True)) != prepared_key(segment, settings)
    assert prepared_key(SegmentRange(1, 3), settings) != prepared_key(segment, settings)


class _Renderer:
    def __init__(self, tmp_path, *, block: bool = False) -> None:
        self.tmp_path = tmp_path
        self.block = block
        self.started = threading.Event()
        self.released = 0
        self.closed = False

    def render(self, segment, seed, settings, *, writer_for, report, cancel):
        self.started.set()
        if self.block:
            cancel.wait(5)
            raise Cancelled()
        report("denoise", 0.5, 12.0)
        writer = writer_for(self.tmp_path, (0, 1000))
        writer.frames.append(SeedFrame(0.5, self.tmp_path / "50.jpg"))
        return writer

    def release(self):
        self.released += 1

    def close(self):
        self.closed = True


def _worker(monkeypatch, renderer) -> tuple[LtxSeedPreviewWorker, threading.Event]:
    stopped = threading.Event()
    worker = LtxSeedPreviewWorker(
        "video.mp4", METADATA, SimpleNamespace(), renderer.tmp_path, on_stopped=stopped.set
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
    deadline = time.monotonic() + 5
    while not renderer.released and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.close()
    worker.join(timeout=5)
    assert renderer.released == 1 and renderer.closed and stopped.is_set()


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


def test_seed_renderer_deletes_its_temp_dir_on_close(tmp_path) -> None:
    renderer = ltx_seed_preview.SeedRenderer(tmp_path / "video.mp4", METADATA, SimpleNamespace(), tmp_path)
    (temp,) = tmp_path.glob(".jasna-seeds-*")

    renderer.close()

    assert not temp.exists()


def test_prepared_key_never_mixes_ltx_models() -> None:
    segment = SegmentRange(1, 2)
    settings = replace(AppSettings(), restoration_model="ltx")

    assert prepared_key(segment, replace(settings, ltx_model="undistilled")) != prepared_key(segment, settings)

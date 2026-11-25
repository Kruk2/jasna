from __future__ import annotations

import json
from pathlib import Path

import pytest

from jasna.gui.models import JobItem, JobStatus
from jasna.gui.processor import ProgressUpdate
from jasna.gui.saved_queue import get_queue_path, job_from_dict, job_to_dict, load_queue, save_queue
from jasna.segments import SegmentRange, SegmentRestoration


def _video(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def _job_options(job: JobItem) -> tuple:
    return (
        job.path,
        job.duration_seconds,
        job.segments,
        job.detection_model,
        job.detection_score_threshold,
        job.vr_projection,
        job.unfinished_output,
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "duration_seconds": 90.5,
            "segments": (
                SegmentRange(1, 2),
                SegmentRange(3, 4.5, SegmentRestoration("ltx", 7)),
                SegmentRange(5, 6, SegmentRestoration("basicvsrpp", None)),
            ),
            "detection_model": "rfdetr-v5",
            "detection_score_threshold": 0.5,
            "vr_projection": "fisheye",
            "unfinished_output": Path("out") / "a_restored (1).mp4",
        },
    ],
    ids=["plain", "edited"],
)
def test_job_keeps_its_options_through_the_saved_queue(options) -> None:
    job = JobItem(path=Path("videos") / "a.mp4", status=JobStatus.ERROR, progress=0.4, **options)

    restored = job_from_dict(json.loads(json.dumps(job_to_dict(job))))

    assert _job_options(restored) == _job_options(job)
    assert (restored.status, restored.progress) == (JobStatus.PENDING, 0.0)


def test_saved_queue_keeps_unfinished_items_in_order_as_pending(user_config_dir, tmp_path) -> None:
    save_queue([JobItem(path=_video(tmp_path / f"{status.value}.mp4"), status=status) for status in JobStatus])

    restored = load_queue()

    assert [job.path.name for job in restored] == ["pending.mp4", "processing.mp4", "error.mp4", "skipped.mp4"]
    assert {job.status for job in restored} == {JobStatus.PENDING}


def test_restored_queue_leaves_out_files_that_are_gone(user_config_dir, tmp_path) -> None:
    kept = _video(tmp_path / "kept.mp4")
    save_queue([JobItem(path=tmp_path / "gone.mp4"), JobItem(path=kept)])

    assert [job.path for job in load_queue()] == [kept]


def test_no_saved_queue_restores_nothing(user_config_dir) -> None:
    assert load_queue() == []


def _with_job(**changes) -> dict:
    job = job_to_dict(JobItem(path=Path("a.mp4"), segments=(SegmentRange(1, 2, SegmentRestoration("ltx", 3)),)))
    return {"format": 1, "jobs": [{**job, **changes}]}


def _with_segment(**changes) -> dict:
    data = _with_job()
    data["jobs"][0]["segments"][0].update(changes)
    return data


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe",
        json.dumps([]).encode(),
        json.dumps({"format": 2, "jobs": []}).encode(),
        json.dumps({"format": 1, "jobs": [{"path": "a.mp4"}]}).encode(),
        json.dumps(_with_job(path=None)).encode(),
        json.dumps(_with_job(detection_score_threshold="high")).encode(),
        json.dumps(_with_segment(end=0.5)).encode(),
        json.dumps(_with_segment(restoration={"model": "other", "ltx_seed": None})).encode(),
        json.dumps(_with_segment(restoration={"model": "ltx", "ltx_seed": None})).encode(),
    ],
    ids=[
        "not-json", "not-utf8", "not-an-object", "other-format", "missing-keys", "no-path",
        "bad-threshold", "empty-range", "unknown-model", "ltx-without-seed",
    ],
)
def test_unreadable_saved_queue_restores_nothing(user_config_dir, tmp_path, content) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    get_queue_path().parent.mkdir(parents=True)
    get_queue_path().write_bytes(content)

    assert load_queue() == []


def test_queue_comes_back_in_a_new_window(make_jasna_app, tmp_path) -> None:
    first = make_jasna_app()
    videos = [_video(tmp_path / name) for name in ("b.mp4", "a.mp4", "c.mp4")]
    segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 3)),)
    first._queue_panel.add_jobs([
        JobItem(path=videos[0], segments=segments, detection_score_threshold=0.6),
        JobItem(path=videos[1]),
        JobItem(path=videos[2]),
    ])
    first._on_close()
    videos[2].unlink()

    second = make_jasna_app()

    restored = [(job.path, job.segments, job.detection_score_threshold) for job in second._queue_panel.get_jobs()]
    assert restored == [(videos[0], segments, 0.6), (videos[1], (), None)]


def test_finished_job_leaves_the_saved_queue_while_the_queue_runs(jasna_app, tmp_path) -> None:
    videos = [_video(tmp_path / name) for name in ("a.mp4", "b.mp4")]
    jasna_app._queue_panel.add_jobs(JobItem(path=video) for video in videos)
    finished = jasna_app._queue_panel.get_jobs()[0]

    jasna_app._handle_progress(ProgressUpdate(job_id=finished.id, status=JobStatus.COMPLETED, progress=100.0))

    assert [job.path for job in load_queue()] == videos[1:]


def test_job_keeps_accumulated_elapsed_seconds_through_saved_queue() -> None:
    plain = JobItem(path=Path("videos") / "plain.mp4")
    assert "elapsed_seconds" not in job_to_dict(plain)

    paused = JobItem(path=Path("videos") / "paused.mp4", status=JobStatus.PENDING, elapsed_seconds=125.5)
    restored = job_from_dict(json.loads(json.dumps(job_to_dict(paused))))

    assert restored.elapsed_seconds == 125.5
    assert restored.status is JobStatus.PENDING


def test_old_saved_queue_without_elapsed_seconds_loads_with_none(user_config_dir, tmp_path) -> None:
    video = _video(tmp_path / "legacy.mp4")
    legacy_job = job_to_dict(JobItem(path=video))
    assert "elapsed_seconds" not in legacy_job
    get_queue_path().parent.mkdir(parents=True)
    get_queue_path().write_text(json.dumps({"format": 1, "jobs": [legacy_job]}), encoding="utf-8")

    restored = load_queue()

    assert len(restored) == 1
    assert restored[0].path == video
    assert restored[0].elapsed_seconds is None


@pytest.mark.parametrize("bad_elapsed", [True, False, -1, -0.01, "10", float("nan"), float("inf"), float("-inf")])
def test_invalid_elapsed_seconds_is_rejected(bad_elapsed) -> None:
    data = _with_job(elapsed_seconds=bad_elapsed)["jobs"][0]
    with pytest.raises((TypeError, ValueError)):
        job_from_dict(data)


@pytest.mark.parametrize("bad_elapsed", [True, -5, "10"])
def test_corrupt_elapsed_seconds_in_queue_file_restores_nothing(user_config_dir, tmp_path, bad_elapsed) -> None:
    _video(tmp_path / "a.mp4")
    get_queue_path().parent.mkdir(parents=True)
    get_queue_path().write_bytes(json.dumps(_with_job(elapsed_seconds=bad_elapsed)).encode("utf-8"))

    assert load_queue() == []


@pytest.mark.usefixtures("nvidia_vendor")
def test_paused_job_elapsed_seconds_survives_restart_and_accumulates(make_jasna_app, monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from jasna.gui import app as app_module

    clock = [100.0]
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    video = _video(tmp_path / "paused.mp4")
    segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 3)),)
    first = make_jasna_app()
    first._queue_panel.add_jobs([JobItem(path=video, segments=segments)])
    job = first._queue_panel.get_jobs()[0]

    first._handle_progress(ProgressUpdate(job_id=job.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 145.0
    first._handle_progress(ProgressUpdate(job_id=job.id, status=JobStatus.PENDING))
    assert job.elapsed_seconds == pytest.approx(45.0)
    first._on_close()

    second = make_jasna_app()
    restored = second._queue_panel.get_jobs()[0]
    assert restored.status is JobStatus.PENDING
    assert restored.elapsed_seconds == pytest.approx(45.0)

    monkeypatch.setattr(second, "_settings_panel", second._settings_panel)
    monkeypatch.setattr("jasna.gui.app.start_blocker", lambda *_a, **_k: None)
    monkeypatch.setattr("jasna.gui.engine_preflight.run_engine_preflight", lambda *_a, **_k: SimpleNamespace(missing=()))
    monkeypatch.setattr(second._processor, "start", lambda *_a, **_k: None)

    clock[0] = 500.0
    second._on_start()
    assert restored.elapsed_seconds == pytest.approx(45.0)

    second._handle_progress(ProgressUpdate(job_id=restored.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 520.0
    second._handle_progress(ProgressUpdate(job_id=restored.id, status=JobStatus.COMPLETED, progress=100.0))
    second._handle_complete(queue_finished=True)

    assert restored.elapsed_seconds == pytest.approx(65.0)
    assert second._queue_panel._job_widgets[0]._status_label.cget("text").endswith("1m 5s")
    assert second._control_bar._fps_label.cget("text").endswith("1m 5s")


@pytest.mark.usefixtures("nvidia_vendor")
def test_stop_and_quit_while_processing_flushes_resumable_elapsed_time(
    make_jasna_app, fake_job_pipelines, monkeypatch, tmp_path
) -> None:
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from jasna.gui import app as app_module
    import jasna.gui.processor as processor_module

    clock = [100.0]
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("jasna.gui.app.start_blocker", lambda *_a, **_k: None)
    monkeypatch.setattr("jasna.gui.engine_preflight.run_engine_preflight", lambda *_a, **_k: SimpleNamespace(missing=()))
    monkeypatch.setattr(processor_module, "splice_source", MagicMock())
    monkeypatch.setattr(
        processor_module, "plan_segment_job", MagicMock(return_value=MagicMock(reencodes_whole_video=False))
    )

    entered = threading.Event()
    second_started = threading.Event()

    def restore(pipeline):
        if len(fake_job_pipelines.built) == 1:
            entered.set()
            while not pipeline.cancel_requested:
                threading.Event().wait(0.01)
        else:
            assert second_started.wait(5.0)
            clock[0] += 10.0

    fake_job_pipelines.restore = restore
    video = _video(tmp_path / "quit_paused.mp4")
    segments = (SegmentRange(1, 2),)

    first = make_jasna_app()
    first._queue_panel.add_jobs([JobItem(path=video, segments=segments)])
    first._on_start()
    assert entered.wait(5.0)
    first._main_thread.drain_pending()

    clock[0] = 135.0
    first._on_close()

    saved = load_queue()
    assert len(saved) == 1
    assert saved[0].elapsed_seconds == pytest.approx(35.0)

    second = make_jasna_app()
    restored = second._queue_panel.get_jobs()[0]
    assert restored.elapsed_seconds == pytest.approx(35.0)

    original_handle = second._handle_progress

    def track_second(update: ProgressUpdate):
        original_handle(update)
        if update.status is JobStatus.PROCESSING:
            second_started.set()

    monkeypatch.setattr(second, "_handle_progress", track_second)

    clock[0] = 300.0
    second._on_start()
    while second._processor.is_running():
        second._main_thread.drain_pending()
        threading.Event().wait(0.005)
    second._processor.join(timeout=5.0)
    second._main_thread.drain_pending()

    assert restored.status is JobStatus.COMPLETED
    assert restored.elapsed_seconds == pytest.approx(45.0)
    assert second._queue_panel._job_widgets[0]._status_label.cget("text").endswith("45s")
    assert second._control_bar._fps_label.cget("text").endswith("45s")

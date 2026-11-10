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

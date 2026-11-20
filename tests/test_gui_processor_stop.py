import threading
from pathlib import Path

import pytest

from jasna.gui.models import DEFAULT_OUTPUT_PATTERN, AppSettings, JobItem, JobStatus
from jasna.gui.processor import ProgressUpdate
from factories import gui_processor, job_queue, video_metadata

pytestmark = pytest.mark.usefixtures("no_gpu_cleanup")


def _run_queue(processor, jobs, tmp_path, settings, output_pattern="{original}_restored.mp4"):
    processor.start(job_queue(jobs), settings, str(tmp_path), output_pattern)
    processor.join()
    assert not processor.is_running()


def test_stop_cancels_the_running_job_and_leaves_it_queued(tmp_path, fake_job_pipelines):
    started, release = threading.Event(), threading.Event()

    def restore(_pipeline):
        started.set()
        release.wait(5)

    fake_job_pipelines.restore = restore
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []
    processor = gui_processor(on_progress=updates.append)
    processor.start(job_queue([job]), AppSettings(), str(tmp_path), "{original}_restored.mp4")
    assert started.wait(5)
    assert job.status is JobStatus.PROCESSING

    processor.stop()
    release.set()
    processor.join()

    assert not processor.is_running()
    assert fake_job_pipelines.built[0].cancel_requested
    assert len(fake_job_pipelines.built) == 1  # a job left pending must not be picked up again
    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING


def test_completed_job_stays_completed(tmp_path, fake_job_pipelines):
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []

    _run_queue(gui_processor(on_progress=updates.append), [job], tmp_path, AppSettings())

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "clip_restored.mp4"
    assert updates[-1].status is JobStatus.COMPLETED


def test_failing_job_still_marked_error(tmp_path, fake_job_pipelines):
    def restore(_pipeline):
        raise RuntimeError("boom")

    fake_job_pipelines.restore = restore
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []

    _run_queue(gui_processor(on_progress=updates.append), [job], tmp_path, AppSettings())

    assert job.status is JobStatus.ERROR
    assert updates[-1].status is JobStatus.ERROR


@pytest.mark.parametrize(
    "choice,queued_after,answer,asked,replaced",
    [
        ("ask", 0, True, True, True),
        ("ask", 0, False, True, False),
        ("ask", 1, True, False, False),
        ("start_over", 1, False, False, True),
        ("skip", 0, True, False, False),
    ],
)
def test_outdated_progress_never_holds_up_videos_queued_after_it(
    tmp_path, fake_job_pipelines, choice, queued_after, answer, asked, replaced
):
    """#755: a question about outdated saved progress must not stall an unattended queue."""
    from concurrent.futures import Future

    from jasna.job_resume import ResumeMismatchError

    saved = tmp_path / "saved-progress"
    saved.mkdir()
    jobs = [JobItem(path=tmp_path / f"clip{index}.mp4") for index in range(1 + queued_after)]
    for job in jobs:
        job.path.touch()

    def restore(pipeline):
        if pipeline.input_path == jobs[0].path and saved.exists():
            raise ResumeMismatchError(saved)

    fake_job_pipelines.restore = restore
    questions: list[Path] = []

    def answer_question(directory):
        questions.append(directory)
        future = Future()
        future.set_result(answer)
        return future

    completions = []
    processor = gui_processor(on_resume_conflict=answer_question, on_complete=completions.append)
    processor.start(job_queue(jobs), AppSettings(saved_progress_conflict=choice), str(tmp_path), DEFAULT_OUTPUT_PATTERN)
    processor.join()

    assert not processor.is_running()
    assert questions == ([saved] if asked else [])
    assert saved.exists() is not replaced
    assert jobs[0].status is (JobStatus.COMPLETED if replaced else JobStatus.SKIPPED)
    assert all(job.status is JobStatus.COMPLETED for job in jobs[1:])
    assert completions == [True]


@pytest.mark.parametrize("file_conflict", ["auto_rename", "skip"])
@pytest.mark.parametrize("ranges", ["editor", "auto_split", "none", "basicvsrpp_editor"])
def test_saved_progress_wins_over_file_conflict(tmp_path, monkeypatch, file_conflict, ranges):
    """The processor's saved-work lookup derives the job exactly as running it does, also
    when ranges keep a source codec other than the chosen one."""
    import importlib
    from fractions import Fraction
    from unittest.mock import MagicMock

    import torch

    from jasna.media.splice import KeyframeIndex
    from jasna.mosaic_scan import MosaicSamples
    from jasna.pipeline import Pipeline
    from jasna.segments import SegmentRange

    metadata = video_metadata(fps=30, num_frames=180, video_width=1920, video_height=1080)
    for target, value in {
        "jasna.media.probe.get_video_meta_data": lambda *_a: metadata,
        "jasna.pipeline.get_video_meta_data": lambda *_a: metadata,
        "jasna.media.splice.validate_smart_render": lambda *_a, **_k: None,
        "jasna.media.splice.probe_keyframes": lambda *_a: KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180),
        "jasna.mosaic_scan.scan_mosaic_sizes": lambda *_a, **_k: MosaicSamples(
            times=(1.0, 3.0), scores=(0.9, 0.9), sizes=(0.05, 0.3), stride=1.0
        ),
        "jasna.mosaic.detection_registry.require_detection_model_weights": lambda name: tmp_path / "det.onnx",
        "jasna.session_factory.provide_restoration_models": lambda *_a, **_k: None,
        "jasna.pipeline.require_encodable_size": lambda *_a: None,
        "jasna.pipeline.device_used_bytes": lambda _device: 0,
        "jasna.pipeline.device_free_bytes": lambda _device: 0,
    }.items():
        module, name = target.rsplit(".", 1)
        monkeypatch.setattr(importlib.import_module(module), name, value)
    resumed = []

    def restore(pipeline, _metadata):
        work = pipeline.resume.directory / (
            "segments/0000-raw.nut" if ranges == "basicvsrpp_editor" else "ltx/000000.0-1-0.reference.pt"
        )
        resumed.append(work.exists())
        work.parent.mkdir(parents=True, exist_ok=True)
        work.write_bytes(b"work")
        if len(resumed) == 1:
            pipeline.cancel()

    monkeypatch.setattr(Pipeline, "_run_smart", restore)
    monkeypatch.setattr(Pipeline, "_run_ltx", restore)
    monkeypatch.setattr(Pipeline, "validate_metadata", lambda *_a: None)
    monkeypatch.setattr(Pipeline, "configure_vr", lambda *_a: None)

    monkeypatch.setattr(
        "jasna.gui.processor.build_video_session", lambda *_a, **_k: MagicMock(device=torch.device("cpu"))
    )
    monkeypatch.setattr("jasna.engine_compiler.ensure_engines_compiled", lambda *_a, **_k: None)
    job = JobItem(path=tmp_path / "clip.mp4")
    job.path.touch()
    if ranges in ("editor", "basicvsrpp_editor"):
        job.segments = (SegmentRange(2.5, 3.0),)
    settings = AppSettings(
        restoration_model="basicvsrpp" if ranges == "basicvsrpp_editor" else "ltx", ltx_auto_split=ranges == "auto_split", codec="av1", encoder_cq=38,
        vr_mode="off", file_conflict=file_conflict,
    )
    processor = gui_processor()

    _run_queue(processor, [job], tmp_path, settings)
    assert job.status is JobStatus.PENDING
    output = tmp_path / "clip_restored.mp4"
    output.touch()
    _run_queue(processor, [job], tmp_path, settings)

    assert resumed == [False, True]
    assert job.status is JobStatus.COMPLETED
    assert job.output_path == output


def test_control_bar_shows_pause_for_ltx_and_uses_pause_confirmation(tk_root, monkeypatch):
    from jasna.gui.control_bar import ControlBar
    from jasna.gui.locales import t

    asked: list[tuple[str, str]] = []
    stopped: list[bool] = []
    monkeypatch.setattr(
        "jasna.gui.control_bar.messagebox.askyesno", lambda title, message, **_kwargs: asked.append((title, message)) or True
    )
    bar = ControlBar(tk_root)
    bar.set_callbacks(on_stop=lambda: stopped.append(True))

    bar.set_resumable(True)
    assert bar._stop_btn.cget("text") == "Ⅱ"
    bar._handle_stop()
    assert asked == [(t("pause_confirm_title"), t("pause_confirm"))]
    assert stopped == [True]

    bar.set_resumable(False)
    assert bar._stop_btn.cget("text") == "■"


@pytest.mark.parametrize(
    ("name", "model", "segments", "resumable"),
    [
        ("clip.mp4", "basicvsrpp", (), False),
        ("clip.mp4", "basicvsrpp", ((1.0, 2.0),), True),
        ("clip.mp4", "ltx", (), True),
        ("photo.png", "ltx", (), False),
    ],
)
def test_ltx_and_segment_video_jobs_pause_instead_of_stopping(name, model, segments, resumable):
    from jasna.gui.validation import job_is_resumable
    from jasna.segments import SegmentRange

    job = JobItem(path=Path(name), segments=tuple(SegmentRange(*times) for times in segments))
    assert job_is_resumable(job, AppSettings(restoration_model=model)) is resumable


@pytest.mark.parametrize("cancel_before_show", [False, True])
def test_resume_confirmation_is_posted_to_main_thread_and_can_be_cancelled(jasna_app, monkeypatch, cancel_before_show):
    asked: list[bool] = []
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *_args, **_kwargs: asked.append(True) or True)

    answer = jasna_app._confirm_resume_restart(None)
    assert not answer.done()
    if cancel_before_show:
        assert answer.cancel()
    jasna_app._main_thread._run_pending()

    assert asked == ([] if cancel_before_show else [True])
    if not cancel_before_show:
        assert answer.result() is True


@pytest.mark.parametrize("queue_finished", [False, True])
def test_only_a_finished_queue_is_reported_as_completed(jasna_app, queue_finished):
    bar = jasna_app._control_bar
    bar.set_running(True)
    bar.update_progress("clip.mp4", 40.0)

    jasna_app._handle_complete(queue_finished)

    assert (bar._percent_label.cget("text") == "100%") is queue_finished
    assert ("All jobs completed" in [message for _, _, message in jasna_app._log_panel._entries]) is queue_finished


def test_changing_the_queue_keeps_the_running_jobs_progress(jasna_app, tmp_path):
    bar = jasna_app._control_bar
    bar.update_progress("clip.mp4", 40.0, queue_current=1, queue_total=1)
    later = tmp_path / "later.mp4"
    later.touch()

    jasna_app._queue_panel.add_jobs([JobItem(path=later)])

    texts = (bar._filename_label.cget("text"), bar._percent_label.cget("text"), bar._queue_progress.cget("text"))
    assert texts == ("clip.mp4", "40%", "1 / 1")


def test_a_failed_license_check_fails_that_job_and_stops_the_queue(tmp_path, fake_job_pipelines, caplog):
    from jasna.gui.locales import t
    from jasna.license_api import LicenseCheckStopped

    class ConnectionLost(LicenseCheckStopped):
        code = "expired"

    def restore(_pipeline):
        raise ConnectionLost("lost the connection")

    fake_job_pipelines.restore = restore
    job, later = JobItem(path=tmp_path / "clip.mp4"), JobItem(path=tmp_path / "later.mp4")
    updates: list[ProgressUpdate] = []
    completions = []
    processor = gui_processor(on_progress=updates.append, on_complete=completions.append)

    _run_queue(processor, [job, later], tmp_path, AppSettings())

    assert [pipeline.input_path for pipeline in fake_job_pipelines.built] == [job.path]
    assert job.status is JobStatus.ERROR and later.status is JobStatus.PENDING
    assert updates[-1].message == t("license_problem_expired")
    assert updates[-1].license_problem == "expired"
    assert completions == [False]
    assert [(record.levelname, record.getMessage()) for record in caplog.records if record.name == "jasna.gui.processor"] == [
        ("WARNING", "Failed to process clip.mp4: lost the connection")
    ]


def test_a_range_job_that_reencodes_the_whole_video_offers_stop_not_pause(tmp_path, fake_job_pipelines, monkeypatch):
    from unittest.mock import MagicMock

    import jasna.gui.processor as processor_module
    from jasna.segments import SegmentRange

    monkeypatch.setattr(processor_module, "splice_source", MagicMock())
    monkeypatch.setattr(
        processor_module, "plan_segment_job", MagicMock(return_value=MagicMock(reencodes_whole_video=True))
    )
    updates: list[ProgressUpdate] = []
    processor = gui_processor(on_progress=updates.append)
    job = JobItem(path=tmp_path / "clip.mp4", segments=(SegmentRange(2.5, 3.0),))
    job.path.touch()

    _run_queue(processor, [job], tmp_path, AppSettings())

    assert job.status is JobStatus.COMPLETED
    assert [update.resumable for update in updates if update.resumable is not None] == [True, False]


def test_rerun_replaces_the_output_its_stopped_run_left(tmp_path, fake_job_pipelines):
    def restore(pipeline):
        pipeline.output_path.touch()
        if len(fake_job_pipelines.built) == 1:
            pipeline.cancel()

    fake_job_pipelines.restore = restore
    job = JobItem(path=tmp_path / "clip.mp4")
    job.path.touch()
    processor = gui_processor()

    _run_queue(processor, [job], tmp_path, AppSettings(file_conflict="skip"))
    assert job.status is JobStatus.PENDING
    _run_queue(processor, [job], tmp_path, AppSettings(file_conflict="skip"))

    assert [pipeline.output_path for pipeline in fake_job_pipelines.built] == [tmp_path / "clip_restored.mp4"] * 2
    assert job.status is JobStatus.COMPLETED


def test_overwrite_never_writes_over_the_input(tmp_path, fake_job_pipelines):
    job = JobItem(path=tmp_path / "clip.mp4")
    job.path.touch()

    _run_queue(
        gui_processor(), [job], tmp_path,
        AppSettings(file_conflict="overwrite"), output_pattern="{original}.mp4",
    )

    assert [pipeline.output_path for pipeline in fake_job_pipelines.built] == [tmp_path / "clip (1).mp4"]
    assert job.status is JobStatus.COMPLETED

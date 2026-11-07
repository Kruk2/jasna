import threading
from pathlib import Path

import pytest

from jasna.gui.models import DEFAULT_OUTPUT_PATTERN, AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor, ProgressUpdate
from factories import no_resume_conflict

pytestmark = pytest.mark.usefixtures("no_gpu_cleanup", "tk_garbage_collected")


class _FakePipeline:
    def __init__(self):
        self.cancel_requested = False
        self.completed = False

    def cancel(self):
        self.cancel_requested = True


def _processor_with_job(tmp_path, run_pipeline):
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []
    processor = Processor(on_resume_conflict=no_resume_conflict, on_progress=updates.append)
    processor._run_pipeline = run_pipeline
    processor._jobs = [job]
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    return processor, job, updates


def _run_queue(processor, job, tmp_path, settings, output_pattern=DEFAULT_OUTPUT_PATTERN):
    processor.start([job], settings, str(tmp_path), output_pattern)
    processor.join()
    assert not processor.is_running()


def test_stopped_job_leaves_processing_state(tmp_path):
    from jasna.gui.processor import ProcessingStopped

    started = threading.Event()
    release = threading.Event()

    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        started.set()
        release.wait(5)
        if processor._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    worker = threading.Thread(target=processor._run, daemon=True)
    worker.start()
    assert started.wait(5)
    assert job.status is JobStatus.PROCESSING

    processor.stop()
    release.set()
    worker.join(5)

    assert not worker.is_alive()
    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING


def test_completed_job_stays_completed(tmp_path):
    processor, job, updates = _processor_with_job(
        tmp_path,
        lambda job_id, input_path, output_path, **kwargs: None,
    )

    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "clip_restored.mp4"
    assert updates[-1].status is JobStatus.COMPLETED


def test_stop_cancels_the_running_pipeline(tmp_path):
    processor, _job, _updates = _processor_with_job(
        tmp_path,
        lambda *args, **kwargs: None,
    )
    pipeline = _FakePipeline()
    processor._current_pipeline = pipeline

    processor.stop()

    assert pipeline.cancel_requested


def test_interrupted_pipeline_marks_job_pending(tmp_path):
    from jasna.gui.processor import ProcessingStopped

    runs = []

    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        runs.append(job_id)
        raise ProcessingStopped("Processing stopped")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    processor._run()

    assert runs == [job.id]  # a job left pending must not be picked up again

    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING


def test_failing_job_still_marked_error(tmp_path):
    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        raise RuntimeError("boom")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    processor._run()

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
    tmp_path, monkeypatch, choice, queued_after, answer, asked, replaced
):
    """#755: a question about outdated saved progress must not stall an unattended queue."""
    from concurrent.futures import Future
    from unittest.mock import MagicMock

    import torch

    from jasna.job_resume import ResumeMismatchError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir()
    (tmp_path / "model_weights" / "rfdetr-v6.onnx").touch()
    saved = tmp_path / "saved-progress"
    saved.mkdir()
    jobs = [JobItem(path=tmp_path / f"clip{index}.mp4") for index in range(1 + queued_after)]
    for job in jobs:
        job.path.touch()

    class SavedProgressPipeline:
        cancel_requested = False
        completed = True

        def __init__(self, input_path):
            self.input_path = input_path

        def run(self):
            if self.input_path == jobs[0].path and saved.exists():
                raise ResumeMismatchError(saved)

    monkeypatch.setattr(
        "jasna.gui.processor.build_job_pipeline",
        lambda _config, _session, input_path, *_a, **_k: SavedProgressPipeline(input_path),
    )
    monkeypatch.setattr(
        "jasna.gui.processor.build_video_session", lambda *_a, **_k: MagicMock(device=torch.device("cpu"))
    )
    monkeypatch.setattr("jasna.engine_compiler.ensure_engines_compiled", lambda *_a, **_k: None)
    questions: list[Path] = []

    def answer_question(directory):
        questions.append(directory)
        future = Future()
        future.set_result(answer)
        return future

    completions = []
    processor = Processor(on_resume_conflict=answer_question, on_complete=completions.append)
    processor.start(jobs, AppSettings(saved_progress_conflict=choice), str(tmp_path), DEFAULT_OUTPUT_PATTERN)
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
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    from jasna.media.splice import KeyframeIndex
    from jasna.mosaic_scan import MosaicSamples
    from jasna.pipeline import Pipeline
    from jasna.segments import SegmentRange

    metadata = SimpleNamespace(
        codec_name="hevc", duration=6.0, num_frames=180, time_base=Fraction(1, 30),
        video_width=1920, video_height=1080, sample_aspect_ratio=Fraction(1, 1),
    )
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

    processor, job, _updates = _processor_with_job(tmp_path, None)
    del processor._run_pipeline
    monkeypatch.setattr(
        "jasna.gui.processor.build_video_session", lambda *_a, **_k: MagicMock(device=torch.device("cpu"))
    )
    processor._prepare_job_detector = MagicMock()
    job.path.touch()
    if ranges in ("editor", "basicvsrpp_editor"):
        job.segments = (SegmentRange(2.5, 3.0),)
    processor._settings = AppSettings(
        restoration_model="basicvsrpp" if ranges == "basicvsrpp_editor" else "ltx", ltx_auto_split=ranges == "auto_split", codec="av1", encoder_cq=38,
        vr_mode="off", file_conflict=file_conflict,
    )

    processor._run()
    assert job.status is JobStatus.PENDING
    output = tmp_path / "clip_restored.mp4"
    output.touch()
    processor._run()

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


@pytest.mark.parametrize("cancel_requested,completed,expected", [
    (False, True, False),
    (True, False, True),
    (True, True, False),
])
def test_pipeline_stop_detection(cancel_requested, completed, expected):
    from jasna.gui.processor import _pipeline_was_stopped

    pipeline = _FakePipeline()
    pipeline.cancel_requested = cancel_requested
    pipeline.completed = completed

    assert _pipeline_was_stopped(pipeline) is expected


@pytest.mark.parametrize("queue_finished", [False, True])
def test_only_a_finished_queue_is_reported_as_completed(jasna_app, queue_finished):
    bar = jasna_app._control_bar
    bar.set_running(True)
    bar.update_progress("clip.mp4", 40.0)

    jasna_app._handle_complete(queue_finished)

    assert (bar._percent_label.cget("text") == "100%") is queue_finished
    assert ("All jobs completed" in [message for _, _, message in jasna_app._log_panel._entries]) is queue_finished


def test_a_failed_license_check_fails_that_job_and_stops_the_queue(tmp_path, caplog):
    from jasna.gui.locales import t
    from jasna.license_api import LicenseCheckStopped

    class ConnectionLost(LicenseCheckStopped):
        code = "expired"

    runs = []

    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        runs.append(job_id)
        raise ConnectionLost("lost the connection")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)
    later = JobItem(path=tmp_path / "later.mp4")
    processor._jobs.append(later)
    completions = []
    processor._on_complete = completions.append

    processor._run()

    assert runs == [job.id]
    assert job.status is JobStatus.ERROR and later.status is JobStatus.PENDING
    assert updates[-1].message == t("license_problem_expired")
    assert updates[-1].license_problem == "expired"
    assert completions == [False]
    assert [(record.levelname, record.getMessage()) for record in caplog.records if record.name == "jasna.gui.processor"] == [
        ("WARNING", "Failed to process clip.mp4: lost the connection")
    ]


def test_a_range_job_that_reencodes_the_whole_video_offers_stop_not_pause(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import torch

    import jasna.gui.processor as processor_module
    from jasna.segments import SegmentRange

    monkeypatch.setattr(processor_module, "splice_source", MagicMock())
    monkeypatch.setattr(
        processor_module, "plan_segment_job", MagicMock(return_value=MagicMock(reencodes_whole_video=True))
    )
    monkeypatch.setattr(
        processor_module, "build_job_pipeline", MagicMock(return_value=MagicMock(cancel_requested=False))
    )
    monkeypatch.setattr(
        "jasna.gui.processor.build_video_session", lambda *_a, **_k: MagicMock(device=torch.device("cpu"))
    )
    monkeypatch.setattr("jasna.engine_compiler.ensure_engines_compiled", lambda *_a, **_k: None)
    updates: list[ProgressUpdate] = []
    processor = Processor(on_resume_conflict=no_resume_conflict, on_progress=updates.append)
    job = JobItem(path=tmp_path / "clip.mp4", segments=(SegmentRange(2.5, 3.0),))
    job.path.touch()

    _run_queue(processor, job, tmp_path, AppSettings())

    assert job.status is JobStatus.COMPLETED
    assert [update.resumable for update in updates if update.resumable is not None] == [True, False]


def test_rerun_replaces_the_output_its_stopped_run_left(tmp_path):
    from jasna.gui.processor import ProcessingStopped

    outputs: list[Path] = []

    def run_pipeline(job_id, input_path, output_path, **kwargs):
        outputs.append(output_path)
        output_path.touch()
        if len(outputs) == 1:
            raise ProcessingStopped("Processing stopped")

    processor, job, _updates = _processor_with_job(tmp_path, run_pipeline)
    job.path.touch()

    _run_queue(processor, job, tmp_path, AppSettings(file_conflict="skip"))
    assert job.status is JobStatus.PENDING
    _run_queue(processor, job, tmp_path, AppSettings(file_conflict="skip"))

    assert outputs == [tmp_path / "clip_restored.mp4"] * 2
    assert job.status is JobStatus.COMPLETED


def test_overwrite_never_writes_over_the_input(tmp_path):
    outputs: list[Path] = []
    processor, job, _updates = _processor_with_job(tmp_path, lambda _id, _input, output, **_kw: outputs.append(output))
    job.path.touch()

    _run_queue(processor, job, tmp_path, AppSettings(file_conflict="overwrite"), output_pattern="{original}.mp4")

    assert outputs == [tmp_path / "clip (1).mp4"]
    assert job.status is JobStatus.COMPLETED

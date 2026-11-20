import threading
from pathlib import Path

from jasna.gui.processor import Processor
from jasna.gui.models import JobItem, AppSettings
from factories import gui_processor, job_queue


def _start(processor: Processor, folder: Path) -> None:
    processor.start(job_queue([JobItem(path=folder / "test.mp4")]), AppSettings(), output_folder=str(folder), output_pattern="{original}_restored.mp4")


def test_join_waits_for_the_running_job_and_is_safe_before_and_after(tmp_path, fake_job_pipelines):
    entered = threading.Event()
    release = threading.Event()
    fake_job_pipelines.restore = lambda _pipeline: entered.set() or release.wait(5)
    processor = gui_processor()
    processor.join(timeout=1.0)
    assert not processor.is_running()

    _start(processor, tmp_path)
    assert entered.wait(timeout=5.0)
    assert processor.is_running()
    processor.stop()
    release.set()
    processor.join(timeout=5.0)

    assert not processor.is_running()
    processor.join(timeout=1.0)
    assert not processor.is_running()


def test_processor_error_logs_full_traceback(tmp_path, fake_job_pipelines):
    def restore(_pipeline):
        raise ValueError("test error")

    fake_job_pipelines.restore = restore
    logs: list[tuple[str, str]] = []
    processor = gui_processor(on_log=lambda level, message: logs.append((level, message)))

    _start(processor, tmp_path)
    processor.join(timeout=5.0)

    error_logs = [message for level, message in logs if level == "ERROR"]
    assert len(error_logs) == 1
    assert "Failed to process test.mp4: test error" in error_logs[0]
    assert "Traceback (most recent call last):" in error_logs[0]
    assert "ValueError: test error" in error_logs[0]

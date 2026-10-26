import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from jasna.gui.processor import Processor
from jasna.gui.models import JobItem, AppSettings
from factories import no_resume_conflict

pytestmark = pytest.mark.usefixtures("no_gpu_cleanup", "tk_garbage_collected")


def _start(processor: Processor) -> None:
    processor.start([JobItem(path=Path("test.mp4"))], AppSettings(), output_folder="", output_pattern="{original}_restored.mp4")


def test_join_waits_for_the_running_job_and_is_safe_before_and_after():
    entered = threading.Event()
    release = threading.Event()
    processor = Processor(on_resume_conflict=no_resume_conflict)
    processor.join(timeout=1.0)
    assert not processor.is_running()

    with patch.object(processor, "_run_pipeline", side_effect=lambda *_args, **_kwargs: entered.set() or release.wait()):
        _start(processor)
        entered.wait(timeout=5.0)
        assert processor.is_running()
        processor.stop()
        release.set()
        processor.join(timeout=5.0)

    assert not processor.is_running()
    processor.join(timeout=1.0)
    assert not processor.is_running()


def test_processor_error_logs_full_traceback():
    logs: list[tuple[str, str]] = []
    processor = Processor(on_resume_conflict=no_resume_conflict, on_log=lambda level, message: logs.append((level, message)))

    with patch.object(processor, "_run_pipeline", side_effect=ValueError("test error")):
        _start(processor)
        processor.join(timeout=5.0)

    error_logs = [message for level, message in logs if level == "ERROR"]
    assert len(error_logs) == 1
    assert "Failed to process test.mp4: test error" in error_logs[0]
    assert "Traceback (most recent call last):" in error_logs[0]
    assert "ValueError: test error" in error_logs[0]

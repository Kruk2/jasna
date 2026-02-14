import threading
import time
from pathlib import Path
from unittest.mock import patch

from jasna.gui.processor import Processor, ProgressUpdate
from jasna.gui.models import JobStatus, JobItem, AppSettings


def test_processor_join_waits_for_thread():
    """stop() + join() should wait for the background thread to finish."""
    entered = threading.Event()
    p = Processor()

    class FakeJob:
        path = None
        filename = "a.mp4"

    class FakeSettings:
        pass

    # Monkey-patch _run so we control when the thread finishes
    original_run = p._run

    def slow_run():
        entered.set()
        time.sleep(0.3)

    p._run = slow_run
    p._stop_event.clear()
    p._thread = threading.Thread(target=p._run, daemon=True)
    p._thread.start()

    entered.wait(timeout=2)
    p.stop()
    p.join(timeout=5.0)

    assert not p.is_running()


def test_processor_join_noop_when_not_started():
    """join() should be safe to call when no thread was started."""
    p = Processor()
    p.join(timeout=1.0)


def test_processor_join_noop_after_thread_finished():
    """join() should be safe to call after thread has already exited."""
    p = Processor()

    p._run = lambda: None
    p._thread = threading.Thread(target=p._run, daemon=True)
    p._thread.start()
    p._thread.join()

    p.stop()
    p.join(timeout=1.0)
    assert not p.is_running()


def test_processor_error_logs_full_traceback():
    logs: list[tuple[str, str]] = []

    def on_log(level: str, message: str) -> None:
        logs.append((level, message))

    p = Processor(on_log=on_log)
    job = JobItem(path=Path("test.mp4"))
    settings = AppSettings()

    with patch.object(p, "_run_pipeline", side_effect=ValueError("test error")):
        p.start(
            [job],
            settings,
            output_folder="",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
            allow_unsafe_basicvsrpp_compile=False,
        )
        p.join(timeout=5.0)

    error_logs = [m for lvl, m in logs if lvl == "ERROR"]
    assert len(error_logs) == 1
    msg = error_logs[0]
    assert "Failed to process test.mp4: test error" in msg
    assert "Traceback (most recent call last):" in msg
    assert "ValueError: test error" in msg

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from jasna.gui import app as app_module
from jasna.gui import run_log as run_log_module
from jasna.gui.run_log import (
    AsyncRunLog,
    build_run_log_path,
    format_run_telemetry,
    read_run_telemetry,
)
from jasna.media.media_files import folder_media_in_processing_order


def _bare_app():
    app = object.__new__(app_module.JasnaApp)
    app._run_log = None
    app._run_telemetry_sampler = None
    app._closing_run_logs = []
    app._processor = None
    return app


def _run_log_settings(*, enabled: bool):
    return SimpleNamespace(
        save_run_log=enabled,
        detection_model="rfdetr",
        codec="hevc_amf",
        fp16_mode=True,
        vr_mode="2d",
        secondary_restoration="off",
        api_key="must-not-be-logged",
    )


def test_run_log_path_is_collision_resistant_and_scoped_to_output(tmp_path: Path) -> None:
    path = build_run_log_path(
        tmp_path,
        now=lambda: datetime(2026, 8, 16, 12, 34, 56, 123456),
        pid=42,
        token="abc-123",
    )

    assert path.parent == tmp_path / ".jasna-logs"
    assert path.name == "jasna-run-20260816-123456-123456-pid42-abc123.log"


def test_async_run_log_flushes_and_syncs_on_close(tmp_path: Path) -> None:
    writer = AsyncRunLog.start(
        tmp_path,
        flush_interval_seconds=0.05,
        sync_interval_seconds=0.05,
        token="test",
    )
    writer.enqueue("INFO", "first line\nsecond line")
    writer.close(timeout=2.0)

    assert not writer.is_running
    text = writer.path.read_text(encoding="utf-8")
    assert "Persistent run log requested" in text
    assert "first line" in text
    assert "second line" in text
    assert "Persistent run log closing" in text


def test_async_run_log_periodically_syncs_before_close(tmp_path: Path) -> None:
    synced = threading.Event()
    writer = AsyncRunLog.start(
        tmp_path,
        flush_interval_seconds=0.02,
        sync_interval_seconds=0.02,
        fsync=lambda _fd: synced.set(),
        token="periodic-sync",
    )
    writer.enqueue("INFO", "durable before a crash")

    assert synced.wait(2.0)
    writer.close(timeout=2.0)


def test_bounded_run_log_reports_dropped_events(tmp_path: Path) -> None:
    release_open = threading.Event()

    def delayed_open(path: Path):
        assert release_open.wait(2.0)
        return path.open("x", encoding="utf-8", buffering=1)

    writer = AsyncRunLog.start(
        tmp_path,
        queue_capacity=2,
        open_stream=delayed_open,
        token="bounded",
    )
    for index in range(8):
        writer.enqueue("INFO", f"storm-{index}")
    release_open.set()
    writer.close(timeout=2.0)

    text = writer.path.read_text(encoding="utf-8")
    assert "Run log dropped" in text


def test_run_log_failure_is_reported_without_raising(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    statuses: list[tuple[str, str]] = []

    writer = AsyncRunLog.start(blocked, on_status=lambda *item: statuses.append(item))
    deadline = time.monotonic() + 2.0
    while not writer.failed and time.monotonic() < deadline:
        time.sleep(0.01)
    writer.close(timeout=1.0)

    assert writer.failed
    assert any(level == "WARNING" for level, _message in statuses)


def test_linux_telemetry_reads_proc_and_amd_sysfs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_log_module.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 1024 kB\nMemAvailable: 256 kB\n",
        encoding="utf-8",
    )
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("1.00 2.00 3.00 1/2 3\n", encoding="utf-8")
    device = tmp_path / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "gpu_busy_percent").write_text("73\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("1024\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("4096\n", encoding="utf-8")

    telemetry = read_run_telemetry(
        proc_meminfo_path=meminfo,
        proc_loadavg_path=loadavg,
        drm_class_path=tmp_path / "drm",
    )

    assert telemetry.memory_used_bytes == 768 * 1024
    assert telemetry.gpu_busy_percent == 73
    assert telemetry.vram_total_bytes == 4096
    rendered = format_run_telemetry(telemetry, {"gui_pid": 7, "job": "video.mp4"})
    assert "gui_pid=7" in rendered
    assert "gpu_busy=73%" in rendered


def test_recursive_media_scan_ignores_diagnostic_log_directory(tmp_path: Path) -> None:
    visible = tmp_path / "video.mp4"
    visible.touch()
    hidden_dir = tmp_path / ".jasna-logs"
    hidden_dir.mkdir()
    (hidden_dir / "misleading.mp4").touch()

    assert folder_media_in_processing_order(tmp_path) == [visible]


def test_app_run_log_disabled_does_not_start_writer(monkeypatch) -> None:
    app = _bare_app()

    def unexpected_start(*_args, **_kwargs):
        raise AssertionError("run-log writer must stay disabled")

    monkeypatch.setattr(app_module.AsyncRunLog, "start", unexpected_start)

    assert app._start_run_log(_run_log_settings(enabled=False), [], "output", "{stem}") is None
    assert app._run_log is None
    assert app._run_telemetry_sampler is None


def test_app_run_log_lifecycle_is_bounded_and_does_not_log_secret_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeRunLog:
        path = tmp_path / "run.log"
        failed = False
        accepts_events = True

        def __init__(self):
            self.events: list[tuple[str, str]] = []
            self.close_calls: list[float] = []
            self.is_running = True

        def enqueue(self, level: str, message: str) -> bool:
            self.events.append((level, message))
            return True

        def close(self, timeout: float = 1.0) -> None:
            self.close_calls.append(timeout)
            if len(self.close_calls) > 1:
                self.is_running = False

    class FakeSampler:
        instance = None

        def __init__(self, on_log, *, is_active, context_provider):
            self.on_log = on_log
            self.is_active = is_active
            self.context_provider = context_provider
            self.started = False
            self.stop_calls: list[float] = []
            FakeSampler.instance = self

        def start(self) -> None:
            self.started = True

        def stop(self, timeout: float = 0.5) -> None:
            self.stop_calls.append(timeout)

    run_log = FakeRunLog()
    monkeypatch.setattr(app_module.AsyncRunLog, "start", lambda *_args, **_kwargs: run_log)
    monkeypatch.setattr(app_module, "RunTelemetrySampler", FakeSampler)
    app = _bare_app()
    app._on_run_log_status = lambda *_args: None

    path = app._start_run_log(
        _run_log_settings(enabled=True),
        [SimpleNamespace(path=tmp_path / "input.mp4")],
        str(tmp_path),
        "{stem}-restored",
    )

    assert path == run_log.path
    assert app._run_log is run_log
    assert FakeSampler.instance is app._run_telemetry_sampler
    assert FakeSampler.instance.started
    assert "must-not-be-logged" not in "\n".join(message for _level, message in run_log.events)
    messages = "\n".join(message for _level, message in run_log.events)
    if app_module.sys.platform == "win32":
        assert "Event Viewer System log" in messages
        assert "journalctl" not in messages
    else:
        assert "journalctl and pstore" in messages

    app._close_run_log()

    assert app._run_log is None
    assert app._run_telemetry_sampler is None
    assert FakeSampler.instance.stop_calls == [0.0]
    assert run_log.close_calls == [0.0]
    assert app._closing_run_logs == [run_log]

    app._wait_for_run_log_shutdown(timeout=0.25)

    assert len(run_log.close_calls) == 2
    assert 0.0 <= run_log.close_calls[-1] <= 0.25
    assert not run_log.is_running
    assert app._closing_run_logs == []


def test_app_run_log_start_failure_is_fail_open(monkeypatch) -> None:
    statuses: list[tuple[str, str]] = []
    app = _bare_app()
    app._on_run_log_status = lambda *item: statuses.append(item)
    monkeypatch.setattr(
        app_module.AsyncRunLog,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    assert app._start_run_log(_run_log_settings(enabled=True), [], "output", "{stem}") is None
    assert app._run_log is None
    assert app._run_telemetry_sampler is None
    assert statuses == [("WARNING", "Run log unavailable: disk unavailable")]

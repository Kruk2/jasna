from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_staged_run_log():
    path = Path(__file__).parents[1] / "jasna" / "gui" / "run_log.py"
    name = "_jasna_staged_run_log_windows_adapter"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_windows_run_log_adapts_valid_existing_system_stats(monkeypatch) -> None:
    run_log = _load_staged_run_log()
    monkeypatch.setattr(run_log.sys, "platform", "win32")
    from jasna.gui import system_stats

    calls: list[str] = []

    def read_stats():
        calls.append("read")
        return SimpleNamespace(gpu_util=71, vram_util=19, ram_util=46, cpu_util=23)

    monkeypatch.setattr(system_stats, "read_system_stats", read_stats)
    telemetry = run_log.read_run_telemetry()

    assert calls == ["read"]
    assert telemetry.telemetry_source == "windows-system-stats"
    assert telemetry.gpu_busy_percent == 71
    assert telemetry.cpu_util_percent == 23
    assert telemetry.ram_util_percent == 46
    assert telemetry.vram_util_percent == 19
    assert telemetry.memory_used_bytes is None
    assert telemetry.memory_total_bytes is None
    assert telemetry.vram_used_bytes is None
    assert telemetry.vram_total_bytes is None
    rendered = run_log.format_run_telemetry(telemetry)
    assert "telemetry_source=windows-system-stats" in rendered
    assert "gpu_busy=71%" in rendered
    assert "cpu_util=23%" in rendered
    assert "ram_util=46%" in rendered
    assert "vram_util=19%" in rendered
    assert "vram=unavailable/unavailable" in rendered


def test_windows_run_log_keeps_partial_or_invalid_stats_explicitly_unknown(monkeypatch) -> None:
    run_log = _load_staged_run_log()
    monkeypatch.setattr(run_log.sys, "platform", "win32")

    partial = run_log.read_run_telemetry(
        system_stats_reader=lambda: SimpleNamespace(
            gpu_util=None,
            vram_util=None,
            ram_util=42,
            cpu_util="unknown",
        )
    )
    invalid = run_log.read_run_telemetry(
        system_stats_reader=lambda: SimpleNamespace(
            gpu_util=True,
            vram_util=101,
            ram_util=42.5,
            cpu_util=False,
        )
    )

    assert partial.gpu_busy_percent is None
    assert partial.vram_util_percent is None
    assert partial.ram_util_percent == 42
    assert partial.cpu_util_percent is None
    rendered = run_log.format_run_telemetry(partial)
    assert "gpu_busy=unavailable" in rendered
    assert "cpu_util=unavailable" in rendered
    assert "ram_util=42%" in rendered
    assert "vram_util=unavailable" in rendered
    assert "=0%" not in rendered
    assert invalid.gpu_busy_percent is None
    assert invalid.vram_util_percent is None
    assert invalid.ram_util_percent is None
    assert invalid.cpu_util_percent is None
    invalid_rendered = run_log.format_run_telemetry(invalid)
    assert "gpu_busy=unavailable" in invalid_rendered
    assert "cpu_util=unavailable" in invalid_rendered
    assert "ram_util=unavailable" in invalid_rendered
    assert "vram_util=unavailable" in invalid_rendered


def test_windows_run_log_import_failure_is_fail_open(monkeypatch) -> None:
    run_log = _load_staged_run_log()
    monkeypatch.setattr(run_log.sys, "platform", "win32")
    original_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jasna.gui.system_stats":
            raise ImportError("simulated missing provider")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    telemetry = run_log.read_run_telemetry()

    assert telemetry.telemetry_source == "windows-system-stats"
    assert telemetry.gpu_busy_percent is None
    assert telemetry.cpu_util_percent is None
    assert telemetry.ram_util_percent is None
    assert telemetry.vram_util_percent is None
    rendered = run_log.format_run_telemetry(telemetry)
    assert "telemetry_source=windows-system-stats" in rendered
    assert "gpu_busy=unavailable" in rendered
    assert "cpu_util=unavailable" in rendered


def test_windows_run_log_keeps_missing_or_failed_stats_explicitly_unknown(monkeypatch) -> None:
    run_log = _load_staged_run_log()
    monkeypatch.setattr(run_log.sys, "platform", "win32")

    missing = run_log.read_run_telemetry(system_stats_reader=lambda: None)

    def failing_reader():
        raise RuntimeError("PDH unavailable")

    failed = run_log.read_run_telemetry(system_stats_reader=failing_reader)
    for telemetry in (missing, failed):
        assert telemetry.telemetry_source == "windows-system-stats"
        assert telemetry.gpu_busy_percent is None
        assert telemetry.cpu_util_percent is None
        assert telemetry.ram_util_percent is None
        assert telemetry.vram_util_percent is None
        rendered = run_log.format_run_telemetry(telemetry)
        assert "gpu_busy=unavailable" in rendered
        assert "cpu_util=unavailable" in rendered
        assert "ram_util=unavailable" in rendered
        assert "vram_util=unavailable" in rendered
        assert "=0%" not in rendered


def test_linux_run_log_path_does_not_call_windows_system_stats(monkeypatch, tmp_path: Path) -> None:
    run_log = _load_staged_run_log()
    monkeypatch.setattr(run_log.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1024 kB\nMemAvailable: 256 kB\n", encoding="utf-8")
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("1.00 2.00 3.00 1/2 3\n", encoding="utf-8")

    def unexpected_reader():
        raise AssertionError("Linux telemetry must not use the Windows provider")

    telemetry = run_log.read_run_telemetry(
        proc_meminfo_path=meminfo,
        proc_loadavg_path=loadavg,
        drm_class_path=tmp_path / "missing-drm",
        system_stats_reader=unexpected_reader,
    )

    assert telemetry.memory_used_bytes == 768 * 1024
    assert telemetry.load_1 == 1.0
    assert telemetry.telemetry_source is None
    assert "telemetry_source=" not in run_log.format_run_telemetry(telemetry)

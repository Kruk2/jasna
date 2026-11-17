from __future__ import annotations

from types import SimpleNamespace
import subprocess

import pytest

from jasna.gui import system_stats
from jasna.gui.components import format_duration as _format_duration
from jasna.gui.control_bar import _color_for_percent


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (45, "45s"), (90, "1m 30s"), (3599, "59m 59s"), (3600, "1h 0m"), (7323, "2h 2m")],
)
def test_format_duration_shows_the_two_largest_units(seconds, text) -> None:
    assert _format_duration(seconds) == text


def _rgb(color: str) -> tuple[int, ...]:
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


@pytest.mark.parametrize(("percent", "lower", "upper"), [(25, 0, 50), (75, 50, 100)])
def test_color_for_percent_blends_the_neighbouring_stops(percent, lower, upper) -> None:
    blended, low, high = _rgb(_color_for_percent(percent)), _rgb(_color_for_percent(lower)), _rgb(_color_for_percent(upper))

    assert all(min(a, b) <= value <= max(a, b) for value, a, b in zip(blended, low, high))
    assert blended not in (low, high)


def test_color_for_percent_clamps_outside_0_to_100() -> None:
    assert _color_for_percent(-20) == _color_for_percent(0)
    assert _color_for_percent(150) == _color_for_percent(100)


def test_parse_nvidia_smi_csv_line_parses_gpu_and_vram_pct() -> None:
    gpu, vram = system_stats._parse_nvidia_smi_csv_line("85, 1200, 2400")
    assert gpu == 85
    assert vram == 50


def test_read_gpu_vram_returns_none_when_gpu_tools_and_amd_sysfs_are_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)
    monkeypatch.setattr(system_stats, "_DRM_CLASS_PATH", tmp_path)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when nvidia-smi is missing")

    monkeypatch.setattr(system_stats.subprocess, "run", _should_not_run)

    gpu, vram = system_stats.read_gpu_vram()
    assert gpu is None
    assert vram is None


def test_read_gpu_vram_reads_amd_sysfs_without_torch(monkeypatch, tmp_path) -> None:
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "gpu_busy_percent").write_text("73\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("300\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("1200\n", encoding="utf-8")
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)
    monkeypatch.setattr(system_stats, "_DRM_CLASS_PATH", tmp_path)

    assert system_stats.read_gpu_vram() == (73, 25)


def test_read_gpu_vram_parses_first_device(monkeypatch) -> None:
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: "nvidia-smi")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="85, 1200, 2400\n", stderr="")

    monkeypatch.setattr(system_stats.subprocess, "run", _fake_run)

    gpu, vram = system_stats.read_gpu_vram()
    assert gpu == 85
    assert vram == 50


def test_read_cpu_ram_uses_psutil(monkeypatch) -> None:
    import psutil
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 23.4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=45.6))

    cpu, ram = system_stats.read_cpu_ram()
    assert cpu == 23
    assert ram == 46

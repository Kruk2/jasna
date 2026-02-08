from __future__ import annotations

from types import SimpleNamespace
import subprocess

from jasna.gui import system_stats
from jasna.gui.control_bar import _color_for_percent


def test_color_for_percent_green_at_zero() -> None:
    assert _color_for_percent(0) == "#34d399"


def test_color_for_percent_amber_at_50() -> None:
    assert _color_for_percent(50) == "#fbbf24"


def test_color_for_percent_rose_at_100() -> None:
    assert _color_for_percent(100) == "#fb7185"


def test_color_for_percent_interpolates_midpoints() -> None:
    c25 = _color_for_percent(25)
    assert c25.startswith("#") and len(c25) == 7
    c75 = _color_for_percent(75)
    assert c75.startswith("#") and len(c75) == 7
    assert c25 != c75


def test_parse_nvidia_smi_csv_line_parses_gpu_and_vram_pct() -> None:
    gpu, vram = system_stats._parse_nvidia_smi_csv_line("85, 1200, 2400")
    assert gpu == 85
    assert vram == 50


def test_read_gpu_vram_returns_none_when_nvidia_smi_missing(monkeypatch) -> None:
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when nvidia-smi is missing")

    monkeypatch.setattr(system_stats.subprocess, "run", _should_not_run)

    gpu, vram = system_stats.read_gpu_vram()
    assert gpu is None
    assert vram is None


def test_read_gpu_vram_parses_first_device(monkeypatch) -> None:
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: "nvidia-smi")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="85, 1200, 2400\n", stderr="")

    monkeypatch.setattr(system_stats.subprocess, "run", _fake_run)

    gpu, vram = system_stats.read_gpu_vram()
    assert gpu == 85
    assert vram == 50


def test_read_cpu_ram_uses_psutil(monkeypatch) -> None:
    monkeypatch.setattr(system_stats.psutil, "cpu_percent", lambda interval=None: 23.4)
    monkeypatch.setattr(system_stats.psutil, "virtual_memory", lambda: SimpleNamespace(percent=45.6))

    cpu, ram = system_stats.read_cpu_ram()
    assert cpu == 23
    assert ram == 46


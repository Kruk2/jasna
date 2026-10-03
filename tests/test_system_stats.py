from __future__ import annotations

from types import SimpleNamespace
import subprocess

from jasna.gui import system_stats
from jasna.gui.components import format_duration as _format_duration
from jasna.gui.control_bar import _color_for_percent


def test_format_duration_seconds_only() -> None:
    assert _format_duration(45) == "45s"
    assert _format_duration(0) == "0s"


def test_format_duration_minutes_and_seconds() -> None:
    assert _format_duration(90) == "1m 30s"
    assert _format_duration(3599) == "59m 59s"


def test_format_duration_hours() -> None:
    assert _format_duration(3600) == "1h 0m"
    assert _format_duration(7323) == "2h 2m"


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


def test_read_gpu_vram_returns_none_when_gpu_tools_and_amd_sysfs_are_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(system_stats.sys, "platform", "linux")
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
    monkeypatch.setattr(system_stats.sys, "platform", "linux")
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


def test_windows_gpu_percentages_match_amd_luid_and_aggregate_engines() -> None:
    marker = "luid_0x00000000_0x0000f7bf_phys_"
    gpu, vram = system_stats._windows_gpu_percentages(
        marker,
        24 * 1024**3,
        (
            ("pid_1_luid_0x00000000_0x0000F7BF_phys_0_eng_2_engtype_Compute", 62),
            ("pid_2_luid_0x00000000_0x0000F7BF_phys_0_eng_2_engtype_Compute", 23),
            ("pid_1_luid_0x00000000_0x0000F7BF_phys_0_eng_7_engtype_Video", 41),
            ("pid_3_luid_0x00000000_0x000114DA_phys_0_eng_0_engtype_3D", 99),
        ),
        (
            ("luid_0x00000000_0x0000F7BF_phys_0", 6 * 1024**3),
            ("luid_0x00000000_0x000114DA_phys_0", 12 * 1024**3),
        ),
    )

    assert gpu == 85
    assert vram == 25


def test_read_gpu_vram_uses_windows_amd_reader_without_nvidia_smi(
    monkeypatch,
) -> None:
    monkeypatch.setattr(system_stats.sys, "platform", "win32")
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)
    monkeypatch.setattr(system_stats, "_read_windows_amd_gpu", lambda: (71, 19))

    assert system_stats.read_gpu_vram() == (71, 19)


def test_read_cpu_ram_uses_psutil(monkeypatch) -> None:
    import psutil
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 23.4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=45.6))

    cpu, ram = system_stats.read_cpu_ram()
    assert cpu == 23
    assert ram == 46

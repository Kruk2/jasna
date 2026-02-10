import pytest

from jasna import os_utils


def test_parse_ffmpeg_major_version_parses_plain_semver() -> None:
    out = "ffmpeg version 8.0.1 Copyright (c) ..."
    assert os_utils._parse_ffmpeg_major_version(out) == 8


def test_parse_ffmpeg_major_version_parses_n_prefix() -> None:
    out = "ffprobe version n8.1.2-12-gdeadbeef Copyright (c) ..."
    assert os_utils._parse_ffmpeg_major_version(out) == 8


def test_parse_ffmpeg_major_version_parses_nightly_build_from_libavutil() -> None:
    out = "\n".join(
        [
            "ffmpeg version N-113224-gdeadbeef Copyright (c) ...",
            "libavutil      60.  3.100 / 60.  3.100",
        ]
    )
    assert os_utils._parse_ffmpeg_major_version(out) == 8


def test_check_required_executables_uses_expected_version_commands(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffmpeg":
            return type("R", (), {"returncode": 0, "stdout": "ffmpeg version 8.0.0", "stderr": ""})()
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 8.1.0", "stderr": ""})()
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    os_utils.check_required_executables()

    assert calls == [
        ["/fake/ffprobe", "-version"],
        ["/fake/ffmpeg", "-version"],
        ["/fake/mkvmerge", "--version"],
    ]


def test_check_required_executables_skips_ffmpeg_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        exe = os_utils.Path(cmd[0]).name
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    os_utils.check_required_executables(disable_ffmpeg_check=True)

    assert calls == [["/fake/mkvmerge", "--version"]]


def test_check_required_executables_logs_stdout_stderr_when_exe_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 8.0.0", "stderr": ""})()
        if exe == "ffmpeg":
            return type("R", (), {"returncode": 1, "stdout": "ffmpeg stdout", "stderr": "ffmpeg stderr"})()
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            os_utils.check_required_executables()

    assert any("ffmpeg failed" in rec.message and "ffmpeg stdout" in rec.message and "ffmpeg stderr" in rec.message for rec in caplog.records)


def test_check_required_executables_errors_on_old_ffmpeg(monkeypatch, capsys) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 8.0.0", "stderr": ""})()
        if exe == "ffmpeg":
            return type("R", (), {"returncode": 0, "stdout": "ffmpeg version 7.1.0", "stderr": ""})()
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as e:
        os_utils.check_required_executables()
    assert int(e.value.code) == 1

    captured = capsys.readouterr()
    assert "major version must be exactly 8" in captured.out


def test_check_required_executables_errors_on_newer_ffmpeg(monkeypatch, capsys) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 8.0.0", "stderr": ""})()
        if exe == "ffmpeg":
            return type("R", (), {"returncode": 0, "stdout": "ffmpeg version 9.0.0", "stderr": ""})()
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as e:
        os_utils.check_required_executables()
    assert int(e.value.code) == 1

    captured = capsys.readouterr()
    assert "major version must be exactly 8" in captured.out


def test_check_required_executables_errors_when_version_cannot_be_detected(monkeypatch, capsys) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version N-113224-gdeadbeef", "stderr": ""})()
        if exe == "ffmpeg":
            return type("R", (), {"returncode": 0, "stdout": "ffmpeg version N-113224-gdeadbeef", "stderr": ""})()
        if exe == "mkvmerge":
            return type("R", (), {"returncode": 0, "stdout": "mkvmerge v82.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as e:
        os_utils.check_required_executables()
    assert int(e.value.code) == 1

    captured = capsys.readouterr()
    assert "could not detect major version" in captured.out


def test_get_subprocess_startup_info_non_nt_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.os, "name", "posix", raising=False)
    assert os_utils.get_subprocess_startup_info() is None


def test_get_subprocess_startup_info_nt_sets_startf_flag(monkeypatch) -> None:
    class _StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0

    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTUPINFO", _StartupInfo, raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTF_USESHOWWINDOW", 1 << 0, raising=False)

    si = os_utils.get_subprocess_startup_info()
    assert si is not None
    assert si.dwFlags & (1 << 0)


def test_find_executable_prefers_bundled_when_frozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna.exe"), raising=False)
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: None)

    ffmpeg = tmp_path / "_internal" / "tools" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.write_bytes(b"")

    assert os_utils.find_executable("ffmpeg") == str(ffmpeg)


def test_find_executable_finds_bundled_mkvmerge_recursive(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna.exe"), raising=False)
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: None)

    mkvmerge = tmp_path / "_internal" / "mkvtoolnix" / "nested" / "mkvmerge.exe"
    mkvmerge.parent.mkdir(parents=True, exist_ok=True)
    mkvmerge.write_bytes(b"")

    assert os_utils.find_executable("mkvmerge") == str(mkvmerge)


def test_warn_if_windows_hardware_accelerated_gpu_scheduling_enabled_prints_when_enabled(monkeypatch, capsys) -> None:
    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Winreg:
        HKEY_LOCAL_MACHINE = object()

        @staticmethod
        def OpenKey(root, path):
            return _Key()

        @staticmethod
        def QueryValueEx(key, name):
            assert name == "HwSchMode"
            return (2, None)

    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setitem(os_utils.sys.modules, "winreg", _Winreg)

    os_utils.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled()
    out = capsys.readouterr().out
    assert "Hardware-accelerated GPU scheduling" in out


def test_check_windows_hardware_accelerated_gpu_scheduling_returns_false_when_enabled(monkeypatch) -> None:
    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Winreg:
        HKEY_LOCAL_MACHINE = object()

        @staticmethod
        def OpenKey(root, path):
            return _Key()

        @staticmethod
        def QueryValueEx(key, name):
            assert name == "HwSchMode"
            return (2, None)

    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setitem(os_utils.sys.modules, "winreg", _Winreg)

    ok, info = os_utils.check_windows_hardware_accelerated_gpu_scheduling()
    assert ok is False
    assert "Enabled" in info


def test_check_nvidia_gpu_returns_name_when_available_and_compute_ok(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (8, 0),
            get_device_name=lambda device: "RTX 4090",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_nvidia_gpu()
    assert ok is True
    assert result == "RTX 4090"


def test_check_nvidia_gpu_returns_no_cuda_when_unavailable(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_nvidia_gpu()
    assert ok is False
    assert result == "no_cuda"


def test_check_nvidia_gpu_returns_compute_too_low_when_below_min(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (6, 1),
            get_device_name=lambda device: "GTX 1060",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_nvidia_gpu()
    assert ok is False
    assert result == ("compute_too_low", 6, 1)


def test_check_nvidia_gpu_returns_ok_at_exactly_min_compute(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (7, 5),
            get_device_name=lambda device: "RTX 2070",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_nvidia_gpu()
    assert ok is True
    assert result == "RTX 2070"


def test_min_gpu_compute_constant() -> None:
    assert os_utils.MIN_GPU_COMPUTE == (7, 5)


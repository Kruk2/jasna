import os
import sys
import types

import pytest

from jasna import os_utils


class _FakeKernel32:
    def __init__(self) -> None:
        self.free = 0
        self.set_std: list[int] = []

    def FreeConsole(self) -> None:
        self.free += 1

    def SetStdHandle(self, which, handle) -> int:
        self.set_std.append(which)
        return 1


def _fake_windll(monkeypatch) -> _FakeKernel32:
    """Replace ctypes.windll so _redirect_std_streams_to_null's SetStdHandle/FreeConsole
    calls hit a recorder, never the live test process's real OS std handles."""
    import ctypes

    kernel32 = _FakeKernel32()
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    return kernel32


def test_redirect_std_streams_to_null_discards_writes(monkeypatch) -> None:
    # After FreeConsole the console handles are invalid; writes to the real streams raise
    # WinError 6. The redirect must make stray print()/writes no-ops, not crash.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "stdin", sys.stdin)
    _fake_windll(monkeypatch)

    os_utils._redirect_std_streams_to_null()

    print("discarded")            # must not raise
    sys.stderr.write("discarded")  # must not raise
    assert sys.stdin.read() == ""


def test_redirect_std_streams_to_null_repoints_os_std_handles_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "stdin", sys.stdin)
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: 0))
    kernel32 = _fake_windll(monkeypatch)

    os_utils._redirect_std_streams_to_null()

    assert kernel32.set_std == [
        os_utils.STD_INPUT_HANDLE,
        os_utils.STD_OUTPUT_HANDLE,
        os_utils.STD_ERROR_HANDLE,
    ]


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
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 8.1.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    os_utils.check_required_executables()

    assert calls == [
        ["/fake/ffprobe", "-version"],
    ]


def test_check_required_executables_logs_stdout_stderr_when_exe_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 1, "stdout": "ffprobe stdout", "stderr": "ffprobe stderr"})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            os_utils.check_required_executables()

    assert any("ffprobe failed" in rec.message and "ffprobe stdout" in rec.message and "ffprobe stderr" in rec.message for rec in caplog.records)


def test_check_required_executables_errors_on_old_ffprobe(monkeypatch, capsys) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 7.1.0", "stderr": ""})()
        raise AssertionError(f"Unexpected exe {exe!r}")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as e:
        os_utils.check_required_executables()
    assert int(e.value.code) == 1

    captured = capsys.readouterr()
    assert "major version must be exactly 8" in captured.out


def test_check_required_executables_errors_on_newer_ffprobe(monkeypatch, capsys) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")

    def fake_run(cmd, **kwargs):
        exe = os_utils.Path(cmd[0]).name
        if exe == "ffprobe":
            return type("R", (), {"returncode": 0, "stdout": "ffprobe version 9.0.0", "stderr": ""})()
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


def test_drop_console_window_non_win_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    os_utils.drop_console_window()  # must not touch ctypes/raise off Windows


def test_drop_console_window_dev_win_is_noop(monkeypatch) -> None:
    # In dev (not frozen) the console is the developer's terminal — must not detach it.
    import ctypes

    calls = {"free": 0}

    class _Windll:
        class kernel32:
            @staticmethod
            def FreeConsole() -> None:
                calls["free"] += 1

    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: False)
    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)

    os_utils.drop_console_window()

    assert calls["free"] == 0


def test_drop_console_window_frozen_win_calls_freeconsole(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "stdin", sys.stdin)
    kernel32 = _fake_windll(monkeypatch)

    os_utils.drop_console_window()

    assert kernel32.free == 1


def test_freeconsole_dangling_std_handles_break_subprocess_until_redirect(tmp_path) -> None:
    if sys.platform != "win32":
        pytest.skip("FreeConsole and the dangling-std-handle bug are Windows-only")

    import subprocess

    result_path = tmp_path / "result.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import ctypes, subprocess, sys\n"
        "from jasna.os_utils import _redirect_std_streams_to_null\n"
        "k = ctypes.windll.kernel32\n"
        "def popen_ok():\n"
        "    try:\n"
        "        p = subprocess.Popen([sys.executable, '-c', 'pass'],\n"
        "                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
        "        p.communicate()\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "k.FreeConsole(); k.AllocConsole(); k.FreeConsole()\n"  # OS std handles now dangle
        "before = popen_ok()\n"
        "_redirect_std_streams_to_null()\n"
        f"after = popen_ok()\n"
        f"open(r'{result_path}', 'w').write(f'{{before}},{{after}}')\n"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, str(child)], check=True, env=env)

    before, after = result_path.read_text().split(",")
    assert before == "False"  # bug reproduces: dangling stdin handle breaks Popen(stdin=None)
    assert after == "True"     # fix: NUL OS std handles let the child duplicate them


def test_subprocess_no_window_kwargs_non_nt_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.os, "name", "posix", raising=False)
    assert os_utils.subprocess_no_window_kwargs() == {}


def test_subprocess_no_window_kwargs_nt_sets_create_no_window(monkeypatch) -> None:
    class _StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0

    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTUPINFO", _StartupInfo, raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTF_USESHOWWINDOW", 1 << 0, raising=False)
    monkeypatch.setattr(os_utils.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    kwargs = os_utils.subprocess_no_window_kwargs()

    assert kwargs["creationflags"] == 0x08000000
    assert kwargs["startupinfo"] is not None
    assert kwargs["startupinfo"].dwFlags & (1 << 0)


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
    # Nuitka dist has no _internal/; bundled tools sit at the dist root (tools/).
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna"), raising=False)
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: None)
    monkeypatch.setattr(os_utils, "_bundled_exe_filename", lambda name: name)

    ffmpeg = tmp_path / "tools" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.write_bytes(b"")

    assert os_utils.find_executable("ffmpeg") == str(ffmpeg)


def test_find_executable_bundled_wins_over_system_path(monkeypatch, tmp_path) -> None:
    # A frozen release ships its own ffmpeg; it must use those even when a different
    # copy is on the user's PATH (otherwise a wrong-version system ffmpeg would be picked).
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna"), raising=False)
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: "/usr/bin/ffmpeg")
    monkeypatch.setattr(os_utils, "_bundled_exe_filename", lambda name: name)

    ffmpeg = tmp_path / "tools" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.write_bytes(b"")

    assert os_utils.find_executable("ffmpeg") == str(ffmpeg)


def test_check_sysmem_fallback_returns_true_when_prefer_no_sysmem(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        os_utils, "_read_drs_setting", lambda setting_id: os_utils._PREFER_NO_SYSMEM_FALLBACK
    )

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is True
    assert "Prefer No Sysmem Fallback" in info


def test_check_sysmem_fallback_returns_false_when_driver_default(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(os_utils, "_read_drs_setting", lambda setting_id: 0)

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is False
    assert "Driver Default" in info


def test_check_sysmem_fallback_returns_false_when_prefer_sysmem(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        os_utils, "_read_drs_setting", lambda setting_id: os_utils._PREFER_SYSMEM_FALLBACK
    )

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is False
    assert "Prefer Sysmem Fallback" in info


def test_check_sysmem_fallback_returns_false_when_setting_not_found(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(os_utils, "_read_drs_setting", lambda setting_id: None)

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is False
    assert "Driver Default" in info


def test_check_sysmem_fallback_returns_false_on_oserror(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)

    def _raise(setting_id):
        raise OSError("nvdrsdb0.bin not found")

    monkeypatch.setattr(os_utils, "_read_drs_setting", _raise)

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is False
    assert "nvdrsdb0.bin not found" in info


def test_check_sysmem_fallback_returns_na_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)

    ok, info = os_utils.check_windows_nvidia_sysmem_fallback_policy()
    assert ok is True
    assert info == "N/A"


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


def test_check_gpu_driver_version_passes_when_580(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "580.65\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is True
    assert info == "580.65"


def test_check_gpu_driver_version_passes_when_590(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "590.18\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is True
    assert info == "590.18"


def test_check_gpu_driver_version_passes_when_600(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "600.01\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is True
    assert info == "600.01"


def test_check_gpu_driver_version_fails_when_old(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "566.36\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert "566.36" in info
    assert "580" in info


def test_check_gpu_driver_version_fails_when_nvidia_smi_not_found(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: None)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert "not found" in info


def test_find_executable_falls_back_to_common_location_when_not_on_path(monkeypatch, tmp_path) -> None:
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text("")

    monkeypatch.setattr(os_utils.shutil, "which", lambda name: None)
    monkeypatch.setitem(os_utils._COMMON_EXECUTABLE_LOCATIONS, "nvidia-smi", (str(fake_smi),))

    assert os_utils.find_executable("nvidia-smi") == str(fake_smi)


def test_find_executable_returns_none_when_neither_path_nor_common(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda name: None)
    monkeypatch.setitem(os_utils._COMMON_EXECUTABLE_LOCATIONS, "nvidia-smi", ("/nope/nvidia-smi",))

    assert os_utils.find_executable("nvidia-smi") is None


def test_find_executable_prefers_path_over_common_locations(monkeypatch, tmp_path) -> None:
    on_path = tmp_path / "from-path"
    on_path.write_text("")
    fallback = tmp_path / "fallback"
    fallback.write_text("")

    monkeypatch.setattr(os_utils.shutil, "which", lambda name: str(on_path))
    monkeypatch.setitem(os_utils._COMMON_EXECUTABLE_LOCATIONS, "nvidia-smi", (str(fallback),))

    assert os_utils.find_executable("nvidia-smi") == str(on_path)


def test_check_gpu_driver_version_fails_when_nvidia_smi_errors(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert "exited with code" in info


def test_check_gpu_driver_version_fails_on_oserror(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert "permission denied" in info


def test_check_gpu_driver_version_fails_on_unparseable_output(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "garbage\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert "Could not parse" in info


def test_check_ascii_install_path_passes_for_ascii(monkeypatch, tmp_path) -> None:
    ascii_path = tmp_path / "jasna"
    ascii_path.mkdir()
    monkeypatch.setattr(os_utils, "__file__", str(ascii_path / "os_utils.py"))
    ok, info = os_utils.check_ascii_install_path()
    assert ok is True


def test_check_ascii_install_path_fails_for_non_ascii(monkeypatch, tmp_path) -> None:
    non_ascii_path = tmp_path / "プロジェクト"
    non_ascii_path.mkdir()
    monkeypatch.setattr(os_utils, "__file__", str(non_ascii_path / "os_utils.py"))
    ok, info = os_utils.check_ascii_install_path()
    assert ok is False
    assert "プロジェクト" in info


def test_check_ascii_install_path_uses_executable_when_frozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    exe_path = tmp_path / "jasna.exe"
    monkeypatch.setattr(os_utils.sys, "executable", str(exe_path), raising=False)
    ok, info = os_utils.check_ascii_install_path()
    assert ok is True

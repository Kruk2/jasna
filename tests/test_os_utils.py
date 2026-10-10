import os
import sys
import types

import pytest
import torch

from jasna import os_utils


@pytest.fixture(autouse=True)
def _pin_vendor_for_cuda_tests(request, monkeypatch):
    """Pin the vendor to NVIDIA for the CUDA-side tests of this module.

    `check_supported_gpu` / `check_gpu_driver_version` branch on the active
    vendor. The tests for the NVIDIA branch (compute capability, `nvidia-smi`
    parsing) mock `nvidia-smi` and CUDA capabilities, so on a ROCm build they
    would take the AMD branch instead and fail on the missing attributes. The
    AMD-branch tests set their own fake through `_as_amd_build`, which overrides
    this fixture, and are skipped here by name.
    """
    name = request.node.name
    if "amd" in name or "rocm" in name:
        return
    import jasna.accelerator as accelerator

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(
        accelerator, "vendor_for_device",
        lambda _device: accelerator.AcceleratorVendor.NVIDIA,
    )


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
    if before != "False":
        # The dangling-handle failure depends on the console/session the tests run
        # in; where Windows still lets the child inherit the freed handle there is
        # nothing to assert, so the precondition (not the fix) is what is missing.
        pytest.skip("the dangling-std-handle failure does not reproduce in this environment")
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


def test_check_supported_gpu_returns_name_when_available_and_compute_ok(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (8, 0),
            get_device_name=lambda device: "RTX 4090",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_supported_gpu()
    assert ok is True
    assert result == "RTX 4090"


def test_check_supported_gpu_returns_no_cuda_when_unavailable(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_supported_gpu()
    assert ok is False
    assert result == "no_cuda"


def test_check_supported_gpu_returns_compute_too_low_when_below_min(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (6, 1),
            get_device_name=lambda device: "GTX 1060",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_supported_gpu()
    assert ok is False
    assert result == ("compute_too_low", 6, 1)


def _fake_rocm_torch(gcn_arch_name: str, arch_list: list[str]):
    def _no_kernels(*_args, **_kwargs):
        raise AssertionError("the GPU check must not launch a kernel")

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(gcnArchName=gcn_arch_name),
            get_arch_list=lambda: arch_list,
            get_device_name=lambda device: "AMD Radeon 780M Graphics",
        ),
        zeros=_no_kernels,
    )


def _as_amd_build(monkeypatch, fake_torch) -> None:
    import jasna.accelerator as accelerator

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "vendor_for_device", lambda _device: accelerator.AcceleratorVendor.AMD)


def test_check_supported_gpu_rejects_amd_arch_missing_from_build(monkeypatch) -> None:
    _as_amd_build(monkeypatch, _fake_rocm_torch("gfx1103", ["gfx1100", "gfx1101", "gfx1102", "gfx1201"]))
    ok, result = os_utils.check_supported_gpu()
    assert ok is False
    assert result == ("arch_unsupported", "gfx1103")


def test_check_supported_gpu_accepts_amd_arch_ignoring_feature_suffixes(monkeypatch) -> None:
    _as_amd_build(monkeypatch, _fake_rocm_torch("gfx90a:sramecc+:xnack-", ["gfx90a:xnack-", "gfx1100"]))
    ok, result = os_utils.check_supported_gpu()
    assert ok is True
    assert result == "AMD Radeon 780M Graphics"


def test_check_supported_gpu_accepts_amd_when_build_reports_no_arch_list(monkeypatch) -> None:
    _as_amd_build(monkeypatch, _fake_rocm_torch("gfx1103", []))
    ok, _ = os_utils.check_supported_gpu()
    assert ok is True


def test_gpu_check_error_names_each_failure() -> None:
    assert "No compatible GPU" in os_utils.gpu_check_error("no_cuda")
    assert "gfx1103" in os_utils.gpu_check_error(("arch_unsupported", "gfx1103"))
    assert "6.1" in os_utils.gpu_check_error(("compute_too_low", 6, 1))


def test_check_supported_gpu_returns_ok_at_exactly_min_compute(monkeypatch) -> None:
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (7, 5),
            get_device_name=lambda device: "RTX 2070",
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    ok, result = os_utils.check_supported_gpu()
    assert ok is True
    assert result == "RTX 2070"


def test_nvidia_compatibility_alias_is_not_exposed() -> None:
    assert not hasattr(os_utils, "check_nvidia_gpu")


def test_min_gpu_compute_constant() -> None:
    assert os_utils.MIN_GPU_COMPUTE == (7, 5)


def test_min_driver_version_is_platform_specific() -> None:
    import sys
    assert os_utils.MIN_DRIVER_VERSION == (580 if sys.platform == "linux" else 610)


def test_check_gpu_driver_version_passes_at_minimum(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")
    version = f"{os_utils.MIN_DRIVER_VERSION}.00"

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": f"{version}\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is True
    assert info == version


def test_check_gpu_driver_version_passes_when_newer(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "611.12\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is True
    assert info == "611.12"


def test_check_gpu_driver_version_fails_below_minimum(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: "/fake/nvidia-smi")
    below = f"{os_utils.MIN_DRIVER_VERSION - 1}.99"

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": f"{below}\n", "stderr": ""})()

    monkeypatch.setattr(os_utils.subprocess, "run", fake_run)
    ok, info = os_utils.check_gpu_driver_version()
    assert ok is False
    assert below in info
    assert str(os_utils.MIN_DRIVER_VERSION) in info


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


def test_env_flag_unset_and_empty_use_default(monkeypatch) -> None:
    monkeypatch.delenv("JASNA_TEST_FLAG", raising=False)
    assert os_utils.env_flag("JASNA_TEST_FLAG") is False
    assert os_utils.env_flag("JASNA_TEST_FLAG", default=True) is True
    monkeypatch.setenv("JASNA_TEST_FLAG", "")
    assert os_utils.env_flag("JASNA_TEST_FLAG", default=True) is True


def test_env_flag_accepts_common_spellings(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("JASNA_TEST_FLAG", value)
        assert os_utils.env_flag("JASNA_TEST_FLAG") is True, value
    for value in ("0", "false", "NO", "off"):
        monkeypatch.setenv("JASNA_TEST_FLAG", value)
        assert os_utils.env_flag("JASNA_TEST_FLAG", default=True) is False, value


def test_env_flag_tolerates_cmd_exe_trailing_space(monkeypatch) -> None:
    # ``set JASNA_TEST_FLAG=1 && prog`` binds "1 " (cmd.exe folds the space before
    # &&); a literal `== "1"` read would silently keep the flag off, and the mirror
    # case would refuse to turn a default-on flag off.
    monkeypatch.setenv("JASNA_TEST_FLAG", "1 ")
    assert os_utils.env_flag("JASNA_TEST_FLAG") is True
    monkeypatch.setenv("JASNA_TEST_FLAG", "0 ")
    assert os_utils.env_flag("JASNA_TEST_FLAG", default=True) is False
    monkeypatch.setenv("JASNA_TEST_FLAG", "  TRUE  ")
    assert os_utils.env_flag("JASNA_TEST_FLAG") is True


def test_env_flag_unrecognised_value_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_TEST_FLAG", "maybe")
    assert os_utils.env_flag("JASNA_TEST_FLAG") is False
    assert os_utils.env_flag("JASNA_TEST_FLAG", default=True) is True


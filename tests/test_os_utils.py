import io
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from jasna import accelerator, os_utils

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def test_frozen_linux_application_executable_ignores_the_python_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils.sys, "executable", "/moved/app/python")
    monkeypatch.setattr(os_utils.os, "readlink", lambda path: "/moved/app/jasna.bin")

    assert os_utils.get_application_executable() == "/moved/app/jasna.bin"


@pytest.mark.parametrize("platform,frozen", [("linux", False), ("win32", False), ("darwin", True)])
def test_other_application_executables_keep_sys_executable(monkeypatch, platform, frozen) -> None:
    monkeypatch.setattr(os_utils, "is_frozen", lambda: frozen)
    monkeypatch.setattr(os_utils.sys, "platform", platform)
    monkeypatch.setattr(os_utils.sys, "executable", "/application/python")

    assert os_utils.get_application_executable() == "/application/python"


@pytest.mark.parametrize("path", ["C:/relocated/app/jasna.exe", "C:/" + "folder/" * 100 + "jasna.exe"])
def test_frozen_windows_application_executable_uses_complete_running_module_path(monkeypatch, path) -> None:
    import ctypes
    from ctypes import wintypes

    calls = []
    def module_path(module, buffer, size):
        calls.append((module, size))
        buffer.value = path[:size - 1]
        return len(path) if len(path) < size else size
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: types.SimpleNamespace(GetModuleFileNameW=module_path), raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.sys, "executable", "C:/relocated/app/python.exe")

    assert os_utils.get_application_executable() == path
    assert all(module is None for module, size in calls)
    assert calls[-1][1] > len(path)
    assert module_path.argtypes == [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    assert module_path.restype is wintypes.DWORD


def test_frozen_windows_application_executable_reports_module_lookup_failure(monkeypatch) -> None:
    import ctypes

    def module_path(module, buffer, size):
        return 0
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: types.SimpleNamespace(GetModuleFileNameW=module_path), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda error: OSError(error, "module path unavailable"), raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.sys, "executable", "C:/relocated/app/python.exe")

    with pytest.raises(OSError, match="module path unavailable") as failure:
        os_utils.get_application_executable()
    assert failure.value.errno == 5


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native module path API")
def test_frozen_windows_application_executable_ignores_placeholder_using_native_api(monkeypatch) -> None:
    expected = sys.executable
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "executable", "C:/relocated/app/python.exe")

    assert os_utils.get_application_executable().casefold() == expected.casefold()


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


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.mark.parametrize(
    "output",
    [
        "ffmpeg version 8.0.1 Copyright (c) ...",
        "ffprobe version n8.1.2-12-gdeadbeef Copyright (c) ...",
        "ffmpeg version N-113224-gdeadbeef Copyright (c) ...\nlibavutil      60.  3.100 / 60.  3.100",
    ],
    ids=["semver", "n-prefix", "nightly-libavutil"],
)
def test_parse_ffmpeg_major_version(output) -> None:
    assert os_utils._parse_ffmpeg_major_version(output) == 8


def test_check_required_executables_accepts_ffprobe_8(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")
    calls = []
    monkeypatch.setattr(os_utils.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or _completed("ffprobe version 8.1.0"))

    os_utils.check_required_executables()

    assert calls == [["/fake/ffprobe", "-version"]]


@pytest.mark.parametrize(
    ("version_output", "message"),
    [
        ("ffprobe version 7.1.0", "major version must be exactly 8"),
        ("ffprobe version 9.0.0", "major version must be exactly 8"),
        ("ffprobe version N-113224-gdeadbeef", "could not detect major version"),
    ],
)
def test_check_required_executables_rejects_other_ffprobe_versions(monkeypatch, capsys, version_output, message) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")
    monkeypatch.setattr(os_utils.subprocess, "run", lambda cmd, **kwargs: _completed(version_output))

    with pytest.raises(SystemExit) as exit_info:
        os_utils.check_required_executables()

    assert int(exit_info.value.code) == 1
    assert message in capsys.readouterr().out


def test_check_required_executables_logs_stdout_stderr_when_exe_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda exe: f"/fake/{exe}")
    monkeypatch.setattr(
        os_utils.subprocess, "run", lambda cmd, **kwargs: _completed("ffprobe stdout", 1, "ffprobe stderr")
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        os_utils.check_required_executables()

    assert any(
        "ffprobe failed" in rec.message and "ffprobe stdout" in rec.message and "ffprobe stderr" in rec.message
        for rec in caplog.records
    )


@pytest.mark.parametrize(("platform", "frozen", "frees"), [("linux", True, 0), ("win32", False, 0), ("win32", True, 1)])
def test_drop_console_window_detaches_only_a_frozen_windows_app(monkeypatch, platform, frozen, frees) -> None:
    # In dev the console is the developer's terminal and must stay attached.
    monkeypatch.setattr(os_utils.sys, "platform", platform, raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: frozen)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "stdin", sys.stdin)
    kernel32 = _fake_windll(monkeypatch)

    os_utils.drop_console_window()

    assert kernel32.free == frees


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


def test_subprocess_window_options_are_empty_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.os, "name", "posix", raising=False)

    assert os_utils.subprocess_no_window_kwargs() == {}
    assert os_utils.get_subprocess_startup_info() is None


def test_subprocess_window_options_hide_the_window_on_windows(monkeypatch) -> None:
    class _StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0

    monkeypatch.setattr(os_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTUPINFO", _StartupInfo, raising=False)
    monkeypatch.setattr(os_utils.subprocess, "STARTF_USESHOWWINDOW", 1 << 0, raising=False)
    monkeypatch.setattr(os_utils.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    kwargs = os_utils.subprocess_no_window_kwargs()

    assert kwargs["creationflags"] == 0x08000000
    assert kwargs["startupinfo"].dwFlags & (1 << 0)
    assert os_utils.get_subprocess_startup_info().dwFlags & (1 << 0)


@pytest.mark.parametrize("system_ffmpeg", [None, "/usr/bin/ffmpeg"])
def test_frozen_app_uses_its_bundled_executable_even_with_one_on_path(monkeypatch, tmp_path, system_ffmpeg) -> None:
    # A frozen release ships its own ffmpeg; a wrong-version system copy must not win.
    monkeypatch.setattr(os_utils.sys, "frozen", True, raising=False)
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna"), raising=False)
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: system_ffmpeg)
    monkeypatch.setattr(os_utils, "_bundled_exe_filename", lambda name: name)
    ffmpeg = tmp_path / "tools" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"")

    assert os_utils.find_executable("ffmpeg") == str(ffmpeg)


@pytest.mark.parametrize(
    ("on_path", "fallback_exists", "found"),
    [(False, True, "fallback"), (False, False, None), (True, True, "from-path")],
    ids=["common-location", "missing", "path-first"],
)
def test_find_executable_falls_back_to_common_locations(monkeypatch, tmp_path, on_path, fallback_exists, found) -> None:
    (tmp_path / "from-path").write_text("")
    if fallback_exists:
        (tmp_path / "fallback").write_text("")
    monkeypatch.setattr(os_utils.shutil, "which", lambda name: str(tmp_path / "from-path") if on_path else None)
    monkeypatch.setitem(os_utils._COMMON_EXECUTABLE_LOCATIONS, "nvidia-smi", (str(tmp_path / "fallback"),))

    assert os_utils.find_executable("nvidia-smi") == (str(tmp_path / found) if found else None)


def _raise_oserror(setting_id):
    raise OSError("nvdrsdb0.bin not found")


@pytest.mark.parametrize(
    ("read_setting", "ok", "info"),
    [
        (lambda setting_id: os_utils._PREFER_NO_SYSMEM_FALLBACK, True, "Prefer No Sysmem Fallback"),
        (lambda setting_id: 0, False, "Driver Default"),
        (lambda setting_id: os_utils._PREFER_SYSMEM_FALLBACK, False, "Prefer Sysmem Fallback"),
        (lambda setting_id: None, False, "Driver Default"),
        (_raise_oserror, False, "nvdrsdb0.bin not found"),
    ],
    ids=["prefer-no-sysmem", "driver-default", "prefer-sysmem", "not-set", "unreadable"],
)
def test_check_sysmem_fallback_policy_reads_the_nvidia_setting(monkeypatch, read_setting, ok, info) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(os_utils, "_read_drs_setting", read_setting)

    result_ok, result_info = os_utils.check_windows_nvidia_sysmem_fallback_policy()

    assert result_ok is ok
    assert info in result_info


def test_check_sysmem_fallback_skips_nvidia_drs_on_amd(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: True)

    def _read_drs_setting(setting_id):
        raise AssertionError("AMD must not read the NVIDIA DRS database")

    monkeypatch.setattr(os_utils, "_read_drs_setting", _read_drs_setting)

    assert os_utils.check_windows_nvidia_sysmem_fallback_policy() == (True, "N/A (AMD)")


def test_check_sysmem_fallback_returns_na_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)

    assert os_utils.check_windows_nvidia_sysmem_fallback_policy() == (True, "N/A")


@pytest.mark.parametrize(
    ("available", "capability", "name", "expected"),
    [
        (True, (8, 0), "RTX 4090", (True, "RTX 4090")),
        (True, (7, 5), "RTX 2070", (True, "RTX 2070")),
        (True, (6, 1), "GTX 1060", (False, ("compute_too_low", 6, 1))),
        (False, None, None, (False, "no_cuda")),
    ],
    ids=["supported", "minimum", "too-old", "no-cuda"],
)
def test_check_supported_gpu_needs_compute_capability_7_5(monkeypatch, available, capability, name, expected) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            get_device_capability=lambda device: capability,
            get_device_name=lambda device: name,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert os_utils.check_supported_gpu() == expected


def _fake_rocm_torch(gcn_arch_name: str, arch_list: list[str], torch_dir: Path):
    def _no_kernels(*_args, **_kwargs):
        raise AssertionError("the GPU check must not launch a kernel")

    return types.SimpleNamespace(
        __file__=str(torch_dir / "__init__.py"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(gcnArchName=gcn_arch_name),
            get_arch_list=lambda: arch_list,
            get_device_name=lambda device: "AMD Radeon 780M Graphics",
        ),
        zeros=_no_kernels,
    )


@pytest.mark.parametrize(
    ("arch", "arch_list", "kernel_packs", "expected"),
    [
        ("gfx1103", ["gfx1100", "gfx1101", "gfx1102", "gfx1201"], (), (False, ("arch_unsupported", "gfx1103"))),
        ("gfx90a:sramecc+:xnack-", ["gfx90a:xnack-", "gfx1100"], (), (True, "AMD Radeon 780M Graphics")),
        ("gfx1103", [], (), (True, "AMD Radeon 780M Graphics")),
        ("gfx1031", ["gfx1030", "gfx1031", "gfx1201"], ("gfx1030", "gfx1201"), (False, ("arch_unsupported", "gfx1031"))),
        ("gfx1031", ["gfx1030", "gfx1031", "gfx1201"], ("gfx1031",), (True, "AMD Radeon 780M Graphics")),
    ],
    ids=["missing-from-build", "feature-suffixes", "no-arch-list", "kernel-pack-missing", "kernel-pack-installed"],
)
def test_check_supported_gpu_needs_an_amd_arch_the_build_contains(
    monkeypatch, tmp_path, arch, arch_list, kernel_packs, expected,
) -> None:
    (tmp_path / ".kpack").mkdir()
    for architecture in kernel_packs:
        (tmp_path / ".kpack" / f"torch_{architecture}.kpack").touch()
    fake_torch = _fake_rocm_torch(arch, arch_list, tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "vendor_for_device", lambda _device: accelerator.AcceleratorVendor.AMD)

    assert os_utils.check_supported_gpu() == expected


def test_gpu_check_error_names_each_failure() -> None:
    assert "No compatible GPU" in os_utils.gpu_check_error("no_cuda")
    assert "gfx1103" in os_utils.gpu_check_error(("arch_unsupported", "gfx1103"))
    assert "6.1" in os_utils.gpu_check_error(("compute_too_low", 6, 1))


def _nvidia_smi_raises(cmd, **kwargs):
    raise OSError("permission denied")


@pytest.mark.parametrize(
    ("smi", "run", "ok", "info"),
    [
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(f"{os_utils.MIN_DRIVER_VERSION}.00\n"), True, f"{os_utils.MIN_DRIVER_VERSION}.00"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed("611.12\n"), True, "611.12"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(f"{os_utils.MIN_DRIVER_VERSION - 1}.99\n"), False,
         f"{os_utils.MIN_DRIVER_VERSION - 1}.99 (requires {os_utils.MIN_DRIVER_VERSION}+)"),
        (None, None, False, "nvidia-smi not found"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(returncode=1), False, "nvidia-smi exited with code 1"),
        ("/fake/nvidia-smi", _nvidia_smi_raises, False, "permission denied"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed("garbage\n"), False, "Could not parse driver version: 'garbage'"),
    ],
    ids=["minimum", "newer", "too-old", "no-nvidia-smi", "smi-fails", "smi-unrunnable", "unparseable"],
)
def test_check_gpu_driver_version(monkeypatch, smi, run, ok, info) -> None:
    monkeypatch.setattr(os_utils, "find_executable", lambda name: smi)
    monkeypatch.setattr(os_utils.subprocess, "run", run)

    assert os_utils.check_gpu_driver_version() == (ok, info)


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
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils, "__file__", str(tmp_path / "プロジェクト" / "os_utils.py"))
    monkeypatch.setattr(os_utils.sys, "executable", str(tmp_path / "jasna.exe"))
    assert os_utils.check_ascii_install_path() == (True, str(tmp_path))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows module lookup")
def test_windows_module_loaded_reports_modules_of_this_process() -> None:
    assert os_utils.windows_module_loaded("kernel32.dll") is True
    assert os_utils.windows_module_loaded("jasna-not-a-loaded-module.dll") is False


def test_windows_module_loaded_is_false_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux")

    assert os_utils.windows_module_loaded("kernel32.dll") is False


def _sysfs_disk(root, name: str, rotational: str):
    disk = root / "devices" / name
    (disk / "queue").mkdir(parents=True)
    (disk / "queue" / "rotational").write_text(rotational + "\n")
    partition = disk / f"{name}1"
    partition.mkdir()
    (partition / "partition").write_text("1\n")
    return disk, partition


@pytest.mark.skipif(sys.platform == "win32", reason="sysfs device names like 8:1 are not valid Windows file names")
def test_partition_on_a_spinning_disk_reports_its_disk_as_rotational(tmp_path) -> None:
    _hdd, hdd_partition = _sysfs_disk(tmp_path, "sda", "1")
    ssd, _ssd_partition = _sysfs_disk(tmp_path, "nvme0n1", "0")
    links = tmp_path / "dev" / "block"
    links.mkdir(parents=True)
    (links / "8:1").symlink_to(hdd_partition)
    (links / "259:0").symlink_to(ssd)

    assert os_utils.sysfs_block_is_rotational(links / "8:1")
    assert not os_utils.sysfs_block_is_rotational(links / "259:0")
    assert not os_utils.sysfs_block_is_rotational(links / "0:79")


def test_mount_source_is_the_innermost_mount_holding_the_path() -> None:
    mountinfo = "\n".join([
        "29 1 0:26 / / rw,relatime shared:1 - btrfs /dev/mapper/root rw,subvol=/root",
        "40 29 8:2 / /mnt/media rw,relatime shared:5 - ext4 /dev/sda2 rw",
        "41 40 0:50 / /mnt/media/old\\040disk rw,relatime shared:6 - btrfs /dev/sdb1 rw",
        "42 29 0:79 / /mnt/pool rw,relatime shared:7 - fuse.mergerfs pool rw",
    ])

    assert os_utils.mount_source(mountinfo, Path("/mnt/media/old disk/jasna")) == "/dev/sdb1"
    assert os_utils.mount_source(mountinfo, Path("/mnt/media/jasna")) == "/dev/sda2"
    assert os_utils.mount_source(mountinfo, Path("/mnt/mediaX/jasna")) == "/dev/mapper/root"
    assert os_utils.mount_source(mountinfo, Path("/mnt/pool/jasna")) == "pool"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mounts")
def test_a_mount_device_missing_in_a_container_leaves_the_btrfs_device(monkeypatch, tmp_path) -> None:
    anonymous = os.makedev(0, 79)
    real_stat = os.stat

    def stat(path, **kwargs):
        if Path(path) == tmp_path:
            return os.stat_result((0o40755, 1, anonymous, 1, 0, 0, 0, 0, 0, 0))
        return real_stat(path, **kwargs)

    monkeypatch.setattr(os_utils.os, "stat", stat)
    monkeypatch.setattr(os_utils, "mount_source", lambda mountinfo, path: "/dev/mapper/luks-not-in-this-container")
    assert os_utils._linux_block_device(tmp_path) == anonymous


@pytest.mark.skipif(sys.platform != "win32", reason="Windows storage query")
def test_windows_seek_penalty_query_answers_for_a_drive_and_skips_network_paths() -> None:
    assert isinstance(os_utils._windows_drive_has_seek_penalty(Path(os.environ["SystemDrive"] + "\\")), bool)
    assert not os_utils._windows_drive_has_seek_penalty(Path(r"\\server\share\jasna"))


@pytest.mark.parametrize("platform,library", [("linux", "libamfrt64.so.1"), ("win32", "amfrt64.dll")])
def test_amf_runtime_check_loads_the_platform_library(monkeypatch, platform, library) -> None:
    import ctypes

    loaded = []
    monkeypatch.setattr(os_utils.sys, "platform", platform)
    monkeypatch.setattr(ctypes, "CDLL", loaded.append)

    assert os_utils.check_amd_amf_runtime() == (True, library)
    assert loaded == [library]


@pytest.mark.parametrize('error', [OSError, ImportError])
def test_amf_runtime_check_reports_the_load_error(monkeypatch, error) -> None:
    import ctypes

    def missing(name):
        raise error(f"{name}: cannot open shared object file")

    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", missing)

    assert os_utils.check_amd_amf_runtime() == (False, "libamfrt64.so.1: cannot open shared object file")


def test_exit_without_joining_threads_runs_atexit_handlers_and_skips_stuck_threads(tmp_path) -> None:
    marker = tmp_path / "released"
    script = (
        "import atexit, threading, pathlib\n"
        "from jasna import os_utils\n"
        "threading.Thread(target=threading.Event().wait).start()\n"
        f"atexit.register(pathlib.Path({str(marker)!r}).write_text, 'released')\n"
        "os_utils.exit_without_joining_threads()\n"
    )

    result = subprocess.run([sys.executable, "-c", script], timeout=60)

    assert result.returncode == 0
    assert marker.read_text() == "released"


def test_escaped_output_keeps_the_code_page_and_escapes_what_it_cannot_encode() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")

    os_utils.escape_unencodable_output(stream)
    print("Running post-export command for café ż.mp4", file=stream)
    stream.flush()

    assert raw.getvalue() == b"Running post-export command for caf\xe9 \\u017c.mp4\n"


def test_tvai_setup_problems_name_each_missing_folder_and_ffmpeg(tmp_path, monkeypatch):
    from jasna.os_utils import tvai_setup_problems

    monkeypatch.delenv("TVAI_MODEL_DATA_DIR", raising=False)
    monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path / "models"))
    ffmpeg = tmp_path / "ffmpeg.exe"

    assert [(problem.key, problem.path) for problem in tvai_setup_problems(str(ffmpeg))] == [
        ("data_dir_not_set", ""),
        ("model_dir_missing", str(tmp_path / "models")),
        ("ffmpeg_not_found", str(ffmpeg)),
    ]
    (tmp_path / "models").mkdir()
    ffmpeg.touch()
    monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
    assert tvai_setup_problems(str(ffmpeg)) == []

import io
import os
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath

import pytest

from jasna import os_utils

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

    def module_path(module, buffer, size):
        buffer.value = path[:size - 1]
        return len(path) if len(path) < size else size
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: types.SimpleNamespace(GetModuleFileNameW=module_path), raising=False)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.sys, "executable", "C:/relocated/app/python.exe")

    assert os_utils.get_application_executable() == path


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

    def FreeConsole(self) -> None:
        self.free += 1

    def SetStdHandle(self, which, handle) -> int:
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


@pytest.mark.parametrize(
    ("library_path", "expected"),
    [
        ("/opt/jasna/nvvfx/libs:/usr/local/cuda/lib64", "/usr/local/cuda/lib64"),
        ("/home/me/lib:/opt/jasna-other/lib", "/home/me/lib:/opt/jasna-other/lib"),
        # OpenCV's loader adds an unnormalized path and a trailing empty entry.
        ("/opt/jasna/cv2/../../lib64:/opt/jasna/nvvfx/libs:", None),
        (None, None),
    ],
)
def test_library_path_keeps_only_folders_outside_the_app(library_path, expected) -> None:
    environment = {"HOME": "/home/me"}
    if library_path is not None:
        environment["LD_LIBRARY_PATH"] = library_path

    child = os_utils.without_app_library_path(environment, PurePosixPath("/opt/jasna"))

    assert child.get("LD_LIBRARY_PATH") == expected
    assert child["HOME"] == "/home/me"


def test_frozen_linux_host_programs_lose_the_bundle_library_folders(monkeypatch) -> None:
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils.sys, "executable", "/opt/jasna/jasna")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/jasna/nvvfx/libs:/usr/local/cuda/lib64")

    assert os_utils.host_program_environment()["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"


def test_windows_host_programs_inherit_the_environment(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32")

    assert os_utils.host_program_environment() is None


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows module lookup")
def test_windows_module_loaded_reports_modules_of_this_process() -> None:
    assert os_utils.windows_module_loaded("kernel32.dll") is True
    assert os_utils.windows_module_loaded("jasna-not-a-loaded-module.dll") is False


def test_windows_module_loaded_is_false_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux")

    assert os_utils.windows_module_loaded("kernel32.dll") is False


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


def _closed_stdin(prompt):
    raise EOFError


@pytest.mark.parametrize(
    ("isatty", "answer", "expected"),
    [
        (True, lambda prompt: " Yes ", True),
        (True, lambda prompt: "", False),
        (False, lambda prompt: "y", None),
        (True, _closed_stdin, None),
    ],
    ids=["yes", "default-no", "no-terminal", "nul-stdin"],
)
def test_confirm_in_terminal(monkeypatch, isatty, answer, expected):
    monkeypatch.setattr(os_utils.sys.stdin, "isatty", lambda: isatty)
    monkeypatch.setattr("builtins.input", answer)
    assert os_utils.confirm_in_terminal("Continue? [y/N]: ") is expected


def test_scratch_folder_is_removed_after_its_body_and_when_the_body_raises(tmp_path):
    with os_utils.scratch_folder(tmp_path, ".ltx-") as folder:
        (folder / "latents.npy").write_bytes(b"x")
        assert folder.parent == tmp_path and folder.name.startswith(".ltx-")
    with pytest.raises(RuntimeError, match="crashed"):
        with os_utils.scratch_folder(tmp_path, ".ltx-"):
            raise RuntimeError("crashed")

    assert list(tmp_path.iterdir()) == []


def test_scratch_folder_removes_leftovers_with_its_prefix_only(tmp_path):
    (tmp_path / ".ltx-crashed").mkdir()
    (tmp_path / ".ltx-crashed" / "plans.npz").write_bytes(b"x")
    (tmp_path / ".jasna-seeds-other").mkdir()
    (tmp_path / "video.ltx-notes.txt").write_text("keep")

    folder = os_utils.make_scratch_folder(tmp_path, ".ltx-")

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [folder.name, ".jasna-seeds-other", "video.ltx-notes.txt"]
    )


def test_a_leftover_that_cannot_be_removed_logs_a_warning_and_a_folder_is_still_made(monkeypatch, tmp_path, caplog):
    (tmp_path / ".ltx-locked").mkdir()

    def locked(path):
        raise PermissionError(f"in use: {path}")

    monkeypatch.setattr(os_utils.shutil, "rmtree", locked)

    folder = os_utils.make_scratch_folder(tmp_path, ".ltx-")

    assert folder.is_dir() and (tmp_path / ".ltx-locked").is_dir()
    assert any(record.levelname == "WARNING" and ".ltx-locked" in record.getMessage() for record in caplog.records)


def _dacl_is_protected(path: Path) -> bool:
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD),
    ]
    file_object, dacl_information, dacl_protected = 1, 4, 0x1000
    descriptor = ctypes.c_void_p()
    assert advapi32.GetNamedSecurityInfoW(
        str(path), file_object, dacl_information, None, None, None, None, ctypes.byref(descriptor)
    ) == 0
    control, revision = wintypes.WORD(), wintypes.DWORD()
    try:
        assert advapi32.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision))
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)
    return bool(control.value & dacl_protected)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows folder permissions")
def test_scratch_folder_inherits_the_parent_permissions_on_windows(tmp_path):
    with os_utils.scratch_folder(tmp_path, ".ltx-") as folder:
        assert not _dacl_is_protected(folder)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX folder modes")
def test_scratch_folder_has_the_default_folder_mode(tmp_path):
    (tmp_path / "plain").mkdir()
    with os_utils.scratch_folder(tmp_path, ".ltx-") as folder:
        assert folder.stat().st_mode == (tmp_path / "plain").stat().st_mode

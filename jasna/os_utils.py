import atexit
import logging
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from jasna._frozen import is_frozen

logger = logging.getLogger(__name__)

# A100, H100, B200. TensorRT builds engines only with its builder resource for the GPU's own SM,
# and releases may leave these out.
DATACENTER_NVIDIA_COMPUTE = frozenset({(8, 0), (9, 0), (10, 0)})


def get_application_executable() -> str:
    """Return the running application for subprocess launches."""
    if sys.platform == "linux" and is_frozen():
        # Without Nuitka's multiprocessing plugin, sys.executable names a
        # CPython placeholder in the distribution rather than the compiled app.
        return os.readlink("/proc/self/exe")
    if sys.platform == "win32" and is_frozen():
        import ctypes
        from ctypes import wintypes

        module_path = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
        module_path.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        module_path.restype = wintypes.DWORD
        size = 260
        while True:
            buffer = ctypes.create_unicode_buffer(size)
            length = module_path(None, buffer, size)
            if not length:
                raise ctypes.WinError(ctypes.get_last_error())
            if length < size:
                return buffer.value
            size *= 2
    return sys.executable


def _bundled_exe_filename(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _find_bundled_executable(name: str) -> Path | None:
    if not is_frozen():
        return None

    exe = _bundled_exe_filename(name)
    # Nuitka standalone has no _internal/ (a PyInstaller-ism); bundled tools sit at the
    # dist root next to the binary, alongside model_weights/ and assets/.
    base = Path(sys.executable).parent

    if name in {"ffmpeg", "ffprobe"}:
        p = base / "tools" / exe
        return p if p.is_file() else None

    return None


_COMMON_EXECUTABLE_LOCATIONS: dict[str, tuple[str, ...]] = {
    "nvidia-smi": (
        # Windows
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        # Linux / WSL
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
    ),
}


def _find_in_common_locations(name: str) -> str | None:
    for candidate in _COMMON_EXECUTABLE_LOCATIONS.get(name, ()):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_executable(name: str) -> str | None:
    bundled = _find_bundled_executable(name)
    if bundled is not None:
        return str(bundled)
    found = shutil.which(name)
    if found is not None:
        return found
    return _find_in_common_locations(name)


def resolve_executable(name: str) -> str:
    found = find_executable(name)
    return found if found is not None else name



def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12


def _redirect_std_streams_to_null() -> None:
    """Point sys.stdout/stderr/stdin AND the process's OS-level standard handles at NUL.

    After FreeConsole the inherited console handles are dead, and two things break.
    First, a stray `print()` or a logging StreamHandler writing to the real stream raises
    `OSError: [WinError 6] The handle is invalid` — repointing sys.* at NUL discards those
    writes. Second — and this is what randomly fails the system check and later crashes
    ffprobe — `subprocess.Popen` calls `GetStdHandle()` for every stream the
    caller left as None (our tools run with stdout/stderr=PIPE but stdin unset), then
    `DuplicateHandle`s that dangling handle and dies with `[WinError 6]`/`[WinError 50]`.
    Repointing the OS std handles at NUL gives each child a valid, inheritable handle to
    duplicate; fixing sys.* alone does not."""
    null_r = open(os.devnull, "r")
    null_w = open(os.devnull, "w")
    sys.stdin = null_r
    sys.stdout = null_w
    sys.stderr = null_w
    if os.name != "nt":
        return
    import ctypes
    import msvcrt

    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle(STD_INPUT_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_r.fileno())))
    kernel32.SetStdHandle(STD_OUTPUT_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_w.fileno())))
    kernel32.SetStdHandle(STD_ERROR_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_w.fileno())))


def drop_console_window() -> None:
    """Detach the console on a frozen GUI launch (Windows). The frozen binary is built
    console-subsystem (`--windows-console-mode=force`) so the CLI blocks cmd and stdout/stderr
    work; for a GUI/double-click launch we don't want that console window, so release it.
    No-op off Windows and in dev, where the console is the developer's own terminal."""
    if sys.platform != "win32" or not is_frozen():
        return
    import ctypes

    ctypes.windll.kernel32.FreeConsole()
    _redirect_std_streams_to_null()


def escape_unencodable_output(*streams) -> None:
    """Keep each stream's encoding but write characters it cannot encode as escapes. A Windows
    pipe uses the locale code page (e.g. cp1252), so printing a non-ASCII filename would crash."""
    for stream in streams:
        if stream is not None:
            stream.reconfigure(errors="backslashreplace")


def confirm_in_terminal(prompt: str) -> bool | None:
    """``True`` for a y/yes answer, ``None`` when there is no terminal to ask in (scheduled or
    scripted runs). Windows reports a NUL stdin as a terminal; reading it ends at once."""
    if not sys.stdin.isatty():
        return None
    try:
        answer = input(prompt)
    except EOFError:
        return None
    return answer.strip().lower() in ("y", "yes")


def exit_without_joining_threads() -> None:
    """End the process without waiting for non-daemon threads that CUDA/TensorRT may leave running, after the atexit
    handlers a normal exit runs (freeing the online license slot, stopping LTX workers, removing code caches)."""
    atexit._run_exitfuncs()
    os._exit(0)


def subprocess_no_window_kwargs() -> dict:
    """Popen/run kwargs that suppress a child's console window on Windows.

    The GUI drops its own console (FreeConsole), so a console-subsystem child like
    ffmpeg/ffprobe would otherwise pop its own cmd window. We capture their output
    via pipes, so they never need an inherited console. No-op (empty) off Windows."""
    if os.name != "nt":
        return {}
    return {
        "startupinfo": get_subprocess_startup_info(),
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


def without_app_library_path(environment: Mapping[str, str], app_dir: PurePosixPath) -> dict[str, str]:
    """``environment`` without the empty ``LD_LIBRARY_PATH`` entries and those written as paths
    inside ``app_dir``.

    Bundled packages add them when imported: NVIDIA VFX its library folder, OpenCV a folder next to
    the app plus an empty entry, which means the working directory, often the app folder. A system
    program started with them loads Jasna's copies, e.g. systemctl fails on the bundled libcrypto."""
    child = dict(environment)
    user_entries = [
        entry
        for entry in child.get("LD_LIBRARY_PATH", "").split(":")
        if entry and not PurePosixPath(entry).is_relative_to(app_dir)
    ]
    if user_entries:
        child["LD_LIBRARY_PATH"] = ":".join(user_entries)
    else:
        child.pop("LD_LIBRARY_PATH", None)
    return child


def host_program_environment() -> dict[str, str] | None:
    """Environment for programs of the user's system that Jasna starts (shutdown, the user's own
    commands, the file manager); ``None`` keeps the inherited one off Linux."""
    if sys.platform != "linux":
        return None
    app_dir = PurePosixPath(sys.executable).parent if is_frozen() else PurePosixPath(sys.prefix)
    return without_app_library_path(os.environ, app_dir)


def windows_module_loaded(name: str) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    get_module_handle = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleHandleW
    get_module_handle.argtypes = [ctypes.c_wchar_p]
    get_module_handle.restype = ctypes.c_void_p
    return bool(get_module_handle(name))


def windows_powershell() -> str:
    return str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe")


def check_amd_amf_runtime() -> tuple[bool, str]:
    """AMD decode/encode needs AMD's AMF runtime: the Windows driver ships it, Linux needs AMD's AMF installer."""
    import ctypes

    library = "amfrt64.dll" if sys.platform == "win32" else "libamfrt64.so.1"
    try:
        ctypes.CDLL(library)
    except (OSError, ImportError) as e:
        return False, str(e)
    return True, library


class TvaiSetupProblem(NamedTuple):
    key: str
    """data_dir_not_set, data_dir_missing, model_dir_not_set, model_dir_missing or ffmpeg_not_found"""
    message: str
    path: str


def tvai_setup_problems(ffmpeg_path: str) -> list[TvaiSetupProblem]:
    """What stops Topaz Video from running: its model folder variables and its ffmpeg."""
    problems = []
    for variable, name in (("TVAI_MODEL_DATA_DIR", "data_dir"), ("TVAI_MODEL_DIR", "model_dir")):
        directory = os.environ.get(variable)
        if not directory:
            problems.append(TvaiSetupProblem(f"{name}_not_set", f"{variable} environment variable is not set", ""))
        elif not Path(directory).is_dir():
            problems.append(TvaiSetupProblem(f"{name}_missing", f"{variable} is not a directory: {directory}", directory))
    if not Path(ffmpeg_path).is_file():
        problems.append(TvaiSetupProblem("ffmpeg_not_found", f"TVAI ffmpeg not found: {ffmpeg_path}", ffmpeg_path))
    return problems


def get_user_config_dir(app_name: str) -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / app_name
        return Path.home() / app_name

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / app_name
    return Path.home() / ".config" / app_name


def make_scratch_folder(parent: Path, prefix: str) -> Path:
    """Make a new ``prefix…`` folder in ``parent`` after removing the ones crashed runs left there.
    A plain mkdir inherits the parent's permissions; tempfile's owner-only folders cannot be
    deleted from Explorer on Windows."""
    for leftover in parent.glob(f"{prefix}*"):
        if leftover.is_dir():
            try:
                shutil.rmtree(leftover)
            except OSError as error:
                logger.warning("Could not remove leftover folder %s: %s", leftover, error)
    folder = parent / f"{prefix}{secrets.token_hex(4)}"
    folder.mkdir()
    return folder


@contextmanager
def scratch_folder(parent: Path, prefix: str):
    folder = make_scratch_folder(parent, prefix)
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


@contextmanager
def atomic_write(path: Path, mode: str):
    """Write ``path`` through a temporary file renamed over it, so a reader or a crash never
    leaves a partial ``path``."""
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        with partial.open(mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    commit_file(partial, path)


def commit_file(partial: Path, path: Path) -> None:
    partial.replace(path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

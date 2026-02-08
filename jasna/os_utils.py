import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _bundled_exe_filename(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _find_bundled_executable(name: str) -> Path | None:
    if not getattr(sys, "frozen", False):
        return None

    exe = _bundled_exe_filename(name)
    base = Path(sys.executable).parent
    internal = base / "_internal"

    if name in {"ffmpeg", "ffprobe"}:
        p = internal / "tools" / exe
        return p if p.is_file() else None

    if name == "mkvmerge":
        root = internal / "mkvtoolnix"
        direct = root / exe
        if direct.is_file():
            return direct
        if root.is_dir():
            for candidate in root.rglob(exe):
                if candidate.is_file():
                    return candidate
        return None

    return None


def find_executable(name: str) -> str | None:
    bundled = _find_bundled_executable(name)
    if bundled is not None:
        return str(bundled)
    return shutil.which(name)


def resolve_executable(name: str) -> str:
    found = find_executable(name)
    return found if found is not None else name



def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


def _parse_ffmpeg_major_version(version_output: str) -> int:
    def parse_major_from_libavutil(out: str) -> int:
        m = re.search(r"(?m)^\s*libavutil\s+(\d+)\.", out)
        if not m:
            raise ValueError("Could not parse libavutil version from ffmpeg output")
        libavutil_major = int(m.group(1))
        return libavutil_major - 52

    first_line = version_output.splitlines()[0] if version_output else ""
    m = re.match(r"^\s*(?:ffmpeg|ffprobe)\s+version\s+(\S+)", first_line)
    if not m:
        raise ValueError(f"Unexpected ffmpeg/ffprobe version output: {first_line!r}")

    ver = m.group(1)
    if ver.startswith(("N-", "git-", "GIT-")):
        return parse_major_from_libavutil(version_output)

    m = re.search(r"(\d+)", ver)
    if not m:
        raise ValueError(f"Could not parse ffmpeg major version from: {ver!r}")
    return int(m.group(1))


FFMPEG_DOWNLOAD_LINKS = (
    "Linux: https://www.ffmpeg.org/download.html  "
    "Windows: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z"
)


def check_required_executables(disable_ffmpeg_check: bool = False) -> None:
    """Check that required external tools are available in PATH and callable."""
    missing: list[str] = []
    wrong_version: list[str] = []
    checks = {
        "ffprobe": ["-version"],
        "ffmpeg": ["-version"],
        "mkvmerge": ["--version"],
    }
    if disable_ffmpeg_check:
        checks = {k: v for k, v in checks.items() if k != "ffprobe" and k != "ffmpeg"}
    for exe, args in checks.items():
        exe_path = find_executable(exe)
        if exe_path is None:
            missing.append(exe)
            continue
        cmd = [exe_path] + list(args)
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            missing.append(exe)
            continue
        if completed.returncode != 0:
            missing.append(exe)
            continue

        if exe in {"ffprobe", "ffmpeg"}:
            try:
                major = _parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            except ValueError:
                wrong_version.append(f"{exe} (could not detect major version)")
                continue
            if major != 8:
                wrong_version.append(f"{exe} (detected major={major})")

    if missing:
        msg = f"Error: Required executable(s) not found in PATH or not callable: {', '.join(missing)}"
        print(msg)
        logger.info("%s", msg)
        print("Please install them and ensure they are available in your system PATH and runnable.")
        logger.info("Please install them and ensure they are available in your system PATH and runnable.")
        if "ffmpeg" in missing or "ffprobe" in missing:
            print(FFMPEG_DOWNLOAD_LINKS)
            logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)
    if wrong_version:
        msg = (
            "Error: ffmpeg/ffprobe major version must be exactly 8 (or detectable as 8): "
            + ", ".join(wrong_version)
        )
        print(msg)
        logger.info("%s", msg)
        print(FFMPEG_DOWNLOAD_LINKS)
        logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)


def warn_if_windows_hardware_accelerated_gpu_scheduling_enabled() -> None:
    ok, _ = check_windows_hardware_accelerated_gpu_scheduling()
    if not ok:
        msg = (
            "Warning: Windows 'Hardware-accelerated GPU scheduling' is enabled. "
            "This will make Jasna slower and might add artifacts to the output video."
        )
        print(msg)
        logger.info("%s", msg)


def check_windows_hardware_accelerated_gpu_scheduling() -> tuple[bool, str]:
    if sys.platform != "win32":
        return True, "N/A"

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
        ) as key:
            mode, _ = winreg.QueryValueEx(key, "HwSchMode")
    except OSError:
        return True, "Not detected"

    if int(mode) == 2:
        return False, "Enabled (recommended OFF: can slow Jasna and add artifacts)"
    return True, "Off"


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


import os
import shutil
import subprocess
import sys


def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


def _parse_ffmpeg_major_version(version_output: str) -> int:
    first_line = version_output.splitlines()[0] if version_output else ""
    parts = first_line.split()
    if len(parts) < 3 or parts[1] != "version":
        raise ValueError(f"Unexpected ffmpeg/ffprobe version output: {first_line!r}")
    raw = parts[2]
    raw = raw.lstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    raw = raw.split("-", 1)[0]
    major_str = raw.split(".", 1)[0]
    return int(major_str)


def check_required_executables() -> None:
    """Check that required external tools are available in PATH and callable."""
    missing: list[str] = []
    wrong_version: list[str] = []
    checks = {
        "ffprobe": ["ffprobe", "-version"],
        "ffmpeg": ["ffmpeg", "-version"],
        "mkvmerge": ["mkvmerge", "--version"],
    }
    for exe, cmd in checks.items():
        if shutil.which(exe) is None:
            missing.append(exe)
            continue
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
            major = _parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            if major != 8:
                wrong_version.append(f"{exe} (detected major={major})")

    if missing:
        print(f"Error: Required executable(s) not found in PATH or not callable: {', '.join(missing)}")
        print("Please install them and ensure they are available in your system PATH and runnable.")
        sys.exit(1)
    if wrong_version:
        print(f"Error: ffmpeg/ffprobe major version must be exactly 8: {', '.join(wrong_version)}")
        sys.exit(1)


def warn_if_windows_hardware_accelerated_gpu_scheduling_enabled() -> None:
    if sys.platform != "win32":
        return

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
        ) as key:
            mode, _ = winreg.QueryValueEx(key, "HwSchMode")
    except OSError:
        return

    if int(mode) == 2:
        print(
            "Warning: Windows 'Hardware-accelerated GPU scheduling' is enabled. "
            "This will make Jasna slower and might add artifacts to the output video."
        )


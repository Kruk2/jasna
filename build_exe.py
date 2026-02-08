import os
import shutil
import subprocess
import sys
from pathlib import Path
import stat
import urllib.request
import zipfile

distpath = "dist_linux" if os.name != "nt" else "dist"
env = os.environ.copy()
subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", distpath, "jasna.spec"], check=True, env=env)
if os.name == "nt":
    env["BUILD_CLI"] = "1"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", distpath, "jasna.spec"], check=True, env=env)

if os.name == "nt":
    cli_exe = "jasna-cli.exe"
    shutil.copy(Path(distpath) / "jasna-cli" / cli_exe, Path(distpath) / "jasna" / cli_exe)

out = Path(distpath) / "jasna"
(out / "model_weights").mkdir(parents=True, exist_ok=True)
for name in [
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "rfdetr-v3.onnx",
]:
    shutil.copy(Path("model_weights") / name, out / "model_weights" / name)

internal = out / "_internal"
tools_dir = internal / "tools"
tools_dir.mkdir(parents=True, exist_ok=True)

def _parse_ldd_paths(output: str) -> list[str]:
    paths: list[str] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if line == "" or "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        right = right.strip()
        if right.startswith("not found"):
            continue
        p = right.split("(", 1)[0].strip()
        if p.startswith("/"):
            paths.append(p)
    return paths


for tool in ["ffmpeg", "ffprobe"]:
    tool_path = shutil.which(tool)
    if tool_path is None:
        raise FileNotFoundError(f"Could not find {tool!r} in PATH while building.")
    dst = tools_dir / Path(tool_path).name
    shutil.copy2(tool_path, dst)
    if os.name != "nt":
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

if os.name != "nt":
    tools_lib_dir = tools_dir / "lib"
    tools_lib_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("Could not find 'ffmpeg' in PATH while building.")
    prefix = Path(ffmpeg_path).resolve().parent.parent

    deps: set[Path] = set()
    for tool in ["ffmpeg", "ffprobe"]:
        tool_path = shutil.which(tool)
        if tool_path is None:
            raise FileNotFoundError(f"Could not find {tool!r} in PATH while building.")
        completed = subprocess.run(["ldd", tool_path], capture_output=True, text=True, check=True)
        for p in _parse_ldd_paths(completed.stdout):
            dep = Path(p)
            if dep.exists() and dep.is_file() and dep.is_relative_to(prefix):
                deps.add(dep)

    for dep in sorted(deps):
        shutil.copy2(dep, tools_lib_dir / dep.name)

if os.name == "nt":
    mkvtoolnix_url = "https://github.com/Kruk2/jasna/releases/download/0.1/mkvtoolnix.zip"
    zip_path = internal / "mkvtoolnix.zip"
    mkvtoolnix_dir = internal / "mkvtoolnix"
    if mkvtoolnix_dir.exists():
        shutil.rmtree(mkvtoolnix_dir)
    if zip_path.exists():
        zip_path.unlink()
    urllib.request.urlretrieve(mkvtoolnix_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(mkvtoolnix_dir)
    zip_path.unlink()

if os.name != "nt":
    for f in (internal / "tensorrt_libs").glob("libnvinfer_builder_resource_win.so.*"):
        f.unlink()

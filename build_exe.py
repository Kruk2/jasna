import os
import shutil
import subprocess
import sys
from pathlib import Path
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
    "rfdetr-v4.onnx",
    "lada_mosaic_detection_model_v4_fast.pt",
]:
    shutil.copy(Path("model_weights") / name, out / "model_weights" / name)

(out / "assets").mkdir(parents=True, exist_ok=True)
shutil.copy(Path("assets") / "test_clip1.mp4", out / "assets" / "test_clip1_1080p.mp4")
shutil.copy(Path("assets") / "test_clip1_2160p.mp4", out / "assets" / "test_clip1_2160p.mp4")

internal = out / "_internal"

if os.name == "nt":
    tools_dir = internal / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    tool_path = Path(r'C:\Program Files\ffmpeg8\bin')
    for item in tool_path.iterdir():
        dst = tools_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    mkvtoolnix_url = "https://github.com/Kruk2/jasna/releases/download/0.1/mkvtoolnix.zip"
    zip_path = internal / "mkvtoolnix.zip"
    if zip_path.exists():
        zip_path.unlink()
    urllib.request.urlretrieve(mkvtoolnix_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(internal)
    zip_path.unlink()

if os.name != "nt":
    for f in (internal / "tensorrt_libs").glob("libnvinfer_builder_resource_win.so.*"):
        f.unlink()

import os
import shutil
import subprocess
import sys
from pathlib import Path

distpath = "dist_linux" if os.name != "nt" else "dist"
env = os.environ.copy()
subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", distpath, "jasna.spec"], check=True, env=env)
env["BUILD_CLI"] = "1"
subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", distpath, "jasna.spec"], check=True, env=env)

cli_exe = "jasna-cli.exe" if os.name == "nt" else "jasna-cli"
shutil.copy(Path(distpath) / "jasna-cli" / cli_exe, Path(distpath) / "jasna" / cli_exe)

out = Path(distpath) / "jasna"
(out / "model_weights").mkdir(parents=True, exist_ok=True)
for name in [
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "rfdetr-v3.onnx",
]:
    shutil.copy(Path("model_weights") / name, out / "model_weights" / name)

if os.name != "nt":
    internal = out / "_internal"
    for f in (internal / "tensorrt_libs").glob("libnvinfer_builder_resource_win.so.*"):
        f.unlink()

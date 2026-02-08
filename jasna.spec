# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_all
from importlib.util import find_spec
import os
import sys

_build_cli = os.environ.get("BUILD_CLI", "").lower() in ("1", "true", "yes")


def _collect(name: str, *, required: bool):
    if find_spec(name) is None:
        if required:
            raise RuntimeError(
                f"PyInstaller build requires {name!r} to be installed in this Python environment. "
                f"Current sys.executable is: {sys.executable}"
            )
        return [], [], []
    return collect_all(name)

datas, binaries, hiddenimports = [], [], []
required_pkgs = ["psutil"] + ([] if _build_cli else ["customtkinter"])
optional_pkgs = ["torch", "torch_tensorrt", "av", "PyNvVideoCodec", "python_vali", "tensorrt", "tensorrt_libs"]

for pkg in required_pkgs:
    d, b, h = _collect(pkg, required=True)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in optional_pkgs:
    d, b, h = _collect(pkg, required=False)
    datas += d
    binaries += b
    hiddenimports += h

# torch_tensorrt's custom ops are registered by its compiled extension module.
# `collect_all("torch_tensorrt")` collects `torchtrt.dll` but does not always pick up the `_C*.pyd`.
trt_ext = find_spec("torch_tensorrt._C")
if trt_ext is not None and trt_ext.origin:
    binaries += [(trt_ext.origin, "torch_tensorrt")]
    hiddenimports += ["torch_tensorrt._C"]

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH") or r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    cuda_bin = os.path.join(cuda_path, "bin", "x64")
    cuda_dlls = [
        "nppig64_13.dll",
        "nppicc64_13.dll",
        "nppc64_13.dll",
        "nppidei64_13.dll",
        "nppial64_13.dll",
        "nvjpeg64_13.dll",
    ]
    for dll in cuda_dlls:
        dll_path = os.path.join(cuda_bin, dll)
        if os.path.isfile(dll_path):
            binaries += [(dll_path, ".")]

runtime_hooks = []
if os.name == "nt":
    runtime_hooks.append(os.path.join("jasna", "packaging", "pyinstaller_runtime_hook_windows_dll_paths.py"))

a = Analysis(
    ["jasna/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if os.name == "nt" and not _build_cli:
    _exe_name = "jasna-gui"
    _collect_name = "jasna"
else:
    _exe_name = "jasna-cli" if _build_cli else "jasna"
    _collect_name = _exe_name
_console = True if _build_cli else (os.name != "nt")

exe = EXE(
    pyz,
    a.scripts,
    [],
    name=_exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=["_C*.pyd", "torchtrt.dll"],
    runtime_tmpdir=None,
    console=_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["_C*.pyd", "torchtrt.dll"],
    name=_collect_name,
)


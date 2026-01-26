# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_all
from importlib.util import find_spec
import os

def _collect(name: str):
    if find_spec(name) is None:
        return [], [], []
    return collect_all(name)

datas, binaries, hiddenimports = [], [], []
for pkg in ["torch", "torch_tensorrt", "av", "PyNvVideoCodec", "python_vali", "tensorrt", "tensorrt_libs"]:
    d, b, h = _collect(pkg)
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

a = Analysis(
    ["jasna/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="jasna",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=["_C*.pyd", "torchtrt.dll"],
    runtime_tmpdir=None,
    console=True,
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
    name="jasna",
)


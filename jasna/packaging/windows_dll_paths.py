from __future__ import annotations

import os
import sys
from pathlib import Path


_WINDOWS_PATHSEP = ";"
_ADDED_DLL_DIR_HANDLES: list[object] = []


def _norm(p: str) -> str:
    p = p.strip().strip('"').strip()
    if p == "":
        return ""
    return os.path.normcase(os.path.normpath(p)).lower()


def _split_path(path_value: str) -> list[str]:
    if not path_value:
        return []
    return [p for p in path_value.split(_WINDOWS_PATHSEP) if p.strip() != ""]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        k = _norm(item)
        if k == "" or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _is_under_root(path_entry: str, root: str) -> bool:
    p = _norm(path_entry)
    r = _norm(root)
    if p == "" or r == "":
        return False
    if p == r:
        return True
    sep = "\\" if ("\\" in r or ":" in r) else os.sep
    prefix = r + sep
    return p.startswith(prefix)


def sanitize_windows_path_for_cuda(path_value: str, *, preferred_dirs: list[str], cuda_roots: list[str]) -> str:
    preferred = [p for p in preferred_dirs if p and p.strip() != ""]
    blocked_roots = [r for r in cuda_roots if r and r.strip() != ""]

    keep: list[str] = []
    for entry in _split_path(path_value):
        entry_norm = _norm(entry)
        if entry_norm == "":
            continue

        if "nvidia gpu computing toolkit\\cuda" in entry_norm.replace("/", "\\"):
            continue
        if any(_is_under_root(entry, r) for r in blocked_roots):
            continue
        keep.append(entry)

    merged = _dedupe_keep_order(preferred + keep)
    return _WINDOWS_PATHSEP.join(merged)


def _iter_top_level_lib_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []

    out: list[Path] = [root]

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower().endswith(".libs"):
            out.append(child)

    torch_lib = root / "torch" / "lib"
    if torch_lib.is_dir():
        out.append(torch_lib)

    for name in ["tensorrt_libs", "PyNvVideoCodec", "python_vali"]:
        p = root / name
        if p.is_dir():
            out.append(p)

    return out


def configure_windows_dll_search_paths() -> None:
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    app_dir = Path(sys.executable).resolve().parent
    meipass = Path(getattr(sys, "_MEIPASS", str(app_dir)))

    candidates: list[Path] = []
    for base in [meipass, app_dir / "_internal", app_dir]:
        if base.is_dir():
            candidates.append(base)

    preferred_dirs: list[str] = []
    for base in candidates:
        for p in _iter_top_level_lib_dirs(base):
            preferred_dirs.append(str(p))

    cuda_roots: list[str] = []
    for key in ["CUDA_PATH", "CUDA_HOME"]:
        v = os.environ.get(key)
        if v:
            cuda_roots.append(v)
    for k, v in list(os.environ.items()):
        if k.upper().startswith("CUDA_PATH_V") and v:
            cuda_roots.append(v)

    for k in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"]:
        os.environ.pop(k, None)
    for k in list(os.environ.keys()):
        if k.upper().startswith("CUDA_PATH_V"):
            os.environ.pop(k, None)

    os.environ["PATH"] = sanitize_windows_path_for_cuda(
        os.environ.get("PATH", ""),
        preferred_dirs=preferred_dirs,
        cuda_roots=cuda_roots,
    )

    for d in _dedupe_keep_order(preferred_dirs):
        p = Path(d)
        if p.is_dir():
            _ADDED_DLL_DIR_HANDLES.append(os.add_dll_directory(str(p)))

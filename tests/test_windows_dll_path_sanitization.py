from __future__ import annotations

from jasna.packaging.windows_dll_paths import (
    _iter_top_level_lib_dirs,
    sanitize_windows_path_for_cuda,
)


def test_sanitize_windows_path_removes_cuda_toolkit_and_prepends_preferred() -> None:
    preferred = [r"C:\App", r"C:\App\torch\lib"]
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    original = ";".join(
        [
            r"C:\Windows\System32",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
            r"C:\SomethingElse\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64",
        ]
    )

    out = sanitize_windows_path_for_cuda(original, preferred_dirs=preferred, cuda_roots=[cuda_root])

    assert out.split(";")[0:2] == preferred
    assert r"NVIDIA GPU Computing Toolkit\CUDA" not in out
    assert r"C:\Windows\System32" in out
    assert r"C:\SomethingElse\bin" in out


def test_iter_top_level_lib_dirs_finds_nuitka_root_layout(tmp_path) -> None:
    # Nuitka dist: torch/lib, the *_libs vendor dirs, and *.libs siblings all live at the
    # dist root (no _internal/). configure_windows_dll_search_paths relies on this set.
    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "tensorrt_libs").mkdir()
    (tmp_path / "nvvfx" / "libs").mkdir(parents=True)
    (tmp_path / "numpy.libs").mkdir()

    dirs = {str(p) for p in _iter_top_level_lib_dirs(tmp_path)}

    assert str(tmp_path) in dirs
    assert str(tmp_path / "torch" / "lib") in dirs
    assert str(tmp_path / "tensorrt_libs") in dirs
    assert str(tmp_path / "nvvfx" / "libs") in dirs
    assert str(tmp_path / "numpy.libs") in dirs


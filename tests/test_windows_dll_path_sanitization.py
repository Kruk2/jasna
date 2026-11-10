from __future__ import annotations

from jasna.packaging.windows_dll_paths import (
    frozen_dll_directories,
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


def test_frozen_dll_directories_find_every_packaged_dll_folder(tmp_path) -> None:
    for name in ("torch/lib", "tensorrt_libs", "nvvfx/libs", "numpy.libs", "torch_tensorrt/lib", "tools", "vlc",
                 "triton/backends"):
        (tmp_path / name).mkdir(parents=True)

    dirs = frozen_dll_directories(tmp_path)

    assert dirs[0] == tmp_path
    assert {p.relative_to(tmp_path).as_posix() for p in dirs[1:]} == {
        "numpy.libs", "torch/lib", "tensorrt_libs", "nvvfx", "nvvfx/libs", "torch_tensorrt/lib", "tools", "vlc",
    }

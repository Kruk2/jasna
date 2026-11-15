from __future__ import annotations

from jasna.packaging.windows_dll_paths import (
    ambiguous_dll_names,
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


def test_ambiguous_dll_names_are_the_dlls_more_than_one_search_folder_provides(tmp_path) -> None:
    for name in ("c10.dll", "torch/lib/c10.dll", "torch/lib/torch_cuda.dll", "av.libs/avcodec-62.dll",
                 "avcodec-62.dll", "msvcp140.dll", "scipy/linalg/msvcp140.dll"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"image")

    assert ambiguous_dll_names(tmp_path) == ["av.libs/avcodec-62.dll = avcodec-62.dll", "c10.dll = torch/lib/c10.dll"]

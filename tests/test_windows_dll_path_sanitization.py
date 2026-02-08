from __future__ import annotations

from jasna.packaging.linux_lib_paths import sanitize_linux_ld_library_path_for_cuda
from jasna.packaging.windows_dll_paths import sanitize_windows_path_for_cuda


def test_sanitize_windows_path_removes_cuda_toolkit_and_prepends_preferred() -> None:
    preferred = [r"C:\App\_internal", r"C:\App\_internal\torch\lib"]
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


def test_sanitize_linux_ld_library_path_removes_usr_local_cuda_and_prepends_preferred() -> None:
    preferred = ["/opt/jasna/_internal", "/opt/jasna/_internal/torch/lib"]
    original = ":".join(
        [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/other/lib",
            "/usr/local/cuda/extras/CUPTI/lib64",
        ]
    )

    out = sanitize_linux_ld_library_path_for_cuda(original, preferred_dirs=preferred, cuda_roots=["/usr/local/cuda"])

    assert out.split(":")[0:2] == preferred
    assert "/usr/local/cuda" not in out
    assert "/usr/lib/x86_64-linux-gnu" in out
    assert "/opt/other/lib" in out


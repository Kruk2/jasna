from __future__ import annotations

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


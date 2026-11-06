import os
import sys

from factories import write_sparse_file


def _allocated_bytes(path) -> int:
    if sys.platform == "win32":
        import ctypes

        high = ctypes.c_ulong()
        low = ctypes.windll.kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
        return (high.value << 32) | low
    return os.stat(path).st_blocks * 512


def test_sparse_file_has_its_size_without_taking_disk_space(tmp_path):
    path = tmp_path / "model.safetensors.enc"

    write_sparse_file(path, 1 << 30)

    assert path.stat().st_size == 1 << 30
    assert _allocated_bytes(path) < 1 << 20

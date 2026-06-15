from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def read_image_rgb_chw(path: str | Path) -> np.ndarray:
    """Read an image as a ``(C, H, W)`` uint8 RGB array.

    Uses ``np.fromfile`` + ``cv2.imdecode`` so unicode paths work on Windows
    (``cv2.imread`` mangles non-ASCII paths there).
    """
    path = Path(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def write_image_rgb_chw(path: str | Path, chw_uint8: np.ndarray) -> None:
    """Write a ``(C, H, W)`` uint8 RGB array to ``path``.

    Uses ``cv2.imencode`` + ``tofile`` for unicode-safe writes on Windows. The
    output format is chosen from the path suffix.
    """
    path = Path(path)
    if chw_uint8.dtype != np.uint8 or chw_uint8.ndim != 3 or chw_uint8.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) uint8 RGB, got {chw_uint8.shape} {chw_uint8.dtype}")
    rgb = np.ascontiguousarray(chw_uint8.transpose(1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise ValueError(f"Failed to encode image for suffix {path.suffix!r}: {path}")
    buf.tofile(str(path))

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jasna.media.image_io import (
    IMAGE_EXTENSIONS,
    is_image_path,
    read_image_rgb_chw,
    write_image_rgb_chw,
)


class TestIsImagePath:
    @pytest.mark.parametrize("name", ["a.png", "b.JPG", "c.jpeg", "d.webp", "e.tif", "f.tiff", "g.bmp"])
    def test_image_suffixes(self, name: str):
        assert is_image_path(name) is True

    @pytest.mark.parametrize("name", ["a.mp4", "b.mkv", "c.txt"])
    def test_non_image_suffixes(self, name: str):
        assert is_image_path(name) is False

    def test_extension_set(self):
        assert ".png" in IMAGE_EXTENSIONS and ".jpg" in IMAGE_EXTENSIONS


class TestRoundTrip:
    def test_png_roundtrip_preserves_pixels(self, tmp_path: Path):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (3, 17, 23), dtype=np.uint8)
        path = tmp_path / "x.png"
        write_image_rgb_chw(path, img)
        back = read_image_rgb_chw(path)
        assert back.shape == img.shape
        assert back.dtype == np.uint8
        assert np.array_equal(back, img)

    def test_write_rejects_bad_shape(self, tmp_path: Path):
        with pytest.raises(ValueError):
            write_image_rgb_chw(tmp_path / "y.png", np.zeros((17, 23, 3), dtype=np.uint8))

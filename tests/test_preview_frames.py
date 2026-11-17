from __future__ import annotations

import pytest
import torch

from jasna.gui.preview_frames import downscaled_size, gpu_frame_image


def test_gpu_frame_image_resizes_bytes_with_float_rounding() -> None:
    frame = torch.tensor([0, 255, 0, 255], dtype=torch.uint8).view(1, 1, 4).expand(3, 2, 4).contiguous()

    image = gpu_frame_image(frame, (3, 2))

    assert [image.getpixel((x, 0)) for x in range(3)] == [(43, 43, 43), (128, 128, 128), (213, 213, 213)]


@pytest.mark.parametrize(
    ("frame_size", "size"),
    [((8, 4), (16, 8)), ((720, 576), (768, 576)), ((8, 4), (8, 4))],
    ids=["upscale", "anamorphic", "same-size"],
)
def test_gpu_frame_image_has_exactly_the_asked_size(frame_size, size) -> None:
    frame = torch.full((3, frame_size[1], frame_size[0]), 20, dtype=torch.uint8)

    image = gpu_frame_image(frame, size)

    assert image.size == size
    assert image.getpixel((0, 0)) == (20, 20, 20)


@pytest.mark.parametrize(
    ("size", "max_size", "expected"),
    [((3840, 2160), (640, 360), (640, 360)), ((1080, 1920), (640, 360), (202, 360)), ((320, 180), (640, 360), (320, 180))],
)
def test_downscaled_size_fits_without_enlarging(size, max_size, expected) -> None:
    assert downscaled_size(size, max_size) == expected

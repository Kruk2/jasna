import numpy as np
import torch

from jasna.crop_buffer import extract_crop


def test_left_eye_crop_expansion_cannot_cross_sbs_seam(monkeypatch) -> None:
    import jasna.crop_buffer as crop_buffer

    monkeypatch.setattr(crop_buffer, "MIN_BORDER", 20)
    frame = torch.zeros((3, 64, 128), dtype=torch.uint8)

    crop = extract_crop(
        frame,
        np.array([58.0, 20.0, 63.0, 30.0], dtype=np.float32),
        64,
        128,
        x_bounds=(0, 64),
    )

    assert crop.enlarged_bbox[2] <= 64


def test_right_eye_crop_expansion_cannot_cross_sbs_seam(monkeypatch) -> None:
    import jasna.crop_buffer as crop_buffer

    monkeypatch.setattr(crop_buffer, "MIN_BORDER", 20)
    frame = torch.zeros((3, 64, 128), dtype=torch.uint8)

    crop = extract_crop(
        frame,
        np.array([65.0, 20.0, 70.0, 30.0], dtype=np.float32),
        64,
        128,
        x_bounds=(64, 128),
    )

    assert crop.enlarged_bbox[0] >= 64

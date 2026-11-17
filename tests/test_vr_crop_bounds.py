import numpy as np
import pytest
import torch

import jasna.crop_buffer as crop_buffer
from jasna.crop_buffer import extract_crop


@pytest.mark.parametrize(
    ("frame_hw", "bbox", "eye"),
    [
        ((64, 128), (58.0, 20.0, 63.0, 30.0), (0, 0, 64, 64)),
        ((64, 128), (65.0, 20.0, 70.0, 30.0), (64, 0, 128, 64)),
        ((128, 64), (20.0, 65.0, 30.0, 70.0), (0, 64, 64, 128)),
    ],
    ids=["left-eye-sbs", "right-eye-sbs", "bottom-eye-tb"],
)
def test_crop_expansion_stays_inside_its_eye(monkeypatch, frame_hw, bbox, eye) -> None:
    monkeypatch.setattr(crop_buffer, "MIN_BORDER", 20)
    height, width = frame_hw
    frame = torch.zeros((3, height, width), dtype=torch.uint8)

    x1, y1, x2, y2 = extract_crop(frame, np.array(bbox, dtype=np.float32), height, width, bounds=eye).enlarged_bbox

    assert eye[0] <= x1 and eye[1] <= y1 and x2 <= eye[2] and y2 <= eye[3]

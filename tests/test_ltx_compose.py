import cv2
import numpy as np
import torch

from jasna.ltx.camera import letterbox
from jasna.ltx.compose import (
    Candidate,
    canvas_to_source,
    composite_frame,
    crop_to_canvas,
    feather_alpha,
    polygon_mask,
)


def _frame(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, 360, 640), dtype=torch.uint8, generator=generator)


def test_feather_alpha_matches_opencv():
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(mask, (150, 100), 40, 255, -1)
    feather = 12
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1,) * 2)
    expected = cv2.GaussianBlur(cv2.dilate(mask, kernel).astype(np.float32) / 255.0, (2 * feather + 1,) * 2, 0.0)
    actual = feather_alpha(torch.from_numpy(mask > 0), feather).numpy()
    assert np.abs(actual - expected).max() < 1e-5


def test_crop_round_trip_at_unit_scale():
    frame = _frame()
    crop = letterbox((100, 50, 356, 306), 256)
    canvas = crop_to_canvas(frame, crop)
    assert canvas.shape == (3, 256, 256)
    assert torch.equal(canvas_to_source(canvas, crop).to(torch.uint8), frame[:, 50:306, 100:356])


def test_crop_pads_by_reflection():
    frame = _frame()
    crop = letterbox((0, 0, 200, 100), 200)
    canvas = crop_to_canvas(frame, crop)
    top = crop.pad[0]
    assert torch.equal(canvas[:, top - 1], canvas[:, top + 1])


def test_composite_touches_only_the_feathered_mask():
    source = _frame(1)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 128, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    out = composite_frame(source, [Candidate(restored, crop, 1.0, [polygon])], feather=8)
    changed = (out != source).any(dim=0)
    ys, xs = torch.nonzero(changed, as_tuple=True)
    assert changed[230, 330]
    assert xs.min() >= 300 - 2 * 8 - 1 and xs.max() <= 360 + 2 * 8 + 1
    assert ys.min() >= 200 - 2 * 8 - 1 and ys.max() <= 260 + 2 * 8 + 1


def test_polygon_mask_is_the_union_of_overlapping_polygons():
    outer = [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]]
    nested = [[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]]
    crossing = [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]]
    mask = polygon_mask([outer, nested, crossing], (160, 160), (0, 0), torch.device("cpu"))
    assert mask[40, 40] and mask[55, 55] and mask[80, 80] and mask[140, 140]
    assert mask.sum() == 100 * 100 + 101 * 101 - 50 * 50


def test_zero_weight_and_no_candidates_keep_the_source():
    source = _frame(2)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.zeros((3, 256, 256), dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0]]
    assert torch.equal(composite_frame(source, [], feather=8), source)
    assert torch.equal(composite_frame(source, [Candidate(restored, crop, 0.0, [polygon])], feather=8), source)


def test_two_agreeing_windows_equal_one():
    source = _frame(3)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 90, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    one = composite_frame(source, [Candidate(restored, crop, 1.0, [polygon])], feather=8)
    two = composite_frame(
        source, [Candidate(restored, crop, 0.3, [polygon]), Candidate(restored, crop, 0.7, [polygon])], feather=8
    )
    assert (one.int() - two.int()).abs().max() <= 1

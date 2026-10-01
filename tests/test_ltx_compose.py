import cv2
import numpy as np
import pytest
import torch

from jasna.ltx.camera import letterbox
from jasna.ltx.compose import (
    Candidate,
    canvas_to_source,
    composite_frame,
    crop_to_canvas,
    feather_alpha,
    masked_median,
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
    out = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8)
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
    assert torch.equal(composite_frame(source, [Candidate(restored, crop, 0.0, 0, [polygon], 1.0)], feather=8), source)


def test_two_agreeing_windows_equal_one():
    source = _frame(3)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 90, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    one = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8)
    two = composite_frame(
        source, [Candidate(restored, crop, 0.3, 0, [polygon], 1.0), Candidate(restored, crop, 0.7, 0, [polygon], 1.0)], feather=8
    )
    assert (one.int() - two.int()).abs().max() <= 1


def _quantile_median(difference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return torch.zeros(difference.shape[0])
    return torch.quantile(difference[:, mask], 0.5, dim=1)


@pytest.mark.parametrize(
    "pixels",
    [
        [],
        [(5, 7)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 0), (1, 1), (2, 2), (3, 3)],
        [(y, 4) for y in range(16)],
    ],
    ids=["empty", "single", "odd", "even", "column"],
)
def test_masked_median_equals_quantile(pixels):
    generator = torch.Generator().manual_seed(len(pixels))
    difference = torch.randint(-255, 256, (3, 16, 16), generator=generator).float()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    for y, x in pixels:
        mask[y, x] = True
    assert torch.equal(masked_median(difference, mask), _quantile_median(difference, mask))


def test_masked_median_equals_quantile_on_random_rings():
    generator = torch.Generator().manual_seed(0)
    for spread in (0, 1, 3, 40, 255):
        for _ in range(20):
            difference = torch.randint(-spread, spread + 1, (3, 40, 50), generator=generator).float()
            difference[1] = -difference[1].abs()
            mask = torch.rand(40, 50, generator=generator) < torch.rand(1, generator=generator).item()
            assert torch.equal(masked_median(difference, mask), _quantile_median(difference, mask))


def test_masked_median_takes_the_mean_of_the_two_central_values():
    difference = torch.tensor([[-255.0, -4.0, 7.0, 255.0], [3.0, 3.0, 3.0, 3.0], [-1.0, 0.0, 1.0, 2.0]])[:, None, :]
    assert masked_median(difference, torch.ones(1, 4, dtype=torch.bool)).tolist() == [1.5, 3.0, 0.5]


def test_composite_shifts_the_restored_colour_towards_the_ring():
    source = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 140, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    out = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8)
    assert out[:, 230, 330].tolist() == [110, 110, 110]


def test_two_tracks_each_paint_their_own_mask():
    source = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    crop = letterbox((100, 50, 580, 330), 480)
    left = [[200.0, 150.0], [260.0, 150.0], [260.0, 210.0], [200.0, 210.0]]
    right = [[400.0, 150.0], [460.0, 150.0], [460.0, 210.0], [400.0, 210.0]]
    dark = torch.full((3, 480, 480), 20, dtype=torch.uint8)
    bright = torch.full((3, 480, 480), 220, dtype=torch.uint8)
    out = composite_frame(source, [Candidate(dark, crop, 1.0, 0, [left], 1.0), Candidate(bright, crop, 1.0, 1, [right], 1.0)], feather=8)
    assert out[0, 180, 230] < 100 < out[0, 180, 430]


def _handover(presence: float) -> torch.Tensor:
    source = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    crop = letterbox((100, 50, 580, 330), 480)
    parent = [[150.0, 100.0], [500.0, 100.0], [500.0, 280.0], [150.0, 280.0]]
    child = [[200.0, 150.0], [300.0, 150.0], [300.0, 230.0], [200.0, 230.0]]
    dark = torch.full((3, 480, 480), 20, dtype=torch.uint8)
    bright = torch.full((3, 480, 480), 220, dtype=torch.uint8)
    return composite_frame(
        source, [Candidate(dark, crop, 1.0, 0, [parent], 1.0), Candidate(bright, crop, 1.0, 1, [child], presence)], feather=8
    )


def test_handover_shares_the_child_area_by_presence():
    start, half, done = _handover(1e-6), _handover(0.5), _handover(1.0 - 1e-6)
    inside, edge, parent_only = (190, 250), (150, 200 - 9), (190, 450)
    assert start[0, 190, 250] < 100 < done[0, 190, 250]
    for y, x in (inside, edge):
        assert abs(int(half[0, y, x]) - (int(start[0, y, x]) + int(done[0, y, x])) / 2) <= 1
    assert torch.equal(half[:, parent_only[0], parent_only[1]], start[:, parent_only[0], parent_only[1]])
    assert done[0, edge[0], edge[1]] > start[0, edge[0], edge[1]]

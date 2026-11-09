import cv2
import numpy as np
import pytest
import torch

from jasna.vr180 import StereoLayout
from jasna.ltx.camera import letterbox
from jasna.ltx.compose import (
    Candidate,
    canvas_to_source,
    composite_frame,
    crop_edge_ramp,
    crop_to_canvas,
    feather_alpha,
    masked_median,
    polygon_box,
    polygon_mask,
)


def _frame(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, 360, 640), dtype=torch.uint8, generator=generator)


@pytest.mark.parametrize(
    ("shape", "feather"),
    [((200, 300), 12), ((64, 65), 6), ((129, 128), 30), ((33, 257), 3), ((513, 70), 20)],
)
def test_feather_alpha_matches_opencv(shape, feather):
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (width // 5, height // 5), 0, 0, 360, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1,) * 2)
    expected = cv2.GaussianBlur(cv2.dilate(mask, kernel).astype(np.float32) / 255.0, (2 * feather + 1,) * 2, 0.0)
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    actual = feather_alpha(torch.from_numpy(mask > 0), feather, box).numpy()
    assert np.abs(actual - expected).max() < 1e-5


_REGION_POLYGONS = [
    [[80, 40], [110, 40], [110, 70], [80, 70]],
    [[0, 0], [30, 0], [30, 30], [0, 30]],
    [[170, 90], [199, 90], [199, 119], [170, 119]],
    [[150, 50], [197, 50], [197, 80], [150, 80]],
    [[24, 30], [60, 30], [60, 94], [24, 94]],
    [[25, 30], [60, 30], [60, 95], [25, 95]],
    [[26, 26], [60, 26], [60, 93], [26, 93]],
    [[-40, 60], [90, 10], [240, 150]],
    [[0, 0], [199, 0], [199, 119], [0, 119]],
]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("polygon", _REGION_POLYGONS)
def test_feather_alpha_inside_the_polygon_box_matches_the_whole_region(device, polygon):
    shape, feather = (120, 200), 12
    mask = polygon_mask([polygon], shape, (0, 0), torch.device(device))
    whole = feather_alpha(mask, feather, (0, 0, shape[1], shape[0]))
    in_box = feather_alpha(mask, feather, polygon_box([polygon], shape, (0, 0)))
    assert whole.any()
    assert torch.equal(in_box > 0, whole > 0)
    # The blur's matrix products sum the taps in an order that depends on the cut-out's size, so it differs by a few float ulps.
    torch.testing.assert_close(in_box, whole, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("seed", range(4))
def test_outside_the_union_feather_is_outside_every_track_feather(device, seed):
    rng = np.random.default_rng(seed)
    shape, feather = (120, 200), 9
    polygons = [(rng.integers(-20, 220, size=(5, 2))).tolist() for _ in range(3)]
    masks = [polygon_mask([p], shape, (0, 0), torch.device(device)) for p in polygons]
    union = torch.stack(masks).any(dim=0)
    union_alpha = feather_alpha(union, feather, polygon_box(polygons, shape, (0, 0)))
    each = torch.stack([feather_alpha(m, feather, polygon_box([p], shape, (0, 0))) <= 0.0 for m, p in zip(masks, polygons)])
    assert torch.equal(union_alpha <= 0.0, each.all(dim=0))


@pytest.mark.parametrize("box", [(30, 20, 150, 100), (0, 0, 200, 120), (0, 40, 90, 120)])
def test_crop_edge_ramp_window_is_that_part_of_the_whole_ramp(box):
    shape, window = (120, 200), (25, 15, 160, 110)
    whole = crop_edge_ramp(box, shape, 12, (0, 0, shape[1], shape[0]), torch.device("cpu"))
    x1, y1, x2, y2 = window
    assert torch.equal(crop_edge_ramp(box, shape, 12, window, torch.device("cpu")), whole[y1:y2, x1:x2])


def test_composite_blends_the_whole_feathered_mask():
    source = torch.full((3, 360, 640), 20, dtype=torch.uint8)
    crop = letterbox((100, 40, 420, 360), 320)
    polygon = [[200.0, 120.0], [300.0, 120.0], [300.0, 220.0], [200.0, 220.0]]
    canvas = torch.full((3, 320, 320), 220, dtype=torch.uint8)
    out = composite_frame(source, [Candidate(canvas, crop, 1.0, 0, [polygon], 1.0)], feather=12, projector=None)
    alpha = feather_alpha(polygon_mask([polygon], (360, 640), (0, 0), torch.device("cpu")), 12, (0, 0, 640, 360))
    # The colour match moves the canvas 3/4 of the way to the ring: 220 - 0.75 * 200 = 70.
    expected = 20 + 50 * alpha
    assert (out[0].float() - expected).abs().max() <= 0.5 + 1e-4


def test_crop_round_trip_at_unit_scale():
    frame = _frame()
    crop = letterbox((100, 50, 356, 306), 256)
    canvas = crop_to_canvas(frame, crop, None)
    assert canvas.shape == (3, 256, 256)
    assert torch.equal(canvas_to_source(canvas, crop, frame, None).to(torch.uint8), frame[:, 50:306, 100:356])


def test_crop_pads_by_reflection():
    frame = _frame()
    crop = letterbox((0, 0, 200, 100), 200)
    canvas = crop_to_canvas(frame, crop, None)
    top = crop.pad[0]
    assert torch.equal(canvas[:, top - 1], canvas[:, top + 1])


def test_composite_touches_only_the_feathered_mask():
    source = _frame(1)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 128, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    out = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8, projector=None)
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
    assert torch.equal(composite_frame(source, [], feather=8, projector=None), source)
    assert torch.equal(composite_frame(source, [Candidate(restored, crop, 0.0, 0, [polygon], 1.0)], feather=8, projector=None), source)


def test_two_agreeing_windows_equal_one():
    source = _frame(3)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 256, 256), 90, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    one = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8, projector=None)
    two = composite_frame(
        source,
        [Candidate(restored, crop, 0.3, 0, [polygon], 1.0), Candidate(restored, crop, 0.7, 0, [polygon], 1.0)],
        feather=8,
        projector=None,
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
    out = composite_frame(source, [Candidate(restored, crop, 1.0, 0, [polygon], 1.0)], feather=8, projector=None)
    assert out[:, 230, 330].tolist() == [110, 110, 110]


def test_two_tracks_each_paint_their_own_mask():
    source = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    crop = letterbox((100, 50, 580, 330), 480)
    left = [[200.0, 150.0], [260.0, 150.0], [260.0, 210.0], [200.0, 210.0]]
    right = [[400.0, 150.0], [460.0, 150.0], [460.0, 210.0], [400.0, 210.0]]
    dark = torch.full((3, 480, 480), 20, dtype=torch.uint8)
    bright = torch.full((3, 480, 480), 220, dtype=torch.uint8)
    out = composite_frame(source, [Candidate(dark, crop, 1.0, 0, [left], 1.0), Candidate(bright, crop, 1.0, 1, [right], 1.0)], feather=8, projector=None)
    assert out[0, 180, 230] < 100 < out[0, 180, 430]


def _handover(presence: float) -> torch.Tensor:
    source = torch.full((3, 360, 640), 100, dtype=torch.uint8)
    crop = letterbox((100, 50, 580, 330), 480)
    parent = [[150.0, 100.0], [500.0, 100.0], [500.0, 280.0], [150.0, 280.0]]
    child = [[200.0, 150.0], [300.0, 150.0], [300.0, 230.0], [200.0, 230.0]]
    dark = torch.full((3, 480, 480), 20, dtype=torch.uint8)
    bright = torch.full((3, 480, 480), 220, dtype=torch.uint8)
    return composite_frame(
        source,
        [Candidate(dark, crop, 1.0, 0, [parent], 1.0), Candidate(bright, crop, 1.0, 1, [child], presence)],
        feather=8,
        projector=None,
    )


def test_handover_shares_the_child_area_by_presence():
    start, half, done = _handover(1e-6), _handover(0.5), _handover(1.0 - 1e-6)
    inside, edge, parent_only = (190, 250), (150, 200 - 9), (190, 450)
    assert start[0, 190, 250] < 100 < done[0, 190, 250]
    for y, x in (inside, edge):
        assert abs(int(half[0, y, x]) - (int(start[0, y, x]) + int(done[0, y, x])) / 2) <= 1
    assert torch.equal(half[:, parent_only[0], parent_only[1]], start[:, parent_only[0], parent_only[1]])
    assert done[0, edge[0], edge[1]] > start[0, edge[0], edge[1]]


def test_only_windows_compose_enlarges_get_super_resolution():
    from jasna.ltx.plan import Window
    from jasna.ltx.super_res import enlarged_windows

    fits = letterbox((0, 0, 400, 400), 512)
    shrunk = Window(index=0, track_id=0, start=0, real_frames=2, crops=(fits, fits), noise=0)
    enlarged = Window(index=1, track_id=0, start=1, real_frames=2, crops=(fits, letterbox((0, 0, 700, 700), 512)), noise=1)
    assert enlarged_windows([shrunk, enlarged]) == {1}


def test_a_2x_canvas_maps_back_onto_the_same_crop():
    from jasna.ltx.super_res import upscaled_crop

    crop = letterbox((100, 50, 1000, 650), 512)
    ramp = torch.linspace(0, 255, 512)
    canvas = torch.stack([ramp[None, :].expand(512, 512), ramp[:, None].expand(512, 512), torch.full((512, 512), 90.0)])
    canvas = canvas.round().to(torch.uint8)
    doubled = torch.nn.functional.interpolate(canvas[None].float(), scale_factor=2, mode="bilinear")[0]
    frame = torch.zeros((3, 700, 1100), dtype=torch.uint8)

    direct = canvas_to_source(canvas, crop, frame, None)
    upscaled = canvas_to_source(doubled.round().to(torch.uint8), upscaled_crop(crop), frame, None)

    assert upscaled.shape == direct.shape == (3, 600, 900)
    assert (upscaled - direct).abs().mean() < 1.0


@pytest.mark.parametrize("layout", ["sbs", "tb"])
@pytest.mark.parametrize("kind", ["fisheye", "gnomonic"])
def test_vr_projected_crops_change_only_what_the_restoration_changed(kind, layout):
    from jasna.vr_projection import build_vr_projector

    stereo = StereoLayout("sbs", 640, 360) if layout == "sbs" else StereoLayout("tb", 320, 720)
    dx, dy = stereo.right_offset
    source = torch.randint(
        0, 256, (3, stereo.frame_h, stereo.frame_w), dtype=torch.uint8, generator=torch.Generator().manual_seed(3)
    )
    projector = build_vr_projector(kind, stereo, device=torch.device("cpu"))
    crop = letterbox((dx + 60, dy + 80, dx + 260, dy + 280), 256)
    polygon = [[dx + 120.0, dy + 140.0], [dx + 200.0, dy + 140.0], [dx + 200.0, dy + 220.0], [dx + 120.0, dy + 220.0]]
    view = crop_to_canvas(source, crop, projector)

    kept = composite_frame(source, [Candidate(view, crop, 1.0, 0, [polygon], 1.0)], feather=8, projector=projector)
    brighter = (view.int() + 40).clamp(max=255).to(torch.uint8)
    changed = composite_frame(source, [Candidate(brighter, crop, 1.0, 0, [polygon], 1.0)], feather=8, projector=projector)

    assert view.shape == (3, 256, 256)
    assert torch.equal(kept, source)
    moved = (changed != source).any(dim=0)
    assert moved[dy + 180, dx + 160]
    ys, xs = torch.nonzero(moved, as_tuple=True)
    assert xs.min() >= dx + 60 and xs.max() < dx + 260 and ys.min() >= dy + 80 and ys.max() < dy + 280


@pytest.mark.parametrize("polygon", _REGION_POLYGONS)
def test_polygon_mask_fills_what_opencv_fills_on_the_whole_region(polygon):
    shape, (dx, dy) = (120, 200), (7, 5)
    expected = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(expected, [np.array(polygon, dtype=np.int32)], 255)
    mask = polygon_mask([[[x + dx, y + dy] for x, y in polygon]], shape, (dx, dy), torch.device("cpu"))
    assert np.array_equal(mask.numpy(), expected > 0)

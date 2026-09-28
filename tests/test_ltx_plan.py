import numpy as np
import pytest

from jasna.ltx.camera import PLAN_CANVAS, ZOOM_PER_WINDOW, letterbox, track_camera, zoom_crop_sides
from jasna.ltx.plan import (
    LARGE_CANVAS,
    LEFT_SHARED_LATENTS,
    RIGHT_SHARED_LATENTS,
    WINDOW_HOP,
    Region,
    RegionGroup,
    Track,
    build_tracks,
    crossfade_weights,
    group_regions,
    merge_continuations,
    plan_video,
    track_polygons,
    window_starts,
)


def _square(x: float, y: float, side: float) -> Region:
    box = (x, y, x + side, y + side)
    return Region(box=box, polygon=[[x, y], [x + side, y], [x + side, y + side], [x, y + side]])


def _group(x: float, y: float, side: float) -> RegionGroup:
    region = _square(x, y, side)
    return RegionGroup(region.box, (region.polygon,))


def test_group_regions_merges_touching_boxes_only():
    groups = group_regions([_square(0, 0, 100), _square(90, 0, 100), _square(500, 500, 50)])
    assert sorted(len(g.polygons) for g in groups) == [1, 2]
    merged = next(g for g in groups if len(g.polygons) == 2)
    assert merged.box == (0, 0, 190, 100)


def test_build_tracks_links_across_short_gaps_but_not_cuts():
    frames = [[_group(100, 100, 80)] for _ in range(10)]
    frames[3] = []
    frames[4] = []
    assert len(build_tracks(frames, cuts=set())) == 1
    assert len(build_tracks(frames, cuts={6})) == 2
    frames[5] = frames[6] = frames[7] = []
    frames[3] = frames[4] = []
    assert len(build_tracks(frames, cuts=set())) == 2


def test_merge_continuations_folds_a_split_mosaic():
    parent = Track(0, {i: _group(100, 100, 200) for i in range(10)})
    child = Track(1, {i: _group(150, 150, 60) for i in range(5, 15)})
    elsewhere = Track(2, {i: _group(900, 900, 60) for i in range(5, 15)})
    merged = merge_continuations([parent, child, elsewhere], cuts=set())
    assert [t.track_id for t in merged] == [0, 2]
    assert merged[0].first == 0 and merged[0].last == 14
    assert len(merged[0].frames[6].polygons) == 2
    assert len(merge_continuations([parent, child], cuts={5})) == 2


def test_track_polygons_interpolate_gaps():
    track = Track(0, {0: _group(0, 0, 100), 4: _group(40, 0, 100)})
    polygons = track_polygons(track)
    assert len(polygons) == 5
    xs = [p[0][0][0] for p in polygons]
    assert xs == pytest.approx([0, 10, 20, 30, 40])


def test_window_starts_cover_with_hop():
    assert window_starts(1) == [0]
    assert window_starts(121) == [0]
    assert window_starts(122) == [0, WINDOW_HOP]
    assert window_starts(385) == [0, 88, 176, 264]
    assert window_starts(386) == [0, 88, 176, 264, 352]


def test_shared_latents_encode_the_same_frames():
    def first_pixel(latent: int) -> int:
        return 0 if latent == 0 else 8 * latent - 7

    for left, right in zip(LEFT_SHARED_LATENTS, RIGHT_SHARED_LATENTS):
        assert first_pixel(left) == WINDOW_HOP + first_pixel(right)


def test_crossfade_weights_partition_unity():
    plans = plan_video([[_square(300 + t, 300, 120)] for t in range(300)], set(), frame_w=1920, frame_h=1080, large_canvas=True)
    windows = plans[0].windows
    total = np.zeros(300)
    for window, weight in zip(windows, crossfade_weights(windows)):
        total[window.start : window.stop] += weight
    assert np.allclose(total, 1.0)


def test_zoom_sides_cover_raw_and_respect_rate():
    raw = np.array([200] * 100 + [800] * 100 + [200] * 200)
    sides = zoom_crop_sides(raw)
    assert (sides >= raw).all()
    span = sides[120:] / sides[:-120]
    assert max(span.max(), (1 / span).max()) <= ZOOM_PER_WINDOW * (1 + 1 / raw.min())


def test_track_camera_contains_support_and_letterboxes():
    boxes = [(100.0 + 3 * t, 200.0, 400.0 + 3 * t, 500.0) for t in range(150)]
    camera = track_camera(boxes, frame_w=1920, frame_h=1080, feather=30)
    for crop, box in zip(camera, boxes):
        x1, y1, x2, y2 = crop.crop_box
        assert x1 <= box[0] and y1 <= box[1] and x2 >= box[2] and y2 >= box[3]
        assert crop.canvas == PLAN_CANVAS


def test_large_crops_use_the_large_canvas():
    plans = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, large_canvas=True)
    assert plans[0].windows[0].canvas == LARGE_CANVAS
    capped = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, large_canvas=False)
    assert capped[0].windows[0].canvas == PLAN_CANVAS
    assert letterbox((0, 0, 900, 900), LARGE_CANVAS).resize_hw == (LARGE_CANVAS, LARGE_CANVAS)

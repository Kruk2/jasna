import hashlib
import time

import numpy as np
import pytest
import torch

from jasna.ltx.camera import PLAN_CANVAS, ZOOM_PER_WINDOW, letterbox, track_camera, zoom_crop_sides
from jasna.ltx.plan import (
    BLEND_FRAMES,
    LARGE_CANVAS,
    LEAD_IN_FRAMES,
    LESS_OVERLAP_WINDOW_HOP,
    WINDOW_HOP,
    Region,
    RegionGroup,
    Track,
    build_tracks,
    crossfade_weights,
    filled_frames,
    group_regions,
    handover_leads,
    handover_presence,
    merge_continuations,
    plan_video,
    rejoin_across_cover,
    restart_cameras,
    shared_latents,
    window_hop,
    window_starts,
)
from jasna.ltx.regions import trace_region


def _square(x: float, y: float, side: float) -> Region:
    box = (x, y, x + side, y + side)
    return Region(box=box, polygon=np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.int32))


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
    merged = merge_continuations([parent, child, elsewhere], cuts=set(), frame_w=1920, frame_h=1080)
    assert [t.track_id for t in merged] == [0, 2]
    assert merged[0].first == 0 and merged[0].last == 14
    assert len(merged[0].frames[6].polygons) == 2
    assert len(merge_continuations([parent, child], cuts={5}, frame_w=1920, frame_h=1080)) == 2


def test_merge_continuations_keeps_the_gap_fill_of_both_tracks():
    parent = Track(0, {i: _group(100, 100, 200) for i in range(10) if i != 7})
    child = Track(1, {i: _group(150, 150, 60) for i in range(5, 15) if i != 8})
    merged = merge_continuations([parent, child], cuts=set(), frame_w=1920, frame_h=1080)
    assert len(merged) == 1
    assert [len(merged[0].frames[i].polygons) for i in (6, 7, 8, 9)] == [2, 2, 2, 2]
    assert merged[0].frames[7].box == (100, 100, 300, 300)


def _split_video() -> list[list[Region]]:
    """A 900 px mosaic that splits at frame 100 into a piece leaving by frame 129 and a
    100 px piece staying until frame 499."""
    frames = [[_square(100, 90, 900)] for _ in range(100)]
    frames += [[_square(100, 500, 400), _square(700, 150, 100)] for _ in range(30)]
    return frames + [[_square(700, 150, 100)] for _ in range(370)]


def test_a_lasting_split_gets_its_own_camera_with_a_handover():
    plans = plan_video(_split_video(), set(), frame_w=1920, frame_h=1080, large_canvas=False, window_hop=WINDOW_HOP)
    assert [(p.start, p.stop) for p in plans] == [(0, 130), (100 - LEAD_IN_FRAMES, 500)]
    piece = plans[1]
    assert len(piece.presence) == len(piece.polygons) == piece.stop - piece.start
    blend = piece.presence[LEAD_IN_FRAMES - BLEND_FRAMES : LEAD_IN_FRAMES]
    assert set(piece.presence[: LEAD_IN_FRAMES - BLEND_FRAMES]) == {0.0}
    assert 0.0 < blend[0] and all(a < b for a, b in zip(blend, blend[1:])) and blend[-1] < 1.0
    assert set(piece.presence[LEAD_IN_FRAMES:]) == {1.0}
    assert piece.polygons[0] is piece.polygons[LEAD_IN_FRAMES]
    window = piece.windows[1]
    assert window.crops[200 - window.start].side <= PLAN_CANVAS


def test_handover_leads_stop_at_cuts_and_hand_back_to_a_covering_track():
    parent = Track(0, {i: _group(100, 100, 800) for i in range(300)})
    child = Track(1, {i: _group(150, 150, 100) for i in range(100, 200)})
    assert handover_leads([parent, child], set(), 300) == [(0, 0), (LEAD_IN_FRAMES, BLEND_FRAMES)]
    assert handover_leads([parent, child], {90}, 300)[1] == (10, BLEND_FRAMES)
    assert handover_leads([parent, child], {200}, 300)[1] == (LEAD_IN_FRAMES, 0)
    assert handover_leads([parent, child], {202}, 300)[1] == (LEAD_IN_FRAMES, 2)


def test_handover_presence_ramps_in_and_out():
    presence = handover_presence(5, LEAD_IN_FRAMES, BLEND_FRAMES)
    ramp_in = presence[LEAD_IN_FRAMES - BLEND_FRAMES : LEAD_IN_FRAMES]
    ramp_out = presence[LEAD_IN_FRAMES + 5 :]
    assert len(presence) == LEAD_IN_FRAMES + 5 + BLEND_FRAMES
    assert presence[LEAD_IN_FRAMES : LEAD_IN_FRAMES + 5] == (1.0,) * 5
    assert ramp_in == pytest.approx(tuple(1.0 - v for v in ramp_out))
    assert handover_presence(3, 0, 0) == (1.0, 1.0, 1.0)


def test_rejoin_across_cover_bridges_a_brief_join_only_while_covered():
    joined = Track(0, {i: _group(100, 100, 800) for i in range(300)})
    before = Track(1, {i: _group(150, 150, 200) for i in range(100)})
    after = Track(2, {i: _group(150, 150, 200) for i in range(104, 300)})
    rejoined = rejoin_across_cover([joined, before, after], set())
    assert [(t.first, t.last) for t in rejoined] == [(0, 299), (0, 299)]
    assert len(rejoined[1].frames) == 300
    short = Track(0, {i: _group(100, 100, 800) for i in range(102)})
    assert len(rejoin_across_cover([short, before, after], set())) == 3
    assert len(rejoin_across_cover([joined, before, after], {102})) == 3


def test_restart_cameras_cut_at_a_lasting_size_change_only():
    shrinking = Track(0, {i: _group(100, 100, 800 if i < 200 else 100) for i in range(600)})
    assert [(t.first, t.last) for t in restart_cameras([shrinking], frame_w=1920, frame_h=1080)] == [(0, 199), (200, 599)]
    spiking = Track(0, {i: _group(100, 100, 800 if 300 <= i < 303 else 100) for i in range(600)})
    assert len(restart_cameras([spiking], frame_w=1920, frame_h=1080)) == 1


def test_filled_frames_interpolate_gaps():
    track = Track(0, {0: _group(0, 0, 100), 4: _group(40, 0, 100)})
    frames = filled_frames(track)
    assert list(frames) == [0, 1, 2, 3, 4]
    xs = [group.polygons[0][0][0] for group in frames.values()]
    assert xs == pytest.approx([0, 10, 20, 30, 40])


def test_window_starts_cover_with_hop():
    assert window_starts(1, WINDOW_HOP) == [0]
    assert window_starts(121, WINDOW_HOP) == [0]
    assert window_starts(122, WINDOW_HOP) == [0, WINDOW_HOP]
    assert window_starts(385, WINDOW_HOP) == [0, 88, 176, 264]
    assert window_starts(386, WINDOW_HOP) == [0, 88, 176, 264, 352]


def test_less_overlap_window_starts_step_104():
    assert window_starts(121, LESS_OVERLAP_WINDOW_HOP) == [0]
    assert window_starts(122, LESS_OVERLAP_WINDOW_HOP) == [0, 104]
    assert window_starts(600, LESS_OVERLAP_WINDOW_HOP) == [0, 104, 208, 312, 416, 520]


def test_window_hop_from_the_less_overlap_switch():
    assert (window_hop(False), window_hop(True)) == (88, 104)


def test_default_hop_shares_four_latents():
    shared = shared_latents(WINDOW_HOP)
    assert (shared.left, shared.right) == (range(12, 16), range(1, 5))
    assert shared.ramp == (0.2, 0.4, 0.6, 0.8)


def test_less_overlap_hop_shares_two_latents():
    shared = shared_latents(LESS_OVERLAP_WINDOW_HOP)
    assert (shared.left, shared.right) == (range(14, 16), range(1, 3))
    assert shared.ramp == pytest.approx((1 / 3, 2 / 3))


@pytest.mark.parametrize("hop", [WINDOW_HOP, LESS_OVERLAP_WINDOW_HOP])
def test_shared_latents_encode_the_same_frames(hop):
    def first_pixel(latent: int) -> int:
        return 0 if latent == 0 else 8 * latent - 7

    shared = shared_latents(hop)
    for left, right in zip(shared.left, shared.right, strict=True):
        assert first_pixel(left) == hop + first_pixel(right)


def test_plan_windows_follow_the_hop():
    regions = [[_square(300, 300, 120)] for _ in range(300)]
    (plan,) = plan_video(regions, set(), frame_w=1920, frame_h=1080, large_canvas=False, window_hop=LESS_OVERLAP_WINDOW_HOP)
    assert [w.start for w in plan.windows] == [0, 104, 208]
    assert plan.window_hop == LESS_OVERLAP_WINDOW_HOP


def test_crossfade_weights_partition_unity():
    plans = plan_video([[_square(300 + t, 300, 120)] for t in range(300)], set(), frame_w=1920, frame_h=1080, large_canvas=True, window_hop=WINDOW_HOP)
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
    plans = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, large_canvas=True, window_hop=WINDOW_HOP)
    assert plans[0].windows[0].canvas == LARGE_CANVAS
    capped = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, large_canvas=False, window_hop=WINDOW_HOP)
    assert capped[0].windows[0].canvas == PLAN_CANVAS
    assert letterbox((0, 0, 900, 900), LARGE_CANVAS).resize_hw == (LARGE_CANVAS, LARGE_CANVAS)


def test_traced_polygons_are_compact_int32_points():
    mask = torch.zeros(16, 16)
    mask[4:10, 5:12] = 1.0
    region = trace_region(np.array([20.7, 16.2, 50.0, 41.9]), mask, 64, 64)
    assert region.box == (20.0, 16.0, 50.0, 41.0)
    assert isinstance(region.polygon, np.ndarray)
    assert region.polygon.dtype == np.int32 and region.polygon.ndim == 2 and region.polygon.shape[1] == 2
    assert RegionGroup(region.box, (region.polygon,)).extent == (0.0, 0.0, 63.0, 59.0)


def test_gap_polygons_map_points_exactly_as_float64_arithmetic():
    polygon = np.array([[3, 7], [250, 9], [101, 333]], dtype=np.int32)
    later = np.array([[11, 2], [301, 20], [140, 401]], dtype=np.int32)
    track = Track(0, {0: RegionGroup((0.0, 0.0, 1.0, 1.0), (polygon,)), 2: RegionGroup((0.0, 0.0, 1.0, 1.0), (later,))})
    (mapped,) = filled_frames(track)[1].polygons
    source = (3.0, 7.0, 250.0, 333.0)
    extent = tuple(a + (b - a) * 0.5 for a, b in zip(source, (11.0, 2.0, 301.0, 401.0)))
    scale_x = (extent[2] - extent[0]) / (source[2] - source[0])
    scale_y = (extent[3] - extent[1]) / (source[3] - source[1])
    expected = [[extent[0] + (x - source[0]) * scale_x, extent[1] + (y - source[1]) * scale_y] for x, y in polygon.tolist()]
    assert mapped.dtype == np.float64 and mapped.tolist() == expected


def _random_video(seed: int, frames: int) -> tuple[list[list[Region]], set[int]]:
    """Drifting mosaics of a few sizes with dropped detections, size jumps, nested
    continuations and random scene cuts."""
    rng = np.random.default_rng(seed)
    regions: list[list[Region]] = [[] for _ in range(frames)]
    for _ in range(frames // 60):
        start = int(rng.integers(0, frames - 10))
        length = int(rng.integers(5, 400))
        x, y = rng.uniform(0, 1500), rng.uniform(0, 800)
        side = float(rng.choice([40.0, 120.0, 400.0]))
        child = rng.random() < 0.4
        for f in range(start, min(frames, start + length)):
            if rng.random() < 0.08:
                continue
            x = min(max(x + rng.normal(0, 3), 0.0), 1400.0)
            y = min(max(y + rng.normal(0, 3), 0.0), 600.0)
            if rng.random() < 0.01:
                side = float(min(max(side * rng.choice([0.5, 2.0]), 40.0), 400.0))
            regions[f].append(_square(round(x), round(y), side))
            if child and f > start + 30:
                regions[f].append(_square(round(x + side / 3), round(y + side / 3), side / 6))
    cuts = {int(c) for c in rng.choice(np.arange(1, frames), size=frames // 400, replace=False)}
    return regions, cuts


def _plan_digest(plans) -> str:
    digest = hashlib.sha256()
    for plan in plans:
        digest.update(repr((plan.track_id, plan.start, plan.window_hop, np.round(plan.presence, 6).tolist())).encode())
        for polygons in plan.polygons:
            for polygon in polygons:
                digest.update(np.round(np.asarray(polygon, dtype=np.float64), 3).tobytes())
        for w in plan.windows:
            crops = [(*crop.crop_box, crop.canvas) for crop in w.crops]
            digest.update(repr((w.index, w.track_id, w.start, w.real_frames, crops)).encode())
    return digest.hexdigest()


def test_plan_video_output_matches_the_recorded_plan():
    regions, cuts = _random_video(0, 4000)
    plans = plan_video(regions, cuts, frame_w=1920, frame_h=1080, large_canvas=True, window_hop=window_hop(True))
    assert _plan_digest(plans) == "1099f21ad72e3a86d0a9e3b1e2d94dc4b126535674cfdddf8d70d4c3e12f3e30"


def test_plan_video_scales_linearly_with_video_length():
    frames = 64_000
    regions = [[_square(100 + f % 50, 100, 200), _square(900, 500, 200)] for f in range(frames)]
    cuts = set(range(150, frames, 150))
    started = time.perf_counter()
    plan_video(regions, cuts, frame_w=1920, frame_h=1080, large_canvas=True, window_hop=window_hop(True))
    assert time.perf_counter() - started < 15.0


def test_gap_fill_keeps_outlines_inside_the_frame_when_boxes_leave_it():
    outline = np.array([[0, 0], [60, 0], [60, 10], [0, 10]], dtype=np.int32)
    tall = Region(box=(0.0, -800.0, 100.0, 100.0), polygon=outline)
    hugging = Region(box=(0.0, -5.0, 100.0, 100.0), polygon=outline)
    regions = [[tall], [], [], []] + [[hugging] for _ in range(200)]
    (plan,) = plan_video(regions, set(), frame_w=1280, frame_h=720, large_canvas=False, window_hop=WINDOW_HOP)
    points = np.concatenate([polygon for polygons in plan.polygons for polygon in polygons])
    assert points.min() >= 0

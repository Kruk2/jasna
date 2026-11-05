import hashlib

import cv2
import numpy as np
import pytest
import torch

from jasna.ltx.camera import PLAN_CANVAS, ZOOM_PER_WINDOW, letterbox, track_camera, zoom_crop_sides
from jasna.ltx.plan import (
    BLEND_FRAMES,
    LARGE_CANVAS,
    LEAD_IN_FRAMES,
    LESS_OVERLAP_WINDOW_HOP,
    MAX_GAP,
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
    track_max_gap,
    window_hop,
    window_starts,
)
from jasna.ltx.regions import trace_region
from jasna.vr180 import StereoLayout


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
    assert len(build_tracks(frames, cuts=set(), max_gap=MAX_GAP)) == 1
    assert len(build_tracks(frames, cuts={6}, max_gap=MAX_GAP)) == 2
    frames[5] = frames[6] = frames[7] = []
    frames[3] = frames[4] = []
    assert len(build_tracks(frames, cuts=set(), max_gap=MAX_GAP)) == 2


def test_track_max_gap_only_widens_what_ltx_always_bridges():
    assert [track_max_gap(n) for n in (0, 2, MAX_GAP, 9)] == [MAX_GAP, MAX_GAP, MAX_GAP, 9]


def test_a_wider_max_gap_keeps_a_longer_dropout_in_one_filled_track():
    regions = [[_square(300, 300, 120)] for _ in range(200)]
    regions[100:106] = [[] for _ in range(6)]

    def spans(cuts, max_gap):
        plans = plan_video(regions, cuts, frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=max_gap, min_duration=0)
        return [(p.start, p.stop) for p in plans], plans

    assert spans(set(), MAX_GAP)[0] == [(0, 100), (106, 200)]
    bridged, (plan,) = spans(set(), 6)
    assert bridged == [(0, 200)]
    assert all(len(polygons) == 1 for polygons in plan.polygons[100:106])
    assert spans({103}, 6)[0] == [(0, 100), (106, 200)]


def test_merge_continuations_reaches_back_max_gap_frames():
    parent = Track(0, {i: _group(100, 100, 200) for i in range(10)})
    child = Track(1, {i: _group(150, 150, 60) for i in range(15, 25)})
    assert len(merge_continuations([parent, child], cuts=set(), frame_w=1920, frame_h=1080, max_gap=MAX_GAP)) == 2
    assert len(merge_continuations([parent, child], cuts=set(), frame_w=1920, frame_h=1080, max_gap=5)) == 1


def test_handover_leads_look_max_gap_frames_past_a_track():
    track = Track(0, {i: _group(150, 150, 100) for i in range(100, 200)})
    heir = Track(1, {i: _group(100, 100, 800) for i in range(205, 300)})
    assert handover_leads([track, heir], set(), 300, MAX_GAP)[0] == (0, 0)
    assert handover_leads([track, heir], set(), 300, 6)[0] == (0, BLEND_FRAMES)


def test_merge_continuations_folds_a_split_mosaic():
    parent = Track(0, {i: _group(100, 100, 200) for i in range(10)})
    child = Track(1, {i: _group(150, 150, 60) for i in range(5, 15)})
    elsewhere = Track(2, {i: _group(900, 900, 60) for i in range(5, 15)})
    merged = merge_continuations([parent, child, elsewhere], cuts=set(), frame_w=1920, frame_h=1080, max_gap=MAX_GAP)
    assert [t.track_id for t in merged] == [0, 2]
    assert merged[0].first == 0 and merged[0].last == 14
    assert len(merged[0].frames[6].polygons) == 2
    assert len(merge_continuations([parent, child], cuts={5}, frame_w=1920, frame_h=1080, max_gap=MAX_GAP)) == 2


def test_merge_continuations_keeps_the_gap_fill_of_both_tracks():
    parent = Track(0, {i: _group(100, 100, 200) for i in range(10) if i != 7})
    child = Track(1, {i: _group(150, 150, 60) for i in range(5, 15) if i != 8})
    merged = merge_continuations([parent, child], cuts=set(), frame_w=1920, frame_h=1080, max_gap=MAX_GAP)
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
    plans = plan_video(_split_video(), set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
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


def _rectangle(x1: int, y1: int, x2: int, y2: int) -> Region:
    return Region((x1, y1, x2, y2), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))


def _outline(*points: tuple[int, int]) -> Region:
    polygon = np.array(points, dtype=np.int32)
    (x1, y1), (x2, y2) = polygon.min(axis=0), polygon.max(axis=0)
    return Region((float(x1), float(y1), float(x2), float(y2)), polygon)


def _plan(frames: list[list[Region]]):
    return plan_video(frames, set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)


def _widest_crop(plan) -> int:
    return max(crop.side for window in plan.windows for crop in window.crops)


def _mask(polygons) -> np.ndarray:
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


_LEFT, _RIGHT, _BOTH = _rectangle(200, 300, 400, 500), _rectangle(420, 300, 620, 500), _rectangle(200, 300, 620, 500)


def test_a_detection_briefly_spanning_two_mosaics_keeps_both_cameras():
    apart = [[_LEFT, _RIGHT] for _ in range(300)]
    spanned = apart[:150] + [[_BOTH] for _ in range(20)] + [[_LEFT, _RIGHT, _BOTH] for _ in range(5)] + apart[175:]
    plans = _plan(spanned)
    assert [(p.start, p.stop) for p in plans] == [(0, 300), (0, 300)]
    assert all(set(p.presence) == {1.0} for p in plans)
    for plan, alone in zip(plans, _plan(apart), strict=True):
        assert _widest_crop(plan) <= 1.02 * _widest_crop(alone)
    for frame in (150, 160, 172):
        left, right = (_mask(p.polygons[frame]) for p in plans)
        assert not (left & right).any()
        assert np.array_equal(left | right, _mask([_BOTH.polygon]))


def test_tiny_mosaics_keep_their_tracks_through_a_spanning_detection():
    """Real outlines are dilated around the model box; a cut part's box must not keep that margin."""

    def dilated(x1, y1, x2, y2, margin=20):
        outline = [[x1 - margin, y1 - margin], [x2 + margin, y1 - margin], [x2 + margin, y2 + margin], [x1 - margin, y2 + margin]]
        return Region((x1, y1, x2, y2), np.array(outline, dtype=np.int32))

    left, right, both = dilated(300, 300, 310, 310), dilated(360, 300, 370, 310), dilated(300, 300, 370, 310)
    frames = [[left, right] for _ in range(150)] + [[both] for _ in range(20)] + [[left, right] for _ in range(130)]
    tracks = build_tracks([group_regions(regions) for regions in frames], set(), MAX_GAP)
    assert [(t.first, t.last) for t in tracks] == [(0, 299), (0, 299)]


def test_a_mosaic_breaking_up_beside_another_keeps_both_tracks():
    """For a few frames a piece of the big mosaic comes glued to its neighbour, then one detection covers both."""
    mosaic, neighbour = _rectangle(560, 580, 1780, 1080), _rectangle(1490, 250, 1920, 700)
    piece, rest = _rectangle(600, 625, 1170, 915), _rectangle(1220, 580, 1780, 1080)
    glued = _outline((1220, 580), (1490, 580), (1490, 250), (1920, 250), (1920, 700), (1780, 700), (1780, 1080), (1220, 1080))
    both = _outline((560, 580), (1490, 580), (1490, 250), (1920, 250), (1920, 700), (1780, 700), (1780, 1080), (560, 1080))
    apart = [[mosaic, neighbour] for _ in range(300)]
    plans = _plan(apart[:150] + [[piece, rest, neighbour]] + [[piece, glued] for _ in range(6)] + [[both]] + apart[158:])
    assert [(p.start, p.stop) for p in plans] == [(0, 300), (0, 300)]
    assert all(set(p.presence) == {1.0} for p in plans)
    for plan, alone in zip(plans, _plan(apart), strict=True):
        assert _widest_crop(plan) <= 1.02 * _widest_crop(alone)


def test_a_lasting_merge_still_hands_over():
    plans = _plan([[_LEFT, _RIGHT] for _ in range(150)] + [[_BOTH] for _ in range(200)])
    assert [(p.start, p.stop) for p in plans] == [(0, 150 + BLEND_FRAMES), (0, 350)]
    assert plans[0].presence[-1] < 1.0
    assert set(plans[1].presence) == {1.0}


def test_a_mosaic_seen_for_a_few_frames_gets_no_part_of_a_spanning_detection():
    frames = [[_LEFT] for _ in range(100)] + [[_LEFT, _RIGHT] for _ in range(5)] + [[_BOTH] for _ in range(10)] + [[_LEFT, _RIGHT] for _ in range(185)]
    outlines = [_mask(p.polygons[110 - p.start]) for p in _plan(frames) if p.start <= 110 < p.stop]
    assert any(np.array_equal(outline, _mask([_BOTH.polygon])) for outline in outlines)


def test_a_small_detection_inside_a_mosaic_keeps_its_own_outline():
    """The mosaic around the small detection is missed now and then; its neighbour overlaps it."""
    mosaic, neighbour, inner = _rectangle(700, 140, 860, 300), _rectangle(820, 200, 940, 320), _rectangle(775, 195, 801, 221)
    plans = _plan([[neighbour, inner] if t % 40 in (10, 11) else [mosaic, neighbour, inner] for t in range(300)])
    widths = [max(np.ptp(np.concatenate(polygons)[:, 0]) for polygons in p.polygons) for p in plans]
    assert min(widths) == 26


def test_handover_leads_stop_at_cuts_and_hand_back_to_a_covering_track():
    parent = Track(0, {i: _group(100, 100, 800) for i in range(300)})
    child = Track(1, {i: _group(150, 150, 100) for i in range(100, 200)})
    assert handover_leads([parent, child], set(), 300, MAX_GAP) == [(0, 0), (LEAD_IN_FRAMES, BLEND_FRAMES)]
    assert handover_leads([parent, child], {90}, 300, MAX_GAP)[1] == (10, BLEND_FRAMES)
    assert handover_leads([parent, child], {200}, 300, MAX_GAP)[1] == (LEAD_IN_FRAMES, 0)
    assert handover_leads([parent, child], {202}, 300, MAX_GAP)[1] == (LEAD_IN_FRAMES, 2)


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


def test_min_duration_drops_short_tracks_but_not_a_short_size_run_of_a_long_one():
    regions = [[_square(100, 100, 800 if t < 200 else 400)] for t in range(600)]
    regions[400] = [*regions[400], _square(1500, 700, 100)]

    def spans(min_duration):
        plans = plan_video(regions, set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=min_duration)
        return [(p.start, p.stop) for p in plans]

    assert (400, 401) in spans(0)
    assert spans(2) == [span for span in spans(0) if span != (400, 401)]
    assert spans(300) == [(0, 200), (200 - LEAD_IN_FRAMES, 600)]


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


def test_window_hop_from_the_overlap_frames():
    assert (window_hop(33), window_hop(17)) == (88, 104)


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
    (plan,) = plan_video(regions, set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=LESS_OVERLAP_WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
    assert [w.start for w in plan.windows] == [0, 104, 208]
    assert plan.window_hop == LESS_OVERLAP_WINDOW_HOP


def test_crossfade_weights_partition_unity():
    plans = plan_video([[_square(300 + t, 300, 120)] for t in range(300)], set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=True, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
    windows = plans[0].windows
    total = np.zeros(300)
    for window, weight in zip(windows, crossfade_weights(windows)):
        total[window.start : window.stop] += weight
    assert np.allclose(total, 1.0)


@pytest.mark.parametrize("hop", [WINDOW_HOP, LESS_OVERLAP_WINDOW_HOP])
def test_crossfade_hides_a_later_windows_unfused_lead_frames(hop):
    """A later window's free latent 0 sits in front of fused latents, and the decoder
    garbles the colour of that latent and the next one (#580): its first 9 frames get no weight."""
    plans = plan_video([[_square(300, 300, 120)] for _ in range(400)], set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=hop, max_gap=MAX_GAP, min_duration=0)
    windows = plans[0].windows
    weights = crossfade_weights(windows)
    overlap = windows[0].stop - windows[1].start
    for weight in weights[1:]:
        assert set(weight[:9]) == {0.0}
        assert weight[9] > 0.0
        assert np.all(np.diff(weight[:overlap]) >= 0.0) and weight[overlap - 1] == 1.0


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


def test_track_camera_eases_around_a_brief_box_spike():
    boxes = [(833.0, 235.0, 1059.0, 731.0)] * 95 + [(833.0, 295.0, 1059.0, 761.0)] * 10 + [(833.0, 235.0, 1059.0, 731.0)] * 95
    boxes[100] = (833.0, 295.0, 1059.0, 784.0)
    camera = track_camera(boxes, frame_w=1920, frame_h=1080, feather=30)
    tops = np.array([crop.crop_box[1] for crop in camera])
    assert np.abs(np.diff(tops, 2)).max() <= 2
    for crop, box in zip(camera, boxes):
        x1, y1, x2, y2 = crop.crop_box
        assert x1 <= box[0] and y1 <= box[1] and x2 >= box[2] and y2 >= box[3]


def test_large_crops_use_the_large_canvas():
    plans = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=True, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
    assert plans[0].windows[0].canvas == LARGE_CANVAS
    capped = plan_video([[_square(200, 100, 700)] for _ in range(50)], set(), frame_w=1920, frame_h=1080, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
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
    plans = plan_video(regions, cuts, frame_w=1920, frame_h=1080, stereo=None, large_canvas=True, window_hop=window_hop(17), max_gap=MAX_GAP, min_duration=0)
    assert _plan_digest(plans) == "29215532cc269f4f95eaad4cf5b42dee38008dd256276efd1c0b0bee6b8fb408"


def test_gap_fill_keeps_outlines_inside_the_frame_when_boxes_leave_it():
    outline = np.array([[0, 0], [60, 0], [60, 10], [0, 10]], dtype=np.int32)
    tall = Region(box=(0.0, -800.0, 100.0, 100.0), polygon=outline)
    hugging = Region(box=(0.0, -5.0, 100.0, 100.0), polygon=outline)
    regions = [[tall], [], [], []] + [[hugging] for _ in range(200)]
    (plan,) = plan_video(regions, set(), frame_w=1280, frame_h=720, stereo=None, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)
    points = np.concatenate([polygon for polygons in plan.polygons for polygon in polygons])
    assert points.min() >= 0


def _eye_square(stereo: StereoLayout, eye: int, x: float, y: float, side: float) -> Region:
    """A square at (x, y) in eye ``eye`` (0 left, 1 right)."""
    dx, dy = stereo.right_offset
    return _square(x + eye * dx, y + eye * dy, side)


@pytest.mark.parametrize("kind", ["sbs", "tb"])
def test_stereo_mosaics_too_far_apart_for_one_subject_are_planned_per_eye(kind):
    eye = 1000
    stereo = StereoLayout(kind, eye * (2 if kind == "sbs" else 1), eye * (2 if kind == "tb" else 1))
    axis = 0 if kind == "sbs" else 1
    regions = [
        [_eye_square(stereo, 0, *np.roll((eye - 160 + t % 3, 300), axis), 150), _eye_square(stereo, 1, *np.roll((5 + t % 3, 300), axis), 150)]
        for t in range(250)
    ]
    settings = dict(frame_w=stereo.frame_w, frame_h=stereo.frame_h, large_canvas=False, window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0)

    flat = plan_video(regions, set(), stereo=None, **settings)
    plans = plan_video(regions, set(), stereo=stereo, **settings)

    assert any(c.crop_box[axis] < eye < c.crop_box[axis + 2] for p in flat for w in p.windows for c in w.crops)
    assert [p.track_id for p in plans] == [0, 1]
    windows = [w for p in plans for w in p.windows]
    assert [w.index for w in windows] == list(range(len(windows)))
    assert [w.noise for w in windows] == [w.index for w in windows]
    for plan, (low, high) in zip(plans, ((0, eye), (eye, 2 * eye))):
        assert all(low <= c.crop_box[axis] and c.crop_box[axis + 2] <= high for w in plan.windows for c in w.crops)
        points = np.concatenate([polygon for polygons in plan.polygons for polygon in polygons])
        assert low <= points[:, axis].min() and points[:, axis].max() <= high


@pytest.mark.parametrize("kind", ["sbs", "tb"])
def test_stereo_right_eye_restores_each_left_track_as_a_twin_with_its_noise(kind):
    eye, disparity = 1000, -40
    stereo = StereoLayout(kind, eye * (2 if kind == "sbs" else 1), eye * (2 if kind == "tb" else 1))
    regions = [
        [
            _eye_square(stereo, 0, 300 + t // 10, 400, 150),
            _eye_square(stereo, 1, 300 + disparity + t // 10, 400, 150),
            _eye_square(stereo, 1, 700, 700, 150),
        ]
        for t in range(250)
    ]
    left, twin, right_only = plan_video(
        regions, set(), frame_w=stereo.frame_w, frame_h=stereo.frame_h, stereo=stereo, large_canvas=False,
        window_hop=WINDOW_HOP, max_gap=MAX_GAP, min_duration=0,
    )

    assert [w.noise for w in twin.windows] == [w.index for w in left.windows]
    assert [w.noise for w in right_only.windows] == [w.index for w in right_only.windows]
    windows = [w for plan in (left, twin, right_only) for w in plan.windows]
    assert [w.index for w in windows] == list(range(len(windows)))
    dx, dy = stereo.right_offset
    for left_window, twin_window in zip(left.windows, twin.windows):
        for left_crop, twin_crop in zip(left_window.crops, twin_window.crops):
            x1, y1, x2, y2 = left_crop.crop_box
            assert twin_crop.crop_box == (x1 + dx + disparity, y1 + dy, x2 + dx + disparity, y2 + dy)

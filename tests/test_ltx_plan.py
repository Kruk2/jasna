import numpy as np
import pytest

from jasna.ltx.camera import PLAN_CANVAS, ZOOM_PER_WINDOW, letterbox, track_camera, zoom_crop_sides
from jasna.ltx.plan import (
    BLEND_FRAMES,
    LARGE_CANVAS,
    LEAD_IN_FRAMES,
    LEFT_SHARED_LATENTS,
    RIGHT_SHARED_LATENTS,
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
    plans = plan_video(_split_video(), set(), frame_w=1920, frame_h=1080, large_canvas=False)
    assert [(p.start, p.stop) for p in plans] == [(0, 130), (100 - LEAD_IN_FRAMES, 500)]
    piece = plans[1]
    assert len(piece.presence) == len(piece.polygons) == piece.stop - piece.start
    blend = piece.presence[LEAD_IN_FRAMES - BLEND_FRAMES : LEAD_IN_FRAMES]
    assert set(piece.presence[: LEAD_IN_FRAMES - BLEND_FRAMES]) == {0.0}
    assert 0.0 < blend[0] and all(a < b for a, b in zip(blend, blend[1:])) and blend[-1] < 1.0
    assert set(piece.presence[LEAD_IN_FRAMES:]) == {1.0}
    assert piece.polygons[0] == piece.polygons[LEAD_IN_FRAMES]
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

"""Mosaic tracks and their 121-frame model windows for LTX restoration.

Detections are grouped per frame, linked into cut-safe tracks, continuations are folded
into their track, and each track gets one zooming camera sliced into windows at hop 88
(104 with less overlap). Windows of a track are denoised together and share their
overlapping latent frames.

A group covering the last outlines of two or more established tracks that show up apart
again within half a window (one detection spanning neighbouring mosaics, or a piece of
one mosaic glued to its neighbour) is cut between them by nearest outline
(``share_groups``) instead of widening one track and ending or swapping the other.

A continuation stays its own track when folding it would cost its camera
``SEPARATE_GAIN`` sharpness for at least a window (``worth_separating``), and a track is
cut where its size changes for good (``restart_cameras``). A track starting inside
another hands over from it: it starts ``LEAD_IN_FRAMES`` early on its first outline and
takes the area over during the last ``BLEND_FRAMES`` of them; a track ending inside
another hands the area back over ``BLEND_FRAMES``.

Stereo VR (side by side or top/bottom) plans the left eye on its own; the right eye restores each left track as a twin
shifted by its disparity and denoised with the same noise, so both eyes invent the same
detail.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from jasna.ltx.camera import (
    PLAN_CANVAS,
    SMOOTH_SIGMA_FRAMES,
    WINDOW_FRAMES,
    Box,
    CropFrame,
    gaussian_smooth,
    lada_crop_boxes,
    letterbox,
    track_camera,
)
from jasna.tracking.clip_tracker import compute_iou_matrix, shorter_than_min_duration
from jasna.vr180 import StereoLayout

Polygon = np.ndarray

WINDOW_HOP = 88
LESS_OVERLAP_WINDOW_HOP = 104
TEMPORAL_SCALE = 8
GROUP_IOU = 0.05
LINK_IOU = 0.1
MAX_GAP = 3
CONTINUATION_COVER = 0.5
SEPARATE_GAIN = 1.3
LEAD_IN_FRAMES = 25
BLEND_FRAMES = 16
LARGE_CANVAS = 768
LARGE_CANVAS_MIN_SIDE = 640
FEATHER_FRACTION = 0.028
SHARE_COVER = 0.3
SHARE_OWN = 0.1
SHARE_MIN_FRAMES = 16
SHARE_HORIZON = WINDOW_FRAMES // 2
OUTLINE_SCALE = 4
SCRAP_SHARE = 0.01


@dataclass(frozen=True)
class Region:
    """One detected mosaic: model box and the traced outline of its dilated mask, ``(N, 2)``
    int32 points (float64 once mapped onto a gap frame)."""

    box: Box
    polygon: Polygon


@dataclass(frozen=True)
class RegionGroup:
    box: Box
    polygons: tuple[Polygon, ...]

    @functools.cached_property
    def extent(self) -> Box:
        """Bounds of the polygon points."""
        return _extent(self.polygons)

    @functools.cached_property
    def largest_area(self) -> float:
        """Area of the largest polygon, at most the area of the outline."""
        import cv2  # the LTX worker imports this module and must not load OpenCV
        return max(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32))) for polygon in self.polygons)


@dataclass
class Track:
    track_id: int
    frames: dict[int, RegionGroup] = field(default_factory=dict)  # in ascending frame order

    @property
    def first(self) -> int:
        return next(iter(self.frames))

    @property
    def last(self) -> int:
        return next(reversed(self.frames))


@dataclass(frozen=True)
class Window:
    index: int  # global across tracks; also the noise seed offset
    track_id: int
    start: int
    real_frames: int  # the model always sees WINDOW_FRAMES; the tail is padded
    crops: tuple[CropFrame, ...]  # one per real frame
    noise: int  # noise seed offset: the index, or the left window's for a right-eye twin

    @property
    def stop(self) -> int:
        return self.start + self.real_frames

    @property
    def canvas(self) -> int:
        return self.crops[0].canvas


@dataclass(frozen=True)
class TrackPlan:
    track_id: int
    start: int
    polygons: tuple[tuple[Polygon, ...], ...]  # per frame from ``start``
    presence: tuple[float, ...]  # per frame: the track's share of its area, below 1 during a handover
    windows: tuple[Window, ...]
    window_hop: int

    @property
    def stop(self) -> int:
        return self.start + len(self.polygons)


def plans_to_arrays(segments: Sequence[Sequence[TrackPlan]]) -> dict[str, np.ndarray]:
    """Every segment's track plans as flat arrays for ``np.savez``; ``plans_from_arrays`` reads them back."""
    plans = [(s, plan) for s, segment in enumerate(segments) for plan in segment]
    polygons = [polygon for _, plan in plans for frame in plan.polygons for polygon in frame]
    windows = [(t, window) for t, (_, plan) in enumerate(plans) for window in plan.windows]
    return {
        "segments": np.array(len(segments)),
        "tracks": np.array(
            [(s, plan.track_id, plan.start, plan.window_hop, len(plan.polygons)) for s, plan in plans], dtype=np.int64
        ).reshape(-1, 5),
        "presence": np.array([value for _, plan in plans for value in plan.presence], dtype=np.float64),
        "frame_polygons": np.array([len(frame) for _, plan in plans for frame in plan.polygons], dtype=np.int64),
        "polygon_points": np.array([len(polygon) for polygon in polygons], dtype=np.int64),
        "polygon_float": np.array([polygon.dtype == np.float64 for polygon in polygons], dtype=bool),
        "points": np.concatenate([np.zeros((0, 2)), *polygons]),
        "windows": np.array(
            [(t, w.index, w.track_id, w.start, w.real_frames, w.noise) for t, w in windows], dtype=np.int64
        ).reshape(-1, 6),
        "crops": np.array(
            [(*c.crop_box, *c.resize_hw, *c.pad) for _, w in windows for c in w.crops], dtype=np.int64
        ).reshape(-1, 10),
    }


def plans_from_arrays(arrays: Mapping[str, np.ndarray]) -> list[list[TrackPlan]]:
    points = np.split(arrays["points"], np.cumsum(arrays["polygon_points"])[:-1])
    polygons = iter(
        polygon if is_float else polygon.astype(np.int32) for polygon, is_float in zip(points, arrays["polygon_float"])
    )
    frame_polygons = iter(arrays["frame_polygons"].tolist())
    presence = iter(arrays["presence"].tolist())
    crops = iter(CropFrame(tuple(c[:4]), tuple(c[4:6]), tuple(c[6:])) for c in arrays["crops"].tolist())
    windows: dict[int, list[Window]] = {}
    for t, index, track_id, start, real_frames, noise in arrays["windows"].tolist():
        windows.setdefault(t, []).append(
            Window(index, track_id, start, real_frames, tuple(itertools.islice(crops, real_frames)), noise)
        )
    segments: list[list[TrackPlan]] = [[] for _ in range(int(arrays["segments"]))]
    for t, (s, track_id, start, window_hop, frames) in enumerate(arrays["tracks"].tolist()):
        frame_tuples = tuple(tuple(itertools.islice(polygons, next(frame_polygons))) for _ in range(frames))
        segments[s].append(TrackPlan(
            track_id, start, frame_tuples, tuple(itertools.islice(presence, frames)), tuple(windows.get(t, ())), window_hop,
        ))
    return segments


def feather_pixels(frame_h: int) -> int:
    return max(1, int(round(FEATHER_FRACTION * frame_h)))


def _union_box(boxes: Sequence[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _first_cut_after(sorted_cuts: Sequence[int], frame_idx: int, default: int) -> int:
    position = bisect.bisect_right(sorted_cuts, frame_idx)
    return sorted_cuts[position] if position < len(sorted_cuts) else default


def _cut_between(sorted_cuts: Sequence[int], last: int, frame_idx: int) -> bool:
    """Whether a scene cut separates frame ``last`` from the later ``frame_idx``."""
    return _first_cut_after(sorted_cuts, last, frame_idx + 1) <= frame_idx


def _last_cut_at_or_before(sorted_cuts: Sequence[int], frame_idx: int) -> int:
    position = bisect.bisect_right(sorted_cuts, frame_idx)
    return sorted_cuts[position - 1] if position > 0 else 0


def group_regions(regions: Sequence[Region]) -> list[RegionGroup]:
    """Union-find merge of regions whose boxes overlap by IoU >= ``GROUP_IOU``."""
    count = len(regions)
    if count == 0:
        return []
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    boxes = np.asarray([r.box for r in regions], dtype=np.float64)
    iou = compute_iou_matrix(boxes, boxes)
    for i in range(count):
        for j in range(i + 1, count):
            if iou[i, j] >= GROUP_IOU:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_i] = root_j
    buckets: dict[int, list[Region]] = {}
    for i in range(count):
        buckets.setdefault(find(i), []).append(regions[i])
    return [
        RegionGroup(box=_union_box([r.box for r in members]), polygons=tuple(r.polygon for r in members))
        for members in buckets.values()
    ]


def track_max_gap(max_detection_gap: int) -> int:
    """Max missing frames an LTX track bridges: ``max(MAX_GAP, --max-detection-gap)``
    (the default of 2 is for BasicVSR++, below LTX's 3-frame minimum)."""
    return max(MAX_GAP, max_detection_gap)


class _Raster:
    """Outlines rasterized ``OUTLINE_SCALE`` times coarser on one grid covering ``extents``."""

    def __init__(self, extents: Sequence[Box]) -> None:
        self.x0 = math.floor(min(extent[0] for extent in extents))
        self.y0 = math.floor(min(extent[1] for extent in extents))
        self.x1 = math.ceil(max(extent[2] for extent in extents)) + 1
        self.y1 = math.ceil(max(extent[3] for extent in extents)) + 1
        self.shape = ((self.y1 - self.y0) // OUTLINE_SCALE + 2, (self.x1 - self.x0) // OUTLINE_SCALE + 2)

    def touches(self, box: Box) -> bool:
        return box[0] < self.x1 and box[2] > self.x0 and box[1] < self.y1 and box[3] > self.y0

    def mask(self, polygons: Sequence[Polygon]) -> np.ndarray:
        """uint8 cells inside any of ``polygons`` (one fill each, so overlaps stay filled)."""
        import cv2
        mask = np.zeros(self.shape, dtype=np.uint8)
        origin = np.array((self.x0, self.y0), dtype=np.float64)
        for polygon in polygons:
            cell_polygon = ((np.asarray(polygon, dtype=np.float64) - origin) / OUTLINE_SCALE).round().astype(np.int32)
            cv2.fillPoly(mask, [cell_polygon], 1)
        return mask

    def pixels(self, grid: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """``grid`` values over the pixels ``x0..x1``, ``y0..y1`` (half-open) inside the raster,
        each pixel taking its nearest cell as ``mask`` rounds polygon points."""
        half = OUTLINE_SCALE // 2
        top, left = (y0 - self.y0 + half) // OUTLINE_SCALE, (x0 - self.x0 + half) // OUTLINE_SCALE
        bottom, right = (y1 - 1 - self.y0 + half) // OUTLINE_SCALE + 1, (x1 - 1 - self.x0 + half) // OUTLINE_SCALE + 1
        block = grid[top:bottom, left:right].repeat(OUTLINE_SCALE, axis=0).repeat(OUTLINE_SCALE, axis=1)
        row = y0 - self.y0 - top * OUTLINE_SCALE + half
        col = x0 - self.x0 - left * OUTLINE_SCALE + half
        return block[row : row + y1 - y0, col : col + x1 - x0].copy()


def _covered(mask: np.ndarray, outline: np.ndarray) -> float:
    """Share of the cells of ``outline`` inside ``mask``."""
    area = np.count_nonzero(outline)
    return np.count_nonzero(mask & outline) / area if area else 0.0


def _seen_apart(raster: _Raster, outlines: Sequence[np.ndarray], ahead: Sequence[Sequence[RegionGroup]]) -> list[bool]:
    """Per outline: whether a frame of ``ahead`` covers it (``SHARE_COVER``) with groups that
    cover none of the other outlines."""
    apart = [False] * len(outlines)
    for groups in ahead:
        holds = [
            [_covered(mask, outline) >= SHARE_COVER for outline in outlines]
            for mask in (raster.mask(group.polygons) for group in groups if raster.touches(group.extent))
        ]
        for k in range(len(outlines)):
            holders = [held for held in holds if held[k]]
            if holders and not any(held[j] for held in holders for j in range(len(outlines)) if j != k):
                apart[k] = True
        if all(apart):
            break
    return apart


def _cut(group: RegionGroup, raster: _Raster, outlines: Sequence[np.ndarray]) -> list[RegionGroup | None]:
    """The group's outlines cut so each pixel goes to the nearest of ``outlines`` (signed
    distance on the coarse grid); a cut piece under ``SCRAP_SHARE`` of its outline stays with
    that outline's largest part. One part per outline, None when it gets no pixel."""
    import cv2
    nearest_cell = np.argmin(
        [cv2.distanceTransform(1 - outline, cv2.DIST_L2, 3) - cv2.distanceTransform(outline, cv2.DIST_L2, 3) for outline in outlines],
        axis=0,
    ).astype(np.uint8)
    outside = len(outlines)
    pieces: list[list[Polygon]] = [[] for _ in outlines]
    for polygon in group.polygons:
        owners = np.unique(nearest_cell[raster.mask((polygon,)) > 0])
        if len(owners) == 1:
            pieces[owners[0]].append(polygon)
            continue
        left, top, right, bottom = _extent((polygon,))
        x0, y0, x1, y1 = math.floor(left), math.floor(top), math.ceil(right) + 1, math.ceil(bottom) + 1
        inside = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(inside, [np.asarray(polygon, dtype=np.float64).round().astype(np.int32) - (x0, y0)], 1)
        nearest = raster.pixels(nearest_cell, x0, y0, x1, y1)
        nearest[inside == 0] = outside
        counts = np.bincount(nearest.ravel(), minlength=outside + 1)[:outside]
        largest = int(counts.argmax())
        for k in np.flatnonzero(counts):
            if k == largest:
                continue
            _, labels, stats, _ = cv2.connectedComponentsWithStats((nearest == k).view(np.uint8), connectivity=8)
            scrap = stats[:, cv2.CC_STAT_AREA] < SCRAP_SHARE * counts.sum()
            scrap[0] = False
            nearest[scrap[labels]] = largest
        if np.count_nonzero(nearest == largest) == counts.sum():
            pieces[largest].append(polygon)
            continue
        for k in range(outside):
            contours, _ = cv2.findContours((nearest == k).view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pieces[k].extend(contour[:, 0, :] + np.array((x0, y0), dtype=np.int32) for contour in contours if len(contour) >= 3)
    return [RegionGroup(_model_box(found, group.box), tuple(found)) if found else None for found in pieces]


def _model_box(polygons: Sequence[Polygon], box: Box) -> Box:
    """A cut part's box: its outline bounds clipped to the detection's model box, since the
    outlines are dilated (``EXPAND_PIXELS``) and other frames link on model boxes."""
    left, top, right, bottom = _extent(polygons)
    clipped = max(left, box[0]), max(top, box[1]), min(right, box[2]), min(bottom, box[3])
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else (left, top, right, bottom)


def _may_share(group: RegionGroup, last: RegionGroup) -> bool:
    """Whether the outline bounds overlap enough for ``_share``'s outline tests to pass."""
    overlap = _intersection(group.extent, last.extent)
    return overlap >= SHARE_COVER * last.largest_area and overlap >= SHARE_OWN * group.largest_area


def _share(
    group: RegionGroup, tracks: Sequence[Track], ahead: Sequence[Sequence[RegionGroup]]
) -> tuple[list[tuple[Track, RegionGroup]], RegionGroup | None] | None:
    """``share_groups`` for one group and the tracks near it, None when it is not shared."""
    raster = _Raster([group.extent, *(track.frames[track.last].extent for track in tracks)])
    group_mask = raster.mask(group.polygons)
    owners = []
    for track in tracks:
        outline = raster.mask(track.frames[track.last].polygons)
        if _covered(group_mask, outline) >= SHARE_COVER and _covered(outline, group_mask) >= SHARE_OWN:
            owners.append((track, outline))
    outlines = [outline for _, outline in owners]
    if len(owners) < 2 or any(_covered(a, b) >= CONTINUATION_COVER for a, b in itertools.permutations(outlines, 2)):
        return None
    apart = _seen_apart(raster, outlines, ahead)
    if not any(apart):
        return None
    parts = _cut(group, raster, outlines)
    given = [(track, part) for (track, _), part, alone in zip(owners, parts, apart) if alone and part is not None]
    rest = [polygon for part, alone in zip(parts, apart) if not alone and part is not None for polygon in part.polygons]
    return given, RegionGroup(_model_box(rest, group.box), tuple(rest)) if rest else None


def share_groups(
    groups: Sequence[RegionGroup], tracks: Sequence[Track], ahead: Sequence[Sequence[RegionGroup]]
) -> tuple[list[tuple[Track, RegionGroup]], list[RegionGroup]]:
    """Cut each group shared by tracks seen on at least ``SHARE_MIN_FRAMES`` frames: it covers
    ``SHARE_COVER`` of their last outlines, each of those covers ``SHARE_OWN`` of it, and no two
    overlap by ``CONTINUATION_COVER`` (one mosaic). A track seen apart again within ``ahead``
    takes its part (``_cut``); the parts of the others stay one group. Returns the parts given
    to tracks and the groups left to link."""
    established = [track for track in tracks if len(track.frames) >= SHARE_MIN_FRAMES]
    given: list[tuple[Track, RegionGroup]] = []
    remaining: list[RegionGroup] = []
    for group in groups:
        near = [
            track
            for track in established
            if all(track is not taken for taken, _ in given) and _may_share(group, track.frames[track.last])
        ]
        shared = _share(group, near, ahead) if len(near) >= 2 else None
        if shared is None:
            remaining.append(group)
            continue
        parts, rest = shared
        given.extend(parts)
        if rest is not None:
            remaining.append(rest)
    return given, remaining


def build_tracks(
    groups_per_frame: Sequence[Sequence[RegionGroup]], cuts: set[int], max_gap: int
) -> list[Track]:
    """Greedy best-IoU linking; a track tolerates ``max_gap`` missing frames and never
    crosses a scene cut (``cuts`` holds first-frame-after-cut indices). A group shared by
    tracks is first cut between them (``share_groups``, looking ``SHARE_HORIZON`` frames ahead)."""
    sorted_cuts = sorted(cuts)
    tracks: list[Track] = []
    active: list[Track] = []
    for frame_idx, groups in enumerate(groups_per_frame):
        active = [
            track
            for track in active
            if frame_idx - track.last - 1 <= max_gap and not _cut_between(sorted_cuts, track.last, frame_idx)
        ]
        ahead_stop = min(_first_cut_after(sorted_cuts, frame_idx, len(groups_per_frame)), frame_idx + 1 + SHARE_HORIZON)
        given, groups = share_groups(groups, active, groups_per_frame[frame_idx + 1 : ahead_stop])
        for track, part in given:
            track.frames[frame_idx] = part
        linkable = [track for track in active if frame_idx not in track.frames]
        candidates: list[tuple[float, int, int]] = []
        if linkable and groups:
            iou = compute_iou_matrix(
                np.asarray([track.frames[track.last].box for track in linkable], dtype=np.float64),
                np.asarray([group.box for group in groups], dtype=np.float64),
            )
            candidates = [
                (float(iou[ti, gi]), ti, gi)
                for ti in range(len(linkable))
                for gi in range(len(groups))
                if iou[ti, gi] >= LINK_IOU
            ]
            candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_groups: set[int] = set()
        for _, ti, gi in candidates:
            if ti in used_tracks or gi in used_groups:
                continue
            linkable[ti].frames[frame_idx] = groups[gi]
            used_tracks.add(ti)
            used_groups.add(gi)
        for gi, group in enumerate(groups):
            if gi not in used_groups:
                track = Track(track_id=len(tracks), frames={frame_idx: group})
                tracks.append(track)
                active.append(track)
    return tracks


def _intersection(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def _cover(outer: Box, inner: Box) -> float:
    """Intersection of two boxes over the area of ``inner``."""
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return _intersection(outer, inner) / area if area > 0 else 0.0


def _fold(into: dict[int, RegionGroup], frames: dict[int, RegionGroup]) -> dict[int, RegionGroup]:
    out = dict(into)
    for idx, group in frames.items():
        other = out.get(idx)
        out[idx] = (
            group
            if other is None
            else RegionGroup(_union_box([other.box, group.box]), other.polygons + group.polygons)
        )
    return dict(sorted(out.items()))


def camera_sides(frames: dict[int, RegionGroup], frame_w: int, frame_h: int) -> np.ndarray:
    """The track camera's crop side on every frame of a gap-free track."""
    boxes = [support_box(group, frame_w, frame_h) for group in frames.values()]
    camera = track_camera(boxes, frame_w=frame_w, frame_h=frame_h, feather=feather_pixels(frame_h))
    return np.asarray([crop.side for crop in camera])


def _gains(whole: dict[int, RegionGroup], part: dict[int, RegionGroup], frame_w: int, frame_h: int) -> bool:
    """Whether ``part`` alone gets a ``SEPARATE_GAIN`` sharper camera than inside ``whole``
    on at least a window of its frames."""
    first = next(iter(whole))
    sides = camera_sides(whole, frame_w, frame_h)[[idx - first for idx in part]]
    return int((sides / camera_sides(part, frame_w, frame_h) >= SEPARATE_GAIN).sum()) >= WINDOW_FRAMES


def worth_separating(parent: dict[int, RegionGroup], child: dict[int, RegionGroup], frame_w: int, frame_h: int) -> bool:
    """Whether the child's own camera beats the folded track's (``_gains``), judged on the
    parent's frames within a window of the child."""
    first, last = next(iter(child)), next(reversed(child))
    near = {idx: parent[idx] for idx in range(first - WINDOW_FRAMES, last + WINDOW_FRAMES + 1) if idx in parent}
    return _gains(filled_frames(Track(-1, _fold(near, child))), child, frame_w, frame_h)


def _size_runs(frames: dict[int, RegionGroup], frame_w: int, frame_h: int) -> list[tuple[int, int]]:
    """Half-open frame ranges of a gap-free track split at ``SEPARATE_GAIN`` jumps of its
    crop size, keeping only runs of at least a window (neighbours of similar median size
    with at most ``LEAD_IN_FRAMES`` between count as one: a cut there would cost more
    than it gains); the frames between two runs join the one whose median size is closer."""
    boxes = lada_crop_boxes([support_box(group, frame_w, frame_h) for group in frames.values()], frame_w, frame_h)
    sizes = np.log(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))
    bounds = [0, *(np.flatnonzero(np.abs(np.diff(sizes)) >= np.log(SEPARATE_GAIN)) + 1), len(sizes)]
    lasting: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < WINDOW_FRAMES:
            continue
        if (
            lasting
            and a - lasting[-1][1] <= LEAD_IN_FRAMES
            and abs(np.median(sizes[a:b]) - np.median(sizes[slice(*lasting[-1])])) < np.log(SEPARATE_GAIN)
        ):
            lasting[-1] = (lasting[-1][0], int(b))
        else:
            lasting.append((int(a), int(b)))
    first = next(iter(frames))
    if len(lasting) < 2:
        return [(first, first + len(sizes))]
    runs = [[0, lasting[0][1]]]
    for (prev_a, prev_b), (a, b) in zip(lasting, lasting[1:]):
        if prev_b < a:
            between = np.median(sizes[prev_b:a])
            if abs(between - np.median(sizes[prev_a:prev_b])) <= abs(between - np.median(sizes[a:b])):
                runs[-1][1] = a
            else:
                a = prev_b
        runs.append([a, b])
    runs[-1][1] = len(sizes)
    return [(first + a, first + b) for a, b in runs]


def restart_cameras(tracks: Sequence[Track], *, frame_w: int, frame_h: int) -> list[Track]:
    """Cut a track between two lasting size runs (``_size_runs``) when either run alone
    gets a sharper camera (``_gains``, judged on the two runs); the later part hands over
    from the earlier one like a separated continuation."""
    out = []
    for track in tracks:
        frames = filled_frames(track)
        runs = _size_runs(frames, frame_w, frame_h)
        start = runs[0][0]
        for (left_a, left_b), (right_a, right_b) in zip(runs, runs[1:]):
            left = {idx: frames[idx] for idx in range(left_a, left_b)}
            right = {idx: frames[idx] for idx in range(right_a, right_b)}
            if _gains(left | right, left, frame_w, frame_h) or _gains(left | right, right, frame_w, frame_h):
                out.append(Track(track.track_id, {idx: frames[idx] for idx in range(start, right_a)}))
                start = right_a
        out.append(Track(track.track_id, {idx: frames[idx] for idx in range(start, runs[-1][1])}))
    return out


def merge_continuations(
    tracks: Sequence[Track], cuts: set[int], *, frame_w: int, frame_h: int, max_gap: int
) -> list[Track]:
    """Fold a track whose first box lies >= ``CONTINUATION_COVER`` inside an earlier
    track's box from <= ``max_gap`` frames before (no cut between) into that track:
    a mosaic the detector split in two, or a link lost for a few frames. A continuation
    whose own camera would be much sharper stays a track (``worth_separating``)."""
    sorted_cuts = sorted(cuts)
    merged: list[Track] = []
    continuable: list[Track] = []
    for track in sorted(tracks, key=lambda item: (item.first, item.track_id)):
        head_idx = track.first
        head = track.frames[head_idx]
        continuable = [candidate for candidate in continuable if head_idx - candidate.last - 1 <= max_gap]
        parent, best = None, CONTINUATION_COVER
        for candidate in continuable:
            last = next((idx for idx in range(head_idx - 1, head_idx - 2 - max_gap, -1) if idx in candidate.frames), None)
            if last is None or _cut_between(sorted_cuts, last, head_idx):
                continue
            cover = _cover(candidate.frames[last].box, head.box)
            if cover >= best:
                parent, best = candidate, cover
        frames = filled_frames(track)
        if parent is None or worth_separating(parent.frames, frames, frame_w, frame_h):
            merged.append(Track(track_id=track.track_id, frames=frames))
            continuable.append(merged[-1])
            continue
        parent.frames = _fold(parent.frames, frames)
    return merged


def _tracks_by_frame(tracks: Sequence[Track]) -> dict[int, list[Track]]:
    by_frame: dict[int, list[Track]] = {}
    for track in tracks:
        for frame_idx in track.frames:
            by_frame.setdefault(frame_idx, []).append(track)
    return by_frame


def _covered_at(tracks_by_frame: dict[int, list[Track]], track: Track, box: Box, frame_idx: int) -> Track | None:
    """Another track whose box at ``frame_idx`` holds ``box`` >= ``CONTINUATION_COVER``."""
    for other in tracks_by_frame.get(frame_idx, ()):
        if other is not track and _cover(other.frames[frame_idx].box, box) >= CONTINUATION_COVER:
            return other
    return None


def rejoin_across_cover(tracks: Sequence[Track], cuts: set[int]) -> list[Track]:
    """Fold a track into an earlier one that ended <= ``LEAD_IN_FRAMES`` before it on the
    same spot (cover >= ``CONTINUATION_COVER``, no cut) while other tracks covered that
    spot on every frame between: one mosaic briefly detected as part of a bigger one,
    which would otherwise hand over twice."""
    sorted_cuts = sorted(cuts)
    ordered = sorted(tracks, key=lambda item: (item.first, item.track_id))
    ordered_by_frame = _tracks_by_frame(ordered)
    kept: list[Track] = []
    recent: list[Track] = []
    for track in ordered:
        head = track.frames[track.first].box
        recent = [other for other in recent if track.first - other.last <= LEAD_IN_FRAMES + 1]
        earlier = next(
            (
                other
                for other in recent
                if 1 < track.first - other.last
                and not _cut_between(sorted_cuts, other.last, track.first)
                and _cover(other.frames[other.last].box, head) >= CONTINUATION_COVER
                and all(
                    _covered_at(ordered_by_frame, other, other.frames[other.last].box, idx) is not None
                    for idx in range(other.last + 1, track.first)
                )
            ),
            None,
        )
        if earlier is None:
            kept.append(Track(track_id=track.track_id, frames=dict(track.frames)))
            recent.append(kept[-1])
        else:
            earlier.frames = filled_frames(Track(earlier.track_id, _fold(earlier.frames, track.frames)))
    return kept


def handover_leads(
    tracks: Sequence[Track], cuts: set[int], frame_count: int, max_gap: int
) -> list[tuple[int, int]]:
    """Per track the frames it renders before its first and after its last detection:
    a lead-in when it starts inside another track (a separated continuation), a lead-out
    when another track covers it right after it ends. Leads stop at cuts and at the
    covering track's ends."""
    sorted_cuts = sorted(cuts)
    by_frame = _tracks_by_frame(tracks)
    leads = []
    for track in tracks:
        first, last = track.first, track.last
        lead_in = lead_out = 0
        for idx in range(first - 1, max(first - 2 - max_gap, -1), -1):
            if idx + 1 in cuts:
                break
            parent = _covered_at(by_frame, track, track.frames[first].box, idx)
            if parent is not None:
                since_cut = _last_cut_at_or_before(sorted_cuts, first)
                lead_in = min(LEAD_IN_FRAMES, first - max(parent.first, since_cut))
                break
        for idx in range(last + 1, min(last + 2 + max_gap, frame_count)):
            if idx in cuts:
                break
            heir = _covered_at(by_frame, track, track.frames[last].box, idx)
            if heir is not None:
                next_cut = _first_cut_after(sorted_cuts, last, frame_count)
                lead_out = min(BLEND_FRAMES, heir.last - last, next_cut - 1 - last)
                break
        leads.append((lead_in, lead_out))
    return leads


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def handover_presence(frames: int, lead_in: int, lead_out: int) -> tuple[float, ...]:
    """1 on the track's own frames; over a lead-in 0, then a smoothstep up across the
    last ``BLEND_FRAMES``; over a lead-out a smoothstep down."""
    blend_in = min(BLEND_FRAMES, lead_in)
    ramp_in = [_smoothstep((k + 1) / (blend_in + 1)) for k in range(blend_in)]
    ramp_out = [1.0 - _smoothstep((k + 1) / (lead_out + 1)) for k in range(lead_out)]
    return tuple([0.0] * (lead_in - blend_in) + ramp_in + [1.0] * frames + ramp_out)


def _affine_polygons(polygons: Sequence[Polygon], source: Box, target: Box) -> tuple[Polygon, ...]:
    scale_x = (target[2] - target[0]) / max(source[2] - source[0], 1e-6)
    scale_y = (target[3] - target[1]) / max(source[3] - source[1], 1e-6)
    scale = np.array((scale_x, scale_y))
    return tuple(np.array(target[:2]) + (polygon - np.array(source[:2])) * scale for polygon in polygons)


def filled_frames(track: Track) -> dict[int, RegionGroup]:
    """Every frame of the track span; a detection gap gets the box interpolated between
    its neighbours and the nearer one's polygons mapped onto the interpolated polygon
    bounds (model boxes may reach past the frame, the polygons never do)."""
    anchors = sorted(track.frames)
    out = {}
    for frame_idx in range(anchors[0], anchors[-1] + 1):
        group = track.frames.get(frame_idx)
        if group is not None:
            out[frame_idx] = group
            continue
        position = bisect.bisect_left(anchors, frame_idx)
        left, right = track.frames[anchors[position - 1]], track.frames[anchors[position]]
        alpha = (frame_idx - anchors[position - 1]) / (anchors[position] - anchors[position - 1])
        box = tuple(a + (b - a) * alpha for a, b in zip(left.box, right.box))
        extent = tuple(a + (b - a) * alpha for a, b in zip(left.extent, right.extent))
        near = left if alpha <= 0.5 else right
        out[frame_idx] = RegionGroup(box, _affine_polygons(near.polygons, near.extent, extent))
    return out


def support_box(group: RegionGroup, frame_w: int, frame_h: int) -> Box:
    """The group's polygon bounds clamped to the frame, at least a pixel wide."""
    x1, y1, x2, y2 = group.extent
    return (
        max(0.0, min(x1, frame_w - 1.0)),
        max(0.0, min(y1, frame_h - 1.0)),
        min(float(frame_w), max(x2, x1 + 1.0)),
        min(float(frame_h), max(y2, y1 + 1.0)),
    )


def window_hop(overlap_frames: int) -> int:
    return WINDOW_FRAMES - overlap_frames


def window_starts(frames: int, window_hop: int) -> list[int]:
    starts = [0]
    while starts[-1] + WINDOW_FRAMES < frames:
        starts.append(starts[-1] + window_hop)
    return starts


def track_canvas(camera: Sequence[CropFrame], *, large_canvas: bool) -> int:
    """768 for a track whose largest crop is over 640 px (when allowed), else 512."""
    if large_canvas and max(crop.side for crop in camera) > LARGE_CANVAS_MIN_SIDE:
        return LARGE_CANVAS
    return PLAN_CANVAS


def plan_track(
    track: Track,
    *,
    lead: tuple[int, int],
    first_index: int,
    frame_w: int,
    frame_h: int,
    large_canvas: bool,
    window_hop: int,
) -> TrackPlan:
    lead_in, lead_out = lead
    own = list(filled_frames(track).values())
    groups = [own[0]] * lead_in + own + [own[-1]] * lead_out
    polygons = [group.polygons for group in groups]
    start = track.first - lead_in
    boxes = [support_box(group, frame_w, frame_h) for group in groups]
    camera = track_camera(boxes, frame_w=frame_w, frame_h=frame_h, feather=feather_pixels(frame_h))
    canvas = track_canvas(camera, large_canvas=large_canvas)
    if canvas != PLAN_CANVAS:
        camera = [letterbox(crop.crop_box, canvas) for crop in camera]
    windows = tuple(
        Window(
            index=first_index + offset,
            track_id=track.track_id,
            start=start + local,
            real_frames=min(WINDOW_FRAMES, len(polygons) - local),
            crops=tuple(camera[local : local + WINDOW_FRAMES]),
            noise=first_index + offset,
        )
        for offset, local in enumerate(window_starts(len(polygons), window_hop))
    )
    return TrackPlan(
        track_id=track.track_id,
        start=start,
        polygons=tuple(polygons),
        presence=handover_presence(len(own), lead_in, lead_out),
        windows=windows,
        window_hop=window_hop,
    )


def _plan_eye(
    regions_per_frame: Sequence[Sequence[Region]],
    cuts: set[int],
    *,
    first_track: int,
    first_index: int,
    frame_w: int,
    frame_h: int,
    large_canvas: bool,
    window_hop: int,
    max_gap: int,
    min_duration: int,
) -> list[TrackPlan]:
    groups = [group_regions(regions) for regions in regions_per_frame]
    tracks = rejoin_across_cover(
        merge_continuations(
            build_tracks(groups, cuts, max_gap), cuts, frame_w=frame_w, frame_h=frame_h, max_gap=max_gap
        ),
        cuts,
    )
    tracks = [track for track in tracks if not shorter_than_min_duration(track.last - track.first + 1, min_duration)]
    tracks = [
        Track(first_track + track_id, track.frames)
        for track_id, track in enumerate(restart_cameras(tracks, frame_w=frame_w, frame_h=frame_h))
    ]
    plans = []
    for track, lead in zip(tracks, handover_leads(tracks, cuts, len(regions_per_frame), max_gap)):
        plan = plan_track(
            track,
            lead=lead,
            first_index=first_index + sum(len(p.windows) for p in plans),
            frame_w=frame_w,
            frame_h=frame_h,
            large_canvas=large_canvas,
            window_hop=window_hop,
        )
        plans.append(plan)
    return plans


def _shift_region(region: Region, dx: int, dy: int) -> Region:
    x1, y1, x2, y2 = region.box
    return Region((x1 + dx, y1 + dy, x2 + dx, y2 + dy), region.polygon + np.array((dx, dy), dtype=region.polygon.dtype))


def _shift_crop(crop: CropFrame, dx: int, dy: int) -> CropFrame:
    x1, y1, x2, y2 = crop.crop_box
    return replace(crop, crop_box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy))


def _shift_plan(plan: TrackPlan, dx: int, dy: int) -> TrackPlan:
    return replace(
        plan,
        polygons=tuple(
            tuple(polygon + np.array((dx, dy), dtype=polygon.dtype) for polygon in frame) for frame in plan.polygons
        ),
        windows=tuple(replace(w, crops=tuple(_shift_crop(c, dx, dy) for c in w.crops)) for w in plan.windows),
    )


def _window_count(plans: Sequence[TrackPlan]) -> int:
    return sum(len(plan.windows) for plan in plans)


def plan_video(
    regions_per_frame: Sequence[Sequence[Region]],
    cuts: set[int],
    *,
    frame_w: int,
    frame_h: int,
    stereo: StereoLayout | None,
    large_canvas: bool,
    window_hop: int,
    max_gap: int,
    min_duration: int,
) -> list[TrackPlan]:
    """Every mosaic track of a video with its windows, in window-index order.
    Tracks bridge detection gaps up to ``max_gap`` frames (``track_max_gap``); a track
    spanning fewer than ``min_duration`` frames is dropped before size-run cuts, so a short
    piece of a longer track is never dropped. Stereo VR (``stereo``) plans the left eye on
    its own, gives each left track a right-eye twin (``right_eye_twins``) and plans the
    right-eye mosaics no twin covers on their own; no track or crop crosses into the other eye."""
    settings = dict(large_canvas=large_canvas, window_hop=window_hop, max_gap=max_gap, min_duration=min_duration)
    if stereo is None:
        return _plan_eye(regions_per_frame, cuts, first_track=0, first_index=0, frame_w=frame_w, frame_h=frame_h, **settings)
    dx, dy = stereo.right_offset
    eye = dict(frame_w=stereo.eye_w, frame_h=stereo.eye_h, **settings)
    left_regions = [[r for r in regions if stereo.eye_of(r.box)[:2] == (0, 0)] for regions in regions_per_frame]
    right_regions = [
        [_shift_region(r, -dx, -dy) for r in regions if stereo.eye_of(r.box)[:2] == (dx, dy)]
        for regions in regions_per_frame
    ]
    left = _plan_eye(left_regions, cuts, first_track=0, first_index=0, **eye)
    twins, unpaired = right_eye_twins(
        left, right_regions, eye_width=stereo.eye_w, first_track=len(left), first_index=_window_count(left)
    )
    right = twins + _plan_eye(
        unpaired, cuts, first_track=len(left) + len(twins), first_index=_window_count(left + twins), **eye
    )
    return left + [_shift_plan(plan, dx, dy) for plan in right]


def _extent(polygons: Sequence[Polygon]) -> Box:
    """Bounds of the polygon points."""
    if not polygons:
        raise ValueError("a tracked frame carries no polygon")
    points = np.concatenate(polygons)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


STEREO_MAX_LEFT_SHIFT = 0.12
"""Eye-width fraction a mosaic may sit further left in the right eye (a near subject)."""
STEREO_MAX_RIGHT_SHIFT = 0.02
STEREO_MAX_ROW_SHIFT = 0.25
"""Fraction of the mosaic height the two eyes' mosaic centres may differ by vertically."""
STEREO_MIN_MATCHED = 0.5


def _right_eye_match(left: Box, regions: Sequence[Region], eye_width: int) -> tuple[int, float] | None:
    """The right-eye region showing the same mosaic as the left box (about the same row and
    size, a plausible disparity), nearest first, with its horizontal disparity."""
    lx1, ly1, lx2, ly2 = left
    best = None
    for i, region in enumerate(regions):
        rx1, ry1, rx2, ry2 = _extent((region.polygon,))
        dx = (rx1 + rx2 - lx1 - lx2) / 2.0
        dy = abs(ry1 + ry2 - ly1 - ly2) / 2.0
        size = max(rx2 - rx1, ry2 - ry1) / max(lx2 - lx1, ly2 - ly1, 1.0)
        if (
            -STEREO_MAX_LEFT_SHIFT * eye_width <= dx <= STEREO_MAX_RIGHT_SHIFT * eye_width
            and dy <= STEREO_MAX_ROW_SHIFT * (ly2 - ly1)
            and 0.5 <= size <= 2.0
            and (best is None or abs(dx) + dy < best[2])
        ):
            best = (i, dx, abs(dx) + dy)
    return None if best is None else best[:2]


def _twin_crop(crop: CropFrame, shift: int, eye_width: int) -> CropFrame:
    x1, _, x2, _ = crop.crop_box
    return _shift_crop(crop, min(max(shift, -x1), eye_width - x2), 0)


def _outline_covers(polygons: Sequence[Polygon], polygon: Polygon) -> bool:
    """Whether ``CONTINUATION_COVER`` of ``polygon`` lies inside ``polygons``."""
    outline, box = _extent(polygons), _extent((polygon,))
    if _cover(outline, box) < CONTINUATION_COVER:
        return False
    # The box test alone marks a region beside a concave outline as covered.
    raster = _Raster((outline, box))
    return _covered(raster.mask(polygons), raster.mask((polygon,))) >= CONTINUATION_COVER


def right_eye_twins(
    left_plans: Sequence[TrackPlan],
    right_regions: Sequence[Sequence[Region]],
    *,
    eye_width: int,
    first_track: int,
    first_index: int,
) -> tuple[list[TrackPlan], list[list[Region]]]:
    """Each left-eye track the right eye sees on at least ``STEREO_MIN_MATCHED`` of its frames,
    copied to the right eye (eye coordinates) and shifted by its smoothed disparity. A twin's
    windows keep the left windows' ``noise``, so both eyes restore the same detail. Also
    returns the right-eye regions no twin matched or covers by ``CONTINUATION_COVER``; a twin also
    paints the right-eye regions it covers."""
    used: list[set[int]] = [set() for _ in right_regions]
    twins: list[TrackPlan] = []
    index = first_index
    for plan in left_plans:
        frames = len(plan.polygons)
        disparity = np.full(frames, np.nan)
        matched = []
        for k, polygons in enumerate(plan.polygons):
            t = plan.start + k
            match = _right_eye_match(_extent(polygons), right_regions[t], eye_width) if 0 <= t < len(right_regions) else None
            if match is not None:
                matched.append((t, match[0]))
                disparity[k] = match[1]
        if len(matched) < STEREO_MIN_MATCHED * frames:
            continue
        known = np.flatnonzero(~np.isnan(disparity))
        shift = np.round(
            gaussian_smooth(np.interp(np.arange(frames), known, disparity[known]), SMOOTH_SIGMA_FRAMES)
        ).astype(int)
        track_id = first_track + len(twins)
        windows = []
        for window in plan.windows:
            offset = window.start - plan.start
            crops = tuple(_twin_crop(crop, int(shift[offset + j]), eye_width) for j, crop in enumerate(window.crops))
            windows.append(replace(window, index=index, track_id=track_id, crops=crops))
            index += 1
        polygons = []
        for k, frame in enumerate(plan.polygons):
            t = plan.start + k
            outline = tuple(polygon + np.array((shift[k], 0), dtype=polygon.dtype) for polygon in frame)
            # A rigid shift misses where the eyes see the mosaic's shape differently (depth across a close-up),
            # so the twin also paints the right-eye regions it covers; those never become a second track.
            covered = (
                [i for i, r in enumerate(right_regions[t]) if _outline_covers(outline, r.polygon)]
                if plan.presence[k] > 0 and 0 <= t < len(right_regions)
                else []
            )
            polygons.append(outline + tuple(right_regions[t][i].polygon for i in covered if i not in used[t]))
            used[t].update(covered)
        for t, i in matched:
            used[t].add(i)
        twins.append(replace(plan, track_id=track_id, polygons=tuple(polygons), windows=tuple(windows)))
    # A right-eye detection merging two twins' mosaics is covered by their outlines together.
    for t, regions in enumerate(right_regions):
        outlines = [
            polygon for twin in twins if twin.start <= t < twin.stop and twin.presence[t - twin.start] > 0
            for polygon in twin.polygons[t - twin.start]
        ]
        used[t].update(i for i, r in enumerate(regions) if i not in used[t] and outlines and _outline_covers(outlines, r.polygon))
    unpaired = [[r for i, r in enumerate(regions) if i not in used[t]] for t, regions in enumerate(right_regions)]
    return twins, unpaired


UNFUSED_LEAD_FRAMES = 1 + TEMPORAL_SCALE
"""A later window's free latent 0 sits in front of latents fused with the earlier window, and
the decoder garbles the colour of the frames of that latent and the next one."""


def crossfade_weights(windows: Sequence[Window]) -> list[np.ndarray]:
    """Per-frame compose weights of one track's windows: across each overlap the earlier
    window alone over the later one's ``UNFUSED_LEAD_FRAMES``, then a linear ramp."""
    weights = [np.ones(window.real_frames, dtype=np.float32) for window in windows]
    for index, (left, right) in enumerate(zip(windows, windows[1:])):
        overlap = left.stop - right.start
        ramp = np.zeros(overlap, dtype=np.float32)
        ramp[UNFUSED_LEAD_FRAMES - 1 :] = np.linspace(0.0, 1.0, overlap - UNFUSED_LEAD_FRAMES + 1, dtype=np.float32)
        weights[index][len(weights[index]) - overlap :] = 1.0 - ramp
        weights[index + 1][:overlap] = ramp
    return weights


LATENT_FRAMES = 1 + (WINDOW_FRAMES - 1) // TEMPORAL_SCALE


@dataclass(frozen=True)
class SharedLatents:
    """Window k's latents ``left`` encode the same pixel frames as window k+1's latents
    ``right`` (12..15 and 1..4 at hop 88); ``ramp`` is the later window's fusion weight on each."""

    left: range
    right: range
    ramp: tuple[float, ...]


def shared_latents(window_hop: int) -> SharedLatents:
    shift = window_hop // TEMPORAL_SCALE
    left = range(shift + 1, LATENT_FRAMES)
    ramp = tuple((i + 1) / (len(left) + 1) for i in range(len(left)))
    return SharedLatents(left, range(1, LATENT_FRAMES - shift), ramp)

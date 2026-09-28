"""Mosaic tracks and their 121-frame model windows for LTX restoration.

Detections are grouped per frame, linked into cut-safe tracks, continuations are folded
into their track, and each track gets one zooming camera sliced into windows at hop 88.
Windows of a track are denoised together and share their overlapping latent frames.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES, Box, CropFrame, letterbox, track_camera
from jasna.tracking.clip_tracker import compute_iou_matrix

Polygon = list[list[float]]

WINDOW_HOP = 88
TEMPORAL_SCALE = 8
GROUP_IOU = 0.05
LINK_IOU = 0.1
MAX_GAP = 3
CONTINUATION_COVER = 0.5
LARGE_CANVAS = 768
LARGE_CANVAS_MIN_SIDE = 640
FEATHER_FRACTION = 0.028


@dataclass(frozen=True)
class Region:
    """One detected mosaic: model box and the traced outline of its dilated mask."""

    box: Box
    polygon: Polygon


@dataclass(frozen=True)
class RegionGroup:
    box: Box
    polygons: tuple[Polygon, ...]


@dataclass
class Track:
    track_id: int
    frames: dict[int, RegionGroup] = field(default_factory=dict)

    @property
    def first(self) -> int:
        return min(self.frames)

    @property
    def last(self) -> int:
        return max(self.frames)


@dataclass(frozen=True)
class Window:
    index: int  # global across tracks; also the noise seed offset
    track_id: int
    start: int
    real_frames: int  # the model always sees WINDOW_FRAMES; the tail is padded
    crops: tuple[CropFrame, ...]  # one per real frame

    @property
    def stop(self) -> int:
        return self.start + self.real_frames

    @property
    def canvas(self) -> int:
        return self.crops[0].canvas

    def model_crops(self) -> list[CropFrame]:
        return list(self.crops) + [self.crops[-1]] * (WINDOW_FRAMES - self.real_frames)


@dataclass(frozen=True)
class TrackPlan:
    track_id: int
    start: int
    polygons: tuple[tuple[Polygon, ...], ...]  # per frame from ``start``
    windows: tuple[Window, ...]

    @property
    def stop(self) -> int:
        return self.start + len(self.polygons)


def feather_pixels(frame_h: int) -> int:
    return max(1, int(round(FEATHER_FRACTION * frame_h)))


def _union_box(boxes: Sequence[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


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


def build_tracks(groups_per_frame: Sequence[Sequence[RegionGroup]], cuts: set[int]) -> list[Track]:
    """Greedy best-IoU linking; a track tolerates ``MAX_GAP`` missing frames and never
    crosses a scene cut (``cuts`` holds first-frame-after-cut indices)."""
    tracks: list[Track] = []
    active: list[Track] = []
    for frame_idx, groups in enumerate(groups_per_frame):
        active = [
            track
            for track in active
            if frame_idx - track.last - 1 <= MAX_GAP
            and not any(track.last < cut <= frame_idx for cut in cuts)
        ]
        candidates: list[tuple[float, int, int]] = []
        if active and groups:
            iou = compute_iou_matrix(
                np.asarray([track.frames[track.last].box for track in active], dtype=np.float64),
                np.asarray([group.box for group in groups], dtype=np.float64),
            )
            candidates = [
                (float(iou[ti, gi]), ti, gi)
                for ti in range(len(active))
                for gi in range(len(groups))
                if iou[ti, gi] >= LINK_IOU
            ]
            candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_groups: set[int] = set()
        for _, ti, gi in candidates:
            if ti in used_tracks or gi in used_groups:
                continue
            active[ti].frames[frame_idx] = groups[gi]
            used_tracks.add(ti)
            used_groups.add(gi)
        for gi, group in enumerate(groups):
            if gi not in used_groups:
                track = Track(track_id=len(tracks), frames={frame_idx: group})
                tracks.append(track)
                active.append(track)
    return tracks


def _cover(outer: Box, inner: Box) -> float:
    """Intersection of two boxes over the area of ``inner``."""
    width = min(outer[2], inner[2]) - max(outer[0], inner[0])
    height = min(outer[3], inner[3]) - max(outer[1], inner[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0.0, width) * max(0.0, height) / area if area > 0 else 0.0


def merge_continuations(tracks: Sequence[Track], cuts: set[int]) -> list[Track]:
    """Fold a track whose first box lies >= ``CONTINUATION_COVER`` inside an earlier
    track's box from <= ``MAX_GAP`` frames before (no cut between) into that track:
    a mosaic the detector split in two, or a link lost for a few frames."""
    merged: list[Track] = []
    for track in sorted(tracks, key=lambda item: (item.first, item.track_id)):
        head_idx = track.first
        head = track.frames[head_idx]
        parent, best = None, CONTINUATION_COVER
        for candidate in merged:
            before = [idx for idx in candidate.frames if idx < head_idx]
            if not before:
                continue
            last = max(before)
            if head_idx - last - 1 > MAX_GAP or any(last < cut <= head_idx for cut in cuts):
                continue
            cover = _cover(candidate.frames[last].box, head.box)
            if cover >= best:
                parent, best = candidate, cover
        if parent is None:
            merged.append(Track(track_id=track.track_id, frames=dict(track.frames)))
            continue
        for idx, group in track.frames.items():
            other = parent.frames.get(idx)
            parent.frames[idx] = (
                group
                if other is None
                else RegionGroup(_union_box([other.box, group.box]), other.polygons + group.polygons)
            )
        parent.frames = dict(sorted(parent.frames.items()))
    return merged


def _affine_polygons(polygons: Sequence[Polygon], source: Box, target: Box) -> tuple[Polygon, ...]:
    scale_x = (target[2] - target[0]) / max(source[2] - source[0], 1e-6)
    scale_y = (target[3] - target[1]) / max(source[3] - source[1], 1e-6)
    return tuple(
        [[target[0] + (x - source[0]) * scale_x, target[1] + (y - source[1]) * scale_y] for x, y in polygon]
        for polygon in polygons
    )


def track_polygons(track: Track) -> list[tuple[Polygon, ...]]:
    """Per-frame polygons over the whole track span; a detection gap gets the box
    interpolated between its neighbours and the nearer one's polygons mapped onto it."""
    anchors = sorted(track.frames)
    out = []
    for frame_idx in range(anchors[0], anchors[-1] + 1):
        group = track.frames.get(frame_idx)
        if group is not None:
            out.append(group.polygons)
            continue
        position = bisect.bisect_left(anchors, frame_idx)
        left, right = track.frames[anchors[position - 1]], track.frames[anchors[position]]
        alpha = (frame_idx - anchors[position - 1]) / (anchors[position] - anchors[position - 1])
        box = tuple(a + (b - a) * alpha for a, b in zip(left.box, right.box))
        near = left if alpha <= 0.5 else right
        out.append(_affine_polygons(near.polygons, near.box, box))
    return out


def polygons_box(polygons: Sequence[Polygon], frame_w: int, frame_h: int) -> Box:
    points = np.asarray([point for polygon in polygons for point in polygon], dtype=np.float64)
    if not len(points):
        raise ValueError("a tracked frame carries no polygon")
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return (
        float(max(0.0, min(x1, frame_w - 1.0))),
        float(max(0.0, min(y1, frame_h - 1.0))),
        float(min(float(frame_w), max(x2, x1 + 1.0))),
        float(min(float(frame_h), max(y2, y1 + 1.0))),
    )


def window_starts(frames: int) -> list[int]:
    starts = [0]
    while starts[-1] + WINDOW_FRAMES < frames:
        starts.append(starts[-1] + WINDOW_HOP)
    return starts


def track_canvas(camera: Sequence[CropFrame], *, large_canvas: bool) -> int:
    """768 for a track whose largest crop is over 640 px (when allowed), else 512."""
    if large_canvas and max(crop.side for crop in camera) > LARGE_CANVAS_MIN_SIDE:
        return LARGE_CANVAS
    return PLAN_CANVAS


def plan_track(track: Track, *, first_index: int, frame_w: int, frame_h: int, large_canvas: bool) -> TrackPlan:
    polygons = track_polygons(track)
    boxes = [polygons_box(item, frame_w, frame_h) for item in polygons]
    camera = track_camera(boxes, frame_w=frame_w, frame_h=frame_h, feather=feather_pixels(frame_h))
    canvas = track_canvas(camera, large_canvas=large_canvas)
    if canvas != PLAN_CANVAS:
        camera = [letterbox(crop.crop_box, canvas) for crop in camera]
    windows = tuple(
        Window(
            index=first_index + offset,
            track_id=track.track_id,
            start=track.first + local,
            real_frames=min(WINDOW_FRAMES, len(polygons) - local),
            crops=tuple(camera[local : local + WINDOW_FRAMES]),
        )
        for offset, local in enumerate(window_starts(len(polygons)))
    )
    return TrackPlan(track_id=track.track_id, start=track.first, polygons=tuple(polygons), windows=windows)


def plan_video(
    regions_per_frame: Sequence[Sequence[Region]],
    cuts: set[int],
    *,
    frame_w: int,
    frame_h: int,
    large_canvas: bool,
) -> list[TrackPlan]:
    """Every mosaic track of a video with its windows, in window-index order."""
    groups = [group_regions(regions) for regions in regions_per_frame]
    tracks = merge_continuations(build_tracks(groups, cuts), cuts)
    plans = []
    for track in tracks:
        plan = plan_track(
            track,
            first_index=sum(len(p.windows) for p in plans),
            frame_w=frame_w,
            frame_h=frame_h,
            large_canvas=large_canvas,
        )
        plans.append(plan)
    return plans


def crossfade_weights(windows: Sequence[Window]) -> list[np.ndarray]:
    """Per-frame compose weights of one track's windows: linear ramps across each overlap."""
    weights = [np.ones(window.real_frames, dtype=np.float32) for window in windows]
    for index, (left, right) in enumerate(zip(windows, windows[1:])):
        overlap = left.stop - right.start
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        weights[index][len(weights[index]) - overlap :] = 1.0 - ramp
        weights[index + 1][:overlap] = ramp
    return weights


LATENT_FRAMES = 1 + (WINDOW_FRAMES - 1) // TEMPORAL_SCALE
SHARED_LATENT_SHIFT = WINDOW_HOP // TEMPORAL_SCALE
# Window k's latents 12..15 encode the same pixel frames as window k+1's latents 1..4.
LEFT_SHARED_LATENTS = range(SHARED_LATENT_SHIFT + 1, LATENT_FRAMES)
RIGHT_SHARED_LATENTS = range(1, LATENT_FRAMES - SHARED_LATENT_SHIFT)
FUSION_RAMP = tuple((i + 1) / (len(LEFT_SHARED_LATENTS) + 1) for i in range(len(LEFT_SHARED_LATENTS)))

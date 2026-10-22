"""Mosaic tracks and their 121-frame model windows for LTX restoration.

Detections are grouped per frame, linked into cut-safe tracks, continuations are folded
into their track, and each track gets one zooming camera sliced into windows at hop 88
(104 with less overlap). Windows of a track are denoised together and share their
overlapping latent frames.

A continuation stays its own track when folding it would cost its camera
``SEPARATE_GAIN`` sharpness for at least a window (``worth_separating``), and a track is
cut where its size changes for good (``restart_cameras``). A track starting inside
another hands over from it: it starts ``LEAD_IN_FRAMES`` early on its first outline and
takes the area over during the last ``BLEND_FRAMES`` of them; a track ending inside
another hands the area back over ``BLEND_FRAMES``.
"""

from __future__ import annotations

import bisect
import functools
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES, Box, CropFrame, lada_crop_boxes, letterbox, track_camera
from jasna.tracking.clip_tracker import compute_iou_matrix

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
        if not self.polygons:
            raise ValueError("a tracked frame carries no polygon")
        points = np.concatenate(self.polygons)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)


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
            [(t, w.index, w.track_id, w.start, w.real_frames) for t, w in windows], dtype=np.int64
        ).reshape(-1, 5),
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
    for t, index, track_id, start, real_frames in arrays["windows"].tolist():
        windows.setdefault(t, []).append(
            Window(index, track_id, start, real_frames, tuple(itertools.islice(crops, real_frames)))
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


def build_tracks(groups_per_frame: Sequence[Sequence[RegionGroup]], cuts: set[int]) -> list[Track]:
    """Greedy best-IoU linking; a track tolerates ``MAX_GAP`` missing frames and never
    crosses a scene cut (``cuts`` holds first-frame-after-cut indices)."""
    sorted_cuts = sorted(cuts)
    tracks: list[Track] = []
    active: list[Track] = []
    for frame_idx, groups in enumerate(groups_per_frame):
        active = [
            track
            for track in active
            if frame_idx - track.last - 1 <= MAX_GAP and not _cut_between(sorted_cuts, track.last, frame_idx)
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


def merge_continuations(tracks: Sequence[Track], cuts: set[int], *, frame_w: int, frame_h: int) -> list[Track]:
    """Fold a track whose first box lies >= ``CONTINUATION_COVER`` inside an earlier
    track's box from <= ``MAX_GAP`` frames before (no cut between) into that track:
    a mosaic the detector split in two, or a link lost for a few frames. A continuation
    whose own camera would be much sharper stays a track (``worth_separating``)."""
    sorted_cuts = sorted(cuts)
    merged: list[Track] = []
    continuable: list[Track] = []
    for track in sorted(tracks, key=lambda item: (item.first, item.track_id)):
        head_idx = track.first
        head = track.frames[head_idx]
        continuable = [candidate for candidate in continuable if head_idx - candidate.last - 1 <= MAX_GAP]
        parent, best = None, CONTINUATION_COVER
        for candidate in continuable:
            last = next((idx for idx in range(head_idx - 1, head_idx - 2 - MAX_GAP, -1) if idx in candidate.frames), None)
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


def handover_leads(tracks: Sequence[Track], cuts: set[int], frame_count: int) -> list[tuple[int, int]]:
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
        for idx in range(first - 1, max(first - 2 - MAX_GAP, -1), -1):
            if idx + 1 in cuts:
                break
            parent = _covered_at(by_frame, track, track.frames[first].box, idx)
            if parent is not None:
                since_cut = _last_cut_at_or_before(sorted_cuts, first)
                lead_in = min(LEAD_IN_FRAMES, first - max(parent.first, since_cut))
                break
        for idx in range(last + 1, min(last + 2 + MAX_GAP, frame_count)):
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


def window_hop(less_overlap: bool) -> int:
    return LESS_OVERLAP_WINDOW_HOP if less_overlap else WINDOW_HOP


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


def plan_video(
    regions_per_frame: Sequence[Sequence[Region]],
    cuts: set[int],
    *,
    frame_w: int,
    frame_h: int,
    large_canvas: bool,
    window_hop: int,
) -> list[TrackPlan]:
    """Every mosaic track of a video with its windows, in window-index order."""
    groups = [group_regions(regions) for regions in regions_per_frame]
    tracks = rejoin_across_cover(
        merge_continuations(build_tracks(groups, cuts), cuts, frame_w=frame_w, frame_h=frame_h), cuts
    )
    tracks = [
        Track(track_id, track.frames)
        for track_id, track in enumerate(restart_cameras(tracks, frame_w=frame_w, frame_h=frame_h))
    ]
    plans = []
    for track, lead in zip(tracks, handover_leads(tracks, cuts, len(regions_per_frame))):
        plan = plan_track(
            track,
            lead=lead,
            first_index=sum(len(p.windows) for p in plans),
            frame_w=frame_w,
            frame_h=frame_h,
            large_canvas=large_canvas,
            window_hop=window_hop,
        )
        plans.append(plan)
    return plans


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

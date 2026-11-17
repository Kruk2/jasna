"""The LTX restoration plan of a video: its mosaic tracks (``tracks.py``), each with one zooming
camera sliced into 121-frame windows (``windows.py``) on a 512 px canvas, or 768 px for large
mosaics when allowed. Windows of a track are denoised together and share their overlapping
latent frames. Stereo VR (side by side or top/bottom) plans the left eye on its own and gives
the right eye twins of its tracks (``stereo_plan.py``).
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from jasna.ltx.camera import (
    PLAN_CANVAS,
    WINDOW_FRAMES,
    CropFrame,
    feather_pixels,
    letterbox,
    track_camera,
)
from jasna.ltx.stereo_plan import right_eye_twins, shifted_plan, shifted_region
from jasna.ltx.tracks import (
    BLEND_FRAMES,
    Polygon,
    Region,
    Track,
    build_tracks,
    filled_frames,
    group_regions,
    handover_leads,
    merge_continuations,
    rejoin_across_cover,
    restart_cameras,
    support_box,
)
from jasna.ltx.windows import Window, window_starts

# kept importable from this module for callers outside this package
from jasna.ltx.windows import window_hop as window_hop
from jasna.tracking.clip_tracker import shorter_than_min_duration
from jasna.vr180 import StereoLayout

LARGE_CANVAS = 768
LARGE_CANVAS_MIN_SIDE = 640


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


def plan_summary(frames: int, cuts: int, plans: Sequence[TrackPlan]) -> str:
    canvases = Counter(window.canvas for plan in plans for window in plan.windows)
    summary = f"LTX plan: {frames} frames, {cuts} scene cuts, {len(plans)} tracks, {canvases.total()} windows"
    if canvases:
        summary += ": " + ", ".join(f"{count} at {canvas}" for canvas, count in sorted(canvases.items()))
    return summary


def plan_window_lines(plans: Sequence[TrackPlan], cuts: Collection[int], frame_numbers: Sequence[int]) -> list[str]:
    """One line per window with its first and last frame in the whole input (``frame_numbers``
    of the planned frames), its track, canvas and the frames it shares with the track's next
    window, then the scene cuts."""
    windows = [(track, window, following)
               for track, plan in enumerate(plans, 1)
               for window, following in itertools.zip_longest(plan.windows, plan.windows[1:])]
    lines = []
    for number, (track, window, following) in enumerate(windows, 1):
        overlap = "" if following is None else f", overlap with next {window.stop - following.start}"
        lines.append(
            f"LTX window {number}/{len(windows)}: frames {frame_numbers[window.start]}-{frame_numbers[window.stop - 1]} "
            f"(track {track}, canvas {window.canvas}{overlap})"
        )
    if cuts:
        lines.append("LTX scene cuts: " + ", ".join(str(frame_numbers[cut]) for cut in sorted(cuts)))
    return lines


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def handover_presence(frames: int, lead_in: int, lead_out: int) -> tuple[float, ...]:
    """1 on the track's own frames; over a lead-in 0, then a smoothstep up across the
    last ``BLEND_FRAMES``; over a lead-out a smoothstep down."""
    blend_in = min(BLEND_FRAMES, lead_in)
    ramp_in = [_smoothstep((k + 1) / (blend_in + 1)) for k in range(blend_in)]
    ramp_out = [1.0 - _smoothstep((k + 1) / (lead_out + 1)) for k in range(lead_out)]
    return tuple([0.0] * (lead_in - blend_in) + ramp_in + [1.0] * frames + ramp_out)


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
        [shifted_region(r, -dx, -dy) for r in regions if stereo.eye_of(r.box)[:2] == (dx, dy)]
        for regions in regions_per_frame
    ]
    left = _plan_eye(left_regions, cuts, first_track=0, first_index=0, **eye)
    twins, unpaired = right_eye_twins(
        left, right_regions, eye_width=stereo.eye_w, first_track=len(left), first_index=_window_count(left)
    )
    right = twins + _plan_eye(
        unpaired, cuts, first_track=len(left) + len(twins), first_index=_window_count(left + twins), **eye
    )
    return left + [shifted_plan(plan, dx, dy) for plan in right]



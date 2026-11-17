"""Stereo VR (side by side or top/bottom): the right eye restores each left-eye track as a twin
shifted by its disparity and denoised with the same noise, so both eyes invent the same detail."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from jasna.ltx.camera import (
    SMOOTH_SIGMA_FRAMES,
    Box,
    CropFrame,
    gaussian_smooth,
)
from jasna.ltx.tracks import (
    CONTINUATION_COVER,
    OutlineRaster,
    Polygon,
    Region,
    box_cover,
    covered_share,
    outline_bounds,
)

if TYPE_CHECKING:
    from jasna.ltx.plan import TrackPlan


def shifted_region(region: Region, dx: int, dy: int) -> Region:
    x1, y1, x2, y2 = region.box
    return Region((x1 + dx, y1 + dy, x2 + dx, y2 + dy), region.polygon + np.array((dx, dy), dtype=region.polygon.dtype))


def _shift_crop(crop: CropFrame, dx: int, dy: int) -> CropFrame:
    x1, y1, x2, y2 = crop.crop_box
    return replace(crop, crop_box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy))


def shifted_plan(plan: TrackPlan, dx: int, dy: int) -> TrackPlan:
    return replace(
        plan,
        polygons=tuple(
            tuple(polygon + np.array((dx, dy), dtype=polygon.dtype) for polygon in frame) for frame in plan.polygons
        ),
        windows=tuple(replace(w, crops=tuple(_shift_crop(c, dx, dy) for c in w.crops)) for w in plan.windows),
    )


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
        rx1, ry1, rx2, ry2 = outline_bounds((region.polygon,))
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
    outline, box = outline_bounds(polygons), outline_bounds((polygon,))
    if box_cover(outline, box) < CONTINUATION_COVER:
        return False
    # The box test alone marks a region beside a concave outline as covered.
    raster = OutlineRaster((outline, box))
    return covered_share(raster.mask(polygons), raster.mask((polygon,))) >= CONTINUATION_COVER


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
            match = _right_eye_match(outline_bounds(polygons), right_regions[t], eye_width) if 0 <= t < len(right_regions) else None
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

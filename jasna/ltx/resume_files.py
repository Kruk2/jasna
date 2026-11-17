"""What an LTX run keeps on disk between passes and to resume: each span's track plans and each
window's latents."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from jasna.job_resume import write_json
from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import LARGE_CANVAS, TrackPlan
from jasna.ltx.spans import LtxSpan
from jasna.ltx.windows import Window
from jasna.os_utils import atomic_write

if TYPE_CHECKING:
    from jasna.ltx.restore import PreparedSpan

class LatentStore:
    """Window latents of one segment on disk, keyed by window index."""

    def __init__(self, directory: Path, kind: str) -> None:
        self._directory = directory
        self._kind = kind

    def _path(self, index: int) -> Path:
        return self._directory / f"{index:06d}.{self._kind}.pt"

    def put(self, index: int, latent: torch.Tensor) -> None:
        with atomic_write(self._path(index), "wb") as stream:
            torch.save(latent.to("cpu", torch.bfloat16).contiguous(), stream)

    def has(self, index: int) -> bool:
        return self._path(index).is_file()

    def has_track(self, plan: TrackPlan) -> bool:
        return self._path(plan.track_id).with_suffix(".done").is_file() and all(self.has(w.index) for w in plan.windows)

    def commit_track(self, plan: TrackPlan) -> None:
        write_json(self._path(plan.track_id).with_suffix(".done"), [w.index for w in plan.windows])

    def get(self, index: int) -> torch.Tensor:
        return torch.load(self._path(index), weights_only=True)

    def delete(self, index: int) -> None:
        self._path(index).unlink()


def pts_key(start_pts: int, end_pts: int) -> str:
    return f"{start_pts}-{end_pts}"


def span_key(span: LtxSpan) -> str:
    return pts_key(span.segments[0].start_pts, span.segments[-1].end_pts)


def plans_file(key: str, directory: Path) -> Path:
    return directory / f"{key}.plans.npz"


def plan_path(span: LtxSpan, directory: Path) -> Path:
    return plans_file(span_key(span), directory)


@dataclass(frozen=True)
class SavedPlans:
    frames: int
    plans: list[list[TrackPlan]]

    @property
    def large_canvas(self) -> bool:
        return any(w.canvas == LARGE_CANVAS for segment in self.plans for plan in segment for w in plan.windows)


def saved_plans(span: LtxSpan, directory: Path) -> SavedPlans | None:
    return read_plans(plan_path(span, directory))


def read_plans(path: Path) -> SavedPlans | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        return SavedPlans(int(arrays["frames"]), plans_from_arrays(arrays))


def saved_frames(span: LtxSpan, directory: Path) -> int:
    with np.load(plan_path(span, directory), allow_pickle=False) as arrays:
        return int(arrays["frames"])


def span_stores(span: LtxSpan, directory: Path, kind: str) -> tuple[LatentStore, ...]:
    return tuple(LatentStore(directory, f"{span_key(span)}-{g}.{kind}") for g in range(len(span.segments)))


def final_stores(prepared: Sequence[PreparedSpan], directory: Path) -> list[tuple[LatentStore, ...]]:
    return [span_stores(p.span, directory, "final") for p in prepared]


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

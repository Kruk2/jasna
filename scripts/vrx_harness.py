"""Phase-2 CPU harness primitives for the VR projection benchmark.

Deterministic, GPU-free building blocks used by the discovery/restore driver:
seek planning, track-stability scoring, angular position binning, sample-record
(de)serialization, and blind A/B/C labelling. Kept script-local until routing is
decided (per the execution brief). No torch/jasna imports here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np


def stable_seed(*parts) -> int:
    """Deterministic 63-bit seed from arbitrary parts (stable across runs)."""
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def seek_candidates(duration: float, seed: int, n: int = 12,
                    skip_head: float = 0.08, skip_tail: float = 0.08) -> list[float]:
    """Deterministic, sorted seek timestamps inside the useful duration
    (credits trimmed by skip_head/skip_tail)."""
    lo, hi = duration * skip_head, duration * (1.0 - skip_tail)
    if hi <= lo:
        return [duration * 0.5]
    rng = np.random.default_rng(seed)
    return sorted(float(t) for t in rng.uniform(lo, hi, size=n))


def angular_center(u: float, v: float) -> tuple[float, float]:
    """Half-equirect normalized eye coord -> (lon_deg, lat_deg)."""
    return ((u - 0.5) * 180.0, (0.5 - v) * 180.0)


def position_bin(lat_deg: float, lon_deg: float,
                 center_band: float = 25.0, side_lon: float = 30.0) -> str:
    """Bin by angular position. 'center' = middle latitude band; 'bottom' = low
    latitude (real lower-body mosaics), flagged off-axis when |lon|>side_lon."""
    if lat_deg <= -center_band:
        return "bottom_offaxis" if abs(lon_deg) > side_lon else "bottom"
    if lat_deg >= center_band:
        return "top"
    return "center"


def track_stability(centers, areas, coverage: float) -> dict:
    """centers: (N,2) normalized eye coords; areas: (N,) normalized; coverage:
    fraction of the discovery window the track spans. Higher = stabler."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(areas, dtype=np.float64).reshape(-1)
    cen_jit = float(np.hypot(*c.std(axis=0))) if len(c) > 1 else 0.0
    area_jit = float(a.std() / max(a.mean(), 1e-6)) if len(a) > 1 else 0.0
    stable = bool(coverage >= 0.8 and cen_jit < 0.03 and area_jit < 0.5)
    score = float(coverage) - 4.0 * cen_jit - area_jit
    return dict(center_jitter=cen_jit, area_jitter=area_jit,
                coverage=float(coverage), stable=stable, score=score)


@dataclass
class SampleRecord:
    sample_id: str
    studio: str
    title: str
    source_path: str
    seek_ts: float
    pts_start: int
    pts_end: int
    width: int
    height: int
    fps: float
    eye: str
    track_frames: list = field(default_factory=list)
    bboxes: list = field(default_factory=list)          # full-frame xyxy per frame
    mask_ref: str = ""                                   # path to a .npy of low-res masks
    mask_shape: list = field(default_factory=list)
    center_lon: float = 0.0
    center_lat: float = 0.0
    position: str = ""
    stability: dict = field(default_factory=dict)
    vr_reason: str = ""
    zelefans_prior: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def write_jsonl(path, records) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def read_jsonl(path) -> list[SampleRecord]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(SampleRecord(**json.loads(line)))
    return out


def blind_labels(sample_id: str, variants: list[str]) -> dict[str, str]:
    """Deterministic per-sample shuffle: {label -> variant}. The inverse is the
    key saved alongside; never expose variant names in the rating video."""
    rng = np.random.default_rng(stable_seed(sample_id, "blind-v1"))
    order = list(variants)
    rng.shuffle(order)
    return {chr(ord("A") + i): v for i, v in enumerate(order)}

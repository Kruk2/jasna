"""LTX speed learned from finished runs, so a later run knows its time left once it is planned.

A profile (one JSON file) maps a key (Jasna version, GPU, transformer file) to rates in
seconds: per pass a fixed part (model load, first-call compile), per frame-megapixel
for the passes that decode the video, and per window of each canvas. Rates are warm: the
first batch of a pass and the first window of each canvas are left out, and denoise
leaves out each track's first step. Every finished run replaces the rates it measured.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from jasna import __version__
from jasna.engine_paths import model_weights_dir

TIMED_PASSES = ("encode", "denoise", "compose")


def profile_path() -> Path:
    return model_weights_dir() / "ltx-speed-profile.json"


def profile_key(gpu: str, transformer: str) -> str:
    return f"{__version__} | {gpu} | {transformer}"


def load_rates(path: Path, key: str) -> dict[str, float]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get(key, {})


def save_rates(path: Path, key: str, rates: dict[str, float]) -> None:
    """Merge ``rates`` into ``key``'s profile; written to a temporary file and renamed, so a
    reader never sees a partial file."""
    profiles = json.loads(path.read_text()) if path.is_file() else {}
    profiles[key] = {**profiles.get(key, {}), **rates}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(profiles, indent=1, sort_keys=True))
    os.replace(temp, path)


@dataclass(frozen=True)
class Work:
    """What a run restores, known once the scan has planned it."""

    frames: int
    megapixels: float
    windows: dict[int, int]

    def units(self, stage: str) -> dict[str, float]:
        units: dict[str, float] = {f"{stage}/window{canvas}": count for canvas, count in self.windows.items()}
        if stage != "denoise":
            units[f"{stage}/frame"] = self.frames * self.megapixels
        return units


@dataclass
class PassTiming:
    """One pass of a run: ``seconds`` since the previous pass ended (model load included);
    ``warm_seconds`` and ``warm_frames`` after its first batch; ``windows`` the
    ``(canvas, seconds)`` of each window in order (denoise: seconds per window of a track
    without its first step)."""

    seconds: float = 0.0
    warm_seconds: float = 0.0
    warm_frames: int = 0
    windows: list[tuple[int, float]] = field(default_factory=list)


def pass_estimates(rates: dict[str, float], work: Work) -> dict[str, tuple[float, float]] | None:
    """``(fixed, per-unit)`` seconds of each timed pass, or None when a rate is missing."""
    estimates = {}
    for stage in TIMED_PASSES:
        units = work.units(stage)
        if any(name not in rates for name in [*units, f"{stage}/fixed"]):
            return None
        estimates[stage] = (rates[f"{stage}/fixed"], sum(rates[name] * count for name, count in units.items()))
    return estimates


def learned_rates(timings: dict[str, PassTiming], work: Work, known: dict[str, float]) -> dict[str, float]:
    """The rates this run measured; a pass's fixed part needs its other rates, from this run
    or else from ``known``."""
    rates: dict[str, float] = {}
    for stage, timing in timings.items():
        by_canvas = defaultdict(list)
        for canvas, seconds in timing.windows:
            by_canvas[canvas].append(seconds)
        cold = 0 if stage == "denoise" else 1
        for canvas, samples in by_canvas.items():
            if len(samples) > cold:
                rates[f"{stage}/window{canvas}"] = mean(samples[cold:])
        if stage != "denoise" and timing.warm_frames:
            frame_seconds = max(0.0, timing.warm_seconds - sum(seconds for _, seconds in timing.windows))
            rates[f"{stage}/frame"] = frame_seconds / (timing.warm_frames * work.megapixels)
    for stage, timing in timings.items():
        merged = {**known, **rates}
        units = work.units(stage)
        if all(name in merged for name in units):
            variable = sum(merged[name] * count for name, count in units.items())
            rates[f"{stage}/fixed"] = max(0.0, timing.seconds - variable)
    return rates

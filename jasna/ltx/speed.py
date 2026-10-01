"""LTX speed learned from finished runs, so a later run knows its time left once it is planned.

A profile (one JSON file) maps a key (Jasna version, GPU, transformer file, decode
budget: whole videos decode within the free VRAM, segments within a fixed budget, so their
decode tiling differs) to rates in
seconds: per pass a fixed part (model load, first-call compile), per frame-megapixel
for the passes that decode the video, and per window of each canvas. Rates are warm: the
first batch of a pass and the first window of each canvas are left out, and denoise
leaves out each track's first step. Every finished run replaces the rates it measured.
The same pass timings make the summary logged at the end of every run (``run_summary``).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from jasna import __version__
from jasna.engine_paths import model_weights_dir

TIMED_PASSES = ("encode", "denoise", "compose")
SUMMARY_PASSES = ("scan", *TIMED_PASSES)


def profile_path() -> Path:
    return model_weights_dir() / "ltx-speed-profile.json"


def profile_key(gpu: str, transformer: str, decode_budget: str) -> str:
    return f"{__version__} | {gpu} | {transformer} | decode {decode_budget}"


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
    ``units`` it counted (frames, or window-steps for denoise); ``warm_seconds`` and
    ``warm_frames`` after its first batch; ``windows`` the ``(canvas, seconds)`` of each
    window in order (denoise: seconds per window of a track without its first step);
    ``steps`` (denoise) the ``(canvas, seconds per window-step)`` of every window group's
    step after a track's first."""

    seconds: float = 0.0
    units: int = 0
    warm_seconds: float = 0.0
    warm_frames: int = 0
    windows: list[tuple[int, float]] = field(default_factory=list)
    steps: list[tuple[int, float]] = field(default_factory=list)


def _by_canvas(samples: list[tuple[int, float]]) -> dict[int, list[float]]:
    by_canvas = defaultdict(list)
    for canvas, seconds in samples:
        by_canvas[canvas].append(seconds)
    return by_canvas


def warm_windows(stage: str, timing: PassTiming) -> dict[int, list[float]]:
    """Seconds per window of each canvas without its cold first window (denoise windows
    already leave out each track's first step)."""
    cold = 0 if stage == "denoise" else 1
    return {
        canvas: samples[cold:] for canvas, samples in _by_canvas(timing.windows).items() if len(samples) > cold
    }


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
    timings = {stage: timing for stage, timing in timings.items() if stage in TIMED_PASSES}
    rates: dict[str, float] = {}
    for stage, timing in timings.items():
        for canvas, samples in warm_windows(stage, timing).items():
            rates[f"{stage}/window{canvas}"] = mean(samples)
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


def _medians(samples: dict[int, list[float]], unit: str) -> list[str]:
    return [f"{median(values):.2f} s/{unit} at {canvas}" for canvas, values in sorted(samples.items())]


def run_summary(
    profile_key: str, timings: dict[str, PassTiming], work: Work, peak_vram_bytes: int, blocks: tuple[int, int] | None
) -> list[str]:
    """Log lines for a finished run: totals and the speed profile key its rates belong to,
    then per pass its wall time and warm rates (medians per canvas); ``blocks`` is
    ``(resident, total)`` transformer blocks when denoise ran."""
    windows = ", ".join(f"{count} at {canvas}" for canvas, count in sorted(work.windows.items())) or "0"
    window_steps = timings["denoise"].units if "denoise" in timings else 0
    lines = [
        f"[timing] LTX: {sum(t.seconds for t in timings.values()):.1f}s, {work.frames} frames, "
        f"windows {windows}, {window_steps} window-steps, peak VRAM {peak_vram_bytes / 2**20:.1f} MiB",
        f"[timing] LTX profile: {profile_key}",
    ]
    for stage in SUMMARY_PASSES:
        timing = timings.get(stage)
        if timing is None:
            continue
        parts = [f"{timing.seconds:.1f}s"]
        if stage == "denoise":
            parts += _medians(_by_canvas(timing.steps), "window-step")
            if blocks is not None:
                parts.append(f"blocks {blocks[0]}/{blocks[1]} resident")
        else:
            if timing.warm_seconds > 0:
                parts.append(f"{timing.warm_frames / timing.warm_seconds:.1f} frames/s")
            decode = "decode " if stage == "compose" else ""
            parts += [decode + part for part in _medians(warm_windows(stage, timing), "window")]
        lines.append(f"[timing] LTX {stage}: {', '.join(parts)}")
    return lines

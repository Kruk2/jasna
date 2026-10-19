"""LTX speed learned from finished runs, so a later run knows its time left once it is planned.

A profile (one JSON file) maps a key (Jasna version, GPU, transformer file, decode
budget: whole videos decode within the free VRAM, segments within a fixed budget, so their
decode tiling differs) to rates in
seconds: per pass a fixed part (model load, first-call compile), per frame-megapixel for
encode, per megapixel of window crop for compose (blending, which a frame without a mosaic
skips), and per window of each canvas. Compose decodes the next window while it blends
frames, so it takes the longer of its decodes and its blending, and blending is learned
from the time its frame loop did not wait for a decode. Rates are warm: the
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
    """What a run restores, known once the scan has planned it; ``crop_megapixels`` sums
    every window's crop over the frames it covers."""

    frames: int
    megapixels: float
    windows: dict[int, int]
    crop_megapixels: float

    def frame_units(self, stage: str, frames: int) -> dict[str, float]:
        """The units ``frames`` of the video stand for in ``stage``; compose's are their even
        share of the crops."""
        if stage == "encode":
            return {"encode/frame": frames * self.megapixels}
        if stage == "compose":
            return {"compose/crop": self.crop_megapixels * frames / self.frames if self.frames else 0.0}
        return {}

    def units(self, stage: str) -> dict[str, float]:
        units: dict[str, float] = {f"{stage}/window{canvas}": count for canvas, count in self.windows.items()}
        return {**units, **self.frame_units(stage, self.frames)}


@dataclass
class PassTiming:
    """One pass of a run: ``seconds`` since the previous pass ended (model load included);
    ``units`` it counted (frames, or window-steps for denoise); ``warm_seconds`` and
    ``warm_frames`` after its first batch; ``waited_seconds`` (compose) how long its frame
    loop waited for a window decode; ``windows`` the ``(canvas, seconds)`` of each
    window in order (denoise: seconds per window of a track without its first step);
    ``steps`` (denoise) the ``(canvas, seconds per window-step)`` of every window group's
    step after a track's first."""

    seconds: float = 0.0
    units: int = 0
    warm_seconds: float = 0.0
    warm_frames: int = 0
    waited_seconds: float = 0.0
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


def variable_seconds(rates: dict[str, float], stage: str, units: dict[str, float]) -> float:
    """Seconds of ``stage``'s ``units`` without its fixed part; compose decodes windows while
    it blends frames, so it takes the longer of the two."""
    seconds = {name: rates[name] * count for name, count in units.items()}
    if stage == "compose":
        blending = seconds.pop("compose/crop", 0.0)
        return max(blending, sum(seconds.values()))
    return sum(seconds.values())


def pass_estimates(rates: dict[str, float], work: Work) -> dict[str, tuple[float, float]] | None:
    """``(fixed, per-unit)`` seconds of each timed pass, or None when a rate is missing."""
    estimates = {}
    for stage in TIMED_PASSES:
        units = work.units(stage)
        if any(name not in rates for name in [*units, f"{stage}/fixed"]):
            return None
        estimates[stage] = (rates[f"{stage}/fixed"], variable_seconds(rates, stage, units))
    return estimates


UNPROFILED_FIXED_SECONDS = 30.0
UNPROFILED_SECONDS_PER_VIDEO_SECOND = 5.0


def segment_estimate_seconds(
    path: Path, gpu: str, transformer: str, *, frames: int, fps: float, megapixels: float, window_hop: int
) -> float:
    """About how long a segment render of ``frames`` takes, for one mosaic at the 512 canvas:
    from this GPU's learned rates when the profile has them, else 30 s plus 5x the range
    (an RTX 5090 run)."""
    from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES
    from jasna.ltx.plan import window_starts

    starts = window_starts(frames, window_hop)
    window_frames = sum(min(WINDOW_FRAMES, frames - start) for start in starts)
    crop_megapixels = window_frames * min(PLAN_CANVAS * PLAN_CANVAS / 1e6, megapixels)
    work = Work(frames, megapixels, {512: len(starts)}, crop_megapixels)
    estimates = pass_estimates(load_rates(path, profile_key(gpu, transformer, "segment")), work)
    if estimates is None:
        return UNPROFILED_FIXED_SECONDS + UNPROFILED_SECONDS_PER_VIDEO_SECOND * frames / fps
    return sum(fixed + per_unit for fixed, per_unit in estimates.values())


def learned_rates(timings: dict[str, PassTiming], work: Work, known: dict[str, float]) -> dict[str, float]:
    """The rates this run measured; a pass's fixed part needs its other rates, from this run
    or else from ``known``."""
    timings = {stage: timing for stage, timing in timings.items() if stage in TIMED_PASSES}
    rates: dict[str, float] = {}
    for stage, timing in timings.items():
        for canvas, samples in warm_windows(stage, timing).items():
            rates[f"{stage}/window{canvas}"] = mean(samples)
        window_seconds = timing.waited_seconds if stage == "compose" else sum(s for _, s in timing.windows)
        frame_seconds = max(0.0, timing.warm_seconds - window_seconds)
        for name, count in work.frame_units(stage, timing.warm_frames).items():
            if count:
                rates[name] = frame_seconds / count
    for stage, timing in timings.items():
        merged = {**known, **rates}
        units = work.units(stage)
        if all(name in merged for name in units):
            rates[f"{stage}/fixed"] = max(0.0, timing.seconds - variable_seconds(merged, stage, units))
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

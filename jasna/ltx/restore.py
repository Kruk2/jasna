"""Restore video with the LTX model in four passes, one model on the GPU at a time.

1. scan: decode, detect mosaics and scene cuts, plan tracks, cameras and windows;
2. encode: crop every window's 121 frames onto its canvas and VAE-encode them;
3. denoise: run each track's windows through the transformer in lockstep;
4. compose: decode each window's latent when the video reaches it, composite, write.

The unit of work is a segment: the frames of one pts range, planned on their own with
frame indices from the segment's first frame, so a segment restores the same way whatever
surrounds it. A whole video is one segment. Spans are the decoded stretches holding the
segments; every pass decodes each span again (the camera and fusion need a whole track
before its first window can be denoised). Per-window latents (1-2 MB) are kept in a
disk directory between passes, so memory does not grow with length. Offline jobs keep
that directory to resume; previews and speed tests use temporary directories.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from tqdm import tqdm

from jasna.accelerator import device_free_bytes
from jasna.job_resume import write_json
from jasna.os_utils import atomic_write, scratch_folder
from jasna.ltx import speed
from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES
from jasna.ltx.compose import Candidate, composite_frame, crop_to_canvas
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.inference import ModelRuntime, WorkerDecoder, WorkerEncoder, WorkerTransformer
from jasna.ltx.plan import (
    LARGE_CANVAS,
    WINDOW_HOP,
    Region,
    TrackPlan,
    Window,
    crossfade_weights,
    feather_pixels,
    plan_summary,
    plan_video,
    plan_window_lines,
    plans_from_arrays,
    plans_to_arrays,
)
from jasna.ltx.regions import regions_per_frame
from jasna.ltx.sampler import LtxTransformer
from jasna.ltx.super_res import LtxSuperRes, enlarged_windows, upscaled_crop
from jasna.models.ltx_vae import decode_latent
from jasna.mosaic.detections import Detections
from jasna.progressbar import StageProgress
from jasna.session_config import DecodeBackendName
from jasna.tensor_utils import half_dtype
from jasna.tracking.scene_detector import SceneCutDetector
from jasna.vr180 import StereoLayout
from jasna.vr_projection import RegionProjector

logger = logging.getLogger(__name__)

FrameBatches = Iterator[tuple[torch.Tensor, list[int]]]
FrameSource = Callable[[], FrameBatches]
DecodeBudget = Callable[[int], int]
LARGE_CANVAS_MIN_FREE_BYTES = 12 << 30
LARGE_CANVAS_MIN_TOTAL_BYTES = 15 << 30
SEGMENT_DECODE_BUDGETS = (
    (22 << 30, {PLAN_CANVAS: 8 << 30, LARGE_CANVAS: 16 << 30}),
    (15 << 30, {PLAN_CANVAS: 8 << 30, LARGE_CANVAS: 10 << 30}),
    (9 << 30, {PLAN_CANVAS: 6 << 30, LARGE_CANVAS: 8 << 30}),
    (7 << 30, {PLAN_CANVAS: 5 << 30, LARGE_CANVAS: 8 << 30}),
    (0, {PLAN_CANVAS: 4 << 30, LARGE_CANVAS: 8 << 30}),
)
"""(least total VRAM, decode budget per canvas): 8 GiB decodes a 512 window in one tile,
16 GiB a 768 window; 5 GiB keeps an 8 GB card's whole job under 7 GiB while BasicVSR++
and NVENC share it (#697), and a 6 GB card gets less again for the same reason (768 px
windows need 15 GiB, see ``segment_large_canvas``)."""
DECODE_BUDGET_STEP_BYTES = 512 << 20
DECODE_BUDGET_EXACT_BELOW = 4 << 30
DECODE_AHEAD = 2
"""Windows a compose pass keeps decoding or decoded ahead of the frames; each waits in host
memory (~95 MB at 512 px), or on the GPU from ``GPU_WINDOWS_MIN_FREE_BYTES`` of budget on."""
GPU_WINDOWS_MIN_FREE_BYTES = 18 << 30
"""Decode budget from which decoded windows stay on the GPU: the few held at once (~95 MB each
at 512 px, ~210 MB at 768) still leave every later decode the 14.75 GiB that decodes a 768 px
window in one tile, so no tile layout changes, and each frame skips uploading its canvases."""
ENCODE_AHEAD = 2
"""Filled windows an encode pass keeps encoding or queued while its frame loop fills the next
ones; each keeps its pixels (~95 MB at 512 px) until it is encoded."""
ENCODE_GPU_RESERVE_BYTES = 10 << 30
"""VRAM an encode pass leaves free beside the window pixels it keeps on the GPU: the encoder's
activations (up to ~5.7 GiB for a bf16 768 px window, an eager AMD one reserves up to ~8.3 GiB)."""
WHOLE_VIDEO_PTS = (-(1 << 63), (1 << 63) - 1)
SPEED_TEST_FRAMES = WINDOW_FRAMES + WINDOW_HOP
"""Two windows per track: the speed profile leaves out each canvas's cold first window."""


class Cancelled(Exception):
    pass


class FrameWriter(Protocol):
    """``finish`` once every frame is written; ``close`` (also after ``finish``) releases it."""

    def write(self, frame: torch.Tensor, pts: int) -> None: ...

    def finish(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LtxSegment:
    """The frames with ``start_pts <= pts < end_pts``; window ``i`` draws noise from ``seed + i``."""

    start_pts: int
    end_pts: int
    seed: int

    def contains(self, pts: int) -> bool:
        return self.start_pts <= pts < self.end_pts


@dataclass(frozen=True)
class FrameNumbers:
    """A frame's 0-based number in the whole input from its pts, whatever stretch was decoded."""

    start_pts: int
    time_base: Fraction
    fps: float

    @classmethod
    def of_video(cls, metadata) -> FrameNumbers:
        return cls(metadata.start_pts, metadata.time_base, metadata.mean_fps)

    def frame_number(self, pts: int) -> int:
        return round((int(pts) - self.start_pts) * self.time_base * self.fps)


@dataclass(frozen=True)
class LtxSpan:
    """``frames`` decodes a stretch of video holding ``segments``; every frame goes to the
    writer ``open_writer`` makes when the span is composed, a frame outside the segments
    unchanged and without the LUT."""

    frames: FrameSource
    segments: tuple[LtxSegment, ...]
    open_writer: Callable[[], FrameWriter]


LtxStage = Literal["scan", "encode", "denoise", "compose"]
ProgressReport = Callable[[float, float, StageProgress], None]
_STAGE_NUMBER = {"scan": 1, "encode": 2, "denoise": 3, "compose": 4}
_STAGE_SHARE = {"scan": 0.02, "encode": 0.08, "denoise": 0.8, "compose": 0.1}


class _WallClockMark:
    """``torch.cuda.Event``'s timing interface on the wall clock."""

    def __init__(self) -> None:
        self._at = 0.0

    def record(self) -> None:
        self._at = time.monotonic()

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end: _WallClockMark) -> float:
        return (end._at - self._at) * 1000.0


def _cuda_mark() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


TimingMark = torch.cuda.Event | _WallClockMark


def timing_marks(model) -> Callable[[], TimingMark]:
    """How to time ``model``'s windows: the LTX worker's GPU work never reaches this process's
    stream (and Windows batches the event records), but its requests return only once that
    work is done, so the wall clock times it; local models are timed by CUDA events."""
    if isinstance(model, (WorkerEncoder, WorkerTransformer, WorkerDecoder)):
        return _WallClockMark
    return _cuda_mark


class Progress:
    """One console bar per pass; ``frames`` sizes the frame-based passes. ``report`` gets
    ``(fraction of the whole run, seconds left, the pass's own progress)``. When the speed
    profile's rates (``expect``) cover what the scan planned, seconds left come from them
    from the end of the scan on; else they are 0 until the denoise pass has timed a step.
    A pass's own seconds left are the profile's seconds of work it has left, scaled by how
    long its work so far took against the profile (clock changes, uneven mosaics), else
    its pace so far. Every pass records its timing for the profile."""

    _bar_label = "LTX"

    def __init__(self, frames: int, *, disable: bool, report: ProgressReport | None) -> None:
        self.frames = frames
        self._disable = disable
        self._report = report
        self._started = time.monotonic()
        self._pass_end = self._started
        self._done_share = 0.0
        self._reported_fraction = 0.0
        self._timed = False
        self._resumed = False
        self._rates: dict[str, float] = {}
        self._work: speed.Work | None = None
        self._estimates: dict[str, tuple[float, float]] | None = None
        self.timings: dict[str, speed.PassTiming] = {}

    def expect(self, rates: dict[str, float]) -> None:
        """The profile's ``rates`` for the run, whose passes start now."""
        self._rates = rates
        self._started = self._pass_end = time.monotonic()

    def speed_test(self) -> Progress:
        """The progress of the speed test run before this run, reported as its own step."""
        report = self._report
        speed_test_report = None
        if report is not None:
            speed_test_report = lambda fraction, _left, _step: report(
                0.0, 0.0, StageProgress("speed_test", 1, 1, fraction, 0.0)
            )
        speed_test = Progress(min(self.frames, SPEED_TEST_FRAMES), disable=self._disable, report=speed_test_report)
        speed_test._bar_label = "LTX speed test"
        return speed_test

    def plan(self, work: speed.Work) -> None:
        self._work = work
        self._estimates = speed.pass_estimates(self._rates, work) if work.windows else None

    def learned_rates(self) -> dict[str, float]:
        return speed.learned_rates(self.timings, self._work, self._rates)

    def summary(
        self, profile_key: str, peak_vram_bytes: tuple[int, int | None], blocks: tuple[int, int] | None
    ) -> list[str]:
        return speed.run_summary(profile_key, self.timings, self._work, peak_vram_bytes, blocks)

    def bar(
        self, stage: LtxStage, total: int, unit: str = "frame", marks: Callable[[], TimingMark] = _cuda_mark
    ) -> _StageBar:
        """``marks`` times the pass's windows and steps (``timing_marks``)."""
        tqdm_bar = tqdm(
            total=total,
            desc=f"{self._bar_label} {_STAGE_NUMBER[stage]}/4 {stage}",
            unit=unit,
            dynamic_ncols=True,
            disable=self._disable,
        )
        return _StageBar(self, stage, tqdm_bar, marks)

    def begin(self, stage: LtxStage) -> None:
        """Report ``stage`` as started before its model loads, so loading shows as that pass."""
        self._advanced(stage, 0, 1, None, 0.0, 1.0)

    def skip(self, *stages: LtxStage) -> None:
        """Count ``stages`` as done, for a run that reuses their results."""
        for stage in stages:
            self._done_share += _STAGE_SHARE[stage]

    def skip_saved(self, *stages: LtxStage) -> None:
        """Count ``stages`` as done and left with no time, for a run resuming their saved results."""
        self._resumed = True
        self.skip(*stages)
        for stage in stages:
            if self._estimates is not None:
                self._estimates[stage] = (0.0, 0.0)
        self._reported_fraction = max(self._reported_fraction, self._done_share)

    def profiled_seconds(self, stage: LtxStage, frames: int, windows: dict[int, float]) -> float:
        """The profile's seconds for ``frames`` and ``windows`` (count per canvas) of
        ``stage``, 0 when the profile does not cover this run."""
        if self._estimates is None or stage not in self._estimates:
            return 0.0
        units = {f"{stage}/window{canvas}": count for canvas, count in windows.items()}
        return speed.variable_seconds(self._rates, stage, {**units, **self._work.frame_units(stage, frames)})

    def _profile_stage_left(self, stage: LtxStage, profiled_done: float, pace_ratio: float) -> float | None:
        if self._estimates is None or stage not in self._estimates:
            return None
        return max(0.0, self._estimates[stage][1] - profiled_done) * pace_ratio

    def _advanced(
        self, stage: LtxStage, done: int, total: int, paced_left: float | None, profiled_done: float, pace_ratio: float
    ) -> float | None:
        """Report the run's progress; ``paced_left`` is the pass's seconds left at its pace
        so far, ``profiled_done`` the profile's seconds of its work done and ``pace_ratio``
        its measured seconds per profile second. Returns the run's seconds left the
        profile gives, if any."""
        stage_fraction = min(1.0, done / total) if total else 1.0
        profile_stage_left = self._profile_stage_left(stage, profiled_done, pace_ratio)
        left = None
        if profile_stage_left is not None:
            later = speed.TIMED_PASSES[speed.TIMED_PASSES.index(stage) + 1 :]
            left = profile_stage_left + sum(sum(self._estimates[s]) for s in later)
        if self._report is not None:
            stage_left = profile_stage_left if profile_stage_left is not None else paced_left or 0.0
            step = StageProgress(stage, _STAGE_NUMBER[stage], len(_STAGE_NUMBER), stage_fraction, stage_left)
            elapsed = time.monotonic() - self._started
            if left is not None:
                fraction, seconds_left = elapsed / (elapsed + left), left
            else:
                fraction = self._done_share + _STAGE_SHARE[stage] * stage_fraction
                self._timed = self._timed or (stage == "denoise" and done > 0)
                seconds_left = elapsed * (1.0 - fraction) / fraction if self._timed else 0.0
            if self._resumed:
                fraction = max(fraction, self._done_share + _STAGE_SHARE[stage] * stage_fraction)
            self._reported_fraction = max(self._reported_fraction, fraction)
            self._report(self._reported_fraction, seconds_left, step)
        return left

    def _finished(self, stage: LtxStage, timing: speed.PassTiming) -> None:
        self._done_share += _STAGE_SHARE[stage]
        now = time.monotonic()
        timing.seconds = now - self._pass_end
        self._pass_end = now
        self.timings[stage] = timing


class _StageBar:
    def __init__(self, progress: Progress, stage: LtxStage, bar: tqdm, marks: Callable[[], TimingMark]) -> None:
        self._progress = progress
        self._stage = stage
        self._bar = bar
        self._marks = marks
        self._done = 0
        self._frames_done = 0
        self._windows_done: Counter[int] = Counter()
        self._warm_from: tuple[float, int, float] | None = None
        self._windows: list[tuple[int, TimingMark, TimingMark]] = []
        self._waits: list[tuple[TimingMark, TimingMark]] = []
        self._tracks: list[tuple[int, int, TimingMark, list[tuple[int, TimingMark]]]] = []
        progress._advanced(stage, 0, bar.total, None, 0.0, 1.0)

    def update(self, n: int) -> None:
        self._frames_done += n
        self._count(n)

    def reuse(self, done: int, frames: int, windows: Counter[int]) -> None:
        """Count saved work a resumed run skips: ``done`` units holding ``frames`` frames and
        ``windows`` (count per canvas). Its pace is timed from the next work it does."""
        if done == 0 and not windows:
            return
        self._progress._resumed = True
        self._frames_done += frames
        self._windows_done.update(windows)
        if done:
            self._count(done)
            self._warm_from = None

    def _count(self, n: int) -> None:
        self._bar.update(n)
        self._done += n
        now = time.monotonic()
        profiled_done = self._progress.profiled_seconds(self._stage, self._frames_done, self._windows_done)
        if self._warm_from is None:
            self._warm_from = (now, self._done, profiled_done)
        started, warm_done, warm_profiled = self._warm_from
        paced_left = (
            (now - started) / (self._done - warm_done) * (self._bar.total - self._done)
            if self._done > warm_done
            else None
        )
        profiled_since_warm = profiled_done - warm_profiled
        pace_ratio = (now - started) / profiled_since_warm if profiled_since_warm > 0 else 1.0
        left = self._progress._advanced(self._stage, self._done, self._bar.total, paced_left, profiled_done, pace_ratio)
        if left is not None:
            self._bar.set_postfix_str(f"run {tqdm.format_interval(left)} left", refresh=False)

    @contextmanager
    def window(self, canvas: int) -> Iterator[None]:
        """Time the GPU work of one window with marks, read when the pass closes; the marks go
        on the calling thread's stream, so a decode thread can time its own windows."""
        start, end = self._marks(), self._marks()
        start.record()
        yield
        end.record()
        self._windows.append((canvas, start, end))

    @contextmanager
    def waiting(self) -> Iterator[None]:
        """Time the frame loop waiting for a window decoded on another thread; on CUDA the
        marks go on the loop's stream, so they time how long it ran dry."""
        start, end = self._marks(), self._marks()
        start.record()
        yield
        end.record()
        self._waits.append((start, end))

    def windows_done(self, canvas: int, count: float) -> None:
        """Count ``count`` windows of ``canvas`` as done toward the time left."""
        self._windows_done[canvas] += count

    def track(self, canvas: int, steps: int) -> Callable[[int], None]:
        """``advance`` for one track's denoise: counts its window-steps and marks the stream
        at the track's start and after each window group's step, for ``PassTiming.add_track``."""
        start = self._marks()
        start.record()
        marks: list[tuple[int, TimingMark]] = []
        self._tracks.append((canvas, steps, start, marks))

        def advance(n: int) -> None:
            self.windows_done(canvas, n / steps)
            self._count(n)
            mark = self._marks()
            mark.record()
            marks.append((n, mark))

        return advance

    def close(self) -> None:
        self._bar.close()
        timing = speed.PassTiming(units=self._done)
        if self._warm_from is not None:
            started, done, _ = self._warm_from
            timing.warm_seconds = time.monotonic() - started
            timing.warm_frames = self._done - done
        for canvas, start, end in self._windows:
            end.synchronize()
            timing.windows.append((canvas, start.elapsed_time(end) / 1000))
        for start, end in self._waits:
            end.synchronize()
            timing.waited_seconds += start.elapsed_time(end) / 1000
        for canvas, steps, start, marks in self._tracks:
            if marks:
                marks[-1][1].synchronize()
            timing.add_track(canvas, steps, [(n, start.elapsed_time(mark) / 1000) for n, mark in marks])
        self._progress._finished(self._stage, timing)

    def __enter__(self) -> _StageBar:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


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


def video_frames(
    path: Path,
    metadata,
    *,
    batch_size: int,
    device: torch.device,
    frame_stride: int,
    decode_backend: DecodeBackendName,
    seek_ts: float | None,
    end_pts: int | None,
) -> FrameSource:
    """Decoded batches of ``path`` from the keyframe at ``seek_ts`` (the start when None),
    stopping before ``end_pts`` (the end when None)."""
    from jasna.media.video_decoder import VideoReader

    def frames() -> FrameBatches:
        with VideoReader(
            str(path), batch_size, device, metadata, frame_stride=frame_stride, decode_backend=decode_backend
        ) as reader:
            for batch, pts in reader.frames(seek_ts):
                if end_pts is None:
                    yield batch, pts
                    continue
                keep = next((i for i, p in enumerate(pts) if int(p) >= end_pts), len(pts))
                if keep:
                    yield batch[:keep], pts[:keep]
                if keep < len(pts):
                    return

    return frames


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def segment_batches(
    batches: FrameBatches, segments: Sequence[LtxSegment], batch_size: int
) -> Iterator[tuple[int | None, torch.Tensor, list[int]]]:
    """``(segment index, frames, pts)`` in decode order. A segment's frames come in batches
    of ``batch_size`` counted from its first frame (the last may be short), so detection
    sees the same batches wherever decoding started; frames outside every segment come as
    decoded with index None."""
    owner: int | None = None
    pending: list[tuple[torch.Tensor, list[int]]] = []

    def full_batches(flush: bool) -> Iterator[tuple[int | None, torch.Tensor, list[int]]]:
        count = sum(len(pts) for _, pts in pending)
        if count == 0 or (count < batch_size and not flush):
            return
        if len(pending) == 1:
            frames, pts = pending[0]
        else:
            frames = torch.cat([chunk for chunk, _ in pending])
            pts = [p for _, chunk_pts in pending for p in chunk_pts]
        pending.clear()
        whole = len(pts) if flush else len(pts) - len(pts) % batch_size
        for start in range(0, whole, batch_size):
            yield owner, frames[start : start + batch_size], pts[start : start + batch_size]
        if whole < len(pts):
            pending.append((frames[whole:], pts[whole:]))

    for frames, pts in batches:
        owners = [next((i for i, s in enumerate(segments) if s.contains(int(p))), None) for p in pts]
        start = 0
        while start < len(pts):
            stop = start + 1
            while stop < len(pts) and owners[stop] == owners[start]:
                stop += 1
            if owners[start] != owner:
                yield from full_batches(flush=True)
                owner = owners[start]
            if owner is None:
                yield None, frames[start:stop], list(pts[start:stop])
            else:
                pending.append((frames[start:stop], list(pts[start:stop])))
                yield from full_batches(flush=False)
            start = stop
    yield from full_batches(flush=True)


def scan_span(
    frames: FrameSource,
    segments: Sequence[LtxSegment],
    detector,
    *,
    batch_size: int,
    frame_h: int,
    frame_w: int,
    stereo: StereoLayout | None,
    large_canvas: bool,
    window_hop: int,
    bar: _StageBar,
    cancel: threading.Event,
    max_gap: int,
    min_duration: int,
    frame_numbers: FrameNumbers,
) -> list[list[TrackPlan]]:
    """The track plans of each segment, frame indices counted from the segment's first frame.
    Each plan is logged, its windows (debug) in ``frame_numbers`` of the whole input."""
    regions: list[list[list[Region]]] = [[] for _ in segments]
    cuts: list[set[int]] = [set() for _ in segments]
    input_frames: list[list[int]] = [[] for _ in segments]
    scenes = [SceneCutDetector() for _ in segments]
    for owner, batch, pts in segment_batches(frames(), segments, batch_size):
        _check(cancel)
        if owner is not None:
            cuts[owner].update(len(regions[owner]) + offset for offset in scenes[owner].find_cuts(batch))
            regions[owner].extend(regions_per_frame(detector(batch, target_hw=(frame_h, frame_w)), frame_h, frame_w))
            input_frames[owner].extend(frame_numbers.frame_number(p) for p in pts)
        bar.update(len(batch))
    plans = []
    for segment_regions, segment_cuts, segment_frames in zip(regions, cuts, input_frames):
        segment_plans = plan_video(
            segment_regions,
            segment_cuts,
            frame_w=frame_w,
            frame_h=frame_h,
            stereo=stereo,
            large_canvas=large_canvas,
            window_hop=window_hop,
            max_gap=max_gap,
            min_duration=min_duration,
        )
        logger.info(plan_summary(len(segment_regions), len(segment_cuts), segment_plans))
        for line in plan_window_lines(segment_plans, segment_cuts, segment_frames):
            logger.debug(line)
        plans.append(segment_plans)
    return plans


def _vae_pixels(window_pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    return window_pixels.to(device).float().div_(127.5).sub_(1.0).to(half_dtype(device)).unsqueeze(0)


def encode_gpu_window_bytes(device: torch.device) -> int:
    """Bytes of window pixels an encode pass keeps on ``device``: what the GPU has free beyond
    ``ENCODE_GPU_RESERVE_BYTES``, none on the CPU."""
    if device.type != "cuda":
        return 0
    return max(0, device_free_bytes(device) - ENCODE_GPU_RESERVE_BYTES)


class _WindowBuffers:
    """``[3, WINDOW_FRAMES, canvas, canvas]`` uint8 pixels for the open windows, reused once a
    window is encoded, so no window stacks its frames or faults in fresh pages. The first ones up
    to ``gpu_bytes`` live on ``device``, where the frame loop queues its crops without waiting for
    each to reach host memory behind the encoder's kernels; the rest live in host memory."""

    def __init__(self, device: torch.device, gpu_bytes: int) -> None:
        self._device = device
        self._gpu_bytes_left = gpu_bytes
        self._free: dict[int, list[torch.Tensor]] = {}

    def take(self, canvas: int) -> torch.Tensor:
        free = self._free.setdefault(canvas, [])
        if free:
            return free.pop()
        shape = (3, WINDOW_FRAMES, canvas, canvas)
        size = math.prod(shape)
        if size <= self._gpu_bytes_left:
            self._gpu_bytes_left -= size
            return torch.empty(shape, dtype=torch.uint8, device=self._device)
        return torch.empty(shape, dtype=torch.uint8)

    def give_back(self, canvas: int, pixels: torch.Tensor) -> None:
        self._free[canvas].append(pixels)


def _copy_canvas(window_pixels: torch.Tensor, position: int, real_frames: int, canvas: torch.Tensor) -> None:
    """``canvas`` into frame ``position``; the last real frame also fills the padding frames."""
    last = position == real_frames - 1
    window_pixels[:, position : WINDOW_FRAMES if last else position + 1].copy_(canvas.unsqueeze(1))


class _WindowEncodes:
    """Encodes filled windows on a thread of its own (on CUDA on a stream of its own), so the frame
    loop decodes and crops the next windows while the encoder runs; once ``ENCODE_AHEAD`` windows
    wait, ``submit`` first finishes the oldest. Pixels on the GPU are read once the frame loop's
    stream has cropped them. A window's buffer goes back once its latent is in host memory; the
    latent then goes to its store on a thread of its own, so the encoder never waits for the disk
    (each write ends with an fsync, ~30 ms on a hard disk)."""

    def __init__(self, encoder, buffers: _WindowBuffers, device: torch.device, bar: _StageBar) -> None:
        self._encoder = encoder
        self._buffers = buffers
        self._device = device
        self._bar = bar
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxEncode")
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxLatentWrite")
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._queued: list[tuple[Future[None], int, torch.Tensor]] = []
        self._writes: list[Future[None]] = []

    def submit(self, store: LatentStore, index: int, canvas: int, window_pixels: torch.Tensor) -> None:
        while len(self._queued) >= ENCODE_AHEAD:
            self._finish_oldest()
        cropped = None
        if window_pixels.is_cuda:
            cropped = torch.cuda.Event()
            cropped.record()
        encode = self._thread.submit(self._encode, store, index, canvas, window_pixels, cropped)
        self._queued.append((encode, canvas, window_pixels))

    def _encode(
        self, store: LatentStore, index: int, canvas: int, window_pixels: torch.Tensor, cropped: torch.cuda.Event | None
    ) -> None:
        with torch.cuda.stream(self._stream), self._bar.window(canvas), torch.inference_mode():
            if cropped is not None:
                self._stream.wait_event(cropped)
                window_pixels.record_stream(self._stream)
            pixels = (
                window_pixels.unsqueeze(0).cpu()
                if isinstance(self._encoder, WorkerEncoder) else _vae_pixels(window_pixels, self._device)
            )
            latent = self._encoder(pixels).cpu()
        self._writes.append(self._writer.submit(store.put, index, latent))

    def _finish_oldest(self) -> None:
        future, canvas, window_pixels = self._queued.pop(0)
        future.result()
        self._bar.windows_done(canvas, 1)
        self._buffers.give_back(canvas, window_pixels)

    def finish(self) -> None:
        while self._queued:
            self._finish_oldest()
        for write in self._writes:
            write.result()

    def close(self) -> None:
        self._thread.shutdown(cancel_futures=True)
        self._writer.shutdown()


class _SegmentEncoder:
    def __init__(
        self,
        plans: list[TrackPlan],
        store: LatentStore,
        buffers: _WindowBuffers,
        encodes: _WindowEncodes,
        projector: RegionProjector | None,
    ) -> None:
        self._pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
        self._upcoming = next(self._pending, None)
        self._active: dict[int, tuple[Window, torch.Tensor]] = {}
        self._projector = projector
        self._store = store
        self._buffers = buffers
        self._encodes = encodes
        self._frame_idx = 0

    def frame(self, frame: torch.Tensor) -> None:
        while self._upcoming is not None and self._upcoming.start == self._frame_idx:
            self._active[self._upcoming.index] = (self._upcoming, self._buffers.take(self._upcoming.canvas))
            self._upcoming = next(self._pending, None)
        for index, (window, window_pixels) in list(self._active.items()):
            position = self._frame_idx - window.start
            _copy_canvas(window_pixels, position, window.real_frames, crop_to_canvas(frame, window.crops[position], self._projector))
            if position == window.real_frames - 1:
                self._encodes.submit(self._store, index, window.canvas, window_pixels)
                del self._active[index]
        self._frame_idx += 1

    def finish(self) -> None:
        if self._active or self._upcoming is not None:
            raise RuntimeError("the video ended before every planned window was filled")


def encode_span(
    frames: FrameSource,
    segments: Sequence[LtxSegment],
    plans: Sequence[list[TrackPlan]],
    encoder,
    stores: Sequence[LatentStore],
    *,
    batch_size: int,
    device: torch.device,
    bar: _StageBar,
    cancel: threading.Event,
    projector: RegionProjector | None,
) -> None:
    """VAE-encode every planned window's reference crops into its segment's store."""
    buffers = _WindowBuffers(device, encode_gpu_window_bytes(device))
    encodes = _WindowEncodes(encoder, buffers, device, bar)
    try:
        encoders = [_SegmentEncoder(p, store, buffers, encodes, projector) for p, store in zip(plans, stores)]
        for owner, batch, _pts in segment_batches(frames(), segments, batch_size):
            _check(cancel)
            bar.update(len(batch))
            if owner is not None:
                for frame in batch:
                    encoders[owner].frame(frame)
        encodes.finish()
        for segment_encoder in encoders:
            segment_encoder.finish()
    finally:
        encodes.close()


def denoise_track(
    plan: TrackPlan, transformer: LtxTransformer, references: LatentStore, *, seed: int, advance: Callable[[int], None]
) -> list[torch.Tensor]:
    """The final latents of one track's windows, in window order."""
    return transformer.denoise_chain(
        [references.get(w.index) for w in plan.windows],
        [seed + w.noise for w in plan.windows],
        plan.window_hop,
        advance,
    )


def free_vram_budget(device: torch.device) -> DecodeBudget:
    """Free VRAM rounded down to ``DECODE_BUDGET_STEP_BYTES``: decode tiling follows the budget and
    each tile shape is a compiled graph, so an exact figure that wanders (8 GB cards) runs past
    the compile limit (#690). Under ``DECODE_BUDGET_EXACT_BELOW`` the exact figure stays, so the
    step never drops a decode that the smallest tiles (~3.3 GiB at 512) would fit."""

    def budget(_canvas: int) -> int:
        free = device_free_bytes(device)
        rounded = free // DECODE_BUDGET_STEP_BYTES * DECODE_BUDGET_STEP_BYTES
        return rounded if rounded >= DECODE_BUDGET_EXACT_BELOW else free

    return budget


def segment_decode_budget(total_bytes: int) -> DecodeBudget:
    """A decode budget fixed by the GPU's size, so a segment's tile layout (and pixels) never
    depend on the VRAM free at that moment."""
    return next(budgets for least, budgets in SEGMENT_DECODE_BUDGETS if total_bytes >= least).__getitem__


def _decode_window(
    decoder, latent: torch.Tensor, seed: int, canvas: int, budget: DecodeBudget, device: torch.device
) -> torch.Tensor:
    """Decoded window as uint8 ``[F, 3, S, S]``, in host memory while the budget is under
    ``GPU_WINDOWS_MIN_FREE_BYTES``: a frame can need every open window of its tracks, and holding
    them on the GPU would shrink the next decode's budget. From there on it stays on the GPU."""
    torch.cuda.empty_cache()
    free = budget(canvas)
    held = device if device.type == "cuda" and free >= GPU_WINDOWS_MIN_FREE_BYTES else torch.device("cpu")
    if isinstance(decoder, WorkerDecoder):
        return decoder.decode_window(latent, seed, free).to(held)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        video = decode_latent(decoder, latent.to(device), free_bytes=free, generator=generator)
        video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)[0].float()
        return video.mul_(255.0).round_().to(torch.uint8).permute(1, 0, 2, 3).contiguous().to(held)


def _decode_shares(windows: list[Window]) -> dict[int, list[tuple[int, float]]]:
    """Per segment frame, the ``(canvas, share)`` of each window decode it counts: a window
    decodes from the start of the window ``DECODE_AHEAD`` before it (the first ones from
    frame 0) to its own start, so its decode counts evenly over those frames, or all at its
    start when there are none."""
    shares: dict[int, list[tuple[int, float]]] = {}
    submitted = [0] * DECODE_AHEAD + [window.start for window in windows]
    for window, start in zip(windows, submitted):
        frames = range(start, window.start) or range(window.start, window.start + 1)
        for frame_idx in frames:
            shares.setdefault(frame_idx, []).append((window.canvas, 1 / len(frames)))
    return shares


class _SegmentComposer:
    def __init__(
        self,
        plans: list[TrackPlan],
        decoder,
        finals: LatentStore,
        *,
        seed: int,
        feather: int,
        budget: DecodeBudget,
        device: torch.device,
        bar: _StageBar,
        decodes: ThreadPoolExecutor,
        stream: torch.cuda.Stream | None,
        super_res: LtxSuperRes | None,
        projector: RegionProjector | None,
    ) -> None:
        self._weights: dict[int, torch.Tensor] = {}
        self._tracks: dict[int, TrackPlan] = {}
        for plan in plans:
            for window, weight in zip(plan.windows, crossfade_weights(plan.windows)):
                self._weights[window.index] = weight
                self._tracks[window.index] = plan
        windows = sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start)
        self._super_res = super_res
        self._projector = projector
        self._upscaled = enlarged_windows(windows) if super_res is not None else set()
        self._pending = iter(windows)
        self._upcoming = next(self._pending, None)
        self._covering: list[Window] = []
        self._decoded: dict[int, torch.Tensor] = {}
        self._decoder = decoder
        self._finals = finals
        self._seed = seed
        self._feather = feather
        self._budget = budget
        self._device = device
        self._bar = bar
        self._frame_idx = 0
        self._decodes = decodes
        self._stream = stream
        self._decode_order = iter(windows)
        self._decode_shares = _decode_shares(windows)
        self._decoding: dict[int, Future[torch.Tensor]] = {}

    def _decode(self, window: Window) -> torch.Tensor:
        with torch.cuda.stream(self._stream), self._bar.window(window.canvas):
            video = _decode_window(
                self._decoder,
                self._finals.get(window.index),
                self._seed + window.noise,
                window.canvas,
                self._budget,
                self._device,
            )
        if self._stream is not None:
            self._stream.synchronize()
        return video

    def _decode_ahead(self) -> None:
        """Queue the next windows in start order until ``DECODE_AHEAD`` are decoding or wait
        unused, so they decode back to back while the frames before their starts are
        composited and written."""
        while len(self._decoding) < DECODE_AHEAD and (window := next(self._decode_order, None)) is not None:
            self._decoding[window.index] = self._decodes.submit(self._decode, window)

    def _decoded_window(self, window: Window) -> torch.Tensor:
        with self._bar.waiting():
            video = self._decoding.pop(window.index).result()
        if video.is_cuda:
            # decoded on the decode stream, read by this stream's frames: keep its memory from the
            # next decode until they ran
            video.record_stream(torch.cuda.current_stream(video.device))
        self._decode_ahead()
        return video

    def frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame_idx = self._frame_idx
        if frame_idx == 0:
            self._decode_ahead()
        while self._upcoming is not None and self._upcoming.start == frame_idx:
            self._covering.append(self._upcoming)
            self._upcoming = next(self._pending, None)
        for window in [w for w in self._covering if w.stop <= frame_idx]:
            self._covering.remove(window)
            self._decoded.pop(window.index, None)
        candidates = []
        for window in self._covering:
            if window.index not in self._decoded:
                self._decoded[window.index] = self._decoded_window(window)
            local = frame_idx - window.start
            track = self._tracks[window.index]
            canvas, crop = self._decoded[window.index][local].to(self._device, non_blocking=True), window.crops[local]
            if window.index in self._upscaled:
                canvas, crop = self._super_res.upscale(canvas), upscaled_crop(crop)
            candidates.append(
                Candidate(
                    canvas=canvas,
                    crop=crop,
                    weight=float(self._weights[window.index][local]),
                    track=track.track_id,
                    polygons=track.polygons[frame_idx - track.start],
                    presence=track.presence[frame_idx - track.start],
                )
            )
        for canvas, share in self._decode_shares.get(frame_idx, ()):
            self._bar.windows_done(canvas, share)
        self._frame_idx += 1
        return composite_frame(frame, candidates, feather=self._feather, projector=self._projector)


def compose_span(
    span: LtxSpan,
    plans: Sequence[list[TrackPlan]],
    decoder,
    finals: Sequence[LatentStore],
    *,
    frame_h: int,
    batch_size: int,
    budget: DecodeBudget,
    device: torch.device,
    bar: _StageBar,
    cancel: threading.Event,
    super_res: LtxSuperRes | None,
    projector: RegionProjector | None,
) -> None:
    """Write every frame of ``span``: segment frames composited, the rest unchanged. Windows
    decode ``DECODE_AHEAD`` ahead on a thread; on CUDA on a stream of their own, below the priority of the
    frames' stream, whose short kernels would otherwise queue behind the decode's."""
    feather = feather_pixels(frame_h)
    decodes = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxDecode")
    cuda = device.type == "cuda"
    decode_stream = torch.cuda.Stream(device) if cuda else None
    try:
        composers = [
            _SegmentComposer(
                p, decoder, store, seed=segment.seed, feather=feather, budget=budget, device=device, bar=bar,
                decodes=decodes, stream=decode_stream, super_res=super_res, projector=projector,
            )
            for p, store, segment in zip(plans, finals, span.segments)
        ]
        with torch.cuda.stream(torch.cuda.Stream(device, priority=-1) if cuda else None):
            writer = span.open_writer()
            try:
                for owner, batch, pts in segment_batches(span.frames(), span.segments, batch_size):
                    _check(cancel)
                    bar.update(len(batch))
                    for frame, frame_pts in zip(batch, pts):
                        if owner is None:
                            writer.write(frame, frame_pts)
                        else:
                            writer.write(composers[owner].frame(frame), frame_pts)
                writer.finish()
            finally:
                writer.close()
    finally:
        decodes.shutdown(cancel_futures=True)


def large_canvas_fits(requested: bool, free_bytes: int) -> bool:
    """768 px windows need ~7.5 GiB of VAE-encoder activations alone; below 12 GiB free
    every window runs at 512 px instead."""
    if requested and free_bytes < LARGE_CANVAS_MIN_FREE_BYTES:
        logger.warning("LTX: %.1f GiB free VRAM, restoring large mosaics at 512 px", free_bytes / 2**30)
        return False
    return requested


def whole_video_large_canvas(requested: bool, device: torch.device, resume_dir: Path) -> bool:
    """The 768 px decision of a whole-video run: saved work keeps the one it was made with,
    else the VRAM free now decides."""
    saved = _read_plans(_plans_file(_pts_key(*WHOLE_VIDEO_PTS), resume_dir))
    if saved is None:
        return large_canvas_fits(requested, device_free_bytes(device))
    free = device_free_bytes(device)
    if saved.large_canvas and free < LARGE_CANVAS_MIN_FREE_BYTES:
        raise RuntimeError(
            f"The saved LTX work for this video uses 768 px windows, which need "
            f"{LARGE_CANVAS_MIN_FREE_BYTES / 2**30:.0f} GiB of free GPU memory; {free / 2**30:.1f} GiB is free. "
            "Close other programs that use the GPU and try again."
        )
    return saved.large_canvas


def segment_large_canvas(requested: bool, total_bytes: int) -> bool:
    """The 768 px decision for segments, from the GPU's size so it is the same every run."""
    if requested and total_bytes < LARGE_CANVAS_MIN_TOTAL_BYTES:
        logger.warning("LTX: %.1f GiB GPU, restoring large mosaics at 512 px", total_bytes / 2**30)
        return False
    return requested


@dataclass(frozen=True)
class LtxRender:
    """What every pass needs besides the spans: the detector, the model files, the frame
    size, where a stereo VR frame keeps its eyes (None for flat video), detection batch
    size, the 768 px decision, the frames between window starts, the decode budget per
    canvas and its name (decode tiling follows the budget, so it keys the speed profile),
    the GPU, the speed profile file (``speed.py``), the max detection gap a track bridges
    (``track_max_gap``), the fewest frames a kept track spans, the super resolution of
    windows compose enlarges (``super_res.py``), if any, the VR fisheye/gnomonic
    projector crops are seen through, if any, and the input's frame numbers the plan logs."""

    detector: Callable
    files: LtxModelFiles
    frame_h: int
    frame_w: int
    stereo: StereoLayout | None
    batch_size: int
    large_canvas: bool
    window_hop: int
    budget: DecodeBudget
    budget_name: str
    device: torch.device
    speed_profile: Path
    max_gap: int
    min_duration: int
    super_res: LtxSuperRes | None
    vr_projector: RegionProjector | None
    frame_numbers: FrameNumbers


@dataclass(frozen=True)
class PreparedSpan:
    """A span planned and encoded: per segment its track plans and reference latents.
    Neither depends on the seed, so one prepared span can be denoised with many seeds."""

    span: LtxSpan
    plans: tuple[list[TrackPlan], ...]
    references: tuple[LatentStore, ...]

    @property
    def windows(self) -> int:
        return sum(len(plan.windows) for segment_plans in self.plans for plan in segment_plans)

    @property
    def largest_canvas(self) -> int:
        return max(window.canvas for segment_plans in self.plans for plan in segment_plans for window in plan.windows)


def prepare_spans(
    spans: Sequence[LtxSpan], render: LtxRender, directory: Path, *, progress: Progress, cancel: threading.Event,
    runtime: ModelRuntime | None = None,
) -> list[PreparedSpan]:
    """Scan and VAE-encode every span; reference latents go to ``directory``. A ``runtime`` passed
    in will denoise next, so its worker starts and opens the transformer file during the scan; it
    holds no model on the GPU before the encoder loads."""
    worker_ready = None
    if runtime is not None and not all(_plan_path(span, directory).is_file() for span in spans):
        starting = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxWorkerStart")
        worker_ready = starting.submit(runtime.prepare_transformer)
        starting.shutdown(wait=False)
    try:
        prepared = _scan_spans(spans, render, directory, progress=progress, cancel=cancel, worker_ready=worker_ready)
    except BaseException:
        _settle(worker_ready)
        raise
    finals = final_stores(prepared, directory)
    pending = []
    reused: Counter[int] = Counter()
    for p, span_finals in zip(prepared, finals):
        segment_plans = []
        for plans, references, store in zip(p.plans, p.references, span_finals):
            remaining = []
            for plan in plans:
                complete = store.has_track(plan)
                windows = []
                for window in plan.windows:
                    if complete or references.has(window.index):
                        reused[window.canvas] += 1
                    else:
                        windows.append(window)
                if windows:
                    remaining.append(replace(plan, windows=tuple(windows)))
            segment_plans.append(remaining)
        pending.append(replace(p, plans=tuple(segment_plans)))
    if any(p.windows for p in pending):
        progress.begin("encode")
        owned_runtime = runtime is None
        runtime = runtime or ModelRuntime(render.files, render.device, cancel)
        try:
            if worker_ready is not None:
                worker_ready.result()
            encoder = runtime.load_encoder()
            try:
                with progress.bar("encode", progress.frames, marks=timing_marks(encoder)) as bar:
                    encoded = [p.windows > 0 or not done.windows for p, done in zip(pending, prepared)]
                    saved_frames = sum(
                        _saved_frames(p.span, directory) for p, encode in zip(prepared, encoded) if not encode
                    )
                    bar.reuse(saved_frames, saved_frames, reused)
                    for p, encode in zip(pending, encoded):
                        if encode:
                            encode_span(
                                p.span.frames, p.span.segments, p.plans, encoder, p.references,
                                batch_size=render.batch_size, device=render.device, bar=bar, cancel=cancel,
                                projector=render.vr_projector,
                            )
            finally:
                del encoder
                runtime.unload_encoder()
        finally:
            if owned_runtime:
                runtime.close()
    else:
        if reused:
            progress.skip_saved("encode")
        _settle(worker_ready)
    return prepared


def _settle(worker_ready: Future | None) -> None:
    """Waits for a worker start nothing needs any more (so the caller can close the worker);
    its error is only logged."""
    if worker_ready is not None and (error := worker_ready.exception()) is not None:
        logger.debug("LTX worker started ahead of a pass that did not need it failed: %r", error)


def _scan_spans(
    spans: Sequence[LtxSpan], render: LtxRender, directory: Path, *, progress: Progress, cancel: threading.Event,
    worker_ready: Future | None,
) -> list[PreparedSpan]:
    plans = []
    cached = [_saved_plans(span, directory) for span in spans]
    if all(item is not None for item in cached):
        progress.skip_saved("scan")
        plans = [item.plans for item in cached]
    else:
        with progress.bar("scan", progress.frames) as bar:
            for span, saved in zip(spans, cached):
                _check(cancel)
                if saved is not None:
                    plans.append(saved.plans)
                    bar.reuse(saved.frames, saved.frames, Counter())
                    continue
                count = 0

                def counted_frames():
                    nonlocal count
                    with closing(span.frames()) as batches:
                        for batch, pts in batches:
                            # A worker that failed to start ends the scan with its error.
                            if worker_ready is not None and worker_ready.done():
                                worker_ready.result()
                            count += len(pts)
                            yield batch, pts

                span_plans = scan_span(
                    counted_frames, span.segments, render.detector,
                    batch_size=render.batch_size, frame_h=render.frame_h, frame_w=render.frame_w, stereo=render.stereo,
                    large_canvas=render.large_canvas, window_hop=render.window_hop, bar=bar, cancel=cancel,
                    max_gap=render.max_gap, min_duration=render.min_duration, frame_numbers=render.frame_numbers,
                )
                _check(cancel)
                with atomic_write(_plan_path(span, directory), "wb") as stream:
                    np.savez(stream, frames=np.array(count), **plans_to_arrays(span_plans))
                plans.append(span_plans)
    prepared = [
        PreparedSpan(span, tuple(span_plans), _span_stores(span, directory, "reference"))
        for span, span_plans in zip(spans, plans)
    ]
    planned = [w for p in prepared for track_plans in p.plans for plan in track_plans for w in plan.windows]
    crop_pixels = sum(
        (c.crop_box[2] - c.crop_box[0]) * (c.crop_box[3] - c.crop_box[1]) for w in planned for c in w.crops
    )
    progress.plan(
        speed.Work(
            progress.frames,
            render.frame_h * render.frame_w / 1e6,
            dict(Counter(w.canvas for w in planned)),
            crop_pixels / 1e6,
        )
    )
    return prepared


def _pts_key(start_pts: int, end_pts: int) -> str:
    return f"{start_pts}-{end_pts}"


def _span_key(span: LtxSpan) -> str:
    return _pts_key(span.segments[0].start_pts, span.segments[-1].end_pts)


def _plans_file(key: str, directory: Path) -> Path:
    return directory / f"{key}.plans.npz"


def _plan_path(span: LtxSpan, directory: Path) -> Path:
    return _plans_file(_span_key(span), directory)


@dataclass(frozen=True)
class _SavedPlans:
    frames: int
    plans: list[list[TrackPlan]]

    @property
    def large_canvas(self) -> bool:
        return any(w.canvas == LARGE_CANVAS for segment in self.plans for plan in segment for w in plan.windows)


def _saved_plans(span: LtxSpan, directory: Path) -> _SavedPlans | None:
    return _read_plans(_plan_path(span, directory))


def _read_plans(path: Path) -> _SavedPlans | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        return _SavedPlans(int(arrays["frames"]), plans_from_arrays(arrays))


def _saved_frames(span: LtxSpan, directory: Path) -> int:
    with np.load(_plan_path(span, directory), allow_pickle=False) as arrays:
        return int(arrays["frames"])


def _span_stores(span: LtxSpan, directory: Path, kind: str) -> tuple[LatentStore, ...]:
    return tuple(LatentStore(directory, f"{_span_key(span)}-{g}.{kind}") for g in range(len(span.segments)))


def final_stores(prepared: Sequence[PreparedSpan], directory: Path) -> list[tuple[LatentStore, ...]]:
    return [_span_stores(p.span, directory, "final") for p in prepared]


def denoise_spans(
    prepared: Sequence[PreparedSpan],
    finals: Sequence[tuple[LatentStore, ...]],
    transformer: LtxTransformer,
    *,
    keep_references: bool,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    """Denoise every track with its segment's seed into ``finals``; ``keep_references``
    keeps the reference latents for another seed."""
    windows = sum(p.windows for p in prepared)
    with progress.bar(
        "denoise", len(transformer.conditions) * windows, unit="step", marks=timing_marks(transformer)
    ) as bar:
        reused = Counter(
            window.canvas
            for p, stores in zip(prepared, finals)
            for plans, store in zip(p.plans, stores)
            for plan in plans if not keep_references and store.has_track(plan)
            for window in plan.windows
        )
        bar.reuse(sum(reused.values()) * len(transformer.conditions), 0, reused)
        for p, span_finals in zip(prepared, finals):
            for segment, segment_plans, references, store in zip(p.span.segments, p.plans, p.references, span_finals):
                for plan in segment_plans:
                    _check(cancel)
                    if not keep_references and store.has_track(plan):
                        continue
                    advance = bar.track(plan.windows[0].canvas, len(transformer.conditions))
                    latents = denoise_track(plan, transformer, references, seed=segment.seed, advance=advance)
                    for window, latent in zip(plan.windows, latents):
                        store.put(window.index, latent)
                    if not keep_references:
                        store.commit_track(plan)
                        for window in plan.windows:
                            references.delete(window.index)


def compose_spans(
    prepared: Sequence[PreparedSpan],
    finals: Sequence[tuple[LatentStore, ...]],
    decoder,
    render: LtxRender,
    *,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    with progress.bar("compose", progress.frames, marks=timing_marks(decoder)) as bar:
        for p, span_finals in zip(prepared, finals):
            compose_span(
                p.span,
                p.plans,
                decoder,
                span_finals,
                frame_h=render.frame_h,
                batch_size=render.batch_size,
                budget=render.budget,
                device=render.device,
                bar=bar,
                cancel=cancel,
                super_res=render.super_res,
                projector=render.vr_projector,
            )


class _SpeedTestMosaics:
    """A detector that finds the same two mosaics on every frame: a small one and one big
    enough for the 768 px canvas (when the run allows it)."""

    def __call__(self, batch: torch.Tensor, target_hw: tuple[int, int]) -> Detections:
        frame_h, frame_w = target_hw
        side = min(frame_h, frame_w)
        small, big = side // 12, side * 9 // 10
        boxes = np.array([[0, 0, small, small], [frame_w - big, frame_h - big, frame_w, frame_h]], dtype=np.float32)
        masks = torch.zeros((len(boxes), frame_h // 4, frame_w // 4), dtype=torch.bool, device=batch.device)
        for mask, (x0, y0, x1, y1) in zip(masks, (boxes // 4).astype(int)):
            mask[y0:y1, x0:x1] = True
        return Detections([boxes] * len(batch), [masks] * len(batch))


class _DiscardedFrames:
    def write(self, frame: torch.Tensor, pts: int) -> None:
        pass

    def finish(self) -> None:
        pass

    def close(self) -> None:
        pass


def _first_frames(frames: FrameSource, count: int) -> FrameSource:
    def first() -> FrameBatches:
        left = count
        with closing(frames()) as batches:
            for batch, pts in batches:
                yield batch[:left], pts[:left]
                left -= len(pts)
                if left <= 0:
                    return

    return first


def _profile_key(render: LtxRender) -> str:
    return speed.profile_key(torch.cuda.get_device_name(render.device), render.files.transformer_name, render.budget_name)


def restore_spans(
    spans: Sequence[LtxSpan], render: LtxRender, *, work_dir: Path, resume_dir: Path, progress: Progress,
    cancel: threading.Event,
) -> None:
    """Restore every segment of ``spans`` and write each span's frames to a writer of its own
    (closed once the span is written); plans and latents are kept in ``resume_dir`` to resume.
    Without a speed profile for this GPU and model, a speed test runs first in a temporary
    directory under ``work_dir``: the first frames of the first span with two made-up mosaics,
    restored and thrown away, so the run's time left is known once its scan has planned it."""
    if not speed.load_rates(render.speed_profile, _profile_key(render)):
        logger.info("LTX speed test: no speed profile for this GPU and model yet")
        speed_test = LtxSpan(
            _first_frames(spans[0].frames, SPEED_TEST_FRAMES),
            (LtxSegment(*WHOLE_VIDEO_PTS, 0),),
            _DiscardedFrames,
        )
        with scratch_folder(work_dir, ".ltx-") as temp:
            _restore_spans(
                [speed_test],
                replace(render, detector=_SpeedTestMosaics()),
                temp,
                progress=progress.speed_test(),
                cancel=cancel,
            )
    resume_dir.mkdir(parents=True, exist_ok=True)
    _restore_spans(spans, render, resume_dir, progress=progress, cancel=cancel)


def _restore_spans(
    spans: Sequence[LtxSpan], render: LtxRender, directory: Path, *, progress: Progress, cancel: threading.Event
) -> None:
    profile_key = _profile_key(render)
    progress.expect(speed.load_rates(render.speed_profile, profile_key))
    torch.cuda.reset_peak_memory_stats(render.device)
    blocks = None
    resuming = any(directory.glob("*.plans.npz"))
    with ModelRuntime(render.files, render.device, cancel) as runtime:
        prepared = prepare_spans(spans, render, directory, progress=progress, cancel=cancel, runtime=runtime)
        finals = final_stores(prepared, directory)
        windows = sum(p.windows for p in prepared)
        pending_tracks = any(
            not store.has_track(plan)
            for p, stores in zip(prepared, finals) for plans, store in zip(p.plans, stores) for plan in plans
        )
        if pending_tracks:
            progress.begin("denoise")
            transformer = runtime.load_transformer()
            blocks = (transformer.blocks.resident, transformer.blocks.count)
            try:
                denoise_spans(prepared, finals, transformer, keep_references=False, progress=progress, cancel=cancel)
            finally:
                del transformer
                runtime.unload_transformer()
        elif windows:
            progress.skip_saved("denoise")
        progress.begin("compose")
        decoder = runtime.load_decoder() if windows else None
        try:
            compose_spans(prepared, finals, decoder, render, progress=progress, cancel=cancel)
        finally:
            del decoder
            runtime.unload_decoder()
        peak_vram_bytes = runtime.peak_vram_bytes()
    for line in progress.summary(profile_key, peak_vram_bytes, blocks):
        logger.info(line)
    if windows and not resuming:
        speed.save_rates(render.speed_profile, profile_key, progress.learned_rates())


def restore_video(
    frames: FrameSource,
    open_writer: Callable[[], FrameWriter],
    *,
    detector,
    files: LtxModelFiles,
    frame_h: int,
    frame_w: int,
    stereo: StereoLayout | None,
    batch_size: int,
    large_canvas: bool,
    window_hop: int,
    seed: int,
    device: torch.device,
    work_dir: Path,
    speed_profile: Path,
    progress: Progress,
    resume_dir: Path,
    cancel: threading.Event,
    max_gap: int,
    min_duration: int,
    super_res: LtxSuperRes | None,
    vr_projector: RegionProjector | None,
    frame_numbers: FrameNumbers,
) -> None:
    """Restore every mosaic of the video ``frames`` yields as one segment. ``large_canvas``
    is ``whole_video_large_canvas``'s decision; the decode tiles follow the free VRAM."""
    span = LtxSpan(frames, (LtxSegment(*WHOLE_VIDEO_PTS, seed),), open_writer)
    restore_spans(
        [span],
        LtxRender(
            detector=detector,
            files=files,
            frame_h=frame_h,
            frame_w=frame_w,
            stereo=stereo,
            batch_size=batch_size,
            large_canvas=large_canvas,
            window_hop=window_hop,
            budget=free_vram_budget(device),
            budget_name="free VRAM",
            device=device,
            speed_profile=speed_profile,
            max_gap=max_gap,
            min_duration=min_duration,
            super_res=super_res,
            vr_projector=vr_projector,
            frame_numbers=frame_numbers,
        ),
        work_dir=work_dir,
        progress=progress,
        cancel=cancel,
        resume_dir=resume_dir,
    )

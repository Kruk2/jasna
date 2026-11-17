"""Progress of an LTX run: one bar per pass, the run's fraction and seconds left (from the speed
profile once the scan has planned the run), and each pass's timing for the profile."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import torch
from tqdm import tqdm

from jasna.ltx import speed
from jasna.ltx.camera import WINDOW_FRAMES
from jasna.ltx.inference import (
    WorkerDecoder,
    WorkerEncoder,
    WorkerTransformer,
)
from jasna.ltx.windows import WINDOW_HOP
from jasna.progressbar import StageProgress

SPEED_TEST_FRAMES = WINDOW_FRAMES + WINDOW_HOP
"""At least two windows per track at every overlap: the speed profile leaves out each canvas's cold first
window."""


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
    its pace so far. Every pass records its timing for the profile.
    ``writer_adds_frame_work``: the compose pass writes through an upscaler or frame
    generation, per-frame work the profile does not time, so that pass is timed by its frame
    pace once known and its rates are not learned."""

    _bar_label = "LTX"

    def __init__(
        self, frames: int, *, disable: bool, writer_adds_frame_work: bool, report: ProgressReport | None
    ) -> None:
        self.frames = frames
        self._disable = disable
        self._writer_adds_frame_work = writer_adds_frame_work
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
        speed_test = Progress(
            min(self.frames, SPEED_TEST_FRAMES),
            disable=self._disable,
            writer_adds_frame_work=False,
            report=speed_test_report,
        )
        speed_test._bar_label = "LTX speed test"
        return speed_test

    def plan(self, work: speed.Work) -> None:
        self._work = work
        self._estimates = speed.pass_estimates(self._rates, work) if work.windows else None

    def learned_rates(self) -> dict[str, float]:
        timings = {
            stage: timing for stage, timing in self.timings.items() if not self._paced_by_frames(stage)
        }
        return speed.learned_rates(timings, self._work, self._rates)

    def _paced_by_frames(self, stage: LtxStage) -> bool:
        return stage == "compose" and self._writer_adds_frame_work

    def summary(
        self, profile_key: str, peak_vram_bytes: tuple[int, int | None], blocks: tuple[int, int] | None
    ) -> list[str]:
        return speed.run_summary(profile_key, self.timings, self._work, peak_vram_bytes, blocks)

    def bar(
        self, stage: LtxStage, total: int, unit: str = "frame", marks: Callable[[], TimingMark] = _cuda_mark
    ) -> StageBar:
        """``marks`` times the pass's windows and steps (``timing_marks``)."""
        tqdm_bar = tqdm(
            total=total,
            desc=f"{self._bar_label} {_STAGE_NUMBER[stage]}/4 {stage}",
            unit=unit,
            dynamic_ncols=True,
            disable=self._disable,
        )
        return StageBar(self, stage, tqdm_bar, marks)

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

    def _profile_stage_left(
        self, stage: LtxStage, paced_left: float | None, profiled_done: float, pace_ratio: float
    ) -> float | None:
        """The pass's seconds left when the profile covers the run: its frame pace once known
        for a pass the profile does not time, else the profile's work left at the measured pace."""
        if self._estimates is None or stage not in self._estimates:
            return None
        if self._paced_by_frames(stage) and paced_left is not None:
            return paced_left
        return max(0.0, self._estimates[stage][1] - profiled_done) * pace_ratio

    def _advanced(
        self, stage: LtxStage, done: int, total: int, paced_left: float | None, profiled_done: float, pace_ratio: float
    ) -> float | None:
        """Report the run's progress; ``paced_left`` is the pass's seconds left at its pace
        so far, ``profiled_done`` the profile's seconds of its work done and ``pace_ratio``
        its measured seconds per profile second. Returns the run's seconds left the
        profile gives, if any."""
        stage_fraction = min(1.0, done / total) if total else 1.0
        profile_stage_left = self._profile_stage_left(stage, paced_left, profiled_done, pace_ratio)
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


class StageBar:
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

    def __enter__(self) -> StageBar:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

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
temporary directory between passes, so memory does not grow with length.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
from tqdm import tqdm

from jasna.ltx import speed
from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES
from jasna.ltx.compose import Candidate, composite_frame, crop_to_canvas
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.plan import LARGE_CANVAS, Region, TrackPlan, Window, crossfade_weights, feather_pixels, plan_video
from jasna.ltx.regions import regions_per_frame
from jasna.ltx.sampler import LtxTransformer
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder
from jasna.progressbar import StageProgress
from jasna.tracking.scene_detector import SceneCutDetector

logger = logging.getLogger(__name__)

FrameBatches = Iterator[tuple[torch.Tensor, list[int]]]
FrameSource = Callable[[], FrameBatches]
DecodeBudget = Callable[[int], int]
LARGE_CANVAS_MIN_FREE_BYTES = 10 << 30
LARGE_CANVAS_MIN_TOTAL_BYTES = 15 << 30
SEGMENT_DECODE_BUDGET_BYTES = {PLAN_CANVAS: 6 << 30, LARGE_CANVAS: 8 << 30}
WHOLE_VIDEO_PTS = (-(1 << 63), (1 << 63) - 1)


class Cancelled(Exception):
    pass


class FrameWriter(Protocol):
    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None: ...

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


class Progress:
    """One console bar per pass; ``frames`` sizes the frame-based passes. ``report`` gets
    ``(fraction of the whole run, seconds left, the pass's own progress)``. When the speed
    profile's rates (``expect``) cover what the scan planned, seconds left come from them
    from the end of the scan on; else they are 0 until the denoise pass has timed a step.
    A pass's own seconds left come from its pace so far, else from the profile. Every
    pass records its timing for the profile."""

    def __init__(self, frames: int, *, disable: bool, report: ProgressReport | None) -> None:
        self.frames = frames
        self._disable = disable
        self._report = report
        self._started = time.monotonic()
        self._pass_end = self._started
        self._done_share = 0.0
        self._timed = False
        self._rates: dict[str, float] = {}
        self._work: speed.Work | None = None
        self._estimates: dict[str, tuple[float, float]] | None = None
        self.timings: dict[str, speed.PassTiming] = {}

    def expect(self, rates: dict[str, float]) -> None:
        self._rates = rates

    def plan(self, work: speed.Work) -> None:
        self._work = work
        self._estimates = speed.pass_estimates(self._rates, work) if work.windows else None

    def learned_rates(self) -> dict[str, float]:
        return speed.learned_rates(self.timings, self._work, self._rates)

    def summary(self, profile_key: str, peak_vram_bytes: int, blocks: tuple[int, int] | None) -> list[str]:
        return speed.run_summary(profile_key, self.timings, self._work, peak_vram_bytes, blocks)

    def bar(self, stage: LtxStage, total: int, unit: str = "frame") -> _StageBar:
        tqdm_bar = tqdm(
            total=total,
            desc=f"LTX {_STAGE_NUMBER[stage]}/4 {stage}",
            unit=unit,
            dynamic_ncols=True,
            disable=self._disable,
        )
        return _StageBar(self, stage, tqdm_bar)

    def begin(self, stage: LtxStage) -> None:
        """Report ``stage`` as started before its model loads, so loading shows as that pass."""
        self._advanced(stage, 0, 1, None)

    def skip(self, *stages: LtxStage) -> None:
        """Count ``stages`` as done, for a run that reuses their results."""
        for stage in stages:
            self._done_share += _STAGE_SHARE[stage]

    def _profile_stage_left(self, stage: LtxStage, stage_fraction: float) -> float | None:
        if self._estimates is None or stage not in self._estimates:
            return None
        return self._estimates[stage][1] * (1.0 - stage_fraction)

    def _advanced(self, stage: LtxStage, done: int, total: int, paced_left: float | None) -> float | None:
        """Report the run's progress; ``paced_left`` is the pass's seconds left at its pace
        so far. Returns the run's seconds left the profile gives, if any."""
        stage_fraction = min(1.0, done / total) if total else 1.0
        profile_stage_left = self._profile_stage_left(stage, stage_fraction)
        left = None
        if profile_stage_left is not None:
            later = speed.TIMED_PASSES[speed.TIMED_PASSES.index(stage) + 1 :]
            left = profile_stage_left + sum(sum(self._estimates[s]) for s in later)
        if self._report is not None:
            stage_left = paced_left if paced_left is not None else profile_stage_left or 0.0
            step = StageProgress(stage, _STAGE_NUMBER[stage], len(_STAGE_NUMBER), stage_fraction, stage_left)
            elapsed = time.monotonic() - self._started
            if left is not None:
                self._report(elapsed / (elapsed + left), left, step)
            else:
                fraction = self._done_share + _STAGE_SHARE[stage] * stage_fraction
                self._timed = self._timed or (stage == "denoise" and done > 0)
                self._report(fraction, elapsed * (1.0 - fraction) / fraction if self._timed else 0.0, step)
        return left

    def _finished(self, stage: LtxStage, timing: speed.PassTiming) -> None:
        self._done_share += _STAGE_SHARE[stage]
        now = time.monotonic()
        timing.seconds = now - self._pass_end
        self._pass_end = now
        self.timings[stage] = timing


class _StageBar:
    def __init__(self, progress: Progress, stage: LtxStage, bar: tqdm) -> None:
        self._progress = progress
        self._stage = stage
        self._bar = bar
        self._done = 0
        self._warm_from: tuple[float, int] | None = None
        self._windows: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._tracks: list[tuple[int, int, list[tuple[int, torch.cuda.Event]]]] = []
        progress._advanced(stage, 0, bar.total, None)

    def update(self, n: int) -> None:
        self._bar.update(n)
        self._done += n
        now = time.monotonic()
        if self._warm_from is None:
            self._warm_from = (now, self._done)
        started, warm_done = self._warm_from
        paced_left = (
            (now - started) / (self._done - warm_done) * (self._bar.total - self._done)
            if self._done > warm_done
            else None
        )
        left = self._progress._advanced(self._stage, self._done, self._bar.total, paced_left)
        if left is not None:
            self._bar.set_postfix_str(f"run {tqdm.format_interval(left)} left", refresh=False)

    @contextmanager
    def window(self, canvas: int) -> Iterator[None]:
        """Time the GPU work of one window with stream events, read when the pass closes."""
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._windows.append((canvas, start, end))

    def track(self, canvas: int, steps: int) -> Callable[[int], None]:
        """``advance`` for one track's denoise: counts its window-steps and marks the stream
        after each, so the pass can time the track without its first step."""
        marks: list[tuple[int, torch.cuda.Event]] = []
        self._tracks.append((canvas, steps, marks))

        def advance(n: int) -> None:
            self.update(n)
            mark = torch.cuda.Event(enable_timing=True)
            mark.record()
            marks.append((n, mark))

        return advance

    def close(self) -> None:
        self._bar.close()
        timing = speed.PassTiming(units=self._done)
        if self._warm_from is not None:
            started, done = self._warm_from
            timing.warm_seconds = time.monotonic() - started
            timing.warm_frames = self._done - done
        for canvas, start, end in self._windows:
            end.synchronize()
            timing.windows.append((canvas, start.elapsed_time(end) / 1000))
        for canvas, steps, marks in self._tracks:
            if len(marks) > 1:
                marks[-1][1].synchronize()
                for (_, previous), (n, mark) in zip(marks, marks[1:]):
                    timing.steps.append((canvas, previous.elapsed_time(mark) / 1000 / n))
                seconds = marks[0][1].elapsed_time(marks[-1][1]) / 1000
                timing.windows.append((canvas, seconds / sum(n for n, _ in marks[1:]) * steps))
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
        torch.save(latent.to("cpu", torch.bfloat16).contiguous(), self._path(index))

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
    seek_ts: float | None,
    end_pts: int | None,
) -> FrameSource:
    """Decoded batches of ``path`` from the keyframe at ``seek_ts`` (the start when None),
    stopping before ``end_pts`` (the end when None)."""
    from jasna.media.video_decoder import VideoReader

    def frames() -> FrameBatches:
        with VideoReader(str(path), batch_size, device, metadata, frame_stride=frame_stride) as reader:
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
    large_canvas: bool,
    bar: _StageBar,
    cancel: threading.Event,
) -> list[list[TrackPlan]]:
    """The track plans of each segment, frame indices counted from the segment's first frame."""
    regions: list[list[list[Region]]] = [[] for _ in segments]
    cuts: list[set[int]] = [set() for _ in segments]
    scenes = [SceneCutDetector() for _ in segments]
    for owner, batch, _pts in segment_batches(frames(), segments, batch_size):
        _check(cancel)
        if owner is not None:
            cuts[owner].update(len(regions[owner]) + offset for offset in scenes[owner].find_cuts(batch))
            regions[owner].extend(regions_per_frame(detector(batch, target_hw=(frame_h, frame_w)), frame_h, frame_w))
        bar.update(len(batch))
    plans = []
    for segment_regions, segment_cuts in zip(regions, cuts):
        segment_plans = plan_video(segment_regions, segment_cuts, frame_w=frame_w, frame_h=frame_h, large_canvas=large_canvas)
        logger.info(
            "LTX plan: %d frames, %d scene cuts, %d tracks, %d windows",
            len(segment_regions),
            len(segment_cuts),
            len(segment_plans),
            sum(len(plan.windows) for plan in segment_plans),
        )
        plans.append(segment_plans)
    return plans


def _vae_pixels(canvases: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    stacked = torch.stack(canvases + [canvases[-1]] * (WINDOW_FRAMES - len(canvases)), dim=1)
    return stacked.to(device).float().div_(127.5).sub_(1.0).to(torch.bfloat16).unsqueeze(0)


class _SegmentEncoder:
    def __init__(
        self, plans: list[TrackPlan], encoder, store: LatentStore, device: torch.device, bar: _StageBar
    ) -> None:
        self._pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
        self._upcoming = next(self._pending, None)
        self._active: dict[int, tuple[Window, list[torch.Tensor]]] = {}
        self._encoder = encoder
        self._store = store
        self._device = device
        self._bar = bar
        self._frame_idx = 0

    def frame(self, frame: torch.Tensor) -> None:
        while self._upcoming is not None and self._upcoming.start == self._frame_idx:
            self._active[self._upcoming.index] = (self._upcoming, [])
            self._upcoming = next(self._pending, None)
        for index, (window, canvases) in list(self._active.items()):
            canvases.append(crop_to_canvas(frame, window.crops[self._frame_idx - window.start]).cpu())
            if len(canvases) == window.real_frames:
                with self._bar.window(window.canvas), torch.inference_mode():
                    self._store.put(index, self._encoder(_vae_pixels(canvases, self._device)))
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
) -> None:
    """VAE-encode every planned window's reference crops into its segment's store."""
    encoders = [_SegmentEncoder(p, encoder, store, device, bar) for p, store in zip(plans, stores)]
    for owner, batch, _pts in segment_batches(frames(), segments, batch_size):
        _check(cancel)
        bar.update(len(batch))
        if owner is not None:
            for frame in batch:
                encoders[owner].frame(frame)
    for segment_encoder in encoders:
        segment_encoder.finish()


def denoise_track(
    plan: TrackPlan, transformer: LtxTransformer, references: LatentStore, *, seed: int, advance: Callable[[int], None]
) -> list[torch.Tensor]:
    """The final latents of one track's windows, in window order."""
    return transformer.denoise_chain(
        [references.get(w.index) for w in plan.windows], [seed + w.index for w in plan.windows], advance
    )


def free_vram_budget(device: torch.device) -> DecodeBudget:
    return lambda _canvas: torch.cuda.mem_get_info(device)[0]


def segment_decode_budget(canvas: int) -> int:
    """A fixed decode budget, so a segment's tile layout (and pixels) never depend on the
    VRAM free at that moment."""
    return SEGMENT_DECODE_BUDGET_BYTES[canvas]


def _decode_window(
    decoder, latent: torch.Tensor, seed: int, canvas: int, budget: DecodeBudget, device: torch.device
) -> torch.Tensor:
    """Decoded window as uint8 ``[F, 3, S, S]`` on ``device``."""
    torch.cuda.empty_cache()
    free = budget(canvas)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        video = decode_latent(decoder, latent.to(device), free_bytes=free, generator=generator)
        video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)[0].float()
        return video.mul_(255.0).round_().to(torch.uint8).permute(1, 0, 2, 3).contiguous()


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
    ) -> None:
        self._weights: dict[int, torch.Tensor] = {}
        self._tracks: dict[int, TrackPlan] = {}
        for plan in plans:
            for window, weight in zip(plan.windows, crossfade_weights(plan.windows)):
                self._weights[window.index] = weight
                self._tracks[window.index] = plan
        self._pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
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

    def frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame_idx = self._frame_idx
        while self._upcoming is not None and self._upcoming.start == frame_idx:
            self._covering.append(self._upcoming)
            self._upcoming = next(self._pending, None)
        for window in [w for w in self._covering if w.stop <= frame_idx]:
            self._covering.remove(window)
            self._decoded.pop(window.index, None)
        candidates = []
        for window in self._covering:
            if window.index not in self._decoded:
                with self._bar.window(window.canvas):
                    self._decoded[window.index] = _decode_window(
                        self._decoder,
                        self._finals.get(window.index),
                        self._seed + window.index,
                        window.canvas,
                        self._budget,
                        self._device,
                    )
            local = frame_idx - window.start
            track = self._tracks[window.index]
            candidates.append(
                Candidate(
                    canvas=self._decoded[window.index][local],
                    crop=window.crops[local],
                    weight=float(self._weights[window.index][local]),
                    track=track.track_id,
                    polygons=track.polygons[frame_idx - track.start],
                    presence=track.presence[frame_idx - track.start],
                )
            )
        self._frame_idx += 1
        return composite_frame(frame, candidates, feather=self._feather)


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
) -> None:
    """Write every frame of ``span``: segment frames composited, the rest unchanged."""
    feather = feather_pixels(frame_h)
    composers = [
        _SegmentComposer(
            p, decoder, store, seed=segment.seed, feather=feather, budget=budget, device=device, bar=bar
        )
        for p, store, segment in zip(plans, finals, span.segments)
    ]
    writer = span.open_writer()
    try:
        for owner, batch, pts in segment_batches(span.frames(), span.segments, batch_size):
            _check(cancel)
            bar.update(len(batch))
            for frame, frame_pts in zip(batch, pts):
                if owner is None:
                    writer.write(frame, frame_pts, apply_lut=False)
                else:
                    writer.write(composers[owner].frame(frame), frame_pts)
    finally:
        writer.close()


def large_canvas_fits(requested: bool, free_bytes: int) -> bool:
    """768 px windows need ~7.5 GiB of VAE-encoder activations alone; below 10 GiB free
    every window runs at 512 px instead."""
    if requested and free_bytes < LARGE_CANVAS_MIN_FREE_BYTES:
        logger.warning("LTX: %.1f GiB free VRAM, restoring large mosaics at 512 px", free_bytes / 2**30)
        return False
    return requested


def segment_large_canvas(requested: bool, total_bytes: int) -> bool:
    """The 768 px decision for segments, from the GPU's size so it is the same every run."""
    if requested and total_bytes < LARGE_CANVAS_MIN_TOTAL_BYTES:
        logger.warning("LTX: %.1f GiB GPU, restoring large mosaics at 512 px", total_bytes / 2**30)
        return False
    return requested


@dataclass(frozen=True)
class LtxRender:
    """What every pass needs besides the spans: the detector, the model files, the frame
    size, detection batch size, the 768 px decision, the decode budget per canvas and its
    name (decode tiling follows the budget, so it keys the speed profile), the GPU and the
    speed profile file (``speed.py``)."""

    detector: Callable
    files: LtxModelFiles
    frame_h: int
    frame_w: int
    batch_size: int
    large_canvas: bool
    budget: DecodeBudget
    budget_name: str
    device: torch.device
    speed_profile: Path


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


def prepare_spans(
    spans: Sequence[LtxSpan], render: LtxRender, directory: Path, *, progress: Progress, cancel: threading.Event
) -> list[PreparedSpan]:
    """Scan and VAE-encode every span; reference latents go to ``directory``."""
    with progress.bar("scan", progress.frames) as bar:
        plans = [
            scan_span(
                span.frames,
                span.segments,
                render.detector,
                batch_size=render.batch_size,
                frame_h=render.frame_h,
                frame_w=render.frame_w,
                large_canvas=render.large_canvas,
                bar=bar,
                cancel=cancel,
            )
            for span in spans
        ]
    prepared = [
        PreparedSpan(
            span,
            tuple(span_plans),
            tuple(LatentStore(directory, f"{s}-{g}.reference") for g in range(len(span.segments))),
        )
        for s, (span, span_plans) in enumerate(zip(spans, plans))
    ]
    windows = Counter(
        w.canvas for p in prepared for track_plans in p.plans for plan in track_plans for w in plan.windows
    )
    progress.plan(speed.Work(progress.frames, render.frame_h * render.frame_w / 1e6, dict(windows)))
    if sum(p.windows for p in prepared):
        progress.begin("encode")
        encoder = load_video_encoder(render.files.vae, render.device)
        with progress.bar("encode", progress.frames) as bar:
            for p in prepared:
                encode_span(
                    p.span.frames,
                    p.span.segments,
                    p.plans,
                    encoder,
                    p.references,
                    batch_size=render.batch_size,
                    device=render.device,
                    bar=bar,
                    cancel=cancel,
                )
        del encoder
        torch.cuda.empty_cache()
    return prepared


def final_stores(prepared: Sequence[PreparedSpan], directory: Path) -> list[tuple[LatentStore, ...]]:
    return [
        tuple(LatentStore(directory, f"{s}-{g}.final") for g in range(len(p.span.segments)))
        for s, p in enumerate(prepared)
    ]


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
    with progress.bar("denoise", len(transformer.conditions) * windows, unit="step") as bar:
        for p, span_finals in zip(prepared, finals):
            for segment, segment_plans, references, store in zip(p.span.segments, p.plans, p.references, span_finals):
                for plan in segment_plans:
                    _check(cancel)
                    advance = bar.track(plan.windows[0].canvas, len(transformer.conditions))
                    latents = denoise_track(plan, transformer, references, seed=segment.seed, advance=advance)
                    for window, latent in zip(plan.windows, latents):
                        store.put(window.index, latent)
                        if not keep_references:
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
    with progress.bar("compose", progress.frames) as bar:
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
            )


def restore_spans(
    spans: Sequence[LtxSpan], render: LtxRender, *, work_dir: Path, progress: Progress, cancel: threading.Event
) -> None:
    """Restore every segment of ``spans`` and write each span's frames to a writer of its own
    (closed once the span is written)."""
    profile_key = speed.profile_key(
        torch.cuda.get_device_name(render.device), render.files.transformer_name, render.budget_name
    )
    progress.expect(speed.load_rates(render.speed_profile, profile_key))
    torch.cuda.reset_peak_memory_stats(render.device)
    blocks = None
    with tempfile.TemporaryDirectory(dir=work_dir, prefix=".ltx-") as temp:
        prepared = prepare_spans(spans, render, Path(temp), progress=progress, cancel=cancel)
        finals = final_stores(prepared, Path(temp))
        windows = sum(p.windows for p in prepared)
        if windows:
            progress.begin("denoise")
            transformer = LtxTransformer(render.files.transformer, render.device)
            blocks = (transformer.blocks.resident, transformer.blocks.count)
            try:
                denoise_spans(prepared, finals, transformer, keep_references=False, progress=progress, cancel=cancel)
            finally:
                transformer.close()
                del transformer
                torch.cuda.empty_cache()
        progress.begin("compose")
        decoder = load_video_decoder(render.files.vae, render.files.tuned_decoder, render.device) if windows else None
        try:
            compose_spans(prepared, finals, decoder, render, progress=progress, cancel=cancel)
        finally:
            del decoder
            torch.cuda.empty_cache()
    for line in progress.summary(profile_key, torch.cuda.max_memory_allocated(render.device), blocks):
        logger.info(line)
    if windows:
        speed.save_rates(render.speed_profile, profile_key, progress.learned_rates())


def restore_video(
    frames: FrameSource,
    writer: FrameWriter,
    *,
    detector,
    files: LtxModelFiles,
    frame_h: int,
    frame_w: int,
    batch_size: int,
    large_canvas: bool,
    seed: int,
    device: torch.device,
    work_dir: Path,
    speed_profile: Path,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    """Restore every mosaic of the video ``frames`` yields as one segment; the 768 px
    decision and the decode tiles follow the VRAM free when they are made."""
    restore_spans(
        [LtxSpan(frames, (LtxSegment(*WHOLE_VIDEO_PTS, seed),), lambda: writer)],
        LtxRender(
            detector=detector,
            files=files,
            frame_h=frame_h,
            frame_w=frame_w,
            batch_size=batch_size,
            large_canvas=large_canvas_fits(large_canvas, torch.cuda.mem_get_info(device)[0]),
            budget=free_vram_budget(device),
            budget_name="free VRAM",
            device=device,
            speed_profile=speed_profile,
        ),
        work_dir=work_dir,
        progress=progress,
        cancel=cancel,
    )

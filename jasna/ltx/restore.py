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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
from tqdm import tqdm

from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES
from jasna.ltx.compose import Candidate, composite_frame, crop_to_canvas
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.plan import LARGE_CANVAS, Region, TrackPlan, Window, crossfade_weights, feather_pixels, plan_video
from jasna.ltx.regions import regions_per_frame
from jasna.ltx.sampler import LtxTransformer
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder
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
ProgressReport = Callable[[LtxStage, float, float], None]
_STAGE_NUMBER = {"scan": 1, "encode": 2, "denoise": 3, "compose": 4}
_STAGE_SHARE = {"scan": 0.02, "encode": 0.08, "denoise": 0.8, "compose": 0.1}


class Progress:
    """One console bar per pass; ``frames`` sizes the frame-based passes. ``report`` gets
    ``(stage, fraction of the whole run, seconds left)``; seconds left is 0 until the
    denoise pass has timed a step."""

    def __init__(self, frames: int, *, disable: bool, report: ProgressReport | None) -> None:
        self.frames = frames
        self._disable = disable
        self._report = report
        self._started = time.monotonic()
        self._done_share = 0.0
        self._timed = False

    def bar(self, stage: LtxStage, total: int, unit: str = "frame") -> _StageBar:
        tqdm_bar = tqdm(
            total=total,
            desc=f"LTX {_STAGE_NUMBER[stage]}/4 {stage}",
            unit=unit,
            dynamic_ncols=True,
            disable=self._disable,
        )
        return _StageBar(self, stage, tqdm_bar)

    def _advanced(self, stage: LtxStage, done: int, total: int) -> None:
        if self._report is None:
            return
        fraction = self._done_share + _STAGE_SHARE[stage] * (min(1.0, done / total) if total else 1.0)
        self._timed = self._timed or (stage == "denoise" and done > 0)
        elapsed = time.monotonic() - self._started
        self._report(stage, fraction, elapsed * (1.0 - fraction) / fraction if self._timed else 0.0)

    def _finished(self, stage: LtxStage) -> None:
        self._done_share += _STAGE_SHARE[stage]


class _StageBar:
    def __init__(self, progress: Progress, stage: LtxStage, bar: tqdm) -> None:
        self._progress = progress
        self._stage = stage
        self._bar = bar
        self._done = 0
        progress._advanced(stage, 0, bar.total)

    def update(self, n: int) -> None:
        self._bar.update(n)
        self._done += n
        self._progress._advanced(self._stage, self._done, self._bar.total)

    def close(self) -> None:
        self._bar.close()
        self._progress._finished(self._stage)

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
    def __init__(self, plans: list[TrackPlan], encoder, store: LatentStore, device: torch.device) -> None:
        self._pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
        self._upcoming = next(self._pending, None)
        self._active: dict[int, tuple[Window, list[torch.Tensor]]] = {}
        self._encoder = encoder
        self._store = store
        self._device = device
        self._frame_idx = 0

    def frame(self, frame: torch.Tensor) -> None:
        while self._upcoming is not None and self._upcoming.start == self._frame_idx:
            self._active[self._upcoming.index] = (self._upcoming, [])
            self._upcoming = next(self._pending, None)
        for index, (window, canvases) in list(self._active.items()):
            canvases.append(crop_to_canvas(frame, window.crops[self._frame_idx - window.start]).cpu())
            if len(canvases) == window.real_frames:
                with torch.inference_mode():
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
    encoders = [_SegmentEncoder(p, encoder, store, device) for p, store in zip(plans, stores)]
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
                    polygons=track.polygons[frame_idx - track.start],
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
        _SegmentComposer(p, decoder, store, seed=segment.seed, feather=feather, budget=budget, device=device)
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


def restore_spans(
    spans: Sequence[LtxSpan],
    *,
    detector,
    files: LtxModelFiles,
    frame_h: int,
    frame_w: int,
    batch_size: int,
    large_canvas: bool,
    budget: DecodeBudget,
    device: torch.device,
    work_dir: Path,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    """Restore every segment of ``spans`` and write each span's frames to a writer of its own
    (closed once the span is written). ``large_canvas`` allows 768 px windows for large mosaics;
    ``budget`` gives the decode memory budget per canvas size."""
    with progress.bar("scan", progress.frames) as bar:
        plans = [
            scan_span(
                span.frames,
                span.segments,
                detector,
                batch_size=batch_size,
                frame_h=frame_h,
                frame_w=frame_w,
                large_canvas=large_canvas,
                bar=bar,
                cancel=cancel,
            )
            for span in spans
        ]
    windows = sum(len(plan.windows) for span_plans in plans for segment_plans in span_plans for plan in segment_plans)
    with tempfile.TemporaryDirectory(dir=work_dir, prefix=".ltx-") as temp:
        stores = [
            [
                (LatentStore(Path(temp), f"{s}-{g}.reference"), LatentStore(Path(temp), f"{s}-{g}.final"))
                for g in range(len(span.segments))
            ]
            for s, span in enumerate(spans)
        ]
        if windows:
            encoder = load_video_encoder(files.vae, device)
            with progress.bar("encode", progress.frames) as bar:
                for span, span_plans, span_stores in zip(spans, plans, stores):
                    encode_span(
                        span.frames,
                        span.segments,
                        span_plans,
                        encoder,
                        [references for references, _ in span_stores],
                        batch_size=batch_size,
                        device=device,
                        bar=bar,
                        cancel=cancel,
                    )
            del encoder
            torch.cuda.empty_cache()
            transformer = LtxTransformer(files.transformer, device)
            try:
                with progress.bar("denoise", len(transformer.conditions) * windows, unit="step") as bar:
                    for span, span_plans, span_stores in zip(spans, plans, stores):
                        for segment, segment_plans, (references, finals) in zip(span.segments, span_plans, span_stores):
                            for plan in segment_plans:
                                _check(cancel)
                                latents = denoise_track(plan, transformer, references, seed=segment.seed, advance=bar.update)
                                for window, latent in zip(plan.windows, latents):
                                    finals.put(window.index, latent)
                                    references.delete(window.index)
            finally:
                transformer.close()
                del transformer
                torch.cuda.empty_cache()
        decoder = load_video_decoder(files.vae, files.tuned_decoder, device) if windows else None
        with progress.bar("compose", progress.frames) as bar:
            for span, span_plans, span_stores in zip(spans, plans, stores):
                compose_span(
                    span,
                    span_plans,
                    decoder,
                    [finals for _, finals in span_stores],
                    frame_h=frame_h,
                    batch_size=batch_size,
                    budget=budget,
                    device=device,
                    bar=bar,
                    cancel=cancel,
                )


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
    progress: Progress,
    cancel: threading.Event,
) -> None:
    """Restore every mosaic of the video ``frames`` yields as one segment; the 768 px
    decision and the decode tiles follow the VRAM free when they are made."""
    restore_spans(
        [LtxSpan(frames, (LtxSegment(*WHOLE_VIDEO_PTS, seed),), lambda: writer)],
        detector=detector,
        files=files,
        frame_h=frame_h,
        frame_w=frame_w,
        batch_size=batch_size,
        large_canvas=large_canvas_fits(large_canvas, torch.cuda.mem_get_info(device)[0]),
        budget=free_vram_budget(device),
        device=device,
        work_dir=work_dir,
        progress=progress,
        cancel=cancel,
    )

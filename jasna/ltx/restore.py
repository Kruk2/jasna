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
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from jasna.accelerator import release_blas_workspaces
from jasna.ltx import speed
from jasna.ltx.compose_pass import compose_span
from jasna.ltx.encode_pass import encode_span
from jasna.ltx.inference import ModelRuntime
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.plan import TrackPlan, plan_summary, plan_video, plan_window_lines
from jasna.ltx.resume_files import plans_to_arrays
from jasna.ltx.tracks import Region
from jasna.ltx.progress import SPEED_TEST_FRAMES, Progress, StageBar, timing_marks
from jasna.ltx.regions import regions_per_frame
from jasna.ltx.resume_files import (
    LatentStore,
    final_stores,
    plan_path,
    saved_frames,
    saved_plans,
    span_stores,
)
from jasna.ltx.sampler import LtxTransformer
from jasna.ltx.spans import (
    WHOLE_VIDEO_PTS,
    FrameBatches,
    FrameNumbers,
    FrameSource,
    FrameWriter,
    LtxSegment,
    LtxSpan,
    check_cancel,
    segment_batches,
)
from jasna.ltx.super_res import LtxSuperRes
from jasna.ltx.vram_budget import DecodeBudget, free_vram_budget
from jasna.mosaic.detections import Detections
from jasna.os_utils import atomic_write, scratch_folder
from jasna.tracking.scene_detector import SceneCutDetector
from jasna.vr180 import StereoLayout
from jasna.vr_projection import RegionProjector

logger = logging.getLogger(__name__)


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
    bar: StageBar,
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
        check_cancel(cancel)
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
    runtime: ModelRuntime,
) -> list[PreparedSpan]:
    """Scan and VAE-encode every span; reference latents go to ``directory``. ``runtime`` denoises
    next, so its worker starts and opens the transformer file during the scan; it holds no model on
    the GPU before the encoder loads."""
    worker_ready = None
    if not all(plan_path(span, directory).is_file() for span in spans):
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
        if worker_ready is not None:
            worker_ready.result()
        encoder = runtime.load_encoder()
        try:
            with progress.bar("encode", progress.frames, marks=timing_marks(encoder)) as bar:
                encoded = [p.windows > 0 or not done.windows for p, done in zip(pending, prepared)]
                saved_frames = sum(
                    saved_frames(p.span, directory) for p, encode in zip(prepared, encoded) if not encode
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
    cached = [saved_plans(span, directory) for span in spans]
    if all(item is not None for item in cached):
        progress.skip_saved("scan")
        plans = [item.plans for item in cached]
    else:
        with progress.bar("scan", progress.frames) as bar:
            for span, saved in zip(spans, cached):
                check_cancel(cancel)
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
                check_cancel(cancel)
                with atomic_write(plan_path(span, directory), "wb") as stream:
                    np.savez(stream, frames=np.array(count), **plans_to_arrays(span_plans))
                plans.append(span_plans)
    prepared = [
        PreparedSpan(span, tuple(span_plans), span_stores(span, directory, "reference"))
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
                    check_cancel(cancel)
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
    cancel: threading.Event, runtime: ModelRuntime,
) -> None:
    """Restore every segment of ``spans`` with the models of ``runtime`` (the caller closes it)
    and write each span's frames to a writer of its own (closed once the span is written);
    plans and latents are kept in ``resume_dir`` to resume.
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
                runtime=runtime,
            )
    resume_dir.mkdir(parents=True, exist_ok=True)
    _restore_spans(spans, render, resume_dir, progress=progress, cancel=cancel, runtime=runtime)


def _restore_spans(
    spans: Sequence[LtxSpan], render: LtxRender, directory: Path, *, progress: Progress, cancel: threading.Event,
    runtime: ModelRuntime,
) -> None:
    profile_key = _profile_key(render)
    progress.expect(speed.load_rates(render.speed_profile, profile_key))
    torch.cuda.reset_peak_memory_stats(render.device)
    blocks = None
    resuming = any(directory.glob("*.plans.npz"))
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
        release_blas_workspaces()
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
    runtime: ModelRuntime,
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
        runtime=runtime,
    )

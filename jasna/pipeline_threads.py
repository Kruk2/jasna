from __future__ import annotations

import gc
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Protocol

import torch

from jasna.accelerator import empty_cache, ipc_collect
from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue
from jasna.media.probe import VideoMetadata
from jasna.media.video_decoder import VideoReader
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryLoopStats, _SENTINEL
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.pipeline_timing import LoopTimer
from jasna.progressbar import Progressbar
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.session_config import DecodeBackendName
from jasna.tracking.clip_tracker import ClipTracker
from jasna.tracking.scene_detector import SceneCutDetector
from jasna.vr180 import StereoLayout
from jasna.vram_offloader import VramOffloader

log = logging.getLogger(__name__)


class PassFrameWriter(Protocol):
    def write(self, frame: torch.Tensor, pts: int) -> None: ...
    def after_write(self, frames_written: int, stop_event: threading.Event) -> None: ...


@dataclass(frozen=True)
class PassContext:
    """What the worker threads of one restoration pass share.

    ``stop_event`` is pass-owned: set on cancel or on the first error in ``error_holder``.
    """

    device: torch.device
    restoration_pipeline: RestorationPipeline
    input_video: str
    decode_backend: DecodeBackendName
    batch_size: int
    metadata: VideoMetadata
    seek_ts: float | None
    frame_stride: int
    blend_buffer: BlendBuffer
    crop_buffers: dict[int, CropBuffer]
    clip_queue: FrameQueue
    secondary_queue: FrameQueue
    encode_queue: FrameQueue
    metadata_queue: Queue
    primary_idle_event: threading.Event
    stop_event: threading.Event
    error_holder: list[BaseException]
    debug_memory: PipelineDebugMemoryLogger

    def open_reader(self) -> VideoReader:
        return VideoReader(
            self.input_video,
            batch_size=self.batch_size,
            device=self.device,
            metadata=self.metadata,
            frame_stride=self.frame_stride,
            decode_backend=self.decode_backend,
        )


def effect_runs(
    pts_list: list[int], effect_ranges: tuple[tuple[int, int], ...] | None
) -> list[tuple[int, int, bool]]:
    """Split a batch into maximal ``(start, end, inside)`` runs of frames that are all inside
    or all outside the half-open ``effect_ranges`` (``None`` means everything is inside)."""
    runs = []
    start = 0
    inside_flags = (
        effect_ranges is None or any(low <= int(pts) < high for low, high in effect_ranges) for pts in pts_list
    )
    for inside, group in itertools.groupby(inside_flags):
        end = start + sum(1 for _ in group)
        runs.append((start, end, inside))
        start = end
    return runs


def decode_detect_loop(
    ctx: PassContext,
    pipeline,
    *,
    end_pts: int | None,
    effect_ranges: tuple[tuple[int, int], ...] | None,
    output_frame_count: int | None,
    output_fps: float | None,
    progress: Progressbar | None,
) -> None:
    """Decode, detect and track; ``pipeline`` supplies the detector, tracker settings and VR layout."""
    timer = LoopTimer("decode-detect")
    try:
        torch.cuda.set_device(ctx.device)
        with ctx.open_reader() as reader, torch.inference_mode():
            if progress is not None:
                progress.init()
            metadata = ctx.metadata
            log.info(
                "Processing %s: %d frames @ %s fps, %dx%d",
                ctx.input_video,
                metadata.num_frames if output_frame_count is None else output_frame_count,
                metadata.video_fps if output_fps is None else output_fps,
                metadata.video_width,
                metadata.video_height,
            )
            try:
                _detect_batches(
                    ctx, pipeline, reader, timer, end_pts=end_pts, effect_ranges=effect_ranges, progress=progress
                )
            except Exception:
                if progress is not None:
                    progress.error = True
                raise
    except BaseException as e:
        if not ctx.stop_event.is_set():
            log.exception("[decode] thread crashed")
            ctx.error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[decode] thread exiting")
        ctx.clip_queue.put(_SENTINEL)
        ctx.metadata_queue.put(_SENTINEL)


def _detect_batches(
    ctx: PassContext,
    pipeline,
    reader: VideoReader,
    timer: LoopTimer,
    *,
    end_pts: int | None,
    effect_ranges: tuple[tuple[int, int], ...] | None,
    progress: Progressbar | None,
) -> None:
    tracker = ClipTracker(
        max_clip_size=pipeline.max_clip_size,
        temporal_overlap=pipeline.temporal_overlap,
        max_detection_gap=pipeline.max_detection_gap,
    )
    scene_detector = SceneCutDetector() if pipeline.scene_detection else None
    discard_margin = pipeline.temporal_overlap
    blend_frames = pipeline.temporal_overlap // 3
    target_hw = (int(ctx.metadata.video_height), int(ctx.metadata.video_width))
    frame_idx = 0 if ctx.seek_ts is None else _estimate_start_frame(ctx.metadata, ctx.seek_ts)
    frame_shape = target_hw
    effect_active = effect_ranges is None
    stop_after_batch = False

    def _finalize_tracker() -> None:
        nonlocal effect_active
        if not effect_active:
            return
        finalize_processing(
            tracker=tracker,
            blend_buffer=ctx.blend_buffer,
            crop_buffers=ctx.crop_buffers,
            clip_queue=ctx.clip_queue,
            frame_shape=frame_shape,
            discard_margin=discard_margin,
            blend_frames=blend_frames,
            min_detection_duration=pipeline.min_detection_duration,
        )
        if scene_detector is not None:
            scene_detector.reset()
        effect_active = False

    for frames, pts_list in timer.timed_iter(reader.frames(seek_ts=ctx.seek_ts), "decode"):
        if ctx.stop_event.is_set():
            break
        if end_pts is not None:
            keep_count = next((i for i, pts in enumerate(pts_list) if int(pts) >= end_pts), len(pts_list))
            if keep_count < len(pts_list):
                stop_after_batch = True
                frames = frames[:keep_count]
                pts_list = pts_list[:keep_count]
        effective_bs = len(pts_list)
        if effective_bs == 0:
            if stop_after_batch:
                break
            continue

        frame_shape = (int(frames.shape[-2]), int(frames.shape[-1]))
        if ctx.error_holder:
            raise ctx.error_holder[0]

        batch_start = frame_idx
        with timer.measure("detect-track"):
            for start, end, inside in effect_runs(pts_list, effect_ranges):
                if not inside:
                    _finalize_tracker()
                    for pts in pts_list[start:end]:
                        ctx.metadata_queue.put(FrameMeta(frame_idx=frame_idx, pts=int(pts), apply_effect=False))
                        frame_idx += 1
                    continue
                effect_active = True
                frame_idx = process_frame_batch(
                    frames=frames[start:end],
                    pts_list=[int(p) for p in pts_list[start:end]],
                    start_frame_idx=frame_idx,
                    target_hw=target_hw,
                    detections_fn=pipeline.job_detection_model,
                    tracker=tracker,
                    blend_buffer=ctx.blend_buffer,
                    crop_buffers=ctx.crop_buffers,
                    clip_queue=ctx.clip_queue,
                    metadata_queue=ctx.metadata_queue,
                    discard_margin=discard_margin,
                    blend_frames=blend_frames,
                    stereo=pipeline.stereo,
                    min_detection_duration=pipeline.min_detection_duration,
                    scene_detector=scene_detector,
                    vr_projector=pipeline.vr_projector,
                ).next_frame_idx
        ctx.debug_memory.snapshot("decode", f"frame_start={batch_start} batch={effective_bs}")
        if progress is not None:
            progress.update(effective_bs)
        if stop_after_batch:
            break

    if not ctx.stop_event.is_set():
        _finalize_tracker()
        ctx.debug_memory.snapshot("decode", "finalized")


def primary_restore_loop(ctx: PassContext) -> None:
    timer = LoopTimer("primary")
    restoration_pipeline = ctx.restoration_pipeline
    try:
        torch.cuda.set_device(ctx.device)
        log.debug("[primary] thread starting")
        while not ctx.stop_event.is_set():
            ctx.primary_idle_event.set()
            try:
                with timer.measure("queue-wait"):
                    item = ctx.clip_queue.get(timeout=0.1)
            except Empty:
                continue
            ctx.primary_idle_event.clear()
            if item is _SENTINEL:
                break
            clip_item: ClipRestoreItem = item
            with timer.measure("restore"):
                result = restoration_pipeline.prepare_and_run_primary(
                    clip_item.clip,
                    clip_item.raw_crops,
                    clip_item.frame_shape,
                    clip_item.keep_start,
                    clip_item.keep_end,
                    clip_item.crossfade_weights,
                )
                if restoration_pipeline.secondary_prefers_cpu_input:
                    result.primary_raw = result.primary_raw.cpu()
            with timer.measure("queue-put"):
                ctx.secondary_queue.put(result, frame_count=result.keep_end - result.keep_start)
            ctx.debug_memory.snapshot(
                "primary",
                f"clip={clip_item.clip.track_id} frames={len(clip_item.raw_crops)}",
            )
    except BaseException as e:
        if not ctx.stop_event.is_set():
            log.exception("[primary] thread crashed")
            ctx.error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[primary] thread exiting")
        ctx.secondary_queue.put(_SENTINEL)


def secondary_restore_loop(ctx: PassContext) -> None:
    timer = LoopTimer("secondary")
    restoration_pipeline = ctx.restoration_pipeline
    try:
        torch.cuda.set_device(ctx.device)
        log.debug("[secondary] thread starting")
        while not ctx.stop_event.is_set():
            try:
                with timer.measure("queue-wait"):
                    item = ctx.secondary_queue.get(timeout=0.1)
            except Empty:
                continue
            if item is _SENTINEL:
                break
            pr: PrimaryRestoreResult = item
            with timer.measure("restore"):
                restored_frames = restoration_pipeline._run_secondary(
                    pr.primary_raw,
                    pr.keep_start,
                    pr.keep_end,
                )
                del pr.primary_raw
                sr = restoration_pipeline.build_secondary_result(pr, restored_frames)
            with timer.measure("queue-put"):
                ctx.encode_queue.put(sr, frame_count=sr.keep_end)
            ctx.debug_memory.snapshot(
                "secondary",
                f"clip={pr.track_id} frames={sr.frame_count}",
            )
    except BaseException as e:
        if not ctx.stop_event.is_set():
            log.exception("[secondary] thread crashed")
            ctx.error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[secondary] thread exiting")
        ctx.encode_queue.put(_SENTINEL)


def blend_encode_loop(ctx: PassContext, frame_writer: PassFrameWriter, vram_offloader: VramOffloader) -> None:
    timer = LoopTimer("blend-encode")
    blend_buffer = ctx.blend_buffer
    encode_queue = ctx.encode_queue
    try:
        torch.cuda.set_device(ctx.device)

        def _flat_frames(rdr: VideoReader):
            for batch, pts in rdr.frames(seek_ts=ctx.seek_ts):
                for i in range(len(pts)):
                    yield batch[i]

        with ctx.open_reader() as reader2:
            frame_gen = _flat_frames(reader2)
            secondary_done = False
            frames_encoded = 0

            def _drain_encode_queue():
                nonlocal secondary_done
                while not secondary_done:
                    try:
                        sr_item = encode_queue.get_nowait()
                        if sr_item is _SENTINEL:
                            secondary_done = True
                        else:
                            blend_buffer.add_result(sr_item)
                    except Empty:
                        break

            while not ctx.stop_event.is_set():
                _drain_encode_queue()
                try:
                    with timer.measure("queue-wait"):
                        meta_item = ctx.metadata_queue.get(timeout=0.05)
                except Empty:
                    continue
                if meta_item is _SENTINEL:
                    break
                meta: FrameMeta = meta_item
                with timer.measure("decode"):
                    original_frame = next(frame_gen)

                with timer.measure("result-wait"):
                    while meta.apply_effect and not blend_buffer.is_frame_ready(meta.frame_idx):
                        if ctx.stop_event.is_set():
                            break
                        if ctx.error_holder:
                            raise ctx.error_holder[0]
                        if secondary_done:
                            log.error("[blend-encode] frame %d not ready but secondary is done", meta.frame_idx)
                            break
                        try:
                            sr_item = encode_queue.get(timeout=0.1)
                            if sr_item is _SENTINEL:
                                secondary_done = True
                                continue
                            blend_buffer.add_result(sr_item)
                        except Empty:
                            pass

                with timer.measure("blend"):
                    if not meta.apply_effect:
                        blended = original_frame
                    else:
                        blended = blend_buffer.blend_frame(
                            meta.frame_idx,
                            original_frame,
                        )
                with timer.measure("write"):
                    frame_writer.write(blended, meta.pts)
                    frames_encoded += 1
                    frame_writer.after_write(frames_encoded, ctx.stop_event)

            vram_offloader.pause_stall_check()

    except BaseException as e:
        if not ctx.stop_event.is_set():
            log.exception("[blend-encode] thread crashed")
            ctx.error_holder.append(e)
    finally:
        log.info(timer.summary())


def _estimate_start_frame(metadata, seek_ts: float) -> int:
    return int(seek_ts * metadata.video_fps)


_ASYNC_POLL_TIMEOUT = 0.05
_FLUSH_DELAY = 2.0
_FLUSH_RETRY_TIMEOUT = 5.0


def earliest_blocking_seqs(pending_prs: dict[int, PrimaryRestoreResult]) -> set[int] | None:
    if not pending_prs:
        return None
    earliest_frame = min(pr.start_frame + pr.keep_start for pr in pending_prs.values())
    return {
        seq for seq, pr in pending_prs.items()
        if pr.start_frame + pr.keep_start <= earliest_frame <= pr.start_frame + pr.keep_end - 1
    }


def starvation_action(
    *, starved: bool, flushed_since_last_push: bool, since_push: float, since_flush: float
) -> str | None:
    """``"flush"`` once buffered clips have waited ``_FLUSH_DELAY`` with no clip arriving,
    ``"retry"`` when a flush forwarded nothing for ``_FLUSH_RETRY_TIMEOUT``, else ``None``."""
    if not starved:
        return None
    if not flushed_since_last_push:
        return "flush" if since_push > _FLUSH_DELAY else None
    return "retry" if since_flush > _FLUSH_RETRY_TIMEOUT else None


def run_async_secondary(ctx: PassContext) -> SecondaryLoopStats:
    """Feed an AsyncSecondaryRestorer (TVAI) from one thread and forward results from this one.

    The restorer buffers frames internally, so when no clip is arriving the
    loop flushes the clip that blocks the earliest unblended frame.
    """
    restoration_pipeline = ctx.restoration_pipeline
    restorer: AsyncSecondaryRestorer = restoration_pipeline.secondary_restorer  # type: ignore[assignment]
    pending_prs: dict[int, PrimaryRestoreResult] = {}
    push_done = threading.Event()
    pusher_error: list[BaseException] = []
    last_push_time = time.monotonic()
    flushed_since_last_push = False
    last_flush_time = 0.0
    pusher_stall_seconds = 0.0
    clips_pushed = 0

    def _pusher():
        nonlocal last_push_time, flushed_since_last_push, pusher_stall_seconds, clips_pushed
        try:
            while not ctx.stop_event.is_set():
                try:
                    item = ctx.secondary_queue.get(timeout=0.1)
                except Empty:
                    continue
                if item is _SENTINEL:
                    break
                pr: PrimaryRestoreResult = item  # type: ignore[assignment]
                t0 = time.monotonic()
                seq = restorer.push_clip(
                    pr.primary_raw,
                    keep_start=pr.keep_start,
                    keep_end=pr.keep_end,
                )
                push_elapsed = time.monotonic() - t0
                pusher_stall_seconds += push_elapsed
                clips_pushed += 1
                del pr.primary_raw
                pending_prs[seq] = pr
                last_push_time = time.monotonic()
                flushed_since_last_push = False
                if push_elapsed > 0.05:
                    log.debug("[secondary] push_clip seq=%d took %.0fms", seq, push_elapsed * 1000)
        except BaseException as e:
            pusher_error.append(e)
        finally:
            push_done.set()

    clips_popped = 0

    def _forward_completed() -> int:
        nonlocal clips_popped
        forwarded = 0
        for seq, frames_np in restorer.pop_completed():
            pr = pending_prs.pop(seq)
            batch = restorer._to_tensors(frames_np)
            if batch.numel() > 0 and pr.frame_device.type != "cpu":
                batch = batch.to(pr.frame_device, non_blocking=True)
            tensors = list(batch.unbind(0)) if batch.numel() > 0 else []
            sr = restoration_pipeline.build_secondary_result(pr, tensors)
            ctx.encode_queue.put(sr, frame_count=sr.keep_end)
            ctx.debug_memory.snapshot("secondary", f"clip={pr.track_id} frames={sr.frame_count}")
            forwarded += 1
            clips_popped += 1
        return forwarded

    def _no_clips_incoming() -> bool:
        return ctx.primary_idle_event.is_set() and ctx.clip_queue.qsize() == 0

    pusher_thread = threading.Thread(target=_pusher, name="SecondaryPusher", daemon=True)
    pusher_thread.start()

    starvation_count = 0
    starvation_seconds = 0.0
    starvation_start: float | None = None

    while not push_done.is_set():
        if pusher_error:
            raise pusher_error[0]

        if _forward_completed() > 0:
            if starvation_start is not None:
                starvation_seconds += time.monotonic() - starvation_start
                starvation_start = None
            flushed_since_last_push = False
            continue

        now = time.monotonic()
        action = starvation_action(
            starved=restorer.has_pending and _no_clips_incoming(),
            flushed_since_last_push=flushed_since_last_push,
            since_push=now - last_push_time,
            since_flush=now - last_flush_time,
        )
        if action == "flush":
            if starvation_start is None:
                starvation_start = now
            target_seqs = earliest_blocking_seqs(dict(pending_prs))
            log.debug("[secondary] starvation flush target_seqs=%s", target_seqs)
            if restorer.flush_pending(target_seqs=target_seqs):
                flushed_since_last_push = True
                last_flush_time = now
            starvation_count += 1
        elif action == "retry":
            log.warning(
                "[secondary] flush retry: no clips forwarded for %.0fs after flush, pending=%d",
                now - last_flush_time, len(pending_prs),
            )
            flushed_since_last_push = False

        time.sleep(_ASYNC_POLL_TIMEOUT)

    if starvation_start is not None:
        starvation_seconds += time.monotonic() - starvation_start
    pusher_thread.join()
    if pusher_error:
        raise pusher_error[0]
    if not ctx.stop_event.is_set():
        restorer.flush_all()
        for _ in range(100):
            if not pending_prs:
                break
            _forward_completed()
            if pending_prs:
                time.sleep(_ASYNC_POLL_TIMEOUT)
    return SecondaryLoopStats(
        starvation_flushes=starvation_count,
        starvation_seconds=starvation_seconds,
        pusher_stall_seconds=pusher_stall_seconds,
        clips_pushed=clips_pushed,
        clips_popped=clips_popped,
    )


def async_secondary_restore_loop(ctx: PassContext) -> None:
    try:
        torch.cuda.set_device(ctx.device)
        stats = run_async_secondary(ctx)
        log.info(
            "Secondary — clips: %d pushed / %d popped, pusher stall: %.1fs, starvation flushes: %d (%.1fs)",
            stats.clips_pushed, stats.clips_popped, stats.pusher_stall_seconds,
            stats.starvation_flushes, stats.starvation_seconds,
        )
    except BaseException as e:
        if not ctx.stop_event.is_set():
            log.exception("[secondary-async] thread crashed")
            ctx.error_holder.append(e)
    finally:
        ctx.encode_queue.put(_SENTINEL)


def run_restoration_pass(
    pipeline,
    metadata,
    frame_writer: PassFrameWriter,
    cancel_event: threading.Event,
    *,
    seek_ts: float | None,
    use_async_secondary: bool,
    end_pts: int | None = None,
    effect_ranges: tuple[tuple[int, int], ...] | None = None,
    frame_stride: int = 1,
    output_frame_count: int | None = None,
    output_fps: float | None = None,
    progress: Progressbar | None = None,
    encode_heartbeat: list[float] | None = None,
    poll: Callable[[], bool] | None = None,
) -> BaseException | None:
    """Run the decode/detect -> primary -> secondary -> blend threads over one span.

    ``poll`` is called while the threads run; returning True cancels the pass.
    A cancel or the first thread error stops every thread, and the queues are
    drained so blocked producers can exit. Returns that first error (errors are
    not recorded after a cancel); each caller decides whether it matters.
    """
    device = pipeline.device
    restoration_pipeline = pipeline.restoration_pipeline
    max_clip_size = pipeline.max_clip_size
    secondary_workers = max(1, int(restoration_pipeline.secondary_num_workers))

    clip_queue = FrameQueue(max_frames=max_clip_size)
    secondary_queue = FrameQueue(max_frames=max_clip_size * secondary_workers)
    encode_queue = FrameQueue(max_frames=max_clip_size)
    metadata_queue: Queue[FrameMeta | object] = Queue(maxsize=max_clip_size * 5)
    queues = (clip_queue, secondary_queue, encode_queue, metadata_queue)

    blend_buffer = BlendBuffer(device=device, vr_projector=pipeline.vr_projector)
    crop_buffers: dict[int, CropBuffer] = {}
    vram_offloader = VramOffloader(device=device, blend_buffer=blend_buffer, crop_buffers=crop_buffers)
    if encode_heartbeat is not None:
        vram_offloader.set_encode_heartbeat(encode_heartbeat)
    vram_offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)
    ctx = PassContext(
        device=device,
        restoration_pipeline=restoration_pipeline,
        input_video=str(pipeline.input_video),
        decode_backend=pipeline.decode_backend,
        batch_size=pipeline.batch_size,
        metadata=metadata,
        seek_ts=seek_ts,
        frame_stride=frame_stride,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        clip_queue=clip_queue,
        secondary_queue=secondary_queue,
        encode_queue=encode_queue,
        metadata_queue=metadata_queue,
        primary_idle_event=threading.Event(),
        stop_event=threading.Event(),
        error_holder=[],
        debug_memory=PipelineDebugMemoryLogger(
            logger=log,
            blend_buffer=blend_buffer,
            clip_queue=clip_queue,
            secondary_queue=secondary_queue,
            encode_queue=encode_queue,
        ),
    )

    if use_async_secondary:
        log.debug("Using async secondary restore path")
        secondary_loop = async_secondary_restore_loop
    else:
        secondary_loop = secondary_restore_loop
    decode_options = dict(
        end_pts=end_pts,
        effect_ranges=effect_ranges,
        output_frame_count=output_frame_count,
        output_fps=output_fps,
        progress=progress,
    )
    threads = [
        threading.Thread(
            target=decode_detect_loop, args=(ctx, pipeline), kwargs=decode_options, name="DecodeDetect", daemon=True
        ),
        threading.Thread(target=primary_restore_loop, args=(ctx,), name="PrimaryRestore", daemon=True),
        threading.Thread(target=secondary_loop, args=(ctx,), name="SecondaryRestore", daemon=True),
        threading.Thread(
            target=blend_encode_loop, args=(ctx, frame_writer, vram_offloader), name="BlendEncode", daemon=True
        ),
    ]
    vram_offloader.start()
    for thread in threads:
        thread.start()

    try:
        while any(thread.is_alive() for thread in threads):
            if poll is not None and poll():
                cancel_event.set()
            if cancel_event.wait(0.05) or ctx.error_holder:
                break
    finally:
        # Also on Ctrl+C: the workers stop before the caller closes the encoder and models they use.
        ctx.stop_event.set()
        for thread in threads:
            while thread.is_alive():
                _drain(queues)
                thread.join(timeout=0.02)
        vram_offloader.stop()

    error = ctx.error_holder[0] if ctx.error_holder else None
    del ctx, queues, clip_queue, secondary_queue, encode_queue, metadata_queue
    del blend_buffer, crop_buffers, threads
    gc.collect()
    empty_cache(device)
    ipc_collect(device)
    return error


def _drain(queues) -> None:
    for pipeline_queue in queues:
        try:
            while True:
                pipeline_queue.get_nowait()
        except Empty:
            continue

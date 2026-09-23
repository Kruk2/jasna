from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Protocol

import torch

from jasna.accelerator import empty_cache, ipc_collect
from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue
from jasna.media.video_decoder import VideoReader
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryLoopStats, _SENTINEL
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.pipeline_timing import LoopTimer
from jasna.progressbar import Progressbar
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.tracking.clip_tracker import ClipTracker
from jasna.tracking.scene_detector import SceneCutDetector
from jasna.vram_offloader import VramOffloader

log = logging.getLogger(__name__)


class FrameWriter(Protocol):
    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None: ...
    def after_write(self, frames_written: int) -> None: ...


def decode_detect_loop(
    *,
    input_video: str,
    batch_size: int,
    device: torch.device,
    metadata,
    detection_model,
    max_clip_size: int,
    temporal_overlap: int,
    max_detection_gap: int,
    min_detection_duration: int,
    enable_crossfade: bool,
    scene_detection: bool,
    blend_buffer: BlendBuffer,
    crop_buffers: dict[int, CropBuffer],
    clip_queue: FrameQueue,
    metadata_queue: Queue,
    error_holder: list[BaseException],
    cancel_event: threading.Event,
    debug_memory: PipelineDebugMemoryLogger,
    vr_mode: str,
    vr_projector,
    seek_ts: float | None,
    end_pts: int | None,
    effect_ranges: tuple[tuple[int, int], ...] | None,
    frame_stride: int,
    output_frame_count: int | None,
    output_fps: float | None,
    progress: Progressbar | None,
) -> None:
    timer = LoopTimer("decode-detect")
    try:
        torch.cuda.set_device(device)
        tracker = ClipTracker(
            max_clip_size=max_clip_size,
            temporal_overlap=temporal_overlap,
            max_detection_gap=max_detection_gap,
        )
        scene_detector = SceneCutDetector() if scene_detection else None
        discard_margin = temporal_overlap
        blend_frames = (temporal_overlap // 3) if enable_crossfade else 0

        with (
            VideoReader(
                input_video,
                batch_size=batch_size,
                device=device,
                metadata=metadata,
                frame_stride=frame_stride,
            ) as reader,
            torch.inference_mode(),
        ):
            if progress is not None:
                progress.init()
            target_hw = (int(metadata.video_height), int(metadata.video_width))
            crop_eye_width = (
                int(metadata.video_width) // 2 if vr_mode == "sbs" else None
            )
            frame_idx = 0 if seek_ts is None else _estimate_start_frame(metadata, seek_ts)
            frame_shape = target_hw
            effect_active = effect_ranges is None
            stop_after_batch = False

            def _selected(pts: int) -> bool:
                if effect_ranges is None:
                    return True
                return any(start <= pts < end for start, end in effect_ranges)

            def _finalize_tracker() -> None:
                nonlocal effect_active
                if not effect_active:
                    return
                finalize_processing(
                    tracker=tracker,
                    blend_buffer=blend_buffer,
                    crop_buffers=crop_buffers,
                    clip_queue=clip_queue,
                    frame_shape=frame_shape,
                    discard_margin=discard_margin,
                    blend_frames=blend_frames,
                    min_detection_duration=min_detection_duration,
                )
                if scene_detector is not None:
                    scene_detector.reset()
                effect_active = False
            log.info(
                "Processing %s: %d frames @ %s fps, %dx%d",
                input_video,
                metadata.num_frames if output_frame_count is None else output_frame_count,
                metadata.video_fps if output_fps is None else output_fps,
                metadata.video_width,
                metadata.video_height,
            )

            try:
                for frames, pts_list in timer.timed_iter(reader.frames(seek_ts=seek_ts), "decode"):
                    if cancel_event.is_set():
                        break
                    if end_pts is not None:
                        keep_count = next(
                            (i for i, pts in enumerate(pts_list) if int(pts) >= end_pts),
                            len(pts_list),
                        )
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
                    if error_holder:
                        raise error_holder[0]

                    batch_start = frame_idx

                    with timer.measure("detect-track"):
                        offset = 0
                        while offset < effective_bs:
                            selected = _selected(int(pts_list[offset]))
                            group_end = offset + 1
                            while (
                                group_end < effective_bs
                                and _selected(int(pts_list[group_end])) == selected
                            ):
                                group_end += 1

                            if selected:
                                effect_active = True
                                selected_frames = frames[offset:group_end]
                                res = process_frame_batch(
                                    frames=selected_frames,
                                    pts_list=[int(p) for p in pts_list[offset:group_end]],
                                    start_frame_idx=frame_idx,
                                    target_hw=target_hw,
                                    detections_fn=detection_model,
                                    tracker=tracker,
                                    blend_buffer=blend_buffer,
                                    crop_buffers=crop_buffers,
                                    clip_queue=clip_queue,
                                    metadata_queue=metadata_queue,
                                    discard_margin=discard_margin,
                                    blend_frames=blend_frames,
                                    crop_eye_width=crop_eye_width,
                                    min_detection_duration=min_detection_duration,
                                    scene_detector=scene_detector,
                                    vr_projector=vr_projector,
                                )
                                frame_idx = res.next_frame_idx
                            else:
                                _finalize_tracker()
                                for pts in pts_list[offset:group_end]:
                                    metadata_queue.put(
                                        FrameMeta(
                                            frame_idx=frame_idx,
                                            pts=int(pts),
                                            apply_effect=False,
                                        )
                                    )
                                    frame_idx += 1
                            offset = group_end
                    debug_memory.snapshot("decode", f"frame_start={batch_start} batch={effective_bs}")
                    if progress is not None:
                        progress.update(effective_bs)
                    if stop_after_batch:
                        break

                if not cancel_event.is_set():
                    _finalize_tracker()
                    debug_memory.snapshot("decode", "finalized")
            except Exception:
                if progress is not None:
                    progress.error = True
                raise
    except BaseException as e:
        if not cancel_event.is_set():
            log.exception("[decode] thread crashed")
            error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[decode] thread exiting")
        clip_queue.put(_SENTINEL)
        metadata_queue.put(_SENTINEL)


def primary_restore_loop(
    *,
    device: torch.device,
    restoration_pipeline: RestorationPipeline,
    clip_queue: FrameQueue,
    secondary_queue: FrameQueue,
    error_holder: list[BaseException],
    primary_idle_event: threading.Event,
    cancel_event: threading.Event,
    debug_memory: PipelineDebugMemoryLogger,
) -> None:
    timer = LoopTimer("primary")
    try:
        torch.cuda.set_device(device)
        log.debug("[primary] thread starting")
        while not cancel_event.is_set():
            primary_idle_event.set()
            try:
                with timer.measure("queue-wait"):
                    item = clip_queue.get(timeout=0.1)
            except Empty:
                continue
            primary_idle_event.clear()
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
                secondary_queue.put(result, frame_count=result.keep_end - result.keep_start)
            debug_memory.snapshot(
                "primary",
                f"clip={clip_item.clip.track_id} frames={len(clip_item.raw_crops)}",
            )
    except BaseException as e:
        if not cancel_event.is_set():
            log.exception("[primary] thread crashed")
            error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[primary] thread exiting")
        secondary_queue.put(_SENTINEL)


def secondary_restore_loop(
    *,
    device: torch.device,
    restoration_pipeline: RestorationPipeline,
    secondary_queue: FrameQueue,
    encode_queue: FrameQueue,
    error_holder: list[BaseException],
    cancel_event: threading.Event,
    debug_memory: PipelineDebugMemoryLogger,
) -> None:
    timer = LoopTimer("secondary")
    try:
        torch.cuda.set_device(device)
        log.debug("[secondary] thread starting")
        while not cancel_event.is_set():
            try:
                with timer.measure("queue-wait"):
                    item = secondary_queue.get(timeout=0.1)
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
                encode_queue.put(sr, frame_count=sr.keep_end)
            debug_memory.snapshot(
                "secondary",
                f"clip={pr.track_id} frames={sr.frame_count}",
            )
    except BaseException as e:
        if not cancel_event.is_set():
            log.exception("[secondary] thread crashed")
            error_holder.append(e)
    finally:
        log.info(timer.summary())
        log.debug("[secondary] thread exiting")
        encode_queue.put(_SENTINEL)


def blend_encode_loop(
    *,
    input_video: str,
    batch_size: int,
    device: torch.device,
    metadata,
    blend_buffer: BlendBuffer,
    encode_queue: FrameQueue,
    metadata_queue: Queue,
    error_holder: list[BaseException],
    frame_writer: FrameWriter,
    cancel_event: threading.Event,
    vram_offloader: VramOffloader,
    seek_ts: float | None,
    frame_stride: int,
) -> None:
    timer = LoopTimer("blend-encode")
    try:
        torch.cuda.set_device(device)

        def _flat_frames(rdr: VideoReader):
            for batch, pts in rdr.frames(seek_ts=seek_ts):
                for i in range(len(pts)):
                    yield batch[i]

        with VideoReader(
            input_video,
            batch_size=batch_size,
            device=device,
            metadata=metadata,
            frame_stride=frame_stride,
        ) as reader2:
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

            while not cancel_event.is_set():
                _drain_encode_queue()
                try:
                    with timer.measure("queue-wait"):
                        meta_item = metadata_queue.get(timeout=0.05)
                except Empty:
                    continue
                if meta_item is _SENTINEL:
                    break
                meta: FrameMeta = meta_item
                with timer.measure("decode"):
                    original_frame = next(frame_gen)

                with timer.measure("result-wait"):
                    while meta.apply_effect and not blend_buffer.is_frame_ready(meta.frame_idx):
                        if cancel_event.is_set():
                            break
                        if error_holder:
                            raise error_holder[0]
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
                    if meta.apply_effect:
                        frame_writer.write(blended, meta.pts)
                    else:
                        frame_writer.write(blended, meta.pts, apply_lut=False)
                    frames_encoded += 1
                    frame_writer.after_write(frames_encoded)

            vram_offloader.pause_stall_check()

    except BaseException as e:
        if not cancel_event.is_set():
            log.exception("[blend-encode] thread crashed")
            error_holder.append(e)
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


def run_async_secondary(
    *,
    restoration_pipeline: RestorationPipeline,
    secondary_queue: FrameQueue,
    encode_queue: FrameQueue,
    clip_queue: FrameQueue,
    primary_idle_event: threading.Event,
    cancel_event: threading.Event,
    debug_memory: PipelineDebugMemoryLogger,
) -> SecondaryLoopStats:
    """Feed an AsyncSecondaryRestorer (TVAI) from one thread and forward results from this one.

    The restorer buffers frames internally, so when no clip is arriving the
    loop flushes the clip that blocks the earliest unblended frame.
    """
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
            while not cancel_event.is_set():
                try:
                    item = secondary_queue.get(timeout=0.1)
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
            encode_queue.put(sr, frame_count=sr.keep_end)
            debug_memory.snapshot("secondary", f"clip={pr.track_id} frames={sr.frame_count}")
            forwarded += 1
            clips_popped += 1
        return forwarded

    def _no_clips_incoming() -> bool:
        return primary_idle_event.is_set() and clip_queue.qsize() == 0

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
        if (
            restorer.has_pending
            and _no_clips_incoming()
            and not flushed_since_last_push
            and now - last_push_time > _FLUSH_DELAY
        ):
            if starvation_start is None:
                starvation_start = now
            target_seqs = earliest_blocking_seqs(dict(pending_prs))
            log.debug("[secondary] starvation flush target_seqs=%s", target_seqs)
            if restorer.flush_pending(target_seqs=target_seqs):
                flushed_since_last_push = True
                last_flush_time = now
            starvation_count += 1
        elif (
            flushed_since_last_push
            and restorer.has_pending
            and _no_clips_incoming()
            and now - last_flush_time > _FLUSH_RETRY_TIMEOUT
        ):
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
    if not cancel_event.is_set():
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


def async_secondary_restore_loop(
    *,
    device: torch.device,
    restoration_pipeline: RestorationPipeline,
    secondary_queue: FrameQueue,
    encode_queue: FrameQueue,
    clip_queue: FrameQueue,
    primary_idle_event: threading.Event,
    error_holder: list[BaseException],
    cancel_event: threading.Event,
    debug_memory: PipelineDebugMemoryLogger,
) -> None:
    try:
        torch.cuda.set_device(device)
        stats = run_async_secondary(
            restoration_pipeline=restoration_pipeline,
            secondary_queue=secondary_queue,
            encode_queue=encode_queue,
            clip_queue=clip_queue,
            primary_idle_event=primary_idle_event,
            cancel_event=cancel_event,
            debug_memory=debug_memory,
        )
        log.info(
            "Secondary — clips: %d pushed / %d popped, pusher stall: %.1fs, starvation flushes: %d (%.1fs)",
            stats.clips_pushed, stats.clips_popped, stats.pusher_stall_seconds,
            stats.starvation_flushes, stats.starvation_seconds,
        )
    except BaseException as e:
        if not cancel_event.is_set():
            log.exception("[secondary-async] thread crashed")
            error_holder.append(e)
    finally:
        encode_queue.put(_SENTINEL)


def run_restoration_pass(
    pipeline,
    metadata,
    frame_writer: FrameWriter,
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
    After a cancel the queues are drained so blocked producers can exit. Returns
    the first error a thread recorded (errors are not recorded after a cancel);
    each caller decides whether it matters.
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

    error_holder: list[BaseException] = []
    blend_buffer = BlendBuffer(device=device, vr_projector=pipeline.vr_projector)
    crop_buffers: dict[int, CropBuffer] = {}
    primary_idle_event = threading.Event()
    vram_offloader = VramOffloader(device=device, blend_buffer=blend_buffer, crop_buffers=crop_buffers)
    if encode_heartbeat is not None:
        vram_offloader.set_encode_heartbeat(encode_heartbeat)
    vram_offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)
    debug_memory = PipelineDebugMemoryLogger(
        logger=log,
        blend_buffer=blend_buffer,
        clip_queue=clip_queue,
        secondary_queue=secondary_queue,
        encode_queue=encode_queue,
    )

    secondary_kwargs = dict(
        device=device,
        restoration_pipeline=restoration_pipeline,
        secondary_queue=secondary_queue,
        encode_queue=encode_queue,
        error_holder=error_holder,
        cancel_event=cancel_event,
        debug_memory=debug_memory,
    )
    if use_async_secondary:
        log.debug("Using async secondary restore path")
        secondary_target = lambda: async_secondary_restore_loop(  # noqa: E731
            clip_queue=clip_queue, primary_idle_event=primary_idle_event, **secondary_kwargs
        )
    else:
        secondary_target = lambda: secondary_restore_loop(**secondary_kwargs)  # noqa: E731

    threads = [
        threading.Thread(
            target=lambda: decode_detect_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                detection_model=pipeline.job_detection_model,
                max_clip_size=max_clip_size,
                temporal_overlap=pipeline.temporal_overlap,
                max_detection_gap=pipeline.max_detection_gap,
                min_detection_duration=pipeline.min_detection_duration,
                enable_crossfade=pipeline.enable_crossfade,
                scene_detection=pipeline.scene_detection,
                blend_buffer=blend_buffer,
                crop_buffers=crop_buffers,
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                cancel_event=cancel_event,
                debug_memory=debug_memory,
                vr_mode=pipeline.vr_resolution.resolved,
                vr_projector=pipeline.vr_projector,
                seek_ts=seek_ts,
                end_pts=end_pts,
                effect_ranges=effect_ranges,
                frame_stride=frame_stride,
                output_frame_count=output_frame_count,
                output_fps=output_fps,
                progress=progress,
            ),
            name="DecodeDetect", daemon=True,
        ),
        threading.Thread(
            target=lambda: primary_restore_loop(
                device=device,
                restoration_pipeline=restoration_pipeline,
                clip_queue=clip_queue,
                secondary_queue=secondary_queue,
                error_holder=error_holder,
                primary_idle_event=primary_idle_event,
                cancel_event=cancel_event,
                debug_memory=debug_memory,
            ),
            name="PrimaryRestore", daemon=True,
        ),
        threading.Thread(target=secondary_target, name="SecondaryRestore", daemon=True),
        threading.Thread(
            target=lambda: blend_encode_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_writer=frame_writer,
                cancel_event=cancel_event,
                vram_offloader=vram_offloader,
                seek_ts=seek_ts,
                frame_stride=frame_stride,
            ),
            name="BlendEncode", daemon=True,
        ),
    ]
    vram_offloader.start()
    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        if poll is not None and poll():
            cancel_event.set()
        if cancel_event.wait(0.05):
            break

    for thread in threads:
        while thread.is_alive():
            _drain(queues)
            thread.join(timeout=0.02)
    vram_offloader.stop()

    error = error_holder[0] if error_holder else None
    del queues, clip_queue, secondary_queue, encode_queue, metadata_queue
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

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Protocol

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.pipeline_timing import LoopTimer
from jasna.progressbar import Progressbar
from jasna.restorer import RestorationPipeline
from jasna.tracking import ClipTracker

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
    enable_crossfade: bool,
    blend_buffer: BlendBuffer,
    crop_buffers: dict[int, CropBuffer],
    clip_queue: FrameQueue,
    metadata_queue: Queue,
    error_holder: list[BaseException],
    frame_shape: list[tuple[int, int]],
    cancel_event: threading.Event | None = None,
    seek_ts: float | None = None,
    end_pts: int | None = None,
    effect_ranges: tuple[tuple[int, int], ...] | None = None,
    frame_stride: int = 1,
    output_frame_count: int | None = None,
    output_fps: float | None = None,
    progress: Progressbar | None = None,
    close_progress: bool = True,
    debug_memory: PipelineDebugMemoryLogger | None = None,
) -> None:
    timer = LoopTimer("decode-detect")
    try:
        torch.cuda.set_device(device)
        tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=temporal_overlap)
        discard_margin = temporal_overlap
        blend_frames = (temporal_overlap // 3) if enable_crossfade else 0

        with (
            NvidiaVideoReader(
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
            frame_idx = 0 if seek_ts is None else _estimate_start_frame(metadata, seek_ts)
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
                fs = frame_shape[0] if frame_shape else target_hw
                finalize_processing(
                    tracker=tracker,
                    blend_buffer=blend_buffer,
                    crop_buffers=crop_buffers,
                    clip_queue=clip_queue,
                    frame_shape=fs,
                    discard_margin=discard_margin,
                    blend_frames=blend_frames,
                )
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
                    if cancel_event is not None and cancel_event.is_set():
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

                    if not frame_shape:
                        _, fh, fw = frames[0].shape
                        frame_shape.append((int(fh), int(fw)))

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
                                res = process_frame_batch(
                                    frames=frames[offset:group_end],
                                    pts_list=[int(p) for p in pts_list[offset:group_end]],
                                    start_frame_idx=frame_idx,
                                    batch_size=batch_size,
                                    target_hw=target_hw,
                                    detections_fn=detection_model,
                                    tracker=tracker,
                                    blend_buffer=blend_buffer,
                                    crop_buffers=crop_buffers,
                                    clip_queue=clip_queue,
                                    metadata_queue=metadata_queue,
                                    discard_margin=discard_margin,
                                    blend_frames=blend_frames,
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
                    if debug_memory is not None:
                        debug_memory.snapshot("decode", f"frame_start={batch_start} batch={effective_bs}")
                    if progress is not None:
                        progress.update(effective_bs)
                    if stop_after_batch:
                        break

                if cancel_event is None or not cancel_event.is_set():
                    _finalize_tracker()
                    if debug_memory is not None:
                        debug_memory.snapshot("decode", "finalized")
            except Exception:
                if progress is not None:
                    progress.error = True
                raise
            finally:
                if progress is not None and close_progress:
                    progress.close(ensure_completed_bar=True)
    except BaseException as e:
        if cancel_event is None or not cancel_event.is_set():
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
    cancel_event: threading.Event | None = None,
    debug_memory: PipelineDebugMemoryLogger | None = None,
) -> None:
    timer = LoopTimer("primary")
    try:
        torch.cuda.set_device(device)
        log.debug("[primary] thread starting")
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            primary_idle_event.set()
            if cancel_event is not None:
                try:
                    with timer.measure("queue-wait"):
                        item = clip_queue.get(timeout=0.1)
                except Empty:
                    continue
            else:
                with timer.measure("queue-wait"):
                    item = clip_queue.get()
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
            if debug_memory is not None:
                debug_memory.snapshot(
                    "primary",
                    f"clip={clip_item.clip.track_id} frames={len(clip_item.raw_crops)}",
                )
    except BaseException as e:
        if cancel_event is None or not cancel_event.is_set():
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
    cancel_event: threading.Event | None = None,
    debug_memory: PipelineDebugMemoryLogger | None = None,
) -> None:
    timer = LoopTimer("secondary")
    try:
        torch.cuda.set_device(device)
        log.debug("[secondary] thread starting")
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            if cancel_event is not None:
                try:
                    with timer.measure("queue-wait"):
                        item = secondary_queue.get(timeout=0.1)
                except Empty:
                    continue
            else:
                with timer.measure("queue-wait"):
                    item = secondary_queue.get()
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
            if debug_memory is not None:
                debug_memory.snapshot(
                    "secondary",
                    f"clip={pr.track_id} frames={sr.frame_count}",
                )
    except BaseException as e:
        if cancel_event is None or not cancel_event.is_set():
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
    cancel_event: threading.Event | None = None,
    seek_ts: float | None = None,
    frame_stride: int = 1,
    vram_offloader=None,
) -> None:
    timer = LoopTimer("blend-encode")
    try:
        torch.cuda.set_device(device)

        def _flat_frames(rdr: NvidiaVideoReader):
            for batch, pts in rdr.frames(seek_ts=seek_ts):
                for i in range(len(pts)):
                    yield batch[i]

        with NvidiaVideoReader(
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

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    break
                _drain_encode_queue()
                try:
                    with timer.measure("queue-wait"):
                        meta_item = metadata_queue.get(timeout=0.1 if cancel_event is not None else 0.05)
                except Empty:
                    continue
                if meta_item is _SENTINEL:
                    break
                meta: FrameMeta = meta_item
                with timer.measure("decode"):
                    original_frame = next(frame_gen)

                with timer.measure("result-wait"):
                    while meta.apply_effect and not blend_buffer.is_frame_ready(meta.frame_idx):
                        if cancel_event is not None and cancel_event.is_set():
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
                    blended = (
                        blend_buffer.blend_frame(meta.frame_idx, original_frame)
                        if meta.apply_effect
                        else original_frame
                    )
                with timer.measure("write"):
                    if meta.apply_effect:
                        frame_writer.write(blended, meta.pts)
                    else:
                        frame_writer.write(blended, meta.pts, apply_lut=False)
                    frames_encoded += 1
                    frame_writer.after_write(frames_encoded)

            if vram_offloader is not None:
                vram_offloader.pause_stall_check()

    except BaseException as e:
        if cancel_event is None or not cancel_event.is_set():
            log.exception("[blend-encode] thread crashed")
            error_holder.append(e)
    finally:
        log.info(timer.summary())


def _estimate_start_frame(metadata, seek_ts: float) -> int:
    return int(seek_ts * metadata.video_fps)

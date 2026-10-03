from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from queue import Empty, Queue
from typing import Any, Protocol

import torch

from jasna.accelerator import AcceleratorVendor, current_stream, empty_cache, ipc_collect, vendor_for_device
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
from jasna.vram_offloader import (
    VramOffloader, default_host_memory_limit_bytes,
    restoration_vram_safetynet, restoration_vram_startup_budget,
)
from jasna.native_worker import (
    HostMemoryPressureError, NativeWorkerRecycleRequested,
    amf_encoder_stall_timeout_seconds, is_isolated_video_job,
)

log = logging.getLogger(__name__)

AMF_READER_CALLER_HANDOFF_ENV = "JASNA_AMF_READER_CALLER_HANDOFF"
_AMF_READER_CALLER_HANDOFF_MODES = frozenset(
    {"off", "record-stream", "record-stream-clone-batch"}
)


def _amf_reader_caller_handoff_mode(reader: VideoReader) -> str:
    """Return the storage handoff policy for one opened reader.

    Explicit AMF interop readers produce batches on a private Torch stream, so
    their caller must register the subsequent cross-stream use with the
    caching allocator.  Other decoder routes retain their existing behavior.
    The environment override remains available for rollback and diagnosis.
    """

    configured_mode = os.environ.get(AMF_READER_CALLER_HANDOFF_ENV)
    if configured_mode is None:
        return (
            "record-stream"
            if (
                getattr(reader, "_amf_interop_enabled", False) is True
                or getattr(reader, "_windows_resident_enabled", False) is True
            )
            else "off"
        )
    mode = configured_mode.strip().casefold()
    if mode not in _AMF_READER_CALLER_HANDOFF_MODES:
        raise ValueError(
            f"Invalid {AMF_READER_CALLER_HANDOFF_ENV} value {mode!r}; expected "
            "'off', 'record-stream', or 'record-stream-clone-batch'"
        )
    return mode


def _handoff_reader_batch_to_caller(
    batch: torch.Tensor,
    *,
    role: str,
    mode: str,
) -> torch.Tensor:
    """Register one yielded decode batch with its actual caller stream.

    AMF uploads and RGB conversion run on a reader-private stream.  A producer
    synchronization makes the pixels readable, but it does not tell Torch's
    caching allocator that the decode tensor is subsequently consumed on the
    caller stream.  The experimental clone mode first records that source use,
    then gives the caller storage allocated on its own stream.
    """

    if mode == "off":
        return batch
    if not isinstance(batch, torch.Tensor) or batch.device.type != "cuda":
        raise RuntimeError(
            f"{AMF_READER_CALLER_HANDOFF_ENV}={mode} requires a CUDA/HIP "
            f"reader tensor for {role}; got {type(batch).__name__} on "
            f"{getattr(getattr(batch, 'device', None), 'type', None)!r}"
        )
    caller_stream = current_stream(batch.device)
    batch.record_stream(caller_stream)
    if mode == "record-stream-clone-batch":
        batch = batch.clone()
    return batch


def record_worker_error(
    label: str,
    error: BaseException,
    error_holder: list[BaseException],
    cancel_event: threading.Event | None,
) -> None:
    """Record the first worker failure and ask the other workers to stop."""
    if cancel_event is not None and cancel_event.is_set():
        return
    log.exception("[%s] thread crashed", label)
    error_holder.append(error)
    if cancel_event is not None:
        cancel_event.set()


def _drain_pipeline_queues(queues: Iterable[Any]) -> None:
    for pipeline_queue in queues:
        while True:
            try:
                pipeline_queue.get_nowait()
            except Empty:
                break


def wait_for_worker_threads(
    threads: Sequence[threading.Thread],
    queues: Iterable[Any],
    cancel_event: threading.Event,
    *,
    poll_interval: float = 0.02,
) -> None:
    """Join workers while releasing blocked producers during cancellation."""
    pipeline_queues = tuple(queues)
    while True:
        alive = [thread for thread in threads if thread.is_alive()]
        if not alive:
            return
        if cancel_event.is_set():
            _drain_pipeline_queues(pipeline_queues)
        for thread in alive:
            thread.join(timeout=poll_interval)

PTS_RESYNC_HARDWARE_RETRIES = 2
PTS_RESYNC_FORWARD_SCAN_LIMIT = 64


class _PtsRecoveryCancelled(RuntimeError):
    pass


class _PtsAlignedFrameReader:
    """Read exact source PTS and reopen the selected product route on divergence."""

    def __init__(
        self,
        *,
        input_video: str,
        batch_size: int,
        device: torch.device,
        metadata,
        frame_stride: int,
        seek_ts: float | None,
        cancel_event: threading.Event | None,
        resident_coordinator: object | None = None,
    ) -> None:
        self.input_video = input_video
        self.batch_size = int(batch_size)
        self.device = device
        self.metadata = metadata
        self.frame_stride = int(frame_stride)
        self.seek_ts = seek_ts
        self.cancel_event = cancel_event
        self.resident_coordinator = resident_coordinator
        self._reader: VideoReader | None = None
        self._frames = None
        self._retry_backend: str | None = None
        self._stream_start_pts = int(getattr(metadata, "start_pts", 0) or 0)

    @staticmethod
    def _flat_frames(
        reader: VideoReader,
        seek_ts: float | None,
        handoff_mode: str,
    ):
        logged_handoff = False
        for batch, pts in reader.frames(seek_ts=seek_ts):
            batch = _handoff_reader_batch_to_caller(
                batch,
                role="blend-encode",
                mode=handoff_mode,
            )
            if handoff_mode != "off" and not logged_handoff:
                log.info(
                    "AMF reader caller handoff: role=blend-encode mode=%s",
                    handoff_mode,
                )
                logged_handoff = True
            for index, frame_pts in enumerate(pts):
                yield batch[index], int(frame_pts)

    def __enter__(self):
        self._open(self.seek_ts, decode_backend=None)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(exc_type=exc_type, exc_value=exc_value, traceback=traceback)

    def close(
        self,
        *,
        exc_type=None,
        exc_value=None,
        traceback=None,
    ) -> None:
        reader, self._reader = self._reader, None
        frames, self._frames = self._frames, None
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        if reader is not None:
            reader.__exit__(exc_type, exc_value, traceback)

    def _open(self, seek_ts: float | None, *, decode_backend: str | None) -> None:
        self.close()
        reader = VideoReader(
            self.input_video,
            batch_size=self.batch_size,
            device=self.device,
            metadata=self.metadata,
            frame_stride=self.frame_stride,
            decode_backend=(
                "amf-d3d11-hip-resident"
                if self.resident_coordinator is not None
                else decode_backend
            ),
            resident_coordinator=self.resident_coordinator,
            resident_role=(
                "blend-encode"
                if self.resident_coordinator is not None
                else None
            ),
        )
        try:
            entered = reader.__enter__()
        except BaseException:
            reader.__exit__(None, None, None)
            raise
        self._reader = reader
        self._frames = self._flat_frames(
            entered,
            seek_ts,
            _amf_reader_caller_handoff_mode(entered),
        )
        self._stream_start_pts = int(entered.start_pts)
        self._retry_backend = str(entered._decode_backend)

    def _next(self) -> tuple[torch.Tensor | None, int | None]:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise _PtsRecoveryCancelled("PTS recovery cancelled")
        if self._frames is None:
            return None, None
        try:
            return next(self._frames)
        except StopIteration:
            return None, None

    def _scan_for_pts(
        self,
        expected_pts: int,
        *,
        first_frame: torch.Tensor | None = None,
        first_pts: int | None = None,
    ) -> tuple[torch.Tensor | None, int | None, int]:
        frame, actual_pts = first_frame, first_pts
        discarded = 0
        for _ in range(PTS_RESYNC_FORWARD_SCAN_LIMIT + 1):
            if actual_pts is None:
                frame, actual_pts = self._next()
            if actual_pts is None or actual_pts >= expected_pts:
                return frame, actual_pts, discarded
            discarded += 1
            frame, actual_pts = self._next()
        return frame, actual_pts, discarded

    def _target_seek_seconds(self, expected_pts: int) -> float:
        return max(
            0.0,
            float(
                (int(expected_pts) - self._stream_start_pts)
                * self.metadata.time_base
            ),
        )

    def read_exact(self, expected_pts: int) -> torch.Tensor:
        frame, actual_pts = self._next()
        if actual_pts == expected_pts and frame is not None:
            return frame

        first_actual = actual_pts
        if actual_pts is not None and actual_pts < expected_pts:
            frame, actual_pts, discarded = self._scan_for_pts(
                expected_pts,
                first_frame=frame,
                first_pts=actual_pts,
            )
            if actual_pts == expected_pts and frame is not None:
                log.warning(
                    "[blend-encode] PTS resynchronized by discarding %d stale "
                    "secondary-reader frame(s): expected=%d first_actual=%d",
                    discarded,
                    expected_pts,
                    first_actual,
                )
                return frame

        seek_ts = self._target_seek_seconds(expected_pts)
        if self.resident_coordinator is not None:
            self.close()
            raise RuntimeError(
                "The explicit Windows D3D11/HIP resident route does not reopen "
                "a decoder after a PTS mismatch; refusing an unvalidated native "
                f"session transition at expected PTS {expected_pts}"
            )
        observations: list[str] = []
        retry_backend = self._retry_backend
        attempts = [
            (retry_backend, f"{retry_backend or 'decoder'} retry {attempt}")
            for attempt in range(1, PTS_RESYNC_HARDWARE_RETRIES + 1)
        ]
        for backend, description in attempts:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise _PtsRecoveryCancelled("PTS recovery cancelled")
            try:
                self._open(seek_ts, decode_backend=backend)
                frame, recovered_pts, discarded = self._scan_for_pts(expected_pts)
            except _PtsRecoveryCancelled:
                raise
            except Exception as error:
                observations.append(f"{description}: {type(error).__name__}: {error}")
                log.warning(
                    "[blend-encode] PTS recovery %s failed while reopening at "
                    "PTS %d: %s",
                    description,
                    expected_pts,
                    error,
                )
                continue

            if recovered_pts == expected_pts and frame is not None:
                log.warning(
                    "[blend-encode] PTS mismatch recovered by %s at PTS %d "
                    "(first_actual=%s, discarded=%d)",
                    description,
                    expected_pts,
                    first_actual,
                    discarded,
                )
                return frame
            observations.append(
                f"{description}: observed "
                f"{recovered_pts if recovered_pts is not None else 'EOF'}"
            )

        self.close()
        details = "; ".join(observations)
        raise RuntimeError(
            "[blend-encode] could not recover secondary-reader PTS mismatch: "
            f"expected PTS {expected_pts}, initial actual PTS "
            f"{first_actual if first_actual is not None else 'EOF'}; {details}"
        )


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
    vr_mode: str = "off",
    vr_projector=None,
    resident_coordinator: object | None = None,
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

        reader_context = VideoReader(
            input_video,
            batch_size=batch_size,
            device=device,
            metadata=metadata,
            frame_stride=frame_stride,
            decode_backend=(
                "amf-d3d11-hip-resident"
                if resident_coordinator is not None
                else None
            ),
            resident_coordinator=resident_coordinator,
            resident_role=(
                "decode-detect" if resident_coordinator is not None else None
            ),
        )
        with reader_context as reader, torch.inference_mode():
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

            frame_batches = reader.frames(seek_ts=seek_ts)
            handoff_mode = _amf_reader_caller_handoff_mode(reader)
            logged_handoff = False
            try:
                for frames, pts_list in timer.timed_iter(frame_batches, "decode"):
                    frames = _handoff_reader_batch_to_caller(
                        frames,
                        role="decode-detect",
                        mode=handoff_mode,
                    )
                    if handoff_mode != "off" and not logged_handoff:
                        log.info(
                            "AMF reader caller handoff: role=decode-detect mode=%s",
                            handoff_mode,
                        )
                        logged_handoff = True
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
            finally:
                close_batches = getattr(frame_batches, "close", None)
                if callable(close_batches):
                    close_batches()
                if progress is not None and close_progress:
                    progress.close(ensure_completed_bar=True)
    except BaseException as e:
        record_worker_error("decode", e, error_holder, cancel_event)
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
        record_worker_error("primary", e, error_holder, cancel_event)
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
        record_worker_error("secondary", e, error_holder, cancel_event)
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
    resident_coordinator: object | None = None,
) -> None:
    timer = LoopTimer("blend-encode")
    try:
        torch.cuda.set_device(device)

        with _PtsAlignedFrameReader(
            input_video=input_video,
            batch_size=batch_size,
            device=device,
            metadata=metadata,
            frame_stride=frame_stride,
            seek_ts=seek_ts,
            cancel_event=cancel_event,
            resident_coordinator=resident_coordinator,
        ) as reader2:
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
                    original_frame = reader2.read_exact(meta.pts)

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
        record_worker_error("blend-encode", e, error_holder, cancel_event)
    finally:
        log.info(timer.summary())


def _estimate_start_frame(metadata, seek_ts: float) -> int:
    return int(seek_ts * metadata.video_fps)


_ASYNC_POLL_TIMEOUT = 0.05
_ASYNC_CANCEL_JOIN_TIMEOUT = 5.0
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
    set_cancel_event = getattr(restorer, "set_cancel_event", None)
    if callable(set_cancel_event):
        set_cancel_event(cancel_event)
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
            if not cancel_event.is_set():
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

    try:
        while not push_done.is_set():
            if cancel_event.is_set():
                break
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
        if pusher_error:
            raise pusher_error[0]
        if not cancel_event.is_set():
            restorer.flush_all()
            for _ in range(100):
                if cancel_event.is_set() or not pending_prs:
                    break
                _forward_completed()
                if pending_prs:
                    cancel_event.wait(_ASYNC_POLL_TIMEOUT)
    except BaseException:
        if not cancel_event.is_set():
            cancel_event.set()
            raise
    finally:
        if cancel_event.is_set():
            cancel_restorer = getattr(restorer, "cancel", None)
            if callable(cancel_restorer):
                try:
                    if cancel_restorer() is False:
                        log.warning("[secondary] async restorer cancellation timed out")
                except BaseException:
                    log.exception("[secondary] async restorer cancellation failed")
            pusher_thread.join(timeout=_ASYNC_CANCEL_JOIN_TIMEOUT)
            if pusher_thread.is_alive():
                log.warning("[secondary] async pusher remained alive after cancellation timeout")
        else:
            pusher_thread.join()
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
        # The async helper cancels its pusher before re-raising a genuine
        # failure; do not mistake that cleanup for a user cancellation.
        log.exception("[secondary-async] thread crashed")
        if not error_holder:
            error_holder.append(e)
        cancel_event.set()
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
    encode_heartbeat: list[float | None] | None = None,
    poll: Callable[[], bool] | None = None,
    resident_coordinator: object | None = None,
    recycle_on_host_memory_pressure: bool = False,
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
    vendor = vendor_for_device(device)
    amd = vendor is AcceleratorVendor.AMD
    memory_geometry = dict(
        frame_width=int(metadata.video_width), frame_height=int(metadata.video_height),
        batch_size=int(pipeline.batch_size), ten_bit=bool(metadata.is_10bit), amd=amd,
    )
    owned_vram_options = {}
    if os.name == "nt" and amd and os.environ.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") == "1":
        from jasna.windows_global_vram import create_windows_hip_vram_reader

        def required_vram_failure(error):
            if not error_holder:
                error_holder.append(error)
            cancel_event.set()
            log.error("Required whole-card VRAM monitoring failed", exc_info=(
                type(error), error, error.__traceback__,
            ))

        owned_vram_options = dict(
            system_vram_reader_factory=lambda: create_windows_hip_vram_reader(device),
            on_system_vram_error=required_vram_failure,
        )
    vram_offloader = VramOffloader(
        device=device, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        safetynet=restoration_vram_safetynet(**memory_geometry),
        system_vram_startup_budget=(restoration_vram_startup_budget(**memory_geometry) if amd else None),
        host_memory_limit_bytes=(default_host_memory_limit_bytes() if amd and is_isolated_video_job() else None),
        cancel_event=cancel_event,
        terminate_on_encode_stall=amd and is_isolated_video_job(),
        encode_stall_timeout_seconds=amf_encoder_stall_timeout_seconds(),
        **owned_vram_options,
    )
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

    def worker_thread(*, target, name, daemon=True):
        # Also cover failures before a loop enters its own try block (for
        # example argument binding). Peers must never wait forever for a
        # sentinel from a worker that failed to start its loop.
        def guarded_target():
            try:
                target()
            except BaseException as worker_error:
                record_worker_error(name, worker_error, error_holder, cancel_event)

        return threading.Thread(target=guarded_target, name=name, daemon=daemon)

    threads = [
        worker_thread(
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
                close_progress=False,
                resident_coordinator=resident_coordinator,
            ),
            name="DecodeDetect", daemon=True,
        ),
        worker_thread(
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
        worker_thread(target=secondary_target, name="SecondaryRestore", daemon=True),
        worker_thread(
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
                resident_coordinator=resident_coordinator,
            ),
            name="BlendEncode", daemon=True,
        ),
    ]
    started_threads: list[threading.Thread] = []
    try:
        vram_offloader.start()
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        while any(thread.is_alive() for thread in started_threads):
            if poll is not None and poll():
                cancel_event.set()
            if cancel_event.wait(0.05):
                break
    except BaseException as error:
        if not error_holder:
            error_holder.append(error)
        cancel_event.set()
    finally:
        try:
            wait_for_worker_threads(started_threads, queues, cancel_event)
        finally:
            try:
                vram_offloader.stop()
            except BaseException as error:
                if not error_holder:
                    error_holder.append(error)
                else:
                    log.exception("VRAM monitor cleanup failed after a pipeline failure")

    error = error_holder[0] if error_holder else None
    pipeline._last_pass_vram_stats = vram_offloader.stats
    if error is None and getattr(vram_offloader, "host_memory_pressure", False) is True:
        message = "Host memory pressure stopped the isolated worker before the operating system OOM killer"
        error = (
            NativeWorkerRecycleRequested(message, reason="native_pressure")
            if recycle_on_host_memory_pressure and is_isolated_video_job()
            else HostMemoryPressureError(message)
        )
    try:
        if resident_coordinator is not None:
            resident_coordinator.decoder_backend.collect_telemetry()
            resident_coordinator.decoder_backend.validate_telemetry()
            resident_coordinator.validate_encoder_telemetry()
    except BaseException as cleanup_error:
        if error is None:
            error = cleanup_error
        else:
            log.exception("Resident telemetry validation failed after a pipeline failure")
    finally:
        del queues, clip_queue, secondary_queue, encode_queue, metadata_queue
        del blend_buffer, crop_buffers, threads, started_threads
        gc.collect()
        for cleanup in (empty_cache, ipc_collect):
            try:
                cleanup(device)
            except BaseException as cleanup_error:
                if error is None:
                    error = cleanup_error
                else:
                    log.exception("GPU cache cleanup failed after a pipeline failure")
    return error


def _drain(queues) -> None:
    for pipeline_queue in queues:
        try:
            while True:
                pipeline_queue.get_nowait()
        except Empty:
            continue

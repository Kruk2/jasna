from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable, Sequence
from queue import Empty, Queue
from typing import Any, Protocol

import torch

from jasna.accelerator import current_stream
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
from jasna.tracking.scene_detector import SceneCutDetector

log = logging.getLogger(__name__)

AMF_READER_CALLER_HANDOFF_ENV = "JASNA_AMF_READER_CALLER_HANDOFF"
_AMF_READER_CALLER_HANDOFF_MODES = frozenset(
    {"off", "record-stream", "record-stream-clone-batch"}
)


def _amf_reader_caller_handoff_mode(reader: NvidiaVideoReader) -> str:
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
        self._reader: NvidiaVideoReader | None = None
        self._frames = None
        self._retry_backend: str | None = None
        self._stream_start_pts = int(getattr(metadata, "start_pts", 0) or 0)

    @staticmethod
    def _flat_frames(
        reader: NvidiaVideoReader,
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
        reader = NvidiaVideoReader(
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

        reader_context = NvidiaVideoReader(
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
                    original_frame = reader2.read_exact(meta.pts)

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

            if vram_offloader is not None:
                vram_offloader.pause_stall_check()

    except BaseException as e:
        record_worker_error("blend-encode", e, error_holder, cancel_event)
    finally:
        log.info(timer.summary())


def _estimate_start_frame(metadata, seek_ts: float) -> int:
    return int(seek_ts * metadata.video_fps)

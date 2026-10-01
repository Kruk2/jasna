from __future__ import annotations

import bisect
import gc
import logging
import os
import threading
import time
from fractions import Fraction
from pathlib import Path
from queue import Empty, Queue

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue

import psutil
import torch

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.media import UnsupportedColorspaceError, get_video_meta_data
from jasna.media.video_encoder import NvidiaVideoEncoder, resolve_hevc_smart_render_vui
from jasna.media.windows_d3d11_hip_resident import (
    WindowsD3D11HipResidentCoordinator,
    windows_d3d11_hip_resident_requested,
)
from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.native_worker import (
    HostMemoryPressureError,
    NativeWorkerRecycleRequested,
    amf_encoder_stall_timeout_seconds,
    amf_render_session_seconds,
    is_isolated_video_job,
)
from jasna.media.splice import (
    SmartRenderCompatibilityError,
    SplicePlan,
    SpliceSpan,
    build_splice_plan,
    create_copy_fragment,
    create_normalized_copy_fragment,
    hevc_copy_fragment_timeline_matches_source,
    mux_fragments_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
    split_render_spans,
    validate_hevc_fragment_parameter_sets,
    validate_smart_render,
)
from jasna.mosaic.detection_registry import build_detection_model
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import FrameMeta, PrimaryRestoreResult, SecondaryLoopStats, _SENTINEL
from jasna.pipeline_threads import (
    blend_encode_loop,
    decode_detect_loop,
    primary_restore_loop,
    record_worker_error,
    secondary_restore_loop,
    wait_for_worker_threads,
)
from jasna.progressbar import Progressbar
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.segments import SegmentRange
from jasna.smart_render_workspace import SmartRenderWorkspace, workspace_signature
from jasna.vram_offloader import (
    VramOffloader,
    VramStats,
    default_host_memory_limit_bytes,
    restoration_vram_safetynet,
    restoration_vram_startup_budget,
)
from jasna.vr180 import (
    SbsDetectionAdapter,
    resolve_vr_mode,
)
from jasna.vr_projection import build_vr_projector

log = logging.getLogger(__name__)


def _dual_gop_smart_encoder_settings(
    settings: dict[str, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    """Keep the proven dual-session GOP contract after Smart Render matching."""

    if not enabled:
        return settings
    from jasna.media.dual_gop_encoder import AMD_DUAL_GOP_SIZE

    resolved = dict(settings)
    resolved.update({"g": AMD_DUAL_GOP_SIZE, "bf": 0})
    return resolved


def _bounded_amf_render_session_seconds(
    *,
    metadata,
    codec: str,
    vendor: AcceleratorVendor,
    batch_size: int,
    dual_gop_enabled: bool,
    retarget_high_fps: bool = False,
) -> float | None:
    """Bound affected Linux AMF render spans before native state can stall."""

    if not is_isolated_video_job():
        return None
    if vendor is not AcceleratorVendor.AMD:
        return None
    normalized_codec = str(codec).casefold()
    width = int(getattr(metadata, "video_width", 0))
    height = int(getattr(metadata, "video_height", 0))
    ten_bit = bool(getattr(metadata, "is_10bit", False))
    if not (
        normalized_codec == "hevc"
        and not bool(retarget_high_fps)
        # Main10 keeps its established corruption boundary.  Main8 needs the
        # same boundary only when two persistent AMF encoders are requested:
        # otherwise a long render span can carry the native working set into
        # later decoder/encoder groups and exhaust whole-card headroom.
        and (ten_bit or bool(dual_gop_enabled))
        and width == 8192
        and height == 4096
        and int(batch_size) == 4
    ):
        return None
    return amf_render_session_seconds()


class _OfflineFrameWriter:
    def __init__(
        self,
        encoder_ctx: NvidiaVideoEncoder,
        encode_heartbeat: list[float | None],
        *,
        amd_dual_gop_encode: bool = False,
    ):
        self._encoder_ctx = encoder_ctx
        self._encode_heartbeat = encode_heartbeat
        self._entered = False
        self._dual_gop = None

        from jasna.media.dual_gop_encoder import (
            AmdDualGopFrameWriter,
            use_dual_gop_writer,
        )

        if use_dual_gop_writer(
            encoder_ctx,
            enabled=bool(amd_dual_gop_encode),
        ):
            self._dual_gop = AmdDualGopFrameWriter(encoder_ctx)

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        # Arm the stall watchdog only when the encoder receives its first
        # write.  Updating before the call also preserves diagnostics if the
        # very first native encode submission itself blocks.
        self._encode_heartbeat[0] = time.monotonic()
        if self._dual_gop is not None:
            self._dual_gop.write(frame, pts, apply_lut=apply_lut)
            self._encode_heartbeat[0] = time.monotonic()
            return
        if not self._entered:
            self._encoder_ctx.__enter__()
            self._entered = True
        self._encoder_ctx.encode(frame, pts, apply_lut=apply_lut)
        self._encode_heartbeat[0] = time.monotonic()

    def after_write(self, frames_written: int) -> None:
        pass

    def close(self, *, abort: bool = False) -> None:
        if self._dual_gop is not None:
            if abort:
                self._dual_gop.abort()
            else:
                self._dual_gop.close()
            self._dual_gop = None
            return
        if self._entered:
            self._encoder_ctx.__exit__(None, None, None)
            self._entered = False


class Pipeline:
    def __init__(
        self,
        *,
        input_video: Path,
        output_video: Path,
        detection_model_name: str,
        detection_model_path: Path,
        detection_score_threshold: float,
        restoration_pipeline: RestorationPipeline,
        codec: str,
        encoder_settings: dict[str, object],
        batch_size: int,
        device: torch.device,
        max_clip_size: int,
        temporal_overlap: int,
        max_detection_gap: int,
        min_detection_duration: int,
        enable_crossfade: bool = True,
        scene_detection: bool = True,
        vr_mode: str = "auto",
        vr_projection: str = "auto",
        fp16: bool,
        disable_progress: bool = False,
        progress_callback: callable | None = None,
        lut_path: str | Path | None = None,
        sharpen_strength: float = 0.0,
        retarget_high_fps: bool = False,
        auto_source_rate: bool = False,
        amd_dual_gop_encode: bool = False,
        fmp4: bool = False,
        segments: tuple[SegmentRange, ...] | None = None,
        splice_plan: SplicePlan | None = None,
        effect_ranges: tuple[tuple[int, int], ...] | None = None,
        working_dir: Path | None = None,
        workspace_output: Path | None = None,
        processing_signature: dict[str, object] | None = None,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.working_dir = working_dir
        # Full-video Linux AMD retries render into a per-attempt staging path
        # so the final publish remains atomic.  Keep the durable workspace
        # identity bound to the canonical destination instead of that UUID-
        # suffixed staging path; otherwise every recycled worker starts a new
        # workspace and renders fragment zero again.
        self.workspace_output = workspace_output
        self.codec = str(codec)
        self.encoder_settings = dict(encoder_settings)
        self.batch_size = int(batch_size)
        self.device = device
        self.max_clip_size = int(max_clip_size)
        self.temporal_overlap = int(temporal_overlap)
        self.max_detection_gap = int(max_detection_gap)
        self.min_detection_duration = int(min_detection_duration)
        self.enable_crossfade = bool(enable_crossfade)
        self.scene_detection = bool(scene_detection)
        self.vr_mode = str(vr_mode)
        self.vr_projection = str(vr_projection)
        self.detection_model_path = Path(detection_model_path)
        self.processing_signature = dict(processing_signature or {
            "detection_model": str(detection_model_name),
            "detection_score_threshold": float(detection_score_threshold),
            "batch_size": int(batch_size),
            "max_clip_size": int(max_clip_size),
            "temporal_overlap": int(temporal_overlap),
            "max_detection_gap": int(max_detection_gap),
            "min_detection_duration": int(min_detection_duration),
            "enable_crossfade": bool(enable_crossfade),
            "scene_detection": bool(scene_detection),
            "vr_mode": str(vr_mode),
            "vr_projection": str(vr_projection),
            "fp16": bool(fp16),
            "sharpen_strength": float(sharpen_strength),
        })

        self.detection_model = build_detection_model(
            detection_model_name,
            detection_model_path,
            batch_size=self.batch_size,
            device=self.device,
            score_threshold=float(detection_score_threshold),
            fp16=bool(fp16),
        )
        self.restoration_pipeline = restoration_pipeline
        self.disable_progress = bool(disable_progress)
        self.progress_callback = progress_callback
        self.lut_path = lut_path
        self.sharpen_strength = float(sharpen_strength)
        self.retarget_high_fps = bool(retarget_high_fps)
        self.auto_source_rate = bool(auto_source_rate)
        self.amd_dual_gop_encode = bool(amd_dual_gop_encode)
        self.fmp4 = bool(fmp4)
        self.segments = tuple(segments) if segments else None
        self.splice_plan = splice_plan
        self.effect_ranges = tuple(effect_ranges) if effect_ranges else None
        self._vr_resolution = None
        self._vr_projector = None
        self._job_detection_model = self.detection_model
        self._cancel_event = threading.Event()
        self.completed = False

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Ask the running pipeline to stop as soon as the worker threads notice."""
        self._cancel_event.set()

    def configure_vr(self, metadata) -> None:
        self._vr_resolution = resolve_vr_mode(
            self.vr_mode,
            metadata,
            self.input_video,
            projection=self.vr_projection,
        )
        self._job_detection_model = (
            SbsDetectionAdapter(self.detection_model)
            if self._vr_resolution.is_sbs
            else self.detection_model
        )
        self._vr_projector = (
            build_vr_projector(
                self._vr_resolution.projection,
                eye_width=int(metadata.video_width) // 2,
                height=int(metadata.video_height),
                device=self.device,
            )
            if self._vr_resolution.is_sbs
            else None
        )

    def close(self) -> None:
        if hasattr(self, "detection_model") and self.detection_model is not None:
            if hasattr(self.detection_model, "close"):
                self.detection_model.close()
            self.detection_model = None
        self.restoration_pipeline = None

    _ASYNC_POLL_TIMEOUT = 0.05
    _ASYNC_CANCEL_JOIN_TIMEOUT = 5.0

    @staticmethod
    def _earliest_blocking_seqs(pending_prs: dict[int, PrimaryRestoreResult]) -> set[int] | None:
        if not pending_prs:
            return None
        earliest_frame = min(
            pr.start_frame + pr.keep_start for pr in pending_prs.values()
        )
        return {
            seq for seq, pr in pending_prs.items()
            if pr.start_frame + pr.keep_start <= earliest_frame <= pr.start_frame + pr.keep_end - 1
        }

    _FLUSH_DELAY = 2.0
    _FLUSH_RETRY_TIMEOUT = 5.0

    def _run_secondary_loop(
        self,
        secondary_queue: FrameQueue,
        encode_queue: FrameQueue,
        debug_memory: PipelineDebugMemoryLogger | None = None,
        clip_queue: FrameQueue | None = None,
        primary_idle_event: threading.Event | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SecondaryLoopStats:
        restorer: AsyncSecondaryRestorer = self.restoration_pipeline.secondary_restorer  # type: ignore[assignment]
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
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    try:
                        item = secondary_queue.get(timeout=self._ASYNC_POLL_TIMEOUT)
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
                # A push interrupted by user Stop is a cancellation outcome,
                # not a worker root cause.  Real errors observed before Stop
                # remain eligible for first-error propagation below.
                if cancel_event is None or not cancel_event.is_set():
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
                sr = self.restoration_pipeline.build_secondary_result(pr, tensors)
                encode_queue.put(sr, frame_count=sr.keep_end)
                if debug_memory is not None:
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.track_id} frames={sr.frame_count}",
                    )
                forwarded += 1
                clips_popped += 1
            return forwarded

        def _no_clips_incoming() -> bool:
            if primary_idle_event is None or clip_queue is None:
                return False
            return primary_idle_event.is_set() and clip_queue.qsize() == 0

        pusher_thread = threading.Thread(target=_pusher, daemon=True)
        pusher_thread.start()

        starvation_count = 0
        starvation_seconds = 0.0
        starvation_start: float | None = None

        while not push_done.is_set():
            if cancel_event is not None and cancel_event.is_set():
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
                and now - last_push_time > self._FLUSH_DELAY
            ):
                if starvation_start is None:
                    starvation_start = now
                target_seqs = self._earliest_blocking_seqs(dict(pending_prs))
                log.debug("[secondary] starvation flush target_seqs=%s", target_seqs)
                if restorer.flush_pending(target_seqs=target_seqs):
                    flushed_since_last_push = True
                    last_flush_time = now
                starvation_count += 1
            elif (
                flushed_since_last_push
                and restorer.has_pending
                and _no_clips_incoming()
                and now - last_flush_time > self._FLUSH_RETRY_TIMEOUT
            ):
                log.warning(
                    "[secondary] flush retry: no clips forwarded for %.0fs after flush, pending=%d",
                    now - last_flush_time, len(pending_prs),
                )
                flushed_since_last_push = False

            time.sleep(self._ASYNC_POLL_TIMEOUT)

        if starvation_start is not None:
            starvation_seconds += time.monotonic() - starvation_start
        cancelled = cancel_event is not None and cancel_event.is_set()
        if cancelled:
            cancel_restorer = getattr(restorer, "cancel", None)
            if callable(cancel_restorer):
                try:
                    stopped = cancel_restorer()
                except BaseException:
                    # User Stop remains authoritative.  Keep an incomplete
                    # third-party shutdown observable in logs without
                    # reclassifying it as a worker failure.
                    log.exception("[secondary] async restorer cancellation failed")
                else:
                    if stopped is False:
                        log.warning("[secondary] async restorer cancellation timed out")
            pusher_thread.join(timeout=self._ASYNC_CANCEL_JOIN_TIMEOUT)
            if pusher_thread.is_alive():
                log.warning("[secondary] async pusher remained alive after cancellation timeout")
        else:
            pusher_thread.join()
        if pusher_error:
            raise pusher_error[0]
        if cancelled:
            return SecondaryLoopStats(
                starvation_flushes=starvation_count,
                starvation_seconds=starvation_seconds,
                pusher_stall_seconds=pusher_stall_seconds,
                clips_pushed=clips_pushed,
                clips_popped=clips_popped,
            )
        restorer.flush_all()
        for _ in range(100):
            if not pending_prs:
                break
            _forward_completed()
            if pending_prs:
                time.sleep(self._ASYNC_POLL_TIMEOUT)
        return SecondaryLoopStats(
            starvation_flushes=starvation_count,
            starvation_seconds=starvation_seconds,
            pusher_stall_seconds=pusher_stall_seconds,
            clips_pushed=clips_pushed,
            clips_popped=clips_popped,
        )

    def _run_pass(
        self,
        *,
        metadata,
        encoder_ctx: NvidiaVideoEncoder,
        progress: Progressbar,
        seek_ts: float | None = None,
        end_pts: int | None = None,
        effect_ranges: tuple[tuple[int, int], ...] | None = None,
        output_frame_count: int | None = None,
        recycle_on_host_memory_pressure: bool = False,
        resident_coordinator: WindowsD3D11HipResidentCoordinator | None = None,
    ) -> VramStats:
        device = self.device
        secondary_workers = max(1, int(self.restoration_pipeline.secondary_num_workers))
        frame_rate = resolve_frame_rate_retarget(
            metadata.video_fps_exact,
            enabled=self.retarget_high_fps,
            measured_fps=metadata.average_fps,
        )
        if output_frame_count is None:
            output_frame_count = frame_rate.output_frame_count(metadata.num_frames)

        clip_queue = FrameQueue(max_frames=self.max_clip_size)
        secondary_queue = FrameQueue(max_frames=self.max_clip_size * secondary_workers)
        encode_queue = FrameQueue(max_frames=self.max_clip_size)
        metadata_queue: Queue[FrameMeta | object] = Queue(maxsize=self.max_clip_size * 5)

        error_holder: list[BaseException] = []
        blend_buffer = BlendBuffer(device=device, vr_projector=self._vr_projector)
        crop_buffers: dict[int, CropBuffer] = {}
        crop_lock = threading.Lock()
        primary_idle_event = threading.Event()
        frame_shape: list[tuple[int, int]] = []

        encode_heartbeat: list[float | None] = [None]
        frame_writer = _OfflineFrameWriter(
            encoder_ctx,
            encode_heartbeat,
            amd_dual_gop_encode=self.amd_dual_gop_encode,
        )
        pipeline_vendor = vendor_for_device(device)
        vram_safetynet = restoration_vram_safetynet(
            frame_width=int(metadata.video_width),
            frame_height=int(metadata.video_height),
            batch_size=self.batch_size,
            ten_bit=bool(metadata.is_10bit),
            amd=pipeline_vendor is AcceleratorVendor.AMD,
        )
        vram_startup_budget = restoration_vram_startup_budget(
            frame_width=int(metadata.video_width),
            frame_height=int(metadata.video_height),
            batch_size=self.batch_size,
            ten_bit=bool(metadata.is_10bit),
            amd=pipeline_vendor is AcceleratorVendor.AMD,
        )
        owned_vram_options = {}
        if (os.name == "nt" and pipeline_vendor is AcceleratorVendor.AMD
                and os.environ.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") == "1"):
            from jasna.windows_global_vram import create_windows_hip_vram_reader

            def required_vram_failure(error):
                # Mandatory telemetry failure is fatal even when another
                # cancellation is already in flight. Publish before logging.
                if not error_holder:
                    error_holder.append(error)
                self._cancel_event.set()
                log.error("Required whole-card VRAM monitoring failed", exc_info=(
                    type(error), error, error.__traceback__,
                ))

            owned_vram_options = dict(
                system_vram_reader_factory=lambda: create_windows_hip_vram_reader(device),
                on_system_vram_error=required_vram_failure,
            )
        vram_offloader = VramOffloader(
            device=device,
            blend_buffer=blend_buffer,
            crop_buffers=crop_buffers,
            crop_lock=crop_lock,
            safetynet=vram_safetynet,
            system_vram_startup_budget=(
                vram_startup_budget
                if pipeline_vendor is AcceleratorVendor.AMD
                else None
            ),
            host_memory_limit_bytes=(
                default_host_memory_limit_bytes()
                if (
                    pipeline_vendor is AcceleratorVendor.AMD
                    and is_isolated_video_job()
                )
                else None
            ),
            cancel_event=self._cancel_event,
            terminate_on_encode_stall=(
                pipeline_vendor is AcceleratorVendor.AMD
                and is_isolated_video_job()
            ),
            encode_stall_timeout_seconds=amf_encoder_stall_timeout_seconds(),
            **owned_vram_options,
        )
        vram_offloader.set_encode_heartbeat(encode_heartbeat)
        vram_offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)

        debug_memory = PipelineDebugMemoryLogger(
            logger=log,
            blend_buffer=blend_buffer,
            clip_queue=clip_queue,
            secondary_queue=secondary_queue,
            encode_queue=encode_queue,
        )

        starvation_stats = SecondaryLoopStats()

        def _async_secondary_thread():
            nonlocal starvation_stats
            try:
                torch.cuda.set_device(device)
                starvation_stats = self._run_secondary_loop(
                    secondary_queue,
                    encode_queue,
                    debug_memory,
                    clip_queue,
                    primary_idle_event,
                    self._cancel_event,
                )
            except BaseException as e:
                record_worker_error(
                    "secondary-async",
                    e,
                    error_holder,
                    self._cancel_event,
                )
            finally:
                encode_queue.put(_SENTINEL)

        use_async_secondary = isinstance(self.restoration_pipeline.secondary_restorer, AsyncSecondaryRestorer)
        if use_async_secondary:
            log.debug("Using async secondary restore path")
            secondary_target = _async_secondary_thread
        else:
            secondary_target = lambda: secondary_restore_loop(
                device=device,
                restoration_pipeline=self.restoration_pipeline,
                secondary_queue=secondary_queue,
                encode_queue=encode_queue,
                error_holder=error_holder,
                debug_memory=debug_memory,
                cancel_event=self._cancel_event,
            )

        threads = [
            threading.Thread(
                target=lambda: decode_detect_loop(
                    input_video=str(self.input_video),
                    batch_size=self.batch_size,
                    device=device,
                    metadata=metadata,
                    detection_model=self._job_detection_model,
                    max_clip_size=self.max_clip_size,
                    temporal_overlap=self.temporal_overlap,
                    max_detection_gap=self.max_detection_gap,
                    min_detection_duration=self.min_detection_duration,
                    enable_crossfade=self.enable_crossfade,
                    scene_detection=self.scene_detection,
                    blend_buffer=blend_buffer,
                    crop_buffers=crop_buffers,
                    clip_queue=clip_queue,
                    metadata_queue=metadata_queue,
                    error_holder=error_holder,
                    frame_shape=frame_shape,
                    progress=progress,
                    close_progress=False,
                    seek_ts=seek_ts,
                    end_pts=end_pts,
                    effect_ranges=effect_ranges,
                    debug_memory=debug_memory,
                    frame_stride=frame_rate.frame_stride,
                    output_frame_count=output_frame_count,
                    output_fps=float(frame_rate.output_fps),
                    vr_mode=self._vr_resolution.resolved,
                    vr_projector=self._vr_projector,
                    cancel_event=self._cancel_event,
                    resident_coordinator=resident_coordinator,
                ),
                name="DecodeDetect", daemon=True,
            ),
            threading.Thread(
                target=lambda: primary_restore_loop(
                    device=device,
                    restoration_pipeline=self.restoration_pipeline,
                    clip_queue=clip_queue,
                    secondary_queue=secondary_queue,
                    error_holder=error_holder,
                    primary_idle_event=primary_idle_event,
                    debug_memory=debug_memory,
                    cancel_event=self._cancel_event,
                ),
                name="PrimaryRestore", daemon=True,
            ),
            threading.Thread(target=secondary_target, name="SecondaryRestore", daemon=True),
            threading.Thread(
                target=lambda: blend_encode_loop(
                    input_video=str(self.input_video),
                    batch_size=self.batch_size,
                    device=device,
                    metadata=metadata,
                    blend_buffer=blend_buffer,
                    encode_queue=encode_queue,
                    metadata_queue=metadata_queue,
                    error_holder=error_holder,
                    frame_writer=frame_writer,
                    vram_offloader=vram_offloader,
                    frame_stride=frame_rate.frame_stride,
                    seek_ts=seek_ts,
                    cancel_event=self._cancel_event,
                    resident_coordinator=resident_coordinator,
                ),
                name="BlendEncode", daemon=True,
            ),
        ]
        frame_writer_error: BaseException | None = None
        try:
            vram_offloader.start()
            for t in threads:
                t.start()
            wait_for_worker_threads(
                threads,
                (clip_queue, secondary_queue, encode_queue, metadata_queue),
                self._cancel_event,
            )
        except BaseException as error:
            if not error_holder:
                error_holder.append(error)
            self._cancel_event.set()
            wait_for_worker_threads(
                threads,
                (clip_queue, secondary_queue, encode_queue, metadata_queue),
                self._cancel_event,
            )
        finally:
            try:
                vram_offloader.stop()
            except BaseException as error:
                if not error_holder:
                    error_holder.append(error)
                self._cancel_event.set()
        try:
            frame_writer.close(
                abort=bool(error_holder) or self._cancel_event.is_set(),
            )
        except BaseException as error:
            # Native encoder teardown can fail after all pipeline threads have
            # already stopped.  Keep the failure visible, but continue the
            # common allocator/queue cleanup below so a failed session cannot
            # poison the next bounded fragment or queued job.
            frame_writer_error = error
            self._cancel_event.set()
            log.error(
                "Frame writer teardown failed; continuing isolated cleanup",
                exc_info=True,
            )
            if not error_holder:
                error_holder.append(error)
        if resident_coordinator is not None:
            try:
                resident_stats = resident_coordinator.close()
                log.info("Windows D3D11/HIP resident audit: %s", resident_stats)
            except BaseException as error:
                if not error_holder:
                    error_holder.append(error)

        _process = psutil.Process(os.getpid())
        try:
            free, total = torch.cuda.mem_get_info(device)
            vram_used = total - free
            log.info("VRAM usage at end — %.1f MiB", vram_used / (1024 ** 2))
        except Exception:
            log.debug("Could not read end-of-run VRAM usage", exc_info=True)
        try:
            rss = _process.memory_info().rss
            log.info("RAM usage at end — %.1f MiB", rss / (1024 ** 2))
        except Exception:
            log.debug("Could not read end-of-run RAM usage", exc_info=True)

        ss = starvation_stats
        if ss.clips_pushed > 0 or ss.clips_popped > 0:
            log.info(
                "Secondary — clips: %d pushed / %d popped, pusher stall: %.1fs, starvation flushes: %d (%.1fs)",
                ss.clips_pushed, ss.clips_popped, ss.pusher_stall_seconds, ss.starvation_flushes, ss.starvation_seconds,
            )

        err = error_holder[0] if error_holder else frame_writer_error
        if err is not None and frame_writer_error is not None and err is not frame_writer_error:
            log.error(
                "Frame writer teardown also failed after the primary pipeline error: %s",
                frame_writer_error,
            )
        if err is not None:
            err.__traceback__ = None

        del clip_queue, secondary_queue, encode_queue, metadata_queue
        del blend_buffer, crop_buffers
        del error_holder, threads
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(self.device)

        if err is not None:
            raise err
        if getattr(vram_offloader, "host_memory_pressure", False):
            message = (
                "Host memory pressure forced an isolated worker shutdown before "
                "the operating system OOM killer could terminate it"
            )
            if recycle_on_host_memory_pressure and is_isolated_video_job():
                raise NativeWorkerRecycleRequested(
                    message,
                    reason="native_pressure",
                )
            raise HostMemoryPressureError(message)
        return vram_offloader.stats

    def _validate_metadata(self, metadata) -> None:
        from av.video.reformatter import Colorspace as AvColorspace

        if metadata.color_space not in (
            AvColorspace.ITU709,
            AvColorspace.ITU601,
            AvColorspace.BT2020,
        ):
            raise UnsupportedColorspaceError(
                f"Unsupported color space: {metadata.color_space!r} in {self.input_video.name}. "
                "Only BT.709, BT.601, and BT.2020 non-constant-luminance are supported."
            )

    def _run_full(self, metadata) -> None:
        frame_rate = resolve_frame_rate_retarget(
            metadata.video_fps_exact,
            enabled=self.retarget_high_fps,
            measured_fps=metadata.average_fps,
        )
        if frame_rate.active:
            log.info(
                "Retargeting frame rate: %s fps -> %s fps (keeping every %dth frame)",
                frame_rate.source_fps,
                frame_rate.output_fps,
                frame_rate.frame_stride,
            )
        elif frame_rate.rate_mismatch:
            log.warning(
                "Frame-rate retargeting skipped: the container reports %s fps but the measured "
                "frame rate is %.3f fps; keeping the source rate",
                frame_rate.source_fps,
                metadata.average_fps,
            )
        elif self.retarget_high_fps:
            log.info(
                "Frame-rate retargeting requested, but %s fps is not a supported source rate; keeping source rate",
                frame_rate.source_fps,
            )
        output_frame_count = frame_rate.output_frame_count(metadata.num_frames)
        progress = Progressbar(
            total_frames=output_frame_count,
            video_fps=float(frame_rate.output_fps),
            disable=self.disable_progress,
            callback=self.progress_callback,
        )
        bounded_amf_session_seconds = _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec=self.codec,
            vendor=vendor_for_device(self.device),
            batch_size=int(getattr(self, "batch_size", 0) or 0),
            dual_gop_enabled=bool(
                getattr(self, "amd_dual_gop_encode", False)
            ),
            retarget_high_fps=bool(self.retarget_high_fps),
        )
        if bounded_amf_session_seconds is not None:
            try:
                self._run_bounded_full(
                    metadata,
                    frame_rate=frame_rate,
                    progress=progress,
                    output_frame_count=output_frame_count,
                    max_duration_seconds=bounded_amf_session_seconds,
                )
            finally:
                progress.close(ensure_completed_bar=True)
            return
        if self.fmp4 and self.output_video.suffix.lower() not in {".mp4", ".mov"}:
            log.info(
                "Fragmented MP4 has no effect on %s output; it is already playable while it grows",
                self.output_video.suffix,
            )
        resident_coordinator = None
        try:
            if windows_d3d11_hip_resident_requested():
                if self.amd_dual_gop_encode:
                    raise RuntimeError(
                        "The explicit Windows D3D11/HIP resident route cannot be "
                        "combined with dual-GOP encoding"
                    )
                resident_coordinator = WindowsD3D11HipResidentCoordinator(
                    device=self.device,
                    metadata=metadata,
                    batch_size=self.batch_size,
                    output_codec=self.codec,
                )
            encoder_ctx = NvidiaVideoEncoder(
                str(self.output_video),
                device=self.device,
                metadata=metadata,
                codec=self.codec,
                encoder_settings=self.encoder_settings,
                lut_path=self.lut_path,
                sharpen_strength=self.sharpen_strength,
                output_fps=frame_rate.output_fps,
                match_input_bit_depth=True,
                auto_source_rate=getattr(self, "auto_source_rate", False),
                prefer_amf_host_native=bool(
                    getattr(self, "amd_dual_gop_encode", False)
                ),
                fmp4=self.fmp4,
                resident_coordinator=resident_coordinator,
            )
            self._run_pass(
                metadata=metadata,
                encoder_ctx=encoder_ctx,
                progress=progress,
                effect_ranges=self.effect_ranges,
                output_frame_count=output_frame_count,
                resident_coordinator=resident_coordinator,
            )
        finally:
            try:
                if resident_coordinator is not None:
                    resident_coordinator.close()
            finally:
                progress.close(ensure_completed_bar=True)

    def _run_bounded_full(
        self,
        metadata,
        *,
        frame_rate,
        progress: Progressbar,
        output_frame_count: int,
        max_duration_seconds: float,
    ) -> None:
        """Render a full video in short AMF epochs and assemble the fragments.

        Linux AMF/Vulkan resources are process-global and can retain native
        surfaces beyond Python object lifetime.  A single full-video pass can
        therefore grow the host/native working set for hours even though each
        queue is bounded.  Closed-GOP render fragments put a hard lifetime
        boundary around every decoder and dual-encoder session while keeping
        the final output frame/PTS contract unchanged.
        """

        index = probe_keyframes(self.input_video, metadata)
        plan = split_render_spans(
            SplicePlan(
                index=index,
                spans=(
                    SpliceSpan(
                        "render",
                        index.start_pts,
                        index.end_pts,
                        tuple(self.effect_ranges or ()),
                    ),
                ),
                segments=(),
            ),
            video_fps=frame_rate.output_fps,
            max_duration_seconds=max_duration_seconds,
        )
        self._validate_bounded_full_plan(plan)
        log.info(
            "Bounded Linux AMD full render sessions: %.1fs maximum, "
            "%d render fragments",
            max_duration_seconds,
            len(plan.render_spans),
        )
        work_root = self.working_dir or self.output_video.parent
        work_root.mkdir(parents=True, exist_ok=True)
        # A bounded full render is resumable.  The isolated worker is
        # recycled after each completed fragment so process-global AMF/Vulkan
        # allocations cannot accumulate across the source video; the next
        # child reopens this signature-bound workspace and skips its completed
        # prefix instead of rendering it again.
        resolved_projection = (
            self._vr_resolution.projection
            if getattr(self, "_vr_resolution", None) is not None
            else getattr(self, "vr_projection", "auto")
        )
        workspace_output = getattr(self, "workspace_output", None) or self.output_video
        signature = workspace_signature(
            source=self.input_video,
            output=workspace_output,
            plan=plan,
            processing={
                **dict(getattr(self, "processing_signature", {}) or {}),
                "route": "bounded-full",
                "amf_session_seconds": float(max_duration_seconds),
                # Bump the durable workspace identity along with the decode
                # sentinel fix below.  Fragments produced by the old bug are
                # structurally valid but contain no restoration, so they must
                # never be reused after upgrading this code.
                "bounded_full_effect_ranges_semantics": "all-frames-none-v1",
            },
            model_files={
                "detection": getattr(self, "detection_model_path", None),
                "restoration": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "restorer",
                        None,
                    ),
                    "checkpoint_path",
                    None,
                ),
                "secondary": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "secondary_restorer",
                        None,
                    ),
                    "engine_path",
                    None,
                ),
                "lut": self.lut_path,
            },
            codec=self.codec,
            encoder_settings=self.encoder_settings,
            resolved_projection=resolved_projection,
        )
        workspace = SmartRenderWorkspace.open(
            work_root,
            output=workspace_output,
            signature=signature,
        )
        fragments: list[tuple[Path, float]] = []
        succeeded = False
        try:
            fragment_suffix = ".ts" if self.codec in {"h264", "hevc"} else ".mkv"
            for span_index, span in enumerate(plan.render_spans):
                if self._cancel_event.is_set():
                    return
                duration = float((span.end_pts - span.start_pts) * index.time_base)
                expected_frames = max(
                    1,
                    round(
                        Fraction(span.end_pts - span.start_pts)
                        * Fraction(frame_rate.output_fps)
                        * index.time_base
                    ),
                )
                reusable = workspace.reusable_fragment(span_index)
                if reusable is not None:
                    log.info(
                        "Reusing bounded full-render fragment %s from %s",
                        span_index,
                        reusable,
                    )
                    fragments.append((reusable, duration))
                    progress.mark_completed(expected_frames)
                    continue

                workspace.mark_running(span_index)
                raw = workspace.raw_path(span_index)
                normalized = workspace.fragment_path(span_index, fragment_suffix)
                raw.unlink(missing_ok=True)
                normalized.unlink(missing_ok=True)
                encoder_ctx = NvidiaVideoEncoder(
                    str(raw),
                    device=self.device,
                    metadata=metadata,
                    codec=self.codec,
                    encoder_settings=self.encoder_settings,
                    lut_path=self.lut_path,
                    sharpen_strength=self.sharpen_strength,
                    output_fps=frame_rate.output_fps,
                    mux_audio=False,
                    pts_origin=span.start_pts,
                    match_input_bit_depth=True,
                    smart_fragment=True,
                    auto_source_rate=getattr(self, "auto_source_rate", False),
                    prefer_amf_host_native=bool(
                        getattr(self, "amd_dual_gop_encode", False)
                    ),
                )
                self._run_pass(
                    metadata=metadata,
                    encoder_ctx=encoder_ctx,
                    progress=progress,
                    seek_ts=index.seconds_for_pts(span.start_pts),
                    end_pts=span.end_pts,
                    # A bounded-full render is still a full-video pass.  The
                    # empty tuple stored on its SpliceSpan is only plan
                    # metadata; passing it to decode_detect_loop would mean
                    # "no selected frames" and silently turn every fragment
                    # into a no-op.  None is the sentinel for all frames;
                    # preserve an explicitly supplied non-empty range.
                    effect_ranges=span.effect_ranges or None,
                    output_frame_count=expected_frames,
                    recycle_on_host_memory_pressure=True,
                )
                if self._cancel_event.is_set():
                    return
                normalize_fragment(
                    raw,
                    normalized,
                    codec=self.codec,
                    # Full bounded renders contain no copied source packets;
                    # each fragment starts at an encoder PTS origin of zero.
                    # Applying the source decoder's B-frame delay here would
                    # create an artificial DTS offset (and negative DTS) on
                    # the independently encoded, closed-GOP fragment.
                    decode_delay=Fraction(0, 1),
                )
                workspace.mark_complete(span_index, normalized)
                raw.unlink(missing_ok=True)
                fragments.append((normalized, duration))

                # AMF/Vulkan allocations are process-global on the affected
                # Linux AMD runtime.  Reopening an encoder in this process is
                # not a hard lifetime boundary: the previous run reached
                # 24.48 GiB whole-card usage after only three fragments.  End
                # the isolated worker after every completed fragment and let
                # the parent resume from the workspace in a fresh process.
                if (
                    is_isolated_video_job()
                    and span_index + 1 < len(plan.render_spans)
                ):
                    message = (
                        "Linux AMD 8K HEVC bounded full fragment "
                        f"{span_index} completed; recycling the isolated worker "
                        "before the next AMF decoder/encoder session"
                    )
                    log.warning(message)
                    raise NativeWorkerRecycleRequested(
                        message,
                        reason="amf_session_limit",
                    )

            if self._cancel_event.is_set():
                return
            if self.codec == "hevc":
                validate_hevc_fragment_parameter_sets(
                    [(fragment, "render") for fragment, _duration in fragments]
                )
            mux_fragments_final_output(
                fragments,
                self.input_video,
                self.output_video,
                manifest=workspace.path / "fragments.ffconcat",
                codec=self.codec,
            )
            succeeded = True
        finally:
            if succeeded:
                workspace.cleanup()
            else:
                log.error(
                    "Preserving failed/resumable bounded full-render workspace: %s",
                    workspace.path,
                )

    @staticmethod
    def _validate_bounded_full_plan(plan: SplicePlan) -> None:
        """Reject fragment plans that could silently drop or duplicate PTS.

        The bounded full route is intentionally all-render: unlike Smart
        Render it has no copied spans that can bridge a gap.  Keep an explicit
        contiguous PTS check next to the splitter so a future change to
        keyframe probing or duration rounding cannot create a valid-looking
        output with a missing boundary frame.
        """

        spans = tuple(plan.render_spans)
        if not spans:
            raise RuntimeError("bounded full render produced no fragments")
        if len(spans) != len(plan.spans):
            raise RuntimeError(
                "bounded full render plan contains a non-render span"
            )
        cursor = int(plan.index.start_pts)
        for position, span in enumerate(spans):
            if not span.is_render:
                raise RuntimeError(
                    f"bounded full fragment {position} is not a render span"
                )
            if int(span.start_pts) != cursor:
                raise RuntimeError(
                    "bounded full render has a PTS gap or overlap before "
                    f"fragment {position}: expected {cursor}, got {span.start_pts}"
                )
            if int(span.end_pts) <= int(span.start_pts):
                raise RuntimeError(
                    f"bounded full fragment {position} has non-positive PTS range"
                )
            cursor = int(span.end_pts)
        if cursor != int(plan.index.end_pts):
            raise RuntimeError(
                "bounded full render does not cover the probed source PTS range: "
                f"ended at {cursor}, expected {plan.index.end_pts}"
            )

    def _run_smart(self, metadata) -> None:
        if windows_d3d11_hip_resident_requested():
            raise RuntimeError(
                "The explicit Windows D3D11/HIP resident route is not admitted "
                "for Smart Render spans"
            )
        codec = validate_smart_render(
            metadata,
            output_path=self.output_video,
            codec=self.codec,
            retarget_high_fps=self.retarget_high_fps,
        )
        if self.splice_plan is None:
            index = probe_keyframes(self.input_video, metadata)
            plan = build_splice_plan(self.segments or (), index, duration=metadata.duration)
        else:
            plan = self.splice_plan
            if plan.segments != tuple(self.segments or ()):
                raise ValueError("Precomputed splice plan does not match pipeline segments")
            index = plan.index
        vendor = vendor_for_device(self.device)
        recycle_h264_render_sessions = bool(
            is_isolated_video_job()
            and vendor is AcceleratorVendor.AMD
            and codec == "h264"
        )
        bounded_amf_session_seconds = _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec=codec,
            vendor=vendor,
            batch_size=int(getattr(self, "batch_size", 0) or 0),
            dual_gop_enabled=bool(
                getattr(self, "amd_dual_gop_encode", False)
            ),
        )
        if bounded_amf_session_seconds is not None:
            original_render_spans = len(plan.render_spans)
            plan = split_render_spans(
                plan,
                video_fps=metadata.video_fps_exact,
                max_duration_seconds=bounded_amf_session_seconds,
            )
            log.info(
                "Bounded Linux AMD %s render sessions: %.1fs maximum, "
                "%d -> %d render spans",
                codec.upper(),
                bounded_amf_session_seconds,
                original_render_spans,
                len(plan.render_spans),
            )
        smart_encoder_settings = resolve_smart_encoder_settings(
            codec,
            metadata,
            index,
            self.encoder_settings,
            vendor=vendor,
        )
        smart_encoder_settings = _dual_gop_smart_encoder_settings(
            smart_encoder_settings,
            enabled=bool(getattr(self, "amd_dual_gop_encode", False)),
        )
        total_frames = max(
            1,
            sum(
                round((span.end_pts - span.start_pts) * index.time_base * metadata.video_fps)
                for span in plan.render_spans
            ),
        )
        progress = Progressbar(
            total_frames=total_frames,
            video_fps=metadata.video_fps,
            disable=self.disable_progress,
            callback=self.progress_callback,
        )
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        work_root = self.working_dir or self.output_video.parent
        work_root.mkdir(parents=True, exist_ok=True)

        vr_resolution = getattr(self, "_vr_resolution", None)
        resolved_projection = (
            vr_resolution.projection
            if vr_resolution is not None
            else getattr(self, "vr_projection", "auto")
        )
        signature = workspace_signature(
            source=self.input_video,
            output=self.output_video,
            plan=plan,
            processing=getattr(self, "processing_signature", {}),
            model_files={
                "detection": getattr(self, "detection_model_path", None),
                "restoration": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "restorer",
                        None,
                    ),
                    "checkpoint_path",
                    None,
                ),
                "secondary": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "secondary_restorer",
                        None,
                    ),
                    "engine_path",
                    None,
                ),
                "lut": self.lut_path,
            },
            codec=codec,
            encoder_settings=smart_encoder_settings,
            resolved_projection=resolved_projection,
        )
        workspace = SmartRenderWorkspace.open(
            work_root,
            output=self.output_video,
            signature=signature,
        )
        render_metadata = None
        render_output_fps = None
        try:
            fragments: list[tuple[Path, float]] = []
            remaining_render_spans = len(plan.render_spans)
            fragment_suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
            for span_index, span in enumerate(plan.spans):
                if self._cancel_event.is_set():
                    return
                duration = float((span.end_pts - span.start_pts) * index.time_base)
                expected_frames = max(1, round(duration * metadata.video_fps))
                reusable = workspace.reusable_fragment(span_index)
                if (
                    reusable is not None
                    and codec == "hevc"
                    and not span.is_render
                    and not hevc_copy_fragment_timeline_matches_source(
                        reusable,
                        self.input_video,
                        span,
                        index,
                    )
                ):
                    log.warning(
                        "Discarding stale HEVC Smart Render copy span %s with "
                        "a source-mismatched packet timeline: %s",
                        span_index,
                        reusable,
                    )
                    reusable = None
                if reusable is not None:
                    log.info("Reusing smart-render span %s from %s", span_index, reusable)
                    fragments.append((reusable, duration))
                    if span.is_render:
                        remaining_render_spans -= 1
                        progress.mark_completed(expected_frames)
                    continue

                workspace.mark_running(span_index)
                raw = workspace.raw_path(span_index)
                normalized = workspace.fragment_path(span_index, fragment_suffix)
                raw.unlink(missing_ok=True)
                normalized.unlink(missing_ok=True)
                if span.is_render:
                    if render_metadata is None:
                        render_metadata = metadata
                        render_output_fps = metadata.video_fps_exact
                        if codec == "hevc":
                            render_metadata, render_output_fps = (
                                resolve_hevc_smart_render_vui(metadata)
                            )
                    encoder_ctx = NvidiaVideoEncoder(
                        str(raw),
                        device=self.device,
                        metadata=render_metadata,
                        codec=codec,
                        encoder_settings=smart_encoder_settings,
                        lut_path=self.lut_path,
                        sharpen_strength=self.sharpen_strength,
                        output_fps=render_output_fps,
                        mux_audio=False,
                        pts_origin=span.start_pts,
                        match_input_bit_depth=True,
                        smart_fragment=True,
                        auto_source_rate=getattr(self, "auto_source_rate", False),
                        prefer_amf_host_native=bool(
                            getattr(self, "amd_dual_gop_encode", False)
                        ),
                    )
                    pass_stats = self._run_pass(
                        metadata=metadata,
                        encoder_ctx=encoder_ctx,
                        progress=progress,
                        seek_ts=index.seconds_for_pts(span.start_pts),
                        end_pts=span.end_pts,
                        effect_ranges=span.effect_ranges,
                        output_frame_count=expected_frames,
                        recycle_on_host_memory_pressure=True,
                    )
                    recycle_after_pressure = bool(
                        is_isolated_video_job()
                        and (
                            pass_stats.system_pressure_episodes > 0
                            or pass_stats.system_reclaim_count > 0
                        )
                    )
                else:
                    recycle_after_pressure = False
                    if codec in {"h264", "hevc"}:
                        create_normalized_copy_fragment(
                            self.input_video,
                            span,
                            index,
                            normalized,
                            codec=codec,
                        )
                    else:
                        create_copy_fragment(
                            self.input_video,
                            span,
                            index,
                            raw,
                            codec=codec,
                        )
                if self._cancel_event.is_set():
                    return
                if span.is_render:
                    normalize_fragment(
                        raw,
                        normalized,
                        codec=codec,
                        decode_delay=index.decode_delay_pts * index.time_base,
                    )
                elif codec == "av1":
                    normalize_fragment(raw, normalized, codec=codec)
                workspace.mark_complete(span_index, normalized)
                if span.is_render:
                    remaining_render_spans -= 1
                try:
                    raw.unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not clean raw smart-render span %s", raw)
                fragments.append((normalized, duration))
                recycle_for_session_limit = bool(
                    span.is_render
                    and (
                        bounded_amf_session_seconds is not None
                        or recycle_h264_render_sessions
                    )
                    and remaining_render_spans > 0
                )
                if (
                    (recycle_after_pressure or recycle_for_session_limit)
                    and remaining_render_spans > 0
                ):
                    if recycle_after_pressure:
                        message = (
                            "Linux AMD whole-card pressure occurred while completing "
                            f"Smart Render span {span_index}; recycling the isolated "
                            "worker before another AMF decoder pair is opened"
                        )
                        recycle_reason = "native_pressure"
                    else:
                        if recycle_h264_render_sessions:
                            message = (
                                "Linux AMD H.264 Smart Render span "
                                f"{span_index} completed; recycling the isolated "
                                "worker before another AMF encoder session opens"
                            )
                        else:
                            message = (
                                "Linux AMD 8K HEVC AMF render-session limit completed "
                                f"at Smart Render span {span_index}; recycling the "
                                "isolated worker before the next bounded span"
                            )
                        recycle_reason = "amf_session_limit"
                    if recycle_after_pressure:
                        log.warning(message)
                    else:
                        # This is an expected resource boundary, not a
                        # user-facing warning.  The GUI parent keeps progress
                        # and ETA stable while the fresh worker resumes the
                        # durable workspace.
                        log.debug(message)
                    raise NativeWorkerRecycleRequested(
                        message,
                        reason=recycle_reason,
                    )

            if self._cancel_event.is_set():
                return
            if codec == "hevc":
                log.info(
                    "Finalizing HEVC Smart Render: checking boundary "
                    "VPS/SPS/PPS for %d fragments",
                    len(fragments),
                )
                validate_hevc_fragment_parameter_sets(
                    [
                        (fragment, span.kind)
                        for (fragment, _duration), span in zip(
                            fragments,
                            plan.spans,
                        )
                    ]
                )
            mux_fragments_final_output(
                fragments,
                self.input_video,
                self.output_video,
                manifest=workspace.path / "fragments.ffconcat",
                codec=codec,
                copy_validation_ranges=(
                    self._hevc_copy_validation_ranges(plan)
                    if codec == "hevc"
                    else ()
                ),
            )
            if self._cancel_event.is_set():
                return
            try:
                workspace.cleanup()
            except OSError:
                log.warning(
                    "Could not clean completed smart-render workspace %s",
                    workspace.path,
                )
        except SmartRenderCompatibilityError:
            log.error(
                "Preserving rejected Smart Render workspace for diagnosis or "
                "resume: %s",
                workspace.path,
            )
            raise
        finally:
            progress.close(ensure_completed_bar=True)

    @staticmethod
    def _hevc_copy_validation_ranges(
        plan: SplicePlan,
    ) -> tuple[tuple[float, float], ...]:
        """Return at most one second on every untouched side of a render seam."""

        keyframes = plan.index.pts
        one_second_pts = max(1, int(1 / plan.index.time_base))
        ranges: set[tuple[int, int]] = set()
        for span_index, span in enumerate(plan.spans):
            if span.is_render:
                continue
            if span_index > 0 and plan.spans[span_index - 1].is_render:
                next_keyframe = next(
                    (pts for pts in keyframes if pts > span.start_pts),
                    span.end_pts,
                )
                ranges.add(
                    (
                        span.start_pts,
                        min(
                            span.end_pts,
                            next_keyframe,
                            span.start_pts + one_second_pts,
                        ),
                    )
                )
            if span_index + 1 < len(plan.spans) and plan.spans[span_index + 1].is_render:
                previous_index = bisect.bisect_left(keyframes, span.end_pts) - 1
                start_pts = keyframes[max(0, previous_index)]
                ranges.add(
                    (
                        max(
                            span.start_pts,
                            start_pts,
                            span.end_pts - one_second_pts,
                        ),
                        span.end_pts,
                    )
                )
        return tuple(
            (
                plan.index.seconds_for_pts(start_pts),
                float((end_pts - start_pts) * plan.index.time_base),
            )
            for start_pts, end_pts in sorted(ranges)
            if end_pts > start_pts
        )

    def run(self) -> None:
        metadata = get_video_meta_data(str(self.input_video))
        self._validate_metadata(metadata)
        self.configure_vr(metadata)
        if self.segments:
            if self.fmp4:
                log.warning(
                    "Fragmented MP4 is not available with segment processing; "
                    "the output is assembled after processing finishes"
                )
                self.fmp4 = False
            self._run_smart(metadata)
        else:
            self._run_full(metadata)
        self.completed = not self._cancel_event.is_set()

    def run_streaming(
        self,
        port: int = 8765,
        segment_duration: float = 4.0,
        hls_server=None,
    ) -> None:
        if self.segments:
            raise ValueError("Segment processing is not supported in streaming mode")
        from jasna.streaming_pipeline import run_streaming
        run_streaming(self, port=port, segment_duration=segment_duration, hls_server=hls_server)

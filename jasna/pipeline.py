from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue

import psutil
import torch

from jasna.media import UnsupportedColorspaceError, get_video_meta_data
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.media.splice import (
    SplicePlan,
    build_splice_plan,
    concatenate_fragments,
    create_copy_fragment,
    mux_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
    validate_smart_render,
)
from jasna.mosaic.detection_registry import build_detection_model
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import FrameMeta, PrimaryRestoreResult, SecondaryLoopStats, _SENTINEL
from jasna.pipeline_threads import decode_detect_loop, primary_restore_loop, secondary_restore_loop, blend_encode_loop
from jasna.progressbar import Progressbar
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.segments import SegmentRange
from jasna.vram_offloader import VramOffloader
from jasna.vr180 import (
    FisheyeProjector,
    SbsDetectionAdapter,
    resolve_vr_mode,
)

log = logging.getLogger(__name__)


class _OfflineFrameWriter:
    def __init__(self, encoder_ctx: NvidiaVideoEncoder, encode_heartbeat: list[float]):
        self._encoder_ctx = encoder_ctx
        self._encode_heartbeat = encode_heartbeat
        self._entered = False

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if not self._entered:
            self._encoder_ctx.__enter__()
            self._entered = True
        self._encoder_ctx.encode(frame, pts, apply_lut=apply_lut)
        self._encode_heartbeat[0] = time.monotonic()

    def after_write(self, frames_written: int) -> None:
        pass

    def close(self) -> None:
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
        vr_mode: str = "auto",
        fp16: bool,
        disable_progress: bool = False,
        progress_callback: callable | None = None,
        lut_path: str | Path | None = None,
        retarget_high_fps: bool = False,
        segments: tuple[SegmentRange, ...] | None = None,
        splice_plan: SplicePlan | None = None,
        working_dir: Path | None = None,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.working_dir = working_dir
        self.codec = str(codec)
        self.encoder_settings = dict(encoder_settings)
        self.batch_size = int(batch_size)
        self.device = device
        self.max_clip_size = int(max_clip_size)
        self.temporal_overlap = int(temporal_overlap)
        self.max_detection_gap = int(max_detection_gap)
        self.min_detection_duration = int(min_detection_duration)
        self.enable_crossfade = bool(enable_crossfade)
        self.vr_mode = str(vr_mode)

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
        self.retarget_high_fps = bool(retarget_high_fps)
        self.segments = tuple(segments) if segments else None
        self.splice_plan = splice_plan
        self._vr_resolution = None
        self._vr_projector = None
        self._job_detection_model = self.detection_model

    def configure_vr(self, metadata) -> None:
        self._vr_resolution = resolve_vr_mode(
            self.vr_mode,
            metadata,
            self.input_video,
        )
        self._job_detection_model = (
            SbsDetectionAdapter(self.detection_model)
            if self._vr_resolution.is_sbs
            else self.detection_model
        )
        self._vr_projector = (
            FisheyeProjector(
                eye_width=int(metadata.video_width) // 2,
                height=int(metadata.video_height),
                device=self.device,
            )
            if self._vr_resolution.uses_fisheye
            else None
        )

    def close(self) -> None:
        if hasattr(self, "detection_model") and self.detection_model is not None:
            if hasattr(self.detection_model, "close"):
                self.detection_model.close()
            self.detection_model = None
        self.restoration_pipeline = None

    _ASYNC_POLL_TIMEOUT = 0.05

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
    ) -> SecondaryLoopStats:
        restorer: AsyncSecondaryRestorer = self.restoration_pipeline.secondary_restorer  # type: ignore[assignment]
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
                    item = secondary_queue.get()
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
        pusher_thread.join()
        if pusher_error:
            raise pusher_error[0]
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
    ) -> None:
        device = self.device
        secondary_workers = max(1, int(self.restoration_pipeline.secondary_num_workers))
        frame_rate = resolve_frame_rate_retarget(metadata.video_fps_exact, enabled=self.retarget_high_fps)
        if output_frame_count is None:
            output_frame_count = frame_rate.output_frame_count(metadata.num_frames)

        clip_queue = FrameQueue(max_frames=self.max_clip_size)
        secondary_queue = FrameQueue(max_frames=self.max_clip_size * secondary_workers)
        encode_queue = FrameQueue(max_frames=self.max_clip_size)
        metadata_queue: Queue[FrameMeta | object] = Queue(maxsize=self.max_clip_size * 5)

        error_holder: list[BaseException] = []
        blend_buffer = BlendBuffer(device=device)
        crop_buffers: dict[int, CropBuffer] = {}
        crop_lock = threading.Lock()
        primary_idle_event = threading.Event()
        frame_shape: list[tuple[int, int]] = []

        encode_heartbeat: list[float] = [time.monotonic()]
        frame_writer = _OfflineFrameWriter(encoder_ctx, encode_heartbeat)
        vram_offloader = VramOffloader(
            device=device,
            blend_buffer=blend_buffer,
            crop_buffers=crop_buffers,
            crop_lock=crop_lock,
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
                starvation_stats = self._run_secondary_loop(secondary_queue, encode_queue, debug_memory, clip_queue, primary_idle_event)
            except BaseException as e:
                log.exception("[secondary-async] thread crashed")
                error_holder.append(e)
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
                    vr_projector=self._vr_projector,
                ),
                name="BlendEncode", daemon=True,
            ),
        ]
        vram_offloader.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        vram_offloader.stop()
        frame_writer.close()

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

        err = error_holder[0] if error_holder else None
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
        )
        if frame_rate.active:
            log.info(
                "Retargeting frame rate: %s fps -> %s fps (keeping every %dth frame)",
                frame_rate.source_fps,
                frame_rate.output_fps,
                frame_rate.frame_stride,
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
        encoder_ctx = NvidiaVideoEncoder(
            str(self.output_video),
            device=self.device,
            metadata=metadata,
            codec=self.codec,
            encoder_settings=self.encoder_settings,
            lut_path=self.lut_path,
            output_fps=frame_rate.output_fps,
        )
        try:
            self._run_pass(
                metadata=metadata,
                encoder_ctx=encoder_ctx,
                progress=progress,
                output_frame_count=output_frame_count,
            )
        finally:
            progress.close(ensure_completed_bar=True)

    def _run_smart(self, metadata) -> None:
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
        smart_encoder_settings = resolve_smart_encoder_settings(
            codec,
            metadata,
            index,
            self.encoder_settings,
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

        try:
            with TemporaryDirectory(
                dir=work_root,
                prefix=f".{self.output_video.stem}.segments-",
            ) as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                fragments: list[tuple[Path, float]] = []
                fragment_suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
                for span_index, span in enumerate(plan.spans):
                    raw = temp_dir / f"{span_index:04d}-raw.nut"
                    normalized = temp_dir / f"{span_index:04d}{fragment_suffix}"
                    duration = float((span.end_pts - span.start_pts) * index.time_base)
                    if span.is_render:
                        encoder_ctx = NvidiaVideoEncoder(
                            str(raw),
                            device=self.device,
                            metadata=metadata,
                            codec=codec,
                            encoder_settings=smart_encoder_settings,
                            lut_path=self.lut_path,
                            output_fps=metadata.video_fps_exact,
                            mux_audio=False,
                            pts_origin=span.start_pts,
                            match_input_bit_depth=True,
                            smart_fragment=True,
                        )
                        self._run_pass(
                            metadata=metadata,
                            encoder_ctx=encoder_ctx,
                            progress=progress,
                            seek_ts=index.seconds_for_pts(span.start_pts),
                            end_pts=span.end_pts,
                            effect_ranges=span.effect_ranges,
                            output_frame_count=max(1, round(duration * metadata.video_fps)),
                        )
                    else:
                        create_copy_fragment(self.input_video, span, index, raw, codec=codec)
                    normalize_fragment(raw, normalized, codec=codec)
                    fragments.append((normalized, duration))

                assembled = temp_dir / f"assembled{fragment_suffix}"
                concatenate_fragments(
                    fragments,
                    manifest=temp_dir / "fragments.ffconcat",
                    destination=assembled,
                    codec=codec,
                )
                mux_final_output(
                    assembled,
                    self.input_video,
                    self.output_video,
                    codec=codec,
                )
        finally:
            progress.close(ensure_completed_bar=True)

    def run(self) -> None:
        metadata = get_video_meta_data(str(self.input_video))
        self._validate_metadata(metadata)
        self.configure_vr(metadata)
        if self.segments:
            self._run_smart(metadata)
        else:
            self._run_full(metadata)

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

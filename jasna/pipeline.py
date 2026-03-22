from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from jasna.frame_queue import FrameQueue

import psutil
import torch

logger = logging.getLogger(__name__)

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic import RfDetrMosaicDetectionModel, YoloMosaicDetectionModel
from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model, coerce_detection_model_name
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryLoopStats, SecondaryRestoreResult, _SENTINEL
from jasna.progressbar import Progressbar
from jasna.tracking import ClipTracker, FrameBuffer
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.pipeline_processing import process_frame_batch, finalize_processing

log = logging.getLogger(__name__)


class Pipeline:
    _DECODE_FB_STALL_WAIT_TIMEOUT_SECONDS = 0.05
    _VRAM_FREE_HEADROOM_BYTES = 1024 * 1024 ** 2
    _VRAM_LIMIT_OVERRIDE_GB: float | None = None
    _RAM_PRESSURE_PERCENT = 94.0

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
        enable_crossfade: bool = True,
        fp16: bool,
        disable_progress: bool = False,
        progress_callback: callable | None = None,
        working_directory: Path | None = None,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.codec = str(codec)
        self.encoder_settings = dict(encoder_settings)
        self.batch_size = int(batch_size)
        self.device = device
        self.max_clip_size = int(max_clip_size)
        self.temporal_overlap = int(temporal_overlap)
        self.enable_crossfade = bool(enable_crossfade)

        det_name = coerce_detection_model_name(detection_model_name)
        if is_rfdetr_model(det_name):
            self.detection_model = RfDetrMosaicDetectionModel(
                onnx_path=detection_model_path,
                batch_size=self.batch_size,
                device=self.device,
                score_threshold=float(detection_score_threshold),
                fp16=bool(fp16),
            )
        elif is_yolo_model(det_name):
            self.detection_model = YoloMosaicDetectionModel(
                model_path=detection_model_path,
                batch_size=self.batch_size,
                device=self.device,
                score_threshold=float(detection_score_threshold),
                fp16=bool(fp16),
            )
        self.restoration_pipeline = restoration_pipeline
        self.disable_progress = bool(disable_progress)
        self.progress_callback = progress_callback
        self.working_directory = working_directory

    def _wait_for_decode_fb_drain(self, drained_event: threading.Event) -> None:
        drained_event.wait(timeout=self._DECODE_FB_STALL_WAIT_TIMEOUT_SECONDS)
        drained_event.clear()

    def _should_offload_frames(self) -> tuple[bool, int, int]:
        free, total = torch.cuda.mem_get_info(self.device)
        used = total - free
        if self._VRAM_LIMIT_OVERRIDE_GB is not None:
            cap = int(self._VRAM_LIMIT_OVERRIDE_GB * (1024 ** 3))
            threshold = cap - self._VRAM_FREE_HEADROOM_BYTES
            return used > threshold, used, threshold
        threshold = total - self._VRAM_FREE_HEADROOM_BYTES
        return used > threshold, used, threshold

    _ASYNC_POLL_TIMEOUT = 0.05

    @staticmethod
    def _earliest_blocking_seqs(pending_prs: dict[int, PrimaryRestoreResult]) -> set[int] | None:
        if not pending_prs:
            return None
        earliest_frame = min(
            pr.clip.start_frame + pr.keep_start for pr in pending_prs.values()
        )
        return {
            seq for seq, pr in pending_prs.items()
            if pr.clip.start_frame + pr.keep_start <= earliest_frame <= pr.clip.start_frame + pr.keep_end - 1
        }

    _FLUSH_DELAY = 2.0

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
                        logger.debug("[secondary] push_clip seq=%d took %.0fms", seq, push_elapsed * 1000)
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
                encode_queue.put(sr, frame_count=sr.frame_count)
                if debug_memory is not None:
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.clip.track_id} frames={sr.frame_count}",
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

            if (
                restorer.has_pending
                and _no_clips_incoming()
                and not flushed_since_last_push
                and time.monotonic() - last_push_time > self._FLUSH_DELAY
            ):
                if starvation_start is None:
                    starvation_start = time.monotonic()
                target_seqs = self._earliest_blocking_seqs(dict(pending_prs))
                logger.debug("[secondary] starvation flush target_seqs=%s", target_seqs)
                restorer.flush_pending(target_seqs=target_seqs)
                starvation_count += 1
                flushed_since_last_push = True

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

    def run(self) -> None:
        device = self.device
        metadata = get_video_meta_data(str(self.input_video))
        secondary_workers = max(1, int(self.restoration_pipeline.secondary_num_workers))
        decode_bp_gap_threshold = int(self.max_clip_size * 1.2)

        clip_queue = FrameQueue(max_frames=self.max_clip_size)
        secondary_queue = FrameQueue(max_frames=self.max_clip_size * secondary_workers)
        encode_queue = FrameQueue(max_frames=self.max_clip_size)

        error_holder: list[BaseException] = []
        frame_buffer = FrameBuffer(device=device)
        fb_drained_event = threading.Event()
        primary_idle_event = threading.Event()
        decode_backpressure_event = threading.Event()
        ram_pressure = threading.Event()
        debug_memory = PipelineDebugMemoryLogger(
            logger=log,
            frame_buffer=frame_buffer,
            clip_queue=clip_queue,
            secondary_queue=secondary_queue,
            encode_queue=encode_queue,
        )

        def _decode_detect_thread():
            nonlocal peak_fb_size, bp_stall_count, bp_stall_seconds, ram_stall_count, ram_stall_seconds
            try:
                torch.cuda.set_device(device)
                tracker = ClipTracker(max_clip_size=self.max_clip_size, temporal_overlap=int(self.temporal_overlap))
                discard_margin = int(self.temporal_overlap)
                blend_frames = (self.temporal_overlap // 3) if self.enable_crossfade else 0
                raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}

                with (
                    NvidiaVideoReader(str(self.input_video), batch_size=self.batch_size, device=device, metadata=metadata) as reader,
                    torch.inference_mode(),
                ):
                    pb = Progressbar(
                        total_frames=metadata.num_frames,
                        video_fps=metadata.video_fps,
                        disable=self.disable_progress,
                        callback=self.progress_callback,
                    )
                    pb.init()
                    target_hw = (int(metadata.video_height), int(metadata.video_width))
                    frame_idx = 0
                    frames_since_last_clip_emit = 0
                    log.info(
                        "Processing %s: %d frames @ %s fps, %dx%d",
                        self.input_video.name, metadata.num_frames, metadata.video_fps, metadata.video_width, metadata.video_height,
                    )

                    try:
                        for frames, pts_list in reader.frames():
                            effective_bs = len(pts_list)
                            if effective_bs == 0:
                                continue

                            fb_size = len(frame_buffer.frames)
                            peak_fb_size = max(peak_fb_size, fb_size)

                            if ram_pressure.is_set():
                                t_ram = time.monotonic()
                                while ram_pressure.is_set():
                                    if error_holder:
                                        raise error_holder[0]
                                    time.sleep(0.05)
                                ram_stall_seconds += time.monotonic() - t_ram
                                ram_stall_count += 1

                            if frames_since_last_clip_emit >= decode_bp_gap_threshold:
                                log.debug(
                                    "[decode] gap backpressure enter gap=%d fb=%d",
                                    frames_since_last_clip_emit,
                                    fb_size,
                                )
                                t_bp = time.monotonic()
                                decode_backpressure_event.set()
                                while len(frame_buffer.frames) > decode_bp_gap_threshold:
                                    if error_holder:
                                        raise error_holder[0]
                                    self._wait_for_decode_fb_drain(fb_drained_event)
                                decode_backpressure_event.clear()
                                bp_stall_seconds += time.monotonic() - t_bp
                                bp_stall_count += 1
                                frames_since_last_clip_emit = 0
                                log.debug(
                                    "[decode] backpressure exit fb=%d",
                                    len(frame_buffer.frames),
                                )

                            batch_start = frame_idx

                            res = process_frame_batch(
                                frames=frames,
                                pts_list=[int(p) for p in pts_list],
                                start_frame_idx=frame_idx,
                                batch_size=self.batch_size,
                                target_hw=target_hw,
                                detections_fn=self.detection_model,
                                tracker=tracker,
                                frame_buffer=frame_buffer,
                                clip_queue=clip_queue,
                                discard_margin=discard_margin,
                                blend_frames=blend_frames,
                                raw_frame_context=raw_frame_context,
                            )

                            frame_idx = res.next_frame_idx
                            if res.clips_emitted > 0:
                                frames_since_last_clip_emit = 0
                            else:
                                frames_since_last_clip_emit += effective_bs

                            debug_memory.snapshot(
                                "decode",
                                f"frame_start={batch_start} batch={effective_bs} gap={frames_since_last_clip_emit}",
                            )
                            pb.update(effective_bs)

                        finalize_processing(
                            tracker=tracker,
                            frame_buffer=frame_buffer,
                            clip_queue=clip_queue,
                            discard_margin=discard_margin,
                            blend_frames=blend_frames,
                            raw_frame_context=raw_frame_context,
                        )
                        debug_memory.snapshot("decode", "finalized")
                    except Exception:
                        pb.error = True
                        raise
                    finally:
                        pb.close(ensure_completed_bar=True)
            except BaseException as e:
                log.exception("[decode] thread crashed")
                error_holder.append(e)
            finally:
                clip_queue.put(_SENTINEL)

        def _primary_restore_thread():
            try:
                torch.cuda.set_device(device)
                while True:
                    primary_idle_event.set()
                    item = clip_queue.get()
                    primary_idle_event.clear()
                    if item is _SENTINEL:
                        break
                    clip_item: ClipRestoreItem = item  # type: ignore[assignment]
                    result = self.restoration_pipeline.prepare_and_run_primary(
                        clip_item.clip,
                        clip_item.frames,
                        clip_item.keep_start,
                        clip_item.keep_end,
                        clip_item.crossfade_weights,
                    )
                    if self.restoration_pipeline.secondary_prefers_cpu_input:
                        result.primary_raw = result.primary_raw.cpu()
                    secondary_queue.put(result, frame_count=result.keep_end - result.keep_start)
                    debug_memory.snapshot(
                        "primary",
                        f"clip={clip_item.clip.track_id} frames={len(clip_item.frames)}",
                    )
            except BaseException as e:
                log.exception("[primary] thread crashed")
                error_holder.append(e)
            finally:
                secondary_queue.put(_SENTINEL)

        def _secondary_restore_thread():
            try:
                torch.cuda.set_device(device)
                while True:
                    item = secondary_queue.get()
                    if item is _SENTINEL:
                        break
                    pr: PrimaryRestoreResult = item  # type: ignore[assignment]
                    restored_frames = self.restoration_pipeline._run_secondary(
                        pr.primary_raw,
                        pr.keep_start,
                        pr.keep_end,
                    )
                    del pr.primary_raw
                    sr = self.restoration_pipeline.build_secondary_result(pr, restored_frames)
                    encode_queue.put(sr, frame_count=sr.frame_count)
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.clip.track_id} frames={sr.frame_count}",
                    )
            except BaseException as e:
                log.exception("[secondary] thread crashed")
                error_holder.append(e)
            finally:
                encode_queue.put(_SENTINEL)

        starvation_stats = SecondaryLoopStats()

        def _async_secondary_restore_thread():
            nonlocal starvation_stats
            try:
                torch.cuda.set_device(device)
                starvation_stats = self._run_secondary_loop(secondary_queue, encode_queue, debug_memory, clip_queue, primary_idle_event)
            except BaseException as e:
                log.exception("[secondary-async] thread crashed")
                error_holder.append(e)
            finally:
                encode_queue.put(_SENTINEL)

        def _encode_thread():
            try:
                torch.cuda.set_device(device)

                with NvidiaVideoEncoder(
                    str(self.output_video),
                    device=device,
                    metadata=metadata,
                    codec=self.codec,
                    encoder_settings=self.encoder_settings,
                    stream_mode=False,
                    working_directory=self.working_directory,
                ) as encoder:
                    while True:
                        encoded_count = 0
                        try:
                            item = encode_queue.get(timeout=0.1)
                            if item is _SENTINEL:
                                break
                            sr: SecondaryRestoreResult = item  # type: ignore[assignment]
                            for blended_idx in self.restoration_pipeline.blend_secondary_result(sr, frame_buffer):
                                for ready_idx, ready_frame, ready_pts in frame_buffer.get_ready_frames():
                                    encoder.encode(ready_frame, ready_pts)
                                    encoded_count += 1
                            debug_memory.snapshot("encode", f"clip={sr.clip.track_id} blended")
                        except Empty:
                            pass

                        for ready_idx, ready_frame, ready_pts in frame_buffer.get_ready_frames():
                            encoder.encode(ready_frame, ready_pts)
                            encoded_count += 1
                        if encoded_count > 0:
                            fb_drained_event.set()

                    for ready_idx, ready_frame, ready_pts in frame_buffer.flush():
                        encoder.encode(ready_frame, ready_pts)
                        #log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)
            except BaseException as e:
                log.exception("[encode] thread crashed")
                error_holder.append(e)

        stop_offload = threading.Event()
        vram_max = 0
        vram_sum = 0
        vram_samples = 0
        offload_count = 0
        ram_max = 0
        ram_sum = 0
        ram_samples = 0
        _process = psutil.Process(os.getpid())
        peak_fb_size = 0
        bp_stall_count = 0
        bp_stall_seconds = 0.0
        ram_stall_count = 0
        ram_stall_seconds = 0.0

        def _vram_offload_thread():
            nonlocal vram_max, vram_sum, vram_samples, offload_count, ram_max, ram_sum, ram_samples
            try:
                while not stop_offload.is_set():
                    over_limit, used, threshold = self._should_offload_frames()
                    vram_max = max(vram_max, used)
                    vram_sum += used
                    vram_samples += 1
                    try:
                        rss = _process.memory_info().rss
                        ram_max = max(ram_max, rss)
                        ram_sum += rss
                        ram_samples += 1
                    except Exception:
                        pass
                    if over_limit:
                        excess = int((used - threshold) * 1.2)
                        offloaded = frame_buffer.offload_gpu_frames(excess)
                        if offloaded > 0:
                            offload_count += offloaded
                            torch.cuda.empty_cache()
                            continue

                    try:
                        ram_pct = psutil.virtual_memory().percent
                        if ram_pct >= self._RAM_PRESSURE_PERCENT:
                            ram_pressure.set()
                        else:
                            ram_pressure.clear()
                    except Exception:
                        pass

                    headroom = threshold - used
                    if headroom > 2 * (1024 ** 3):
                        stop_offload.wait(timeout=0.2)
                    else:
                        stop_offload.wait(timeout=0.05)
            except BaseException:
                log.exception("[offload] thread crashed")

        use_async_secondary = isinstance(self.restoration_pipeline.secondary_restorer, AsyncSecondaryRestorer)
        secondary_fn = _async_secondary_restore_thread if use_async_secondary else _secondary_restore_thread
        if use_async_secondary:
            log.debug("Using async secondary restore path")

        threads = [
            threading.Thread(target=_decode_detect_thread, name="DecodeDetect", daemon=True),
            threading.Thread(target=_primary_restore_thread, name="PrimaryRestore", daemon=True),
            threading.Thread(target=secondary_fn, name="SecondaryRestore", daemon=True),
            threading.Thread(target=_encode_thread, name="Encode", daemon=True),
            threading.Thread(target=_vram_offload_thread, name="VramOffload", daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads[:4]:
            t.join()
        stop_offload.set()
        threads[4].join(timeout=1)

        if vram_samples > 0:
            vram_avg = vram_sum / vram_samples
            log.info(
                "VRAM usage — max: %.1f MiB, avg: %.1f MiB (%d samples), offloaded frames: %d",
                vram_max / (1024 ** 2), vram_avg / (1024 ** 2), vram_samples, offload_count,
            )
        if ram_samples > 0:
            ram_avg = ram_sum / ram_samples
            log.info(
                "RAM usage — max: %.1f MiB, avg: %.1f MiB (%d samples)",
                ram_max / (1024 ** 2), ram_avg / (1024 ** 2), ram_samples,
            )
        log.info("Frame buffer — peak: %d frames", peak_fb_size)
        if bp_stall_count > 0:
            log.info("Decode backpressure — stalls: %d, total: %.1fs", bp_stall_count, bp_stall_seconds)
        if ram_stall_count > 0:
            log.info("Decode RAM stall — stalls: %d, total: %.1fs", ram_stall_count, ram_stall_seconds)
        ss = starvation_stats
        if ss.clips_pushed > 0 or ss.clips_popped > 0:
            log.info(
                "Secondary — clips: %d pushed / %d popped, pusher stall: %.1fs, starvation flushes: %d (%.1fs)",
                ss.clips_pushed, ss.clips_popped, ss.pusher_stall_seconds, ss.starvation_flushes, ss.starvation_seconds,
            )

        frame_buffer.frames.clear()
        frame_buffer._gpu_pinned.clear()
        del frame_buffer

        err = error_holder[0] if error_holder else None
        if err is not None:
            err.__traceback__ = None

        del clip_queue, secondary_queue, encode_queue
        del error_holder, threads
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(self.device)

        if err is not None:
            raise err

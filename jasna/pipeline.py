from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue

import psutil
import torch

from jasna.media import UnsupportedColorspaceError, get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
from jasna.mosaic.yolo import YoloMosaicDetectionModel
from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model, coerce_detection_model_name
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryLoopStats, SecondaryRestoreResult, _SENTINEL
from jasna.progressbar import Progressbar
from jasna.tracking import ClipTracker
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.vram_offloader import VramOffloader

log = logging.getLogger(__name__)


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

    def run(self) -> None:
        from av.video.reformatter import Colorspace as AvColorspace
        device = self.device
        metadata = get_video_meta_data(str(self.input_video))
        if metadata.color_space != AvColorspace.ITU709:
            raise UnsupportedColorspaceError(
                f"Unsupported color space: {metadata.color_space!r} in {self.input_video.name}. Only BT.709 is supported."
            )
        secondary_workers = max(1, int(self.restoration_pipeline.secondary_num_workers))

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

        def _decode_detect_thread():
            try:
                torch.cuda.set_device(device)
                tracker = ClipTracker(max_clip_size=self.max_clip_size, temporal_overlap=int(self.temporal_overlap))
                discard_margin = int(self.temporal_overlap)
                blend_frames = (self.temporal_overlap // 3) if self.enable_crossfade else 0

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
                    log.info(
                        "Processing %s: %d frames @ %s fps, %dx%d",
                        self.input_video.name, metadata.num_frames, metadata.video_fps, metadata.video_width, metadata.video_height,
                    )

                    try:
                        for frames, pts_list in reader.frames():
                            effective_bs = len(pts_list)
                            if effective_bs == 0:
                                continue

                            if not frame_shape:
                                _, fh, fw = frames[0].shape
                                frame_shape.append((int(fh), int(fw)))

                            if error_holder:
                                raise error_holder[0]

                            batch_start = frame_idx

                            res = process_frame_batch(
                                frames=frames,
                                pts_list=[int(p) for p in pts_list],
                                start_frame_idx=frame_idx,
                                batch_size=self.batch_size,
                                target_hw=target_hw,
                                detections_fn=self.detection_model,
                                tracker=tracker,
                                blend_buffer=blend_buffer,
                                crop_buffers=crop_buffers,
                                clip_queue=clip_queue,
                                metadata_queue=metadata_queue,
                                discard_margin=discard_margin,
                                blend_frames=blend_frames,
                            )

                            frame_idx = res.next_frame_idx
                            debug_memory.snapshot(
                                "decode",
                                f"frame_start={batch_start} batch={effective_bs}",
                            )
                            pb.update(effective_bs)

                        fs = frame_shape[0] if frame_shape else (int(metadata.video_height), int(metadata.video_width))
                        finalize_processing(
                            tracker=tracker,
                            blend_buffer=blend_buffer,
                            crop_buffers=crop_buffers,
                            clip_queue=clip_queue,
                            frame_shape=fs,
                            discard_margin=discard_margin,
                            blend_frames=blend_frames,
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
                metadata_queue.put(_SENTINEL)

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
                        clip_item.raw_crops,
                        clip_item.frame_shape,
                        clip_item.keep_start,
                        clip_item.keep_end,
                        clip_item.crossfade_weights,
                    )
                    if self.restoration_pipeline.secondary_prefers_cpu_input:
                        result.primary_raw = result.primary_raw.cpu()
                    secondary_queue.put(result, frame_count=result.keep_end - result.keep_start)
                    debug_memory.snapshot(
                        "primary",
                        f"clip={clip_item.clip.track_id} frames={len(clip_item.raw_crops)}",
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
                    encode_queue.put(sr, frame_count=sr.keep_end)
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.track_id} frames={sr.frame_count}",
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

        def _blend_encode_thread():
            try:
                torch.cuda.set_device(device)

                def _flat_frames(rdr: NvidiaVideoReader):
                    for batch, pts in rdr.frames():
                        for i in range(len(pts)):
                            yield batch[i]

                with (
                    NvidiaVideoReader(str(self.input_video), batch_size=self.batch_size, device=device, metadata=metadata) as reader2,
                    NvidiaVideoEncoder(
                        str(self.output_video),
                        device=device,
                        metadata=metadata,
                        codec=self.codec,
                        encoder_settings=self.encoder_settings,
                        stream_mode=False,
                        working_directory=self.working_directory,
                    ) as encoder,
                ):
                    frame_gen = _flat_frames(reader2)

                    secondary_done = False

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
                        _drain_encode_queue()
                        try:
                            meta_item = metadata_queue.get(timeout=0.05)
                        except Empty:
                            continue
                        if meta_item is _SENTINEL:
                            break
                        meta: FrameMeta = meta_item  # type: ignore[assignment]
                        original_frame = next(frame_gen)

                        while not blend_buffer.is_frame_ready(meta.frame_idx):
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

                        blended = blend_buffer.blend_frame(meta.frame_idx, original_frame)
                        encoder.encode(blended, meta.pts)
                        encode_heartbeat[0] = time.monotonic()

            except BaseException as e:
                log.exception("[blend-encode] thread crashed")
                error_holder.append(e)

        _process = psutil.Process(os.getpid())

        use_async_secondary = isinstance(self.restoration_pipeline.secondary_restorer, AsyncSecondaryRestorer)
        secondary_fn = _async_secondary_restore_thread if use_async_secondary else _secondary_restore_thread
        if use_async_secondary:
            log.debug("Using async secondary restore path")

        threads = [
            threading.Thread(target=_decode_detect_thread, name="DecodeDetect", daemon=True),
            threading.Thread(target=_primary_restore_thread, name="PrimaryRestore", daemon=True),
            threading.Thread(target=secondary_fn, name="SecondaryRestore", daemon=True),
            threading.Thread(target=_blend_encode_thread, name="BlendEncode", daemon=True),
        ]
        vram_offloader.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        vram_offloader.stop()

        try:
            free, total = torch.cuda.mem_get_info(device)
            vram_used = total - free
            log.info("VRAM usage at end — %.1f MiB", vram_used / (1024 ** 2))
        except Exception:
            pass
        try:
            rss = _process.memory_info().rss
            log.info("RAM usage at end — %.1f MiB", rss / (1024 ** 2))
        except Exception:
            pass

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

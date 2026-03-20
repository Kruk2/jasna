from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from queue import Empty, Queue

import torch

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic import RfDetrMosaicDetectionModel, YoloMosaicDetectionModel
from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model, coerce_detection_model_name
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.progressbar import Progressbar
from jasna.tracking import ClipTracker, FrameBuffer
from jasna.restorer import RestorationPipeline
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.pipeline_processing import process_frame_batch, finalize_processing

log = logging.getLogger(__name__)


class Pipeline:
    _SECONDARY_QUEUE_MAXSIZE = 2
    _DECODE_FB_STALL_WAIT_TIMEOUT_SECONDS = 0.05
    _VRAM_FREE_HEADROOM_BYTES = 750 * 1024 ** 2
    _VRAM_LIMIT_OVERRIDE_GB: float | None = None

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

    _ASYNC_FLUSH_TIMEOUT = 0.1
    _ASYNC_POLL_TIMEOUT = 0.05

    def _run_secondary_loop(
        self,
        secondary_queue: Queue,
        encode_queue: Queue,
        debug_memory: PipelineDebugMemoryLogger | None = None,
    ) -> None:
        restorer: AsyncSecondaryRestorer = self.restoration_pipeline.secondary_restorer  # type: ignore[assignment]
        pending_prs: dict[int, PrimaryRestoreResult] = {}
        idle_seconds = 0.0

        def _forward_completed() -> int:
            forwarded = 0
            for seq, frames_np in restorer.pop_completed():
                pr = pending_prs.pop(seq)
                tensors = restorer._to_tensors(frames_np)
                if pr.frame_device.type != "cpu" and tensors:
                    tensors = list(torch.stack(tensors).to(pr.frame_device, non_blocking=True).unbind(0))
                sr = self.restoration_pipeline.build_secondary_result(pr, tensors)
                encode_queue.put(sr)
                if debug_memory is not None:
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.clip.track_id} frames={sr.frame_count}",
                    )
                forwarded += 1
            return forwarded

        done = False
        while not done:
            try:
                item = secondary_queue.get(timeout=self._ASYNC_POLL_TIMEOUT)
                if item is _SENTINEL:
                    done = True
                else:
                    pr = item  # type: ignore[assignment]
                    seq = restorer.push_clip(
                        pr.primary_raw,
                        keep_start=pr.keep_start,
                        keep_end=pr.keep_end,
                    )
                    del pr.primary_raw
                    pending_prs[seq] = pr
                    idle_seconds = 0.0
            except Empty:
                idle_seconds += self._ASYNC_POLL_TIMEOUT

            _forward_completed()

            if not done and idle_seconds >= self._ASYNC_FLUSH_TIMEOUT and restorer.has_pending:
                restorer.flush_pending()
                idle_seconds = 0.0

        restorer.flush_all()
        _forward_completed()

    def run(self) -> None:
        device = self.device
        metadata = get_video_meta_data(str(self.input_video))
        secondary_workers = max(1, int(self.restoration_pipeline.secondary_num_workers))
        decode_fb_low_watermark = int(self.max_clip_size)
        decode_fb_high_watermark = int(self.max_clip_size) * 3 + int(self.batch_size)
        if decode_fb_high_watermark <= decode_fb_low_watermark:
            decode_fb_high_watermark = decode_fb_low_watermark + 1

        clip_queue: Queue[ClipRestoreItem | object] = Queue(maxsize=1)
        secondary_queue: Queue[PrimaryRestoreResult | object] = Queue(maxsize=self._SECONDARY_QUEUE_MAXSIZE)
        encode_queue: Queue[SecondaryRestoreResult | object] = Queue(maxsize=secondary_workers + 1)

        error_holder: list[BaseException] = []
        frame_buffer = FrameBuffer(device=device)
        fb_drained_event = threading.Event()
        debug_memory = PipelineDebugMemoryLogger(
            logger=log,
            frame_buffer=frame_buffer,
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
                    log.info(
                        "Processing %s: %d frames @ %s fps, %dx%d",
                        self.input_video.name, metadata.num_frames, metadata.video_fps, metadata.video_width, metadata.video_height,
                    )

                    try:
                        for frames, pts_list in reader.frames():
                            effective_bs = len(pts_list)
                            if effective_bs == 0:
                                continue

                            if len(frame_buffer.frames) >= decode_fb_high_watermark:
                                log.debug(
                                    "[decode] fb backpressure enter fb=%d hwm=%d lwm=%d",
                                    len(frame_buffer.frames),
                                    decode_fb_high_watermark,
                                    decode_fb_low_watermark,
                                )
                                while len(frame_buffer.frames) > decode_fb_low_watermark:
                                    if error_holder:
                                        raise error_holder[0]
                                    self._wait_for_decode_fb_drain(fb_drained_event)
                                log.debug(
                                    "[decode] fb backpressure exit fb=%d hwm=%d lwm=%d",
                                    len(frame_buffer.frames),
                                    decode_fb_high_watermark,
                                    decode_fb_low_watermark,
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

                            debug_memory.snapshot(
                                "decode",
                                f"frame_start={batch_start} batch={effective_bs}",
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
                    item = clip_queue.get()
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
                    secondary_queue.put(result)
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
                    encode_queue.put(sr)
                    debug_memory.snapshot(
                        "secondary",
                        f"clip={pr.clip.track_id} frames={sr.frame_count}",
                    )
            except BaseException as e:
                log.exception("[secondary] thread crashed")
                error_holder.append(e)
            finally:
                encode_queue.put(_SENTINEL)

        def _async_secondary_restore_thread():
            try:
                torch.cuda.set_device(device)
                self._run_secondary_loop(secondary_queue, encode_queue, debug_memory)
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

        def _vram_offload_thread():
            nonlocal vram_max, vram_sum, vram_samples, offload_count
            try:
                while not stop_offload.is_set():
                    over_limit, used, threshold = self._should_offload_frames()
                    vram_max = max(vram_max, used)
                    vram_sum += used
                    vram_samples += 1
                    if over_limit:
                        offloaded = frame_buffer.offload_gpu_frames()
                        if offloaded > 0:
                            offload_count += offloaded
                            torch.cuda.empty_cache()
                            continue
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
            log.info("Using async secondary restore path")

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

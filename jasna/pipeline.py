from __future__ import annotations

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
from jasna.pipeline_processing import process_frame_batch, finalize_processing

log = logging.getLogger(__name__)


class Pipeline:
    _SECONDARY_QUEUE_MAXSIZE = 2
    _DECODE_FB_STALL_WAIT_TIMEOUT_SECONDS = 0.05

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
                            self.restoration_pipeline.blend_secondary_result(sr, frame_buffer)
                            debug_memory.snapshot("encode", f"clip={sr.clip.track_id} blended")
                        except Empty:
                            pass

                        for ready_idx, ready_frame, ready_pts in frame_buffer.get_ready_frames():
                            encoder.encode(ready_frame, ready_pts)
                            encoded_count += 1
                            #log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)
                        if encoded_count > 0:
                            fb_drained_event.set()
                        if encoded_count > 0:
                            debug_memory.snapshot("encode", f"encoded={encoded_count}")

                    for ready_idx, ready_frame, ready_pts in frame_buffer.flush():
                        encoder.encode(ready_frame, ready_pts)
                        #log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)
            except BaseException as e:
                error_holder.append(e)

        threads = [
            threading.Thread(target=_decode_detect_thread, name="DecodeDetect", daemon=True),
            threading.Thread(target=_primary_restore_thread, name="PrimaryRestore", daemon=True),
            threading.Thread(target=_secondary_restore_thread, name="SecondaryRestore", daemon=True),
            threading.Thread(target=_encode_thread, name="Encode", daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if error_holder:
            raise error_holder[0]

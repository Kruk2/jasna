from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import torch

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic import RfDetrMosaicDetectionModel, YoloMosaicDetectionModel
from jasna.mosaic import Detections
from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model, coerce_detection_model_name
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.progressbar import Progressbar
from jasna.tracking import ClipTracker, FrameBuffer
from jasna.restorer import RestorationPipeline
from jasna.pipeline_processing import process_frame_batch, finalize_processing

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

    def run(self) -> None:
        device = self.device
        metadata = get_video_meta_data(str(self.input_video))

        clip_queue: Queue[ClipRestoreItem | object] = Queue(maxsize=1)
        secondary_queue: Queue[PrimaryRestoreResult | object] = Queue(maxsize=1)
        encode_queue: Queue[SecondaryRestoreResult | object] = Queue(maxsize=8)

        error_holder: list[BaseException] = []
        frame_buffer = FrameBuffer(device=device)

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
                            pb.update(effective_bs)

                        finalize_processing(
                            tracker=tracker,
                            frame_buffer=frame_buffer,
                            clip_queue=clip_queue,
                            discard_margin=discard_margin,
                            blend_frames=blend_frames,
                            raw_frame_context=raw_frame_context,
                        )
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
            except BaseException as e:
                error_holder.append(e)
            finally:
                secondary_queue.put(_SENTINEL)

        def _secondary_restore_thread():
            try:
                torch.cuda.set_device(device)
                restorer = self.restoration_pipeline.secondary_restorer
                if restorer is not None and hasattr(restorer, 'push_clip'):
                    self._run_async_secondary(secondary_queue, encode_queue)
                else:
                    num_workers = self.restoration_pipeline.secondary_num_workers
                    self._run_pooled_secondary(num_workers, secondary_queue, encode_queue)
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
                        try:
                            item = encode_queue.get(timeout=0.1)
                            if item is _SENTINEL:
                                break
                            sr: SecondaryRestoreResult = item  # type: ignore[assignment]
                            self.restoration_pipeline.blend_secondary_result(sr, frame_buffer)
                        except Empty:
                            pass

                        for ready_idx, ready_frame, ready_pts in frame_buffer.get_ready_frames():
                            encoder.encode(ready_frame, ready_pts)
                            #log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)

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

    _FLUSH_GAP_SECONDS = 2.0

    def _run_async_secondary(
        self,
        secondary_queue: Queue[PrimaryRestoreResult | object],
        encode_queue: Queue[SecondaryRestoreResult | object],
    ) -> None:
        restorer = self.restoration_pipeline.secondary_restorer
        rp = self.restoration_pipeline
        pending_prs: dict[int, PrimaryRestoreResult] = {}
        last_push_time = time.monotonic()

        def _drain_completed():
            for seq, restored_frames in restorer.pop_completed():
                pr = pending_prs.pop(seq)
                encode_queue.put(rp.build_secondary_result(pr, restored_frames))

        while True:
            _drain_completed()

            try:
                item = secondary_queue.get(timeout=0.5)
            except Empty:
                if pending_prs and (time.monotonic() - last_push_time) > self._FLUSH_GAP_SECONDS:
                    log.info("TVAI async: gap detected (%d pending), flushing workers", len(pending_prs))
                    restorer.flush_all()
                    _drain_completed()
                continue

            if item is _SENTINEL:
                if pending_prs:
                    restorer.flush_all()
                    _drain_completed()
                break

            pr: PrimaryRestoreResult = item  # type: ignore[assignment]
            seq = restorer.push_clip(pr.primary_raw, pr.keep_start, pr.keep_end)
            del pr.primary_raw
            pending_prs[seq] = pr
            last_push_time = time.monotonic()

        if pending_prs:
            log.warning("TVAI async: %d clips still pending after flush", len(pending_prs))

    def _run_pooled_secondary(
        self,
        num_workers: int,
        secondary_queue: Queue[PrimaryRestoreResult | object],
        encode_queue: Queue[SecondaryRestoreResult | object],
    ) -> None:
        def _worker(pr: PrimaryRestoreResult) -> SecondaryRestoreResult:
            return self.restoration_pipeline.run_secondary_from_primary(pr)

        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="SecondaryWorker") as pool:
            pending: dict[int, Future[SecondaryRestoreResult]] = {}
            next_submit_seq = 0
            next_drain_seq = 0
            first_error: BaseException | None = None

            def _drain_seq(seq: int) -> None:
                nonlocal first_error

                fut = pending.pop(seq)
                if first_error is not None:
                    fut.cancel()
                    return
                try:
                    encode_queue.put(fut.result())
                except BaseException as exc:
                    first_error = exc
                    log.error("Pooled secondary failed, draining remaining work", exc_info=True)

            while True:
                # Block taking new sequences if we reach max in-flight capacity.
                # We cap at num_workers so all worker slots are full but no extra GPU tensors
                # pile up in pending. The secondary_queue maxsize=1 provides the 1-ahead buffer.
                while len(pending) >= num_workers:
                    _drain_seq(next_drain_seq)
                    next_drain_seq += 1

                while next_drain_seq in pending and pending[next_drain_seq].done():
                    _drain_seq(next_drain_seq)
                    next_drain_seq += 1

                item = secondary_queue.get()
                if item is _SENTINEL:
                    break
                if first_error is not None:
                    continue
                pr: PrimaryRestoreResult = item  # type: ignore[assignment]

                seq = next_submit_seq
                next_submit_seq += 1
                pending[seq] = pool.submit(_worker, pr)

            for seq in range(next_drain_seq, next_submit_seq):
                if seq in pending:
                    _drain_seq(seq)

            if first_error is not None:
                raise first_error

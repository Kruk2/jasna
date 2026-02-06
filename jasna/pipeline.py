from __future__ import annotations

import logging
from pathlib import Path

import torch

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic import RfDetrMosaicDetectionModel
from jasna.mosaic import Detections
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
        detection_model_path: Path,
        detection_score_threshold: float,
        restoration_pipeline: RestorationPipeline,
        codec: str,
        encoder_settings: dict[str, object],
        stream: torch.cuda.Stream,
        batch_size: int,
        device: torch.device,
        max_clip_size: int,
        temporal_overlap: int,
        enable_crossfade: bool = True,
        fp16: bool,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.codec = str(codec)
        self.encoder_settings = dict(encoder_settings)
        self.stream = stream
        self.batch_size = int(batch_size)
        self.device = device
        self.max_clip_size = int(max_clip_size)
        self.temporal_overlap = int(temporal_overlap)
        self.enable_crossfade = bool(enable_crossfade)

        self.detection_model = RfDetrMosaicDetectionModel(
            onnx_path=detection_model_path,
            stream=self.stream,
            batch_size=self.batch_size,
            device=self.device,
            score_threshold=float(detection_score_threshold),
            fp16=bool(fp16),
        )
        self.restoration_pipeline = restoration_pipeline

    def run(self) -> None:
        stream = self.stream
        metadata = get_video_meta_data(str(self.input_video))

        tracker = ClipTracker(max_clip_size=self.max_clip_size, temporal_overlap=int(self.temporal_overlap))
        frame_buffer = FrameBuffer(device=self.device)

        discard_margin = int(self.temporal_overlap)
        blend_frames = (self.temporal_overlap // 3) if self.enable_crossfade else 0
        raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}

        with (
            NvidiaVideoReader(str(self.input_video), batch_size=self.batch_size, device=self.device, stream=stream, metadata=metadata) as reader,
            NvidiaVideoEncoder(
                str(self.output_video),
                device=self.device,
                stream=stream,
                metadata=metadata,
                codec=self.codec,
                encoder_settings=self.encoder_settings,
                stream_mode=False,
            ) as encoder,
            torch.inference_mode(),
            torch.cuda.stream(stream),
        ):
            pb = Progressbar(total_frames=metadata.num_frames, video_fps=metadata.video_fps)
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
                        restoration_pipeline=self.restoration_pipeline,
                        discard_margin=discard_margin,
                        blend_frames=blend_frames,
                        raw_frame_context=raw_frame_context,
                    )
                    for ready_idx, ready_frame, ready_pts in res.ready_frames:
                        encoder.encode(ready_frame, ready_pts)
                        log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)

                    frame_idx = res.next_frame_idx
                    pb.update(effective_bs)

                remaining_frames = finalize_processing(
                    tracker=tracker,
                    frame_buffer=frame_buffer,
                    restoration_pipeline=self.restoration_pipeline,
                    discard_margin=discard_margin,
                    blend_frames=blend_frames,
                    raw_frame_context=raw_frame_context,
                )
                if remaining_frames:
                    log.debug("encoding %d remaining frame(s)", len(remaining_frames))
                for ready_idx, ready_frame, ready_pts in remaining_frames:
                    encoder.encode(ready_frame, ready_pts)
                    log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)
            except Exception:
                pb.error = True
                raise
            finally:
                pb.close(ensure_completed_bar=True)


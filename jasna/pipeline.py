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

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        *,
        input_video: Path,
        output_video: Path,
        detection_model_path: Path,
        restoration_pipeline: RestorationPipeline,
        codec: str,
        encoder_settings: dict[str, object],
        stream: torch.cuda.Stream,
        batch_size: int,
        device: torch.device,
        max_clip_size: int,
        temporal_overlap: int,
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

        self.detection_model = RfDetrMosaicDetectionModel(
            onnx_path=detection_model_path,
            stream=self.stream,
            batch_size=self.batch_size,
            device=self.device,
            fp16=bool(fp16),
        )
        self.restoration_pipeline = restoration_pipeline

    def run(self) -> None:
        stream = self.stream
        metadata = get_video_meta_data(str(self.input_video))

        tracker = ClipTracker(max_clip_size=self.max_clip_size, temporal_overlap=self.temporal_overlap)
        frame_buffer = FrameBuffer(device=self.device)
        active_tracks: set[int] = set()
        continuation_context: dict[int, list[torch.Tensor]] = {}

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

            try:
                for frames, pts_list in reader.frames():
                    effective_bs = len(pts_list)
                    if effective_bs == 0:
                        continue

                    frames_eff = frames[:effective_bs]
                    if effective_bs < self.batch_size:
                        pad = frames_eff[-1:].expand(self.batch_size - effective_bs, -1, -1, -1)
                        frames_in = torch.cat([frames_eff, pad], dim=0)
                    else:
                        frames_in = frames

                    detections: Detections = self.detection_model(frames_in, target_hw=target_hw)

                    for i in range(effective_bs):
                        current_frame_idx = frame_idx + i
                        pts = int(pts_list[i])
                        frame = frames_eff[i]

                        valid_boxes = detections.boxes_xyxy[i]
                        valid_masks = detections.masks[i]
                        n_detections = valid_boxes.shape[0]

                        if n_detections > 0:
                            log.debug("frame %d: %d detection(s)", current_frame_idx, n_detections)

                        ended_clips, active_track_ids = tracker.update(
                            current_frame_idx, valid_boxes, valid_masks
                        )

                        new_tracks = active_track_ids - active_tracks
                        for track_id in new_tracks:
                            log.debug("clip %d started at frame %d", track_id, current_frame_idx)
                        active_tracks = (active_tracks | active_track_ids) - {ec.clip.track_id for ec in ended_clips}

                        frame_buffer.add_frame(current_frame_idx, pts, frame, active_track_ids)

                        for ended_clip in ended_clips:
                            clip = ended_clip.clip
                            log.debug("clip %d ended: frames %d-%d (%d frames)", clip.track_id, clip.start_frame, clip.end_frame, clip.frame_count)
                            frames_for_clip = [frame_buffer.get_frame(fi) for fi in clip.frame_indices()]
                            frames_for_clip = [f for f in frames_for_clip if f is not None]
                            if frames_for_clip:
                                # Check if this clip is a continuation of a previously split clip
                                source_track_id = tracker.get_continuation_source(clip.track_id)
                                prefix_frames = None
                                if source_track_id is not None and source_track_id in continuation_context:
                                    prefix_frames = continuation_context.pop(source_track_id)
                                    log.debug("clip %d using %d prefix frames from clip %d", clip.track_id, len(prefix_frames), source_track_id)
                                    tracker.clear_continuation(clip.track_id)
                                
                                restored_clip = self.restoration_pipeline.restore_clip(
                                    clip, frames_for_clip, prefix_restored_frames=prefix_frames
                                )
                                log.debug("clip %d restored", clip.track_id)
                                frame_buffer.blend_clip(clip, restored_clip)
                                log.debug("clip %d blended onto frames %d-%d", clip.track_id, clip.start_frame, clip.end_frame)
                                
                                # Store context for potential continuation if this clip was split
                                if ended_clip.split_due_to_max_size and self.temporal_overlap > 0:
                                    n_context = min(self.temporal_overlap, len(restored_clip.restored_frames))
                                    continuation_context[clip.track_id] = restored_clip.restored_frames[-n_context:]
                                    log.debug("clip %d storing %d frames for continuation", clip.track_id, n_context)

                        ready_frames = frame_buffer.get_ready_frames()
                        for ready_idx, ready_frame, ready_pts in ready_frames:
                            encoder.encode(ready_frame, ready_pts)
                            log.debug("frame %d encoded (pts=%d)", ready_idx, ready_pts)

                    frame_idx += effective_bs
                    pb.update(effective_bs)

                final_clips = tracker.flush()
                if final_clips:
                    log.debug("flushing %d remaining clip(s)", len(final_clips))
                for ended_clip in final_clips:
                    clip = ended_clip.clip
                    log.debug("clip %d ended: frames %d-%d (%d frames)", clip.track_id, clip.start_frame, clip.end_frame, clip.frame_count)
                    frames_for_clip = [frame_buffer.get_frame(fi) for fi in clip.frame_indices()]
                    frames_for_clip = [f for f in frames_for_clip if f is not None]
                    if frames_for_clip:
                        # Check if this clip is a continuation of a previously split clip
                        source_track_id = tracker.get_continuation_source(clip.track_id)
                        prefix_frames = None
                        if source_track_id is not None and source_track_id in continuation_context:
                            prefix_frames = continuation_context.pop(source_track_id)
                            log.debug("clip %d using %d prefix frames from clip %d", clip.track_id, len(prefix_frames), source_track_id)
                        
                        restored_clip = self.restoration_pipeline.restore_clip(
                            clip, frames_for_clip, prefix_restored_frames=prefix_frames
                        )
                        log.debug("clip %d restored", clip.track_id)
                        frame_buffer.blend_clip(clip, restored_clip)
                        log.debug("clip %d blended onto frames %d-%d", clip.track_id, clip.start_frame, clip.end_frame)

                remaining_frames = frame_buffer.flush()
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


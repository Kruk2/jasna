from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Queue

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, RawCrop, extract_crop
from jasna.mosaic.detections import Detections
from jasna.pipeline_items import ClipRestoreItem, FrameMeta
from jasna.pipeline_overlap import compute_crossfade_weights, compute_keep_range, compute_overlap_and_tail_indices, compute_parent_crossfade_weights
from jasna.tensor_utils import pad_batch_with_last
from jasna.tracking.clip_tracker import ClipTracker, EndedClip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchProcessResult:
    next_frame_idx: int
    clips_emitted: int


def _process_ended_clips(
    *,
    ended_clips: list[EndedClip],
    discard_margin: int,
    blend_frames: int,
    max_clip_size: int,
    blend_buffer: BlendBuffer,
    crop_buffers: dict[int, CropBuffer],
    clip_queue: Queue[ClipRestoreItem | object],
    frame_shape: tuple[int, int],
) -> None:
    bf = min(int(blend_frames), int(discard_margin)) if discard_margin > 0 else 0
    if bf > 0 and discard_margin > 0:
        max_bf = max(0, (int(max_clip_size) - 2 * int(discard_margin)) // 2)
        bf = min(bf, max_bf)

    for ended_clip in ended_clips:
        clip = ended_clip.clip
        crop_buf = crop_buffers.pop(clip.track_id, None)
        if crop_buf is None:
            raise RuntimeError(f"missing CropBuffer for clip {clip.track_id}")

        if ended_clip.split_due_to_max_size and discard_margin > 0:
            child_id = ended_clip.continuation_track_id
            if child_id is None:
                raise RuntimeError("split clip is missing continuation_track_id")

            overlap_len = 2 * int(discard_margin)
            crop_buffers[child_id] = crop_buf.split_overlap(
                overlap_len, child_id, clip.end_frame - overlap_len + 1,
            )

            overlap_indices, tail_indices = compute_overlap_and_tail_indices(
                end_frame=clip.end_frame, discard_margin=discard_margin
            )
            blend_buffer.add_pending_clip(overlap_indices, child_id)

            if bf > 0:
                non_crossfade_tail = list(range(clip.end_frame - discard_margin + 1 + bf, clip.end_frame + 1))
                if non_crossfade_tail:
                    blend_buffer.remove_pending_clip(non_crossfade_tail, clip.track_id)
            else:
                blend_buffer.remove_pending_clip(tail_indices, clip.track_id)

        keep_start, keep_end = compute_keep_range(
            frame_count=clip.frame_count,
            is_continuation=clip.is_continuation,
            split_due_to_max_size=ended_clip.split_due_to_max_size,
            discard_margin=discard_margin,
            blend_frames=bf,
        )

        crossfade_weights = None
        if clip.is_continuation and bf > 0 and discard_margin > 0:
            crossfade_weights = compute_crossfade_weights(
                discard_margin=discard_margin,
                blend_frames=bf,
            )
        if ended_clip.split_due_to_max_size and bf > 0 and discard_margin > 0:
            parent_weights = compute_parent_crossfade_weights(
                frame_count=clip.frame_count,
                discard_margin=discard_margin,
                blend_frames=bf,
            )
            if crossfade_weights is None:
                crossfade_weights = parent_weights
            else:
                crossfade_weights.update(parent_weights)

        item = ClipRestoreItem(
            clip=clip,
            raw_crops=crop_buf.crops,
            frame_shape=frame_shape,
            keep_start=int(keep_start),
            keep_end=int(keep_end),
            crossfade_weights=crossfade_weights,
        )
        clip_queue.put(item, frame_count=int(keep_end) - int(keep_start))


def process_frame_batch(
    *,
    frames: torch.Tensor,
    pts_list: list[int],
    start_frame_idx: int,
    batch_size: int,
    target_hw: tuple[int, int],
    detections_fn,
    tracker: ClipTracker,
    blend_buffer: BlendBuffer,
    crop_buffers: dict[int, CropBuffer],
    clip_queue: Queue[ClipRestoreItem | object],
    metadata_queue: Queue[FrameMeta | object],
    discard_margin: int,
    blend_frames: int = 0,
) -> BatchProcessResult:
    effective_bs = len(pts_list)
    if effective_bs == 0:
        return BatchProcessResult(next_frame_idx=int(start_frame_idx), clips_emitted=0)

    frames_eff = frames[:effective_bs]
    frames_in = pad_batch_with_last(frames_eff, batch_size=int(batch_size))

    detections: Detections = detections_fn(frames_in, target_hw=target_hw)
    _, frame_h, frame_w = frames_eff[0].shape

    clips_emitted = 0
    for i in range(effective_bs):
        current_frame_idx = int(start_frame_idx) + i
        pts = int(pts_list[i])
        frame = frames_eff[i]

        valid_boxes = detections.boxes_xyxy[i]
        valid_masks = detections.masks[i]

        ended_clips, active_track_ids = tracker.update(current_frame_idx, valid_boxes, valid_masks)

        blend_buffer.register_frame(current_frame_idx, active_track_ids)
        metadata_queue.put(FrameMeta(frame_idx=current_frame_idx, pts=pts))

        for track_id in active_track_ids:
            clip = tracker.active_clips.get(track_id)
            if clip is None:
                continue
            if track_id not in crop_buffers:
                crop_buffers[track_id] = CropBuffer(track_id=track_id, start_frame=clip.start_frame)
            raw_crop = extract_crop(frame, clip.bboxes[-1], frame_h, frame_w)
            crop_buffers[track_id].add(raw_crop)

        for ec in ended_clips:
            tid = ec.clip.track_id
            if tid not in crop_buffers:
                crop_buffers[tid] = CropBuffer(track_id=tid, start_frame=ec.clip.start_frame)
            if crop_buffers[tid].frame_count < ec.clip.frame_count:
                raw_crop = extract_crop(frame, ec.clip.bboxes[-1], frame_h, frame_w)
                crop_buffers[tid].add(raw_crop)

        clips_emitted += len(ended_clips)

        _process_ended_clips(
            ended_clips=ended_clips,
            discard_margin=int(discard_margin),
            blend_frames=int(blend_frames),
            max_clip_size=tracker.max_clip_size,
            blend_buffer=blend_buffer,
            crop_buffers=crop_buffers,
            clip_queue=clip_queue,
            frame_shape=(frame_h, frame_w),
        )

    return BatchProcessResult(
        next_frame_idx=int(start_frame_idx) + effective_bs,
        clips_emitted=clips_emitted,
    )


def finalize_processing(
    *,
    tracker: ClipTracker,
    blend_buffer: BlendBuffer,
    crop_buffers: dict[int, CropBuffer],
    clip_queue: Queue[ClipRestoreItem | object],
    frame_shape: tuple[int, int],
    discard_margin: int,
    blend_frames: int = 0,
) -> None:
    ended_clips = tracker.flush()
    _process_ended_clips(
        ended_clips=ended_clips,
        discard_margin=int(discard_margin),
        blend_frames=int(blend_frames),
        max_clip_size=tracker.max_clip_size,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        clip_queue=clip_queue,
        frame_shape=frame_shape,
    )

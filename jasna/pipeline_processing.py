from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from jasna.mosaic.detections import Detections
from jasna.pipeline_overlap import compute_crossfade_weights, compute_keep_range, compute_overlap_and_tail_indices, compute_parent_crossfade_weights
from jasna.tensor_utils import pad_batch_with_last
from jasna.tracking.clip_tracker import ClipTracker, EndedClip
from jasna.tracking.frame_buffer import FrameBuffer
from jasna.restorer.restoration_pipeline import RestorationPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchProcessResult:
    next_frame_idx: int
    ready_frames: list[tuple[int, torch.Tensor, int]]


def _process_ended_clips(
    *,
    ended_clips: list[EndedClip],
    discard_margin: int,
    blend_frames: int,
    frame_buffer: FrameBuffer,
    restoration_pipeline: RestorationPipeline,
    raw_frame_context: dict[int, dict[int, torch.Tensor]],
) -> None:
    bf = min(int(blend_frames), int(discard_margin)) if discard_margin > 0 else 0

    for ended_clip in ended_clips:
        clip = ended_clip.clip
        ctx = raw_frame_context.get(clip.track_id, {})
        frames_for_clip: list[torch.Tensor] = []
        for fi in clip.frame_indices():
            f = frame_buffer.get_frame(fi)
            if f is None:
                f = ctx.get(fi)
            if f is None:
                raise RuntimeError(f"missing frame {fi} for clip {clip.track_id}")
            frames_for_clip.append(f)

        if ended_clip.split_due_to_max_size and discard_margin > 0:
            child_id = ended_clip.continuation_track_id
            if child_id is None:
                raise RuntimeError("split clip is missing continuation_track_id")

            overlap_indices, tail_indices = compute_overlap_and_tail_indices(
                end_frame=clip.end_frame, discard_margin=discard_margin
            )
            child_ctx: dict[int, torch.Tensor] = {}
            for fi in overlap_indices:
                f = frame_buffer.get_frame(fi)
                if f is None:
                    raise RuntimeError(f"missing overlap frame {fi} for continuation clip {child_id}")
                child_ctx[fi] = f
            raw_frame_context[child_id] = child_ctx

            if bf > 0:
                seam_frame = clip.end_frame - discard_margin + 1
                child_pending_indices = list(range(seam_frame - bf, clip.end_frame + 1))
                frame_buffer.add_pending_clip(child_pending_indices, child_id)
                non_crossfade_tail = list(range(seam_frame + bf, clip.end_frame + 1))
                if non_crossfade_tail:
                    frame_buffer.remove_pending_clip(non_crossfade_tail, clip.track_id)
            else:
                frame_buffer.add_pending_clip(tail_indices, child_id)
                frame_buffer.remove_pending_clip(tail_indices, clip.track_id)

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

        restoration_pipeline.restore_and_blend_clip(
            clip,
            frames_for_clip,
            keep_start=int(keep_start),
            keep_end=int(keep_end),
            frame_buffer=frame_buffer,
            crossfade_weights=crossfade_weights,
        )
        raw_frame_context.pop(clip.track_id, None)

        if ended_clip.split_due_to_max_size:
            # Transfer worker mapping to continuation so TVAI outputs flow naturally.
            restoration_pipeline.transfer_track(clip.track_id, ended_clip.continuation_track_id)
        else:
            # True end: push padding frames to drain TVAI's ~25 buffered frames.
            restoration_pipeline.flush_track(clip.track_id)
        
        # Poll to drain any ready outputs and reduce frame_buffer growth.
        restoration_pipeline.poll_secondary(frame_buffer=frame_buffer)


def process_frame_batch(
    *,
    frames: torch.Tensor,
    pts_list: list[int],
    start_frame_idx: int,
    batch_size: int,
    target_hw: tuple[int, int],
    detections_fn,
    tracker: ClipTracker,
    frame_buffer: FrameBuffer,
    restoration_pipeline: RestorationPipeline,
    discard_margin: int,
    blend_frames: int = 0,
    raw_frame_context: dict[int, dict[int, torch.Tensor]],
) -> BatchProcessResult:
    effective_bs = len(pts_list)
    if effective_bs == 0:
        return BatchProcessResult(next_frame_idx=int(start_frame_idx), ready_frames=[])

    frames_eff = frames[:effective_bs]
    frames_in = pad_batch_with_last(frames_eff, batch_size=int(batch_size))

    detections: Detections = detections_fn(frames_in, target_hw=target_hw)

    ready_frames: list[tuple[int, torch.Tensor, int]] = []
    for i in range(effective_bs):
        current_frame_idx = int(start_frame_idx) + i
        pts = int(pts_list[i])
        frame = frames_eff[i]

        valid_boxes = detections.boxes_xyxy[i]
        valid_masks = detections.masks[i]

        ended_clips, active_track_ids = tracker.update(current_frame_idx, valid_boxes, valid_masks)
        frame_buffer.add_frame(current_frame_idx, pts, frame, active_track_ids)

        _process_ended_clips(
            ended_clips=ended_clips,
            discard_margin=int(discard_margin),
            blend_frames=int(blend_frames),
            frame_buffer=frame_buffer,
            restoration_pipeline=restoration_pipeline,
            raw_frame_context=raw_frame_context,
        )
        restoration_pipeline.poll_secondary(frame_buffer=frame_buffer)
        ready_frames.extend(frame_buffer.get_ready_frames())

    buffered_count = len(frame_buffer.frames)
    if buffered_count > 100:
        pending_tracks = set()
        for pf in frame_buffer.frames.values():
            pending_tracks.update(pf.pending_clips)
        logger.debug(
            "frame_buffer has %d frames buffered, waiting for tracks: %s",
            buffered_count, pending_tracks,
        )

    return BatchProcessResult(next_frame_idx=int(start_frame_idx) + effective_bs, ready_frames=ready_frames)


def finalize_processing(
    *,
    tracker: ClipTracker,
    frame_buffer: FrameBuffer,
    restoration_pipeline: RestorationPipeline,
    discard_margin: int,
    blend_frames: int = 0,
    raw_frame_context: dict[int, dict[int, torch.Tensor]],
) -> list[tuple[int, torch.Tensor, int]]:
    ended_clips = tracker.flush()
    _process_ended_clips(
        ended_clips=ended_clips,
        discard_margin=int(discard_margin),
        blend_frames=int(blend_frames),
        frame_buffer=frame_buffer,
        restoration_pipeline=restoration_pipeline,
        raw_frame_context=raw_frame_context,
    )
    restoration_pipeline.flush_secondary(frame_buffer=frame_buffer)
    return frame_buffer.flush()


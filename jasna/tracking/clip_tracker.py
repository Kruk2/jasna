from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrackedClip:
    track_id: int
    start_frame: int
    mask_resolution: tuple[int, int]  # (Hm, Wm) model mask resolution
    bboxes: list[np.ndarray] = field(default_factory=list)  # each (4,) xyxy, CPU
    masks: list[torch.Tensor] = field(default_factory=list)  # each (Hm, Wm) bool, GPU
    is_continuation: bool = False

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.bboxes) - 1

    @property
    def frame_count(self) -> int:
        return len(self.bboxes)

    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.start_frame + len(self.bboxes)))


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    boxes1: (N, 4) xyxy, CPU
    boxes2: (M, 4) xyxy, CPU
    Returns: (N, M) IoU matrix
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    b1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    b2 = boxes2[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def merge_overlapping_boxes(
    bboxes: np.ndarray, masks: torch.Tensor, iou_threshold: float
) -> tuple[np.ndarray, torch.Tensor]:
    """
    bboxes: (K, 4) xyxy, CPU
    masks: (K, Hm, Wm) bool, GPU
    Returns: merged (N, 4) bboxes (CPU) and (N, Hm, Wm) masks (GPU) where N <= K
    """
    n = bboxes.shape[0]
    if n == 0:
        return bboxes, masks
    if n == 1:
        return bboxes, masks

    iou_matrix = compute_iou_matrix(bboxes, bboxes)
    adjacency = iou_matrix > iou_threshold

    labels = np.arange(n)
    for _ in range(n):
        for i in range(n):
            neighbors = np.where(adjacency[i])[0]
            if len(neighbors) > 0:
                min_label = labels[neighbors].min()
                if min_label < labels[i]:
                    labels[i] = min_label

    unique_labels = np.unique(labels)
    merged_bboxes = []
    merged_masks = []

    for label in unique_labels:
        group_indices = np.where(labels == label)[0]
        group_boxes = bboxes[group_indices]
        x1 = group_boxes[:, 0].min()
        y1 = group_boxes[:, 1].min()
        x2 = group_boxes[:, 2].max()
        y2 = group_boxes[:, 3].max()
        merged_bboxes.append(np.array([x1, y1, x2, y2]))
        merged_masks.append(masks[group_indices].any(dim=0))

    return np.stack(merged_bboxes), torch.stack(merged_masks)


@dataclass
class EndedClip:
    """Wrapper for ended clips with metadata about why they ended."""
    clip: TrackedClip
    split_due_to_max_size: bool
    continuation_track_id: int | None = None


class ClipTracker:
    def __init__(self, max_clip_size: int, temporal_overlap: int = 0, iou_threshold: float = 0.3):
        self.max_clip_size = int(max_clip_size)
        self.temporal_overlap = int(temporal_overlap)
        self.iou_threshold = iou_threshold
        self.active_clips: dict[int, TrackedClip] = {}
        self.next_track_id = 0
        self.last_frame_boxes: np.ndarray | None = None  # (T, 4) xyxy, CPU
        self.track_ids: list[int] = []  # track_id for each row in last_frame_boxes
        if self.temporal_overlap < 0:
            raise ValueError("temporal_overlap must be >= 0")
        if self.temporal_overlap >= self.max_clip_size:
            raise ValueError("temporal_overlap must be < max_clip_size")
        if self.temporal_overlap > 0 and (2 * self.temporal_overlap) >= self.max_clip_size:
            raise ValueError("temporal_overlap must satisfy 2*temporal_overlap < max_clip_size")

    def update(
        self, frame_idx: int, bboxes: np.ndarray, masks: torch.Tensor
    ) -> tuple[list[EndedClip], set[int]]:
        """
        bboxes: (K, 4) xyxy, CPU
        masks: (K, Hm, Wm) bool, GPU
        Returns: (ended_clips, active_track_ids)
        """
        if bboxes.shape[0] > 0:
            bboxes, masks = merge_overlapping_boxes(bboxes, masks, self.iou_threshold)

        ended_clips: list[EndedClip] = []
        active_track_ids: set[int] = set()

        if bboxes.shape[0] == 0:
            for track_id in self.track_ids:
                ended_clips.append(EndedClip(clip=self.active_clips.pop(track_id), split_due_to_max_size=False))
            self.last_frame_boxes = None
            self.track_ids = []
            return ended_clips, active_track_ids

        n_detections = bboxes.shape[0]
        matched_det = np.zeros(n_detections, dtype=bool)
        matched_track_indices: set[int] = set()
        det_to_track: dict[int, int] = {}

        if self.last_frame_boxes is not None and len(self.track_ids) > 0:
            iou_matrix = compute_iou_matrix(bboxes, self.last_frame_boxes)  # (K, T)

            for _ in range(min(n_detections, len(self.track_ids))):
                valid_mask = ~matched_det[:, np.newaxis] & ~np.array([i in matched_track_indices for i in range(len(self.track_ids))])
                masked_iou = np.where(valid_mask, iou_matrix, 0.0)

                max_iou = masked_iou.max()
                if max_iou <= self.iou_threshold:
                    break

                flat_idx = masked_iou.argmax()
                det_idx = flat_idx // iou_matrix.shape[1]
                track_idx = flat_idx % iou_matrix.shape[1]

                matched_det[det_idx] = True
                matched_track_indices.add(track_idx)
                det_to_track[det_idx] = track_idx
        for det_idx, track_idx in det_to_track.items():
            track_id = self.track_ids[track_idx]
            clip = self.active_clips[track_id]
            clip.bboxes.append(bboxes[det_idx])
            clip.masks.append(masks[det_idx])
            active_track_ids.add(track_id)

            if clip.frame_count >= self.max_clip_size:
                if self.temporal_overlap > 0:
                    overlap_len = 2 * self.temporal_overlap
                    overlap_bboxes = clip.bboxes[-overlap_len:]
                    overlap_masks = clip.masks[-overlap_len:]
                    new_start_frame = clip.end_frame - overlap_len + 1

                    new_track_id = self.next_track_id
                    self.next_track_id += 1
                    new_clip = TrackedClip(
                        track_id=new_track_id,
                        start_frame=new_start_frame,
                        mask_resolution=clip.mask_resolution,
                        bboxes=overlap_bboxes.copy(),
                        masks=overlap_masks.copy(),
                        is_continuation=True,
                    )
                    self.active_clips[new_track_id] = new_clip
                    active_track_ids.add(new_track_id)
                    active_track_ids.discard(track_id)

                    ended_clips.append(
                        EndedClip(clip=clip, split_due_to_max_size=True, continuation_track_id=new_track_id)
                    )
                    del self.active_clips[track_id]
                else:
                    ended_clips.append(EndedClip(clip=clip, split_due_to_max_size=True))
                    del self.active_clips[track_id]

        for track_idx, track_id in enumerate(self.track_ids):
            if track_idx not in matched_track_indices and track_id in self.active_clips:
                ended_clips.append(EndedClip(clip=self.active_clips.pop(track_id), split_due_to_max_size=False))

        for det_idx in range(n_detections):
            if not matched_det[det_idx]:
                track_id = self.next_track_id
                self.next_track_id += 1
                clip = TrackedClip(
                    track_id=track_id,
                    start_frame=frame_idx,
                    mask_resolution=(masks.shape[1], masks.shape[2]),
                    bboxes=[bboxes[det_idx]],
                    masks=[masks[det_idx]],
                    is_continuation=False,
                )
                self.active_clips[track_id] = clip
                active_track_ids.add(track_id)

        new_boxes = []
        new_track_ids = []
        for track_id in active_track_ids:
            clip = self.active_clips.get(track_id)
            if clip:
                new_boxes.append(clip.bboxes[-1])
                new_track_ids.append(track_id)

        if new_boxes:
            self.last_frame_boxes = np.stack(new_boxes)
            self.track_ids = new_track_ids
        else:
            self.last_frame_boxes = None
            self.track_ids = []

        return ended_clips, active_track_ids

    def flush(self) -> list[EndedClip]:
        clips = [EndedClip(clip=c, split_due_to_max_size=False) for c in self.active_clips.values()]
        self.active_clips.clear()
        self.last_frame_boxes = None
        self.track_ids = []
        return clips

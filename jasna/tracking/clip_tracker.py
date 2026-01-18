from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrackedClip:
    track_id: int
    start_frame: int
    bboxes: list[np.ndarray] = field(default_factory=list)  # each (4,) xyxy, CPU
    masks: list[torch.Tensor] = field(default_factory=list)  # each (H, W) bool, GPU

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.bboxes) - 1

    @property
    def frame_count(self) -> int:
        return len(self.bboxes)

    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.start_frame + len(self.bboxes)))


def compute_iou_matrix_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes (xyxy format).
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    n, m = boxes1.shape[0], boxes2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    b1 = boxes1.astype(np.float32)[:, None, :]  # (N, 1, 4)
    b2 = boxes2.astype(np.float32)[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union_area = area1 + area2 - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def merge_overlapping_boxes(
    bboxes: np.ndarray, masks: torch.Tensor, iou_threshold: float
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Merge overlapping bboxes within a single frame.
    bboxes: (K, 4) xyxy numpy array on CPU
    masks: (K, H, W) bool tensor on GPU
    Returns merged (N, 4) bboxes and (N, H, W) masks where N <= K
    """
    n = bboxes.shape[0]
    if n <= 1:
        return bboxes, masks

    iou_matrix = compute_iou_matrix_np(bboxes, bboxes)
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
        group_idx = np.where(labels == label)[0]
        group_boxes = bboxes[group_idx]
        x1 = group_boxes[:, 0].min()
        y1 = group_boxes[:, 1].min()
        x2 = group_boxes[:, 2].max()
        y2 = group_boxes[:, 3].max()
        merged_bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        merged_masks.append(masks[group_idx].any(dim=0))

    return np.stack(merged_bboxes), torch.stack(merged_masks)


class ClipTracker:
    def __init__(self, max_clip_size: int, iou_threshold: float = 0.3):
        self.max_clip_size = max_clip_size
        self.iou_threshold = iou_threshold
        self.active_clips: dict[int, TrackedClip] = {}
        self.next_track_id = 0
        self.last_frame_boxes: np.ndarray | None = None  # (T, 4)
        self.track_ids: list[int] = []

    def update(
        self, frame_idx: int, bboxes: torch.Tensor, masks: torch.Tensor
    ) -> tuple[list[TrackedClip], set[int]]:
        """
        Update tracker with detections from a new frame.
        
        bboxes: (K, 4) xyxy format, CPU tensor
        masks: (K, H, W) bool, GPU tensor
        
        Returns:
            ended_clips: clips that ended this frame (max size or no match)
            active_track_ids: track ids that cover this frame
        """
        bboxes_np = bboxes.numpy().astype(np.float32)

        if bboxes_np.shape[0] > 0:
            bboxes_np, masks = merge_overlapping_boxes(bboxes_np, masks, self.iou_threshold)

        ended_clips: list[TrackedClip] = []
        active_track_ids: set[int] = set()

        if bboxes_np.shape[0] == 0:
            for track_id in self.track_ids:
                ended_clips.append(self.active_clips.pop(track_id))
            self.last_frame_boxes = None
            self.track_ids = []
            return ended_clips, active_track_ids

        n_det = bboxes_np.shape[0]
        det_to_track: dict[int, int] = {}
        matched_tracks: set[int] = set()

        if self.last_frame_boxes is not None and self.track_ids:
            iou = compute_iou_matrix_np(bboxes_np, self.last_frame_boxes)
            matched_det = [False] * n_det
            n_tracks = len(self.track_ids)

            for _ in range(min(n_det, n_tracks)):
                best_iou, best_d, best_t = self.iou_threshold, -1, -1
                for d in range(n_det):
                    if matched_det[d]:
                        continue
                    for t in range(n_tracks):
                        if t in matched_tracks:
                            continue
                        if iou[d, t] > best_iou:
                            best_iou, best_d, best_t = iou[d, t], d, t
                if best_d < 0:
                    break
                matched_det[best_d] = True
                matched_tracks.add(best_t)
                det_to_track[best_d] = best_t

        for det_idx, track_idx in det_to_track.items():
            track_id = self.track_ids[track_idx]
            clip = self.active_clips[track_id]
            clip.bboxes.append(bboxes_np[det_idx])
            clip.masks.append(masks[det_idx])
            active_track_ids.add(track_id)

            if clip.frame_count >= self.max_clip_size:
                ended_clips.append(clip)
                del self.active_clips[track_id]

        for track_idx, track_id in enumerate(self.track_ids):
            if track_idx not in matched_tracks and track_id in self.active_clips:
                ended_clips.append(self.active_clips.pop(track_id))

        for det_idx in range(n_det):
            if det_idx not in det_to_track:
                track_id = self.next_track_id
                self.next_track_id += 1
                clip = TrackedClip(
                    track_id=track_id,
                    start_frame=frame_idx,
                    bboxes=[bboxes_np[det_idx]],
                    masks=[masks[det_idx]],
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

        self.last_frame_boxes = np.stack(new_boxes) if new_boxes else None
        self.track_ids = new_track_ids

        return ended_clips, active_track_ids

    def flush(self) -> list[TrackedClip]:
        """End all active clips and return them."""
        clips = list(self.active_clips.values())
        self.active_clips.clear()
        self.last_frame_boxes = None
        self.track_ids = []
        return clips

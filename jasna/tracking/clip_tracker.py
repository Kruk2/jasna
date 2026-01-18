from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrackedClip:
    track_id: int
    start_frame: int
    bboxes: list[torch.Tensor] = field(default_factory=list)  # each (4,) xyxy
    masks: list[torch.Tensor] = field(default_factory=list)  # each (H, W) bool

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.bboxes) - 1

    @property
    def frame_count(self) -> int:
        return len(self.bboxes)

    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.start_frame + len(self.bboxes)))


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes (xyxy format)."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def merge_overlapping_boxes(
    bboxes: torch.Tensor, masks: torch.Tensor, iou_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge overlapping bboxes within a single frame.
    bboxes: (K, 4) xyxy
    masks: (K, H, W) bool
    Returns merged (N, 4) bboxes and (N, H, W) masks where N <= K
    """
    if bboxes.shape[0] == 0:
        return bboxes, masks

    n = bboxes.shape[0]
    merged_indices: list[set[int]] = [{i} for i in range(n)]

    changed = True
    while changed:
        changed = False
        for i in range(len(merged_indices)):
            for j in range(i + 1, len(merged_indices)):
                for idx_i in merged_indices[i]:
                    for idx_j in merged_indices[j]:
                        if compute_iou(bboxes[idx_i], bboxes[idx_j]) > iou_threshold:
                            merged_indices[i] = merged_indices[i] | merged_indices[j]
                            merged_indices[j] = set()
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break

    merged_indices = [s for s in merged_indices if s]

    merged_bboxes = []
    merged_masks = []
    for group in merged_indices:
        group_list = list(group)
        group_boxes = bboxes[group_list]
        x1 = group_boxes[:, 0].min()
        y1 = group_boxes[:, 1].min()
        x2 = group_boxes[:, 2].max()
        y2 = group_boxes[:, 3].max()
        merged_bboxes.append(torch.tensor([x1, y1, x2, y2], device=bboxes.device))

        union_mask = masks[group_list].any(dim=0)
        merged_masks.append(union_mask)

    return torch.stack(merged_bboxes), torch.stack(merged_masks)


class ClipTracker:
    def __init__(self, max_clip_size: int, iou_threshold: float = 0.3):
        self.max_clip_size = max_clip_size
        self.iou_threshold = iou_threshold
        self.active_clips: dict[int, TrackedClip] = {}
        self.next_track_id = 0
        self.last_frame_boxes: dict[int, torch.Tensor] = {}  # track_id -> last bbox

    def update(
        self, frame_idx: int, bboxes: torch.Tensor, masks: torch.Tensor
    ) -> tuple[list[TrackedClip], set[int]]:
        """
        Update tracker with detections from a new frame.
        
        bboxes: (K, 4) xyxy format
        masks: (K, H, W) bool
        
        Returns:
            ended_clips: clips that ended this frame (max size or no match)
            active_track_ids: track ids that cover this frame
        """
        if bboxes.shape[0] > 0:
            bboxes, masks = merge_overlapping_boxes(bboxes, masks, self.iou_threshold)

        ended_clips: list[TrackedClip] = []
        active_track_ids: set[int] = set()

        if bboxes.shape[0] == 0:
            for track_id, clip in list(self.active_clips.items()):
                ended_clips.append(clip)
                del self.active_clips[track_id]
            self.last_frame_boxes.clear()
            return ended_clips, active_track_ids

        n_detections = bboxes.shape[0]
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()

        for det_idx in range(n_detections):
            det_box = bboxes[det_idx]
            best_track_id = None
            best_iou = self.iou_threshold

            for track_id, last_box in self.last_frame_boxes.items():
                if track_id in matched_tracks:
                    continue
                iou = compute_iou(det_box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                clip = self.active_clips[best_track_id]
                clip.bboxes.append(det_box)
                clip.masks.append(masks[det_idx])
                self.last_frame_boxes[best_track_id] = det_box
                matched_detections.add(det_idx)
                matched_tracks.add(best_track_id)
                active_track_ids.add(best_track_id)

                if clip.frame_count >= self.max_clip_size:
                    ended_clips.append(clip)
                    del self.active_clips[best_track_id]
                    del self.last_frame_boxes[best_track_id]

        for track_id in list(self.active_clips.keys()):
            if track_id not in matched_tracks:
                ended_clips.append(self.active_clips[track_id])
                del self.active_clips[track_id]
                if track_id in self.last_frame_boxes:
                    del self.last_frame_boxes[track_id]

        for det_idx in range(n_detections):
            if det_idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                clip = TrackedClip(
                    track_id=track_id,
                    start_frame=frame_idx,
                    bboxes=[bboxes[det_idx]],
                    masks=[masks[det_idx]],
                )
                self.active_clips[track_id] = clip
                self.last_frame_boxes[track_id] = bboxes[det_idx]
                active_track_ids.add(track_id)

        return ended_clips, active_track_ids

    def flush(self) -> list[TrackedClip]:
        """End all active clips and return them."""
        clips = list(self.active_clips.values())
        self.active_clips.clear()
        self.last_frame_boxes.clear()
        return clips

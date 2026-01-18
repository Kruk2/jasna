from __future__ import annotations

import torch

from jasna.restorer.frame_restorer import FrameRestorer
from jasna.tracking.clip_tracker import TrackedClip


class RestorationPipeline:
    def __init__(self, restorers: list[FrameRestorer]) -> None:
        self.restorers = restorers

    def restore_clip(
        self, clip: TrackedClip, frames: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Restore a clip by running all restorers sequentially.
        
        clip: TrackedClip with bbox/mask info
        frames: list of (C, H, W) original frames corresponding to clip frames
        
        Returns: list of (C, H_crop, W_crop) restored regions for each frame
        """
        crops: list[torch.Tensor] = []

        for i, frame in enumerate(frames):
            bbox = clip.bboxes[i].astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
            x2 = min(frame.shape[2], x2)
            y2 = min(frame.shape[1], y2)
            crops.append(frame[:, y1:y2, x1:x2])

        for restorer in self.restorers:
            crops = restorer.restore(crops)

        return crops


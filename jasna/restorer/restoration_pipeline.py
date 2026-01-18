from __future__ import annotations

import torch

from jasna.tracking.clip_tracker import TrackedClip


class RestorationPipeline:
    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)

    def restore_clip(
        self, clip: TrackedClip, frames: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Restore a clip.
        
        clip: TrackedClip with bbox/mask info
        frames: list of (C, H, W) original frames corresponding to clip frames
        
        Returns: list of (C, H_crop, W_crop) restored regions for each frame
        """
        restored_regions: list[torch.Tensor] = []

        for i, frame in enumerate(frames):
            bbox = clip.bboxes[i]
            x1, y1, x2, y2 = bbox.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[2], x2)
            y2 = min(frame.shape[1], y2)

            crop = frame[:, y1:y2, x1:x2]
            restored = self._restore_region(crop)
            restored_regions.append(restored)

        return restored_regions

    def _restore_region(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Restore a cropped region. Currently applies red tint as placeholder.
        Override or replace with actual restoration model.
        """
        crop_f16 = crop.to(dtype=torch.float16)
        red = torch.tensor(
            [255.0, 0.0, 0.0], device=crop.device, dtype=torch.float16
        )[:, None, None]
        blended = crop_f16 * (1.0 - self.alpha) + red * self.alpha
        return blended.clamp(0, 255).to(torch.uint8)


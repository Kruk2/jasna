from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FrameRestorer(ABC):
    @abstractmethod
    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Restore a list of cropped regions.
        
        crops: list of (C, H, W) uint8 tensors
        Returns: list of (C, H, W) uint8 restored tensors
        """
        ...


class RedTintRestorer(FrameRestorer):
    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        restored = []
        for crop in crops:
            crop_f16 = crop.to(dtype=torch.float16)
            red = torch.tensor(
                [255.0, 0.0, 0.0], device=crop.device, dtype=torch.float16
            )[:, None, None]
            blended = crop_f16 * (1.0 - self.alpha) + red * self.alpha
            restored.append(blended.clamp(0, 255).to(torch.uint8))
        return restored

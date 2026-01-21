import torch
import torch.nn.functional as F
from torch import Tensor

from jasna.models.basicvsrpp.inference import load_model

INFERENCE_SIZE = 256


class BasicvsrppMosaicRestorer:
    def __init__(self, checkpoint_path: str, device: torch.device, fp16: bool, config: str | dict | None = None):
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = load_model(config, checkpoint_path, self.device, fp16)

    def restore(self, video: list[Tensor]) -> list[Tensor]:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            list of (256, 256, C) uint8 tensors in RGB format
        """
        with torch.inference_mode():
            resized = []
            for frame in video:
                # RGB to BGR
                f = frame.flip(-1).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype).div_(255.0)
                f = F.interpolate(f, size=(INFERENCE_SIZE, INFERENCE_SIZE), mode='bilinear', align_corners=False)
                resized.append(f.squeeze(0))
            stacked = torch.stack(resized, dim=0)

            result = self.model(inputs=stacked.unsqueeze(0))

            result = result.squeeze(0)
            # BGR to RGB
            result = result.flip(1).mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1)

        return list(torch.unbind(result, 0))
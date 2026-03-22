import logging

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.models.basicvsrpp.inference import load_model
from jasna.restorer.basicvsrpp_sub_engines import create_split_forward

INFERENCE_SIZE = 256


class BasicvsrppMosaicRestorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        max_clip_size: int,
        use_tensorrt: bool,
        fp16: bool,
        config: str | dict | None = None,
    ):
        self.device = torch.device(device)
        self.max_clip_size = int(max_clip_size)
        self.use_tensorrt = bool(use_tensorrt)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.input_dtype = self.dtype

        self._split_forward = None
        self.model = None

        if self.use_tensorrt and self.device.type == "cuda":
            pytorch_model = load_model(config, checkpoint_path, self.device, fp16)
            self._split_forward = create_split_forward(
                model=pytorch_model,
                model_weights_path=checkpoint_path,
                device=self.device,
                fp16=fp16,
                max_clip_size=self.max_clip_size,
            )
            if self._split_forward is not None:
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        else:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            stacked = torch.stack(video).permute(0, 3, 1, 2).to(device=self.device, dtype=self.input_dtype, memory_format=torch.contiguous_format).div_(255.0)

            if self._split_forward is not None:
                result = self._split_forward(stacked.unsqueeze(0))
            else:
                result = self.model(inputs=stacked.unsqueeze(0))
            return result.squeeze(0)

    def restore(self, video: list[Tensor]) -> list[Tensor]:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            list of (256, 256, C) uint8 tensors in RGB format
        """
        result = self.raw_process(video)
        result = result.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1)
        return list(torch.unbind(result, 0))

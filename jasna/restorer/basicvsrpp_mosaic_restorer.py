import logging

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import is_amd_device, is_nvidia_device
from jasna.models.basicvsrpp.inference import load_model
from jasna.restorer.basicvsrpp_hip_graph import HipGraphReplay, graph_lengths



class BasicvsrppMosaicRestorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        max_clip_size: int,
        use_tensorrt: bool,
        fp16: bool,
        config: str | dict | None = None,
        temporal_overlap: int = 0,
    ):
        self.device = torch.device(device)
        self.max_clip_size = int(max_clip_size)
        self.input_dtype = torch.float16 if fp16 else torch.float32

        self._split_forward = None
        self.model = None
        self._hip_graph = HipGraphReplay(self.device)

        if use_tensorrt and is_nvidia_device(self.device):
            from jasna.restorer.basicvsrpp_sub_engines import create_split_forward

            pytorch_model = load_model(config, checkpoint_path, self.device, fp16)
            self._split_forward = create_split_forward(
                model=pytorch_model,
                model_weights_path=checkpoint_path,
                device=self.device,
                fp16=fp16,
            )
            if self._split_forward is not None:
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        elif is_amd_device(self.device):
            from jasna.restorer.basicvsrpp_sub_engines import create_eager_split_forward

            self._split_forward = create_eager_split_forward(load_model(config, checkpoint_path, self.device, fp16))
            logger.info("BasicVSR++ using the split forward with PyTorch sub-modules (fp16=%s)", fp16)
        else:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

        # Capture before any pipeline thread exists: capture poisons concurrent
        # CUDA calls from other threads, replay is thread-safe.
        if self._split_forward is not None and self._hip_graph.enabled:
            self._hip_graph.capture(
                self._split_forward,
                graph_lengths(self.max_clip_size, temporal_overlap),
                dtype=self.input_dtype,
            )

    def close(self) -> None:
        self._hip_graph.close()
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = None
        self.model = None

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (C, H, W) tensors in RGB format, [0, 255]
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            stacked = torch.stack(video).to(device=self.device, dtype=self.input_dtype, memory_format=torch.contiguous_format).div_(255.0)

            batched = stacked.unsqueeze(0)
            if self._split_forward is not None:
                replayed = self._hip_graph.replay(batched)
                if replayed is not None:
                    return replayed.squeeze(0)
                result = self._split_forward(batched)
            else:
                result = self.model(inputs=batched)
            return result.squeeze(0)

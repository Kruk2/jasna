import logging
import os

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import is_amd_device, is_nvidia_device
from jasna.models.basicvsrpp.inference import load_model

AMD_GRAPHS_ENV = "JASNA_AMD_GRAPHS"


class _ClipGraph:
    """The split forward captured once for full-size clips, which carry almost every restored frame.

    Replay skips the per-kernel launch cost that dominates the eager forward on AMD. Capture runs
    before the pipeline threads start; other lengths keep running eager.
    """

    def __init__(self, forward, clip_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._input = torch.zeros((1, clip_size, 3, 256, 256), device=device, dtype=dtype)
        with torch.inference_mode():
            forward(self._input)  # kernel selection has to happen outside the capture
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
                self._output = forward(self._input)

    def fits(self, lqs: Tensor) -> bool:
        return lqs.shape == self._input.shape

    def __call__(self, lqs: Tensor) -> Tensor:
        self._input.copy_(lqs)
        self._graph.replay()
        return self._output.clone()


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
        self.input_dtype = torch.float16 if fp16 else torch.float32

        self._split_forward = None
        self._clip_graph = None
        self.model = None

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
            if os.environ.get(AMD_GRAPHS_ENV, "1") != "0":
                self._clip_graph = _ClipGraph(self._split_forward, self.max_clip_size, self.device, self.input_dtype)
            logger.info(
                "BasicVSR++ using the split forward with PyTorch sub-modules (fp16=%s, graph=%s)",
                fp16, self._clip_graph is not None,
            )
        else:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

    def close(self) -> None:
        self._clip_graph = None
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

            clip = stacked.unsqueeze(0)
            if self._clip_graph is not None and self._clip_graph.fits(clip):
                result = self._clip_graph(clip)
            elif self._split_forward is not None:
                result = self._split_forward(clip)
            else:
                result = self.model(inputs=clip)
            return result.squeeze(0)

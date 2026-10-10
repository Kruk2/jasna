import logging
import os

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import is_nvidia_device
from jasna.models.basicvsrpp.inference import load_model
from jasna.os_utils import env_flag

# On AMD (no TensorRT sub-engines) the BasicVSR++ forward runs ~150k small CUDA
# ops per call, which makes it launch/dispatch bound: HIP-graph replay of the
# captured forward measured 534 -> 292 ms at T=60 (bit-identical output) on an
# RX 7900 XT (torch 2.14+rocm10.1, Windows).
#
# Graphs need static shapes AND a quiet GPU: capturing while the pipeline's
# other threads submit CUDA work poisons their calls ("operation not permitted
# when stream is capturing" aborts the decode/encode threads on this ROCm
# build). Capture therefore happens once here, at construction time - before
# any pipeline thread exists - for the two clip lengths that cover virtually
# every clip of a run: max_clip_size and max_clip_size - 2*temporal_overlap.
# Any other length (final partial clip, scene-cut clip) runs eager.
# JASNA_BV_HIP_GRAPH=0 disables the feature entirely.
_HIP_GRAPH_ENV = "JASNA_BV_HIP_GRAPH"
# Extra clip lengths to capture, comma separated (e.g. "52,40"). Any clip whose
# length was not captured runs eager, ~1.8x slower per frame on the same card, so
# a run whose clips end early (scene cuts, mosaics that disappear) pays that on
# every such clip. Capture cost is one eager forward plus the graph's reserved
# activations per length, so add only the lengths the clip histogram shows.
_HIP_GRAPH_LENGTHS_ENV = "JASNA_BV_HIP_GRAPH_LENGTHS"


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
        self._graphs: dict[int, tuple[Tensor, Tensor, "torch.cuda.CUDAGraph"]] = {}
        self._graphs_enabled = (
            env_flag(_HIP_GRAPH_ENV, default=True)
            and self.device.type == "cuda"
        )

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
                self._graphs_enabled = False
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        else:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

        if self._graphs_enabled:
            self._prewarm_graphs(int(temporal_overlap))

    def _prewarm_graphs(self, temporal_overlap: int) -> None:
        """Capture graphs for the two dominant clip lengths before the
        pipeline threads start (mid-pipeline capture is not safe on ROCm)."""
        lengths: list[int] = []
        t1 = self.max_clip_size
        t2 = t1 - 2 * int(temporal_overlap)
        if t1 > 1:
            lengths.append(t1)
        if 1 < t2 < t1:
            lengths.append(t2)
        for token in os.environ.get(_HIP_GRAPH_LENGTHS_ENV, "").split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            extra = int(token)
            if extra > 1 and extra not in lengths:
                lengths.append(extra)
        for t in lengths:
            reserved_before = torch.cuda.memory_reserved(self.device)
            try:
                with torch.inference_mode():
                    static_input = torch.zeros(
                        (1, t, 3, 256, 256), device=self.device, dtype=self.input_dtype
                    )
                    # the eager call doubles as the capture warmup (MIOpen
                    # kernel selection must happen outside the capture)
                    self.model(inputs=static_input)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_output = self.model(inputs=static_input)
            except Exception:
                logger.warning(
                    "BasicVSR++ HIP graph capture failed for T=%d; that clip"
                    " length will run eager.", t, exc_info=True,
                )
                torch.cuda.synchronize(self.device)
                continue
            self._graphs[t] = (static_input, static_output, graph)
            logger.info(
                "BasicVSR++ HIP graph captured for T=%d (+%.0f MiB reserved)",
                t,
                (torch.cuda.memory_reserved(self.device) - reserved_before) / (1024 * 1024),
            )
        if self._graphs:
            logger.info(
                "BasicVSR++ HIP graphs ready for clip lengths %s (JASNA_BV_HIP_GRAPH=0 disables)",
                sorted(self._graphs),
            )

    def close(self) -> None:
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = None
        self._graphs.clear()
        self._graphs_enabled = False
        self.model = None

    def _forward_graphed(self, batched: Tensor) -> Tensor:
        entry = self._graphs.get(int(batched.shape[1]))
        if entry is None:
            return self.model(inputs=batched)
        static_input, static_output, graph = entry
        static_input.copy_(batched)
        graph.replay()
        # static_output is a fixed buffer that the next replay overwrites;
        # callers may hold the result until long after, so hand out a copy.
        return static_output.clone()

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
                result = self._split_forward(batched)
            else:
                result = self._forward_graphed(batched)
            return result.squeeze(0)

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from torch import Tensor

from jasna.models.basicvsrpp.inference import load_model
from jasna.restorer.basicvrspp_tenorrt_compilation import (
    SMALL_TRT_CLIP_LENGTH,
    SMALL_TRT_CLIP_LENGTH_TRIGGER,
    get_compiled_mosaic_restoration_model_path_for_clip,
    load_engine,
)

INFERENCE_SIZE = 256


def _try_parse_trt_clip_length(checkpoint_path: str) -> int | None:
    m = re.search(r"_clip(\d+)\.trt_", checkpoint_path)
    if m is None:
        return None
    return int(m.group(1))


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

        self._engine_small = None
        self._engine_main = None
        self._engine_main_len: int | None = None
        self._trt_pre_event: torch.cuda.Event | None = None
        self._trt_post_event: torch.cuda.Event | None = None

        if checkpoint_path.endswith(".engine"):
            self._engine_main = load_engine(checkpoint_path, self.device)
            self._engine_main_len = _try_parse_trt_clip_length(checkpoint_path)
            self.model = None
            return

        if self.use_tensorrt and self.device.type == "cuda":
            main_path = get_compiled_mosaic_restoration_model_path_for_clip(
                checkpoint_path=checkpoint_path,
                clip_length=self.max_clip_size,
                fp16=fp16,
            )
            if os.path.isfile(main_path):
                self._engine_main = load_engine(main_path, self.device)
                self._engine_main_len = self.max_clip_size

            if self.max_clip_size > SMALL_TRT_CLIP_LENGTH_TRIGGER:
                small_path = get_compiled_mosaic_restoration_model_path_for_clip(
                    checkpoint_path=checkpoint_path,
                    clip_length=SMALL_TRT_CLIP_LENGTH,
                    fp16=fp16,
                )
                if os.path.isfile(small_path):
                    self._engine_small = load_engine(small_path, self.device)

        if self._engine_main is None:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)
        else:
            self.model = None
            logger.info("BasicVSR++ using TensorRT engine: clip_len=%s fp16=%s", self._engine_main_len, fp16)

        if (self._engine_main is not None or self._engine_small is not None) and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._trt_pre_event = torch.cuda.Event()
                self._trt_post_event = torch.cuda.Event()

    def _select_engine(self, t: int):
        if self._engine_main is None and self._engine_small is None:
            return None, 0

        if self._engine_small is not None and t <= SMALL_TRT_CLIP_LENGTH:
            return self._engine_small, SMALL_TRT_CLIP_LENGTH

        if self._engine_main is not None:
            pad_to = int(self._engine_main_len) if self._engine_main_len is not None else self.max_clip_size
            return self._engine_main, pad_to

        return None, 0

    def _engine_infer(self, engine, stacked: torch.Tensor, t: int, pad_to: int) -> torch.Tensor:
        if t < pad_to:
            if t == 1:
                idx = torch.zeros((pad_to,), dtype=torch.long, device=stacked.device)
            else:
                base = list(range(t)) + list(range(t - 2, 0, -1))
                reps = (pad_to + len(base) - 1) // len(base)
                idx = torch.tensor((base * reps)[:pad_to], dtype=torch.long, device=stacked.device)
            stacked = stacked.index_select(0, idx)

        cur_stream = torch.cuda.current_stream(stacked.device)
        default_stream = torch.cuda.default_stream(stacked.device)

        self._trt_pre_event.record(cur_stream)
        default_stream.wait_event(self._trt_pre_event)
        with torch.cuda.stream(default_stream):
            result = engine(stacked.unsqueeze(0))
        self._trt_post_event.record(default_stream)
        cur_stream.wait_event(self._trt_post_event)
        result = result.squeeze(0)
        return result[:t]

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            resized = []
            for frame in video:
                f = frame.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.input_dtype).div_(255.0)
                f = F.interpolate(f, size=(INFERENCE_SIZE, INFERENCE_SIZE), mode="bilinear", align_corners=False)
                resized.append(f.squeeze(0))
            stacked = torch.stack(resized, dim=0)

            t = int(stacked.shape[0])
            engine, pad_to = self._select_engine(t)
            if engine is not None:
                if t > pad_to:
                    raise RuntimeError(f"clip length {t} exceeds TensorRT compiled clip length {pad_to}")
                result = self._engine_infer(engine, stacked, t, pad_to)
            else:
                result = self.model(inputs=stacked.unsqueeze(0))

            if engine is None:
                result = result.squeeze(0)
            return result

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

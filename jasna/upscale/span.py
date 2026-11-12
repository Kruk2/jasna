"""Whole-frame upscaling with the SPAN 2x network: TensorRT on NVIDIA (one engine per input
size), PyTorch on AMD. Upscales needing more than 2x run it twice."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch

from jasna.engine_paths import engine_precision_name, engine_system_suffix, model_weights_dir
from jasna.upscale import Size, finish_frame

logger = logging.getLogger(__name__)

SPAN_SCALE = 2
SPAN_WEIGHTS_PATH = model_weights_dir() / "span-nomosuni-2x.safetensors"
_MAX_PASSES = 2
_ENGINE_WORKSPACE_GB = 8


def span_input_sizes(source_size: Size, output_size: Size) -> tuple[Size, ...]:
    """The input size of each SPAN pass: one 2x pass, or two when the output is more than
    twice as tall (the second pass upscales the first one's output)."""
    width, height = source_size
    passes = 1
    while passes < _MAX_PASSES and height * SPAN_SCALE**passes < output_size[1]:
        passes += 1
    return tuple((width * SPAN_SCALE**index, height * SPAN_SCALE**index) for index in range(passes))


def span_engine_path(input_size: Size, *, fp16: bool) -> Path:
    width, height = input_size
    stem = SPAN_WEIGHTS_PATH.with_suffix("").name
    return SPAN_WEIGHTS_PATH.with_name(
        f"{stem}.{width}x{height}.{engine_precision_name(fp16=fp16)}{engine_system_suffix()}.engine"
    )


def missing_span_engines(input_sizes: tuple[Size, ...], *, fp16: bool) -> tuple[Size, ...]:
    return tuple(size for size in input_sizes if not span_engine_path(size, fp16=fp16).exists())


def compile_span_engine(input_size: Size, device: torch.device, *, fp16: bool) -> Path:
    from jasna.models.span import load_span
    from jasna.trt import _build_serialized_engine

    width, height = input_size
    network = load_span(SPAN_WEIGHTS_PATH, scale=SPAN_SCALE)
    onnx = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(
            network, (torch.zeros(1, 3, height, width),), onnx,
            input_names=["image"], output_names=["upscaled"], opset_version=17, dynamo=False,
        )
    engine = _build_serialized_engine(
        onnx.getvalue(), device, batch_size=None, fp16=fp16, optimization_level=5, workspace_gb=_ENGINE_WORKSPACE_GB
    )
    path = span_engine_path(input_size, fp16=fp16)
    path.write_bytes(engine)
    return path


class SpanFrameUpscaler:
    def __init__(
        self, device: torch.device, *, source_size: Size, output_size: Size, fp16: bool, strength: float
    ) -> None:
        from jasna.accelerator import is_nvidia_device

        self._output_size = output_size
        self._strength = strength
        self._runners = []
        input_sizes = span_input_sizes(source_size, output_size)
        if is_nvidia_device(device):
            from jasna.trt.trt_runner import TrtRunner

            for width, height in input_sizes:
                runner = TrtRunner(span_engine_path((width, height), fp16=fp16), {"image": (1, 3, height, width)}, device)
                self._runners.append(runner)
            self._dtype = self._runners[0].input_dtypes["image"]
            self._passes = [lambda image, runner=runner: runner.infer({"image": image})["upscaled"] for runner in self._runners]
        else:
            from jasna.models.span import load_span

            self._dtype = torch.float16 if fp16 else torch.float32
            network = load_span(SPAN_WEIGHTS_PATH, scale=SPAN_SCALE).to(device=device, dtype=self._dtype)
            self._passes = [network] * len(input_sizes)

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        image = frame[None].to(self._dtype).div_(255.0)
        for index, upscale in enumerate(self._passes):
            if index:
                image = image.clamp_(0.0, 1.0)
            image = upscale(image)
        return finish_frame(image, frame, self._output_size, self._strength)

    def close(self) -> None:
        for runner in self._runners:
            runner.close()
        self._runners.clear()
        self._passes = []

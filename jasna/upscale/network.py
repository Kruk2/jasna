"""Whole-frame upscaling with a super-resolution network: TensorRT on NVIDIA (one engine per
network and input size), PyTorch on AMD. A network runs twice when one pass leaves the frame
much smaller than the output."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from jasna.engine_paths import engine_precision_name, engine_system_suffix, model_weights_dir
from jasna.upscale import Size, finish_frame

_MAX_PASSES = 2
# The plain resize at the end may enlarge this much; more needs another network pass.
_RESIZE_FINISH_MAX = 1.25
_ENGINE_WORKSPACE_GB = 8


def _load_span(path: Path, scale: int) -> nn.Module:
    from jasna.models.span import load_span

    return load_span(path, scale=scale)


def _load_realplksr(path: Path, scale: int) -> nn.Module:
    from jasna.models.realplksr import load_realplksr

    return load_realplksr(path, scale=scale)


@dataclass(frozen=True)
class UpscaleNetwork:
    weights_path: Path
    scale: int
    load: Callable[[Path, int], nn.Module]

    def network(self) -> nn.Module:
        return self.load(self.weights_path, self.scale)


UPSCALE_NETWORKS: dict[str, UpscaleNetwork] = {
    "span": UpscaleNetwork(model_weights_dir() / "span-nomosuni-2x.safetensors", 2, _load_span),
    "realplksr": UpscaleNetwork(model_weights_dir() / "realplksr-nomoswebphoto-4x.safetensors", 4, _load_realplksr),
}
"""The ``SessionConfig.upscale_model`` values that run a network, and their network."""


def network_input_sizes(network: UpscaleNetwork, source_size: Size, output_size: Size) -> tuple[Size, ...]:
    """The input size of each pass: one, or two when the output is more than
    ``_RESIZE_FINISH_MAX`` taller than one pass makes it (the second pass upscales the first
    one's output)."""
    width, height = source_size
    passes = 1
    while passes < _MAX_PASSES and output_size[1] > height * network.scale**passes * _RESIZE_FINISH_MAX:
        passes += 1
    return tuple((width * network.scale**index, height * network.scale**index) for index in range(passes))


def network_engine_path(network: UpscaleNetwork, input_size: Size, *, fp16: bool) -> Path:
    width, height = input_size
    stem = network.weights_path.with_suffix("").name
    return network.weights_path.with_name(
        f"{stem}.{width}x{height}.{engine_precision_name(fp16=fp16)}{engine_system_suffix()}.engine"
    )


def missing_network_engines(network: UpscaleNetwork, input_sizes: tuple[Size, ...], *, fp16: bool) -> tuple[Size, ...]:
    return tuple(size for size in input_sizes if not network_engine_path(network, size, fp16=fp16).exists())


def compile_network_engine(network: UpscaleNetwork, input_size: Size, device: torch.device, *, fp16: bool) -> Path:
    from jasna.trt import _build_serialized_engine

    width, height = input_size
    onnx = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            network.network(), (torch.zeros(1, 3, height, width),), onnx,
            input_names=["image"], output_names=["upscaled"], opset_version=17, dynamo=False,
        )
    engine = _build_serialized_engine(
        onnx.getvalue(), device, batch_size=None, fp16=fp16, optimization_level=5, workspace_gb=_ENGINE_WORKSPACE_GB
    )
    path = network_engine_path(network, input_size, fp16=fp16)
    path.write_bytes(engine)
    return path


class NetworkFrameUpscaler:
    def __init__(
        self,
        network: UpscaleNetwork,
        device: torch.device,
        *,
        source_size: Size,
        output_size: Size,
        fp16: bool,
        strength: float,
    ) -> None:
        from jasna.accelerator import is_nvidia_device

        self._output_size = output_size
        self._strength = strength
        self._runners = []
        input_sizes = network_input_sizes(network, source_size, output_size)
        if is_nvidia_device(device):
            from jasna.trt.trt_runner import TrtRunner

            for width, height in input_sizes:
                path = network_engine_path(network, (width, height), fp16=fp16)
                self._runners.append(TrtRunner(path, {"image": (1, 3, height, width)}, device))
            self._dtype = self._runners[0].input_dtypes["image"]
            self._passes = [lambda image, runner=runner: runner.infer({"image": image})["upscaled"] for runner in self._runners]
        else:
            self._dtype = torch.float16 if fp16 else torch.float32
            module = network.network().to(device=device, dtype=self._dtype)
            self._passes = [module] * len(input_sizes)

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

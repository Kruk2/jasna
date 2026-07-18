from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from jasna.accelerator import device_name

logger = logging.getLogger(__name__)

_MIGRAPHX_PROVIDER = "MIGraphXExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_ORT_DTYPES: dict[str, tuple[torch.dtype, np.dtype]] = {
    "tensor(float)": (torch.float32, np.dtype(np.float32)),
    "tensor(float16)": (torch.float16, np.dtype(np.float16)),
    "tensor(int64)": (torch.int64, np.dtype(np.int64)),
    "tensor(int32)": (torch.int32, np.dtype(np.int32)),
    "tensor(uint8)": (torch.uint8, np.dtype(np.uint8)),
    "tensor(bool)": (torch.bool, np.dtype(np.bool_)),
}


@dataclass(frozen=True)
class MigraphxTensorInfo:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


def _model_digest(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_model_digest(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=16)
def _cached_model_digest(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as model:
        while chunk := model.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _gpu_arch(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "") or device_name(device)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(architecture)).strip("-") or "amd-gpu"


def migraphx_cache_dir(
    onnx_path: Path,
    device: torch.device,
    *,
    fp16: bool,
) -> Path:
    precision = "fp16" if fp16 else "fp32"
    key = f"{_model_digest(onnx_path)}-{_gpu_arch(device)}-{precision}"
    return onnx_path.parent / f"{onnx_path.stem}.migraphx" / key


def migraphx_cache_is_ready(
    onnx_path: Path,
    device: torch.device,
    *,
    fp16: bool,
) -> bool:
    directory = migraphx_cache_dir(onnx_path, device, fp16=fp16)
    return directory.is_dir() and any(path.is_file() for path in directory.rglob("*"))


def migraphx_provider_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return _MIGRAPHX_PROVIDER in ort.get_available_providers()


def _shape(node, *, kind: str) -> tuple[int, ...]:
    dimensions = tuple(node.shape)
    if any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise RuntimeError(
            f"MIGraphX requires fixed positive {kind} dimensions for {node.name}: "
            f"{dimensions}"
        )
    return tuple(int(value) for value in dimensions)


def _dtype(node) -> tuple[torch.dtype, np.dtype]:
    try:
        return _ORT_DTYPES[str(node.type)]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported ONNX Runtime tensor type for {node.name}: {node.type}"
        ) from exc


class MigraphxRunner:
    def __init__(
        self,
        onnx_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        *,
        fp16: bool,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR on AMD requires the onnxruntime package"
            ) from exc

        available = set(ort.get_available_providers())
        use_migraphx = _MIGRAPHX_PROVIDER in available
        if not use_migraphx and _CPU_PROVIDER not in available:
            raise RuntimeError(
                "RF-DETR on AMD requires the MIGraphX or CPU ONNX Runtime provider; "
                "available providers: "
                + ", ".join(sorted(available))
            )

        self.device = torch.device(device)
        self.execution_provider = (
            _MIGRAPHX_PROVIDER if use_migraphx else _CPU_PROVIDER
        )
        self.cache_dir: Path | None = None
        if use_migraphx:
            self.cache_dir = migraphx_cache_dir(
                onnx_path,
                self.device,
                fp16=fp16,
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            provider_options = {
                "device_id": str(self.device.index or 0),
                "migraphx_fp16_enable": "1" if fp16 else "0",
                "migraphx_model_cache_dir": str(self.cache_dir),
            }
            providers = [
                (_MIGRAPHX_PROVIDER, provider_options),
                _CPU_PROVIDER,
            ]
        else:
            providers = [_CPU_PROVIDER]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
        )
        active = self.session.get_providers()
        if not active or active[0] != self.execution_provider:
            raise RuntimeError(
                f"{self.execution_provider} did not become the primary "
                "RF-DETR execution provider: "
                + ", ".join(active)
            )

        input_nodes = self.session.get_inputs()
        output_nodes = self.session.get_outputs()
        self.input_names = [node.name for node in input_nodes]
        self.output_names = [node.name for node in output_nodes]
        if isinstance(input_shapes, list):
            input_shapes = dict(zip(self.input_names, input_shapes, strict=True))

        self.input_dtypes: dict[str, torch.dtype] = {}
        self._input_numpy_dtypes: dict[str, np.dtype] = {}
        self._host_inputs: dict[str, torch.Tensor] = {}
        for node in input_nodes:
            model_shape = _shape(node, kind="input")
            requested_shape = tuple(int(value) for value in input_shapes[node.name])
            if requested_shape != model_shape:
                raise ValueError(
                    f"RF-DETR ONNX input {node.name} is fixed at {model_shape}; "
                    f"requested {requested_shape}"
                )
            torch_dtype, numpy_dtype = _dtype(node)
            self.input_dtypes[node.name] = torch_dtype
            self._input_numpy_dtypes[node.name] = numpy_dtype
            self._host_inputs[node.name] = torch.empty(
                model_shape,
                dtype=torch_dtype,
                device="cpu",
                pin_memory=self.device.type != "cpu",
            )

        self.outputs: dict[str, MigraphxTensorInfo] = {}
        for node in output_nodes:
            torch_dtype, _ = _dtype(node)
            self.outputs[node.name] = MigraphxTensorInfo(
                shape=_shape(node, kind="output"),
                dtype=torch_dtype,
            )

        logger.info(
            "ONNX model loaded: %s on %s (provider=%s, cache=%s)",
            onnx_path,
            device_name(self.device),
            self.execution_provider,
            self.cache_dir,
        )

    def close(self) -> None:
        self._host_inputs.clear()
        self.outputs.clear()
        self.session = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.session is None:
            raise RuntimeError("MIGraphX runner is closed")

        feeds: dict[str, np.ndarray] = {}
        for name, tensor in inputs.items():
            host = self._host_inputs[name]
            host.copy_(
                tensor.detach().to(dtype=self.input_dtypes[name]).contiguous(),
                non_blocking=False,
            )
            feeds[name] = host.numpy()

        arrays = self.session.run(self.output_names, feeds)
        return {
            name: torch.from_numpy(array).to(device=self.device, non_blocking=False)
            for name, array in zip(self.output_names, arrays, strict=True)
        }

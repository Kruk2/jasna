from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from jasna.accelerator import is_amd_device
from jasna.migraphx_artifact import MigraphxArtifactRunner


logger = logging.getLogger(__name__)

_PURPOSE = "jasna-rfdetr-v6-medium-segmentation"
_PLATFORM = "linux"
_DEVICE_ARCH = "gfx1100"
PRODUCT_MANIFEST_FILENAME = f"rfdetr-v6.migraphx-{_DEVICE_ARCH}.json"
_SUPPORTED_STATIC_BATCHES = frozenset({1, 2})
_SUPPORTED_INTERNAL_PRECISIONS = frozenset(
    {
        "fp16",
        "mixed-convolution-dot-fp16",
        "mixed-dot-projector-convolution-fp16",
    }
)
_INPUT_SHAPE_TAIL = (3, 576, 576)
_INPUT_STRIDES = (995328, 331776, 576, 1)
_OUTPUT_CONTRACTS = {
    "dets": {
        "parameter_name": "main:#output_0",
        "shape_tail": (200, 4),
        "strides": (800, 4, 1),
    },
    "labels": {
        "parameter_name": "main:#output_1",
        "shape_tail": (200, 3),
        "strides": (600, 3, 1),
    },
    "masks": {
        "parameter_name": "main:#output_2",
        "shape_tail": (200, 144, 144),
        "strides": (4147200, 20736, 144, 1),
    },
}


def discover_product_rfdetr_migraphx_manifest(
    *,
    detection_model_name: str,
    weights_path: Path,
    device: torch.device,
    fp16: bool,
) -> Path | None:
    """Return the installed product artifact only for its proven host scope.

    Ineligibility or absence is a selection decision and preserves the existing
    PyTorch RF-DETR route. Once a path is returned, the strict runner owns every
    file/hash/runtime/ABI check and failures must propagate without fallback.
    """

    resolved_device = torch.device(device)
    if (
        sys.platform != _PLATFORM
        or not bool(fp16)
        or str(detection_model_name).strip().casefold() != "rfdetr-v6"
        or not is_amd_device(resolved_device)
        or resolved_device.type != "cuda"
        or getattr(torch.version, "hip", None) is None
        or not torch.cuda.is_available()
    ):
        return None

    candidate = Path(weights_path).parent / PRODUCT_MANIFEST_FILENAME
    if not candidate.is_file():
        return None
    properties = torch.cuda.get_device_properties(resolved_device)
    if getattr(properties, "gcnArchName", None) != _DEVICE_ARCH:
        return None
    return candidate


class RfDetrMigraphxRunner:
    """Linux gfx1100 AMD RF-DETR runner backed by a verified MXR sidecar.

    Selection is deliberately outside this module. Once an explicit sidecar
    reaches this runner, every validation or execution failure is terminal and
    must not fall back to the PyTorch detector.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        weights_path: Path,
        device: torch.device,
        fp16: bool,
        resolution: int,
        variant: str | None,
    ) -> None:
        if not fp16:
            raise RuntimeError(
                "The experimental RF-DETR MIGraphX artifact requires "
                "FP16-enabled execution; --no-fp16 is incompatible"
            )
        if int(resolution) != 576 or variant != "medium":
            raise RuntimeError(
                "The experimental RF-DETR MIGraphX artifact only supports "
                "rfdetr-v6 medium at 576x576"
            )
        self._validate_host(device)
        runner = MigraphxArtifactRunner(
            manifest_path,
            source_path=weights_path,
            device=device,
            expected_purpose=_PURPOSE,
        )
        try:
            self._validate_contract(runner)
        except BaseException:
            runner.close()
            raise
        self._runner = runner
        self.input_names = runner.input_names
        self.input_dtypes = runner.input_dtypes
        self.output_names = runner.output_names
        self.outputs = runner.outputs
        self.batch_size = runner.batch_size
        self.artifact_path = runner.artifact_path
        logger.info(
            "AMD RF-DETR direct MIGraphX artifact loaded: %s "
            "(static_batch=%d, internal_precision=%s, fallback=False)",
            self.artifact_path,
            self.batch_size,
            runner.manifest.internal_precision,
        )

    @staticmethod
    def _validate_host(device: torch.device) -> None:
        """Reject ineligible hosts before importing/loading an MXR runtime."""

        resolved = torch.device(device)
        if sys.platform != _PLATFORM:
            raise RuntimeError("RF-DETR MIGraphX artifacts are supported on Linux only")
        if not is_amd_device(resolved):
            raise RuntimeError("RF-DETR MIGraphX artifacts require an AMD/ROCm device")
        if resolved.type != "cuda" or getattr(torch.version, "hip", None) is None:
            raise RuntimeError("RF-DETR MIGraphX artifacts require a ROCm CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("RF-DETR MIGraphX artifacts require an available ROCm GPU")
        properties = torch.cuda.get_device_properties(resolved)
        if getattr(properties, "gcnArchName", None) != _DEVICE_ARCH:
            raise RuntimeError(
                "RF-DETR MIGraphX artifacts require "
                f"{_DEVICE_ARCH}, got {getattr(properties, 'gcnArchName', None)!r}"
            )

    @staticmethod
    def _validate_contract(runner: MigraphxArtifactRunner) -> None:
        manifest = runner.manifest
        if manifest.platform != _PLATFORM or manifest.device_arch != _DEVICE_ARCH:
            raise RuntimeError(
                "RF-DETR MIGraphX target mismatch: "
                f"platform={manifest.platform!r}, device_arch={manifest.device_arch!r}; "
                f"expected {_PLATFORM!r}/{_DEVICE_ARCH!r}"
            )
        if manifest.internal_precision not in _SUPPORTED_INTERNAL_PRECISIONS:
            raise RuntimeError(
                "RF-DETR MIGraphX internal precision is outside the proven set: "
                f"{manifest.internal_precision!r}; expected one of "
                f"{sorted(_SUPPORTED_INTERNAL_PRECISIONS)!r}"
            )
        if manifest.stream_type != "ihipStream_t":
            raise RuntimeError(
                f"RF-DETR MIGraphX stream ABI mismatch: {manifest.stream_type!r}"
            )
        if len(manifest.inputs) != 1:
            raise RuntimeError("RF-DETR MIGraphX requires exactly one input contract")
        input_contract = manifest.inputs[0]
        if len(input_contract.shape) != 4:
            raise RuntimeError(
                "RF-DETR MIGraphX input contract must have rank 4: "
                f"{input_contract!r}"
            )
        batch_size = int(input_contract.shape[0])
        if batch_size not in _SUPPORTED_STATIC_BATCHES:
            raise RuntimeError(
                "RF-DETR MIGraphX static batch is outside the proven set: "
                f"{batch_size}; expected one of {sorted(_SUPPORTED_STATIC_BATCHES)}"
            )
        expected_input = {
            "logical_name": "input",
            "parameter_name": "input",
            "dtype_name": "float32",
            "shape": (batch_size, *_INPUT_SHAPE_TAIL),
            "strides": _INPUT_STRIDES,
        }
        actual_input = {
            "logical_name": input_contract.logical_name,
            "parameter_name": input_contract.parameter_name,
            "dtype_name": input_contract.dtype_name,
            "shape": input_contract.shape,
            "strides": input_contract.strides,
        }
        if actual_input != expected_input:
            raise RuntimeError(
                "RF-DETR MIGraphX input contract mismatch: "
                f"{actual_input!r} != {expected_input!r}"
            )
        if [item.logical_name for item in manifest.outputs] != list(_OUTPUT_CONTRACTS):
            raise RuntimeError(
                "RF-DETR MIGraphX logical output names/order must be "
                "['dets', 'labels', 'masks']"
            )
        for contract in manifest.outputs:
            expected = _OUTPUT_CONTRACTS[contract.logical_name]
            if (
                contract.dtype_name != "float32"
                or contract.parameter_name != expected["parameter_name"]
                or contract.shape != (batch_size, *expected["shape_tail"])
                or contract.strides != expected["strides"]
            ):
                raise RuntimeError(
                    f"RF-DETR MIGraphX output contract mismatch: {contract!r}"
                )

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._runner is None:
            raise RuntimeError("RF-DETR MIGraphX runner is closed")
        return self._runner.infer(inputs)

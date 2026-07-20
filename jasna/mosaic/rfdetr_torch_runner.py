from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from jasna.accelerator import device_name

logger = logging.getLogger(__name__)

# RF-DETR-Seg size -> rfdetr package class. AMD runs the .pt through the rfdetr
# torch model (no ONNX/MIGraphX); the size cannot be inferred from the trimmed
# deploy checkpoint args, so the detection registry supplies it.
_VARIANT_CLASSES: dict[str, str] = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}

_ONNX_OUTPUT_NAMES = ("dets", "labels", "masks")


@dataclass(frozen=True)
class TorchTensorInfo:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


class RfDetrTorchRunner:
    """AMD/ROCm RF-DETR path: run the trained checkpoint through the rfdetr torch
    model instead of ONNX + MIGraphX. Mirrors the runner contract consumed by
    ``RfDetrMosaicDetectionModel`` (``input_names``/``input_dtypes``/``outputs``/
    ``output_names``/``infer``/``close``) and emits the same ``dets``/``labels``/
    ``masks`` tensors the ONNX export produces (LWDETR ``pred_boxes``/
    ``pred_logits``/``pred_masks``)."""

    def __init__(
        self,
        weights_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        *,
        fp16: bool,
        resolution: int,
        variant: str,
    ) -> None:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR on AMD requires the rfdetr package (jasna[amd])"
            ) from exc

        cls_name = _VARIANT_CLASSES.get(variant)
        if cls_name is None:
            raise RuntimeError(
                f"RF-DETR torch runner: unsupported variant {variant!r}; "
                f"known: {', '.join(sorted(_VARIANT_CLASSES))}"
            )

        self.device = torch.device(device)
        self.fp16 = bool(fp16)

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        num_classes = int(state["class_embed.weight"].shape[0]) - 1

        model_cls = getattr(rfdetr, cls_name)
        wrapper = model_cls(
            num_classes=num_classes,
            resolution=int(resolution),
            pretrain_weights=str(weights_path),
            device=str(self.device),
        )
        core = wrapper.model.model
        if core is None:
            raise RuntimeError(
                "rfdetr model is unavailable after load (inplace-optimized checkpoint)"
            )
        self._wrapper = wrapper
        self._core = core.to(self.device).eval()

        if isinstance(input_shapes, dict):
            input_shapes = list(input_shapes.values())
        batch_size = int(input_shapes[0][0])

        self.input_names = ["input"]
        self.input_dtypes: dict[str, torch.dtype] = {"input": torch.float32}
        self.output_names = list(_ONNX_OUTPUT_NAMES)
        self.outputs: dict[str, TorchTensorInfo] = {
            "dets": TorchTensorInfo((batch_size, -1, 4), torch.float32),
            "labels": TorchTensorInfo((batch_size, -1, num_classes + 1), torch.float32),
            "masks": TorchTensorInfo((batch_size, -1, -1, -1), torch.float32),
        }

        logger.info(
            "RF-DETR torch model loaded: %s (%s) on %s (num_classes=%d, fp16=%s)",
            weights_path,
            cls_name,
            device_name(self.device),
            num_classes,
            self.fp16,
        )

    def close(self) -> None:
        self._core = None
        self._wrapper = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._core is None:
            raise RuntimeError("RF-DETR torch runner is closed")
        x = inputs["input"].to(device=self.device, dtype=torch.float32)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.fp16
        ):
            out = self._core(x)
        return {
            "dets": out["pred_boxes"].float(),
            "labels": out["pred_logits"].float(),
            "masks": out["pred_masks"].float(),
        }

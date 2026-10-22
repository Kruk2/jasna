from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from jasna.accelerator import device_name
from jasna.mosaic.rfdetr_fused_head import fuse_segmentation_head
from jasna.mosaic.rfdetr_sync_free import export_without_host_syncs

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


def load_first_batch_libraries(device: torch.device, dtype: torch.dtype) -> None:
    """Load MIOpen and the ``x @ wᵀ`` GEMM library of ``dtype``, which a process's first conv and
    linear otherwise load on the first detection batch (1.2 s + 0.5 s on an R9700); a thread
    started before the model is built overlaps them with the import and build."""
    square = torch.zeros(8, 8, device=device, dtype=dtype)
    F.conv2d(square[None, None], square[None, None, :3, :3])
    F.linear(square, square)


class RfDetrTorchRunner:
    """AMD/ROCm RF-DETR path: run the trained checkpoint through the rfdetr torch
    model instead of ONNX + MIGraphX. Mirrors the runner contract consumed by
    ``RfDetrMosaicDetectionModel`` (``input_names``/``input_dtypes``/``outputs``/
    ``output_names``/``infer``/``close``) and runs the export forward the ONNX
    graph is traced from, so it emits the same ``dets``/``labels``/``masks``."""

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
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self._dtype = torch.float16 if self.fp16 else torch.float32
        threading.Thread(
            target=load_first_batch_libraries, args=(self.device, self._dtype), name="RfDetrLibraries", daemon=False
        ).start()
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

        state = torch.load(weights_path, map_location="cpu", weights_only=True)["model"]
        num_classes = int(state["class_embed.weight"].shape[0]) - 1

        # rfdetr's own checkpoint loader unpickles arbitrary objects, so the model
        # is built empty and gets the weights-only state the way that loader would.
        from rfdetr.config import PretrainWeightsCompatibilityWarning
        from rfdetr.models.weights import interpolate_position_embeddings

        model_cls = getattr(rfdetr, cls_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
            wrapper = model_cls(
                num_classes=num_classes,
                resolution=int(resolution),
                pretrain_weights=None,
                device=str(self.device),
            )
        core = wrapper.model.model
        if core is None:
            raise RuntimeError(
                "rfdetr model is unavailable after load (inplace-optimized checkpoint)"
            )
        interpolate_position_embeddings(state, wrapper.model_config.positional_encoding_size)
        core.load_state_dict(state)
        self._wrapper = wrapper
        self._core = core.to(device=self.device, dtype=self._dtype).eval()
        export_without_host_syncs(self._core)
        fuse_segmentation_head(self._core, int(resolution))

        if isinstance(input_shapes, dict):
            input_shapes = list(input_shapes.values())
        batch_size = int(input_shapes[0][0])

        self.input_names = ["input"]
        self.input_dtypes: dict[str, torch.dtype] = {"input": torch.float32}
        self.output_names = list(_ONNX_OUTPUT_NAMES)
        self.outputs: dict[str, TorchTensorInfo] = {
            "dets": TorchTensorInfo((batch_size, -1, 4), torch.float32),
            "labels": TorchTensorInfo((batch_size, -1, num_classes + 1), torch.float32),
            "masks": TorchTensorInfo((batch_size, -1, -1, -1), self._dtype),
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
        x = inputs["input"].to(device=self.device, dtype=self._dtype)
        with torch.no_grad():
            boxes, logits, masks = self._core(x)
        return {"dets": boxes.float(), "labels": logits.float(), "masks": masks}

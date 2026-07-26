from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from jasna.accelerator import is_amd_device, is_nvidia_device
from jasna.engine_paths import get_onnx_tensorrt_engine_path
from jasna.media.resize_normalize import ResizeNormalizer
from jasna.mosaic.detections import Detections
from jasna.tensor_utils import pad_batch_with_last


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def TrtRunner(*args, **kwargs):
    from jasna.trt.trt_runner import TrtRunner as Runner

    return Runner(*args, **kwargs)


def compile_rfdetr_engine(
    weights_path: Path,
    device: torch.device,
    *,
    batch_size: int,
    resolution: int,
    dynamic_batch: bool,
    fp16: bool,
) -> Path:
    if is_amd_device(device):
        # AMD runs the trained checkpoint through the rfdetr torch model
        # (RfDetrTorchRunner); there is no ahead-of-time engine to build.
        return weights_path
    if not is_nvidia_device(device):
        raise RuntimeError(
            f"RF-DETR is not supported on device backend {device.type!r}"
        )

    from jasna.trt import compile_onnx_to_tensorrt_engine
    return compile_onnx_to_tensorrt_engine(
        weights_path,
        device,
        batch_size=int(batch_size),
        fp16=bool(fp16),
        workspace_gb=20,
        dynamic_batch=dynamic_batch,
    )


class RfDetrMosaicDetectionModel:
    DEFAULT_SCORE_THRESHOLD = 0.25
    DEFAULT_MAX_SELECT = 16

    def __init__(
        self,
        *,
        weights_path: Path,
        batch_size: int,
        device: torch.device,
        resolution: int,
        dynamic_batch: bool,
        torch_variant: str | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_select: int = DEFAULT_MAX_SELECT,
        fp16: bool = True,
    ) -> None:
        self.weights_path = weights_path
        self.batch_size = int(batch_size)
        self.device = device
        self.resolution = int(resolution)
        self.dynamic_batch = bool(dynamic_batch)
        self.score_threshold = float(score_threshold)
        self.max_select = int(max_select)
        self._normalization_cache: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if is_amd_device(self.device):
            if torch_variant is None:
                raise RuntimeError(
                    f"RF-DETR on AMD requires a torch variant for {weights_path.name}"
                )
            from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner

            self.runner = RfDetrTorchRunner(
                self.weights_path,
                input_shapes=[
                    (self.batch_size, 3, self.resolution, self.resolution)
                ],
                device=self.device,
                fp16=bool(fp16),
                resolution=self.resolution,
                variant=torch_variant,
            )
            self.engine_path = self.weights_path
        elif is_nvidia_device(self.device):
            self.engine_path = get_onnx_tensorrt_engine_path(
                self.weights_path, batch_size=self.batch_size, fp16=bool(fp16),
                dynamic_batch=self.dynamic_batch,
            )
            if not self.engine_path.exists():
                raise FileNotFoundError(
                    f"RF-DETR engine not found: {self.engine_path}. "
                    "Run engine compilation first via ensure_engines_compiled()."
                )
            self.runner = TrtRunner(
                self.engine_path,
                input_shapes=[
                    (self.batch_size, 3, self.resolution, self.resolution)
                ],
                device=self.device,
            )
        else:
            raise RuntimeError(
                f"RF-DETR is not supported on device backend {self.device.type!r}"
            )
        self._input_name = self.runner.input_names[0]
        self.input_dtype = self.runner.input_dtypes[self._input_name]
        resizer = ResizeNormalizer(
            device=self.device,
            dtype=self.input_dtype,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            fill=(0.0, 0.0, 0.0),
        )
        self._resizer = resizer if resizer.available else None

        self.boxes_out = next(
            name
            for name in self.runner.output_names
            if self.runner.outputs[name].ndim == 3
            and (
                self.runner.outputs[name].shape[-1] == 4
                or name.lower() == "dets"
                or "box" in name.lower()
            )
        )
        self.masks_out = next(k for k in self.runner.output_names if self.runner.outputs[k].ndim == 4)
        self.logits_out = next(k for k in self.runner.output_names if k not in {self.boxes_out, self.masks_out})
        logger.info("RF-DETR detection model loaded: %s (batch_size=%d)", self.engine_path, self.batch_size)

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None

    def _normalization(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (x.device, x.dtype)
        cached = self._normalization_cache.get(key)
        if cached is None:
            mean = x.new_tensor(_IMAGENET_MEAN)[:, None, None]
            std = x.new_tensor(_IMAGENET_STD)[:, None, None]
            cached = (mean, std)
            self._normalization_cache[key] = cached
        return cached

    def _preprocess(self, frames_uint8_bchw: torch.Tensor) -> torch.Tensor:
        if self._resizer is not None and frames_uint8_bchw.dtype is torch.uint8:
            return self._resizer.run(
                frames_uint8_bchw,
                out_hw=(self.resolution, self.resolution),
                content=(0, 0, self.resolution, self.resolution),
            )
        x = frames_uint8_bchw.to(device=self.device, dtype=self.input_dtype).div_(255.0)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        mean, std = self._normalization(x)
        return (x - mean) / std

    def _infer(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.dynamic_batch:
            return self.runner.infer({self._input_name: x})

        output_parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.runner.output_names
        }
        for chunk in x.split(self.batch_size):
            effective_batch_size = int(chunk.shape[0])
            padded = pad_batch_with_last(chunk, batch_size=self.batch_size)
            outputs = self.runner.infer({self._input_name: padded})
            for name in output_parts:
                output_parts[name].append(
                    outputs[name][:effective_batch_size].clone()
                )
        return {
            name: torch.cat(parts, dim=0)
            for name, parts in output_parts.items()
        }

    @staticmethod
    def _postprocess(
        *,
        pred_boxes: torch.Tensor,  # (B, Q, 4) cxcywh normalized
        pred_logits: torch.Tensor,  # (B, Q, C)
        pred_masks: torch.Tensor,  # (B, Q, Hm, Wm)
        target_hw: tuple[int, int],
        score_threshold: float,
        max_select: int,
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        b, q, c = pred_logits.shape
        prob = pred_logits.sigmoid()
        k = min(max_select, q)
        topk_values, topk_indexes = torch.topk(prob.view(b, -1), k, dim=1)
        topk_boxes = topk_indexes // c

        x_c, y_c, w, h = pred_boxes.unbind(-1)
        boxes = torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)
        boxes = boxes.gather(1, topk_boxes.unsqueeze(-1).expand(b, k, 4))

        th, tw = target_hw
        boxes = boxes * boxes.new_tensor((tw, th, tw, th))

        hm, wm = pred_masks.shape[-2], pred_masks.shape[-1]
        masks = pred_masks.gather(1, topk_boxes[:, :, None, None].expand(b, k, hm, wm)) > 0.0

        valid_mask = topk_values > score_threshold  # (B, K)
        boxes_cpu = boxes.to(device='cpu', dtype=torch.float32).numpy()  # (B, K, 4)
        valid_mask_cpu = valid_mask.cpu().numpy()  # (B, K)
        
        boxes_list: list[np.ndarray] = []
        masks_list: list[torch.Tensor] = []
        for i in range(b):
            valid_i = valid_mask_cpu[i]
            boxes_list.append(boxes_cpu[i][valid_i])  # (N_i, 4) CPU
            masks_list.append(masks[i][valid_mask[i]])  # (N_i, Hm, Wm) GPU
        
        return boxes_list, masks_list

    def scan_scores_masks(
        self, frames_uint8_bchw: torch.Tensor, *, mask_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU-only fast path for whole-video scanning: per-frame best score
        (B,) float32 and merged low-res mask (B, mask_h, mask_w) bool, with no
        host synchronization."""

        x = self._preprocess(frames_uint8_bchw)
        outs = self._infer(x)
        per_query = outs[self.logits_out].sigmoid().amax(dim=-1)  # (B, Q)
        scores = per_query.amax(dim=-1).float()  # (B,)
        pred_masks = outs[self.masks_out]  # (B, Q, Hm, Wm)
        active = (pred_masks > 0.0) & (per_query > self.score_threshold)[:, :, None, None]
        merged = active.any(dim=1, keepdim=True).float()
        merged = F.interpolate(merged, size=mask_hw, mode="area") > 0.0
        return scores, merged[:, 0]

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        x = self._preprocess(frames_uint8_bchw)
        outs = self._infer(x)
        boxes_list, masks_list = self._postprocess(
            pred_boxes=outs[self.boxes_out],
            pred_logits=outs[self.logits_out],
            pred_masks=outs[self.masks_out],
            target_hw=target_hw,
            score_threshold=self.score_threshold,
            max_select=self.max_select,
        )
        return Detections(
            boxes_xyxy=boxes_list,
            masks=masks_list,
        )

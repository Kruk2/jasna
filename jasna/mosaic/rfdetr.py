from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
from torch.nn import functional as F

from jasna.trt import compile_onnx_to_tensorrt_engine
from jasna.trt.trt_runner import TrtRunner
from jasna.mosaic.detections import Detections


class RfDetrMosaicDetectionModel:
    DEFAULT_RESOLUTION = 768
    DEFAULT_SCORE_THRESHOLD = 0.2
    DEFAULT_MAX_SELECT = 16

    def __init__(
        self,
        *,
        onnx_path: Path,
        stream: torch.cuda.Stream,
        batch_size: int,
        device: torch.device,
        resolution: int = DEFAULT_RESOLUTION,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_select: int = DEFAULT_MAX_SELECT,
        fp16: bool = True,
    ) -> None:
        self.onnx_path = onnx_path
        self.stream = stream
        self.batch_size = int(batch_size)
        self.device = device
        self.resolution = int(resolution)
        self.score_threshold = float(score_threshold)
        self.max_select = int(max_select)

        self.engine_path = compile_onnx_to_tensorrt_engine(
            self.onnx_path,
            batch_size=self.batch_size,
            fp16=bool(fp16),
        )
        self.runner = TrtRunner(
            self.engine_path,
            stream=self.stream,
            input_shape=(self.batch_size, 3, self.resolution, self.resolution),
            device=self.device,
        )
        self.input_dtype = self.runner.input_dtype

        self.boxes_out = next(
            k for k in self.runner.output_names if self.runner.outputs[k].ndim == 3 and self.runner.outputs[k].shape[-1] == 4
        )
        self.masks_out = next(k for k in self.runner.output_names if self.runner.outputs[k].ndim == 4)
        self.logits_out = next(k for k in self.runner.output_names if k not in {self.boxes_out, self.masks_out})
        logger.info("RF-DETR detection model loaded: %s (batch_size=%d)", self.engine_path, self.batch_size)

    def _preprocess(self, frames_uint8_bchw: torch.Tensor) -> torch.Tensor:
        x = frames_uint8_bchw.to(device=self.device, dtype=self.input_dtype).div_(255.0)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std

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
        topk_values, topk_indexes = torch.topk(prob.view(b, -1), q, dim=1)
        keep = topk_values > score_threshold
        topk_values = topk_values.masked_fill(~keep, float("-inf"))

        k = min(max_select, q)
        topk_values, sel = torch.topk(topk_values, k, dim=1)
        topk_indexes = topk_indexes.gather(1, sel)
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

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        x = self._preprocess(frames_uint8_bchw)
        outs = self.runner.infer(x)
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


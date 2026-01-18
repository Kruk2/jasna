from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from jasna.trt import compile_onnx_to_tensorrt_engine
from jasna.trt.trt_runner import TrtRunner
from jasna.mosaic.detections import Detections


class RfDetrMosaicDetectionModel:
    DEFAULT_RESOLUTION = 768
    DEFAULT_SCORE_THRESHOLD = 0.3
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
    ) -> None:
        self.onnx_path = onnx_path
        self.stream = stream
        self.batch_size = int(batch_size)
        self.device = device
        self.resolution = int(resolution)
        self.score_threshold = float(score_threshold)
        self.max_select = int(max_select)

        self.engine_path = compile_onnx_to_tensorrt_engine(self.onnx_path, fp16=True)
        self.runner = TrtRunner(
            self.engine_path,
            stream=self.stream,
            input_shape=(self.batch_size, 3, self.resolution, self.resolution),
            device=self.device,
        )

        self.boxes_out = next(
            k for k in self.runner.output_names if self.runner.outputs[k].ndim == 3 and self.runner.outputs[k].shape[-1] == 4
        )
        self.masks_out = next(k for k in self.runner.output_names if self.runner.outputs[k].ndim == 4)
        self.logits_out = next(k for k in self.runner.output_names if k not in {self.boxes_out, self.masks_out})

    def _preprocess(self, frames_uint8_bchw: torch.Tensor) -> torch.Tensor:
        x = frames_uint8_bchw.to(device=self.device, dtype=torch.float16).div_(255.0)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std

    @staticmethod
    def _postprocess_same_hw(
        *,
        pred_boxes: torch.Tensor,  # (B, Q, 4) cxcywh normalized
        pred_logits: torch.Tensor,  # (B, Q, C)
        pred_masks: torch.Tensor,  # (B, Q, Hm, Wm)
        target_hw: tuple[int, int],
        score_threshold: float,
        max_select: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, q, c = pred_logits.shape
        prob = pred_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(b, -1), int(q), dim=1)
        keep = topk_values > float(score_threshold)
        topk_values = topk_values.masked_fill(~keep, float("-inf"))

        k = min(int(max_select), int(q))
        topk_values, sel = torch.topk(topk_values, k, dim=1)
        topk_indexes = topk_indexes.gather(1, sel)
        topk_boxes = topk_indexes // c
        labels = topk_indexes % c

        x_c, y_c, w, h = pred_boxes.unbind(-1)
        boxes = torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)
        boxes = boxes.gather(1, topk_boxes.unsqueeze(-1).expand(b, k, 4))

        th, tw = target_hw
        scale = boxes.new_tensor((tw, th, tw, th))
        boxes = boxes * scale

        hm, wm = pred_masks.shape[-2], pred_masks.shape[-1]
        masks = pred_masks.gather(1, topk_boxes[:, :, None, None].expand(b, k, hm, wm))
        masks = F.interpolate(masks.reshape(b * k, 1, hm, wm), size=(th, tw), mode="bilinear", align_corners=False)
        masks = masks.reshape(b, k, th, tw) > 0.0

        return topk_values, labels, boxes, masks

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        x = self._preprocess(frames_uint8_bchw)
        outs = self.runner.infer(x)
        scores, labels, boxes, masks = self._postprocess_same_hw(
            pred_boxes=outs[self.boxes_out],
            pred_logits=outs[self.logits_out],
            pred_masks=outs[self.masks_out],
            target_hw=target_hw,
            score_threshold=self.score_threshold,
            max_select=self.max_select,
        )
        return Detections(scores=scores, labels=labels, boxes_xyxy=boxes, masks=masks)


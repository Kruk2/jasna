from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import nms, ops

from jasna.mosaic.detections import Detections
from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine
from jasna.trt.trt_runner import TrtRunner

logger = logging.getLogger(__name__)


_YOLO_LETTERBOX_PAD_VALUE = 114.0 / 255.0
_MASK_MAX_SIDE = 256


def _mask_hw_for_frame(target_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = (int(target_hw[0]), int(target_hw[1]))
    m = max(h, w)
    scale = _MASK_MAX_SIDE / m
    mh = max(1, int(round(h * scale)))
    mw = max(1, int(round(w * scale)))
    return mh, mw


def _letterbox_normalized_bchw(
    x: torch.Tensor,
    *,
    new_shape: tuple[int, int],
    stride: int,
) -> tuple[torch.Tensor, tuple[tuple[float, float], tuple[int, int]]]:
    # Expect BCHW
    del stride
    _, _, h, w = x.shape
    new_h, new_w = (int(new_shape[0]), int(new_shape[1]))

    gain = min(new_h / h, new_w / w)
    unpad_w = int(round(w * gain))
    unpad_h = int(round(h * gain))

    dw = new_w - unpad_w
    dh = new_h - unpad_h

    if (unpad_h, unpad_w) != (h, w):
        x = F.interpolate(x, size=(unpad_h, unpad_w), mode="bilinear", align_corners=False)

    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top
    if dw or dh:
        x = F.pad(x, (left, right, top, bottom), value=_YOLO_LETTERBOX_PAD_VALUE)

    ratio_pad = ((float(gain), float(gain)), (int(left), int(top)))
    return x, ratio_pad


class YoloMosaicDetectionModel:
    DEFAULT_SCORE_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.7
    DEFAULT_MAX_DET = 32
    DEFAULT_IMGSZ = 640

    def __init__(
        self,
        *,
        model_path: Path,
        stream: torch.cuda.Stream,
        batch_size: int,
        device: torch.device,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_det: int = DEFAULT_MAX_DET,
        fp16: bool = True,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        self.stream = stream
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.device = device
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if self.max_det <= 0:
            raise ValueError(f"max_det must be > 0, got {max_det}")

        self.fp16 = bool(fp16) and (self.device.type == "cuda")
        self.imgsz = int(imgsz)
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {imgsz}")
        self.stride = 32
        self.end2end = False
        self.runner: TrtRunner | None = None
        self.input_dtype = torch.float16 if self.fp16 else torch.float32

        runtime_path = self.model_path
        if self.device.type == "cuda":
            runtime_path = compile_yolo_to_tensorrt_engine(
                self.model_path,
                batch=self.batch_size,
                fp16=self.fp16,
                imgsz=self.imgsz,
            )

        if runtime_path.suffix.lower() == ".engine":
            self.runner = TrtRunner(
                runtime_path,
                stream=self.stream,
                input_shape=(self.batch_size, 3, self.imgsz, self.imgsz),
                device=self.device,
            )
            self.input_dtype = self.runner.input_dtype
        else:
            from ultralytics.nn.autobackend import AutoBackend

            self.model = AutoBackend(
                model=str(runtime_path),
                device=self.device,
                fp16=bool(self.fp16),
                fuse=True,
                verbose=False,
            )
            self.fp16 = bool(getattr(self.model, "fp16", self.fp16))
            self.names = getattr(self.model, "names", {})
            self.end2end = bool(getattr(self.model, "end2end", False))
            self.input_dtype = torch.float16 if self.fp16 else torch.float32
            stride_attr = getattr(self.model, "stride", None)
            if isinstance(stride_attr, torch.Tensor):
                self.stride = int(stride_attr.max().item())
            elif isinstance(stride_attr, (tuple, list)):
                self.stride = int(max(int(x) for x in stride_attr))
            elif stride_attr is not None:
                self.stride = int(stride_attr)
            self.model.eval()
        self._empty_masks_cache: dict[tuple[int, int], torch.Tensor] = {}
        logger.info("YOLO detection model loaded: %s (batch_size=%d)", runtime_path, self.batch_size)

    def _get_empty_masks(self, mask_h: int, mask_w: int) -> torch.Tensor:
        key = (mask_h, mask_w)
        t = self._empty_masks_cache.get(key)
        if t is None:
            t = torch.zeros((0, mask_h, mask_w), dtype=torch.bool, device=self.device)
            self._empty_masks_cache[key] = t
        return t

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        x = frames_uint8_bchw.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        x /= 255.0
        x, ratio_pad = _letterbox_normalized_bchw(
            x,
            new_shape=(self.imgsz, self.imgsz),
            stride=self.stride,
        )

        with torch.inference_mode():
            if self.runner is not None:
                outs = self.runner.infer(x)
                pred = next(t for t in outs.values() if t.ndim == 3)
                proto = next(t for t in outs.values() if t.ndim == 4)
                raw = (pred, proto)
            else:
                raw = self.model(x)

        if not isinstance(raw, (tuple, list)) or len(raw) < 2:
            raise RuntimeError(f"Unexpected YOLO output type/shape: {type(raw)}")

        pred_raw = raw[0] if not isinstance(raw[0], tuple) else raw[0][0]
        protos = raw[1]
        if protos.ndim == 4 and protos.shape[1] not in {8, 16, 32, 64, 128} and protos.shape[-1] in {8, 16, 32, 64, 128}:
            protos = protos.permute(0, 3, 1, 2).contiguous()

        if pred_raw.ndim == 3 and pred_raw.shape[1] > pred_raw.shape[2]:
            pred_raw = pred_raw.permute(0, 2, 1).contiguous()

        mask_dim = int(protos.shape[1]) if protos.ndim == 4 else 0
        nc = int(pred_raw.shape[1]) - 4 - mask_dim
        preds = nms.non_max_suppression(
            pred_raw,
            conf_thres=float(self.score_threshold),
            iou_thres=float(self.iou_threshold),
            classes=None,
            agnostic=False,
            max_det=int(self.max_det),
            nc=max(0, nc),
            end2end=bool(self.end2end),
        )

        out_h, out_w = (int(target_hw[0]), int(target_hw[1]))
        mask_h, mask_w = _mask_hw_for_frame((out_h, out_w))
        empty_masks = self._get_empty_masks(mask_h, mask_w)

        scale_x = float(mask_w) / float(out_w)
        scale_y = float(mask_h) / float(out_h)
        img_shape = x.shape[2:]

        boxes_list: list[np.ndarray] = []
        masks_list: list[torch.Tensor] = []

        for pred, proto in zip(preds, protos):
            if pred.numel() == 0:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                masks_list.append(empty_masks)
                continue

            boxes = ops.scale_boxes(img_shape, pred[:, :4], (out_h, out_w), ratio_pad=ratio_pad)

            boxes_target = boxes.new_empty(boxes.shape)
            boxes_target[:, 0] = boxes[:, 0] * scale_x
            boxes_target[:, 1] = boxes[:, 1] * scale_y
            boxes_target[:, 2] = boxes[:, 2] * scale_x
            boxes_target[:, 3] = boxes[:, 3] * scale_y

            masks_u8 = ops.process_mask_native(proto, pred[:, 6:], boxes_target, (mask_h, mask_w))
            masks = masks_u8.bool()

            keep = masks.flatten(1).any(dim=1)
            if keep.any():
                boxes_list.append(boxes[keep].to(device="cpu", dtype=torch.float32, non_blocking=True).numpy())
                masks_list.append(masks[keep])
            else:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                masks_list.append(empty_masks)

        return Detections(boxes_xyxy=boxes_list, masks=masks_list)


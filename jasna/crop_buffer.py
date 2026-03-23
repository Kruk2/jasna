from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

RESTORATION_SIZE = 256
BORDER_RATIO = 0.06
MIN_BORDER = 20
MAX_EXPANSION_FACTOR = 1.0


def _torch_pad_reflect(image: torch.Tensor, paddings: tuple[int, int, int, int]) -> torch.Tensor:
    paddings_arr = np.array(paddings, dtype=int)
    while np.any(paddings_arr):
        image_limits = np.repeat(np.array(image.shape[::-1][:len(paddings_arr)//2]), 2) - 1
        possible_paddings = np.minimum(paddings_arr, image_limits)
        image = F.pad(image, possible_paddings.tolist(), mode='reflect')
        paddings_arr = paddings_arr - possible_paddings
    return image


def expand_bbox(
    x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int,
) -> tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1

    border = max(MIN_BORDER, int(max(w, h) * BORDER_RATIO)) if BORDER_RATIO > 0.0 else 0
    x1_exp = max(0, x1 - border)
    y1_exp = max(0, y1 - border)
    x2_exp = min(frame_w, x2 + border)
    y2_exp = min(frame_h, y2 + border)

    w = x2_exp - x1_exp
    h = y2_exp - y1_exp
    down_scale_factor = min(RESTORATION_SIZE / w, RESTORATION_SIZE / h) if w > 0 and h > 0 else 1.0
    if down_scale_factor > 1.0:
        down_scale_factor = 1.0

    missing_w = int((RESTORATION_SIZE - (w * down_scale_factor)) / down_scale_factor) if down_scale_factor > 0 else 0
    missing_h = int((RESTORATION_SIZE - (h * down_scale_factor)) / down_scale_factor) if down_scale_factor > 0 else 0

    available_w_l = x1_exp
    available_w_r = frame_w - x2_exp
    available_h_t = y1_exp
    available_h_b = frame_h - y2_exp

    budget_w = int(MAX_EXPANSION_FACTOR * w)
    budget_h = int(MAX_EXPANSION_FACTOR * h)

    expand_w_lr = min(available_w_l, available_w_r, missing_w // 2, budget_w)
    expand_w_l = min(available_w_l - expand_w_lr, missing_w - expand_w_lr * 2, budget_w - expand_w_lr)
    expand_w_r = min(
        available_w_r - expand_w_lr,
        missing_w - expand_w_lr * 2 - expand_w_l,
        budget_w - expand_w_lr - expand_w_l,
    )

    expand_h_tb = min(available_h_t, available_h_b, missing_h // 2, budget_h)
    expand_h_t = min(available_h_t - expand_h_tb, missing_h - expand_h_tb * 2, budget_h - expand_h_tb)
    expand_h_b = min(
        available_h_b - expand_h_tb,
        missing_h - expand_h_tb * 2 - expand_h_t,
        budget_h - expand_h_tb - expand_h_t,
    )

    x1_exp = x1_exp - math.floor(expand_w_lr / 2) - expand_w_l
    x2_exp = x2_exp + math.ceil(expand_w_lr / 2) + expand_w_r
    y1_exp = y1_exp - math.floor(expand_h_tb / 2) - expand_h_t
    y2_exp = y2_exp + math.ceil(expand_h_tb / 2) + expand_h_b

    x1_exp = max(0, min(int(x1_exp), frame_w))
    x2_exp = max(0, min(int(x2_exp), frame_w))
    y1_exp = max(0, min(int(y1_exp), frame_h))
    y2_exp = max(0, min(int(y2_exp), frame_h))

    return x1_exp, y1_exp, x2_exp, y2_exp


def scale_offsets(
    frame_u8: torch.Tensor,
    pad_offset_256: tuple[int, int],
    resize_shape_256: tuple[int, int],
    restoration_size: int = RESTORATION_SIZE,
) -> tuple[tuple[int, int], tuple[int, int]]:
    out_h = int(frame_u8.shape[1])
    out_w = int(frame_u8.shape[2])
    pl, pt = pad_offset_256
    rh, rw = resize_shape_256
    x0 = int(round(pl * out_w / restoration_size))
    x1 = int(round((pl + rw) * out_w / restoration_size))
    y0 = int(round(pt * out_h / restoration_size))
    y1 = int(round((pt + rh) * out_h / restoration_size))
    return (x0, y0), (y1 - y0, x1 - x0)


@dataclass
class RawCrop:
    crop: torch.Tensor          # (C, crop_h, crop_w)
    enlarged_bbox: tuple[int, int, int, int]
    crop_shape: tuple[int, int]  # (crop_h, crop_w)


def extract_crop(
    frame: torch.Tensor,
    bbox: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> RawCrop:
    x1 = int(np.floor(bbox[0]))
    y1 = int(np.floor(bbox[1]))
    x2 = int(np.ceil(bbox[2]))
    y2 = int(np.ceil(bbox[3]))
    x1 = max(0, min(x1, frame_w))
    y1 = max(0, min(y1, frame_h))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))

    x1_exp, y1_exp, x2_exp, y2_exp = expand_bbox(x1, y1, x2, y2, frame_h, frame_w)
    if frame.device.type == "cpu":
        crop = torch.from_numpy(np.array(frame.numpy()[:, y1_exp:y2_exp, x1_exp:x2_exp]))
    else:
        crop = frame[:, y1_exp:y2_exp, x1_exp:x2_exp].clone()

    return RawCrop(
        crop=crop,
        enlarged_bbox=(x1_exp, y1_exp, x2_exp, y2_exp),
        crop_shape=(int(crop.shape[1]), int(crop.shape[2])),
    )


class CropBuffer:
    def __init__(self, track_id: int, start_frame: int):
        self.track_id = track_id
        self.start_frame = start_frame
        self.crops: list[RawCrop] = []

    def add(self, crop: RawCrop) -> None:
        self.crops.append(crop)

    @property
    def frame_count(self) -> int:
        return len(self.crops)

    def split_overlap(self, overlap_len: int, new_track_id: int, new_start_frame: int) -> CropBuffer:
        new_buf = CropBuffer(track_id=new_track_id, start_frame=new_start_frame)
        new_buf.crops = list(self.crops[-overlap_len:])
        return new_buf


def prepare_crops_for_restoration(
    raw_crops: list[RawCrop],
    restoration_size: int = RESTORATION_SIZE,
) -> tuple[list[torch.Tensor], list[tuple[int, int]], list[tuple[int, int]]]:
    crop_shapes = [c.crop_shape for c in raw_crops]
    max_h = max(s[0] for s in crop_shapes)
    max_w = max(s[1] for s in crop_shapes)

    scale_h = restoration_size / max_h
    scale_w = restoration_size / max_w
    if scale_h > 1.0 and scale_w > 1.0:
        scale_h = scale_w = 1.0

    resized_crops: list[torch.Tensor] = []
    resize_shapes: list[tuple[int, int]] = []
    pad_offsets: list[tuple[int, int]] = []

    for raw_crop in raw_crops:
        crop_h, crop_w = raw_crop.crop_shape
        new_h = int(crop_h * scale_h)
        new_w = int(crop_w * scale_w)
        resize_shapes.append((new_h, new_w))

        pad_top = (restoration_size - new_h) // 2
        pad_left = (restoration_size - new_w) // 2
        pad_bottom = restoration_size - new_h - pad_top
        pad_right = restoration_size - new_w - pad_left
        pad_offsets.append((pad_left, pad_top))

        resized = F.interpolate(
            raw_crop.crop.unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        padded = _torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom))
        resized_crops.append(padded.to(raw_crop.crop.dtype).permute(1, 2, 0))

    return resized_crops, pad_offsets, resize_shapes

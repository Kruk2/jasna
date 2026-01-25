from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restored_clip import RestoredClip
from jasna.tracking.clip_tracker import TrackedClip

RESTORATION_SIZE = 256
BORDER_RATIO = 0.06
MIN_BORDER = 20
MAX_EXPANSION_FACTOR = 1.0


def _torch_pad_reflect(image: torch.Tensor, paddings: tuple[int, int, int, int]) -> torch.Tensor:
    paddings_arr = np.array(paddings, dtype=int)
    while np.any(paddings_arr):
        image_limits = np.repeat(np.array(image.shape[::-1][:len(paddings_arr)//2]), 2) - 1
        possible_paddings = np.minimum(paddings_arr, image_limits)
        image = F.pad(image, tuple(possible_paddings), mode='reflect')
        paddings_arr = paddings_arr - possible_paddings
    return image

class RestorationPipeline:
    def __init__(self, restorer: BasicvsrppMosaicRestorer) -> None:
        self.restorer = restorer
        self._resize_dtype = restorer.dtype

    def restore_clip(
        self, clip: TrackedClip, frames: list[torch.Tensor], prefix_restored_frames: list[torch.Tensor] | None = None
    ) -> RestoredClip:
        """
        clip: TrackedClip with bbox/mask info
        frames: list of (C, H, W) original frames, GPU
        prefix_restored_frames: optional list of already-restored (C, 256, 256) frames to prepend for temporal context
        Returns: RestoredClip with restored frames and metadata for blending (excluding prefix frames)
        """
        n_prefix = len(prefix_restored_frames) if prefix_restored_frames else 0
        crops: list[torch.Tensor] = []
        crop_masks: list[torch.Tensor] = []
        enlarged_bboxes: list[tuple[int, int, int, int]] = []
        crop_shapes: list[tuple[int, int]] = []

        for i, frame in enumerate(frames):
            _, frame_h, frame_w = frame.shape
            bbox = clip.bboxes[i]
            x1 = int(np.floor(bbox[0]))
            y1 = int(np.floor(bbox[1]))
            x2 = int(np.ceil(bbox[2]))
            y2 = int(np.ceil(bbox[3]))
            x1 = max(0, min(x1, frame_w))
            y1 = max(0, min(y1, frame_h))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            x1_exp, y1_exp, x2_exp, y2_exp = self._expand_bbox(
                x1, y1, x2, y2, frame_h, frame_w
            )

            enlarged_bboxes.append((x1_exp, y1_exp, x2_exp, y2_exp))
            crop = frame[:, y1_exp:y2_exp, x1_exp:x2_exp]
            crop_shapes.append((crop.shape[1], crop.shape[2]))
            crops.append(crop)

            mask = clip.masks[i].unsqueeze(0).unsqueeze(0).to(dtype=self._resize_dtype)  # (1, 1, Hm, Wm)
            mask_fullres = F.interpolate(mask, size=(frame_h, frame_w), mode="nearest").squeeze(0).squeeze(0)  # (H, W)
            crop_mask = mask_fullres[y1_exp:y2_exp, x1_exp:x2_exp]
            crop_masks.append(crop_mask)

        max_h = max(s[0] for s in crop_shapes)
        max_w = max(s[1] for s in crop_shapes)

        scale_h = RESTORATION_SIZE / max_h
        scale_w = RESTORATION_SIZE / max_w
        if scale_h > 1.0 and scale_w > 1.0:
            scale_h = scale_w = 1.0

        resized_crops: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        resize_shapes: list[tuple[int, int]] = []
        pad_offsets: list[tuple[int, int]] = []

        for crop, crop_mask, (crop_h, crop_w) in zip(crops, crop_masks, crop_shapes):
            new_h = int(crop_h * scale_h)
            new_w = int(crop_w * scale_w)
            resize_shapes.append((new_h, new_w))

            resized = F.interpolate(
                crop.unsqueeze(0).to(dtype=self._resize_dtype),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            pad_top = (RESTORATION_SIZE - new_h) // 2
            pad_left = (RESTORATION_SIZE - new_w) // 2
            pad_bottom = RESTORATION_SIZE - new_h - pad_top
            pad_right = RESTORATION_SIZE - new_w - pad_left
            pad_offsets.append((pad_left, pad_top))

            padded = _torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom))
            resized_crops.append(padded.to(crop.dtype).permute(1, 2, 0))

            resized_mask = F.interpolate(
                crop_mask.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            padded_mask = F.pad(resized_mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            padded_masks.append(padded_mask > 0.0)

        # Prepend prefix frames if provided (already in HWC format, 256x256)
        if prefix_restored_frames:
            prefix_hwc = [f.permute(1, 2, 0) for f in prefix_restored_frames]
            resized_crops = prefix_hwc + resized_crops

        restored = self.restorer.restore(resized_crops)
        
        # Remove prefix frames from output
        if n_prefix > 0:
            restored = restored[n_prefix:]
        
        restored_frames = [r.permute(2, 0, 1) for r in restored]

        _, frame_h, frame_w = frames[0].shape
        return RestoredClip(
            restored_frames=restored_frames,
            masks=clip.masks,
            padded_masks=padded_masks,
            frame_shape=(frame_h, frame_w),
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=pad_offsets,
            resize_shapes=resize_shapes,
        )

    def _expand_bbox(
        self, x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int
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


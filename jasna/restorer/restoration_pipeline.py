from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
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


@dataclass
class RestoredClip:
    restored_frames: list[torch.Tensor]  # each (C, 256, 256), GPU
    masks: list[torch.Tensor]  # each (Hm, Wm) bool, GPU (model resolution)
    frame_shape: tuple[int, int]  # (H, W) original frame shape
    enlarged_bboxes: list[tuple[int, int, int, int]]  # each (x1, y1, x2, y2) after expansion
    crop_shapes: list[tuple[int, int]]  # each (H, W) original crop shape before resize
    pad_offsets: list[tuple[int, int]]  # each (pad_left, pad_top)
    resize_shapes: list[tuple[int, int]]  # each (H, W) shape after resize, before padding


class RestorationPipeline:
    def __init__(self, restorer: BasicvsrppMosaicRestorer) -> None:
        self.restorer = restorer

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
        enlarged_bboxes: list[tuple[int, int, int, int]] = []
        crop_shapes: list[tuple[int, int]] = []

        for i, frame in enumerate(frames):
            _, frame_h, frame_w = frame.shape
            bbox = clip.bboxes[i].astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            x1_exp, y1_exp, x2_exp, y2_exp = self._expand_bbox(
                x1, y1, x2, y2, frame_h, frame_w
            )

            enlarged_bboxes.append((x1_exp, y1_exp, x2_exp, y2_exp))
            crop = frame[:, y1_exp:y2_exp, x1_exp:x2_exp]
            crop_shapes.append((crop.shape[1], crop.shape[2]))
            crops.append(crop)

        max_h = max(s[0] for s in crop_shapes)
        max_w = max(s[1] for s in crop_shapes)

        scale_h = RESTORATION_SIZE / max_h
        scale_w = RESTORATION_SIZE / max_w
        if scale_h > 1.0 and scale_w > 1.0:
            scale_h = scale_w = 1.0

        resized_crops: list[torch.Tensor] = []
        resize_shapes: list[tuple[int, int]] = []
        pad_offsets: list[tuple[int, int]] = []

        for crop, (crop_h, crop_w) in zip(crops, crop_shapes):
            new_h = int(crop_h * scale_h)
            new_w = int(crop_w * scale_w)
            resize_shapes.append((new_h, new_w))

            resized = F.interpolate(
                crop.unsqueeze(0).float(),
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
            frame_shape=(frame_h, frame_w),
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=pad_offsets,
            resize_shapes=resize_shapes,
        )

    def _expand_bbox(
        self, x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int
    ) -> tuple[int, int, int, int]:
        w, h = x2 - x1, y2 - y1

        border = max(MIN_BORDER, int(max(w, h) * BORDER_RATIO))
        x1_exp = max(0, x1 - border)
        y1_exp = max(0, y1 - border)
        x2_exp = min(frame_w, x2 + border)
        y2_exp = min(frame_h, y2 + border)

        curr_w = x2_exp - x1_exp
        curr_h = y2_exp - y1_exp

        if curr_w < RESTORATION_SIZE or curr_h < RESTORATION_SIZE:
            need_w = max(0, RESTORATION_SIZE - curr_w)
            need_h = max(0, RESTORATION_SIZE - curr_h)

            max_expand_w = int(curr_w * MAX_EXPANSION_FACTOR)
            max_expand_h = int(curr_h * MAX_EXPANSION_FACTOR)
            expand_w = min(need_w, max_expand_w)
            expand_h = min(need_h, max_expand_h)

            avail_left = x1_exp
            avail_right = frame_w - x2_exp
            avail_top = y1_exp
            avail_bottom = frame_h - y2_exp

            expand_lr = min(avail_left, avail_right, expand_w // 2)
            expand_left = expand_lr + min(avail_left - expand_lr, expand_w - expand_lr * 2)
            expand_right = expand_lr + min(avail_right - expand_lr, expand_w - expand_lr - expand_left)

            expand_tb = min(avail_top, avail_bottom, expand_h // 2)
            expand_top = expand_tb + min(avail_top - expand_tb, expand_h - expand_tb * 2)
            expand_bottom = expand_tb + min(avail_bottom - expand_tb, expand_h - expand_tb - expand_top)

            x1_exp -= expand_left
            x2_exp += expand_right
            y1_exp -= expand_top
            y2_exp += expand_bottom

        return x1_exp, y1_exp, x2_exp, y2_exp


from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from jasna.tracking.blending import dilate_ellipse
from jasna.tracking.clip_tracker import iou_groups

if TYPE_CHECKING:
    from jasna.session_factory import ImageSession

logger = logging.getLogger(__name__)

SD15_RESTORATION_SIZE = 512
SD15_IOU_THRESHOLD = 0.5
SD15_EXPAND_PIXELS = 20
SD15_MAX_STRENGTH = 0.7


@dataclass
class PreparedImageRestoreGroup:
    mosaic_01: torch.Tensor
    mask_01: torch.Tensor
    crop_bbox: tuple[int, int, int, int]
    content_h: int
    content_w: int
    group_mask: np.ndarray


@dataclass
class PreparedImageRestore:
    img_chw_u8: np.ndarray
    img_rgb: np.ndarray
    groups: list[PreparedImageRestoreGroup]

    @property
    def has_detections(self) -> bool:
        return bool(self.groups)


def clamp_strength(strength: float) -> float:
    """SDEdit strength is clamped to <= 0.7 (DPMSolver crashes near >= 0.75 and
    0.25-0.8 are visually identical; see plan §7)."""
    s = float(strength)
    if s <= 0.0:
        raise ValueError("--sd15-strength must be > 0")
    if s > SD15_MAX_STRENGTH:
        logger.warning("SD 1.5 strength %g is above the maximum; using %g", s, SD15_MAX_STRENGTH)
    return min(s, SD15_MAX_STRENGTH)


def variant_output_paths(base: Path, num_variants: int) -> list[Path]:
    base = Path(base)
    if num_variants <= 1:
        return [base]
    return [base.with_stem(f"{base.stem}_v{i}") for i in range(1, num_variants + 1)]


def _union_bbox(boxes: np.ndarray, indices: list[int]) -> np.ndarray:
    sub = boxes[indices]
    return np.array([sub[:, 0].min(), sub[:, 1].min(), sub[:, 2].max(), sub[:, 3].max()], dtype=np.float32)


def _build_group_mask(masks: torch.Tensor, indices: list[int], h: int, w: int, dilate_px: int) -> np.ndarray:
    union = torch.zeros((h, w), dtype=torch.bool, device=masks.device)
    for idx in indices:
        m = masks[idx].unsqueeze(0).unsqueeze(0).float()
        union |= F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    if dilate_px > 0:
        union = dilate_ellipse(union, dilate_px)
    return union.to(torch.uint8).mul_(255).cpu().numpy()


def prepare_image_restore(
    img_chw_u8: np.ndarray,
    detector,
    *,
    device: torch.device,
    fp16: bool,
) -> PreparedImageRestore:
    from jasna.sd15_crop_utils import compute_crop_bbox, crop_and_resize_np

    _, h, w = img_chw_u8.shape
    dtype = torch.float16 if fp16 else torch.float32

    frame_cpu = torch.from_numpy(img_chw_u8)
    batch = frame_cpu.unsqueeze(0).expand(detector.batch_size, -1, -1, -1).contiguous()
    detections = detector(batch, target_hw=(h, w))
    boxes = detections.boxes_xyxy[0]
    masks = detections.masks[0]
    img_rgb = np.ascontiguousarray(img_chw_u8.transpose(1, 2, 0))  # HWC RGB

    if len(boxes) == 0:
        logger.info("No mosaics detected; writing input unchanged")
        return PreparedImageRestore(img_chw_u8=img_chw_u8.copy(), img_rgb=img_rgb, groups=[])

    groups = []
    for indices in iou_groups(boxes, SD15_IOU_THRESHOLD, inclusive=True):
        group_mask = _build_group_mask(masks, indices, h, w, SD15_EXPAND_PIXELS)
        merged = _union_bbox(boxes, indices)
        crop_bbox = compute_crop_bbox(merged, w, h, SD15_RESTORATION_SIZE)
        crop_rgb, content_h, content_w = crop_and_resize_np(img_rgb, crop_bbox, SD15_RESTORATION_SIZE)
        crop_mask, _, _ = crop_and_resize_np(group_mask, crop_bbox, SD15_RESTORATION_SIZE)
        mosaic_01 = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype) / 255.0
        mask_01 = torch.from_numpy(crop_mask).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype) / 255.0
        groups.append(PreparedImageRestoreGroup(
            mosaic_01=mosaic_01,
            mask_01=mask_01,
            crop_bbox=tuple(int(v) for v in crop_bbox),
            content_h=int(content_h),
            content_w=int(content_w),
            group_mask=group_mask,
        ))
    return PreparedImageRestore(img_chw_u8=img_chw_u8.copy(), img_rgb=img_rgb, groups=groups)


def render_prepared_image(
    prepared: PreparedImageRestore,
    restorer,
    *,
    steps: int,
    strength: float,
    seed: int,
    freeu: dict | None,
) -> np.ndarray:
    from jasna.sd15_crop_utils import paste_back

    if not prepared.groups:
        return prepared.img_chw_u8.copy()

    result = prepared.img_rgb.copy()
    for group in prepared.groups:
        restored = restorer.restore_crop(
            group.mosaic_01, group.mask_01, steps=steps, strength=strength, seed=seed, freeu=freeu,
        )
        restored_rgb = restored.permute(1, 2, 0).cpu().numpy()  # (512, 512, 3) RGB uint8
        paste_back(result, restored_rgb, group.crop_bbox, group.content_h, group.content_w, group.group_mask)
    return np.ascontiguousarray(result.transpose(2, 0, 1))


def mask_overlay_rgb_chw(prepared: PreparedImageRestore, alpha: float = 0.45) -> np.ndarray:
    overlay = prepared.img_rgb.copy().astype(np.float32)
    mask = np.zeros(prepared.img_rgb.shape[:2], dtype=bool)
    for group in prepared.groups:
        mask |= group.group_mask > 0
    if mask.any():
        overlay[mask, 0] = overlay[mask, 0] * (1.0 - alpha) + 255.0 * alpha
        overlay[mask, 1] = overlay[mask, 1] * (1.0 - alpha)
        overlay[mask, 2] = overlay[mask, 2] * (1.0 - alpha)
    return np.ascontiguousarray(overlay.clip(0, 255).round().astype(np.uint8).transpose(2, 0, 1))


def restore_image(
    img_chw_u8: np.ndarray,
    detector,
    restorer,
    *,
    device: torch.device,
    fp16: bool,
    steps: int,
    strength: float,
    seed: int,
    num_variants: int,
    freeu: dict | None,
) -> list[np.ndarray]:
    """Detect mosaics, SDEdit-inpaint each group, composite back.

    Uses the crop geometry the model was trained with
    (``sd15_crop_utils``): square crop (``compute_crop_bbox``) + resize/reflect-pad
    (``crop_and_resize_np``) + hard-mask ``paste_back``.

    Returns one ``(3, H, W)`` uint8 RGB array per variant. Zero detections ->
    copies of the input unchanged."""
    prepared = prepare_image_restore(
        img_chw_u8, detector,
        device=device, fp16=fp16,
    )

    outputs: list[np.ndarray] = []
    for v in range(num_variants):
        outputs.append(render_prepared_image(
            prepared, restorer, steps=steps, strength=strength, seed=seed + v, freeu=freeu,
        ))
    return outputs


@dataclass(frozen=True)
class ImageRestoreOptions:
    """How each still image is restored; ``strength`` is clamped and ``variants`` is at least 1."""

    steps: int
    strength: float
    seed: int
    variants: int
    freeu: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", clamp_strength(self.strength))
        object.__setattr__(self, "variants", max(1, int(self.variants)))


def restore_image_file(
    session: ImageSession, options: ImageRestoreOptions, input_path: Path, output_base: Path
) -> list[Path]:
    """Restore ``input_path`` into ``output_base`` (``name_vN`` files for several variants);
    returns the files written."""
    from jasna.accelerator import device_context
    from jasna.media import image_io
    from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

    image = image_io.read_image_rgb_chw(input_path)
    with device_context(session.device):
        outputs = restore_image(
            image, session.detector, session.restorer,
            device=session.device, fp16=session.fp16,
            steps=options.steps, strength=options.strength, seed=options.seed,
            num_variants=options.variants, freeu=dict(DEFAULT_FREEU) if options.freeu else None,
        )
    paths = variant_output_paths(output_base, options.variants)
    for path, output in zip(paths, outputs):
        image_io.write_image_rgb_chw(path, output)
    return paths

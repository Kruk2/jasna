from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
    return min(s, SD15_MAX_STRENGTH)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def group_boxes_by_iou(boxes: np.ndarray, iou_threshold: float) -> list[list[int]]:
    """Union-find grouping of overlapping boxes (ported from crop_utils)."""
    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i], boxes[j]) >= iou_threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def variant_output_paths(base: Path, num_variants: int) -> list[Path]:
    base = Path(base)
    if num_variants <= 1:
        return [base]
    return [base.with_stem(f"{base.stem}_v{i}") for i in range(1, num_variants + 1)]


def _union_bbox(boxes: np.ndarray, indices: list[int]) -> np.ndarray:
    sub = boxes[indices]
    return np.array([sub[:, 0].min(), sub[:, 1].min(), sub[:, 2].max(), sub[:, 3].max()], dtype=np.float32)


def _build_group_mask(masks: torch.Tensor, indices: list[int], h: int, w: int, dilate_px: int) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.uint8)
    for idx in indices:
        m = masks[idx].unsqueeze(0).unsqueeze(0).float()
        up = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
        acc = np.maximum(acc, (up.cpu().numpy() * 255).astype(np.uint8))
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        acc = cv2.dilate(acc, kernel, iterations=1)
    return acc


def prepare_image_restore(
    img_chw_u8: np.ndarray,
    detector,
    *,
    device: torch.device,
    fp16: bool,
    iou_threshold: float = SD15_IOU_THRESHOLD,
    expand_pixels: int = SD15_EXPAND_PIXELS,
    restoration_size: int = SD15_RESTORATION_SIZE,
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
    for indices in group_boxes_by_iou(boxes, iou_threshold):
        group_mask = _build_group_mask(masks, indices, h, w, expand_pixels)
        merged = _union_bbox(boxes, indices)
        crop_bbox = compute_crop_bbox(merged, w, h, restoration_size)
        crop_rgb, content_h, content_w = crop_and_resize_np(img_rgb, crop_bbox, restoration_size)
        crop_mask, _, _ = crop_and_resize_np(group_mask, crop_bbox, restoration_size)
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
    iou_threshold: float = SD15_IOU_THRESHOLD,
    expand_pixels: int = SD15_EXPAND_PIXELS,
    restoration_size: int = SD15_RESTORATION_SIZE,
) -> list[np.ndarray]:
    """Detect mosaics, SDEdit-inpaint each group, composite back.

    Uses the jav-restoration-v2 crop geometry the model was trained with
    (``sd15_crop_utils``): square crop (``compute_crop_bbox``) + resize/reflect-pad
    (``crop_and_resize_np``) + hard-mask ``paste_back``.

    Returns one ``(3, H, W)`` uint8 RGB array per variant. Zero detections ->
    copies of the input unchanged."""
    prepared = prepare_image_restore(
        img_chw_u8, detector,
        device=device, fp16=fp16,
        iou_threshold=iou_threshold,
        expand_pixels=expand_pixels,
        restoration_size=restoration_size,
    )

    outputs: list[np.ndarray] = []
    for v in range(num_variants):
        outputs.append(render_prepared_image(
            prepared, restorer, steps=steps, strength=strength, seed=seed + v, freeu=freeu,
        ))
    return outputs


def run_image_restoration(args) -> None:
    input_path = Path(args.input)
    output_base = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_out")
    _run_image_jobs(args, [(input_path, output_base)])


def run_image_restoration_folder(
    args,
    input_paths: list[Path],
    output_dir: Path,
    *,
    output_pattern: str | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> None:
    """Restore every image in ``input_paths``, writing into ``output_dir`` (one
    shared model load for the whole batch)."""
    from jasna.media.media_files import folder_output_path

    jobs = [(Path(p), folder_output_path(output_dir, p, output_pattern)) for p in input_paths]
    total = progress_total or len(jobs)

    def progress(i: int, input_path: Path, output_base: Path) -> None:
        print(f"[{progress_offset + i}/{total}] Processing {input_path.name} -> {output_base.name}")

    _run_image_jobs(args, jobs, progress_callback=progress)


def _run_image_jobs(args, jobs: list[tuple[Path, Path]], progress_callback=None) -> None:
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.engine_paths import SD15_DIR
    from jasna.media import image_io
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        require_detection_model_weights,
    )
    from jasna.restorer.sd15_download import ensure_sd15_bundle
    from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU, Sd15InpaintRestorer

    if not jobs:
        return

    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    batch_size = int(args.batch_size)
    score_threshold = float(args.detection_score_threshold)

    if args.license_email and args.license_key:
        from jasna.protection import license_store
        license_store.set_license(args.license_email, args.license_key)

    ensure_sd15_bundle(SD15_DIR)

    detection_model_name = coerce_detection_model_name(str(args.detection_model))
    has_explicit_path = bool(str(args.detection_model_path).strip())
    detection_model_path = (
        Path(str(args.detection_model_path))
        if has_explicit_path
        else require_detection_model_weights(detection_model_name)
    )
    if not detection_model_path.exists():
        raise FileNotFoundError(str(detection_model_path))

    ensure_engines_compiled(EngineCompilationRequest(
        device=str(device),
        fp16=fp16,
        detection=True,
        detection_model_name=detection_model_name,
        detection_model_path=str(detection_model_path),
        detection_batch_size=batch_size,
    ))

    num_variants = max(1, int(args.sd15_variants))
    freeu = dict(DEFAULT_FREEU) if bool(args.sd15_freeu) else None
    strength = clamp_strength(args.sd15_strength)

    detector = build_detection_model(
        detection_model_name,
        detection_model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )
    restorer = Sd15InpaintRestorer(SD15_DIR, device, fp16)
    try:
        for i, (input_path, output_base) in enumerate(jobs, start=1):
            logger.info("[%d/%d] Processing %s", i, len(jobs), input_path.name)
            if progress_callback is not None:
                progress_callback(i, input_path, output_base)
            img = image_io.read_image_rgb_chw(input_path)
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                outputs = restore_image(
                    img, detector, restorer,
                    device=device, fp16=fp16,
                    steps=int(args.sd15_steps), strength=strength, seed=int(args.sd15_seed),
                    num_variants=num_variants, freeu=freeu,
                )
            for path, out in zip(variant_output_paths(output_base, num_variants), outputs):
                image_io.write_image_rgb_chw(path, out)
                logger.info("Wrote %s", path)
    finally:
        detector.close()
        restorer.close()

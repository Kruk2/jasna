"""
Debug script: visualize detection mask, expanded bbox, and blend mask contours.

Decodes input video (NVDEC), runs RF-DETR detection + ClipTracker,
draws 3 annotation layers per frame, encodes output (NVENC).

  Green contour  — detection mask outline (original low-res mask upscaled to bbox region)
  Yellow rect    — expanded bbox from expand_bbox()
  Red contour    — blend mask outer edge (weight > 0)

Usage:
    python debug_blend_contours.py <input_video> [output_video]

If output_video is omitted, writes to <input_stem>_blend_debug.mkv next to input.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from jasna.crop_buffer import expand_bbox
from jasna.media import VideoMetadata, get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    detection_model_weights_path,
    is_rfdetr_model,
    is_yolo_model,
)
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
from jasna.mosaic.yolo import YoloMosaicDetectionModel
from jasna.tensor_utils import pad_batch_with_last
from jasna.tracking.blending import create_blend_mask
from jasna.tracking.clip_tracker import ClipTracker

BATCH_SIZE = 4
MAX_CLIP_SIZE = 180
TEMPORAL_OVERLAP = 15
DETECTION_SCORE_THRESHOLD = 0.25

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
THICKNESS = 2


def _draw_annotations(
    frame_hwc: np.ndarray,
    boxes_xyxy: np.ndarray,
    masks: torch.Tensor,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Draw detection mask contour, expanded bbox, and blend mask contour on frame."""
    n = boxes_xyxy.shape[0]
    if n == 0:
        return frame_hwc

    hm, wm = masks.shape[1], masks.shape[2]

    for i in range(n):
        det_x1, det_y1, det_x2, det_y2 = (
            int(np.floor(boxes_xyxy[i, 0])),
            int(np.floor(boxes_xyxy[i, 1])),
            int(np.ceil(boxes_xyxy[i, 2])),
            int(np.ceil(boxes_xyxy[i, 3])),
        )
        det_x1 = max(0, min(det_x1, frame_w))
        det_y1 = max(0, min(det_y1, frame_h))
        det_x2 = max(0, min(det_x2, frame_w))
        det_y2 = max(0, min(det_y2, frame_h))

        det_w = det_x2 - det_x1
        det_h = det_y2 - det_y1
        if det_w <= 0 or det_h <= 0:
            continue

        mask_lr = masks[i].cpu()

        # --- 1) Green: detection mask contour in original bbox region ---
        y_idx = (torch.arange(det_y1, det_y2) * hm) // frame_h
        x_idx = (torch.arange(det_x1, det_x2) * wm) // frame_w
        det_crop_mask = mask_lr.float().index_select(0, y_idx).index_select(1, x_idx)
        det_mask_np = (det_crop_mask.numpy() > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(det_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c[:, :, 0] += det_x1
            c[:, :, 1] += det_y1
        cv2.drawContours(frame_hwc, contours, -1, GREEN, THICKNESS)

        # --- 2) Cyan: detection bbox ---
        cv2.rectangle(frame_hwc, (det_x1, det_y1), (det_x2, det_y2), CYAN, THICKNESS)

        # --- 3) Yellow: expanded bbox ---
        ex1, ey1, ex2, ey2 = expand_bbox(det_x1, det_y1, det_x2, det_y2, frame_h, frame_w)
        cv2.rectangle(frame_hwc, (ex1, ey1), (ex2, ey2), YELLOW, THICKNESS)

        exp_w = ex2 - ex1
        exp_h = ey2 - ey1
        if exp_w <= 0 or exp_h <= 0:
            continue

        # --- 4) Red: blend mask outer contour (weight > 0) ---
        ey_idx = (torch.arange(ey1, ey2) * hm) // frame_h
        ex_idx = (torch.arange(ex1, ex2) * wm) // frame_w
        crop_mask = mask_lr.float().index_select(0, ey_idx).index_select(1, ex_idx)
        blend_mask = create_blend_mask(crop_mask, frame_h)
        blend_np = (blend_mask.numpy() > 0).astype(np.uint8) * 255
        contours_blend, _ = cv2.findContours(blend_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_blend:
            c[:, :, 0] += ex1
            c[:, :, 1] += ey1
        cv2.drawContours(frame_hwc, contours_blend, -1, RED, THICKNESS)

    return frame_hwc


def _build_detector(batch_size: int, device: torch.device):
    det_name = DEFAULT_DETECTION_MODEL_NAME
    det_path = detection_model_weights_path(det_name)
    if is_rfdetr_model(det_name):
        return RfDetrMosaicDetectionModel(
            onnx_path=det_path,
            batch_size=batch_size,
            device=device,
            score_threshold=DETECTION_SCORE_THRESHOLD,
        )
    elif is_yolo_model(det_name):
        return YoloMosaicDetectionModel(
            model_path=det_path,
            batch_size=batch_size,
            device=device,
            score_threshold=DETECTION_SCORE_THRESHOLD,
        )
    raise RuntimeError(f"Unknown detection model: {det_name}")


def main():
    input_path = Path(r"K:\lada\new_tests2\KV-109-short.mp4")
    output_path = input_path.with_stem(input_path.stem + "_blend_debug").with_suffix(".mp4")

    device = torch.device("cuda:0")
    metadata: VideoMetadata = get_video_meta_data(str(input_path))
    frame_h, frame_w = metadata.video_height, metadata.video_width
    target_hw = (frame_h, frame_w)

    print(f"Input : {input_path}  ({frame_w}x{frame_h}, {metadata.num_frames} frames)")
    print(f"Output: {output_path}")

    detector = _build_detector(BATCH_SIZE, device)
    tracker = ClipTracker(max_clip_size=MAX_CLIP_SIZE, temporal_overlap=TEMPORAL_OVERLAP)

    frame_idx = 0
    with (
        NvidiaVideoReader(str(input_path), BATCH_SIZE, device, metadata) as reader,
        NvidiaVideoEncoder(
            str(output_path), device, metadata,
            codec="hevc", encoder_settings={}, stream_mode=True,
        ) as encoder,
    ):
        for batch_tensor, pts_list in reader.frames():
            effective_bs = len(pts_list)
            if effective_bs == 0:
                break

            frames_eff = batch_tensor[:effective_bs]
            frames_in = pad_batch_with_last(frames_eff, batch_size=BATCH_SIZE)

            detections = detector(frames_in, target_hw=target_hw)

            for i in range(effective_bs):
                current_idx = frame_idx + i
                pts = pts_list[i]

                boxes = detections.boxes_xyxy[i]
                masks = detections.masks[i]

                tracker.update(current_idx, boxes, masks)

                # Collect all active track boxes/masks for this frame
                all_boxes = []
                all_masks = []
                for clip in tracker.active_clips.values():
                    if clip.start_frame <= current_idx <= clip.end_frame:
                        local = current_idx - clip.start_frame
                        all_boxes.append(clip.bboxes[local])
                        all_masks.append(clip.masks[local])

                frame_gpu = frames_eff[i]  # (3, H, W) uint8 GPU

                if all_boxes:
                    frame_hwc = frame_gpu.permute(1, 2, 0).cpu().numpy().copy()
                    boxes_np = np.stack(all_boxes)
                    masks_t = torch.stack(all_masks)
                    frame_hwc = _draw_annotations(frame_hwc, boxes_np, masks_t, frame_h, frame_w)
                    frame_gpu = torch.from_numpy(frame_hwc).permute(2, 0, 1).to(device)

                encoder.encode(frame_gpu, pts)

                if current_idx % 100 == 0:
                    print(f"  frame {current_idx}/{metadata.num_frames}")

            frame_idx += effective_bs

    detector.close()
    print(f"Done. {frame_idx} frames written to {output_path}")


if __name__ == "__main__":
    main()

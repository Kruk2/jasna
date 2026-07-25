"""Render one mosaic track under raw / flat(gnomonic) / fisheye projections.

Diagnostic for VR180 restoration: picks a single LEFT-eye track near the middle
of a clip and, for a window of consecutive frames, writes an mp4 of a labelled
montage of the same mosaic ROI cropped three ways so you can see how the mosaic
grid aligns:

  * raw     - axis-aligned crop straight from the (equirectangular) eye
  * flat    - per-region gnomonic dewarp fed to the 2D restoration model
  * fisheye - the same ROI seen through the legacy whole-eye 180 fisheye remap

Usage (needs a GPU, decode + detection run on cuda:0):

  python scripts/vr_render_tracks.py --input CLIP.mp4 --out DIR [--frames 180]

Nothing is restored; this only visualises the projections used before
restoration. NEAREST resampling is used for the montage so mosaic block edges
stay crisp.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="VR180 SBS clip")
    ap.add_argument("--out", required=True, help="output directory for montage PNGs")
    ap.add_argument("--detection-model", default="rfdetr-vr-v1")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=180,
                    help="consecutive frames from the chosen track rendered to mp4 (default: %(default)s)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="decode stride for the detect/track pass (default: %(default)s)")
    ap.add_argument("--no-fisheye", action="store_true", help="skip the fisheye panel")
    ap.add_argument("--panel-height", type=int, default=512)
    return ap.parse_args()


def _pixel_centers(length: int, dev, dtype):
    import torch
    return (torch.arange(length, device=dev, dtype=dtype) + 0.5) / length


def _build_fisheye_forward_grid(eye_width: int, height: int, device, fov_degrees: float = 180.0):
    """Grid mapping each fisheye-output pixel to its equirect source coord
    (legacy FisheyeProjector._build_forward_grid, whole eye)."""
    import torch
    half_fov = math.radians(fov_degrees) * 0.5
    oy, ox = torch.meshgrid(
        _pixel_centers(height, device, torch.float64),
        _pixel_centers(eye_width, device, torch.float64),
        indexing="ij",
    )
    fx = ox * 2.0 - 1.0
    fy = oy * 2.0 - 1.0
    radius = torch.sqrt(fx * fx + fy * fy)
    theta = radius * half_fov
    phi = torch.atan2(fy, fx)
    dir_x = torch.sin(theta) * torch.cos(phi)
    dir_y = torch.sin(theta) * torch.sin(phi)
    dir_z = torch.cos(theta)
    longitude = torch.atan2(dir_x, dir_z)
    latitude = torch.asin(dir_y.clamp(-1.0, 1.0))
    grid_x = (longitude / math.pi + 0.5) * 2.0 - 1.0
    grid_y = (latitude / math.pi + 0.5) * 2.0 - 1.0
    outside = radius > 1.0
    grid_x = torch.where(outside, torch.full_like(grid_x, 2.0), grid_x)
    grid_y = torch.where(outside, torch.full_like(grid_y, 2.0), grid_y)
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()


def _equirect_to_fisheye_px(px, py, eye_width, height, fov_degrees=180.0):
    """Where an equirect eye pixel lands in the fisheye output (pixel coords)."""
    half_fov = math.radians(fov_degrees) * 0.5
    lon = ((px + 0.5) / eye_width - 0.5) * math.pi
    lat = ((py + 0.5) / height - 0.5) * math.pi
    dir_x = math.cos(lat) * math.sin(lon)
    dir_y = math.sin(lat)
    dir_z = math.cos(lat) * math.cos(lon)
    theta = math.acos(max(-1.0, min(1.0, dir_z)))
    phi = math.atan2(dir_y, dir_x)
    radius = theta / half_fov
    fx = radius * math.cos(phi)
    fy = radius * math.sin(phi)
    return (
        (fx + 1.0) * 0.5 * (eye_width - 1),
        (fy + 1.0) * 0.5 * (height - 1),
    )


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    from jasna.crop_buffer import compute_enlarged_bbox, extract_crop
    from jasna.media import get_video_meta_data
    from jasna.media.image_io import write_image_rgb_chw
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        detection_model_weights_path,
        precompile_detection_engine,
        recommended_score_threshold,
    )
    from jasna.pipeline_processing import _eye_bounds
    from jasna.tracking.clip_tracker import ClipTracker
    from jasna.vr180 import SbsDetectionAdapter, resolve_vr_mode
    from jasna.vr_projection import GnomonicProjector

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    meta = get_video_meta_data(str(inp))
    height, width = int(meta.video_height), int(meta.video_width)
    vr = resolve_vr_mode("auto", meta, inp)
    is_sbs = vr.is_sbs
    eye_width = width // 2 if is_sbs else None
    print(f"[vr] {inp.name}: {width}x{height} resolved={vr.resolved} sbs={is_sbs}", flush=True)

    name = coerce_detection_model_name(args.detection_model)
    wpath = detection_model_weights_path(name)
    precompile_detection_engine(name, wpath, batch_size=args.batch, device=device, fp16=True)
    base_model = build_detection_model(
        name, wpath, batch_size=args.batch, device=device,
        score_threshold=recommended_score_threshold(name), fp16=True,
    )
    detector = SbsDetectionAdapter(base_model) if is_sbs else base_model

    # --- pass 1: detect + track (frames discarded, only bboxes kept) ---
    tracker = ClipTracker(
        max_clip_size=10_000_000, temporal_overlap=0,
        iou_threshold=0.3, max_detection_gap=6,
    )
    clips = []
    frame_idx = 0
    with NvidiaVideoReader(str(inp), args.batch, device, meta, frame_stride=args.frame_stride) as reader, \
            torch.inference_mode():
        for batch, pts in reader.frames():
            det = detector(batch, target_hw=(height, width))
            for i in range(len(pts)):
                ended, _active = tracker.update(frame_idx, det.boxes_xyxy[i], det.masks[i])
                clips.extend(e.clip for e in ended)
                frame_idx += 1
    clips.extend(e.clip for e in tracker.flush())
    total = frame_idx
    if not clips:
        print("[!] no tracks detected", file=sys.stderr)
        return 1

    mid = total // 2

    def _is_left(c):
        cx = (float(c.bboxes[0][0]) + float(c.bboxes[0][2])) * 0.5
        return eye_width is None or cx < eye_width

    pool = [c for c in clips if _is_left(c)] or clips
    chosen = max(pool, key=lambda c: (c.start_frame <= mid <= c.end_frame, c.frame_count))
    print(f"[track] id={chosen.track_id} eye={'left' if _is_left(chosen) else 'right'} "
          f"frames {chosen.start_frame}..{chosen.end_frame} count={chosen.frame_count} "
          f"(video middle={mid})", flush=True)

    # A window of consecutive frames (for a smooth mp4), centred on the video
    # middle when the track covers it, else on the track's own middle.
    frames_all = list(chosen.frame_indices())
    n = min(args.frames, len(frames_all))
    center = mid if chosen.start_frame <= mid <= chosen.end_frame else (chosen.start_frame + chosen.end_frame) // 2
    ci = center - chosen.start_frame
    lo = max(0, min(ci - n // 2, len(frames_all) - n))
    wanted = frames_all[lo:lo + n]
    wanted_set = set(wanted)
    order = {fi: k for k, fi in enumerate(wanted)}
    bbox_of = {fi: chosen.bboxes[fi - chosen.start_frame] for fi in wanted}

    gnomonic = GnomonicProjector(eye_width=eye_width, height=height, device=device) if is_sbs else None
    fisheye_grid = (
        _build_fisheye_forward_grid(eye_width, height, device).to(device)
        if (is_sbs and not args.no_fisheye) else None
    )

    def _to_hwc_rgb(chw_u8):
        return chw_u8.permute(1, 2, 0).contiguous().cpu().numpy()

    def _fisheye_region(frame, ebox, x_bounds):
        offset = x_bounds[0] if x_bounds is not None else 0
        x1, y1, x2, y2 = ebox
        eye = frame[:, :, offset:offset + eye_width].unsqueeze(0).float()
        fisheye_eye = F.grid_sample(
            eye, fisheye_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )[0]
        xs, ys = [], []
        for px in (x1 - offset, x2 - offset):
            for py in (y1, y2):
                fx, fy = _equirect_to_fisheye_px(px, py, eye_width, height)
                xs.append(fx)
                ys.append(fy)
        fx1 = max(0, int(math.floor(min(xs))))
        fx2 = min(eye_width, int(math.ceil(max(xs))))
        fy1 = max(0, int(math.floor(min(ys))))
        fy2 = min(height, int(math.ceil(max(ys))))
        if fx2 <= fx1 or fy2 <= fy1:
            return fisheye_eye.round().clamp(0, 255).to(torch.uint8)
        return fisheye_eye[:, fy1:fy2, fx1:fx2].round().clamp(0, 255).to(torch.uint8)

    def _panel(label, chw_u8, size):
        img = _to_hwc_rgb(chw_u8)
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((size, size, 3), 30, np.uint8)
        oy, ox = (size - nh) // 2, (size - nw) // 2
        canvas[oy:oy + nh, ox:ox + nw] = r
        bar = np.zeros((28, size, 3), np.uint8)
        cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    frames_dir = out / f"frames_track{chosen.track_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    size = args.panel_height - (args.panel_height % 2)  # even dims for yuv420p
    saved = 0
    fidx = 0
    with NvidiaVideoReader(str(inp), args.batch, device, meta, frame_stride=args.frame_stride) as reader, \
            torch.inference_mode():
        for batch, pts in reader.frames():
            for i in range(len(pts)):
                if fidx in wanted_set:
                    frame = batch[i]
                    bbox = bbox_of[fidx]
                    x_bounds = _eye_bounds(bbox, eye_width, width) if is_sbs else None
                    ebox = compute_enlarged_bbox(bbox, height, width, x_bounds)

                    views = [("raw", extract_crop(frame, bbox, height, width, x_bounds=x_bounds).crop)]
                    if gnomonic is not None:
                        views.append(("flat", gnomonic.extract_region_crop(
                            frame, bbox, height, width, x_bounds=x_bounds).crop))
                    if fisheye_grid is not None:
                        views.append(("fisheye", _fisheye_region(frame, ebox, x_bounds)))

                    tiles = [_panel(lbl, t, size) for lbl, t in views]
                    sep = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
                    row = []
                    for k, t in enumerate(tiles):
                        if k:
                            row.append(sep)
                        row.append(t)
                    montage = np.hstack(row)
                    path = frames_dir / f"{order[fidx]:05d}.png"
                    write_image_rgb_chw(str(path), montage.transpose(2, 0, 1))
                    saved += 1
                fidx += 1
                if saved == len(wanted_set):
                    break
            if saved == len(wanted_set):
                break

    fps = float(meta.video_fps) or 30.0
    mp4 = out / f"{inp.stem}_track{chosen.track_id}_lefteye.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps:.4f}",
            "-i", str(frames_dir / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(mp4),
        ],
        check=True,
    )
    print(f"[done] track {chosen.track_id}: {saved} frames -> {mp4}", flush=True)
    if hasattr(detector, "close"):
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

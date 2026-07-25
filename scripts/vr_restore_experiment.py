"""Restore a mosaic track under many projections to find the best input space.

For one LEFT-eye track (near the video middle) of each clip, builds several
projections of the mosaic ROI, restores each through the real pipeline
(BasicVSR++ primary + unet-4x secondary), and writes an mp4 whose every frame is
a grid: one column per projection, top tile = the 256 crop fed to the restorer,
bottom tile = the restored result. Goal: see which projection the 2D model
handles best (i.e. what space the VR mosaic actually lives in).

Columns:
  raw            equirect axis-aligned crop (baseline)
  flat           per-region gnomonic dewarp (rectilinear)
  fisheye        whole-eye 180 equirect->fisheye, cropped
  flat+fe        flat, then fisheye-warp the whole patch
  flat+fe(mask)  flat, then fisheye-warp only the masked mosaic
  raw+fe(mask)   equirect crop, fisheye-warp only the masked mosaic

The '+fe' fisheye is an equidistant barrel warp of tunable strength
(--fisheye-fov); it is a probe, not a calibrated lens.

Usage (GPU):
  python scripts/vr_restore_experiment.py --input CLIP.mp4 --out DIR
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detection-model", default="rfdetr-vr-v1")
    ap.add_argument("--restoration-model", default="lada_mosaic_restoration_model_generic_v1.2.pth")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--fisheye-fov", type=float, default=120.0, help="strength of the +fe barrel warp (degrees)")
    ap.add_argument("--panel", type=int, default=256)
    return ap.parse_args()


def _pixel_centers(n, dev, dtype):
    import torch
    return (torch.arange(n, device=dev, dtype=dtype) + 0.5) / n


def _fisheye_forward_grid(eye_width, height, device, fov_degrees=180.0):
    """Whole-eye: fisheye-output pixel -> equirect source coord."""
    import torch
    half = math.radians(fov_degrees) * 0.5
    oy, ox = torch.meshgrid(_pixel_centers(height, device, torch.float64),
                            _pixel_centers(eye_width, device, torch.float64), indexing="ij")
    fx, fy = ox * 2 - 1, oy * 2 - 1
    r = torch.sqrt(fx * fx + fy * fy)
    theta = r * half
    phi = torch.atan2(fy, fx)
    dx, dy, dz = torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)
    lon, lat = torch.atan2(dx, dz), torch.asin(dy.clamp(-1, 1))
    gx, gy = (lon / math.pi + 0.5) * 2 - 1, (lat / math.pi + 0.5) * 2 - 1
    outside = r > 1.0
    gx = torch.where(outside, torch.full_like(gx, 2.0), gx)
    gy = torch.where(outside, torch.full_like(gy, 2.0), gy)
    return torch.stack((gx, gy), -1).unsqueeze(0).float()


def _equirect_to_fisheye_px(px, py, eye_width, height, fov_degrees=180.0):
    half = math.radians(fov_degrees) * 0.5
    lon = ((px + 0.5) / eye_width - 0.5) * math.pi
    lat = ((py + 0.5) / height - 0.5) * math.pi
    dx = math.cos(lat) * math.sin(lon)
    dy = math.sin(lat)
    dz = math.cos(lat) * math.cos(lon)
    theta = math.acos(max(-1.0, min(1.0, dz)))
    phi = math.atan2(dy, dx)
    r = theta / half
    return (
        (r * math.cos(phi) + 1) * 0.5 * (eye_width - 1),
        (r * math.sin(phi) + 1) * 0.5 * (height - 1),
    )


def _fisheye_patch_grid(h, w, fov_degrees, device):
    """Barrel warp: output(fisheye) pixel -> input(rectilinear) sample, in [-1,1]."""
    import torch
    half = math.radians(fov_degrees) * 0.5
    oy, ox = torch.meshgrid(_pixel_centers(h, device, torch.float64) * 2 - 1,
                            _pixel_centers(w, device, torch.float64) * 2 - 1, indexing="ij")
    r = torch.sqrt(ox * ox + oy * oy).clamp_min(1e-9)
    theta = r * half
    r_rect = torch.tan(theta) / math.tan(half)
    scale = r_rect / r
    return torch.stack((ox * scale, oy * scale), -1).unsqueeze(0).float()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    from jasna.crop_buffer import RawCrop, compute_enlarged_bbox, extract_crop, prepare_crops_for_restoration
    from jasna.engine_paths import model_weights_dir
    from jasna.media import get_video_meta_data
    from jasna.media.image_io import write_image_rgb_chw
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.mosaic.detection_registry import (
        build_detection_model, coerce_detection_model_name, detection_model_weights_path,
        recommended_score_threshold)
    from jasna.pipeline_processing import _eye_bounds
    from jasna.session_config import SessionConfig
    from jasna.session_factory import build_restoration_session
    from jasna.tracking.clip_tracker import ClipTracker
    from jasna.vr180 import SbsDetectionAdapter, resolve_vr_mode
    from jasna.vr_projection import GnomonicProjector

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fov = float(args.fisheye_fov)
    P = args.panel - (args.panel % 2)

    meta = get_video_meta_data(str(inp))
    H, W = int(meta.video_height), int(meta.video_width)
    vr = resolve_vr_mode("auto", meta, inp)
    is_sbs = vr.is_sbs
    eye_width = W // 2 if is_sbs else W
    print(f"[vr] {inp.name}: {W}x{H} resolved={vr.resolved} fov={fov}", flush=True)

    det_name = coerce_detection_model_name(args.detection_model)
    det_path = detection_model_weights_path(det_name)
    base_model = build_detection_model(
        det_name, det_path, batch_size=args.batch, device=device,
        score_threshold=recommended_score_threshold(det_name), fp16=True)
    detector = SbsDetectionAdapter(base_model) if is_sbs else base_model

    gnom = GnomonicProjector(eye_width=eye_width, height=H, device=device) if is_sbs else None
    fe_eye_grid = _fisheye_forward_grid(eye_width, H, device).to(device) if is_sbs else None

    def _to_np(chw_u8):
        return chw_u8.permute(1, 2, 0).contiguous().cpu().numpy()

    def _warp_fe(crop_chw_u8):
        h, w = int(crop_chw_u8.shape[1]), int(crop_chw_u8.shape[2])
        grid = _fisheye_patch_grid(h, w, fov, device).to(crop_chw_u8.device)
        s = F.grid_sample(crop_chw_u8.unsqueeze(0).float(), grid, mode="bilinear",
                          padding_mode="border", align_corners=True)[0]
        return s.round().clamp(0, 255).to(torch.uint8)

    def _composite(base, warped, mask_chw):
        alpha = (mask_chw[:1] > 127).float()
        return (warped.float() * alpha + base.float() * (1 - alpha)).round().clamp(0, 255).to(torch.uint8)

    def _fisheye_region(frame, ebox, x_bounds):
        offset = x_bounds[0] if x_bounds is not None else 0
        x1, y1, x2, y2 = ebox
        eye = frame[:, :, offset:offset + eye_width].unsqueeze(0).float()
        fe = F.grid_sample(eye, fe_eye_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        xs, ys = [], []
        for px in (x1 - offset, x2 - offset):
            for py in (y1, y2):
                fx, fy = _equirect_to_fisheye_px(px, py, eye_width, H)
                xs.append(fx); ys.append(fy)
        fx1, fx2 = max(0, int(min(xs))), min(eye_width, int(max(xs)) + 1)
        fy1, fy2 = max(0, int(min(ys))), min(H, int(max(ys)) + 1)
        if fx2 <= fx1 or fy2 <= fy1:
            return fe.round().clamp(0, 255).to(torch.uint8)
        return fe[:, fy1:fy2, fx1:fx2].round().clamp(0, 255).to(torch.uint8)

    VARIANTS = ["raw", "flat", "fisheye", "flat+fe", "flat+fe(mask)", "raw+fe(mask)"]

    def _build_crops(frame, mask_frame, bbox):
        xb = _eye_bounds(bbox, eye_width, W) if is_sbs else None
        ebox = compute_enlarged_bbox(bbox, H, W, xb)
        raw = extract_crop(frame, bbox, H, W, x_bounds=xb).crop
        raw_m = extract_crop(mask_frame, bbox, H, W, x_bounds=xb).crop
        out = {"raw": raw, "raw+fe(mask)": _composite(raw, _warp_fe(raw), raw_m)}
        if gnom is not None:
            flat = gnom.extract_region_crop(frame, bbox, H, W, x_bounds=xb).crop
            flat_m = gnom.extract_region_crop(mask_frame, bbox, H, W, x_bounds=xb).crop
            out["flat"] = flat
            out["flat+fe"] = _warp_fe(flat)
            out["flat+fe(mask)"] = _composite(flat, _warp_fe(flat), flat_m)
            out["fisheye"] = _fisheye_region(frame, ebox, xb)
        return ebox, out

    # --- pass 1: detect + track ---
    tracker = ClipTracker(max_clip_size=10_000_000, temporal_overlap=0, iou_threshold=0.3, max_detection_gap=6)
    clips = []
    fidx = 0
    with NvidiaVideoReader(str(inp), args.batch, device, meta, frame_stride=args.frame_stride) as reader, \
            torch.inference_mode():
        for batch, pts in reader.frames():
            det = detector(batch, target_hw=(H, W))
            for i in range(len(pts)):
                ended, _ = tracker.update(fidx, det.boxes_xyxy[i], det.masks[i])
                clips.extend(e.clip for e in ended)
                fidx += 1
    clips.extend(e.clip for e in tracker.flush())
    total = fidx
    if not clips:
        print("[!] no tracks", file=sys.stderr)
        return 1

    mid = total // 2

    def _is_left(c):
        cx = (float(c.bboxes[0][0]) + float(c.bboxes[0][2])) * 0.5
        return eye_width == W or cx < eye_width

    pool = [c for c in clips if _is_left(c)] or clips
    chosen = max(pool, key=lambda c: (c.start_frame <= mid <= c.end_frame, c.frame_count))
    frames_all = list(chosen.frame_indices())
    n = min(args.frames, len(frames_all))
    center = mid if chosen.start_frame <= mid <= chosen.end_frame else (chosen.start_frame + chosen.end_frame) // 2
    ci = center - chosen.start_frame
    lo = max(0, min(ci - n // 2, len(frames_all) - n))
    wanted = frames_all[lo:lo + n]
    wanted_set = set(wanted)
    print(f"[track] id={chosen.track_id} frames {wanted[0]}..{wanted[-1]} ({len(wanted)})", flush=True)

    # --- pass 2: build the per-variant RawCrop sequences ---
    seqs = {v: [] for v in VARIANTS}
    fidx = 0
    saved = 0
    with NvidiaVideoReader(str(inp), args.batch, device, meta, frame_stride=args.frame_stride) as reader, \
            torch.inference_mode():
        for batch, pts in reader.frames():
            for i in range(len(pts)):
                if fidx in wanted_set:
                    off = fidx - chosen.start_frame
                    bbox = chosen.bboxes[off]
                    m = chosen.masks[off].float()[None, None]
                    mask_frame = F.interpolate(m, size=(H, W), mode="nearest")[0].mul(255).to(torch.uint8)
                    ebox, crops = _build_crops(batch[i], mask_frame, bbox)
                    cs = (int(ebox[3] - ebox[1]), int(ebox[2] - ebox[0]))
                    for v in VARIANTS:
                        c = crops.get(v)
                        if c is None:
                            continue
                        seqs[v].append(RawCrop(crop=c.cpu(), enlarged_bbox=ebox,
                                               crop_shape=(int(c.shape[1]), int(c.shape[2]))))
                    saved += 1
                fidx += 1
                if saved == len(wanted_set):
                    break
            if saved == len(wanted_set):
                break

    base_model.close()

    # --- build the restoration session after detection is freed ---
    config = SessionConfig(
        device="cuda:0", fp16=True, batch_size=args.batch,
        detection_model_name=det_name, detection_model_path=det_path,
        detection_score_threshold=recommended_score_threshold(det_name),
        max_detection_gap=0, min_detection_duration=0, scene_detection=False,
        restoration_model_path=model_weights_dir() / args.restoration_model,
        compile_basicvsrpp=True, max_clip_size=args.frames, temporal_overlap=0,
        enable_crossfade=False, denoise_strength="none", denoise_step="after_primary",
        secondary_restoration="unet-4x", tvai_ffmpeg_path="", tvai_model="", tvai_scale=1,
        tvai_args="", tvai_workers=1, rtx_scale=2, rtx_quality="high", rtx_denoise="none",
        rtx_deblur="none", vr_mode="auto", codec="hevc", encoder_settings={}, lut_path=None,
        retarget_high_fps=False, disable_progress=True, working_dir=None)
    session = build_restoration_session(config, disable_basicvsrpp_tensorrt=False, log_callback=print)
    primary = session.restoration_pipeline.restorer
    secondary = session.secondary_restorer

    # --- restore each variant (primary + secondary) ---
    inputs = {}
    restored = {}
    for v in VARIANTS:
        raws = seqs[v]
        if not raws:
            continue
        r256, _pad, _rs = prepare_crops_for_restoration(raws, device=device, dtype=torch.float16)
        prim = primary.raw_process([c for c in r256])  # (T,C,256,256) [0,1]
        sec = secondary.restore(prim, keep_start=0, keep_end=prim.shape[0])  # list (C,1024,1024) uint8
        inputs[v] = [c.round().clamp(0, 255).to(torch.uint8) for c in r256]
        restored[v] = [F.interpolate(s.unsqueeze(0).float(), size=(P, P), mode="area")[0]
                       .round().clamp(0, 255).to(torch.uint8) for s in sec]

    # --- montage -> mp4 ---
    frames_dir = out / f"frames_track{chosen.track_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _tile(img_np, label):
        r = cv2.resize(img_np, (P, P), interpolation=cv2.INTER_NEAREST)
        bar = np.zeros((22, P, 3), np.uint8)
        cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return bar, r

    ncols = [v for v in VARIANTS if v in inputs]
    for k in range(len(wanted)):
        cols = []
        for v in ncols:
            bar, inp_t = _tile(_to_np(inputs[v][k]), v)
            _, res_t = _tile(_to_np(restored[v][k]), v)
            gap = np.full((2, P, 3), 60, np.uint8)
            cols.append(np.vstack([bar, inp_t, gap, res_t]))
        sep = np.full((cols[0].shape[0], 4, 3), 40, np.uint8)
        row = []
        for j, c in enumerate(cols):
            if j:
                row.append(sep)
            row.append(c)
        montage = np.hstack(row)
        if montage.shape[1] % 2:
            montage = montage[:, :-1]
        write_image_rgb_chw(str(frames_dir / f"{k:05d}.png"), montage.transpose(2, 0, 1))

    fps = float(meta.video_fps) or 30.0
    mp4 = out / f"{inp.stem}_track{chosen.track_id}_restore-experiment.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", f"{fps:.4f}", "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
    print(f"[done] {len(wanted)} frames, {len(ncols)} variants (top=input, bottom=restored) -> {mp4}", flush=True)
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

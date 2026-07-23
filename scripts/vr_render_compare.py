"""Render one mosaic track cropped from source / old-restored / new-restored.

Old-vs-new restoration diagnostic: detects a LEFT-eye track near the middle of
the source clip, then for a window of consecutive frames crops the SAME region
(axis-aligned, in place) from three frame-aligned videos and writes an mp4 of a
labelled montage:

  * source - the untouched input frame ROI
  * old    - previous restoration (whole-eye fisheye / plain)
  * new    - per-region flat-projection restoration

Usage (GPU):

  python scripts/vr_render_compare.py --source SRC.mp4 --old OLD.mp4 \
      --new NEW.mp4 --out DIR [--frames 180]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detection-model", default="rfdetr-vr-v1")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--panel-height", type=int, default=512)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import numpy as np
    import torch

    from jasna.crop_buffer import extract_crop
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

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    src = Path(args.source).expanduser().resolve()
    old = Path(args.old).expanduser().resolve()
    new = Path(args.new).expanduser().resolve()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    meta = get_video_meta_data(str(src))
    height, width = int(meta.video_height), int(meta.video_width)
    vr = resolve_vr_mode("auto", meta, src)
    is_sbs = vr.is_sbs
    eye_width = width // 2 if is_sbs else None
    print(f"[vr] {src.name}: {width}x{height} resolved={vr.resolved}", flush=True)

    name = coerce_detection_model_name(args.detection_model)
    wpath = detection_model_weights_path(name)
    precompile_detection_engine(name, wpath, batch_size=args.batch, device=device, fp16=True)
    base_model = build_detection_model(
        name, wpath, batch_size=args.batch, device=device,
        score_threshold=recommended_score_threshold(name), fp16=True)
    detector = SbsDetectionAdapter(base_model) if is_sbs else base_model

    # --- pass 1: detect + track on the source ---
    tracker = ClipTracker(max_clip_size=10_000_000, temporal_overlap=0,
                          iou_threshold=0.3, max_detection_gap=6)
    clips = []
    frame_idx = 0
    with NvidiaVideoReader(str(src), args.batch, device, meta, frame_stride=args.frame_stride) as reader, \
            torch.inference_mode():
        for batch, pts in reader.frames():
            det = detector(batch, target_hw=(height, width))
            for i in range(len(pts)):
                ended, _ = tracker.update(frame_idx, det.boxes_xyxy[i], det.masks[i])
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
          f"frames {chosen.start_frame}..{chosen.end_frame} count={chosen.frame_count}", flush=True)

    frames_all = list(chosen.frame_indices())
    n = min(args.frames, len(frames_all))
    center = mid if chosen.start_frame <= mid <= chosen.end_frame else (chosen.start_frame + chosen.end_frame) // 2
    ci = center - chosen.start_frame
    lo = max(0, min(ci - n // 2, len(frames_all) - n))
    wanted = frames_all[lo:lo + n]
    wanted_set = set(wanted)
    order = {fi: k for k, fi in enumerate(wanted)}
    bbox_of = {fi: chosen.bboxes[fi - chosen.start_frame] for fi in wanted}

    size = args.panel_height - (args.panel_height % 2)
    frames_dir = out / f"frames_track{chosen.track_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _panel(label, chw_u8):
        img = chw_u8.permute(1, 2, 0).contiguous().cpu().numpy()
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

    # --- pass 2: crop the same ROI from source / old / new in lockstep ---
    saved = 0
    fidx = 0
    with NvidiaVideoReader(str(src), args.batch, device, meta, frame_stride=args.frame_stride) as rs, \
            NvidiaVideoReader(str(old), args.batch, device, meta, frame_stride=args.frame_stride) as ro, \
            NvidiaVideoReader(str(new), args.batch, device, meta, frame_stride=args.frame_stride) as rn, \
            torch.inference_mode():
        for (bs, _p), (bo, _), (bn, _) in zip(rs.frames(), ro.frames(), rn.frames()):
            m = min(bs.shape[0], bo.shape[0], bn.shape[0])
            for i in range(m):
                if fidx in wanted_set:
                    bbox = bbox_of[fidx]
                    xb = _eye_bounds(bbox, eye_width, width) if is_sbs else None
                    crops = [
                        ("source", extract_crop(bs[i], bbox, height, width, x_bounds=xb).crop),
                        ("old", extract_crop(bo[i], bbox, height, width, x_bounds=xb).crop),
                        ("new", extract_crop(bn[i], bbox, height, width, x_bounds=xb).crop),
                    ]
                    tiles = [_panel(lbl, t) for lbl, t in crops]
                    sep = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
                    row = []
                    for k, t in enumerate(tiles):
                        if k:
                            row.append(sep)
                        row.append(t)
                    montage = np.hstack(row)
                    write_image_rgb_chw(str(frames_dir / f"{order[fidx]:05d}.png"),
                                        montage.transpose(2, 0, 1))
                    saved += 1
                fidx += 1
                if saved == len(wanted_set):
                    break
            if saved == len(wanted_set):
                break

    fps = float(meta.video_fps) or 30.0
    mp4 = out / f"{src.stem}_track{chosen.track_id}_oldvsnew.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-framerate", f"{fps:.4f}", "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
        check=True,
    )
    print(f"[done] track {chosen.track_id}: {saved} frames -> {mp4}", flush=True)
    if hasattr(detector, "close"):
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

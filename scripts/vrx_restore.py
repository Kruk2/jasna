"""Phase-2 restore: 3-variant projection restoration + source-space composite.

Loads BasicVSR++ (+unet-4x available but the composite uses the primary delta)
once. For each discovered sample: PTS-aligned re-seek to reproduce the source
eyes, build raw / gnomonic / fisheye conditioning patches (equal spherical
support), restore each temporally, delta-composite back into source space, render
a gnomonic headset viewport, and emit BLIND A/B/C final-view videos + a key.

Projection-space crops are diagnostic; ranking uses the source-space composite
and the viewport. Uses scripts.vr_projection (FFmpeg-validated) for all grids.

Usage:
  python scripts/vrx_restore.py --root VR_PROJECTION_DIR [--only SAMPLE_ID]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH = 256
CORE = ["raw", "gnomonic", "fisheye"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", default=None, help="restore a single sample_id")
    ap.add_argument("--restoration-model", default="lada_mosaic_restoration_model_generic_v1.2.pth")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--viewport-fov", type=float, default=0.0, help="0 = auto from region")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    from jasna.crop_buffer import compute_enlarged_bbox
    from jasna.engine_paths import model_weights_dir
    from jasna.media import get_video_meta_data
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.session_config import SessionConfig
    from jasna.session_factory import build_restoration_session
    from scripts import vr_projection as vp
    from scripts.vrx_harness import SampleRecord, blind_labels

    root = Path(args.root)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    recs = [SampleRecord(**json.loads(l)) for l in (root / "discovery.jsonl").read_text().splitlines() if l.strip()]
    if args.only:
        recs = [r for r in recs if r.sample_id == args.only]

    cfg = SessionConfig(
        device="cuda:0", fp16=True, batch_size=args.batch,
        detection_model_name="rfdetr-vr-v1",
        detection_model_path=model_weights_dir() / "rfdetr-vr-v1.onnx",
        detection_score_threshold=0.4, max_detection_gap=0, min_detection_duration=0,
        scene_detection=False, restoration_model_path=model_weights_dir() / args.restoration_model,
        compile_basicvsrpp=True, max_clip_size=180, temporal_overlap=0, enable_crossfade=False,
        denoise_strength="none", denoise_step="after_primary", secondary_restoration="none",
        tvai_ffmpeg_path="", tvai_model="", tvai_scale=1, tvai_args="", tvai_workers=1,
        rtx_scale=2, rtx_quality="high", rtx_denoise="none", rtx_deblur="none",
        vr_mode="auto", codec="hevc", encoder_settings={}, lut_path=None,
        retarget_high_fps=False, disable_progress=True, working_dir=None)
    session = build_restoration_session(cfg, disable_basicvsrpp_tensorrt=False, log_callback=print)
    primary = session.restoration_pipeline.restorer

    def sample_eye(eye_f, uv_np):  # eye_f (3,H,Ew) float; uv (h,w,2) in [0,1] -> (3,h,w)
        grid = torch.from_numpy(uv_np * 2 - 1).float().unsqueeze(0).to(device)
        return F.grid_sample(eye_f.unsqueeze(0), grid, mode="bilinear",
                             padding_mode="border", align_corners=False)[0]

    def variant_grids(bbox_uv, region_wh):
        """Return dict variant -> (fwd_uv (P,P,2) eye-uv, inv_uv (rh,rw,2) patch-uv, cov)."""
        rw, rh = region_wh
        u1, v1, u2, v2 = bbox_uv
        tx = (np.arange(PATCH) + 0.5) / PATCH
        pyy, pxx = np.meshgrid(tx, tx, indexing="ij")
        # region eye-uv grid (source pixels inside the enlarged bbox)
        ru = u1 + (u2 - u1) * (np.arange(rw) + 0.5) / rw
        rv = v1 + (v2 - v1) * (np.arange(rh) + 0.5) / rh
        rvv, ruu = np.meshgrid(rv, ru, indexing="ij")
        out = {}
        # raw
        fwd = np.stack((u1 + (u2 - u1) * pxx, v1 + (v2 - v1) * pyy), -1)
        inv = np.stack((np.broadcast_to((np.arange(rw) + 0.5) / rw, (rh, rw)),
                        np.broadcast_to(((np.arange(rh) + 0.5) / rh)[:, None], (rh, rw))), -1)
        out["raw"] = (fwd, inv, np.ones((rh, rw), bool))
        # gnomonic (square FOV from max extent, no X/Y stretch)
        lon0, lat0, hf, vf, xmax, ymax = vp.region_gnomonic_spec(bbox_uv)
        fov = 2 * math.degrees(math.atan(max(xmax, ymax)))
        guv, _ = vp.v360_map("flat", PATCH, PATCH, h_fov=fov, v_fov=fov, yaw=lon0, pitch=lat0)
        vec_raw = vp.hequirect_uv_to_xyz(ruu, rvv)           # un-rotated region directions
        vecr = vp.rotate_inv(vec_raw, lon0, lat0, 0.0)        # gnomonic-centred
        gpuv, gcov = vp.xyz_to_flat_uv(vecr, fov, fov)
        out["gnomonic"] = (guv, gpuv, gcov)
        # fisheye: sub-window of the fixed 180 disc covering the ROI (squared)
        pu = np.concatenate([np.linspace(u1, u2, 33), np.full(33, u1), np.full(33, u2), np.linspace(u1, u2, 33)])
        pv = np.concatenate([np.full(33, v1), np.linspace(v1, v2, 33), np.linspace(v1, v2, 33), np.full(33, v2)])
        fdir = vp.hequirect_uv_to_xyz(pu, pv)
        fuv, _ = vp.xyz_to_fisheye_uv(fdir, 180, 180)
        cxx, cyy = (fuv[:, 0].min() + fuv[:, 0].max()) / 2, (fuv[:, 1].min() + fuv[:, 1].max()) / 2
        half = max(np.ptp(fuv[:, 0]), np.ptp(fuv[:, 1])) / 2 * 1.06
        f1u, f2u, f1v, f2v = cxx - half, cxx + half, cyy - half, cyy + half
        fdir2 = vp.fisheye_uv_to_xyz(f1u + (f2u - f1u) * pxx, f1v + (f2v - f1v) * pyy)
        feye, _ = vp.xyz_to_hequirect_uv(fdir2)
        finv, fcov = vp.xyz_to_fisheye_uv(vec_raw, 180, 180)
        finvn = np.stack(((finv[..., 0] - f1u) / (f2u - f1u), (finv[..., 1] - f1v) / (f2v - f1v)), -1)
        out["fisheye"] = (feye, finvn, fcov)
        return out, (lon0, lat0)

    for rec in recs:
        sdir = root / "samples" / rec.sample_id
        outdir = root / "blind" / rec.sample_id
        outdir.mkdir(parents=True, exist_ok=True)
        (root / "keys").mkdir(parents=True, exist_ok=True)
        meta = get_video_meta_data(rec.source_path)
        H, W = rec.height, rec.width
        eye_w = W // 2
        pts_want = list(np.load(sdir / "pts.npy"))
        masks = np.load(rec.mask_ref)  # (F,Hm,Wm) bool
        bboxes = rec.bboxes
        print(f"[restore] {rec.sample_id} pos={rec.position} frames={len(pts_want)}", flush=True)

        # PTS-aligned reproduce: build patches/regions inline, discard each 8K eye.
        F_ = len(pts_want)
        want = {int(p): k for k, p in enumerate(pts_want)}
        inputs = {v: [None] * F_ for v in CORE}
        composites = {v: [] for v in CORE}
        grids_per_frame = [None] * F_
        regions = [None] * F_
        with NvidiaVideoReader(rec.source_path, args.batch, device, meta) as r, torch.inference_mode():
            for batch, pts in r.frames(seek_ts=rec.seek_ts):
                for i in range(len(pts)):
                    k = want.get(int(pts[i]))
                    if k is None or grids_per_frame[k] is not None:
                        continue
                    eye = batch[i, :, :, :eye_w].float()
                    x1, y1, x2, y2 = compute_enlarged_bbox(np.array(bboxes[k], np.float32), H, W, (0, eye_w))
                    rw, rh = x2 - x1, y2 - y1
                    bbox_uv = (x1 / eye_w, y1 / H, x2 / eye_w, y2 / H)
                    gr, center = variant_grids(bbox_uv, (rw, rh))
                    grids_per_frame[k] = (gr, (x1, y1, x2, y2), center)
                    regions[k] = eye[:, y1:y2, x1:x2].clone()
                    for v in CORE:
                        inputs[v][k] = sample_eye(eye, gr[v][0]).clone()
                if all(g is not None for g in grids_per_frame):
                    break
        if any(g is None for g in grids_per_frame):
            print(f"  [!] PTS align failed ({sum(g is None for g in grids_per_frame)} missing) -> skip", flush=True)
            continue

        # restore each variant (primary), delta-composite into source region
        mask_t = torch.from_numpy(masks.astype(np.float32)).to(device)
        for v in CORE:
            prim = primary.raw_process([p for p in inputs[v]])  # (F,3,256,256) [0,1]
            prim = prim.clamp(0, 1) * 255.0
            for k in range(F_):
                gr, (x1, y1, x2, y2), center = grids_per_frame[k]
                rw, rh = x2 - x1, y2 - y1
                inp = inputs[v][k]
                delta = prim[k] - inp                                  # patch-space delta [0..255]
                inv = gr[v][1]; cov = gr[v][2]
                ig = torch.from_numpy(inv * 2 - 1).float().unsqueeze(0).to(device)
                region_delta = F.grid_sample(delta.unsqueeze(0), ig, mode="bilinear",
                                             padding_mode="zeros", align_corners=False)[0]
                covm = torch.from_numpy(cov.astype(np.float32)).to(device)
                # crop the full-frame detection mask to this region, then upsample
                Hm, Wm = mask_t.shape[1], mask_t.shape[2]
                mx1, mx2 = max(0, round(x1 / W * Wm)), min(Wm, round(x2 / W * Wm))
                my1, my2 = max(0, round(y1 / H * Hm)), min(Hm, round(y2 / H * Hm))
                msub = mask_t[k][my1:my2, mx1:mx2]
                if msub.numel() == 0:
                    msub = mask_t[k]
                m = F.interpolate(msub[None, None], size=(rh, rw), mode="bilinear",
                                  align_corners=False)[0, 0]
                blend = (m * covm).clamp(0, 1)
                out = regions[k].clone()
                out = out + region_delta * blend
                composites[v].append(out.clamp(0, 255))

        # render: source-region composite + gnomonic viewport, per variant -> mp4 frames
        variant_frames = {v: [] for v in CORE}
        for v in CORE:
            for k in range(F_):
                gr, (x1, y1, x2, y2), (lon0, lat0) = grids_per_frame[k]
                comp = composites[v][k]  # (3,rh,rw) float [0,255], equirect region
                # viewport: gnomonic view centred on mosaic, sampled from the region
                rw, rh = x2 - x1, y2 - y1
                u1, v1 = x1 / eye_w, y1 / H
                du, dv = (x2 - x1) / eye_w, (y2 - y1) / H
                lon0g, lat0g, hf, vf, xmax, ymax = vp.region_gnomonic_spec((u1, v1, x2 / eye_w, y2 / H))
                fov = 2 * math.degrees(math.atan(max(xmax, ymax)))
                vuv, _ = vp.v360_map("flat", PATCH, PATCH, h_fov=fov, v_fov=fov, yaw=lon0g, pitch=lat0g)
                # eye-uv -> region-local normalized
                ruvn = np.stack(((vuv[..., 0] - u1) / du, (vuv[..., 1] - v1) / dv), -1)
                rg = torch.from_numpy(ruvn * 2 - 1).float().unsqueeze(0).to(device)
                vp_img = F.grid_sample(comp.unsqueeze(0), rg, mode="bilinear",
                                       padding_mode="zeros", align_corners=False)[0]
                # montage [region resized-square | viewport]
                reg_sq = F.interpolate(comp.unsqueeze(0), size=(PATCH, PATCH), mode="bilinear",
                                       align_corners=False)[0]
                tile = torch.cat([reg_sq, vp_img], dim=2)  # side by side
                variant_frames[v].append(tile.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

        # blind A/B/C
        labels = blind_labels(rec.sample_id, CORE)  # label -> variant
        (root / "keys" / f"{rec.sample_id}.json").write_text(json.dumps(labels))
        fps = rec.fps or 30.0
        for label, v in labels.items():
            fdir_ = outdir / f"frames_{label}"
            fdir_.mkdir(exist_ok=True)
            for k, im in enumerate(variant_frames[v]):
                cv2.imwrite(str(fdir_ / f"{k:04d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            mp4 = outdir / f"{label}.mp4"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-framerate", f"{fps:.3f}", "-i", str(fdir_ / "%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
        print(f"  [done] blind A/B/C -> {outdir} (key hidden in keys/)", flush=True)

    session.close()
    print("[all done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

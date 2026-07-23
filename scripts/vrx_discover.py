"""Phase-2 discovery: deterministic random-seek track sampling (GPU).

Loads rfdetr-vr-v1 once. For each input clip, generates deterministic seek
candidates, decodes a bounded window at each, detects+tracks, scores tracks, and
retains one stable `center` and one stable `bottom/off-axis` LEFT-eye sample
where available. Serializes discovery.jsonl + per-sample bboxes/masks/pts so the
restore phase can reproduce crops without re-running discovery.

Usage:
  python scripts/vrx_discover.py --root VR_PROJECTION_DIR \
      --inputs 'VIDEO_DIR/*.mp4'
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RESTORE_FRAMES = 60
DISCOVERY_WINDOW = 80
DETECT_BATCH = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--detection-model", default="rfdetr-vr-v1")
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--restore-frames", type=int, default=RESTORE_FRAMES)
    ap.add_argument("--window", type=int, default=DISCOVERY_WINDOW)
    return ap.parse_args()


def studio_of(name: str, fisheye_tokens, direct_tokens) -> tuple[str, str]:
    s = name.upper()
    m = re.match(r"^([0-9]?[A-Z]{2,7})", s)
    studio = m.group(1) if m else s
    if any(t in s for t in fisheye_tokens):
        prior = "fisheye"
    elif any(t in s for t in direct_tokens):
        prior = "raw"
    else:
        prior = "unknown"
    return studio, prior


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import numpy as np
    import torch

    from jasna.media import get_video_meta_data
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.mosaic.detection_registry import (
        build_detection_model, coerce_detection_model_name,
        detection_model_weights_path, recommended_score_threshold)
    from jasna.tracking.clip_tracker import ClipTracker
    from jasna.vr180 import (DIRECT_STUDIO_TOKENS, FISHEYE_STUDIO_TOKENS,
                             SbsDetectionAdapter, resolve_vr_mode)
    from scripts.vrx_harness import (SampleRecord, angular_center, position_bin,
                                     seek_candidates, stable_seed, track_stability,
                                     write_jsonl)

    root = Path(args.root)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    name = coerce_detection_model_name(args.detection_model)
    wpath = detection_model_weights_path(name)
    base = build_detection_model(name, wpath, batch_size=DETECT_BATCH, device=device,
                                 score_threshold=recommended_score_threshold(name), fp16=True)

    records: list[SampleRecord] = []

    for inp in [Path(p) for p in args.inputs]:
        meta = get_video_meta_data(str(inp))
        H, W = int(meta.video_height), int(meta.video_width)
        vr = resolve_vr_mode("auto", meta, inp)
        if not vr.is_sbs:
            print(f"[skip] {inp.name}: not SBS ({vr.resolved})", flush=True)
            continue
        eye_w = W // 2
        detector = SbsDetectionAdapter(base)
        studio, prior = studio_of(inp.stem, FISHEYE_STUDIO_TOKENS, DIRECT_STUDIO_TOKENS)
        dur = float(meta.duration)
        seed = stable_seed(studio, inp.stem)
        cands = seek_candidates(dur, seed, n=args.candidates)
        print(f"[clip] {inp.name} {W}x{H} {vr.resolved} studio={studio} prior={prior} "
              f"dur={dur:.0f}s cands={len(cands)}", flush=True)

        picked: dict[str, dict] = {}   # bin_group -> best sample dict
        for ci, ts in enumerate(cands):
            if "center" in picked and "bottom" in picked:
                break
            tracker = ClipTracker(max_clip_size=10_000_000, temporal_overlap=0,
                                  iou_threshold=0.3, max_detection_gap=6)
            frames_meta = []   # (frame_idx, pts)
            n = 0
            with NvidiaVideoReader(str(inp), DETECT_BATCH, device, meta) as reader, torch.inference_mode():
                for batch, pts in reader.frames(seek_ts=ts):
                    det = detector(batch, target_hw=(H, W))
                    for i in range(len(pts)):
                        tracker.update(n, det.boxes_xyxy[i], det.masks[i])
                        frames_meta.append((n, int(pts[i])))
                        n += 1
                        if n >= args.window:
                            break
                    if n >= args.window:
                        break
            clips = [e.clip for e in tracker.flush()]
            pts_by_idx = dict(frames_meta)

            for clip in clips:
                if clip.frame_count < args.restore_frames:
                    continue
                cx = np.array([(b[0] + b[2]) * 0.5 for b in clip.bboxes])
                cyc = np.array([(b[1] + b[3]) * 0.5 for b in clip.bboxes])
                if cx.mean() >= eye_w:      # left eye only
                    continue
                # eye-normalized centres / areas
                un = cx / eye_w; vn = cyc / H
                areas = np.array([((b[2] - b[0]) * (b[3] - b[1])) / (eye_w * H) for b in clip.bboxes])
                coverage = clip.frame_count / args.window
                stab = track_stability(np.stack([un, vn], 1), areas, coverage)
                if not stab["stable"]:
                    continue
                lon, lat = angular_center(float(un.mean()), float(vn.mean()))
                pos = position_bin(lat, lon)
                grp = "bottom" if pos.startswith("bottom") else ("center" if pos == "center" else None)
                if grp is None:
                    continue
                if grp in picked and picked[grp]["score"] >= stab["score"]:
                    continue
                # take the central restore-frames window of the track
                lo = max(0, (clip.frame_count - args.restore_frames) // 2)
                idxs = list(range(clip.start_frame + lo, clip.start_frame + lo + args.restore_frames))
                bboxes = [clip.bboxes[k - clip.start_frame].tolist() for k in idxs]
                masks = torch.stack([clip.masks[k - clip.start_frame] for k in idxs]).cpu().numpy()
                ptss = [pts_by_idx[k] for k in idxs]
                picked[grp] = dict(
                    score=stab["score"], seek_ts=ts, pos=pos, lon=lon, lat=lat,
                    bboxes=bboxes, masks=masks, pts=ptss, stab=stab,
                    W=W, H=H, fps=float(meta.video_fps),
                )
                print(f"  [cand {ci}] {grp} pos={pos} lat={lat:.1f} lon={lon:.1f} "
                      f"cov={coverage:.2f} score={stab['score']:.2f}", flush=True)

        for grp, s in picked.items():
            sid = f"{studio}__{inp.stem}__{grp}"
            sdir = root / "samples" / sid
            sdir.mkdir(parents=True, exist_ok=True)
            np.save(sdir / "masks.npy", s["masks"])
            rec = SampleRecord(
                sample_id=sid, studio=studio, title=inp.stem, source_path=str(inp),
                seek_ts=float(s["seek_ts"]), pts_start=int(s["pts"][0]), pts_end=int(s["pts"][-1]),
                width=s["W"], height=s["H"], fps=s["fps"], eye="left",
                track_frames=list(range(len(s["pts"]))), bboxes=s["bboxes"],
                mask_ref=str(sdir / "masks.npy"), mask_shape=list(s["masks"].shape[1:]),
                center_lon=float(s["lon"]), center_lat=float(s["lat"]), position=s["pos"],
                stability=s["stab"], vr_reason=vr.reason, zelefans_prior=prior,
            )
            np.save(sdir / "pts.npy", np.array(s["pts"], dtype=np.int64))
            records.append(rec)
            print(f"  [saved] {sid}", flush=True)

    if hasattr(base, "close"):
        base.close()
    write_jsonl(root / "discovery.jsonl", records)
    print(f"[done] {len(records)} samples -> {root/'discovery.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

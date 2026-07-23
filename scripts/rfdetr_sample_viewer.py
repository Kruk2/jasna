#!/usr/bin/env python
"""Interactive detection-model sample viewer.

Classifies each sample image by resolution and runs the matching detector set:

- non-VR (anything that is not exact 2:1)  -> non-VR models on the whole frame.
- VR      (width == 2 * height, jasna's    -> split down the middle and run the VR
           auto side-by-side gate)             models on each eye separately, then
                                                composite the two eyes back together.

Several models can be attached to each category; press Q / E to swap between them on
the current frame. TensorRT engines are built through jasna's own logic
(``precompile_detection_engine``), so the viewer sees exactly what the pipeline sees.
Any detector jasna supports works here (RF-DETR and YOLO share one interface).

Window controls:
  A / D   previous / next image
  Q / E   previous / next model (within the current frame's category)
  M       toggle mask(s)
  Esc     quit

Examples:
  python scripts/rfdetr_sample_viewer.py /path/to/samples
  python scripts/rfdetr_sample_viewer.py /path/to/samples \\
      --non-vr-models rfdetr-v5,lada-yolo-v4,rfdetr-v6,rfdetr-v6-large \\
      --vr-models zelefans-vr-yolo-v2,rfdetr-vr-v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NON_VR = "rfdetr-v5,lada-yolo-v4,rfdetr-v6,rfdetr-v6-large"
DEFAULT_VR = "zelefans-vr-yolo-v2,rfdetr-vr-v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="directory of sample images (searched recursively)")
    ap.add_argument("--non-vr-models", default=DEFAULT_NON_VR, help="comma-separated detectors for non-VR frames")
    ap.add_argument("--vr-models", default=DEFAULT_VR, help="comma-separated detectors for VR frames (run per eye)")
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help=(
            "maximum/optimal batch for dynamic engines; legacy static RF-DETR "
            "models use their fixed batch (default: %(default)s)"
        ),
    )
    ap.add_argument("--exts", default="jpg,jpeg,png,webp,bmp", help="comma-separated image extensions")
    ap.add_argument("--include-mask-files", action="store_true",
                    help="also load files named mask.* (excluded by default; sample dirs pair frame/mask)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (0 = all)")
    ap.add_argument("--headless", action="store_true", help="build engines + run every model once + print, no window")
    ap.add_argument("--disp-max", default="1600x900", help="max display WxH before downscaling (default: %(default)s)")
    return ap.parse_args()


def collect_images(samples: Path, exts: set[str], include_mask: bool) -> list[Path]:
    files: list[Path] = []
    for p in sorted(samples.rglob("*")):
        if not p.is_file() or p.suffix.lower().lstrip(".") not in exts:
            continue
        if not include_mask and p.stem.lower() == "mask":
            continue
        files.append(p)
    return files


def main() -> int:
    args = parse_args()
    samples = Path(args.samples).expanduser().resolve()
    if not samples.is_dir():
        print(f"not a directory: {samples}", file=sys.stderr)
        return 2
    exts = {e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}
    disp_max_w, disp_max_h = (int(v) for v in args.disp_max.lower().split("x"))

    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import numpy as np
    import torch

    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        detection_model_weights_path,
        precompile_detection_engine,
        recommended_score_threshold,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    files = collect_images(samples, exts, args.include_mask_files)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images found under {samples}", file=sys.stderr)
        return 1

    items = []
    for file in files:
        bgr = cv2.imread(str(file))
        if bgr is None:
            print(f"[skip] unreadable: {file}", file=sys.stderr)
            continue
        height, width = bgr.shape[:2]
        items.append(
            {
                "bgr": bgr,
                "is_vr": width == 2 * height,
                "name": (
                    file.parent.name
                    if file.name == "frame.jpg"
                    else file.name
                ),
            }
        )
    if not items:
        return 1

    def available(names: list[str]) -> list[str]:
        out = []
        for raw in names:
            name = coerce_detection_model_name(raw)
            weights_path = detection_model_weights_path(name)
            if weights_path.is_file():
                out.append(name)
            else:
                print(
                    f"[skip] {name}: weights not found ({weights_path})",
                    file=sys.stderr,
                )
        return out

    non_vr_models = available([m.strip() for m in args.non_vr_models.split(",") if m.strip()])
    vr_models = available([m.strip() for m in args.vr_models.split(",") if m.strip()])
    needs_non_vr = any(not item["is_vr"] for item in items)
    needs_vr = any(item["is_vr"] for item in items)
    if needs_non_vr and not non_vr_models:
        print("need at least one available non-VR model", file=sys.stderr)
        return 2
    if needs_vr and not vr_models:
        print("need at least one available VR model", file=sys.stderr)
        return 2

    models_to_prepare = (
        (non_vr_models if needs_non_vr else [])
        + (vr_models if needs_vr else [])
    )
    for name in dict.fromkeys(models_to_prepare):
        path = detection_model_weights_path(name)
        print(
            f"[engine] ensuring TensorRT engine for {name} "
            f"({path.name}, requested batch={args.batch}) ...",
            flush=True,
        )
        precompile_detection_engine(
            name,
            path,
            batch_size=args.batch,
            device=device,
            fp16=True,
        )

    _model_cache: dict[str, object] = {}

    def close_models() -> None:
        for model in _model_cache.values():
            model.close()

    def get_model(name: str):
        if name not in _model_cache:
            path = detection_model_weights_path(name)
            _model_cache[name] = build_detection_model(
                name, path, batch_size=args.batch, device=device,
                score_threshold=recommended_score_threshold(name), fp16=True,
            )
        return _model_cache[name]

    @torch.no_grad()
    def run_region(model, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        det = model(t, target_hw=(h, w))
        boxes = np.asarray(det.boxes_xyxy[0], dtype=np.float32).reshape(-1, 4)
        masks = det.masks[0]
        if masks.dtype != torch.bool:
            masks = masks > 0.5
        if masks.shape[0] > 0:
            union = masks.any(dim=0).to("cpu").numpy().astype(np.uint8)
            mask = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask = np.zeros((h, w), dtype=bool)
        return boxes, mask

    def infer(bgr, is_vr, model_name):
        model = get_model(model_name)
        h, w = bgr.shape[:2]
        if not is_vr:
            return run_region(model, bgr)
        half = w // 2
        bl, ml = run_region(model, bgr[:, :half])
        br, mr = run_region(model, bgr[:, half:])
        if br.shape[0]:
            br = br.copy()
            br[:, [0, 2]] += half
        boxes = np.concatenate([bl, br], axis=0) if (bl.shape[0] or br.shape[0]) else bl
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :half] = ml
        mask[:, half:half + mr.shape[1]] = mr
        return boxes, mask

    _result_cache: dict[tuple[int, str], tuple] = {}

    def result(idx, model_name):
        key = (idx, model_name)
        if key not in _result_cache:
            it = items[idx]
            _result_cache[key] = infer(it["bgr"], it["is_vr"], model_name)
        return _result_cache[key]

    def draw_hud(img, lines):
        font, scale, th, pad, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1, 10, 26
        w_max = max(cv2.getTextSize(ln, font, scale, th)[0][0] for ln in lines)
        pw = min(img.shape[1], w_max + 2 * pad)
        ph = pad + lh * len(lines)
        roi = img[0:ph, 0:pw]
        roi[:] = cv2.addWeighted(roi, 0.25, np.full_like(roi, (15, 15, 15)), 0.75, 0)
        y = pad + 17
        for ln in lines:
            cv2.putText(img, ln, (pad, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
            y += lh

    def render(idx, model_name, show_mask):
        it = items[idx]
        boxes, mask = result(idx, model_name)
        bgr = it["bgr"].copy()
        h, w = bgr.shape[:2]
        if show_mask and mask.any():
            overlay = bgr.copy()
            overlay[mask] = (0, 0, 255)
            bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)
        thick = max(2, round(w / 640))
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), thick)
        if it["is_vr"]:
            cv2.line(bgr, (w // 2, 0), (w // 2, h), (255, 255, 0), thick)
        scale = min(1.0, disp_max_w / w, disp_max_h / h)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cat = "VR per-eye" if it["is_vr"] else "non-VR"
        mlist = vr_models if it["is_vr"] else non_vr_models
        lines = [
            f"[{idx + 1}/{len(items)}]  {it['name'][:8]}  {w}x{h}  {cat}",
            f"model {mlist.index(model_name) + 1}/{len(mlist)}:  {model_name}   dets={len(boxes)}   mask={'ON' if show_mask else 'off'}",
            "A/D image    Q/E model    M mask    Esc quit",
        ]
        draw_hud(bgr, lines)
        return bgr

    if args.headless:
        for idx, it in enumerate(items):
            mlist = vr_models if it["is_vr"] else non_vr_models
            counts = {m: len(result(idx, m)[0]) for m in mlist}
            h, w = it["bgr"].shape[:2]
            cat = "VR    " if it["is_vr"] else "non-VR"
            print(f"[{idx + 1}/{len(items)}] {w}x{h} {cat} {it['name']}  " +
                  "  ".join(f"{m}={c}" for m, c in counts.items()))
        n_vr = sum(it["is_vr"] for it in items)
        print(f"[done] {len(items)} images  VR={n_vr}  non-VR={len(items) - n_vr}")
        close_models()
        return 0

    active = {"vr": 0, "non_vr": 0}

    def current_model(idx):
        it = items[idx]
        mlist = vr_models if it["is_vr"] else non_vr_models
        key = "vr" if it["is_vr"] else "non_vr"
        return mlist[active[key] % len(mlist)], mlist, key

    win = "detection sample viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    idx, show_mask = 0, True
    while True:
        model_name, mlist, cat_key = current_model(idx)
        cv2.imshow(win, render(idx, model_name, show_mask))
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        elif k in (ord("d"), 83):
            idx = (idx + 1) % len(items)
        elif k in (ord("a"), 81):
            idx = (idx - 1) % len(items)
        elif k == ord("e"):
            active[cat_key] = (active[cat_key] + 1) % len(mlist)
        elif k == ord("q"):
            active[cat_key] = (active[cat_key] - 1) % len(mlist)
        elif k == ord("m"):
            show_mask = not show_mask
        elif cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    close_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

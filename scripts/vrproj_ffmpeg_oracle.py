"""Conformance oracle: validate scripts/vr_projection against FFmpeg v360.

Encodes a linear coordinate ramp into a 16-bit half-equirect input, runs real
`v360` (input=hequirect, output=flat/fisheye) with explicit params, decodes the
output->input mapping FFmpeg actually used, and compares it to vr_projection's
map. Reports mean/max pixel error over the valid interior.

CPU only. Requires ffmpeg 8.x with v360. Run:
  ~/.virtualenvs/jasna-linux/bin/python scripts/vrproj_ffmpeg_oracle.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vr_projection import v360_map  # noqa: E402

IN = 1600            # square hequirect input side
OUT = 512            # output side
BORDER = 3           # px margin excluded near output edges
MAX16 = 65535.0

CASES = [
    dict(name="flat-center", out_proj="flat", h_fov=90, v_fov=45, yaw=0, pitch=0, roll=0),
    dict(name="flat-narrow", out_proj="flat", h_fov=40, v_fov=40, yaw=0, pitch=0, roll=0),
    dict(name="flat-yaw40",  out_proj="flat", h_fov=60, v_fov=60, yaw=40, pitch=0, roll=0),
    dict(name="flat-pitch-30", out_proj="flat", h_fov=60, v_fov=60, yaw=0, pitch=-30, roll=0),
    dict(name="flat-pitch+30", out_proj="flat", h_fov=60, v_fov=60, yaw=0, pitch=30, roll=0),
    dict(name="fisheye-180", out_proj="fisheye", h_fov=180, v_fov=180, yaw=0, pitch=0, roll=0),
]


def make_coord_input(path: Path) -> None:
    ys = (np.arange(IN) + 0.5) / IN
    xs = (np.arange(IN) + 0.5) / IN
    v, u = np.meshgrid(ys, xs, indexing="ij")
    r = np.round(u * MAX16).astype(np.uint16)
    g = np.round(v * MAX16).astype(np.uint16)
    b = np.zeros_like(r)
    bgr = np.stack((b, g, r), axis=-1)  # cv2 BGR
    cv2.imwrite(str(path), bgr)


def run_v360(src: Path, dst: Path, case: dict) -> None:
    vf = (
        f"v360=input=hequirect:output={case['out_proj']}"
        f":h_fov={case['h_fov']}:v_fov={case['v_fov']}"
        f":yaw={case['yaw']}:pitch={case['pitch']}:roll={case['roll']}"
        f":w={OUT}:h={OUT}:interp=linear:rorder=ypr"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-vf", vf, "-pix_fmt", "rgb48be", "-frames:v", "1", str(dst)],
        check=True,
    )


def evaluate(case: dict, work: Path) -> dict:
    src = work / "coord_in.png"
    dst = work / f"out_{case['name']}.png"
    if not src.exists():
        make_coord_input(src)
    run_v360(src, dst, case)

    out = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)  # BGR uint16
    if out is None or out.dtype != np.uint16:
        raise RuntimeError(f"failed to read 16-bit output for {case['name']} (dtype={None if out is None else out.dtype})")
    u_ff = out[..., 2].astype(np.float64) / MAX16
    v_ff = out[..., 1].astype(np.float64) / MAX16

    uv, valid = v360_map(
        case["out_proj"], OUT, OUT,
        h_fov=case["h_fov"], v_fov=case["v_fov"],
        yaw=case["yaw"], pitch=case["pitch"], roll=case["roll"],
    )
    u_me, v_me = uv[..., 0], uv[..., 1]

    # ffmpeg fills invalid/border with black (0,0) or clamps; keep interior valid.
    mask = valid.copy()
    mask[:BORDER, :] = mask[-BORDER:, :] = mask[:, :BORDER] = mask[:, -BORDER:] = False
    # exclude samples that land near the hequirect input edge, where ffmpeg
    # edge-clamps but our map extrapolates (a legitimate projection border).
    edge = 0.02
    mask &= (u_me > edge) & (u_me < 1 - edge) & (v_me > edge) & (v_me < 1 - edge)
    # drop pixels ffmpeg likely clamped: near-zero on both channels where we predict interior
    ff_black = (out[..., 2] == 0) & (out[..., 1] == 0)
    mask &= ~ff_black
    # the fisheye disc rim is a legitimate projection border (singular, steep);
    # judge the interior separately per the brief.
    if case["out_proj"] == "fisheye":
        cy, cx = np.meshgrid((2 * np.arange(OUT) + 1) / OUT - 1,
                             (2 * np.arange(OUT) + 1) / OUT - 1, indexing="ij")
        rad = np.hypot(cx, cy)
        # Interior disc (r<0.7) drives the mean/p99 gate. r in [0.7,0.9) is a
        # HIGH-JACOBIAN OUTER ANNULUS, not an invalid border: real bottom mosaics
        # may fall here. Directions still match FFmpeg to ~80 arcsec but the
        # hequirect Jacobian amplifies it. We record annular error and assert the
        # max stays <1px across r<0.9; reopen only if sampled ROIs reach r>=0.9
        # or show artifacts.
        we = np.hypot((u_ff - u_me) * IN, (v_ff - v_me) * IN)
        base = valid & ~ff_black
        wide = base & (rad < 0.9)
        ann = base & (rad >= 0.7) & (rad < 0.9)
        case["_wide_max"] = float(we[wide].max()) if wide.any() else float("nan")
        case["_ann_mean"] = float(we[ann].mean()) if ann.any() else float("nan")
        case["_ann_max"] = float(we[ann].max()) if ann.any() else float("nan")
        mask &= rad < 0.7

    ex = (u_ff - u_me) * IN
    ey = (v_ff - v_me) * IN
    err = np.hypot(ex, ey)[mask]
    if err.size == 0:
        return dict(name=case["name"], n=0, mean=float("nan"), p99=float("nan"), mx=float("nan"))
    return dict(
        name=case["name"], n=int(err.size),
        mean=float(err.mean()), p99=float(np.percentile(err, 99)), mx=float(err.max()),
        wide_max=case.get("_wide_max", float(err.max())),
        ann_mean=case.get("_ann_mean", float("nan")),
        ann_max=case.get("_ann_max", float("nan")),
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        rows = [evaluate(c, work) for c in CASES]
    print(f"{'case':<16}{'N':>10}{'mean_px':>10}{'p99_px':>10}{'max_px':>10}{'wideMax':>9}  verdict")
    ok = True
    for r in rows:
        good = r["mean"] < 0.25 and r["p99"] < 1.0 and r["wide_max"] < 1.0
        ok &= good
        print(f"{r['name']:<16}{r['n']:>10}{r['mean']:>10.3f}{r['p99']:>10.3f}{r['mx']:>10.3f}{r['wide_max']:>9.3f}  {'PASS' if good else 'FAIL'}")
    ann = next((r for r in rows if r["ann_mean"] == r["ann_mean"]), None)  # first non-nan
    if ann:
        print(f"\nfisheye high-Jacobian annulus r[0.7,0.9): mean={ann['ann_mean']:.3f}px max={ann['ann_max']:.3f}px "
              f"(recorded, not gated; reopen if ROIs reach r>=0.9)")
    print("\nGATE:", "PASS" if ok else "FAIL (fix conventions before continuing)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

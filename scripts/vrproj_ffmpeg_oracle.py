"""Conformance oracle: validate scripts/vr_projection against FFmpeg v360.

Encodes a linear coordinate ramp into a 16-bit half-equirect input, runs real
`v360` (input=hequirect, output=flat/fisheye) with explicit params, decodes the
output->input sampling positions FFmpeg actually used, and compares them to
vr_projection's map using FFmpeg's ``uv * (size - 1)`` source-pixel convention.
Reports mean/max pixel error over the valid interior.

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

IN = 1600            # square hequirect input side
OUT = 512            # output side
BORDER = 3           # px margin excluded near output edges
MAX16 = 65535.0
MAX_MEAN_ERROR = 0.10
MAX_P99_ERROR = 0.25
MAX_ERROR = 0.30

CASES = [
    dict(name="flat-center", out_proj="flat", h_fov=90, v_fov=45, yaw=0, pitch=0, roll=0),
    dict(name="flat-narrow", out_proj="flat", h_fov=40, v_fov=40, yaw=0, pitch=0, roll=0),
    dict(name="flat-yaw40",  out_proj="flat", h_fov=60, v_fov=60, yaw=40, pitch=0, roll=0),
    dict(name="flat-pitch-30", out_proj="flat", h_fov=60, v_fov=60, yaw=0, pitch=-30, roll=0),
    dict(name="flat-pitch+30", out_proj="flat", h_fov=60, v_fov=60, yaw=0, pitch=30, roll=0),
    dict(name="fisheye-180", out_proj="fisheye", h_fov=180, v_fov=180, yaw=0, pitch=0, roll=0),
]


def load_v360_map():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.vr_projection import v360_map

    return v360_map


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


def evaluate(case: dict, work: Path, v360_map) -> dict:
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

    geometry_uv, valid = v360_map(
        case["out_proj"], OUT, OUT,
        h_fov=case["h_fov"], v_fov=case["v_fov"],
        yaw=case["yaw"], pitch=case["pitch"], roll=case["roll"],
    )
    expected_uv = (geometry_uv * (IN - 1) + 0.5) / IN
    u_me, v_me = expected_uv[..., 0], expected_uv[..., 1]

    # ffmpeg fills invalid/border with black (0,0) or clamps; keep interior valid.
    mask = valid.copy()
    mask[:BORDER, :] = mask[-BORDER:, :] = mask[:, :BORDER] = mask[:, -BORDER:] = False
    # exclude samples that land near the hequirect input edge, where ffmpeg
    # edge-clamps but our map extrapolates (a legitimate projection border).
    edge = 0.02
    mask &= (
        (geometry_uv[..., 0] > edge)
        & (geometry_uv[..., 0] < 1 - edge)
        & (geometry_uv[..., 1] > edge)
        & (geometry_uv[..., 1] < 1 - edge)
    )
    # drop pixels ffmpeg likely clamped: near-zero on both channels where we predict interior
    ff_black = (out[..., 2] == 0) & (out[..., 1] == 0)
    mask &= ~ff_black
    if case["out_proj"] == "fisheye":
        cy, cx = np.meshgrid((2 * np.arange(OUT) + 1) / OUT - 1,
                             (2 * np.arange(OUT) + 1) / OUT - 1, indexing="ij")
        mask &= np.hypot(cx, cy) < 0.9

    ex = (u_ff - u_me) * IN
    ey = (v_ff - v_me) * IN
    err = np.hypot(ex, ey)[mask]
    if err.size == 0:
        return dict(name=case["name"], n=0, mean=float("nan"), p99=float("nan"), mx=float("nan"))
    return dict(
        name=case["name"], n=int(err.size),
        mean=float(err.mean()), p99=float(np.percentile(err, 99)), mx=float(err.max()),
    )


def main() -> int:
    v360_map = load_v360_map()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        rows = [evaluate(c, work, v360_map) for c in CASES]
    print(f"{'case':<16}{'N':>10}{'mean_px':>10}{'p99_px':>10}{'max_px':>10}  verdict")
    ok = True
    for r in rows:
        good = (
            r["mean"] < MAX_MEAN_ERROR
            and r["p99"] < MAX_P99_ERROR
            and r["mx"] < MAX_ERROR
        )
        ok &= good
        print(f"{r['name']:<16}{r['n']:>10}{r['mean']:>10.3f}{r['p99']:>10.3f}{r['mx']:>10.3f}  {'PASS' if good else 'FAIL'}")
    print("\nGATE:", "PASS" if ok else "FAIL (fix conventions before continuing)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

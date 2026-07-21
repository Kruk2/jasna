"""Benchmark full jasna e2e restoration across decode backends.

Runs the jasna CLI once per (clip, backend) in a subprocess with
``jasna.media.video_decoder.DECODE_BACKEND`` patched, and reports wall time +
throughput. Fixed settings: ``--max-clip-size 180 --temporal-overlap 15
--secondary-restoration none``.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_decode_backends.py
    ... [--clips CLIP ...] [--backends vali pyav-hw pyav-sw] [--repeats N]
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench_memory import MemorySampler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_DIR = REPO_ROOT / "assets" / "benchmark"
BACKENDS = ("vali", "pyav-hw", "pyav-sw")

RUNNER_SHIM = """
import sys
import jasna.media.video_decoder as video_decoder
video_decoder.DECODE_BACKEND = sys.argv[1]
sys.argv = ["jasna"] + sys.argv[2:]
from jasna.main import main
main()
"""

FALLBACK_MARKERS = ("falling back", "software decoding")


def probe_frames(clip: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "json", str(clip),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["nb_read_packets"])


def run_once(clip: Path, backend: str, workdir: Path) -> tuple[float, str, dict[str, float]]:
    output = workdir / f"{clip.stem}_{backend}_out.mp4"
    cmd = [
        sys.executable, "-c", RUNNER_SHIM, backend,
        "--input", str(clip),
        "--output", str(output),
        "--max-clip-size", "180",
        "--temporal-overlap", "15",
        "--secondary-restoration", "none",
        "--log-level", "warning",
        "--no-progress",
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = MemorySampler(proc.pid)
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    memory = sampler.stop()
    if proc.returncode != 0:
        tail = "\n".join((stdout + stderr).strip().splitlines()[-5:])
        print(f"{clip.name} [{backend}] rc={proc.returncode}:\n{tail}", file=sys.stderr)
        return float("nan"), "FAILED", memory
    output.unlink(missing_ok=True)
    log = stdout + stderr
    flags = ""
    if backend != "pyav-sw" and any(marker in log.lower() for marker in FALLBACK_MARKERS):
        flags = "FALLBACK"
    return elapsed, flags, memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", type=Path, default=None)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["vali"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    clips = args.clips or sorted(DEFAULT_CLIP_DIR.glob("*_bench_*.mp4"))
    if not clips:
        raise SystemExit(f"no clips found in {DEFAULT_CLIP_DIR}")

    workdir = Path(tempfile.mkdtemp(prefix="jasna_decode_bench_"))
    rows = []
    if not args.no_warmup:
        print(f"warmup: {clips[0].name} [{args.backends[0]}] (engine compile + caches)")
        run_once(clips[0], args.backends[0], workdir)

    for clip in clips:
        frames = probe_frames(clip)
        for backend in args.backends:
            times = []
            flags = ""
            memory = {}
            for _ in range(args.repeats):
                elapsed, run_flags, run_memory = run_once(clip, backend, workdir)
                times.append(elapsed)
                flags = flags or run_flags
                memory = memory or run_memory
                if run_flags == "FAILED":
                    break
            wall = statistics.median(times)
            fps = 0.0 if flags == "FAILED" else frames / wall
            rows.append((
                clip.name, backend, frames, wall, fps,
                memory["ram_med_mb"], memory["ram_peak_mb"],
                memory["vram_med_mb"], memory["vram_peak_mb"], flags,
            ))
            print(
                f"{clip.name} [{backend}]: {wall:.1f}s {fps:.1f}fps "
                f"ram {memory['ram_med_mb']:.0f}/{memory['ram_peak_mb']:.0f}MB "
                f"vram {memory['vram_med_mb']:.0f}/{memory['vram_peak_mb']:.0f}MB {flags}",
                flush=True,
            )

    print(flush=True)
    print("| clip | backend | frames | wall s | fps | ram med/peak MB | vram med/peak MB | flags |")
    print("|---|---|---|---|---|---|---|---|")
    for name, backend, frames, wall, fps, ram_med, ram_peak, vram_med, vram_peak, flags in rows:
        print(
            f"| {name} | {backend} | {frames} | {wall:.1f} | {fps:.1f} "
            f"| {ram_med:.0f}/{ram_peak:.0f} | {vram_med:.0f}/{vram_peak:.0f} | {flags} |"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "clip", "backend", "frames", "wall_s", "fps",
                "ram_med_mb", "ram_peak_mb", "vram_med_mb", "vram_peak_mb", "flags",
            ])
            writer.writerows(rows)
        print(f"csv written to {args.csv}", flush=True)


if __name__ == "__main__":
    main()

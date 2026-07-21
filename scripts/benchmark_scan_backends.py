"""Benchmark the GUI every-frame mosaic scan across decode backends.

Runs ``MosaicScanWorker``'s scan (detection only, ``stride_seconds=0.0`` =
every frame, results collected into one preallocated GPU tensor) once per
(clip, backend) in a subprocess with
``jasna.media.video_decoder.DECODE_BACKEND`` patched. Detector build and
engine compilation happen before the timed section; the reported wall time
covers decode + detection + the final CPU sync only.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_scan_backends.py
    ... [--clips CLIP ...] [--backends vali pyav-hw pyav-sw] [--repeats N]
"""

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

from bench_memory import MemorySampler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_DIR = REPO_ROOT / "assets" / "benchmark"
BACKENDS = ("vali", "pyav-hw", "pyav-sw")

RUNNER_SHIM = """
import queue
import sys
import time

import jasna.media.video_decoder as video_decoder

video_decoder.DECODE_BACKEND = sys.argv[1]
path = sys.argv[2]

from jasna.media import get_video_meta_data
from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import MosaicScanWorker, ScanCompleted

metadata = get_video_meta_data(path)
worker = MosaicScanWorker(path, metadata, AppSettings(), stride_seconds=0.0)
detector = worker._build_detector()
start = time.perf_counter()
worker._scan(detector)
elapsed = time.perf_counter() - start

completed = None
while True:
    try:
        event = worker.events.get_nowait()
    except queue.Empty:
        break
    if isinstance(event, ScanCompleted):
        completed = event
if completed is None:
    raise SystemExit("scan did not complete")
result = completed.result
print(f"RESULT wall={elapsed:.3f} samples={len(result.times)} "
      f"completed_until={result.completed_until:.2f}", flush=True)
"""


def run_once(clip: Path, backend: str) -> tuple[float, int, str, dict[str, float]]:
    cmd = [sys.executable, "-c", RUNNER_SHIM, backend, str(clip)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = MemorySampler(proc.pid)
    stdout, stderr = proc.communicate()
    memory = sampler.stop()
    if proc.returncode != 0:
        tail = "\n".join((stdout + stderr).strip().splitlines()[-5:])
        print(f"{clip.name} [{backend}] rc={proc.returncode}:\n{tail}", file=sys.stderr)
        return float("nan"), 0, "FAILED", memory
    result_line = next(
        line for line in stdout.splitlines() if line.startswith("RESULT ")
    )
    fields = dict(part.split("=") for part in result_line.split()[1:])
    log = (stdout + stderr).lower()
    flags = ""
    if backend != "pyav-sw" and ("falling back" in log or "software decoding" in log):
        flags = "FALLBACK"
    return float(fields["wall"]), int(fields["samples"]), flags, memory


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

    rows = []
    if not args.no_warmup:
        print(f"warmup: {clips[0].name} [{args.backends[0]}]", flush=True)
        run_once(clips[0], args.backends[0])

    for clip in clips:
        for backend in args.backends:
            times = []
            samples = 0
            flags = ""
            memory = {}
            for _ in range(args.repeats):
                elapsed, run_samples, run_flags, run_memory = run_once(clip, backend)
                times.append(elapsed)
                samples = samples or run_samples
                flags = flags or run_flags
                memory = memory or run_memory
                if run_flags == "FAILED":
                    break
            wall = statistics.median(times)
            fps = 0.0 if flags == "FAILED" else samples / wall
            rows.append((
                clip.name, backend, samples, wall, fps,
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
    for name, backend, samples, wall, fps, ram_med, ram_peak, vram_med, vram_peak, flags in rows:
        print(
            f"| {name} | {backend} | {samples} | {wall:.1f} | {fps:.1f} "
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

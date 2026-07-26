"""Benchmark the Lada Flatpak on Jasna's canonical benchmark assets.

The Flatpak app and all of its subprocesses share an app-specific cgroup.
Memory sampling sums RSS and process VRAM across that cgroup.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_lada_flatpak.py \
        --workdir BENCHMARK_WORKDIR/lada-flatpak \
        --csv benchmarks/lada-flatpak.csv
"""

import argparse
import csv
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from benchmark_releases import probe_frames

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_DIR = REPO_ROOT / "assets" / "benchmark"
DEFAULT_APP_ID = "io.github.ladaapp.lada"
FAILURE_FLAGS = frozenset({"FAILED", "TIMEOUT"})


def canonical_clips(clip_dir: Path) -> list[Path]:
    return sorted(clip_dir.glob("SONE-610_bench_*.mp4"))


def flatpak_cgroup_pids(app_id: str, proc_root: Path = Path("/proc")) -> set[int]:
    marker = f"app-flatpak-{app_id}-"
    pids = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cgroup = (entry / "cgroup").read_text()
        except OSError:
            continue
        if marker in cgroup:
            pids.add(int(entry.name))
    return pids


class FlatpakMemorySampler:
    def __init__(self, app_id: str, interval_seconds: float = 0.5) -> None:
        self.app_id = app_id
        self.interval_seconds = interval_seconds
        self._ram_mb: list[float] = []
        self._vram_mb: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample_ram(self, pids: set[int]) -> None:
        total_kib = 0
        for pid in pids:
            try:
                lines = Path(f"/proc/{pid}/status").read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        if total_kib:
            self._ram_mb.append(total_kib / 1024)

    def _sample_vram(self, pids: set[int]) -> None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        total_mb = 0.0
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if (
                len(parts) == 2
                and parts[0].isdigit()
                and int(parts[0]) in pids
                and parts[1].isdigit()
            ):
                total_mb += float(parts[1])
        if total_mb:
            self._vram_mb.append(total_mb)

    def _run(self) -> None:
        while not self._stop.is_set():
            pids = flatpak_cgroup_pids(self.app_id)
            self._sample_ram(pids)
            self._sample_vram(pids)
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()

        def med_peak(samples: list[float]) -> tuple[float, float]:
            if not samples:
                return 0.0, 0.0
            return statistics.median(samples), max(samples)

        ram_med, ram_peak = med_peak(self._ram_mb)
        vram_med, vram_peak = med_peak(self._vram_mb)
        return {
            "ram_med_mb": ram_med,
            "ram_peak_mb": ram_peak,
            "vram_med_mb": vram_med,
            "vram_peak_mb": vram_peak,
        }


def build_command(
    app_id: str,
    clip: Path,
    output: Path,
    workdir: Path,
    *,
    device: str,
    detection_model: str,
    encoding_preset: str,
    max_clip_length: int,
    extra_args: tuple[str, ...],
) -> list[str]:
    return [
        "flatpak",
        "run",
        f"--filesystem={clip.parent}:ro",
        f"--filesystem={workdir}",
        "--command=lada-cli",
        app_id,
        "--input",
        str(clip),
        "--output",
        str(output),
        "--temporary-directory",
        str(workdir),
        "--device",
        device,
        "--fp16",
        "--max-clip-length",
        str(max_clip_length),
        "--mosaic-detection-model",
        detection_model,
        "--encoding-preset",
        encoding_preset,
        *extra_args,
    ]


def remove_output_artifacts(output: Path, workdir: Path) -> None:
    output.unlink(missing_ok=True)
    (workdir / f"{output.stem}.tmp{output.suffix}").unlink(missing_ok=True)


def stop_process(proc: subprocess.Popen[str], *, force: bool) -> None:
    if sys.platform == "win32":
        (proc.kill if force else proc.terminate)()
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return


def run_once(
    app_id: str,
    clip: Path,
    output: Path,
    workdir: Path,
    log_path: Path,
    *,
    device: str,
    detection_model: str,
    encoding_preset: str,
    max_clip_length: int,
    extra_args: tuple[str, ...],
    timeout_seconds: float | None,
) -> tuple[float, str, dict[str, float]]:
    command = build_command(
        app_id,
        clip,
        output,
        workdir,
        device=device,
        detection_model=detection_model,
        encoding_preset=encoding_preset,
        max_clip_length=max_clip_length,
        extra_args=extra_args,
    )
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    sampler = FlatpakMemorySampler(app_id)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_process(proc, force=False)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stop_process(proc, force=True)
            stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    memory = sampler.stop()
    log = stdout + stderr
    if timed_out:
        assert timeout_seconds is not None
        log += f"\nBenchmark timed out after {timeout_seconds:g} seconds.\n"
    log_path.write_text(log)
    output_missing = not output.is_file() or output.stat().st_size == 0
    remove_output_artifacts(output, workdir)
    if timed_out:
        return elapsed, "TIMEOUT", memory
    if proc.returncode != 0 or output_missing:
        tail = "\n".join(log.strip().splitlines()[-8:])
        print(
            f"Lada {clip.name} rc={proc.returncode} "
            f"output_missing={output_missing}:\n{tail}",
            file=sys.stderr,
        )
        return elapsed, "FAILED", memory
    return elapsed, "", memory


def write_csv(path: Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "version",
                "clip",
                "frames",
                "max_clip_length",
                "wall_s",
                "fps",
                "ram_med_mb",
                "ram_peak_mb",
                "vram_med_mb",
                "vram_peak_mb",
                "flags",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", default=DEFAULT_APP_ID)
    parser.add_argument("--label", default="Lada Flatpak")
    parser.add_argument("--clips", nargs="+", type=Path, default=None)
    parser.add_argument("--warmup-clip", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detection-model", default="v2")
    parser.add_argument("--encoding-preset", default="hevc-nvidia-gpu-hq")
    parser.add_argument("--max-clip-length", type=int, default=180)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_clip_length < 1:
        parser.error("--max-clip-length must be at least 1")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than 0")

    installed = subprocess.run(
        ["flatpak", "info", "--user", args.app_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if installed.returncode != 0:
        parser.error(f"user Flatpak is not installed: {args.app_id}")

    clips = [
        path.resolve()
        for path in (args.clips or canonical_clips(DEFAULT_CLIP_DIR))
    ]
    if not clips:
        parser.error(f"no clips found in {DEFAULT_CLIP_DIR}")
    missing_clips = [clip for clip in clips if not clip.is_file()]
    if missing_clips:
        parser.error(f"clips not found: {', '.join(map(str, missing_clips))}")

    if args.workdir:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="lada_flatpak_bench_"))
    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    extra_args = tuple(args.extra_arg)
    if not args.no_warmup:
        warmup_clip = (args.warmup_clip or clips[0]).resolve()
        print(f"warmup: {args.label} {warmup_clip.name}", flush=True)
        elapsed, flags, _ = run_once(
            args.app_id,
            warmup_clip,
            workdir / "warmup_lada_flatpak.mp4",
            workdir,
            log_dir / "warmup_lada_flatpak.log",
            device=args.device,
            detection_model=args.detection_model,
            encoding_preset=args.encoding_preset,
            max_clip_length=args.max_clip_length,
            extra_args=extra_args,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"warmup: {args.label} {elapsed:.1f}s {flags}", flush=True)
        if flags in FAILURE_FLAGS:
            for clip in clips:
                rows.append(
                    (
                        args.label,
                        clip.name,
                        probe_frames(clip),
                        args.max_clip_length,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        f"WARMUP_{flags}",
                    )
                )
            write_csv(args.csv, rows)
            return

    for clip in clips:
        frames = probe_frames(clip)
        times = []
        memories = []
        flags = ""
        for repeat in range(1, args.repeats + 1):
            elapsed, flags, memory = run_once(
                args.app_id,
                clip,
                workdir / f"{clip.stem}_lada_flatpak_out.mp4",
                workdir,
                log_dir / f"{clip.stem}_{repeat}.log",
                device=args.device,
                detection_model=args.detection_model,
                encoding_preset=args.encoding_preset,
                max_clip_length=args.max_clip_length,
                extra_args=extra_args,
                timeout_seconds=args.timeout_seconds,
            )
            times.append(elapsed)
            memories.append(memory)
            if flags in FAILURE_FLAGS:
                break
        wall = statistics.median(times)
        fps = 0.0 if flags in FAILURE_FLAGS else frames / wall
        ram_med = statistics.median(item["ram_med_mb"] for item in memories)
        ram_peak = max(item["ram_peak_mb"] for item in memories)
        vram_med = statistics.median(item["vram_med_mb"] for item in memories)
        vram_peak = max(item["vram_peak_mb"] for item in memories)
        rows.append(
            (
                args.label,
                clip.name,
                frames,
                args.max_clip_length,
                wall,
                fps,
                ram_med,
                ram_peak,
                vram_med,
                vram_peak,
                flags,
            )
        )
        print(
            f"{args.label} {clip.name}: {wall:.1f}s {fps:.1f}fps "
            f"ram {ram_med:.0f}/{ram_peak:.0f}MB "
            f"vram {vram_med:.0f}/{vram_peak:.0f}MB {flags}",
            flush=True,
        )
        write_csv(args.csv, rows)
        print(f"checkpoint written to {args.csv}", flush=True)


if __name__ == "__main__":
    main()

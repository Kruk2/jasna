"""Benchmark end-to-end restoration across Jasna releases and source trees.

Each target runs in a fresh subprocess with the canonical release benchmark
settings: clip size 180, temporal overlap 15, and no secondary restoration.
Frozen releases use their bundled executable. Source targets force the VALI
decode backend.

Usage:
    ~/.virtualenvs/jasna-linux/bin/python scripts/benchmark_releases.py \
        --release v0.7.2=/path/to/v0.7.2/jasna \
        --source HEAD=/path/to/jasna \
        --workdir BENCHMARK_WORKDIR/work \
        --csv benchmarks/releases.csv
"""

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from bench_memory import MemorySampler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_DIR = REPO_ROOT / "assets" / "benchmark"
FALLBACK_MARKERS = ("falling back", "software decoding")
FAILURE_FLAGS = frozenset({"FAILED", "TIMEOUT"})

SOURCE_RUNNER_SHIM = """
import sys
import jasna.media.video_decoder as video_decoder
video_decoder.DECODE_BACKEND = "vali"
sys.argv = ["jasna"] + sys.argv[1:]
from jasna.main import main
main()
"""


@dataclass(frozen=True)
class Target:
    label: str
    command_prefix: tuple[str, ...]
    cwd: Path


@dataclass
class Measurements:
    times: list[float] = field(default_factory=list)
    memories: list[dict[str, float]] = field(default_factory=list)
    flags: str = ""


def parse_target_spec(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("target must use LABEL=PATH")
    return label, Path(raw_path).expanduser().resolve()


def release_target(spec: tuple[str, Path]) -> Target:
    label, executable = spec
    if not executable.is_file():
        raise argparse.ArgumentTypeError(f"release executable not found: {executable}")
    return Target(label, (str(executable),), executable.parent)


def source_target(spec: tuple[str, Path]) -> Target:
    label, source_dir = spec
    if not (source_dir / "jasna" / "main.py").is_file():
        raise argparse.ArgumentTypeError(f"Jasna source tree not found: {source_dir}")
    return Target(
        label,
        (sys.executable, "-c", SOURCE_RUNNER_SHIM),
        source_dir,
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def targets_for_repeat(targets: list[Target], repeat: int) -> list[Target]:
    return targets if repeat % 2 else list(reversed(targets))


def probe_frames(clip: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["nb_read_packets"])


def target_supports(target: Target, option: str) -> bool:
    proc = subprocess.run(
        [*target.command_prefix, "--help"],
        cwd=target.cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return option in proc.stdout + proc.stderr


def remove_output_artifacts(output: Path) -> None:
    """Remove the requested output and known partial encoder artifacts."""
    artifacts = (
        output,
        output.with_suffix(".hevc"),
        output.with_name(f"{output.stem}_temp_video{output.suffix}"),
        output.with_name(f"{output.stem}_temp_video.txt"),
    )
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)


def combine_flags(current: str, new: str) -> str:
    if new in FAILURE_FLAGS:
        return new
    if not current:
        return new
    if not new or new == current:
        return current
    return f"{current}|{new}"


def stop_process(proc: subprocess.Popen[str], *, force: bool) -> None:
    if sys.platform == "win32":
        (proc.kill if force else proc.terminate)()
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def build_command(
    target: Target,
    clip: Path,
    output: Path,
    *,
    no_progress: bool,
    disable_ffmpeg_check: bool,
    max_clip_size: int = 180,
    temporal_overlap: int = 15,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    command = [
        *target.command_prefix,
        "--input",
        str(clip),
        "--output",
        str(output),
        "--max-clip-size",
        str(max_clip_size),
        "--temporal-overlap",
        str(temporal_overlap),
        "--secondary-restoration",
        "none",
        "--log-level",
        "warning",
        *extra_args,
    ]
    if disable_ffmpeg_check:
        command.append("--disable-ffmpeg-check")
    if no_progress:
        command.append("--no-progress")
    return command


def run_once(
    target: Target,
    clip: Path,
    output: Path,
    log_path: Path,
    *,
    no_progress: bool,
    disable_ffmpeg_check: bool,
    max_clip_size: int,
    temporal_overlap: int,
    extra_args: tuple[str, ...],
    timeout_seconds: float | None,
) -> tuple[float, str, dict[str, float]]:
    command = build_command(
        target,
        clip,
        output,
        no_progress=no_progress,
        disable_ffmpeg_check=disable_ffmpeg_check,
        max_clip_size=max_clip_size,
        temporal_overlap=temporal_overlap,
        extra_args=extra_args,
    )
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=target.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    sampler = MemorySampler(proc.pid)
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
    if timed_out:
        remove_output_artifacts(output)
        return elapsed, "TIMEOUT", memory
    output_missing = not output.is_file() or output.stat().st_size == 0
    if proc.returncode != 0 or output_missing:
        tail = "\n".join(log.strip().splitlines()[-8:])
        print(
            f"{target.label} {clip.name} rc={proc.returncode} "
            f"output_missing={output_missing}:\n{tail}",
            file=sys.stderr,
        )
        remove_output_artifacts(output)
        return elapsed, "FAILED", memory
    remove_output_artifacts(output)
    flags = ""
    if any(marker in log.lower() for marker in FALLBACK_MARKERS):
        flags = "FALLBACK"
    return elapsed, flags, memory


def write_csv(path: Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "version",
                "clip",
                "frames",
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
    parser.add_argument(
        "--release",
        action="append",
        type=parse_target_spec,
        default=[],
        metavar="LABEL=EXECUTABLE",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_target_spec,
        default=[],
        metavar="LABEL=SOURCE_DIR",
    )
    parser.add_argument("--clips", nargs="+", type=Path, default=None)
    parser.add_argument("--warmup-clip", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--interleave-targets",
        action="store_true",
        help="alternate target order within each clip to reduce run-order drift",
    )
    parser.add_argument("--max-clip-size", type=int, default=180)
    parser.add_argument("--temporal-overlap", type=int, default=15)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_clip_size < 1:
        parser.error("--max-clip-size must be at least 1")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than 0")

    targets = [
        *(release_target(spec) for spec in args.release),
        *(source_target(spec) for spec in args.source),
    ]
    if not targets:
        parser.error("at least one --release or --source target is required")

    clips = [
        path.resolve()
        for path in (args.clips or sorted(DEFAULT_CLIP_DIR.glob("*_bench_*.mp4")))
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
        workdir = Path(tempfile.mkdtemp(prefix="jasna_release_bench_"))
    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[object, ...]] = []
    extra_args = tuple(args.extra_arg)
    target_options = {
        target: (
            target_supports(target, "--no-progress"),
            target_supports(target, "--disable-ffmpeg-check"),
        )
        for target in targets
    }

    def warm_target(target: Target) -> bool:
        if args.no_warmup:
            return True
        warmup_clip = (args.warmup_clip or clips[0]).resolve()
        no_progress, disable_ffmpeg_check = target_options[target]
        print(f"warmup: {target.label} {warmup_clip.name}", flush=True)
        elapsed, flags, _ = run_once(
            target,
            warmup_clip,
            workdir / f"warmup_{safe_name(target.label)}.mp4",
            log_dir / f"warmup_{safe_name(target.label)}.log",
            no_progress=no_progress,
            disable_ffmpeg_check=disable_ffmpeg_check,
            max_clip_size=args.max_clip_size,
            temporal_overlap=args.temporal_overlap,
            extra_args=extra_args,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"warmup: {target.label} {elapsed:.1f}s {flags}", flush=True)
        if flags not in FAILURE_FLAGS:
            return True
        for clip in clips:
            rows.append(
                (
                    target.label,
                    clip.name,
                    probe_frames(clip),
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
        return False

    def run_sample(
        target: Target, clip: Path, repeat: int
    ) -> tuple[float, str, dict[str, float]]:
        no_progress, disable_ffmpeg_check = target_options[target]
        return run_once(
            target,
            clip,
            workdir / f"{safe_name(target.label)}_{clip.stem}_out.mp4",
            log_dir / f"{safe_name(target.label)}_{clip.stem}_{repeat}.log",
            no_progress=no_progress,
            disable_ffmpeg_check=disable_ffmpeg_check,
            max_clip_size=args.max_clip_size,
            temporal_overlap=args.temporal_overlap,
            extra_args=extra_args,
            timeout_seconds=args.timeout_seconds,
        )

    def record_result(
        target: Target,
        clip: Path,
        frames: int,
        times: list[float],
        memories: list[dict[str, float]],
        flags: str,
    ) -> None:
        wall = statistics.median(times)
        fps = 0.0 if flags in FAILURE_FLAGS else frames / wall
        ram_med = statistics.median(item["ram_med_mb"] for item in memories)
        ram_peak = max(item["ram_peak_mb"] for item in memories)
        vram_med = statistics.median(item["vram_med_mb"] for item in memories)
        vram_peak = max(item["vram_peak_mb"] for item in memories)
        rows.append(
            (
                target.label,
                clip.name,
                frames,
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
            f"{target.label} {clip.name}: {wall:.1f}s {fps:.1f}fps "
            f"ram {ram_med:.0f}/{ram_peak:.0f}MB "
            f"vram {vram_med:.0f}/{vram_peak:.0f}MB {flags}",
            flush=True,
        )
        write_csv(args.csv, rows)
        print(f"checkpoint written to {args.csv}", flush=True)

    if args.interleave_targets:
        active_targets = [target for target in targets if warm_target(target)]
        for clip in clips:
            frames = probe_frames(clip)
            measurements = {target: Measurements() for target in active_targets}
            for repeat in range(1, args.repeats + 1):
                for target in targets_for_repeat(active_targets, repeat):
                    measurement = measurements[target]
                    if measurement.flags in FAILURE_FLAGS:
                        continue
                    elapsed, run_flags, memory = run_sample(target, clip, repeat)
                    measurement.times.append(elapsed)
                    measurement.memories.append(memory)
                    measurement.flags = combine_flags(measurement.flags, run_flags)
            for target in active_targets:
                measurement = measurements[target]
                record_result(
                    target,
                    clip,
                    frames,
                    measurement.times,
                    measurement.memories,
                    measurement.flags,
                )
    else:
        for target in targets:
            if not warm_target(target):
                continue
            for clip in clips:
                frames = probe_frames(clip)
                times = []
                memories = []
                flags = ""
                for repeat in range(1, args.repeats + 1):
                    elapsed, run_flags, memory = run_sample(target, clip, repeat)
                    times.append(elapsed)
                    memories.append(memory)
                    flags = combine_flags(flags, run_flags)
                    if run_flags in FAILURE_FLAGS:
                        break
                record_result(target, clip, frames, times, memories, flags)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path
from argparse import Namespace

import torch

from jasna.benchmark.basicvsrpp_restoration import benchmark_basicvsrpp_restoration
from jasna.benchmark.lada_yolo_detection_speed import benchmark_lada_yolo_detection_speed
from jasna.benchmark.rfdetr_detection_speed import benchmark_rfdetr_detection_speed
from jasna.os_utils import check_nvidia_gpu, check_required_executables

BENCHMARK_VIDEO_DEFAULTS: list[Path] = [
    Path("assets/test_clip1_1080p.mp4"),
    Path("assets/test_clip1_2160p.mp4"),
]

BENCHMARKS = [
    benchmark_basicvsrpp_restoration,
    benchmark_rfdetr_detection_speed,
    benchmark_lada_yolo_detection_speed,
]


def run_benchmarks(
    *,
    device: torch.device,
    batch_size: int = 4,
    fp16: bool = True,
    benchmark_videos: list[Path],
    detection_score_threshold: float = 0.2,
    restoration_model_path: Path | None = None,
    compile_basicvsrpp: bool = True,
) -> None:
    results: dict[str, dict[str, tuple[float, float]]] = {}
    videos_to_run = [p for p in benchmark_videos if p.resolve().exists()]

    with torch.cuda.device(device):
        for benchmark_fn in BENCHMARKS:
            table_rows = benchmark_fn(
                device=device,
                batch_size=batch_size,
                fp16=fp16,
                benchmark_videos=videos_to_run,
                detection_score_threshold=detection_score_threshold,
                restoration_model_path=restoration_model_path,
                compile_basicvsrpp=compile_basicvsrpp,
            )
            if table_rows is not None:
                results[benchmark_fn.__name__.replace("benchmark_", "")] = table_rows

    _print_results_table(results, videos_to_run)


def _print_results_table(
    results: dict[str, dict[str, tuple[float, float]]],
    videos: list[Path],
) -> None:
    col_fnames = [p.name for p in videos]
    model_names = list(results.keys())
    model_width = max(len(m) for m in model_names) if model_names else 10
    model_width = max(model_width, 6)
    cell_content = "XXX.X fps (X.XXs)"
    cell_width = max(max(len(f) for f in col_fnames) if col_fnames else 0, len(cell_content))

    header = "model".ljust(model_width)
    for fname in col_fnames:
        header += f"  {fname.ljust(cell_width)}"
    print(header)

    sep = "-" * model_width
    for _ in col_fnames:
        sep += "  " + "-" * cell_width
    print(sep)

    for model in model_names:
        row = model.ljust(model_width)
        for fname in col_fnames:
            if fname in results[model]:
                median_s, fps = results[model][fname]
                cell = f"{fps:.1f} fps ({median_s:.2f}s)"
                row += f"  {cell.ljust(cell_width)}"
            else:
                row += f"  {'-'.ljust(cell_width)}"
        print(row)


def run_benchmark_cli(args: Namespace) -> None:
    check_required_executables(disable_ffmpeg_check=args.disable_ffmpeg_check)
    gpu_ok, gpu_result = check_nvidia_gpu()
    if not gpu_ok:
        if gpu_result == "no_cuda":
            print("Error: No CUDA device. An NVIDIA GPU with compute capability 7.5+ is required.")
        else:
            _, major, minor = gpu_result
            print(f"Error: Compute capability 7.5+ required (GPU: {major}.{minor}).")
        sys.exit(1)
    benchmark_videos = (
        [Path(p) for p in args.benchmark_video] if args.benchmark_video else BENCHMARK_VIDEO_DEFAULTS
    )
    run_benchmarks(
        device=torch.device(str(args.device)),
        batch_size=int(args.batch_size),
        fp16=bool(args.fp16),
        benchmark_videos=benchmark_videos,
        detection_score_threshold=float(args.detection_score_threshold),
        restoration_model_path=Path(args.restoration_model_path),
        compile_basicvsrpp=bool(args.compile_basicvsrpp),
    )

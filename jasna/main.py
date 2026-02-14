import argparse
import logging
import sys
from pathlib import Path

from jasna import __version__
from jasna.os_utils import (
    check_nvidia_gpu,
    check_required_executables,
    warn_if_windows_hardware_accelerated_gpu_scheduling_enabled,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks instead of processing video")
    parser.add_argument("--input", required=False, type=str, default=None, help="Path to input video")
    parser.add_argument("--output", required=False, type=str, default=None, help="Path to output video")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--disable-ffmpeg-check",
        action="store_true",
        help="Skip checking for ffmpeg/ffprobe in PATH and their version.",
    )

    restoration = parser.add_argument_group("Restoration")
    restoration.add_argument(
        "--restoration-model-name",
        type=str,
        default="basicvsrpp",
        choices=["basicvsrpp"],
        help='Restoration model name (only "basicvsrpp" supported for now)',
    )
    restoration.add_argument(
        "--restoration-model-path",
        type=str,
        default=str(Path("model_weights") / "lada_mosaic_restoration_model_generic_v1.2.pth"),
        help="Path to restoration model (default: %(default)s)",
    )
    restoration.add_argument(
        "--compile-basicvsrpp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Compile BasicVSR++ for big performance boost (at cost of VRAM usage). Not recommended to use big clip sizes. (default: %(default)s)",
    )
    restoration.add_argument(
        "--max-clip-size",
        type=int,
        default=60,
        help="Maximum clip size for tracking (default: %(default)s)",
    )
    restoration.add_argument(
        "--temporal-overlap",
        type=int,
        default=8,
        help="Discard margin for overlap+discard clip splitting. Each split uses 2*temporal_overlap input overlap and discards temporal_overlap frames at each split boundary (default: %(default)s)",
    )
    restoration.add_argument(
        "--enable-crossfade",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Cross-fade between clip boundaries to reduce flickering at seams. Uses frames that are already processed but otherwise discarded, so no extra GPU cost. (default: %(default)s)",
    )
    restoration.add_argument(
        "--denoise",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help="Spatial denoising strength applied to restored crops. Reduces noise artifacts. (default: %(default)s)",
    )
    restoration.add_argument(
        "--denoise-step",
        type=str,
        default="after_primary",
        choices=["after_primary", "after_secondary"],
        help="When to apply denoising: after_primary (before secondary) or after_secondary (right before blend). (default: %(default)s)",
    )

    secondary = parser.add_argument_group("2nd restoration")
    secondary.add_argument(
        "--secondary-restoration",
        type=str,
        default="none",
        choices=["none", "swin2sr", "tvai"],
        help='Secondary restoration after primary model (default: %(default)s)',
    )

    swin2sr = parser.add_argument_group("Swin2SR")
    swin2sr.add_argument(
        "--swin2sr-batch-size",
        type=int,
        default=8,
        help="Batch size for Swin2SR secondary restoration (default: %(default)s)",
    )
    swin2sr.add_argument(
        "--swin2sr-compilation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable Swin2SR TensorRT compilation/usage where supported (default: %(default)s)",
    )

    tvai = parser.add_argument_group("Topaz Video AI")
    tvai.add_argument(
        "--tvai-ffmpeg-path",
        type=str,
        default="C:\\Program Files\\Topaz Labs LLC\\Topaz Video AI\\ffmpeg.exe",
        help="Path to Topaz Video AI ffmpeg.exe (default: %(default)s)",
    )
    tvai.add_argument(
        "--tvai-model",
        type=str,
        default="iris-2",
        help='Topaz model name for tvai_up (e.g. "iris-2", "prob-4", "iris-3") (default: %(default)s)',
    )
    tvai.add_argument(
        "--tvai-scale",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help='Topaz tvai_up scale (1=no scale). Output size is 256*scale (default: %(default)s)',
    )
    tvai.add_argument(
        "--tvai-args",
        type=str,
        default="preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1",
        help='Extra params for tvai_up. (default: %(default)s)',
    )
    tvai.add_argument(
        "--tvai-workers",
        type=int,
        default=2,
        help="Number of parallel TVAI ffmpeg workers (default: %(default)s)",
    )

    detection = parser.add_argument_group("Detection")
    detection.add_argument(
        "--detection-model",
        type=str,
        default="rfdetr-v4",
        choices=["rfdetr-v4", "rfdetr-v3", "rfdetr-v2", "lada-yolo-v4", "lada-yolo-v2"],
        help="Detection model name (default: %(default)s)",
    )
    detection.add_argument(
        "--detection-model-path",
        type=str,
        default="",
        help='Optional path to detection weights. If not set, uses "model_weights/<detection-model>.onnx" (RF-DETR) or ".pt" (YOLO).',
    )
    detection.add_argument(
        "--detection-score-threshold",
        type=float,
        default=0.2,
        help="Detection score threshold (default: %(default)s)",
    )

    encoding = parser.add_argument_group("Encoding")
    encoding.add_argument(
        "--codec",
        type=str,
        default="hevc",
        help='Output video codec (only "hevc" supported for now)',
    )
    encoding.add_argument(
        "--encoder-settings",
        type=str,
        default="",
        help='Encoder settings, as JSON object or comma-separated key=value pairs (e.g. {"cq":22} or cq=22,lookahead=32)',
    )
    encoding.add_argument(
        "--working-directory",
        type=str,
        default="",
        help="Directory for encoder temp files (.hevc, temp video). Default: same as output.",
    )

    benchmark_group = parser.add_argument_group("Benchmark")
    benchmark_group.add_argument(
        "--benchmark-video",
        type=str,
        action="append",
        default=None,
        help="Video path for benchmark (can be repeated). Default: test_clip1_1080p.mp4, test_clip1_2160p.mp4",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.benchmark:
        from jasna.benchmark import run_benchmark_cli
        run_benchmark_cli(args)
        return

    if args.input is None or args.output is None:
        parser.error("--input and --output are required when not using --benchmark")

    check_required_executables(disable_ffmpeg_check=args.disable_ffmpeg_check)
    warn_if_windows_hardware_accelerated_gpu_scheduling_enabled()

    gpu_ok, gpu_result = check_nvidia_gpu()
    if not gpu_ok:
        if gpu_result == "no_cuda":
            print("Error: No CUDA device. An NVIDIA GPU with compute capability 7.5+ is required.")
        else:
            _, major, minor = gpu_result
            print(f"Error: Compute capability 7.5+ required (GPU: {major}.{minor}).")
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    from jasna.pipeline import Pipeline
    from jasna.media import parse_encoder_settings, validate_encoder_settings

    input_video = Path(args.input)
    if not input_video.exists():
        raise FileNotFoundError(str(input_video))

    output_video = Path(args.output)

    from jasna.mosaic.detection_registry import RFDETR_MODEL_NAMES, YOLO_MODEL_NAMES, coerce_detection_model_name, detection_model_weights_path

    detection_model_name = coerce_detection_model_name(str(args.detection_model))
    detection_model_path = Path(str(args.detection_model_path)) if str(args.detection_model_path).strip() else detection_model_weights_path(detection_model_name)
    if not detection_model_path.exists():
        raise FileNotFoundError(str(detection_model_path))

    restoration_model_name = str(args.restoration_model_name)
    restoration_model_path = Path(args.restoration_model_path)
    if not restoration_model_path.exists():
        raise FileNotFoundError(str(restoration_model_path))

    codec = str(args.codec).lower()
    if codec != "hevc":
        raise ValueError(f"Unsupported codec: {codec} (only hevc supported)")

    encoder_settings = validate_encoder_settings(parse_encoder_settings(str(args.encoder_settings)))

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    max_clip_size = int(args.max_clip_size)
    if max_clip_size <= 0:
        raise ValueError("--max-clip-size must be > 0")

    temporal_overlap = int(args.temporal_overlap)
    if temporal_overlap < 0:
        raise ValueError("--temporal-overlap must be >= 0")
    if temporal_overlap >= max_clip_size:
        raise ValueError("--temporal-overlap must be < --max-clip-size")
    if temporal_overlap > 0 and (2 * temporal_overlap) >= max_clip_size:
        raise ValueError("--temporal-overlap must satisfy 2*--temporal-overlap < --max-clip-size")

    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    detection_score_threshold = float(args.detection_score_threshold)
    if not (0.0 <= detection_score_threshold <= 1.0):
        raise ValueError("--detection-score-threshold must be in [0, 1]")

    if restoration_model_name != "basicvsrpp":
        raise ValueError(f"Unsupported restoration model: {restoration_model_name}")

    from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.swin2sr_secondary_restorer import Swin2srSecondaryRestorer
    from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer, _parse_tvai_args_kv

    with torch.cuda.device(device):
        use_tensorrt = basicvsrpp_startup_policy(
            restoration_model_path=str(restoration_model_path),
            max_clip_size=max_clip_size,
            device=device,
            fp16=fp16,
            compile_basicvsrpp=bool(args.compile_basicvsrpp),
        )

        secondary_name = str(args.secondary_restoration).lower()
        if secondary_name == "none":
            secondary_restorer = None
        elif secondary_name == "swin2sr":
            swin2sr_batch_size = int(args.swin2sr_batch_size)
            if swin2sr_batch_size <= 0:
                raise ValueError("--swin2sr-batch-size must be > 0")
            secondary_restorer = Swin2srSecondaryRestorer(
                device=device,
                fp16=fp16,
                batch_size=swin2sr_batch_size,
                use_tensorrt=bool(args.swin2sr_compilation),
            )
        elif secondary_name == "tvai":
            tvai_model = str(args.tvai_model).strip()
            if tvai_model == "":
                raise ValueError("--tvai-model must be non-empty")
            tvai_scale = int(args.tvai_scale)
            tvai_workers = int(args.tvai_workers)
            if tvai_workers <= 0:
                raise ValueError("--tvai-workers must be > 0")

            tvai_args_rest = str(args.tvai_args)
            tvai_kv = _parse_tvai_args_kv(tvai_args_rest)
            if "model" in tvai_kv:
                raise ValueError('Do not pass "model" in --tvai-args; use --tvai-model instead')
            if "scale" in tvai_kv:
                raise ValueError('Do not pass "scale" in --tvai-args; use --tvai-scale instead')
            if ("w" in tvai_kv) or ("h" in tvai_kv):
                raise ValueError('Do not pass "w" or "h" in --tvai-args; use --tvai-scale instead')

            tvai_args = f"model={tvai_model}:scale={tvai_scale}"
            if tvai_args_rest.strip() != "":
                tvai_args = f"{tvai_args}:{tvai_args_rest}"
            secondary_restorer = TvaiSecondaryRestorer(
                device=device,
                ffmpeg_path=str(args.tvai_ffmpeg_path),
                tvai_args=tvai_args,
                max_clip_size=max_clip_size,
                num_workers=tvai_workers,
            )
        else:
            raise ValueError(f"Unsupported secondary restoration: {secondary_name}")

        denoise_strength = DenoiseStrength(str(args.denoise).lower())
        denoise_step = DenoiseStep(str(args.denoise_step).lower())

        restoration_pipeline = RestorationPipeline(
            restorer=BasicvsrppMosaicRestorer(
                checkpoint_path=str(restoration_model_path),
                device=device,
                max_clip_size=max_clip_size,
                use_tensorrt=use_tensorrt,
                fp16=fp16,
            ),
            secondary_restorer=secondary_restorer,
            denoise_strength=denoise_strength,
            denoise_step=denoise_step,
        )

        stream = torch.cuda.Stream()
        working_directory = Path(args.working_directory) if args.working_directory else None
        Pipeline(
            input_video=input_video,
            output_video=output_video,
            detection_model_name=detection_model_name,
            detection_model_path=detection_model_path,
            detection_score_threshold=detection_score_threshold,
            restoration_pipeline=restoration_pipeline,
            codec=codec,
            encoder_settings=encoder_settings,
            stream=stream,
            batch_size=batch_size,
            device=device,
            max_clip_size=max_clip_size,
            temporal_overlap=temporal_overlap,
            enable_crossfade=bool(args.enable_crossfade),
            fp16=fp16,
            working_directory=working_directory,
        ).run()


if __name__ == "__main__":
    main()


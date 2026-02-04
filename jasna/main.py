import argparse
import logging
from pathlib import Path

from jasna import __version__
from jasna.os_utils import check_required_executables, warn_if_windows_hardware_accelerated_gpu_scheduling_enabled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--input", required=True, type=str, help="Path to input video")
    parser.add_argument("--output", required=True, type=str, help="Path to output video")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

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
        "--compile-tensorrt",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable TensorRT compilation/usage where supported (default: %(default)s)",
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

    tvai = parser.add_argument_group("Topaz Video AI")
    tvai.add_argument(
        "--tvai-ffmpeg-path",
        type=str,
        default="C:\\Program Files\\Topaz Labs LLC\\Topaz Video AI\\ffmpeg.exe",
        help="Path to Topaz Video AI ffmpeg.exe (default: %(default)s)",
    )
    tvai.add_argument(
        "--tvai-args",
        type=str,
        default="model=iris-3:scale=0:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1",
        help='Arguments for tvai_up filter (passed as "tvai_up=<args>" to -filter_complex) (default: %(default)s)',
    )

    detection = parser.add_argument_group("Detection")
    detection.add_argument(
        "--detection-model",
        type=str,
        default="rfdetr",
        choices=["rfdetr"],
        help='Detection model name (only "rfdetr" supported for now)',
    )
    detection.add_argument(
        "--detection-model-path",
        type=str,
        default=str(Path("model_weights") / "rfdetr-v3.onnx"),
        help="Path to detection ONNX model (default: %(default)s)",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check_required_executables()
    warn_if_windows_hardware_accelerated_gpu_scheduling_enabled()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
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

    detection_model_name = str(args.detection_model)
    detection_model_path = Path(args.detection_model_path)
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

    if detection_model_name != "rfdetr":
        raise ValueError(f"Unsupported detection model: {detection_model_name}")

    if restoration_model_name != "basicvsrpp":
        raise ValueError(f"Unsupported restoration model: {restoration_model_name}")

    from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.swin2sr_secondary_restorer import Swin2srSecondaryRestorer
    from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer

    use_tensorrt = basicvsrpp_startup_policy(
        restoration_model_path=str(restoration_model_path),
        max_clip_size=max_clip_size,
        device=device,
        fp16=fp16,
        compile_basicvsrpp=bool(args.compile_tensorrt) and bool(args.compile_basicvsrpp),
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
            use_tensorrt=bool(args.compile_tensorrt),
        )
    elif secondary_name == "tvai":
        secondary_restorer = TvaiSecondaryRestorer(
            device=device,
            ffmpeg_path=str(args.tvai_ffmpeg_path),
            tvai_args=str(args.tvai_args),
            max_clip_size=max_clip_size,
        )
    else:
        raise ValueError(f"Unsupported secondary restoration: {secondary_name}")

    restoration_pipeline = RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(restoration_model_path),
            device=device,
            max_clip_size=max_clip_size,
            use_tensorrt=use_tensorrt,
            fp16=fp16,
        ),
        secondary_restorer=secondary_restorer,
    )

    stream = torch.cuda.Stream()
    Pipeline(
        input_video=input_video,
        output_video=output_video,
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
        fp16=fp16,
    ).run()


if __name__ == "__main__":
    main()


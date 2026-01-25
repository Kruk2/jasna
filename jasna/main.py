import argparse
import logging
import shutil
import sys
from pathlib import Path


def check_required_executables() -> None:
    """Check that required external tools are available in PATH."""
    missing = []
    for exe in ("ffmpeg", "mkvmerge"):
        if shutil.which(exe) is None:
            missing.append(exe)
    
    if missing:
        print(f"Error: Required executable(s) not found in PATH: {', '.join(missing)}")
        print("Please install them and ensure they are available in your system PATH.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--input", required=True, type=str, help="Path to input video")
    parser.add_argument("--output", required=True, type=str, help="Path to output video")
    parser.add_argument(
        "--detection-model",
        type=str,
        default="rfdetr",
        choices=["rfdetr"],
        help='Detection model name (only "rfdetr" supported for now)',
    )
    parser.add_argument(
        "--detection-model-path",
        type=str,
        default=str(Path("model_weights") / "rfdetr.onnx"),
        help='Path to detection ONNX model (default: "model_weights/rfdetr.onnx")',
    )
    parser.add_argument(
        "--restoration-model-name",
        type=str,
        default="basicvsrpp",
        choices=["basicvsrpp"],
        help='Restoration model name (only "basicvsrpp" supported for now)',
    )
    parser.add_argument(
        "--restoration-model-path",
        type=str,
        default=str(Path("model_weights") / "lada_mosaic_restoration_model_generic_v1.2.pth"),
        help='Path to restoration model (default: "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")',
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance.",
    )
    parser.add_argument(
        "--compile-basicvsrpp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Compile BasicVSR++ for big performance boost (at cost of VRAM usage). Not recommended to use big clip sizes.",
    )
    parser.add_argument("--max-clip-size", type=int, default=30, help="Maximum clip size for tracking")
    parser.add_argument("--temporal-overlap", type=int, default=3, help="Number of restored frames to use as context for split clips")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    check_required_executables()
    
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    from jasna.pipeline import Pipeline

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

    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)

    if detection_model_name != "rfdetr":
        raise ValueError(f"Unsupported detection model: {detection_model_name}")

    if restoration_model_name != "basicvsrpp":
        raise ValueError(f"Unsupported restoration model: {restoration_model_name}")

    from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    use_tensorrt = basicvsrpp_startup_policy(
        restoration_model_path=str(restoration_model_path),
        max_clip_size=max_clip_size,
        device=device,
        fp16=fp16,
        compile_basicvsrpp=bool(args.compile_basicvsrpp),
    )
    
    restoration_pipeline = RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(restoration_model_path),
            device=device,
            max_clip_size=max_clip_size,
            use_tensorrt=use_tensorrt,
            fp16=fp16,
        )
    )

    stream = torch.cuda.Stream()
    Pipeline(
        input_video=input_video,
        output_video=output_video,
        detection_model_path=detection_model_path,
        restoration_pipeline=restoration_pipeline,
        stream=stream,
        batch_size=batch_size,
        device=device,
        max_clip_size=max_clip_size,
        temporal_overlap=temporal_overlap,
        fp16=fp16,
    ).run()


if __name__ == "__main__":
    main()


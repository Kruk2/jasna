import argparse
import logging
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--input", required=True, type=str, help="Path to input video")
    parser.add_argument("--output", required=True, type=str, help="Path to output video")
    parser.add_argument(
        "--restorer-model",
        type=str,
        default="rfdetr",
        choices=["rfdetr"],
        help='Restorer model name (only "rfdetr" supported for now)',
    )
    parser.add_argument(
        "--restorer-model-path",
        type=str,
        default=str(Path("model_weights") / "rfdetr.onnx"),
        help='Path to ONNX model (default: "model_weights/rfdetr.onnx")',
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-clip-size", type=int, default=30, help="Maximum clip size for tracking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    from jasna.mosaic import RfDetrMosaicDetectionModel
    from jasna.pipeline import Pipeline
    from jasna.restorer import RedTintRestorer, RestorationPipeline

    input_video = Path(args.input)
    if not input_video.exists():
        raise FileNotFoundError(str(input_video))

    output_video = Path(args.output)
    restorer_model = str(args.restorer_model)
    restorer_model_path = Path(args.restorer_model_path)
    if not restorer_model_path.exists():
        raise FileNotFoundError(str(restorer_model_path))

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    max_clip_size = int(args.max_clip_size)
    if max_clip_size <= 0:
        raise ValueError("--max-clip-size must be > 0")

    device = torch.device(str(args.device))

    if restorer_model != "rfdetr":
        raise ValueError(f"Unsupported restorer model: {restorer_model}")

    stream = torch.cuda.Stream()
    detection_model = RfDetrMosaicDetectionModel(
        onnx_path=restorer_model_path,
        stream=stream,
        batch_size=batch_size,
        device=device,
    )
    restoration_pipeline = RestorationPipeline(restorers=[RedTintRestorer(alpha=0.3)])

    Pipeline(
        input_video=input_video,
        output_video=output_video,
        detection_model=detection_model,
        restoration_pipeline=restoration_pipeline,
        stream=stream,
        batch_size=batch_size,
        device=device,
        max_clip_size=max_clip_size,
    ).run()


if __name__ == "__main__":
    main()


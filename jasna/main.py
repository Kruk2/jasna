import argparse
import logging
import sys
from pathlib import Path

from jasna import __version__
from jasna.engine_paths import model_weights_dir
from jasna.media import UnsupportedColorspaceError
from jasna.os_utils import (
    MIN_DRIVER_VERSION,
    check_ascii_install_path,
    check_gpu_driver_version,
    check_required_executables,
    check_supported_gpu,
    check_windows_nvidia_sysmem_fallback_policy,
)


def _path_collision_key(path: Path) -> str:
    absolute = path.resolve(strict=False)
    key = str(absolute)
    return key.casefold() if sys.platform == "win32" else key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks instead of processing video")
    parser.add_argument("--input", required=False, type=str, default=None, help="Path to input video, image, or folder")
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=None,
        help="Path to output file, or output folder when --input is a folder",
    )
    parser.add_argument(
        "--output-pattern",
        type=str,
        default=None,
        help=(
            "Filename template for folder input, matching the GUI pattern behavior. "
            "Use {original} for the input stem. Default: {original}_out with each input extension. "
            "Images keep their source extension; videos use the template extension when provided."
        ),
    )
    parser.add_argument(
        "--working-directory",
        type=str,
        default=None,
        help="Directory for temporary files created while assembling segment output (default: the output video's directory)",
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
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )

    restoration = parser.add_argument_group("Restoration")
    restoration.add_argument(
        "--restoration-model-name",
        type=str,
        default="basicvsrpp",
        choices=["basicvsrpp"],
        help='Restoration model for video input (only "basicvsrpp" supported for now).',
    )
    restoration.add_argument(
        "--restoration-model-path",
        type=str,
        default=str(model_weights_dir() / "lada_mosaic_restoration_model_generic_v1.2.pth"),
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
        default=90,
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
        choices=["none", "unet-4x", "tvai", "rtx-super-res"],
        help='Secondary restoration after primary model (default: %(default)s)',
    )

    sd15 = parser.add_argument_group("SD 1.5 image restoration")
    sd15.add_argument(
        "--image-restoration-model-name",
        type=str,
        default="sd-15-jav",
        choices=["sd-15-jav"],
        help="Restoration model for still-image input (image-only). Images auto-route here; no need to set --restoration-model-name.",
    )
    sd15.add_argument(
        "--sd15-steps",
        type=int,
        default=25,
        help="SD15 inpaint diffusion steps (default: %(default)s)",
    )
    sd15.add_argument(
        "--sd15-strength",
        type=float,
        default=0.6,
        help="SDEdit denoise strength, clamped to <= 0.7 (default: %(default)s)",
    )
    sd15.add_argument(
        "--sd15-freeu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply FreeU (s1=0.9, s2=0.2, b1=1.2, b2=1.4) to the SD15 UNet (default: %(default)s)",
    )
    sd15.add_argument(
        "--sd15-seed",
        type=int,
        default=0,
        help="Base seed for SD15 inpaint (default: %(default)s)",
    )
    sd15.add_argument(
        "--sd15-variants",
        type=int,
        default=1,
        help="Generate N stochastic variants with seeds seed..seed+N-1 (default: %(default)s)",
    )

    license_group = parser.add_argument_group("License")
    license_group.add_argument(
        "--license-email",
        type=str,
        default="",
        help="Supporter email tied to your license key (enables unet-4x).",
    )
    license_group.add_argument(
        "--license-key",
        type=str,
        default="",
        help="License key issued for your supporter email.",
    )

    rtx = parser.add_argument_group("RTX Super Res")
    rtx.add_argument(
        "--rtx-scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="RTX Super Res upscale factor (default: %(default)s)",
    )
    rtx.add_argument(
        "--rtx-quality",
        type=str,
        default="high",
        choices=["low", "medium", "high", "ultra"],
        help="RTX Super Res upscale quality (default: %(default)s)",
    )
    rtx.add_argument(
        "--rtx-denoise",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high", "ultra"],
        help="RTX Super Res denoise level, none to disable (default: %(default)s)",
    )
    rtx.add_argument(
        "--rtx-deblur",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high", "ultra"],
        help="RTX Super Res deblur level, none to disable (default: %(default)s)",
    )

    tvai = parser.add_argument_group("Topaz Video")
    tvai.add_argument(
        "--tvai-ffmpeg-path",
        type=str,
        default="C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        help="Path to Topaz Video ffmpeg.exe (default: %(default)s)",
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
        default="rfdetr-v5",
        help=(
            "Detection model name. Installed models are discovered from model_weights/; "
            "zelefans-vr-yolo-v2 is bundled with Jasna "
            "(default: %(default)s)"
        ),
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
        default=0.25,
        help="Detection score threshold (default: %(default)s)",
    )

    projection = parser.add_argument_group("VR projection")
    projection.add_argument(
        "--vr-mode",
        type=str,
        default="auto",
        choices=["auto", "off", "sbs", "sbs-fisheye"],
        help=(
            "VR180 handling: auto uses conservative studio/metadata detection; "
            "sbs processes each eye separately; sbs-fisheye also reprojects each "
            "eye for detection/restoration and preserves source projection on output. "
            "(default: %(default)s)"
        ),
    )

    streaming = parser.add_argument_group("Streaming")
    streaming.add_argument(
        "--stream",
        action="store_true",
        help="Enable HLS streaming mode (no file output). Serves processed video via HTTP for playback in VLC/browser.",
    )
    streaming.add_argument(
        "--stream-port",
        type=int,
        default=8765,
        help="HTTP port for HLS streaming server (default: %(default)s)",
    )
    streaming.add_argument(
        "--stream-segment-duration",
        type=float,
        default=4.0,
        help="HLS segment duration in seconds (default: %(default)s)",
    )
    streaming.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window when starting streaming mode.",
    )

    encoding = parser.add_argument_group("Encoding")
    encoding.add_argument(
        "--codec",
        type=lambda value: str(value).lower(),
        default="hevc",
        choices=["hevc", "h264", "av1"],
        help="Offline output video codec (HLS streaming always uses H.264). Default: %(default)s",
    )
    encoding.add_argument(
        "--encoder-settings",
        type=str,
        default="",
        help='Encoder settings, as JSON object or comma-separated key=value pairs (e.g. {"cq":22} or cq=22,rc-lookahead=32)',
    )
    encoding.add_argument(
        "--lut",
        type=str,
        default="",
        help="Path to a .cube color LUT (1D or 3D) applied on GPU before encoding.",
    )
    encoding.add_argument(
        "--retarget-high-fps",
        action="store_true",
        help=(
            "For offline exports, map 60 fps to 30 fps and 59.94 fps to 29.97 fps "
            "by processing every second frame. Other source rates are unchanged."
        ),
    )
    encoding.add_argument(
        "--segments",
        type=str,
        default="",
        help=(
            "Restore only selected ranges and smart-render the rest, for example "
            "10-25,01:10-01:30. Output codec must match the H.264, HEVC, or AV1 input."
        ),
    )

    post_export = parser.add_argument_group("Post-export action")
    post_export.add_argument(
        "--post-export-action",
        type=str,
        default="none",
        choices=["none", "shutdown", "command"],
        help="Action to run after all non-streaming exports finish.",
    )
    post_export.add_argument(
        "--post-export-command",
        type=str,
        default="",
        help="Shell command to run when --post-export-action=command.",
    )

    benchmark_group = parser.add_argument_group("Benchmark")
    benchmark_group.add_argument(
        "--benchmark-filter",
        type=str,
        default=None,
        help="Only run benchmarks whose name contains this string (e.g. 'basicvsrpp')",
    )
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
    codec_was_explicit = any(
        value == "--codec" or value.startswith("--codec=")
        for value in sys.argv[1:]
    )

    if args.benchmark:
        from jasna.benchmark import run_benchmark_cli
        run_benchmark_cli(args)
        return

    is_streaming = bool(args.stream)
    if is_streaming and args.retarget_high_fps:
        parser.error("--retarget-high-fps is only supported for offline exports")
    from jasna.post_export_action import validate_post_export_action, run_post_export_action_safely
    validate_post_export_action(str(args.post_export_action), str(args.post_export_command))

    def _run_post_export_action() -> None:
        run_post_export_action_safely(
            str(args.post_export_action),
            str(args.post_export_command),
            lambda message: print(f"Warning: {message}"),
        )

    if args.input is None and not is_streaming:
        parser.error("--input is required when not using --benchmark or --stream")
    if args.output is None and not is_streaming:
        parser.error("--output is required when not using --benchmark or --stream")

    path_ok, path_info = check_ascii_install_path()
    if not path_ok:
        print(f"Error: Jasna must be installed in a path with ASCII characters only.")
        print(f"Current path: {path_info}")
        sys.exit(1)

    check_required_executables()

    gpu_ok, gpu_result = check_supported_gpu(str(args.device))
    if not gpu_ok:
        if gpu_result == "no_cuda":
            print("Error: No compatible GPU was found for this Jasna build.")
        else:
            _, major, minor = gpu_result
            print(f"Error: Compute capability 7.5+ required (GPU: {major}.{minor}).")
        sys.exit(1)

    driver_ok, driver_info = check_gpu_driver_version()
    if not driver_ok:
        print(f"Error: GPU driver version check failed: {driver_info}")
        if "ROCm" not in driver_info:
            print(f"Please update your NVIDIA driver to version {MIN_DRIVER_VERSION} or newer.")
        sys.exit(1)

    from jasna.accelerator import is_nvidia_device

    if sys.platform == "win32" and is_nvidia_device():
        sysmem_ok, sysmem_info = check_windows_nvidia_sysmem_fallback_policy()
        if not sysmem_ok:
            print(f"Warning: CUDA Sysmem Fallback Policy: {sysmem_info}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()

    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()
    import torch

    from jasna.pipeline import Pipeline
    from jasna.media import parse_encoder_settings, validate_encoder_settings

    input_video = Path(args.input) if args.input else None
    if input_video is not None and not input_video.exists():
        raise FileNotFoundError(str(input_video))

    output_video = Path(args.output) if args.output else (input_video.with_stem(input_video.stem + "_out") if input_video else None)

    from jasna.media.image_io import is_image_path
    input_is_image = input_video is not None and is_image_path(input_video)
    input_is_dir = input_video is not None and input_video.is_dir()
    segments_spec = str(args.segments).strip()
    if segments_spec:
        if is_streaming:
            parser.error("--segments cannot be combined with --stream")
        if input_is_image:
            parser.error("--segments requires a single video input, not an image")
        if input_is_dir:
            parser.error("--segments requires a single video input, not a folder")

    folder_videos: list[Path] = []
    folder_output_dir: Path | None = None
    if input_is_dir:
        if is_streaming:
            parser.error("--stream does not support folder input")
        from jasna.media.media_files import classify_folder, folder_output_path, is_media
        folder_images, folder_videos = classify_folder(input_video)
        folder_total = len(folder_images) + len(folder_videos)
        if not folder_images and not folder_videos:
            parser.error(f"No image or video files found in folder: {input_video}")
        if output_video is None:
            parser.error("--output (a folder) is required when --input is a folder")
        folder_output_dir = output_video
        if folder_output_dir.exists() and not folder_output_dir.is_dir():
            parser.error(f"--output must be a folder when --input is a folder (got existing file: {folder_output_dir})")
        if not folder_output_dir.exists() and is_media(folder_output_dir):
            parser.error(
                f"--output must be a folder when --input is a folder; got a media filename: {folder_output_dir}"
            )
        folder_inputs = [*folder_images, *folder_videos]
        planned_outputs: dict[str, tuple[Path, Path]] = {}
        input_keys = {_path_collision_key(path) for path in folder_inputs}
        for path in folder_inputs:
            out_path = folder_output_path(folder_output_dir, path, args.output_pattern)
            out_key = _path_collision_key(out_path)
            if out_key in planned_outputs:
                other_input, other_output = planned_outputs[out_key]
                parser.error(
                    "--output-pattern maps multiple inputs to the same output: "
                    f"{other_input.name} and {path.name} -> {other_output}"
                )
            if out_key in input_keys:
                parser.error(f"--output-pattern would overwrite an input file: {out_path}")
            planned_outputs[out_key] = (path, out_path)
        folder_output_dir.mkdir(parents=True, exist_ok=True)
        # Images first, then videos.
        if folder_images:
            from jasna.image_restore import run_image_restoration_folder
            run_image_restoration_folder(
                args,
                folder_images,
                folder_output_dir,
                output_pattern=args.output_pattern,
                progress_total=folder_total,
            )
        if not folder_videos:
            _run_post_export_action()
            return

    if input_is_image:
        if is_streaming:
            parser.error("Image input does not support --stream")
        from jasna.image_restore import run_image_restoration
        run_image_restoration(args)
        _run_post_export_action()
        return

    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        discover_available_detection_models,
        require_detection_model_weights,
    )

    detection_model_name = coerce_detection_model_name(str(args.detection_model))
    has_explicit_path = bool(str(args.detection_model_path).strip())
    if not has_explicit_path:
        available = discover_available_detection_models()
        if available and detection_model_name not in available:
            print(f"Warning: detection model '{detection_model_name}' not found in model_weights/. Available: {', '.join(available)}")
    detection_model_path = (
        Path(str(args.detection_model_path))
        if has_explicit_path
        else require_detection_model_weights(detection_model_name)
    )
    if not detection_model_path.exists():
        raise FileNotFoundError(str(detection_model_path))

    restoration_model_name = str(args.restoration_model_name)
    restoration_model_path = Path(args.restoration_model_path)
    if not restoration_model_path.exists():
        raise FileNotFoundError(str(restoration_model_path))

    segments = None
    splice_plan = None
    codec = str(args.codec).lower()
    if segments_spec:
        from jasna.media import get_video_meta_data
        from jasna.media.splice import (
            SmartRenderCompatibilityError,
            build_splice_plan,
            probe_keyframes,
            validate_smart_render,
        )
        from jasna.segments import parse_segments

        metadata = get_video_meta_data(str(input_video))
        try:
            segments = parse_segments(segments_spec, duration=metadata.duration)
        except ValueError as exc:
            parser.error(f"invalid --segments: {exc}")
        input_codec = {
            "avc": "h264",
            "h265": "hevc",
            "av01": "av1",
        }.get(metadata.codec_name.lower(), metadata.codec_name.lower())
        if codec_was_explicit and codec != input_codec:
            parser.error(
                f"with --segments output codec must match input; pass --codec {input_codec}"
            )
        codec = input_codec
        try:
            validate_smart_render(
                metadata,
                output_path=output_video,
                codec=codec,
                retarget_high_fps=bool(args.retarget_high_fps),
            )
            splice_plan = build_splice_plan(
                segments,
                probe_keyframes(input_video, metadata),
                duration=metadata.duration,
            )
        except SmartRenderCompatibilityError as exc:
            parser.error(str(exc))
    if codec not in {"hevc", "h264", "av1"}:
        raise ValueError(f"Unsupported codec: {codec} (supported: hevc, h264, av1)")

    encoder_settings = validate_encoder_settings(parse_encoder_settings(str(args.encoder_settings)), codec=codec)

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
    from jasna.accelerator import device_context, is_amd_device

    fp16 = bool(args.fp16)
    detection_score_threshold = float(args.detection_score_threshold)
    if not (0.0 <= detection_score_threshold <= 1.0):
        raise ValueError("--detection-score-threshold must be in [0, 1]")

    if restoration_model_name != "basicvsrpp":
        raise ValueError(f"Unsupported restoration model: {restoration_model_name}")

    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    secondary_name = str(args.secondary_restoration).lower()
    if is_amd_device(device) and secondary_name != "none":
        raise ValueError(
            f"Secondary restoration '{secondary_name}' is not available in the AMD build yet"
        )

    if args.license_email and args.license_key:
        from jasna.protection import license_store
        license_store.set_license(args.license_email, args.license_key)

    compile_result = ensure_engines_compiled(EngineCompilationRequest(
        device=str(device),
        fp16=fp16,
        basicvsrpp=bool(args.compile_basicvsrpp) and not is_amd_device(device),
        basicvsrpp_model_path=str(restoration_model_path),
        basicvsrpp_max_clip_size=max_clip_size,
        detection=True,
        detection_model_name=detection_model_name,
        detection_model_path=str(detection_model_path),
        detection_batch_size=batch_size,
        unet4x=(secondary_name == "unet-4x"),
    ))
    use_tensorrt = compile_result.use_basicvsrpp_tensorrt

    with device_context(device):
        if secondary_name == "none":
            secondary_restorer = None
        elif secondary_name == "tvai":
            from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
            tvai_args_str = f"model={args.tvai_model}:scale={args.tvai_scale}:{args.tvai_args}"
            secondary_restorer = TvaiSecondaryRestorer(
                ffmpeg_path=args.tvai_ffmpeg_path,
                tvai_args=tvai_args_str,
                scale=int(args.tvai_scale),
                num_workers=int(args.tvai_workers),
            )
        elif secondary_name == "unet-4x":
            from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer
            secondary_restorer = Unet4xSecondaryRestorer(device=device, fp16=fp16)
        elif secondary_name == "rtx-super-res":
            from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
            rtx_denoise = str(args.rtx_denoise).lower()
            rtx_deblur = str(args.rtx_deblur).lower()
            secondary_restorer = RtxSuperresSecondaryRestorer(
                device=device,
                scale=int(args.rtx_scale),
                quality=str(args.rtx_quality).lower(),
                denoise=None if rtx_denoise == "none" else rtx_denoise,
                deblur=None if rtx_deblur == "none" else rtx_deblur,
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

        lut_arg = str(args.lut).strip()
        if lut_arg and not Path(lut_arg).exists():
            raise FileNotFoundError(lut_arg)
        lut_path = lut_arg or None

        working_dir = Path(args.working_directory) if args.working_directory else None

        def _make_pipeline(vid_input: Path, out_path: Path) -> Pipeline:
            return Pipeline(
                input_video=vid_input,
                output_video=out_path,
                detection_model_name=detection_model_name,
                detection_model_path=detection_model_path,
                detection_score_threshold=detection_score_threshold,
                restoration_pipeline=restoration_pipeline,
                codec=codec,
                encoder_settings=encoder_settings,
                batch_size=batch_size,
                device=device,
                max_clip_size=max_clip_size,
                temporal_overlap=temporal_overlap,
                enable_crossfade=bool(args.enable_crossfade),
                vr_mode=str(args.vr_mode),
                fp16=fp16,
                disable_progress=args.no_progress,
                lut_path=lut_path,
                retarget_high_fps=bool(args.retarget_high_fps),
                segments=segments,
                splice_plan=splice_plan,
                working_dir=working_dir,
            )

        video_inputs = folder_videos if input_is_dir else ([input_video] if input_video is not None else [])

        def _video_output_path(vid: Path) -> Path:
            if input_is_dir:
                return folder_output_path(folder_output_dir, vid, args.output_pattern)
            return output_video or vid.with_stem(vid.stem + "_out")

        pipeline: Pipeline | None = None
        try:
            if is_streaming and input_video is None:
                from jasna.streaming import HlsStreamingServer
                pipeline = _make_pipeline(Path("__streaming__"), Path("__streaming___out__"))
                hls_server = HlsStreamingServer(
                    segment_duration=float(args.stream_segment_duration),
                    port=int(args.stream_port),
                )
                hls_server.start()
                if not args.no_browser:
                    import webbrowser
                    webbrowser.open(f"http://localhost:{args.stream_port}/")
                try:
                    while True:
                        video_path = hls_server.wait_for_video()
                        pipeline.input_video = video_path
                        try:
                            pipeline.run_streaming(
                                hls_server=hls_server,
                                segment_duration=float(args.stream_segment_duration),
                            )
                        except UnsupportedColorspaceError as e:
                            print(f"Error: {e}")
                        hls_server.unload_video()
                except KeyboardInterrupt:
                    pass
                finally:
                    hls_server.stop()
            elif is_streaming:
                pipeline = _make_pipeline(input_video, _video_output_path(input_video))
                if not args.no_browser:
                    import webbrowser
                    webbrowser.open(f"http://localhost:{args.stream_port}/")
                pipeline.run_streaming(
                    port=int(args.stream_port),
                    segment_duration=float(args.stream_segment_duration),
                )
            else:
                video_start = len(folder_images) + 1 if input_is_dir else 1
                video_total = len(folder_images) + len(video_inputs) if input_is_dir else len(video_inputs)
                for i, vid in enumerate(video_inputs, start=video_start):
                    out_path = _video_output_path(vid)
                    if input_is_dir:
                        print(f"[{i}/{video_total}] Processing {vid.name} -> {out_path.name}")
                    pipeline = _make_pipeline(vid, out_path)
                    try:
                        pipeline.run()
                    except UnsupportedColorspaceError as e:
                        # In a folder batch, skip the bad file and keep going.
                        print(f"Error processing {vid.name}: {e}")
                        if not input_is_dir:
                            sys.exit(1)
                    finally:
                        pipeline.close()
                        pipeline = None
                _run_post_export_action()
        except UnsupportedColorspaceError as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            if pipeline is not None:
                pipeline.close()
            restoration_pipeline.restorer.close()
            if secondary_restorer is not None and hasattr(secondary_restorer, "close"):
                secondary_restorer.close()


if __name__ == "__main__":
    main()

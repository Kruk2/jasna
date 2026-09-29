import argparse
import logging
import sys
from pathlib import Path
from typing import get_args

from jasna import __version__
from jasna.cli_help import CLI_HELP
from jasna.engine_paths import default_restoration_model_path
from jasna.media.probe import UnsupportedColorspaceError
from jasna.os_utils import (
    MIN_DRIVER_VERSION,
    check_ascii_install_path,
    check_gpu_driver_version,
    check_required_executables,
    check_supported_gpu,
    check_windows_nvidia_sysmem_fallback_policy,
    gpu_check_error,
)
from jasna.session_config import LTX_DEFAULT_MODEL, LTX_DEFAULT_SEED, LtxModelName, SessionConfig



def _session_config_from_args(
    args: argparse.Namespace,
    *,
    codec: str,
    encoder_settings: dict[str, object],
    detection_model_name: str,
    detection_model_path: Path,
    detection_score_threshold: float,
    restoration_model_path: Path,
    lut_path: str | None,
) -> SessionConfig:
    return SessionConfig(
        device=str(args.device),
        fp16=bool(args.fp16),
        batch_size=int(args.batch_size),
        detection_model_name=detection_model_name,
        detection_model_path=detection_model_path,
        detection_score_threshold=detection_score_threshold,
        max_detection_gap=int(args.max_detection_gap),
        min_detection_duration=int(args.min_detection_duration),
        scene_detection=bool(args.scene_detection),
        restoration_model_name=str(args.restoration_model_name),
        restoration_model_path=restoration_model_path,
        ltx_large_canvas=bool(args.ltx_large_canvas),
        ltx_seed=int(args.ltx_seed),
        ltx_fast=bool(args.ltx_fast),
        ltx_model=str(args.ltx_model),
        compile_basicvsrpp=bool(args.compile_basicvsrpp),
        max_clip_size=int(args.max_clip_size),
        temporal_overlap=int(args.temporal_overlap),
        enable_crossfade=bool(args.enable_crossfade),
        denoise_strength=str(args.denoise).lower(),
        denoise_step=str(args.denoise_step).lower(),
        secondary_restoration=str(args.secondary_restoration).lower(),
        tvai_ffmpeg_path=str(args.tvai_ffmpeg_path),
        tvai_model=str(args.tvai_model),
        tvai_scale=int(args.tvai_scale),
        tvai_args=str(args.tvai_args),
        tvai_workers=int(args.tvai_workers),
        tvai_denoise=bool(args.tvai_denoise),
        rtx_scale=int(args.rtx_scale),
        rtx_quality=str(args.rtx_quality).lower(),
        rtx_denoise=str(args.rtx_denoise).lower(),
        rtx_deblur=str(args.rtx_deblur).lower(),
        vr_mode=str(args.vr_mode),
        vr_projection="auto",
        codec=codec,
        encoder_settings=encoder_settings,
        lut_path=lut_path,
        sharpen_strength=float(args.sharpen),
        retarget_high_fps=bool(args.retarget_high_fps),
        fmp4=bool(args.fmp4),
        disable_progress=bool(args.no_progress),
        working_dir=Path(args.working_directory) if args.working_directory else None,
    )


def _path_collision_key(path: Path) -> str:
    absolute = path.resolve(strict=False)
    key = str(absolute)
    return key.casefold() if sys.platform == "win32" else key


def _resolve_cli_encoder_settings(
    raw_settings: str,
    *,
    cq: int | None,
    codec: str,
    vendor,
) -> dict[str, object]:
    from jasna.accelerator import AcceleratorVendor
    from jasna.media.encoder_settings import parse_encoder_settings, validate_encoder_settings
    from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_cq

    resolved_vendor = AcceleratorVendor(str(vendor))
    settings = parse_encoder_settings(raw_settings)
    cq_aliases = {"cq"}
    if resolved_vendor is AcceleratorVendor.AMD:
        cq_aliases.add("qvbr_quality_level")
    duplicates = sorted(cq_aliases & settings.keys())
    if len(duplicates) > 1:
        raise ValueError(
            "--encoder-settings contains multiple CQ controls: "
            f"{', '.join(duplicates)}; use only one"
        )
    if cq is not None and duplicates:
        raise ValueError(
            "--cq conflicts with --encoder-settings "
            f"{', '.join(duplicates)}; use only one CQ control"
        )

    if cq is not None:
        settings["cq"] = validate_encoder_cq(
            cq,
            codec=codec,
            vendor=resolved_vendor,
        )
    elif "cq" in settings:
        validate_encoder_cq(
            settings["cq"],
            codec=codec,
            vendor=resolved_vendor,
        )
    elif (
        resolved_vendor is AcceleratorVendor.AMD
        and "qvbr_quality_level" in settings
    ):
        settings["cq"] = validate_encoder_cq(
            settings.pop("qvbr_quality_level"),
            codec=codec,
            vendor=resolved_vendor,
        )
    else:
        settings["cq"] = encoder_cq_spec(codec, resolved_vendor).default

    return validate_encoder_settings(
        settings,
        codec=codec,
        vendor=resolved_vendor,
    )


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
        help=CLI_HELP["fp16"],
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
        choices=["basicvsrpp", "ltx"],
        help=CLI_HELP["restoration_model_name"],
    )
    restoration.add_argument(
        "--restoration-model-path",
        type=str,
        default=None,
        help=CLI_HELP["restoration_model_path"],
    )
    restoration.add_argument(
        "--ltx-large-canvas",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["ltx_large_canvas"],
    )
    restoration.add_argument(
        "--ltx-seed",
        type=int,
        default=LTX_DEFAULT_SEED,
        help=CLI_HELP["ltx_seed"],
    )
    restoration.add_argument(
        "--ltx-model",
        choices=list(get_args(LtxModelName)),
        default=LTX_DEFAULT_MODEL,
        help=CLI_HELP["ltx_model"],
    )
    restoration.add_argument(
        "--ltx-fast",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["ltx_fast"],
    )
    restoration.add_argument(
        "--compile-basicvsrpp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["compile_basicvsrpp"],
    )
    restoration.add_argument(
        "--max-clip-size",
        type=int,
        default=90,
        help=CLI_HELP["max_clip_size"],
    )
    restoration.add_argument(
        "--temporal-overlap",
        type=int,
        default=8,
        help=CLI_HELP["temporal_overlap"],
    )
    restoration.add_argument(
        "--enable-crossfade",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["enable_crossfade"],
    )
    restoration.add_argument(
        "--denoise",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help=CLI_HELP["denoise"],
    )
    restoration.add_argument(
        "--denoise-step",
        type=str,
        default="after_primary",
        choices=["after_primary", "after_secondary"],
        help=CLI_HELP["denoise_step"],
    )

    secondary = parser.add_argument_group("2nd restoration")
    secondary.add_argument(
        "--secondary-restoration",
        type=str,
        default="none",
        choices=["none", "unet-4x", "tvai", "rtx-super-res"],
        help=CLI_HELP["secondary_restoration"],
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
        help=CLI_HELP["tvai_ffmpeg_path"],
    )
    tvai.add_argument(
        "--tvai-model",
        type=str,
        default="iris-2",
        help=CLI_HELP["tvai_model"],
    )
    tvai.add_argument(
        "--tvai-scale",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help=CLI_HELP["tvai_scale"],
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
        help=CLI_HELP["tvai_workers"],
    )
    tvai.add_argument(
        "--tvai-denoise",
        default=False,
        action="store_true",
        help=CLI_HELP["tvai_denoise"],
    )

    detection = parser.add_argument_group("Detection")
    detection.add_argument(
        "--detection-model",
        type=str,
        default="rfdetr-v6",
        help=(
            "Detection model name. Installed models are discovered from model_weights/; "
            "rfdetr-v6 (fast) and rfdetr-vr-v1 (VR180) are bundled with Jasna, "
            "rfdetr-v6-large (higher quality) and zelefans-vr-yolo-v2 are optional downloads "
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
        default=None,
        help=CLI_HELP["detection_score_threshold"],
    )
    detection.add_argument(
        "--max-detection-gap",
        type=int,
        default=2,
        help=CLI_HELP["max_detection_gap"],
    )
    detection.add_argument(
        "--min-detection-duration",
        type=int,
        default=2,
        help=CLI_HELP["min_detection_duration"],
    )
    detection.add_argument(
        "--scene-detection",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["scene_detection"],
    )

    projection = parser.add_argument_group("VR projection")
    projection.add_argument(
        "--vr-mode",
        type=str,
        default="auto",
        choices=["auto", "off", "sbs", "sbs-fisheye"],
        help=CLI_HELP["vr_mode"],
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
        help=CLI_HELP["codec"],
    )
    encoding.add_argument(
        "--cq",
        type=int,
        default=None,
        help=CLI_HELP["cq"],
    )
    encoding.add_argument(
        "--encoder-settings",
        type=str,
        default="",
        help=CLI_HELP["encoder_settings"],
    )
    encoding.add_argument(
        "--lut",
        type=str,
        default="",
        help="Path to a .cube color LUT (1D or 3D) applied on GPU before encoding.",
    )
    encoding.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help=(
            "Sharpen the picture on GPU before encoding, from 0 (off) to 1 "
            "(strongest). Matches the ffmpeg cas filter."
        ),
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
        "--fmp4",
        action="store_true",
        help=(
            "Write .mp4/.mov output as fragmented MP4, so the file can be played "
            "while processing runs and stays playable after an interruption. "
            "Not available with --stream or --segments."
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
        help=CLI_HELP["post_export_action"],
    )
    post_export.add_argument(
        "--post-export-command",
        type=str,
        default="",
        help="Shell command to run when --post-export-action=command.",
    )
    post_export.add_argument(
        "--post-export-video-command",
        type=str,
        default="",
        help=CLI_HELP["post_export_video_command"],
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


def _check_system(args: argparse.Namespace) -> None:
    path_ok, path_info = check_ascii_install_path()
    if not path_ok:
        print("Error: Jasna must be installed in a path with ASCII characters only.")
        print(f"Current path: {path_info}")
        sys.exit(1)

    check_required_executables()

    gpu_ok, gpu_result = check_supported_gpu(str(args.device))
    if not gpu_ok:
        print(f"Error: {gpu_check_error(gpu_result)}")
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


def _configure_runtime(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()

    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()
    from jasna.accelerator import configure_rocm_process_env
    configure_rocm_process_env()


def _plan_folder(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path | None,
) -> tuple[list[Path], list[Path], Path]:
    """Validate a folder batch and return its images, videos and output folder."""
    from jasna.media.media_files import classify_folder, folder_output_path, is_media

    images, videos = classify_folder(input_dir)
    if not images and not videos:
        parser.error(f"No image or video files found in folder: {input_dir}")
    if output_dir is None:
        parser.error("--output (a folder) is required when --input is a folder")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output must be a folder when --input is a folder (got existing file: {output_dir})")
    if not output_dir.exists() and is_media(output_dir):
        parser.error(f"--output must be a folder when --input is a folder; got a media filename: {output_dir}")
    inputs = [*images, *videos]
    planned_outputs: dict[str, tuple[Path, Path]] = {}
    input_keys = {_path_collision_key(path) for path in inputs}
    for path in inputs:
        out_path = folder_output_path(output_dir, path, args.output_pattern)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    return images, videos, output_dir


def _resolve_segments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_video: Path,
    output_video: Path,
    *,
    codec_was_explicit: bool,
):
    """Parse --segments and return (codec, segments, splice_plan); smart rendering keeps the input codec."""
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import (
        SmartRenderCompatibilityError,
        build_splice_plan,
        canonical_codec,
        probe_keyframes,
        validate_smart_render,
    )
    from jasna.segments import parse_segments

    metadata = get_video_meta_data(str(input_video))
    try:
        segments = parse_segments(str(args.segments).strip(), duration=metadata.duration)
    except ValueError as exc:
        parser.error(f"invalid --segments: {exc}")
    input_codec = canonical_codec(metadata.codec_name)
    if codec_was_explicit and str(args.codec).lower() != input_codec:
        parser.error(f"with --segments output codec must match input; pass --codec {input_codec}")
    try:
        validate_smart_render(
            metadata,
            output_path=output_video,
            codec=input_codec,
            retarget_high_fps=bool(args.retarget_high_fps),
        )
        splice_plan = build_splice_plan(
            segments,
            probe_keyframes(input_video, metadata),
            duration=metadata.duration,
        )
    except SmartRenderCompatibilityError as exc:
        parser.error(str(exc))
    return input_codec, segments, splice_plan


def _run_streaming(args: argparse.Namespace, make_pipeline, input_video: Path | None) -> None:
    from jasna.streaming import HlsStreamingServer
    from jasna.streaming_pipeline import run_streaming

    hls_server = HlsStreamingServer(
        segment_duration=float(args.stream_segment_duration),
        port=int(args.stream_port),
    )
    print(f"HLS stream: {hls_server.start()}")
    print(f"Browser:    http://localhost:{args.stream_port}/")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.stream_port}/")
    try:
        if input_video is not None:
            run_streaming(make_pipeline(input_video, input_video), hls_server)
            return
        while True:
            video_path = hls_server.wait_for_video()
            try:
                run_streaming(make_pipeline(video_path, video_path), hls_server)
            except UnsupportedColorspaceError as e:
                print(f"Error: {e}")
            hls_server.unload_video()
    except KeyboardInterrupt:
        print("Streaming stopped")
    finally:
        hls_server.stop()


def _run_videos(
    args: argparse.Namespace,
    make_pipeline,
    videos: list[Path],
    output_path_for,
    *,
    in_folder: bool,
    first_index: int,
    total: int,
) -> bool:
    """Process videos one by one; returns False when a per-video command failed in a folder batch."""
    from jasna.post_export_action import PostExportVideoCommandError, run_post_export_video_command

    post_export_video_command = str(args.post_export_video_command).strip()
    all_commands_ok = True
    for i, vid in enumerate(videos, start=first_index):
        out_path = output_path_for(vid)
        if in_folder:
            print(f"[{i}/{total}] Processing {vid.name} -> {out_path.name}")
        try:
            make_pipeline(vid, out_path).run()
        except UnsupportedColorspaceError as e:
            print(f"Error processing {vid.name}: {e}")
            if not in_folder:
                sys.exit(1)
            continue
        if not post_export_video_command:
            continue
        print(f"Running post-export command for {out_path.name}")
        try:
            run_post_export_video_command(post_export_video_command, vid, out_path, lambda: False)
        except PostExportVideoCommandError as e:
            print(f"Error post-processing {vid.name}: {e}")
            if not in_folder:
                sys.exit(1)
            all_commands_ok = False
    return all_commands_ok


def _ensure_ltx_model(config: SessionConfig, segments) -> None:
    """Offer to download the selected LTX model when the job needs it and it is missing.
    Without a terminal to ask in, fail with what to download instead of waiting."""
    from jasna.ltx.model_files import download_files, download_size_text, missing_downloads

    uses_ltx = config.restoration_model_name == "ltx" or any(
        segment.restoration is not None and segment.restoration.model == "ltx" for segment in segments or ()
    )
    if not uses_ltx:
        return
    directory = config.restoration_model_path if config.restoration_model_name == "ltx" else default_restoration_model_path("ltx")
    missing = missing_downloads(directory, config.ltx_model, fast=config.ltx_fast)
    if not missing:
        return
    names = ", ".join(file.name for file in missing)
    size = download_size_text(missing)
    if not sys.stdin.isatty():
        raise FileNotFoundError(f"LTX model files missing in {directory}: {names}. Run jasna in a terminal to download them ({size}).")
    answer = input(f"The LTX {config.ltx_model} model is missing in {directory}:\n  {names}\nDownload {size} now? [y/N]: ")
    if answer.strip().lower() not in ("y", "yes"):
        raise FileNotFoundError(f"LTX model files missing in {directory}: {names}")
    from tqdm import tqdm

    with tqdm(total=sum(file.size_bytes for file in missing), unit="B", unit_scale=True, desc="LTX download") as bar:
        download_files(directory, missing, lambda done, total: bar.update(done - bar.n))


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
    if is_streaming and args.fmp4:
        parser.error("--fmp4 is only supported for offline exports")
    from jasna.post_export_action import run_post_export_action_safely, validate_post_export_action
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

    _check_system(args)
    _configure_runtime(args)

    input_video = Path(args.input) if args.input else None
    if input_video is not None and not input_video.exists():
        raise FileNotFoundError(str(input_video))
    output_video = Path(args.output) if args.output else (input_video.with_stem(input_video.stem + "_out") if input_video else None)

    from jasna.media.media_files import folder_output_path, is_image
    input_is_image = input_video is not None and is_image(input_video)
    input_is_dir = input_video is not None and input_video.is_dir()
    segments_spec = str(args.segments).strip()
    if segments_spec:
        if is_streaming:
            parser.error("--segments cannot be combined with --stream")
        if input_is_image:
            parser.error("--segments requires a single video input, not an image")
        if input_is_dir:
            parser.error("--segments requires a single video input, not a folder")
        if args.fmp4:
            parser.error("--fmp4 cannot be combined with --segments")

    folder_images: list[Path] = []
    folder_videos: list[Path] = []
    folder_output_dir: Path | None = None
    if input_is_dir:
        if is_streaming:
            parser.error("--stream does not support folder input")
        folder_images, folder_videos, folder_output_dir = _plan_folder(parser, args, input_video, output_video)
        if folder_images:
            from jasna.image_restore import run_image_restoration_folder
            run_image_restoration_folder(
                args,
                folder_images,
                folder_output_dir,
                output_pattern=args.output_pattern,
                progress_total=len(folder_images) + len(folder_videos),
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
        resolve_detection_model,
    )

    if not str(args.detection_model_path).strip():
        available = discover_available_detection_models()
        requested = coerce_detection_model_name(str(args.detection_model))
        if available and requested not in available:
            print(f"Warning: detection model '{requested}' not found in model_weights/. Available: {', '.join(available)}")
    detection_model_name, detection_model_path, detection_score_threshold = resolve_detection_model(
        str(args.detection_model), str(args.detection_model_path), args.detection_score_threshold
    )

    if args.restoration_model_name == "ltx":
        if is_streaming:
            parser.error("--restoration-model-name ltx does not support --stream")
        if args.secondary_restoration != "none" or args.denoise != "none":
            parser.error("--restoration-model-name ltx does not support --secondary-restoration or --denoise")
    restoration_model_path = Path(
        args.restoration_model_path or default_restoration_model_path(args.restoration_model_name)
    )
    if not restoration_model_path.exists():
        raise FileNotFoundError(str(restoration_model_path))

    segments = None
    splice_plan = None
    codec = str(args.codec).lower()
    if segments_spec:
        codec, segments, splice_plan = _resolve_segments(
            parser, args, input_video, output_video, codec_was_explicit=codec_was_explicit
        )

    from jasna.accelerator import device_context, vendor_for_device

    encoder_settings = _resolve_cli_encoder_settings(
        str(args.encoder_settings),
        cq=args.cq,
        codec=codec,
        vendor=vendor_for_device(str(args.device)),
    )

    if args.license_email and args.license_key:
        from jasna.protection import license_store
        license_store.set_license(args.license_email, args.license_key)

    lut_arg = str(args.lut).strip()
    if lut_arg and not Path(lut_arg).exists():
        raise FileNotFoundError(lut_arg)

    config = _session_config_from_args(
        args,
        codec=codec,
        encoder_settings=encoder_settings,
        detection_model_name=detection_model_name,
        detection_model_path=detection_model_path,
        detection_score_threshold=detection_score_threshold,
        restoration_model_path=restoration_model_path,
        lut_path=lut_arg or None,
    )

    _ensure_ltx_model(config, segments)

    from jasna.session_factory import build_pipeline, build_restoration_session

    with device_context(config.device):
        session = build_restoration_session(config, log_callback=None)

        def make_pipeline(vid_input: Path, out_path: Path):
            return build_pipeline(config, session, vid_input, out_path, segments=segments, splice_plan=splice_plan)

        def output_path_for(vid: Path) -> Path:
            if input_is_dir:
                return folder_output_path(folder_output_dir, vid, args.output_pattern)
            return output_video

        try:
            if is_streaming:
                _run_streaming(args, make_pipeline, input_video)
                return
            all_commands_ok = _run_videos(
                args,
                make_pipeline,
                folder_videos if input_is_dir else [input_video],
                output_path_for,
                in_folder=input_is_dir,
                first_index=len(folder_images) + 1,
                total=len(folder_images) + len(folder_videos) if input_is_dir else 1,
            )
            _run_post_export_action()
            if not all_commands_ok:
                sys.exit(1)
        except UnsupportedColorspaceError as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            session.close()


if __name__ == "__main__":
    main()

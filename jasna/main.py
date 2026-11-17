import argparse
import logging
import os
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from jasna.cli_activation import run_activation_options
from jasna.cli_downloads import ensure_dlss5_runtime, ensure_ltx_model
from jasna.cli_parser import BASICVSRPP_ONLY_ARG_DESTS, CLI_RESTORATION_MODELS, LTX_ONLY_ARG_DESTS, build_parser
from jasna.engine_paths import default_restoration_model_path
from jasna.media.probe import UnsupportedVideoError
from jasna.os_utils import check_amd_amf_runtime, confirm_in_terminal, escape_unencodable_output
from jasna.system_requirements import (
    AMD_DRIVER_URL,
    AMF_LINUX_INSTALL_URL,
    LTX_MIN_LINUX_KERNEL,
    MIN_DRIVER_VERSION,
    check_ascii_install_path,
    check_gpu_driver_version,
    check_required_executables,
    check_supported_gpu,
    check_windows_nvidia_sysmem_fallback_policy,
    gpu_check_error,
    install_on_hard_disk,
    leftover_install_dll,
    old_linux_kernel_for_ltx,
)
from jasna.session_config import (
    DEFAULT_CODEC,
    DecodeBackendName,
    SessionConfig,
    ltx_unsupported_options,
)

if TYPE_CHECKING:
    from jasna.session_factory import SegmentJob


def batch_size_from_args(args: argparse.Namespace) -> int:
    from jasna.mosaic.detection_registry import default_detection_batch_size

    return int(args.batch_size) if args.batch_size is not None else default_detection_batch_size(args.device)


def _env_choice(name: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, "auto")
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(choices)}")
    return value


def decode_backend_from_env() -> DecodeBackendName:
    return _env_choice("JASNA_DECODE_BACKEND", get_args(DecodeBackendName))


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
    from jasna.ltx.windows import window_hop

    return SessionConfig(
        device=str(args.device),
        fp16=bool(args.fp16),
        batch_size=batch_size_from_args(args),
        detection_model_name=detection_model_name,
        detection_model_path=detection_model_path,
        detection_score_threshold=detection_score_threshold,
        max_detection_gap=int(args.max_detection_gap),
        min_detection_duration=int(args.min_detection_duration),
        scene_detection=bool(args.scene_detection),
        restoration_model_name=CLI_RESTORATION_MODELS[args.restoration_model_name][0],
        restoration_model_path=restoration_model_path,
        ltx_large_canvas=bool(args.ltx_large_canvas),
        ltx_window_hop=window_hop(int(args.ltx_overlap)),
        ltx_seed=int(args.ltx_seed),
        ltx_fast=bool(args.ltx_fast),
        ltx_model=CLI_RESTORATION_MODELS[args.restoration_model_name][1],
        ltx_vae_encoder=str(args.ltx_vae_encoder),
        ltx_vae_decoder=str(args.ltx_vae_decoder),
        ltx_trial=bool(args.ltx_trial),
        ltx_auto_split_percent=args.ltx_auto_split,
        compile_basicvsrpp=bool(args.compile_basicvsrpp),
        max_clip_size=int(args.max_clip_size),
        temporal_overlap=int(args.temporal_overlap),
        denoise_strength=str(args.denoise).lower(),
        secondary_restoration=str(args.secondary_restoration).lower(),
        tvai_ffmpeg_path=str(args.tvai_ffmpeg_path),
        tvai_model=str(args.tvai_model),
        tvai_scale=int(args.tvai_scale),
        tvai_args=str(args.tvai_args),
        tvai_workers=int(args.tvai_workers),
        tvai_denoise=bool(args.tvai_denoise),
        decode_backend=decode_backend_from_env(),
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
        frame_generation=args.frame_generation,
        upscale_model=args.upscale,
        upscale_height=int(args.upscale_height),
        upscale_strength=float(args.upscale_strength),
        upscale_rtx_quality=args.upscale_rtx_quality,
        upscale_rtx_denoise=args.upscale_rtx_denoise,
        upscale_rtx_deblur=args.upscale_rtx_deblur,
        upscale_dlss5_preset=args.upscale_dlss5_preset,
        upscale_dlss5_style=args.upscale_dlss5_style,
        upscale_dlss5_intensity=float(args.upscale_dlss5_intensity),
        fmp4=bool(args.fmp4),
        disable_progress=bool(args.no_progress),
        working_dir=Path(args.working_directory) if args.working_directory else None,
    )


@contextmanager
def _usage_errors(parser: argparse.ArgumentParser):
    try:
        yield
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


def _check_system(args: argparse.Namespace) -> None:
    path_ok, path_info = check_ascii_install_path()
    if not path_ok:
        print("Error: Jasna must be installed in a path with ASCII characters only.")
        print(f"Current path: {path_info}")
        sys.exit(1)

    leftover = leftover_install_dll()
    if leftover is not None:
        print(f"Error: the Jasna folder still has files from an older version, such as {leftover}. "
              "Extract Jasna into a new, empty folder, then move the model_weights folder there.")
        sys.exit(1)

    check_required_executables()

    driver_ok, driver_info = check_gpu_driver_version()
    if driver_ok is False:
        print(f"Error: GPU driver version check failed: {driver_info}")
        if "ROCm" not in driver_info:
            print(f"Please update your NVIDIA driver to version {MIN_DRIVER_VERSION} or newer.")
        sys.exit(1)
    if driver_ok is None:
        print(f"Warning: {driver_info}")

    gpu_ok, gpu_result = check_supported_gpu(str(args.device))
    if not gpu_ok:
        print(f"Error: {gpu_check_error(gpu_result)}")
        sys.exit(1)

    from jasna.accelerator import is_amd_device, is_nvidia_device

    if is_amd_device():
        amf_ok, amf_info = check_amd_amf_runtime()
        if not amf_ok:
            print(f"Error: AMD's AMF video runtime is missing, so Jasna can't decode or encode video ({amf_info}).")
            if sys.platform == "win32":
                print(f"Update or reinstall your AMD graphics driver: {AMD_DRIVER_URL}")
            else:
                print(f"Install it with AMD's AMF installer: open the newest version folder at {AMF_LINUX_INSTALL_URL}, "
                      "download amf_installer_<version>.zip, unpack it and run sudo ./amf_installer_<version>.sh")
            sys.exit(1)

    if sys.platform == "win32" and is_nvidia_device():
        sysmem_ok, sysmem_info = check_windows_nvidia_sysmem_fallback_policy()
        if not sysmem_ok:
            print(f"Warning: CUDA Sysmem Fallback Policy: {sysmem_info}")

    if install_on_hard_disk():
        print("Warning: Jasna is installed on a hard disk drive (HDD), so it starts and loads models slowly. "
              "Move the Jasna folder to an SSD.")


def _configure_logging(level: int) -> None:
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().setLevel(level)
    from jasna.media.ffmpeg_log import log_ffmpeg_errors
    log_ffmpeg_errors()


def _configure_runtime(args: argparse.Namespace) -> None:
    _configure_logging(getattr(logging, args.log_level.upper()))
    from jasna.bootstrap import configure_process

    configure_process()


def _plan_folder(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path | None,
) -> tuple[list[Path], list[Path], Path]:
    """Validate a folder batch and return its images, videos and output folder."""
    from jasna.media.media_files import classify_folder, folder_output_path, is_media, path_collision_key

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
    input_keys = {path_collision_key(path) for path in inputs}
    for path in inputs:
        out_path = folder_output_path(output_dir, path, args.output_pattern)
        out_key = path_collision_key(out_path)
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


def _plan_segments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: SessionConfig,
    input_video: Path,
    output_video: Path,
    *,
    codec_was_explicit: bool,
) -> "SegmentJob":
    """The job of --segments; its output keeps the input codec."""
    from jasna.media.splice import SmartRenderCompatibilityError
    from jasna.segments import parse_segments
    from jasna.session_factory import plan_segment_job, splice_source

    try:
        spliced = splice_source(config, input_video, output_video)
    except SmartRenderCompatibilityError as exc:
        parser.error(str(exc))
    if codec_was_explicit and config.codec != spliced.codec:
        parser.error(f"with --segments output codec must match input; pass --codec {spliced.codec}")
    try:
        segments = parse_segments(str(args.segments).strip(), duration=spliced.metadata.duration)
    except ValueError as exc:
        parser.error(f"invalid --segments: {exc}")
    try:
        return plan_segment_job(config, input_video, spliced, segments)
    except SmartRenderCompatibilityError as exc:
        parser.error(str(exc))


def _restore_images(args: argparse.Namespace, jobs: list[tuple[Path, Path]], *, folder_total: int | None) -> None:
    """Restore each ``(input, output)`` still image with one model load; a folder batch
    (``folder_total`` inputs, its images first) prints its progress."""
    import torch

    from jasna.engine_paths import SD15_DIR
    from jasna.image_restore import ImageRestoreOptions, restore_image_file
    from jasna.mosaic.detection_registry import resolve_detection_model
    from jasna.restorer.sd15_download import ensure_sd15_bundle
    from jasna.session_factory import build_image_session

    ensure_sd15_bundle(SD15_DIR)
    detection_model_name, detection_model_path, score_threshold = resolve_detection_model(
        str(args.detection_model), str(args.detection_model_path), args.detection_score_threshold
    )
    options = ImageRestoreOptions(
        steps=int(args.sd15_steps),
        strength=float(args.sd15_strength),
        seed=int(args.sd15_seed),
        variants=int(args.sd15_variants),
        freeu=bool(args.sd15_freeu),
    )
    session = build_image_session(
        detection_model_name,
        detection_model_path,
        device=torch.device(str(args.device)),
        batch_size=batch_size_from_args(args),
        fp16=bool(args.fp16),
        score_threshold=score_threshold,
        log_callback=None,
    )
    try:
        for index, (input_path, output_base) in enumerate(jobs, start=1):
            if folder_total is not None:
                print(f"[{index}/{folder_total}] Processing {input_path.name} -> {output_base.name}")
            for path in restore_image_file(session, options, input_path, output_base):
                logging.getLogger(__name__).info("Wrote %s", path)
    finally:
        session.close()


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
            except UnsupportedVideoError as e:
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
    from jasna.job_resume import ResumeMismatchError, run_with_resume_prompt
    from jasna.post_export_action import PostExportVideoCommandError, run_post_export_video_command

    post_export_video_command = str(args.post_export_video_command).strip()
    all_commands_ok = True
    for i, vid in enumerate(videos, start=first_index):
        out_path = output_path_for(vid)
        if in_folder:
            print(f"[{i}/{total}] Processing {vid.name} -> {out_path.name}")
        try:
            run_with_resume_prompt(lambda: make_pipeline(vid, out_path).run(), _ask_to_replace_progress)
        except UnsupportedVideoError as e:
            print(f"Error processing {vid.name}: {e}")
            if not in_folder:
                sys.exit(1)
            continue
        except ResumeMismatchError as e:
            all_commands_ok = False
            print(f"Error processing {vid.name}: {e}. Run jasna in a terminal to replace it, or delete that folder.")
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


def _ask_to_replace_progress(error) -> bool:
    """Without a terminal to ask in, saved progress is kept."""
    return confirm_in_terminal(f"{error}\nReplace saved progress and start over? [y/N]: ") is True


def _reject_options_of_other_restoration_backend(
    parser: argparse.ArgumentParser, args: argparse.Namespace, restoration_backend: str, *, auto_split: bool
) -> None:
    if restoration_backend != "ltx":
        foreign_dests, needed = LTX_ONLY_ARG_DESTS, "ltx-basic, ltx, ltx-4step, ltx-8step or ltx-undistilled"
    elif auto_split:  # its smaller mosaics run BasicVSR++
        return
    else:
        foreign_dests, needed = BASICVSRPP_ONLY_ARG_DESTS, "basicvsrpp or --ltx-auto-split"
    for dest in foreign_dests:
        if getattr(args, dest) != parser.get_default(dest):
            option = "--" + dest.replace("_", "-")
            parser.error(f"{option} needs --restoration-model-name {needed}")


@dataclass(frozen=True)
class _CliPaths:
    input: Path | None
    output: Path | None
    lut: str | None
    input_is_image: bool
    input_is_dir: bool


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> _CliPaths:
    """Option checks that need no GPU or runtime setup; each mistake is a usage error."""
    is_streaming = bool(args.stream)
    if is_streaming and args.retarget_high_fps:
        parser.error("--retarget-high-fps is only supported for offline exports")
    if is_streaming and args.fmp4:
        parser.error("--fmp4 is only supported for offline exports")
    if is_streaming and args.frame_generation != "none":
        parser.error("--frame-generation is only supported for offline exports")
    if is_streaming and args.upscale != "none":
        parser.error("--upscale is only supported for offline exports")
    from jasna.post_export_action import validate_post_export_action
    with _usage_errors(parser):
        validate_post_export_action(str(args.post_export_action), str(args.post_export_command))
    if args.input is None and not is_streaming:
        parser.error("--input is required when not using --benchmark or --stream")
    if args.output is None and not is_streaming:
        parser.error("--output is required when not using --benchmark or --stream")

    input_video = Path(args.input) if args.input else None
    if input_video is not None and not input_video.exists():
        parser.error(f"input not found: {input_video}")
    output_video = Path(args.output) if args.output else (input_video.with_stem(input_video.stem + "_out") if input_video else None)
    lut_arg = str(args.lut).strip()
    if lut_arg and not Path(lut_arg).exists():
        parser.error(f"--lut file not found: {lut_arg}")

    from jasna.media.media_files import is_image, path_collision_key
    input_is_image = input_video is not None and is_image(input_video)
    input_is_dir = input_video is not None and input_video.is_dir()
    if (
        input_video is not None and output_video is not None and not input_is_dir
        and path_collision_key(output_video) == path_collision_key(input_video)
    ):
        parser.error("--output must not be the --input file")
    segments_spec = str(args.segments).strip()
    auto_split = args.ltx_auto_split is not None
    if auto_split:
        if segments_spec:
            parser.error("--segments cannot be combined with --ltx-auto-split")
        if not 0 < args.ltx_auto_split <= 100:
            parser.error("--ltx-auto-split must be a percentage above 0 and at most 100")
        if args.fmp4:
            parser.error("--fmp4 cannot be combined with --ltx-auto-split")
        if args.frame_generation != "none":
            parser.error("--frame-generation cannot be combined with --ltx-auto-split")
        if args.upscale != "none":
            parser.error("--upscale cannot be combined with --ltx-auto-split")
    if segments_spec:
        if is_streaming:
            parser.error("--segments cannot be combined with --stream")
        if input_is_image:
            parser.error("--segments requires a single video input, not an image")
        if input_is_dir:
            parser.error("--segments requires a single video input, not a folder")
        if args.fmp4:
            parser.error("--fmp4 cannot be combined with --segments")
        if args.frame_generation != "none":
            parser.error("--frame-generation cannot be combined with --segments")
        if args.upscale != "none":
            parser.error("--upscale cannot be combined with --segments")
    if is_streaming and input_is_dir:
        parser.error("--stream does not support folder input")
    if is_streaming and input_is_image:
        parser.error("Image input does not support --stream")

    restoration_backend = CLI_RESTORATION_MODELS[args.restoration_model_name][0]
    if not input_is_image:
        if restoration_backend == "ltx":
            model_option = f"--restoration-model-name {args.restoration_model_name}"
            if is_streaming:
                parser.error(f"{model_option} does not support --stream")
            if (kernel := old_linux_kernel_for_ltx()) is not None:
                minimum = ".".join(map(str, LTX_MIN_LINUX_KERNEL))
                parser.error(f"LTX needs Linux kernel {minimum} or newer; this system runs {kernel}. "
                             "Update the system to use LTX.")
            unsupported = ltx_unsupported_options(
                auto_split=auto_split, secondary_restoration=args.secondary_restoration, denoise_strength=args.denoise
            )
            if "secondary_restoration" in unsupported:
                parser.error(
                    f"{model_option} supports --secondary-restoration {args.secondary_restoration} "
                    "only with --ltx-auto-split"
                )
            if "denoise_strength" in unsupported:
                parser.error(f"{model_option} supports --denoise only with --ltx-auto-split")
        _reject_options_of_other_restoration_backend(parser, args, restoration_backend, auto_split=auto_split)

    return _CliPaths(input_video, output_video, lut_arg or None, input_is_image, input_is_dir)


def _stop_like_ctrl_c(signum, frame) -> None:
    raise KeyboardInterrupt


def stop_on_termination_signals() -> None:
    """SIGTERM, and on Windows Ctrl+Break (the only signal a parent can send a console process group), stop Jasna
    like Ctrl+C, so exit handlers still release resources."""
    signal.signal(signal.SIGTERM, _stop_like_ctrl_c)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _stop_like_ctrl_c)


def main() -> None:
    stop_on_termination_signals()
    try:
        _main()
    except KeyboardInterrupt:
        print("Stopped")
        sys.exit(130)


def _main() -> None:
    escape_unencodable_output(sys.stdout, sys.stderr)
    parser = build_parser()
    args = parser.parse_args()
    if run_activation_options(parser, args, _configure_logging):
        return

    if not args.benchmark:
        from jasna.license_status import start_pc_license_refresh

        start_pc_license_refresh()
    if args.benchmark:
        from jasna.benchmark import run_benchmark_cli

        _configure_runtime(args)
        run_benchmark_cli(args)
        return

    paths = _validate_args(parser, args)
    restoration_backend = CLI_RESTORATION_MODELS[args.restoration_model_name][0]

    from jasna.media.media_files import folder_output_path
    from jasna.post_export_action import run_post_export_action_safely

    def _run_post_export_action() -> None:
        run_post_export_action_safely(
            str(args.post_export_action),
            str(args.post_export_command),
            lambda message: print(f"Warning: {message}"),
        )

    _configure_runtime(args)
    _check_system(args)
    if args.frame_generation != "none" and not paths.input_is_image:
        from jasna.accelerator import supports_frame_generation
        from jasna.session_factory import FRAME_GENERATION_NEEDS_RTX_40

        if not supports_frame_generation(args.device):
            parser.error(FRAME_GENERATION_NEEDS_RTX_40)

    folder_images: list[Path] = []
    folder_videos: list[Path] = []
    folder_output_dir: Path | None = None
    if paths.input_is_dir:
        folder_images, folder_videos, folder_output_dir = _plan_folder(parser, args, paths.input, paths.output)
        if folder_images:
            _restore_images(
                args,
                [(image, folder_output_path(folder_output_dir, image, args.output_pattern)) for image in folder_images],
                folder_total=len(folder_images) + len(folder_videos),
            )
        if not folder_videos:
            _run_post_export_action()
            return

    if paths.input_is_image:
        _restore_images(args, [(paths.input, paths.output)], folder_total=None)
        _run_post_export_action()
        return

    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        discover_available_detection_models,
        resolve_detection_model,
    )

    with _usage_errors(parser):
        if not str(args.detection_model_path).strip():
            available = discover_available_detection_models()
            requested = coerce_detection_model_name(str(args.detection_model))
            if available and requested not in available:
                print(f"Warning: detection model '{requested}' not found in model_weights/. Available: {', '.join(available)}")
        detection_model_name, detection_model_path, detection_score_threshold = resolve_detection_model(
            str(args.detection_model), str(args.detection_model_path), args.detection_score_threshold
        )

    restoration_model_path = Path(
        args.restoration_model_path or default_restoration_model_path(restoration_backend)
    )
    if restoration_backend != "ltx" and not restoration_model_path.exists():
        parser.error(f"restoration model not found: {restoration_model_path}")

    codec = DEFAULT_CODEC if args.codec is None else args.codec
    from jasna.accelerator import device_context, vendor_for_device

    with _usage_errors(parser):
        from jasna.media.encoder_settings import resolve_custom_encoder_settings

        encoder_settings = resolve_custom_encoder_settings(
            str(args.encoder_settings),
            cq=args.cq,
            codec=codec,
            vendor=vendor_for_device(str(args.device)),
        )
        config = _session_config_from_args(
            args,
            codec=codec,
            encoder_settings=encoder_settings,
            detection_model_name=detection_model_name,
            detection_model_path=detection_model_path,
            detection_score_threshold=detection_score_threshold,
            restoration_model_path=restoration_model_path,
            lut_path=paths.lut,
        )

    segment_job = (
        _plan_segments(parser, args, config, paths.input, paths.output, codec_was_explicit=args.codec is not None)
        if str(args.segments).strip() else None
    )
    ensure_ltx_model(config, segment_job.splice_plan.segments if segment_job is not None else None)
    ensure_dlss5_runtime(config)

    from jasna.session_factory import build_job_pipeline, build_restoration_session, plan_job

    with device_context(config.device):
        session = build_restoration_session(config, log_callback=None)

        def make_pipeline(vid_input: Path, out_path: Path):
            job = plan_job(config, session, vid_input, out_path, segment_job, progress_callback=None)
            return build_job_pipeline(config, session, vid_input, out_path, job, progress_callback=None)

        def output_path_for(vid: Path) -> Path:
            if paths.input_is_dir:
                return folder_output_path(folder_output_dir, vid, args.output_pattern)
            return paths.output

        try:
            if args.stream:
                _run_streaming(args, make_pipeline, paths.input)
                return
            all_commands_ok = _run_videos(
                args,
                make_pipeline,
                folder_videos if paths.input_is_dir else [paths.input],
                output_path_for,
                in_folder=paths.input_is_dir,
                first_index=len(folder_images) + 1,
                total=len(folder_images) + len(folder_videos) if paths.input_is_dir else 1,
            )
            _run_post_export_action()
            if not all_commands_ok:
                sys.exit(1)
        except UnsupportedVideoError as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            session.close()


if __name__ == "__main__":
    main()

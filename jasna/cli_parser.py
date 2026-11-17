"""The ``jasna`` command line: options, their defaults and help (``jasna.cli_help``)."""

import argparse
from typing import get_args

from jasna import __version__
from jasna.cli_help import CLI_HELP
from jasna.session_config import (
    DEFAULT_DETECTION_MODEL_NAME,
    DEFAULT_MAX_CLIP_SIZE,
    DEFAULT_MAX_DETECTION_GAP,
    DEFAULT_MIN_DETECTION_DURATION,
    DEFAULT_TEMPORAL_OVERLAP,
    DEFAULT_UPSCALE_HEIGHT,
    DEFAULT_UPSCALE_STRENGTH,
    LTX_DEFAULT_MODEL,
    LTX_DEFAULT_OVERLAP,
    LTX_DEFAULT_SEED,
    LTX_OVERLAP_CHOICES,
    RTX_DEFAULT_DEBLUR,
    RTX_DEFAULT_DENOISE,
    RTX_DEFAULT_QUALITY,
    RTX_DEFAULT_SCALE,
    SD15_DEFAULT_SEED,
    SD15_DEFAULT_STEPS,
    SD15_DEFAULT_STRENGTH,
    SD15_DEFAULT_VARIANTS,
    TVAI_DEFAULT_ARGS,
    TVAI_DEFAULT_FFMPEG_PATH,
    TVAI_DEFAULT_MODEL,
    TVAI_DEFAULT_SCALE,
    TVAI_DEFAULT_WORKERS,
    UPSCALE_DLSS5_DEFAULT_INTENSITY,
    UPSCALE_DLSS5_DEFAULT_PRESET,
    UPSCALE_DLSS5_DEFAULT_STYLE,
    UPSCALE_HEIGHTS,
    UPSCALE_RTX_DEFAULT_DEBLUR,
    UPSCALE_RTX_DEFAULT_DENOISE,
    UPSCALE_RTX_DEFAULT_QUALITY,
    Dlss5PresetName,
    Dlss5StyleName,
    LtxModelName,
    LtxVaeName,
    RestorationModelName,
    RtxLevelName,
    RtxQualityName,
    UpscaleModelName,
)


# CLI restoration model name -> (restoration backend, LTX model variant).
CLI_RESTORATION_MODELS: dict[str, tuple[RestorationModelName, LtxModelName]] = {
    "basicvsrpp": ("basicvsrpp", LTX_DEFAULT_MODEL),
    "ltx-basic": ("ltx", "distilled1"),
    "ltx": ("ltx", "distilled2"),
    "ltx-4step": ("ltx", "distilled4"),
    "ltx-8step": ("ltx", "distilled"),
    "ltx-undistilled": ("ltx", "undistilled"),
}

class _ShortErrorParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.exit(2, f"{self.prog}: error: {message}\nRun '{self.prog} --help' to see all options.\n")

def build_parser() -> argparse.ArgumentParser:
    parser = _ShortErrorParser(prog="jasna")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--benchmark", action="store_true", help=CLI_HELP["benchmark"])
    parser.add_argument("--input", required=False, type=str, default=None, help=CLI_HELP["input"])
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=None,
        help=CLI_HELP["output"],
    )
    parser.add_argument(
        "--output-pattern",
        type=str,
        default=None,
        help=(
            CLI_HELP["output_pattern"]
        ),
    )
    parser.add_argument(
        "--working-directory",
        type=str,
        default=None,
        help=CLI_HELP["working_directory"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
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
        help=CLI_HELP["log_level"],
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=CLI_HELP["no_progress"],
    )

    restoration = parser.add_argument_group("Restoration")
    restoration.add_argument(
        "--restoration-model-name",
        type=str,
        default="basicvsrpp",
        choices=list(CLI_RESTORATION_MODELS),
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
        "--ltx-overlap",
        type=int,
        default=LTX_DEFAULT_OVERLAP,
        choices=LTX_OVERLAP_CHOICES,
        help=CLI_HELP["ltx_overlap"],
    )
    for part in ("encoder", "decoder"):
        restoration.add_argument(
            f"--ltx-vae-{part}",
            default="auto",
            choices=get_args(LtxVaeName),
            help=CLI_HELP[f"ltx_vae_{part}"],
        )
    restoration.add_argument(
        "--ltx-seed",
        type=int,
        default=LTX_DEFAULT_SEED,
        help=CLI_HELP["ltx_seed"],
    )
    restoration.add_argument(
        "--ltx-fast",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["ltx_fast"],
    )
    restoration.add_argument(
        "--ltx-trial",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["ltx_trial"],
    )
    restoration.add_argument(
        "--ltx-auto-split",
        type=float,
        default=None,
        metavar="PERCENT",
        help=CLI_HELP["ltx_auto_split"],
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
        default=DEFAULT_MAX_CLIP_SIZE,
        help=CLI_HELP["max_clip_size"],
    )
    restoration.add_argument(
        "--temporal-overlap",
        type=int,
        default=DEFAULT_TEMPORAL_OVERLAP,
        help=CLI_HELP["temporal_overlap"],
    )
    restoration.add_argument(
        "--denoise",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help=CLI_HELP["denoise"],
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
        help=CLI_HELP["image_restoration_model_name"],
    )
    sd15.add_argument(
        "--sd15-steps",
        type=int,
        default=SD15_DEFAULT_STEPS,
        help=CLI_HELP["sd15_steps"],
    )
    sd15.add_argument(
        "--sd15-strength",
        type=float,
        default=SD15_DEFAULT_STRENGTH,
        help=CLI_HELP["sd15_strength"],
    )
    sd15.add_argument(
        "--sd15-freeu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=CLI_HELP["sd15_freeu"],
    )
    sd15.add_argument(
        "--sd15-seed",
        type=int,
        default=SD15_DEFAULT_SEED,
        help=CLI_HELP["sd15_seed"],
    )
    sd15.add_argument(
        "--sd15-variants",
        type=int,
        default=SD15_DEFAULT_VARIANTS,
        help=CLI_HELP["sd15_variants"]    )

    license_group = parser.add_argument_group("Activation")
    license_group.add_argument(
        "--license-code",
        type=str,
        help=CLI_HELP["license_code"],
    )
    license_group.add_argument(
        "--online",
        action="store_true",
        help=CLI_HELP["online"],
    )
    license_group.add_argument(
        "--activation-status",
        action="store_true",
        help=CLI_HELP["activation_status"],
    )

    rtx = parser.add_argument_group("RTX Super Res")
    rtx.add_argument(
        "--rtx-scale",
        type=int,
        default=RTX_DEFAULT_SCALE,
        choices=[2, 4],
        help=CLI_HELP["rtx_scale"],
    )
    rtx.add_argument(
        "--rtx-quality",
        type=str,
        default=RTX_DEFAULT_QUALITY,
        choices=["low", "medium", "high", "ultra"],
        help=CLI_HELP["rtx_quality"],
    )
    rtx.add_argument(
        "--rtx-denoise",
        type=str,
        default=RTX_DEFAULT_DENOISE,
        choices=["none", "low", "medium", "high", "ultra"],
        help=CLI_HELP["rtx_denoise"],
    )
    rtx.add_argument(
        "--rtx-deblur",
        type=str,
        default=RTX_DEFAULT_DEBLUR,
        choices=["none", "low", "medium", "high", "ultra"],
        help=CLI_HELP["rtx_deblur"],
    )

    upscale = parser.add_argument_group("Upscale whole video")
    upscale.add_argument(
        "--upscale",
        type=str,
        default="none",
        choices=list(get_args(UpscaleModelName)),
        help=CLI_HELP["upscale"],
    )
    upscale.add_argument(
        "--upscale-height",
        type=int,
        default=DEFAULT_UPSCALE_HEIGHT,
        choices=list(UPSCALE_HEIGHTS),
        help=CLI_HELP["upscale_height"],
    )
    upscale.add_argument(
        "--upscale-strength",
        type=float,
        default=DEFAULT_UPSCALE_STRENGTH,
        help=CLI_HELP["upscale_strength"],
    )
    upscale.add_argument(
        "--upscale-rtx-quality",
        type=str,
        default=UPSCALE_RTX_DEFAULT_QUALITY,
        choices=list(get_args(RtxQualityName)),
        help=CLI_HELP["upscale_rtx_quality"],
    )
    upscale.add_argument(
        "--upscale-rtx-denoise",
        type=str,
        default=UPSCALE_RTX_DEFAULT_DENOISE,
        choices=list(get_args(RtxLevelName)),
        help=CLI_HELP["upscale_rtx_denoise"],
    )
    upscale.add_argument(
        "--upscale-rtx-deblur",
        type=str,
        default=UPSCALE_RTX_DEFAULT_DEBLUR,
        choices=list(get_args(RtxLevelName)),
        help=CLI_HELP["upscale_rtx_deblur"],
    )
    upscale.add_argument(
        "--upscale-dlss5-preset",
        type=str,
        default=UPSCALE_DLSS5_DEFAULT_PRESET,
        choices=list(get_args(Dlss5PresetName)),
        help=CLI_HELP["upscale_dlss5_preset"],
    )
    upscale.add_argument(
        "--upscale-dlss5-style",
        type=str,
        default=UPSCALE_DLSS5_DEFAULT_STYLE,
        choices=list(get_args(Dlss5StyleName)),
        help=CLI_HELP["upscale_dlss5_style"],
    )
    upscale.add_argument(
        "--upscale-dlss5-intensity",
        type=float,
        default=UPSCALE_DLSS5_DEFAULT_INTENSITY,
        help=CLI_HELP["upscale_dlss5_intensity"],
    )

    tvai = parser.add_argument_group("Topaz Video")
    tvai.add_argument(
        "--tvai-ffmpeg-path",
        type=str,
        default=TVAI_DEFAULT_FFMPEG_PATH,
        help=CLI_HELP["tvai_ffmpeg_path"],
    )
    tvai.add_argument(
        "--tvai-model",
        type=str,
        default=TVAI_DEFAULT_MODEL,
        help=CLI_HELP["tvai_model"],
    )
    tvai.add_argument(
        "--tvai-scale",
        type=int,
        default=TVAI_DEFAULT_SCALE,
        choices=[1, 2, 4],
        help=CLI_HELP["tvai_scale"],
    )
    tvai.add_argument(
        "--tvai-args",
        type=str,
        default=TVAI_DEFAULT_ARGS,
        help=CLI_HELP["tvai_args"],
    )
    tvai.add_argument(
        "--tvai-workers",
        type=int,
        default=TVAI_DEFAULT_WORKERS,
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
        default=DEFAULT_DETECTION_MODEL_NAME,
        help=(
            CLI_HELP["detection_model"]
        ),
    )
    detection.add_argument(
        "--detection-model-path",
        type=str,
        default="",
        help=(
            CLI_HELP["detection_model_path"]
        ),
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
        default=DEFAULT_MAX_DETECTION_GAP,
        help=CLI_HELP["max_detection_gap"],
    )
    detection.add_argument(
        "--min-detection-duration",
        type=int,
        default=DEFAULT_MIN_DETECTION_DURATION,
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
        choices=["auto", "off", "sbs", "sbs-fisheye", "tb"],
        help=CLI_HELP["vr_mode"],
    )

    streaming = parser.add_argument_group("Streaming")
    streaming.add_argument(
        "--stream",
        action="store_true",
        help=CLI_HELP["stream"],
    )
    streaming.add_argument(
        "--stream-port",
        type=int,
        default=8765,
        help=CLI_HELP["stream_port"],
    )
    streaming.add_argument(
        "--stream-segment-duration",
        type=float,
        default=4.0,
        help=CLI_HELP["stream_segment_duration"],
    )
    streaming.add_argument(
        "--no-browser",
        action="store_true",
        help=CLI_HELP["no_browser"],
    )

    encoding = parser.add_argument_group("Encoding")
    encoding.add_argument(
        "--codec",
        type=lambda value: str(value).lower(),
        default=None,
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
        help=CLI_HELP["lut"],
    )
    encoding.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help=(
            CLI_HELP["sharpen"]
        ),
    )
    encoding.add_argument(
        "--retarget-high-fps",
        action="store_true",
        help=(
            CLI_HELP["retarget_high_fps"]
        ),
    )
    encoding.add_argument(
        "--frame-generation",
        type=str,
        default="none",
        choices=["none", "2x"],
        help=CLI_HELP["frame_generation"],
    )
    encoding.add_argument(
        "--fmp4",
        action="store_true",
        help=(
            CLI_HELP["fmp4"]
        ),
    )
    encoding.add_argument(
        "--segments",
        type=str,
        default="",
        help=(
            CLI_HELP["segments"]
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
        help=CLI_HELP["post_export_command"],
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
        help=CLI_HELP["benchmark_filter"],
    )
    benchmark_group.add_argument(
        "--benchmark-video",
        type=str,
        action="append",
        default=None,
        help=CLI_HELP["benchmark_video"],
    )
    return parser

LTX_ONLY_ARG_DESTS = (
    "ltx_fast", "ltx_large_canvas", "ltx_overlap", "ltx_vae_encoder", "ltx_vae_decoder", "ltx_seed", "ltx_trial",
    "ltx_auto_split",
)

BASICVSRPP_ONLY_ARG_DESTS = ("compile_basicvsrpp",)

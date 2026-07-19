from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

if TYPE_CHECKING:
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

logger = logging.getLogger(__name__)

# ffmpeg *_nvenc option names shared by every codec
_COMMON_ENCODER_SETTINGS: frozenset[str] = frozenset(
    {
        "preset",
        "tune",
        "rc",
        "cq",
        "qmin",
        "qmax",
        "nonref_p",
        "g",
        "temporal-aq",
        "rc-lookahead",
        "lookahead_level",
        "aq-strength",
        "init_qpI",
        "init_qpP",
        "init_qpB",
        "bf",
        "b_ref_mode",
        "maxrate",
        "bufsize",
        "multipass",
        "b_adapt",
        "weighted_pred",
        "tf_level",
    }
)

# hevc_nvenc/h264_nvenc accept both AQ spellings; av1_nvenc only the hyphen one.
SUPPORTED_ENCODER_SETTINGS_BY_CODEC: dict[str, frozenset[str]] = {
    "hevc": _COMMON_ENCODER_SETTINGS | {"profile", "tier", "spatial_aq", "spatial-aq"},
    "h264": _COMMON_ENCODER_SETTINGS | {"profile", "coder", "spatial_aq", "spatial-aq"},
    "av1": _COMMON_ENCODER_SETTINGS | {"tier", "spatial-aq", "tile-rows", "tile-columns"},
}

SUPPORTED_ENCODER_SETTINGS: frozenset[str] = frozenset().union(
    *SUPPORTED_ENCODER_SETTINGS_BY_CODEC.values()
)

# User-facing AMF settings. ``cq`` is kept as a portable Jasna option and is
# translated to AMF's qvbr_quality_level by the encoder.
_COMMON_AMF_ENCODER_SETTINGS: frozenset[str] = frozenset(
    {
        "preset",
        "usage",
        "quality",
        "rc",
        "cq",
        "qvbr_quality_level",
        "g",
        "bf",
        "preanalysis",
        "vbaq",
        "maxrate",
        "bufsize",
        "profile",
        "level",
    }
)

AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC: dict[str, frozenset[str]] = {
    "hevc": _COMMON_AMF_ENCODER_SETTINGS | {"tier", "bitdepth"},
    "h264": _COMMON_AMF_ENCODER_SETTINGS | {"coder"},
    "av1": _COMMON_AMF_ENCODER_SETTINGS | {"bitdepth"},
}

AMF_SUPPORTED_ENCODER_SETTINGS: frozenset[str] = frozenset().union(
    *AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC.values()
)


def _parse_encoder_setting_scalar(value: str) -> object:
    v = value.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_encoder_settings(value: str) -> dict[str, object]:
    value = (value or "").strip()
    if value == "":
        return {}

    if value.startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("--encoder-settings JSON must be an object")
        return parsed

    settings: dict[str, object] = {}
    for part in value.split(","):
        part = part.strip()
        if part == "":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --encoder-settings item: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        k = k.strip()
        if k == "":
            raise ValueError(f"Invalid --encoder-settings item: {part!r} (empty key)")
        settings[k] = _parse_encoder_setting_scalar(v)

    return settings


def validate_encoder_settings(
    settings: dict[str, object],
    codec: str | None = None,
    *,
    vendor: AcceleratorVendor | str | None = None,
) -> dict[str, object]:
    resolved_vendor = (
        vendor_for_device()
        if vendor is None
        else AcceleratorVendor(str(vendor))
    )
    by_codec = (
        AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC
        if resolved_vendor is AcceleratorVendor.AMD
        else SUPPORTED_ENCODER_SETTINGS_BY_CODEC
    )
    supported_all = (
        AMF_SUPPORTED_ENCODER_SETTINGS
        if resolved_vendor is AcceleratorVendor.AMD
        else SUPPORTED_ENCODER_SETTINGS
    )
    if "spatial_aq" in settings and "spatial-aq" in settings:
        raise ValueError(
            "Conflicting encoder settings: spatial_aq and spatial-aq are aliases; use only one"
        )
    if codec is None:
        supported = supported_all
        scope = "Supported"
    else:
        if codec not in by_codec:
            raise ValueError(f"Unsupported codec: {codec}")
        supported = by_codec[codec]
        scope = f"Supported for {codec}"
    invalid = sorted(set(settings.keys()) - set(supported))
    if invalid:
        raise ValueError(
            f"Unsupported encoder setting(s){'' if codec is None else f' for codec {codec}'}: "
            + ", ".join(invalid)
            + f". {scope}: "
            + ", ".join(sorted(supported))
        )
    return settings


class UnsupportedColorspaceError(Exception):
    pass


@dataclass
class VideoMetadata:
    video_file: str
    video_height: int
    video_width: int
    video_fps: float
    average_fps: float
    video_fps_exact: Fraction
    codec_name: str
    duration: float
    time_base: Fraction
    start_pts: int
    color_range: AvColorRange
    color_space: AvColorspace
    num_frames: int
    is_10bit: bool
    sample_aspect_ratio: Fraction = Fraction(1, 1)
    pixel_format: str = ""
    profile: str = ""
    field_order: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    stereo_layout: str = ""
    spherical_projection: str = ""


def resolve_video_start_pts(
    stream_start_time: int | None,
    metadata_start_pts: int | None,
) -> int:
    if stream_start_time is not None:
        return int(stream_start_time)
    return int(metadata_start_pts or 0)

def _get_frame_count_by_counting(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count


def is_stream_10bit(json_video_stream: dict) -> bool:
    bprs = json_video_stream.get('bits_per_raw_sample')
    if isinstance(bprs, (int, float)):
        return int(bprs) == 10
    if isinstance(bprs, str):
        try:
            if int(bprs) == 10:
                return True
        except ValueError:
            pass
    pix_fmt = (json_video_stream.get('pix_fmt') or '').lower()
    ten_bit_markers = (
        'p10',
        'p010',
        'v210',
        'rgb10', 'bgr10', 'x2rgb10', 'x2bgr10', 'yuv10', 'gray10'
    )
    return any(marker in pix_fmt for marker in ten_bit_markers)

def parse_sample_aspect_ratio(json_video_stream: dict) -> Fraction:
    text = json_video_stream.get('sample_aspect_ratio') or ''
    num, sep, den = text.partition(':')
    if sep and num.isdigit() and den.isdigit() and int(num) > 0 and int(den) > 0:
        return Fraction(int(num), int(den))
    return Fraction(1, 1)


def parse_spatial_metadata(json_video_stream: dict) -> tuple[str, str]:
    stereo_layout = ""
    spherical_projection = ""
    for side_data in json_video_stream.get("side_data_list") or ():
        side_data_type = str(side_data.get("side_data_type") or "").lower()
        if side_data_type == "stereo 3d":
            stereo_layout = str(side_data.get("type") or "")
        elif side_data_type == "spherical mapping":
            spherical_projection = str(side_data.get("projection") or "")
    tags = json_video_stream.get("tags") or {}
    if not stereo_layout:
        stereo_layout = str(tags.get("stereo_mode") or "")
    if not spherical_projection:
        spherical_projection = str(tags.get("projection") or "")
    return stereo_layout, spherical_projection


def get_video_meta_data(path: str) -> VideoMetadata:
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
    ffprobe = resolve_executable("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-select_streams",
        "v",
        "-show_streams",
        "-show_format",
        path,
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **subprocess_no_window_kwargs(),
    )
    out, err = p.communicate()
    if p.returncode != 0:
        stdout_text = (out or b"").decode(errors="replace")
        stderr_text = (err or b"").decode(errors="replace")
        logger.error(
            "ffprobe failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            p.returncode,
            stdout_text,
            stderr_text,
        )
        raise Exception(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    value = [int(num) for num in json_video_stream['avg_frame_rate'].split("/")]
    # Can be 0/0 for some files for ffprobe isn't able to determine the number of frames nb_frames
    average_fps = value[0]/value[1] if len(value) == 2 and value[1] != 0 else value[0]

    value = [int(num) for num in json_video_stream['r_frame_rate'].split("/")]
    fps = value[0]/value[1] if len(value) == 2 else value[0]
    fps_exact = Fraction(value[0], value[1])

    value = [int(num) for num in json_video_stream['time_base'].split("/")]
    time_base = Fraction(value[0], value[1])

    start_pts = json_video_stream.get('start_pts')
    range_name = (json_video_stream.get("color_range") or "").lower()
    color_range = (
        AvColorRange.JPEG
        if range_name in {"pc", "jpeg", "full"}
        else AvColorRange.MPEG
    )
    color_space_name = (json_video_stream.get("color_space") or "").lower()
    if color_space_name in {"bt601", "bt470bg", "smpte170m"}:
        color_space = AvColorspace.ITU601
    elif color_space_name in {"bt2020", "bt2020nc", "bt2020_ncl"}:
        color_space = AvColorspace.BT2020
    else:
        color_space = AvColorspace.ITU709


    num_frames = int(json_video_stream.get('nb_frames', 0))
    if num_frames == 0:
        num_frames = _get_frame_count_by_counting(path)
    is_10bit = is_stream_10bit(json_video_stream)
    stereo_layout, spherical_projection = parse_spatial_metadata(json_video_stream)

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts,
        color_range=color_range,
        color_space=color_space,
        num_frames=num_frames,
        is_10bit=is_10bit,
        sample_aspect_ratio=parse_sample_aspect_ratio(json_video_stream),
        pixel_format=str(json_video_stream.get("pix_fmt") or ""),
        profile=str(json_video_stream.get("profile") or ""),
        field_order=str(json_video_stream.get("field_order") or ""),
        color_primaries=str(json_video_stream.get("color_primaries") or ""),
        color_transfer=str(json_video_stream.get("color_transfer") or ""),
        stereo_layout=stereo_layout,
        spherical_projection=spherical_projection,
    )
    return metadata

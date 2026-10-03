"""Video stream metadata read through ffprobe."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

if TYPE_CHECKING:
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

logger = logging.getLogger(__name__)


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
    video_bitrate: int = 0
    hevc_level: int | None = None


def resolve_video_start_pts(stream_start_time: int | None, metadata_start_pts: int) -> int:
    if stream_start_time is not None:
        return int(stream_start_time)
    return metadata_start_pts

def _frame_count_from_container(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(frame_count)


def is_stream_10bit(json_video_stream: dict) -> bool:
    bprs = json_video_stream.get('bits_per_raw_sample')
    if isinstance(bprs, (int, float)):
        return int(bprs) == 10
    if isinstance(bprs, str) and bprs.strip() == "10":
        return True
    pix_fmt = _stream_pixel_format(json_video_stream)
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


def parse_video_bitrate(json_video_stream: dict, json_video_format: dict) -> int:
    """Video bitrate in bits/s, or 0 when no source reports one."""
    tags = json_video_stream.get("tags") or {}
    candidates = (
        json_video_stream.get("bit_rate"),
        tags.get("BPS"),
        tags.get("BPS-eng"),
        # Container rate counts audio too, so it slightly overstates the video
        # stream. Matroska muxers routinely omit the per-stream rate, and an
        # ceiling that is a few percent loose beats having none at all.
        json_video_format.get("bit_rate"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = int(float(candidate))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


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
        raise RuntimeError(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    # avg_frame_rate is 0/0 when ffprobe cannot count the frames.
    avg_num, avg_den = (int(num) for num in json_video_stream['avg_frame_rate'].split("/"))
    average_fps = avg_num / avg_den if avg_den else float(avg_num)
    fps_exact = Fraction(json_video_stream['r_frame_rate'])
    fps = float(fps_exact)
    time_base = Fraction(json_video_stream['time_base'])
    start_pts = int(json_video_stream.get('start_pts') or 0)
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
        num_frames = _frame_count_from_container(path)
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
        pixel_format=_stream_pixel_format(json_video_stream),
        profile=str(json_video_stream.get("profile") or ""),
        field_order=str(json_video_stream.get("field_order") or ""),
        color_primaries=str(json_video_stream.get("color_primaries") or ""),
        color_transfer=str(json_video_stream.get("color_transfer") or ""),
        stereo_layout=stereo_layout,
        spherical_projection=spherical_projection,
        video_bitrate=parse_video_bitrate(json_video_stream, json_video_format),
        hevc_level=parse_hevc_level_idc(json_video_stream),
    )
    return metadata


_HEVC_LEVEL_IDC_TO_AMF_OPTION: dict[int, str] = {
    30: "1.0",
    60: "2.0",
    63: "2.1",
    90: "3.0",
    93: "3.1",
    120: "4.0",
    123: "4.1",
    150: "5.0",
    153: "5.1",
    156: "5.2",
    180: "6.0",
    183: "6.1",
    186: "6.2",
}


def _stream_pixel_format(json_video_stream: dict) -> str:
    """Return ffprobe's pixel format or a standardized AV1 codec-string inference.

    The minimal unified AMF runtime exposes only the hardware AV1 decoder.  Its
    stream probe reports the standardized ``av01`` MIME codec string but no
    ``pix_fmt`` until decode.  Preserve a fixed-format decision by translating
    only the explicit 8/10-bit 4:2:0 depths this reader supports.
    """

    pixel_format = str(json_video_stream.get("pix_fmt") or "").lower()
    if pixel_format or str(json_video_stream.get("codec_name") or "").lower() != "av1":
        return pixel_format
    codec_parts = str(json_video_stream.get("mime_codec_string") or "").split(".")
    if len(codec_parts) < 4 or codec_parts[0].lower() != "av01":
        return ""
    return {
        "08": "yuv420p",
        "10": "yuv420p10le",
    }.get(codec_parts[3], "")


def parse_hevc_level_idc(json_video_stream: dict) -> int | None:
    """Return FFprobe's integral HEVC level_idc, or None when unavailable."""

    if str(json_video_stream.get("codec_name") or "").lower() != "hevc":
        return None
    value = json_video_stream.get("level")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError:
            return None
    return None


def hevc_level_to_amf_option(level_idc: object) -> str | None:
    """Translate FFprobe HEVC level_idc to hevc_amf dotted option text."""

    if isinstance(level_idc, bool):
        return None
    try:
        value = int(level_idc)
    except (TypeError, ValueError):
        return None
    if isinstance(level_idc, float) and not level_idc.is_integer():
        return None
    return _HEVC_LEVEL_IDC_TO_AMF_OPTION.get(value)

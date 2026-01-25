import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
import json

SUPPORTED_ENCODER_SETTINGS: frozenset[str] = frozenset(
    {
        "preset",
        "tuning_info",
        "rc",
        "cq",
        "qmin",
        "qmax",
        "nonrefp",
        "gop",
        "maxbitrate",
        "vbvinit",
        "vbvbufsize",
        "temporalaq",
        "lookahead",
        "lookahead_level",
        "aq",
        "initqp",
        "tflevel",
    }
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


def validate_encoder_settings(settings: dict[str, object]) -> dict[str, object]:
    invalid = sorted(set(settings.keys()) - set(SUPPORTED_ENCODER_SETTINGS))
    if invalid:
        raise ValueError(
            "Unsupported encoder setting(s): "
            + ", ".join(invalid)
            + ". Supported: "
            + ", ".join(sorted(SUPPORTED_ENCODER_SETTINGS))
        )
    return settings


def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info

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
        except Exception:
            pass
    pix_fmt = (json_video_stream.get('pix_fmt') or '').lower()
    ten_bit_markers = (
        'p10',
        'p010',
        'v210',
        'rgb10', 'bgr10', 'x2rgb10', 'x2bgr10', 'yuv10', 'gray10'
    )
    return any(marker in pix_fmt for marker in ten_bit_markers)

def get_video_meta_data(path: str) -> VideoMetadata:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=get_subprocess_startup_info())
    out, err =  p.communicate()
    if p.returncode != 0:
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
    color_range = AvColorRange.MPEG if 'color_range' not in json_video_stream or json_video_stream['color_range'] == 'tv' else AvColorRange.JPEG if json_video_stream['color_range'] == 'jpeg' else AvColorRange.MPEG
    color_space = AvColorspace.ITU709 if 'color_space' not in json_video_stream or json_video_stream['color_space'] == 'bt709' else AvColorspace.ITU601 if json_video_stream['color_space'] == 'bt601' else AvColorspace.ITU709

    num_frames = int(json_video_stream.get('nb_frames', 0))
    if num_frames == 0:
        num_frames = _get_frame_count_by_counting(path)
    is_10bit = is_stream_10bit(json_video_stream)

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
    )
    return metadata
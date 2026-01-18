import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
import json

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
        color_space=color_space
    )
    return metadata
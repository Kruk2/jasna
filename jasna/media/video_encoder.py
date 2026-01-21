import torch
import logging

import PyNvVideoCodec as nvc
from pathlib import Path
from jasna.media import VideoMetadata, get_subprocess_startup_info
from jasna.media.rgb_to_p010 import chw_rgb_to_p010_bt709_limited
import av
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
import heapq
from collections  import deque
import subprocess
av.logging.set_level(logging.ERROR)

def _parse_hevc_nal_units(data: bytes):
    """Parse HEVC NAL units from Annex B bitstream. Returns list of (nal_type, start, end)."""
    nal_units = []
    i = 0
    n = len(data)
    
    while i < n - 3:
        # Find start code (0x000001 or 0x00000001)
        if data[i:i+3] == b'\x00\x00\x01':
            start = i + 3
            sc_len = 3
        elif i < n - 4 and data[i:i+4] == b'\x00\x00\x00\x01':
            start = i + 4
            sc_len = 4
        else:
            i += 1
            continue
        
        # Find next start code
        end = start
        while end < n - 3:
            if data[end:end+3] == b'\x00\x00\x01' or (end < n - 4 and data[end:end+4] == b'\x00\x00\x00\x01'):
                break
            end += 1
        if end >= n - 3:
            end = n
        
        if start < n:
            # HEVC NAL unit type is bits 1-6 of first byte
            nal_type = (data[start] >> 1) & 0x3F
            nal_units.append((nal_type, i, end))
        
        i = end
    
    return nal_units


def _is_hevc_keyframe(data: bytes) -> bool:
    """Check if HEVC bitstream contains an IDR or CRA frame."""
    # HEVC NAL types for keyframes: IDR_W_RADL=19, IDR_N_LP=20, CRA_NUT=21, BLA types=16-18
    keyframe_types = {16, 17, 18, 19, 20, 21}
    for nal_type, _, _ in _parse_hevc_nal_units(data):
        if nal_type in keyframe_types:
            return True
    return False


def _extract_hevc_extradata(data: bytes) -> bytes:
    """Extract VPS, SPS, PPS NAL units for codec extradata."""
    # VPS=32, SPS=33, PPS=34
    param_types = {32, 33, 34}
    extradata_parts = []
    
    for nal_type, start, end in _parse_hevc_nal_units(data):
        if nal_type in param_types:
            # Include the start code
            extradata_parts.append(data[start-4:end] if data[start-4:start] == b'\x00\x00\x00\x01' else b'\x00\x00\x00\x01' + data[start:end])
    
    return b''.join(extradata_parts)


def mux_hevc_to_mkv(hevc_path: Path, output_path: Path, pts_list, time_base):
    timecodes_path = output_path.with_suffix('.txt')
    with open(timecodes_path, 'w') as f:
        f.write("# timestamp format v4\n")
        for pts in pts_list:
            timestamp_ms = float(pts * time_base * 1000)
            f.write(f"{timestamp_ms:.6f}\n")
    
    cmd = [
        'mkvmerge', '-o', str(output_path),
        '--timestamps', f'0:{timecodes_path}',
        str(hevc_path)
    ]
    result = subprocess.run(cmd, capture_output=True, startupinfo=get_subprocess_startup_info())
    if result.returncode != 0:
        raise RuntimeError(f"mkvmerge failed with code {result.returncode}: {' '.join(cmd)}\n{result.stderr.decode()}")
    timecodes_path.unlink()


def remux_with_audio_and_metadata(video_input: Path, output_path: Path, metadata: VideoMetadata):
    colorspace_map = {
        AvColorspace.ITU709: 'bt709',
        AvColorspace.ITU601: 'smpte170m',
    }
    color_range_map = {
        AvColorRange.MPEG: 'tv',
        AvColorRange.JPEG: 'pc',
    }
    ffmpeg_colorspace = colorspace_map.get(metadata.color_space, 'bt709')
    ffmpeg_color_range = color_range_map.get(metadata.color_range, 'tv')

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_input),
        '-i', metadata.video_file,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-map_metadata', '1',
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-color_primaries', ffmpeg_colorspace,
        '-color_trc', ffmpeg_colorspace,
        '-colorspace', ffmpeg_colorspace,
        '-color_range', ffmpeg_color_range,
    ]
    if output_path.suffix.lower() in {'.mp4', '.mov'}:
        cmd += ['-movflags', '+faststart']
    cmd.append(str(output_path))
    result = subprocess.run(cmd, capture_output=True, startupinfo=get_subprocess_startup_info())
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {' '.join(cmd)}\n{result.stderr.decode()}")


class NvidiaVideoEncoder:
    def __init__(self, file: str, device: torch.device, stream: torch.cuda.Stream, metadata: VideoMetadata, stream_mode: bool = False):
        self.metadata = metadata
        self.stream = stream
        self.device = device
        self.file = file
        self.output_path = Path(file)
        self.stream_mode = stream_mode
        bf = 1 if stream_mode else 4 # 1 or 2?

        #todo for streaming mode enable tuning low latency, disable qpass
        encoder_options = {
            'codec': 'hevc',
            'preset': 'P5',
            'tuning_info': 'high_quality',
            'profile': 'main10',
            'rc': 'vbr',
            "cq": 25,
            "qmin": 17,
            "qmax": 34,
            # 'rc': 'constqp',
            # 'constqp': 21,
            'nonrefp': 1,
            # 'multipass': 'qres', # lower psnr
            'gop': 250,
            'fps': float(metadata.video_fps_exact),
            "maxbitrate": 0,
            # "maxbitrate": 153600,
            "vbvinit": 0,
            "vbvbufsize": 0,
            'temporalaq': 1,
            'lookahead': 32,
            'lookahead_level': 1,
            'aq': 8,
            "initqp": 17,
            'bf': bf,
            'tflevel': 0,
            "bref": 2 if not stream_mode else 0,
        }

        self.encoder = nvc.CreateEncoder(
            width=metadata.video_width,
            height=metadata.video_height,
            cudastream=stream.cuda_stream,
            fmt="P010",
            usecpuinputbuffer=False,
            **encoder_options
        )

        self.BUFFER_MAX_SIZE = 8
        self.pts_heap = []
        self.frame_buffer = deque()
        self.pts_set = set()
        self.reordered_pts_queue = deque()

        if metadata.color_space != AvColorspace.ITU709 and metadata.color_range != AvColorRange.MPEG:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")

        self.temp_video_path = self.output_path.with_name(self.output_path.stem + '_temp_video' + self.output_path.suffix)

        if self.stream_mode:
            dst_file = av.open(str(self.temp_video_path), 'w')
            out_stream = dst_file.add_stream('hevc', rate=metadata.video_fps_exact)
            out_stream.width = metadata.video_width
            out_stream.height = metadata.video_height
            out_stream.time_base = metadata.time_base
            out_stream.color_range = metadata.color_range
            out_stream.colorspace = metadata.color_space
            out_stream.codec_context.width = metadata.video_width
            out_stream.codec_context.height = metadata.video_height
            out_stream.codec_context.time_base = metadata.time_base
            out_stream.codec_context.color_range = metadata.color_range
            out_stream.codec_context.colorspace = metadata.color_space
            out_stream.options.update({'x265-params': 'log_level=error'})
            self.dst_file = dst_file
            self.out_stream = out_stream
            self.extradata_set = False
        else:
            self.hevc_path = self.output_path.with_suffix('.hevc')
            self.raw_hevc = open(self.hevc_path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        while self.frame_buffer:
            self._process_buffer(flush_all=True)

        while True:
            with torch.cuda.stream(self.stream):
                bitstream = self.encoder.EndEncode()
            if len(bitstream) == 0:
                break
            data = bytearray(bitstream)
            if self.stream_mode:
                pts = self.reordered_pts_queue.popleft()
                self._mux_packet_pyav(data, pts)
            else:
                self.raw_hevc.write(data)

        if self.stream_mode:
            self.dst_file.close()
            remux_with_audio_and_metadata(self.temp_video_path, self.output_path, self.metadata)
            self.temp_video_path.unlink()
        else:
            self.raw_hevc.close()
            mux_hevc_to_mkv(self.hevc_path, self.temp_video_path, self.reordered_pts_queue, self.metadata.time_base)
            remux_with_audio_and_metadata(self.temp_video_path, self.output_path, self.metadata)
            self.hevc_path.unlink()
            self.temp_video_path.unlink()

        del self.encoder

    def _mux_packet_pyav(self, data: bytearray, pts: int):
        data_bytes = bytes(data)
        
        if not self.extradata_set:
            extradata = _extract_hevc_extradata(data_bytes)
            if extradata:
                self.out_stream.codec_context.extradata = extradata
                self.extradata_set = True
        
        pkt = av.packet.Packet(data_bytes)
        pkt.stream = self.out_stream
        pkt.time_base = self.out_stream.time_base
        pkt.pts = pts
        
        if _is_hevc_keyframe(data_bytes):
            pkt.is_keyframe = True
        
        self.dst_file.mux(pkt)

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            self._encode_frame(frame_to_encode, pts_to_assign)

    def _encode_frame(self, frame: torch.Tensor, pts: int):
        self.reordered_pts_queue.append(pts)

        with torch.cuda.stream(self.stream):
            p010 = chw_rgb_to_p010_bt709_limited(frame)
            bitstream = self.encoder.Encode(p010)

        if len(bitstream) > 0:
            data = bytearray(bitstream)
            if self.stream_mode:
                pts = self.reordered_pts_queue.popleft()
                self._mux_packet_pyav(data, pts)
            else:
                self.raw_hevc.write(data)

    def encode(self, frame: torch.Tensor, pts: int):
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self.pts_set.add(pts)
        self._process_buffer()

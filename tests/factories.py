"""Builders shared by tests that need a SessionConfig, a Pipeline, or test media."""
from __future__ import annotations

import re
import struct
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media.probe import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import PrimaryRestoreResult
from jasna.os_utils import resolve_executable
from jasna.session_config import SessionConfig


class TemporalUpscaler(torch.nn.Module):
    """Small recurrent network for exercising secondary restoration on the CPU."""

    def forward(self, previous, current, high_previous):
        low = (current + previous) / 2
        high = torch.nn.functional.interpolate(low.permute(0, 3, 1, 2), scale_factor=4, mode="nearest")
        return (high.permute(0, 2, 3, 1) + high_previous) / 2


def session_config(**overrides) -> SessionConfig:
    base = dict(
        device="cuda:0",
        fp16=True,
        batch_size=4,
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        detection_score_threshold=0.25,
        max_detection_gap=2,
        min_detection_duration=2,
        scene_detection=True,
        restoration_model_name="basicvsrpp",
        restoration_model_path=Path("restore.pth"),
        ltx_large_canvas=True,
        ltx_window_hop=88,
        ltx_seed=0,
        ltx_fast=False,
        ltx_model="distilled",
        ltx_trial=False,
        ltx_auto_split_percent=None,
        compile_basicvsrpp=True,
        max_clip_size=90,
        temporal_overlap=8,
        denoise_strength="none",
        secondary_restoration="none",
        tvai_ffmpeg_path="ffmpeg.exe",
        tvai_model="iris-2",
        tvai_scale=4,
        tvai_args="noise=0",
        tvai_workers=2,
        rtx_scale=4,
        rtx_quality="high",
        rtx_denoise="medium",
        rtx_deblur="none",
        vr_mode="auto",
        codec="hevc",
        encoder_settings={"cq": 25},
        lut_path=None,
        retarget_high_fps=False,
        disable_progress=False,
        working_dir=None,
        vr_projection="auto",
        fmp4=False,
        sharpen_strength=0.0,
        tvai_denoise=False,
    )
    base.update(overrides)
    return SessionConfig(**base)


def video_metadata(*, fps: int = 24, num_frames: int = 2, **overrides) -> VideoMetadata:
    """Probe result of an 8x8 8-bit BT.709 video of ``num_frames`` frames at ``fps``."""
    values = dict(
        video_file="input.mkv",
        video_height=8,
        video_width=8,
        video_fps=float(fps),
        average_fps=float(fps),
        video_fps_exact=Fraction(fps),
        codec_name="hevc",
        duration=num_frames / fps,
        time_base=Fraction(1, fps),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=num_frames,
        is_10bit=False,
    )
    values.update(overrides)
    return VideoMetadata(**values)


def primary_result(
    *, track_id: int = 1, start_frame: int = 0, frames: int = 2, keep_start: int = 0, keep_end: int | None = None
) -> PrimaryRestoreResult:
    """A primary-restored clip of an 8x8 video: ``frames`` 256x256 crops of the box (1, 1, 5, 5)."""
    return PrimaryRestoreResult(
        track_id=track_id,
        start_frame=start_frame,
        frame_count=frames,
        frame_shape=(8, 8),
        frame_device=torch.device("cpu"),
        masks=[torch.ones((8, 8), dtype=torch.bool)] * frames,
        primary_raw=torch.ones((frames, 3, 256, 256)),
        keep_start=keep_start,
        keep_end=frames if keep_end is None else keep_end,
        crossfade_weights=None,
        enlarged_bboxes=[(1, 1, 5, 5)] * frames,
        crop_shapes=[(4, 4)] * frames,
        pad_offsets=[(126, 126)] * frames,
        resize_shapes=[(4, 4)] * frames,
    )


def cli_args(tmp_path: Path, *extra: str, missing: tuple[str, ...] = ()) -> list[str]:
    """CLI arguments for one video job in ``tmp_path``; every file exists unless its flag is ``missing``."""
    paths = {
        "--input": tmp_path / "in.mp4",
        "--output": tmp_path / "out.mkv",
        "--restoration-model-path": tmp_path / "model_weights" / "restore.pth",
        "--detection-model-path": tmp_path / "model_weights" / "det.onnx",
    }
    paths["--restoration-model-path"].parent.mkdir(exist_ok=True)
    for flag in ("--input", "--restoration-model-path", "--detection-model-path"):
        if flag not in missing:
            paths[flag].touch()
    return [arg for flag, path in paths.items() for arg in (flag, str(path))] + list(extra)


@contextmanager
def cli_environment(pipeline=None):
    """Passing system checks, no GPU context, no engine builds and a mocked ``Pipeline`` class, which it yields."""
    pipeline_cls = MagicMock(return_value=MagicMock() if pipeline is None else pipeline)
    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.accelerator.device_context", side_effect=lambda _device: nullcontext()),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", pipeline_cls),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        yield pipeline_cls


def run_cli(argv: list[str], *, pipeline=None) -> MagicMock:
    """Run ``jasna`` with ``argv`` in ``cli_environment``; returns the mocked ``Pipeline`` class."""
    from jasna.main import main

    with cli_environment(pipeline) as pipeline_cls, patch.object(sys, "argv", ["jasna", *argv]):
        main()
    return pipeline_cls


def no_resume_conflict(directory: Path):
    raise AssertionError(f"unexpected saved-progress conflict: {directory}")


def fake_session(
    *, device: torch.device, restoration_pipeline, detection_model=None, ltx_files=None
) -> SimpleNamespace:
    detection_model = MagicMock() if detection_model is None else detection_model
    return SimpleNamespace(
        device=device,
        restoration_pipeline=restoration_pipeline,
        ltx_files=ltx_files,
        detection_model_for=lambda config: detection_model,
    )


def make_pipeline(
    *,
    device: torch.device = torch.device("cpu"),
    restoration_pipeline=None,
    detection_model=None,
    input_video: Path = Path("in.mp4"),
    output_video: Path = Path("out.mkv"),
    progress_callback=None,
    segments=None,
    splice_plan=None,
    ltx_files=None,
    **config_overrides,
) -> Pipeline:
    if restoration_pipeline is None:
        restoration_pipeline = MagicMock(
            secondary_restorer=None,
            secondary_num_workers=1,
            secondary_prefers_cpu_input=False,
        )
    return Pipeline(
        config=session_config(**config_overrides),
        session=fake_session(
            device=device,
            restoration_pipeline=restoration_pipeline,
            detection_model=detection_model,
            ltx_files=ltx_files,
        ),
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
    )


def _adts_header(frame_length: int) -> bytes:
    return bytes([
        0xFF, 0xF1,
        0x50,
        0x40 | (frame_length >> 11),
        (frame_length >> 3) & 0xFF,
        ((frame_length & 7) << 5) | 0x1F,
        0xFC,
    ])


def write_double_adts_aac_source(tmp_path: Path) -> Path:
    """H.264 + AAC (LC, 44.1 kHz, mono) NUT file whose audio packets each carry two ADTS headers."""
    plain = tmp_path / "plain_aac.mp4"
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ar", "44100", "-ac", "1",
            "-c:v", "libx264", "-c:a", "aac", str(plain),
        ],
        check=True,
    )
    source = tmp_path / "double_adts.nut"
    with av.open(str(plain)) as src, av.open(str(source), "w") as dst:
        video_out = dst.add_stream_from_template(src.streams.video[0])
        audio_out = dst.add_stream_from_template(src.streams.audio[0])
        for packet in src.demux():
            if not packet.size:
                continue
            if packet.stream.type == "audio":
                payload = bytes(packet)
                doubled = av.Packet(_adts_header(len(payload) + 14) * 2 + payload)
                doubled.pts, doubled.dts = packet.pts, packet.dts
                doubled.time_base, doubled.duration = packet.time_base, packet.duration
                doubled.is_keyframe = True
                doubled.stream = audio_out
                dst.mux(doubled)
            else:
                packet.stream = video_out
                dst.mux(packet)
    return source


def _length_prefixed_units(data: bytes) -> list[bytes]:
    units, offset = [], 0
    while offset + 4 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "big")
        units.append(data[offset + 4:offset + 4 + size])
        offset += 4 + size
    return units


def write_variable_frame_rate_source(path: Path) -> Path:
    """H.264 + AAC MP4 on a 30 fps timestamp grid that drops a varying share of frames,
    so frame gaps alternate between 1/30 s and 2/30 s."""
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=4",
            "-vf", "select='not(eq(mod(n,5),2))*not(eq(mod(n,7),4))'", "-fps_mode", "vfr",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
            "-g", "20", "-keyint_min", "20", "-sc_threshold", "0", "-bf", "3",
            "-c:a", "aac", str(path),
        ],
        check=True,
    )
    return path


def write_hrd_timing_sei_source(path: Path) -> Path:
    """29.97 fps H.264 + AAC MP4 with HRD buffering_period/pic_timing SEI whose samples
    end with the next picture's access unit delimiter and SEI, as some muxers store them."""
    plain = path.with_name(f"{path.stem}-plain.mp4")
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30000/1001:duration=4.004",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=4.004",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-bf", "3", "-b:v", "500k",
            "-x264-params", "nal-hrd=vbr:vbv-maxrate=1000:vbv-bufsize=2000:aud=1",
            "-bsf:v", "h264_metadata=fixed_frame_rate_flag=1",
            "-c:a", "aac", str(plain),
        ],
        check=True,
    )
    with av.open(str(plain)) as src:
        samples = []
        for packet in src.demux(src.streams.video[0]):
            if packet.size:
                units = _length_prefixed_units(bytes(packet))
                first_slice = next(i for i, unit in enumerate(units) if unit[0] & 0x1F in {1, 5})
                samples.append((units[:first_slice], units[first_slice:]))
    moved = [samples[0][0] + samples[0][1] + samples[1][0]]
    moved += [pictures + following[0] for (_, pictures), following in zip(samples[1:], samples[2:])]
    moved.append(samples[-1][1])
    with av.open(str(plain)) as src, av.open(str(path), "w") as dst:
        outputs = {stream.index: dst.add_stream_from_template(stream) for stream in src.streams}
        video_samples = iter(moved)
        for packet in src.demux():
            if not packet.size:
                continue
            output = outputs[packet.stream.index]
            if packet.stream.type == "video":
                units = next(video_samples)
                rewritten = av.Packet(b"".join(len(unit).to_bytes(4, "big") + unit for unit in units))
                rewritten.pts, rewritten.dts, rewritten.duration = packet.pts, packet.dts, packet.duration
                rewritten.time_base, rewritten.is_keyframe = packet.time_base, packet.is_keyframe
                packet = rewritten
            packet.stream = output
            dst.mux(packet)
    plain.unlink()
    return path


def _sps_without_bitstream_restriction(sps: bytes, flag_bit: int) -> bytes:
    """``sps`` cut at its bitstream_restriction_flag (a bit offset into the RBSP),
    which is set to 0 and followed by the RBSP stop bit."""
    rbsp = re.sub(rb"\x00\x00\x03", b"\x00\x00", sps)
    bits = "".join(f"{byte:08b}" for byte in rbsp)[:flag_bit] + "01"
    bits += "0" * (-len(bits) % 8)
    cut = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return re.sub(rb"\x00\x00(?=[\x00-\x03])", b"\x00\x00\x03", cut)


def write_unbounded_reorder_h264_source(path: Path) -> Path:
    """352x288 level 1.1 H.264 MP4 with three B-frames whose SPS leaves out
    bitstream_restriction, so a decoder must hold the level's whole DPB for
    reordering. Every frame is a flat gray level of its own."""
    ffmpeg = resolve_executable("ffmpeg")
    plain = path.with_name(f"{path.stem}-plain.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "color=black:size=352x288:rate=30:duration=1.6,format=gray,geq=lum='mod(N*23,200)+20',format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "main", "-level", "1.1",
            "-x264-params", "bframes=3:b-pyramid=0:b-adapt=0:ref=4:keyint=12:scenecut=0",
            str(plain),
        ],
        check=True,
    )
    trace = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(plain), "-c", "copy", "-bsf:v", "trace_headers",
         "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    flag_bit = int(re.search(r"(\d+)\s+bitstream_restriction_flag", trace).group(1))
    with av.open(str(plain)) as src, av.open(str(path), "w") as dst:
        video = src.streams.video[0]
        avcc = bytes(video.codec_context.extradata)
        sps_end = 8 + int.from_bytes(avcc[6:8], "big")
        sps = _sps_without_bitstream_restriction(avcc[8:sps_end], flag_bit)
        output = dst.add_stream_from_template(video)
        output.codec_context.extradata = avcc[:6] + len(sps).to_bytes(2, "big") + sps + avcc[sps_end:]
        for packet in src.demux(video):
            if packet.size:
                packet.stream = output
                dst.mux(packet)
    plain.unlink()
    return path


def ltx_models(root, directory: Path, *, installed: bool, on_change=lambda: None):
    """A real ``LtxModels`` over ``directory``; ``installed`` puts every model file there."""
    from jasna.gui.ltx_models import LtxModels
    from jasna.gui.queues import MainThreadCalls
    from jasna.ltx.model_files import LTX_MODELS, bundle_names

    directory.mkdir(parents=True, exist_ok=True)
    if installed:
        for model in LTX_MODELS:
            for fast in (False, True):
                for name in bundle_names(model, fast=fast):
                    (directory / name).touch()
    return LtxModels(directory, MainThreadCalls(root, 20), on_change)


def write_non_utf8_metadata_source(path: Path) -> Path:
    """H.264 + AAC file whose container, stream and chapter titles hold a raw 0x90 byte."""
    marker = b"JASNA-TAG"
    chapters = path.with_name(f"{path.stem}.ffmetadata")
    chapters.write_text(f";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle={marker.decode()}\n")
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            # 384 px wide: the narrowest frame hevc_amf encodes.
            "-f", "lavfi", "-i", "testsrc2=size=384x256:rate=12:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-i", str(chapters), "-map", "0", "-map", "1", "-map_chapters", "2",
            "-metadata", f"title={marker.decode()}",
            "-metadata:s:v:0", f"title={marker.decode()}",
            "-c:v", "libx264", "-g", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True,
    )
    data = path.read_bytes()
    assert data.count(marker) >= 2
    path.write_bytes(data.replace(marker, b"JASNA\x90TAG"))
    return path


def offload_bundle_targets(path: Path) -> set[str]:
    """Target IDs in an uncompressed clang offload bundle (``.hipfb``)."""
    data = path.read_bytes()
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    assert data.startswith(magic), f"{path.name} is not an uncompressed offload bundle"
    (count,) = struct.unpack_from("<Q", data, len(magic))
    offset = len(magic) + 8
    targets = set()
    for _ in range(count):
        _, _, id_length = struct.unpack_from("<QQQ", data, offset)
        offset += 24
        targets.add(data[offset:offset + id_length].decode())
        offset += id_length
    return targets


def weight_stub(dtype, device: str = "cuda:0") -> SimpleNamespace:
    """A model weight's dtype and device, for kernel checks that look at nothing else."""
    return SimpleNamespace(dtype=dtype, device=torch.device(device))


class ReadObserver:
    """Wraps a reader's ``read_tensors_into``: records each call's tensor names and every
    destination it fills, then raises ``failure`` once the read is done, when given."""

    def __init__(self, reader, failure: Exception | None = None):
        self.operations, self.destinations = [], []
        load = reader.read_tensors_into

        def observe(group_names, groups):
            self.operations.append(group_names)
            load(group_names, self._record(groups))
            if failure is not None:
                raise failure

        reader.read_tensors_into = observe

    def _record(self, groups):
        for group in groups:
            self.destinations.extend(group.values())
            yield group

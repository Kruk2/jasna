"""Builders shared by tests that need a SessionConfig, a Pipeline, or test media."""
from __future__ import annotations

import logging
import re
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.blend_buffer import BlendBuffer
from jasna.frame_queue import FrameQueue
from jasna.media.probe import VideoMetadata
from jasna.media.yuv_to_rgb import YuvToRgbConverter
from jasna.pipeline import Pipeline
from jasna.pipeline_debug_logging import PipelineDebugMemoryLogger
from jasna.pipeline_items import PrimaryRestoreResult, SecondaryRestoreResult
from jasna.pipeline_threads import PassContext
from jasna.os_utils import resolve_executable
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession


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
        ltx_vae_encoder="auto",
        ltx_vae_decoder="auto",
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
        frame_generation="none",
        upscale_model="none",
        upscale_height=2160,
        upscale_strength=1.0,
        upscale_rtx_quality="high",
        upscale_rtx_denoise="none",
        upscale_rtx_deblur="none",
        upscale_dlss5_preset="m",
        upscale_dlss5_style="default",
        upscale_dlss5_intensity=1.0,
        disable_progress=False,
        working_dir=None,
        vr_projection="auto",
        fmp4=False,
        sharpen_strength=0.0,
        tvai_denoise=False,
        decode_backend="auto",
    )
    base.update(overrides)
    return SessionConfig(**base)


GUI_WEIGHTS_DIR = Path("model_weights")


def gui_session_config(settings) -> SessionConfig:
    """The session config the GUI builds from ``settings``, with every model under ``GUI_WEIGHTS_DIR``
    and the detector's weights at ``GUI_WEIGHTS_DIR / "<name>.onnx"``."""
    from jasna.gui.video_session import video_session_config

    with (
        patch("jasna.engine_paths.model_weights_dir", return_value=GUI_WEIGHTS_DIR),
        patch(
            "jasna.mosaic.detection_registry.require_detection_model_weights",
            side_effect=lambda name: GUI_WEIGHTS_DIR / f"{name}.onnx",
        ),
    ):
        return video_session_config(settings, codec="hevc", encoder_settings={})


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


def yuv_to_rgb(converter: YuvToRgbConverter, y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """``converter``'s RGB frame for one Y/UV plane pair, in a fresh buffer."""
    out = torch.empty((3, converter.height, converter.width), device=y.device, dtype=torch.uint8)
    converter.convert_into(y, uv, out)
    return out


def pass_context(**overrides) -> PassContext:
    """The shared state of one CPU restoration pass over ``video_metadata()``, with unbounded queues."""
    values = dict(
        device=torch.device("cpu"),
        restoration_pipeline=MagicMock(),
        input_video="fake.mkv",
        decode_backend="auto",
        batch_size=2,
        metadata=video_metadata(),
        seek_ts=None,
        frame_stride=1,
        blend_buffer=BlendBuffer(vr_projector=None),
        crop_buffers={},
        clip_queue=FrameQueue(max_frames=9999),
        secondary_queue=FrameQueue(max_frames=9999),
        encode_queue=FrameQueue(max_frames=9999),
        metadata_queue=Queue(),
        primary_idle_event=threading.Event(),
        stop_event=threading.Event(),
        error_holder=[],
    )
    values.update(overrides)
    values.setdefault(
        "debug_memory",
        PipelineDebugMemoryLogger(
            logger=logging.getLogger("tests.pass_context"),
            blend_buffer=values["blend_buffer"],
            clip_queue=values["clip_queue"],
            secondary_queue=values["secondary_queue"],
            encode_queue=values["encode_queue"],
        ),
    )
    return PassContext(**values)


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


def restore_synchronously(
    restoration_pipeline, clip, raw_crops, frame_shape, keep_start: int, keep_end: int, crossfade_weights=None
) -> SecondaryRestoreResult:
    """One clip through primary and secondary restoration on the calling thread, as the
    primary and secondary worker threads of a pass run it."""
    primary = restoration_pipeline.prepare_and_run_primary(
        clip, raw_crops, frame_shape, keep_start, keep_end, crossfade_weights
    )
    restored = restoration_pipeline.run_secondary(primary.primary_raw, primary.keep_start, primary.keep_end)
    return restoration_pipeline.build_secondary_result(primary, restored)


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
    """Passing system checks, no GPU context, no engine builds and a mocked ``Pipeline`` class, which it yields.
    The root log level ``main`` sets is restored, so later tests still capture warnings."""
    pipeline_cls = MagicMock(return_value=MagicMock() if pipeline is None else pipeline)
    root_level = logging.getLogger().level
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
        try:
            yield pipeline_cls
        finally:
            logging.getLogger().setLevel(root_level)


def run_cli(argv: list[str], *, pipeline=None) -> MagicMock:
    """Run ``jasna`` with ``argv`` in ``cli_environment``; returns the mocked ``Pipeline`` class."""
    from jasna.main import main

    with cli_environment(pipeline) as pipeline_cls, patch.object(sys, "argv", ["jasna", *argv]):
        main()
    return pipeline_cls


@contextmanager
def cli_image_models():
    """The CLI's still-image models without weights: the restorer hands each image back unchanged."""
    from jasna.session_factory import ImageSession

    session = ImageSession(MagicMock(), MagicMock(), torch.device("cpu"), fp16=False)
    with (
        patch("jasna.restorer.sd15_download.ensure_sd15_bundle"),
        patch("jasna.session_factory.build_image_session", return_value=session),
        patch("jasna.image_restore.restore_image", lambda image, *_a, num_variants, **_k: [image] * num_variants),
    ):
        yield


def write_test_image(path: Path) -> Path:
    import numpy as np

    from jasna.media.image_io import write_image_rgb_chw

    write_image_rgb_chw(path, np.zeros((3, 8, 8), dtype=np.uint8))
    return path


def no_resume_conflict(directory: Path):
    raise AssertionError(f"unexpected saved-progress conflict: {directory}")


def job_queue(jobs):
    from jasna.gui.job_queue import JobQueue

    queue = JobQueue()
    queue.add_unique(jobs)
    return queue


def gui_processor(**callbacks):
    """A GUI ``Processor`` whose callbacks ignore everything except the ones given."""
    from jasna.gui.processor import Processor

    return Processor(**{
        "on_progress": lambda _update: None,
        "on_log": lambda _level, _message: None,
        "on_complete": lambda _queue_finished: None,
        "on_resume_conflict": no_resume_conflict,
        **callbacks,
    })


class CallsAtOnce:
    """Stands in for ``gui.queues.MainThreadCalls``: runs each posted callback right away."""

    def post(self, callback) -> None:
        callback()


class FakeJobPipeline:
    """A job's pipeline for GUI processor tests: ``run()`` calls its builder's ``restore(pipeline)``
    and, like the real pipeline, counts as completed unless it was cancelled."""

    def __init__(self, builder: "FakeJobPipelines", input_path: Path, output_path: Path):
        self.input_path, self.output_path = input_path, output_path
        self.cancel_requested = self.completed = False
        self._builder = builder

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        self._builder.restore(self)
        self.completed = not self.cancel_requested


class FakeJobPipelines:
    """Stands in for ``session_factory.build_job_pipeline``; a test sets ``restore`` to act inside each run."""

    def __init__(self):
        self.restore = lambda pipeline: None
        self.built: list[FakeJobPipeline] = []

    def build(self, _config, _session, input_path, output_path, _segment_job, *, progress_callback) -> FakeJobPipeline:
        self.built.append(FakeJobPipeline(self, input_path, output_path))
        return self.built[-1]


@dataclass
class FixedDetectorSession(RestorationSession):
    """A restoration session whose detector is given instead of built from the config."""

    detector: object = None

    def detection_model_for(self, config: SessionConfig):
        return self.detector


class RecordingWriter:
    """A frame writer that keeps every ``(frame, pts)`` it is given; ``close`` raises when ``fail_on_close``."""

    def __init__(self, *, fail_on_close: bool = False):
        self.written: list[tuple[torch.Tensor, int]] = []
        self.finished = self.closed = False
        self._fail_on_close = fail_on_close

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self.written.append((frame, pts))

    def after_write(self, frames_written: int, stop_event: threading.Event) -> None:
        pass

    def finish(self) -> None:
        self.finished = True

    def close(self) -> None:
        self.closed = True
        if self._fail_on_close:
            raise RuntimeError("encoder failed")


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
        session=FixedDetectorSession(
            device=device,
            restoration_pipeline=restoration_pipeline,
            log_callback=None,
            ltx_files=ltx_files,
            detector=MagicMock() if detection_model is None else detection_model,
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


def write_two_block_wma_source(tmp_path: Path) -> Path:
    """WMV whose WMA payloads each carry two blocks, so every second decoded frame has no
    timestamp; one payload is stamped a block early, as ASF send times can be."""
    plain = tmp_path / "plain_wma.wmv"
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ar", "44100", "-ac", "1",
            "-c:v", "wmv2", "-c:a", "wmav2", str(plain),
        ],
        check=True,
    )
    source = tmp_path / "two_block_wma.wmv"
    with av.open(str(plain)) as src, av.open(str(source), "w") as dst:
        video_out = dst.add_stream_from_template(src.streams.video[0])
        audio_out = dst.add_stream_from_template(src.streams.audio[0])
        audio_packets = []
        for packet in src.demux():
            if not packet.size:
                continue
            if packet.stream.type == "audio":
                audio_packets.append(packet)
            else:
                packet.stream = video_out
                dst.mux(packet)
        block_ms = audio_packets[1].pts - audio_packets[0].pts
        for pair_index, first in enumerate(range(0, len(audio_packets) - 1, 2)):
            head, tail = audio_packets[first], audio_packets[first + 1]
            payload = av.Packet(bytes(head) + bytes(tail))
            payload.pts = payload.dts = head.pts - (block_ms * 6 // 5 if pair_index == 3 else 0)
            payload.time_base = head.time_base
            payload.duration = head.duration + tail.duration
            payload.is_keyframe = True
            payload.stream = audio_out
            dst.mux(payload)
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


def write_dated_source(path: Path) -> Path:
    """H.264 + AAC file whose container and streams claim they were made in 2001."""
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            # 384 px wide: the narrowest frame hevc_amf encodes.
            "-f", "lavfi", "-i", "testsrc2=size=384x256:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-metadata", "creation_time=2001-02-03T04:05:06Z",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True,
    )
    return path


def write_transport_stream_with_empty_end(path: Path) -> Path:
    """Two-second 12 fps H.264 MPEG-TS followed by 17 MB of zeros, like an unfinished download.
    ffprobe looks for the last timestamps only in the final 16 MB, so it reports no duration."""
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    with path.open("ab") as stream:
        stream.write(bytes(17_000_000))
    return path


def creation_times(path: Path) -> list[datetime]:
    """Every ``creation_time`` tag of ``path``: its container's, then its streams'."""
    with av.open(str(path)) as container:
        tags = [container.metadata, *(stream.metadata for stream in container.streams)]
        return [datetime.fromisoformat(tag["creation_time"]) for tag in tags if "creation_time" in tag]


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


class AnsweredReply:
    """A worker reply that is already in, as ``ModelRuntime.send`` returns it."""

    def __init__(self, tensors: list):
        self._tensors = tensors

    def answered(self) -> None:
        pass

    def tensors(self) -> list:
        return self._tensors


def answered_at_once(request):
    """``ModelRuntime.send`` for a fake ``request(runtime, op, params, tensors)``: its reply is in at once."""
    return lambda runtime, op, params, tensors: AnsweredReply(request(runtime, op, params, tensors)[1])


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


class TensorDictReader:
    """A model-file reader over in-memory tensors, stored in the dict's order."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors

    def tensor_specs(self):
        return {name: (tensor.dtype, tuple(tensor.shape)) for name, tensor in self._tensors.items()}

    def offset_keys(self):
        return list(self._tensors)

    def read_tensors_into(self, group_names, groups) -> None:
        for names, destinations in zip(group_names, groups, strict=True):
            assert tuple(destinations) == tuple(names)
            for name, destination in destinations.items():
                destination.copy_(self._tensors[name])


def block_store(blocks: list[dict[str, torch.Tensor]], device: torch.device, *, resident: int):
    """A ``BlockStore`` holding ``blocks`` (per block ``{name: tensor}``), read like a model file."""
    from jasna.ltx.blocks import BlockStore

    tensors = {f"blocks.{index}.{name}": tensor for index, block in enumerate(blocks) for name, tensor in block.items()}
    names = [{name: f"blocks.{index}.{name}" for name in block} for index, block in enumerate(blocks)]
    return BlockStore.from_reader(TensorDictReader(tensors), names, device, resident=resident)


def shown_pixel(canvas, x: int, y: int) -> tuple[int, ...]:
    """RGB at ``x``, ``y`` of the one image a canvas shows, in that image's own pixels."""
    (item,) = canvas.find_all()
    assert canvas.type(item) == "image"
    return tuple(int(value) for value in canvas.tk.call(canvas.itemcget(item, "image"), "get", x, y))


def shown_message(canvas) -> str:
    """Text of the one message a canvas shows instead of an image."""
    (item,) = canvas.find_all()
    assert canvas.type(item) == "text"
    return canvas.itemcget(item, "text")


def scheduled_after_ids(widget) -> set[str]:
    """Ids of the Tk ``after`` callbacks still waiting to run."""
    return set(widget.tk.splitlist(widget.tk.call("after", "info")))


class FakeFrameGeneration:
    """Stands in for RtxFrameGeneration: makes each in-between frame with ``blend`` and records the calls."""

    def __init__(self, blend):
        self.blend, self.calls, self.closed = blend, 0, False

    def __call__(self, previous, current):
        self.calls += 1
        return self.blend(previous, current)

    def close(self):
        self.closed = True


class NullProgressBar:
    """Stands in for a stage bar of the LTX ``Progress``: counts the frames or steps done, times nothing."""

    def __init__(self, total: int = 0):
        self.total, self.n = total, 0

    def update(self, n):
        self.n += n

    def reuse(self, done, frames, windows):
        self.n += done

    def window(self, canvas):
        return nullcontext()

    def waiting(self):
        return nullcontext()

    def windows_done(self, canvas, count):
        pass

    def track(self, canvas, steps):
        return self.update

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def track_plan(windows, *, window_hop: int = 3):
    """An LTX ``TrackPlan`` of track 0 from frame 0 to the end of its last window, with no
    polygons and the track owning every frame."""
    from jasna.ltx.plan import TrackPlan

    windows = tuple(windows)
    frames = max(window.stop for window in windows)
    return TrackPlan(0, 0, ((),) * frames, (1.0,) * frames, windows, window_hop)


def ltx_block_weights(dim: int, heads: int, generator: torch.Generator, linear_weights=None) -> dict[str, torch.Tensor]:
    """Random weights of one LTX transformer block. ``linear_weights(name, out_features, in_features)``
    gives each linear's weight tensors (bf16 ``.weight`` when omitted)."""

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    def bf16_linear(name: str, out_features: int, in_features: int) -> dict[str, torch.Tensor]:
        return {f"{name}.weight": rand(out_features, in_features)}

    linear_weights = linear_weights or bf16_linear
    weights = {}
    for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            weights |= linear_weights(f"{attn}.{proj}", dim, dim)
            weights[f"{attn}.{proj}.bias"] = rand(dim)
        weights |= linear_weights(f"{attn}.to_gate_logits", heads, dim)
        weights[f"{attn}.to_gate_logits.bias"] = rand(heads)
        weights[f"{attn}.q_norm.weight"] = rand(dim) + 1
        weights[f"{attn}.k_norm.weight"] = rand(dim) + 1
    weights |= linear_weights("ff.net.0.proj", 4 * dim, dim)
    weights |= linear_weights("ff.net.2", dim, 4 * dim)
    weights["scale_shift_table"] = rand(9, dim)
    weights["prompt_scale_shift_table"] = rand(2, dim)
    return weights


def ltx_block_inputs(dim: int, heads: int, generator: torch.Generator, *, rows: int = 24, context_rows: int = 5,
                     latent_shape: tuple[int, int, int] = (3, 2, 2), device="cpu"):
    """``(x, cond, context, rope)`` of one LTX transformer block on ``device``: ``rows`` tokens, one
    step's conditioning, ``context_rows`` prompt tokens and the rope of a ``latent_shape`` latent.
    Values are drawn on ``generator``'s device in that order."""
    from jasna.ltx import transformer as T

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, device=generator.device) * scale).to(device, torch.bfloat16)

    cond = T.StepConditioning(
        block_rows=randn(1, 2, 9, dim, scale=0.1),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device=device),
        prompt_rows=randn(1, 1, 2, dim, scale=0.1),
    )
    x = randn(1, rows, dim)
    context = randn(1, context_rows, dim)
    rope = T.build_rope(*latent_shape, dim=dim, heads=heads, device=torch.device(device))
    return x, cond, context, rope


def write_sparse_file(path: Path, size: int) -> None:
    """A file of ``size`` zero bytes that takes no disk space. Windows truncate() writes the zeros,
    so the size comes from one byte written at the end of a sparse file."""
    with path.open("wb") as stream:
        if sys.platform == "win32":
            _mark_sparse(stream)
        stream.seek(size - 1)
        stream.write(bytes(1))


def _mark_sparse(stream) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    fsctl_set_sparse = 0x000900C4
    returned = wintypes.DWORD()
    if not ctypes.windll.kernel32.DeviceIoControl(wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno())), fsctl_set_sparse,
                                                  None, 0, None, 0, ctypes.byref(returned), None):
        raise ctypes.WinError()

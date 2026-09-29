"""Builders shared by tests that need a SessionConfig, a Pipeline, or test media."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import torch

from jasna.pipeline import Pipeline
from jasna.os_utils import resolve_executable
from jasna.session_config import SessionConfig


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
        ltx_seed=0,
        ltx_fast=False,
        ltx_model="distilled",
        compile_basicvsrpp=True,
        max_clip_size=90,
        temporal_overlap=8,
        enable_crossfade=True,
        denoise_strength="none",
        denoise_step="after_primary",
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


def fake_session(*, device: torch.device, restoration_pipeline, detection_model=None) -> SimpleNamespace:
    detection_model = MagicMock() if detection_model is None else detection_model
    return SimpleNamespace(
        device=device,
        restoration_pipeline=restoration_pipeline,
        ltx_files=None,
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

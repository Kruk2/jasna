"""Builders shared by tests that need a SessionConfig, a Pipeline, or test media."""
from __future__ import annotations

import struct
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
        ltx_window_hop=88,
        ltx_seed=0,
        ltx_fast=False,
        ltx_model="distilled",
        ltx_trial=False,
        ltx_auto_split_percent=None,
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


def no_resume_conflict(directory: Path):
    raise AssertionError(f"unexpected saved-progress conflict: {directory}")


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


def _length_prefixed_units(data: bytes) -> list[bytes]:
    units, offset = [], 0
    while offset + 4 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "big")
        units.append(data[offset + 4:offset + 4 + size])
        offset += 4 + size
    return units


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

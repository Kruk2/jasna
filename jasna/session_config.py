"""Core-owned typed settings model for pipeline composition.

Holds every parameter the shared composition root (``jasna.session_factory``)
needs. CLI (``jasna.main``) and GUI (``jasna.gui.video_session``) each map
their own settings representation into this one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

SecondaryRestorationName = Literal["none", "unet-4x", "tvai", "rtx-super-res"]
DenoiseStrengthName = Literal["none", "low", "medium", "high"]
DenoiseStepName = Literal["after_primary", "after_secondary"]
VrModeName = Literal["auto", "off", "sbs", "sbs-fisheye"]
RtxQualityName = Literal["low", "medium", "high", "ultra"]
RtxLevelName = Literal["none", "low", "medium", "high", "ultra"]
CodecName = Literal["hevc", "h264", "av1"]


@dataclass(frozen=True)
class SessionConfig:
    device: str
    fp16: bool
    batch_size: int
    detection_model_name: str
    detection_model_path: Path
    detection_score_threshold: float
    max_detection_gap: int
    min_detection_duration: int
    restoration_model_path: Path
    compile_basicvsrpp: bool
    max_clip_size: int
    temporal_overlap: int
    enable_crossfade: bool
    denoise_strength: DenoiseStrengthName
    denoise_step: DenoiseStepName
    secondary_restoration: SecondaryRestorationName
    tvai_ffmpeg_path: str
    tvai_model: str
    tvai_scale: int
    tvai_args: str
    tvai_workers: int
    rtx_scale: int
    rtx_quality: RtxQualityName
    rtx_denoise: RtxLevelName
    rtx_deblur: RtxLevelName
    vr_mode: VrModeName
    codec: CodecName
    encoder_settings: Mapping[str, object]
    lut_path: str | None
    retarget_high_fps: bool
    disable_progress: bool
    working_dir: Path | None

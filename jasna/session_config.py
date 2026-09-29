"""Core-owned typed settings model for pipeline composition.

Holds every parameter the shared composition root (``jasna.session_factory``)
needs. CLI (``jasna.main``) and GUI (``jasna.gui.video_session``) each map
their own settings representation into this one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RestorationModelName = Literal["basicvsrpp", "ltx"]
LtxModelName = Literal["distilled", "undistilled"]
LTX_DEFAULT_MODEL: LtxModelName = "distilled"

LTX_DEFAULT_SEED = 20260923
SecondaryRestorationName = Literal["none", "unet-4x", "tvai", "rtx-super-res"]
DenoiseStrengthName = Literal["none", "low", "medium", "high"]
DenoiseStepName = Literal["after_primary", "after_secondary"]
VrModeName = Literal["auto", "off", "sbs", "sbs-fisheye"]
VrProjectionName = Literal["auto", "raw", "fisheye", "gnomonic"]
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
    scene_detection: bool
    restoration_model_name: RestorationModelName
    restoration_model_path: Path
    ltx_large_canvas: bool
    ltx_seed: int
    ltx_fast: bool
    ltx_model: LtxModelName
    ltx_trial: bool
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
    vr_projection: VrProjectionName
    fmp4: bool
    sharpen_strength: float
    tvai_denoise: bool

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Batch size must be > 0")
        if self.max_clip_size <= 0:
            raise ValueError("Max clip size must be > 0")
        if self.temporal_overlap < 0:
            raise ValueError("Temporal overlap must be >= 0")
        if self.temporal_overlap > 0 and 2 * self.temporal_overlap >= self.max_clip_size:
            raise ValueError("Temporal overlap must satisfy 2 * temporal overlap < max clip size")
        if not 0 <= self.max_detection_gap < self.max_clip_size:
            raise ValueError("Max detection gap must be >= 0 and < max clip size")
        if not 0 <= self.min_detection_duration < self.max_clip_size:
            raise ValueError("Min detection duration must be >= 0 and < max clip size")
        if not 0.0 <= self.detection_score_threshold <= 1.0:
            raise ValueError("Detection score threshold must be in [0, 1]")
        if not 0.0 <= self.sharpen_strength <= 1.0:
            raise ValueError("Sharpening must be in [0, 1]")

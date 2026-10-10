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
SecondaryRestorationName = Literal["none", "unet-4x", "tvai", "rtx-super-res", "amd-upscale"]
AmdUpscaleEngineName = Literal["amf-sr", "real-esr", "realesrgan"]
AmdUpscaleModelName = Literal[
    "auto", "x4v3", "wdn-x4v3", "lsdir-c3", "lsdir-v2", "hfa2k-2x",
    "x4plus", "anime-6b", "bsrnet",
]
DenoiseStrengthName = Literal["none", "low", "medium", "high"]
DenoiseStepName = Literal["after_primary", "after_secondary"]
VrModeName = Literal["auto", "off", "sbs", "sbs-fisheye"]
VrProjectionName = Literal["auto", "raw", "fisheye", "gnomonic"]
RtxQualityName = Literal["low", "medium", "high", "ultra"]
RtxLevelName = Literal["none", "low", "medium", "high", "ultra"]
CodecName = Literal["hevc", "h264", "av1"]

# Row 1 of the AMD upscale panel ("engine") is the class, row 2 ("model") picks the
# concrete checkpoint inside that class. The classes are disjoint on purpose, so a
# checkpoint never sits under a class it does not belong to:
#   real-esr   - the SRVGGNetCompact family (VGG stack, no convolution at output size,
#                16-32 convs): realesr-general-x4v3 / -wdn-x4v3, 4xLSDIRCompactC3,
#                4xLSDIRCompactv2 and 2xHFA2kCompact.
#   realesrgan - the RRDBNet family (23-block / 6-block, quality ceiling):
#                RealESRGAN_x4plus / x4plus_anime_6B and BSRNet.
#   amf-sr     - AMD's driver-level Video SR (AMF filter sr_amf): no checkpoint at all.
AMD_UPSCALE_ENGINE_ORDER: tuple[str, ...] = ("amf-sr", "real-esr", "realesrgan")
AMD_UPSCALE_ENGINE_MODELS: dict[str, tuple[str, ...]] = {
    "amf-sr": (),
    "real-esr": ("auto", "x4v3", "wdn-x4v3", "lsdir-c3", "lsdir-v2", "hfa2k-2x"),
    "realesrgan": ("auto", "x4plus", "anime-6b", "bsrnet"),
}
# "auto" fits every class (it takes the first checkpoint found), so the reverse map
# only covers the explicit presets, which is what repairs stale preset files.
AMD_UPSCALE_MODEL_ENGINE: dict[str, str] = {
    model: engine
    for engine, models in AMD_UPSCALE_ENGINE_MODELS.items()
    for model in models
    if model != "auto"
}
AMD_UPSCALE_MODEL_DEFAULT: dict[str, str] = {
    "amf-sr": "auto",
    "real-esr": "x4v3",
    "realesrgan": "x4plus",
}


def amd_upscale_engine_for(value: str, model: str) -> str:
    """The engine a checkpoint belongs to, so a stale engine value stays consistent.

    Presets written before the model row was split into classes record ``realesrgan``
    for every checkpoint, SRVGGNetCompact ones included. The checkpoint decides here,
    so such a preset shows up under the right class again instead of under a class
    whose name does not cover it.
    """
    engine = str(value or "").strip().lower()
    owner = AMD_UPSCALE_MODEL_ENGINE.get(str(model or "").strip().lower())
    if owner is not None:
        return owner
    return engine if engine in AMD_UPSCALE_ENGINE_ORDER else "real-esr"


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
    amd_upscale_engine: AmdUpscaleEngineName
    amd_upscale_model: AmdUpscaleModelName
    amd_upscale_scale: int
    amd_upscale_algorithm: str
    amd_upscale_sharpness: float
    amd_upscale_ffmpeg_path: str | None
    amd_upscale_model_path: str | None
    amd_upscale_timeout_s: float
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

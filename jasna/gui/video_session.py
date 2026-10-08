"""GUI adapter around the shared composition root (``jasna.session_factory``).

Maps ``AppSettings`` to the core ``SessionConfig`` (the only place a
GUI-settings-to-internal-value mapping may live) and builds the heavy video
restoration session for both the background job Processor and the
segment-editor restoration preview, plus the still-image session.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from jasna.gui.models import AppSettings
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession

if TYPE_CHECKING:
    import torch


def video_session_key(settings: AppSettings) -> tuple:
    key = (
        settings.detection_model,
        settings.detection_score_threshold,
        settings.batch_size,
        settings.fp16_mode,
        settings.max_clip_size,
        settings.temporal_overlap,  # defines the clip lengths the restorer pre-captures
        settings.compile_basicvsrpp,
        settings.denoise_strength,
        settings.denoise_step,
        settings.secondary_restoration,
        settings.restoration_model,
        settings.ltx_model,
        settings.ltx_fast,
        settings.ltx_trial,
    )
    if settings.secondary_restoration == "tvai":
        key += (
            settings.tvai_ffmpeg_path,
            settings.tvai_model,
            settings.tvai_scale,
            settings.tvai_workers,
            settings.tvai_args,
            settings.tvai_denoise,
        )
    elif settings.secondary_restoration == "rtx-super-res":
        key += (
            settings.rtx_scale,
            settings.rtx_quality,
            settings.rtx_denoise,
            settings.rtx_deblur,
        )
    elif settings.secondary_restoration == "amd-upscale":
        key += (
            settings.amd_upscale_engine,
            settings.amd_upscale_scale,
            settings.amd_upscale_algorithm,
            settings.amd_upscale_sharpness,
            settings.amd_upscale_ffmpeg_path,
            settings.amd_upscale_timeout_s,
        )
    return key


def video_session_config(
    settings: AppSettings,
    *,
    codec: str,
    encoder_settings: Mapping[str, object],
) -> SessionConfig:
    from jasna.engine_paths import default_restoration_model_path
    from jasna.mosaic.detection_registry import coerce_detection_model_name, require_detection_model_weights

    det_name = coerce_detection_model_name(str(settings.detection_model))
    standard_model = settings.restoration_model == "basicvsrpp"
    secondary_restoration = settings.secondary_restoration if standard_model else "none"
    return SessionConfig(
        device="cuda:0",
        fp16=bool(settings.fp16_mode),
        batch_size=int(settings.batch_size),
        detection_model_name=det_name,
        detection_model_path=require_detection_model_weights(det_name),
        detection_score_threshold=float(settings.detection_score_threshold),
        max_detection_gap=int(settings.max_detection_gap),
        min_detection_duration=int(settings.min_detection_duration),
        scene_detection=bool(settings.scene_detection),
        restoration_model_name=settings.restoration_model,
        restoration_model_path=default_restoration_model_path(settings.restoration_model),
        ltx_large_canvas=bool(settings.ltx_large_canvas),
        ltx_seed=int(settings.ltx_seed),
        ltx_fast=bool(settings.ltx_fast),
        ltx_model=settings.ltx_model,
        ltx_trial=bool(settings.ltx_trial),
        compile_basicvsrpp=bool(settings.compile_basicvsrpp),
        max_clip_size=int(settings.max_clip_size),
        temporal_overlap=int(settings.temporal_overlap),
        enable_crossfade=bool(settings.enable_crossfade),
        denoise_strength=settings.denoise_strength if standard_model else "none",
        denoise_step=settings.denoise_step,
        secondary_restoration=secondary_restoration,
        tvai_ffmpeg_path=settings.tvai_ffmpeg_path,
        tvai_model=settings.tvai_model,
        tvai_scale=int(settings.tvai_scale),
        tvai_args=settings.tvai_args,
        tvai_workers=int(settings.tvai_workers),
        tvai_denoise=bool(settings.tvai_denoise and secondary_restoration == "tvai"),
        rtx_scale=int(settings.rtx_scale),
        rtx_quality=settings.rtx_quality.lower(),
        rtx_denoise=settings.rtx_denoise.lower(),
        rtx_deblur=settings.rtx_deblur.lower(),
        amd_upscale_engine=str(settings.amd_upscale_engine).lower(),
        amd_upscale_scale=int(settings.amd_upscale_scale),
        amd_upscale_algorithm=str(settings.amd_upscale_algorithm).lower(),
        amd_upscale_sharpness=float(settings.amd_upscale_sharpness),
        amd_upscale_ffmpeg_path=(str(settings.amd_upscale_ffmpeg_path).strip() or None),
        amd_upscale_model_path=None,
        amd_upscale_timeout_s=float(settings.amd_upscale_timeout_s),
        vr_mode=settings.vr_mode,
        vr_projection=settings.vr_projection,
        codec=codec,
        encoder_settings=dict(encoder_settings),
        lut_path=(settings.lut_path or "").strip() or None,
        sharpen_strength=float(settings.sharpen_strength),
        retarget_high_fps=bool(settings.retarget_high_fps),
        fmp4=bool(settings.fmp4),
        disable_progress=True,
        working_dir=Path(settings.working_directory) if settings.working_directory else None,
    )


def build_video_session(
    settings: AppSettings,
    *,
    log: Callable[[str], None],
) -> RestorationSession:
    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    from jasna.session_factory import build_restoration_session

    config = video_session_config(settings, codec=settings.codec, encoder_settings={})
    return build_restoration_session(config, log_callback=log)


def build_image_session(
    settings: AppSettings,
    *,
    log: Callable[[str], None] | None,
) -> tuple:
    """Load the mosaic detector and SD 1.5 inpaint restorer used for still images."""
    import torch

    from jasna._suppress_noise import install as _install_noise_filters
    from jasna.engine_paths import SD15_DIR
    from jasna.gui.locales import t
    from jasna.mosaic.detection_registry import resolve_detection_model
    from jasna.restorer.sd15_download import bundle_present
    from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer
    from jasna.session_factory import build_compiled_detection_model

    _install_noise_filters()
    if not bundle_present(SD15_DIR):
        raise FileNotFoundError(t("interactive_model_missing"))
    device = torch.device("cuda:0")
    detection_model_name, detection_model_path, _ = resolve_detection_model(
        str(settings.detection_model), "", None
    )
    detector = build_compiled_detection_model(
        detection_model_name,
        detection_model_path,
        device=device,
        batch_size=settings.batch_size,
        fp16=settings.fp16_mode,
        score_threshold=settings.detection_score_threshold,
        log_callback=log,
    )
    return detector, Sd15InpaintRestorer(SD15_DIR, device, settings.fp16_mode), device


def release_session_memory(device: "torch.device") -> None:
    import gc
    import torch
    from jasna.accelerator import empty_cache, ipc_collect, synchronize

    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        synchronize(device)
        empty_cache(device)
        ipc_collect(device)

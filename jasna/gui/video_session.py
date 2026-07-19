"""GUI adapter around the shared composition root (``jasna.session_factory``).

Maps ``AppSettings`` to the core ``SessionConfig`` (the only place a
GUI-settings-to-internal-value mapping may live) and builds the heavy video
restoration session for both the background job Processor and the
segment-editor restoration preview.
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
        settings.compile_basicvsrpp,
        settings.denoise_strength,
        settings.denoise_step,
        settings.secondary_restoration,
    )
    if settings.secondary_restoration == "tvai":
        key += (
            settings.tvai_ffmpeg_path,
            settings.tvai_model,
            settings.tvai_scale,
            settings.tvai_workers,
            settings.tvai_args,
        )
    elif settings.secondary_restoration == "rtx-super-res":
        key += (
            settings.rtx_scale,
            settings.rtx_quality,
            settings.rtx_denoise,
            settings.rtx_deblur,
        )
    return key


def video_session_config(
    settings: AppSettings,
    *,
    codec: str,
    encoder_settings: Mapping[str, object],
) -> SessionConfig:
    from jasna.engine_paths import model_weights_dir
    from jasna.mosaic.detection_registry import coerce_detection_model_name, require_detection_model_weights

    det_name = coerce_detection_model_name(str(settings.detection_model))
    return SessionConfig(
        device="cuda:0",
        fp16=bool(settings.fp16_mode),
        batch_size=int(settings.batch_size),
        detection_model_name=det_name,
        detection_model_path=require_detection_model_weights(det_name),
        detection_score_threshold=float(settings.detection_score_threshold),
        max_detection_gap=int(settings.max_detection_gap),
        min_detection_duration=int(settings.min_detection_duration),
        restoration_model_path=model_weights_dir() / "lada_mosaic_restoration_model_generic_v1.2.pth",
        compile_basicvsrpp=bool(settings.compile_basicvsrpp),
        max_clip_size=int(settings.max_clip_size),
        temporal_overlap=int(settings.temporal_overlap),
        enable_crossfade=bool(settings.enable_crossfade),
        denoise_strength=settings.denoise_strength,
        denoise_step=settings.denoise_step,
        secondary_restoration=settings.secondary_restoration,
        tvai_ffmpeg_path=settings.tvai_ffmpeg_path,
        tvai_model=settings.tvai_model,
        tvai_scale=int(settings.tvai_scale),
        tvai_args=settings.tvai_args,
        tvai_workers=int(settings.tvai_workers),
        rtx_scale=int(settings.rtx_scale),
        rtx_quality=settings.rtx_quality.lower(),
        rtx_denoise=settings.rtx_denoise.lower(),
        rtx_deblur=settings.rtx_deblur.lower(),
        vr_mode=settings.vr_mode,
        codec=codec,
        encoder_settings=dict(encoder_settings),
        lut_path=(settings.lut_path or "").strip() or None,
        retarget_high_fps=bool(settings.retarget_high_fps),
        disable_progress=True,
        working_dir=Path(settings.working_directory) if settings.working_directory else None,
    )


def build_video_session(
    settings: AppSettings,
    *,
    disable_basicvsrpp_tensorrt: bool,
    log: Callable[[str], None],
) -> RestorationSession:
    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    from jasna.session_factory import build_restoration_session

    config = video_session_config(settings, codec=settings.codec, encoder_settings={})
    return build_restoration_session(
        config,
        disable_basicvsrpp_tensorrt=disable_basicvsrpp_tensorrt,
        log_callback=log,
    )


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

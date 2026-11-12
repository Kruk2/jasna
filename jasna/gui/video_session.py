"""GUI adapter around the shared composition root (``jasna.session_factory``).

Maps ``AppSettings`` to the core ``SessionConfig`` (the only place a
GUI-settings-to-internal-value mapping may live) and builds the heavy video
restoration session for both the background job Processor and the
segment-editor restoration preview, plus the still-image session.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Hashable, Mapping

from jasna.gui.models import AppSettings
from jasna.session_config import SessionConfig, ltx_unsupported_options
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
        settings.secondary_restoration,
        settings.restoration_model,
        settings.ltx_model,
        settings.ltx_fast,
        settings.ltx_vae_encoder,
        settings.ltx_vae_decoder,
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
    return key


def video_session_config(
    settings: AppSettings,
    *,
    codec: str,
    encoder_settings: Mapping[str, object],
) -> SessionConfig:
    from jasna.engine_paths import default_restoration_model_path
    from jasna.ltx.plan import window_hop
    from jasna.mosaic.detection_registry import coerce_detection_model_name, require_detection_model_weights

    det_name = coerce_detection_model_name(str(settings.detection_model))
    standard_model = settings.restoration_model == "basicvsrpp"
    ltx_unsupported = () if standard_model else ltx_unsupported_options(
        auto_split=settings.ltx_auto_split,
        secondary_restoration=settings.secondary_restoration,
        denoise_strength=settings.denoise_strength,
    )
    secondary_restoration = "none" if "secondary_restoration" in ltx_unsupported else settings.secondary_restoration
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
        ltx_window_hop=window_hop(int(settings.ltx_overlap)),
        ltx_seed=int(settings.ltx_seed),
        ltx_fast=bool(settings.ltx_fast),
        ltx_model=settings.ltx_model,
        ltx_vae_encoder=settings.ltx_vae_encoder,
        ltx_vae_decoder=settings.ltx_vae_decoder,
        ltx_trial=bool(settings.ltx_trial),
        ltx_auto_split_percent=float(settings.ltx_auto_split_percent) if settings.ltx_auto_split else None,
        compile_basicvsrpp=bool(settings.compile_basicvsrpp),
        max_clip_size=int(settings.max_clip_size),
        temporal_overlap=int(settings.temporal_overlap),
        denoise_strength="none" if "denoise_strength" in ltx_unsupported else settings.denoise_strength,
        secondary_restoration=secondary_restoration,
        tvai_ffmpeg_path=settings.tvai_ffmpeg_path,
        tvai_model=settings.tvai_model,
        tvai_scale=int(settings.tvai_scale),
        tvai_args=settings.tvai_args,
        tvai_workers=int(settings.tvai_workers),
        tvai_denoise=bool(settings.tvai_denoise and secondary_restoration == "tvai"),
        decode_backend=settings.decode_backend,
        rtx_scale=int(settings.rtx_scale),
        rtx_quality=settings.rtx_quality.lower(),
        rtx_denoise=settings.rtx_denoise.lower(),
        rtx_deblur=settings.rtx_deblur.lower(),
        vr_mode=settings.vr_mode,
        vr_projection=settings.vr_projection,
        codec=codec,
        encoder_settings=dict(encoder_settings),
        lut_path=(settings.lut_path or "").strip() or None,
        sharpen_strength=float(settings.sharpen_strength),
        retarget_high_fps=bool(settings.retarget_high_fps),
        frame_generation=settings.frame_generation,
        upscale_model=settings.upscale_model,
        upscale_height=int(settings.upscale_height),
        upscale_strength=float(settings.upscale_strength),
        upscale_rtx_quality=settings.upscale_rtx_quality,
        upscale_rtx_denoise=settings.upscale_rtx_denoise,
        upscale_rtx_deblur=settings.upscale_rtx_deblur,
        upscale_dlss5_preset=settings.upscale_dlss5_preset,
        upscale_dlss5_style=settings.upscale_dlss5_style,
        upscale_dlss5_intensity=float(settings.upscale_dlss5_intensity),
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


class SessionCache:
    """One thread's video session, rebuilt when ``key(settings)`` changes.

    Not thread-safe: each worker owns its cache and calls it from its own thread.
    ``on_close`` drops what still holds the old session (a pipeline, an LTX runtime)
    before it is closed and its GPU memory released.
    """

    def __init__(
        self,
        *,
        key: Callable[[AppSettings], Hashable] = video_session_key,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._key_of = key
        self._on_close = on_close
        self._key: Hashable = None
        self.session: RestorationSession | None = None

    def get(self, settings: AppSettings, build: Callable[[], RestorationSession]) -> RestorationSession:
        key = self._key_of(settings)
        if self.session is None or key != self._key:
            self.close()
            self.session = build()
            self._key = key
        return self.session

    def close(self) -> None:
        session = self.session
        if session is None:
            return
        if self._on_close is not None:
            self._on_close()
        self.session = None
        self._key = None
        try:
            session.close()
        finally:
            release_session_memory(session.device)

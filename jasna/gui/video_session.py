"""Shared factory for the heavy video-restoration session (engines + restorers).

Used by both the background job Processor and the segment-editor restoration
preview so both build the exact same restorer stack from AppSettings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jasna.gui.models import AppSettings

if TYPE_CHECKING:
    import torch
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.secondary_restorer import SecondaryRestorer


@dataclass
class VideoSession:
    device: "torch.device"
    det_name: str
    detection_model_path: Path
    restoration_pipeline: "RestorationPipeline"
    secondary_restorer: "SecondaryRestorer | None"
    lut_path: str | None

    def close(self) -> None:
        self.restoration_pipeline.restorer.close()
        if self.secondary_restorer is not None and hasattr(self.secondary_restorer, "close"):
            self.secondary_restorer.close()
        import gc
        import torch

        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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


def build_video_session(
    settings: AppSettings,
    *,
    disable_basicvsrpp_tensorrt: bool,
    log: Callable[[str], None],
) -> VideoSession:
    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    import torch
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.engine_paths import model_weights_dir
    from jasna.mosaic.detection_registry import coerce_detection_model_name, require_detection_model_weights
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    device = torch.device("cuda:0")
    restoration_model_path = model_weights_dir() / "lada_mosaic_restoration_model_generic_v1.2.pth"
    det_name = coerce_detection_model_name(str(settings.detection_model))
    detection_model_path = require_detection_model_weights(det_name)

    compile_basicvsrpp = bool(settings.compile_basicvsrpp) and (not disable_basicvsrpp_tensorrt)
    compile_result = ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=settings.fp16_mode,
            basicvsrpp=compile_basicvsrpp,
            basicvsrpp_model_path=str(restoration_model_path),
            basicvsrpp_max_clip_size=int(settings.max_clip_size),
            detection=True,
            detection_model_name=det_name,
            detection_model_path=str(detection_model_path),
            detection_batch_size=settings.batch_size,
            unet4x=(settings.secondary_restoration == "unet-4x"),
        ),
        log_callback=log,
    )
    use_tensorrt = compile_result.use_basicvsrpp_tensorrt

    secondary_restorer = None
    if settings.secondary_restoration == "tvai":
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
        tvai_args_str = f"model={settings.tvai_model}:scale={settings.tvai_scale}:{settings.tvai_args}"
        secondary_restorer = TvaiSecondaryRestorer(
            ffmpeg_path=settings.tvai_ffmpeg_path,
            tvai_args=tvai_args_str,
            scale=settings.tvai_scale,
            num_workers=settings.tvai_workers,
        )
    elif settings.secondary_restoration == "unet-4x":
        from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer
        secondary_restorer = Unet4xSecondaryRestorer(device=device, fp16=settings.fp16_mode)
    elif settings.secondary_restoration == "rtx-super-res":
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
        rtx_denoise = settings.rtx_denoise.lower()
        rtx_deblur = settings.rtx_deblur.lower()
        secondary_restorer = RtxSuperresSecondaryRestorer(
            device=device,
            scale=settings.rtx_scale,
            quality=settings.rtx_quality.lower(),
            denoise=None if rtx_denoise == "none" else rtx_denoise,
            deblur=None if rtx_deblur == "none" else rtx_deblur,
        )

    restoration_pipeline = RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(restoration_model_path),
            device=device,
            max_clip_size=settings.max_clip_size,
            use_tensorrt=use_tensorrt,
            fp16=settings.fp16_mode,
        ),
        secondary_restorer=secondary_restorer,
        denoise_strength=DenoiseStrength(settings.denoise_strength),
        denoise_step=DenoiseStep(settings.denoise_step),
    )

    return VideoSession(
        device=device,
        det_name=det_name,
        detection_model_path=detection_model_path,
        restoration_pipeline=restoration_pipeline,
        secondary_restorer=secondary_restorer,
        lut_path=(settings.lut_path or "").strip() or None,
    )

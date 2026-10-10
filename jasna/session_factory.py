"""Shared composition root for the video restoration pipeline.

Builds the heavy restoration session (engine compilation, primary and
secondary restorers, and a detector cached across videos) and per-video
``Pipeline`` instances from one ``SessionConfig``. Consumed by both the CLI (``jasna.main``) and the GUI
(``jasna.gui.video_session`` / ``jasna.gui.processor``).

All heavy imports (torch, restorers, pipeline) stay inside the functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jasna.session_config import RestorationModelName, SessionConfig, amd_upscale_engine_for

if TYPE_CHECKING:
    import torch

    from jasna.ltx.model_files import LtxModelFiles
    from jasna.media.splice import SplicePlan
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    from jasna.pipeline import Pipeline
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.segments import SegmentRange

    DetectionModel = RfDetrMosaicDetectionModel | YoloMosaicDetectionModel


logger = logging.getLogger(__name__)

# Models that stay resident in VRAM across sessions: the secondary restorer
# (Real-ESRGAN / TVAI / RTX / UNet4x) and the mosaic detector. Each entry is
# keyed by everything that defines the instance, so an unrelated settings
# change (batch size, denoise, clip length, LTX model...) reuses the resident
# copy instead of paying the disk load and the MIOpen warm-up again. They are
# released when their own settings change, or by release_shared_models() on
# app exit.
_SHARED_MODELS: dict[tuple, object] = {}


def release_shared_models() -> None:
    """Close the models shared across sessions (app exit / explicit request)."""
    for key, model in list(_SHARED_MODELS.items()):
        closer = getattr(model, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover - best effort
                logger.debug("releasing shared model %s failed", key, exc_info=True)
        _SHARED_MODELS.pop(key, None)
    logger.info("Shared models released")


@dataclass
class RestorationSession:
    device: "torch.device"
    restoration_pipeline: "RestorationPipeline | None"
    ltx_files: "LtxModelFiles | None" = None

    def detection_model_for(self, config: SessionConfig) -> "DetectionModel":
        """The detector for ``config``, shared across sessions until its settings change."""
        from jasna.mosaic.detection_registry import build_detection_model

        key = (
            "detection",
            config.detection_model_name,
            str(config.detection_model_path),
            config.detection_score_threshold,
            config.batch_size,
            config.fp16,
            str(self.device),
        )
        model = _SHARED_MODELS.get(key)
        if model is not None:
            logger.debug("Detection model reused from VRAM (shared across sessions)")
            return model
        model = build_detection_model(
            config.detection_model_name,
            config.detection_model_path,
            batch_size=config.batch_size,
            device=self.device,
            score_threshold=config.detection_score_threshold,
            fp16=config.fp16,
        )
        _SHARED_MODELS[key] = model
        logger.info("Detection model loaded (shared across sessions, resident in VRAM)")
        return model

    def close(self) -> None:
        # The detector and the secondary restorer are shared across sessions (see
        # _SHARED_MODELS): they are released by release_shared_models(), not here,
        # so an unrelated settings change does not reload them from disk.
        if self.restoration_pipeline is not None:
            self.restoration_pipeline.restorer.close()


def build_compiled_detection_model(
    detection_model_name: str,
    detection_model_path: Path,
    *,
    device: "torch.device",
    batch_size: int,
    fp16: bool,
    score_threshold: float,
    log_callback: Callable[[str], None] | None,
) -> "DetectionModel":
    """Compile the detector's TensorRT engine if it is missing, then load the detector."""
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.mosaic.detection_registry import build_detection_model

    ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=fp16,
            detection=True,
            detection_model_name=detection_model_name,
            detection_model_path=str(detection_model_path),
            detection_batch_size=batch_size,
        ),
        log_callback=log_callback,
    )
    return build_detection_model(
        detection_model_name,
        detection_model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )


def _secondary_cache_key(config: SessionConfig, device: "torch.device") -> tuple:
    """Everything that defines the secondary restorer instance."""
    return (
        "secondary",
        config.secondary_restoration,
        config.amd_upscale_engine,
        config.amd_upscale_model,
        str(config.amd_upscale_model_path),
        config.amd_upscale_scale,
        config.amd_upscale_algorithm,
        config.amd_upscale_sharpness,
        str(config.amd_upscale_ffmpeg_path),
        config.amd_upscale_timeout_s,
        config.tvai_ffmpeg_path,
        config.tvai_model,
        config.tvai_scale,
        config.tvai_workers,
        config.tvai_args,
        config.tvai_denoise,
        config.rtx_scale,
        config.rtx_quality,
        config.rtx_denoise,
        config.rtx_deblur,
        config.fp16,
        str(device),
    )


def _build_secondary_restorer(config: SessionConfig, device: "torch.device"):
    """Build the secondary restorer, or reuse the one already resident in VRAM.

    Keyed by the restorer's own settings only: an unrelated settings change
    (batch size, denoise, clip length, LTX model...) keeps it resident instead
    of paying the disk load and the MIOpen warm-up again. Released when its own
    settings change, or by release_shared_models() on app exit.
    """
    if config.secondary_restoration == "none":
        return None
    key = _secondary_cache_key(config, device)
    cached = _SHARED_MODELS.get(key)
    if cached is not None:
        logger.info("Secondary restorer reused from VRAM (shared, not reloaded)")
        return cached
    restorer = _build_secondary_restorer_uncached(config, device)
    for old_key in [k for k in _SHARED_MODELS if k[0] == "secondary" and k != key]:
        old = _SHARED_MODELS.pop(old_key)
        closer = getattr(old, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover - best effort
                logger.debug("releasing the previous secondary restorer failed", exc_info=True)
    _SHARED_MODELS[key] = restorer
    logger.info("Secondary restorer loaded (shared across sessions, resident in VRAM)")
    return restorer


def _build_secondary_restorer_uncached(config: SessionConfig, device: "torch.device"):
    if config.secondary_restoration == "none":
        return None
    if config.secondary_restoration == "tvai":
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer

        tvai_args = f"model={config.tvai_model}:scale={config.tvai_scale}:{config.tvai_args}"
        return TvaiSecondaryRestorer(
            ffmpeg_path=config.tvai_ffmpeg_path,
            tvai_args=tvai_args,
            tvai_denoise=bool(config.tvai_denoise),
            scale=int(config.tvai_scale),
            num_workers=int(config.tvai_workers),
        )
    if config.secondary_restoration == "unet-4x":
        from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer

        return Unet4xSecondaryRestorer(device=device, fp16=bool(config.fp16))
    if config.secondary_restoration == "rtx-super-res":
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer

        return RtxSuperresSecondaryRestorer(
            device=device,
            scale=int(config.rtx_scale),
            quality=config.rtx_quality,
            denoise=None if config.rtx_denoise == "none" else config.rtx_denoise,
            deblur=None if config.rtx_deblur == "none" else config.rtx_deblur,
        )
    if config.secondary_restoration == "amd-upscale":
        # Row 1 of the AMD panel picks the class: amf-sr is AMD's D3D11 Video SR
        # filter, while real-esr and realesrgan both run a network in-process on the
        # ROCm device (SRVGGNetCompact vs RRDBNet). The weight in row 2 selects which
        # checkpoint; ``amd_upscale_engine_for`` keeps the two in sync for old presets.
        engine = amd_upscale_engine_for(config.amd_upscale_engine, config.amd_upscale_model)
        if engine == "amf-sr":
            from jasna.restorer.amd_upscale_secondary_restorer import AmdUpscaleSecondaryRestorer

            return AmdUpscaleSecondaryRestorer(
                device=device,
                scale=int(config.amd_upscale_scale),
                engine=engine,
                algorithm=config.amd_upscale_algorithm,
                sharpness=float(config.amd_upscale_sharpness),
                ffmpeg_path=config.amd_upscale_ffmpeg_path,
                timeout_s=float(config.amd_upscale_timeout_s),
            )

        from jasna.restorer.realesrgan_secondary_restorer import (
            RealEsrganSecondaryRestorer,
        )

        return RealEsrganSecondaryRestorer(
            device=device,
            scale=int(config.amd_upscale_scale),
            model_path=config.amd_upscale_model_path,
            model=str(config.amd_upscale_model),
            fp16=bool(config.fp16),
        )
    raise ValueError(f"Unsupported secondary restoration: {config.secondary_restoration}")


def _resolve_device(requested: str) -> "torch.device":
    """The device to run on, preferring a discrete Radeon over an iGPU at index 0.

    Some driver builds enumerate a Ryzen iGPU first, and everything that used ``cuda:0``
    then targeted it: the whole pipeline would run on the slowest device in the machine,
    and the MIGraphX engine aborts there because ``gfx103x`` has no device code
    (``RUNTIME_EXCEPTION ... Failed to call function``). Only the default (``cuda`` /
    ``cuda:0``) is remapped, so an explicit ``--device cuda:1`` keeps its meaning, and a
    machine with no discrete Radeon (Strix Halo, NVIDIA, CPU) is unaffected.
    """
    import torch

    from jasna.accelerator import preferred_gpu_index

    device = torch.device(requested)
    if device.type != "cuda" or device.index not in (None, 0):
        return device
    index = preferred_gpu_index()
    return torch.device(f"cuda:{index}") if index else device


def build_restoration_session(
    config: SessionConfig,
    *,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device

    device = _resolve_device(config.device)
    if log_callback is not None and device != torch.device(config.device):
        log_callback(
            f"device 0 is not a discrete Radeon; running on {device} instead "
            f"(the requested '{config.device}' was the default, pass --device to override)"
        )
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    if config.secondary_restoration == "amd-upscale" and not is_amd_device(device):
        raise ValueError(
            "Secondary restoration 'amd-upscale' requires an AMD GPU (it uses the AMF video upscaler)"
        )
    if is_amd_device(device) and config.secondary_restoration not in ("none", "amd-upscale"):
        raise ValueError(
            f"Secondary restoration '{config.secondary_restoration}' is not available in the AMD build yet"
        )
    if config.restoration_model_name == "ltx" and (
        config.secondary_restoration != "none" or config.denoise_strength != "none"
    ):
        raise ValueError("LTX restoration does not support secondary restoration or denoise")
    session = RestorationSession(device=device, restoration_pipeline=None)
    provide_restoration_models(
        config, session, frozenset({config.restoration_model_name}), log_callback=log_callback
    )
    return session


def provide_restoration_models(
    config: SessionConfig,
    session: RestorationSession,
    models: frozenset[RestorationModelName],
    *,
    log_callback: Callable[[str], None] | None,
) -> None:
    """Load each of ``models`` the session does not hold yet. ``--restoration-model-path``
    belongs to the job's model; another model loads from its default path."""
    if "basicvsrpp" in models and session.restoration_pipeline is None:
        session.restoration_pipeline = _build_basicvsrpp_pipeline(config, session.device, log_callback=log_callback)
    if "ltx" in models and session.ltx_files is None:
        session.ltx_files = _ltx_model_files(config, session.device, log_callback=log_callback)


def _restoration_model_path(config: SessionConfig, name: RestorationModelName) -> Path:
    from jasna.engine_paths import default_restoration_model_path

    if name == config.restoration_model_name:
        return config.restoration_model_path
    return default_restoration_model_path(name)


def _build_basicvsrpp_pipeline(
    config: SessionConfig, device: "torch.device", *, log_callback: Callable[[str], None] | None
) -> "RestorationPipeline":
    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    model_path = _restoration_model_path(config, "basicvsrpp")
    compile_result = ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=bool(config.fp16),
            basicvsrpp=config.compile_basicvsrpp and not is_amd_device(device),
            basicvsrpp_model_path=str(model_path),
            detection=True,
            detection_model_name=config.detection_model_name,
            detection_model_path=str(config.detection_model_path),
            detection_batch_size=int(config.batch_size),
            unet4x=(config.secondary_restoration == "unet-4x"),
        ),
        log_callback=log_callback,
    )
    return RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(model_path),
            device=device,
            max_clip_size=int(config.max_clip_size),
            use_tensorrt=compile_result.use_basicvsrpp_tensorrt,
            fp16=bool(config.fp16),
            temporal_overlap=int(config.temporal_overlap),
        ),
        secondary_restorer=_build_secondary_restorer(config, device),
        denoise_strength=DenoiseStrength(config.denoise_strength),
        denoise_step=DenoiseStep(config.denoise_step),
    )


def _ltx_model_files(
    config: SessionConfig, device: "torch.device", *, log_callback: Callable[[str], None] | None
) -> "LtxModelFiles":
    import torch

    from jasna.accelerator import is_nvidia_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.ltx.model_files import LtxModelFiles

    if not is_nvidia_device(device):
        raise ValueError("LTX restoration needs an NVIDIA GPU")
    if config.ltx_fast and torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("The fast LTX model needs an RTX 50-series (Blackwell) GPU")
    if config.ltx_trial:
        from jasna.ltx.model_files import LTX_TRIAL_NOTICE

        logger.warning(LTX_TRIAL_NOTICE)
        files = LtxModelFiles.placeholder(config.ltx_model, fast=config.ltx_fast)
    else:
        files = LtxModelFiles.from_dir(_restoration_model_path(config, "ltx"), config.ltx_model, fast=config.ltx_fast)
    ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=bool(config.fp16),
            detection=True,
            detection_model_name=config.detection_model_name,
            detection_model_path=str(config.detection_model_path),
            detection_batch_size=int(config.batch_size),
        ),
        log_callback=log_callback,
    )
    return files


def build_pipeline(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    progress_callback: Callable | None = None,
    segments: "tuple[SegmentRange, ...] | None" = None,
    splice_plan: "SplicePlan | None" = None,
) -> "Pipeline":
    """A per-video ``Pipeline``; the session first loads any model the segments ask for."""
    from jasna.pipeline import Pipeline
    from jasna.segments import job_restoration, resolve_restorations

    default = job_restoration(config.restoration_model_name, config.ltx_seed)
    models = frozenset(
        segment.restoration.model for segment in resolve_restorations(tuple(segments or ()), default)
    ) or frozenset({config.restoration_model_name})
    provide_restoration_models(config, session, models, log_callback=None)
    return Pipeline(
        config=config,
        session=session,
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
    )

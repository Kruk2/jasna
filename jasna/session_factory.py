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

from jasna.session_config import RestorationModelName, SessionConfig

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


@dataclass
class RestorationSession:
    device: "torch.device"
    restoration_pipeline: "RestorationPipeline | None"
    ltx_files: "LtxModelFiles | None" = None
    _detection_key: tuple | None = None
    _detection_model: "DetectionModel | None" = None

    def detection_model_for(self, config: SessionConfig) -> "DetectionModel":
        """The detector for ``config``, reused across videos until its settings change."""
        from jasna.mosaic.detection_registry import build_detection_model

        key = (
            config.detection_model_name,
            config.detection_model_path,
            config.detection_score_threshold,
            config.batch_size,
            config.fp16,
        )
        if key != self._detection_key:
            if self._detection_model is not None:
                self._detection_model.close()
            self._detection_model = build_detection_model(
                config.detection_model_name,
                config.detection_model_path,
                batch_size=config.batch_size,
                device=self.device,
                score_threshold=config.detection_score_threshold,
                fp16=config.fp16,
            )
            self._detection_key = key
        return self._detection_model

    def close(self) -> None:
        if self._detection_model is not None:
            self._detection_model.close()
            self._detection_model = None
        if self.restoration_pipeline is not None:
            self.restoration_pipeline.restorer.close()
            if self.restoration_pipeline.secondary_restorer is not None:
                self.restoration_pipeline.secondary_restorer.close()


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


def _build_secondary_restorer(config: SessionConfig, device: "torch.device"):
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
    raise ValueError(f"Unsupported secondary restoration: {config.secondary_restoration}")


def build_restoration_session(
    config: SessionConfig,
    *,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device

    device = torch.device(config.device)
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    if is_amd_device(device) and config.secondary_restoration != "none":
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
    workspace_output: Path | None = None,
    progress_callback: Callable | None = None,
    segments: "tuple[SegmentRange, ...] | None" = None,
    splice_plan: "SplicePlan | None" = None,
    effect_ranges: "tuple[tuple[int, int], ...] | None" = None,
) -> "Pipeline":
    """A per-video ``Pipeline``; the session first loads any model the segments ask for."""
    from jasna import __version__
    from jasna.pipeline import Pipeline
    from jasna.segments import job_restoration, resolve_restorations

    default = job_restoration(config.restoration_model_name, config.ltx_seed)
    models = frozenset(
        segment.restoration.model for segment in resolve_restorations(tuple(segments or ()), default)
    ) or frozenset({config.restoration_model_name})
    provide_restoration_models(config, session, models, log_callback=None)
    secondary_signature: dict[str, object] = {}
    if config.secondary_restoration == "tvai":
        secondary_signature = {
            "ffmpeg_path": config.tvai_ffmpeg_path,
            "model": config.tvai_model,
            "scale": int(config.tvai_scale),
            "args": config.tvai_args,
            "denoise": bool(config.tvai_denoise),
            "workers": int(config.tvai_workers),
        }
    elif config.secondary_restoration == "unet-4x":
        secondary_signature = {
            "fp16": bool(config.fp16),
        }
    elif config.secondary_restoration == "rtx-super-res":
        secondary_signature = {
            "scale": int(config.rtx_scale),
            "quality": config.rtx_quality,
            "denoise": config.rtx_denoise,
            "deblur": config.rtx_deblur,
        }

    processing_signature = {
        "jasna_version": __version__,
        "device": config.device,
        "fp16": bool(config.fp16),
        "batch_size": int(config.batch_size),
        "detection_model_name": config.detection_model_name,
        "detection_score_threshold": float(config.detection_score_threshold),
        "max_detection_gap": int(config.max_detection_gap),
        "min_detection_duration": int(config.min_detection_duration),
        "scene_detection": bool(config.scene_detection),
        "restoration_model_name": config.restoration_model_name,
        "ltx_model": config.ltx_model,
        "ltx_seed": config.ltx_seed,
        "ltx_large_canvas": config.ltx_large_canvas,
        "ltx_fast": config.ltx_fast,
        "ltx_trial": config.ltx_trial,
        "segment_restorations": [
            {
                "start": segment.start, "end": segment.end,
                "model": segment.restoration.model,
                "ltx_seed": segment.restoration.ltx_seed,
            }
            for segment in resolve_restorations(tuple(segments or ()), default)
        ],
        "max_clip_size": int(config.max_clip_size),
        "temporal_overlap": int(config.temporal_overlap),
        "enable_crossfade": bool(config.enable_crossfade),
        "denoise_strength": config.denoise_strength,
        "denoise_step": config.denoise_step,
        "primary_tensorrt": bool(
            getattr(getattr(session.restoration_pipeline, "restorer", None), "use_tensorrt", False)
        ),
        "secondary_restoration": config.secondary_restoration,
        "secondary_restoration_settings": secondary_signature,
        "vr_mode": config.vr_mode,
        "vr_projection": config.vr_projection,
        "sharpen_strength": float(config.sharpen_strength),
        "retarget_high_fps": bool(config.retarget_high_fps),
        "auto_source_rate": bool(config.auto_source_rate),
        "amd_dual_gop_encode": bool(config.amd_dual_gop_encode),
    }

    return Pipeline(
        config=config,
        session=session,
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
        effect_ranges=effect_ranges,
        workspace_output=workspace_output,
        processing_signature=processing_signature,
    )

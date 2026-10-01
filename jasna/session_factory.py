"""Shared composition root for the video restoration pipeline.

Builds the heavy restoration session (engine compilation, primary and
secondary restorers) and per-video ``Pipeline`` instances from one
``SessionConfig``. Consumed by both the CLI (``jasna.main``) and the GUI
(``jasna.gui.video_session`` / ``jasna.gui.processor``).

All heavy imports (torch, restorers, pipeline) stay inside the functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jasna.session_config import SessionConfig

if TYPE_CHECKING:
    import torch

    from jasna.media.splice import SplicePlan
    from jasna.pipeline import Pipeline
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.secondary_restorer import SecondaryRestorer
    from jasna.segments import SegmentRange


@dataclass
class RestorationSession:
    device: "torch.device"
    detection_model_name: str
    detection_model_path: Path
    restoration_pipeline: "RestorationPipeline"
    secondary_restorer: "SecondaryRestorer | None"

    def close(self) -> None:
        self.restoration_pipeline.restorer.close()
        if self.secondary_restorer is not None and hasattr(self.secondary_restorer, "close"):
            self.secondary_restorer.close()


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
    disable_basicvsrpp_tensorrt: bool,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    device = torch.device(config.device)
    amd = is_amd_device(device)
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    if amd and config.secondary_restoration != "none":
        raise ValueError(
            f"Secondary restoration '{config.secondary_restoration}' is not available in the AMD build yet"
        )

    compile_result = ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=bool(config.fp16),
            basicvsrpp=bool(config.compile_basicvsrpp) and not disable_basicvsrpp_tensorrt and not amd,
            basicvsrpp_model_path=str(config.restoration_model_path),
            detection=True,
            detection_model_name=config.detection_model_name,
            detection_model_path=str(config.detection_model_path),
            detection_batch_size=int(config.batch_size),
            unet4x=(config.secondary_restoration == "unet-4x"),
        ),
        log_callback=log_callback,
    )

    secondary_restorer = _build_secondary_restorer(config, device)
    restoration_pipeline = RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(config.restoration_model_path),
            device=device,
            max_clip_size=int(config.max_clip_size),
            use_tensorrt=compile_result.use_basicvsrpp_tensorrt,
            fp16=bool(config.fp16),
        ),
        secondary_restorer=secondary_restorer,
        denoise_strength=DenoiseStrength(config.denoise_strength),
        denoise_step=DenoiseStep(config.denoise_step),
    )

    return RestorationSession(
        device=device,
        detection_model_name=config.detection_model_name,
        detection_model_path=Path(config.detection_model_path),
        restoration_pipeline=restoration_pipeline,
        secondary_restorer=secondary_restorer,
    )


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
    from jasna import __version__
    from jasna.pipeline import Pipeline

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
        "max_clip_size": int(config.max_clip_size),
        "temporal_overlap": int(config.temporal_overlap),
        "enable_crossfade": bool(config.enable_crossfade),
        "denoise_strength": config.denoise_strength,
        "denoise_step": config.denoise_step,
        "primary_tensorrt": bool(
            getattr(session.restoration_pipeline.restorer, "use_tensorrt", False)
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
        input_video=input_video,
        output_video=output_video,
        detection_model_name=config.detection_model_name,
        detection_model_path=config.detection_model_path,
        detection_score_threshold=float(config.detection_score_threshold),
        restoration_pipeline=session.restoration_pipeline,
        codec=config.codec,
        encoder_settings=dict(config.encoder_settings),
        batch_size=int(config.batch_size),
        device=session.device,
        max_clip_size=int(config.max_clip_size),
        temporal_overlap=int(config.temporal_overlap),
        max_detection_gap=int(config.max_detection_gap),
        min_detection_duration=int(config.min_detection_duration),
        enable_crossfade=bool(config.enable_crossfade),
        scene_detection=bool(config.scene_detection),
        vr_mode=config.vr_mode,
        vr_projection=config.vr_projection,
        fp16=bool(config.fp16),
        disable_progress=bool(config.disable_progress),
        progress_callback=progress_callback,
        lut_path=config.lut_path,
        sharpen_strength=config.sharpen_strength,
        retarget_high_fps=bool(config.retarget_high_fps),
        auto_source_rate=bool(config.auto_source_rate),
        amd_dual_gop_encode=bool(config.amd_dual_gop_encode),
        fmp4=bool(config.fmp4),
        segments=segments,
        splice_plan=splice_plan,
        effect_ranges=effect_ranges,
        working_dir=config.working_dir,
        workspace_output=workspace_output,
        processing_signature=processing_signature,
    )

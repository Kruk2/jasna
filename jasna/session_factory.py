"""Shared composition root for the video restoration pipeline.

Builds the heavy restoration session (engine compilation, primary and
secondary restorers, and a detector cached across videos) and per-video
``Pipeline`` instances from one ``SessionConfig``. Consumed by both the CLI (``jasna.main``) and the GUI
(``jasna.gui.video_session`` / ``jasna.gui.processor``).

All heavy imports (torch, restorers, pipeline) stay inside the functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jasna.session_config import RestorationModelName, SessionConfig, ltx_unsupported_options

if TYPE_CHECKING:
    import torch

    from jasna.ltx.model_files import LtxModelFiles
    from jasna.ltx.super_res import LtxSuperRes
    from jasna.media.probe import VideoMetadata
    from jasna.media.splice import KeyframeIndex, SplicePlan
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    from jasna.pipeline import Pipeline
    from jasna.progressbar import ProgressCallback
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.segments import SegmentRange
    from jasna.upscale import FrameUpscaler, Size

    DetectionModel = RfDetrMosaicDetectionModel | YoloMosaicDetectionModel


logger = logging.getLogger(__name__)


@dataclass
class RestorationSession:
    device: "torch.device"
    restoration_pipeline: "RestorationPipeline | None"
    ltx_files: "LtxModelFiles | None" = None
    ltx_super_res: "LtxSuperRes | None" = None
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
        if self.ltx_super_res is not None:
            self.ltx_super_res.close()
            self.ltx_super_res = None
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


def frame_upscale_size(config: SessionConfig, width: int, height: int) -> "Size | None":
    """The size offline exports of a ``width`` x ``height`` video are upscaled to, or ``None``."""
    from jasna.upscale import upscaled_size

    if config.upscale_model == "none":
        return None
    return upscaled_size(width, height, config.upscale_height)


def ensure_frame_upscale_engines(
    config: SessionConfig,
    source_size: "Size",
    output_size: "Size",
    *,
    log_callback: Callable[[str], None] | None,
) -> None:
    """Compile the TensorRT engines a network upscaler needs for this source size (NVIDIA)."""
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.upscale.network import UPSCALE_NETWORKS, network_input_sizes

    network = UPSCALE_NETWORKS.get(config.upscale_model)
    if network is None:
        return
    ensure_engines_compiled(
        EngineCompilationRequest(
            device=config.device,
            fp16=bool(config.fp16),
            upscale_model=config.upscale_model,
            upscale_input_sizes=list(network_input_sizes(network, source_size, output_size)),
        ),
        log_callback=log_callback,
    )


def build_frame_upscaler(
    config: SessionConfig, device: "torch.device", source_size: "Size", output_size: "Size"
) -> "FrameUpscaler":
    if config.upscale_model == "rtx-super-res":
        from jasna.upscale.rtx import RtxFrameUpscaler

        return RtxFrameUpscaler(
            device,
            source_size=source_size,
            output_size=output_size,
            quality=config.upscale_rtx_quality,
            denoise=config.upscale_rtx_denoise,
            deblur=config.upscale_rtx_deblur,
            strength=config.upscale_strength,
        )
    from jasna.upscale.network import UPSCALE_NETWORKS, NetworkFrameUpscaler

    if config.upscale_model in UPSCALE_NETWORKS:
        return NetworkFrameUpscaler(
            UPSCALE_NETWORKS[config.upscale_model],
            device,
            source_size=source_size,
            output_size=output_size,
            fp16=bool(config.fp16),
            strength=config.upscale_strength,
        )
    raise ValueError(f"Unsupported upscale model: {config.upscale_model}")


def build_restoration_session(
    config: SessionConfig,
    *,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device, log_amd_backend, supports_frame_generation

    device = torch.device(config.device)
    if is_amd_device(device):
        log_amd_backend(device)
    if config.frame_generation != "none" and not supports_frame_generation(device):
        raise ValueError(FRAME_GENERATION_NEEDS_RTX_40)
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    if is_amd_device(device) and config.secondary_restoration not in ("none", "unet-4x"):
        raise ValueError(
            f"Secondary restoration '{config.secondary_restoration}' is not available in the AMD build yet"
        )
    if is_amd_device(device) and config.upscale_model == "rtx-super-res":
        raise ValueError(UPSCALE_RTX_NEEDS_NVIDIA)
    if config.restoration_model_name == "ltx" and ltx_unsupported_options(
        auto_split=config.ltx_auto_split_percent is not None,
        secondary_restoration=config.secondary_restoration,
        denoise_strength=config.denoise_strength,
    ):
        raise ValueError(
            "LTX restoration supports Unet 4x, TVAI or denoise only with LTX auto split, "
            "where they apply to the BasicVSR++ parts"
        )
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
    """Load each of ``models`` the session does not hold yet."""
    if "basicvsrpp" in models and session.restoration_pipeline is None:
        session.restoration_pipeline = _build_basicvsrpp_pipeline(config, session.device, log_callback=log_callback)
    if "ltx" in models and session.ltx_files is None:
        session.ltx_files = _ltx_model_files(config, session.device, log_callback=log_callback)
    if "ltx" in models and config.secondary_restoration == "rtx-super-res" and session.ltx_super_res is None:
        from jasna.ltx.super_res import LtxSuperRes

        session.ltx_super_res = LtxSuperRes(
            device=session.device, quality=config.rtx_quality, denoise=config.rtx_denoise, deblur=config.rtx_deblur
        )


def _build_basicvsrpp_pipeline(
    config: SessionConfig, device: "torch.device", *, log_callback: Callable[[str], None] | None
) -> "RestorationPipeline":
    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    model_path = config.model_path("basicvsrpp")
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
    )


LTX_FAST_NEEDS_BLACKWELL = "The fast LTX model needs an RTX 50-series (Blackwell) GPU"
FRAME_GENERATION_NEEDS_RTX_40 = "Frame generation needs an NVIDIA RTX 40-series or newer GPU"
UPSCALE_RTX_NEEDS_NVIDIA = "RTX Super Res upscaling needs an NVIDIA GPU"


def _ltx_model_files(
    config: SessionConfig, device: "torch.device", *, log_callback: Callable[[str], None] | None
) -> "LtxModelFiles":
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.ltx.model_files import LtxModelFiles

    from jasna.accelerator import TRITON_HIP_ARCHITECTURES, amd_gpu_architecture, is_amd_device, supports_nvfp4
    from jasna.ltx.transformer import require_kernels

    if config.ltx_fast and not supports_nvfp4(device):
        raise ValueError(LTX_FAST_NEEDS_BLACKWELL)
    if is_amd_device(device) and (architecture := amd_gpu_architecture(device)) not in TRITON_HIP_ARCHITECTURES:
        raise ValueError(
            f"LTX can't run on this AMD GPU ({architecture}): it needs a Radeon RX 7000 or 9000 series GPU. "
            "Use BasicVSR++ instead."
        )
    require_kernels(device)
    if config.ltx_trial:
        from jasna.ltx.model_files import LTX_TRIAL_NOTICE

        logger.warning(LTX_TRIAL_NOTICE)
        files = LtxModelFiles.placeholder(
            config.ltx_model, fast=config.ltx_fast, encoder=config.ltx_vae_encoder, decoder=config.ltx_vae_decoder
        )
    else:
        files = LtxModelFiles.from_dir(
            config.model_path("ltx"),
            config.ltx_model,
            fast=config.ltx_fast,
            encoder=config.ltx_vae_encoder,
            decoder=config.ltx_vae_decoder,
        )
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


def segment_job_config(config: SessionConfig, source_codec: str, input_video: Path) -> SessionConfig:
    """A segment job copies the parts it leaves alone, so its output keeps the source codec.
    The CQ and custom encoder settings were chosen for ``config.codec`` (``tune=uhq`` of
    av1_nvenc fails on h264_nvenc), so another codec gets only its own default CQ."""
    if source_codec == config.codec:
        return config
    from jasna.accelerator import vendor_for_device
    from jasna.media.encoder_settings import encoder_cq_spec

    cq = encoder_cq_spec(source_codec, vendor_for_device(config.device)).default
    logger.warning(
        "%s: the output keeps the source codec %s so the parts without changes can be copied; "
        "the CQ and custom encoder settings are for %s and are not used (%s CQ %d)",
        input_video.name,
        source_codec.upper(),
        config.codec.upper(),
        source_codec.upper(),
        cq,
    )
    return replace(config, codec=source_codec, encoder_settings={"cq": cq})


@dataclass(frozen=True)
class SpliceSource:
    """A video that can be restored in parts: its metadata, codec and keyframes."""

    metadata: "VideoMetadata"
    codec: str
    index: "KeyframeIndex"


def splice_source(config: SessionConfig, source: Path, output: Path) -> SpliceSource:
    """Raises ``SmartRenderCompatibilityError`` when ``source`` cannot be restored in parts
    into ``output``."""
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import (
        SmartRenderCompatibilityError,
        canonical_codec,
        probe_keyframes,
        validate_smart_render,
    )

    if config.frame_generation != "none":
        # Copied parts keep the source frame rate, so they cannot sit next to doubled ones.
        raise SmartRenderCompatibilityError("Smart rendering cannot be combined with frame generation")
    if config.upscale_model != "none":
        # Copied parts keep the source size, so they cannot sit next to upscaled ones.
        raise SmartRenderCompatibilityError("Smart rendering cannot be combined with upscaling")
    metadata = get_video_meta_data(str(source))
    codec = canonical_codec(metadata.codec_name)
    validate_smart_render(metadata, output_path=output, codec=codec, retarget_high_fps=config.retarget_high_fps)
    return SpliceSource(metadata, codec, probe_keyframes(source, metadata))


@dataclass(frozen=True)
class SegmentJob:
    """A job that restores ranges and copies the rest: its config in the source codec and
    the plan of its ranges."""

    config: SessionConfig
    splice_plan: "SplicePlan"

    @property
    def reencodes_whole_video(self) -> bool:
        """Past the encoder's B-frame cap the ranges cannot be cut out, so the pipeline
        re-encodes the whole video with BasicVSR++ and keeps no finished parts."""
        from jasna.accelerator import vendor_for_device
        from jasna.media.splice import h264_b_frames_over_cap

        return h264_b_frames_over_cap(self.config.codec, self.splice_plan.index, vendor_for_device(self.config.device))


def _require_ltx_cuts(config: SessionConfig, spliced: SpliceSource) -> None:
    """LTX restores ranges only as cut-out parts, which the encoder cannot match past its B-frame cap."""
    from jasna.accelerator import vendor_for_device
    from jasna.media.splice import SmartRenderCompatibilityError, h264_b_frames_over_cap

    if h264_b_frames_over_cap(spliced.codec, spliced.index, vendor_for_device(config.device)):
        raise SmartRenderCompatibilityError(
            f"This H.264 video uses {spliced.index.max_b_frames} consecutive B-frames, more than the GPU encoder "
            "can match, so LTX ranges cannot be cut out of it; restore the ranges with BasicVSR++, "
            "or clear them to restore the whole video with LTX"
        )


def plan_segment_job(
    config: SessionConfig, source: Path, spliced: SpliceSource, segments: "tuple[SegmentRange, ...]"
) -> SegmentJob:
    from jasna.media.splice import build_splice_plan
    from jasna.segments import job_models

    if "ltx" in job_models(config.restoration_model_name, segments):
        _require_ltx_cuts(config, spliced)
    plan = build_splice_plan(segments, spliced.index, duration=spliced.metadata.duration)
    if config.lut_path is not None or config.sharpen_strength > 0:
        logger.info("%s: a color LUT or sharpening is set, so the parts without ranges are encoded again", source.name)
        plan = plan.reencoding_copies()
    return SegmentJob(segment_job_config(config, spliced.codec, source), plan)


def ltx_auto_split(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    progress_callback: "ProgressCallback | None",
) -> SegmentJob | None:
    """For an LTX job with ``ltx_auto_split_percent`` and no ranges of its own: scan the
    video, give big mosaics LTX and the rest BasicVSR++, copy the parts without mosaics.
    None when the job does not split, including a video segment processing cannot handle
    or with no mosaics found; those are logged and restored whole with LTX."""
    from jasna.mosaic_scan import ltx_auto_segments, scan_mosaic_sizes
    from jasna.progressbar import StageProgress
    from jasna.vr180 import resolve_vr_mode

    def on_progress(fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(0.0, 0.0, 0.0, 0, 0, StageProgress("split", 1, 1, fraction, 0.0))

    def scan(spliced: SpliceSource) -> "tuple[SegmentRange, ...]":
        stereo = resolve_vr_mode(config.vr_mode, spliced.metadata, input_video).layout(spliced.metadata)
        samples = scan_mosaic_sizes(
            session.detection_model_for(config),
            input_video,
            spliced.metadata,
            stereo=stereo,
            batch_size=config.batch_size,
            device=session.device,
            decode_backend=config.decode_backend,
            disable_progress=config.disable_progress,
            on_progress=on_progress,
        )
        return ltx_auto_segments(
            samples.times,
            samples.scores,
            samples.sizes,
            threshold=config.detection_score_threshold,
            ltx_min_size=config.ltx_auto_split_percent / 100,
            stride=samples.stride,
            duration=spliced.metadata.duration,
            keyframes=spliced.index.keyframe_seconds(),
            ltx_seed=config.ltx_seed,
        )

    return _auto_split_job(config, input_video, output_video, scan)


def _auto_split_job(
    config: SessionConfig,
    input_video: Path,
    output_video: Path,
    scan: "Callable[[SpliceSource], tuple[SegmentRange, ...]] | None",
) -> SegmentJob | None:
    """The auto split job from the saved split, else from ``scan``; with ``scan`` None only
    a saved split counts."""
    from jasna.job_resume import ResumeJob
    from jasna.media.splice import SmartRenderCompatibilityError

    if config.ltx_auto_split_percent is None or config.restoration_model_name != "ltx":
        return None
    try:
        spliced = splice_source(config, input_video, output_video)
        _require_ltx_cuts(config, spliced)
    except SmartRenderCompatibilityError as exc:
        logger.warning("%s: LTX auto split skipped (%s); restoring the whole video with LTX", input_video.name, exc)
        return None
    checkpoint = ResumeJob(config, input_video, output_video, spliced.metadata, None)
    segments = checkpoint.saved_split()
    if segments is not None:
        logger.info("Reusing saved LTX auto split for %s", input_video.name)
    elif scan is None:
        return None
    else:
        segments = scan(spliced)
        checkpoint.save_split(segments)
    if not segments:
        logger.warning("%s: LTX auto split found no mosaics; restoring the whole video with LTX", input_video.name)
        return None
    try:
        job = plan_segment_job(config, input_video, spliced, segments)
    except SmartRenderCompatibilityError as exc:
        logger.warning("%s: LTX auto split skipped (%s); restoring the whole video with LTX", input_video.name, exc)
        return None
    mosaic_seconds = sum(segment.duration for segment in segments)
    ltx_seconds = sum(segment.duration for segment in segments if segment.restoration.model == "ltx")
    logger.info(
        "%s: LTX auto split: %.0f s with mosaics, %.0f s (%.0f%%) of it with LTX",
        input_video.name,
        mosaic_seconds,
        ltx_seconds,
        ltx_seconds / mosaic_seconds * 100,
    )
    return job


def find_resume_output(
    config: SessionConfig, source: Path, output: Path, segments: "tuple[SegmentRange, ...]"
) -> Path | None:
    """The output whose saved work continues this job, so it wins over GUI file conflicts
    (idea from #415), including an earlier auto-rename. The job is derived as a run derives
    it; ranges that cannot be restored in parts raise as the run would."""
    from jasna.job_resume import ResumeJob, ResumeMismatchError, saved_outputs
    from jasna.media.probe import get_video_meta_data

    for saved in saved_outputs(config, output):
        try:
            job = (
                plan_segment_job(config, source, splice_source(config, source, saved), segments) if segments
                else _auto_split_job(config, source, saved, None)
            )
        except ResumeMismatchError:
            continue
        metadata = get_video_meta_data(str(source))
        checkpoint = (
            ResumeJob(config, source, saved, metadata, None) if job is None
            else ResumeJob(job.config, source, saved, metadata, job.splice_plan)
        )
        if checkpoint.matches():
            return saved
    return None


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
    from jasna.segments import job_models

    provide_restoration_models(
        config, session, job_models(config.restoration_model_name, segments), log_callback=None
    )
    return Pipeline(
        config=config,
        session=session,
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
    )


def build_job_pipeline(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    segment_job: SegmentJob | None,
    *,
    progress_callback: Callable | None,
) -> "Pipeline":
    """The pipeline of a job: ``segment_job`` when it restores ranges, else the whole video."""
    if segment_job is None:
        return build_pipeline(config, session, input_video, output_video, progress_callback=progress_callback)
    return build_pipeline(
        segment_job.config,
        session,
        input_video,
        output_video,
        progress_callback=progress_callback,
        segments=segment_job.splice_plan.segments,
        splice_plan=segment_job.splice_plan,
    )

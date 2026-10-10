from __future__ import annotations

import bisect
import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from fractions import Fraction

import psutil
import torch

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.media.container_utils import MOV_SUFFIXES
from jasna.media.probe import UnsupportedColorspaceError, get_video_meta_data
from jasna.media.video_encoder import VideoEncoder
from jasna.media.video_encoder import resolve_hevc_smart_render_vui
from jasna.media.windows_d3d11_hip_resident import WindowsD3D11HipResidentCoordinator, windows_d3d11_hip_resident_requested
from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.native_worker import (
    HostMemoryPressureError,
    NativeWorkerRecycleRequested,
    amf_encoder_stall_timeout_seconds,
    amf_render_session_seconds,
    is_isolated_video_job,
)
from jasna.media.splice import (
    KeyframeIndex,
    SmartRenderCompatibilityError,
    SplicePlan,
    SpliceSpan,
    build_splice_plan,
    create_copy_fragment,
    create_normalized_copy_fragment,
    hevc_copy_fragment_timeline_matches_source,
    mux_fragments_final_output,
    normalize_fragment,
    probe_keyframes,
    require_one_model,
    resolve_smart_encoder_settings,
    split_render_spans,
    validate_hevc_fragment_parameter_sets,
    validate_smart_render,
)
from jasna.pipeline_threads import run_restoration_pass
from jasna.progressbar import LTX_WORK_PER_FRAME, JobProgress, ProgressCallback, Progressbar
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.segments import SegmentRange, job_restoration, resolve_restorations
from jasna.session_config import SessionConfig
from jasna.smart_render_workspace import SmartRenderWorkspace, workspace_signature
from jasna.vram_offloader import VramStats
from jasna.vr180 import (
    SbsDetectionAdapter,
    resolve_vr_mode,
)
from jasna.vr_projection import build_vr_projector

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter, LtxRender, LtxSpan
    from jasna.session_factory import RestorationSession

log = logging.getLogger(__name__)


def _span_frames(span: SpliceSpan, index: KeyframeIndex, metadata) -> int:
    return round((span.end_pts - span.start_pts) * index.time_base * metadata.video_fps)


def _dual_gop_smart_encoder_settings(
    settings: dict[str, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    """Keep the proven dual-session GOP contract after Smart Render matching."""

    if not enabled:
        return settings
    from jasna.media.dual_gop_encoder import AMD_DUAL_GOP_SIZE

    resolved = dict(settings)
    resolved.update({"g": AMD_DUAL_GOP_SIZE, "bf": 0})
    return resolved


def _bounded_amf_render_session_seconds(
    *,
    metadata,
    codec: str,
    vendor: AcceleratorVendor,
    batch_size: int,
    dual_gop_enabled: bool,
    retarget_high_fps: bool = False,
) -> float | None:
    """Bound affected Linux AMF render spans before native state can stall."""

    if not is_isolated_video_job():
        return None
    if vendor is not AcceleratorVendor.AMD:
        return None
    normalized_codec = str(codec).casefold()
    width = int(getattr(metadata, "video_width", 0))
    height = int(getattr(metadata, "video_height", 0))
    ten_bit = bool(getattr(metadata, "is_10bit", False))
    if not (
        normalized_codec == "hevc"
        and not bool(retarget_high_fps)
        # Main10 keeps its established corruption boundary.  Main8 needs the
        # same boundary only when two persistent AMF encoders are requested:
        # otherwise a long render span can carry the native working set into
        # later decoder/encoder groups and exhaust whole-card headroom.
        and (ten_bit or bool(dual_gop_enabled))
        and width == 8192
        and height == 4096
        and int(batch_size) == 4
    ):
        return None
    return amf_render_session_seconds()


class _OfflineFrameWriter:
    def __init__(
        self,
        encoder_ctx: VideoEncoder,
        encode_heartbeat: list[float | None],
        *,
        amd_dual_gop_encode: bool = False,
    ):
        self._encoder_ctx = encoder_ctx
        self._encode_heartbeat = encode_heartbeat
        self._entered = False
        self._dual_gop = None

        from jasna.media.dual_gop_encoder import (
            AmdDualGopFrameWriter,
            use_dual_gop_writer,
        )

        if use_dual_gop_writer(
            encoder_ctx,
            enabled=bool(amd_dual_gop_encode),
        ):
            self._dual_gop = AmdDualGopFrameWriter(encoder_ctx)

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        # Arm the stall watchdog only when the encoder receives its first
        # write.  Updating before the call also preserves diagnostics if the
        # very first native encode submission itself blocks.
        self._encode_heartbeat[0] = time.monotonic()
        if self._dual_gop is not None:
            self._dual_gop.write(frame, pts, apply_lut=apply_lut)
            self._encode_heartbeat[0] = time.monotonic()
            return
        if not self._entered:
            self._encoder_ctx.__enter__()
            self._entered = True
        self._encoder_ctx.encode(frame, pts, apply_lut=apply_lut)
        self._encode_heartbeat[0] = time.monotonic()

    def after_write(self, frames_written: int) -> None:
        pass

    def close(self, *, abort: bool = False) -> None:
        if self._dual_gop is not None:
            if abort:
                self._dual_gop.abort()
            else:
                self._dual_gop.close()
            self._dual_gop = None
            return
        if self._entered:
            self._encoder_ctx.__exit__(None, None, None)
            self._entered = False


class Pipeline:
    def __init__(
        self,
        *,
        config: SessionConfig,
        session: RestorationSession,
        input_video: Path,
        output_video: Path,
        progress_callback: ProgressCallback | None,
        segments: tuple[SegmentRange, ...] | None,
        splice_plan: SplicePlan | None,
        effect_ranges: tuple[tuple[int, int], ...] | None = None,
        workspace_output: Path | None = None,
        processing_signature: dict[str, object] | None = None,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.working_dir = config.working_dir
        self.workspace_output = workspace_output
        self.effect_ranges = effect_ranges
        self.processing_signature = dict(processing_signature or {})
        self.detection_model_path = config.detection_model_path
        self.auto_source_rate = config.auto_source_rate
        self.amd_dual_gop_encode = config.amd_dual_gop_encode
        self.codec = config.codec
        self.encoder_settings = dict(config.encoder_settings)
        self.batch_size = config.batch_size
        self.device = session.device
        self.max_clip_size = config.max_clip_size
        self.temporal_overlap = config.temporal_overlap
        self.max_detection_gap = config.max_detection_gap
        self.min_detection_duration = config.min_detection_duration
        self.enable_crossfade = config.enable_crossfade
        self.scene_detection = config.scene_detection
        self.vr_mode = config.vr_mode
        self.vr_projection = config.vr_projection
        self.detection_model = session.detection_model_for(config)
        self.restoration_pipeline = session.restoration_pipeline
        self.ltx_files = session.ltx_files
        self.restoration_model_name = config.restoration_model_name
        self.ltx_large_canvas = config.ltx_large_canvas
        self.ltx_seed = config.ltx_seed
        self.disable_progress = config.disable_progress
        self.progress_callback = progress_callback
        self.lut_path = config.lut_path
        self.sharpen_strength = config.sharpen_strength
        self.retarget_high_fps = config.retarget_high_fps
        self.fmp4 = config.fmp4
        self.segments = tuple(segments) if segments else None
        self.splice_plan = splice_plan
        self.vr_resolution = None
        self.vr_projector = None
        self.job_detection_model = self.detection_model
        self._cancel_event = threading.Event()
        self.completed = False

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Ask the running pipeline to stop as soon as the worker threads notice."""
        self._cancel_event.set()

    def configure_vr(self, metadata) -> None:
        self.vr_resolution = resolve_vr_mode(
            self.vr_mode,
            metadata,
            self.input_video,
            projection=self.vr_projection,
        )
        self.job_detection_model = (
            SbsDetectionAdapter(self.detection_model)
            if self.vr_resolution.is_sbs
            else self.detection_model
        )
        self.vr_projector = (
            build_vr_projector(
                self.vr_resolution.projection,
                eye_width=int(metadata.video_width) // 2,
                height=int(metadata.video_height),
                device=self.device,
            )
            if self.vr_resolution.is_sbs
            else None
        )

    def close(self) -> None:
        """Release per-video references; the shared session owns its cached models."""
        self.job_detection_model = None
        self.detection_model = None
        self.restoration_pipeline = None
        self.vr_projector = None

    def _run_pass(
        self,
        *,
        metadata,
        encoder_ctx: VideoEncoder,
        progress: Progressbar,
        seek_ts: float | None = None,
        end_pts: int | None = None,
        effect_ranges: tuple[tuple[int, int], ...] | None = None,
        output_frame_count: int | None = None,
        recycle_on_host_memory_pressure: bool = False,
        resident_coordinator: WindowsD3D11HipResidentCoordinator | None = None,
    ) -> VramStats:
        frame_rate = resolve_frame_rate_retarget(
            metadata.video_fps_exact,
            enabled=self.retarget_high_fps,
            measured_fps=metadata.average_fps,
        )
        if output_frame_count is None:
            output_frame_count = frame_rate.output_frame_count(metadata.num_frames)

        encode_heartbeat: list[float | None] = [None]
        frame_writer = _OfflineFrameWriter(encoder_ctx, encode_heartbeat, amd_dual_gop_encode=self.amd_dual_gop_encode)
        error = None
        try:
            error = run_restoration_pass(
                self,
                metadata,
                frame_writer,
                self._cancel_event,
                seek_ts=seek_ts,
                use_async_secondary=isinstance(
                    self.restoration_pipeline.secondary_restorer, AsyncSecondaryRestorer
                ),
                end_pts=end_pts,
                effect_ranges=effect_ranges,
                frame_stride=frame_rate.frame_stride,
                output_frame_count=output_frame_count,
                output_fps=float(frame_rate.output_fps),
                progress=progress,
                encode_heartbeat=encode_heartbeat,
                resident_coordinator=resident_coordinator,
                recycle_on_host_memory_pressure=recycle_on_host_memory_pressure,
            )
        except BaseException as pass_error:
            error = pass_error
        finally:
            try:
                frame_writer.close(abort=error is not None or self._cancel_event.is_set())
            except BaseException as close_error:
                if error is None:
                    error = close_error
                else:
                    log.exception("Encoder cleanup failed after a pipeline failure")

        if error is not None:
            raise error

        free, total = torch.cuda.mem_get_info(self.device)
        log.info("VRAM usage at end — %.1f MiB", (total - free) / (1024 ** 2))
        log.info("RAM usage at end — %.1f MiB", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
        return getattr(self, "_last_pass_vram_stats", VramStats())

    def validate_metadata(self, metadata) -> None:
        from av.video.reformatter import Colorspace as AvColorspace

        if metadata.color_space not in (
            AvColorspace.ITU709,
            AvColorspace.ITU601,
            AvColorspace.BT2020,
        ):
            raise UnsupportedColorspaceError(
                f"Unsupported color space: {metadata.color_space!r} in {self.input_video.name}. "
                "Only BT.709, BT.601, and BT.2020 non-constant-luminance are supported."
            )

    def _resolve_frame_rate(self, metadata):
        frame_rate = resolve_frame_rate_retarget(
            metadata.video_fps_exact,
            enabled=self.retarget_high_fps,
            measured_fps=metadata.average_fps,
        )
        if frame_rate.active:
            log.info(
                "Retargeting frame rate: %s fps -> %s fps (keeping every %dth frame)",
                frame_rate.source_fps,
                frame_rate.output_fps,
                frame_rate.frame_stride,
            )
        elif frame_rate.rate_mismatch:
            log.warning(
                "Frame-rate retargeting skipped: the container reports %s fps but the measured "
                "frame rate is %.3f fps; keeping the source rate",
                frame_rate.source_fps,
                metadata.average_fps,
            )
        elif self.retarget_high_fps:
            log.info(
                "Frame-rate retargeting requested, but %s fps is not a supported source rate; keeping source rate",
                frame_rate.source_fps,
            )
        return frame_rate

    def _video_encoder(self, metadata, frame_rate) -> VideoEncoder:
        if self.fmp4 and self.output_video.suffix.lower() not in MOV_SUFFIXES:
            log.info(
                "Fragmented MP4 has no effect on %s output; it is already playable while it grows",
                self.output_video.suffix,
            )
        return VideoEncoder(
            str(self.output_video),
            device=self.device,
            metadata=metadata,
            codec=self.codec,
            encoder_settings=self.encoder_settings,
            lut_path=self.lut_path,
            sharpen_strength=self.sharpen_strength,
            output_fps=frame_rate.output_fps,
            match_input_bit_depth=True, auto_source_rate=self.auto_source_rate,
            prefer_amf_host_native=self.amd_dual_gop_encode,
            fmp4=self.fmp4,
        )

    def _run_full(
        self, metadata, *, effect_ranges: tuple[tuple[int, int], ...] | None = None
    ) -> None:
        frame_rate = self._resolve_frame_rate(metadata)
        output_frame_count = frame_rate.output_frame_count(metadata.num_frames)
        progress = Progressbar(
            total_frames=output_frame_count,
            video_fps=float(frame_rate.output_fps),
            disable=self.disable_progress,
            callback=self.progress_callback,
        )
        if effect_ranges is None:
            effect_ranges = self.effect_ranges
        bounded_seconds = _bounded_amf_render_session_seconds(
            metadata=metadata, codec=self.codec, vendor=vendor_for_device(self.device),
            batch_size=self.batch_size, dual_gop_enabled=self.amd_dual_gop_encode,
            retarget_high_fps=self.retarget_high_fps,
        )
        if bounded_seconds is not None:
            try:
                self._run_bounded_full(metadata, frame_rate=frame_rate, progress=progress,
                    output_frame_count=output_frame_count, max_duration_seconds=bounded_seconds,
                    effect_ranges=effect_ranges)
            finally:
                progress.close(ensure_completed_bar=True)
            return
        resident_coordinator = None
        try:
            if windows_d3d11_hip_resident_requested():
                if self.amd_dual_gop_encode:
                    raise RuntimeError(
                        "The explicit Windows D3D11/HIP resident route cannot be "
                        "combined with dual-GOP encoding"
                    )
                resident_coordinator = WindowsD3D11HipResidentCoordinator(
                    device=self.device,
                    metadata=metadata,
                    batch_size=self.batch_size,
                    output_codec=self.codec,
                )
            encoder_ctx = VideoEncoder(
                str(self.output_video),
                device=self.device,
                metadata=metadata,
                codec=self.codec,
                encoder_settings=self.encoder_settings,
                lut_path=self.lut_path,
                sharpen_strength=self.sharpen_strength,
                output_fps=frame_rate.output_fps,
                match_input_bit_depth=True,
                auto_source_rate=getattr(self, "auto_source_rate", False),
                prefer_amf_host_native=bool(
                    getattr(self, "amd_dual_gop_encode", False)
                ),
                fmp4=self.fmp4,
                resident_coordinator=resident_coordinator,
            )
            self._run_pass(
                metadata=metadata,
                encoder_ctx=encoder_ctx,
                progress=progress,
                effect_ranges=effect_ranges,
                output_frame_count=output_frame_count,
                resident_coordinator=resident_coordinator,
            )
        finally:
            try:
                if resident_coordinator is not None:
                    resident_coordinator.close()
            finally:
                progress.close(ensure_completed_bar=True)

    def _run_bounded_full(
        self,
        metadata,
        *,
        frame_rate,
        progress: Progressbar,
        output_frame_count: int,
        max_duration_seconds: float,
        effect_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Render a full video in short AMF epochs and assemble the fragments.

        Linux AMF/Vulkan resources are process-global and can retain native
        surfaces beyond Python object lifetime.  A single full-video pass can
        therefore grow the host/native working set for hours even though each
        queue is bounded.  Closed-GOP render fragments put a hard lifetime
        boundary around every decoder and dual-encoder session while keeping
        the final output frame/PTS contract unchanged.
        """

        index = probe_keyframes(self.input_video, metadata)
        plan = split_render_spans(
            SplicePlan(
                index=index,
                spans=(
                    SpliceSpan(
                        "render",
                        index.start_pts,
                        index.end_pts,
                        tuple(effect_ranges or ()),
                    ),
                ),
                segments=(),
            ),
            video_fps=frame_rate.output_fps,
            max_duration_seconds=max_duration_seconds,
        )
        self._validate_bounded_full_plan(plan)
        log.info(
            "Bounded Linux AMD full render sessions: %.1fs maximum, "
            "%d render fragments",
            max_duration_seconds,
            len(plan.render_spans),
        )
        work_root = self.working_dir or self.output_video.parent
        work_root.mkdir(parents=True, exist_ok=True)
        # A bounded full render is resumable.  The isolated worker is
        # recycled after each completed fragment so process-global AMF/Vulkan
        # allocations cannot accumulate across the source video; the next
        # child reopens this signature-bound workspace and skips its completed
        # prefix instead of rendering it again.
        resolved_projection = (
            self.vr_resolution.projection
            if getattr(self, "vr_resolution", None) is not None
            else getattr(self, "vr_projection", "auto")
        )
        workspace_output = getattr(self, "workspace_output", None) or self.output_video
        signature = workspace_signature(
            source=self.input_video,
            output=workspace_output,
            plan=plan,
            processing={
                **dict(getattr(self, "processing_signature", {}) or {}),
                "route": "bounded-full",
                "amf_session_seconds": float(max_duration_seconds),
                # Bump the durable workspace identity along with the decode
                # sentinel fix below.  Fragments produced by the old bug are
                # structurally valid but contain no restoration, so they must
                # never be reused after upgrading this code.
                "bounded_full_effect_ranges_semantics": "all-frames-none-v1",
            },
            model_files={
                "detection": getattr(self, "detection_model_path", None),
                "restoration": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "restorer",
                        None,
                    ),
                    "checkpoint_path",
                    None,
                ),
                "secondary": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "secondary_restorer",
                        None,
                    ),
                    "engine_path",
                    None,
                ),
                "lut": self.lut_path,
            },
            codec=self.codec,
            encoder_settings=self.encoder_settings,
            resolved_projection=resolved_projection,
        )
        workspace = SmartRenderWorkspace.open(
            work_root,
            output=workspace_output,
            signature=signature,
        )
        fragments: list[tuple[Path, float]] = []
        succeeded = False
        session_recycled = False
        try:
            fragment_suffix = ".ts" if self.codec in {"h264", "hevc"} else ".mkv"
            for span_index, span in enumerate(plan.render_spans):
                if self._cancel_event.is_set():
                    return
                duration = float((span.end_pts - span.start_pts) * index.time_base)
                expected_frames = max(
                    1,
                    round(
                        Fraction(span.end_pts - span.start_pts)
                        * Fraction(frame_rate.output_fps)
                        * index.time_base
                    ),
                )
                reusable = workspace.reusable_fragment(span_index)
                if reusable is not None:
                    log.info(
                        "Reusing bounded full-render fragment %s from %s",
                        span_index,
                        reusable,
                    )
                    fragments.append((reusable, duration))
                    progress.mark_completed(expected_frames)
                    continue

                workspace.mark_running(span_index)
                raw = workspace.raw_path(span_index)
                normalized = workspace.fragment_path(span_index, fragment_suffix)
                raw.unlink(missing_ok=True)
                normalized.unlink(missing_ok=True)
                encoder_ctx = VideoEncoder(
                    str(raw),
                    device=self.device,
                    metadata=metadata,
                    codec=self.codec,
                    encoder_settings=self.encoder_settings,
                    lut_path=self.lut_path,
                    sharpen_strength=self.sharpen_strength,
                    output_fps=frame_rate.output_fps,
                    mux_audio=False,
                    pts_origin=span.start_pts,
                    match_input_bit_depth=True,
                    smart_fragment=True,
                    auto_source_rate=getattr(self, "auto_source_rate", False),
                    prefer_amf_host_native=bool(
                        getattr(self, "amd_dual_gop_encode", False)
                    ),
                )
                self._run_pass(
                    metadata=metadata,
                    encoder_ctx=encoder_ctx,
                    progress=progress,
                    seek_ts=index.seconds_for_pts(span.start_pts),
                    end_pts=span.end_pts,
                    # A bounded-full render is still a full-video pass.  The
                    # empty tuple stored on its SpliceSpan is only plan
                    # metadata; passing it to decode_detect_loop would mean
                    # "no selected frames" and silently turn every fragment
                    # into a no-op.  None is the sentinel for all frames;
                    # preserve an explicitly supplied non-empty range.
                    effect_ranges=span.effect_ranges or None,
                    output_frame_count=expected_frames,
                    recycle_on_host_memory_pressure=True,
                )
                if self._cancel_event.is_set():
                    return
                normalize_fragment(
                    raw,
                    normalized,
                    codec=self.codec,
                    # Full bounded renders contain no copied source packets;
                    # each fragment starts at an encoder PTS origin of zero.
                    # Applying the source decoder's B-frame delay here would
                    # create an artificial DTS offset (and negative DTS) on
                    # the independently encoded, closed-GOP fragment.
                    decode_delay=Fraction(0, 1),
                )
                workspace.mark_complete(span_index, normalized)
                raw.unlink(missing_ok=True)
                fragments.append((normalized, duration))

                # AMF/Vulkan allocations are process-global on the affected
                # Linux AMD runtime.  Reopening an encoder in this process is
                # not a hard lifetime boundary: the previous run reached
                # 24.48 GiB whole-card usage after only three fragments.  End
                # the isolated worker after every completed fragment and let
                # the parent resume from the workspace in a fresh process.
                if (
                    is_isolated_video_job()
                    and span_index + 1 < len(plan.render_spans)
                ):
                    message = (
                        "Linux AMD 8K HEVC bounded full fragment "
                        f"{span_index} completed; recycling the isolated worker "
                        "before the next AMF decoder/encoder session"
                    )
                    log.debug(message)
                    raise NativeWorkerRecycleRequested(
                        message,
                        reason="amf_session_limit",
                    )

            if self._cancel_event.is_set():
                return
            if self.codec == "hevc":
                validate_hevc_fragment_parameter_sets(
                    [(fragment, "render") for fragment, _duration in fragments]
                )
            mux_fragments_final_output(
                fragments,
                self.input_video,
                self.output_video,
                manifest=workspace.path / "fragments.ffconcat",
                codec=self.codec,
            )
            succeeded = True
        except NativeWorkerRecycleRequested as request:
            session_recycled = request.reason == "amf_session_limit"
            raise
        finally:
            if succeeded:
                workspace.cleanup()
            elif session_recycled:
                log.debug(
                    "Retaining resumable workspace for normal AMF session recycle: %s",
                    workspace.path,
                )
            else:
                log.error(
                    "Preserving failed/resumable bounded full-render workspace: %s",
                    workspace.path,
                )

    @staticmethod
    def _validate_bounded_full_plan(plan: SplicePlan) -> None:
        """Reject fragment plans that could silently drop or duplicate PTS.

        The bounded full route is intentionally all-render: unlike Smart
        Render it has no copied spans that can bridge a gap.  Keep an explicit
        contiguous PTS check next to the splitter so a future change to
        keyframe probing or duration rounding cannot create a valid-looking
        output with a missing boundary frame.
        """

        spans = tuple(plan.render_spans)
        if not spans:
            raise RuntimeError("bounded full render produced no fragments")
        if len(spans) != len(plan.spans):
            raise RuntimeError(
                "bounded full render plan contains a non-render span"
            )
        cursor = int(plan.index.start_pts)
        for position, span in enumerate(spans):
            if not span.is_render:
                raise RuntimeError(
                    f"bounded full fragment {position} is not a render span"
                )
            if int(span.start_pts) != cursor:
                raise RuntimeError(
                    "bounded full render has a PTS gap or overlap before "
                    f"fragment {position}: expected {cursor}, got {span.start_pts}"
                )
            if int(span.end_pts) <= int(span.start_pts):
                raise RuntimeError(
                    f"bounded full fragment {position} has non-positive PTS range"
                )
            cursor = int(span.end_pts)
        if cursor != int(plan.index.end_pts):
            raise RuntimeError(
                "bounded full render does not cover the probed source PTS range: "
                f"ended at {cursor}, expected {plan.index.end_pts}"
            )

    def _ltx_progress(self, frames: int, callback: ProgressCallback | None):
        from jasna.ltx.restore import Progress

        report = None
        if callback is not None:
            report = lambda stage, fraction, eta: callback(fraction * 100.0, 0.0, eta, 0, 0, stage)
        return Progress(frames, disable=self.disable_progress, report=report)

    def _require_ltx_video(self) -> None:
        if self.vr_resolution.is_sbs:
            raise ValueError("LTX restoration does not support VR180 side-by-side video")

    def _run_ltx(self, metadata) -> None:
        from jasna.ltx.restore import Cancelled, restore_video, video_frames

        self._require_ltx_video()
        frame_rate = self._resolve_frame_rate(metadata)
        writer = _OfflineFrameWriter(self._video_encoder(metadata, frame_rate), [time.monotonic()])
        try:
            restore_video(
                video_frames(
                    self.input_video,
                    metadata,
                    batch_size=self.batch_size,
                    device=self.device,
                    frame_stride=frame_rate.frame_stride,
                    seek_ts=None,
                    end_pts=None,
                ),
                writer,
                detector=self.job_detection_model,
                files=self.ltx_files,
                frame_h=int(metadata.video_height),
                frame_w=int(metadata.video_width),
                batch_size=self.batch_size,
                large_canvas=self.ltx_large_canvas,
                seed=self.ltx_seed,
                device=self.device,
                work_dir=self.working_dir or self.output_video.parent,
                progress=self._ltx_progress(
                    frame_rate.output_frame_count(metadata.num_frames), self.progress_callback
                ),
                cancel=self._cancel_event,
            )
        except Cancelled:
            log.info("LTX restoration cancelled")
        finally:
            writer.close()

    def ltx_segment_render(self, metadata) -> "LtxRender":
        """How LTX restores segments: the 768 px decision and the decode tiles follow the
        GPU's size, never the VRAM free at that moment, so a segment renders the same in a
        job and in a seed preview."""
        from jasna.ltx.restore import LtxRender, segment_decode_budget, segment_large_canvas

        self._require_ltx_video()
        return LtxRender(
            detector=self.job_detection_model,
            files=self.ltx_files,
            frame_h=int(metadata.video_height),
            frame_w=int(metadata.video_width),
            batch_size=self.batch_size,
            large_canvas=segment_large_canvas(
                self.ltx_large_canvas, torch.cuda.get_device_properties(self.device).total_memory
            ),
            budget=segment_decode_budget,
            device=self.device,
        )

    def ltx_span(
        self,
        metadata,
        index: KeyframeIndex,
        span: SpliceSpan,
        segments: tuple[SegmentRange, ...],
        open_writer: Callable[[], "FrameWriter"],
    ) -> "LtxSpan":
        """The LTX work of one render span: decoded from its keyframe, one LTX segment per
        effect range with that range's seed."""
        from jasna.ltx.restore import LtxSegment, LtxSpan, video_frames

        return LtxSpan(
            video_frames(
                self.input_video,
                metadata,
                batch_size=self.batch_size,
                device=self.device,
                frame_stride=1,
                seek_ts=index.seconds_for_pts(span.start_pts),
                end_pts=span.end_pts,
            ),
            tuple(
                LtxSegment(start, end, segment.restoration.ltx_seed)
                for (start, end), segment in zip(span.effect_ranges, segments)
            ),
            open_writer,
        )

    def _run_ltx_spans(
        self,
        metadata,
        index: KeyframeIndex,
        spans: list[tuple[SpliceSpan, tuple[SegmentRange, ...], Callable[[], _OfflineFrameWriter]]],
        work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Restore the LTX render spans in one batched run, so every model loads once."""
        from jasna.ltx.restore import Cancelled, restore_spans

        render = self.ltx_segment_render(metadata)
        frames = sum(_span_frames(span, index, metadata) for span, _, _ in spans)
        try:
            restore_spans(
                [self.ltx_span(metadata, index, span, segments, open_writer) for span, segments, open_writer in spans],
                render,
                work_dir=work_dir,
                progress=self._ltx_progress(frames, progress_callback),
                cancel=self._cancel_event,
            )
        except Cancelled:
            log.info("LTX restoration cancelled")

    def _run_smart(self, metadata) -> None:
        if windows_d3d11_hip_resident_requested():
            raise RuntimeError(
                "The explicit Windows D3D11/HIP resident route is not admitted "
                "for Smart Render spans"
            )
        codec = validate_smart_render(
            metadata,
            output_path=self.output_video,
            codec=self.codec,
            retarget_high_fps=self.retarget_high_fps,
        )
        if self.splice_plan is None:
            index = probe_keyframes(self.input_video, metadata)
            plan = build_splice_plan(self.segments or (), index, duration=metadata.duration)
        else:
            plan = self.splice_plan
            if plan.segments != tuple(self.segments or ()):
                raise ValueError("Precomputed splice plan does not match pipeline segments")
            index = plan.index
        vendor = vendor_for_device(self.device)
        recycle_h264_render_sessions = bool(
            is_isolated_video_job()
            and vendor is AcceleratorVendor.AMD
            and codec == "h264"
        )
        bounded_amf_session_seconds = _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec=codec,
            vendor=vendor,
            batch_size=int(getattr(self, "batch_size", 0) or 0),
            dual_gop_enabled=bool(
                getattr(self, "amd_dual_gop_encode", False)
            ),
        )
        if bounded_amf_session_seconds is not None:
            original_render_spans = len(plan.render_spans)
            plan = split_render_spans(
                plan,
                video_fps=metadata.video_fps_exact,
                max_duration_seconds=bounded_amf_session_seconds,
            )
            log.info(
                "Bounded Linux AMD %s render sessions: %.1fs maximum, "
                "%d -> %d render spans",
                codec.upper(),
                bounded_amf_session_seconds,
                original_render_spans,
                len(plan.render_spans),
            )
        default = job_restoration(self.restoration_model_name, self.ltx_seed)
        segments_of = {
            span: resolve_restorations(segments, default)
            for span, segments in zip(plan.render_spans, plan.render_span_segments())
        }
        for segments in segments_of.values():
            require_one_model(segments)
        ltx_spans = {span for span, segments in segments_of.items() if segments and segments[0].restoration.model == "ltx"}
        # AMF's H.264 encoder caps at 3 consecutive B-frames, so it cannot
        # match sources using more; re-render segments would not stitch
        # cleanly against the stream-copied ones. Fall back to a full
        # re-encode instead of failing the job (NVIDIA NVENC has no such cap).
        if (
            vendor_for_device(self.device) is AcceleratorVendor.AMD
            and codec == "h264"
            and index.max_b_frames > 3
        ):
            log.warning(
                "%s uses %d consecutive B-frames; AMF H.264 smart rendering supports "
                "at most 3, falling back to a full re-encode",
                self.input_video,
                index.max_b_frames,
            )
            self._run_full(
                metadata,
                effect_ranges=tuple(
                    effect_range
                    for span in plan.render_spans
                    for effect_range in span.effect_ranges
                ),
            )
            return
        smart_encoder_settings = resolve_smart_encoder_settings(
            codec,
            metadata,
            index,
            self.encoder_settings,
            vendor=vendor,
        )
        smart_encoder_settings = _dual_gop_smart_encoder_settings(
            smart_encoder_settings,
            enabled=bool(getattr(self, "amd_dual_gop_encode", False)),
        )
        standard_frames = sum(
            _span_frames(span, index, metadata) for span in plan.render_spans if span not in ltx_spans
        )
        ltx_callback = standard_callback = self.progress_callback
        if self.progress_callback is not None and ltx_spans and standard_frames:
            job_progress = JobProgress(
                self.progress_callback,
                {
                    "ltx": LTX_WORK_PER_FRAME * sum(_span_frames(span, index, metadata) for span in ltx_spans),
                    "standard": float(standard_frames),
                },
            )
            ltx_callback, standard_callback = job_progress.part("ltx"), job_progress.part("standard")
        progress = Progressbar(
            total_frames=max(1, standard_frames),
            video_fps=metadata.video_fps,
            disable=self.disable_progress,
            callback=standard_callback,
        )
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        work_root = self.working_dir or self.output_video.parent
        work_root.mkdir(parents=True, exist_ok=True)

        vr_resolution = getattr(self, "vr_resolution", None)
        resolved_projection = (
            vr_resolution.projection
            if vr_resolution is not None
            else getattr(self, "vr_projection", "auto")
        )
        signature = workspace_signature(
            source=self.input_video,
            output=self.output_video,
            plan=plan,
            processing=getattr(self, "processing_signature", {}),
            model_files={
                "detection": getattr(self, "detection_model_path", None),
                "restoration": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "restorer",
                        None,
                    ),
                    "checkpoint_path",
                    None,
                ),
                "secondary": getattr(
                    getattr(
                        getattr(self, "restoration_pipeline", None),
                        "secondary_restorer",
                        None,
                    ),
                    "engine_path",
                    None,
                ),
                "lut": self.lut_path,
            },
            codec=codec,
            encoder_settings=smart_encoder_settings,
            resolved_projection=resolved_projection,
        )
        workspace = SmartRenderWorkspace.open(
            work_root,
            output=self.output_video,
            signature=signature,
        )
        render_metadata = None
        render_output_fps = None
        try:
            fragments: list[tuple[Path, float]] = []
            remaining_render_spans = len(plan.render_spans)
            fragment_suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
            def raw_path(span_index: int) -> Path:
                return workspace.raw_path(span_index)

            def fragment_encoder(span_index: int, span: SpliceSpan) -> VideoEncoder:
                fragment_metadata, fragment_fps = (
                    resolve_hevc_smart_render_vui(metadata)
                    if codec == "hevc" else (metadata, metadata.video_fps_exact)
                )
                return VideoEncoder(
                    str(raw_path(span_index)), device=self.device, metadata=fragment_metadata,
                    codec=codec, encoder_settings=smart_encoder_settings,
                    lut_path=self.lut_path, sharpen_strength=self.sharpen_strength,
                    output_fps=fragment_fps, pts_origin=span.start_pts, smart_fragment=True,
                    match_input_bit_depth=True, mux_audio=False,
                    auto_source_rate=self.auto_source_rate,
                    prefer_amf_host_native=self.amd_dual_gop_encode,
                )

            def fragment_writer(span_index: int, span: SpliceSpan) -> Callable[[], _OfflineFrameWriter]:
                return lambda: _OfflineFrameWriter(fragment_encoder(span_index, span), [time.monotonic()])

            if ltx_spans:
                self._run_ltx_spans(
                    metadata,
                    index,
                    [
                        (
                            span,
                            segments_of[span],
                            fragment_writer(span_index, span),
                        )
                        for span_index, span in enumerate(plan.spans)
                        if span in ltx_spans and workspace.reusable_fragment(span_index) is None
                    ],
                    work_root,
                    ltx_callback,
                )
            for span_index, span in enumerate(plan.spans):
                if self._cancel_event.is_set():
                    return
                duration = float((span.end_pts - span.start_pts) * index.time_base)
                expected_frames = max(1, round(duration * metadata.video_fps))
                reusable = workspace.reusable_fragment(span_index)
                if (
                    reusable is not None
                    and codec == "hevc"
                    and not span.is_render
                    and not hevc_copy_fragment_timeline_matches_source(
                        reusable,
                        self.input_video,
                        span,
                        index,
                    )
                ):
                    log.warning(
                        "Discarding stale HEVC Smart Render copy span %s with "
                        "a source-mismatched packet timeline: %s",
                        span_index,
                        reusable,
                    )
                    reusable = None
                if reusable is not None:
                    log.info("Reusing smart-render span %s from %s", span_index, reusable)
                    fragments.append((reusable, duration))
                    if span.is_render:
                        remaining_render_spans -= 1
                        if span not in ltx_spans:
                            progress.mark_completed(expected_frames)
                    continue

                workspace.mark_running(span_index)
                raw = workspace.raw_path(span_index)
                normalized = workspace.fragment_path(span_index, fragment_suffix)
                if span not in ltx_spans:
                    raw.unlink(missing_ok=True)
                normalized.unlink(missing_ok=True)
                recycle_after_pressure = False
                if span.is_render and span not in ltx_spans:
                    if render_metadata is None:
                        render_metadata = metadata
                        render_output_fps = metadata.video_fps_exact
                        if codec == "hevc":
                            render_metadata, render_output_fps = (
                                resolve_hevc_smart_render_vui(metadata)
                            )
                    encoder_ctx = VideoEncoder(
                        str(raw),
                        device=self.device,
                        metadata=render_metadata,
                        codec=codec,
                        encoder_settings=smart_encoder_settings,
                        lut_path=self.lut_path,
                        sharpen_strength=self.sharpen_strength,
                        output_fps=render_output_fps,
                        mux_audio=False,
                        pts_origin=span.start_pts,
                        match_input_bit_depth=True,
                        smart_fragment=True,
                        auto_source_rate=getattr(self, "auto_source_rate", False),
                        prefer_amf_host_native=bool(
                            getattr(self, "amd_dual_gop_encode", False)
                        ),
                    )
                    pass_stats = self._run_pass(
                        metadata=metadata,
                        encoder_ctx=encoder_ctx,
                        progress=progress,
                        seek_ts=index.seconds_for_pts(span.start_pts),
                        end_pts=span.end_pts,
                        effect_ranges=span.effect_ranges,
                        output_frame_count=expected_frames,
                        recycle_on_host_memory_pressure=True,
                    )
                    recycle_after_pressure = bool(
                        is_isolated_video_job()
                        and (
                            pass_stats.system_pressure_episodes > 0
                            or pass_stats.system_reclaim_count > 0
                        )
                    )
                elif not span.is_render:
                    if codec in {"h264", "hevc"}:
                        create_normalized_copy_fragment(
                            self.input_video,
                            span,
                            index,
                            normalized,
                            codec=codec,
                        )
                    else:
                        create_copy_fragment(
                            self.input_video,
                            span,
                            index,
                            raw,
                            codec=codec,
                        )
                if self._cancel_event.is_set():
                    return
                if span.is_render:
                    normalize_fragment(
                        raw,
                        normalized,
                        codec=codec,
                        decode_delay=index.decode_delay_pts * index.time_base,
                    )
                elif codec == "av1":
                    normalize_fragment(raw, normalized, codec=codec)
                workspace.mark_complete(span_index, normalized)
                if span.is_render:
                    remaining_render_spans -= 1
                try:
                    raw.unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not clean raw smart-render span %s", raw)
                fragments.append((normalized, duration))
                recycle_for_session_limit = bool(
                    span.is_render
                    and (
                        bounded_amf_session_seconds is not None
                        or recycle_h264_render_sessions
                    )
                    and remaining_render_spans > 0
                )
                if (
                    (recycle_after_pressure or recycle_for_session_limit)
                    and remaining_render_spans > 0
                ):
                    if recycle_after_pressure:
                        message = (
                            "Linux AMD whole-card pressure occurred while completing "
                            f"Smart Render span {span_index}; recycling the isolated "
                            "worker before another AMF decoder pair is opened"
                        )
                        recycle_reason = "native_pressure"
                    else:
                        if recycle_h264_render_sessions:
                            message = (
                                "Linux AMD H.264 Smart Render span "
                                f"{span_index} completed; recycling the isolated "
                                "worker before another AMF encoder session opens"
                            )
                        else:
                            message = (
                                "Linux AMD 8K HEVC AMF render-session limit completed "
                                f"at Smart Render span {span_index}; recycling the "
                                "isolated worker before the next bounded span"
                            )
                        recycle_reason = "amf_session_limit"
                    if recycle_after_pressure:
                        log.warning(message)
                    else:
                        # This is an expected resource boundary, not a
                        # user-facing warning.  The GUI parent keeps progress
                        # and ETA stable while the fresh worker resumes the
                        # durable workspace.
                        log.debug(message)
                    raise NativeWorkerRecycleRequested(
                        message,
                        reason=recycle_reason,
                    )

            if self._cancel_event.is_set():
                return
            if codec == "hevc":
                log.info(
                    "Finalizing HEVC Smart Render: checking boundary "
                    "VPS/SPS/PPS for %d fragments",
                    len(fragments),
                )
                validate_hevc_fragment_parameter_sets(
                    [
                        (fragment, span.kind)
                        for (fragment, _duration), span in zip(
                            fragments,
                            plan.spans,
                        )
                    ]
                )
            mux_fragments_final_output(
                fragments,
                self.input_video,
                self.output_video,
                manifest=workspace.path / "fragments.ffconcat",
                codec=codec,
                copy_validation_ranges=(
                    self._hevc_copy_validation_ranges(plan)
                    if codec == "hevc"
                    else ()
                ),
            )
            if self._cancel_event.is_set():
                return
            try:
                workspace.cleanup()
            except OSError:
                log.warning(
                    "Could not clean completed smart-render workspace %s",
                    workspace.path,
                )
        except SmartRenderCompatibilityError:
            log.error(
                "Preserving rejected Smart Render workspace for diagnosis or "
                "resume: %s",
                workspace.path,
            )
            raise
        finally:
            progress.close(ensure_completed_bar=True)

    @staticmethod
    def _hevc_copy_validation_ranges(
        plan: SplicePlan,
    ) -> tuple[tuple[float, float], ...]:
        """Return at most one second on every untouched side of a render seam."""

        keyframes = plan.index.pts
        one_second_pts = max(1, int(1 / plan.index.time_base))
        ranges: set[tuple[int, int]] = set()
        for span_index, span in enumerate(plan.spans):
            if span.is_render:
                continue
            if span_index > 0 and plan.spans[span_index - 1].is_render:
                next_keyframe = next(
                    (pts for pts in keyframes if pts > span.start_pts),
                    span.end_pts,
                )
                ranges.add(
                    (
                        span.start_pts,
                        min(
                            span.end_pts,
                            next_keyframe,
                            span.start_pts + one_second_pts,
                        ),
                    )
                )
            if span_index + 1 < len(plan.spans) and plan.spans[span_index + 1].is_render:
                previous_index = bisect.bisect_left(keyframes, span.end_pts) - 1
                start_pts = keyframes[max(0, previous_index)]
                ranges.add(
                    (
                        max(
                            span.start_pts,
                            start_pts,
                            span.end_pts - one_second_pts,
                        ),
                        span.end_pts,
                    )
                )
        return tuple(
            (
                plan.index.seconds_for_pts(start_pts),
                float((end_pts - start_pts) * plan.index.time_base),
            )
            for start_pts, end_pts in sorted(ranges)
            if end_pts > start_pts
        )

    def run(self) -> None:
        metadata = get_video_meta_data(str(self.input_video))
        self.validate_metadata(metadata)
        self.configure_vr(metadata)
        if self.segments:
            if self.fmp4:
                log.warning(
                    "Fragmented MP4 is not available with segment processing; "
                    "the output is assembled after processing finishes"
                )
                self.fmp4 = False
            self._run_smart(metadata)
        elif self.restoration_model_name == "ltx":
            self._run_ltx(metadata)
        else:
            self._run_full(metadata)
        self.completed = not self._cancel_event.is_set()

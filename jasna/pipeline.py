from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import psutil
import torch

from jasna.accelerator import vendor_for_device
from jasna.job_resume import ResumeJob, resume_directory
from jasna.media.container_utils import MOV_SUFFIXES
from jasna.media.encoder_settings import require_encodable_size
from jasna.media.probe import UnsupportedVideoError, get_video_meta_data
from jasna.media.video_encoder import VideoEncoder, resolve_encoder_options
from jasna.media.frame_rate import resolve_frame_rate_retarget
from jasna.media.splice import (
    H264_SMART_MAX_B_FRAMES,
    KeyframeIndex,
    SplicePlan,
    SpliceSpan,
    concatenate_fragments,
    create_copy_fragment,
    mux_final_output,
    normalize_fragment,
    require_one_model,
    resolve_smart_encoder_settings,
)
from jasna.os_utils import commit_file
from jasna.pipeline_threads import run_restoration_pass
from jasna.progressbar import JobProgress, ProgressCallback, Progressbar
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.segments import SegmentRange, job_models, job_restoration, resolve_restorations
from jasna.session_config import RestorationModelName, SessionConfig
from jasna.settings_tag import job_settings, settings_comment
from jasna.vr180 import (
    SbsDetectionAdapter,
    require_ltx_input,
    resolve_vr_mode,
)
from jasna.vr_projection import build_vr_projector

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter, LtxRender, LtxSpan
    from jasna.session_factory import RestorationSession

log = logging.getLogger(__name__)


def _span_seconds(span: SpliceSpan, index: KeyframeIndex) -> float:
    return float((span.end_pts - span.start_pts) * index.time_base)


def _span_frames(span: SpliceSpan, index: KeyframeIndex, metadata) -> int:
    return round((span.end_pts - span.start_pts) * index.time_base * metadata.video_fps)


class _OfflineFrameWriter:
    def __init__(self, encoder_ctx: VideoEncoder, encode_heartbeat: list[float]):
        self._encoder_ctx = encoder_ctx
        self._encode_heartbeat = encode_heartbeat
        self._entered = False

    def write(self, frame: torch.Tensor, pts: int) -> None:
        if not self._entered:
            self._encoder_ctx.__enter__()
            self._entered = True
        self._encoder_ctx.encode(frame, pts)
        self._encode_heartbeat[0] = time.monotonic()

    def after_write(self, frames_written: int, stop_event: threading.Event) -> None:
        pass

    def finish(self) -> None:
        self.close()

    def close(self) -> None:
        if self._entered:
            self._encoder_ctx.__exit__(None, None, None)
            self._entered = False


class _FragmentFrameWriter(_OfflineFrameWriter):
    """Writes a smart-render fragment; ``finish`` publishes it once every frame is written."""

    def __init__(self, encoder: VideoEncoder, commit: Callable[[], None]) -> None:
        super().__init__(encoder, [time.monotonic()])
        self._commit = commit

    def finish(self) -> None:
        self.close()
        self._commit()


class _FragmentDir:
    """The raw fragment of each splice span. In a checkpoint (``durable``) a fragment is
    written as ``partial(i)`` and becomes ``final(i)`` only once complete and on disk, so a
    restart skips exactly the finished ones."""

    def __init__(self, directory: Path, *, durable: bool) -> None:
        self.directory = directory
        self.durable = durable

    def final(self, span_index: int) -> Path:
        return self.directory / f"{span_index:04d}-raw.nut"

    def partial(self, span_index: int) -> Path:
        return self.directory / f"{span_index:04d}-partial.nut" if self.durable else self.final(span_index)

    def done(self, span_index: int) -> bool:
        return self.final(span_index).is_file()

    def commit(self, span_index: int) -> None:
        if self.durable:
            with self.partial(span_index).open("r+b") as stream:
                os.fsync(stream.fileno())
            commit_file(self.partial(span_index), self.final(span_index))


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
    ) -> None:
        self.config = config
        self.resume: ResumeJob | None = None
        self.input_video = input_video
        self.output_video = output_video
        self.working_dir = config.working_dir
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
        self.ltx_super_res = session.ltx_super_res if config.secondary_restoration == "rtx-super-res" else None
        self.restoration_model_name = config.restoration_model_name
        self.ltx_large_canvas = config.ltx_large_canvas
        self.ltx_window_hop = config.ltx_window_hop
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
    ) -> None:
        frame_rate = resolve_frame_rate_retarget(
            metadata.video_fps_exact,
            enabled=self.retarget_high_fps,
            measured_fps=metadata.average_fps,
        )
        if output_frame_count is None:
            output_frame_count = frame_rate.output_frame_count(metadata.num_frames)

        encode_heartbeat: list[float] = [time.monotonic()]
        frame_writer = _OfflineFrameWriter(encoder_ctx, encode_heartbeat)
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
            )
        finally:
            frame_writer.close()

        self._log_memory_at_end()
        if error is not None:
            error.__traceback__ = None
            raise error

    def _log_memory_at_end(self) -> None:
        free, total = torch.cuda.mem_get_info(self.device)
        log.info("VRAM usage at end — %.1f MiB", (total - free) / (1024 ** 2))
        log.info("RAM usage at end — %.1f MiB", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))

    def validate_metadata(self, metadata) -> None:
        from av.video.reformatter import Colorspace as AvColorspace

        if metadata.color_space not in (
            AvColorspace.ITU709,
            AvColorspace.ITU601,
            AvColorspace.BT2020,
        ):
            raise UnsupportedVideoError(
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

    def _job_settings(
        self, models: frozenset[RestorationModelName], *, ltx_large_canvas: bool | None
    ) -> dict[str, object]:
        default = job_restoration(self.restoration_model_name, self.ltx_seed)
        return job_settings(
            self.config,
            models=models,
            segments=resolve_restorations(self.segments, default) if self.segments else None,
            vr=self.vr_resolution,
            ltx_weights=self.ltx_files.transformer_name if "ltx" in models else None,
            ltx_large_canvas=ltx_large_canvas,
        )

    def _video_encoder(self, metadata, frame_rate, settings: dict[str, object]) -> VideoEncoder:
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
            fmp4=self.fmp4,
            settings=settings,
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
        encoder_ctx = self._video_encoder(
            metadata,
            frame_rate,
            self._job_settings(job_models(self.restoration_model_name, self.segments), ltx_large_canvas=None),
        )
        try:
            self._run_pass(
                metadata=metadata,
                encoder_ctx=encoder_ctx,
                progress=progress,
                effect_ranges=effect_ranges,
                output_frame_count=output_frame_count,
            )
        finally:
            progress.close(ensure_completed_bar=True)

    def _ltx_progress(self, frames: int, callback: ProgressCallback | None):
        from jasna.ltx.restore import Progress

        report = None
        if callback is not None:
            report = lambda fraction, eta, stage: callback(fraction * 100.0, 0.0, eta, 0, 0, stage)
        return Progress(frames, disable=self.disable_progress, report=report)

    def _run_ltx(self, metadata) -> None:
        from jasna.ltx.plan import track_max_gap
        from jasna.ltx.restore import Cancelled, restore_video, video_frames, whole_video_large_canvas
        from jasna.ltx.speed import profile_path

        frame_rate = self._resolve_frame_rate(metadata)
        resume_dir = self.resume.directory / "ltx"
        large_canvas = whole_video_large_canvas(self.ltx_large_canvas, self.device, resume_dir)
        settings = self._job_settings(frozenset({"ltx"}), ltx_large_canvas=large_canvas)
        writer = _OfflineFrameWriter(self._video_encoder(metadata, frame_rate, settings), [time.monotonic()])
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
                large_canvas=large_canvas,
                window_hop=self.ltx_window_hop,
                seed=self.ltx_seed,
                device=self.device,
                work_dir=self.working_dir or self.output_video.parent,
                speed_profile=profile_path(),
                resume_dir=resume_dir,
                progress=self._ltx_progress(
                    frame_rate.output_frame_count(metadata.num_frames), self.progress_callback
                ),
                cancel=self._cancel_event,
                max_gap=track_max_gap(self.max_detection_gap),
                min_duration=self.min_detection_duration,
                super_res=self.ltx_super_res,
            )
        except Cancelled:
            log.info("LTX restoration cancelled")
        finally:
            writer.close()
        self._log_memory_at_end()

    def ltx_segment_render(self, metadata) -> "LtxRender":
        """How LTX restores segments: the 768 px decision and the decode tiles follow the
        GPU's size, never the VRAM free at that moment, so a segment renders the same in a
        job and in a seed preview."""
        from jasna.ltx.plan import track_max_gap
        from jasna.ltx.restore import LtxRender, segment_decode_budget, segment_large_canvas
        from jasna.ltx.speed import profile_path

        require_ltx_input(self.vr_resolution)
        total_bytes = torch.cuda.get_device_properties(self.device).total_memory
        return LtxRender(
            detector=self.job_detection_model,
            files=self.ltx_files,
            frame_h=int(metadata.video_height),
            frame_w=int(metadata.video_width),
            batch_size=self.batch_size,
            large_canvas=segment_large_canvas(self.ltx_large_canvas, total_bytes),
            window_hop=self.ltx_window_hop,
            budget=segment_decode_budget(total_bytes),
            budget_name="segment",
            device=self.device,
            speed_profile=profile_path(),
            max_gap=track_max_gap(self.max_detection_gap),
            min_duration=self.min_detection_duration,
            super_res=self.ltx_super_res,
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

    def _ltx_estimate_seconds(self, metadata, frames: int) -> float:
        """About how long LTX takes on ``frames``, before its scan has planned them."""
        from jasna.ltx.speed import profile_path, segment_estimate_seconds

        return segment_estimate_seconds(
            profile_path(),
            torch.cuda.get_device_name(self.device),
            self.ltx_files.transformer_name,
            frames=frames,
            fps=metadata.video_fps,
            megapixels=metadata.video_height * metadata.video_width / 1e6,
            window_hop=self.ltx_window_hop,
        )

    def _run_ltx_spans(
        self,
        render: "LtxRender",
        metadata,
        index: KeyframeIndex,
        spans: list[tuple[SpliceSpan, tuple[SegmentRange, ...], Callable[[], _OfflineFrameWriter]]],
        work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Restore the LTX render spans in one batched run, so every model loads once."""
        from jasna.ltx.restore import Cancelled, restore_spans

        frames = sum(_span_frames(span, index, metadata) for span, _, _ in spans)
        try:
            restore_spans(
                [self.ltx_span(metadata, index, span, segments, open_writer) for span, segments, open_writer in spans],
                render,
                work_dir=work_dir,
                resume_dir=self.resume.directory / "ltx",
                progress=self._ltx_progress(frames, progress_callback),
                cancel=self._cancel_event,
            )
        except Cancelled:
            log.info("LTX restoration cancelled")
        self._log_memory_at_end()

    def _reencode_span(
        self, metadata, index: KeyframeIndex, span: SpliceSpan, writer: _OfflineFrameWriter, progress: Progressbar
    ) -> None:
        """A span with nothing to restore, encoded again so the LUT and sharpening cover it as well."""
        from jasna.ltx.restore import video_frames

        frames = video_frames(
            self.input_video,
            metadata,
            batch_size=self.batch_size,
            device=self.device,
            frame_stride=1,
            seek_ts=index.seconds_for_pts(span.start_pts),
            end_pts=span.end_pts,
        )
        try:
            for batch, pts in frames():
                if self._cancel_event.is_set():
                    return
                for frame, frame_pts in zip(batch, pts):
                    writer.write(frame, frame_pts)
                progress.update(len(pts))
            writer.finish()
        finally:
            writer.close()

    def _run_smart(self, metadata) -> None:
        plan = self.splice_plan
        index = plan.index
        default = job_restoration(self.restoration_model_name, self.ltx_seed)
        segments_of = {
            span: resolve_restorations(segments, default)
            for span, segments in zip(plan.render_spans, plan.render_span_segments())
        }
        for segments in segments_of.values():
            require_one_model(segments)
        ltx_spans = {span for span, segments in segments_of.items() if segments[0].restoration.model == "ltx"}
        # Re-rendered segments must match the source's B-frames to stitch
        # against the stream-copied ones; past the encoder's cap, re-encode all.
        vendor = vendor_for_device(self.device)
        if (
            self.codec == "h264"
            and vendor in H264_SMART_MAX_B_FRAMES
            and index.max_b_frames > H264_SMART_MAX_B_FRAMES[vendor]
        ):
            log.warning(
                "%s uses %d consecutive B-frames; %s H.264 smart rendering supports "
                "at most %d, falling back to a full re-encode",
                self.input_video,
                index.max_b_frames,
                vendor.value,
                H264_SMART_MAX_B_FRAMES[vendor],
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
            self.codec,
            metadata,
            index,
            self.encoder_settings,
            vendor=vendor,
        )
        smart_spec, smart_encoder_options = resolve_encoder_options(
            vendor, self.codec, metadata, smart_encoder_settings, smart_fragment=True
        )
        ltx_render = self.ltx_segment_render(metadata) if ltx_spans else None
        comment = settings_comment(
            self._job_settings(
                frozenset(segments[0].restoration.model for segments in segments_of.values()),
                ltx_large_canvas=None if ltx_render is None else ltx_render.large_canvas,
            ),
            encoder=smart_spec.encoder_name,
            options=smart_encoder_options,
        )
        standard_spans = {span for span in plan.spans if span.kind != "copy" and span not in ltx_spans}
        standard_frames = sum(_span_frames(span, index, metadata) for span in standard_spans)
        ltx_callback = standard_callback = self.progress_callback
        if self.progress_callback is not None and ltx_spans and standard_frames:
            ltx_frames = sum(_span_frames(span, index, metadata) for span in ltx_spans)
            job_progress = JobProgress(
                self.progress_callback,
                {"standard": 0.0, "ltx": self._ltx_estimate_seconds(metadata, ltx_frames)},
            )
            standard_callback, ltx_callback = job_progress.part("standard"), job_progress.part("ltx")
        progress = Progressbar(
            total_frames=max(1, standard_frames),
            video_fps=metadata.video_fps,
            disable=self.disable_progress,
            callback=standard_callback,
        )
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        work_root = self.working_dir or self.output_video.parent
        work_root.mkdir(parents=True, exist_ok=True)

        def fragment_encoder(fragments: _FragmentDir, span_index: int, span: SpliceSpan) -> VideoEncoder:
            return VideoEncoder(
                str(fragments.partial(span_index)),
                device=self.device,
                metadata=metadata,
                codec=self.codec,
                encoder_settings=smart_encoder_settings,
                lut_path=self.lut_path,
                sharpen_strength=self.sharpen_strength,
                output_fps=metadata.video_fps_exact,
                pts_origin=span.start_pts,
                smart_fragment=True,
            )

        def fragment_writer(fragments: _FragmentDir, span_index: int, span: SpliceSpan):
            return lambda: _FragmentFrameWriter(
                fragment_encoder(fragments, span_index, span), lambda: fragments.commit(span_index)
            )

        directory = (
            nullcontext(self.resume.directory / "segments") if self.resume is not None
            else TemporaryDirectory(dir=work_root, prefix=f".{self.output_video.stem}.segments-")
        )
        try:
            with directory as directory_name:
                fragments = _FragmentDir(Path(directory_name), durable=self.resume is not None)
                fragments.directory.mkdir(parents=True, exist_ok=True)
                for span_index, span in enumerate(plan.spans):
                    if self._cancel_event.is_set():
                        return
                    if fragments.done(span_index):
                        if span in standard_spans:
                            progress.update(_span_frames(span, index, metadata))
                    elif span.kind == "copy":
                        create_copy_fragment(
                            self.input_video, span, index, fragments.partial(span_index), codec=self.codec
                        )
                        fragments.commit(span_index)
                    elif span.kind == "reencode":
                        self._reencode_span(
                            metadata, index, span, fragment_writer(fragments, span_index, span)(), progress
                        )
                    elif span not in ltx_spans:
                        self._run_pass(
                            metadata=metadata,
                            encoder_ctx=fragment_encoder(fragments, span_index, span),
                            progress=progress,
                            seek_ts=index.seconds_for_pts(span.start_pts),
                            end_pts=span.end_pts,
                            effect_ranges=span.effect_ranges,
                            output_frame_count=max(1, round(_span_seconds(span, index) * metadata.video_fps)),
                        )
                        if not self._cancel_event.is_set():
                            fragments.commit(span_index)
                pending_ltx = [
                    (span, segments_of[span], fragment_writer(fragments, span_index, span))
                    for span_index, span in enumerate(plan.spans)
                    if span in ltx_spans and not fragments.done(span_index)
                ]
                if pending_ltx:
                    self._run_ltx_spans(ltx_render, metadata, index, pending_ltx, work_root, ltx_callback)
                if self._cancel_event.is_set():
                    return
                self._assemble(fragments, plan, comment)
        finally:
            progress.close(ensure_completed_bar=True)

    def _assemble(self, fragments: _FragmentDir, plan: SplicePlan, comment: str) -> None:
        suffix = ".ts" if self.codec in {"h264", "hevc"} else ".mkv"
        normalized = []
        for span_index, span in enumerate(plan.spans):
            path = fragments.directory / f"{span_index:04d}{suffix}"
            normalize_fragment(fragments.final(span_index), path, codec=self.codec)
            normalized.append((path, _span_seconds(span, plan.index)))
        assembled = fragments.directory / f"assembled{suffix}"
        concatenate_fragments(
            normalized, manifest=fragments.directory / "fragments.ffconcat", destination=assembled, codec=self.codec
        )
        mux_final_output(assembled, self.input_video, self.output_video, codec=self.codec, comment=comment)

    def run(self) -> None:
        metadata = get_video_meta_data(str(self.input_video))
        self.validate_metadata(metadata)
        require_encodable_size(vendor_for_device(self.device), self.codec, metadata.video_width, metadata.video_height)
        self.configure_vr(metadata)
        uses_ltx = "ltx" in job_models(self.restoration_model_name, self.segments)
        if uses_ltx:
            require_ltx_input(self.vr_resolution)
        auto_split = self.restoration_model_name == "ltx" and self.config.ltx_auto_split_percent is not None
        if uses_ltx or auto_split or (resume_directory(self.config, self.output_video) / "render").exists():
            self.resume = ResumeJob(self.config, self.input_video, self.output_video, metadata, self.splice_plan)
            self.resume.prepare()
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
        if self.completed and self.resume is not None:
            self.resume.finish()

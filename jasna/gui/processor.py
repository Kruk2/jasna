"""Background processor for video processing jobs."""

import logging
import threading
import traceback
import queue
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable
from uuid import uuid4

from jasna.gui.models import JobItem, JobStatus, AppSettings
from jasna.gui.video_session import build_video_session, release_session_memory, video_session_config
from jasna.media import UnsupportedColorspaceError
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline

logger = logging.getLogger(__name__)

_OutputFingerprint = tuple[int, int, int, int, int, int]


@dataclass
class ProgressUpdate:
    job_id: int
    status: JobStatus
    progress: float = 0.0
    fps: float = 0.0
    eta_seconds: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    message: str = ""


class ProcessingStopped(Exception):
    """Raised inside a job when the user stopped processing."""


def _pipeline_was_stopped(pipeline) -> bool:
    return bool(pipeline.cancel_requested) and not bool(pipeline.completed)


def _cleanup_torch(torch_mod) -> None:
    import gc

    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.ipc_collect()
        torch_mod.cuda.reset_peak_memory_stats()


class Processor:
    """Handles video processing in a background thread."""
    
    def __init__(
        self,
        on_progress: Callable[[ProgressUpdate], None] = None,
        on_log: Callable[[str, str], None] = None,
        on_complete: Callable[[], None] = None,
    ):
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._completion_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self._jobs: list[JobItem] = []
        self._settings: AppSettings | None = None
        self._output_folder: str = ""
        self._output_pattern: str = "{original}_restored.mp4"
        self._disable_basicvsrpp_tensorrt_for_run = False

        # Heavy models are loaded once and reused across consecutive jobs of the
        # same type; the other session is unloaded when the type switches.
        self._img_session: tuple | None = None      # (detector, restorer, device)
        self._video_session: RestorationSession | None = None
        self._current_pipeline = None
        
    def start(
        self,
        jobs: list[JobItem],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
        *,
        disable_basicvsrpp_tensorrt: bool,
    ):
        if self._thread and self._thread.is_alive():
            return
            
        self._jobs = jobs
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._disable_basicvsrpp_tensorrt_for_run = bool(disable_basicvsrpp_tensorrt)
        
        with self._completion_lock:
            self._stop_event.clear()
        self._pause_event.set()
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()
            
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()
        
    def stop(self):
        # Linearize Stop against the final job-state commit. If Stop wins this
        # lock after output validation, the current job remains pending; if the
        # completion commit wins, the completed job remains authoritative.
        with self._completion_lock:
            self._stop_event.set()
        self._pause_event.set()  # Unpause to allow thread to exit
        pipeline = self._current_pipeline
        if pipeline is not None:
            pipeline.cancel()

    def join(self, timeout: float = 5.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
        
    def _log(self, level: str, message: str):
        if self._on_log:
            self._on_log(level, message)
            
    def _progress(self, update: ProgressUpdate):
        if self._on_progress:
            self._on_progress(update)
            
    def _next_pending_job(self) -> JobItem | None:
        for job in self._jobs:
            if job.status == JobStatus.PENDING:
                return job
        return None

    def _validate_completed_video_output(
        self,
        input_path: Path,
        output_path: Path,
        *,
        codec: str,
        smart_render: bool,
        previous_fingerprint: _OutputFingerprint | None,
    ) -> None:
        from jasna.media.splice import (
            sync_and_validate_final_output,
            validate_video_output,
        )

        self._require_completed_output_changed(output_path, previous_fingerprint)
        if smart_render:
            # Smart-render muxing commits through _commit_smart_output, which
            # already validates and syncs the final output before returning.
            validate_video_output(output_path, source=input_path)
        else:
            sync_and_validate_final_output(
                output_path,
                source=input_path,
                expected_codec=codec,
            )

    @staticmethod
    def _output_fingerprint(path: Path) -> _OutputFingerprint | None:
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return (
            int(info.st_mode),
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
        )

    @classmethod
    def _require_completed_output_changed(
        cls,
        output_path: Path,
        previous_fingerprint: _OutputFingerprint | None,
    ) -> None:
        current_fingerprint = cls._output_fingerprint(output_path)
        if current_fingerprint is None:
            raise ValueError(f"completed output is missing: {output_path}")
        if (
            previous_fingerprint is not None
            and current_fingerprint == previous_fingerprint
        ):
            raise ValueError(
                f"completed output was not created or changed by this job: {output_path}"
            )

    def _commit_completed_job(self, job: JobItem, output_path: Path) -> None:
        """Commit terminal job fields atomically with respect to ``stop()``."""

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            job.output_path = output_path
            job.status = JobStatus.COMPLETED

    def _begin_job_unless_stopped(self, job: JobItem):
        """Claim a pending job in the same ordering domain as ``stop()``."""

        with self._completion_lock:
            if self._stop_event.is_set():
                return None
            return job.begin_processing()

    def _create_output_parent_unless_stopped(self, output_path: Path) -> None:
        """Create job output directories only while the job may still start."""

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _full_render_staging_path(output_path: Path) -> Path:
        return output_path.with_name(
            f".{output_path.stem}.jasna-full-{uuid4().hex}{output_path.suffix}"
        )

    def _publish_full_render_unless_stopped(
        self,
        staging_path: Path,
        output_path: Path,
        *,
        input_path: Path,
        codec: str,
    ) -> None:
        """Atomically publish a full render in the same ordering domain as Stop."""

        from jasna.media.splice import commit_video_output

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            commit_video_output(
                staging_path,
                output_path,
                source=input_path,
                codec=codec,
            )

    def _run(self):
        self._log("INFO", "Processing started")

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                job = self._next_pending_job()
                if job is None:
                    break

                self._process_job(job)
                if job.status is JobStatus.PENDING:
                    break  # stopped mid-job; it stays queued for the next run
        finally:
            self._close_image_session()
            self._close_video_session()

        if self._stop_event.is_set():
            self._log("INFO", "Processing stopped by user")
        else:
            self._log("INFO", "Processing completed")
            self._run_post_export_action()
        if self._on_complete:
            self._on_complete()

    def _run_post_export_action(self):
        settings = self._settings
        if settings is None:
            return
        from jasna.post_export_action import run_post_export_action_safely

        action = settings.post_export_action
        command = settings.post_export_command
        if action == "none":
            return

        self._log("INFO", f"Running post-export action: {action}")
        run_post_export_action_safely(action, command, lambda message: self._log("ERROR", message))
            
    def _process_job(self, job: JobItem):
        snapshot = self._begin_job_unless_stopped(job)
        if snapshot is None:
            return
        segments = snapshot.segments
        self._log("INFO", f"Started processing {job.filename}")
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PROCESSING,
            message=f"Starting {job.filename}",
        ))
        
        input_path = job.path
        from jasna.media.image_io import IMAGE_EXTENSIONS
        is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS
        job_settings = self._settings
        if not is_image:
            overrides = {}
            if snapshot.detection_model is not None:
                overrides["detection_model"] = snapshot.detection_model
            if snapshot.detection_score_threshold is not None:
                overrides["detection_score_threshold"] = snapshot.detection_score_threshold
            if snapshot.vr_projection is not None:
                overrides["vr_projection"] = snapshot.vr_projection
            if overrides:
                job_settings = replace(job_settings, **overrides)

        # Determine output path
        if self._output_folder:
            output_dir = Path(self._output_folder)
        else:
            output_dir = input_path.parent

        output_name = self._output_pattern.replace("{original}", input_path.stem)
        output_path = output_dir / output_name
        if is_image:
            # The video output pattern carries a video extension; images keep their own.
            output_path = output_path.with_suffix(input_path.suffix)
        
        # Handle file conflict based on settings
        file_conflict = self._settings.file_conflict if self._settings else "auto_rename"
        
        if output_path.exists():
            if file_conflict == "skip":
                job.status = JobStatus.SKIPPED
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.SKIPPED,
                    message=f"Output file already exists: {output_path.name}",
                ))
                self._log("WARNING", f"Skipped {job.filename}: output file already exists")
                return
            elif file_conflict == "auto_rename":
                output_path = self._get_unique_output_path(output_path)
                self._log("INFO", f"Renamed output to {output_path.name} to avoid overwrite")
            # "overwrite" - just proceed and let the file be replaced
        
        try:
            self._create_output_parent_unless_stopped(output_path)
            previous_output_fingerprint = (
                self._output_fingerprint(output_path) if not is_image else None
            )
            if is_image:
                self._close_video_session()
            else:
                self._close_image_session()
            pipeline_options = {}
            if segments:
                pipeline_options["segments"] = segments
            if job_settings is not self._settings:
                pipeline_options["settings"] = job_settings
            self._run_pipeline(
                job.id,
                input_path,
                output_path,
                **pipeline_options,
            )
            if not is_image:
                self._validate_completed_video_output(
                    input_path,
                    output_path,
                    codec=job_settings.codec,
                    smart_render=bool(segments),
                    previous_fingerprint=previous_output_fingerprint,
                )

            if not is_image:
                self._run_post_export_video_command(input_path, output_path)
            self._commit_completed_job(job, output_path)
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                progress=100.0,
            ))
            self._log("INFO", f"Finished processing {job.filename}")

        except ProcessingStopped:
            self._mark_stopped(job)

        except UnsupportedColorspaceError as e:
            e.__traceback__ = None
            job.status = JobStatus.SKIPPED
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.SKIPPED,
                message=str(e),
            ))
            self._log("WARNING", f"Skipped {job.filename}: {e}")

        except Exception as e:
            tb = traceback.format_exc()
            e.__traceback__ = None
            job.status = JobStatus.ERROR
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.ERROR,
                message=str(e),
            ))
            self._log("ERROR", f"Failed to process {job.filename}: {e}\n{tb}")

        try:
            import torch
            _cleanup_torch(torch)
        except Exception:
            logger.warning("Torch cleanup failed after job", exc_info=True)

    def _run_post_export_video_command(self, input_path: Path, output_path: Path) -> None:
        settings = self._settings
        if settings is None:
            return
        command = settings.post_export_video_command.strip()
        if not command:
            return
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        from jasna.post_export_action import (
            PostExportVideoCommandCancelled,
            run_post_export_video_command,
        )

        self._log("INFO", f"Running post-export command for {output_path.name}")
        try:
            run_post_export_video_command(
                command,
                input_path,
                output_path,
                self._stop_event.is_set,
            )
        except PostExportVideoCommandCancelled as exc:
            raise ProcessingStopped("Processing stopped") from exc

    def _mark_stopped(self, job: JobItem):
        job.status = JobStatus.PENDING
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PENDING,
        ))
        self._log("INFO", f"Stopped processing {job.filename}")

    def _run_pipeline(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
    ):
        """Run one job; raises ProcessingStopped when the user stopped it."""
        from jasna.media.image_io import IMAGE_EXTENSIONS

        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            self._run_image_job(job_id, input_path, output_path)
            return
        self._run_video_job(
            job_id,
            input_path,
            output_path,
            segments=segments,
            settings=settings or self._settings,
        )


    def _ensure_video_session(self, settings: AppSettings | None = None):
        """Compile engines + build the BasicVSR++ (and optional secondary) restorer
        once; reused across consecutive video jobs."""
        if self._video_session is not None:
            return
        self._video_session = build_video_session(
            settings or self._settings,
            disable_basicvsrpp_tensorrt=self._disable_basicvsrpp_tensorrt_for_run,
            log=lambda msg: self._log("INFO", msg),
        )
        self._log("INFO", "Restoration models loaded (reused across video jobs)")

    def _build_encoder_settings(self, codec: str) -> dict:
        # Built per job (not cached in the video session) so a codec change
        # between queued jobs is always validated against the selected codec.
        from jasna.accelerator import AcceleratorVendor, vendor_for_device
        from jasna.media import parse_encoder_settings, validate_encoder_settings
        from jasna.media.encoder_quality import (
            encoder_cq_spec,
            validate_encoder_cq,
        )

        settings = self._settings
        vendor = vendor_for_device()
        cq = (
            encoder_cq_spec(codec, vendor).default
            if settings.encoder_cq is None
            else settings.encoder_cq
        )
        validate_encoder_cq(cq, codec=codec, vendor=vendor)
        encoder_settings = {"cq": cq}
        if settings.encoder_custom_args:
            custom_settings = parse_encoder_settings(settings.encoder_custom_args)
            cq_aliases = {"cq"}
            if vendor is AcceleratorVendor.AMD:
                cq_aliases.add("qvbr_quality_level")
            duplicates = sorted(cq_aliases & custom_settings.keys())
            if duplicates:
                raise ValueError(
                    "CQ is controlled by the quality slider; remove "
                    f"{', '.join(duplicates)} from custom encoder settings"
                )
            encoder_settings.update(custom_settings)
        return validate_encoder_settings(encoder_settings, codec=codec, vendor=vendor)

    def _run_video_job(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
    ):
        settings = settings or self._settings
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        codec = settings.codec
        splice_plan = None
        if segments:
            from jasna.media import get_video_meta_data
            from jasna.media.splice import build_splice_plan, probe_keyframes, validate_smart_render
            metadata = get_video_meta_data(str(input_path))
            codec = {
                "avc": "h264",
                "h265": "hevc",
                "av01": "av1",
            }.get(metadata.codec_name.lower(), metadata.codec_name.lower())
            validate_smart_render(
                metadata,
                output_path=output_path,
                codec=codec,
                retarget_high_fps=settings.retarget_high_fps,
            )
            splice_plan = build_splice_plan(
                tuple(segments),
                probe_keyframes(input_path, metadata),
                duration=metadata.duration,
            )
        encoder_settings = self._build_encoder_settings(codec)
        config = video_session_config(settings, codec=codec, encoder_settings=encoder_settings)
        self._ensure_video_session(settings)
        s = self._video_session
        self._prepare_job_detector(config, s)
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        last_update_time = [0.0]

        def progress_callback(progress_pct: float, fps: float, eta_seconds: float, frames_done: int, total: int):
            current_time = time.time()
            if current_time - last_update_time[0] < 0.1:
                return
            last_update_time[0] = current_time

            self._pause_event.wait()
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")

            self._progress(ProgressUpdate(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=progress_pct,
                fps=fps,
                eta_seconds=eta_seconds,
                frames_processed=frames_done,
                total_frames=total,
            ))

        pipeline = None
        full_render_staging = (
            self._full_render_staging_path(output_path) if not segments else None
        )
        pipeline_output = full_render_staging or output_path
        try:
            pipeline = build_pipeline(
                config,
                s,
                input_path,
                pipeline_output,
                progress_callback=progress_callback,
                segments=tuple(segments) or None,
                splice_plan=splice_plan,
            )
            self._current_pipeline = pipeline
            if self._stop_event.is_set():
                pipeline.cancel()
            pipeline.run()
            if _pipeline_was_stopped(pipeline):
                raise ProcessingStopped("Processing stopped")
            if full_render_staging is not None:
                self._publish_full_render_unless_stopped(
                    full_render_staging,
                    output_path,
                    input_path=input_path,
                    codec=codec,
                )
        finally:
            self._current_pipeline = None
            if pipeline is not None:
                pipeline.close()
            if full_render_staging is not None:
                try:
                    full_render_staging.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove full-render staging output %s",
                        full_render_staging,
                    )

    def _prepare_job_detector(
        self,
        config: SessionConfig,
        session: RestorationSession,
    ) -> None:
        if (
            config.detection_model_name == session.detection_model_name
            and config.detection_model_path == session.detection_model_path
        ):
            return

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled

        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(session.device),
                fp16=config.fp16,
                detection=True,
                detection_model_name=config.detection_model_name,
                detection_model_path=str(config.detection_model_path),
                detection_batch_size=config.batch_size,
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )

    def _close_video_session(self):
        if self._video_session is None:
            return
        s = self._video_session
        self._video_session = None
        s.close()
        release_session_memory(s.device)
        self._log("INFO", "Restoration models unloaded")

    def _ensure_image_session(self):
        """Load the rf-detr detector + SD 1.5 restorer once; reused across image jobs."""
        if self._img_session is not None:
            return
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.engine_paths import SD15_DIR
        from jasna.mosaic.detection_registry import build_detection_model, coerce_detection_model_name, require_detection_model_weights
        from jasna.restorer.sd15_download import bundle_present
        from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer

        settings = self._settings
        device = torch.device("cuda:0")
        if not bundle_present(SD15_DIR):
            raise FileNotFoundError(
                f"SD 1.5 model not found at {SD15_DIR}. Use 'Download model' in the "
                "Image Restoration settings."
            )

        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = require_detection_model_weights(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=settings.detection_score_threshold,
            fp16=settings.fp16_mode,
        )
        restorer = Sd15InpaintRestorer(SD15_DIR, device, settings.fp16_mode)
        self._img_session = (detector, restorer, device)
        self._log("INFO", "SD 1.5 model loaded (reused across image jobs)")

    def _run_image_job(self, job_id: int, input_path: Path, output_path: Path):
        """Restore a still image with the (shared) SD 1.5 inpaint session."""
        from jasna.image_restore import clamp_strength, restore_image, variant_output_paths
        from jasna.media import image_io
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        self._ensure_image_session()
        detector, restorer, device = self._img_session
        settings = self._settings

        self._pause_event.wait()
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        self._progress(ProgressUpdate(job_id=job_id, status=JobStatus.PROCESSING, progress=20.0, message="Detecting mosaics"))

        num_variants = max(1, int(settings.image_restore_variants))
        freeu = dict(DEFAULT_FREEU) if bool(settings.image_restore_freeu) else None
        strength = clamp_strength(float(settings.image_restore_strength))

        img = image_io.read_image_rgb_chw(input_path)
        outputs = restore_image(
            img, detector, restorer,
            device=device, fp16=settings.fp16_mode,
            steps=int(settings.image_restore_steps),
            strength=strength, seed=int(settings.image_restore_seed),
            num_variants=num_variants, freeu=freeu,
        )
        for path, out in zip(variant_output_paths(output_path, num_variants), outputs):
            image_io.write_image_rgb_chw(path, out)
            self._log("INFO", f"Wrote {path.name}")
        self._progress(ProgressUpdate(job_id=job_id, status=JobStatus.PROCESSING, progress=100.0))

    def _close_image_session(self):
        if self._img_session is None:
            return
        detector, restorer, _ = self._img_session
        self._img_session = None
        detector.close()
        restorer.close()
        import gc
        import torch
        for _ in range(3):
            gc.collect()
        _cleanup_torch(torch)
        self._log("INFO", "SD 1.5 model unloaded")

    def _get_unique_output_path(self, output_path: Path) -> Path:
        """Find a unique output path by adding a counter suffix if file exists."""
        if not output_path.exists():
            return output_path
            
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem} ({counter}){suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
            if counter > 9999:
                raise RuntimeError(f"Could not find unique filename after 9999 attempts: {output_path}")

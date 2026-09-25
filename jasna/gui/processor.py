"""Background processor for video processing jobs.

MODIFIED: adds crash-resume support. If Jasna is killed/crashes mid-job
(power loss, GPU driver crash, process kill, etc.), the next time the same
input file is queued and started, the processor will detect a resume
checkpoint and continue from the last safely-encoded timestamp instead of
re-processing the whole file from frame 0. This works regardless of whether
the remaining footage contains mosaics or not, because it operates purely on
the timeline (seconds already rendered vs. seconds still to render), not on
video content.

How it works
------------
1. While a video job runs, every ~15s of wall-clock time we record how many
   seconds of *output* have been safely encoded so far (with an 8s safety
   margin subtracted, so we never assume the last couple of GOPs made it to
   disk). This is written to a small JSON file next to the output, under
   ``<output_dir>/.jasna_resume/<hash>.json``.
2. If the process crashes, that checkpoint file survives on disk (the
   partially-written output video also survives, since NvidiaVideoEncoder
   writes to it incrementally).
3. Next time this exact job (same input path + same output path) is started,
   ``_run_video_job`` notices the checkpoint, cuts the old (partial) output
   at the safe timestamp with ffmpeg (stream copy, no re-encoding) to keep
   what was already restored, then runs the Pipeline again with
   ``seek_ts=<safe_seconds>`` so it only restores the remaining footage into a
   temporary file, and finally concatenates (stream copy) the kept prefix
   with the freshly rendered remainder into the final output path.
4. On successful completion the checkpoint (and any temp files) are removed.
   If the user explicitly stops the queue (not a crash), the checkpoint is
   *kept* on disk so that pressing Start again later resumes instead of
   restarting — this matches "當機後打開能接續，不用重新來過".
"""

import hashlib
import json
import logging
import shutil
import subprocess
import threading
import traceback
import queue
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable

from jasna.gui.models import JobItem, JobStatus, AppSettings
from jasna.gui.video_session import build_video_session, release_session_memory, video_session_config
from jasna.media import UnsupportedColorspaceError
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Resume checkpoint support
# --------------------------------------------------------------------------

_RESUME_DIR_NAME = ".jasna_resume"
_CHECKPOINT_INTERVAL_SECONDS = 15.0
_SAFETY_MARGIN_SECONDS = 8.0
_MIN_RESUMABLE_SECONDS = 5.0


def _resume_dir_for(output_path: Path) -> Path:
    d = output_path.parent / _RESUME_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_key(input_path: Path, output_path: Path) -> str:
    raw = f"{input_path.resolve(strict=False)}|{output_path.resolve(strict=False)}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _checkpoint_path(input_path: Path, output_path: Path) -> Path:
    return _resume_dir_for(output_path) / f"{_checkpoint_key(input_path, output_path)}.json"


def _partial_output_path(output_path: Path) -> Path:
    """Where we keep the not-yet-finalized output while a job is running."""
    return _resume_dir_for(output_path) / f"{_checkpoint_key(output_path, output_path)}.partial{output_path.suffix}"


class ResumeCheckpoint:
    """Reads/writes the small JSON file that records processing progress."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.path = _checkpoint_path(input_path, output_path)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            logger.warning("Resume checkpoint unreadable, ignoring: %s", self.path, exc_info=True)
            return None
        partial = data.get("partial_output")
        if not partial or not Path(partial).exists():
            return None
        safe_seconds = float(data.get("safe_seconds", 0.0))
        if safe_seconds < _MIN_RESUMABLE_SECONDS:
            return None
        return data

    def write(self, *, safe_seconds: float, partial_output: Path) -> None:
        data = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "partial_output": str(partial_output),
            "safe_seconds": float(safe_seconds),
            "updated_at": time.time(),
        }
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(self.path)
        except OSError:
            logger.warning("Failed to write resume checkpoint: %s", self.path, exc_info=True)

    def clear(self) -> None:
        for p in (self.path,):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                logger.debug("Failed to remove resume checkpoint file: %s", p, exc_info=True)


def _run_ffmpeg(args: list[str]) -> None:
    from jasna.os_utils import ffmpeg_executable_path  # existing helper used elsewhere in Jasna

    exe = None
    try:
        exe = str(ffmpeg_executable_path())
    except Exception:
        exe = "ffmpeg"
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed (resume support): "
            f"{' '.join(cmd)}\n{result.stdout.decode('utf-8', errors='replace')}"
        )


def _probe_duration_seconds(path: Path) -> float:
    from jasna.media import get_video_meta_data

    metadata = get_video_meta_data(str(path))
    return float(metadata.duration)


def _cut_prefix(source: Path, destination: Path, seconds: float) -> None:
    """Stream-copy the first `seconds` of `source` into `destination`, no re-encode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "-i", str(source),
        "-t", f"{seconds:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(destination),
    ])


def _concat_stream_copy(prefix: Path, suffix: Path, destination: Path) -> None:
    """Concatenate two files with matching codecs via stream copy (no re-encode)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.parent / f"{destination.stem}.resume_concat.ffconcat"
    with open(manifest, "w", encoding="utf-8") as f:
        for p in (prefix, suffix):
            escaped = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(manifest),
            "-c", "copy",
            str(destination),
        ])
    finally:
        try:
            manifest.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------


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
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default

        self._jobs: list[JobItem] = []
        self._settings: AppSettings | None = None
        self._output_folder: str = ""
        self._output_pattern: str = "{original}_restored.mp4"
        self._disable_basicvsrpp_tensorrt_for_run = False

        # Heavy models are loaded once and reused across consecutive jobs of the
        # same type; the other session is unloaded when the type switches.
        self._img_session: tuple | None = None  # (detector, restorer, device)
        self._video_session: RestorationSession | None = None
        self._current_pipeline = None

        # True only when a crash is suspected to have been avoided by a clean
        # stop; used to decide whether to keep or clear the resume checkpoint.
        self._current_job_crash_safe = False

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
        snapshot = job.begin_processing()
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

        # Handle file conflict based on settings.
        # NOTE: if a resume checkpoint exists for this exact input+output pair,
        # we must NOT auto-rename/skip — that would orphan the crash checkpoint
        # and force a restart from scratch. Resume takes priority.
        file_conflict = self._settings.file_conflict if self._settings else "auto_rename"
        has_resume = (not is_image) and ResumeCheckpoint(input_path, output_path).load() is not None

        if output_path.exists() and not has_resume:
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

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
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
                self._run_post_export_video_command(input_path, output_path)

            job.output_path = output_path
            job.status = JobStatus.COMPLETED
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
        self._log("INFO", f"Stopped processing {job.filename} (resume checkpoint kept)")

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

        # ---- Resume support: segmented (--segments) jobs are not resumed
        # automatically; only plain full-file jobs are, to keep the smart-
        # render splice-plan logic untouched and low-risk.
        resume_checkpoint = None if segments else ResumeCheckpoint(input_path, output_path)
        resume_data = resume_checkpoint.load() if resume_checkpoint is not None else None

        encoder_settings = self._build_encoder_settings(codec)
        config = video_session_config(settings, codec=codec, encoder_settings=encoder_settings)
        self._ensure_video_session(settings)
        s = self._video_session
        self._prepare_job_detector(config, s)
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")

        if resume_data is not None:
            self._run_video_job_resumed(
                job_id, input_path, output_path, config, s,
                resume_checkpoint=resume_checkpoint, resume_data=resume_data,
            )
        else:
            self._run_video_job_fresh(
                job_id, input_path, output_path, config, s,
                segments=segments, splice_plan=splice_plan,
                resume_checkpoint=resume_checkpoint,
            )

    def _make_progress_callback(
        self,
        job_id: int,
        *,
        resume_checkpoint: "ResumeCheckpoint | None",
        partial_output: Path | None,
        time_offset_seconds: float,
        total_duration_seconds: float,
        last_update_time: list,
        last_checkpoint_time: list,
    ):
        def progress_callback(progress_pct: float, fps: float, eta_seconds: float, frames_done: int, total: int):
            current_time = time.time()
            if current_time - last_update_time[0] < 0.1:
                return
            last_update_time[0] = current_time

            self._pause_event.wait()
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")

            # Combined progress across the already-kept prefix + the part
            # being rendered right now, so the UI still shows 0-100% for the
            # whole file even when resuming partway through.
            if total_duration_seconds > 0:
                rendered_now_seconds = (progress_pct / 100.0) * max(0.0, total_duration_seconds - time_offset_seconds)
                combined_pct = min(100.0, ((time_offset_seconds + rendered_now_seconds) / total_duration_seconds) * 100.0)
            else:
                combined_pct = progress_pct

            self._progress(ProgressUpdate(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=combined_pct,
                fps=fps,
                eta_seconds=eta_seconds,
                frames_processed=frames_done,
                total_frames=total,
            ))

            if resume_checkpoint is not None and partial_output is not None:
                if current_time - last_checkpoint_time[0] >= _CHECKPOINT_INTERVAL_SECONDS:
                    last_checkpoint_time[0] = current_time
                    if total_duration_seconds > 0:
                        rendered_now_seconds = (progress_pct / 100.0) * max(0.0, total_duration_seconds - time_offset_seconds)
                        safe_seconds = time_offset_seconds + rendered_now_seconds - _SAFETY_MARGIN_SECONDS
                        if safe_seconds >= _MIN_RESUMABLE_SECONDS and partial_output.exists():
                            resume_checkpoint.write(safe_seconds=safe_seconds, partial_output=partial_output)

        return progress_callback

    def _run_video_job_fresh(
        self, job_id, input_path, output_path, config, s,
        *, segments, splice_plan, resume_checkpoint,
    ):
        """Normal path: no resume checkpoint found, process the whole file.

        Encodes directly into a `.partial` file under the resume dir so that
        if the process dies mid-job, the partial output + checkpoint survive
        for next time. On success, the partial file is moved to the real
        output path and the checkpoint is cleared.
        """
        use_resume = resume_checkpoint is not None
        target_output = _partial_output_path(output_path) if use_resume else output_path

        total_duration_seconds = 0.0
        if use_resume:
            try:
                total_duration_seconds = _probe_duration_seconds(input_path)
            except Exception:
                logger.warning("Could not probe input duration for resume tracking", exc_info=True)
                use_resume = False
                resume_checkpoint = None
                target_output = output_path

        last_update_time = [0.0]
        last_checkpoint_time = [0.0]
        progress_callback = self._make_progress_callback(
            job_id,
            resume_checkpoint=resume_checkpoint,
            partial_output=target_output if use_resume else None,
            time_offset_seconds=0.0,
            total_duration_seconds=total_duration_seconds,
            last_update_time=last_update_time,
            last_checkpoint_time=last_checkpoint_time,
        )

        pipeline = None
        try:
            pipeline = build_pipeline(
                config,
                s,
                input_path,
                target_output,
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
        finally:
            self._current_pipeline = None
            if pipeline is not None:
                pipeline.close()

        # Completed successfully: promote partial -> real output, clear checkpoint.
        if use_resume:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            shutil.move(str(target_output), str(output_path))
            resume_checkpoint.clear()

    def _run_video_job_resumed(
        self, job_id, input_path, output_path, config, s,
        *, resume_checkpoint: "ResumeCheckpoint", resume_data: dict,
    ):
        """Resume path: a checkpoint says we already safely rendered
        `safe_seconds` of output into `partial_output`. Keep that prefix,
        render only the remainder, then stitch the two together.
        """
        safe_seconds = float(resume_data["safe_seconds"])
        old_partial = Path(resume_data["partial_output"])
        self._log(
            "INFO",
            f"Resuming {input_path.name} from {safe_seconds:.1f}s "
            "(crash/interruption checkpoint found)",
        )

        try:
            total_duration_seconds = _probe_duration_seconds(input_path)
        except Exception as exc:
            self._log("WARNING", f"Could not probe duration for resume, restarting from scratch: {exc}")
            resume_checkpoint.clear()
            self._run_video_job_fresh(
                job_id, input_path, output_path, config, s,
                segments=(), splice_plan=None, resume_checkpoint=ResumeCheckpoint(input_path, output_path),
            )
            return

        kept_prefix = _resume_dir_for(output_path) / f"{_checkpoint_key(input_path, output_path)}.prefix{output_path.suffix}"
        try:
            _cut_prefix(old_partial, kept_prefix, safe_seconds)
        except Exception:
            logger.warning("Failed to cut resume prefix, restarting from scratch", exc_info=True)
            resume_checkpoint.clear()
            self._run_video_job_fresh(
                job_id, input_path, output_path, config, s,
                segments=(), splice_plan=None, resume_checkpoint=ResumeCheckpoint(input_path, output_path),
            )
            return

        remainder_output = _resume_dir_for(output_path) / f"{_checkpoint_key(input_path, output_path)}.remainder{output_path.suffix}"
        new_partial_output = _partial_output_path(output_path)

        last_update_time = [0.0]
        last_checkpoint_time = [0.0]
        progress_callback = self._make_progress_callback(
            job_id,
            resume_checkpoint=resume_checkpoint,
            partial_output=new_partial_output,
            time_offset_seconds=safe_seconds,
            total_duration_seconds=total_duration_seconds,
            last_update_time=last_update_time,
            last_checkpoint_time=last_checkpoint_time,
        )

        pipeline = None
        try:
            pipeline = build_pipeline(
                config,
                s,
                input_path,
                remainder_output,
                progress_callback=progress_callback,
                seek_ts=safe_seconds,
            )
            self._current_pipeline = pipeline
            if self._stop_event.is_set():
                pipeline.cancel()
            pipeline.run()
            if _pipeline_was_stopped(pipeline):
                # Stopped again mid-resume: merge what we have so far into a
                # fresh partial + checkpoint so the *next* resume continues
                # from here rather than losing the kept prefix.
                self._merge_partial_resume(
                    input_path, output_path, kept_prefix, remainder_output,
                    resume_checkpoint, safe_seconds,
                )
                raise ProcessingStopped("Processing stopped")
        finally:
            self._current_pipeline = None
            if pipeline is not None:
                pipeline.close()

        # Remainder finished: stitch kept prefix + freshly rendered remainder.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _concat_stream_copy(kept_prefix, remainder_output, output_path)
        resume_checkpoint.clear()
        for p in (kept_prefix, remainder_output, old_partial, new_partial_output):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    def _merge_partial_resume(
        self, input_path, output_path, kept_prefix, remainder_partial,
        resume_checkpoint: "ResumeCheckpoint", previous_safe_seconds: float,
    ) -> None:
        """Best-effort: if the resumed remainder itself got interrupted again,
        fold whatever of it got rendered back into a new merged partial file
        so the checkpoint chain doesn't reset to zero."""
        if not remainder_partial.exists():
            return
        try:
            remainder_duration = _probe_duration_seconds(remainder_partial)
        except Exception:
            logger.warning("Could not probe interrupted remainder; keeping previous checkpoint as-is", exc_info=True)
            return
        if remainder_duration < _SAFETY_MARGIN_SECONDS:
            return

        safe_remainder_seconds = max(0.0, remainder_duration - _SAFETY_MARGIN_SECONDS)
        trimmed_remainder = _resume_dir_for(output_path) / f"{_checkpoint_key(input_path, output_path)}.remainder_trim{output_path.suffix}"
        merged_partial = _partial_output_path(output_path)
        try:
            _cut_prefix(remainder_partial, trimmed_remainder, safe_remainder_seconds)
            _concat_stream_copy(kept_prefix, trimmed_remainder, merged_partial)
            new_safe_seconds = previous_safe_seconds + safe_remainder_seconds
            resume_checkpoint.write(safe_seconds=new_safe_seconds, partial_output=merged_partial)
            self._log("INFO", f"Checkpoint updated at {new_safe_seconds:.1f}s after stop")
        except Exception:
            logger.warning("Failed to merge partial resume progress", exc_info=True)
        finally:
            try:
                if trimmed_remainder.exists():
                    trimmed_remainder.unlink()
            except OSError:
                pass

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

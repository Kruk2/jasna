"""Background processor for video processing jobs."""

import logging
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable
from uuid import uuid4

from jasna.gui.models import (
    ENCODER_RATE_MODE_AUTO_SOURCE,
    ENCODER_RATE_MODE_MANUAL_CQ,
    ENCODER_RATE_MODES,
    DEFAULT_OUTPUT_PATTERN,
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.output_paths import OutputPathError, job_output_path
from jasna.gui.video_session import build_image_session, build_video_session, release_session_memory, video_session_config
from jasna.media.probe import UnsupportedColorspaceError
from jasna.native_worker import (
    HostMemoryPressureError,
    NATIVE_ENCODE_STALL_EXIT_CODE,
    NATIVE_OPEN_STALL_EXIT_CODE,
    NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
    NativeWorkerRecycleRequested,
)
from jasna.media import media_files
from jasna.media.media_files import unique_path
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline

logger = logging.getLogger(__name__)

_ISOLATED_STOP_GRACE_SECONDS = 5.0
_ISOLATED_TERMINATE_GRACE_SECONDS = 1.0
_ISOLATED_NATIVE_PRESSURE_RECYCLE_LIMIT = 32
_ISOLATED_AMF_SESSION_RECYCLE_LIMIT = 256
_ISOLATED_NATIVE_OPEN_STALL_RETRY_LIMIT = 2
_ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT = 2
# A native HIP failure can abort the child before it can emit the structured
# ``retry`` event.  Keep this separate from pressure/session recycling: the
# workspace may still be resumable, but repeated aborts must fail closed rather
# than loop indefinitely on the same fragment.
_ISOLATED_NATIVE_ABORT_RETRY_LIMIT = 2
_ISOLATED_GPU_RECOVERY_TIMEOUT_SECONDS = 60.0
_ISOLATED_GPU_RECOVERY_POLL_SECONDS = 0.25
_ISOLATED_GPU_RECOVERY_STABLE_SAMPLES = 2
_ISOLATED_NATIVE_ABORT_SIGNALS = frozenset(
    signal_value
    for signal_value in (
        getattr(signal, "SIGABRT", None),
        getattr(signal, "SIGBUS", None),
        getattr(signal, "SIGILL", None),
        getattr(signal, "SIGSEGV", None),
    )
    if signal_value is not None
)
_ISOLATED_FAILURE_DETAIL_MAX_CHARS = 2048
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
    stage: str = ""
    phase: str = ""


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


def build_job_encoder_settings(settings: AppSettings, codec: str) -> dict:
    # Built per job (not cached in the video session) so a codec change
    # between queued jobs is always validated against the selected codec.
    from jasna.accelerator import AcceleratorVendor, vendor_for_device
    from jasna.media.encoder_settings import parse_encoder_settings, validate_encoder_settings
    from jasna.media.encoder_settings import (
        encoder_cq_spec,
        validate_encoder_cq,
    )

    vendor = vendor_for_device()
    rate_mode = settings.encoder_rate_mode
    if rate_mode not in ENCODER_RATE_MODES:
        raise ValueError(f"Unsupported encoder rate mode: {rate_mode!r}")
    if settings.amd_dual_gop_encode:
        if codec != "hevc":
            raise ValueError("Dual AMD GOP encoding requires HEVC output")
        if rate_mode != ENCODER_RATE_MODE_AUTO_SOURCE:
            raise ValueError(
                "Dual AMD GOP encoding requires automatic source-rate control"
            )
    cq = (
        encoder_cq_spec(codec, vendor).default
        if settings.encoder_cq is None
        else settings.encoder_cq
    )
    validate_encoder_cq(cq, codec=codec, vendor=vendor)
    encoder_settings = {} if settings.amd_dual_gop_encode else {"cq": cq}
    if settings.encoder_custom_args:
        from jasna.gui.hardware_policy import split_batch_size_custom_arg

        _batch_size, encoder_custom_args = split_batch_size_custom_arg(
            settings.encoder_custom_args
        )
        custom_settings = parse_encoder_settings(encoder_custom_args)
        if settings.amd_dual_gop_encode:
            conflicting_rate_options = sorted(
                {
                    "cq",
                    "qvbr_quality_level",
                    "rc",
                    "maxrate",
                    "bufsize",
                    "qp_i",
                    "qp_p",
                }
                & custom_settings.keys()
            )
            if conflicting_rate_options:
                raise ValueError(
                    "Dual AMD GOP encoding derives VBR Peak from the source; "
                    "remove custom " + ", ".join(conflicting_rate_options)
                )
            custom_gop = custom_settings.get("g")
            if custom_gop is not None and int(custom_gop) != 250:
                raise ValueError("Dual AMD GOP encoding requires g=250")
            custom_b_frames = custom_settings.get("bf")
            if custom_b_frames is not None and int(custom_b_frames) != 0:
                raise ValueError("Dual AMD GOP encoding requires bf=0")
        if (
            rate_mode == ENCODER_RATE_MODE_MANUAL_CQ
            and vendor is AcceleratorVendor.AMD
            and codec == "hevc"
            and custom_settings.get("rc") not in {None, "cqp", 0, "0"}
        ):
            raise ValueError(
                "Manual CQ for AMD HEVC requires rc=cqp; remove the "
                "conflicting rc from custom encoder settings"
            )
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
    if (
        rate_mode == ENCODER_RATE_MODE_MANUAL_CQ
        and vendor is AcceleratorVendor.AMD
        and codec == "hevc"
    ):
        # An explicit CQP contract disables Linux AMD HEVC automatic
        # source-rate vbr_peak for both full and Smart Render, making the
        # visible CQ slider authoritative. It is also valid on the existing
        # Windows AMD CQP route, without claiming source-rate support there.
        encoder_settings["rc"] = "cqp"
    return validate_encoder_settings(encoder_settings, codec=codec, vendor=vendor)


def _gpu_failure_requires_restart(exc: BaseException) -> bool:
    """Return whether continuing the queue in this GPU process is unsafe."""

    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except (AttributeError, ImportError):
        pass
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "hip out of memory",
            "cuda out of memory",
            "not enough memory for command submission",
        )
    )


def _is_linux_amd_runtime() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


def _bounded_isolated_failure_detail(detail: object) -> str:
    """Return one bounded, display-safe worker diagnostic line."""

    normalized = "".join(
        character if character.isprintable() else " " for character in str(detail)
    )
    normalized = " ".join(normalized.split())
    if len(normalized) > _ISOLATED_FAILURE_DETAIL_MAX_CHARS:
        return normalized[: _ISOLATED_FAILURE_DETAIL_MAX_CHARS - 3] + "..."
    return normalized


def _isolated_video_job_terminal_failure_message(
    returncode: int | None,
    protocol_error: str | None,
) -> str | None:
    """Preserve bounded backend detail alongside an isolated-worker exit code."""

    if returncode != 0:
        message = f"isolated video job exited with code {returncode}"
        if protocol_error is not None:
            detail = _bounded_isolated_failure_detail(protocol_error)
            if detail:
                return f"{message}: {detail}"
        return message
    if protocol_error is not None:
        return "invalid isolated video job protocol: " + _bounded_isolated_failure_detail(
            protocol_error
        )
    return None


class Processor:
    """Handles video processing in a background thread."""
    
    def __init__(
        self,
        on_progress: Callable[[ProgressUpdate], None] = None,
        on_log: Callable[[str, str], None] = None,
        on_complete: Callable[[bool], None] = None,
        *,
        video_job_isolation: str | None = None,
        video_job_attempt_backend=None,
    ):
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        self._video_job_isolation = video_job_isolation
        # Explicit backend injection only; automatic Windows selection remains
        # disabled until runtime packaging/recovery/full-video acceptance.
        self._video_job_attempt_backend = video_job_attempt_backend
        
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._completion_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self._jobs: list[JobItem] = []
        self._settings: AppSettings | None = None
        self._output_folder: str = ""
        self._output_pattern: str = DEFAULT_OUTPUT_PATTERN
        self._preserve_input_structure = False
        self._disable_basicvsrpp_tensorrt_for_run = False

        # Heavy models are loaded once and reused across consecutive jobs of the
        # same type; the other session is unloaded when the type switches.
        self._img_session: tuple | None = None      # (detector, restorer, device)
        self._video_session: RestorationSession | None = None
        self._current_pipeline = None
        self._pre_scan_coordinator = None
        self._current_aux_process: subprocess.Popen | None = None
        self._isolated_process: subprocess.Popen[str] | None = None
        self._isolated_process_lock = threading.Lock()
        self._isolated_stop_reaper: threading.Thread | None = None
        self._completed_processing_paths: dict[int, str] = {}
        # Two folder imports can contain the same relative path.  Reserve each
        # destination before checking the filesystem so the second job is not
        # mistaken for the first job's resumable output.
        self._reserved_output_paths: dict[int, Path] = {}
        self._reserved_output_owners: dict[str, int] = {}
        # A late Smart Render compatibility failure after Linux AMD native GPU
        # work may leave driver-owned allocations alive until process exit.
        # Never run another queued job (or a later Start) in that process.
        self._restart_required_reason: str | None = None
        
    def start(
        self,
        jobs: list[JobItem],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
        *,
        disable_basicvsrpp_tensorrt: bool = False,
        preserve_input_structure: bool = False,
    ):
        if self._restart_required_reason is not None:
            raise RuntimeError(self._restart_required_reason)
        if self._thread and self._thread.is_alive():
            return
            
        self._jobs = jobs
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._preserve_input_structure = bool(preserve_input_structure)
        self._disable_basicvsrpp_tensorrt_for_run = bool(disable_basicvsrpp_tensorrt)
        self._completed_processing_paths.clear()
        self._reserved_output_paths.clear()
        self._reserved_output_owners.clear()

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
        self._send_isolated_command(
            {"command": "set_paused", "paused": self.is_paused()}
        )

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
        coordinator = self._pre_scan_coordinator
        if coordinator is not None:
            coordinator.stop()
        auxiliary = self._current_aux_process
        if auxiliary is not None and auxiliary.poll() is None:
            try:
                auxiliary.terminate()
            except OSError:
                logger.debug("Could not terminate auxiliary media process", exc_info=True)
        self._send_isolated_command({"command": "stop"})
        self._start_isolated_stop_reaper()

    def completed_processing_path(self, job_id: int) -> str | None:
        return self._completed_processing_paths.get(int(job_id))

    def restart_required_reason(self) -> str | None:
        """Return the terminal native-resource reason for this GUI process."""

        return self._restart_required_reason

    def join(self, timeout: float = 5.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive() and self._stop_event.is_set():
                self._terminate_isolated_process()
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    self._terminate_isolated_process(force=True)
                    self._thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def was_stopped(self) -> bool:
        """Return whether Stop was requested for the current or last run."""

        return self._stop_event.is_set()

    def active_job(self) -> JobItem | None:
        """Return the current job for read-only diagnostic context."""

        return next(
            (job for job in self._jobs if job.status is JobStatus.PROCESSING),
            None,
        )
        
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

    def isolated_worker_pid(self) -> int | None:
        """Expose the active child PID for parent-side diagnostics only."""

        with self._isolated_process_lock:
            process = self._isolated_process
            pid = getattr(process, "pid", None) if process is not None else None
        return int(pid) if isinstance(pid, int) else None

    def _send_isolated_command(self, command: dict) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            if process is None or process.poll() is not None or process.stdin is None:
                return
            try:
                process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                logger.debug("Could not send command to isolated video job", exc_info=True)

    def _start_isolated_stop_reaper(self) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            reaper = self._isolated_stop_reaper
            if process is None or (reaper is not None and reaper.is_alive()):
                return

            def reap_stopped_process() -> None:
                try:
                    try:
                        process.wait(timeout=_ISOLATED_STOP_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    # The worker leader may exit while an ffmpeg descendant still
                    # owns stdout. Always retire the dedicated process group.
                    self._terminate_isolated_process(expected=process)
                    time.sleep(_ISOLATED_TERMINATE_GRACE_SECONDS)
                    self._terminate_isolated_process(force=True, expected=process)
                finally:
                    with self._isolated_process_lock:
                        if self._isolated_stop_reaper is threading.current_thread():
                            self._isolated_stop_reaper = None

            self._isolated_stop_reaper = threading.Thread(
                target=reap_stopped_process,
                daemon=True,
                name="isolated-video-job-stop-reaper",
            )
            self._isolated_stop_reaper.start()

    def _terminate_isolated_process(
        self,
        *,
        force: bool = False,
        expected: subprocess.Popen[str] | None = None,
    ) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            if process is None or (expected is not None and process is not expected):
                return
            try:
                if os.name == "posix" and getattr(process, "pid", None) is not None:
                    os.killpg(
                        process.pid,
                        signal.SIGKILL if force else signal.SIGTERM,
                    )
                elif process.poll() is not None:
                    return
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                logger.debug("Could not terminate isolated video job", exc_info=True)

    def _should_isolate_video_job(self, job: JobItem) -> bool:
        if self._video_job_isolation != "linux-amd" or not _is_linux_amd_runtime():
            return False
        return not media_files.is_image(job.path)

    def _wait_for_isolated_gpu_recovery(self) -> bool:
        """Wait for a retired AMD worker's native allocations to disappear."""

        from jasna.gui.gpu_recovery import wait_for_vram_recovery

        if self._stop_event.is_set():
            return False
        recovery_options = dict(
            stop_event=self._stop_event,
            on_log=self._log,
            timeout_seconds=_ISOLATED_GPU_RECOVERY_TIMEOUT_SECONDS,
            poll_seconds=_ISOLATED_GPU_RECOVERY_POLL_SECONDS,
            stable_samples=_ISOLATED_GPU_RECOVERY_STABLE_SAMPLES,
        )
        backend = getattr(self, "_video_job_attempt_backend", None)
        if backend is not None:
            try:
                with backend.open_recovery_reader() as reader:
                    return wait_for_vram_recovery(
                        reader, min_headroom_bytes=1024 ** 3, **recovery_options,
                    )
            except Exception as error:
                self._log("ERROR", f"Cannot observe retired Windows worker GPU memory: {error}"[:2048])
                return False

        from jasna.vram_offloader import (
            AMD_MIN_VRAM_STARTUP_BUDGET,
            read_linux_amd_system_vram,
        )

        return wait_for_vram_recovery(
            read_linux_amd_system_vram,
            min_headroom_bytes=AMD_MIN_VRAM_STARTUP_BUDGET,
            allow_unavailable=True,
            **recovery_options,
        )

    def _final_output_path(self, job: JobItem) -> Path:
        """Resolve the exact final output path for a queued job."""

        input_path = job.path
        output_dir = (
            Path(self._output_folder)
            if self._output_folder
            else input_path.parent
        )
        return job_output_path(
            output_dir,
            input_path,
            self._output_pattern,
            input_root=job.input_root,
            preserve_structure=(
                bool(self._output_folder) and self._preserve_input_structure
            ),
        )

    @staticmethod
    def _output_path_identity(output_path: Path) -> str:
        """Return a platform-normalized key for a prospective output path."""

        return os.path.normcase(
            str(output_path.expanduser().resolve(strict=False))
        )

    def _is_preserved_folder_job(self, job: JobItem) -> bool:
        return (
            job.input_root is not None
            and bool(self._output_folder)
            and self._preserve_input_structure
        )

    def _next_reserved_output_path(self, output_path: Path) -> Path:
        """Find a non-existing, non-reserved sibling of ``output_path``."""

        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        for counter in range(1, 10000):
            candidate = parent / f"{stem} ({counter}){suffix}"
            if (
                not candidate.exists()
                and self._output_path_identity(candidate)
                not in self._reserved_output_owners
            ):
                return candidate
        raise RuntimeError(
            f"Could not find unique filename after 9999 attempts: {output_path}"
        )

    def _reserved_final_output_path(
        self,
        job: JobItem,
        *,
        file_conflict: str,
    ) -> Path:
        """Reserve one final destination for this run's job."""

        existing = self._reserved_output_paths.get(job.id)
        if existing is not None:
            return existing

        output_path = self._final_output_path(job)
        identity = self._output_path_identity(output_path)
        owner = self._reserved_output_owners.get(identity)
        if (
            owner is not None
            and owner != job.id
            and file_conflict == "auto_rename"
            and self._is_preserved_folder_job(job)
        ):
            output_path = self._next_reserved_output_path(output_path)
            identity = self._output_path_identity(output_path)

        self._reserved_output_paths[job.id] = output_path
        self._reserved_output_owners[identity] = job.id
        return output_path

    @staticmethod
    def _is_auto_renamed_output(canonical: Path, output: Path) -> bool:
        if not (
            output.parent == canonical.parent
            and output.suffix == canonical.suffix
            and output.stem.startswith(f"{canonical.stem} (")
            and output.stem.endswith(")")
        ):
            return False
        counter_text = output.stem[len(canonical.stem) + 2 : -1]
        return (
            counter_text.isdigit()
            and str(int(counter_text)) == counter_text
            and 1 <= int(counter_text) <= 9999
        )

    def _snapshot_isolated_output_candidates(
        self,
        canonical_path: Path,
    ) -> dict[Path, _OutputFingerprint]:
        canonical = canonical_path.parent.resolve() / canonical_path.name
        candidates = [canonical]
        try:
            candidates.extend(
                candidate.resolve()
                for candidate in canonical.parent.iterdir()
                if self._is_auto_renamed_output(canonical, candidate)
            )
        except FileNotFoundError:
            pass

        fingerprints = {}
        for candidate in candidates:
            fingerprint = self._output_fingerprint(candidate)
            if fingerprint is not None:
                fingerprints[candidate] = fingerprint
        return fingerprints

    def _validate_isolated_output_path(
        self,
        job: JobItem,
        raw_path: object,
        *,
        file_conflict: str,
        preexisting_outputs: dict[Path, _OutputFingerprint],
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("isolated video job did not report its completed output path")
        output = Path(raw_path).expanduser().resolve()
        canonical_path = self._reserved_output_paths.get(job.id)
        if canonical_path is None:
            canonical_path = self._final_output_path(job)
        canonical_path = canonical_path.expanduser()
        canonical = canonical_path.parent.resolve() / canonical_path.name
        if output.parent != canonical.parent:
            raise ValueError("isolated video job reported an output outside the expected folder")
        if output == canonical and not (
            file_conflict == "auto_rename" and canonical in preexisting_outputs
        ):
            return output
        if (
            file_conflict != "auto_rename"
            or canonical not in preexisting_outputs
            or not self._is_auto_renamed_output(canonical, output)
        ):
            raise ValueError("isolated video job reported an unexpected output filename")
        return output

    def _validate_completed_video_output(
        self,
        input_path: Path,
        output_path: Path,
        *,
        codec: str | None,
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

    def _commit_completed_job(
        self,
        job: JobItem,
        output_path: Path,
        *,
        processing_path: str,
    ) -> None:
        """Commit terminal job fields atomically with respect to ``stop()``."""

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            self._completed_processing_paths[job.id] = processing_path
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

    def _handle_existing_final_output(
        self,
        job: JobItem,
        output_path: Path,
        *,
        file_conflict: str,
        is_image: bool,
        configured_codec: str,
    ) -> str:
        """Return ``process``, ``replace`` or ``skip`` for an existing output."""

        preserved_folder_batch = (
            job.input_root is not None
            and bool(self._output_folder)
            and self._preserve_input_structure
        )
        if not output_path.is_file():
            return "process"
        if file_conflict == "skip":
            action = "skip"
        elif preserved_folder_batch and file_conflict == "auto_rename":
            action = "skip"
            if not is_image:
                from jasna.gui.resume_validation import (
                    ResumeOutputValidationError,
                    validate_resume_video_output,
                )

                try:
                    validate_resume_video_output(
                        job.path,
                        output_path,
                        configured_codec=configured_codec,
                    )
                except ResumeOutputValidationError as exc:
                    self._log(
                        "WARNING",
                        f"Existing output is incomplete; replacing {output_path.name}: {exc}",
                    )
                    action = "replace"
        else:
            return "process"

        # Linearize resume skip/replace decisions with Stop just like the final
        # completion commit. If Stop wins while validation is running, the job
        # remains pending and no output work starts.
        with self._completion_lock:
            stopped = self._stop_event.is_set()
            if not stopped and action == "skip":
                job.status = JobStatus.SKIPPED
        if stopped:
            self._mark_stopped(job)
            return "stopped"
        if action == "replace":
            return action
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.SKIPPED,
            message=f"Output file already exists: {output_path.name}",
        ))
        self._log("WARNING", f"Skipped {job.filename}: output file already exists")
        return action

    def _run(self):
        self._log("INFO", "Processing started")

        try:
            while not self._stop_event.is_set():
                job = self._next_pending_job()
                if job is None:
                    break

                self._process_job(job)
                if self._restart_required_reason is not None:
                    break
                if job.status is JobStatus.PENDING:
                    break  # stopped mid-job; it stays queued for the next run
        finally:
            self._close_image_session()
            self._close_video_session()

        if self._stop_event.is_set():
            self._log("INFO", "Processing stopped by user")
        elif self._restart_required_reason is not None:
            self._log("ERROR", self._restart_required_reason)
        else:
            self._log("INFO", "Processing completed")
        queue_finished = (
            not self._stop_event.is_set()
            and self._restart_required_reason is None
            and all(job.status in {JobStatus.COMPLETED, JobStatus.SKIPPED} for job in self._jobs)
        )
        if self._on_complete:
            self._on_complete(queue_finished)

    def _process_job(self, job: JobItem):
        if self._should_isolate_video_job(job):
            self._process_isolated_video_job(job)
            return
        snapshot = self._begin_job_unless_stopped(job)
        if snapshot is None:
            return
        segments = snapshot.segments
        self._log("INFO", f"Started processing {job.filename}")
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PROCESSING,
            message=f"Starting {job.filename}",
            phase="preparing",
        ))
        
        input_path = job.path
        is_image = media_files.is_image(input_path)
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

        try:
            output_path = self._reserved_final_output_path(
                job,
                file_conflict=job_settings.file_conflict,
            )
        except (OutputPathError, OSError, RuntimeError, ValueError) as error:
            job.status = JobStatus.ERROR
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.ERROR,
                message=str(error),
            ))
            self._log("ERROR", f"Failed to process {job.filename}: {error}")
            return
        
        # Handle file conflict based on settings
        file_conflict = job_settings.file_conflict

        existing_output_action = self._handle_existing_final_output(
            job,
            output_path,
            file_conflict=file_conflict,
            is_image=is_image,
            configured_codec=job_settings.codec,
        )
        if existing_output_action in {"skip", "stopped"}:
            return
        if output_path.exists():
            if (
                file_conflict == "auto_rename"
                and existing_output_action != "replace"
            ):
                output_path = self._next_reserved_output_path(output_path)
                self._reserved_output_paths[job.id] = output_path
                self._reserved_output_owners[self._output_path_identity(output_path)] = job.id
                self._log("INFO", f"Renamed output to {output_path.name} to avoid overwrite")
            # "overwrite" - just proceed and let the file be replaced
        
        try:
            self._create_output_parent_unless_stopped(output_path)
            previous_output_fingerprint = (
                self._output_fingerprint(output_path) if not is_image else None
            )
            processing_path = "smart" if segments else "full"
            automatic_segments = False
            if is_image:
                self._close_video_session()
            else:
                self._close_image_session()
                explicit_segments = bool(segments)
                explicit_full = (
                    snapshot.segment_selection_mode is SegmentSelectionMode.FULL
                )
                should_pre_scan = (
                    not explicit_segments
                    and not explicit_full
                    and str(job_settings.pre_scan_policy).strip().lower() != "off"
                )
                if should_pre_scan:
                    from jasna.gui.pre_scan_routing import (
                        PreScanCoordinator,
                        PreScanFailed,
                        PreScanStopped,
                    )
                    from jasna.media.probe import get_video_meta_data

                    self._close_video_session()
                    coordinator = None
                    try:
                        coordinator = PreScanCoordinator(
                            input_path,
                            output_path,
                            get_video_meta_data(str(input_path)),
                            job_settings,
                            stopped=self._stop_event.is_set,
                            log=self._log,
                            progress=lambda stage, fraction, fps, eta: self._progress(
                                ProgressUpdate(
                                    job_id=job.id,
                                    status=JobStatus.PROCESSING,
                                    progress=min(15.0, max(0.0, fraction * 15.0)),
                                    fps=fps,
                                    eta_seconds=eta,
                                    message="Scanning for mosaic ranges",
                                    phase=f"{stage}_scan",
                                )
                            ),
                        )
                        self._pre_scan_coordinator = coordinator
                        outcome = coordinator.run()
                    except PreScanStopped as exc:
                        raise ProcessingStopped("Processing stopped") from exc
                    except PreScanFailed as exc:
                        if str(job_settings.pre_scan_policy).strip().lower() != "auto":
                            raise
                        self._log(
                            "WARNING",
                            f"Automatic scan failed; falling back to full processing: {exc}",
                        )
                        outcome = None
                    finally:
                        self._pre_scan_coordinator = None
                        if coordinator is not None:
                            coordinator.close()
                    if outcome is not None:
                        processing_path = outcome.processing_path
                        segments = outcome.segments
                        automatic_segments = processing_path == "smart"

            if processing_path == "copy":
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.PROCESSING,
                    progress=15.0,
                    phase="source_copy",
                ))
                try:
                    self._copy_source_video(input_path, output_path)
                except ProcessingStopped:
                    raise
                except Exception as exc:
                    if str(job_settings.pre_scan_policy).strip().lower() != "auto":
                        raise
                    self._log(
                        "WARNING",
                        f"Source copy failed; falling back to full processing: {exc}",
                    )
                    processing_path = "full"
                    segments = ()
            if processing_path != "copy":
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.PROCESSING,
                    phase="restoring",
                ))
                pipeline_options = {}
                if automatic_segments:
                    pipeline_options["automatic_segments"] = True
                if segments:
                    pipeline_options["segments"] = segments
                if job_settings is not self._settings:
                    pipeline_options["settings"] = job_settings
                actual_path = self._run_pipeline(
                    job.id,
                    input_path,
                    output_path,
                    **pipeline_options,
                )
                if actual_path in {"full", "smart"}:
                    processing_path = actual_path
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.PROCESSING,
                progress=99.9,
                phase="finalizing",
            ))
            if not is_image:
                self._validate_completed_video_output(
                    input_path,
                    output_path,
                    codec=(None if processing_path == "copy" else job_settings.codec),
                    smart_render=(processing_path == "smart"),
                    previous_fingerprint=previous_output_fingerprint,
                )

            if not is_image:
                self._run_post_export_video_command(input_path, output_path)
            self._commit_completed_job(
                job,
                output_path,
                processing_path=processing_path,
            )
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

        except NativeWorkerRecycleRequested:
            # The isolated Linux AMD worker is the native-resource recovery
            # boundary. Let its protocol entry point request a fresh child;
            # marking the job failed here would discard automatic resume.
            raise

        except HostMemoryPressureError:
            # Preserve the structured pre-OOM reason for the isolated-worker
            # protocol.  Swallowing it here would turn a controlled host-RAM
            # shutdown into an unclassified result=error event.
            raise

        except Exception as e:
            tb = traceback.format_exc()
            e.__traceback__ = None
            if _gpu_failure_requires_restart(e):
                self._restart_required_reason = (
                    "GPU memory was exhausted. The remaining queue was not "
                    "started because native decoder/encoder resources may no "
                    "longer be safe to reuse in this process. Close and restart "
                    "Jasna before trying again."
                )
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

    def _fail_isolated_video_job(self, job: JobItem, message: str) -> None:
        job.status = JobStatus.ERROR
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.ERROR,
            message=message,
        ))
        self._log("ERROR", f"Failed to process {job.filename}: {message}")

    def _apply_isolated_event(
        self,
        job: JobItem,
        event: dict,
        *,
        resume_state: dict[str, object] | None = None,
    ) -> dict | bool:
        event_type = event.get("type")
        if event_type == "log":
            if resume_state is not None and bool(
                resume_state.get("quiet_resume", False)
            ):
                # Re-opening an isolated worker is an internal native-resource
                # boundary.  Hide repeated model/scan/session setup while the
                # durable Smart Render workspace catches back up; fatal events
                # and the parent-side recovery outcome remain visible.
                level = str(event.get("level", "INFO")).upper()
                if level not in {"ERROR", "CRITICAL"}:
                    return False
            self._log(
                str(event.get("level", "INFO")),
                str(event.get("message", "")),
            )
            return False
        if event_type == "progress":
            raw = event["update"]
            status = JobStatus(raw["status"])
            # Completion is authoritative only after parent-side path and
            # freshness validation plus the Stop/commit ordering gate.
            if status is JobStatus.COMPLETED:
                status = JobStatus.PROCESSING
            progress = min(99.9, float(raw.get("progress", 0.0)))
            phase = str(raw.get("phase", ""))
            if resume_state is not None and status is JobStatus.PROCESSING:
                high_water = float(resume_state.get("progress_high_water", 0.0))
                quiet_resume = bool(resume_state.get("quiet_resume", False))
                if quiet_resume:
                    if phase not in {"restoring", "finalizing"}:
                        return False
                    if progress <= high_water and phase != "finalizing":
                        return False
                    resume_state["quiet_resume"] = False
                    resume_state["frame_speed_warmup"] = True
                # Scan and restoration percentages describe different stages.
                # Only restoration/finalization progress is comparable across
                # isolated-worker resumes; treating the scan's 0-15% display
                # as a global high-water mark leaves a live restoration job
                # mislabeled as scanning until it exceeds 15%.
                if phase in {"restoring", "finalizing"}:
                    resume_state["progress_high_water"] = max(
                        high_water,
                        progress,
                    )
            fps = float(raw.get("fps", 0.0))
            eta_seconds = float(raw.get("eta_seconds", 0.0))
            frames_done = int(raw.get("frames_processed", 0))
            total_frames = int(raw.get("total_frames", 0))
            stage = str(raw.get("stage", ""))
            if (
                resume_state is not None
                and status is JobStatus.PROCESSING
                and phase == "restoring"
                and not stage
                and total_frames > 0
            ):
                if fps > 0:
                    resume_state["last_frame_fps"] = fps
                    resume_state["frame_speed_warmup"] = False
                elif bool(resume_state.get("frame_speed_warmup", False)):
                    last_fps = float(resume_state.get("last_frame_fps", 0.0))
                    if last_fps > 0:
                        # Keep an estimate while this fresh worker collects
                        # real timing samples. Never carry FPS into LTX stages
                        # or another queued job, nor replace a new valid sample.
                        fps = last_fps
                        eta_seconds = max(0, total_frames - frames_done) / fps
            update = ProgressUpdate(
                job_id=job.id,
                status=status,
                progress=progress,
                fps=fps,
                eta_seconds=eta_seconds,
                frames_processed=frames_done,
                total_frames=total_frames,
                message=str(raw.get("message", "")),
                stage=stage,
                phase=phase,
            )
            job.status = status
            self._progress(update)
            return False
        if event_type == "fatal":
            detail = str(event.get("message", "isolated video job failed"))
            reason = str(event.get("reason", "")).strip()
            if reason:
                detail = f"[{reason}] {detail}"
            child_traceback = str(event.get("traceback", "")).strip()
            if child_traceback:
                detail += "\n" + child_traceback
            self._log("ERROR", f"Isolated video job failed: {detail}")
            return False
        if event_type in {"result", "retry"}:
            return event
        return False

    def _run_isolated_video_job_attempt(
        self,
        job: JobItem,
        *,
        command: list[str],
        environment: dict[str, str],
        resume_state: dict[str, object] | None = None,
    ) -> tuple[dict | None, str | None, int]:
        """Run one child attempt and return its terminal protocol state."""

        backend = getattr(self, "_video_job_attempt_backend", None)
        if backend is not None:
            def started(process):
                with self._isolated_process_lock:
                    self._isolated_process = process
                if self.is_paused():
                    self._send_isolated_command({"command": "set_paused", "paused": True})
                if self._stop_event.is_set():
                    self._send_isolated_command({"command": "stop"})
                    self._start_isolated_stop_reaper()

            def finished(process):
                with self._isolated_process_lock:
                    if self._isolated_process is process:
                        self._isolated_process = None

            return backend.run(command, environment,
                on_event=lambda event: self._apply_isolated_event(job, event),
                on_log=self._log, on_started=started, on_finished=finished)

        from jasna.gui.video_job_process import parse_event_line

        terminal_event: dict | None = None
        protocol_error: str | None = None
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            with self._isolated_process_lock:
                self._isolated_process = process
            if self.is_paused():
                self._send_isolated_command(
                    {"command": "set_paused", "paused": True}
                )
            if self._stop_event.is_set():
                self._send_isolated_command({"command": "stop"})
                self._start_isolated_stop_reaper()

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                try:
                    event = parse_event_line(line)
                    if event is None:
                        if line and not bool(
                            resume_state is not None
                            and resume_state.get("quiet_resume", False)
                        ):
                            self._log("WARNING", f"[video worker] {line}")
                        continue
                    applied = self._apply_isolated_event(
                        job,
                        event,
                        resume_state=resume_state,
                    )
                    if isinstance(applied, dict):
                        if terminal_event is not None:
                            raise ValueError(
                                "isolated video job emitted multiple terminal events"
                            )
                        terminal_event = applied
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    protocol_error = str(error)
                    self._log(
                        "ERROR",
                        f"Invalid isolated video job event: {error}: {line}",
                    )
            return terminal_event, protocol_error, process.wait()
        finally:
            with self._isolated_process_lock:
                if self._isolated_process is process:
                    self._isolated_process = None
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is None:
                        continue
                    try:
                        stream.close()
                    except (BrokenPipeError, OSError, ValueError):
                        logger.debug(
                            "Could not close isolated video job pipe",
                            exc_info=True,
                        )

    def _validate_isolated_completed_output(
        self,
        input_path: Path,
        output_path: Path,
        *,
        processing_path: str,
        codec: str,
        previous_fingerprint: _OutputFingerprint | None,
    ) -> None:
        from jasna.media.splice import validate_video_output

        self._require_completed_output_changed(output_path, previous_fingerprint)
        validate_video_output(
            output_path,
            source=input_path,
            expected_codec=(codec if processing_path == "full" else None),
        )

    def _process_isolated_video_job(self, job: JobItem) -> None:
        snapshot = self._begin_job_unless_stopped(job)
        if snapshot is None:
            return
        if self._stop_event.is_set():
            self._mark_stopped(job)
            return

        settings = self._settings
        if settings is None:
            self._fail_isolated_video_job(job, "processor settings are unavailable")
            return
        try:
            canonical_output = self._reserved_final_output_path(
                job,
                file_conflict=(self._settings.file_conflict if self._settings else "auto_rename"),
            )
        except (OutputPathError, OSError, RuntimeError, ValueError) as error:
            self._fail_isolated_video_job(job, str(error))
            return
        try:
            preexisting_outputs = self._snapshot_isolated_output_candidates(
                canonical_output
            )
        except OSError as error:
            self._fail_isolated_video_job(
                job,
                f"could not inspect the output folder: {error}",
            )
            return

        existing_output_action = self._handle_existing_final_output(
            job,
            canonical_output,
            file_conflict=settings.file_conflict,
            is_image=False,
            configured_codec=settings.codec,
        )
        if existing_output_action in {"skip", "stopped"}:
            return
        worker_settings = settings
        if existing_output_action == "replace":
            worker_settings = replace(settings, file_conflict="overwrite")

        try:
            # An image session must never occupy GUI-process VRAM while the
            # isolated video worker owns the GPU.
            self._close_image_session()
        except Exception as error:
            self._fail_isolated_video_job(job, str(error))
            return
        if self._stop_event.is_set():
            self._mark_stopped(job)
            return

        from jasna.gui.video_job_process import (
            build_video_job_request,
            video_job_command,
            write_video_job_request,
        )

        result_event: dict | None = None
        protocol_error: str | None = None
        returncode: int | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="jasna-video-job-") as temporary:
                request_path = Path(temporary) / "request.json"
                request = build_video_job_request(
                    job,
                    snapshot,
                    worker_settings,
                    # Pass the already-resolved per-job destination to the
                    # child. This keeps preserved subfolders authoritative in
                    # the GUI parent and avoids duplicating folder-root state
                    # in the isolated-worker protocol.
                    output_folder=str(canonical_output.parent),
                    output_pattern=canonical_output.name,
                    disable_basicvsrpp_tensorrt=(
                        self._disable_basicvsrpp_tensorrt_for_run
                    ),
                )
                write_video_job_request(request_path, request)
                environment = os.environ.copy()
                environment.pop("JASNA_MAIN_PID", None)
                environment["PYTHONUNBUFFERED"] = "1"
                backend = getattr(self, "_video_job_attempt_backend", None)
                if backend is None:
                    command = video_job_command(request_path)
                else:
                    command, environment = backend.prepare_request(request_path, environment)
                pressure_recycles = 0
                amf_session_recycles = 0
                open_stall_retries = 0
                encode_stall_retries = 0
                native_abort_retries = 0
                resume_state: dict[str, object] = {
                    "progress_high_water": 0.0,
                    "quiet_resume": False,
                }
                while True:
                    terminal_event, protocol_error, returncode = (
                        self._run_isolated_video_job_attempt(
                            job,
                            command=command,
                            environment=environment,
                            resume_state=resume_state,
                        )
                    )
                    if self._stop_event.is_set() and terminal_event is None:
                        self._mark_stopped(job)
                        return

                    if returncode == NATIVE_PRESSURE_RECYCLE_EXIT_CODE:
                        retry_reason = (
                            str(terminal_event.get("reason", "")).strip()
                            if terminal_event is not None
                            and terminal_event.get("type") == "retry"
                            else ""
                        )
                        session_limit_recycle = retry_reason == "amf_session_limit"
                        if session_limit_recycle:
                            # A completed fragment is durable forward progress.
                            # A later fragment therefore starts a new stall
                            # failure chain instead of inheriting retries from
                            # an already completed encoder session.
                            encode_stall_retries = 0
                            amf_session_recycles += 1
                            recycle_count = amf_session_recycles
                            recycle_limit = _ISOLATED_AMF_SESSION_RECYCLE_LIMIT
                        else:
                            pressure_recycles += 1
                            recycle_count = pressure_recycles
                            recycle_limit = _ISOLATED_NATIVE_PRESSURE_RECYCLE_LIMIT
                        if recycle_count > recycle_limit:
                            category = (
                                "bounded AMF sessions"
                                if session_limit_recycle
                                else "native GPU pressure"
                            )
                            self._fail_isolated_video_job(
                                job,
                                f"{category} required too many worker recycles "
                                "while resuming Smart Render",
                            )
                            return
                        reason = (
                            str(terminal_event.get("message", "")).strip()
                            if terminal_event is not None
                            and terminal_event.get("type") == "retry"
                            else "native GPU pressure was reported"
                        )
                        if not session_limit_recycle:
                            self._log(
                                "WARNING",
                                f"{reason}; starting a fresh isolated worker and "
                                "reusing completed Smart Render fragments "
                                f"(recycle {recycle_count}/{recycle_limit})",
                            )
                        if not self._wait_for_isolated_gpu_recovery():
                            if self._stop_event.is_set():
                                self._mark_stopped(job)
                            else:
                                self._fail_isolated_video_job(
                                    job,
                                    "GPU memory did not recover enough to resume "
                                    "the isolated video job",
                                )
                            return
                        resume_state["quiet_resume"] = True
                        continue

                    if returncode == NATIVE_OPEN_STALL_EXIT_CODE:
                        open_stall_retries += 1
                        if (
                            open_stall_retries
                            > _ISOLATED_NATIVE_OPEN_STALL_RETRY_LIMIT
                        ):
                            self._fail_isolated_video_job(
                                job,
                                "AMF decoder open stalled repeatedly after "
                                f"{_ISOLATED_NATIVE_OPEN_STALL_RETRY_LIMIT} "
                                "fresh-worker retries",
                            )
                            return
                        self._log(
                            "WARNING",
                            "AMF decoder open timed out; starting a fresh "
                            "isolated worker and resuming completed Smart Render "
                            f"fragments (retry {open_stall_retries}/"
                            f"{_ISOLATED_NATIVE_OPEN_STALL_RETRY_LIMIT})",
                        )
                        if not self._wait_for_isolated_gpu_recovery():
                            if self._stop_event.is_set():
                                self._mark_stopped(job)
                            else:
                                self._fail_isolated_video_job(
                                    job,
                                    "GPU memory did not recover enough after the "
                                    "stalled AMF decoder worker exited",
                                )
                            return
                        resume_state["quiet_resume"] = True
                        continue

                    if returncode == NATIVE_ENCODE_STALL_EXIT_CODE:
                        encode_stall_retries += 1
                        if (
                            encode_stall_retries
                            > _ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT
                        ):
                            self._fail_isolated_video_job(
                                job,
                                "AMF encoder stalled repeatedly after "
                                f"{_ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT} "
                                "fresh-worker retries; preserving the resumable "
                                "Smart Render workspace",
                            )
                            return
                        self._log(
                            "WARNING",
                            "AMF encoder stopped producing frames; starting a "
                            "fresh isolated worker and resuming completed Smart "
                            f"Render fragments (retry {encode_stall_retries}/"
                            f"{_ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT})",
                        )
                        if not self._wait_for_isolated_gpu_recovery():
                            if self._stop_event.is_set():
                                self._mark_stopped(job)
                            else:
                                self._fail_isolated_video_job(
                                    job,
                                    "GPU memory did not recover enough after the "
                                    "stalled AMF encoder worker exited",
                                )
                            return
                        resume_state["quiet_resume"] = True
                        continue

                    # HIP/MIGraphX failures raised from native code can abort
                    # the isolated child (typically SIGABRT from c10) before
                    # ``run_video_job_file`` can emit a structured retry.  A
                    # fresh child is the only safe recovery boundary because
                    # the crashed process may still own poisoned native
                    # streams or AMF/Vulkan surfaces.  Restrict retries to
                    # signals that identify a native fault; SIGTERM/SIGKILL
                    # remain terminal/stop semantics and must not be retried.
                    if (
                        returncode is not None
                        and returncode < 0
                        and -returncode in _ISOLATED_NATIVE_ABORT_SIGNALS
                    ):
                        native_abort_retries += 1
                        if native_abort_retries > _ISOLATED_NATIVE_ABORT_RETRY_LIMIT:
                            self._restart_required_reason = (
                                "A Linux AMD isolated video worker aborted repeatedly "
                                "in native GPU code. The remaining queue was not "
                                "started because the device may no longer be safe "
                                "to reuse in this process. Close and restart Jasna "
                                "before trying again."
                            )
                            self._fail_isolated_video_job(
                                job,
                                "isolated video worker aborted repeatedly after "
                                f"{_ISOLATED_NATIVE_ABORT_RETRY_LIMIT} fresh-worker "
                                "retries; preserving the resumable workspace",
                            )
                            return
                        try:
                            signal_name = signal.Signals(-returncode).name
                        except ValueError:  # pragma: no cover - defensive
                            signal_name = f"signal {-returncode}"
                        self._log(
                            "WARNING",
                            "isolated video worker terminated by native fault "
                            f"({signal_name}); starting a fresh worker and "
                            "resuming completed Smart Render fragments "
                            f"(retry {native_abort_retries}/"
                            f"{_ISOLATED_NATIVE_ABORT_RETRY_LIMIT})",
                        )
                        if not self._wait_for_isolated_gpu_recovery():
                            if self._stop_event.is_set():
                                self._mark_stopped(job)
                            else:
                                self._restart_required_reason = (
                                    "GPU memory did not recover after a Linux AMD "
                                    "native worker abort. Close and restart Jasna "
                                    "before trying the remaining queue."
                                )
                                self._fail_isolated_video_job(
                                    job,
                                    "GPU memory did not recover after the native "
                                    "worker abort",
                                )
                            return
                        resume_state["quiet_resume"] = True
                        continue

                    if terminal_event is not None and terminal_event.get("type") == "result":
                        result_event = terminal_event
                    elif terminal_event is not None:
                        protocol_error = (
                            "isolated video job emitted a retry request without "
                            "the matching recovery exit code"
                        )
                    break
        except Exception as error:
            if self._stop_event.is_set():
                self._mark_stopped(job)
            else:
                self._fail_isolated_video_job(job, str(error))
            return

        if self._stop_event.is_set() and result_event is None:
            self._mark_stopped(job)
        elif (
            terminal_failure := _isolated_video_job_terminal_failure_message(
                returncode,
                protocol_error,
            )
        ) is not None:
            self._fail_isolated_video_job(
                job,
                terminal_failure,
            )
        elif result_event is None:
            self._fail_isolated_video_job(
                job,
                "isolated video job exited without a final result",
            )
        else:
            try:
                status = JobStatus(result_event["status"])
            except (KeyError, TypeError, ValueError) as error:
                self._fail_isolated_video_job(
                    job,
                    f"invalid isolated video job result: {error}",
                )
                return
            if status is not JobStatus.COMPLETED:
                job.status = status
                return
            try:
                processing_path = str(result_event["processing_path"])
                if processing_path not in {"copy", "full", "smart"}:
                    raise ValueError(
                        f"unexpected processing path: {processing_path!r}"
                    )
                output_path = self._validate_isolated_output_path(
                    job,
                    result_event.get("output_path"),
                    file_conflict=worker_settings.file_conflict,
                    preexisting_outputs=preexisting_outputs,
                )
                self._validate_isolated_completed_output(
                    job.path,
                    output_path,
                    processing_path=processing_path,
                    codec=worker_settings.codec,
                    previous_fingerprint=preexisting_outputs.get(output_path),
                )
                self._commit_completed_job(
                    job,
                    output_path,
                    processing_path=processing_path,
                )
            except ProcessingStopped:
                self._mark_stopped(job)
                return
            except Exception as error:
                self._fail_isolated_video_job(
                    job,
                    f"completed output validation failed: {error}",
                )
                return
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                progress=100.0,
            ))

    def _run_post_export_video_command(self, input_path: Path, output_path: Path) -> None:
        command = self._settings.post_export_video_command.strip()
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

    def _copy_source_video(self, input_path: Path, output_path: Path) -> None:
        """Atomically remux an all-clear scan result without decoding frames."""

        from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

        temporary = output_path.with_name(
            f".{output_path.stem}.source-copy-{os.getpid()}{output_path.suffix}"
        )
        temporary.unlink(missing_ok=True)
        args = [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c",
            "copy",
        ]
        if output_path.suffix.lower() in {".mp4", ".mov"}:
            args += ["-movflags", "+faststart"]
        args += [str(temporary), "-y"]
        self._log("INFO", "No mosaic ranges detected; copying the source video")
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **subprocess_no_window_kwargs(),
        )
        self._current_aux_process = process
        try:
            while process.poll() is None:
                if self._stop_event.wait(0.1):
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ProcessingStopped("Processing stopped")
            detail = process.stderr.read().strip() if process.stderr is not None else ""
            if process.returncode != 0:
                raise RuntimeError(
                    f"source copy failed with code {process.returncode}: "
                    f"{detail or 'unknown ffmpeg error'}"
                )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError("source copy did not produce a non-empty output")
            os.replace(temporary, output_path)
        finally:
            self._current_aux_process = None
            if process.stderr is not None:
                process.stderr.close()
            temporary.unlink(missing_ok=True)

    def _run_pipeline(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
        automatic_segments: bool = False,
    ):
        """Run one job; raises ProcessingStopped when the user stopped it."""
        if media_files.is_image(input_path):
            self._run_image_job(job_id, input_path, output_path)
            return "full"
        return self._run_video_job(
            job_id,
            input_path,
            output_path,
            segments=segments,
            settings=settings or self._settings,
            automatic_segments=automatic_segments,
        )

    def _ensure_video_session(self, settings: AppSettings):
        """Compile engines + build the BasicVSR++ (and optional secondary) restorer
        once; reused across consecutive video jobs."""
        if self._video_session is not None:
            return
        self._video_session = build_video_session(
            settings,
            log=lambda msg: self._log("INFO", msg),
        )
        self._log("INFO", "Restoration models loaded (reused across video jobs)")

    def _build_encoder_settings(self, codec: str, *, settings: AppSettings | None = None) -> dict:
        return build_job_encoder_settings(settings or self._settings, codec)

    def _run_video_job(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
        automatic_segments: bool = False,
    ):
        settings = settings or self._settings
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        if settings.amd_dual_gop_encode:
            from jasna.accelerator import AcceleratorVendor, vendor_for_device

            if sys.platform != "linux":
                raise ValueError("Dual AMD GOP encoding is supported only on Linux")
            if vendor_for_device() is not AcceleratorVendor.AMD:
                raise ValueError("Dual AMD GOP encoding requires an AMD GPU")
            if settings.codec != "hevc":
                raise ValueError("Dual AMD GOP encoding requires HEVC output")
            if settings.encoder_rate_mode != ENCODER_RATE_MODE_AUTO_SOURCE:
                raise ValueError(
                    "Dual AMD GOP encoding requires automatic source-rate control"
                )
            if settings.retarget_high_fps:
                raise ValueError(
                    "Dual AMD GOP encoding cannot reduce 60 FPS to 30 FPS"
                )
            if settings.fmp4:
                raise ValueError("Dual AMD GOP encoding cannot use fMP4 output")
        codec = settings.codec
        metadata = None
        splice_plan = None
        full_effect_ranges = None
        native_linux_amd_smart = False
        if segments:
            from jasna.accelerator import vendor_for_device
            from jasna.media.probe import get_video_meta_data
            from jasna.media.splice import (
                SmartRenderCompatibilityError,
                build_splice_plan,
                probe_keyframes,
                resolve_smart_encoder_settings,
                validate_smart_render,
            )
            from jasna.media.video_decoder import auto_amf_interop_eligible

            metadata = get_video_meta_data(str(input_path))
            native_linux_amd_smart = auto_amf_interop_eligible(
                metadata,
                vendor_for_device(),
            )
            codec = {
                "avc": "h264",
                "h265": "hevc",
                "av01": "av1",
            }.get(metadata.codec_name.lower(), metadata.codec_name.lower())
            try:
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
                if automatic_segments and codec == "h264":
                    # Automatic routing must reject source-GOP contracts that
                    # the selected hardware encoder cannot reproduce before
                    # loading restoration models or starting native GPU work.
                    # The pipeline resolves these settings again for the real
                    # render; this early call exists only to make an automatic
                    # Smart Render decision safely fall back to Full.
                    resolve_smart_encoder_settings(
                        codec,
                        metadata,
                        splice_plan.index,
                        {},
                        vendor=vendor_for_device(),
                    )
            except SmartRenderCompatibilityError as exc:
                if not automatic_segments:
                    raise
                self._log(
                    "WARNING",
                    f"Automatic scan ranges are not Smart Render compatible; "
                    f"falling back to full-video encoding with only the "
                    f"scanned ranges restored: {exc}",
                )
                if splice_plan is not None:
                    full_effect_ranges = tuple(
                        effect_range
                        for span in splice_plan.render_spans
                        for effect_range in span.effect_ranges
                    ) or None
                segments = ()
                splice_plan = None
                codec = settings.codec
        if settings.amd_dual_gop_encode:
            from jasna.media.probe import get_video_meta_data
            from jasna.media.dual_gop_encoder import (
                amd_dual_gop_metadata_eligible,
            )

            if metadata is None:
                metadata = get_video_meta_data(str(input_path))
            dual_gop_eligible = (
                codec == "hevc"
                and amd_dual_gop_metadata_eligible(
                    metadata,
                    smart_fragment=bool(segments),
                )
            )
            if not dual_gop_eligible:
                self._log(
                    "INFO",
                    "Parallel GOP encoding is not beneficial or compatible "
                    "for this output; using the established single-session path",
                )
                settings = replace(settings, amd_dual_gop_encode=False)
        encoder_settings = self._build_encoder_settings(codec, settings=settings)
        config = video_session_config(settings, codec=codec, encoder_settings=encoder_settings)
        self._ensure_video_session(settings)
        s = self._video_session
        self._prepare_job_detector(config, s)
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        last_update_time = [0.0]

        def progress_callback(
            progress_pct: float, fps: float, eta_seconds: float, frames_done: int, total: int, stage: str
        ):
            current_time = time.time()
            if current_time - last_update_time[0] < 0.1:
                return
            last_update_time[0] = current_time

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
                stage=stage,
                phase="restoring",
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
                # The bounded Linux AMD route uses a UUID-suffixed staging
                # output for atomic publication.  Workspace resume must stay
                # keyed to the user's stable destination across worker
                # retries, not to that per-attempt staging filename.
                workspace_output=output_path,
                progress_callback=progress_callback,
                segments=tuple(segments) or None,
                splice_plan=splice_plan,
                effect_ranges=full_effect_ranges,
            )
            self._current_pipeline = pipeline
            if self._stop_event.is_set():
                pipeline.cancel()
            try:
                pipeline.run()
            except Exception as exc:
                if segments and native_linux_amd_smart:
                    from jasna.media.splice import SmartRenderCompatibilityError

                    if isinstance(exc, SmartRenderCompatibilityError):
                        self._restart_required_reason = (
                            "Smart Render became incompatible after native Linux AMD "
                            "GPU processing. The remaining queue was not started to "
                            "avoid reusing driver-owned GPU resources. Close and "
                            "restart Jasna, then choose Full video for this file."
                        )
                raise
            if _pipeline_was_stopped(pipeline):
                raise ProcessingStopped("Processing stopped")
            if full_render_staging is not None:
                self._publish_full_render_unless_stopped(
                    full_render_staging,
                    output_path,
                    input_path=input_path,
                    codec=codec,
                )
            return "smart" if segments else "full"
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
        from jasna.native_worker import is_isolated_video_job

        self._log("DEBUG" if is_isolated_video_job() else "INFO", "Restoration models unloaded")

    def _ensure_image_session(self):
        """Load the rf-detr detector + SD 1.5 restorer once; reused across image jobs."""
        if self._img_session is not None:
            return
        self._img_session = build_image_session(
            self._settings,
            log=lambda msg: self._log("INFO", msg),
        )
        self._log("INFO", "SD 1.5 model loaded (reused across image jobs)")

    def _run_image_job(self, job_id: int, input_path: Path, output_path: Path):
        """Restore a still image with the (shared) SD 1.5 inpaint session."""
        from jasna.image_restore import clamp_strength, restore_image, variant_output_paths
        from jasna.media import image_io
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        self._ensure_image_session()
        detector, restorer, device = self._img_session
        settings = self._settings

        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        self._progress(ProgressUpdate(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=20.0,
            message="Detecting mosaics",
            phase="restoring",
        ))

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
        self._progress(ProgressUpdate(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=100.0,
            phase="finalizing",
        ))

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

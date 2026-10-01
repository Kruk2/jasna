"""Protocol and entry point for one isolated GUI video job.

Linux AMD GUI runs use this worker boundary so ROCm/AMF native allocations
from pre-scan, restoration, and encoding die with the per-video process.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import IO, Any

from jasna._frozen import is_frozen
from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobProcessingSnapshot,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.native_worker import (
    ISOLATED_VIDEO_JOB_ENV,
    HostMemoryPressureError,
    NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
    NativeWorkerRecycleRequested,
)
from jasna.segments import SegmentRange


EVENT_PREFIX = "JASNA_JOB_EVENT\t"
REQUEST_SCHEMA_VERSION = 1
_EVENT_WRITE_LOCK = threading.Lock()


def build_video_job_request(
    job: JobItem,
    snapshot: JobProcessingSnapshot,
    settings: AppSettings,
    *,
    output_folder: str,
    output_pattern: str,
    disable_basicvsrpp_tensorrt: bool,
) -> dict[str, Any]:
    """Return a complete, JSON-safe snapshot for one queued video."""

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "job": {
            "id": job.id,
            "path": str(job.path),
            "duration_seconds": job.duration_seconds,
            "segments": [asdict(segment) for segment in snapshot.segments],
            "segment_selection_mode": snapshot.segment_selection_mode.value,
            "detection_model": snapshot.detection_model,
            "detection_score_threshold": snapshot.detection_score_threshold,
            "vr_projection": snapshot.vr_projection,
        },
        "settings": asdict(settings),
        "output_folder": output_folder,
        "output_pattern": output_pattern,
        "disable_basicvsrpp_tensorrt": bool(disable_basicvsrpp_tensorrt),
    }


def write_video_job_request(path: Path, request: dict[str, Any]) -> None:
    path.write_text(json.dumps(request, ensure_ascii=True), encoding="utf-8")


def video_job_command(request_path: Path) -> list[str]:
    if is_frozen():
        return [sys.executable, "--isolated-video-job", str(request_path)]
    return [sys.executable, "-m", "jasna.gui.video_job_process", str(request_path)]


def parse_event_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    payload = json.loads(line[len(EVENT_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("isolated video job event must be a JSON object")
    return payload


def _emit_event(stream: IO[str], event: dict[str, Any]) -> None:
    encoded = EVENT_PREFIX + json.dumps(event, ensure_ascii=True, separators=(",", ":"))
    with _EVENT_WRITE_LOCK:
        stream.write(encoded + "\n")
        stream.flush()


class _ProtocolLogHandler(logging.Handler):
    def __init__(self, stream: IO[str]):
        super().__init__()
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _emit_event(
                self._stream,
                {
                    "type": "log",
                    "level": record.levelname,
                    "message": self.format(record),
                },
            )
        except Exception:
            self.handleError(record)


def _load_request(path: Path) -> tuple[JobItem, AppSettings, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported isolated video job request schema")
    raw_job = payload["job"]
    job = JobItem(
        id=int(raw_job["id"]),
        path=Path(raw_job["path"]),
        duration_seconds=raw_job.get("duration_seconds"),
        segments=tuple(
            SegmentRange(float(segment["start"]), float(segment["end"]))
            for segment in raw_job.get("segments", ())
        ),
        segment_selection_mode=SegmentSelectionMode(
            raw_job.get("segment_selection_mode", SegmentSelectionMode.DEFAULT.value)
        ),
        detection_model=raw_job.get("detection_model"),
        detection_score_threshold=raw_job.get("detection_score_threshold"),
        vr_projection=raw_job.get("vr_projection"),
    )
    settings = AppSettings(**payload["settings"])
    return job, settings, payload


def run_video_job_file(
    request_path: str | Path,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
) -> int:
    """Run one complete video job and emit line-delimited JSON events."""

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    os.environ["JASNA_MAIN_PID"] = str(os.getpid())
    previous_isolated_job = os.environ.get(ISOLATED_VIDEO_JOB_ENV)
    os.environ[ISOLATED_VIDEO_JOB_ENV] = "1"

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    root_logger.handlers = [_ProtocolLogHandler(output_stream)]
    root_logger.setLevel(logging.INFO)

    try:
        if is_frozen():
            from jasna._frozen import patch_frozen_torch

            patch_frozen_torch()
        job, settings, payload = _load_request(Path(request_path))
        # Queue-level actions belong to the GUI parent and must run only once.
        settings = replace(
            settings,
            post_export_action="none",
            post_export_command="",
        )

        from jasna.windows_native_logs import install_worker_native_logs

        native_log_policy = install_worker_native_logs()
        if native_log_policy is not None:
            root_logger.info("FFmpeg diagnostics use native stderr for this isolated Windows worker")

        from jasna.gui.processor import Processor, ProgressUpdate

        completed = threading.Event()

        def on_progress(update: ProgressUpdate) -> None:
            _emit_event(
                output_stream,
                {
                    "type": "progress",
                    "update": {
                        **asdict(update),
                        "status": update.status.value,
                    },
                },
            )

        processor = Processor(
            on_progress=on_progress,
            on_log=lambda level, message: _emit_event(
                output_stream,
                {"type": "log", "level": level, "message": message},
            ),
            on_complete=completed.set,
        )
        processor._jobs = [job]
        processor._settings = settings
        processor._output_folder = str(payload.get("output_folder", ""))
        processor._output_pattern = str(
            payload.get("output_pattern", "{original}_restored.mp4")
        )
        processor._disable_basicvsrpp_tensorrt_for_run = bool(
            payload.get("disable_basicvsrpp_tensorrt", False)
        )

        if os.environ.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") == "1":
            from jasna.gui.windows_video_worker import report_windows_worker_gpu_identity

            report_windows_worker_gpu_identity(output_stream)

        def control_loop() -> None:
            for line in input_stream:
                try:
                    command = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if command.get("command") == "stop":
                    processor.stop()
                elif command.get("command") == "set_paused":
                    paused = bool(command.get("paused"))
                    if processor.is_paused() != paused:
                        processor.pause()
            if not completed.is_set():
                processor.stop()

        threading.Thread(
            target=control_loop,
            daemon=True,
            name="isolated-video-job-control",
        ).start()
        if native_log_policy is not None:
            native_log_policy.assert_active()
        processor._run()
        if native_log_policy is not None:
            native_log_policy.assert_active()
        completed.set()

        result: dict[str, Any] = {"type": "result", "status": job.status.value}
        if job.status is JobStatus.COMPLETED:
            if job.output_path is None:
                raise RuntimeError("completed video job did not record its output path")
            processing_path = processor.completed_processing_path(job.id)
            if processing_path not in {"copy", "full", "smart"}:
                raise RuntimeError("completed video job did not record its processing path")
            result["output_path"] = str(job.output_path)
            result["processing_path"] = processing_path
        _emit_event(output_stream, result)
        return 0
    except NativeWorkerRecycleRequested as error:
        _emit_event(
            output_stream,
            {
                "type": "retry",
                "reason": error.reason,
                "message": str(error),
            },
        )
        return NATIVE_PRESSURE_RECYCLE_EXIT_CODE
    except HostMemoryPressureError as error:
        _emit_event(
            output_stream,
            {
                "type": "fatal",
                "reason": error.reason,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    except Exception as error:
        _emit_event(
            output_stream,
            {
                "type": "fatal",
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)
        if previous_isolated_job is None:
            os.environ.pop(ISOLATED_VIDEO_JOB_ENV, None)
        else:
            os.environ[ISOLATED_VIDEO_JOB_ENV] = previous_isolated_job


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: python -m jasna.gui.video_job_process REQUEST.json")
    return run_video_job_file(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())

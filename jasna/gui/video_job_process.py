"""Protocol and entry point for an isolated GUI video job."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
)
from jasna.segments import SegmentRange


EVENT_PREFIX = "JASNA_JOB_EVENT\t"
REQUEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
_EVENT_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class VideoJobResult:
    status: JobStatus
    output_path: Path | None
    processing_path: str | None


def build_video_job_request(
    job: JobItem,
    snapshot: JobProcessingSnapshot,
    settings: AppSettings,
    *,
    output_folder: str,
    output_pattern: str,
    disable_basicvsrpp_tensorrt: bool,
) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "job": {
            "id": job.id,
            "path": str(job.path),
            "duration_seconds": job.duration_seconds,
            "segments": [asdict(segment) for segment in snapshot.segments],
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
    try:
        path.chmod(0o600)
    except OSError:
        pass


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


def parse_video_job_result(
    event: dict[str, Any],
    *,
    expected_output_path: Path,
) -> VideoJobResult:
    """Validate a terminal child result before the parent trusts it."""

    if event.get("type") != "result":
        raise ValueError("isolated video job result event has the wrong type")
    if event.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported isolated video job result schema")

    raw_status = event.get("status")
    if not isinstance(raw_status, str):
        raise ValueError("isolated video job result is missing a status")
    try:
        status = JobStatus(raw_status)
    except ValueError as error:
        raise ValueError(
            f"isolated video job result has an invalid status: {raw_status!r}"
        ) from error
    if status not in {
        JobStatus.COMPLETED,
        JobStatus.ERROR,
        JobStatus.PENDING,
        JobStatus.SKIPPED,
    }:
        raise ValueError(
            f"isolated video job result has a non-terminal status: {raw_status!r}"
        )

    raw_output_path = event.get("output_path")
    if status is not JobStatus.COMPLETED:
        if raw_output_path is not None:
            raise ValueError(
                "non-completed isolated video job result must not report an output path"
            )
        if event.get("processing_path") is not None:
            raise ValueError(
                "non-completed isolated video job result must not report a processing path"
            )
        return VideoJobResult(
            status=status,
            output_path=None,
            processing_path=None,
        )

    if not isinstance(raw_output_path, str) or not raw_output_path:
        raise ValueError("completed isolated video job result is missing an output path")
    processing_path = event.get("processing_path")
    if processing_path not in {"full", "smart", "copy"}:
        raise ValueError(
            "completed isolated video job result has an invalid processing path: "
            f"{processing_path!r}"
        )
    output_path = Path(raw_output_path)
    if output_path.resolve(strict=False) != expected_output_path.resolve(strict=False):
        raise ValueError(
            "isolated video job reported an unexpected output path: "
            f"{output_path} (expected {expected_output_path})"
        )
    return VideoJobResult(
        status=status,
        output_path=output_path,
        processing_path=str(processing_path),
    )


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
    """Run one video job and emit progress as line-delimited JSON."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    os.environ["JASNA_MAIN_PID"] = str(os.getpid())

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
            post_export_video_command="",
        )

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
        processor._run()
        completed.set()
        _emit_event(
            output_stream,
            {
                "type": "result",
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": job.status.value,
                "output_path": (
                    str(job.output_path) if job.output_path is not None else None
                ),
            },
        )
        return 0
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: python -m jasna.gui.video_job_process REQUEST.json")
    return run_video_job_file(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())

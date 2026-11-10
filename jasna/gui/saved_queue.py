"""The unfinished GUI queue in queue.json, so it comes back after Jasna restarts."""

import json
import logging
from pathlib import Path
from typing import Iterable, get_args

from jasna import os_utils
from jasna.gui.models import JobItem, JobStatus
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.session_config import RestorationModelName

logger = logging.getLogger(__name__)

QUEUE_FORMAT = 1


def get_queue_path() -> Path:
    return os_utils.get_user_config_dir("jasna") / "queue.json"


def _optional(value, kind: type | tuple[type, ...]):
    if value is None or isinstance(value, kind):
        return value
    raise TypeError(f"unexpected value {value!r}")


def _optional_path(value) -> Path | None:
    text = _optional(value, str)
    return None if text is None else Path(text)


def _segment_to_dict(segment: SegmentRange) -> dict:
    restoration = segment.restoration
    return {
        "start": segment.start,
        "end": segment.end,
        "restoration": None if restoration is None else {"model": restoration.model, "ltx_seed": restoration.ltx_seed},
    }


def _segment_from_dict(data: dict) -> SegmentRange:
    restoration = data["restoration"]
    if restoration is not None:
        if restoration["model"] not in get_args(RestorationModelName):
            raise ValueError(f"unknown restoration model {restoration['model']!r}")
        restoration = SegmentRestoration(restoration["model"], _optional(restoration["ltx_seed"], int))
    return SegmentRange(data["start"], data["end"], restoration)


def job_to_dict(job: JobItem) -> dict:
    """What a queue item keeps across restarts: its input, its own options and the output its
    unfinished run left. Status and progress are left out; every restored item is pending."""
    return {
        "path": str(job.path),
        "duration_seconds": job.duration_seconds,
        "segments": [_segment_to_dict(segment) for segment in job.snapshot_segments()],
        "detection_model": job.detection_model,
        "detection_score_threshold": job.detection_score_threshold,
        "vr_projection": job.vr_projection,
        "unfinished_output": None if job.unfinished_output is None else str(job.unfinished_output),
    }


def job_from_dict(data: dict) -> JobItem:
    """A pending queue item from ``job_to_dict``; raises KeyError, TypeError or ValueError on bad data."""
    threshold = _optional(data["detection_score_threshold"], (int, float))
    return JobItem(
        path=Path(data["path"]),
        duration_seconds=_optional(data["duration_seconds"], (int, float)),
        segments=tuple(_segment_from_dict(segment) for segment in data["segments"]),
        detection_model=_optional(data["detection_model"], str),
        detection_score_threshold=None if threshold is None else float(threshold),
        vr_projection=_optional(data["vr_projection"], str),
        unfinished_output=_optional_path(data["unfinished_output"]),
    )


def queue_to_dict(jobs: Iterable[JobItem]) -> dict:
    """The queue without its finished items."""
    return {
        "format": QUEUE_FORMAT,
        "jobs": [job_to_dict(job) for job in jobs if job.status is not JobStatus.COMPLETED],
    }


def jobs_from_queue(data: dict) -> list[JobItem]:
    """The items of ``queue_to_dict``; raises KeyError, TypeError or ValueError on bad data."""
    if data["format"] != QUEUE_FORMAT:
        raise ValueError(f"unknown queue format {data['format']!r}")
    return [job_from_dict(job) for job in data["jobs"]]


def save_queue(jobs: Iterable[JobItem]) -> None:
    path = get_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with os_utils.atomic_write(path, "wb") as f:
            f.write(json.dumps(queue_to_dict(jobs), indent=2, ensure_ascii=False).encode("utf-8"))
    except OSError as e:
        logger.warning("Could not save the queue to %s: %s", path, e)


def load_queue() -> list[JobItem]:
    """The saved queue without items whose input file is gone; empty when nothing was saved
    or the file cannot be read."""
    path = get_queue_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            jobs = jobs_from_queue(json.load(f))
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("Could not restore the queue from %s, starting with an empty queue: %s", path, e)
        return []
    present = []
    for job in jobs:
        if job.path.is_file():
            present.append(job)
        else:
            logger.info("Left %s out of the restored queue: the file no longer exists", job.path)
    return present

"""Automatic pre-processing scan routing with durable checkpoints.

The segment editor and processing pre-scan intentionally share
``MosaicScanWorker``.  This module only coordinates scan stages, persists the
small score timeline, refines hit boundaries, and chooses an existing full,
smart-render, or source-copy path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    ADAPTIVE_COARSE_POLICY_VERSION,
    ADAPTIVE_COARSE_TOLERANCE_RATIO,
    SCAN_SCORE_FLOOR,
    AdaptiveCoarsePlan,
    MosaicScanResult,
    MosaicScanWorker,
    ScanCheckpoint,
    ScanCompleted,
    ScanFailed,
    ScanProgress,
    ScanStatus,
    plan_adaptive_coarse_scan,
    scan_sample_stride,
    segments_from_scores,
)
from jasna.media.splice import probe_keyframes
from jasna.segments import SegmentRange, normalize_segments


PRE_SCAN_SCHEMA_VERSION = 1
PRE_SCAN_ALGORITHM_VERSION = "jasna-pre-scan-v5-pts-jitter-merge"
CHECKPOINT_SAMPLE_BATCH = 256
CHECKPOINT_MAX_SECONDS = 30.0
TIMELINE_PAD_AUTO = "auto"
TIMELINE_PAD_AUTO_MIN_SECONDS = 0.5
TIMELINE_PAD_AUTO_MAX_SECONDS = 1.0
TIMELINE_MERGE_GAP_SECONDS = 0.0
SHORT_CANDIDATE_IGNORE_SECONDS = 1.0
SHORT_CANDIDATE_MAX_SECONDS = 10.0
SHORT_CANDIDATE_HIGH_CONFIDENCE = 0.90
SHORT_CANDIDATE_CORE_SECONDS = 2.0

PreScanPath = Literal["full", "smart", "copy"]
CoarseExecutionStrategy = Literal["adaptive-direct-gop", "fixed-grid"]


def coarse_execution_strategy(metadata) -> CoarseExecutionStrategy:
    """Choose a coarse reader whose teardown is reliable on this platform."""

    codec = str(getattr(metadata, "codec_name", "")).strip().lower()
    windows_amd_hevc_main10 = (
        sys.platform == "win32"
        and vendor_for_device("cuda:0") is AcceleratorVendor.AMD
        and codec in {"hevc", "h265"}
        and bool(getattr(metadata, "is_10bit", False))
    )
    # Windows AMF cannot currently expose HEVC Main10/P010 frames to PyAV's
    # tensor path, so NvidiaVideoReader falls back to frame-threaded software
    # decoding. Closing a partly consumed per-GOP reader can then block in
    # avcodec_free_context. The fixed-grid reader drains its single decoder.
    return "fixed-grid" if windows_amd_hevc_main10 else "adaptive-direct-gop"


def _write_json_atomic(path: Path, value: dict) -> None:
    """Durably replace a small checkpoint without another candidate dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_identity(path: str | Path) -> dict:
    """Bind a checkpoint to source metadata plus small head/tail hashes."""

    source = Path(path).resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    with source.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return {
        "path": str(source),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_tail_sha256": digest.hexdigest(),
    }


class PreScanStopped(Exception):
    """Raised when processing was stopped during a scan stage."""


class PreScanFailed(RuntimeError):
    """Raised when a forced scan cannot produce a trustworthy result."""


@dataclass(frozen=True)
class PreScanOutcome:
    processing_path: PreScanPath
    segments: tuple[SegmentRange, ...] = ()
    coverage: float = 0.0
    reason: str = ""
    automatic: bool = True


def segment_coverage(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    duration: float,
) -> float:
    duration = float(duration)
    if duration <= 0:
        return 0.0
    normalized = normalize_segments(tuple(segments), duration=duration)
    return min(1.0, sum(segment.duration for segment in normalized) / duration)


def sampled_coverage(
    result: MosaicScanResult,
    *,
    threshold: float,
) -> float:
    """Estimate hit coverage from sample time, not an unweighted hit count.

    The established fixed-grid scanner remains byte-for-byte equivalent to its
    old [sample, sample + stride) bins.  Adaptive coarse samples are irregular
    (keyframes and sparse-GOP targets), so their duration is apportioned by
    midpoint cells; dense scene-cut regions then carry only the short time they
    actually represent.
    """

    times = tuple(float(value) for value in result.times)
    scores = tuple(float(value) for value in result.scores)
    if len(times) != len(scores):
        raise ValueError("scan times and scores must have the same length")
    duration = float(result.duration)
    stride = float(result.stride)
    threshold = float(threshold)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("scan duration must be finite and non-negative")
    if not math.isfinite(stride) or stride <= 0:
        raise ValueError("scan stride must be finite and positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("scan threshold must be between zero and one")
    if not times:
        return 0.0
    if any(
        not math.isfinite(seconds)
        or seconds < 0
        or (index and seconds < times[index - 1])
        for index, seconds in enumerate(times)
    ):
        raise ValueError("scan sample times must be ordered finite non-negative values")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scan scores must be finite values between zero and one")

    # Existing precise and legacy fixed-grid coarse scans retain their exact
    # coverage behavior, including the terminal bin's historical stride.
    grid_epsilon = max(1e-9, min(0.01, stride * 0.001))
    uniform = len(times) <= 1 or all(
        abs((current - previous) - stride) <= grid_epsilon
        for previous, current in zip(times, times[1:])
    )
    if uniform:
        bins = segments_from_scores(
            times,
            scores,
            threshold=threshold,
            stride=stride,
            duration=duration,
            pad=0.0,
        )
        return segment_coverage(bins, duration)

    covered = 0.0
    for index, (seconds, score) in enumerate(zip(times, scores)):
        left = 0.0 if index == 0 else (times[index - 1] + seconds) / 2.0
        right = duration if index + 1 == len(times) else (seconds + times[index + 1]) / 2.0
        if score >= threshold:
            covered += max(0.0, min(duration, right) - max(0.0, left))
    return min(1.0, covered / duration) if duration > 0 else 0.0


def normalize_scan_segments(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    *,
    duration: float,
    pad_seconds: float,
    merge_gap_seconds: float = TIMELINE_MERGE_GAP_SECONDS,
) -> tuple[SegmentRange, ...]:
    """Pad precise ranges and merge only overlapping/touching results."""

    duration = float(duration)
    if duration <= 0 or not segments:
        return ()
    pad_seconds = float(pad_seconds)
    merge_gap_seconds = float(merge_gap_seconds)
    if not math.isfinite(pad_seconds) or pad_seconds < 0:
        raise ValueError("scan range padding must be finite and non-negative")
    if not math.isfinite(merge_gap_seconds) or merge_gap_seconds < 0:
        raise ValueError("scan range merge gap must be finite and non-negative")
    padded: list[SegmentRange] = []
    for segment in normalize_segments(tuple(segments), duration=duration):
        start = max(0.0, segment.start - pad_seconds)
        end = min(duration, segment.end + pad_seconds)
        padded.append(SegmentRange(start, end))

    merged: list[SegmentRange] = []
    for segment in sorted(padded):
        if merged and segment.start <= merged[-1].end + merge_gap_seconds:
            previous = merged[-1]
            merged[-1] = SegmentRange(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return normalize_segments(tuple(merged), duration=duration)


def coarse_route(
    result: MosaicScanResult,
    *,
    detection_threshold: float,
    full_threshold: float,
) -> PreScanPath | Literal["fine"]:
    coverage = sampled_coverage(result, threshold=detection_threshold)
    if not any(score >= detection_threshold for score in result.scores):
        return "copy"
    if coverage >= full_threshold:
        return "full"
    return "fine"


def precise_scan_segments(
    result: MosaicScanResult,
    *,
    threshold: float,
    pad_seconds: float | None = None,
) -> tuple[SegmentRange, ...]:
    """Build restoration ranges from the precise sample grid."""

    _sampled, _filtered, normalized = _precise_scan_segment_stages(
        result,
        threshold=threshold,
        pad_seconds=pad_seconds,
    )
    return normalized


def resolve_timeline_pad_seconds(value: object, *, fine_interval: float) -> float:
    """Resolve Auto padding to the precise interval, clamped to 0.5-1.0s."""

    text = str(value).strip().lower()
    if text == TIMELINE_PAD_AUTO:
        interval = float(fine_interval)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("precise scan interval must be finite and positive")
        return min(
            TIMELINE_PAD_AUTO_MAX_SECONDS,
            max(TIMELINE_PAD_AUTO_MIN_SECONDS, interval),
        )
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scan range padding must be Auto or a number") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("scan range padding must be finite and non-negative")
    return seconds


def _has_sustained_high_confidence_core(
    result: MosaicScanResult,
    segment: SegmentRange,
    *,
    threshold: float = SHORT_CANDIDATE_HIGH_CONFIDENCE,
    core_seconds: float = SHORT_CANDIDATE_CORE_SECONDS,
) -> bool:
    stride = float(result.stride)
    epsilon = max(1e-9, stride * 0.01)
    run_start: float | None = None
    run_end: float | None = None
    for seconds, score in zip(result.times, result.scores):
        seconds = float(seconds)
        if seconds < segment.start - epsilon:
            continue
        if seconds >= segment.end - epsilon:
            break
        sample_start = max(segment.start, seconds)
        sample_end = min(segment.end, seconds + stride)
        if float(score) >= float(threshold) and sample_end > sample_start:
            if run_start is None or run_end is None or sample_start > run_end + epsilon:
                run_start = sample_start
                run_end = sample_end
            else:
                run_end = max(run_end, sample_end)
            if run_end - run_start + epsilon >= float(core_seconds):
                return True
        else:
            run_start = None
            run_end = None
    return False


def filter_short_scan_candidates(
    result: MosaicScanResult,
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    *,
    detection_threshold: float,
) -> tuple[SegmentRange, ...]:
    """Reject isolated or weak short candidates before padding."""

    high_threshold = max(float(detection_threshold), SHORT_CANDIDATE_HIGH_CONFIDENCE)
    kept: list[SegmentRange] = []
    for segment in segments:
        if segment.duration <= SHORT_CANDIDATE_IGNORE_SECONDS + 1e-9:
            continue
        if segment.duration > SHORT_CANDIDATE_MAX_SECONDS + 1e-9:
            kept.append(segment)
            continue
        if _has_sustained_high_confidence_core(
            result,
            segment,
            threshold=high_threshold,
        ):
            kept.append(segment)
    return tuple(kept)


def _precise_scan_segment_stages(
    result: MosaicScanResult,
    *,
    threshold: float,
    pad_seconds: float | None,
) -> tuple[
    tuple[SegmentRange, ...],
    tuple[SegmentRange, ...],
    tuple[SegmentRange, ...],
]:

    sampled = segments_from_scores(
        result.times,
        result.scores,
        threshold=threshold,
        stride=result.stride,
        duration=result.duration,
        pad=0.0,
    )
    filtered = filter_short_scan_candidates(
        result,
        sampled,
        detection_threshold=threshold,
    )
    resolved_pad = (
        resolve_timeline_pad_seconds(TIMELINE_PAD_AUTO, fine_interval=result.stride)
        if pad_seconds is None
        else float(pad_seconds)
    )
    normalized = normalize_scan_segments(
        filtered,
        duration=result.duration,
        pad_seconds=resolved_pad,
    )
    return sampled, filtered, normalized


class _ScanCheckpointStore:
    def __init__(
        self,
        path: Path,
        signature: dict,
        *,
        fps: float,
        time_base: float,
    ) -> None:
        self.path = path
        self.fps = float(fps)
        self.time_base = float(time_base)
        if self.time_base <= 0:
            raise ValueError("scan checkpoint time base must be positive")
        self._dirty_samples = 0
        self._last_write = time.monotonic()
        self.data = self._load(signature)

    def _load(self, signature: dict) -> dict:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == PRE_SCAN_SCHEMA_VERSION
                    and value.get("signature") == signature
                    and self._valid_checkpoint_data(value)
                ):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        value = {
            "schema_version": PRE_SCAN_SCHEMA_VERSION,
            "signature": signature,
            "stages": {},
            "outcome": None,
        }
        _write_json_atomic(self.path, value)
        return value

    @classmethod
    def _valid_checkpoint_data(cls, value: dict) -> bool:
        stages = value.get("stages")
        if not isinstance(stages, dict):
            return False
        for stage in stages.values():
            if not isinstance(stage, dict) or not isinstance(stage.get("complete"), bool):
                return False
            try:
                stride = float(stage["stride"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(stride) or stride <= 0:
                return False
            if not cls._valid_encoded_samples(stage.get("samples")):
                return False
        outcome = value.get("outcome")
        if outcome is None:
            return True
        if not isinstance(outcome, dict):
            return False
        if outcome.get("processing_path") not in {"full", "smart", "copy"}:
            return False
        try:
            coverage = float(outcome.get("coverage", 0.0))
        except (TypeError, ValueError):
            return False
        segments = outcome.get("segments", ())
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            return False
        if not isinstance(segments, list):
            return False
        try:
            return all(
                math.isfinite(float(item["start"]))
                and math.isfinite(float(item["end"]))
                and 0.0 <= float(item["start"]) < float(item["end"])
                for item in segments
                if isinstance(item, dict)
            ) and all(isinstance(item, dict) for item in segments)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _valid_encoded_samples(values) -> bool:
        if not isinstance(values, list):
            return False
        for item in values:
            if not isinstance(item, list) or len(item) != 3:
                return False
            try:
                sample_key = int(item[0])
                seconds = float(item[1])
                score = float(item[2])
            except (TypeError, ValueError):
                return False
            if (
                sample_key < 0
                or not math.isfinite(seconds)
                or seconds < 0
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                return False
        return True

    def _stage(self, name: str, stride: float) -> dict:
        stages = self.data.setdefault("stages", {})
        stage = stages.setdefault(
            name,
            {"stride": float(stride), "complete": False, "samples": []},
        )
        if not math.isclose(float(stage.get("stride", -1)), float(stride), abs_tol=1e-9):
            stage = {"stride": float(stride), "complete": False, "samples": []}
            stages[name] = stage
        return stage

    def samples(self, name: str, stride: float) -> dict[int, tuple[float, float]]:
        stage = self._stage(name, stride)
        return self._decode_samples(stage.get("samples", ()))

    def _decode_samples(self, values) -> dict[int, tuple[float, float]]:
        decoded: dict[int, tuple[float, float]] = {}
        for item in values or ():
            try:
                sample_key, seconds, score = int(item[0]), float(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if sample_key >= 0 and seconds >= 0 and 0.0 <= score <= 1.0:
                decoded[sample_key] = (seconds, score)
        return decoded

    @staticmethod
    def _encode_samples(values: dict[int, tuple[float, float]]) -> list[list[float | int]]:
        return [
            [int(sample_key), round(float(seconds), 9), round(float(score), 7)]
            for sample_key, (seconds, score) in sorted(values.items())
        ]

    def add_stage_samples(
        self,
        name: str,
        stride: float,
        times: tuple[float, ...],
        scores: tuple[float, ...],
        sample_keys: tuple[int, ...] | None = None,
    ) -> None:
        values = self.samples(name, stride)
        self._add(values, times, scores, sample_keys=sample_keys)
        self._stage(name, stride)["samples"] = self._encode_samples(values)
        self._maybe_flush()

    def _add(
        self,
        values: dict[int, tuple[float, float]],
        times: tuple[float, ...],
        scores: tuple[float, ...],
        *,
        sample_keys: tuple[int, ...] | None = None,
    ) -> None:
        if len(times) != len(scores):
            raise ValueError("checkpoint times and scores must have the same length")
        if sample_keys is not None and len(sample_keys) != len(times):
            raise ValueError("checkpoint sample keys and times must have the same length")
        keys = sample_keys or tuple(
            max(0, round(float(seconds) / self.time_base)) for seconds in times
        )
        for sample_key, seconds, score in zip(keys, times, scores):
            sample_key = max(0, int(sample_key))
            values[sample_key] = (float(seconds), float(score))
            self._dirty_samples += 1

    def stage_complete(self, name: str, stride: float) -> bool:
        return bool(self._stage(name, stride).get("complete"))

    def mark_stage_complete(self, name: str, stride: float) -> None:
        self._stage(name, stride)["complete"] = True
        self.flush()

    def set_outcome(self, outcome: PreScanOutcome) -> None:
        self.data["outcome"] = {
            "processing_path": outcome.processing_path,
            "coverage": outcome.coverage,
            "reason": outcome.reason,
            "segments": [
                {"start": segment.start, "end": segment.end}
                for segment in outcome.segments
            ],
        }
        self.flush()

    def completed_outcome(self) -> PreScanOutcome | None:
        value = self.data.get("outcome")
        if not isinstance(value, dict):
            return None
        try:
            return PreScanOutcome(
                processing_path=value["processing_path"],
                segments=tuple(
                    SegmentRange(float(item["start"]), float(item["end"]))
                    for item in value.get("segments", ())
                ),
                coverage=float(value.get("coverage", 0.0)),
                reason=str(value.get("reason", "checkpoint")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _maybe_flush(self) -> None:
        if (
            self._dirty_samples >= CHECKPOINT_SAMPLE_BATCH
            or time.monotonic() - self._last_write >= CHECKPOINT_MAX_SECONDS
        ):
            self.flush()

    def flush(self) -> None:
        _write_json_atomic(self.path, self.data)
        self._dirty_samples = 0
        self._last_write = time.monotonic()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _stat_identity(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_signature(
    source: Path,
    output: Path,
    settings: AppSettings,
    metadata,
) -> dict:
    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        require_detection_model_weights,
    )

    model_name = coerce_detection_model_name(str(settings.detection_model))
    model_path = require_detection_model_weights(model_name)
    resolved_pad_seconds = resolve_timeline_pad_seconds(
        settings.pre_scan_pad_seconds,
        fine_interval=float(settings.pre_scan_fine_interval),
    )
    return {
        "algorithm_version": PRE_SCAN_ALGORITHM_VERSION,
        "source": source_identity(source),
        "output_path": str(output.resolve()),
        "detector": {
            "name": model_name,
            "weights": _stat_identity(Path(model_path)),
            "threshold": float(settings.detection_score_threshold),
            "score_floor": SCAN_SCORE_FLOOR,
            "fp16": bool(settings.fp16_mode),
        },
        "scan": {
            "policy": str(settings.pre_scan_policy),
            "full_threshold": float(settings.pre_scan_full_threshold),
            "coarse_interval": float(settings.pre_scan_coarse_interval),
            "coarse_execution_strategy": coarse_execution_strategy(metadata),
            "fine_interval": float(settings.pre_scan_fine_interval),
            "adaptive_coarse": {
                "policy_version": ADAPTIVE_COARSE_POLICY_VERSION,
                "tolerance_ratio": ADAPTIVE_COARSE_TOLERANCE_RATIO,
                "coverage_policy": "duration-weighted-midpoints-v1",
            },
            "vr_mode": str(settings.vr_mode),
            "pad_seconds": {
                "configured": str(settings.pre_scan_pad_seconds),
                "resolved": resolved_pad_seconds,
            },
            "merge_gap_seconds": TIMELINE_MERGE_GAP_SECONDS,
            "short_candidate_policy": {
                "ignore_at_or_below_seconds": SHORT_CANDIDATE_IGNORE_SECONDS,
                "high_confidence_at_or_below_seconds": SHORT_CANDIDATE_MAX_SECONDS,
                "high_confidence_threshold": SHORT_CANDIDATE_HIGH_CONFIDENCE,
                "high_confidence_core_seconds": SHORT_CANDIDATE_CORE_SECONDS,
            },
            "precise_range_policy": "confidence-filter-plus-sample-padding-v2",
        },
    }


def _checkpoint_path(
    source: Path,
    output: Path,
    settings: AppSettings,
    metadata,
) -> tuple[Path, dict]:
    signature = _checkpoint_signature(source, output, settings, metadata)
    digest = _canonical_hash(signature)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", output.stem).strip("._-")
    slug = (slug or "output")[:64]
    root = (
        Path(settings.working_directory).expanduser()
        if str(settings.working_directory).strip()
        else output.parent
    )
    return root / f".{slug}.pre-scan-{digest[:16]}" / "manifest.json", signature


def _result_from_samples(
    samples: dict[int, tuple[float, float]],
    *,
    stride: float,
    duration: float,
) -> MosaicScanResult:
    ordered = sorted(samples.values())
    return MosaicScanResult(
        times=tuple(item[0] for item in ordered),
        scores=tuple(item[1] for item in ordered),
        masks=(),
        stride=float(stride),
        duration=float(duration),
        completed_until=ordered[-1][0] if ordered else 0.0,
    )


class PreScanCoordinator:
    def __init__(
        self,
        source: Path,
        output: Path,
        metadata,
        settings: AppSettings,
        *,
        stopped: Callable[[], bool],
        log: Callable[[str, str], None],
        progress: Callable[[str, float, float, float], None] | None = None,
        worker_factory=MosaicScanWorker,
    ) -> None:
        self.source = Path(source)
        self.output = Path(output)
        self.metadata = metadata
        self.settings = settings
        self._stopped = stopped
        self._log = log
        self._progress = progress
        self._worker_factory = worker_factory
        self._active_worker = None
        checkpoint_path, signature = _checkpoint_path(
            self.source,
            self.output,
            settings,
            metadata,
        )
        self.checkpoint = _ScanCheckpointStore(
            checkpoint_path,
            signature,
            fps=float(metadata.video_fps),
            time_base=float(metadata.time_base),
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint.path

    def stop(self) -> None:
        worker = self._active_worker
        if worker is not None:
            worker.stop()

    def close(self) -> None:
        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.close()
            worker.join()
        self.checkpoint.flush()

    def run(self) -> PreScanOutcome:
        cached = self.checkpoint.completed_outcome()
        if cached is not None:
            self._log("INFO", f"扫描断点已完成，复用路线：{cached.processing_path}")
            return cached
        threshold = float(self.settings.detection_score_threshold)
        if threshold < SCAN_SCORE_FLOOR:
            raise PreScanFailed(
                f"检测阈值 {threshold:.2f} 低于扫描可记录下限 {SCAN_SCORE_FLOOR:.2f}"
            )

        policy = str(self.settings.pre_scan_policy).strip().lower()
        if policy not in {"auto", "scan"}:
            return PreScanOutcome("full", reason="pre-scan disabled")
        full_threshold = float(self.settings.pre_scan_full_threshold)
        coarse_setting = float(self.settings.pre_scan_coarse_interval)
        fine_setting = float(self.settings.pre_scan_fine_interval)
        if not 0.0 < full_threshold <= 1.0:
            raise PreScanFailed("全片处理阈值必须大于 0% 且不超过 100%")
        if coarse_setting <= 0.0 or fine_setting <= 0.0:
            raise PreScanFailed("扫描间隔必须大于 0 秒")
        if policy == "auto":
            coarse_interval = coarse_setting
            strategy = coarse_execution_strategy(self.metadata)
            adaptive_coarse = strategy == "adaptive-direct-gop"
            if adaptive_coarse:
                self._log("INFO", f"阶段：自动粗扫（目标间隔 {coarse_interval:g} 秒）")
            else:
                self._log(
                    "INFO",
                    f"阶段：自动粗扫（固定网格 {coarse_interval:g} 秒；"
                    "Windows AMD HEVC Main10 兼容路径）",
                )
            coarse, _worker = self._run_stage(
                "coarse",
                coarse_interval,
                adaptive_coarse=adaptive_coarse,
            )
            coverage = sampled_coverage(coarse, threshold=threshold)
            self._log("INFO", f"粗扫有码覆盖率：{coverage:.1%}")
            route = coarse_route(
                coarse,
                detection_threshold=threshold,
                full_threshold=float(self.settings.pre_scan_full_threshold),
            )
            if route == "copy":
                return self._finish(
                    PreScanOutcome("copy", coverage=0.0, reason="coarse scan found no mosaic")
                )
            if route == "full":
                return self._finish(PreScanOutcome("full", coverage=coverage, reason="coarse coverage threshold"))
            self._log("INFO", "自动选择：进入精扫")

        fine_interval = fine_setting
        self._log("INFO", f"阶段：精扫（每 {fine_interval:g} 秒一帧）")
        coarse_known = self.checkpoint.samples(
            "coarse", float(self.settings.pre_scan_coarse_interval)
        )
        fine, _worker = self._run_stage(
            "fine",
            fine_interval,
            seed_samples=coarse_known,
        )
        try:
            pad_seconds = resolve_timeline_pad_seconds(
                self.settings.pre_scan_pad_seconds,
                fine_interval=fine_interval,
            )
        except ValueError as exc:
            raise PreScanFailed(str(exc)) from exc
        sampled, filtered, normalized = _precise_scan_segment_stages(
            fine,
            threshold=threshold,
            pad_seconds=pad_seconds,
        )
        self._log(
            "INFO",
            f"精扫候选区间：{len(sampled)}；短命中过滤后：{len(filtered)}；"
            f"扩边：前后各 {pad_seconds:g} 秒",
        )
        coverage = segment_coverage(normalized, float(self.metadata.duration))
        self._log("INFO", f"最终有码覆盖率：{coverage:.1%}")
        if not normalized:
            return self._finish(PreScanOutcome("copy", reason="precise scan found no mosaic"))
        if policy == "auto" and coverage >= float(self.settings.pre_scan_full_threshold):
            return self._finish(PreScanOutcome("full", coverage=coverage, reason="precise coverage threshold"))
        if coverage >= 1.0 - 1e-9:
            return self._finish(PreScanOutcome("full", coverage=coverage, reason="scan covers the full video"))
        return self._finish(
            PreScanOutcome(
                "smart",
                segments=normalized,
                coverage=coverage,
                reason="precise scan ranges",
            )
        )

    def _finish(self, outcome: PreScanOutcome) -> PreScanOutcome:
        self.checkpoint.set_outcome(outcome)
        self._log("INFO", f"最终路线：{outcome.processing_path}")
        return outcome

    def _adaptive_coarse_plan(self, target_interval: float) -> AdaptiveCoarsePlan:
        """Build a fresh deterministic plan only when coarse work is needed."""

        if self._stopped():
            raise PreScanStopped("pre-scan stopped")
        self._log("INFO", "粗扫：读取安全关键帧索引")
        try:
            index = probe_keyframes(self.source, self.metadata)
            plan = plan_adaptive_coarse_scan(
                index,
                duration=float(self.metadata.duration),
                target_interval=float(target_interval),
                fps=float(self.metadata.video_fps),
                time_base=float(self.metadata.time_base),
            )
        except PreScanStopped:
            raise
        except Exception as exc:
            # No sequential/CPU fallback is trustworthy here: a partial or
            # unsafe keyframe index would invalidate the adaptive coverage.
            raise PreScanFailed(f"粗扫关键帧索引失败：{exc}") from exc
        if self._stopped():
            raise PreScanStopped("pre-scan stopped")
        regular_groups = sum(group.mode == "regular" for group in plan.groups)
        dense_groups = sum(group.mode == "dense" for group in plan.groups)
        sparse_groups = sum(group.mode == "sparse" for group in plan.groups)
        self._log(
            "INFO",
            "粗扫关键帧计划："
            f"{plan.sample_count} 个采样点，{len(plan.groups)} 次 seek，"
            f"常规/密集/稀疏 GOP={regular_groups}/{dense_groups}/{sparse_groups}",
        )
        return plan

    def _run_stage(
        self,
        name: str,
        requested_stride: float,
        *,
        seed_samples: dict[int, tuple[float, float]] | None = None,
        adaptive_coarse: bool = False,
    ) -> tuple[MosaicScanResult, object | None]:
        fps = float(self.metadata.video_fps)
        actual_stride = scan_sample_stride(fps, seconds=requested_stride) / fps
        duration = float(self.metadata.duration)
        # A keyframe plan uses the requested S exactly.  Fine scanning retains
        # its historical frame-rounded stride and is deliberately unchanged.
        stage_stride = float(requested_stride) if adaptive_coarse else actual_stride
        saved = self.checkpoint.samples(name, stage_stride)
        if self.checkpoint.stage_complete(name, stage_stride):
            result = _result_from_samples(saved, stride=stage_stride, duration=duration)
            self._log("INFO", f"扫描断点：复用 {name} 的 {len(saved)} 个采样点")
            return result, None

        known = dict(seed_samples or {})
        known.update(saved)
        adaptive_plan = (
            self._adaptive_coarse_plan(requested_stride) if adaptive_coarse else None
        )
        # Seeking into an incomplete stage re-anchors frame-stride sampling in
        # the decoder and can produce a different sample grid.  Decode from the
        # beginning and let MosaicScanWorker skip detector inference only for
        # checkpointed exact source PTS values.  Decode is cheap compared with
        # inference and this also resumes safely after non-contiguous batches.
        worker_kwargs = {
            "stride_seconds": requested_stride,
            "start_seconds": 0.0,
            "known_sample_scores": {key: value[1] for key, value in known.items()},
            "emit_checkpoints": True,
        }
        if adaptive_plan is not None:
            worker_kwargs["adaptive_coarse_plan"] = adaptive_plan
        worker = self._worker_factory(
            self.source,
            self.metadata,
            self.settings,
            **worker_kwargs,
        )
        self._active_worker = worker
        worker.start()
        stopped_requested = False
        received_checkpoint = False
        try:
            while True:
                if self._stopped() and not stopped_requested:
                    stopped_requested = True
                    worker.stop()
                try:
                    event = worker.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(event, ScanCheckpoint):
                    received_checkpoint = True
                    self.checkpoint.add_stage_samples(
                        name,
                        stage_stride,
                        event.times,
                        event.scores,
                        sample_keys=event.sample_keys,
                    )
                elif isinstance(event, ScanStatus):
                    self._log("INFO", str(event.message))
                elif isinstance(event, ScanProgress) and self._progress is not None:
                    self._progress(name, event.fraction, event.fps, event.eta_seconds)
                elif isinstance(event, ScanFailed):
                    raise PreScanFailed(event.message)
                elif isinstance(event, ScanCompleted):
                    # Adaptive coarse checkpoints carry exact decoded PTS
                    # keys.  Do not append a time-base-rounded copy of the
                    # final result over them.  Fine scanning deliberately
                    # retains its established completion behavior.
                    if not adaptive_coarse or not received_checkpoint:
                        self.checkpoint.add_stage_samples(
                            name,
                            stage_stride,
                            event.result.times,
                            event.result.scores,
                        )
                    self.checkpoint.flush()
                    if event.stopped or self._stopped():
                        raise PreScanStopped("pre-scan stopped")
                    self.checkpoint.mark_stage_complete(name, stage_stride)
                    saved = self.checkpoint.samples(name, stage_stride)
                    result = _result_from_samples(
                        saved, stride=stage_stride, duration=duration
                    )
                    return result, None
        finally:
            self.checkpoint.flush()
            self._close_active_worker()

    def _close_active_worker(self) -> None:
        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.close()
            worker.join()

"""Fast mosaic scan for the segment editor.

Samples the video at a fixed stride, runs the configured detection model on
GPU-decoded frames, and collects per-sample scores plus low-res masks into
preallocated tensors. Scores can be re-thresholded after the scan without
rescanning.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.media import VideoMetadata
from jasna.segments import SegmentRange, normalize_segments

SCAN_SCORE_FLOOR = 0.05
SCAN_MASK_HW = (90, 160)


@dataclass(frozen=True)
class MosaicScanResult:
    """Per-sample detection scores and low-res masks, on CPU after the scan.

    Sample ``i`` was taken at ``times[i]`` seconds. ``scores`` holds the best
    detection score per sample (0.0 when nothing was detected), ``masks`` a
    uint8 [N, H, W] tensor of merged detection masks downscaled to
    ``mask_size``. ``completed_until`` is the last scanned timestamp; earlier
    than ``duration`` when the scan was stopped.
    """

    times: tuple[float, ...]
    scores: tuple[float, ...]
    masks: object
    stride: float
    duration: float
    completed_until: float

    def mask_at(self, seconds: float):
        if not self.times or seconds > self.completed_until + self.stride:
            return None
        last = len(self.times) - 1
        guess = min(last, max(0, round(seconds / self.stride)))
        _, index = min(
            (abs(self.times[i] - seconds), i)
            for i in {max(0, guess - 1), guess, min(last, guess + 1)}
        )
        if abs(self.times[index] - seconds) > self.stride:
            return None
        return self.masks[index]


def scan_sample_stride(fps: float, *, seconds: float = 1.0) -> int:
    """Frame stride for one detection sample roughly every ``seconds``."""

    return max(1, round(float(fps) * float(seconds)))


def segments_from_scores(
    times: tuple[float, ...] | list[float],
    scores: tuple[float, ...] | list[float],
    *,
    threshold: float,
    stride: float,
    duration: float,
    pad: float | None = None,
) -> tuple[SegmentRange, ...]:
    """Merge above-threshold samples into padded, normalized time ranges."""

    if len(times) != len(scores):
        raise ValueError("times and scores must have the same length")
    stride = float(stride)
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if pad is None:
        pad = stride / 2
    hits = []
    for seconds, score in zip(times, scores):
        if score < threshold:
            continue
        start = max(0.0, float(seconds) - pad)
        end = min(float(duration), float(seconds) + stride + pad)
        if end > start:
            hits.append(SegmentRange(start, end))
    return normalize_segments(hits, duration=duration)


@dataclass(frozen=True)
class ScanStatus:
    message: str


@dataclass(frozen=True)
class ScanProgress:
    fraction: float
    fps: float
    eta_seconds: float


@dataclass(frozen=True)
class ScanCompleted:
    result: MosaicScanResult
    stopped: bool


@dataclass(frozen=True)
class ScanFailed:
    message: str


ScanEvent = ScanStatus | ScanProgress | ScanCompleted | ScanFailed


class MosaicScanWorker:
    """One-shot background scan of a whole video with the detection model.

    Decodes with ``NvidiaVideoReader(frame_stride=N)``, runs the configured
    detector on every sampled frame at the ``SCAN_SCORE_FLOOR`` threshold, and
    collects per-sample scores plus merged low-res masks into preallocated GPU
    tensors. Stopping keeps everything scanned so far.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        settings: AppSettings,
        *,
        stride_seconds: float,
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.settings = settings
        self.stride_seconds = float(stride_seconds)
        self._on_stopped = on_stopped
        self.events: queue.Queue[ScanEvent] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"mosaic-scan-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            self.events.put(ScanStatus("loading_models"))
            detector = self._build_detector()
        except Exception as exc:
            self.events.put(ScanFailed(str(exc)))
            if self._on_stopped is not None:
                self._on_stopped()
            return
        try:
            self._scan(detector)
        except Exception as exc:
            self.events.put(ScanFailed(str(exc)))
        finally:
            if hasattr(detector, "close"):
                detector.close()
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            if self._on_stopped is not None:
                self._on_stopped()

    def _build_detector(self):
        from jasna._suppress_noise import install as _install_noise_filters

        _install_noise_filters()
        import torch

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.mosaic.detection_registry import (
            build_detection_model,
            coerce_detection_model_name,
            detection_model_weights_path,
        )

        settings = self.settings
        device = torch.device("cuda:0")
        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = detection_model_weights_path(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
            log_callback=lambda message: self.events.put(ScanStatus(message)),
        )
        return build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=SCAN_SCORE_FLOOR,
            fp16=bool(settings.fp16_mode),
        )

    def _scan(self, detector) -> None:
        import torch

        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        duration = float(metadata.duration)
        time_base = float(metadata.time_base)
        frame_stride = scan_sample_stride(metadata.video_fps, seconds=self.stride_seconds)
        sample_stride_seconds = frame_stride / float(metadata.video_fps)
        capacity = int(duration * float(metadata.video_fps) / frame_stride) + 8

        mask_h, mask_w = SCAN_MASK_HW
        masks = torch.zeros((capacity, mask_h, mask_w), dtype=torch.uint8, device=device)
        scores = torch.zeros((capacity,), dtype=torch.float32, device=device)
        times: list[float] = []
        target_hw = (int(metadata.video_height), int(metadata.video_width))
        batch_size = int(self.settings.batch_size)
        stopped = False
        last_progress = -1.0
        started = time.monotonic()

        reader = NvidiaVideoReader(
            str(self.path),
            batch_size,
            device,
            metadata,
            frame_stride=frame_stride,
        )
        with reader:
            for batch, pts_list in reader.frames():
                if self._stop.is_set():
                    stopped = True
                    break
                if len(times) + len(pts_list) > capacity:
                    break
                if batch.shape[0] < batch_size:
                    pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                    batch = torch.cat((batch, pad))
                batch_scores, batch_masks = detector.scan_scores_masks(
                    batch, mask_hw=(mask_h, mask_w)
                )
                index = len(times)
                count = len(pts_list)
                scores[index : index + count] = batch_scores[:count]
                masks[index : index + count] = batch_masks[:count].to(torch.uint8)
                times.extend(
                    max(0.0, (pts - metadata.start_pts) * time_base) for pts in pts_list
                )
                fraction = min(1.0, times[-1] / duration) if duration > 0 else 1.0
                if fraction - last_progress >= 0.01:
                    last_progress = fraction
                    elapsed = max(1e-6, time.monotonic() - started)
                    fps = len(times) * frame_stride / elapsed
                    video_rate = times[-1] / elapsed
                    eta = (
                        (duration - times[-1]) / video_rate if video_rate > 0 else 0.0
                    )
                    self.events.put(ScanProgress(fraction, fps, eta))

        count = len(times)
        result = MosaicScanResult(
            times=tuple(times),
            scores=tuple(scores[:count].cpu().tolist()),
            masks=masks[:count].cpu(),
            stride=sample_stride_seconds,
            duration=duration,
            completed_until=times[-1] if times else 0.0,
        )
        del masks, scores
        self.events.put(ScanCompleted(result, stopped))

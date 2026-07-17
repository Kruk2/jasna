"""Fast mosaic scan for the segment editor.

Samples the video at a fixed stride, runs the configured detection model on
GPU-decoded frames, and collects per-sample scores plus low-res masks into
preallocated tensors. When GPU headroom is low, completed result chunks spill
to a preallocated CPU tensor and the GPU chunk is reused. Scores can be
re-thresholded after the scan without rescanning.
"""

from __future__ import annotations

import bisect
import math
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
SCAN_VRAM_RESERVE_BYTES = 750 * 1024**2
SCAN_SPILL_CHUNK_BYTES = 64 * 1024**2


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

    def sample_at(self, seconds: float, *, tolerance: float):
        if not self.times:
            return None
        position = bisect.bisect_left(self.times, float(seconds))
        candidates = {
            max(0, position - 1),
            min(len(self.times) - 1, position),
        }
        index = min(candidates, key=lambda candidate: abs(self.times[candidate] - seconds))
        if abs(self.times[index] - seconds) > float(tolerance):
            return None
        return self.times[index], self.scores[index], self.masks[index]


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


@dataclass(frozen=True)
class ScanMaskReady:
    seconds: float
    score: float
    mask: object
    generation: int


@dataclass(frozen=True)
class ScanMaskFailed:
    message: str
    generation: int


@dataclass(frozen=True)
class ScanStorageSpilled:
    pass


@dataclass(frozen=True)
class _MaskRequest:
    seconds: float
    generation: int


@dataclass(frozen=True)
class _Close:
    pass


ScanEvent = (
    ScanStatus
    | ScanProgress
    | ScanCompleted
    | ScanFailed
    | ScanMaskReady
    | ScanMaskFailed
    | ScanStorageSpilled
)


class _ScanTensorCollector:
    """Collect fixed-size scan results with an adaptive CUDA-to-CPU spill."""

    def __init__(
        self,
        torch_mod,
        *,
        capacity: int,
        mask_hw: tuple[int, int],
        batch_size: int,
        device,
        on_spill: Callable[[], None],
    ) -> None:
        self._torch = torch_mod
        self.capacity = int(capacity)
        self.mask_h, self.mask_w = mask_hw
        self.batch_size = int(batch_size)
        self.device = device
        self._on_spill = on_spill
        self.count = 0
        self._buffer_count = 0
        self._spilling = False
        self._cpu_masks = None
        self._cpu_scores = None

        sample_bytes = self.mask_h * self.mask_w + 4
        required_bytes = self.capacity * sample_bytes
        free_bytes, _ = torch_mod.cuda.mem_get_info()
        projected_free = free_bytes - required_bytes
        if projected_free <= SCAN_VRAM_RESERVE_BYTES:
            self._enable_spill()
        else:
            self._gpu_masks, self._gpu_scores = self._allocate_gpu(self.capacity)

    @property
    def spilling(self) -> bool:
        return self._spilling

    def _allocate_gpu(self, capacity: int):
        torch_mod = self._torch
        masks = torch_mod.empty(
            (capacity, self.mask_h, self.mask_w),
            dtype=torch_mod.uint8,
            device=self.device,
        )
        scores = torch_mod.empty(
            (capacity,),
            dtype=torch_mod.float32,
            device=self.device,
        )
        return masks, scores

    def _allocate_cpu(self) -> None:
        torch_mod = self._torch
        self._cpu_masks = torch_mod.empty(
            (self.capacity, self.mask_h, self.mask_w),
            dtype=torch_mod.uint8,
            device="cpu",
        )
        self._cpu_scores = torch_mod.empty(
            (self.capacity,),
            dtype=torch_mod.float32,
            device="cpu",
        )

    def _spill_capacity(self) -> int:
        sample_bytes = self.mask_h * self.mask_w + 4
        return min(
            self.capacity,
            max(self.batch_size, SCAN_SPILL_CHUNK_BYTES // sample_bytes),
        )

    def _enable_spill(self) -> None:
        if self._spilling:
            return
        self._allocate_cpu()
        if hasattr(self, "_gpu_masks"):
            if self.count:
                self._cpu_masks[: self.count].copy_(self._gpu_masks[: self.count])
                self._cpu_scores[: self.count].copy_(self._gpu_scores[: self.count])
            del self._gpu_masks, self._gpu_scores
            self._torch.cuda.empty_cache()
        self._gpu_masks, self._gpu_scores = self._allocate_gpu(self._spill_capacity())
        self._buffer_count = 0
        self._spilling = True
        self._on_spill()

    def _flush(self) -> None:
        if not self._spilling or not self._buffer_count:
            return
        start = self.count - self._buffer_count
        self._cpu_masks[start : self.count].copy_(
            self._gpu_masks[: self._buffer_count]
        )
        self._cpu_scores[start : self.count].copy_(
            self._gpu_scores[: self._buffer_count]
        )
        self._buffer_count = 0

    def add(self, scores, masks, *, count: int) -> None:
        count = int(count)
        if self.count + count > self.capacity:
            raise RuntimeError("Video contains more frames than reported by its metadata")
        if (
            not self._spilling
            and self._torch.cuda.mem_get_info()[0] <= SCAN_VRAM_RESERVE_BYTES
        ):
            self._enable_spill()

        source_offset = 0
        while source_offset < count:
            if self._spilling:
                available = self._gpu_masks.shape[0] - self._buffer_count
                take = min(available, count - source_offset)
                target_start = self._buffer_count
            else:
                take = count - source_offset
                target_start = self.count
            target_end = target_start + take
            source_end = source_offset + take
            self._gpu_scores[target_start:target_end] = scores[source_offset:source_end]
            self._gpu_masks[target_start:target_end] = masks[source_offset:source_end].to(
                self._torch.uint8
            )
            self.count += take
            source_offset = source_end
            if self._spilling:
                self._buffer_count += take
                if self._buffer_count == self._gpu_masks.shape[0]:
                    self._flush()

    def finish(self):
        if self._spilling:
            self._flush()
            scores = tuple(self._cpu_scores[: self.count].tolist())
            masks = self._cpu_masks[: self.count]
        else:
            scores = tuple(self._gpu_scores[: self.count].cpu().tolist())
            masks = self._gpu_masks[: self.count].cpu()
        del self._gpu_masks, self._gpu_scores
        self._torch.cuda.empty_cache()
        return scores, masks


class MosaicScanWorker:
    """One-shot background scan of a whole video with the detection model.

    Decodes with ``NvidiaVideoReader(frame_stride=N)``, runs the configured
    detector on every sampled frame at the ``SCAN_SCORE_FLOOR`` threshold, and
    collects per-sample scores plus merged low-res masks into preallocated
    tensors. A 750 MiB VRAM reserve switches collection to a reusable GPU
    chunk backed by CPU storage. Stopping keeps everything scanned so far.
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
        self._stop_scan = threading.Event()
        self._closed = threading.Event()
        self._commands: queue.Queue[_MaskRequest | _Close] = queue.Queue(maxsize=1)
        self._mask_generation = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"mosaic-scan-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_scan.set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._stop_scan.set()
        self._replace_command(_Close(), allow_closed=True)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def request_mask(self, seconds: float) -> int:
        self._mask_generation += 1
        self._replace_command(_MaskRequest(max(0.0, float(seconds)), self._mask_generation))
        return self._mask_generation

    def _replace_command(
        self,
        command: _MaskRequest | _Close,
        *,
        allow_closed: bool = False,
    ) -> None:
        if self._closed.is_set() and not allow_closed:
            return
        try:
            while True:
                self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            pass

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
            if self._on_stopped is not None:
                self._on_stopped()
            self._serve_mask_requests(detector)
        except Exception as exc:
            if not self._closed.is_set():
                self.events.put(ScanFailed(str(exc)))
            if self._on_stopped is not None:
                self._on_stopped()
        finally:
            if hasattr(detector, "close"):
                detector.close()
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

    def _build_detector(self):
        from jasna._suppress_noise import install as _install_noise_filters

        _install_noise_filters()
        import torch

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.mosaic.detection_registry import (
            build_detection_model,
            coerce_detection_model_name,
            require_detection_model_weights,
        )

        settings = self.settings
        device = torch.device("cuda:0")
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
            log_callback=lambda message: self.events.put(ScanStatus(message)),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=SCAN_SCORE_FLOOR,
            fp16=bool(settings.fp16_mode),
        )
        from jasna.vr180 import (
            FisheyeProjector,
            SbsDetectionAdapter,
            resolve_vr_mode,
        )

        self._vr_resolution = resolve_vr_mode(
            settings.vr_mode,
            self.metadata,
            self.path,
        )
        self._vr_projector = (
            FisheyeProjector(
                eye_width=int(self.metadata.video_width) // 2,
                height=int(self.metadata.video_height),
                device=device,
            )
            if self._vr_resolution.uses_fisheye
            else None
        )
        self._scan_mask_projector = (
            FisheyeProjector(
                eye_width=SCAN_MASK_HW[1] // 2,
                height=SCAN_MASK_HW[0],
                device=device,
            )
            if self._vr_resolution.uses_fisheye
            else None
        )
        return (
            SbsDetectionAdapter(detector)
            if self._vr_resolution.is_sbs
            else detector
        )

    def _prepare_detection_batch(self, batch):
        if self._vr_projector is None:
            return batch
        return self._vr_projector.forward_sbs(batch)

    def _source_projection_masks(self, masks):
        if self._scan_mask_projector is None:
            return masks
        return self._scan_mask_projector.inverse_mask_sbs(masks)

    def _scan(self, detector) -> None:
        import torch

        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        duration = float(metadata.duration)
        time_base = float(metadata.time_base)
        frame_stride = scan_sample_stride(metadata.video_fps, seconds=self.stride_seconds)
        sample_stride_seconds = frame_stride / float(metadata.video_fps)
        batch_size = int(self.settings.batch_size)
        frame_count = int(metadata.num_frames)
        if frame_count > 0:
            capacity = math.ceil(frame_count / frame_stride) + batch_size
        else:
            estimated_rate = max(float(metadata.average_fps), float(metadata.video_fps))
            capacity = math.ceil(duration * estimated_rate / frame_stride) + batch_size

        times: list[float] = []
        collector = None
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
            from jasna.media import resolve_video_start_pts

            start_pts = resolve_video_start_pts(
                reader.video_stream.start_time,
                metadata.start_pts,
            )
            for batch, pts_list in reader.frames():
                if self._stop_scan.is_set():
                    stopped = True
                    break
                if len(times) + len(pts_list) > capacity:
                    raise RuntimeError(
                        "Video contains more frames than reported by its metadata"
                    )
                if batch.shape[0] < batch_size:
                    pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                    batch = torch.cat((batch, pad))
                detection_batch = self._prepare_detection_batch(batch)
                batch_scores, batch_masks = detector.scan_scores_masks(
                    detection_batch, mask_hw=SCAN_MASK_HW
                )
                batch_masks = self._source_projection_masks(batch_masks)
                count = len(pts_list)
                if collector is None:
                    collector = _ScanTensorCollector(
                        torch,
                        capacity=capacity,
                        mask_hw=SCAN_MASK_HW,
                        batch_size=batch_size,
                        device=device,
                        on_spill=lambda: self.events.put(ScanStorageSpilled()),
                    )
                collector.add(batch_scores, batch_masks, count=count)
                times.extend(
                    max(0.0, (pts - start_pts) * time_base) for pts in pts_list
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

        if collector is None:
            masks = torch.empty((0, *SCAN_MASK_HW), dtype=torch.uint8, device="cpu")
            result_scores = ()
        else:
            result_scores, masks = collector.finish()
        result = MosaicScanResult(
            times=tuple(times),
            scores=result_scores,
            masks=masks,
            stride=sample_stride_seconds,
            duration=duration,
            completed_until=times[-1] if times else 0.0,
        )
        self.events.put(ScanCompleted(result, stopped))

    def _serve_mask_requests(self, detector) -> None:
        while not self._closed.is_set():
            try:
                command = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(command, _Close):
                return
            try:
                event = self._detect_mask(detector, command)
            except Exception as exc:
                event = ScanMaskFailed(str(exc), command.generation)
            if not self._closed.is_set():
                self.events.put(event)

    def _detect_mask(self, detector, command: _MaskRequest) -> ScanMaskReady:
        import torch

        from jasna.media import resolve_video_start_pts
        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        batch_size = int(self.settings.batch_size)
        reader = NvidiaVideoReader(
            str(self.path),
            batch_size,
            device,
            metadata,
        )
        with reader:
            batch_and_pts = next(reader.frames(seek_ts=command.seconds), None)
            if batch_and_pts is None:
                raise RuntimeError("Could not decode the requested preview frame")
            batch, pts_list = batch_and_pts
            if batch.shape[0] < batch_size:
                pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                batch = torch.cat((batch, pad))
            detection_batch = self._prepare_detection_batch(batch)
            scores, masks = detector.scan_scores_masks(
                detection_batch,
                mask_hw=SCAN_MASK_HW,
            )
            masks = self._source_projection_masks(masks)
            start_pts = resolve_video_start_pts(
                reader.video_stream.start_time,
                metadata.start_pts,
            )
            seconds = max(
                0.0,
                (pts_list[0] - start_pts) * float(metadata.time_base),
            )
            return ScanMaskReady(
                seconds=seconds,
                score=float(scores[0].cpu()),
                mask=masks[0].to(torch.uint8).cpu(),
                generation=command.generation,
            )

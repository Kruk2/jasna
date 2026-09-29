from __future__ import annotations

import time
from typing import Callable

from tqdm import tqdm

ProgressCallback = Callable[[float, float, float, int, int, str], None]
"""``(percent, fps, eta_seconds, frames_done, total_frames, stage)``; ``stage`` names the LTX
pass running, or is empty for frame-by-frame restoration."""

LTX_WORK_PER_FRAME = 30.0
"""Rough cost of one LTX-restored frame in frames of the standard model."""


class JobProgress:
    """One bar for a job made of parts that report on their own (e.g. LTX ranges, then
    standard ranges). Each part fills its share of the bar by its estimated ``work``, in
    the order given; time left comes from the whole job's elapsed time and fraction, so
    it does not jump when the next part starts."""

    def __init__(self, callback: ProgressCallback, work: dict[str, float]) -> None:
        self._callback = callback
        total = sum(work.values())
        self._shares: dict[str, tuple[float, float]] = {}
        offset = 0.0
        for part, amount in work.items():
            share = amount / total if total > 0 else 0.0
            self._shares[part] = (offset, share)
            offset += share
        self._started = time.monotonic()
        self._timed = False

    def part(self, name: str) -> ProgressCallback:
        offset, share = self._shares[name]

        def report(percent: float, fps: float, eta_seconds: float, done: int, total: int, stage: str) -> None:
            fraction = min(1.0, offset + share * percent / 100.0)
            self._timed = self._timed or eta_seconds > 0
            elapsed = time.monotonic() - self._started
            eta = elapsed * (1.0 - fraction) / fraction if self._timed and fraction > 0 else 0.0
            self._callback(fraction * 100.0, fps, eta, done, total, stage)

        return report


class Progressbar:
    """Progress bar with time remaining estimation and speed display."""

    def __init__(
        self,
        total_frames: int,
        video_fps: float,
        disable: bool = False,
        callback: ProgressCallback | None = None,
    ):
        """Initialize progressbar.
        
        Args:
            total_frames: Total number of frames to process
            video_fps: Video FPS for buffer sizing
            disable: If True, suppress tqdm console output
            callback: Optional ``ProgressCallback``
        """
        self.total_frames = total_frames
        self.callback = callback
        self.frames_processed = 0
        self.frame_processing_durations_buffer: list[float] = []
        self.frame_processing_durations_buffer_min_len = min(
            total_frames - 1, int(video_fps * 2)  # Show FPS/ETA after 2 seconds of data
        )
        self.frame_processing_durations_buffer_max_len = min(
            total_frames - 1, int(video_fps * 30)  # Use last 30 seconds for averaging
        )
        self.error = False
        self.disable = disable

        # Bar format: "Processing video: 50%|████     |Processed: 1:23 (2848f) | Remaining: 0:10 | Speed: 35.0fps"
        bar_format = (
            "Processing video: {percentage:3.0f}%|{bar}|"
            "Processed: {elapsed} ({n_fmt}f){desc}"
        )
        initial_suffix = " | Remaining: ? | Speed: ?"
        self.tqdm = tqdm(
            dynamic_ncols=True,
            total=total_frames,
            bar_format=bar_format,
            desc=initial_suffix,
            disable=disable,
        )
        self.duration_start: float | None = None

    def init(self) -> None:
        """Initialize the timer. Call this right before the processing loop starts."""
        self.duration_start = time.time()

    def close(self, ensure_completed_bar: bool = False) -> None:
        """Close the progress bar.
        
        Args:
            ensure_completed_bar: If True, ensure the bar shows 100% even if
                frame count was estimated incorrectly.
        """
        if ensure_completed_bar:
            if not self.error and self.tqdm.total != self.tqdm.n:
                self.tqdm.total = self.tqdm.n
                self._update_time_remaining_and_speed(completed=True)
                self.tqdm.refresh()
        self.tqdm.close()

    def update(self, n: int = 1) -> None:
        """Update the progress bar after processing frame(s).
        
        Args:
            n: Number of frames processed since last update.
        """
        if self.duration_start is None:
            self.init()

        self.frames_processed += n
        
        duration_end = time.time()
        duration = duration_end - self.duration_start
        self.duration_start = duration_end

        # Store per-frame duration when processing multiple frames
        per_frame_duration = duration / n if n > 0 else duration
        for _ in range(n):
            if len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len:
                self.frame_processing_durations_buffer.pop(0)
            self.frame_processing_durations_buffer.append(per_frame_duration)

        self._update_time_remaining_and_speed()
        self.tqdm.update(n)
        
        if self.callback:
            progress_pct = (self.frames_processed / self.total_frames) * 100 if self.total_frames > 0 else 0
            fps = 0.0
            eta_seconds = 0.0
            if len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len:
                mean_duration = self._get_mean_processing_duration()
                fps = 1.0 / mean_duration if mean_duration > 0 else 0.0
                remaining = self.total_frames - self.frames_processed
                eta_seconds = remaining * mean_duration
            self.callback(progress_pct, fps, eta_seconds, self.frames_processed, self.total_frames, "")

    def _get_mean_processing_duration(self) -> float:
        """Calculate mean frame processing duration from the buffer."""
        return sum(self.frame_processing_durations_buffer) / len(
            self.frame_processing_durations_buffer
        )

    def _format_duration(self, duration_s: float | None) -> str:
        """Format duration in seconds to human-readable string (e.g., '1:23' or '1:02:03')."""
        if not duration_s or duration_s < 0:
            return "0:00"
        seconds = int(duration_s)
        minutes = seconds // 60
        hours = minutes // 60
        seconds = seconds % 60
        minutes = minutes % 60
        if hours == 0:
            return f"{minutes}:{seconds:02d}"
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _update_time_remaining_and_speed(self, completed: bool = False) -> None:
        """Update the description with remaining time and processing speed."""
        frames_remaining = (
            0 if completed else self.tqdm.format_dict["total"] - self.tqdm.format_dict["n"]
        )
        enough_datapoints = (
            len(self.frame_processing_durations_buffer)
            > self.frame_processing_durations_buffer_min_len
        )
        if enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            time_remaining_s = frames_remaining * mean_duration
            time_remaining = self._format_duration(time_remaining_s)
            speed_fps = f"{1.0 / mean_duration:.1f}" if mean_duration > 0 else "?"
            self.tqdm.desc = (
                f" | Remaining: {time_remaining} ({frames_remaining}f) | Speed: {speed_fps}fps"
            )

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


_HALF_RATE_SOURCES: tuple[Fraction, ...] = (
    Fraction(60, 1),
    Fraction(60_000, 1_001),
)

_SOURCE_RATE_TOLERANCE = 0.005
_MEASURED_RATE_TOLERANCE = 0.05


@dataclass(frozen=True)
class FrameRateRetarget:
    source_fps: Fraction
    output_fps: Fraction
    frame_stride: int = 1
    rate_mismatch: bool = False

    @property
    def active(self) -> bool:
        return self.frame_stride > 1

    def output_frame_count(self, source_frame_count: int) -> int:
        source_frame_count = int(source_frame_count)
        if source_frame_count <= 0:
            return source_frame_count
        return (source_frame_count + self.frame_stride - 1) // self.frame_stride


def _is_half_rate_source(source_fps: Fraction) -> bool:
    """ffprobe reports r_frame_rate as the smallest rate that ticks every timestamp exactly, so
    a file whose timestamps drift off the 1001/60000 grid arrives as 19001/317 rather than
    60000/1001; match the standard rates by proximity instead of by exact equality."""
    nominal = float(source_fps)
    return any(
        abs(nominal - float(rate)) / float(rate) <= _SOURCE_RATE_TOLERANCE
        for rate in _HALF_RATE_SOURCES
    )


def _measured_rate_confirms(source_fps: Fraction, measured_fps: float) -> bool:
    """A container can advertise twice the real frame rate; ffprobe's r_frame_rate is the
    lowest rate that represents every timestamp exactly, not the rate frames arrive at."""
    if measured_fps <= 0:
        return True
    nominal = float(source_fps)
    return abs(measured_fps - nominal) / nominal <= _MEASURED_RATE_TOLERANCE


def resolve_frame_rate_retarget(
    source_fps: Fraction,
    *,
    enabled: bool,
    measured_fps: float,
) -> FrameRateRetarget:
    source_fps = Fraction(source_fps)
    if not enabled or not _is_half_rate_source(source_fps):
        return FrameRateRetarget(source_fps=source_fps, output_fps=source_fps)
    if not _measured_rate_confirms(source_fps, measured_fps):
        return FrameRateRetarget(
            source_fps=source_fps,
            output_fps=source_fps,
            rate_mismatch=True,
        )
    return FrameRateRetarget(
        source_fps=source_fps,
        output_fps=source_fps / 2,
        frame_stride=2,
    )

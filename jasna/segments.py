from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SegmentRange:
    """A user-visible half-open time range, in seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        start = float(self.start)
        end = float(self.end)
        if start < 0:
            raise ValueError("segment start must be >= 0")
        if end <= start:
            raise ValueError("segment end must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_timestamp(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if any(number < 0 for number in numbers):
        raise ValueError(f"timestamp must be >= 0: {value!r}")
    if len(numbers) == 1:
        return numbers[0]
    if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
        raise ValueError(f"invalid timestamp: {value!r}")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def format_timestamp(seconds: float, *, milliseconds: bool = True) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if milliseconds:
        return f"{base}.{millis:03d}"
    return base


def normalize_segments(
    segments: list[SegmentRange] | tuple[SegmentRange, ...],
    *,
    duration: float | None = None,
) -> tuple[SegmentRange, ...]:
    ordered = sorted(segments)
    if not ordered:
        return ()
    if duration is not None:
        duration = float(duration)
        for segment in ordered:
            if segment.end > duration + 1e-6:
                raise ValueError(
                    f"segment end {format_timestamp(segment.end)} exceeds video duration "
                    f"{format_timestamp(duration)}"
                )

    merged: list[SegmentRange] = []
    for segment in ordered:
        if merged and segment.start <= merged[-1].end + 1e-9:
            previous = merged[-1]
            merged[-1] = SegmentRange(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return tuple(merged)


def parse_segments(spec: str, *, duration: float | None = None) -> tuple[SegmentRange, ...]:
    text = str(spec or "").strip()
    if not text:
        return ()
    parsed: list[SegmentRange] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        start_text, separator, end_text = item.partition("-")
        if not separator or not start_text.strip() or not end_text.strip():
            raise ValueError(
                f"invalid segment {item!r}; expected START-END, for example 01:20-01:35.5"
            )
        parsed.append(SegmentRange(parse_timestamp(start_text), parse_timestamp(end_text)))
    if not parsed:
        raise ValueError("at least one segment is required")
    return normalize_segments(parsed, duration=duration)


def format_segments(segments: tuple[SegmentRange, ...] | list[SegmentRange]) -> str:
    return ",".join(
        f"{format_timestamp(segment.start)}-{format_timestamp(segment.end)}"
        for segment in segments
    )

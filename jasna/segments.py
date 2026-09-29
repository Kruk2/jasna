from __future__ import annotations

from dataclasses import dataclass, replace

from jasna.session_config import RestorationModelName


@dataclass(frozen=True)
class SegmentRestoration:
    """The model that restores a segment; ``ltx_seed`` is set exactly for ``ltx``."""

    model: RestorationModelName
    ltx_seed: int | None

    def __post_init__(self) -> None:
        if (self.model == "ltx") != (self.ltx_seed is not None):
            raise ValueError("an LTX segment needs a seed and only an LTX segment has one")


def job_restoration(model: RestorationModelName, ltx_seed: int) -> SegmentRestoration:
    return SegmentRestoration(model, ltx_seed if model == "ltx" else None)


@dataclass(frozen=True)
class SegmentRange:
    """A user-visible half-open time range, in seconds. ``restoration`` None means the
    job's model; ``resolve_restorations`` fills it in before processing."""

    start: float
    end: float
    restoration: SegmentRestoration | None = None

    def __lt__(self, other: SegmentRange) -> bool:
        return (self.start, self.end) < (other.start, other.end)

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

    painted: list[SegmentRange] = []
    for segment in segments:
        painted = [piece for old in painted for piece in _outside(old, segment)] + [segment]
    merged: list[SegmentRange] = []
    for segment in sorted(painted):
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and segment.restoration == previous.restoration
            and segment.start <= previous.end + 1e-9
        ):
            merged[-1] = replace(previous, end=max(previous.end, segment.end))
        else:
            merged.append(segment)
    return tuple(merged)


def _outside(segment: SegmentRange, cut: SegmentRange) -> list[SegmentRange]:
    """The parts of ``segment`` that ``cut`` does not cover."""
    pieces = []
    if cut.start - segment.start > 1e-9:
        pieces.append(replace(segment, end=min(segment.end, cut.start)))
    if segment.end - cut.end > 1e-9:
        pieces.append(replace(segment, start=max(segment.start, cut.end)))
    return pieces


def resolve_restorations(
    segments: tuple[SegmentRange, ...], default: SegmentRestoration
) -> tuple[SegmentRange, ...]:
    return tuple(
        segment if segment.restoration is not None else replace(segment, restoration=default)
        for segment in segments
    )


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

"""Read-only validation for preserved folder-batch resume candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


class ResumeOutputValidationError(ValueError):
    """Raised when an existing output cannot represent a completed prior run."""


def _canonical_codec(name: str) -> str:
    value = str(name).lower()
    if value in {"h265", "h.265"}:
        return "hevc"
    if value in {"avc", "h.264"}:
        return "h264"
    if value == "av01":
        return "av1"
    return value


def _stream_duration_seconds(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        import av

        return float(container.duration / av.time_base)
    return 0.0


def _stream_start_seconds(container, stream) -> float:
    if stream.start_time is not None and stream.time_base is not None:
        return float(stream.start_time * stream.time_base)
    if container.start_time is not None:
        import av

        return float(container.start_time / av.time_base)
    return 0.0


def _source_contract(source: Path) -> tuple[str, float]:
    import av

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ResumeOutputValidationError(
                    f"Source has no video stream: {source}"
                )
            stream = container.streams.video[0]
            duration = _stream_duration_seconds(container, stream)
            if duration <= 0:
                raise ResumeOutputValidationError(
                    f"Source has no usable video duration: {source}"
                )
            return _canonical_codec(stream.codec_context.name), duration
    except ResumeOutputValidationError:
        raise
    except Exception as exc:
        raise ResumeOutputValidationError(
            f"Could not inspect source video {source}: {exc}"
        ) from exc


def _validate_video_output_compat(
    output: Path,
    *,
    expected_codec: str,
    expected_duration: float,
) -> None:
    """Upstream-base validator used until the shared durability API is present."""

    import av

    try:
        size = output.stat().st_size
    except OSError as exc:
        raise ResumeOutputValidationError(
            f"Completed output is missing: {output}"
        ) from exc
    if not output.is_file() or size <= 0:
        raise ResumeOutputValidationError(
            f"Completed output is empty or not a file: {output}"
        )

    try:
        with av.open(str(output)) as container:
            if not container.streams.video:
                raise ResumeOutputValidationError(
                    f"Completed output has no video stream: {output}"
                )
            stream = container.streams.video[0]
            actual_codec = _canonical_codec(stream.codec_context.name)
            if actual_codec != _canonical_codec(expected_codec):
                raise ResumeOutputValidationError(
                    f"Completed output codec is {actual_codec}, expected "
                    f"{_canonical_codec(expected_codec)}: {output}"
                )

            actual_duration = _stream_duration_seconds(container, stream)
            if actual_duration <= 0:
                raise ResumeOutputValidationError(
                    f"Completed output has no usable video duration: {output}"
                )
            tolerance = max(0.5, min(2.0, float(expected_duration) * 0.001))
            if abs(actual_duration - float(expected_duration)) > tolerance:
                raise ResumeOutputValidationError(
                    f"Completed output duration is {actual_duration:.6f}s, expected "
                    f"{float(expected_duration):.6f}s (+/- {tolerance:.3f}s): {output}"
                )

            stream_start_seconds = _stream_start_seconds(container, stream)
            seek_seconds = stream_start_seconds + max(0.0, actual_duration - 2.0)
            container.seek(
                int(seek_seconds * av.time_base),
                backward=True,
                any_frame=False,
            )
            tail_seconds: float | None = None
            inspected = 0
            for packet in container.demux(stream):
                if packet.size <= 0:
                    continue
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if timestamp is None or packet.time_base is None:
                    continue
                packet_seconds = (
                    float(timestamp * packet.time_base) - stream_start_seconds
                )
                duration_seconds = float((packet.duration or 0) * packet.time_base)
                tail_seconds = max(
                    tail_seconds or packet_seconds,
                    packet_seconds + duration_seconds,
                )
                inspected += 1
                if inspected >= 4096:
                    break
            if tail_seconds is None or tail_seconds < actual_duration - 1.0:
                raise ResumeOutputValidationError(
                    f"Completed output tail is missing or unreadable: {output}"
                )
    except ResumeOutputValidationError:
        raise
    except Exception as exc:
        raise ResumeOutputValidationError(
            f"Completed output is unreadable: {output}: {exc}"
        ) from exc


def _shared_validator() -> Callable[..., None] | None:
    try:
        from jasna.media.splice import validate_video_output
    except ImportError:
        return None
    return validate_video_output


def validate_resume_video_output(
    source: str | Path,
    output: str | Path,
    *,
    configured_codec: str,
) -> None:
    """Accept outputs compatible with either source-copy or configured full render."""

    source_path = Path(source)
    output_path = Path(output)
    source_codec, source_duration = _source_contract(source_path)
    codecs = tuple(
        dict.fromkeys(
            (_canonical_codec(source_codec), _canonical_codec(configured_codec))
        )
    )
    shared = _shared_validator()
    errors: list[str] = []
    for codec in codecs:
        try:
            if shared is None:
                _validate_video_output_compat(
                    output_path,
                    expected_codec=codec,
                    expected_duration=source_duration,
                )
            else:
                shared(
                    output_path,
                    expected_codec=codec,
                    expected_duration=source_duration,
                )
            return
        except Exception as exc:
            errors.append(str(exc))
    raise ResumeOutputValidationError("; ".join(errors))

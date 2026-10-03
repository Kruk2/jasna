"""GUI processing-batch policy and custom-argument parsing."""

from __future__ import annotations

import re


DEFAULT_DETECTION_BATCH_SIZE = 4
SUPPORTED_GUI_BATCH_SIZES = frozenset({1, 4, 8})
_BATCH_SIZE_ARG = re.compile(r"--batch-size(?:\s+|=)(\S+)")


def split_batch_size_custom_arg(value: str | None) -> tuple[int | None, str]:
    """Extract one GUI batch flag without forwarding it to the encoder."""

    raw = str(value or "").strip()
    if not raw or raw.startswith("{"):
        return None, raw

    explicit_batch_size: int | None = None
    encoder_parts: list[str] = []
    for raw_part in raw.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if not part.startswith("--batch-size"):
            encoder_parts.append(part)
            continue

        match = _BATCH_SIZE_ARG.fullmatch(part)
        if match is None:
            raise ValueError(
                "--batch-size must be written as '--batch-size 1', "
                "'--batch-size 4', or '--batch-size 8'; separate encoder "
                "parameters with commas"
            )
        if explicit_batch_size is not None:
            raise ValueError("--batch-size may only be specified once")
        raw_batch_size = match.group(1)
        if raw_batch_size not in {"1", "4", "8"}:
            raise ValueError("--batch-size only supports 1, 4, or 8")
        explicit_batch_size = int(raw_batch_size)

    return explicit_batch_size, ",".join(encoder_parts)


def gui_batch_size_from_custom_args(value: str | None) -> int:
    explicit, _encoder_args = split_batch_size_custom_arg(value)
    return DEFAULT_DETECTION_BATCH_SIZE if explicit is None else explicit


def recommended_detection_batch_size(
    detection_model: str,
    total_vram_bytes: int | None,
) -> int:
    """Compatibility API: hardware telemetry does not change GUI batching."""

    del detection_model, total_vram_bytes
    return DEFAULT_DETECTION_BATCH_SIZE

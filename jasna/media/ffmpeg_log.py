"""FFmpeg's own messages, routed into Python logging as ``libav.<component>`` loggers."""

import av.logging


def log_ffmpeg_errors() -> None:
    """Log FFmpeg errors such as NVENC's reason for a failed encode. Warnings stay off:
    FFmpeg warns about harmless stream quirks in many real-world files."""
    av.logging.set_level(av.logging.ERROR)

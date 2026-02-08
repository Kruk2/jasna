from __future__ import annotations

from pathlib import Path


LogEntry = tuple[str, str, str]  # (timestamp, level, message)


def format_log_entries(entries: list[LogEntry]) -> str:
    lines = []
    for timestamp, level, message in entries:
        lines.append(f"{timestamp} {level.ljust(8)} {message}\n")
    return "".join(lines)


def export_log_entries_txt(path: Path, entries: list[LogEntry]) -> None:
    path.write_text(format_log_entries(entries), encoding="utf-8")


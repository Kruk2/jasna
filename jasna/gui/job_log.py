"""One video's log, saved next to its output for the post-export command's ``{log}``."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

from jasna.gui.log_export import format_log_entries, log_timestamp
from jasna.media.media_files import path_collision_key


def job_log_path(output_path: Path) -> Path:
    log_path = output_path.with_suffix(".log")
    if path_collision_key(log_path) == path_collision_key(output_path):
        return output_path.with_name(output_path.name + ".log")
    return log_path


class JobLog(logging.Handler):
    """Collects log lines in the log panel's format. Lines wait in memory until
    ``save_to`` names the file (the output path is settled only once the job runs),
    then go straight to it."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))
        self.path: Path | None = None
        self._waiting_lines: list[str] = []
        self._file: TextIO | None = None

    def add_entry(self, level: str, message: str) -> None:
        line = format_log_entries([(log_timestamp(datetime.now()), level.upper(), message)])
        with self.lock:
            if self._file is None:
                self._waiting_lines.append(line)
            else:
                self._file.write(line)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.add_entry(record.levelname, self.format(record))
        except Exception:
            self.handleError(record)

    def save_to(self, path: Path) -> None:
        with self.lock:
            self._file = path.open("w", encoding="utf-8", buffering=1)
            self.path = path
            self._file.writelines(self._waiting_lines)
            self._waiting_lines.clear()

    def close(self) -> None:
        with self.lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        super().close()

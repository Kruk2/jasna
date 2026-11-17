from __future__ import annotations

import logging


def should_include_log_entry(*, level: str, filter_level: str) -> bool:
    levels = logging.getLevelNamesMapping()
    return levels.get(level, logging.NOTSET) >= levels[filter_level.upper()]


def runtime_log_level_for_filter(*, filter_level: str) -> int:
    if filter_level == "debug":
        return logging.DEBUG
    return logging.INFO

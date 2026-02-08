from __future__ import annotations


def should_include_log_entry(*, level: str, filter_level: str) -> bool:
    level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    filter_priority = {"debug": 0, "info": 1, "warning": 2, "error": 3}
    min_priority = filter_priority.get(filter_level, 0)
    return level_priority.get(level, 0) >= min_priority


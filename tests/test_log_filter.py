from __future__ import annotations

import logging

from jasna.gui.log_filter import runtime_log_level_for_filter, should_include_log_entry


def test_should_include_log_entry_respects_filter_level() -> None:
    assert should_include_log_entry(level="DEBUG", filter_level="debug")
    assert should_include_log_entry(level="INFO", filter_level="debug")

    assert not should_include_log_entry(level="DEBUG", filter_level="info")
    assert should_include_log_entry(level="INFO", filter_level="info")
    assert should_include_log_entry(level="WARNING", filter_level="info")
    assert should_include_log_entry(level="ERROR", filter_level="info")

    assert not should_include_log_entry(level="INFO", filter_level="warning")
    assert should_include_log_entry(level="WARNING", filter_level="warning")
    assert should_include_log_entry(level="ERROR", filter_level="warning")

    assert not should_include_log_entry(level="WARNING", filter_level="error")
    assert should_include_log_entry(level="ERROR", filter_level="error")


def test_runtime_log_level_for_filter_only_enables_debug_on_debug_filter() -> None:
    assert runtime_log_level_for_filter(filter_level="debug") == logging.DEBUG
    assert runtime_log_level_for_filter(filter_level="info") == logging.INFO
    assert runtime_log_level_for_filter(filter_level="warning") == logging.INFO
    assert runtime_log_level_for_filter(filter_level="error") == logging.INFO


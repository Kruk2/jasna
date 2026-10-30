"""Filesystem locations for GUI state."""

import json
import logging
from pathlib import Path

from jasna import os_utils

logger = logging.getLogger(__name__)


def get_settings_path() -> Path:
    return os_utils.get_user_config_dir("jasna") / "settings.json"


def read_settings() -> dict:
    """settings.json, or {} when it is missing or unreadable."""
    path = get_settings_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read settings from %s, rewriting it: %s", path, e)
        return {}


def update_settings(**values) -> None:
    """Write only the given keys, so keys changed by other writers survive."""
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_settings()
    data.update(values)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Could not save settings to %s: %s", path, e)

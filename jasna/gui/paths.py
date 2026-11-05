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
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        logger.warning("Could not read settings from %s, rewriting it: %s", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("Settings in %s are not an object, rewriting them", path)
        return {}
    return data


def update_settings(**values) -> None:
    """Write only the given keys, so keys changed by other writers survive."""
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_settings()
    data.update(values)
    try:
        with os_utils.atomic_write(path, "wb") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
    except OSError as e:
        logger.warning("Could not save settings to %s: %s", path, e)

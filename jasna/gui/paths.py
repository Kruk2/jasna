"""Filesystem locations for GUI state."""

from pathlib import Path

from jasna import os_utils


def get_settings_path() -> Path:
    return os_utils.get_user_config_dir("jasna") / "settings.json"

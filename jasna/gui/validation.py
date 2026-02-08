import os
from pathlib import Path

from jasna.gui.models import AppSettings


def validate_gui_start(settings: AppSettings) -> list[str]:
    if settings.secondary_restoration != "tvai":
        return []

    errors: list[str] = []

    data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
    model_dir = os.environ.get("TVAI_MODEL_DIR")

    if not data_dir:
        errors.append("TVAI_MODEL_DATA_DIR env var is not set")
    if not model_dir:
        errors.append("TVAI_MODEL_DIR env var is not set")

    if data_dir and not Path(data_dir).is_dir():
        errors.append(f"TVAI_MODEL_DATA_DIR does not point to an existing directory: {data_dir!r}")
    if model_dir and not Path(model_dir).is_dir():
        errors.append(f"TVAI_MODEL_DIR does not point to an existing directory: {model_dir!r}")

    ffmpeg_path = str(settings.tvai_ffmpeg_path)
    if not Path(ffmpeg_path).is_file():
        errors.append(f"TVAI ffmpeg not found: {ffmpeg_path!r}")

    return errors


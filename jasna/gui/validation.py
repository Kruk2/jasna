import os
from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.gui.locales import t


def validate_gui_start(settings: AppSettings) -> list[str]:
    if settings.secondary_restoration != "tvai":
        return []

    errors: list[str] = []

    data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
    model_dir = os.environ.get("TVAI_MODEL_DIR")

    if not data_dir:
        errors.append(t("error_tvai_data_dir_not_set"))
    if not model_dir:
        errors.append(t("error_tvai_model_dir_not_set"))

    if data_dir and not Path(data_dir).is_dir():
        errors.append(t("error_tvai_data_dir_missing", path=data_dir))
    if model_dir and not Path(model_dir).is_dir():
        errors.append(t("error_tvai_model_dir_missing", path=model_dir))

    ffmpeg_path = str(settings.tvai_ffmpeg_path)
    if not Path(ffmpeg_path).is_file():
        errors.append(t("error_tvai_ffmpeg_not_found", path=ffmpeg_path))

    return errors


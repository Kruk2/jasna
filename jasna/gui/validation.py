import os
from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.gui.locales import t


def _required_model_asset_errors(settings: AppSettings) -> list[str]:
    """Return missing core-model errors without constructing a GPU session."""

    # Keep detector filename and vendor-specific suffix selection in the
    # registry. First use can transitively import torch, but resolving weights
    # does not construct a detection model or GPU session.
    from jasna.mosaic.detection_registry import require_detection_model_weights
    from jasna.engine_paths import model_weights_dir

    errors: list[str] = []
    try:
        require_detection_model_weights(str(settings.detection_model))
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))

    restoration_model = (
        model_weights_dir() / "lada_mosaic_restoration_model_generic_v1.2.pth"
    )
    if not restoration_model.is_file():
        errors.append(f"Restoration model weights not found: {restoration_model}")
    return errors


def validate_gui_start(settings: AppSettings) -> list[str]:
    errors: list[str] = []

    from jasna.gui.hardware_policy import split_batch_size_custom_arg

    try:
        split_batch_size_custom_arg(settings.encoder_custom_args)
    except ValueError:
        errors.append(t("error_batch_size_custom_args"))

    from jasna.post_export_action import validate_post_export_action
    try:
        validate_post_export_action(settings.post_export_action, settings.post_export_command)
    except ValueError:
        errors.append(t("error_post_export_command_required"))

    errors.extend(_required_model_asset_errors(settings))

    if settings.secondary_restoration != "tvai":
        return errors

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

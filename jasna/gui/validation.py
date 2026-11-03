import os
from pathlib import Path

from jasna.gui.models import AppSettings, JobItem
from jasna.gui.locales import t
from jasna.media.media_files import is_image
from jasna.segments import job_models


def _uses_ltx(job: JobItem, settings: AppSettings) -> bool:
    return "ltx" in job_models(settings.restoration_model, job.snapshot_segments())


def any_job_uses_ltx(jobs: list[JobItem], settings: AppSettings) -> bool:
    return any(_uses_ltx(job, settings) for job in jobs)


def job_is_resumable(job: JobItem, settings: AppSettings) -> bool:
    """Whether a stopped run keeps the job's finished work: LTX and segment jobs save it,
    whole-video BasicVSR++ jobs and images do not."""
    return not is_image(job.path) and (bool(job.snapshot_segments()) or _uses_ltx(job, settings))


def validate_gui_start(settings: AppSettings, jobs: list[JobItem], *, ltx_available: bool) -> list[str]:
    errors: list[str] = []

    from jasna.post_export_action import validate_post_export_action
    try:
        validate_post_export_action(settings.post_export_action, settings.post_export_command)
    except ValueError:
        errors.append(t("error_post_export_command_required"))

    if not ltx_available:
        needs_ltx = [job.filename for job in jobs if _uses_ltx(job, settings)]
        if needs_ltx:
            errors.append(t("error_ltx_ranges_unavailable", files=", ".join(needs_ltx)))

    if settings.restoration_model == "ltx":
        return errors

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

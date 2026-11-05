from jasna.gui.models import AppSettings, JobItem
from jasna.gui.locales import t
from jasna.media.media_files import is_image
from jasna.os_utils import tvai_setup_problems
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

    # Without auto split an LTX job drops TVAI (ltx_unsupported_options); auto split keeps it.
    if settings.restoration_model == "ltx" and not settings.ltx_auto_split:
        return errors

    if settings.secondary_restoration != "tvai":
        return errors

    return errors + [
        t(f"error_tvai_{problem.key}", path=problem.path) for problem in tvai_setup_problems(str(settings.tvai_ffmpeg_path))
    ]

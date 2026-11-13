"""Whether a newer Jasna release is out: asked at most once a day, the answer remembered in settings.json."""

import logging
import re

from jasna import __version__
from jasna.license_api import protection_available
from jasna.user_settings import read_settings, update_settings

logger = logging.getLogger(__name__)

RELEASE_VERSION = re.compile(r"[0-9]{1,6}\.[0-9]{1,6}\.[0-9]{1,6}")
CHECK_INTERVAL_SECONDS = 24 * 3600
ENABLED_SETTING = "check_for_updates"
_CHECKED_AT_SETTING = "update_checked_at"
_LATEST_SETTING = "latest_release"


def check_due(checked_at, now: float) -> bool:
    """Whether to ask again: never asked, a day has passed, or the clock went back."""
    if type(checked_at) not in (int, float):
        return True
    return not 0 <= now - checked_at < CHECK_INTERVAL_SECONDS


def is_newer(version: str, current: str) -> bool:
    return tuple(map(int, version.split("."))) > tuple(map(int, current.split(".")))


def release_page_url(version: str) -> str:
    return f"https://github.com/Kruk2/jasna/releases/tag/v{version}"


def _remembered_release(settings: dict) -> str | None:
    version = settings.get(_LATEST_SETTING)
    if isinstance(version, str) and RELEASE_VERSION.fullmatch(version):
        return version
    return None


def fetch_latest_release() -> str | None:
    """The newest stable version from the release service; None without it or when it does not answer."""
    if not protection_available():
        return None
    from jasna.protection import activation

    try:
        return activation.latest_release()
    except activation.ActivationServiceError as error:
        logger.info("Could not check for a new Jasna version (%s)", error.code)
        return None


def newer_release(now: float) -> str | None:
    """The release to tell the user about, or None. Asks the service when a check is due (only an answer counts as a
    check); otherwise uses the version the last check found. Never asks when the user turned the check off."""
    settings = read_settings()
    if not settings.get(ENABLED_SETTING, True):
        return None
    latest = _remembered_release(settings)
    if check_due(settings.get(_CHECKED_AT_SETTING), now):
        fetched = fetch_latest_release()
        if fetched is not None:
            latest = fetched
            update_settings(**{_CHECKED_AT_SETTING: now, _LATEST_SETTING: latest})
    if latest is not None and is_newer(latest, __version__):
        return latest
    return None

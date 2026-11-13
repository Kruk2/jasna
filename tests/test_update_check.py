import pytest

from jasna import __version__, update_check
from jasna.user_settings import read_settings, update_settings

NOW = 1_800_000_000.0
DAY = 24 * 3600


@pytest.mark.parametrize(("checked_at", "due"), [
    (None, True),
    ("yesterday", True),
    (True, True),
    (NOW - 60, False),
    (NOW - DAY + 1, False),
    (NOW - DAY, True),
    (NOW + 60, True),
], ids=["never", "not-a-time", "bool", "a-minute-ago", "almost-a-day", "a-day", "clock-went-back"])
def test_a_check_is_due_once_a_day(checked_at, due):
    assert update_check.check_due(checked_at, NOW) is due


@pytest.mark.parametrize(("version", "current", "newer"), [
    ("0.12.4", "0.12.3", True),
    ("0.13.0", "0.12.9", True),
    ("0.12.10", "0.12.9", True),
    ("1.0.0", "0.99.99", True),
    ("0.12.3", "0.12.3", False),
    ("0.9.0", "0.12.0", False),
    ("0.12.2", "0.12.10", False),
])
def test_versions_compare_by_number(version, current, newer):
    assert update_check.is_newer(version, current) is newer


class FakeReleaseService:
    """Answers the update check with ``answer`` (None: no answer); with ``offline`` set, being asked fails the test."""

    def __init__(self):
        self.answer = "99.0.0"
        self.offline = False
        self.asked = 0

    def fetch(self):
        assert not self.offline, "the update check asked the release service"
        self.asked += 1
        return self.answer


@pytest.fixture
def release_service(monkeypatch, user_config_dir):
    service = FakeReleaseService()
    monkeypatch.setattr(update_check, "fetch_latest_release", service.fetch)
    return service


def test_a_due_check_asks_the_service_and_remembers_the_answer(release_service):
    assert update_check.newer_release(NOW) == "99.0.0"

    assert read_settings() == {"update_checked_at": NOW, "latest_release": "99.0.0"}
    assert release_service.asked == 1


def test_within_a_day_the_remembered_release_shows_without_asking(release_service):
    update_settings(update_checked_at=NOW - 60, latest_release="99.0.0")
    release_service.offline = True

    assert update_check.newer_release(NOW) == "99.0.0"


def test_the_running_version_is_not_announced(release_service):
    release_service.answer = __version__

    assert update_check.newer_release(NOW) is None
    assert read_settings()["latest_release"] == __version__


def test_a_failed_check_keeps_the_remembered_release_and_asks_again_next_time(release_service):
    update_settings(update_checked_at=NOW - 2 * DAY, latest_release="99.0.0")
    release_service.answer = None

    assert update_check.newer_release(NOW) == "99.0.0"
    assert read_settings() == {"update_checked_at": NOW - 2 * DAY, "latest_release": "99.0.0"}


def test_a_tampered_remembered_release_is_ignored(release_service):
    update_settings(update_checked_at=NOW - 60, latest_release="99.0.0/../../evil")
    release_service.offline = True

    assert update_check.newer_release(NOW) is None


def test_turned_off_it_never_asks_and_shows_nothing(release_service):
    update_settings(check_for_updates=False, latest_release="99.0.0")
    release_service.offline = True

    assert update_check.newer_release(NOW) is None


def test_without_the_release_service_there_is_no_answer(monkeypatch):
    monkeypatch.setattr(update_check, "protection_available", lambda: False)

    assert update_check.fetch_latest_release() is None


def test_a_failing_release_service_gives_no_answer(monkeypatch, caplog):
    activation = pytest.importorskip("jasna.protection.activation")

    def unreachable():
        raise activation.ActivationServiceError("network_error")

    monkeypatch.setattr(activation, "latest_release", unreachable)

    assert update_check.fetch_latest_release() is None
    assert not [record for record in caplog.records if record.levelname in ("WARNING", "ERROR")]

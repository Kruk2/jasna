from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.gui import file_actions

VIDEO = Path("/media/video.mp4")


def _command(template: list[str]) -> list[str]:
    return [part.format(file=VIDEO, folder=VIDEO.parent) for part in template]


@pytest.mark.parametrize(
    ("system", "command"),
    [
        ("Windows", ["explorer", "{folder}"]),
        ("Linux", ["xdg-open", "{folder}"]),
        ("Darwin", ["open", "{folder}"]),
    ],
)
def test_open_containing_folder_uses_platform_launcher(
    monkeypatch, system: str, command: list[str]
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)

    file_actions.open_containing_folder(VIDEO, parent=MagicMock())

    launch.assert_called_once_with(_command(command))


@pytest.mark.parametrize(
    ("system", "command"),
    [
        ("Windows", ["explorer", "/select,", "{file}"]),
        ("Linux", ["xdg-open", "{folder}"]),
        ("Darwin", ["open", "-R", "{file}"]),
    ],
)
def test_open_containing_folder_selects_file_when_supported(
    monkeypatch, system: str, command: list[str]
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)

    file_actions.open_containing_folder(VIDEO, parent=MagicMock(), select_file=True)

    launch.assert_called_once_with(_command(command))


def test_open_containing_folder_shows_localized_error_on_failure(monkeypatch) -> None:
    error = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: "Linux")
    monkeypatch.setattr(file_actions.subprocess, "Popen", MagicMock(side_effect=OSError("no opener")))
    monkeypatch.setattr(file_actions.messagebox, "showerror", error)
    monkeypatch.setattr(file_actions, "t", lambda key, **values: key.format(**values))
    parent = MagicMock()

    file_actions.open_containing_folder(VIDEO, parent=parent)

    error.assert_called_once_with(
        "open_containing_folder_failed_title",
        "open_containing_folder_failed",
        parent=parent,
    )


@pytest.mark.parametrize(
    ("system", "command"),
    [
        ("Linux", ["xdg-open", "{file}"]),
        ("Darwin", ["open", "{file}"]),
    ],
)
def test_open_file_uses_platform_launcher(monkeypatch, system: str, command: list[str]) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)

    file_actions.open_file(VIDEO, parent=MagicMock())

    launch.assert_called_once_with(_command(command))


def test_open_file_uses_windows_default_application(monkeypatch) -> None:
    startfile = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: "Windows")
    monkeypatch.setattr(file_actions.os, "startfile", startfile, raising=False)

    file_actions.open_file(VIDEO, parent=MagicMock())

    startfile.assert_called_once_with(str(VIDEO))

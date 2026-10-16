from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.gui import file_actions

VIDEO = Path("/media/video.mp4")


def _command(template, path: Path = VIDEO) -> list[str]:
    return [part.format(file=path, folder=path.parent) for part in template]


def _resolve(build_command, path: Path) -> list[str]:
    if callable(build_command):
        return build_command(path)
    return _command(build_command, path)


@pytest.mark.parametrize(
    ("system", "build_command"),
    [
        ("Windows", ["explorer", "{folder}"]),
        ("Linux", ["xdg-open", "{folder}"]),
        ("Darwin", ["open", "{folder}"]),
        ("Windows", lambda path: ["explorer", str(path.parent)]),
        ("Linux", lambda path: ["xdg-open", str(path.parent)]),
        ("Darwin", lambda path: ["open", str(path.parent)]),
    ],
)
def test_open_containing_folder_uses_platform_launcher(
    monkeypatch, system: str, build_command
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)
    path = Path("/media/video.mp4")

    file_actions.open_containing_folder(path, parent=MagicMock())

    launch.assert_called_once_with(_resolve(build_command, path))


@pytest.mark.parametrize(
    ("system", "build_command"),
    [
        ("Windows", ["explorer", "/select,", "{file}"]),
        ("Linux", ["xdg-open", "{folder}"]),
        ("Darwin", ["open", "-R", "{file}"]),
        ("Windows", lambda path: ["explorer", "/select,", str(path)]),
        ("Linux", lambda path: ["xdg-open", str(path.parent)]),
        ("Darwin", lambda path: ["open", "-R", str(path)]),
    ],
)
def test_open_containing_folder_selects_file_when_supported(
    monkeypatch, system: str, build_command
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)
    path = Path("/media/video.mp4")

    file_actions.open_containing_folder(path, parent=MagicMock(), select_file=True)

    launch.assert_called_once_with(_resolve(build_command, path))


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
    ("system", "launcher"),
    [
        ("Linux", ["xdg-open", "{file}"]),
        ("Darwin", ["open", "{file}"]),
        ("Linux", "xdg-open"),
        ("Darwin", "open"),
    ],
)
def test_open_file_uses_platform_launcher(monkeypatch, system: str, launcher) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)
    path = Path("/media/video.mp4")

    file_actions.open_file(path, parent=MagicMock())

    expected = _command(launcher, path) if isinstance(launcher, list) else [launcher, str(path)]
    launch.assert_called_once_with(expected)


def test_open_file_uses_windows_default_application(monkeypatch) -> None:
    startfile = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: "Windows")
    monkeypatch.setattr(file_actions.os, "startfile", startfile, raising=False)
    path = Path("/media/video.mp4")

    file_actions.open_file(path, parent=MagicMock())

    startfile.assert_called_once_with(str(path))

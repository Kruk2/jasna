from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.gui import file_actions


@pytest.mark.parametrize(
    ("system", "command"),
    [
        ("Windows", ["explorer", "/media"]),
        ("Linux", ["xdg-open", "/media"]),
        ("Darwin", ["open", "/media"]),
    ],
)
def test_open_containing_folder_uses_platform_launcher(
    monkeypatch, system: str, command: list[str]
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: system)
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)

    file_actions.open_containing_folder(Path("/media/video.mp4"), parent=MagicMock())

    launch.assert_called_once_with(command)


def test_open_containing_folder_shows_localized_error_on_failure(monkeypatch) -> None:
    error = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: "Linux")
    monkeypatch.setattr(file_actions.subprocess, "Popen", MagicMock(side_effect=OSError("no opener")))
    monkeypatch.setattr(file_actions.messagebox, "showerror", error)
    monkeypatch.setattr(file_actions, "t", lambda key, **values: key.format(**values))
    parent = MagicMock()

    file_actions.open_containing_folder(Path("/media/video.mp4"), parent=parent)

    error.assert_called_once_with(
        "open_containing_folder_failed_title",
        "open_containing_folder_failed",
        parent=parent,
    )

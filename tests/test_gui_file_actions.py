from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.gui import file_actions

VIDEO = Path("/media/video.mp4")


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


@pytest.mark.parametrize("open_path", [file_actions.open_file, file_actions.open_containing_folder])
def test_linux_file_manager_starts_without_bundled_libraries(
    monkeypatch, linux_bundled_library_path, open_path
) -> None:
    launch = MagicMock()
    monkeypatch.setattr(file_actions.platform, "system", lambda: "Linux")
    monkeypatch.setattr(file_actions.subprocess, "Popen", launch)

    open_path(VIDEO, parent=MagicMock())

    assert launch.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == linux_bundled_library_path


from __future__ import annotations

from pathlib import Path

import threading

import numpy as np
import pytest
from PIL import Image

from jasna.gui import video_session
from jasna.gui.image_restore_worker import interactive_output_path
from jasna.gui.interactive_image_restore import InteractiveImageRestoreDialog, _dialog_geometry_for_image
from jasna.gui.locales import t
from jasna.gui.models import AppSettings


def test_dialog_geometry_fits_entire_image_when_screen_allows() -> None:
    window_w, window_h, preview_w, preview_h = _dialog_geometry_for_image((800, 600), (1920, 1080))

    assert preview_w == 800
    assert preview_h == 600
    assert window_w >= preview_w
    assert window_h >= preview_h


def test_dialog_geometry_caps_large_image_to_screen() -> None:
    window_w, window_h, preview_w, preview_h = _dialog_geometry_for_image((4000, 3000), (1920, 1080))

    assert window_w <= 1840
    assert window_h <= 1000
    assert preview_w < 4000
    assert preview_h < 3000


def test_interactive_output_uses_source_dir_when_output_folder_empty(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"x")

    output = interactive_output_path(source, "", "{original}_restored.mp4")

    assert output == tmp_path / "input_restored.png"


def test_interactive_output_uses_gui_output_folder_and_image_suffix(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "input.jpg"
    source.write_bytes(b"x")

    output = interactive_output_path(source, str(output_dir), "{original}_done.mp4")

    assert output == output_dir / "input_done.jpg"


def test_interactive_output_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"x")
    (tmp_path / "input_restored.png").write_bytes(b"existing")
    (tmp_path / "input_restored (1).png").write_bytes(b"existing")

    output = interactive_output_path(source, "", "{original}_restored.mp4")

    assert output == tmp_path / "input_restored (2).png"


def test_interactive_restore_does_not_open_while_a_job_runs(jasna_app, monkeypatch) -> None:
    from tkinter import filedialog

    asked: list[dict] = []
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **options: asked.append(options) or ())
    monkeypatch.setattr(jasna_app._processor, "is_running", lambda: True)

    jasna_app._open_interactive_image_restore()

    assert asked == []


def test_interactive_restore_holds_the_gpu_until_its_worker_stops(jasna_app, monkeypatch) -> None:
    from tkinter import filedialog

    from jasna.gui import interactive_image_restore

    opened: list[dict] = []
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **_options: ("image.png",))
    monkeypatch.setattr(
        interactive_image_restore, "InteractiveImageRestoreDialog", lambda *_args, **options: opened.append(options)
    )

    jasna_app._open_interactive_image_restore()

    assert jasna_app._preview_gpu_busy
    opened[0]["on_stopped"]()
    assert not jasna_app._preview_gpu_busy


@pytest.fixture
def open_image_dialog(tk_root, tmp_path, monkeypatch):
    """Opens a real dialog over one image whose model never loads; returns it and its log."""

    def missing_model(_settings, *, log):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(video_session, "build_image_session", missing_model)
    dialogs = []

    def open_dialog(image_path):
        logs: list[tuple[str, str]] = []
        dialog = InteractiveImageRestoreDialog(
            tk_root,
            [image_path],
            AppSettings(),
            str(tmp_path / "out"),
            "",
            on_log=lambda level, message: logs.append((level, message)),
            on_stopped=lambda: None,
        )
        dialogs.append(dialog)
        return dialog, logs

    yield open_dialog
    for dialog in dialogs:
        dialog._close()


def _run_next_worker_event(dialog) -> None:
    dialog._main_thread._pending.get(timeout=5)()


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    Image.new("RGB", (800, 600)).save(path)
    return path


def test_dialog_stays_hidden_until_the_first_image_size_arrives(open_image_dialog, tmp_path) -> None:
    dialog, logs = open_image_dialog(_image(tmp_path))
    assert dialog.state() == "withdrawn"

    _run_next_worker_event(dialog)
    assert dialog.state() == "normal"
    assert (dialog._preview_w, dialog._preview_h) == (800, 600)

    _run_next_worker_event(dialog)
    assert dialog._status.cget("text") == "model missing"
    assert logs == [("ERROR", "model missing")]


def test_dialog_shows_an_unreadable_first_image_error(open_image_dialog, tmp_path) -> None:
    dialog, logs = open_image_dialog(tmp_path / "missing.png")

    _run_next_worker_event(dialog)

    assert dialog.state() == "normal"
    assert "missing.png" in dialog._status.cget("text")
    assert logs[0][0] == "ERROR"


def test_save_writes_on_a_background_thread_and_reports_the_path(open_image_dialog, tmp_path) -> None:
    dialog, logs = open_image_dialog(_image(tmp_path))
    _run_next_worker_event(dialog)
    _run_next_worker_event(dialog)
    image = np.zeros((3, 6, 8), dtype=np.uint8)
    dialog._finish_render(dialog._token, 0, 0, image, image, image)

    dialog._save.invoke()
    for thread in threading.enumerate():
        if thread.name == "image-restore-save":
            thread.join()
    _run_next_worker_event(dialog)

    saved = tmp_path / "out" / "input_restored.png"
    assert Image.open(saved).size == (8, 6)
    assert dialog._status.cget("text") == t("interactive_saved", path=str(saved))
    assert logs[-1] == ("INFO", f"Wrote {saved}")

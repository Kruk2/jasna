from __future__ import annotations

from pathlib import Path

from jasna.gui.interactive_image_restore import _dialog_geometry_for_image, interactive_output_path


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

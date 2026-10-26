from pathlib import Path

from scripts.rfdetr_sample_viewer import collect_images


def test_collect_images_filters_mask_files_and_extensions(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "frame.jpg").touch()
    (tmp_path / "nested" / "other.PNG").touch()
    (tmp_path / "nested" / "mask.png").touch()
    (tmp_path / "notes.txt").touch()

    result = collect_images(tmp_path, {"jpg", "png"}, include_mask=False)

    assert result == [
        tmp_path / "frame.jpg",
        tmp_path / "nested" / "other.PNG",
    ]


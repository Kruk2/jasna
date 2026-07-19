from pathlib import Path

from scripts.rfdetr_sample_viewer import collect_images, parse_args


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


def test_parse_args_defaults_dynamic_engine_max_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["rfdetr_sample_viewer.py", str(tmp_path)],
    )

    args = parse_args()

    assert args.batch == 4
    assert "rfdetr-v5" in args.non_vr_models
    assert "rfdetr-v6" in args.non_vr_models

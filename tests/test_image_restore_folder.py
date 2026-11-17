from pathlib import Path

from factories import cli_args, cli_image_models, run_cli, write_test_image


def test_image_folder_restores_every_image_with_progress_and_output_pattern(tmp_path: Path, capsys):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.png", "b.jpg"):
        write_test_image(folder / name)
    model_args = cli_args(tmp_path)[4:]

    with cli_image_models():
        run_cli([
            "--input", str(folder), "--output", str(tmp_path / "out"),
            "--output-pattern", "{original}_restored.mp4", *model_args,
        ])

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["a_restored.png", "b_restored.jpg"]
    printed = capsys.readouterr().out
    assert "[1/2] Processing a.png -> a_restored.png" in printed
    assert "[2/2] Processing b.jpg -> b_restored.jpg" in printed


def test_a_single_image_writes_one_file_per_variant(tmp_path: Path):
    image = write_test_image(tmp_path / "photo.png")
    model_args = cli_args(tmp_path)[4:]

    with cli_image_models():
        run_cli(["--input", str(image), "--output", str(tmp_path / "restored.png"), "--sd15-variants", "2", *model_args])

    assert (tmp_path / "restored_v1.png").is_file() and (tmp_path / "restored_v2.png").is_file()

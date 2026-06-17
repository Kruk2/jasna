from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jasna.image_restore import run_image_restoration_folder


def test_image_folder_progress_and_output_pattern(tmp_path: Path, capsys):
    image = tmp_path / "photo.png"
    out_dir = tmp_path / "out"

    def run_jobs(_args, jobs, progress_callback=None):
        assert jobs == [(image, out_dir / "photo_restored.png")]
        assert progress_callback is not None
        progress_callback(1, jobs[0][0], jobs[0][1])

    with patch("jasna.image_restore._run_image_jobs", side_effect=run_jobs):
        run_image_restoration_folder(
            SimpleNamespace(),
            [image],
            out_dir,
            output_pattern="{original}_restored.mp4",
            progress_offset=2,
            progress_total=4,
        )

    assert "[3/4] Processing photo.png -> photo_restored.png" in capsys.readouterr().out

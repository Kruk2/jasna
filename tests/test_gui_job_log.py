from pathlib import Path

import pytest

from jasna.gui.job_log import job_log_path


@pytest.mark.parametrize(
    ("output_name", "log_name"),
    [("clip_restored.mp4", "clip_restored.log"), ("clip", "clip.log"), ("notes.log", "notes.log.log")],
)
def test_job_log_sits_next_to_the_output_and_never_replaces_it(tmp_path: Path, output_name: str, log_name: str) -> None:
    assert job_log_path(tmp_path / output_name) == tmp_path / log_name

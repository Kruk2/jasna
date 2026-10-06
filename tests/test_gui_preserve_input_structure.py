from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import jasna.gui.queue_panel as queue_panel_module
from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.output_paths import OutputPathError
from jasna.gui.processor import Processor
from jasna.gui.queue_panel import QueuePanel


def _processor(output_root: Path, *, preserve_structure: bool = True) -> Processor:
    processor = Processor()
    processor._output_folder = str(output_root)
    processor._output_pattern = "{original}_restored.mp4"
    processor._preserve_input_structure = preserve_structure
    return processor


def test_folder_import_creates_nested_output_parent(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    source = input_root / "season" / "episode" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    output_root = tmp_path / "output"
    expected = output_root / "season" / "episode" / "clip_restored.mp4"
    job = JobItem(source, input_root=input_root)
    processor = _processor(output_root)
    processor._validate_completed_video_output = MagicMock()

    def write_output(_job_id, _input_path, output_path, **_kwargs):
        output_path.write_bytes(b"finished")
        return "full"

    with (
        patch.object(processor, "_run_pipeline", side_effect=write_output),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor.start(
            [job],
            AppSettings(pre_scan_policy="off"),
            str(output_root),
            "{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
            preserve_input_structure=True,
        )
        processor.join()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == expected
    assert expected.is_file()


def test_direct_file_selection_retains_immediate_containing_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "library" / "season" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    queued: list[tuple[Path, Path | None]] = []
    panel = SimpleNamespace(
        add_job=lambda path, *, input_root=None: queued.append((path, input_root))
    )
    monkeypatch.setattr(
        queue_panel_module.filedialog,
        "askopenfilenames",
        lambda **_kwargs: (str(source),),
    )

    QueuePanel._on_add_files(panel)

    assert queued == [(source, tmp_path / "library")]
    assert _processor(tmp_path / "output")._final_output_path(
        JobItem(source, input_root=queued[0][1])
    ) == tmp_path / "output" / "season" / "clip_restored.mp4"


def test_direct_file_drop_retains_immediate_containing_folder(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input" / "season" / "episode" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    queued: list[tuple[Path, Path | None]] = []
    panel = SimpleNamespace(
        _parse_drop_data=lambda _data: [source],
        add_job=lambda path, *, input_root=None: queued.append((path, input_root)),
    )

    QueuePanel._on_file_drop(panel, SimpleNamespace(data="ignored"))

    assert queued == [(source, tmp_path / "input" / "season")]


def test_pending_duplicate_can_upgrade_to_folder_provenance(tmp_path: Path) -> None:
    source = tmp_path / "input" / "season" / "clip.mp4"
    source.parent.mkdir(parents=True)
    job = JobItem(source)
    panel = SimpleNamespace(
        _jobs=[job],
        _refresh_conflicts=MagicMock(),
        _on_jobs_changed=MagicMock(),
    )

    QueuePanel.add_job(panel, source, input_root=tmp_path / "input")

    assert job.input_root == tmp_path / "input"
    panel._refresh_conflicts.assert_called_once_with()
    panel._on_jobs_changed.assert_called_once_with()


def test_duplicate_relative_folder_outputs_are_reserved_per_run(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = first_root / "season" / "clip.mp4"
    second = second_root / "season" / "clip.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    jobs = [JobItem(first, input_root=first_root), JobItem(second, input_root=second_root)]
    output_root = tmp_path / "output"
    processor = _processor(output_root)
    processor._validate_completed_video_output = MagicMock()
    processed: list[Path] = []

    def write_output(_job_id, _input_path, output_path, **_kwargs):
        processed.append(output_path)
        output_path.write_bytes(output_path.name.encode())
        return "full"

    with (
        patch.object(processor, "_run_pipeline", side_effect=write_output),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor.start(
            jobs,
            AppSettings(pre_scan_policy="off", file_conflict="auto_rename"),
            str(output_root),
            "{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
            preserve_input_structure=True,
        )
        processor.join()

    expected = [
        output_root / "season" / "clip_restored.mp4",
        output_root / "season" / "clip_restored (1).mp4",
    ]
    assert [job.status for job in jobs] == [JobStatus.COMPLETED, JobStatus.COMPLETED]
    assert [job.output_path for job in jobs] == expected
    assert processed == expected
    assert all(path.is_file() for path in expected)


@pytest.mark.parametrize("pattern", ("../escape.mp4", r"..\escape.mp4", "/tmp/escape.mp4"))
def test_output_template_cannot_escape_selected_root(
    tmp_path: Path,
    pattern: str,
) -> None:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir()
    source.touch()
    processor = _processor(tmp_path / "output")
    processor._output_pattern = pattern

    with pytest.raises(OutputPathError, match="escapes"):
        processor._final_output_path(JobItem(source, input_root=source.parent))

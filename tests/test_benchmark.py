import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_benchmark_mode_runs_benchmark_cli() -> None:
    with (
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.benchmark.run_benchmark_cli") as run_benchmark_cli,
    ):
        with patch.object(
            sys,
            "argv",
            ["jasna", "--benchmark", "--benchmark-video", "my_video.mp4", "--detection-score-threshold", "0.5"],
        ):
            from jasna.main import main

            main()

        run_benchmark_cli.assert_called_once()
        passed_args = run_benchmark_cli.call_args[0][0]
        assert passed_args.benchmark_video == ["my_video.mp4"]
        assert passed_args.detection_score_threshold == 0.5


def test_benchmark_rfdetr_detection_speed_file_not_found() -> None:
    from jasna.benchmark.rfdetr_detection_speed import _run_single

    import torch

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        _run_single(
            device=torch.device("cuda:0"),
            batch_size=4,
            fp16=True,
            video_path=Path("/nonexistent/assets/test_clip1_1080p.mp4"),
            score_threshold=0.2,
        )


def test_benchmark_harness_runs_three_times_and_takes_median() -> None:
    from jasna.benchmark.harness import run_repeatedly

    call_count = 0

    def mock_benchmark():
        nonlocal call_count
        call_count += 1
        return (1.0 + call_count * 0.1, {"frames": 100})

    with patch("torch.cuda.synchronize"):
        median_duration, result = run_repeatedly(mock_benchmark, runs=3)

    assert call_count == 3
    assert median_duration == 1.2
    assert result == {"frames": 100}

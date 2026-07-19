import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_benchmark_mode_runs_benchmark_cli() -> None:
    with (
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
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


def test_benchmark_cli_preserves_auto_threshold_for_each_benchmark() -> None:
    from jasna.benchmark import run_benchmark_cli

    args = SimpleNamespace(
        device="cuda:0",
        benchmark_video=[],
        batch_size=4,
        fp16=True,
        detection_score_threshold=None,
        restoration_model_path="model.pth",
        compile_basicvsrpp=True,
        benchmark_filter=None,
    )
    with (
        patch("jasna.benchmark.check_required_executables"),
        patch(
            "jasna.benchmark.check_supported_gpu",
            return_value=(True, "gpu"),
        ),
        patch("jasna.benchmark.run_benchmarks") as run_benchmarks,
    ):
        run_benchmark_cli(args)

    assert run_benchmarks.call_args.kwargs["detection_score_threshold"] is None


def test_rfdetr_benchmark_uses_its_model_recommended_threshold(
    tmp_path,
) -> None:
    from jasna.benchmark.rfdetr_detection_speed import (
        benchmark_rfdetr_detection_speed,
    )

    video = tmp_path / "video.mp4"
    video.touch()
    with (
        patch(
            "jasna.benchmark.rfdetr_detection_speed._run_single",
            return_value=(1.0, {"frames": 1}),
        ) as run_single,
        patch(
            "jasna.benchmark.rfdetr_detection_speed.run_repeatedly",
            side_effect=lambda callback, runs: (1.0, callback()[1]),
        ),
    ):
        benchmark_rfdetr_detection_speed(
            device=MagicMock(),
            batch_size=4,
            fp16=True,
            benchmark_videos=[video],
            detection_score_threshold=None,
        )

    assert run_single.call_args.kwargs["score_threshold"] == pytest.approx(0.35)


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

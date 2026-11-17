from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_benchmark_rfdetr_detection_speed_file_not_found() -> None:
    from jasna.benchmark.detection_speed import _run_single

    import torch

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        _run_single(
            "rfdetr-v6",
            device=torch.device("cuda:0"),
            batch_size=4,
            fp16=True,
            video_path=Path("/nonexistent/assets/test_clip1_1080p.mp4"),
            score_threshold=0.2,
        )


@pytest.mark.parametrize(
    ("argv", "environment", "option", "value"),
    [
        ([], {}, "detection_score_threshold", None),
        ([], {}, "batch_size", 4),
        (["--batch-size", "4"], {"JASNA_DECODE_BACKEND": "pyav-sw"}, "decode_backend", "pyav-sw"),
    ],
    ids=["model-threshold", "vendor-batch-size", "decode-backend-from-environment"],
)
def test_benchmark_options_follow_the_cli(monkeypatch, argv, environment, option, value) -> None:
    import jasna.mosaic.detection_registry as registry
    from jasna.benchmark import benchmark_options
    from jasna.cli_parser import build_parser

    monkeypatch.setattr(registry, "is_amd_device", lambda _device: False)
    for name, setting in environment.items():
        monkeypatch.setenv(name, setting)

    assert benchmark_options(build_parser().parse_args(["--benchmark", *argv]))[option] == value


def test_rfdetr_benchmark_uses_its_model_recommended_threshold(
    tmp_path,
) -> None:
    from jasna.benchmark.detection_speed import (
        benchmark_rfdetr_detection_speed,
    )

    video = tmp_path / "video.mp4"
    video.touch()
    with (
        patch(
            "jasna.benchmark.detection_speed._run_single",
            return_value=(1.0, {"frames": 1}),
        ) as run_single,
        patch(
            "jasna.benchmark.detection_speed.run_repeatedly",
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


def test_detection_benchmark_compiles_missing_engine_before_loading(tmp_path) -> None:
    from jasna.benchmark.detection_speed import _run_single

    import torch

    class _CompileRequested(Exception):
        pass

    video = tmp_path / "video.mp4"
    video.touch()
    weights = tmp_path / "model.onnx"
    weights.touch()
    with (
        patch("jasna.benchmark.detection_speed.get_video_meta_data"),
        patch("jasna.benchmark.detection_speed.detection_model_weights_path", return_value=weights),
        patch("jasna.engine_compiler.ensure_engines_compiled", side_effect=_CompileRequested) as compile_engines,
        pytest.raises(_CompileRequested),
    ):
        _run_single(
            "rfdetr-v6",
            device=torch.device("cpu"),
            batch_size=4,
            fp16=True,
            video_path=video,
            score_threshold=0.2,
        )

    request = compile_engines.call_args.args[0]
    assert request.detection
    assert request.detection_model_name == "rfdetr-v6"
    assert request.detection_model_path == str(weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the benchmark synchronizes CUDA")
def test_basicvsrpp_benchmark_times_the_restorer_without_engines(monkeypatch, tmp_path, capsys) -> None:
    import jasna.benchmark.basicvsrpp_restoration as bench
    import jasna.restorer.basicvsrpp_mosaic_restorer as restorer_module
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    generator = BasicVSRPlusPlusNet(mid_channels=8, num_blocks=1, spynet_pretrained=None).eval()
    monkeypatch.setattr(restorer_module, "load_model", lambda *_args: generator)
    monkeypatch.setattr(bench, "CLIP_LENGTH", 3)
    monkeypatch.setattr(bench, "RUNS", 1)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"")

    bench.benchmark_basicvsrpp_restoration(
        device=torch.device("cpu"), fp16=False, restoration_model_path=weights, compile_basicvsrpp=False,
    )

    assert "raw_process median" in capsys.readouterr().out

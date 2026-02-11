import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_cli_creates_stream_on_chosen_device(tmp_path: Path) -> None:
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    output_path = tmp_path / "out.mkv"
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir()
    restoration_path = model_weights / "lada_mosaic_restoration_model_generic_v1.2.pth"
    restoration_path.touch()
    detection_path = model_weights / "rfdetr-v3.onnx"
    detection_path.touch()

    device_capture: list = []
    fake_stream = MagicMock()

    class NoOpDeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def record_device(device):
        device_capture.append(device)
        return NoOpDeviceContext()

    pipeline_capture: dict = {}

    def capture_pipeline(**kwargs):
        pipeline_capture.update(kwargs)
        mock = MagicMock()
        return mock

    with (
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
        patch(
            "jasna.restorer.basicvrspp_tenorrt_compilation.basicvsrpp_startup_policy",
            return_value=False,
        ),
        patch("jasna.pipeline.Pipeline", side_effect=capture_pipeline),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        import torch

        with patch("torch.cuda.device", side_effect=record_device), patch(
            "torch.cuda.Stream", return_value=fake_stream
        ):
            with patch.object(
                sys,
                "argv",
                [
                    "jasna",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda:1",
                    "--restoration-model-path",
                    str(restoration_path),
                    "--detection-model-path",
                    str(detection_path),
                ],
            ):
                from jasna.main import main

                main()

    assert any(d == torch.device("cuda:1") for d in device_capture)
    assert pipeline_capture["device"] == torch.device("cuda:1")

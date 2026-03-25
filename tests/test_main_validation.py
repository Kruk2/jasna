import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _run_main_with_args(tmp_path, extra_args, *, create_input=True, create_detection=True, create_restoration=True):
    input_path = tmp_path / "in.mp4"
    if create_input:
        input_path.touch()
    output_path = tmp_path / "out.mkv"
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir(exist_ok=True)
    restoration_path = model_weights / "restore.pth"
    if create_restoration:
        restoration_path.touch()
    detection_path = model_weights / "det.onnx"
    if create_detection:
        detection_path.touch()

    base_args = [
        "jasna",
        "--input", str(input_path),
        "--output", str(output_path),
        "--restoration-model-path", str(restoration_path),
        "--detection-model-path", str(detection_path),
    ]

    with (
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", return_value=MagicMock()),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        with patch.object(sys, "argv", base_args + extra_args):
            from jasna.main import main
            main()


class TestMainValidation:
    def test_bad_codec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported codec"):
            _run_main_with_args(tmp_path, ["--codec", "h264"])

    def test_batch_size_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="batch-size must be > 0"):
            _run_main_with_args(tmp_path, ["--batch-size", "0"])

    def test_max_clip_size_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max-clip-size must be > 0"):
            _run_main_with_args(tmp_path, ["--max-clip-size", "0"])

    def test_temporal_overlap_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="temporal-overlap must be >= 0"):
            _run_main_with_args(tmp_path, ["--temporal-overlap", "-1"])

    def test_temporal_overlap_ge_max_clip_size_raises(self, tmp_path):
        with pytest.raises(ValueError, match="temporal-overlap must be < --max-clip-size"):
            _run_main_with_args(tmp_path, ["--max-clip-size", "10", "--temporal-overlap", "10"])

    def test_temporal_overlap_too_large_raises(self, tmp_path):
        with pytest.raises(ValueError, match="2\\*--temporal-overlap < --max-clip-size"):
            _run_main_with_args(tmp_path, ["--max-clip-size", "10", "--temporal-overlap", "5"])

    def test_detection_score_threshold_out_of_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="detection-score-threshold must be in"):
            _run_main_with_args(tmp_path, ["--detection-score-threshold", "1.5"])

    def test_missing_input_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run_main_with_args(tmp_path, [], create_input=False)

    def test_missing_restoration_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run_main_with_args(tmp_path, [], create_restoration=False)

    def test_missing_detection_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run_main_with_args(tmp_path, [], create_detection=False)

    def test_no_gpu_exits(self, tmp_path):
        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mkv"

        with (
            patch("jasna.main.check_nvidia_gpu", return_value=(False, "no_cuda")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
        ):
            with patch.object(sys, "argv", [
                "jasna", "--input", str(input_path), "--output", str(output_path),
            ]):
                from jasna.main import main
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_low_compute_capability_exits(self, tmp_path):
        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mkv"

        with (
            patch("jasna.main.check_nvidia_gpu", return_value=(False, ("GPU", 5, 0))),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
        ):
            with patch.object(sys, "argv", [
                "jasna", "--input", str(input_path), "--output", str(output_path),
            ]):
                from jasna.main import main
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_valid_args_succeed(self, tmp_path):
        _run_main_with_args(tmp_path, [])

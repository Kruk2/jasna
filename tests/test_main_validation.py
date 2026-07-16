import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.segments import SegmentRange


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
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "590.18")),
        patch("jasna.main.check_required_executables"),        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", return_value=MagicMock()) as pipeline_cls,
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        with patch.object(sys, "argv", base_args + extra_args):
            from jasna.main import main
            main()
    return pipeline_cls


class TestMainValidation:
    def test_segments_auto_select_source_codec_and_reach_pipeline(self, tmp_path):
        metadata = MagicMock(codec_name="h264", duration=10.0)
        splice_plan = MagicMock()
        with (
            patch("jasna.media.get_video_meta_data", return_value=metadata),
            patch("jasna.media.splice.validate_smart_render"),
            patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
            patch("jasna.media.splice.build_splice_plan", return_value=splice_plan),
        ):
            pipeline_cls = _run_main_with_args(tmp_path, ["--segments", "1-2"])

        assert pipeline_cls.call_args.kwargs["codec"] == "h264"
        assert pipeline_cls.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
        assert pipeline_cls.call_args.kwargs["splice_plan"] is splice_plan

    def test_segments_reject_explicit_codec_mismatch(self, tmp_path):
        metadata = MagicMock(codec_name="h264", duration=10.0)
        with patch("jasna.media.get_video_meta_data", return_value=metadata):
            with pytest.raises(SystemExit):
                _run_main_with_args(
                    tmp_path,
                    ["--segments", "1-2", "--codec", "hevc"],
                )

    def test_bad_codec_rejected_by_argparse(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--codec", "vp9"])

    def test_h264_and_av1_codecs_accepted(self, tmp_path):
        _run_main_with_args(tmp_path, ["--codec", "h264"])
        _run_main_with_args(tmp_path, ["--codec", "av1"])

    def test_codec_case_normalized(self, tmp_path):
        _run_main_with_args(tmp_path, ["--codec", "AV1"])

    def test_codec_specific_encoder_settings_validated(self, tmp_path):
        with pytest.raises(ValueError, match="for codec av1.*profile"):
            _run_main_with_args(tmp_path, ["--codec", "av1", "--encoder-settings", "profile=main"])

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
            patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
            patch("jasna.main.check_nvidia_gpu", return_value=(False, "no_cuda")),
            patch("jasna.main.check_required_executables"),            patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
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
            patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
            patch("jasna.main.check_nvidia_gpu", return_value=(False, ("GPU", 5, 0))),
            patch("jasna.main.check_required_executables"),            patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
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

    def test_retarget_high_fps_rejected_for_streaming(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--stream", "--retarget-high-fps"])

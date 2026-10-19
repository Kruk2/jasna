import re
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
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
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
            patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
            patch("jasna.media.splice.validate_smart_render"),
            patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
            patch("jasna.media.splice.build_splice_plan", return_value=splice_plan),
        ):
            pipeline_cls = _run_main_with_args(tmp_path, ["--segments", "1-2"])

        assert pipeline_cls.call_args.kwargs["config"].codec == "h264"
        assert pipeline_cls.call_args.kwargs["config"].encoder_settings == {"cq": 25}
        assert pipeline_cls.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
        assert pipeline_cls.call_args.kwargs["splice_plan"] is splice_plan

    def test_ltx_accepts_segments(self, tmp_path):
        metadata = MagicMock(codec_name="h264", duration=10.0)
        with (
            patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
            patch("jasna.media.splice.validate_smart_render"),
            patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
            patch("jasna.media.splice.build_splice_plan", return_value=MagicMock()),
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.model_files.LtxModelFiles.from_dir", return_value=MagicMock()),
            patch("jasna.ltx.model_files.missing_downloads", return_value=[]),
        ):
            pipeline_cls = _run_main_with_args(
                tmp_path, ["--segments", "1-2", "--restoration-model-name", "ltx", "--ltx-seed", "3"]
            )

        config = pipeline_cls.call_args.kwargs["config"]
        assert (config.restoration_model_name, config.ltx_seed) == ("ltx", 3)
        assert pipeline_cls.call_args.kwargs["segments"] == (SegmentRange(1, 2),)

    def test_ltx_auto_split_reaches_pipeline_with_the_split_source_codec(self, tmp_path):
        from jasna.segments import SegmentRestoration
        from jasna.session_factory import AutoSplit

        segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 3)),)
        split = AutoSplit("h264", segments, MagicMock())
        with (
            patch("jasna.session_factory.ltx_auto_split", return_value=split) as auto_split,
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.model_files.LtxModelFiles.from_dir", return_value=MagicMock()),
            patch("jasna.ltx.model_files.missing_downloads", return_value=[]),
        ):
            pipeline_cls = _run_main_with_args(
                tmp_path,
                ["--restoration-model-name", "ltx", "--ltx-seed", "3", "--ltx-auto-split", "12", "--codec", "av1"],
            )

        assert auto_split.call_args.args[0].ltx_auto_split_percent == 12.0
        kwargs = pipeline_cls.call_args.kwargs
        assert kwargs["config"].codec == "h264"
        assert kwargs["config"].encoder_settings == {"cq": 25}
        assert kwargs["segments"] == segments
        assert kwargs["splice_plan"] is split.splice_plan

    def test_ltx_auto_split_without_a_split_runs_the_whole_video(self, tmp_path):
        with (
            patch("jasna.session_factory.ltx_auto_split", return_value=None),
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.model_files.LtxModelFiles.from_dir", return_value=MagicMock()),
            patch("jasna.ltx.model_files.missing_downloads", return_value=[]),
        ):
            pipeline_cls = _run_main_with_args(tmp_path, ["--restoration-model-name", "ltx", "--ltx-auto-split", "12"])

        assert pipeline_cls.call_args.kwargs["segments"] is None
        assert pipeline_cls.call_args.kwargs["config"].codec == "hevc"

    def test_ltx_auto_split_needs_ltx(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--ltx-auto-split", "12"])

    def test_ltx_auto_split_rejects_segments(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(
                tmp_path, ["--restoration-model-name", "ltx", "--ltx-auto-split", "12", "--segments", "1-2"]
            )

    def test_ltx_trial_needs_no_model_files_or_download(self, tmp_path, capsys):
        with (
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.model_files.LtxModelFiles.from_dir", side_effect=AssertionError("reads model files")),
            patch("jasna.ltx.model_files.missing_downloads", side_effect=AssertionError("offers a download")),
        ):
            pipeline_cls = _run_main_with_args(
                tmp_path, ["--restoration-model-name", "ltx-undistilled", "--ltx-trial"], create_restoration=False
            )

        config = pipeline_cls.call_args.kwargs["config"]
        assert (config.restoration_model_name, config.ltx_model, config.ltx_trial) == ("ltx", "undistilled", True)
        assert "LTX trial mode: output is not restored" in capsys.readouterr().out

    def test_ltx_offers_the_download_when_its_model_folder_is_missing(self, tmp_path):
        with patch("jasna.ltx.model_files.missing_downloads", side_effect=RuntimeError("offers a download")):
            with pytest.raises(RuntimeError, match="offers a download"):
                _run_main_with_args(tmp_path, ["--restoration-model-name", "ltx"], create_restoration=False)

    def test_ltx_trial_rejects_basicvsrpp(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--ltx-trial"])

    @pytest.mark.parametrize(
        "ltx_args",
        [["--ltx-fast"], ["--ltx-large-canvas"], ["--no-experimental-ltx-less-overlap"], ["--ltx-seed", "7"], ["--ltx-trial"], ["--ltx-auto-split", "12"]],
    )
    def test_basicvsrpp_rejects_ltx_options(self, tmp_path, capsys, ltx_args):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ltx_args)
        assert "needs --restoration-model-name ltx, ltx-4step, ltx-8step or ltx-undistilled" in capsys.readouterr().err

    @pytest.mark.parametrize("model", ["ltx", "ltx-undistilled"])
    def test_ltx_rejects_disabling_basicvsrpp_compile(self, tmp_path, capsys, model):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--restoration-model-name", model, "--no-compile-basicvsrpp"])
        assert "--compile-basicvsrpp needs --restoration-model-name basicvsrpp" in capsys.readouterr().err

    def test_basicvsrpp_accepts_disabling_its_compile(self, tmp_path):
        pipeline_cls = _run_main_with_args(tmp_path, ["--no-compile-basicvsrpp"])

        assert pipeline_cls.call_args.kwargs["config"].compile_basicvsrpp is False

    def test_ltx_rejects_streaming(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--stream", "--restoration-model-name", "ltx"])

    def test_segments_reject_explicit_codec_mismatch(self, tmp_path):
        metadata = MagicMock(codec_name="h264", duration=10.0)
        with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
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

    def test_codec_specific_encoder_settings_validated(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--codec", "av1", "--encoder-settings", "profile=main"])
        assert re.search("for codec av1.*profile", capsys.readouterr().err)

    def test_batch_size_zero_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--batch-size", "0"])
        assert re.search("Batch size must be > 0", capsys.readouterr().err)

    def test_max_clip_size_zero_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--max-clip-size", "0"])
        assert re.search("Max clip size must be > 0", capsys.readouterr().err)

    def test_temporal_overlap_negative_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--temporal-overlap", "-1"])
        assert re.search("Temporal overlap must be >= 0", capsys.readouterr().err)

    def test_temporal_overlap_ge_max_clip_size_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--max-clip-size", "10", "--temporal-overlap", "10"])
        assert re.search("Temporal overlap must satisfy", capsys.readouterr().err)

    def test_temporal_overlap_too_large_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--max-clip-size", "10", "--temporal-overlap", "5"])
        assert re.search("Temporal overlap must satisfy", capsys.readouterr().err)

    def test_detection_score_threshold_out_of_range_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--detection-score-threshold", "1.5"])
        assert re.search("Detection score threshold must be in", capsys.readouterr().err)

    @pytest.mark.parametrize("value", ["1.5", "-0.1"])
    def test_sharpen_out_of_range_raises(self, tmp_path, value, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--sharpen", value])
        assert re.search("Sharpening must be in", capsys.readouterr().err)

    def test_sharpen_is_forwarded_to_the_pipeline(self, tmp_path):
        pipeline_cls = _run_main_with_args(tmp_path, ["--sharpen", "0.4"])
        assert pipeline_cls.call_args.kwargs["config"].sharpen_strength == 0.4

    def test_missing_input_file_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, [], create_input=False)
        assert "input not found" in capsys.readouterr().err

    def test_missing_restoration_model_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, [], create_restoration=False)
        assert "restoration model not found" in capsys.readouterr().err

    def test_missing_detection_model_raises(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, [], create_detection=False)
        assert "Detection model weights not found" in capsys.readouterr().err

    def test_no_gpu_exits(self, tmp_path):
        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mkv"

        with (
            patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
            patch("jasna.main.check_supported_gpu", return_value=(False, "no_cuda")),
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
            patch("jasna.main.check_supported_gpu", return_value=(False, ("GPU", 5, 0))),
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

    def test_fmp4_rejected_for_streaming(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--stream", "--fmp4"])

    def test_fmp4_rejected_with_segments(self, tmp_path):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--segments", "10-20", "--fmp4"])


class TestCliErrors:
    def test_usage_error_is_one_short_message(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main_with_args(tmp_path, ["--stream", "--fmp4"])
        assert exc.value.code == 2
        assert capsys.readouterr().err.splitlines() == [
            "jasna: error: --fmp4 is only supported for offline exports",
            "Run 'jasna --help' to see all options.",
        ]

    def test_argument_mistakes_are_reported_before_the_gpu_check(self, tmp_path, capsys):
        with patch("jasna.main._check_system", side_effect=AssertionError("checked the GPU first")):
            with pytest.raises(SystemExit):
                _run_main_with_args(tmp_path, ["--restoration-model-name", "ltx-undistilled", "--denoise", "low"])
        assert "--restoration-model-name ltx-undistilled does not support --denoise" in capsys.readouterr().err

    def test_missing_input_is_reported_before_the_gpu_check(self, tmp_path, capsys):
        with patch("jasna.main._check_system", side_effect=AssertionError("checked the GPU first")):
            with pytest.raises(SystemExit):
                _run_main_with_args(tmp_path, [], create_input=False)
        assert "input not found" in capsys.readouterr().err

    def test_post_export_command_without_command_is_a_usage_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_args(tmp_path, ["--post-export-action", "command"])
        assert "--post-export-command is required" in capsys.readouterr().err


@pytest.mark.parametrize("hard_disk", [True, False])
def test_cli_warns_when_installed_on_a_hard_disk(capsys, hard_disk):
    import argparse

    from jasna import main

    with patch.object(main, "check_ascii_install_path", return_value=(True, "C:/jasna")), \
         patch.object(main, "check_required_executables"), \
         patch.object(main, "check_supported_gpu", return_value=(True, "ok")), \
         patch.object(main, "check_gpu_driver_version", return_value=(True, "620")), \
         patch("jasna.accelerator.is_nvidia_device", return_value=False), \
         patch.object(main, "install_on_hard_disk", return_value=hard_disk):
        main._check_system(argparse.Namespace(device="cuda:0"))
    assert ("installed on a hard disk drive (HDD)" in capsys.readouterr().out) is hard_disk


def test_ltx_cli_names_pick_the_two_four_and_eight_step_models():
    from jasna.main import CLI_RESTORATION_MODELS

    assert CLI_RESTORATION_MODELS["ltx"] == ("ltx", "distilled2")
    assert CLI_RESTORATION_MODELS["ltx-4step"] == ("ltx", "distilled4")
    assert "ltx-2step" not in CLI_RESTORATION_MODELS
    assert CLI_RESTORATION_MODELS["ltx-8step"] == ("ltx", "distilled")

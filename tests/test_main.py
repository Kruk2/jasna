import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from jasna.main import build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_files(tmp_path, *, create_input=True, create_detection=True, create_restoration=True):
    input_path = tmp_path / "in.mp4"
    if create_input:
        input_path.touch()
    output_path = tmp_path / "out.mp4"
    restoration_path = tmp_path / "restore.pth"
    if create_restoration:
        restoration_path.touch()
    detection_path = tmp_path / "det.onnx"
    if create_detection:
        detection_path.touch()
    return input_path, output_path, restoration_path, detection_path


def _base_argv(input_path, output_path, restoration_path, detection_path, extra=None):
    args = [
        "jasna",
        "--input", str(input_path),
        "--output", str(output_path),
        "--restoration-model-path", str(restoration_path),
        "--detection-model-path", str(detection_path),
    ]
    if extra:
        args.extend(extra)
    return args


@contextmanager
def _main_patches(pipeline_side_effect=None):
    mock_pipeline_cls = MagicMock()
    if pipeline_side_effect:
        mock_pipeline_cls.side_effect = pipeline_side_effect
    else:
        mock_pipeline_cls.return_value = MagicMock()

    with (
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", mock_pipeline_cls),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        yield mock_pipeline_cls


def _run_main(argv, pipeline_side_effect=None):
    with _main_patches(pipeline_side_effect) as pipeline_cls:
        with patch.object(sys, "argv", argv):
            from jasna.main import main
            main()
    return pipeline_cls


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_defaults(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
        assert args.batch_size == 4
        assert args.device == "cuda:0"
        assert args.fp16 is True
        assert args.log_level == "error"
        assert args.max_clip_size == 90
        assert args.temporal_overlap == 8
        assert args.enable_crossfade is True
        assert args.denoise == "none"
        assert args.denoise_step == "after_primary"
        assert args.secondary_restoration == "none"
        assert args.codec == "hevc"
        assert args.encoder_settings == ""
        assert args.stream is False
        assert args.stream_port == 8765
        assert args.stream_segment_duration == 4.0
        assert args.no_browser is False
        assert args.detection_model == "rfdetr-v5"
        assert args.detection_score_threshold == 0.25
        assert args.benchmark is False

    def test_no_fp16(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4", "--no-fp16"])
        assert args.fp16 is False

    def test_no_enable_crossfade(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4", "--no-enable-crossfade"])
        assert args.enable_crossfade is False

    def test_stream_flag(self):
        args = build_parser().parse_args(["--stream"])
        assert args.stream is True

    def test_benchmark_flag(self):
        args = build_parser().parse_args(["--benchmark"])
        assert args.benchmark is True

    def test_tvai_defaults(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
        assert args.tvai_model == "iris-2"
        assert args.tvai_scale == 4
        assert args.tvai_workers == 2

    def test_rtx_defaults(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
        assert args.rtx_scale == 4
        assert args.rtx_quality == "high"
        assert args.rtx_denoise == "medium"
        assert args.rtx_deblur == "none"


# ---------------------------------------------------------------------------
# Benchmark path
# ---------------------------------------------------------------------------

class TestBenchmarkPath:
    def test_benchmark_dispatches(self, tmp_path):
        with patch("jasna.benchmark.run_benchmark_cli") as mock_bench:
            with patch.object(sys, "argv", ["jasna", "--benchmark"]):
                from jasna.main import main
                main()
            mock_bench.assert_called_once()


# ---------------------------------------------------------------------------
# Output path respected
# ---------------------------------------------------------------------------

class TestOutputPath:
    def test_explicit_output_forwarded_to_pipeline(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_kwargs = {}

        def capture(**kw):
            pipeline_kwargs.update(kw)
            return MagicMock()

        with _main_patches(pipeline_side_effect=capture):
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det)):
                from jasna.main import main
                main()

        assert pipeline_kwargs["output_video"] == out

    def test_streaming_without_output_uses_derived(self, tmp_path):
        inp, _, rest, det = _make_model_files(tmp_path)
        pipeline_kwargs = {}

        def capture(**kw):
            pipeline_kwargs.update(kw)
            mock = MagicMock()
            return mock

        with _main_patches(pipeline_side_effect=capture):
            argv = [
                "jasna",
                "--stream",
                "--input", str(inp),
                "--restoration-model-path", str(rest),
                "--detection-model-path", str(det),
                "--no-browser",
            ]
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                main()

        assert pipeline_kwargs["output_video"] == inp.with_stem(inp.stem + "_out")


# ---------------------------------------------------------------------------
# Secondary restorers
# ---------------------------------------------------------------------------

class TestSecondaryRestorers:
    def test_none_secondary(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_cls = _run_main(_base_argv(inp, out, rest, det))
        pipeline_instance = pipeline_cls.return_value
        pipeline_instance.run.assert_called_once()

    def test_tvai_secondary(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_tvai = MagicMock()
        with patch("jasna.restorer.tvai_secondary_restorer.TvaiSecondaryRestorer", mock_tvai):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "tvai",
                "--tvai-ffmpeg-path", "fake_ffmpeg.exe",
                "--tvai-model", "prob-4",
                "--tvai-scale", "2",
                "--tvai-workers", "1",
                "--tvai-args", "noise=5",
            ]))
        mock_tvai.assert_called_once()
        kw = mock_tvai.call_args
        assert kw.kwargs["ffmpeg_path"] == "fake_ffmpeg.exe"
        assert kw.kwargs["scale"] == 2
        assert kw.kwargs["num_workers"] == 1
        assert "model=prob-4:scale=2:noise=5" in kw.kwargs["tvai_args"]

    def test_unet4x_secondary(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_unet = MagicMock()
        with patch("jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer", mock_unet):
            _run_main(_base_argv(inp, out, rest, det, ["--secondary-restoration", "unet-4x"]))
        mock_unet.assert_called_once()

    def test_rtx_super_res_secondary(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_rtx = MagicMock()
        with patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer", mock_rtx):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "rtx-super-res",
                "--rtx-scale", "2",
                "--rtx-quality", "ultra",
                "--rtx-denoise", "high",
                "--rtx-deblur", "low",
            ]))
        mock_rtx.assert_called_once()
        kw = mock_rtx.call_args.kwargs
        assert kw["scale"] == 2
        assert kw["quality"] == "ultra"
        assert kw["denoise"] == "high"
        assert kw["deblur"] == "low"

    def test_rtx_denoise_none_passes_none(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_rtx = MagicMock()
        with patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer", mock_rtx):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "rtx-super-res",
                "--rtx-denoise", "none",
                "--rtx-deblur", "none",
            ]))
        kw = mock_rtx.call_args.kwargs
        assert kw["denoise"] is None
        assert kw["deblur"] is None


# ---------------------------------------------------------------------------
# Detection model discovery
# ---------------------------------------------------------------------------

class TestDetectionModelDiscovery:
    def test_explicit_detection_path_skips_discovery(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        with patch("jasna.mosaic.detection_registry.discover_available_detection_models") as disc:
            _run_main(_base_argv(inp, out, rest, det))
        disc.assert_not_called()

    def test_auto_discovery_warns_when_model_missing(self, tmp_path, capsys):
        inp, out, rest, _ = _make_model_files(tmp_path, create_detection=False)
        det = tmp_path / "model_weights" / "rfdetr-v5.onnx"
        (tmp_path / "model_weights").mkdir(exist_ok=True)
        det.touch()

        with patch("jasna.mosaic.detection_registry.discover_available_detection_models", return_value=["rfdetr-v3"]):
            with patch("jasna.mosaic.detection_registry.detection_model_weights_path", return_value=det):
                _run_main([
                    "jasna",
                    "--input", str(inp),
                    "--output", str(out),
                    "--restoration-model-path", str(rest),
                ])
        captured = capsys.readouterr()
        assert "not found in model_weights" in captured.out

    def test_auto_discovery_no_warning_when_model_available(self, tmp_path, capsys):
        inp, out, rest, _ = _make_model_files(tmp_path, create_detection=False)
        det = tmp_path / "model_weights" / "rfdetr-v5.onnx"
        (tmp_path / "model_weights").mkdir(exist_ok=True)
        det.touch()

        with patch("jasna.mosaic.detection_registry.discover_available_detection_models", return_value=["rfdetr-v5"]):
            with patch("jasna.mosaic.detection_registry.detection_model_weights_path", return_value=det):
                _run_main([
                    "jasna",
                    "--input", str(inp),
                    "--output", str(out),
                    "--restoration-model-path", str(rest),
                ])
        captured = capsys.readouterr()
        assert "not found in model_weights" not in captured.out


# ---------------------------------------------------------------------------
# Denoise / crossfade / fp16 forwarding
# ---------------------------------------------------------------------------

class TestArgForwarding:
    def _capture_run(self, tmp_path, extra_args):
        inp, out, rest, det = _make_model_files(tmp_path)
        captured = {}

        def capture_pipeline(**kw):
            captured.update(kw)
            return MagicMock()

        rest_pipeline_captured = {}
        orig_rp = None

        def capture_rp(**kw):
            rest_pipeline_captured.update(kw)
            return MagicMock()

        with _main_patches(pipeline_side_effect=capture_pipeline):
            with patch("jasna.restorer.restoration_pipeline.RestorationPipeline", side_effect=capture_rp):
                with patch.object(sys, "argv", _base_argv(inp, out, rest, det, extra_args)):
                    from jasna.main import main
                    main()

        return captured, rest_pipeline_captured

    def test_denoise_forwarded(self, tmp_path):
        _, rp = self._capture_run(tmp_path, ["--denoise", "high", "--denoise-step", "after_secondary"])
        from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
        assert rp["denoise_strength"] == DenoiseStrength.HIGH
        assert rp["denoise_step"] == DenoiseStep.AFTER_SECONDARY

    def test_crossfade_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--no-enable-crossfade"])
        assert pipe["enable_crossfade"] is False

    def test_fp16_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--no-fp16"])
        assert pipe["fp16"] is False

    def test_encoder_settings_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--encoder-settings", "cq=22,lookahead=32"])
        assert pipe["encoder_settings"] == {"cq": 22, "lookahead": 32}

    def test_batch_size_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--batch-size", "8"])
        assert pipe["batch_size"] == 8

    def test_max_clip_size_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--max-clip-size", "120"])
        assert pipe["max_clip_size"] == 120

    def test_temporal_overlap_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--temporal-overlap", "4", "--max-clip-size", "90"])
        assert pipe["temporal_overlap"] == 4

    def test_no_progress_forwarded(self, tmp_path):
        pipe, _ = self._capture_run(tmp_path, ["--no-progress"])
        assert pipe["disable_progress"] is True


# ---------------------------------------------------------------------------
# Streaming paths
# ---------------------------------------------------------------------------

class TestStreamingPaths:
    def test_streaming_with_input_calls_run_streaming(self, tmp_path):
        inp, _, rest, det = _make_model_files(tmp_path)
        pipeline_mock = MagicMock()

        def make_pipeline(**kw):
            return pipeline_mock

        with _main_patches(pipeline_side_effect=make_pipeline):
            argv = [
                "jasna",
                "--stream",
                "--input", str(inp),
                "--restoration-model-path", str(rest),
                "--detection-model-path", str(det),
                "--no-browser",
                "--stream-port", "9999",
                "--stream-segment-duration", "2.0",
            ]
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                main()

        pipeline_mock.run_streaming.assert_called_once_with(port=9999, segment_duration=2.0)
        pipeline_mock.close.assert_called_once()

    def test_streaming_with_input_opens_browser(self, tmp_path):
        inp, _, rest, det = _make_model_files(tmp_path)

        with _main_patches():
            with patch("webbrowser.open") as wb_open:
                argv = [
                    "jasna",
                    "--stream",
                    "--input", str(inp),
                    "--restoration-model-path", str(rest),
                    "--detection-model-path", str(det),
                    "--stream-port", "8888",
                ]
                with patch.object(sys, "argv", argv):
                    from jasna.main import main
                    main()

            wb_open.assert_called_once_with("http://localhost:8888/")

    def test_streaming_with_input_no_browser(self, tmp_path):
        inp, _, rest, det = _make_model_files(tmp_path)

        with _main_patches():
            with patch("webbrowser.open") as wb_open:
                argv = [
                    "jasna",
                    "--stream",
                    "--input", str(inp),
                    "--restoration-model-path", str(rest),
                    "--detection-model-path", str(det),
                    "--no-browser",
                ]
                with patch.object(sys, "argv", argv):
                    from jasna.main import main
                    main()

            wb_open.assert_not_called()

    def test_serverless_streaming_loop(self, tmp_path):
        _, _, rest, det = _make_model_files(tmp_path)
        pipeline_mock = MagicMock()

        def make_pipeline(**kw):
            return pipeline_mock

        mock_hls = MagicMock()
        video_path = tmp_path / "uploaded.mp4"
        video_path.touch()
        call_count = 0

        def wait_for_video_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            return video_path

        mock_hls.wait_for_video.side_effect = wait_for_video_side_effect

        with _main_patches(pipeline_side_effect=make_pipeline):
            with patch("jasna.streaming.HlsStreamingServer", return_value=mock_hls):
                argv = [
                    "jasna",
                    "--stream",
                    "--restoration-model-path", str(rest),
                    "--detection-model-path", str(det),
                    "--no-browser",
                    "--stream-port", "7777",
                    "--stream-segment-duration", "3.0",
                ]
                with patch.object(sys, "argv", argv):
                    from jasna.main import main
                    main()

        mock_hls.start.assert_called_once()
        mock_hls.stop.assert_called_once()
        mock_hls.unload_video.assert_called_once()
        pipeline_mock.run_streaming.assert_called_once()
        assert pipeline_mock.input_video == video_path

    def test_serverless_streaming_opens_browser(self, tmp_path):
        _, _, rest, det = _make_model_files(tmp_path)

        mock_hls = MagicMock()
        mock_hls.wait_for_video.side_effect = KeyboardInterrupt

        with _main_patches():
            with patch("jasna.streaming.HlsStreamingServer", return_value=mock_hls):
                with patch("webbrowser.open") as wb_open:
                    argv = [
                        "jasna",
                        "--stream",
                        "--restoration-model-path", str(rest),
                        "--detection-model-path", str(det),
                        "--stream-port", "6666",
                    ]
                    with patch.object(sys, "argv", argv):
                        from jasna.main import main
                        main()

                wb_open.assert_called_once_with("http://localhost:6666/")

    def test_serverless_streaming_colorspace_error_in_loop(self, tmp_path, capsys):
        _, _, rest, det = _make_model_files(tmp_path)
        pipeline_mock = MagicMock()

        def make_pipeline(**kw):
            return pipeline_mock

        from jasna.media import UnsupportedColorspaceError
        call_count = 0

        def wait_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            return tmp_path / "vid.mp4"

        pipeline_mock.run_streaming.side_effect = UnsupportedColorspaceError("bt2020 not supported")
        mock_hls = MagicMock()
        mock_hls.wait_for_video.side_effect = wait_side_effect

        with _main_patches(pipeline_side_effect=make_pipeline):
            with patch("jasna.streaming.HlsStreamingServer", return_value=mock_hls):
                argv = [
                    "jasna",
                    "--stream",
                    "--restoration-model-path", str(rest),
                    "--detection-model-path", str(det),
                    "--no-browser",
                ]
                with patch.object(sys, "argv", argv):
                    from jasna.main import main
                    main()

        captured = capsys.readouterr()
        assert "bt2020 not supported" in captured.out
        mock_hls.unload_video.assert_called_once()


# ---------------------------------------------------------------------------
# UnsupportedColorspaceError in non-streaming
# ---------------------------------------------------------------------------

class TestColorspaceError:
    def test_colorspace_error_exits_1(self, tmp_path, capsys):
        inp, out, rest, det = _make_model_files(tmp_path)
        from jasna.media import UnsupportedColorspaceError

        pipeline_mock = MagicMock()
        pipeline_mock.run.side_effect = UnsupportedColorspaceError("yuv422 unsupported")

        def make_pipeline(**kw):
            return pipeline_mock

        with pytest.raises(SystemExit) as exc:
            _run_main(
                _base_argv(inp, out, rest, det),
                pipeline_side_effect=make_pipeline,
            )
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "yuv422 unsupported" in captured.out


# ---------------------------------------------------------------------------
# Cleanup / finally block
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_pipeline_and_restorer_closed_on_success(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_mock = MagicMock()
        restorer_mock = MagicMock()
        rp_mock = MagicMock()
        rp_mock.restorer = restorer_mock

        def make_pipeline(**kw):
            return pipeline_mock

        with (
            patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
            patch("jasna.pipeline.Pipeline", side_effect=make_pipeline),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", return_value=restorer_mock),
            patch("jasna.restorer.restoration_pipeline.RestorationPipeline", return_value=rp_mock),
        ):
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det)):
                from jasna.main import main
                main()

        pipeline_mock.close.assert_called_once()
        restorer_mock.close.assert_called_once()

    def test_secondary_restorer_closed(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_tvai = MagicMock()
        tvai_instance = MagicMock()
        mock_tvai.return_value = tvai_instance

        with patch("jasna.restorer.tvai_secondary_restorer.TvaiSecondaryRestorer", mock_tvai):
            _run_main(_base_argv(inp, out, rest, det, ["--secondary-restoration", "tvai"]))

        tvai_instance.close.assert_called_once()

    def test_cleanup_on_error(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_mock = MagicMock()
        pipeline_mock.run.side_effect = RuntimeError("boom")

        def make_pipeline(**kw):
            return pipeline_mock

        with pytest.raises(RuntimeError, match="boom"):
            _run_main(
                _base_argv(inp, out, rest, det),
                pipeline_side_effect=make_pipeline,
            )
        pipeline_mock.close.assert_called_once()


# ---------------------------------------------------------------------------
# Logging level
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_level_configured(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        import logging as _logging
        with patch.object(_logging, "basicConfig") as mock_basic:
            _run_main(_base_argv(inp, out, rest, det, ["--log-level", "debug"]))
        mock_basic.assert_called_once()
        assert mock_basic.call_args.kwargs["level"] == _logging.DEBUG


# ---------------------------------------------------------------------------
# Required arg errors (argparse)
# ---------------------------------------------------------------------------

class TestRequiredArgs:
    def test_input_required_when_not_streaming(self, tmp_path):
        out = tmp_path / "out.mp4"
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["jasna", "--output", str(out)]):
                from jasna.main import main
                main()

    def test_output_required_when_not_streaming(self, tmp_path):
        inp = tmp_path / "in.mp4"
        inp.touch()
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["jasna", "--input", str(inp)]):
                from jasna.main import main
                main()

    def test_stream_does_not_require_input_or_output(self, tmp_path):
        _, _, rest, det = _make_model_files(tmp_path)
        mock_hls = MagicMock()
        mock_hls.wait_for_video.side_effect = KeyboardInterrupt

        with _main_patches():
            with patch("jasna.streaming.HlsStreamingServer", return_value=mock_hls):
                argv = [
                    "jasna",
                    "--stream",
                    "--restoration-model-path", str(rest),
                    "--detection-model-path", str(det),
                    "--no-browser",
                ]
                with patch.object(sys, "argv", argv):
                    from jasna.main import main
                    main()


# ---------------------------------------------------------------------------
# Engine compilation request
# ---------------------------------------------------------------------------

class TestEngineCompilation:
    def test_unet4x_triggers_engine_compilation(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        mock_unet = MagicMock()

        with (
            patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)) as mock_compile,
            patch("jasna.pipeline.Pipeline", return_value=MagicMock()),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
            patch("jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer", mock_unet),
        ):
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det, ["--secondary-restoration", "unet-4x"])):
                from jasna.main import main
                main()

        req = mock_compile.call_args[0][0]
        assert req.unet4x is True

    def test_compile_basicvsrpp_flag(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)

        with (
            patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.warn_if_windows_hardware_accelerated_gpu_scheduling_enabled"),
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)) as mock_compile,
            patch("jasna.pipeline.Pipeline", return_value=MagicMock()),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
        ):
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det, ["--no-compile-basicvsrpp"])):
                from jasna.main import main
                main()

        req = mock_compile.call_args[0][0]
        assert req.basicvsrpp is False


# ---------------------------------------------------------------------------
# Non-streaming pipeline.run() path
# ---------------------------------------------------------------------------

class TestNonStreamingRun:
    def test_run_called(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_cls = _run_main(_base_argv(inp, out, rest, det))
        pipeline_cls.return_value.run.assert_called_once()

    def test_run_streaming_not_called(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        pipeline_cls = _run_main(_base_argv(inp, out, rest, det))
        pipeline_cls.return_value.run_streaming.assert_not_called()

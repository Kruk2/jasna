import logging
import re
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import pytest

from factories import cli_args, cli_environment, run_cli
from jasna.main import batch_size_from_args, build_parser
from jasna.media.probe import UnsupportedVideoError
from jasna.post_export_action import PostExportVideoCommandError
from jasna.restorer.denoise import DenoiseStep, DenoiseStrength

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


@pytest.fixture(autouse=True)
def _mock_device_context():
    """CLI orchestration tests use mocked models and must not initialize CUDA."""
    with patch("jasna.accelerator.device_context", side_effect=lambda _device: nullcontext()):
        yield


def _config(tmp_path, *extra):
    return run_cli(cli_args(tmp_path, *extra)).call_args.kwargs["config"]


def _model_args(tmp_path) -> list[str]:
    """Only the model paths, for runs without an input and output (streaming, folders)."""
    return cli_args(tmp_path)[4:]


def _folder_args(tmp_path, names, *extra, output=None) -> list[str]:
    folder = tmp_path / "in"
    folder.mkdir()
    for name in names:
        (folder / name).touch()
    return ["--input", str(folder), "--output", str(output or tmp_path / "out"), *_model_args(tmp_path), *extra]


def test_benchmark_flag_runs_the_benchmark_instead_of_a_job():
    with patch("jasna.benchmark.run_benchmark_cli") as benchmark:
        pipeline_cls = run_cli(["--benchmark"])
    benchmark.assert_called_once()
    pipeline_cls.assert_not_called()


def test_a_video_job_runs_the_pipeline_and_not_the_stream(tmp_path):
    with patch("jasna.streaming_pipeline.run_streaming") as run_streaming:
        pipeline_cls = run_cli(cli_args(tmp_path))
    pipeline_cls.return_value.run.assert_called_once()
    run_streaming.assert_not_called()


def test_post_export_video_command_runs_after_output_closes(tmp_path):
    events: list[str] = []
    pipeline = MagicMock()
    pipeline.run.side_effect = lambda: events.append("run")
    argv = cli_args(tmp_path, "--post-export-video-command", "remux {output}")

    with patch(
        "jasna.post_export_action.run_post_export_video_command",
        side_effect=lambda *_args: events.append("command"),
    ) as command:
        run_cli(argv, pipeline=pipeline)

    assert events == ["run", "command"]
    assert command.call_args.args[:3] == ("remux {output}", tmp_path / "in.mp4", tmp_path / "out.mkv")


@pytest.mark.parametrize(
    ("args", "threshold"),
    [([], 0.35), (["--detection-model", "rfdetr-v6-large"], 0.40), (["--detection-score-threshold", "0.5"], 0.5)],
    ids=["fast-model", "large-model", "explicit"],
)
def test_detection_threshold_defaults_to_the_models_recommendation(tmp_path, args, threshold):
    assert _config(tmp_path, *args).detection_score_threshold == threshold


@pytest.mark.parametrize(
    ("available", "model", "warns"),
    [
        (["rfdetr-v3"], [], True),
        (["rfdetr-v6"], [], False),
        (["rfdetr-v5", "zelefans-vr-yolo-v2"], ["--detection-model", "zelefans-vr-yolo-v2"], False),
    ],
    ids=["missing", "available", "bundled"],
)
def test_cli_warns_only_when_the_chosen_detection_model_is_missing(tmp_path, capsys, available, model, warns):
    weights = tmp_path / "weights.onnx"
    weights.touch()
    argv = cli_args(tmp_path)[:6] + model

    with (
        patch("jasna.mosaic.detection_registry.discover_available_detection_models", return_value=available),
        patch("jasna.mosaic.detection_registry.require_detection_model_weights", return_value=weights),
    ):
        run_cli(argv)

    assert ("not found in model_weights" in capsys.readouterr().out) is warns


def test_denoise_options_reach_the_restoration_pipeline(tmp_path):
    with patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as restoration_pipeline:
        run_cli(cli_args(tmp_path, "--denoise", "high", "--denoise-step", "after_secondary"))

    kwargs = restoration_pipeline.call_args.kwargs
    assert (kwargs["denoise_strength"], kwargs["denoise_step"]) == (DenoiseStrength.HIGH, DenoiseStep.AFTER_SECONDARY)


@pytest.mark.parametrize(
    ("args", "encoder_settings"),
    [
        ([], {"cq": 28}),
        (["--codec", "h264"], {"cq": 25}),
        (["--codec", "av1"], {"cq": 35}),
        (["--codec", "h264", "--cq", "31"], {"cq": 31}),
        (["--encoder-settings", "cq=22,rc-lookahead=32"], {"cq": 22, "rc-lookahead": 32}),
    ],
    ids=["hevc-default", "h264-default", "av1-default", "direct-cq", "advanced"],
)
def test_encoder_settings_default_per_codec_and_parse_typed_values(tmp_path, args, encoder_settings):
    assert _config(tmp_path, *args).encoder_settings == encoder_settings


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--cq", "28", "--encoder-settings", "cq=22"], "--cq.*--encoder-settings"),
        (["--codec", "h264", "--cq", "52"], r"h264.*1\.\.51"),
    ],
)
def test_cq_conflicts_and_range_are_usage_errors(tmp_path, capsys, args, error):
    with pytest.raises(SystemExit):
        run_cli(cli_args(tmp_path, *args))
    assert re.search(error, capsys.readouterr().err)


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--output", "out.mp4"], "--input is required when not using --benchmark or --stream"),
        (["--input", "."], "--output is required when not using --benchmark or --stream"),
    ],
)
def test_a_video_job_needs_input_and_output(capsys, argv, error):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(argv)
    assert exit_info.value.code == 2
    assert error in capsys.readouterr().err


def test_log_level_sets_the_logging_level(tmp_path):
    root = logging.getLogger()
    previous = root.level
    try:
        run_cli(cli_args(tmp_path, "--log-level", "debug"))
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(previous)


def test_disabling_basicvsrpp_compile_skips_its_engine(tmp_path):
    with cli_environment(), patch("jasna.engine_compiler.ensure_engines_compiled") as compiled, \
            patch("sys.argv", ["jasna", *cli_args(tmp_path, "--no-compile-basicvsrpp")]):
        from jasna.main import main
        main()
    assert compiled.call_args.args[0].basicvsrpp is False


@pytest.mark.parametrize("failure", [None, RuntimeError("boom")])
def test_the_session_is_closed_after_the_job(tmp_path, failure):
    pipeline = MagicMock()
    pipeline.run.side_effect = failure
    with (
        patch("jasna.session_factory.RestorationSession.close") as close,
        pytest.raises(RuntimeError, match="boom") if failure else nullcontext(),
    ):
        run_cli(cli_args(tmp_path), pipeline=pipeline)
    close.assert_called_once()


def test_unsupported_video_exits_1_with_its_reason(tmp_path, capsys):
    pipeline = MagicMock()
    pipeline.run.side_effect = UnsupportedVideoError("yuv422 unsupported")

    with pytest.raises(SystemExit) as exit_info:
        run_cli(cli_args(tmp_path), pipeline=pipeline)

    assert exit_info.value.code == 1
    assert "yuv422 unsupported" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Folder jobs
# ---------------------------------------------------------------------------

def test_folder_input_rejects_file_shaped_output(tmp_path, capsys):
    out_file = tmp_path / "out.mp4"
    with pytest.raises(SystemExit) as exit_info:
        run_cli(_folder_args(tmp_path, ["a.mp4"], output=out_file))

    assert exit_info.value.code == 2
    assert not out_file.exists()
    assert "--output must be a folder when --input is a folder" in capsys.readouterr().err


def test_folder_prints_each_video_and_its_default_output(tmp_path, capsys):
    run_cli(_folder_args(tmp_path, ["a.mp4", "b.mp4"]))

    printed = capsys.readouterr().out
    assert "[1/2] Processing a.mp4 -> a_out.mp4" in printed
    assert "[2/2] Processing b.mp4 -> b_out.mp4" in printed


def test_single_file_does_not_print_batch_line(tmp_path, capsys):
    run_cli(cli_args(tmp_path))
    assert "Processing" not in capsys.readouterr().out


def test_folder_output_pattern_applies_to_videos(tmp_path, capsys):
    pipeline_cls = run_cli(_folder_args(tmp_path, ["a.mkv", "b.mov"], "--output-pattern", "{original}_restored.mp4"))

    assert [call.kwargs["output_video"].name for call in pipeline_cls.call_args_list] == ["a_restored.mp4", "b_restored.mp4"]
    printed = capsys.readouterr().out
    assert "[1/2] Processing a.mkv -> a_restored.mp4" in printed
    assert "[2/2] Processing b.mov -> b_restored.mp4" in printed


@pytest.mark.parametrize(
    ("names", "pattern", "into_input", "error"),
    [
        (["a.mkv", "b.mov"], "restored.mp4", False, "maps multiple inputs to the same output"),
        (["clip.mp4"], "{original}.mp4", True, "would overwrite an input file"),
    ],
)
def test_folder_output_pattern_rejects_colliding_outputs(tmp_path, capsys, names, pattern, into_input, error):
    output = tmp_path / "in" if into_input else None
    with pytest.raises(SystemExit) as exit_info:
        run_cli(_folder_args(tmp_path, names, "--output-pattern", pattern, output=output))

    assert exit_info.value.code == 2
    assert error in capsys.readouterr().err


def test_folder_video_progress_counts_images_too(tmp_path, capsys):
    with patch("jasna.image_restore.run_image_restoration_folder") as image_batch:
        run_cli(_folder_args(tmp_path, ["photo.png", "clip.mp4"]))

    assert image_batch.call_args.kwargs["progress_total"] == 2
    assert "[2/2] Processing clip.mp4 -> clip_out.mp4" in capsys.readouterr().out


def test_post_export_video_command_failure_continues_folder_and_exits_1(tmp_path, capsys):
    command = MagicMock(side_effect=[PostExportVideoCommandError("exit code 7"), None])
    with (
        patch("jasna.post_export_action.run_post_export_video_command", command),
        pytest.raises(SystemExit) as exit_info,
    ):
        run_cli(_folder_args(tmp_path, ["a.mp4", "b.mp4"], "--post-export-video-command", "remux {output}"))

    assert exit_info.value.code == 1
    assert command.call_count == 2
    assert "Error post-processing a.mp4" in capsys.readouterr().out


def test_unsupported_video_is_skipped_and_the_folder_continues(tmp_path, capsys):
    rejected, restored = MagicMock(), MagicMock()
    rejected.run.side_effect = UnsupportedVideoError("LTX does not support VR180 side-by-side video")

    with cli_environment() as pipeline_cls, patch("sys.argv", ["jasna", *_folder_args(tmp_path, ["a_vr.mp4", "b_clip.mp4"])]):
        pipeline_cls.side_effect = [rejected, restored]
        from jasna.main import main
        main()

    restored.run.assert_called_once()
    assert "Error processing a_vr.mp4: LTX does not support VR180 side-by-side video" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@contextmanager
def _stream_server(videos=()):
    """An HLS server mock whose player uploads ``videos`` and then stops; yields it and run_streaming."""
    server = MagicMock()
    server.wait_for_video.side_effect = [*videos, KeyboardInterrupt]
    with (
        patch("jasna.streaming.HlsStreamingServer", return_value=server) as server_cls,
        patch("jasna.streaming_pipeline.run_streaming") as run_streaming,
    ):
        yield server_cls, server, run_streaming


def test_streaming_an_input_serves_it_and_stops_the_server(tmp_path):
    pipeline = MagicMock()
    with _stream_server() as (server_cls, server, run_streaming):
        run_cli(
            ["--stream", "--input", cli_args(tmp_path)[1], *_model_args(tmp_path), "--no-browser",
             "--stream-port", "9999", "--stream-segment-duration", "2.0"],
            pipeline=pipeline,
        )

    server_cls.assert_called_once_with(segment_duration=2.0, port=9999)
    run_streaming.assert_called_once_with(pipeline, server)
    server.stop.assert_called_once()


@pytest.mark.parametrize(
    ("with_input", "no_browser", "opened"),
    [(True, False, True), (True, True, False), (False, False, True)],
    ids=["input", "no-browser", "serverless"],
)
def test_streaming_opens_the_player_unless_told_not_to(tmp_path, with_input, no_browser, opened):
    argv = ["--stream", *_model_args(tmp_path), "--stream-port", "8888"]
    argv += ["--input", cli_args(tmp_path)[1]] if with_input else []
    argv += ["--no-browser"] if no_browser else []

    with _stream_server(), patch("webbrowser.open") as open_browser:
        run_cli(argv)

    assert open_browser.call_args_list == ([(("http://localhost:8888/",),)] if opened else [])


def test_serverless_streaming_restores_each_uploaded_video(tmp_path):
    uploaded = tmp_path / "uploaded.mp4"
    uploaded.touch()

    with _stream_server([uploaded]) as (_, server, run_streaming):
        pipeline_cls = run_cli(["--stream", *_model_args(tmp_path), "--no-browser"])

    assert pipeline_cls.call_args.kwargs["input_video"] == uploaded
    run_streaming.assert_called_once_with(pipeline_cls.return_value, server)
    server.unload_video.assert_called_once()
    server.stop.assert_called_once()


def test_serverless_streaming_reports_an_unsupported_upload_and_waits_for_the_next(tmp_path, capsys):
    with _stream_server([tmp_path / "vid.mp4"]) as (_, server, run_streaming):
        run_streaming.side_effect = UnsupportedVideoError("bt2020 not supported")
        run_cli(["--stream", *_model_args(tmp_path), "--no-browser"])

    assert "bt2020 not supported" in capsys.readouterr().out
    server.unload_video.assert_called_once()


@pytest.mark.parametrize(("argv", "amd", "expected"), [([], True, 8), ([], False, 4), (["--batch-size", "2"], True, 2)])
def test_batch_size_defaults_per_vendor_unless_given(monkeypatch, argv, amd, expected):
    import jasna.mosaic.detection_registry as registry

    monkeypatch.setattr(registry, "is_amd_device", lambda _device: amd)
    args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4", *argv])

    assert batch_size_from_args(args) == expected

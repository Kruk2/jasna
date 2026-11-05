import argparse
import re
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from factories import cli_args, cli_environment, run_cli, session_config, video_metadata
from jasna import main
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.session_factory import SegmentJob

pytestmark = pytest.mark.usefixtures("nvidia_vendor")

H264_SOURCE = video_metadata(codec_name="h264", num_frames=240, pixel_format="yuv420p", profile="High")


@contextmanager
def _h264_source():
    """A 10-second H.264 input; yields the splice-plan builder."""
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=H264_SOURCE),
        patch("jasna.media.splice.probe_keyframes", return_value=MagicMock(max_b_frames=0)),
        patch("jasna.media.splice.build_splice_plan") as plan,
    ):
        yield plan


@contextmanager
def _ltx_installed():
    with (
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.model_files.LtxModelFiles.from_dir", return_value=MagicMock()),
        patch("jasna.ltx.model_files.missing_downloads", return_value=[]),
        patch("jasna.ltx.transformer.require_kernels"),
    ):
        yield


def _segments_config(tmp_path, *extra):
    with _h264_source() as plan:
        pipeline_cls = run_cli(cli_args(tmp_path, "--segments", "1-2", *extra))
    assert plan.call_args.args[0] == (SegmentRange(1, 2),)
    assert pipeline_cls.call_args.kwargs["splice_plan"] is plan.return_value
    return pipeline_cls.call_args.kwargs["config"]


def test_segments_auto_select_source_codec_and_reach_pipeline(tmp_path):
    config = _segments_config(tmp_path)

    assert (config.codec, config.encoder_settings) == ("h264", {"cq": 25})


def test_segments_use_cq_only_when_the_codec_is_the_source_codec(tmp_path):
    assert _segments_config(tmp_path, "--codec", "h264", "--cq", "31").encoder_settings == {"cq": 31}
    assert _segments_config(tmp_path, "--cq", "31").encoder_settings == {"cq": 25}


def test_ltx_accepts_segments(tmp_path):
    with _h264_source(), _ltx_installed():
        pipeline_cls = run_cli(
            cli_args(tmp_path, "--segments", "1-2", "--restoration-model-name", "ltx", "--ltx-seed", "3")
        )

    config = pipeline_cls.call_args.kwargs["config"]
    assert (config.restoration_model_name, config.ltx_seed) == ("ltx", 3)


def test_ltx_auto_split_reaches_pipeline_with_the_split_source_codec(tmp_path):
    segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 3)),)
    split = SegmentJob(session_config(codec="h264"), MagicMock(segments=segments, index=MagicMock(max_b_frames=0)))
    with patch("jasna.session_factory.ltx_auto_split", return_value=split) as auto_split, _ltx_installed():
        pipeline_cls = run_cli(cli_args(
            tmp_path, "--restoration-model-name", "ltx", "--ltx-seed", "3", "--ltx-auto-split", "12", "--codec", "av1",
        ))

    assert auto_split.call_args.args[0].ltx_auto_split_percent == 12.0
    kwargs = pipeline_cls.call_args.kwargs
    assert (kwargs["config"], kwargs["segments"], kwargs["splice_plan"]) == (split.config, segments, split.splice_plan)


def test_ltx_auto_split_without_a_split_runs_the_whole_video(tmp_path):
    with patch("jasna.session_factory.ltx_auto_split", return_value=None), _ltx_installed():
        pipeline_cls = run_cli(cli_args(tmp_path, "--restoration-model-name", "ltx", "--ltx-auto-split", "12"))

    assert pipeline_cls.call_args.kwargs["segments"] is None
    assert pipeline_cls.call_args.kwargs["config"].codec == "hevc"


def test_ltx_auto_split_passes_secondary_restoration_for_the_basicvsrpp_parts(tmp_path):
    with patch("jasna.session_factory.ltx_auto_split", return_value=None), _ltx_installed():
        pipeline_cls = run_cli(cli_args(
            tmp_path, "--restoration-model-name", "ltx", "--ltx-auto-split", "12",
            "--secondary-restoration", "rtx-super-res", "--denoise", "low",
        ))

    config = pipeline_cls.call_args.kwargs["config"]
    assert (config.secondary_restoration, config.denoise_strength) == ("rtx-super-res", "low")


def test_ltx_passes_rtx_super_res_without_auto_split(tmp_path):
    with _ltx_installed():
        pipeline_cls = run_cli(cli_args(
            tmp_path, "--restoration-model-name", "ltx", "--secondary-restoration", "rtx-super-res",
        ))

    assert pipeline_cls.call_args.kwargs["config"].secondary_restoration == "rtx-super-res"


def test_ltx_trial_needs_no_model_files_or_download(tmp_path, capsys):
    with (
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.model_files.LtxModelFiles.from_dir", side_effect=AssertionError("reads model files")),
        patch("jasna.ltx.model_files.missing_downloads", side_effect=AssertionError("offers a download")),
        patch("jasna.ltx.transformer.require_kernels"),
    ):
        pipeline_cls = run_cli(cli_args(
            tmp_path, "--restoration-model-name", "ltx-undistilled", "--ltx-trial", missing=("--restoration-model-path",),
        ))

    config = pipeline_cls.call_args.kwargs["config"]
    assert (config.restoration_model_name, config.ltx_model, config.ltx_trial) == ("ltx", "undistilled", True)
    assert "LTX trial mode: output is not restored" in capsys.readouterr().out


def test_ltx_offers_the_download_when_its_model_folder_is_missing(tmp_path):
    with (
        patch("jasna.ltx.model_files.missing_downloads", side_effect=RuntimeError("offers a download")),
        pytest.raises(RuntimeError, match="offers a download"),
    ):
        run_cli(cli_args(tmp_path, "--restoration-model-name", "ltx", missing=("--restoration-model-path",)))


def test_codec_names_are_case_insensitive(tmp_path):
    assert run_cli(cli_args(tmp_path, "--codec", "AV1")).call_args.kwargs["config"].codec == "av1"


LTX_NEEDED = "needs --restoration-model-name ltx-basic, ltx, ltx-4step, ltx-8step or ltx-undistilled"


@pytest.mark.parametrize(
    ("args", "missing", "error"),
    [
        (["--segments", "1-2", "--codec", "av1"], (), "with --segments output codec must match input; pass --codec h264"),
        (["--segments", "1-2", "--codec", "hevc"], (), "with --segments output codec must match input; pass --codec h264"),
        (["--segments", "1-2", "--codec=hevc"], (), "with --segments output codec must match input; pass --codec h264"),
        (["--ltx-auto-split", "12"], (), f"--ltx-auto-split {LTX_NEEDED}"),
        (["--restoration-model-name", "ltx", "--ltx-auto-split", "12", "--segments", "1-2"], (),
         "--segments cannot be combined with --ltx-auto-split"),
        (["--ltx-trial"], (), f"--ltx-trial {LTX_NEEDED}"),
        (["--ltx-fast"], (), LTX_NEEDED),
        (["--ltx-large-canvas"], (), LTX_NEEDED),
        (["--ltx-overlap", "33"], (), LTX_NEEDED),
        (["--ltx-seed", "7"], (), LTX_NEEDED),
        (["--restoration-model-name", "ltx", "--no-compile-basicvsrpp"], (),
         "--compile-basicvsrpp needs --restoration-model-name basicvsrpp"),
        (["--restoration-model-name", "ltx-undistilled", "--no-compile-basicvsrpp"], (),
         "--compile-basicvsrpp needs --restoration-model-name basicvsrpp"),
        (["--stream", "--restoration-model-name", "ltx"], (), "--restoration-model-name ltx does not support --stream"),
        (["--stream", "--retarget-high-fps"], (), "--retarget-high-fps is only supported for offline exports"),
        (["--stream", "--fmp4"], (), "--fmp4 is only supported for offline exports"),
        (["--segments", "10-20", "--fmp4"], (), "--fmp4 cannot be combined with --segments"),
        (["--codec", "vp9"], (), "argument --codec: invalid choice: 'vp9'"),
        (["--codec", "av1", "--encoder-settings", "profile=main"], (), "for codec av1.*profile"),
        (["--batch-size", "0"], (), "Batch size must be > 0"),
        (["--max-clip-size", "0"], (), "Max clip size must be > 0"),
        (["--temporal-overlap", "-1"], (), "Temporal overlap must be >= 0"),
        (["--max-clip-size", "10", "--temporal-overlap", "10"], (), "Temporal overlap must satisfy"),
        (["--max-clip-size", "10", "--temporal-overlap", "5"], (), "Temporal overlap must satisfy"),
        (["--detection-score-threshold", "1.5"], (), "Detection score threshold must be in"),
        (["--sharpen", "1.5"], (), "Sharpening must be in"),
        (["--sharpen", "-0.1"], (), "Sharpening must be in"),
        (["--post-export-action", "command"], (), "--post-export-command is required"),
        ([], ("--input",), "input not found"),
        ([], ("--restoration-model-path",), "restoration model not found"),
        ([], ("--detection-model-path",), "Detection model weights not found"),
    ],
)
def test_invalid_arguments_are_a_usage_error(tmp_path, capsys, args, missing, error):
    with _h264_source(), pytest.raises(SystemExit) as exit_info:
        run_cli(cli_args(tmp_path, *args, missing=missing))

    assert exit_info.value.code == 2
    assert re.search(error, capsys.readouterr().err)


def test_usage_error_is_one_short_message(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli(cli_args(tmp_path, "--stream", "--fmp4"))
    assert capsys.readouterr().err.splitlines() == [
        "jasna: error: --fmp4 is only supported for offline exports",
        "Run 'jasna --help' to see all options.",
    ]


@pytest.mark.parametrize(
    ("args", "missing", "error"),
    [
        (["--restoration-model-name", "ltx-undistilled", "--denoise", "low"], (),
         "--restoration-model-name ltx-undistilled supports --denoise only with --ltx-auto-split"),
        (["--restoration-model-name", "ltx", "--secondary-restoration", "unet-4x"], (),
         "--restoration-model-name ltx supports --secondary-restoration unet-4x only with --ltx-auto-split"),
        ([], ("--input",), "input not found"),
    ],
)
def test_argument_mistakes_are_reported_before_the_gpu_check(tmp_path, capsys, args, missing, error):
    with (
        patch("jasna.main._check_system", side_effect=AssertionError("checked the GPU first")),
        pytest.raises(SystemExit),
    ):
        run_cli(cli_args(tmp_path, *args, missing=missing))
    assert error in capsys.readouterr().err


@pytest.mark.parametrize(
    ("gpu", "message"),
    [("no_cuda", "No compatible GPU was found"), (("GPU", 5, 0), "Compute capability 7.5+ required (GPU: 5.0)")],
)
def test_an_unsupported_gpu_stops_the_cli(tmp_path, capsys, gpu, message):
    with (
        cli_environment(),
        patch("jasna.main.check_supported_gpu", return_value=(False, gpu)),
        patch("sys.argv", ["jasna", *cli_args(tmp_path)]),
        pytest.raises(SystemExit) as exit_info,
    ):
        main.main()

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().out


@contextmanager
def _system_checks_pass(*, driver: str):
    with (
        patch.object(main, "check_ascii_install_path", return_value=(True, "/jasna")),
        patch.object(main, "check_required_executables"),
        patch.object(main, "check_supported_gpu", return_value=(True, "ok")),
        patch.object(main, "check_gpu_driver_version", return_value=(True, driver)),
        patch.object(main, "install_on_hard_disk", return_value=False),
    ):
        yield


@pytest.mark.parametrize("driver_ok", [False, None])
def test_cli_reports_amd_driver_failures_and_warnings(capsys, driver_ok):
    with (
        _system_checks_pass(driver="ROCm 10.1"),
        patch.object(main, "check_gpu_driver_version", return_value=(driver_ok, "ROCm 10.1; AMD driver needs attention")),
        patch("jasna.accelerator.is_amd_device", return_value=True),
        patch.object(main, "check_amd_amf_runtime", return_value=(True, "AMF")),
    ):
        if driver_ok is False:
            with pytest.raises(SystemExit) as exit_info:
                main._check_system(argparse.Namespace(device="cuda:0"))
            assert exit_info.value.code == 1
        else:
            main._check_system(argparse.Namespace(device="cuda:0"))
    out = capsys.readouterr().out
    assert ("Error:" if driver_ok is False else "Warning:") in out
    assert "AMD driver needs attention" in out
    assert "NVIDIA driver" not in out


@pytest.mark.parametrize("hard_disk", [True, False])
def test_cli_warns_when_installed_on_a_hard_disk(capsys, hard_disk):
    with (
        _system_checks_pass(driver="620"),
        patch("jasna.accelerator.is_nvidia_device", return_value=False),
        patch.object(main, "install_on_hard_disk", return_value=hard_disk),
    ):
        main._check_system(argparse.Namespace(device="cuda:0"))
    assert ("installed on a hard disk drive (HDD)" in capsys.readouterr().out) is hard_disk


@pytest.mark.parametrize(
    "platform,hint",
    [("linux", "https://repo.radeon.com/amf/"), ("win32", "https://www.amd.com/en/support/download/drivers.html")],
)
def test_cli_stops_on_amd_without_the_amf_runtime(capsys, platform, hint):
    with (
        _system_checks_pass(driver="ROCm 10.0"),
        patch("jasna.accelerator.is_amd_device", return_value=True),
        patch.object(main, "check_amd_amf_runtime", return_value=(False, "cannot open libamfrt64.so.1")),
        patch.object(main.sys, "platform", platform),
        pytest.raises(SystemExit) as exit_info,
    ):
        main._check_system(argparse.Namespace(device="cuda:0"))
    out = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "AMF video runtime is missing" in out
    assert hint in out


def test_cli_continues_on_amd_with_the_amf_runtime(capsys):
    with (
        _system_checks_pass(driver="ROCm 10.0"),
        patch("jasna.accelerator.is_amd_device", return_value=True),
        patch.object(main, "check_amd_amf_runtime", return_value=(True, "libamfrt64.so.1")),
    ):
        main._check_system(argparse.Namespace(device="cuda:0"))
    assert "AMF" not in capsys.readouterr().out

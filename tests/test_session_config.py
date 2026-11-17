from __future__ import annotations

from dataclasses import replace

import pytest

from factories import GUI_WEIGHTS_DIR, gui_session_config
from jasna.engine_paths import default_restoration_model_path
from jasna.gui.models import AppSettings
from jasna.gui.video_session import build_image_session
from jasna.cli_parser import build_parser
from jasna.main import _session_config_from_args
from jasna.mosaic.detection_registry import recommended_score_threshold

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _cli_config(extra_args: list[str] | None = None):
    args = build_parser().parse_args(
        ["--input", "a.mp4", "--output", "b.mp4", *(extra_args or [])]
    )
    threshold = args.detection_score_threshold
    return _session_config_from_args(
        args,
        codec="hevc",
        encoder_settings={},
        detection_model_name="rfdetr-v6",
        detection_model_path=GUI_WEIGHTS_DIR / "rfdetr-v6.onnx",
        detection_score_threshold=recommended_score_threshold("rfdetr-v6") if threshold is None else threshold,
        restoration_model_path=GUI_WEIGHTS_DIR / default_restoration_model_path("basicvsrpp").name,
        lut_path=None,
    )


def test_cli_takes_the_video_decoder_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "pyav-sw")

    assert _cli_config().decode_backend == "pyav-sw"


def test_cli_rejects_an_unknown_environment_choice(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "quarter")

    with pytest.raises(ValueError, match="JASNA_DECODE_BACKEND='quarter' is not one of auto, "):
        _cli_config()


def test_cli_ltx_overlap_defaults_to_17_frames_and_sets_the_window_step() -> None:
    assert _cli_config(["--restoration-model-name", "ltx"]).ltx_window_hop == 104
    more = _cli_config(["--restoration-model-name", "ltx", "--ltx-overlap", "33"])
    assert more.ltx_window_hop == 88


def test_gui_defaults_match_cli_defaults() -> None:
    cli = _cli_config()
    gui = gui_session_config(AppSettings())

    # The GUI shows its own progress, never the terminal bar.
    assert replace(cli, disable_progress=True) == gui


def test_gui_image_defaults_match_cli_defaults() -> None:
    args = build_parser().parse_args(["--input", "a.png", "--output", "b.png"])
    gui = AppSettings()

    assert (args.sd15_steps, args.sd15_strength, args.sd15_freeu, args.sd15_seed, args.sd15_variants) == (
        gui.image_restore_steps,
        gui.image_restore_strength,
        gui.image_restore_freeu,
        gui.image_restore_seed,
        gui.image_restore_variants,
    )


def test_gui_config_trims_the_lut_path_and_hides_the_terminal_progress_bar() -> None:
    config = gui_session_config(AppSettings(lut_path="  lut.cube  "))

    assert config.lut_path == "lut.cube"
    assert config.disable_progress is True
    assert gui_session_config(AppSettings(lut_path="   ")).lut_path is None


@pytest.mark.parametrize(("secondary", "auto_split", "expected"), [
    ("rtx-super-res", False, "rtx-super-res"),
    ("unet-4x", False, "none"),
    ("tvai", False, "none"),
    ("unet-4x", True, "unet-4x"),
])
def test_gui_ltx_keeps_only_the_secondary_restoration_it_supports(secondary, auto_split, expected) -> None:
    settings = AppSettings(restoration_model="ltx", secondary_restoration=secondary, ltx_auto_split=auto_split)
    assert gui_session_config(settings).secondary_restoration == expected


def test_gui_image_session_requires_installed_sd15_bundle(monkeypatch) -> None:
    from jasna.gui.locales import t
    from jasna.restorer import sd15_download

    monkeypatch.setattr(sd15_download, "bundle_present", lambda _path: False)

    with pytest.raises(FileNotFoundError, match=t("interactive_model_missing")):
        build_image_session(AppSettings(), log=None)


@pytest.mark.parametrize(
    ("auto_split", "secondary", "denoise", "expected"),
    [
        (False, "rtx-super-res", "none", ()),
        (False, "tvai", "low", ("secondary_restoration", "denoise_strength")),
        (False, "none", "high", ("denoise_strength",)),
        (True, "unet-4x", "high", ()),
    ],
)
def test_ltx_needs_auto_split_for_options_of_the_basicvsrpp_parts(auto_split, secondary, denoise, expected) -> None:
    from jasna.session_config import ltx_unsupported_options

    assert ltx_unsupported_options(
        auto_split=auto_split, secondary_restoration=secondary, denoise_strength=denoise
    ) == expected

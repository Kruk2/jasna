from __future__ import annotations

import json

import pytest

from factories import session_config
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.settings_tag import job_settings, settings_comment, with_source_comment
from jasna.vr180 import VrModeResolution

FLAT = VrModeResolution("auto", "off", "", 16 / 9, "raw")
WEIGHTS = "ltx-restore-alpha1-distill2-int8.safetensors"


def test_whole_video_ltx_records_the_canvas_the_run_used_and_no_basicvsrpp():
    config = session_config(
        restoration_model_name="ltx",
        ltx_seed=7,
        ltx_window_hop=104,
        ltx_large_canvas=True,
        secondary_restoration="rtx-super-res",
        lut_path="/home/someone/looks/warm.cube",
    )

    settings = job_settings(
        config, models=frozenset({"ltx"}), segments=None, vr=FLAT, ltx_weights=WEIGHTS, ltx_large_canvas=False
    )

    assert "basicvsrpp" not in settings and "segments" not in settings
    assert settings["ltx"]["large_canvas"] is False
    assert settings["ltx"]["seed"] == 7
    assert settings["ltx"]["weights"] == WEIGHTS
    assert settings["ltx"]["super_res"] == {"quality": "high", "denoise": "medium", "deblur": "none"}
    assert settings["output"]["lut"] == "warm.cube"


def test_ltx_records_the_vae_sizes_auto_resolved_to():
    config = session_config(restoration_model_name="ltx", ltx_model="distilled1", ltx_vae_decoder="full")

    ltx = job_settings(
        config, models=frozenset({"ltx"}), segments=None, vr=FLAT, ltx_weights=WEIGHTS, ltx_large_canvas=False
    )["ltx"]

    assert (ltx["vae_encoder"], ltx["vae_decoder"]) == ("half", "full")


def test_mixed_job_sums_ranges_per_model_with_every_ltx_seed():
    config = session_config(
        restoration_model_name="ltx",
        ltx_auto_split_percent=12,
        secondary_restoration="tvai",
        tvai_denoise=True,
        denoise_strength="low",
    )
    segments = (
        SegmentRange(1.0, 2.0, SegmentRestoration("basicvsrpp", None)),
        SegmentRange(4.0, 6.5, SegmentRestoration("ltx", 9)),
        SegmentRange(8.0, 8.25, SegmentRestoration("ltx", 3)),
        SegmentRange(9.0, 9.5, SegmentRestoration("ltx", 9)),
    )

    settings = job_settings(
        config,
        models=frozenset({"ltx", "basicvsrpp"}),
        segments=segments,
        vr=FLAT,
        ltx_weights=WEIGHTS,
        ltx_large_canvas=True,
    )

    assert settings["segments"] == {
        "count": 4,
        "basicvsrpp": {"count": 1, "seconds": 1.0},
        "ltx": {"count": 3, "seconds": 3.25, "seeds": [3, 9]},
    }
    assert "seed" not in settings["ltx"]
    assert settings["auto_split_percent"] == 12
    assert settings["basicvsrpp"]["secondary"] == {
        "name": "tvai", "model": "iris-2", "scale": 4, "args": "noise=0", "denoise": True,
    }
    assert "denoise_step" not in settings["basicvsrpp"]


def test_sbs_basicvsrpp_records_the_resolved_projection():
    settings = job_settings(
        session_config(),
        models=frozenset({"basicvsrpp"}),
        segments=None,
        vr=VrModeResolution("auto", "sbs", "", 2.0, "fisheye"),
        ltx_weights=None,
        ltx_large_canvas=None,
    )

    assert (settings["basicvsrpp"]["vr"], settings["basicvsrpp"]["vr_projection"]) == ("sbs", "fisheye")
    assert "ltx" not in settings


def test_comment_is_one_json_line_led_by_the_jasna_version_with_the_encoder_options():
    settings = job_settings(
        session_config(), models=frozenset({"basicvsrpp"}), segments=None, vr=FLAT, ltx_weights=None,
        ltx_large_canvas=None,
    )

    comment = settings_comment(settings, encoder="hevc_nvenc", options={"cq": "28", "preset": "p5"})

    assert "\n" not in comment
    tag = json.loads(comment)
    assert next(iter(tag)) == "jasna"
    assert tag["output"]["encoder"] == "hevc_nvenc"
    assert tag["output"]["encoder_options"] == {"cq": "28", "preset": "p5"}
    assert tag["output"]["codec"] == "hevc"


def test_hundreds_of_ranges_keep_the_comment_short():
    ltx, basicvsrpp = SegmentRestoration("ltx", 7), SegmentRestoration("basicvsrpp", None)
    segments = tuple(SegmentRange(i * 2.0, i * 2.0 + 1.0, ltx if i % 2 else basicvsrpp) for i in range(1000))
    settings = job_settings(
        session_config(restoration_model_name="ltx", ltx_auto_split_percent=12),
        models=frozenset({"ltx", "basicvsrpp"}),
        segments=segments,
        vr=FLAT,
        ltx_weights=WEIGHTS,
        ltx_large_canvas=True,
    )

    assert len(settings_comment(settings, encoder="h264_nvenc", options={"cq": "25"})) < 2000


def test_output_records_frame_generation():
    config = session_config(frame_generation="2x")

    settings = job_settings(
        config, models=frozenset({"basicvsrpp"}), segments=None, vr=FLAT, ltx_weights=None, ltx_large_canvas=None
    )

    assert settings["output"]["frame_generation"] == "2x"


@pytest.mark.parametrize(
    ("source_tags", "expected"),
    [
        ({}, None),
        ({"COMMENT": ""}, None),
        ({"title": "t", "Comment": "my notes"}, "my notes"),
        ({"comment": "x" * 40_000}, "x" * 16_000),
    ],
    ids=["none", "empty", "kept", "cut"],
)
def test_the_source_comment_stays_inside_the_settings_tag(source_tags, expected):
    comment = with_source_comment('{"jasna":"test"}', source_tags)
    assert json.loads(comment).get("source_comment") == expected
    assert next(iter(json.loads(comment))) == "jasna"


@pytest.mark.parametrize(("hip", "cuda", "tensorrt"), [(None, "13.0", True), ("7.15.26333", None, False)])
def test_basicvsrpp_records_tensorrt_only_where_it_runs(monkeypatch, hip, cuda, tensorrt):
    import torch

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    settings = job_settings(
        session_config(compile_basicvsrpp=True),
        models=frozenset({"basicvsrpp"}),
        segments=None,
        vr=FLAT,
        ltx_weights=None,
        ltx_large_canvas=None,
    )

    assert settings["basicvsrpp"]["tensorrt"] is tensorrt

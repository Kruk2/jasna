from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from factories import GUI_WEIGHTS_DIR, gui_session_config
from jasna.gui.models import AppSettings
from jasna.gui.video_session import SessionCache, video_session_key

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


@pytest.mark.parametrize(
    ("changes", "same_session"),
    [
        ({}, True),
        ({"detection_model": "lada-yolo-v2"}, False),
        ({"secondary_restoration": "unet-4x"}, False),
        ({"max_clip_size": 60}, False),
        ({"fp16_mode": False}, False),
        ({"encoder_cq": 30, "codec": "h264"}, True),
        ({"vr_mode": "off"}, True),
        ({"vr_mode": "sbs-fisheye"}, True),
        ({"vr_projection": "fisheye"}, True),
        ({"ltx_vae_encoder": "full"}, False),
        ({"ltx_vae_decoder": "half"}, False),
        ({"decode_backend": "pyav-sw"}, True),
        ({"frame_generation": "2x"}, True),
    ],
)
def test_video_session_key_tracks_only_session_fields(changes, same_session) -> None:
    assert (video_session_key(replace(AppSettings(), **changes)) == video_session_key(AppSettings())) is same_session


def test_video_session_config_forwards_vr_projection() -> None:
    config = gui_session_config(replace(AppSettings(), vr_projection="gnomonic", frame_generation="2x"))

    assert (config.vr_projection, config.frame_generation) == ("gnomonic", "2x")


def test_video_session_key_includes_active_secondary_knobs() -> None:
    tvai = replace(AppSettings(), secondary_restoration="tvai")
    assert video_session_key(replace(tvai, tvai_scale=2)) != video_session_key(tvai)
    assert video_session_key(replace(tvai, rtx_scale=2)) == video_session_key(tvai)

    rtx = replace(AppSettings(), secondary_restoration="rtx-super-res")
    assert video_session_key(replace(rtx, rtx_quality="low")) != video_session_key(rtx)
    assert video_session_key(replace(rtx, tvai_scale=2)) == video_session_key(rtx)


def test_video_session_config_maps_ltx_settings() -> None:
    config = gui_session_config(
        replace(AppSettings(), restoration_model="ltx", ltx_seed=7, ltx_fast=True, ltx_large_canvas=True)
    )

    assert config.restoration_model_name == "ltx"
    assert config.restoration_model_path == GUI_WEIGHTS_DIR / "ltx-restore"
    assert (config.ltx_seed, config.ltx_fast, config.ltx_large_canvas) == (7, True, True)
    assert config.ltx_window_hop == 104


def test_video_session_config_maps_more_ltx_overlap_to_a_shorter_window_hop() -> None:
    config = gui_session_config(replace(AppSettings(), restoration_model="ltx", ltx_overlap=33))

    assert config.ltx_window_hop == 88


def test_video_session_config_keeps_standard_extras_only_for_ltx_auto_split() -> None:
    settings = replace(
        AppSettings(),
        restoration_model="ltx",
        secondary_restoration="tvai",
        tvai_denoise=True,
        denoise_strength="high",
    )

    config = gui_session_config(settings)

    assert (config.secondary_restoration, config.tvai_denoise, config.denoise_strength) == ("none", False, "none")
    standard = gui_session_config(replace(settings, restoration_model="basicvsrpp"))
    assert (standard.secondary_restoration, standard.tvai_denoise, standard.denoise_strength) == ("tvai", True, "high")
    auto_split = gui_session_config(replace(settings, ltx_auto_split=True))
    assert (auto_split.secondary_restoration, auto_split.tvai_denoise, auto_split.denoise_strength) == (
        "tvai", True, "high"
    )


def test_video_session_key_tracks_model_and_ltx_variant() -> None:
    ltx = replace(AppSettings(), restoration_model="ltx")

    assert video_session_key(ltx) != video_session_key(AppSettings())
    assert video_session_key(replace(ltx, ltx_fast=True)) != video_session_key(ltx)
    assert video_session_key(replace(ltx, ltx_seed=1, ltx_large_canvas=True)) == video_session_key(ltx)


def test_video_session_key_tracks_the_ltx_variant_for_ltx_ranges_of_standard_jobs() -> None:
    assert video_session_key(replace(AppSettings(), ltx_fast=True)) != video_session_key(AppSettings())


def test_video_session_config_and_key_follow_the_ltx_model() -> None:
    undistilled = replace(AppSettings(), restoration_model="ltx", ltx_model="undistilled")

    assert gui_session_config(undistilled).ltx_model == "undistilled"
    assert gui_session_config(replace(undistilled, ltx_model="distilled")).ltx_model == "distilled"
    assert video_session_key(undistilled) != video_session_key(replace(undistilled, ltx_model="distilled"))
    standard = replace(AppSettings(), ltx_model="undistilled")
    assert video_session_key(standard) != video_session_key(AppSettings())


class _FakeSession:
    def __init__(self, name: str) -> None:
        self.device = name
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def released(monkeypatch) -> list:
    devices: list = []
    monkeypatch.setattr("jasna.gui.video_session.release_session_memory", devices.append)
    return devices


def test_session_cache_reuses_the_session_for_the_same_key(released) -> None:
    built: list[_FakeSession] = []

    def build() -> _FakeSession:
        built.append(_FakeSession(f"s{len(built)}"))
        return built[-1]

    cache = SessionCache()
    first = cache.get(AppSettings(), build)

    assert cache.get(replace(AppSettings(), encoder_cq=30), build) is first
    assert len(built) == 1 and first.closed == 0 and released == []


def test_session_cache_frees_the_old_session_before_building_for_a_new_key(released) -> None:
    old = _FakeSession("old")
    new = _FakeSession("new")
    dropped: list[int] = []
    cache = SessionCache(on_close=lambda: dropped.append(old.closed))
    cache.get(AppSettings(), lambda: old)

    def build_new() -> _FakeSession:
        assert (old.closed, released) == (1, ["old"])
        return new

    assert cache.get(replace(AppSettings(), max_clip_size=60), build_new) is new
    assert dropped == [0]  # dependents dropped before the old session closed
    assert new.closed == 0 and released == ["old"]


def test_session_cache_close_releases_once_and_empty_close_is_a_noop(released) -> None:
    session = _FakeSession("s")
    dropped: list[None] = []
    cache = SessionCache(on_close=lambda: dropped.append(None))

    cache.close()
    cache.get(AppSettings(), lambda: session)
    cache.close()
    cache.close()

    assert (session.closed, released, len(dropped)) == (1, ["s"], 1)
    assert cache.session is None


def test_session_cache_releases_memory_when_close_fails(released) -> None:
    session = _FakeSession("s")
    session.close = MagicMock(side_effect=RuntimeError("close failed"))
    cache = SessionCache()
    cache.get(AppSettings(), lambda: session)

    with pytest.raises(RuntimeError, match="close failed"):
        cache.close()

    assert released == ["s"] and cache.session is None

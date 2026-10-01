import json

import pytest

from jasna import __version__
from jasna.ltx import speed
from jasna.ltx.model_files import LtxModelFiles


def test_profile_key_names_the_version_gpu_and_transformer():
    assert speed.profile_key("RTX 5090", "m.safetensors") == f"{__version__} | RTX 5090 | m.safetensors"


def test_a_trial_shares_the_key_of_the_model_it_stands_in_for(tmp_path):
    trial = LtxModelFiles.placeholder("distilled", fast=True)
    real = LtxModelFiles(tmp_path / "ltx-restore-alpha1-distill8-nvfp4.safetensors.enc", tmp_path, tmp_path)
    assert trial.transformer_name == real.transformer_name == "ltx-restore-alpha1-distill8-nvfp4.safetensors"


def test_rates_round_trip_per_key(tmp_path):
    path = tmp_path / "profile.json"
    assert speed.load_rates(path, "a") == {}
    speed.save_rates(path, "a", {"denoise/window512": 4.0})
    speed.save_rates(path, "b", {"denoise/window512": 9.0})
    assert speed.load_rates(path, "a") == {"denoise/window512": 4.0}
    assert speed.load_rates(path, "b") == {"denoise/window512": 9.0}
    assert speed.load_rates(path, "c") == {}


def test_a_later_run_replaces_what_it_measured_and_keeps_the_rest(tmp_path):
    path = tmp_path / "profile.json"
    speed.save_rates(path, "a", {"denoise/window512": 4.0, "denoise/window768": 9.0})
    speed.save_rates(path, "a", {"denoise/window512": 3.5})
    assert speed.load_rates(path, "a") == {"denoise/window512": 3.5, "denoise/window768": 9.0}


def test_a_failed_write_leaves_the_old_profile_whole(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    speed.save_rates(path, "a", {"denoise/window512": 4.0})

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(speed.os, "replace", fail)
    with pytest.raises(OSError):
        speed.save_rates(path, "a", {"denoise/window512": 1.0})
    assert json.loads(path.read_text()) == {"a": {"denoise/window512": 4.0}}


WORK = speed.Work(frames=100, megapixels=2.0, windows={512: 3})


def test_learned_rates_leave_out_cold_windows_and_take_the_rest_per_frame():
    timings = {
        "encode": speed.PassTiming(
            seconds=30.0, warm_seconds=12.0, warm_frames=50, windows=[(512, 6.0), (512, 1.0), (512, 3.0)]
        ),
        "denoise": speed.PassTiming(seconds=40.0, windows=[(512, 4.0), (512, 6.0)]),
    }
    rates = speed.learned_rates(timings, WORK, known={})
    assert rates["encode/window512"] == 2.0
    assert rates["encode/frame"] == pytest.approx((12.0 - 10.0) / (50 * 2.0))
    assert rates["encode/fixed"] == pytest.approx(30.0 - 3 * 2.0 - 100 * 2.0 * rates["encode/frame"])
    assert rates["denoise/window512"] == 5.0
    assert rates["denoise/fixed"] == 25.0


def test_the_fixed_part_uses_known_rates_for_what_one_window_cannot_time():
    timings = {"compose": speed.PassTiming(seconds=20.0, warm_seconds=6.0, warm_frames=100, windows=[(512, 9.0)])}
    work = speed.Work(frames=100, megapixels=1.0, windows={512: 1})
    rates = speed.learned_rates(timings, work, known={"compose/window512": 1.0})
    assert "compose/window512" not in rates
    assert rates["compose/frame"] == 0.0
    assert rates["compose/fixed"] == 19.0
    assert speed.learned_rates(timings, work, known={}) == {"compose/frame": 0.0}


def test_pass_estimates_need_every_rate_the_plan_uses():
    rates = {
        "encode/fixed": 1.0, "encode/frame": 0.1, "encode/window512": 2.0,
        "denoise/fixed": 10.0, "denoise/window512": 4.0,
        "compose/fixed": 3.0, "compose/frame": 0.2, "compose/window512": 1.0,
    }
    assert speed.pass_estimates(rates, WORK) == {
        "encode": (1.0, pytest.approx(100 * 2.0 * 0.1 + 3 * 2.0)),
        "denoise": (10.0, 12.0),
        "compose": (3.0, pytest.approx(100 * 2.0 * 0.2 + 3 * 1.0)),
    }
    assert speed.pass_estimates(rates, speed.Work(frames=100, megapixels=2.0, windows={768: 1})) is None


def test_learned_rates_ignore_the_scan():
    timings = {"scan": speed.PassTiming(seconds=5.0, warm_seconds=4.0, warm_frames=90, windows=[(512, 1.0)] * 3)}
    assert speed.learned_rates(timings, WORK, known={}) == {}


def test_run_summary_gives_each_pass_its_wall_time_and_warm_medians_per_canvas():
    timings = {
        "scan": speed.PassTiming(seconds=4.0, units=100, warm_seconds=2.0, warm_frames=90),
        "encode": speed.PassTiming(
            seconds=20.0, units=100, warm_seconds=10.0, warm_frames=90,
            windows=[(512, 9.0), (512, 1.0), (512, 3.0), (512, 2.0), (768, 8.0), (768, 4.0)],
        ),
        "denoise": speed.PassTiming(
            seconds=60.0, units=40, windows=[(512, 30.0)], steps=[(512, 1.0), (512, 5.0), (512, 2.0), (768, 4.0)]
        ),
        "compose": speed.PassTiming(
            seconds=16.0, units=100, warm_seconds=5.0, warm_frames=90, windows=[(512, 7.0), (512, 0.5), (768, 3.0)]
        ),
    }
    work = speed.Work(frames=100, megapixels=2.0, windows={768: 2, 512: 4})
    assert speed.run_summary(timings, work, 3 << 30, (40, 48)) == [
        "[timing] LTX: 100.0s, 100 frames, windows 4 at 512, 2 at 768, 40 window-steps, peak VRAM 3072.0 MiB",
        "[timing] LTX scan: 4.0s, 45.0 frames/s",
        "[timing] LTX encode: 20.0s, 9.0 frames/s, 2.00 s/window at 512, 4.00 s/window at 768",
        "[timing] LTX denoise: 60.0s, 2.00 s/window-step at 512, 4.00 s/window-step at 768, blocks 40/48 resident",
        "[timing] LTX compose: 16.0s, 18.0 frames/s, decode 0.50 s/window at 512",
    ]


def test_run_summary_without_mosaics_has_only_scan_and_compose():
    timings = {
        "scan": speed.PassTiming(seconds=3.0, units=50, warm_seconds=1.0, warm_frames=40),
        "compose": speed.PassTiming(seconds=2.0, units=50),
    }
    assert speed.run_summary(timings, speed.Work(frames=50, megapixels=2.0, windows={}), 0, None) == [
        "[timing] LTX: 5.0s, 50 frames, windows 0, 0 window-steps, peak VRAM 0.0 MiB",
        "[timing] LTX scan: 3.0s, 40.0 frames/s",
        "[timing] LTX compose: 2.0s",
    ]

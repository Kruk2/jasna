"""CPU tests for the deep VRAM consolidation used during a long single pass.

These never touch CUDA: ``release_vram_cache`` is replaced with a recorder, so
the tests are safe to run on a machine that is concurrently using the GPU.
"""
from __future__ import annotations

import pytest

from jasna import accelerator


class _Clock:
    """A monotonic clock the test drives by hand (used by ``MemoryTrend``)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def released(monkeypatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(accelerator, "release_vram_cache", lambda: calls.append(1))
    return calls


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(accelerator.time, "monotonic", fake)
    return fake


class _Video:
    """A restored-video position the test drives by hand (seconds)."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds

    def __call__(self) -> float:
        return self.seconds


class TestVramReleaserVideoCadence:
    """Both sweeps fire on restored *video* position, not wall clock."""

    def test_disabled_without_progress(self, released):
        # Nothing to key on: no progress source means neither sweep can fire.
        releaser = accelerator.VramReleaser(deep_video=3600.0, light_video=1200.0)
        assert not releaser.enabled
        assert releaser.tick() is False
        assert released == []

    def test_light_fires_at_the_light_mark(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=3600.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        assert releaser.light_video == 1200.0

        video.seconds = 1199.0
        assert releaser.tick() is False
        video.seconds = 1200.0
        assert releaser.tick() is True          # light cold-cache release
        assert len(released) == 1
        assert deep_calls == []                  # the heavy hook stays out of it

        # Fires again at the next whole light mark.
        video.seconds = 2399.0
        assert releaser.tick() is False
        video.seconds = 2400.0
        assert releaser.tick() is True
        assert len(released) == 2

    def test_deep_fires_at_the_deep_mark_and_skips_the_light(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=3600.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "pruned_results=0",
        )
        # Clear the two light marks before the deep one.
        video.seconds = 1200.0
        assert releaser.tick() is True           # light
        video.seconds = 2400.0
        assert releaser.tick() is True           # light
        assert deep_calls == []
        assert len(released) == 2

        video.seconds = 3600.0
        assert releaser.tick() is True           # deep, subsuming the light at 3600
        assert deep_calls == ["deep"]
        assert len(released) == 2                # hook owns the tidy-up; no extra release

    def test_light_resumes_after_a_deep_sweep(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=3600.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        video.seconds = 3600.0
        assert releaser.tick() is True           # deep (subsumes the light at 3600)
        assert deep_calls == ["deep"]

        video.seconds = 4800.0                   # next light mark after 3600
        assert releaser.tick() is True
        assert len(released) == 1
        assert deep_calls == ["deep"]

    def test_coarse_update_does_not_double_fire(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=3600.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        # Jump past several marks at once; only one sweep, and the next one is not
        # due until the following whole interval.
        video.seconds = 10_000.0
        assert releaser.tick() is True
        assert deep_calls == ["deep"]
        assert releaser.tick() is False

    def test_zero_light_disables_light_only(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=0.0, deep_video=3600.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        assert releaser.enabled
        video.seconds = 5000.0
        assert releaser.tick() is True           # only the deep sweep can fire
        assert deep_calls == ["deep"]
        assert released == []                     # custom hook; no light release

    def test_zero_deep_disables_deep_only(self, released):
        video = _Video(0.0)
        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=0.0, progress=video,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        assert releaser.enabled
        video.seconds = 10_000.0
        assert releaser.tick() is True            # light fires, deep never does
        assert deep_calls == []
        assert len(released) == 1

    def test_both_zero_disables(self, released):
        releaser = accelerator.VramReleaser(
            light_video=0.0, deep_video=0.0, progress=_Video(1e9),
        )
        assert not releaser.enabled
        assert releaser.tick() is False
        assert released == []

    def test_default_hook_releases_without_custom_hook(self, released):
        video = _Video(600.0)
        releaser = accelerator.VramReleaser(
            light_video=600.0, deep_video=0.0, progress=video,
        )
        assert releaser.tick() is True
        assert len(released) == 1  # fell back to release_vram_cache

    def test_cadences_from_env(self, monkeypatch, released):
        monkeypatch.setenv(accelerator._VRAM_LIGHT_VIDEO_ENV, "900")
        monkeypatch.setenv(accelerator._VRAM_DEEP_VIDEO_ENV, "1800")
        assert accelerator.light_video_seconds() == 900.0
        assert accelerator.deep_video_seconds() == 1800.0
        releaser = accelerator.VramReleaser(progress=_Video(0.0))
        assert releaser.light_video == 900.0
        assert releaser.deep_video == 1800.0

    def test_bad_progress_is_not_a_trigger(self, released):
        def _boom():
            raise RuntimeError("no reading")

        deep_calls: list[str] = []
        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=3600.0, progress=_boom,
            deep_hook=lambda: deep_calls.append("deep") or "",
        )
        assert releaser.tick() is False
        assert deep_calls == []
        assert released == []

    def test_bad_hook_still_releases_and_does_not_raise(self, released):
        def _boom():
            raise RuntimeError("hook failed")

        releaser = accelerator.VramReleaser(
            light_video=1200.0, deep_video=600.0,
            progress=_Video(600.0), deep_hook=_boom,
        )
        # A failed tidy-up must not end the pass: tick reports the sweep and still
        # returns the allocator's cold cache.
        assert releaser.tick() is True
        assert len(released) == 1


class TestMemoryTrend:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(accelerator._MEM_TREND_ENV, raising=False)
        assert accelerator.mem_trend_seconds() == 0.0
        assert not accelerator.MemoryTrend(accelerator.mem_trend_seconds()).enabled

    def test_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv(accelerator._MEM_TREND_ENV, "120")
        assert accelerator.mem_trend_seconds() == 120.0

    def test_disabled_never_fires(self, clock, caplog):
        trend = accelerator.MemoryTrend(0.0)
        clock.advance(10_000)
        assert trend.tick() is False
        assert "MEM TREND" not in caplog.text

    def test_fires_once_per_period_with_extra(self, clock, caplog):
        trend = accelerator.MemoryTrend(10.0, extra=lambda: {"blend_results": 1})
        clock.advance(9)
        assert trend.tick() is False
        clock.advance(1)
        with caplog.at_level("INFO", logger="jasna.accelerator"):
            assert trend.tick() is True
        assert "MEM TREND" in caplog.text
        assert "blend_results=1" in caplog.text

    def test_extra_failure_does_not_raise(self, clock, caplog):
        def _boom():
            raise RuntimeError("nope")

        trend = accelerator.MemoryTrend(1.0, extra=_boom)
        clock.advance(1)
        with caplog.at_level("INFO", logger="jasna.accelerator"):
            assert trend.tick() is True
        assert "extra_failed" in caplog.text

"""MIGraphX EP preflight: a provider that cannot start must cost a fallback, not a crash.

The EP is a native library that fails by terminating the process, so the check runs in a
child process and its verdict is cached. These tests drive that contract with a fake
``subprocess.run`` - no GPU and no ORT session involved.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

from jasna.mosaic import rfdetr as rfdetr_mod
from jasna.mosaic import rfdetr_migraphx_runner as mm

STATUS_ACCESS_VIOLATION = 0xC0000005


def _cache(tmp_path: Path, monkeypatch):
    path = tmp_path / "migraphx-ep-preflight.json"
    monkeypatch.setattr(mm, "migraphx_preflight_cache_path", lambda: path)
    monkeypatch.setattr(
        mm, "migraphx_preflight_fingerprint", lambda: {"ep_dir": "x", "arch": "gfx1100"}
    )
    monkeypatch.delenv(mm.MIGRAPHX_PREFLIGHT_ENV, raising=False)
    return path


def test_probe_ok_is_usable_and_cached(tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr(mm.subprocess, "run", fake_run)
    usable, reason = mm.migraphx_ep_usable()
    assert usable and reason == "probe ok"
    assert cache.is_file()

    usable_again, _ = mm.migraphx_ep_usable()
    assert usable_again
    assert len(calls) == 1, "the cached verdict must spare the second probe"


def test_probe_crash_marks_the_ep_unusable(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mm.subprocess, "run", lambda *a, **k: MagicMock(returncode=STATUS_ACCESS_VIOLATION)
    )
    usable, reason = mm.migraphx_ep_usable()
    assert usable is False
    assert str(STATUS_ACCESS_VIOLATION) in reason


def test_probe_timeout_marks_the_ep_unusable(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise mm.subprocess.TimeoutExpired("probe", 1)

    monkeypatch.setattr(mm.subprocess, "run", boom)
    usable, reason = mm.migraphx_ep_usable()
    assert usable is False
    assert "timed out" in reason


def test_env_flag_skips_the_probe(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    monkeypatch.setenv(mm.MIGRAPHX_PREFLIGHT_ENV, "0")
    called = []
    monkeypatch.setattr(mm.subprocess, "run", lambda *a, **k: called.append(1))
    usable, _ = mm.migraphx_ep_usable()
    assert usable is True
    assert called == []


def test_changed_fingerprint_forces_a_new_probe(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        mm.subprocess, "run",
        lambda *a, **k: (calls.append(1), MagicMock(returncode=1))[1],
    )
    mm.migraphx_ep_usable()
    monkeypatch.setattr(
        mm, "migraphx_preflight_fingerprint", lambda: {"ep_dir": "y", "arch": "gfx1100"}
    )
    mm.migraphx_ep_usable()
    assert len(calls) == 2, "a different EP runtime must be probed again"


def test_engine_falls_back_to_torch_when_the_ep_is_unusable(monkeypatch):
    with patch.object(mm, "migraphx_ep_usable", return_value=(False, "0xC0000005")):
        assert rfdetr_mod.resolve_detection_engine("migraphx") == "torch"


def test_engine_keeps_migraphx_when_the_probe_passes(monkeypatch):
    with patch.object(mm, "migraphx_ep_usable", return_value=(True, "probe ok")):
        assert rfdetr_mod.resolve_detection_engine("migraphx") == "migraphx"


def test_other_engines_are_untouched(monkeypatch):
    monkeypatch.setattr(
        mm, "migraphx_ep_usable",
        lambda **kw: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    assert rfdetr_mod.resolve_detection_engine("torch") == "torch"

import sys
import types

from jasna import startup_timing
from jasna.gui.wizard import _evaluate_check_results


def test_elapsed_ms_is_monotonic_nonnegative():
    first = startup_timing.elapsed_ms()
    second = startup_timing.elapsed_ms()
    assert first >= 0.0
    assert second >= first


def test_evaluate_all_passed():
    results = {"ascii_path": (True, ""), "gpu": (True, ""), "sysmem": (True, "")}
    all_passed, required_failure = _evaluate_check_results(results, results.keys())
    assert all_passed is True
    assert required_failure is False


def test_evaluate_missing_check_counts_as_required_failure():
    # gpu/cuda never ran (e.g. check thread died) -> must read as failure, never "ready".
    results = {"ascii_path": (True, "")}
    all_passed, required_failure = _evaluate_check_results(
        results, ["ascii_path", "gpu", "cuda"]
    )
    assert all_passed is False
    assert required_failure is True


def test_evaluate_sysmem_only_failure_is_warning_not_required():
    results = {"gpu": (True, ""), "sysmem": (False, "")}
    all_passed, required_failure = _evaluate_check_results(results, results.keys())
    assert all_passed is False  # a warning still means "not all passed"
    assert required_failure is False  # but sysmem is warning-only, not blocking


def test_warm_up_cuda_inits_context_when_available(monkeypatch):
    calls = {}
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        zeros=lambda *a, **k: calls.setdefault("device", k.get("device")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from jasna.gui.app import _warm_up_cuda

    _warm_up_cuda()
    assert calls["device"] == "cuda"


def test_warm_up_cuda_skips_when_no_gpu(monkeypatch):
    calls = {}
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        zeros=lambda *a, **k: calls.setdefault("called", True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from jasna.gui.app import _warm_up_cuda

    _warm_up_cuda()
    assert "called" not in calls

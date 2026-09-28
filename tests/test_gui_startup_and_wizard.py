import sys
import types

import pytest

from jasna import startup_timing
from jasna.gui.system_checks import evaluate_check_results


def test_elapsed_ms_is_monotonic_nonnegative():
    first = startup_timing.elapsed_ms()
    second = startup_timing.elapsed_ms()
    assert first >= 0.0
    assert second >= first


def test_evaluate_all_passed():
    results = {"ascii_path": (True, ""), "gpu": (True, ""), "sysmem": (True, "")}
    all_passed, required_failure = evaluate_check_results(results, results.keys())
    assert all_passed is True
    assert required_failure is False


def test_evaluate_missing_check_counts_as_required_failure():
    # gpu/cuda never ran (e.g. check thread died) -> must read as failure, never "ready".
    results = {"ascii_path": (True, "")}
    all_passed, required_failure = evaluate_check_results(
        results, ["ascii_path", "gpu", "cuda"]
    )
    assert all_passed is False
    assert required_failure is True


def test_evaluate_sysmem_only_failure_is_warning_not_required():
    results = {"gpu": (True, ""), "sysmem": (False, "")}
    all_passed, required_failure = evaluate_check_results(results, results.keys())
    assert all_passed is False  # a warning still means "not all passed"
    assert required_failure is False  # but sysmem is warning-only, not blocking


def _warm_up_with_gpu_check(monkeypatch, gpu_check_result) -> dict:
    from jasna import os_utils

    calls = {}
    fake_torch = types.SimpleNamespace(zeros=lambda *a, **k: calls.setdefault("device", k.get("device")))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(os_utils, "check_supported_gpu", lambda: gpu_check_result)
    from jasna.gui.app import _warm_up_cuda

    _warm_up_cuda()
    return calls


def test_warm_up_cuda_inits_context_when_gpu_supported(monkeypatch):
    calls = _warm_up_with_gpu_check(monkeypatch, (True, "RTX 4090"))
    assert calls["device"] == "cuda"


@pytest.mark.parametrize(
    "gpu_check_result",
    [(False, "no_cuda"), (False, ("arch_unsupported", "gfx1103"))],
)
def test_warm_up_cuda_launches_no_kernel_on_unsupported_gpu(monkeypatch, gpu_check_result):
    calls = _warm_up_with_gpu_check(monkeypatch, gpu_check_result)
    assert calls == {}

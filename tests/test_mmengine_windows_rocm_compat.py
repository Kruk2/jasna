from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jasna.models.basicvsrpp.mmengine_compat import (
    _FSDP_MODULE,
    prepare_mmengine_for_windows_rocm,
)


def _fake_torch(
    *,
    hip: str | None = "7.2.1",
    distributed_available: bool = False,
    with_reduce_op: bool = False,
) -> SimpleNamespace:
    distributed = SimpleNamespace(
        is_available=lambda: distributed_available,
    )
    if with_reduce_op:
        distributed.ReduceOp = SimpleNamespace(SUM="existing")
    return SimpleNamespace(
        distributed=distributed,
        version=SimpleNamespace(hip=hip),
    )


@pytest.mark.parametrize(
    ("platform", "torch_module"),
    [
        ("linux", _fake_torch()),
        ("win32", _fake_torch(hip=None)),
        ("win32", _fake_torch(distributed_available=True)),
    ],
)
def test_compatibility_does_not_modify_other_torch_environments(
    monkeypatch,
    platform: str,
    torch_module: SimpleNamespace,
) -> None:
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform=platform,
    )

    assert installed == ()
    assert not hasattr(torch_module.distributed, "ReduceOp")
    assert _FSDP_MODULE not in sys.modules


def test_missing_windows_rocm_distributed_surfaces_are_stubbed(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    torch_module = _fake_torch()

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
    )

    assert installed == (
        "torch.distributed.ReduceOp",
        "mmengine FSDP stub",
    )
    assert not torch_module.distributed.is_available()
    assert torch_module.distributed.ReduceOp.SUM == "sum"
    assert torch_module.distributed.ReduceOp.BXOR == "bxor"
    with pytest.raises(RuntimeError, match="Windows ROCm"):
        sys.modules[_FSDP_MODULE].MMFullyShardedDataParallel()


def test_compatibility_setup_is_idempotent(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    torch_module = _fake_torch()
    prepare_mmengine_for_windows_rocm(torch_module, platform="win32")

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
    )

    assert installed == ()
    assert not torch_module.distributed.is_available()


def test_existing_reduce_op_is_not_replaced(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    torch_module = _fake_torch(with_reduce_op=True)
    original = torch_module.distributed.ReduceOp

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
    )

    assert installed == ("mmengine FSDP stub",)
    assert torch_module.distributed.ReduceOp is original


def test_real_windows_rocm_mmengine_import_keeps_distributed_unavailable() -> None:
    if sys.platform != "win32":
        pytest.skip("Windows ROCm import contract")
    import torch

    if not torch.version.hip or torch.distributed.is_available():
        pytest.skip("requires a Windows ROCm torch build without distributed c10d")

    repo_root = Path(__file__).resolve().parents[1]
    code = """
import torch
import jasna.models.basicvsrpp
from mmengine.runner import load_checkpoint
assert callable(load_checkpoint)
assert not torch.distributed.is_available()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout

import os
from pathlib import Path

import pytest

from jasna.engine_paths import (
    adopt_legacy_engine,
    engine_system_suffix,
    get_basicvsrpp_sub_engine_paths,
    get_onnx_tensorrt_engine_path,
    get_unet4x_encrypted_engine_path,
    get_unet4x_engine_path,
    legacy_engine_path,
)


@pytest.mark.parametrize(("os_name", "suffix"), [("nt", ".sm89.win"), ("posix", ".sm89.linux")])
def test_engine_names_carry_the_gpu_and_operating_system(monkeypatch, os_name, suffix):
    monkeypatch.setattr(os, "name", os_name)
    assert engine_system_suffix("sm89") == suffix


def test_an_nvidia_gpu_is_named_by_its_compute_capability(cuda_capability):
    from jasna.accelerator import cuda_arch_name

    cuda_capability((8, 6))
    assert cuda_arch_name("cuda:0") == "sm86"


@pytest.mark.parametrize("engine_names", [
    lambda gpu: [get_onnx_tensorrt_engine_path("rfdetr-v6.onnx", gpu=gpu, batch_size=4, fp16=True, dynamic_batch=True)],
    lambda gpu: list(get_basicvsrpp_sub_engine_paths("model.pth", True, gpu).values()),
    lambda gpu: [get_unet4x_engine_path(fp16=True, gpu=gpu), get_unet4x_encrypted_engine_path(fp16=True, gpu=gpu)],
], ids=["onnx", "basicvsrpp", "unet4x"])
def test_each_gpu_architecture_gets_its_own_engines(engine_names):
    assert set(engine_names("sm89")).isdisjoint(engine_names("sm86"))


def test_the_legacy_name_is_the_name_without_the_gpu():
    current = get_onnx_tensorrt_engine_path("dir/rfdetr-v6.onnx", gpu="sm89", batch_size=4, fp16=True)

    assert legacy_engine_path(current, "sm89") == (
        Path("dir") / f"rfdetr-v6.bs4.fp16{'.win' if os.name == 'nt' else '.linux'}.engine"
    )


@pytest.mark.parametrize(("runs_here", "current_exists", "adopted"), [
    (True, False, True),
    (False, False, False),
    (True, True, False),
], ids=["runs-here", "built-for-another-gpu", "already-compiled"])
def test_a_legacy_engine_is_renamed_only_when_it_runs_on_this_gpu(tmp_path, runs_here, current_exists, adopted):
    current = get_onnx_tensorrt_engine_path(tmp_path / "model.onnx", gpu="sm89", fp16=True)
    legacy = legacy_engine_path(current, "sm89")
    legacy.write_bytes(b"legacy engine")
    if current_exists:
        current.write_bytes(b"current engine")

    adopt_legacy_engine(current, "sm89", lambda _path: runs_here)

    assert legacy.exists() is not adopted
    assert current.exists() is (adopted or current_exists)
    if adopted:
        assert current.read_bytes() == b"legacy engine"

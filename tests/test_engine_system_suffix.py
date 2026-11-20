import os
from pathlib import Path

import pytest

from jasna import engine_paths
from jasna.engine_paths import (
    adopt_legacy_engine,
    engine_system_suffix,
    get_basicvsrpp_sub_engine_paths,
    get_onnx_tensorrt_engine_path,
    get_unet4x_encrypted_engine_path,
    get_unet4x_engine_path,
    legacy_engine_path,
)


@pytest.fixture
def gpu(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(engine_paths, "engine_gpu_name", lambda: name)

    use("sm89")
    return use


@pytest.mark.parametrize(("os_name", "suffix"), [("nt", ".sm89.win"), ("posix", ".sm89.linux")])
def test_engine_names_carry_the_gpu_and_operating_system(monkeypatch, gpu, os_name, suffix):
    monkeypatch.setattr(os, "name", os_name)
    assert engine_system_suffix() == suffix


@pytest.mark.parametrize("engine_names", [
    lambda: [get_onnx_tensorrt_engine_path("rfdetr-v6.onnx", batch_size=4, fp16=True, dynamic_batch=True)],
    lambda: list(get_basicvsrpp_sub_engine_paths("model.pth", fp16=True).values()),
    lambda: [get_unet4x_engine_path(fp16=True), get_unet4x_encrypted_engine_path(fp16=True)],
], ids=["onnx", "basicvsrpp", "unet4x"])
def test_each_gpu_architecture_gets_its_own_engines(gpu, engine_names):
    gpu("sm89")
    ada = engine_names()
    gpu("sm86")
    ampere = engine_names()

    assert set(ada).isdisjoint(ampere)


def test_the_legacy_name_is_the_name_without_the_gpu(gpu):
    legacy = legacy_engine_path(get_onnx_tensorrt_engine_path("dir/rfdetr-v6.onnx", batch_size=4, fp16=True))

    assert legacy == Path("dir") / f"rfdetr-v6.bs4.fp16{'.win' if os.name == 'nt' else '.linux'}.engine"


@pytest.mark.parametrize(("runs_here", "current_exists", "adopted"), [
    (True, False, True),
    (False, False, False),
    (True, True, False),
], ids=["runs-here", "built-for-another-gpu", "already-compiled"])
def test_a_legacy_engine_is_renamed_only_when_it_runs_on_this_gpu(gpu, tmp_path, runs_here, current_exists, adopted):
    current = get_onnx_tensorrt_engine_path(tmp_path / "model.onnx", fp16=True)
    legacy = legacy_engine_path(current)
    legacy.write_bytes(b"legacy engine")
    if current_exists:
        current.write_bytes(b"current engine")

    adopt_legacy_engine(current, lambda _path: runs_here)

    assert legacy.exists() is not adopted
    assert current.exists() is (adopted or current_exists)
    if adopted:
        assert current.read_bytes() == b"legacy engine"

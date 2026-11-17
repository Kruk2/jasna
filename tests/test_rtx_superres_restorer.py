"""Tests for jasna.restorer.rtx_superres_secondary_restorer covering init, restore, and close."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer


class _Quality:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    DENOISE_LOW = "denoise_low"
    DENOISE_MEDIUM = "denoise_medium"
    DENOISE_HIGH = "denoise_high"
    DENOISE_ULTRA = "denoise_ultra"
    DEBLUR_LOW = "deblur_low"
    DEBLUR_MEDIUM = "deblur_medium"
    DEBLUR_HIGH = "deblur_high"
    DEBLUR_ULTRA = "deblur_ultra"


class _MockVideoSuperRes:
    QualityLevel = _Quality

    def __init__(self, *, device, quality):
        self.device_id = device
        self.quality = quality
        self.output_width = 0
        self.output_height = 0

    def load(self):
        pass

    def run(self, frame, stream_ptr=None):
        result = MagicMock()
        result.image = torch.rand_like(frame)
        return result

    def close(self):
        pass


@pytest.fixture
def fake_nvvfx(monkeypatch):
    """nvvfx (RTX Video Effects) replaced by a fake for this test only."""
    module = ModuleType("nvvfx")
    module.VideoSuperRes = _MockVideoSuperRes
    monkeypatch.setitem(sys.modules, "nvvfx", module)


def _make_restorer(*, scale=4, quality="high", denoise="medium", deblur="none"):
    with patch("torch.cuda.current_stream") as mock_stream:
        mock_stream.return_value.cuda_stream = 12345
        return RtxSuperresSecondaryRestorer(
            device=torch.device("cuda:0"),
            scale=scale,
            quality=quality,
            denoise=denoise,
            deblur=deblur,
        )


TENSORRT_LOAD_ORDER_SCRIPT = """
import ctypes, importlib.abc, importlib.util, os, sys
import jasna.frame_generation, jasna.ltx.super_res, jasna.restorer.rtx_superres_secondary_restorer, jasna.upscale.rtx
from jasna.upscale.nvvfx_effects import import_nvvfx

class FakeNvvfx(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.util.spec_from_loader(name, self) if name == "nvvfx" else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        process = ctypes.CDLL(None)
        module.global_tensorrt = process.getInferLibVersion() if hasattr(process, "getInferLibVersion") else None

imported_by_modules = "nvvfx" in sys.modules
sys.meta_path.insert(0, FakeNvvfx())
nvvfx = import_nvvfx()
import tensorrt_libs
pip_tensorrt = ctypes.CDLL(os.path.join(os.path.dirname(tensorrt_libs.__file__), "libnvinfer.so.10")).getInferLibVersion()
print(imported_by_modules, nvvfx.global_tensorrt == pip_tensorrt)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="RTLD_GLOBAL symbol-scope check is POSIX-specific")
def test_nvvfx_imports_after_the_pip_tensorrt_runtime_holds_the_global_scope():
    """nvvfx bundles its own libnvinfer under the same SONAME as the pip tensorrt our BasicVSR++
    engines are serialized against and loads it with RTLD_GLOBAL; if it loads first, engine
    deserialization fails with a version mismatch. A fresh process, so no nvvfx is loaded yet."""
    result = subprocess.run(
        [sys.executable, "-c", TENSORRT_LOAD_ORDER_SCRIPT],
        capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1], timeout=60,
    )
    assert result.stdout.split("\n")[-2] == "False True"


@pytest.mark.usefixtures("fake_nvvfx")
class TestRtxSuperresRestore:
    @pytest.mark.parametrize(
        ("frame_count", "keep_start", "keep_end"),
        [(0, 0, 0), (5, 3, 3), (5, 4, 2)],
        ids=["no-frames", "empty-keep-range", "keep-end-before-start"],
    )
    def test_restore_returns_nothing_for_an_empty_keep_range(self, frame_count, keep_start, keep_end):
        restorer = _make_restorer()
        frames = torch.rand((frame_count, 3, 256, 256), dtype=torch.float32)
        assert restorer.restore(frames, keep_start=keep_start, keep_end=keep_end) == []

    def test_restore_rejects_wrong_input_size(self):
        restorer = _make_restorer()
        frames = torch.rand((2, 3, 384, 384), dtype=torch.float32)

        with pytest.raises(ValueError, match="expected frames shaped"):
            restorer.restore(frames, keep_start=0, keep_end=2)

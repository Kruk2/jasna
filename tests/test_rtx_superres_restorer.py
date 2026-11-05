"""Tests for jasna.restorer.rtx_superres_secondary_restorer covering init, restore, and close."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

# Install the mock nvvfx module BEFORE importing the rtx module so local
# ``from nvvfx import VideoSuperRes`` calls resolve correctly.

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


_mock_nvvfx = ModuleType("nvvfx")
_mock_nvvfx.VideoSuperRes = _MockVideoSuperRes
sys.modules.setdefault("nvvfx", _mock_nvvfx)

from jasna.restorer.rtx_superres_secondary_restorer import (
    _resolve_quality,
    _resolve_denoise,
    _resolve_deblur,
    RtxSuperresSecondaryRestorer,
)


def _make_restorer(*, scale=4, quality="high", denoise="medium", deblur=None):
    with patch("torch.cuda.current_stream") as mock_stream:
        mock_stream.return_value.cuda_stream = 12345
        return RtxSuperresSecondaryRestorer(
            device=torch.device("cuda:0"),
            scale=scale,
            quality=quality,
            denoise=denoise,
            deblur=deblur,
        )


class TestTensorrtLoadOrder:
    """nvvfx bundles its own libnvinfer under the same SONAME as the pip tensorrt
    our BasicVSR++ engines are serialized against, and loads it with RTLD_GLOBAL.
    The pip runtime must claim the global symbol scope *before* nvvfx loads, so
    the module preloads it at import and only ever imports nvvfx lazily (inside
    functions). If that order breaks, BasicVSR++ engine deserialization fails
    with a TensorRT serialization-version mismatch.
    """

    def test_import_preloads_tensorrt_without_loading_nvvfx(self):
        # A fresh process: this file's fake nvvfx module must not be visible.
        code = (
            "import ctypes, sys\n"
            "import jasna.restorer.rtx_superres_secondary_restorer\n"
            "assert 'nvvfx' not in sys.modules\n"
            "assert sys.platform == 'win32' or hasattr(ctypes.CDLL(None), 'getInferLibVersion')\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1], timeout=60)

    def test_preload_loads_tensorrt_libnvinfer_globally(self):
        import jasna.restorer.rtx_superres_secondary_restorer as mod
        # Importing the module runs the preload; the pip tensorrt libnvinfer
        # must then be resolvable in the process-global symbol scope.
        if sys.platform == "win32":
            pytest.skip("RTLD_GLOBAL symbol-scope check is POSIX-specific")
        mod._preload_tensorrt_runtime()
        get_version = ctypes.CDLL(None).getInferLibVersion
        get_version.restype = ctypes.c_int
        import tensorrt_libs
        libs_dir = os.path.dirname(tensorrt_libs.__file__)
        expected = ctypes.CDLL(os.path.join(libs_dir, "libnvinfer.so.10")).getInferLibVersion
        expected.restype = ctypes.c_int
        assert get_version() == expected()


class TestResolveHelpers:
    def test_resolve_quality(self):
        from nvvfx import VideoSuperRes
        assert _resolve_quality("low") == VideoSuperRes.QualityLevel.LOW
        assert _resolve_quality("HIGH") == VideoSuperRes.QualityLevel.HIGH
        assert _resolve_quality("Ultra") == VideoSuperRes.QualityLevel.ULTRA

    def test_resolve_quality_invalid(self):
        with pytest.raises(KeyError):
            _resolve_quality("invalid")

    def test_resolve_denoise(self):
        from nvvfx import VideoSuperRes
        assert _resolve_denoise("low") == VideoSuperRes.QualityLevel.DENOISE_LOW
        assert _resolve_denoise("MEDIUM") == VideoSuperRes.QualityLevel.DENOISE_MEDIUM

    def test_resolve_denoise_invalid(self):
        with pytest.raises(KeyError):
            _resolve_denoise("invalid")

    def test_resolve_deblur(self):
        from nvvfx import VideoSuperRes
        assert _resolve_deblur("low") == VideoSuperRes.QualityLevel.DEBLUR_LOW
        assert _resolve_deblur("HIGH") == VideoSuperRes.QualityLevel.DEBLUR_HIGH

    def test_resolve_deblur_invalid(self):
        with pytest.raises(KeyError):
            _resolve_deblur("invalid")


class TestRtxSuperresRestore:
    def test_restore_empty_tensor(self):
        restorer = _make_restorer()
        frames = torch.zeros((0, 3, 256, 256), dtype=torch.float32)
        result = restorer.restore(frames, keep_start=0, keep_end=0)
        assert result == []

    def test_restore_empty_keep_range(self):
        restorer = _make_restorer()
        frames = torch.rand((5, 3, 256, 256), dtype=torch.float32)
        result = restorer.restore(frames, keep_start=3, keep_end=3)
        assert result == []

    def test_restore_keep_end_before_start(self):
        restorer = _make_restorer()
        frames = torch.rand((5, 3, 256, 256), dtype=torch.float32)
        result = restorer.restore(frames, keep_start=4, keep_end=2)
        assert result == []

    def test_restore_rejects_wrong_input_size(self):
        restorer = _make_restorer()
        frames = torch.rand((2, 3, 384, 384), dtype=torch.float32)

        with pytest.raises(ValueError, match="expected frames shaped"):
            restorer.restore(frames, keep_start=0, keep_end=2)

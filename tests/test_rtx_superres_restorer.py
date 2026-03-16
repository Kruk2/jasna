"""Tests for jasna.restorer.rtx_superres_secondary_restorer covering init, restore, and close."""
from __future__ import annotations

import sys
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


class TestRtxSuperresInit:
    def test_basic_init(self):
        restorer = _make_restorer(scale=4, quality="high", denoise="medium", deblur=None)
        assert restorer.name == "rtx-super-res"
        assert restorer.num_workers == 1
        assert restorer._sr is not None
        assert restorer._denoise is not None
        assert restorer._deblur is None

    def test_init_with_deblur(self):
        restorer = _make_restorer(scale=2, quality="low", denoise=None, deblur="high")
        assert restorer._sr is not None
        assert restorer._denoise is None
        assert restorer._deblur is not None

    def test_init_no_denoise_no_deblur(self):
        restorer = _make_restorer(scale=4, quality="medium", denoise=None, deblur=None)
        assert restorer._denoise is None
        assert restorer._deblur is None

    def test_init_denoise_none_string(self):
        restorer = _make_restorer(scale=4, quality="high", denoise="none", deblur="none")
        assert restorer._denoise is None
        assert restorer._deblur is None


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

    def test_restore_normal(self):
        restorer = _make_restorer()
        frames = torch.rand((3, 3, 256, 256), dtype=torch.float32)

        mock_sr_result = MagicMock()
        mock_sr_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._sr.run = MagicMock(return_value=mock_sr_result)

        mock_dn_result = MagicMock()
        mock_dn_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._denoise.run = MagicMock(return_value=mock_dn_result)

        result = restorer.restore(frames, keep_start=0, keep_end=3)
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == torch.uint8
            assert frame.shape == (3, 1024, 1024)

    def test_restore_with_keep_slicing(self):
        restorer = _make_restorer()
        frames = torch.rand((10, 3, 256, 256), dtype=torch.float32)

        mock_sr_result = MagicMock()
        mock_sr_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._sr.run = MagicMock(return_value=mock_sr_result)

        mock_dn_result = MagicMock()
        mock_dn_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._denoise.run = MagicMock(return_value=mock_dn_result)

        result = restorer.restore(frames, keep_start=2, keep_end=7)
        assert len(result) == 5
        assert restorer._sr.run.call_count == 5

    def test_restore_with_deblur(self):
        restorer = _make_restorer(denoise="medium", deblur="low")
        frames = torch.rand((2, 3, 256, 256), dtype=torch.float32)

        mock_sr_result = MagicMock()
        mock_sr_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._sr.run = MagicMock(return_value=mock_sr_result)

        mock_dn_result = MagicMock()
        mock_dn_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._denoise.run = MagicMock(return_value=mock_dn_result)

        mock_db_result = MagicMock()
        mock_db_result.image = torch.rand((3, 1024, 1024), dtype=torch.float32)
        restorer._deblur.run = MagicMock(return_value=mock_db_result)

        result = restorer.restore(frames, keep_start=0, keep_end=2)
        assert len(result) == 2
        assert restorer._deblur.run.call_count == 2


class TestRtxSuperresClose:
    def test_close_all(self):
        restorer = _make_restorer(denoise="medium", deblur="low")
        restorer.close()
        assert restorer._sr is None
        assert restorer._denoise is None
        assert restorer._deblur is None

    def test_close_no_denoise_no_deblur(self):
        restorer = _make_restorer(denoise=None, deblur=None)
        restorer.close()
        assert restorer._sr is None
        assert restorer._denoise is None
        assert restorer._deblur is None

    def test_close_idempotent(self):
        restorer = _make_restorer(denoise=None, deblur=None)
        restorer.close()
        restorer.close()
        assert restorer._sr is None

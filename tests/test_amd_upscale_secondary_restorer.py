"""Tests for jasna.restorer.amd_upscale_secondary_restorer.

FFmpeg is never executed: ``subprocess.run`` is replaced with an in-process fake,
so the suite stays fast and GPU-free (mirroring tests/test_rtx_superres_restorer.py).
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jasna.restorer import amd_upscale_secondary_restorer as mod
from jasna.restorer.amd_upscale_secondary_restorer import (
    AMD_UPSCALE_DEFAULT_ALGORITHM,
    AMD_UPSCALE_INPUT_SIZE,
    AmdUpscaleSecondaryRestorer,
    AmdUpscaleTimeout,
    build_filter_chain,
)

_ORIGINAL_RESOLVE = mod._resolve_ffmpeg


class _FakeFfmpeg:
    """Stands in for ``subprocess.run``: records the call, returns fake frames."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.returncode = 0
        self.stderr = b""
        self.raise_timeout = False
        self.frames_override: int | None = None

    def __call__(self, cmd, *, input=None, stdout=None, stderr=None, timeout=None, **kwargs):  # noqa: A002
        self.calls.append({"cmd": cmd, "input": input, "timeout": timeout, "kwargs": kwargs})
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout or 1)
        out_size = 1024
        for i, token in enumerate(cmd):
            if token == "-vf" and "w=" in cmd[i + 1]:
                out_size = int(cmd[i + 1].split("w=")[1].split(":")[0])
                break
        count = len(input) // (AMD_UPSCALE_INPUT_SIZE * AMD_UPSCALE_INPUT_SIZE * 3)
        produced = self.frames_override if self.frames_override is not None else count
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=bytes(out_size * out_size * 3 * produced),
            stderr=self.stderr,
        )


@pytest.fixture
def fake_ffmpeg(monkeypatch) -> _FakeFfmpeg:
    fake = _FakeFfmpeg()
    monkeypatch.setattr(mod.subprocess, "run", fake)
    monkeypatch.setattr(mod, "_resolve_ffmpeg", lambda path: path or "ffmpeg-test")
    return fake


def _restorer(**kwargs) -> AmdUpscaleSecondaryRestorer:
    params = dict(scale=4, engine="amf-sr", algorithm=AMD_UPSCALE_DEFAULT_ALGORITHM)
    params.update(kwargs)
    return AmdUpscaleSecondaryRestorer(**params)


def _frames(count: int, size: int = AMD_UPSCALE_INPUT_SIZE) -> torch.Tensor:
    return torch.rand(count, 3, size, size)


class TestFilterChain:
    def test_amf_chain_uploads_to_the_amf_device_and_back(self):
        chain = build_filter_chain(engine="amf-sr", output_size=1024, algorithm="sr1-0", sharpness=-1.0)
        assert chain.startswith("format=nv12,hwupload=derive_device=amf,")
        assert "sr_amf=w=1024:h=1024:algorithm=sr1-0:sharpness=-1," in chain
        assert chain.endswith("hwdownload,format=nv12,format=rgb24")

    def test_amf_chain_carries_scale_and_sharpness(self):
        chain = build_filter_chain(engine="amf-sr", output_size=512, algorithm="sr1-1", sharpness=0.5)
        assert "sr_amf=w=512:h=512:algorithm=sr1-1:sharpness=0.5," in chain

    def test_libplacebo_engine_is_gone(self):
        with pytest.raises(ValueError, match="Unsupported AMD upscale engine"):
            build_filter_chain(engine="libplacebo", output_size=1024)

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError, match="Unsupported AMD upscale engine"):
            build_filter_chain(engine="magic", output_size=1024)


class TestInit:
    def test_defaults(self):
        r = _restorer()
        assert r.name == "amd-upscale"
        assert r.prefers_cpu_input is True  # frames cross to the FFmpeg process
        assert r.num_workers == 1
        assert (r.input_size, r.output_size) == (256, 1024)

    @pytest.mark.parametrize("scale,expected", [(2, 512), (4, 1024), (6, 1536), (8, 2048)])
    def test_scale_maps_to_output_size(self, scale, expected):
        assert _restorer(scale=scale).output_size == expected

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="Invalid AMD upscale factor"):
            _restorer(scale=3)

    def test_invalid_engine(self):
        with pytest.raises(ValueError, match="Invalid AMD upscale engine"):
            _restorer(engine="cuda")

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Invalid AMD upscale algorithm"):
            _restorer(algorithm="dlss")

    def test_realesrgan_engine_is_not_handled_by_this_class(self):
        # the Real-ESRGAN engine lives in realesrgan_secondary_restorer.py
        with pytest.raises(ValueError, match="Invalid AMD upscale engine"):
            _restorer(engine="realesrgan")

    def test_sharpness_range(self):
        assert _restorer(sharpness=2.0).sharpness == 2.0
        with pytest.raises(ValueError, match="sharpness must be in"):
            _restorer(sharpness=2.5)

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            _restorer(timeout_s=0)


class TestCommand:
    def test_amf_command_initialises_amf_hwdevice(self, fake_ffmpeg):
        cmd = _restorer(scale=2).build_ffmpeg_cmd()
        assert cmd[0] == "ffmpeg-test"
        assert cmd[cmd.index("-init_hw_device") + 1] == "amf=amd"
        assert "sr_amf=w=512:h=512" in cmd[cmd.index("-vf") + 1]
        assert cmd[cmd.index("-s") + 1] == "256x256"
        assert cmd[-1] == "pipe:1"
        assert cmd[cmd.index("-i") + 1] == "pipe:0"

    def test_device_spec_is_passed_through(self, fake_ffmpeg):
        assert "amf=amd:1" in _restorer(device_spec="amd:1").build_ffmpeg_cmd()


class TestRestore:
    def test_returns_chw_uint8_frames(self, fake_ffmpeg):
        out = _restorer(scale=2).restore(_frames(5), keep_start=0, keep_end=5)
        assert len(out) == 5
        for frame in out:
            assert frame.shape == (3, 512, 512)
            assert frame.dtype == torch.uint8

    def test_keep_window_selects_frames(self, fake_ffmpeg):
        out = _restorer(scale=2).restore(_frames(9), keep_start=2, keep_end=7)
        assert len(out) == 5

    def test_empty_window_never_runs_ffmpeg(self, fake_ffmpeg):
        assert _restorer().restore(_frames(4), keep_start=3, keep_end=3) == []
        assert fake_ffmpeg.calls == []

    def test_zero_frames_returns_nothing(self, fake_ffmpeg):
        assert _restorer().restore(torch.empty(0, 3, 256, 256), keep_start=0, keep_end=0) == []
        assert fake_ffmpeg.calls == []

    def test_wrong_shape_rejected(self, fake_ffmpeg):
        with pytest.raises(ValueError, match="expected frames shaped"):
            _restorer().restore(torch.rand(2, 3, 128, 128), keep_start=0, keep_end=2)

    def test_payload_is_contiguous_rgb_uint8(self, fake_ffmpeg):
        frames = torch.zeros(3, 3, 256, 256)
        frames[1] = 1.0
        _restorer(scale=2).restore(frames, keep_start=0, keep_end=3)
        payload = fake_ffmpeg.calls[-1]["input"]
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(3, 256, 256, 3)
        assert arr[1].max() == 255 and arr[0].max() == 0

    def test_hard_timeout_is_handed_to_subprocess(self, fake_ffmpeg):
        _restorer(scale=2, timeout_s=42.5).restore(_frames(2), keep_start=0, keep_end=2)
        assert fake_ffmpeg.calls[-1]["timeout"] == 42.5

    def test_one_ffmpeg_process_per_clip(self, fake_ffmpeg):
        r = _restorer(scale=2)
        r.restore(_frames(4), keep_start=0, keep_end=4)
        r.restore(_frames(4), keep_start=0, keep_end=4)
        assert len(fake_ffmpeg.calls) == 2

    def test_timeout_becomes_amd_upscale_timeout(self, fake_ffmpeg):
        fake_ffmpeg.raise_timeout = True
        with pytest.raises(AmdUpscaleTimeout, match="within 120s"):
            _restorer(scale=2).restore(_frames(2), keep_start=0, keep_end=2)

    def test_nonzero_exit_reports_stderr(self, fake_ffmpeg):
        fake_ffmpeg.returncode = 1
        fake_ffmpeg.stderr = b"Error while filtering: Error number -129 occurred"
        with pytest.raises(RuntimeError, match="exited with 1"):
            _restorer(scale=2).restore(_frames(2), keep_start=0, keep_end=2)

    def test_short_output_is_reported_with_frame_counts(self, fake_ffmpeg):
        fake_ffmpeg.frames_override = 1
        with pytest.raises(RuntimeError, match="returned 1 of 3 frames"):
            _restorer(scale=2).restore(_frames(3), keep_start=0, keep_end=3)

    def test_cpu_float_input_supported(self, fake_ffmpeg):
        assert len(_restorer(scale=2).restore(_frames(2).to(torch.float32), keep_start=0, keep_end=2)) == 2

    def test_cuda_tensor_input_supported(self, fake_ffmpeg, monkeypatch):
        # the pipeline may hand over GPU crops; conversion happens on the GPU first
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: False))
        assert len(_restorer(scale=2).restore(_frames(2), keep_start=0, keep_end=2)) == 2


class TestResolveFfmpeg:
    def test_explicit_missing_path_raises_with_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "_resolve_ffmpeg", _ORIGINAL_RESOLVE)
        missing = tmp_path / "nope" / "ffmpeg.exe"
        with pytest.raises(FileNotFoundError) as excinfo:
            mod._resolve_ffmpeg(str(missing))
        assert str(missing.resolve()) in str(excinfo.value)

    def test_finds_path_ffmpeg(self, monkeypatch):
        import sys

        monkeypatch.setattr(mod, "_resolve_ffmpeg", _ORIGINAL_RESOLVE)
        monkeypatch.setattr(mod, "find_executable", lambda name: sys.executable)
        assert mod._resolve_ffmpeg(None).endswith("python.exe")

    def test_no_ffmpeg_anywhere(self, monkeypatch):
        monkeypatch.setattr(mod, "_resolve_ffmpeg", _ORIGINAL_RESOLVE)
        monkeypatch.setattr(mod, "find_executable", lambda name: None)
        with pytest.raises(FileNotFoundError, match="no bundled copy"):
            mod._resolve_ffmpeg(None)


class TestClose:
    def test_close_is_available_and_idempotent(self, fake_ffmpeg):
        r = _restorer(scale=2)
        r.restore(_frames(2), keep_start=0, keep_end=2)
        r.close()
        r.close()

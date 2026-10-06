"""Tests for the Real-ESRGAN secondary restorer.

Runs on CPU with deliberately tiny networks, so no GPU and no large checkpoint is
needed; the point is shape/scale/loader behaviour, not image quality.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jasna.restorer import realesrgan_secondary_restorer as mod
from jasna.restorer.realesrgan_secondary_restorer import (
    REALESRGAN_WEIGHT_CANDIDATES,
    RealEsrganSecondaryRestorer,
    find_default_weights,
    resolve_weights_path,
)
from jasna.restorer.rrdbnet import RRDBNet, load_rrdbnet, read_checkpoint_spec


def _tiny_checkpoint(
    tmp_path: Path,
    *,
    num_block: int = 1,
    num_feat: int = 8,
    num_grow_ch: int = 4,
    pixel_unshuffle: int = 1,
    wrap: str | None = "params_ema",
    name: str = "tiny.pth",
) -> Path:
    model = RRDBNet(
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        pixel_unshuffle=pixel_unshuffle,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        # constant 0.5 response keeps the output deterministic and non-zero
        model.conv_last.bias.fill_(0.5)
    state: object = {"params_ema": model.state_dict()} if wrap else model.state_dict()
    path = tmp_path / name
    torch.save(state, path)
    return path


def _restorer(tmp_path: Path, **kwargs) -> RealEsrganSecondaryRestorer:
    params = dict(
        device=torch.device("cpu"),
        scale=2,
        model_path=_tiny_checkpoint(tmp_path),
        fp16=False,
    )
    params.update(kwargs)
    return RealEsrganSecondaryRestorer(**params)


def _frames(count: int, size: int = 256) -> torch.Tensor:
    return torch.rand(count, 3, size, size)


class TestCheckpointSpec:
    def test_reads_block_count_feature_width_and_scale(self, tmp_path):
        path = _tiny_checkpoint(tmp_path, num_block=3, num_feat=16, num_grow_ch=8)
        spec = read_checkpoint_spec(path)
        assert (spec.num_block, spec.num_feat, spec.num_grow_ch) == (3, 16, 8)
        assert spec.pixel_unshuffle == 1
        assert spec.native_scale == 4

    def test_pixel_unshuffle_checkpoint_is_2x(self, tmp_path):
        path = _tiny_checkpoint(tmp_path, pixel_unshuffle=2)
        spec = read_checkpoint_spec(path)
        assert spec.pixel_unshuffle == 2
        assert spec.native_scale == 2

    def test_unwrapped_state_dict_is_accepted(self, tmp_path):
        path = _tiny_checkpoint(tmp_path, wrap=None)
        model, spec = load_rrdbnet(path, device=torch.device("cpu"), fp16=False)
        assert spec.num_block == 1
        assert model.conv_first.weight.shape[0] == 8

    def test_mismatched_checkpoint_is_rejected(self, tmp_path):
        path = _tiny_checkpoint(tmp_path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["params_ema"].pop("conv_body.weight")
        torch.save(state, path)
        with pytest.raises(ValueError, match="does not match RRDBNet"):
            load_rrdbnet(path, device=torch.device("cpu"), fp16=False)


class TestWeightResolution:
    def test_explicit_path_wins(self, tmp_path):
        path = _tiny_checkpoint(tmp_path)
        assert resolve_weights_path(path) == path.resolve()

    def test_missing_explicit_path_reports_absolute_path_and_hint(self, tmp_path):
        missing = tmp_path / "sub" / "nope.pth"
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_weights_path(str(missing))
        message = str(excinfo.value)
        assert str(missing.resolve()) in message
        assert "--amd-upscale-model-path" in message

    def test_auto_detect_scans_model_weights(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        assert find_default_weights() is None
        wanted = tmp_path / REALESRGAN_WEIGHT_CANDIDATES[0]
        wanted.write_bytes(b"")
        assert find_default_weights() == wanted

    def test_auto_detect_raises_when_nothing_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="No AMD super-res model found"):
            resolve_weights_path(None)


class TestInit:
    def test_contract_matches_the_rtx_strategy(self, tmp_path):
        r = _restorer(tmp_path, scale=4)
        assert r.name == "amd-upscale"
        assert r.prefers_cpu_input is False  # in-process GPU, like RTX Super Res
        assert r.num_workers == 1
        assert (r.input_size, r.output_size) == (256, 1024)

    @pytest.mark.parametrize("scale,expected", [(2, 512), (4, 1024)])
    def test_scale_sets_output_size(self, tmp_path, scale, expected):
        assert _restorer(tmp_path, scale=scale).output_size == expected

    def test_invalid_scale(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid AMD super-res factor"):
            _restorer(tmp_path, scale=3)

    def test_invalid_batch_size(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _restorer(tmp_path, batch_size=0)


class TestRestore:
    def test_returns_chw_uint8_at_the_requested_size(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        out = r.restore(_frames(5), keep_start=0, keep_end=5)
        assert len(out) == 5
        for frame in out:
            assert frame.shape == (3, 512, 512)
            assert frame.dtype == torch.uint8
            assert frame.max() > 0  # conv_last bias 0.5 -> ~128

    def test_keep_window_is_honoured(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        out = r.restore(_frames(9), keep_start=2, keep_end=7)
        assert len(out) == 5

    def test_empty_window(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        assert r.restore(_frames(4), keep_start=3, keep_end=3) == []

    def test_zero_frames(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        assert r.restore(torch.empty(0, 3, 256, 256), keep_start=0, keep_end=0) == []

    def test_wrong_shape_rejected(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        with pytest.raises(ValueError, match="expected frames shaped"):
            r.restore(torch.rand(2, 3, 128, 128), keep_start=0, keep_end=2)

    def test_batching_splits_work_without_changing_output(self, tmp_path):
        r = _restorer(tmp_path, scale=2, batch_size=2)
        out = r.restore(_frames(5), keep_start=0, keep_end=5)
        assert len(out) == 5
        assert all(f.shape == (3, 512, 512) for f in out)

    def test_4x_native_network_downsampled_to_2x(self, tmp_path):
        # scale=2 with a 4x network must still return 512x512, not 1024x1024
        r = _restorer(tmp_path, scale=2)
        assert r.spec.native_scale == 4
        assert r.restore(_frames(1), keep_start=0, keep_end=1)[0].shape == (3, 512, 512)

    def test_2x_network_upsampled_to_4x_request(self, tmp_path):
        r = _restorer(tmp_path, scale=4, model_path=_tiny_checkpoint(tmp_path, pixel_unshuffle=2, name="u2.pth"))
        assert r.spec.native_scale == 2
        assert r.restore(_frames(1), keep_start=0, keep_end=1)[0].shape == (3, 1024, 1024)

    def test_input_size_must_be_256(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        assert r.input_size == 256


class TestClose:
    def test_close_is_idempotent_and_drops_the_network(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        r.restore(_frames(2), keep_start=0, keep_end=2)
        r.close()
        r.close()
        assert r.model is None

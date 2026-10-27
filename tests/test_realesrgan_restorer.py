"""Tests for the AMD secondary restorer (SRVGGNetCompact family).

Runs on CPU with deliberately tiny networks, so no GPU and no large checkpoint is
needed; the point is shape/scale/loader behaviour, not image quality.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jasna.restorer import realesrgan_secondary_restorer as mod
from jasna.restorer.realesrgan_secondary_restorer import (
    ARCH_SRVGGNET,
    REALESRGAN_MODEL_CHOICES,
    REALESRGAN_MODEL_FILES,
    RealEsrganSecondaryRestorer,
    detect_upscale_arch,
    load_upscale_model,
    read_upscale_spec,
    resolve_weights_path,
)
from jasna.restorer.srvggnet import (
    SRVGGNetCompact,
    load_srvggnet,
    read_srvggnet_spec,
)


def _tiny_srvgg_checkpoint(
    tmp_path: Path,
    *,
    num_conv: int = 1,
    num_feat: int = 8,
    upscale: int = 4,
    act_type: str = "prelu",
    wrap: str | None = "params",
    name: str = "tiny-srvgg.pth",
) -> Path:
    """A zeroed SRVGGNetCompact with a constant residual."""
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=upscale,
        act_type=act_type,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        # body[-1] is the last convolution (feat -> out*up**2); a 0.5 bias keeps
        # the output non-zero, and every real checkpoint ships the "params" wrapper.
        model.body[-1].bias.fill_(0.5)
        if act_type == "prelu":
            model.body[1].weight.fill_(0.25)
    state: object = {wrap: model.state_dict()} if wrap else model.state_dict()
    path = tmp_path / name
    torch.save(state, path)
    return path


def _restorer(tmp_path: Path, **kwargs) -> RealEsrganSecondaryRestorer:
    params = dict(
        device=torch.device("cpu"),
        scale=2,
        model_path=_tiny_srvgg_checkpoint(tmp_path),
        fp16=False,
    )
    params.update(kwargs)
    return RealEsrganSecondaryRestorer(**params)


def _frames(count: int, size: int = 256) -> torch.Tensor:
    return torch.rand(count, 3, size, size)


def _unwrapped_state(path: Path) -> dict:
    return mod._load_state_dict(path)


class TestCheckpointSpec:
    def test_reads_convs_feature_width_and_scale(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, num_conv=3, num_feat=16)
        spec = read_srvggnet_spec(path)
        assert (spec.num_conv, spec.num_feat, spec.upscale) == (3, 16, 4)
        assert spec.native_scale == 4
        assert spec.act_type == "prelu"
        assert (spec.num_in_ch, spec.num_out_ch) == (3, 3)

    def test_relu_checkpoint_has_no_prelu_parameters(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, act_type="relu")
        assert read_srvggnet_spec(path).act_type == "relu"

    def test_params_wrapper_is_accepted(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, wrap="params")
        model, spec = load_srvggnet(path, device=torch.device("cpu"), fp16=False)
        assert spec.num_conv == 1
        assert model.body[0].weight.shape == (8, 3, 3, 3)

    def test_unwrapped_state_dict_is_accepted(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, wrap=None)
        model, spec = load_srvggnet(path, device=torch.device("cpu"), fp16=False)
        assert spec.num_feat == 8

    def test_fp16_flag_downcasts_the_weights(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path)
        model, _ = load_srvggnet(path, device=torch.device("cpu"), fp16=True)
        assert next(model.parameters()).dtype == torch.float16

    def test_mismatched_checkpoint_is_rejected(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Drop a PReLU slope: the architecture is still readable from the remaining
        # keys, but the strict load then reports a missing parameter.
        state["params"].pop("body.1.weight")
        torch.save(state, path)
        with pytest.raises(ValueError, match="does not match SRVGGNetCompact"):
            load_srvggnet(path, device=torch.device("cpu"), fp16=False)

    def test_forward_upscales_and_stays_finite(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, upscale=4)
        model, _ = load_srvggnet(path, device=torch.device("cpu"), fp16=False)
        out = model(torch.rand(2, 3, 16, 16))
        assert out.shape == (2, 3, 64, 64)
        assert torch.isfinite(out).all()

    def test_non_square_output_channels_are_rejected(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, num_conv=1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        # A 4x net emits 3*16 = 48 channels; 46 is neither a multiple of 3 nor square.
        state["params"]["body.4.weight"] = state["params"]["body.4.weight"][:46]
        state["params"]["body.4.bias"] = state["params"]["body.4.bias"][:46]
        torch.save(state, path)
        with pytest.raises(ValueError, match="not a multiple of 3"):
            read_srvggnet_spec(path)


class TestArchDispatch:
    def test_srvgg_checkpoint_is_detected(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path)
        assert detect_upscale_arch(_unwrapped_state(path)) == ARCH_SRVGGNET
        spec = read_upscale_spec(path)
        assert (spec.arch, spec.num_conv, spec.act_type) == (ARCH_SRVGGNET, 1, "prelu")
        assert "SRVGGNetCompact" in spec.describe()

    def test_unknown_checkpoint_is_rejected(self, tmp_path):
        path = tmp_path / "junk.pth"
        torch.save({"foo": torch.zeros(3)}, path)
        with pytest.raises(ValueError, match="unsupported checkpoint"):
            read_upscale_spec(path)

    def test_load_returns_the_network(self, tmp_path):
        model, spec = load_upscale_model(
            _tiny_srvgg_checkpoint(tmp_path), device=torch.device("cpu"), fp16=False
        )
        assert spec.arch == ARCH_SRVGGNET
        assert isinstance(model, SRVGGNetCompact)


class TestWeightResolution:
    def test_explicit_path_wins(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path)
        assert resolve_weights_path(path) == path.resolve()

    def test_missing_explicit_path_reports_absolute_path_and_hint(self, tmp_path):
        missing = tmp_path / "sub" / "nope.pth"
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_weights_path(str(missing))
        message = str(excinfo.value)
        assert str(missing.resolve()) in message
        assert "--amd-upscale-model-path" in message

    def test_preset_parimg_2x_selects_the_compact_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["parimg-2x"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "parimg-2x") == wanted.resolve()

    def test_preset_lsdir_c3_selects_the_compact_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["lsdir-c3"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "lsdir-c3") == wanted.resolve()

    def test_preset_x4v3_selects_the_general_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["x4v3"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "x4v3") == wanted.resolve()

    def test_default_preset_is_parimg_2x(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        (tmp_path / REALESRGAN_MODEL_FILES["x4v3"][0]).write_bytes(b"")
        wanted = tmp_path / REALESRGAN_MODEL_FILES["parimg-2x"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None).name == REALESRGAN_MODEL_FILES["parimg-2x"][0]

    def test_missing_preset_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="4xLSDIRCompactC3.pth"):
            resolve_weights_path(None, "lsdir-c3")

    def test_invalid_preset_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="Invalid AMD super-res model preset"):
            resolve_weights_path(None, "bogus")

    def test_only_the_three_shipped_presets_are_offered(self):
        assert REALESRGAN_MODEL_CHOICES == ("x4v3", "lsdir-c3", "parimg-2x")
        assert set(REALESRGAN_MODEL_FILES) == {"x4v3", "lsdir-c3", "parimg-2x"}


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
            assert frame.max() > 0  # last-conv bias 0.5 -> ~128

    def test_keep_window_is_honoured(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        assert len(r.restore(_frames(9), keep_start=2, keep_end=7)) == 5

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
        # scale=2 with a 4x network must still return 512x512, not 1024x1024.
        r = _restorer(tmp_path, scale=2)
        assert r.spec.native_scale == 4
        assert r.restore(_frames(1), keep_start=0, keep_end=1)[0].shape == (3, 512, 512)

    def test_2x_network_upsampled_to_4x_request(self, tmp_path):
        path = _tiny_srvgg_checkpoint(tmp_path, upscale=2, name="u2.pth")
        r = _restorer(tmp_path, scale=4, model_path=path)
        assert r.spec.native_scale == 2
        assert r.restore(_frames(1), keep_start=0, keep_end=1)[0].shape == (3, 1024, 1024)


class TestClose:
    def test_close_is_idempotent_and_drops_the_network(self, tmp_path):
        r = _restorer(tmp_path, scale=2)
        r.restore(_frames(2), keep_start=0, keep_end=2)
        r.close()
        r.close()
        assert r.model is None

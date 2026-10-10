"""Tests for the Real-ESRGAN secondary restorer.

Runs on CPU with deliberately tiny networks, so no GPU and no large checkpoint is
needed; the point is shape/scale/loader behaviour, not image quality.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from jasna.restorer import realesrgan_secondary_restorer as mod
from jasna.restorer.realesrgan_secondary_restorer import (
    ARCH_RRDBNET,
    ARCH_SRVGGNET,
    REALESRGAN_MODEL_CHOICES,
    REALESRGAN_MODEL_FILES,
    REALESRGAN_WEIGHT_CANDIDATES,
    RealEsrganSecondaryRestorer,
    detect_upscale_arch,
    find_default_weights,
    load_upscale_model,
    read_upscale_spec,
    resolve_weights_path,
)
from jasna.restorer.rrdbnet import RRDBNet, load_rrdbnet, read_checkpoint_spec
from jasna.restorer.srvggnet import (
    SRVGGNetCompact,
    load_srvggnet,
    read_srvggnet_spec,
)


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


_KAIR_PREFIXES = (
    ("conv_body.", "trunk_conv."),
    ("conv_up1.", "upconv1."),
    ("conv_up2.", "upconv2."),
    ("conv_hr.", "HRconv."),
)


def _tiny_kair_checkpoint(
    tmp_path: Path,
    *,
    num_block: int = 1,
    num_feat: int = 8,
    num_grow_ch: int = 4,
    name: str = "tiny-kair.pth",
) -> Path:
    """A zeroed RRDBNet saved under KAIR / BSRGAN key names, as ``BSRNet.pth`` is."""
    model = RRDBNet(num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.conv_last.bias.fill_(0.5)
    state: dict = {}
    for key, value in model.state_dict().items():
        renamed = re.sub(r"^body\.(\d+)\.rdb(\d+)\.", r"RRDB_trunk.\1.RDB\2.", key)
        for old, new in _KAIR_PREFIXES:
            if renamed.startswith(old):
                renamed = new + renamed[len(old):]
                break
        state[renamed] = value
    path = tmp_path / name
    torch.save(state, path)  # raw state dict: BSRNet ships no params_ema wrapper
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
    """A zeroed SRVGGNetCompact with a constant residual, like ``_tiny_checkpoint``."""
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
        # body[-1] is the last convolution (feat -> out*up**2); a 0.5 bias keeps the
        # output non-zero, and every real checkpoint ships the "params" wrapper.
        model.body[-1].bias.fill_(0.5)
        if act_type == "prelu":
            model.body[1].weight.fill_(0.25)
    state: object = {wrap: model.state_dict()} if wrap else model.state_dict()
    path = tmp_path / name
    torch.save(state, path)
    return path


def _srvgg_restorer(tmp_path: Path, **kwargs) -> RealEsrganSecondaryRestorer:
    params = dict(
        device=torch.device("cpu"),
        scale=2,
        model_path=_tiny_srvgg_checkpoint(tmp_path),
        fp16=False,
    )
    params.update(kwargs)
    return RealEsrganSecondaryRestorer(**params)


def _unwrapped_state(path: Path) -> dict:
    return mod._load_state_dict(path)


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

    def test_preset_anime_6b_selects_the_six_block_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["anime-6b"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "anime-6b") == wanted.resolve()

    def test_preset_x4plus_ignores_the_anime_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        (tmp_path / REALESRGAN_MODEL_FILES["anime-6b"][0]).write_bytes(b"")
        wanted = tmp_path / REALESRGAN_MODEL_FILES["x4plus"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "x4plus") == wanted.resolve()

    def test_missing_preset_checkpoint_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="realesrgan_x4plus_anime_6B.pth"):
            resolve_weights_path(None, "anime-6b")

    def test_invalid_preset_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(ValueError, match="Invalid AMD super-res model preset"):
            resolve_weights_path(None, "bogus")


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


class TestSrvggSpec:
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
        # drop a PReLU slope: the architecture is still readable from the remaining
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
        # a 4x net emits 3*16 = 48 channels; 46 is neither a multiple of 3 nor square
        state["params"]["body.4.weight"] = state["params"]["body.4.weight"][:46]
        state["params"]["body.4.bias"] = state["params"]["body.4.bias"][:46]
        torch.save(state, path)
        with pytest.raises(ValueError, match="not a multiple of 3"):
            read_srvggnet_spec(path)


class TestArchDispatch:
    def test_rrdbnet_checkpoint_is_detected(self, tmp_path):
        path = _tiny_checkpoint(tmp_path)
        assert detect_upscale_arch(_unwrapped_state(path)) == ARCH_RRDBNET
        spec = read_upscale_spec(path)
        assert (spec.arch, spec.num_block) == (ARCH_RRDBNET, 1)
        assert "RRDBNet" in spec.describe()

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

    def test_load_returns_the_matching_family(self, tmp_path):
        model, spec = load_upscale_model(
            _tiny_checkpoint(tmp_path), device=torch.device("cpu"), fp16=False
        )
        assert spec.arch == ARCH_RRDBNET
        model2, spec2 = load_upscale_model(
            _tiny_srvgg_checkpoint(tmp_path), device=torch.device("cpu"), fp16=False
        )
        assert spec2.arch == ARCH_SRVGGNET
        assert type(model) is not type(model2)


class TestKairCheckpoint:
    """KAIR/BSRGAN uploads the same RRDBNet under older key names (``BSRNet.pth``)."""

    def test_kair_keys_are_detected_and_read(self, tmp_path):
        path = _tiny_kair_checkpoint(tmp_path, num_block=3, num_feat=16, num_grow_ch=8)
        assert detect_upscale_arch(_unwrapped_state(path)) == ARCH_RRDBNET
        spec = read_checkpoint_spec(path)
        assert (spec.num_block, spec.num_feat, spec.num_grow_ch) == (3, 16, 8)
        assert spec.native_scale == 4

    def test_kair_checkpoint_loads_and_forwards(self, tmp_path):
        path = _tiny_kair_checkpoint(tmp_path)
        model, spec = load_rrdbnet(path, device=torch.device("cpu"), fp16=False)
        assert spec.num_block == 1
        out = model(torch.rand(1, 3, 16, 16))
        assert out.shape == (1, 3, 64, 64)

    def test_kair_restorer_round_trip(self, tmp_path):
        r = _restorer(tmp_path, scale=2, model_path=_tiny_kair_checkpoint(tmp_path))
        out = r.restore(_frames(2), keep_start=0, keep_end=2)
        assert len(out) == 2
        for frame in out:
            assert frame.shape == (3, 512, 512)
            assert frame.dtype == torch.uint8


class TestPresets:
    def test_x4v3_leads_the_search_and_the_choices(self):
        assert REALESRGAN_WEIGHT_CANDIDATES[0] == "realesr-general-x4v3.pth"
        assert "realesr-general-wdn-x4v3.pth" in REALESRGAN_WEIGHT_CANDIDATES
        assert "4xLSDIRCompactC3.pth" in REALESRGAN_WEIGHT_CANDIDATES
        assert "4xLSDIRCompactv2.pth" in REALESRGAN_WEIGHT_CANDIDATES
        assert "2xHFA2kCompact.pth" in REALESRGAN_WEIGHT_CANDIDATES
        assert "BSRNet.pth" in REALESRGAN_WEIGHT_CANDIDATES
        assert REALESRGAN_MODEL_CHOICES == (
            "auto",
            "x4v3",
            "wdn-x4v3",
            "lsdir-c3",
            "lsdir-v2",
            "hfa2k-2x",
            "x4plus",
            "anime-6b",
            "bsrnet",
        )

    def test_preset_lsdir_c3_selects_the_compact_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["lsdir-c3"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "lsdir-c3") == wanted.resolve()

    def test_preset_bsrnet_selects_the_kair_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["bsrnet"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "bsrnet") == wanted.resolve()

    def test_preset_lsdir_v2_selects_the_compact_v2_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["lsdir-v2"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "lsdir-v2") == wanted.resolve()

    def test_preset_hfa2k_2x_selects_the_compact_2x_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["hfa2k-2x"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "hfa2k-2x") == wanted.resolve()

    def test_missing_lsdir_v2_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="4xLSDIRCompactv2.pth"):
            resolve_weights_path(None, "lsdir-v2")

    def test_missing_bsrnet_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="BSRNet.pth"):
            resolve_weights_path(None, "bsrnet")

    def test_preset_x4v3_selects_the_general_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        wanted = tmp_path / REALESRGAN_MODEL_FILES["x4v3"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "x4v3") == wanted.resolve()

    def test_preset_wdn_x4v3_is_distinct_from_plain(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        (tmp_path / REALESRGAN_MODEL_FILES["x4v3"][0]).write_bytes(b"")
        wanted = tmp_path / REALESRGAN_MODEL_FILES["wdn-x4v3"][0]
        wanted.write_bytes(b"")
        assert resolve_weights_path(None, "wdn-x4v3") == wanted.resolve()

    def test_auto_prefers_x4v3_over_the_anime_net(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        (tmp_path / REALESRGAN_MODEL_FILES["anime-6b"][0]).write_bytes(b"")
        (tmp_path / REALESRGAN_MODEL_FILES["x4v3"][0]).write_bytes(b"")
        assert resolve_weights_path(None, "auto").name == REALESRGAN_MODEL_FILES["x4v3"][0]

    def test_missing_x4v3_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="realesr-general-x4v3.pth"):
            resolve_weights_path(None, "x4v3")


class TestSrvggRestorer:
    def test_contract_matches_the_rrdbnet_strategy(self, tmp_path):
        r = _srvgg_restorer(tmp_path, scale=4)
        assert r.name == "amd-upscale"
        assert r.prefers_cpu_input is False
        assert r.spec.arch == ARCH_SRVGGNET
        assert (r.input_size, r.output_size) == (256, 1024)

    def test_round_trip_returns_chw_uint8_at_the_requested_size(self, tmp_path):
        r = _srvgg_restorer(tmp_path, scale=2)
        out = r.restore(_frames(3), keep_start=0, keep_end=3)
        assert len(out) == 3
        for frame in out:
            assert frame.shape == (3, 512, 512)
            assert frame.dtype == torch.uint8
            assert frame.max() > 0

    def test_batching_and_window_are_honoured(self, tmp_path):
        r = _srvgg_restorer(tmp_path, scale=2, batch_size=2)
        out = r.restore(_frames(5), keep_start=1, keep_end=5)
        assert len(out) == 4

    def test_4x_network_downsampled_to_2x_request(self, tmp_path):
        r = _srvgg_restorer(tmp_path, scale=2)
        assert r.spec.native_scale == 4
        assert r.restore(_frames(1), keep_start=0, keep_end=1)[0].shape == (3, 512, 512)

    def test_close_is_idempotent_and_drops_the_network(self, tmp_path):
        r = _srvgg_restorer(tmp_path, scale=2)
        r.restore(_frames(2), keep_start=0, keep_end=2)
        r.close()
        r.close()
        assert r.model is None

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

import jasna.restorer.sd15_inpaint_restorer as sd15
from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH
from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer, _read_checkpoint, use_plaintext_sd15


def _write_bundle(model_dir: Path, *, with_plaintext: bool, with_enc: bool) -> None:
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "unet" / "config.json").write_text("{}")
    if with_plaintext:
        (model_dir / SD15_CKPT_PATH.name).write_bytes(b"x")
    if with_enc:
        (model_dir / SD15_CKPT_ENC_PATH.name).write_bytes(b"x")


class TestModelId:
    def test_model_id(self):
        assert Sd15InpaintRestorer.MODEL_ID == "sd-15-jav"


class TestCheckpointPathSelection:
    def test_plaintext_loaded_directly_when_present(self, tmp_path: Path):
        _write_bundle(tmp_path, with_plaintext=True, with_enc=True)
        with (
            patch.object(sd15, "is_frozen", return_value=False),
            patch.object(sd15.torch, "load", return_value={"state_dict": {}}) as mock_load,
            patch("jasna.protection.protected_model.decrypt_model_to_path") as mock_decrypt,
        ):
            assert use_plaintext_sd15(tmp_path) is True
            _read_checkpoint(tmp_path)
            mock_decrypt.assert_not_called()
            assert str(tmp_path / SD15_CKPT_PATH.name) == mock_load.call_args.args[0]

    def test_encrypted_path_when_no_plaintext(self, tmp_path: Path):
        _write_bundle(tmp_path, with_plaintext=False, with_enc=True)
        with (
            patch.object(sd15, "is_frozen", return_value=False),
            patch.object(sd15.torch, "load", return_value={"state_dict": {}}) as mock_load,
            patch("jasna.protection.protected_model.decrypt_model_to_path") as mock_decrypt,
        ):
            assert use_plaintext_sd15(tmp_path) is False
            _read_checkpoint(tmp_path)
            mock_decrypt.assert_called_once()
            # decrypt target (temp file) is what torch.load reads, not the .enc itself.
            assert mock_load.call_count == 1

    def test_frozen_ignores_plaintext(self, tmp_path: Path):
        _write_bundle(tmp_path, with_plaintext=True, with_enc=True)
        with patch.object(sd15, "is_frozen", return_value=True):
            assert use_plaintext_sd15(tmp_path) is False


class _FakeLatentDist:
    def __init__(self, latents: torch.Tensor):
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class _FakeVae:
    def encode(self, x: torch.Tensor):
        b = x.shape[0]
        return MagicMock(latent_dist=_FakeLatentDist(torch.zeros(b, 4, 64, 64)))

    def decode(self, z: torch.Tensor):
        img = F.interpolate(z[:, :3], size=(512, 512), mode="nearest")
        return MagicMock(sample=img)


class _FakeUnet:
    def enable_freeu(self, **_kw):
        pass

    def disable_freeu(self):
        pass

    def __call__(self, unet_input, t, ehs):
        latents = unet_input[:, :4]
        return MagicMock(sample=torch.zeros_like(latents))


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.zeros(0, dtype=torch.long)

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.arange(steps - 1, -1, -1, dtype=torch.long)

    def add_noise(self, init, noise, t):
        return init + noise

    def step(self, noise_pred, t, latents):
        return MagicMock(prev_sample=latents)


def _make_restorer() -> Sd15InpaintRestorer:
    r = Sd15InpaintRestorer.__new__(Sd15InpaintRestorer)
    r.device = torch.device("cpu")
    r.fp16 = False
    r._dtype = torch.float32
    r.unet = _FakeUnet()
    r.vae = _FakeVae()
    r.scheduler = _FakeScheduler()
    r.null_embedding = torch.zeros(1, 77, 768)
    return r


class TestRestoreCrop:
    def test_output_shape_and_dtype(self):
        r = _make_restorer()
        mosaic = torch.rand(1, 3, 512, 512)
        mask = torch.zeros(1, 1, 512, 512)
        out = r.restore_crop(mosaic, mask, steps=10, strength=0.6, seed=1, freeu=None)
        assert out.shape == (3, 512, 512)
        assert out.dtype == torch.uint8

    def test_seed_determinism(self):
        r = _make_restorer()
        mosaic = torch.rand(1, 3, 512, 512)
        mask = torch.zeros(1, 1, 512, 512)
        a = r.restore_crop(mosaic, mask, steps=10, strength=0.6, seed=7, freeu=None)
        b = r.restore_crop(mosaic, mask, steps=10, strength=0.6, seed=7, freeu=None)
        c = r.restore_crop(mosaic, mask, steps=10, strength=0.6, seed=8, freeu=None)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_freeu_toggle_invoked(self):
        r = _make_restorer()
        r.unet = MagicMock(side_effect=None)
        r.unet.side_effect = lambda unet_input, t, ehs: MagicMock(sample=torch.zeros_like(unet_input[:, :4]))
        mosaic = torch.rand(1, 3, 512, 512)
        mask = torch.zeros(1, 1, 512, 512)
        r.restore_crop(mosaic, mask, steps=5, strength=0.6, seed=1, freeu={"s1": 0.9, "s2": 0.2, "b1": 1.2, "b2": 1.4})
        r.unet.enable_freeu.assert_called_once()
        r.restore_crop(mosaic, mask, steps=5, strength=0.6, seed=1, freeu=None)
        r.unet.disable_freeu.assert_called_once()

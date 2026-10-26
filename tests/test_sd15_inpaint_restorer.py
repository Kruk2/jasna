from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

import jasna.restorer.sd15_inpaint_restorer as sd15
from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH
from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer, _read_checkpoint, use_plaintext_sd15

FREEU = {"s1": 0.9, "s2": 0.2, "b1": 1.2, "b2": 1.4}


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    """An SD 1.5 inpaint model directory laid out like the real one, with a tiny random UNet and VAE."""
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=64, in_channels=9, out_channels=4, layers_per_block=1, block_out_channels=(8, 8),
        down_block_types=("DownBlock2D",) * 2, up_block_types=("UpBlock2D",) * 2,
        cross_attention_dim=8, norm_num_groups=4, attention_head_dim=2,
    )
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text(json.dumps(dict(unet.config)))
    torch.save({"state_dict": {f"unet.{k}": v for k, v in unet.state_dict().items()}}, tmp_path / SD15_CKPT_PATH.name)
    AutoencoderKL(
        block_out_channels=(4,) * 4, norm_num_groups=4, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
    ).save_pretrained(tmp_path / "vae")
    DDPMScheduler().save_pretrained(tmp_path / "scheduler")
    torch.save(torch.zeros(1, 77, 8), tmp_path / "null_embedding.pt")
    return tmp_path


@pytest.mark.parametrize("plaintext, frozen, expected", [
    (True, False, True), (True, True, False), (False, False, False)])
def test_source_runs_load_a_plaintext_checkpoint_and_releases_never_do(tmp_path, monkeypatch, plaintext, frozen, expected):
    monkeypatch.setattr(sd15, "is_frozen", lambda: frozen)
    if plaintext:
        (tmp_path / SD15_CKPT_PATH.name).write_bytes(b"x")
    assert use_plaintext_sd15(tmp_path) is expected


@pytest.mark.parametrize("plaintext", [True, False])
def test_checkpoint_reads_the_plaintext_file_else_the_packaged_one(tmp_path, monkeypatch, plaintext):
    protected_model = pytest.importorskip("jasna.protection.protected_model")
    checkpoint = {"state_dict": {"unet.weight": torch.arange(3.0)}}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    if plaintext:
        (tmp_path / SD15_CKPT_PATH.name).write_bytes(buffer.getvalue())
        monkeypatch.setattr(protected_model, "decrypt_model_to_buffer", lambda *args: pytest.fail("decrypted"))
    else:
        (tmp_path / SD15_CKPT_ENC_PATH.name).write_bytes(b"sealed")
        monkeypatch.setattr(protected_model, "decrypt_model_to_buffer", lambda *args: io.BytesIO(buffer.getvalue()))
    monkeypatch.setattr(sd15, "is_frozen", lambda: False)
    monkeypatch.setattr("jasna.license_api.require_protection", lambda: None)
    assert torch.equal(_read_checkpoint(tmp_path)["state_dict"]["unet.weight"], torch.arange(3.0))


def test_restore_crop_is_a_seeded_uint8_crop(tiny_model_dir):
    restorer = Sd15InpaintRestorer(tiny_model_dir, torch.device("cpu"), fp16=False)
    mosaic = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(0))
    mask = torch.zeros(1, 1, 512, 512)

    def restore(seed, freeu=None):
        return restorer.restore_crop(mosaic, mask, steps=4, strength=0.6, seed=seed, freeu=freeu)

    first = restore(7)
    assert first.shape == (3, 512, 512) and first.dtype == torch.uint8
    assert torch.equal(restore(7), first) and not torch.equal(restore(8), first)
    assert not torch.equal(restore(7, FREEU), first)
    assert torch.equal(restore(7), first)

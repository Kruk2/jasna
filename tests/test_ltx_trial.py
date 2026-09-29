import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import session_config
from jasna.ltx import transformer as T
from jasna.ltx.model_files import LTX_MODELS, LtxModelFiles, bundle_names, open_tensors
from jasna.ltx.sampler import sampler_settings
from jasna.models.ltx_vae.loader import load_video_decoder, load_video_encoder

DEV_FILES = Path(__file__).resolve().parents[1] / "model_weights" / "ltx-restore"
CASES = [(model, fast) for model in LTX_MODELS for fast in (False, True)]


def _header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    with path.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    header.pop("__metadata__", None)
    return {name: (entry["dtype"], tuple(entry["shape"])) for name, entry in header.items()}


def _placeholder_specs(source) -> dict[str, tuple[str, tuple[int, ...]]]:
    names = {torch.bfloat16: "BF16", torch.float32: "F32", torch.int8: "I8", torch.uint8: "U8",
             torch.float8_e4m3fn: "F8_E4M3"}
    with open_tensors(source) as handle:
        return {name: (names[t.dtype], tuple(t.shape)) for name in handle.keys() for t in [handle.get_tensor(name)]}


@pytest.mark.parametrize("model,fast", CASES)
def test_placeholders_match_the_real_files(model, fast):
    names = bundle_names(model, fast=fast)
    if not all((DEV_FILES / name).is_file() for name in names):
        pytest.skip("raw LTX model files not present")
    files = LtxModelFiles.placeholder(model, fast=fast)
    for source, name in zip((files.transformer, files.vae, files.tuned_decoder), names):
        assert _placeholder_specs(source) == _header(DEV_FILES / name)


@pytest.mark.parametrize("model,steps,stg", [("distilled", 8, 0.0), ("undistilled", 15, 1.0)])
def test_placeholder_sampler_has_the_model_step_count(model, steps, stg):
    with open_tensors(LtxModelFiles.placeholder(model, fast=False).transformer) as handle:
        sigmas, stg_scale = sampler_settings(handle.metadata())
    assert len(sigmas) == steps + 1 and stg_scale == stg
    assert sigmas[0] == 1 and sigmas[-1] == 0 and bool((sigmas[1:] < sigmas[:-1]).all())


def test_placeholders_are_deterministic_and_finite():
    source = LtxModelFiles.placeholder("distilled", fast=True).transformer
    with open_tensors(source) as first, open_tensors(source) as second:
        for name in ("blocks.3.attn1.to_q.weight", "blocks.3.attn1.to_q.weight_scale", "blocks.3.attn1.to_q.weight_scale_2",
                     "blocks.3.attn1.to_q.bias", "prompt_context"):
            tensor = first.get_tensor(name)
            assert torch.equal(tensor.reshape(-1).view(torch.uint8), second.get_tensor(name).reshape(-1).view(torch.uint8))
            if tensor.dtype != torch.uint8:
                assert torch.isfinite(tensor.float()).all()
        assert (first.get_tensor("blocks.0.ff.net.2.weight_scale").float() > 0).all()


def test_placeholder_vae_loads_into_the_encoder_and_decoder():
    files = LtxModelFiles.placeholder("distilled", fast=False)
    load_video_encoder(files.vae, torch.device("cpu"))
    load_video_decoder(files.vae, files.tuned_decoder, torch.device("cpu"))


def test_trial_session_uses_placeholders_without_files_or_license():
    from jasna.session_factory import _ltx_model_files

    config = session_config(restoration_model_name="ltx", ltx_trial=True, ltx_model="undistilled")
    with (
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.engine_compiler.ensure_engines_compiled"),
        patch("jasna.ltx.model_files.LtxModelFiles.from_dir", side_effect=AssertionError("reads model files")),
    ):
        files = _ltx_model_files(config, MagicMock(), log_callback=None)
    assert files == LtxModelFiles.placeholder("undistilled", fast=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fast", [False, True])
def test_placeholder_block_runs_on_the_gpu(fast):
    device = torch.device("cuda")
    if fast and torch.cuda.get_device_capability(device)[0] < 10:
        pytest.skip("NVFP4 needs Blackwell")
    with open_tensors(LtxModelFiles.placeholder("distilled", fast=fast).transformer) as handle:
        keys = handle.keys()
        top = {k: handle.get_tensor(k).to(device) for k in keys if not k.startswith("blocks.")}
        block = {k.split(".", 2)[2]: handle.get_tensor(k).to(device) for k in keys if k.startswith("blocks.0.")}
    cond = T.step_conditioning(top, 0.5, device)
    rope = T.build_rope(2, 4, 4, dim=4096, heads=T.HEADS, device=device)
    tokens = torch.randn(1, 64, 128, device=device, dtype=torch.bfloat16)
    x = T.block_forward(block, T.embed_tokens(top, tokens, 16), cond, top["prompt_context"], rope)
    assert torch.isfinite(T.velocity(top, x, cond).float()).all()

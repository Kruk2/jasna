"""LTX super resolution called from compose's own stream, with the real nvvfx (in a subprocess,
since test_rtx_superres_restorer installs a fake nvvfx module)."""

import importlib.util
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("nvvfx") is None, reason="needs CUDA and nvvfx"
)

SCRIPT = """
import torch
from jasna.ltx.super_res import LtxSuperRes

device = torch.device("cuda")
canvas = (torch.rand(3, 512, 512, generator=torch.Generator().manual_seed(0)) * 255).to(torch.uint8).to(device)
expected = LtxSuperRes(device=device, quality="high", denoise="none", deblur="none").upscale(canvas)
with torch.cuda.stream(torch.cuda.Stream(device, priority=-1)):
    upscaled = LtxSuperRes(device=device, quality="high", denoise="none", deblur="none").upscale(canvas)
torch.cuda.synchronize()
print(tuple(upscaled.shape), torch.equal(upscaled, expected))
"""


def test_upscale_from_a_side_stream_matches_the_default_stream():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True)
    assert result.stdout.split("\n")[-2] == "(3, 1024, 1024) True"

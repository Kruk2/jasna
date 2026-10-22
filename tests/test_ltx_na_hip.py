"""The ahead-of-time AMD neighborhood-attention kernels (``na_hip``) and their routing; no GPU needed."""

from pathlib import Path

import pytest
import torch

from jasna.models.ltx_vae import na_hip
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.na_hip_launch import KERNEL_LAUNCH
from factories import offload_bundle_targets

MEDIA = Path(__file__).resolve().parent.parent / "jasna" / "media"


@pytest.mark.parametrize("window", na_hip.WINDOWS)
def test_every_window_ships_a_code_object_for_every_architecture(window):
    module = na_hip.module_name(window)
    targets = offload_bundle_targets(MEDIA / f"{module}.hipfb")

    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == set(KERNEL_LAUNCH)
    assert all(module in kernels for kernels in KERNEL_LAUNCH.values())


def test_without_triton_neighborhood_attention_runs_the_ahead_of_time_kernel(monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "has_triton_package", lambda: False)

    T.enable_fused_attention(torch.device("cpu"))

    assert T._fused_na3d is na_hip.na3d


def test_rejects_inputs_the_kernel_was_not_built_for():
    q = torch.zeros(1, 4, 4, 4, 1, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head dim 64"):
        na_hip.na3d(q, q, q, [3, 3, 3], 0)
    with pytest.raises(ValueError, match="head dim 64"):
        na_hip.na3d(q.half(), q.half(), q.half(), [3, 3, 3], 0)


def test_a_window_without_a_built_kernel_fails_loudly():
    with pytest.raises(RuntimeError, match="no ahead-of-time neighborhood-attention kernel"):
        na_hip._kernel("gfx1201", (5, 5, 5))

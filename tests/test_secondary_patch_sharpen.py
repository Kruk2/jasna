from types import SimpleNamespace

import pytest

from jasna.session_factory import _secondary_patch_sharpen


def _config(**overrides):
    base = dict(
        secondary_restoration="amd-upscale",
        amd_upscale_engine="real-esr",
        amd_upscale_model="x4v3",
        amd_upscale_sharpness=-1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_no_secondary_has_nothing_to_sharpen() -> None:
    assert _secondary_patch_sharpen(_config(secondary_restoration="none", amd_upscale_sharpness=0.5)) == 0.0


def test_amf_sr_sharpens_inside_the_filter_not_the_patch() -> None:
    # For amf-sr the sharpness belongs to the sr_amf filter, so the patch passes
    # through untouched even with a positive value.
    assert _secondary_patch_sharpen(_config(amd_upscale_engine="amf-sr", amd_upscale_model="auto", amd_upscale_sharpness=0.5)) == 0.0


def test_engine_is_resolved_from_the_model_when_the_engine_is_stale() -> None:
    # amd_upscale_engine_for makes the model decide: an amf-sr engine value with a
    # network weight resolves to the network engine, so the patch is sharpened.
    assert _secondary_patch_sharpen(_config(amd_upscale_engine="amf-sr", amd_upscale_model="x4v3", amd_upscale_sharpness=0.4)) == pytest.approx(0.4)


@pytest.mark.parametrize("model", ["x4v3", "wdn-x4v3", "lsdir-v2", "hfa2k-2x"])
def test_real_esr_network_uses_the_patch_sharpen(model: str) -> None:
    assert _secondary_patch_sharpen(_config(amd_upscale_model=model, amd_upscale_sharpness=0.3)) == pytest.approx(0.3)


def test_realesrgan_network_uses_the_patch_sharpen() -> None:
    assert _secondary_patch_sharpen(_config(amd_upscale_engine="realesrgan", amd_upscale_model="x4plus", amd_upscale_sharpness=0.25)) == pytest.approx(0.25)


def test_disabled_and_negative_values_are_off() -> None:
    assert _secondary_patch_sharpen(_config(amd_upscale_sharpness=-1.0)) == 0.0
    assert _secondary_patch_sharpen(_config(amd_upscale_sharpness=0.0)) == 0.0


def test_values_above_one_are_clamped() -> None:
    # The AMF range reaches 2, but the CAS weight is only defined up to 1 (and is
    # harsh there), so the patch clamp keeps it in range.
    assert _secondary_patch_sharpen(_config(amd_upscale_sharpness=2.0)) == 1.0

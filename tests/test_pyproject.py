import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("suffix", [".fatbin", ".hipfb"])
def test_every_precompiled_kernel_ships_as_package_data(suffix) -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        package_data = tomllib.load(file)["tool"]["setuptools"]["package-data"]
    images = sorted((ROOT / "jasna" / "media").glob(f"*{suffix}"))

    assert images
    for image in images:
        assert any(fnmatch(image.name, pattern) for pattern in package_data["jasna.media"])


def test_both_vendors_install_the_same_comfy_kitchen_for_ltx() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        extras = tomllib.load(file)["project"]["optional-dependencies"]

    def kitchen(vendor):
        return [pin for pin in extras[vendor] if pin.startswith("comfy-kitchen")]

    assert kitchen("amd") == kitchen("nvidia") == ["comfy-kitchen==0.2.38"]


def test_amd_extra_pins_the_rocm_101_pytorch_builds() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        amd = tomllib.load(file)["project"]["optional-dependencies"]["amd"]

    assert "torch==2.14.0+rocm10.1.0" in amd
    assert "torchvision==0.29.0a0+rocm10.1.0" in amd

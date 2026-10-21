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

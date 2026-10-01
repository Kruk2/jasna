import tomllib
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_cuda_fatbin_ships_as_package_data() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        package_data = tomllib.load(file)["tool"]["setuptools"]["package-data"]
    fatbins = sorted((ROOT / "jasna" / "media").glob("*.fatbin"))

    assert fatbins
    for fatbin in fatbins:
        assert any(fnmatch(fatbin.name, pattern) for pattern in package_data["jasna.media"])

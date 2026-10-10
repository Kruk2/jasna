from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from jasna.license_api import ProtectionError, license_store


def test_public_source_checkout_is_unlicensed_but_usable() -> None:
    assert license_store.load_license() is None
    assert license_store.is_licensed() is False

    with pytest.raises(ProtectionError, match="unavailable in the public source checkout"):
        license_store.set_license("supporter@example.com", "not-a-real-key")


def test_private_protection_exports_pass_through_unchanged(monkeypatch) -> None:
    fake_protection = ModuleType("jasna.protection")

    class PrivateProtectionError(RuntimeError):
        pass

    private_store = object()
    fake_protection.ProtectionError = PrivateProtectionError
    fake_protection.license_store = private_store
    monkeypatch.setitem(sys.modules, "jasna.protection", fake_protection)

    source = Path(__file__).parents[1] / "jasna" / "license_api.py"
    spec = importlib.util.spec_from_file_location("_license_api_private_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ProtectionError is PrivateProtectionError
    assert module.license_store is private_store


def test_broken_private_dependency_is_not_silently_treated_as_public(monkeypatch) -> None:
    fake_protection = ModuleType("jasna.protection")

    def _missing_dependency(_name: str):
        raise ModuleNotFoundError(
            "No module named 'private_runtime_dependency'",
            name="private_runtime_dependency",
        )

    fake_protection.__getattr__ = _missing_dependency
    monkeypatch.setitem(sys.modules, "jasna.protection", fake_protection)

    source = Path(__file__).parents[1] / "jasna" / "license_api.py"
    spec = importlib.util.spec_from_file_location("_license_api_broken_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ModuleNotFoundError, match="private_runtime_dependency"):
        spec.loader.exec_module(module)

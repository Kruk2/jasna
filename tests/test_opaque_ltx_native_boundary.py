"""Actual extension boundary; licensed GPU loading uses the frozen worker.

Native transaction fixtures cover rollback and ownership. These checks require
a rebuilt extension and verify that Python cannot recreate a writable store.
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="module")
def native():
    if sys.platform not in ("linux", "win32"):
        pytest.skip("native owner boundary applies to Linux and Windows")
    from jasna.protection import opaque_ltx

    if opaque_ltx._opaque_ltx is None:
        pytest.skip("build the native opaque extension first")
    return opaque_ltx._native()


def test_native_extension_has_no_python_store_construction_or_destinations(native):
    for name in (
        "Store", "create_store", "destination", "is_destination",
        "destination_size", "slice_destination", "finalize",
    ):
        assert not hasattr(native, name), f"rebuild required: obsolete native {name} export"


@pytest.mark.parametrize("owner", [None, object(), {}])
@pytest.mark.parametrize("operation", ["close", "device_bytes", "activate", "begin_step"])
def test_native_inference_rejects_objects_that_are_not_private_load_owners(native, owner, operation):
    arguments = {
        "close": (owner,),
        "device_bytes": (owner,),
        "activate": (owner, 0),
        "begin_step": (owner, 8192, 1, False),
    }
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        getattr(native, operation)(*arguments[operation])

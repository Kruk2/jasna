"""Restoration package.

The restorer classes pull in torch and the full restoration stack, a multi-second
import. They are exposed lazily so that reaching a lightweight helper in this package
(e.g. ``sd15_download.bundle_present``, a filesystem check the GUI runs at startup)
does not pay that cost.
"""
import importlib

_LAZY_EXPORTS = {
    "BasicvsrppMosaicRestorer": "jasna.restorer.basicvsrpp_mosaic_restorer",
    "DenoiseStep": "jasna.restorer.denoise",
    "DenoiseStrength": "jasna.restorer.denoise",
    "RestorationPipeline": "jasna.restorer.restoration_pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value

"""LTX model bundle discovery and tensor access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jasna._frozen import is_frozen


def open_tensors(path: Path):
    if not is_frozen() and path.suffix != ".enc" and path.is_file():
        from safetensors import safe_open
        return safe_open(str(path), framework="pt", device="cpu")
    from jasna.protection.protected_tensors import open_ltx_tensors
    return open_ltx_tensors(path)


@dataclass(frozen=True)
class LtxModelFiles:
    transformer: Path
    vae: Path
    tuned_decoder: Path

    @classmethod
    def from_dir(cls, directory: Path, *, fast: bool) -> LtxModelFiles:
        """The model bundle; ``fast`` picks the NVFP4 transformer over the INT8 one."""
        paths = _bundle_paths(directory, fast=fast)
        if any(path.suffix == ".enc" for path in paths):
            from jasna.protection.protected_tensors import check_ltx_license
            check_ltx_license()
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"LTX model files missing: {', '.join(missing)}")
        return cls(*paths)


def _bundle_paths(directory: Path, *, fast: bool) -> list[Path]:
    transformer = "transformer-fast" if fast else "transformer"
    paths = [directory / f"{name}.safetensors" for name in (transformer, "vae", "vae-decoder")]
    return [path.with_name(path.name + ".enc") if is_frozen() or not path.is_file() else path for path in paths]


def bundle_present(directory: Path) -> bool:
    """Whether the quality model files are installed; checks no license."""
    return all(path.is_file() for path in _bundle_paths(directory, fast=False))

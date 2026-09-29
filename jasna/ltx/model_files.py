"""LTX model bundle discovery, tensor access and download.

A bundle is one transformer file (the model choice x the precision) plus the shared VAE
and tuned decoder. Every file name is unique, so all variants can sit in one folder.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from jasna._frozen import is_frozen
from jasna.session_config import LtxModelName

logger = logging.getLogger(__name__)

LTX_MODELS: tuple[LtxModelName, ...] = get_args(LtxModelName)

# Where the model files are published; each file is fetched from f"{LTX_DOWNLOAD_URL}/{name}".
LTX_DOWNLOAD_URL: str | None = None

DownloadProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DownloadableFile:
    name: str
    size_bytes: int
    sha256: str | None


_TRANSFORMER_NAMES: dict[tuple[LtxModelName, bool], str] = {
    ("distilled", False): "ltx-restore-alpha1-distill8-int8.safetensors",
    ("distilled", True): "ltx-restore-alpha1-distill8-nvfp4.safetensors",
    ("undistilled", False): "ltx-restore-alpha1-int8.safetensors",
    ("undistilled", True): "ltx-restore-alpha1-nvfp4.safetensors",
}
_SHARED_NAMES = ("vae.safetensors", "vae-decoder.safetensors")

# The published files (``.enc`` in releases): size for the download prompt, and a sha256
# that is filled in with the release that publishes the file.
LTX_DOWNLOADS: dict[str, DownloadableFile] = {
    name: DownloadableFile(name, size, None)
    for name, size in (
        ("ltx-restore-alpha1-distill8-int8.safetensors", 13_368_044_744),
        ("ltx-restore-alpha1-distill8-nvfp4.safetensors", 7_733_292_672),
        ("ltx-restore-alpha1-int8.safetensors", 13_368_044_792),
        ("ltx-restore-alpha1-nvfp4.safetensors", 7_733_292_712),
        ("vae.safetensors", 1_472_223_346),
        ("vae-decoder.safetensors", 38_718_560),
    )
}


def open_tensors(path: Path):
    if not is_frozen() and path.suffix != ".enc" and path.is_file():
        from safetensors import safe_open
        return safe_open(str(path), framework="pt", device="cpu")
    from jasna.protection.protected_tensors import open_ltx_tensors
    return open_ltx_tensors(path)


def bundle_names(model: LtxModelName, *, fast: bool) -> tuple[str, str, str]:
    """(transformer, vae, tuned decoder) file names of one model choice."""
    return (_TRANSFORMER_NAMES[(model, fast)], *_SHARED_NAMES)


def _bundle_paths(directory: Path, model: LtxModelName, *, fast: bool) -> list[Path]:
    paths = [directory / name for name in bundle_names(model, fast=fast)]
    return [path.with_name(path.name + ".enc") if is_frozen() or not path.is_file() else path for path in paths]


@dataclass(frozen=True)
class LtxModelFiles:
    transformer: Path
    vae: Path
    tuned_decoder: Path

    @classmethod
    def from_dir(cls, directory: Path, model: LtxModelName, *, fast: bool) -> LtxModelFiles:
        """The bundle of ``model``; ``fast`` picks its NVFP4 transformer over the INT8 one."""
        paths = _bundle_paths(directory, model, fast=fast)
        if any(path.suffix == ".enc" for path in paths):
            from jasna.protection.protected_tensors import check_ltx_license
            check_ltx_license()
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"LTX model files missing: {', '.join(missing)}")
        return cls(*paths)


def model_installed(directory: Path, model: LtxModelName, *, fast: bool) -> bool:
    """Whether every file of this model choice is on disk; checks no license."""
    return all(path.is_file() for path in _bundle_paths(directory, model, fast=fast))


def bundle_present(directory: Path) -> bool:
    """Whether any LTX model can run from ``directory``; checks no license."""
    return any(model_installed(directory, model, fast=fast) for model in LTX_MODELS for fast in (False, True))


def missing_downloads(directory: Path, model: LtxModelName, *, fast: bool) -> list[DownloadableFile]:
    """The files this model choice still needs, as published (``.enc`` in releases)."""
    missing = []
    for path in _bundle_paths(directory, model, fast=fast):
        if not path.is_file():
            entry = LTX_DOWNLOADS[path.name.removesuffix(".enc")]
            missing.append(DownloadableFile(path.name, entry.size_bytes, entry.sha256))
    return missing


def download_size_text(files: list[DownloadableFile]) -> str:
    return f"{sum(f.size_bytes for f in files) / 1e9:.1f} GB"


def download_files(
    directory: Path, files: list[DownloadableFile], progress_callback: DownloadProgressCallback | None = None
) -> None:
    """Download ``files`` into ``directory``, verifying each sha256 before it replaces
    anything. ``progress_callback(done_bytes, total_bytes)`` covers all files together."""
    if LTX_DOWNLOAD_URL is None:
        raise RuntimeError("The LTX model download location is not published yet; get the model files manually.")
    unverifiable = [f.name for f in files if f.sha256 is None]
    if unverifiable:
        raise RuntimeError(f"No published checksum for {', '.join(unverifiable)}; refusing to download.")
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(f.size_bytes for f in files)
    done = 0
    for file in files:
        target = directory / file.name
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        logger.info("Downloading %s", file.name)
        with urllib.request.urlopen(f"{LTX_DOWNLOAD_URL}/{file.name}") as response, partial.open("wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress_callback is not None:
                    progress_callback(done, total)
        if digest.hexdigest() != file.sha256:
            partial.unlink()
            raise RuntimeError(f"{file.name} failed its checksum; the download was discarded.")
        partial.replace(target)

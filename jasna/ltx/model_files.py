"""LTX model bundle discovery, tensor access and download.

A bundle is one transformer file (the model choice x the precision) plus the shared VAE
and tuned decoder. Every file name is unique, so all variants can sit in one folder.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import certifi

from jasna._frozen import is_frozen
from jasna.session_config import LtxModelName

logger = logging.getLogger(__name__)

LTX_MODELS: tuple[LtxModelName, ...] = get_args(LtxModelName)

# Where the model files are published; each file is fetched from f"{LTX_DOWNLOAD_URL}/{name}",
# and from the same path on LTX_MIRROR_URL (reachable from mainland China) when that fails.
LTX_DOWNLOAD_URL = "https://huggingface.co/Kruk2/ltx-restore/resolve/main"
LTX_MIRROR_URL = "https://hf-mirror.com/Kruk2/ltx-restore/resolve/main"
_NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)

DownloadProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DownloadableFile:
    name: str
    size_bytes: int
    sha256: str


_TRANSFORMER_NAMES: dict[tuple[LtxModelName, bool], str] = {
    ("distilled2", False): "ltx-restore-alpha1-distill2-int8.safetensors",
    ("distilled2", True): "ltx-restore-alpha1-distill2-nvfp4.safetensors",
    ("distilled4", False): "ltx-restore-alpha1-distill4-int8.safetensors",
    ("distilled4", True): "ltx-restore-alpha1-distill4-nvfp4.safetensors",
    ("distilled", False): "ltx-restore-alpha1-distill8-int8.safetensors",
    ("distilled", True): "ltx-restore-alpha1-distill8-nvfp4.safetensors",
    ("undistilled", False): "ltx-restore-alpha1-int8.safetensors",
    ("undistilled", True): "ltx-restore-alpha1-nvfp4.safetensors",
}
_SHARED_NAMES = ("vae.safetensors", "vae-decoder.safetensors")

# The published files (``.enc`` in releases): size for the download prompt and sha256.
LTX_DOWNLOADS: dict[str, DownloadableFile] = {
    name: DownloadableFile(name, size, checksum)
    for name, size, checksum in (
        ("ltx-restore-alpha1-distill2-int8.safetensors", 13_368_048_749, "c23559f9d120df49266fc2dbe60ec384058d51834cf1a2da049e54b28aea51ad"),
        ("ltx-restore-alpha1-distill2-nvfp4.safetensors", 7_733_294_997, "4aa08c146c8cc335e7f90d19aa293cc0dfe4f14520fd85efc257fe809510693b"),
        ("ltx-restore-alpha1-distill4-int8.safetensors", 13_368_048_733, "1c0f1642232813f690d949b05fe90bdc02d3203e0092950de25e9cc961f30965"),
        ("ltx-restore-alpha1-distill4-nvfp4.safetensors", 7_733_294_981, "57098e622d02598467f903049f4d1a613852f832847a01c43da148f506876973"),
        ("ltx-restore-alpha1-distill8-int8.safetensors", 13_368_048_773, "b006342a109c640a46ae91a5cd10e67186d968c63ef2f715c78bedf36936e057"),
        ("ltx-restore-alpha1-distill8-nvfp4.safetensors", 7_733_295_021, "54f2f4f6f7d3f5d60e18aca7bf6137da206a2b66f282e1a9078743c3173e62e2"),
        ("ltx-restore-alpha1-int8.safetensors", 13_368_048_821, "c72161b1afefe4d8cc0592fa57afef6c20466dcf912912f3628a451468780f83"),
        ("ltx-restore-alpha1-nvfp4.safetensors", 7_733_295_061, "8301d5ba17660b577ebe906d343963208f7e5453e589ef2f7470087a3ba2ccb5"),
        ("vae.safetensors", 1_472_223_815, "b9252d5a799748d1932d4eed60a2d77461949a19f6feb57f2238dbffe08f5e25"),
        ("vae-decoder.safetensors", 38_718_609, "c60c3572871203bd364d6bb2390225fa4c347f7ea5c64fe468256c900caeea2d"),
    )
}


@dataclass(frozen=True)
class PlaceholderTensors:
    """The trial's stand-in for one model file: the real tensor names, shapes and dtypes,
    filled with random values. ``part`` is ``int8``/``nvfp4`` (the transformer of
    ``model``), ``vae`` or ``vae-decoder``."""

    part: str
    model: LtxModelName


LTX_TRIAL_NOTICE = "LTX trial mode: output is not restored. Speed matches a real run, except model loading."

# Chosen so a random INT8 / FP4 weight comes out near 1/sqrt(4096), like a trained one.
_PLACEHOLDER_SCALES = {".scales": 2e-4, ".weight_scale_2": 5e-3, ".weight_scale": 1.0}
_TORCH_DTYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64",
    "F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


def _fill_tensor_groups(reader, group_names, groups) -> None:
    for names, destinations in zip(group_names, groups, strict=True):
        if tuple(destinations) != tuple(names):
            raise ValueError("tensor group does not match its declared names")
        for name, destination in destinations.items():
            reader.get_tensor_into(name, destination)


class _PlaintextHandle:
    def __init__(self, path: Path) -> None:
        from safetensors import safe_open
        self._source = safe_open(str(path), framework="pt", device="cpu")

    def __enter__(self):
        self._source.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._source.__exit__(*exc)

    def keys(self):
        return self._source.keys()

    def offset_keys(self):
        return self._source.offset_keys()

    def metadata(self):
        return self._source.metadata()

    def get_tensor(self, name: str):
        return self._source.get_tensor(name)

    def tensor_specs(self):
        import torch
        result = {}
        for name in self.keys():
            tensor = self._source.get_slice(name)
            result[name] = (getattr(torch, _TORCH_DTYPES[tensor.get_dtype()]), tuple(tensor.get_shape()))
        return result

    def get_tensor_into(self, name: str, destination) -> None:
        destination.copy_(self._source.get_tensor(name))

    def read_tensors_into(self, group_names, groups) -> None:
        _fill_tensor_groups(self, group_names, groups)


class PlaceholderHandle:
    """``safe_open``-like reader of placeholder tensors. Every tensor of one dtype is a
    fresh copy of the same seeded random pool, so it costs what reading the file costs in
    RAM, not what generating 13 GB of random numbers would."""

    def __init__(self, source: PlaceholderTensors) -> None:
        from jasna.ltx import trial_shapes as S

        if source.part in S.TRANSFORMER:
            parts = S.TRANSFORMER[source.part]
            self._specs = dict(parts["top"])
            for index in range(S.BLOCKS):
                self._specs.update({f"blocks.{index}.{name}": spec for name, spec in parts["block"].items()})
            steps, stg = S.STEPS[source.model]
            self._metadata = {
                "sigmas": json.dumps([round(1 - step / steps, 6) for step in range(steps + 1)]),
                "stg_scale": "1.0" if stg else "0.0",
            }
        elif source.part == "vae":
            self._specs, self._metadata = dict(S.VAE), {"config": json.dumps(S.VAE_CONFIG)}
        else:
            self._specs, self._metadata = dict(S.VAE_DECODER), {}
        self._pools: dict = {}

    def __enter__(self) -> PlaceholderHandle:
        return self

    def __exit__(self, *exc) -> None:
        self._pools.clear()

    def metadata(self) -> dict[str, str]:
        return self._metadata

    def keys(self) -> list[str]:
        return list(self._specs)

    def offset_keys(self) -> list[str]:
        return list(self._specs)

    def tensor_specs(self):
        import torch
        return {name: (getattr(torch, _TORCH_DTYPES[dtype]), tuple(shape))
                for name, (dtype, shape) in self._specs.items()}

    def get_tensor_into(self, name: str, destination) -> None:
        dtype, shape = self._specs[name]
        scale = next((value for suffix, value in _PLACEHOLDER_SCALES.items() if name.endswith(suffix)), None)
        if scale is not None:
            destination.fill_(scale)
        else:
            destination.copy_(self._pool(dtype)[:math.prod(shape)].view(shape))

    def get_tensor(self, name: str):
        import torch

        dtype, shape = self._specs[name]
        scale = next((value for suffix, value in _PLACEHOLDER_SCALES.items() if name.endswith(suffix)), None)
        if scale is not None:
            return torch.full(shape, scale).to(getattr(torch, _TORCH_DTYPES[dtype]))
        return self._pool(dtype)[: math.prod(shape)].view(shape).clone()

    def _pool(self, dtype: str):
        import torch

        if dtype not in self._pools:
            numel = max(math.prod(shape) for kind, shape in self._specs.values() if kind == dtype)
            generator = torch.Generator().manual_seed(0)
            torch_dtype = getattr(torch, _TORCH_DTYPES[dtype])
            if dtype in ("I8", "U8"):
                low, high = (-127, 128) if dtype == "I8" else (0, 256)
                self._pools[dtype] = torch.randint(low, high, (numel,), generator=generator, dtype=torch_dtype)
            else:
                self._pools[dtype] = (torch.randn(numel, generator=generator) * 0.02).to(torch_dtype)
        return self._pools[dtype]

    def read_tensors_into(self, group_names, groups) -> None:
        _fill_tensor_groups(self, group_names, groups)


def open_tensors(path: Path | PlaceholderTensors):
    if isinstance(path, PlaceholderTensors):
        return PlaceholderHandle(path)
    if not is_frozen() and path.suffix != ".enc" and path.is_file():
        return _PlaintextHandle(path)
    from jasna.license_api import require_protection

    require_protection()
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
    transformer: Path | PlaceholderTensors
    vae: Path | PlaceholderTensors
    tuned_decoder: Path | PlaceholderTensors

    @property
    def transformer_name(self) -> str:
        """The transformer's published file name (it names the model and precision); a
        trial's placeholder gets the name of the file it stands in for."""
        if isinstance(self.transformer, PlaceholderTensors):
            return _TRANSFORMER_NAMES[(self.transformer.model, self.transformer.part == "nvfp4")]
        return self.transformer.name.removesuffix(".enc")

    @classmethod
    def placeholder(cls, model: LtxModelName, *, fast: bool) -> LtxModelFiles:
        """Random weights shaped like ``model``'s files, for the trial: runs every pass at
        full cost with no model files and no license, and restores nothing."""
        return cls(
            PlaceholderTensors("nvfp4" if fast else "int8", model),
            PlaceholderTensors("vae", model),
            PlaceholderTensors("vae-decoder", model),
        )

    @classmethod
    def from_dir(cls, directory: Path, model: LtxModelName, *, fast: bool) -> LtxModelFiles:
        """The bundle of ``model``; ``fast`` picks its NVFP4 transformer over the INT8 one."""
        paths = _bundle_paths(directory, model, fast=fast)
        if any(path.suffix == ".enc" for path in paths):
            from jasna.license_api import require_protection

            require_protection()
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
    directory: Path,
    files: list[DownloadableFile],
    progress_callback: DownloadProgressCallback | None,
    *,
    mirror_first: bool,
) -> None:
    """Download ``files`` into ``directory``, verifying each sha256 before it replaces
    anything. Each file falls back to the other source when the first is unreachable;
    ``mirror_first`` tries the mirror first. ``progress_callback(done_bytes, total_bytes)``
    covers all files together."""
    context = ssl.create_default_context(cafile=certifi.where())
    sources = (LTX_MIRROR_URL, LTX_DOWNLOAD_URL) if mirror_first else (LTX_DOWNLOAD_URL, LTX_MIRROR_URL)
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(f.size_bytes for f in files)
    done = 0
    for file in files:
        target = directory / file.name
        partial = target.with_name(target.name + ".part")
        for source, fallback in zip(sources, (*sources[1:], None)):
            digest = hashlib.sha256()
            received = 0
            logger.info("Downloading %s from %s", file.name, source)
            try:
                with urllib.request.urlopen(f"{source}/{file.name}", timeout=30, context=context) as response, partial.open("wb") as out:
                    while chunk := response.read(1 << 22):
                        out.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if progress_callback is not None:
                            progress_callback(done + received, total)
                break
            except _NETWORK_ERRORS as error:
                if fallback is None:
                    raise
                logger.warning("Downloading %s from %s failed (%s); trying %s", file.name, source, error, fallback)
        done += received
        if digest.hexdigest() != file.sha256:
            partial.unlink()
            raise RuntimeError(f"{file.name} is corrupted (checksum mismatch). The download was deleted.")
        partial.replace(target)

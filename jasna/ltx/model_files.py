"""LTX model bundle discovery, tensor access and download.

A bundle is one transformer file (the model choice x the precision) plus the shared VAE,
its video encoder and its decoder. Every file name is unique, so all variants can sit in one
folder.
"""
from __future__ import annotations

import functools
import hashlib
import http.client
import json
import logging
import math
import os
import ssl
import stat
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import certifi

from jasna._frozen import is_frozen
from jasna.os_utils import commit_file
from jasna.session_config import LtxModelName

logger = logging.getLogger(__name__)

LTX_MODELS: tuple[LtxModelName, ...] = get_args(LtxModelName)

# The Hugging Face branch holding the files of LTX_DOWNLOADS; bump both together.
LTX_MODEL_BRANCH = "0.12.1"
LTX_DOWNLOAD_URL = f"https://huggingface.co/Kruk2/ltx-restore/resolve/{LTX_MODEL_BRANCH}"
LTX_MIRROR_URL = f"https://hf-mirror.com/Kruk2/ltx-restore/resolve/{LTX_MODEL_BRANCH}"
_NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)

DownloadProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DownloadableFile:
    name: str
    size_bytes: int
    sha256: str


_TRANSFORMER_NAMES: dict[tuple[LtxModelName, bool], str] = {
    ("distilled1", False): "ltx-restore-alpha1-distill1-int8.safetensors",
    ("distilled1", True): "ltx-restore-alpha1-distill1-nvfp4.safetensors",
    ("distilled2", False): "ltx-restore-alpha1-distill2-int8.safetensors",
    ("distilled2", True): "ltx-restore-alpha1-distill2-nvfp4.safetensors",
    ("distilled4", False): "ltx-restore-alpha1-distill4-int8.safetensors",
    ("distilled4", True): "ltx-restore-alpha1-distill4-nvfp4.safetensors",
    ("distilled", False): "ltx-restore-alpha1-distill8-int8.safetensors",
    ("distilled", True): "ltx-restore-alpha1-distill8-nvfp4.safetensors",
    ("undistilled", False): "ltx-restore-alpha1-int8.safetensors",
    ("undistilled", True): "ltx-restore-alpha1-nvfp4.safetensors",
}
_VAE_NAME = "vae.safetensors"

# Video encoders: the VAE's own, or one distilled from it with about half / a third of its residual
# blocks (encode -45 % / -63 % on an RTX 5090).
VAE_ENCODERS = {"full": _VAE_NAME, "half": "vae-encoder-half.safetensors", "max": "vae-encoder-max.safetensors"}
# Decoders: the tuned decoder (8 stage-5 blocks), or one distilled from it with 4 / 2 of them
# (decode -41 % / -61 %).
VAE_DECODERS = {"full": "vae-decoder-full.safetensors", "half": "vae-decoder-half.safetensors",
                "max": "vae-decoder-max.safetensors"}
VAE_ENCODER_ENV, VAE_DECODER_ENV = "JASNA_LTX_VAE_ENCODER", "JASNA_LTX_VAE_DECODER"
# (encoder, decoder) per model. In human A/Bs on real clips the full VAE is slightly sharper than the half encoder
# with the max decoder, which saves 14-29 % of a 1-step job but only 8-14 % of a 4-step one: the fast models take
# the distilled VAE, the quality models the full one (benchmarks/2026-10-28_ltx_vae_choices.md).
DEFAULT_VAES: dict[LtxModelName, tuple[str, str]] = {
    "distilled1": ("half", "max"),
    "distilled2": ("half", "max"),
    "distilled4": ("full", "full"),
    "distilled": ("full", "full"),
    "undistilled": ("full", "full"),
}

# The published files (``.enc`` in releases): size for the download prompt and sha256.
LTX_DOWNLOADS: dict[str, DownloadableFile] = {
    name: DownloadableFile(name, size, checksum)
    for name, size, checksum in (
        ("ltx-restore-alpha1-distill1-int8.safetensors", 13_368_048_765, "440c591479d052916d1d23018477b95f4ccc172e0ac58fc2d7eccc0e4c484a3b"),
        ("ltx-restore-alpha1-distill1-nvfp4.safetensors", 7_733_295_013, "cc8ceb39fe642cf00a492fd25e658b59b020a2bd3ed3bcad6cf5a077bc176919"),
        ("ltx-restore-alpha1-distill2-int8.safetensors", 13_368_048_765, "e7ed596b067efcc23f8442e60a8619b69d97544e577979cc81e5b0c28767f0bc"),
        ("ltx-restore-alpha1-distill2-nvfp4.safetensors", 7_733_295_013, "3facccfc19bade4d6ffe73fa56ae9251152a8495ce2415ca7a5face5543efc9d"),
        ("ltx-restore-alpha1-distill4-int8.safetensors", 13_368_048_765, "61f421368e2087c64efde578c88b083c665513f12ffeea687e42de72b64324de"),
        ("ltx-restore-alpha1-distill4-nvfp4.safetensors", 7_733_295_013, "47f7d2a2de8e4a424de3ad1b2f5d47bd75bbd287bacd4b481dbca3bf82f9fc6e"),
        ("ltx-restore-alpha1-distill8-int8.safetensors", 13_368_048_765, "1bc678374fc0eed0b1cd0a66c6989889f18981a4c5f61ee1ebd20f6f3c63b52f"),
        ("ltx-restore-alpha1-distill8-nvfp4.safetensors", 7_733_295_013, "1a12d93d9fd1e4b689d72f9dc3df6d687b7803c9140b66f6d2170841138a64af"),
        ("ltx-restore-alpha1-int8.safetensors", 13_368_048_765, "09d19c35c3c8c65185b57453c5780c28d41c860ef9f126e15448ebfa46da8b94"),
        ("ltx-restore-alpha1-nvfp4.safetensors", 7_733_295_013, "e2745e1cc7b9fe456838711c9d04584bbfc66b6cfb0235ddcc60bbe3f2f32618"),
        ("vae.safetensors", 1_472_223_815, "491d9bc1dcec2c306f9dfba2ea4706f6edb56b2ca3ac76452b7229e9b8d10df0"),
        ("vae-decoder-full.safetensors", 38_718_609, "df72fa3b8ab7df51694f6543a96b133d2ad02b27f72d5d66c74eb85f82608005"),
        ("vae-decoder-half.safetensors", 29_768_681, "48bf7c7f92f9af7e39d00342116e5c7424db21e9a8dd283eeed460e1db580332"),
        ("vae-decoder-max.safetensors", 25_293_713, "9e2d632987e7caf142fbd400e1aa4f77afe80ff1a7c1fae69e784bbad8181781"),
        ("vae-encoder-half.safetensors", 329_975_931, "ecf1585fe65ce9aa643575031fb0e11e2a52b473df02b829b5d30f8964bc0197"),
        ("vae-encoder-max.safetensors", 292_811_931, "580e5447d7bb686a5003488c2ab14c6d6cdedc5f2d19254f5e57d159f6a45393"),
    )
}


# sha256 of the first MiB of each published file: tells a current file from an older one of
# the same size without reading it all.
LTX_DOWNLOAD_HEADS: dict[str, str] = {
    "ltx-restore-alpha1-distill1-int8.safetensors": "d00e26c6379f6d3e3a8fda2f1fd1ec5a1b39b8c7150578f13ad55596235eed3b",
    "ltx-restore-alpha1-distill1-nvfp4.safetensors": "46fdbdd8a24292850323734c3583281c94a25d82b33e9e1ac7ecf7ee7cab2e07",
    "ltx-restore-alpha1-distill2-int8.safetensors": "7ba5ec34869521ef8ae7d7f8350255d6178af2363e32fcfc80bef14cb9858a9b",
    "ltx-restore-alpha1-distill2-nvfp4.safetensors": "123f38b8e2ff654b7f7c2bac23c7b4c55ef5e1c40bcbb6c66136a29e2cd1267c",
    "ltx-restore-alpha1-distill4-int8.safetensors": "9e7b3eb53da98af5df082f41a6cb0a885a1addd1c40a5e8c207d7ddf7121159b",
    "ltx-restore-alpha1-distill4-nvfp4.safetensors": "f36d5cb65c5b0eb41fa3d80af907b03b75421bf84329cc7b22cc9fd8e0835d58",
    "ltx-restore-alpha1-distill8-int8.safetensors": "bf0b6849a9190c87fe4b12e2a43963e250601a32542680e945b9ba98e1bf9ceb",
    "ltx-restore-alpha1-distill8-nvfp4.safetensors": "4a65ad70c43a9a720df8cdbb1a7ac7cba4750282c7587e5590939faa91f050f2",
    "ltx-restore-alpha1-int8.safetensors": "ba6584ba29299f7afdb6e919f6813e80397862d3a4f5acba5cb2ab93724547e8",
    "ltx-restore-alpha1-nvfp4.safetensors": "9b54d531844e6f3221a68c1f397a8ffdb324dc0209d0fbe7136954fcb1cb5278",
    "vae.safetensors": "d932e58705da2d5d1154b150c0e0a282c013a7d77ef0943b6ea909ed8839b299",
    "vae-decoder-full.safetensors": "a63aa936e9c0bd42eb92beb7cb04a4f693e513760474b3d4712cc3f60c872c10",
    "vae-decoder-half.safetensors": "6386f36f6b270a65b5f243f4250bbad4a519484e4c8249505f68d8a16af4aaed",
    "vae-decoder-max.safetensors": "71fe2ae3bced0d4cc2b57ca8bab1995b47615e343af293cc5b236e60816b1082",
    "vae-encoder-half.safetensors": "ecc51d50c263277379482ea3a31fb911894c93bd8c64a4bcf01c438c20db3a83",
    "vae-encoder-max.safetensors": "9a3c99391e70e9ab02a39a5909c0b15d43c9c52eaae3deaffc45e26f3bbbc759",
}
_HEAD_BYTES = 1 << 20


@dataclass(frozen=True)
class PlaceholderTensors:
    """The trial's stand-in for one model file: the real tensor names, shapes and dtypes,
    filled with random values. ``part`` is ``int8``/``nvfp4`` (the transformer of
    ``model``) or a VAE file's stem (``vae``, ``vae-decoder-max``, ...)."""

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
            self._specs, self._metadata = dict(getattr(S, source.part.upper().replace("-", "_"))), {}
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


def _vae_choice(env: str, choices: dict[str, str], default: str) -> str:
    choice = os.environ.get(env, default)
    if choice not in choices:
        raise ValueError(f"{env}={choice!r} is not one of {', '.join(choices)}")
    return choices[choice]


def bundle_names(model: LtxModelName, *, fast: bool) -> tuple[str, ...]:
    """(transformer, vae, decoder[, video encoder]) file names of one model choice; the encoder is
    named only when it is not the VAE's own. ``JASNA_LTX_VAE_ENCODER`` (full / half / max) and
    ``JASNA_LTX_VAE_DECODER`` (full / half / max) override the model's ``DEFAULT_VAES``."""
    default_encoder, default_decoder = DEFAULT_VAES[model]
    encoder = _vae_choice(VAE_ENCODER_ENV, VAE_ENCODERS, default_encoder)
    decoder = _vae_choice(VAE_DECODER_ENV, VAE_DECODERS, default_decoder)
    return (_TRANSFORMER_NAMES[(model, fast)], _VAE_NAME, decoder, *((encoder,) if encoder != _VAE_NAME else ()))


def bundle_paths(directory: Path, model: LtxModelName, *, fast: bool) -> list[Path]:
    paths = [directory / name for name in bundle_names(model, fast=fast)]
    return [path.with_name(path.name + ".enc") if is_frozen() or not path.is_file() else path for path in paths]


def needs_license(directory: Path, model: LtxModelName, *, fast: bool) -> bool:
    """Whether this model choice runs from the protected (``.enc``) files, which need a license."""
    return any(path.suffix == ".enc" for path in bundle_paths(directory, model, fast=fast))


def _current_model_file(path: Path) -> bool:
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    if not stat.S_ISREG(info.st_mode):
        return False
    if path.suffix != ".enc":
        return True
    # Size and the first MiB detect outdated downloads cheaply; native loading authenticates their bytes.
    name = path.name.removesuffix(".enc")
    return info.st_size == LTX_DOWNLOADS[name].size_bytes and (
        _head_digest(path, info.st_size, info.st_mtime_ns) == LTX_DOWNLOAD_HEADS[name]
    )


@functools.lru_cache(maxsize=64)
def _head_digest(path: Path, size: int, mtime_ns: int) -> str:
    """sha256 of the first MiB; ``size`` and ``mtime_ns`` key the cache to this version of the file."""
    with open(path, "rb") as file:
        return hashlib.sha256(file.read(_HEAD_BYTES)).hexdigest()


@dataclass(frozen=True)
class LtxModelFiles:
    transformer: Path | PlaceholderTensors
    vae: Path | PlaceholderTensors
    tuned_decoder: Path | PlaceholderTensors
    encoder: Path | PlaceholderTensors | None = None
    """The video encoder's file; ``None`` is the VAE's own."""

    @property
    def video_encoder(self) -> Path | PlaceholderTensors:
        return self.vae if self.encoder is None else self.encoder

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
            *(PlaceholderTensors(name.removesuffix(".safetensors"), model) for name in bundle_names(model, fast=fast)[1:]),
        )

    @classmethod
    def from_dir(cls, directory: Path, model: LtxModelName, *, fast: bool) -> LtxModelFiles:
        """The bundle of ``model``; ``fast`` picks its NVFP4 transformer over the INT8 one."""
        paths = bundle_paths(directory, model, fast=fast)
        if any(path.suffix == ".enc" for path in paths):
            from jasna.license_api import require_protection

            require_protection()
            from jasna.protection.protected_tensors import check_ltx_license
            check_ltx_license()
        missing = [str(path) for path in paths if not _current_model_file(path)]
        if missing:
            raise FileNotFoundError(f"LTX model files missing or outdated: {', '.join(missing)}")
        return cls(*paths)


def model_installed(directory: Path, model: LtxModelName, *, fast: bool) -> bool:
    """Whether this model's files are present, with current encrypted sizes; checks no license."""
    return all(_current_model_file(path) for path in bundle_paths(directory, model, fast=fast))


def bundle_present(directory: Path) -> bool:
    """Whether any LTX model can run from ``directory``; checks no license."""
    return any(model_installed(directory, model, fast=fast) for model in LTX_MODELS for fast in (False, True))


def missing_downloads(directory: Path, model: LtxModelName, *, fast: bool) -> list[DownloadableFile]:
    """The files this model choice still needs, as published (``.enc`` in releases)."""
    missing = []
    for path in bundle_paths(directory, model, fast=fast):
        if not _current_model_file(path):
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
    ``mirror_first`` tries the mirror first. An interrupted download resumes from its ``.part``
    file. ``progress_callback(done_bytes, total_bytes)`` covers all files together."""
    context = ssl.create_default_context(cafile=certifi.where())
    sources = (LTX_MIRROR_URL, LTX_DOWNLOAD_URL) if mirror_first else (LTX_DOWNLOAD_URL, LTX_MIRROR_URL)
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(f.size_bytes for f in files)
    done = 0
    for file in files:
        target = directory / file.name
        partial = target.with_name(target.name + ".part")
        manual_hint = f"Download it yourself from {sources[0]}/{file.name} and put it in {directory}"
        for source, fallback in zip(sources, (*sources[1:], None)):
            url = f"{source}/{file.name}"
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset >= file.size_bytes:
                offset = 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            logger.info("Downloading %s", url)
            try:
                with urllib.request.urlopen(request, timeout=30, context=context) as response:
                    if response.status != 206:
                        offset = 0
                    digest = hashlib.sha256()
                    if offset:
                        with partial.open("rb") as resumed:
                            digest = hashlib.file_digest(resumed, "sha256")
                    received = offset
                    with partial.open("ab" if offset else "wb") as out:
                        while chunk := response.read(1 << 22):
                            out.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                            if progress_callback is not None:
                                progress_callback(done + received, total)
                break
            except _NETWORK_ERRORS as error:
                if fallback is None:
                    raise RuntimeError(f"Could not download {file.name} ({error}). {manual_hint}.") from error
                logger.warning("Downloading %s failed (%s); trying %s", url, error, fallback)
        done += received
        if digest.hexdigest() != file.sha256:
            partial.unlink()
            raise RuntimeError(f"{file.name} is corrupted (checksum mismatch). The download was deleted. {manual_hint}.")
        commit_file(partial, target)

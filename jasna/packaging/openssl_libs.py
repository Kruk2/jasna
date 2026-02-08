from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _is_openssl_so(basename: str) -> bool:
    b = basename.lower()
    return b.startswith("libcrypto.so") or b.startswith("libssl.so")


def filter_out_openssl_binaries(binaries: list[tuple]) -> list[tuple]:
    filtered: list[tuple] = []
    for item in binaries:
        src = item[0]
        if _is_openssl_so(Path(str(src)).name):
            continue
        filtered.append(item)
    return filtered


def _find_shared_lib(libname: str, search_roots: Iterable[str | Path]) -> Path:
    for root in search_roots:
        p = Path(root)

        if p.is_file():
            if p.name == libname:
                return p
            continue

        candidate = p / libname
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {libname!r} in search_roots={list(map(str, search_roots))!r}")


def find_openssl_so3(search_roots: Iterable[str | Path]) -> tuple[Path, Path]:
    crypto = _find_shared_lib("libcrypto.so.3", search_roots)
    ssl = _find_shared_lib("libssl.so.3", search_roots)
    return crypto, ssl


def pyinstaller_binaries_for_openssl(search_roots: Iterable[str | Path]) -> list[tuple[str, str]]:
    crypto, ssl = find_openssl_so3(search_roots)
    return [(str(crypto), "."), (str(ssl), ".")]


"""MIOpen tuning that ships with Jasna: the LTX VAE convs' exhaustively tuned solvers per AMD architecture
(``jasna/media/miopen/<arch>.HIP.fdb.txt`` and ``<arch>.db.txt``, written by ``scripts/tune_miopen.py``).

``install`` merges them into MIOpen's system databases, so a fresh install neither searches these convs
(about a minute per canvas size on the first window) nor runs their untuned kernels (~15 % slower encode).
MIOpen reads only the file named after the exact GPU, so every known compute-unit count of an architecture
gets its own copy, under both names seen in the field: ``<arch><CUs in hex>`` (``gfx120140`` on Linux,
``gfx110054`` in a Windows RX 7900 XT log) and ``<arch>_<CUs>`` (``gfx1101_27`` on a Windows V710)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

TUNED_DIR = Path(__file__).resolve().parent / "media" / "miopen"
DATABASE_SUFFIXES = (".HIP.fdb.txt", ".db.txt")
"""MIOpen's find database (solver picks) and performance database (tuned kernel parameters)."""

COMPUTE_UNITS: Mapping[str, Sequence[int]] = {
    "gfx1201": (64, 56, 48),  # RX 9070 XT / AI PRO R9700, RX 9070, RX 9070 GRE
    "gfx1200": (32, 28),  # RX 9060 XT, RX 9060
    "gfx1101": (60, 54, 48, 27),  # RX 7800 XT, RX 7700 XT, PRO W7700, Radeon PRO V710 as Azure exposes it
    "gfx1100": (96, 84, 80),  # RX 7900 XTX / PRO W7900, RX 7900 XT, RX 7900 GRE
    "gfx1102": (32, 28),  # RX 7600 XT / 7600, PRO W7600
    "gfx1150": (16, 12),  # Radeon 890M, 880M
    "gfx1151": (40, 32),  # Radeon 8060S, 8050S
}

TUNED_ON: Mapping[str, str] = {"gfx1100": "gfx1101", "gfx1102": "gfx1101", "gfx1150": "gfx1101", "gfx1151": "gfx1101"}
"""Architectures without a GPU to tune on, and the one whose tuning they take: the same RDNA3 compute units
and MIOpen CK conv kernels, so its solver picks and kernel parameters apply (MIOpen searches again for any
that do not). Unmeasured on these GPUs."""


def database_names(architecture: str, compute_units: int, suffix: str) -> tuple[str, str]:
    """MIOpen's system database files for one GPU, e.g. ``gfx120140.HIP.fdb.txt`` and ``gfx1201_64.HIP.fdb.txt``
    for a 64-CU gfx1201."""
    return f"{architecture}{compute_units:x}{suffix}", f"{architecture}_{compute_units}{suffix}"


def read_entries(text: str) -> dict[str, str]:
    """``key=value`` lines of an MIOpen text database, in file order."""
    entries = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            entries[key] = value
    return entries


def merged(text: str, entries: Mapping[str, str]) -> str:
    """``text`` with ``entries`` added, replacing the lines of the same keys."""
    kept = [line for line in text.splitlines() if line.partition("=")[0] not in entries]
    return "".join(f"{line}\n" for line in [*kept, *(f"{key}={value}" for key, value in entries.items())])


def install(system_db: Path, tuned_dir: Path = TUNED_DIR,
            compute_units: Mapping[str, Sequence[int]] = COMPUTE_UNITS,
            tuned_on: Mapping[str, str] = TUNED_ON) -> list[Path]:
    """Merge the tuned entries into ``system_db`` for every known GPU of each tuned architecture (or of the
    architecture it takes its tuning from); returns the files written."""
    system_db.mkdir(parents=True, exist_ok=True)
    written = []
    for architecture, units in compute_units.items():
        for suffix in DATABASE_SUFFIXES:
            tuned = tuned_dir / f"{tuned_on.get(architecture, architecture)}{suffix}"
            if not tuned.is_file():
                continue
            entries = read_entries(tuned.read_text(encoding="utf-8"))
            for count in units:
                for name in database_names(architecture, count, suffix):
                    target = system_db / name
                    text = target.read_text(encoding="utf-8") if target.is_file() else ""
                    target.write_text(merged(text, entries), encoding="utf-8", newline="\n")
                    written.append(target)
    return written

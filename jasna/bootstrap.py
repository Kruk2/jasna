from __future__ import annotations

import os
import sys
from pathlib import Path


def sanitize_sys_path_for_local_dev(package_dir: Path) -> None:
    package_dir = package_dir.resolve()
    repo_root = package_dir.parent.resolve()

    package_dir_norm = os.path.normcase(os.path.abspath(str(package_dir)))
    repo_root_norm = os.path.normcase(os.path.abspath(str(repo_root)))

    updated: list[str] = []
    for entry in sys.path:
        if not entry:
            cwd_norm = os.path.normcase(os.path.abspath(os.getcwd()))
            if cwd_norm == package_dir_norm:
                updated.append(str(repo_root))
            else:
                updated.append(entry)
            continue

        entry_norm = os.path.normcase(os.path.abspath(entry))
        if entry_norm == package_dir_norm:
            updated.append(str(repo_root))
        else:
            updated.append(entry)

    deduped: list[str] = []
    seen: set[str] = set()
    for entry in updated:
        if not entry:
            deduped.append(entry)
            continue

        entry_norm = os.path.normcase(os.path.abspath(entry))
        if entry_norm in seen:
            continue
        seen.add(entry_norm)
        deduped.append(entry)

    if repo_root_norm not in seen:
        deduped.insert(0, str(repo_root))

    sys.path[:] = deduped

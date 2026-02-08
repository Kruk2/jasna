from __future__ import annotations

import os
import sys
from pathlib import Path

from jasna.bootstrap import sanitize_sys_path_for_local_dev


def test_sanitize_sys_path_replaces_package_dir_with_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "jasna"

    original = list(sys.path)
    try:
        sys.path[0] = str(package_dir)
        sanitize_sys_path_for_local_dev(package_dir)

        normalized = {os.path.normcase(os.path.abspath(p)) for p in sys.path if p}
        assert os.path.normcase(os.path.abspath(str(package_dir))) not in normalized
        assert os.path.normcase(os.path.abspath(str(repo_root))) in normalized
    finally:
        sys.path[:] = original

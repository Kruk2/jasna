#!/usr/bin/env python3
"""Launch Jasna only after the selected native runtime passes preflight."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jasna.runtime_contract import (  # noqa: E402
    RuntimeContractError,
    build_runtime_environment,
    default_runtime_root,
    validate_loaded_runtime,
)


def _preflight_child(runtime_root: Path, repo_root: Path) -> int:
    try:
        result = validate_loaded_runtime(runtime_root, repo_root)
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _product_child(
    runtime_root: Path,
    repo_root: Path,
    jasna_args: list[str],
) -> int:
    """Run Jasna while retaining the selected Windows DLL directory handles."""

    try:
        validate_loaded_runtime(runtime_root, repo_root)
    except Exception as exc:
        print(f"Jasna unified runtime launch failed: {exc}", file=sys.stderr)
        return 1
    os.chdir(repo_root)
    sys.argv = [str(repo_root / "jasna"), *jasna_args]
    runpy.run_module("jasna", run_name="__main__", alter_sys=True)
    return 0


def _write_preflight_record(stdout: str) -> None:
    state_dir = Path(
        os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
    ) / "jasna"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "runtime-preflight.json").write_text(
            stdout.strip() + "\n",
            encoding="utf-8",
        )
    except OSError:
        # A read-only state directory must not turn a passed runtime into a
        # failed launch; the preflight output is still printed on request.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--runtime-root", default=str(default_runtime_root()))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--_preflight-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_product-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args, jasna_args = parser.parse_known_args()
    runtime_root = Path(args.runtime_root).expanduser().resolve(strict=False)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    if jasna_args[:1] == ["--"]:
        jasna_args = jasna_args[1:]
    if args._preflight_child:
        return _preflight_child(runtime_root, repo_root)
    if args._product_child:
        return _product_child(runtime_root, repo_root, jasna_args)
    try:
        environment = build_runtime_environment(
            runtime_root,
            repo_root,
            python_executable=sys.executable,
        )
    except (OSError, RuntimeContractError) as exc:
        print(f"Jasna unified runtime preflight failed: {exc}", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_preflight-child",
        "--runtime-root",
        str(runtime_root),
        "--repo-root",
        str(repo_root),
    ]
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        print(f"Jasna unified runtime preflight failed: {detail}", file=sys.stderr)
        return completed.returncode or 1
    _write_preflight_record(completed.stdout)
    if args.preflight_only:
        print(completed.stdout.strip())
        return 0

    if sys.platform == "win32":
        product_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_product-child",
            "--runtime-root",
            str(runtime_root),
            "--repo-root",
            str(repo_root),
            "--",
            *jasna_args,
        ]
        # CPython's os.execve emulation on Windows can tear down extension
        # runtimes while replacing the current process. In the ROCm/PyAV
        # product environment that has produced an access violation in
        # ucrtbase.dll. A normal child preserves the same validated
        # environment without running that unsafe replacement path. The
        # hidden child validates the loaded runtime again after Windows has
        # resolved its DLLs.
        try:
            completed = subprocess.run(
                product_command,
                env=environment,
                check=False,
            )
        except OSError as exc:
            print(f"Jasna unified runtime launch failed: {exc}", file=sys.stderr)
            return 1
        return completed.returncode

    os.chdir(repo_root)
    product_command = [sys.executable, "-m", "jasna", *jasna_args]
    os.execve(sys.executable, product_command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

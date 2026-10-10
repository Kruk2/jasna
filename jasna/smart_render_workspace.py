"""Persistent, signature-bound workspace for recoverable smart rendering."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_ALGORITHM_VERSION = "jasna-smart-render-workspace-v4"


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    with source.open("rb") as handle:
        head = handle.read(chunk_size)
        digest.update(head)
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return {
        "path": str(source),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_tail_sha256": digest.hexdigest(),
    }


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not str(path).strip():
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(resolved),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _span_payload(span) -> dict[str, Any]:
    return {
        "kind": str(span.kind),
        "start_pts": int(span.start_pts),
        "end_pts": int(span.end_pts),
        "effect_ranges": [list(map(int, values)) for values in span.effect_ranges],
    }


def workspace_signature(
    *,
    source: str | Path,
    output: str | Path,
    plan,
    processing: Mapping[str, Any],
    model_files: Mapping[str, str | Path | None],
    codec: str,
    encoder_settings: Mapping[str, Any],
    resolved_projection: str,
) -> dict[str, Any]:
    return {
        "algorithm_version": WORKSPACE_ALGORITHM_VERSION,
        "source": source_identity(source),
        "output_path": str(Path(output).resolve()),
        "splice": {
            "time_base": str(plan.index.time_base),
            "start_pts": int(plan.index.start_pts),
            "end_pts": int(plan.index.end_pts),
            "keyframes": list(map(int, plan.index.pts)),
            "max_b_frames": int(plan.index.max_b_frames),
            "uses_b_references": bool(plan.index.uses_b_references),
            "decode_delay_pts": int(getattr(plan.index, "decode_delay_pts", 0)),
            "spans": [_span_payload(span) for span in plan.spans],
        },
        "processing": dict(processing),
        "model_files": {
            name: file_identity(path) for name, path in sorted(model_files.items())
        },
        "encoding": {
            "codec": str(codec),
            "settings": dict(encoder_settings),
        },
        "resolved_projection": str(resolved_projection),
    }


@dataclass
class SmartRenderWorkspace:
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        output: str | Path,
        signature: Mapping[str, Any],
    ) -> "SmartRenderWorkspace":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        signature_value = dict(signature)
        signature_hash = _canonical_hash(signature_value)
        output_path = Path(output)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", output_path.stem).strip("._-")
        slug = (slug or "output")[:64]
        work_path = root_path / f".{slug}.segments-{signature_hash[:16]}"
        work_path.mkdir(parents=True, exist_ok=True)
        manifest_path = work_path / "manifest.json"

        manifest: dict[str, Any] | None = None
        expected_spans = signature_value["splice"]["spans"]
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                spans = loaded.get("spans") if isinstance(loaded, dict) else None
                valid_spans = (
                    isinstance(spans, list)
                    and len(spans) == len(expected_spans)
                    and all(
                        isinstance(item, dict)
                        and item.get("index") == index
                        and all(item.get(key) == expected.get(key) for key in expected)
                        and item.get("status") in {"pending", "running", "complete"}
                        for index, (item, expected) in enumerate(
                            zip(spans, expected_spans)
                        )
                    )
                )
                if (
                    isinstance(loaded, dict)
                    and loaded.get("schema_version") == WORKSPACE_SCHEMA_VERSION
                    and loaded.get("signature_sha256") == signature_hash
                    and loaded.get("signature") == signature_value
                    and valid_spans
                ):
                    manifest = loaded
            except (OSError, json.JSONDecodeError):
                manifest = None
            if manifest is None:
                backup = manifest_path.with_name(
                    f"manifest.invalid-{os.getpid()}-{os.urandom(4).hex()}.json"
                )
                os.replace(manifest_path, backup)

        if manifest is None:
            manifest = {
                "schema_version": WORKSPACE_SCHEMA_VERSION,
                "signature_sha256": signature_hash,
                "signature": signature_value,
                "spans": [
                    {
                        "index": index,
                        **span,
                        "status": "pending",
                        "artifact": None,
                    }
                    for index, span in enumerate(expected_spans)
                ],
            }
            _write_json_atomic(manifest_path, manifest)
        else:
            changed = False
            for item in manifest["spans"]:
                if item.get("status") == "running":
                    item["status"] = "pending"
                    item["artifact"] = None
                    changed = True
            if changed:
                _write_json_atomic(manifest_path, manifest)
        return cls(work_path, manifest_path, manifest)

    def raw_path(self, index: int) -> Path:
        return self.path / f"{int(index):04d}-raw.nut"

    def fragment_path(self, index: int, suffix: str) -> Path:
        return self.path / f"{int(index):04d}{suffix}"

    def mark_running(self, index: int) -> None:
        item = self.manifest["spans"][int(index)]
        item["status"] = "running"
        item["artifact"] = None
        _write_json_atomic(self.manifest_path, self.manifest)

    def mark_complete(self, index: int, artifact: Path) -> None:
        artifact = Path(artifact).resolve()
        if artifact.parent != self.path.resolve():
            raise RuntimeError(
                f"Smart-render fragment is outside its workspace: {artifact}"
            )
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise RuntimeError(f"Smart-render fragment is missing or empty: {artifact}")
        stat = artifact.stat()
        item = self.manifest["spans"][int(index)]
        item["status"] = "complete"
        item["artifact"] = {
            "path": str(artifact),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(artifact),
        }
        _write_json_atomic(self.manifest_path, self.manifest)

    def reusable_fragment(self, index: int) -> Path | None:
        item = self.manifest["spans"][int(index)]
        artifact = item.get("artifact")
        if item.get("status") != "complete" or not isinstance(artifact, dict):
            return None
        try:
            path = Path(str(artifact["path"]))
            stat = path.stat()
            if (
                not path.is_absolute()
                or path.resolve().parent != self.path.resolve()
                or not path.is_file()
                or stat.st_size != int(artifact["size_bytes"])
                or stat.st_mtime_ns != int(artifact["mtime_ns"])
                or _file_sha256(path) != str(artifact["sha256"])
            ):
                return None
        except (KeyError, OSError, TypeError, ValueError):
            return None
        return path

    def cleanup(self) -> None:
        if (
            self.manifest_path.parent != self.path
            or self.manifest_path.name != "manifest.json"
            or ".segments-" not in self.path.name
        ):
            raise RuntimeError(f"Refusing to remove invalid smart-render workspace: {self.path}")
        shutil.rmtree(self.path)

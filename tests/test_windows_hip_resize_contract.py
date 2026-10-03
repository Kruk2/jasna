"""CPU-only tests for the exact Windows HIP resize admission contract."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PRODUCT = Path(__file__).resolve().parents[1]
MODULE_PATH = PRODUCT / "jasna" / "media" / "windows_hip_resize_contract.py"
SPEC = importlib.util.spec_from_file_location("windows_hip_resize_contract", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"cannot load contract module from {MODULE_PATH}")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)

EXPECTED_MANIFEST = {
    "schema": "jasna.hip-resize.windows.v1",
    "platform": "win32",
    "architecture": "gfx1100",
    "torch_hip": "7.16.26354",
    "runtime_version": 71626354,
    "runtime_dll_sha256": "37c40daa884de68bb7b5ee1b0575ab97b96866adf304fd76fdbccba772830f5f",
    "parameter_abi": "jasna.resize-normalize.params.v1",
    "source": {
        "file": "resize_normalize.cu",
        "sha256": "206391fa046d3a96b725eaa14e66b9bbfe8e95b903421fe4be74ac270fef63ab",
    },
    "artifact": {
        "file": "resize_normalize.gfx1100.windows.co",
        "sha256": "3c93e066930ad74bfa90f50ffdc059b230bc3ccfcc1ead3b1aa40034d29d0a10",
    },
}

RUNTIME_IDENTITY = {
    "torch_hip": "7.16.26354",
    "runtime_version": 71626354,
    "runtime_dll_sha256": "37c40daa884de68bb7b5ee1b0575ab97b96866adf304fd76fdbccba772830f5f",
    # Extra runtime-location fields are deliberately ignored by the contract.
    "runtime_dll": "amdhip64_7.dll",
    "runtime_path": r"C:\matched\amdhip64_7.dll",
}


def _valid_elf_header() -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4:9] = bytes((2, 1, 1, 64, 4))
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = (224).to_bytes(2, "little")
    header[20:24] = (1).to_bytes(4, "little")
    return bytes(header)


class WindowsHipResizeContractTests(unittest.TestCase):
    def _write_bundle(
        self,
        directory: Path,
        *,
        manifest: dict[str, object] | None = None,
        artifact: bytes | None = None,
        source: bytes | None = None,
    ) -> Path:
        code_object = directory / CONTRACT.CODE_OBJECT
        code_object.write_bytes(_valid_elf_header() if artifact is None else artifact)
        (directory / CONTRACT.MANIFEST).write_text(
            json.dumps(EXPECTED_MANIFEST if manifest is None else manifest),
            encoding="utf-8",
        )
        if source is not None:
            (directory / "resize_normalize.cu").write_bytes(source)
        return code_object

    @staticmethod
    def _expected_digest(path: Path, **_kwargs: object) -> str:
        name = Path(path).name
        if name == CONTRACT.CODE_OBJECT:
            return EXPECTED_MANIFEST["artifact"]["sha256"]
        if name == "resize_normalize.cu":
            return EXPECTED_MANIFEST["source"]["sha256"]
        raise AssertionError(f"unexpected digest request for {path}")

    def _validate_with_expected_hashes(
        self,
        directory: Path,
        *,
        runtime_identity: dict[str, object] | None = None,
        torch_hip: str = "7.16.26354",
        architecture: str = "gfx1100",
    ) -> Path:
        with mock.patch.object(
            CONTRACT, "_sha256_file", side_effect=self._expected_digest
        ):
            return CONTRACT.validate_bundle(
                directory,
                copy.deepcopy(RUNTIME_IDENTITY)
                if runtime_identity is None
                else runtime_identity,
                torch_hip,
                architecture,
            )

    def test_requested_parses_only_the_supplied_mapping(self) -> None:
        for raw in ("", "  ", "0", "False", "NO", "off"):
            with self.subTest(raw=raw):
                self.assertFalse(CONTRACT.requested({CONTRACT.ENV: raw}))
        for raw in ("1", " true ", "YES", "On"):
            with self.subTest(raw=raw):
                self.assertTrue(CONTRACT.requested({CONTRACT.ENV: raw}))
        with mock.patch.dict(os.environ, {CONTRACT.ENV: "1"}):
            self.assertFalse(CONTRACT.requested({}))
        with self.assertRaisesRegex(ValueError, CONTRACT.ENV):
            CONTRACT.requested({CONTRACT.ENV: "maybe"})
        with self.assertRaisesRegex(ValueError, CONTRACT.ENV):
            CONTRACT.requested({CONTRACT.ENV: 1})

    def test_accepted_manifest_is_exact_and_isolated(self) -> None:
        self.assertEqual(CONTRACT.accepted_manifest(), EXPECTED_MANIFEST)
        mutable = CONTRACT.accepted_manifest()
        mutable["artifact"]["file"] = "outside.co"
        self.assertEqual(CONTRACT.accepted_manifest(), EXPECTED_MANIFEST)

    def test_supported_geometry_accepts_the_admitted_matrix(self) -> None:
        admitted = (
            ((4, 3, 1080, 1920), (6220800, 2073600, 1920, 1), (576, 576), (0, 0, 576, 576)),
            ((2, 3, 2160, 3840), (24883200, 8294400, 3840, 1), (640, 640), (0, 140, 640, 360)),
            ((3, 3, 300, 401), (360900, 120300, 401, 1), (640, 640), (0, 80, 640, 479)),
            ((1, 3, 4096, 4096), (100663296, 33554432, 8192, 1), (576, 576), (0, 0, 576, 576)),
            ((2, 3, 17, 47), (15688, 1961, 106, 1), (21, 37), (2, 3, 31, 17)),
            ((1, 3, 1, 1), (1, 1, 1, 1), (1, 1), (0, 0, 1, 1)),
        )
        for shape, strides, out_hw, content in admitted:
            with self.subTest(shape=shape, strides=strides):
                self.assertTrue(
                    CONTRACT.supported_geometry(shape, strides, out_hw, content)
                )

    def test_supported_geometry_rejects_invalid_shapes_strides_and_overflow(self) -> None:
        valid_shape = (1, 3, 2, 4)
        valid_strides = (24, 8, 4, 1)
        valid_out = (16, 16)
        valid_content = (0, 0, 16, 16)
        rejected = (
            ((1, 3, 2), valid_strides, valid_out, valid_content),
            ((True, 3, 2, 4), valid_strides, valid_out, valid_content),
            ((1, 4, 2, 4), valid_strides, valid_out, valid_content),
            ((5, 3, 2, 4), valid_strides, valid_out, valid_content),
            ((1, 3, 8193, 4), valid_strides, valid_out, valid_content),
            (valid_shape, valid_strides, (641, 16), valid_content),
            (valid_shape, (24, 8, 4, 2), valid_out, valid_content),
            (valid_shape, (24, 8, 0, 1), valid_out, valid_content),
            (valid_shape, (24, 8, 3, 1), valid_out, valid_content),
            (valid_shape, (24, 7, 4, 1), valid_out, valid_content),
            ((2, 3, 2, 4), (20, 8, 4, 1), valid_out, valid_content),
            ((1, 3, 1, 1), (1, 1 << 62, 1, 1), (1, 1), (0, 0, 1, 1)),
            ((1, 3, 1, 1), (1, 1, 1, 1 << 63), (1, 1), (0, 0, 1, 1)),
            (valid_shape, valid_strides, valid_out, (0, 0, 0, 1)),
            (valid_shape, valid_strides, valid_out, (-1, 0, 1, 1)),
            (valid_shape, valid_strides, valid_out, (1, 1, 16, 16)),
        )
        for shape, strides, out_hw, content in rejected:
            with self.subTest(shape=shape, strides=strides, out_hw=out_hw, content=content):
                self.assertFalse(
                    CONTRACT.supported_geometry(shape, strides, out_hw, content)
                )

    def test_validate_bundle_accepts_fixed_bundle_and_optional_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            code_object = self._write_bundle(directory)
            self.assertEqual(self._validate_with_expected_hashes(directory), code_object)

            self._write_bundle(directory, source=b"pinned source placeholder")
            self.assertEqual(self._validate_with_expected_hashes(directory), code_object)

    def test_validate_bundle_rejects_manifest_drift_duplicate_keys_and_size(self) -> None:
        mutations = (
            ("unknown key", lambda value: value.update({"unexpected": "value"})),
            ("boolean numeric", lambda value: value.__setitem__("runtime_version", True)),
            (
                "manifest-directed artifact path",
                lambda value: value["artifact"].__setitem__("file", "outside.co"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = copy.deepcopy(EXPECTED_MANIFEST)
                mutate(manifest)
                self._write_bundle(directory, manifest=manifest)
                with self.assertRaisesRegex(RuntimeError, "accepted contract"):
                    self._validate_with_expected_hashes(directory)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory)
            (directory / CONTRACT.MANIFEST).write_text(
                '{"schema":"one","schema":"two"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                self._validate_with_expected_hashes(directory)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory)
            (directory / CONTRACT.MANIFEST).write_bytes(
                b" " * (64 * 1024 + 1)
            )
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                self._validate_with_expected_hashes(directory)

    def test_validate_bundle_rejects_runtime_mismatches(self) -> None:
        cases = (
            ("identity torch", {**RUNTIME_IDENTITY, "torch_hip": "7.16.0"}, "7.16.26354", "gfx1100"),
            ("runtime version", {**RUNTIME_IDENTITY, "runtime_version": 1}, "7.16.26354", "gfx1100"),
            ("runtime digest", {**RUNTIME_IDENTITY, "runtime_dll_sha256": "0" * 64}, "7.16.26354", "gfx1100"),
            ("torch argument", RUNTIME_IDENTITY, "7.16.0", "gfx1100"),
            ("architecture", RUNTIME_IDENTITY, "7.16.26354", "gfx1200"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory)
            for label, identity, torch_hip, architecture in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        self._validate_with_expected_hashes(
                            directory,
                            runtime_identity=copy.deepcopy(identity),
                            torch_hip=torch_hip,
                            architecture=architecture,
                        )

    def test_validate_bundle_rejects_hash_header_and_capacity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory)
            with mock.patch.object(
                CONTRACT, "_sha256_file", return_value="0" * 64
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    CONTRACT.validate_bundle(
                        directory, copy.deepcopy(RUNTIME_IDENTITY), "7.16.26354", "gfx1100"
                    )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bad_header = bytearray(_valid_elf_header())
            bad_header[8] = 3
            self._write_bundle(directory, artifact=bytes(bad_header))
            with self.assertRaisesRegex(RuntimeError, "HSA ABI 4"):
                self._validate_with_expected_hashes(directory)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory, artifact=b"x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                self._validate_with_expected_hashes(directory)

    def test_validate_bundle_verifies_a_present_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory, source=b"source")

            def mismatched_source(path: Path, **_kwargs: object) -> str:
                if Path(path).name == "resize_normalize.cu":
                    return "0" * 64
                return self._expected_digest(path)

            with mock.patch.object(
                CONTRACT, "_sha256_file", side_effect=mismatched_source
            ):
                with self.assertRaisesRegex(RuntimeError, "source SHA256 mismatch"):
                    CONTRACT.validate_bundle(
                        directory, copy.deepcopy(RUNTIME_IDENTITY), "7.16.26354", "gfx1100"
                    )


if __name__ == "__main__":
    unittest.main()

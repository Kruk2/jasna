"""CPU-only tests for the exact-device Windows global VRAM reader."""

from __future__ import annotations

import builtins
import ctypes
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from jasna.windows_global_vram import WindowsGlobalVramReader, WindowsGpuIdentity


class FakeHipDeviceGetLuid:
    def __init__(self, luid: bytes, node_mask: int, status: int = 0):
        self.luid = bytes(luid)
        self.node_mask = node_mask
        self.status = status
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, luid_buffer, mask_pointer, device_index):
        self.calls.append((luid_buffer, mask_pointer, device_index))
        if self.status == 0:
            ctypes.memmove(luid_buffer, self.luid, 8)
            ctypes.cast(mask_pointer, ctypes.POINTER(ctypes.c_uint)).contents.value = (
                self.node_mask
            )
        return self.status


class FakeHipLibrary:
    def __init__(self, luid: bytes, node_mask: int, status: int = 0):
        self.hipDeviceGetLuid = FakeHipDeviceGetLuid(luid, node_mask, status)


class FakeTelemetry:
    def __init__(
        self,
        marker: str,
        total_vram,
        memory_values,
        refresh_results=((1, 2),),
        close_errors=(),
    ):
        self._adapter_marker = marker
        self._total_vram = total_vram
        self._memory_counter = object()
        self._memory_values = list(memory_values)
        self._refresh_results = list(refresh_results)
        self._close_errors = list(close_errors)
        self.read_calls = 0
        self.values_calls = []
        self.close_calls = 0

    def read(self):
        self.read_calls += 1
        if len(self._refresh_results) > 1:
            return self._refresh_results.pop(0)
        return self._refresh_results[0]

    def _values(self, counter):
        self.values_calls.append(counter)
        return list(self._memory_values)

    def close(self):
        self.close_calls += 1
        if self._close_errors:
            raise self._close_errors.pop(0)


class BlockingTelemetry(FakeTelemetry):
    def __init__(self, marker, total_vram, memory_values, entered, release):
        super().__init__(marker, total_vram, memory_values)
        self.entered = entered
        self.release = release

    def read(self):
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise AssertionError("telemetry refresh was not released")
        return super().read()


class WindowsGlobalVramReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luid = bytes.fromhex("b0f70000 00000000")
        cls.marker = "luid_0x00000000_0x0000f7b0_phys_"

    def make_hip(self, *, node_mask=1, status=0, luid=None):
        return FakeHipLibrary(luid or self.luid, node_mask, status)

    def make_telemetry(self, *, marker=None, total=1000, values=None, refresh=((1, 2),)):
        if values is None:
            values = [(f"{marker or self.marker}0", 250)]
        return FakeTelemetry(marker or self.marker, total, values, refresh)

    def make_identity(self, node_index=0):
        return WindowsGpuIdentity(self.marker, node_index)

    def test_binds_and_calls_hip_luid_with_requested_device(self):
        hip = self.make_hip(node_mask=4)
        telemetry = self.make_telemetry(
            values=[(f"{self.marker}2", 250)],
        )
        reader = WindowsGlobalVramReader(
            7,
            hip,
            telemetry_factory=lambda: telemetry,
        )
        self.assertEqual(
            hip.hipDeviceGetLuid.argtypes,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int],
        )
        self.assertIs(hip.hipDeviceGetLuid.restype, ctypes.c_int)
        self.assertEqual(len(hip.hipDeviceGetLuid.calls), 1)
        self.assertEqual(hip.hipDeviceGetLuid.calls[0][2], 7)
        self.assertEqual(reader.adapter_marker, self.marker)
        self.assertEqual(reader.node_index, 2)
        self.assertEqual(reader.total_bytes, 1000)
        self.assertEqual(reader.identity, self.make_identity(2))

    def test_hip_status_failure_does_not_construct_telemetry(self):
        hip = self.make_hip(status=9)
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return self.make_telemetry()

        with self.assertRaisesRegex(RuntimeError, "status 9"):
            WindowsGlobalVramReader(0, hip, telemetry_factory=factory)
        self.assertEqual(factory_calls, [])

    def test_device_index_rejects_bool_and_c_int_overflow_before_hip_call(self):
        for device_index in (True, False, -1, 1 << 31, 1 << 80):
            with self.subTest(device_index=device_index):
                hip = self.make_hip()
                with self.assertRaises((TypeError, ValueError)):
                    WindowsGlobalVramReader(
                        device_index,
                        hip,
                        telemetry_factory=self.make_telemetry,
                    )
                self.assertEqual(hip.hipDeviceGetLuid.calls, [])

    def test_zero_and_ambiguous_node_masks_are_rejected(self):
        for mask, message in (
            (0, "zero"),
            (3, "ambiguous"),
            (0xA, "ambiguous"),
        ):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, message):
                    WindowsGlobalVramReader(
                        0,
                        self.make_hip(node_mask=mask),
                        telemetry_factory=lambda: self.make_telemetry(),
                    )

    def test_mismatched_selected_gpu_is_closed_and_rejected(self):
        telemetry = self.make_telemetry(
            marker="luid_0x11111111_0x22222222_phys_",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            WindowsGlobalVramReader(
                0,
                self.make_hip(),
                telemetry_factory=lambda: telemetry,
            )
        self.assertEqual(telemetry.close_calls, 1)

    def test_refresh_failure_is_not_masked_by_old_raw_values(self):
        telemetry = self.make_telemetry(
            values=[(f"{self.marker}0", 321)],
            refresh=((None, None),),
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        with self.assertRaisesRegex(RuntimeError, "refresh failed"):
            reader()
        self.assertEqual(telemetry.read_calls, 1)
        self.assertEqual(telemetry.values_calls, [])

    def test_missing_sample_and_unrelated_instances_are_handled(self):
        telemetry = self.make_telemetry(
            values=[
                (f"{self.marker}0", 111),
                ("other_adapter_phys_0", 222),
            ],
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(node_mask=2),
            telemetry_factory=lambda: telemetry,
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            reader()

        telemetry._memory_values = [
            ("other_adapter_phys_0", float("nan")),
            (f"{self.marker}1", 333),
        ]
        self.assertEqual(reader(), (333, 1000))

    def test_duplicate_casefolded_samples_are_rejected(self):
        telemetry = self.make_telemetry(
            values=[
                (f"{self.marker}0", 111),
                (f"{self.marker.upper()}0", 222),
            ],
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reader()

    def test_exact_byte_precision_is_preserved(self):
        total = (1 << 53) + 17
        used = (1 << 53) + 9
        telemetry = self.make_telemetry(
            total=total,
            values=[(f"{self.marker}0", used)],
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        self.assertEqual(reader(), (used, total))

    def test_invalid_used_samples_are_rejected_without_rounding(self):
        invalid_samples = (float("nan"), float("inf"), 1.5, -1, 1001)
        for sample in invalid_samples:
            with self.subTest(sample=sample):
                telemetry = self.make_telemetry(
                    values=[(f"{self.marker}0", sample)],
                )
                reader = WindowsGlobalVramReader(
                    0,
                    self.make_hip(),
                    telemetry_factory=lambda telemetry=telemetry: telemetry,
                )
                with self.assertRaises(ValueError):
                    reader()

    def test_invalid_total_is_rejected_and_reader_is_closed(self):
        for total in (0, -1, 10.5, float("nan"), float("inf")):
            with self.subTest(total=total):
                telemetry = self.make_telemetry(total=total)
                with self.assertRaises(ValueError):
                    WindowsGlobalVramReader(
                        0,
                        self.make_hip(),
                        telemetry_factory=lambda telemetry=telemetry: telemetry,
                    )
                self.assertEqual(telemetry.close_calls, 1)

    def test_close_is_terminal_and_idempotent(self):
        telemetry = self.make_telemetry()
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        self.assertEqual(reader(), (250, 1000))
        reader.close()
        reader.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reader()
        self.assertEqual(telemetry.close_calls, 1)
        self.assertEqual(telemetry.read_calls, 1)

    def test_close_failure_retains_resource_for_retry(self):
        error = RuntimeError("close failed")
        telemetry = FakeTelemetry(
            self.marker,
            1000,
            [(f"{self.marker}0", 250)],
            close_errors=(error,),
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            reader.close()
        self.assertFalse(reader._closed)
        self.assertIs(reader._telemetry_reader, telemetry)
        reader.close()
        self.assertTrue(reader._closed)
        self.assertIsNone(reader._telemetry_reader)
        self.assertEqual(telemetry.close_calls, 2)

    def test_call_and_close_are_serialized(self):
        entered = threading.Event()
        release = threading.Event()
        telemetry = BlockingTelemetry(
            self.marker,
            1000,
            [(f"{self.marker}0", 250)],
            entered,
            release,
        )
        reader = WindowsGlobalVramReader(
            0,
            self.make_hip(),
            telemetry_factory=lambda: telemetry,
        )
        call_errors = []
        close_errors = []
        close_finished = threading.Event()

        def run_call():
            try:
                self.assertEqual(reader(), (250, 1000))
            except BaseException as error:  # pragma: no cover - asserted below
                call_errors.append(error)

        def run_close():
            try:
                reader.close()
            except BaseException as error:  # pragma: no cover - asserted below
                close_errors.append(error)
            finally:
                close_finished.set()

        call_thread = threading.Thread(target=run_call)
        close_thread = threading.Thread(target=run_close)
        call_thread.start()
        self.assertTrue(entered.wait(timeout=2))
        close_thread.start()
        self.assertFalse(close_finished.is_set())
        release.set()
        call_thread.join(timeout=2)
        close_thread.join(timeout=2)
        self.assertFalse(call_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(call_errors, [])
        self.assertEqual(close_errors, [])
        self.assertEqual(telemetry.close_calls, 1)

    def test_from_identity_uses_no_hip_or_torch_import(self):
        identity = self.make_identity()
        telemetry = self.make_telemetry()
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch.") or name.startswith("jasna.media.hip_kernel"):
                raise AssertionError(f"from_identity unexpectedly imported {name}")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch(
                "jasna.windows_global_vram._hip_identity",
                side_effect=AssertionError("from_identity unexpectedly accessed HIP"),
            ),
            patch("builtins.__import__", side_effect=guarded_import),
        ):
            reader = WindowsGlobalVramReader.from_identity(
                identity,
                telemetry_factory=lambda: telemetry,
            )
            self.assertEqual(reader(), (250, 1000))

    def test_identity_rejects_noncanonical_marker_and_invalid_nodes(self):
        invalid_markers = (
            "luid_0x00000000_0x0000F7B0_phys_",
            "luid_0x00000000_0x0000f7b0_phys_0",
            "not-a-luid",
            7,
        )
        for marker in invalid_markers:
            with self.subTest(marker=marker):
                with self.assertRaises((TypeError, ValueError)):
                    WindowsGpuIdentity(marker, 0)

        for node_index in (True, False, -1, 32, 1.0):
            with self.subTest(node_index=node_index):
                with self.assertRaises((TypeError, ValueError)):
                    WindowsGpuIdentity(self.marker, node_index)

    def test_reader_identity_is_immutable(self):
        identity = self.make_identity()
        reader = WindowsGlobalVramReader.from_identity(
            identity,
            telemetry_factory=self.make_telemetry,
        )

        self.assertEqual(reader.identity, identity)
        with self.assertRaises(FrozenInstanceError):
            reader.identity.node_index = 1
        with self.assertRaises(AttributeError):
            reader.identity = self.make_identity(1)

    def test_from_identity_mismatch_closes_telemetry(self):
        telemetry = self.make_telemetry(
            marker="luid_0x11111111_0x22222222_phys_",
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            WindowsGlobalVramReader.from_identity(
                self.make_identity(),
                telemetry_factory=lambda: telemetry,
            )

        self.assertEqual(telemetry.close_calls, 1)

    def test_from_identity_reads_exact_node_bytes(self):
        identity = self.make_identity(2)
        telemetry = self.make_telemetry(
            values=[
                (f"{self.marker}0", 111),
                (f"{self.marker}2", 333),
            ],
        )
        reader = WindowsGlobalVramReader.from_identity(
            identity,
            telemetry_factory=lambda: telemetry,
        )

        self.assertEqual(reader.identity, identity)
        self.assertEqual(reader(), (333, 1000))


if __name__ == "__main__":
    unittest.main(verbosity=2)

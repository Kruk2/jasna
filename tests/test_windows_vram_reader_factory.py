"""CPU-only contracts for the pipeline-owned Windows HIP VRAM reader factory."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from jasna import windows_global_vram as global_vram


class _FakeDevice:
    def __init__(self, device_type: str, index: int | None) -> None:
        self.type = device_type
        self.index = index


class _FakeTorch(types.ModuleType):
    def __init__(self, *, hip: object | None, selected: _FakeDevice, current_index: int = 0) -> None:
        super().__init__("torch")
        self.version = types.SimpleNamespace(hip=hip)
        self._selected = selected
        self._current_index = current_index
        self.device_calls: list[object] = []
        self.current_device_calls = 0
        self.device_error: BaseException | None = None
        self.current_device_error: BaseException | None = None
        self.cuda = types.SimpleNamespace(current_device=self.current_device)

    def device(self, value: object) -> _FakeDevice:
        self.device_calls.append(value)
        if self.device_error is not None:
            raise self.device_error
        return self._selected

    def current_device(self) -> int:
        self.current_device_calls += 1
        if self.current_device_error is not None:
            raise self.current_device_error
        return self._current_index


class _FakeHipModule(types.ModuleType):
    def __init__(self, runtime: object | None = None, error: BaseException | None = None) -> None:
        super().__init__("jasna.media.hip_kernel")
        self._runtime = runtime if runtime is not None else object()
        self._error = error
        self.calls = 0

    def hip_runtime(self) -> object:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._runtime


class WindowsHipVramReaderFactoryTests(unittest.TestCase):
    def _call_on_windows(
        self,
        torch_module: _FakeTorch,
        hip_module: _FakeHipModule,
        device: object,
        *,
        reader_return: object | None = None,
        reader_error: BaseException | None = None,
    ):
        fake_media = types.ModuleType("jasna.media")
        fake_media.__path__ = []
        calls: list[tuple[object, object]] = []

        def make_reader(index: object, runtime: object) -> object:
            calls.append((index, runtime))
            if reader_error is not None:
                raise reader_error
            return reader_return

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "jasna.media": fake_media,
                    "jasna.media.hip_kernel": hip_module,
                },
            ),
            patch.object(global_vram, "WindowsGlobalVramReader", side_effect=make_reader),
        ):
            result = global_vram.create_windows_hip_vram_reader(device)
        return result, calls

    def test_explicit_cuda_index_is_forwarded_with_the_selected_hip_runtime(self) -> None:
        expected_reader = object()
        runtime = object()
        torch_module = _FakeTorch(
            hip="6.2",
            selected=_FakeDevice("cuda", 7),
            current_index=3,
        )
        hip_module = _FakeHipModule(runtime=runtime)

        result, reader_calls = self._call_on_windows(
            torch_module,
            hip_module,
            "cuda:7",
            reader_return=expected_reader,
        )

        self.assertIs(result, expected_reader)
        self.assertEqual(torch_module.device_calls, ["cuda:7"])
        self.assertEqual(torch_module.current_device_calls, 0)
        self.assertEqual(hip_module.calls, 1)
        self.assertEqual(reader_calls, [(7, runtime)])

    def test_implicit_cuda_index_uses_current_device(self) -> None:
        expected_reader = object()
        runtime = object()
        torch_module = _FakeTorch(
            hip="6.2",
            selected=_FakeDevice("cuda", None),
            current_index=4,
        )
        hip_module = _FakeHipModule(runtime=runtime)

        result, reader_calls = self._call_on_windows(
            torch_module,
            hip_module,
            "cuda",
            reader_return=expected_reader,
        )

        self.assertIs(result, expected_reader)
        self.assertEqual(torch_module.device_calls, ["cuda"])
        self.assertEqual(torch_module.current_device_calls, 1)
        self.assertEqual(hip_module.calls, 1)
        self.assertEqual(reader_calls, [(4, runtime)])

    def test_non_windows_rejects_before_attempting_to_import_torch(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("non-Windows factory path imported Torch")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.object(sys, "platform", "linux"),
            patch.object(builtins, "__import__", side_effect=guarded_import),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires Windows"):
                global_vram.create_windows_hip_vram_reader("cuda:0")

    def test_non_hip_runtime_rejects_before_device_or_hip_construction(self) -> None:
        torch_module = _FakeTorch(
            hip=None,
            selected=_FakeDevice("cuda", 0),
        )
        hip_module = _FakeHipModule()

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "jasna.media": types.ModuleType("jasna.media"),
                    "jasna.media.hip_kernel": hip_module,
                },
            ),
            patch.object(global_vram, "WindowsGlobalVramReader") as reader_constructor,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires an AMD runtime"):
                global_vram.create_windows_hip_vram_reader("cuda:0")

        self.assertEqual(torch_module.device_calls, [])
        self.assertEqual(torch_module.current_device_calls, 0)
        self.assertEqual(hip_module.calls, 0)
        reader_constructor.assert_not_called()

    def test_cpu_device_is_rejected_before_hip_or_reader_construction(self) -> None:
        torch_module = _FakeTorch(
            hip="6.2",
            selected=_FakeDevice("cpu", None),
        )
        hip_module = _FakeHipModule()

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "jasna.media": types.ModuleType("jasna.media"),
                    "jasna.media.hip_kernel": hip_module,
                },
            ),
            patch.object(global_vram, "WindowsGlobalVramReader") as reader_constructor,
        ):
            with self.assertRaisesRegex(ValueError, "CUDA/HIP device"):
                global_vram.create_windows_hip_vram_reader("cpu")

        self.assertEqual(torch_module.device_calls, ["cpu"])
        self.assertEqual(torch_module.current_device_calls, 0)
        self.assertEqual(hip_module.calls, 0)
        reader_constructor.assert_not_called()

    def test_factory_dependency_errors_propagate_without_substitution(self) -> None:
        cases = (
            ("device", "device_error", ValueError("synthetic device error")),
            ("current device", "current_device_error", RuntimeError("synthetic current device error")),
            ("HIP runtime", "hip", RuntimeError("synthetic HIP runtime error")),
            ("reader", "reader", RuntimeError("synthetic reader error")),
        )

        for name, source, expected in cases:
            with self.subTest(source=name):
                selected = _FakeDevice("cuda", None if source == "current_device_error" else 3)
                torch_module = _FakeTorch(hip="6.2", selected=selected, current_index=4)
                hip_module = _FakeHipModule(
                    error=expected if source == "hip" else None,
                )
                if source == "device_error":
                    torch_module.device_error = expected
                elif source == "current_device_error":
                    torch_module.current_device_error = expected

                fake_media = types.ModuleType("jasna.media")
                fake_media.__path__ = []
                with (
                    patch.object(sys, "platform", "win32"),
                    patch.dict(
                        sys.modules,
                        {
                            "torch": torch_module,
                            "jasna.media": fake_media,
                            "jasna.media.hip_kernel": hip_module,
                        },
                    ),
                    patch.object(
                        global_vram,
                        "WindowsGlobalVramReader",
                        side_effect=expected if source == "reader" else None,
                    ) as reader_constructor,
                ):
                    with self.assertRaisesRegex(type(expected), str(expected)) as raised:
                        global_vram.create_windows_hip_vram_reader("cuda")

                self.assertIs(raised.exception, expected)
                if source == "reader":
                    reader_constructor.assert_called_once_with(3, hip_module._runtime)


class WindowsHipVramReaderFactoryImportTests(unittest.TestCase):
    def test_module_import_has_no_eager_torch_or_hip_runtime_import(self) -> None:
        source_path = PRODUCT_ROOT / "jasna" / "windows_global_vram.py"
        module_name = "_windows_vram_reader_factory_import_probe"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:  # pragma: no cover - repository invariant
            self.fail("could not load the product source")
        probe = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch.") or name == "jasna.media.hip_kernel":
                raise AssertionError(f"module import eagerly requested {name}")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(sys.modules, {module_name: probe}),
            patch.object(builtins, "__import__", side_effect=guarded_import),
        ):
            spec.loader.exec_module(probe)

        self.assertTrue(callable(probe.create_windows_hip_vram_reader))


if __name__ == "__main__":
    unittest.main(verbosity=2)

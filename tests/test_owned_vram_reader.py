"""CPU-only lifecycle contracts for VramOffloader-owned VRAM readers.

This module loads the production offloader source against synthetic ``torch``
and product-buffer modules.  It never imports installed Torch or initializes a
GPU, PDH, media runtime, GUI, FFmpeg, or a native worker.
"""

from __future__ import annotations

import ast
from collections import deque
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
_OFFLOADER_SOURCE = PRODUCT_ROOT / "jasna" / "vram_offloader.py"


class _FakeCuda:
    def get_device_properties(self, _device: object) -> object:
        return types.SimpleNamespace(total_memory=8 * 1024**3)

    def mem_get_info(self, _device: object) -> tuple[int, int]:
        return (7 * 1024**3, 8 * 1024**3)

    def empty_cache(self) -> None:
        return None


def _load_offloader_without_torch() -> tuple[types.ModuleType, types.ModuleType]:
    """Execute the actual source with minimal inert dependency modules."""

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    fake_jasna = types.ModuleType("jasna")
    fake_jasna.__path__ = []
    fake_blend = types.ModuleType("jasna.blend_buffer")
    fake_blend.BlendBuffer = type("BlendBuffer", (), {})
    fake_crop = types.ModuleType("jasna.crop_buffer")
    fake_crop.CropBuffer = type("CropBuffer", (), {})
    module_name = "_owned_vram_offloader_product"
    spec = importlib.util.spec_from_file_location(module_name, _OFFLOADER_SOURCE)
    if spec is None or spec.loader is None:  # pragma: no cover - repository invariant
        raise AssertionError("could not load the product offloader source")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "jasna": fake_jasna,
            "jasna.blend_buffer": fake_blend,
            "jasna.crop_buffer": fake_crop,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module, fake_torch


class _ThreadFactory:
    def __init__(
        self,
        *,
        run_on_start: bool = False,
        complete_on_join: bool = False,
        start_error: BaseException | None = None,
    ) -> None:
        self.run_on_start = run_on_start
        self.complete_on_join = complete_on_join
        self.start_error = start_error
        self.threads: list[_FakeThread] = []

    def __call__(self, *, target, name: str, daemon: bool) -> "_FakeThread":
        thread = _FakeThread(target=target, name=name, daemon=daemon, factory=self)
        self.threads.append(thread)
        return thread

    @property
    def thread(self) -> "_FakeThread":
        if len(self.threads) != 1:
            raise AssertionError("expected exactly one offloader thread")
        return self.threads[0]


class _FakeThread:
    _next_ident = 5000

    def __init__(self, *, target, name: str, daemon: bool, factory: _ThreadFactory) -> None:
        self._target = target
        self.name = name
        self.daemon = daemon
        self._factory = factory
        self.ident: int | None = None
        self._alive = False
        self._finished = False
        self.start_calls = 0
        self.join_timeouts: list[float | None] = []

    def start(self) -> None:
        self.start_calls += 1
        if self.start_calls > 1:
            raise RuntimeError("fake thread cannot be started twice")
        if self._factory.start_error is not None:
            raise self._factory.start_error
        self.ident = self._next_ident
        type(self)._next_ident += 1
        self._alive = True
        if self._factory.run_on_start:
            self.finish()

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if self._factory.complete_on_join and self._alive:
            self.finish()

    def is_alive(self) -> bool:
        return self._alive

    def finish(self) -> None:
        if self._finished:
            return
        if not self._alive:
            raise AssertionError("cannot finish a thread that did not start")
        self._finished = True
        try:
            self._target()
        finally:
            self._alive = False


class _FakeReader:
    def __init__(
        self,
        samples=(),
        *,
        close_error: BaseException | None = None,
        order: list[str] | None = None,
    ) -> None:
        self._samples = deque(samples)
        self._close_error = close_error
        self._order = order
        self.calls = 0
        self.close_calls = 0

    def __call__(self):
        self.calls += 1
        if not self._samples:
            raise AssertionError("reader was sampled more often than the test allowed")
        result = self._samples.popleft()
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self) -> None:
        self.close_calls += 1
        if self._order is not None:
            self._order.append("close")
        if self._close_error is not None:
            raise self._close_error


class _CloseOnlyReader:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _CallableWithoutClose:
    def __call__(self) -> tuple[int, int]:
        return (1, 2)


class OwnedVramReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.offloader_module, cls.fake_torch = _load_offloader_without_torch()

    def _construct(self, thread_factory: _ThreadFactory, **kwargs):
        with patch.object(self.offloader_module.threading, "Thread", thread_factory):
            return self.offloader_module.VramOffloader(
                device=object(),
                blend_buffer=object(),
                crop_buffers={},
                crop_lock=object(),
                vram_limit=1.0,
                safetynet=0,
                **kwargs,
            )

    def _owned(
        self,
        reader_factory,
        errors: list[BaseException],
        thread_factory: _ThreadFactory,
        **kwargs,
    ):
        return self._construct(
            thread_factory,
            system_vram_startup_budget=1,
            system_vram_reader_factory=reader_factory,
            on_system_vram_error=errors.append,
            **kwargs,
        )

    def test_actual_source_is_loaded_against_the_fake_torch_module(self) -> None:
        self.assertIs(self.offloader_module.torch, self.fake_torch)
        self.assertNotEqual(self.offloader_module.__name__, "jasna.vram_offloader")

    def test_owned_reader_initial_sample_updates_stats_then_thread_owned_finally_closes_once(self) -> None:
        order: list[str] = []
        reader = _FakeReader(((400, 1000),), order=order)
        errors: list[BaseException] = []
        threads = _ThreadFactory()
        offloader = self._owned(lambda: reader, errors, threads)
        offloader._run_loop = lambda: order.append("poll finished")

        offloader.start()

        self.assertEqual(reader.calls, 1)
        self.assertEqual(reader.close_calls, 0)
        self.assertTrue(threads.thread.is_alive())
        self.assertEqual(offloader.stats.system_sample_count, 1)
        self.assertEqual(offloader.stats.system_max_used_bytes, 400)
        self.assertEqual(offloader.stats.system_min_headroom_bytes, 600)

        threads.thread.finish()

        self.assertEqual(order, ["poll finished", "close"])
        self.assertEqual(reader.close_calls, 1)
        self.assertEqual(errors, [])
        offloader.stop()
        offloader.stop()
        self.assertEqual(reader.close_calls, 1)
        self.assertEqual(threads.thread.join_timeouts, [5.0, 5.0])

    def test_invalid_initial_samples_fail_closed_report_and_cleanup_before_thread_start(self) -> None:
        invalid_samples = (
            ("unavailable", None),
            ("list", [1, 2]),
            ("boolean", (True, 2)),
            ("wrong arity", (1,)),
            ("zero total", (0, 0)),
            ("negative used", (-1, 2)),
            ("used exceeds total", (3, 2)),
        )

        for name, sample in invalid_samples:
            with self.subTest(sample=name):
                reader = _FakeReader((sample,))
                errors: list[BaseException] = []
                threads = _ThreadFactory()
                offloader = self._owned(lambda reader=reader: reader, errors, threads)

                with self.assertRaisesRegex(ValueError, "required whole-card VRAM sample") as raised:
                    offloader.start()

                self.assertEqual(threads.thread.start_calls, 0)
                self.assertTrue(offloader._stop.is_set())
                self.assertEqual(errors, [raised.exception])
                self.assertEqual(reader.close_calls, 1)
                self.assertIs(offloader._system_vram_failure, raised.exception)
                with self.assertRaisesRegex(RuntimeError, "required whole-card VRAM monitoring failed") as stopped:
                    offloader.stop()
                self.assertIs(stopped.exception.__cause__, raised.exception)
                self.assertEqual(reader.close_calls, 1)

    def test_factory_startup_and_reader_shape_failures_report_first_error_and_cleanup_when_possible(self) -> None:
        with self.subTest(failure="factory"):
            errors: list[BaseException] = []
            threads = _ThreadFactory()
            factory_error = RuntimeError("synthetic factory failure")
            offloader = self._owned(
                lambda: (_ for _ in ()).throw(factory_error), errors, threads
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic factory failure") as raised:
                offloader.start()
            self.assertEqual(errors, [raised.exception])
            self.assertTrue(offloader._stop.is_set())
            self.assertEqual(threads.thread.start_calls, 0)

        with self.subTest(failure="not callable"):
            errors = []
            threads = _ThreadFactory()
            reader = _CloseOnlyReader()
            offloader = self._owned(lambda: reader, errors, threads)
            with self.assertRaisesRegex(TypeError, "callable and closeable") as raised:
                offloader.start()
            self.assertEqual(errors, [raised.exception])
            self.assertEqual(reader.close_calls, 1)
            self.assertEqual(threads.thread.start_calls, 0)

        with self.subTest(failure="not closeable"):
            errors = []
            threads = _ThreadFactory()
            offloader = self._owned(lambda: _CallableWithoutClose(), errors, threads)
            with self.assertRaisesRegex(TypeError, "callable and closeable") as raised:
                offloader.start()
            self.assertEqual(errors, [raised.exception])
            self.assertTrue(offloader._stop.is_set())
            self.assertEqual(threads.thread.start_calls, 0)

        with self.subTest(failure="thread start"):
            errors = []
            reader = _FakeReader(((1, 2),))
            threads = _ThreadFactory(start_error=RuntimeError("synthetic thread start failure"))
            offloader = self._owned(lambda: reader, errors, threads)
            with self.assertRaisesRegex(RuntimeError, "synthetic thread start failure") as raised:
                offloader.start()
            self.assertEqual(errors, [raised.exception])
            self.assertEqual(reader.close_calls, 1)

    def test_late_read_failure_cancels_reports_once_retains_first_error_and_closes_once(self) -> None:
        late_error = RuntimeError("synthetic late sample failure")
        reader = _FakeReader(
            ((200, 1000), late_error),
            close_error=RuntimeError("synthetic secondary close failure"),
        )
        errors: list[BaseException] = []
        threads = _ThreadFactory()
        offloader = self._owned(lambda: reader, errors, threads)
        offloader._run_loop = offloader._read_required_system_vram

        offloader.start()
        threads.thread.finish()

        self.assertTrue(offloader._stop.is_set())
        self.assertEqual(errors, [late_error])
        self.assertIs(offloader._system_vram_failure, late_error)
        self.assertEqual(reader.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "required whole-card VRAM monitoring failed") as stopped:
            offloader.stop()
        self.assertIs(stopped.exception.__cause__, late_error)
        self.assertEqual(reader.close_calls, 1)

    def test_close_failure_is_reported_and_is_not_retried_from_stop(self) -> None:
        close_error = RuntimeError("synthetic close failure")
        reader = _FakeReader(((200, 1000),), close_error=close_error)
        errors: list[BaseException] = []
        threads = _ThreadFactory(run_on_start=True)
        offloader = self._owned(lambda: reader, errors, threads)
        offloader._run_loop = lambda: None

        offloader.start()

        self.assertEqual(errors, [close_error])
        self.assertTrue(offloader._stop.is_set())
        self.assertEqual(reader.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "required whole-card VRAM monitoring failed") as stopped:
            offloader.stop()
        self.assertIs(stopped.exception.__cause__, close_error)
        with self.assertRaisesRegex(RuntimeError, "required whole-card VRAM monitoring failed"):
            offloader.stop()
        self.assertEqual(reader.close_calls, 1)

    def test_stop_refuses_unsafe_cleanup_while_owned_thread_is_still_alive(self) -> None:
        reader = _FakeReader(((200, 1000),))
        errors: list[BaseException] = []
        threads = _ThreadFactory()
        offloader = self._owned(lambda: reader, errors, threads)
        offloader._run_loop = lambda: None

        offloader.start()
        with self.assertRaisesRegex(RuntimeError, "reader remains thread-owned"):
            offloader.stop()
        self.assertEqual(threads.thread.join_timeouts, [5.0])
        self.assertTrue(threads.thread.is_alive())
        self.assertEqual(reader.close_calls, 0)

        threads.thread.finish()
        self.assertEqual(reader.close_calls, 1)
        offloader.stop()
        self.assertEqual(reader.close_calls, 1)
        self.assertEqual(errors, [])

    def test_owned_start_stop_edge_cases_are_bounded_and_do_not_double_close(self) -> None:
        with self.subTest(case="stop before start"):
            factory_calls: list[object] = []
            errors: list[BaseException] = []
            threads = _ThreadFactory()
            offloader = self._owned(
                lambda: factory_calls.append(object()), errors, threads
            )
            offloader.stop()
            offloader.stop()
            self.assertEqual(factory_calls, [])
            self.assertEqual(threads.thread.join_timeouts, [])
            with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                offloader.start()

        with self.subTest(case="double start and normal idempotent stop"):
            reader = _FakeReader(((200, 1000),))
            errors = []
            threads = _ThreadFactory()
            offloader = self._owned(lambda: reader, errors, threads)
            offloader._run_loop = lambda: None
            offloader.start()
            with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                offloader.start()
            threads.thread.finish()
            offloader.stop()
            offloader.stop()
            self.assertEqual(reader.close_calls, 1)
            self.assertEqual(errors, [])

    def test_borrowed_reader_is_never_closed_by_the_new_owned_reader_lifecycle(self) -> None:
        reader = _FakeReader(((200, 1000),))
        threads = _ThreadFactory(run_on_start=True)
        offloader = self._construct(
            threads,
            system_vram_startup_budget=1,
            system_vram_reader=reader,
        )
        offloader._run_loop = lambda: offloader._system_vram_reader()

        offloader.start()
        offloader.stop()

        self.assertEqual(reader.calls, 1)
        self.assertEqual(reader.close_calls, 0)
        self.assertEqual(threads.thread.join_timeouts, [5.0])

    def test_owned_configuration_conflicts_are_rejected_before_reader_or_thread_use(self) -> None:
        base = dict(
            system_vram_startup_budget=1,
            system_vram_reader_factory=lambda: _FakeReader(((1, 2),)),
            on_system_vram_error=lambda _error: None,
        )
        cases = (
            ("factory must be callable", {"system_vram_reader_factory": object()}),
            ("error callback required", {"on_system_vram_error": None}),
            ("budget required", {"system_vram_startup_budget": None}),
            ("borrowed reader conflicts", {"system_vram_reader": lambda: (1, 2)}),
        )

        for name, changes in cases:
            with self.subTest(configuration=name):
                values = dict(base)
                values.update(changes)
                with self.assertRaisesRegex(ValueError, "owned global reader requires"):
                    self._construct(_ThreadFactory(), **values)


class PipelineOwnedReaderSourceTests(unittest.TestCase):
    def test_windows_amd_worker_identity_is_the_only_pipeline_owned_reader_injection(self) -> None:
        source_path = PRODUCT_ROOT / "jasna" / "pipeline.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        pipeline = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pipeline"
        )
        run_pass = next(
            node
            for node in pipeline.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_pass"
        )
        owned_if = next(
            node
            for node in run_pass.body
            if isinstance(node, ast.If)
            and "JASNA_WINDOWS_WORKER_GPU_IDENTITY" in ast.unparse(node.test)
        )
        condition = ast.unparse(owned_if.test)
        self.assertIn("os.name == 'nt'", condition)
        self.assertIn("pipeline_vendor is AcceleratorVendor.AMD", condition)
        self.assertIn("JASNA_WINDOWS_WORKER_GPU_IDENTITY", condition)
        self.assertIn("== '1'", condition)

        imports = [node for node in owned_if.body if isinstance(node, ast.ImportFrom)]
        self.assertTrue(
            any(
                node.module == "jasna.windows_global_vram"
                and any(alias.name == "create_windows_hip_vram_reader" for alias in node.names)
                for node in imports
            )
        )
        options_assignment = next(
            node
            for node in owned_if.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "owned_vram_options"
                for target in node.targets
            )
        )
        self.assertIsInstance(options_assignment.value, ast.Call)
        self.assertIsInstance(options_assignment.value.func, ast.Name)
        self.assertEqual(options_assignment.value.func.id, "dict")
        options = {keyword.arg: keyword.value for keyword in options_assignment.value.keywords}
        self.assertEqual(
            set(options), {"system_vram_reader_factory", "on_system_vram_error"}
        )
        self.assertIsInstance(options["system_vram_reader_factory"], ast.Lambda)
        self.assertIsInstance(options["on_system_vram_error"], ast.Name)
        self.assertEqual(options["on_system_vram_error"].id, "required_vram_failure")
        callback_definition = next(
            node
            for node in owned_if.body
            if isinstance(node, ast.FunctionDef) and node.name == "required_vram_failure"
        )
        callback_module = ast.Module(body=[callback_definition], type_ignores=[])
        ast.fix_missing_locations(callback_module)

        class FakeCancelEvent:
            def __init__(self, initially_set: bool = False) -> None:
                self._set = initially_set
                self.set_calls = 0

            def is_set(self) -> bool:
                return self._set

            def set(self) -> None:
                self.set_calls += 1
                self._set = True

        class FakeLog:
            def __init__(self, cancel_event: FakeCancelEvent, error: BaseException | None = None) -> None:
                self._cancel_event = cancel_event
                self._error = error
                self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def error(self, *args: object, **kwargs: object) -> None:
                self.calls.append((args, kwargs))
                if not self._cancel_event.is_set():
                    raise AssertionError("callback logged before cancellation")
                if self._error is not None:
                    raise self._error

        first_error = RuntimeError("first worker failure")
        already_cancelled = FakeCancelEvent(initially_set=True)
        first_holder: list[BaseException] = []
        first_namespace = {
            "error_holder": first_holder,
            "self": types.SimpleNamespace(_cancel_event=already_cancelled),
            "log": FakeLog(already_cancelled),
        }
        exec(compile(callback_module, str(source_path), "exec"), first_namespace)
        first_namespace["required_vram_failure"](first_error)
        self.assertEqual(first_holder, [first_error])
        self.assertEqual(already_cancelled.set_calls, 1)

        logger_error = RuntimeError("synthetic logging failure")
        second_error = RuntimeError("later worker failure")
        fresh_cancel = FakeCancelEvent()
        retaining_holder: list[BaseException] = [first_error]
        raising_log = FakeLog(fresh_cancel, logger_error)
        second_namespace = {
            "error_holder": retaining_holder,
            "self": types.SimpleNamespace(_cancel_event=fresh_cancel),
            "log": raising_log,
        }
        exec(compile(callback_module, str(source_path), "exec"), second_namespace)
        with self.assertRaisesRegex(RuntimeError, "synthetic logging failure"):
            second_namespace["required_vram_failure"](second_error)
        self.assertEqual(retaining_holder, [first_error])
        self.assertTrue(fresh_cancel.is_set())
        self.assertEqual(fresh_cancel.set_calls, 1)
        self.assertEqual(len(raising_log.calls), 1)

        offloader_call = next(
            node
            for node in ast.walk(run_pass)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VramOffloader"
        )
        self.assertTrue(
            any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "owned_vram_options"
                for keyword in offloader_call.keywords
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

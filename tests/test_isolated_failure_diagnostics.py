"""CPU-only checks for isolated-worker terminal failure diagnostics.

The product module is parsed and the exact terminal formatter/tail are compiled
into a minimal namespace.  This deliberately never imports Processor, Torch,
the GUI, media libraries, or launches a child process.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import subprocess
import unittest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PRODUCT_ROOT / "jasna" / "gui" / "processor.py"


def _processor_tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _processor_method(tree: ast.Module, name: str) -> ast.FunctionDef:
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Processor"
    )
    return next(
        node
        for node in processor.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_terminal_logic() -> tuple[object, object]:
    tree = _processor_tree()
    selected: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_ISOLATED_FAILURE_DETAIL_MAX_CHARS"
                for target in node.targets
            )
        ):
            selected.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_bounded_isolated_failure_detail",
            "_isolated_video_job_terminal_failure_message",
        }:
            selected.append(copy.deepcopy(node))
    if len(selected) != 3:
        raise AssertionError("isolated failure formatter source is incomplete")

    namespace: dict[str, object] = {}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)

    method = _processor_method(tree, "_process_isolated_video_job")
    terminal_if = next(
        node
        for node in reversed(method.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and "result_event" in ast.unparse(node.test)
        and "_stop_event.is_set" in ast.unparse(node.test)
    )
    wrapper = ast.parse(
        "def execute_terminal_tail(self, job, result_event, protocol_error, returncode):\n"
        "    pass\n"
    ).body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = [copy.deepcopy(terminal_if)]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return (
        namespace["_isolated_video_job_terminal_failure_message"],
        namespace["execute_terminal_tail"],
    )


class _StopEvent:
    def __init__(self, stopped: bool = False) -> None:
        self.stopped = stopped

    def is_set(self) -> bool:
        return self.stopped


class _TerminalHarness:
    def __init__(self, *, stopped: bool = False) -> None:
        self._stop_event = _StopEvent(stopped)
        self.marked: list[object] = []
        self.failures: list[tuple[object, str]] = []

    def _mark_stopped(self, job: object) -> None:
        self.marked.append(job)

    def _fail_isolated_video_job(self, job: object, message: str) -> None:
        self.failures.append((job, message))


class IsolatedFailureDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.format_failure, cls.execute_tail = _load_terminal_logic()

    def _tail(
        self,
        *,
        stopped: bool = False,
        result_event: object = None,
        protocol_error: str | None = None,
        returncode: int | None = 1,
    ) -> tuple[_TerminalHarness, object]:
        harness = _TerminalHarness(stopped=stopped)
        job = object()
        type(self).execute_tail(
            harness,
            job,
            result_event,
            protocol_error,
            returncode,
        )
        return harness, job

    def test_nonzero_exit_preserves_bounded_normalized_backend_detail(self) -> None:
        harness, job = self._tail(
            protocol_error="GuardVerificationError: report\ncontains\tcontrols\x00",
            returncode=1,
        )
        self.assertEqual(harness.marked, [])
        self.assertEqual(
            harness.failures,
            [
                (
                    job,
                    "isolated video job exited with code 1: "
                    "GuardVerificationError: report contains controls",
                )
            ],
        )

        message = type(self).format_failure(86, "x" * 3000)
        prefix = "isolated video job exited with code 86: "
        self.assertEqual(message, prefix + "x" * 2045 + "...")
        self.assertLessEqual(len(message) - len(prefix), 2048)

    def test_nonzero_exit_without_detail_keeps_the_exact_legacy_message(self) -> None:
        expected = "isolated video job exited with code 1"
        for detail in (None, "", " \n\t\x00 "):
            with self.subTest(detail=repr(detail)):
                harness, job = self._tail(protocol_error=detail, returncode=1)
                self.assertEqual(harness.marked, [])
                self.assertEqual(harness.failures, [(job, expected)])

    def test_zero_exit_with_protocol_detail_remains_a_protocol_failure(self) -> None:
        harness, job = self._tail(
            protocol_error="event\ncontains\tinvalid JSON",
            returncode=0,
        )
        self.assertEqual(harness.marked, [])
        self.assertEqual(
            harness.failures,
            [(job, "invalid isolated video job protocol: event contains invalid JSON")],
        )
        self.assertIsNone(type(self).format_failure(0, None))

    def test_stop_precedence_skips_terminal_failure_dispatch(self) -> None:
        harness, job = self._tail(
            stopped=True,
            protocol_error="guard runtime failure",
            returncode=1,
        )
        self.assertEqual(harness.marked, [job])
        self.assertEqual(harness.failures, [])

    def test_source_keeps_retry_paths_before_the_terminal_failure_tail(self) -> None:
        tree = _processor_tree()
        method = _processor_method(tree, "_process_isolated_video_job")
        retry_names = {
            "NATIVE_PRESSURE_RECYCLE_EXIT_CODE",
            "NATIVE_OPEN_STALL_EXIT_CODE",
        }
        retry_ifs = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and any(
                isinstance(candidate, ast.Name) and candidate.id in retry_names
                for candidate in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(retry_ifs), 2)
        self.assertTrue(
            all(any(isinstance(node, ast.Continue) for node in ast.walk(branch)) for branch in retry_ifs)
        )

        helper_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_isolated_video_job_terminal_failure_message"
        ]
        self.assertEqual(len(helper_calls), 1)
        self.assertGreater(
            helper_calls[0].lineno,
            max(branch.end_lineno or branch.lineno for branch in retry_ifs),
        )

    def test_cpu_source_execution_did_not_import_forbidden_runtime_modules(self) -> None:
        # Other collected tests legitimately import these modules. Prove this
        # source loader's contract in a fresh interpreter instead of depending
        # on collection order or the process-global import cache.
        script = (
            "import runpy, sys\n"
            f"module = runpy.run_path({str(Path(__file__).resolve())!r})\n"
            "module['_load_terminal_logic']()\n"
            "assert not {'torch', 'av', 'tkinter', 'customtkinter'} & sys.modules.keys()\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

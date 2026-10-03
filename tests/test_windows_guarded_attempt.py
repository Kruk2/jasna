"""CPU-only contract tests for the opt-in Windows guarded attempt backend.

These tests never launch the native guard, a child process, a GUI, or media
code.  They drive the production runner with fake binary pipes and the shared
stream-record interface, while the Processor seam is checked from source so
the test does not import the GUI processor and its runtime dependencies.
"""

from __future__ import annotations

import ast
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from jasna.gui import windows_guarded_attempt as guarded_attempt
from jasna.gui.isolated_worker_streams import StreamRecord


EVENT_PREFIX = "JASNA_JOB_EVENT\t"


class _Digest:
    def hexdigest(self) -> str:
        return guarded_attempt.GUARD_SHA256


class _NoPidKillOs:
    """A narrow stand-in which makes accidental PID-based cleanup observable."""

    name = "nt"

    def __init__(self) -> None:
        self.kill_calls: list[tuple[object, ...]] = []

    def kill(self, *args: object) -> None:
        self.kill_calls.append(args)
        raise AssertionError("guard cleanup must use the owned Popen handle")


class _FakeBinaryPipe:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.flush_count = 0
        self.closed = False
        self.write_seen = threading.Event()
        self.flush_seen = threading.Event()

    def write(self, value: bytes) -> int:
        if self.closed:
            raise ValueError("pipe is closed")
        if not isinstance(value, bytes):
            raise AssertionError("guard pipes must receive binary writes")
        self.writes.append(value)
        self.write_seen.set()
        return len(value)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("pipe is closed")
        self.flush_count += 1
        self.flush_seen.set()

    def close(self) -> None:
        self.closed = True


class _BlockingBinaryPipe(_FakeBinaryPipe):
    """Holds the single writer inside one binary write until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.write_entered = threading.Event()
        self.release_write = threading.Event()

    def write(self, value: bytes) -> int:
        self.write_entered.set()
        if not self.release_write.wait(1):
            raise RuntimeError("test did not release the blocked writer")
        return super().write(value)


class _FakeProcess:
    _next_pid = 7300

    def __init__(
        self,
        *,
        returncode: int = 0,
        running: bool = False,
        wait_failures: tuple[Exception, ...] = (),
        kill_error: Exception | None = None,
    ) -> None:
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self.stdin = _FakeBinaryPipe()
        self.stdout = _FakeBinaryPipe()
        self.stderr = _FakeBinaryPipe()
        self._normal_returncode = returncode
        self.returncode: int | None = None if running else returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []
        self._wait_failures = deque(wait_failures)
        self._kill_error = kill_error

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._wait_failures:
            raise self._wait_failures.popleft()
        if self.returncode is None:
            self.returncode = self._normal_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        if self._kill_error is not None:
            raise self._kill_error
        self.returncode = -9


class _ScriptedStreams:
    """In-memory replacement for the already-tested binary stream transport."""

    def __init__(
        self,
        records: list[StreamRecord] | tuple[StreamRecord, ...] = (),
        *,
        terminal_error: bool = False,
        dropped_stdout: int = 0,
        dropped_stderr: int = 0,
        finished_when_drained: bool = True,
        join_result: bool = True,
        join_error: Exception | None = None,
    ) -> None:
        self._records = deque(records)
        self._terminal_error = terminal_error
        self._dropped_stdout = dropped_stdout
        self._dropped_stderr = dropped_stderr
        self._finished_when_drained = finished_when_drained
        self._join_result = join_result
        self._join_error = join_error
        self.started = False
        self.next_timeouts: list[float] = []
        self.join_timeouts: list[float] = []

    def start(self) -> None:
        self.started = True

    def next_record(self, timeout: float) -> StreamRecord | None:
        self.next_timeouts.append(timeout)
        if self._records:
            return self._records.popleft()
        return None

    @property
    def has_terminal_error(self) -> bool:
        return self._terminal_error

    @property
    def finished(self) -> bool:
        return self._finished_when_drained and not self._records

    @property
    def dropped_records(self) -> dict[str, int]:
        return {
            "stdout": self._dropped_stdout,
            "stderr": self._dropped_stderr,
        }

    def join(self, timeout: float) -> bool:
        self.join_timeouts.append(timeout)
        if self._join_error is not None:
            raise self._join_error
        return self._join_result


class _RunnerHarness:
    def __init__(self, os_probe: _NoPidKillOs) -> None:
        self.os_probe = os_probe
        self.popen_calls: list[tuple[list[str], dict[str, object]]] = []
        self.verification_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.parser_calls: list[str] = []


class _GuardedAttemptTestCase(unittest.TestCase):
    def _config(self, *, timeout_seconds: int = 5) -> guarded_attempt.GuardedAttemptConfig:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        guard_path = root / "GuardedRunner.exe"
        guard_path.write_bytes(b"not-a-native-guard")
        return guarded_attempt.GuardedAttemptConfig(
            guard_path=guard_path,
            working_directory=root,
            timeout_seconds=timeout_seconds,
        )

    @contextmanager
    def _patched_runner(
        self,
        *,
        process_factory,
        streams_factory,
        verified_exit: int = 0,
        report_bytes: bytes = b'{"report":"fake"}',
        parser=None,
    ):
        harness = _RunnerHarness(_NoPidKillOs())

        def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
            command_copy = list(command)
            harness.popen_calls.append((command_copy, dict(kwargs)))
            report_path = Path(command_copy[command_copy.index("--report") + 1])
            report_path.write_bytes(report_bytes)
            return process_factory(command_copy, kwargs)

        def fake_verify(*args: object, **kwargs: object) -> int:
            harness.verification_calls.append((args, dict(kwargs)))
            return verified_exit

        if parser is None:
            def parser(line: str):
                harness.parser_calls.append(line)
                if not line.startswith(EVENT_PREFIX):
                    return None
                return json.loads(line[len(EVENT_PREFIX) :])

        protocol_module = types.ModuleType("jasna.gui.video_job_process")
        protocol_module.parse_event_line = parser
        create_no_window = getattr(guarded_attempt.subprocess, "CREATE_NO_WINDOW", 0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(guarded_attempt, "os", harness.os_probe))
            stack.enter_context(
                patch.object(guarded_attempt.hashlib, "file_digest", return_value=_Digest())
            )
            stack.enter_context(
                patch.object(guarded_attempt.subprocess, "Popen", side_effect=fake_popen)
            )
            stack.enter_context(
                patch.object(
                    guarded_attempt.subprocess,
                    "CREATE_NO_WINDOW",
                    create_no_window,
                    create=True,
                )
            )
            stack.enter_context(
                patch.object(
                    guarded_attempt,
                    "IsolatedWorkerStreams",
                    side_effect=streams_factory,
                )
            )
            stack.enter_context(
                patch.object(
                    guarded_attempt,
                    "verify_guard_exit",
                    side_effect=fake_verify,
                )
            )
            stack.enter_context(
                patch.dict(sys.modules, {"jasna.gui.video_job_process": protocol_module})
            )
            yield harness

    @staticmethod
    def _callbacks(
        *,
        on_event=None,
        on_log=None,
        on_started=None,
        on_finished=None,
    ) -> dict[str, object]:
        return {
            "on_event": on_event or (lambda _event: False),
            "on_log": on_log or (lambda _level, _message: None),
            "on_started": on_started or (lambda _process: None),
            "on_finished": on_finished or (lambda _process: None),
        }


class GuardedAttemptConfigurationTests(_GuardedAttemptTestCase):
    def test_config_is_frozen_and_timeout_is_limited_to_one_through_180_seconds(self) -> None:
        config = self._config(timeout_seconds=1)

        self.assertEqual(config.timeout_seconds, 1)
        with self.assertRaises(FrozenInstanceError):
            config.timeout_seconds = 2  # type: ignore[misc]

        for value in (0, 181, True, 1.0):
            with self.subTest(timeout_seconds=value):
                with self.assertRaises(ValueError):
                    guarded_attempt.GuardedAttemptConfig(
                        guard_path=config.guard_path,
                        working_directory=config.working_directory,
                        timeout_seconds=value,  # type: ignore[arg-type]
                    )

    def test_config_requires_explicit_absolute_path_instances(self) -> None:
        config = self._config()
        with self.assertRaises(ValueError):
            guarded_attempt.GuardedAttemptConfig(
                guard_path=Path("relative-guard.exe"),
                working_directory=config.working_directory,
            )
        with self.assertRaises(ValueError):
            guarded_attempt.GuardedAttemptConfig(
                guard_path=config.guard_path,
                working_directory=Path("relative-working-directory"),
            )


class TextCommandPipeTests(unittest.TestCase):
    def test_text_commands_become_one_utf8_binary_line_and_stay_bounded(self) -> None:
        raw_pipe = _FakeBinaryPipe()
        command_pipe = guarded_attempt._TextCommandPipe(raw_pipe)
        payload = '{"command":"echo","message":"雪"}\n'
        try:
            self.assertEqual(command_pipe.write(payload), len(payload))
            command_pipe.flush()
            self.assertTrue(raw_pipe.write_seen.wait(1), "writer did not receive queued text")
            self.assertTrue(raw_pipe.flush_seen.wait(1), "writer did not flush queued text")
            self.assertEqual(raw_pipe.writes, [payload.encode("utf-8")])
            self.assertEqual(raw_pipe.flush_count, 1)

            with self.assertRaises(TypeError):
                command_pipe.write(b"not-text\n")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                command_pipe.write("no newline")
            with self.assertRaises(ValueError):
                command_pipe.write("two\nlines\n")
            with self.assertRaises(ValueError):
                command_pipe.write("x" * guarded_attempt.MAX_COMMAND_BYTES + "\n")
        finally:
            command_pipe.close()

    def test_full_command_queue_fails_closed_without_blocking_the_gui_writer(self) -> None:
        raw_pipe = _BlockingBinaryPipe()
        command_pipe = guarded_attempt._TextCommandPipe(raw_pipe)
        payload = '{"command":"set_paused","paused":true}\n'
        try:
            command_pipe.write(payload)
            self.assertTrue(raw_pipe.write_entered.wait(1), "writer did not enter raw write")
            for _ in range(guarded_attempt.MAX_PENDING_COMMANDS):
                command_pipe.write(payload)
            with self.assertRaisesRegex(OSError, "queue overflow"):
                command_pipe.write(payload)
            with self.assertRaisesRegex(OSError, "queue overflow"):
                command_pipe.flush()
        finally:
            raw_pipe.release_write.set()
            command_pipe.close()


class WindowsGuardedAttemptRunTests(_GuardedAttemptTestCase):
    def test_launch_uses_fixed_guard_caps_and_a_unique_report_per_attempt(self) -> None:
        config = self._config(timeout_seconds=17)
        attempt = guarded_attempt.WindowsGuardedAttempt(config)
        processes = deque((_FakeProcess(returncode=1), _FakeProcess(returncode=1)))
        streams = deque(
            (
                _ScriptedStreams(
                    [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"run":1}')]
                ),
                _ScriptedStreams(
                    [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"run":2}')]
                ),
            )
        )
        started: list[object] = []
        finished: list[object] = []
        command = ["C:\\Python\\python.exe", "-I", "-B", "worker.py"]
        environment = {"PYTHONUNBUFFERED": "1", "JASNA_TEST": "guarded"}

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: processes.popleft(),
            streams_factory=lambda _stdout, _stderr: streams.popleft(),
            verified_exit=75,
            report_bytes=b'{"report":"bounded"}',
        ) as harness:
            first = attempt.run(
                command,
                environment,
                **self._callbacks(on_started=started.append, on_finished=finished.append),
            )
            second = attempt.run(
                command,
                environment,
                **self._callbacks(on_started=started.append, on_finished=finished.append),
            )

        self.assertEqual(first, (None, None, 75))
        self.assertEqual(second, (None, None, 75))
        self.assertEqual(len(harness.popen_calls), 2)
        self.assertEqual(len(harness.verification_calls), 2)
        self.assertEqual(started, finished)

        report_paths: list[str] = []
        for index, (guard_command, kwargs) in enumerate(harness.popen_calls, start=1):
            with self.subTest(attempt=index):
                self.assertEqual(guard_command[0], str(config.guard_path))
                self.assertEqual(
                    guard_command[guard_command.index("--") + 1 :], command
                )
                expected_options = {
                    "--job-memory-mib": "6144",
                    "--host-commit-reserve-mib": "10240",
                    "--host-physical-reserve-mib": "8192",
                    "--timeout-seconds": "17",
                    "--poll-ms": "1000",
                    "--active-process-limit": "16",
                    "--working-directory": str(config.working_directory),
                }
                for option, expected in expected_options.items():
                    self.assertEqual(guard_command[guard_command.index(option) + 1], expected)
                report_paths.append(guard_command[guard_command.index("--report") + 1])
                self.assertEqual(kwargs["cwd"], config.working_directory)
                # Per-attempt identity tokens must not mutate/reuse caller env.
                self.assertEqual(kwargs["env"], environment)
                self.assertIsNot(kwargs["env"], environment)
                self.assertEqual(kwargs["stdin"], guarded_attempt.subprocess.PIPE)
                self.assertEqual(kwargs["stdout"], guarded_attempt.subprocess.PIPE)
                self.assertEqual(kwargs["stderr"], guarded_attempt.subprocess.PIPE)
                self.assertEqual(kwargs["bufsize"], 0)

        self.assertEqual(len(set(report_paths)), 2)
        self.assertTrue(all(not Path(path).exists() for path in report_paths))
        for (arguments, keywords), expected_record in zip(
            harness.verification_calls, (b'{"run":1}', b'{"run":2}')
        ):
            self.assertEqual(arguments[:2], (b'{"report":"bounded"}', expected_record))
            self.assertEqual(keywords["expected_command"], command)
            self.assertEqual(
                keywords["expected_working_directory"], str(config.working_directory)
            )
            self.assertEqual(keywords["actual_outer_exit_code"], 1)

    def test_stdout_uses_the_shared_parser_and_stderr_never_enters_that_parser(self) -> None:
        config = self._config()
        process = _FakeProcess(returncode=0)
        event = {"type": "result", "status": "completed"}
        stdout_native = "ordinary worker text"
        stdout_event = EVENT_PREFIX + json.dumps(event, separators=(",", ":"))
        stderr_event_lookalike = EVENT_PREFIX + "not-json"
        streams = _ScriptedStreams(
            [
                StreamRecord("stdout", stdout_native),
                StreamRecord("stdout", stdout_event),
                StreamRecord("stderr", stderr_event_lookalike),
                StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":true}'),
            ]
        )
        applied_events: list[dict[str, object]] = []
        logs: list[tuple[str, str]] = []

        def on_event(value: dict[str, object]) -> dict[str, object] | bool:
            applied_events.append(value)
            return value if value["type"] == "result" else False

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
            report_bytes=b'{"report":"ok"}',
        ) as harness:
            result = guarded_attempt.WindowsGuardedAttempt(config).run(
                ["child.exe"],
                {},
                **self._callbacks(on_event=on_event, on_log=lambda *item: logs.append(item)),
            )

        self.assertEqual(result, (event, None, 0))
        self.assertEqual(harness.parser_calls, [stdout_native, stdout_event])
        self.assertEqual(applied_events, [event])
        self.assertIn(("WARNING", "[video worker] " + stdout_native), logs)
        self.assertIn(
            ("WARNING", "[video worker stderr] " + stderr_event_lookalike), logs
        )
        self.assertEqual(harness.verification_calls[0][0][1], b'{"guard":true}')

    def test_transport_loss_or_duplicate_terminal_protocol_never_succeeds(self) -> None:
        cases = (
            (
                "terminal transport error",
                _FakeProcess(returncode=0, running=True),
                _ScriptedStreams(terminal_error=True),
                lambda _event: False,
                "guarded worker communication failed",
            ),
            (
                "lost stdout record",
                _FakeProcess(returncode=0),
                _ScriptedStreams(
                    [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":true}')],
                    dropped_stdout=1,
                ),
                lambda _event: False,
                "worker protocol was lost",
            ),
            (
                "duplicate terminal event",
                _FakeProcess(returncode=0, running=True),
                _ScriptedStreams(
                    [
                        StreamRecord("stdout", EVENT_PREFIX + '{"type":"result"}'),
                        StreamRecord("stdout", EVENT_PREFIX + '{"type":"retry"}'),
                    ]
                ),
                lambda event: event,
                "multiple terminal events",
            ),
            (
                "duplicate guard result",
                _FakeProcess(returncode=0, running=True),
                _ScriptedStreams(
                    [
                        StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":1}'),
                        StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":2}'),
                    ]
                ),
                lambda _event: False,
                "duplicate guard result",
            ),
        )

        for name, process, streams, on_event, expected_error in cases:
            with self.subTest(name=name):
                config = self._config()
                with self._patched_runner(
                    process_factory=lambda _command, _kwargs: process,
                    streams_factory=lambda _stdout, _stderr: streams,
                ) as harness:
                    terminal, error, exit_code = guarded_attempt.WindowsGuardedAttempt(config).run(
                        ["child.exe"],
                        {},
                        **self._callbacks(on_event=on_event),
                    )

                self.assertIsNone(terminal)
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_error, error or "")
                self.assertEqual(harness.verification_calls, [])
                self.assertEqual(streams.join_timeouts, [1])

    def test_callback_exceptions_fail_closed_after_cleaning_the_exact_guard_process(self) -> None:
        cases = (
            (
                "on_started",
                _ScriptedStreams(),
                lambda: self._callbacks(
                    on_started=lambda _process: (_ for _ in ()).throw(RuntimeError("start boom"))
                ),
                "start boom",
            ),
            (
                "on_event",
                _ScriptedStreams(
                    [StreamRecord("stdout", EVENT_PREFIX + '{"type":"progress"}')]
                ),
                lambda: self._callbacks(
                    on_event=lambda _event: (_ for _ in ()).throw(RuntimeError("event boom"))
                ),
                "event boom",
            ),
        )

        for name, streams, callbacks_factory, expected_error in cases:
            with self.subTest(callback=name):
                process = _FakeProcess(returncode=0, running=True)
                finished: list[object] = []
                callbacks = callbacks_factory()
                callbacks["on_finished"] = finished.append
                with self._patched_runner(
                    process_factory=lambda _command, _kwargs: process,
                    streams_factory=lambda _stdout, _stderr: streams,
                ) as harness:
                    terminal, error, exit_code = guarded_attempt.WindowsGuardedAttempt(
                        self._config()
                    ).run(["child.exe"], {}, **callbacks)

                self.assertIsNone(terminal)
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_error, error or "")
                self.assertEqual(process.terminate_calls, 1)
                self.assertEqual(process.kill_calls, 0)
                self.assertEqual(harness.os_probe.kill_calls, [])
                self.assertEqual(len(finished), 1)
                self.assertTrue(process.stdin.closed)
                self.assertTrue(process.stdout.closed)
                self.assertTrue(process.stderr.closed)
                self.assertEqual(streams.join_timeouts, [1])

    def test_on_finished_runs_after_pipe_and_stream_cleanup_and_its_failure_is_fail_closed(self) -> None:
        config = self._config()
        process = _FakeProcess(returncode=0)
        streams = _ScriptedStreams(
            [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":true}')]
        )
        observed: list[object] = []

        def on_finished(handle: object) -> None:
            observed.append(handle)
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
            self.assertEqual(streams.join_timeouts, [1])
            raise RuntimeError("finished boom")

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ):
            terminal, error, exit_code = guarded_attempt.WindowsGuardedAttempt(config).run(
                ["child.exe"],
                {},
                **self._callbacks(on_finished=on_finished),
            )

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("parent deregistration failed: RuntimeError", error or "")
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].pid, process.pid)

    def test_timeout_terminates_only_the_owned_guard_handle_without_pid_killing(self) -> None:
        config = self._config(timeout_seconds=1)
        process = _FakeProcess(returncode=0, running=True)
        streams = _ScriptedStreams(finished_when_drained=False)
        finished: list[object] = []

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ) as harness:
            with patch.object(guarded_attempt.time, "monotonic", side_effect=(0.0, 17.0)):
                terminal, error, exit_code = guarded_attempt.WindowsGuardedAttempt(config).run(
                    ["child.exe"],
                    {},
                    **self._callbacks(on_finished=finished.append),
                )

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("bounded attempt deadline", error or "")
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertEqual(harness.os_probe.kill_calls, [])
        self.assertEqual(process.wait_timeouts, [10])
        self.assertEqual(len(finished), 1)

    def test_failed_terminate_wait_uses_exact_handle_kill_then_allows_a_clean_retry(self) -> None:
        config = self._config()
        process = _FakeProcess(
            returncode=0,
            running=True,
            wait_failures=(
                guarded_attempt.subprocess.TimeoutExpired(["fake-guard"], 10),
            ),
        )
        streams = _ScriptedStreams()
        attempt = guarded_attempt.WindowsGuardedAttempt(config)
        finished: list[object] = []

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ) as harness:
            terminal, error, exit_code = attempt.run(
                ["child.exe"],
                {},
                **self._callbacks(
                    on_started=lambda _handle: (_ for _ in ()).throw(
                        RuntimeError("force cleanup")
                    ),
                    on_finished=finished.append,
                ),
            )

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("force cleanup", error or "")
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.wait_timeouts, [10, 5])
        self.assertEqual(harness.os_probe.kill_calls, [])
        self.assertFalse(attempt._retirement_failed)
        self.assertEqual(len(finished), 1)

    def test_unretired_guard_poisoning_refuses_every_later_attempt(self) -> None:
        config = self._config()
        process = _FakeProcess(
            returncode=0,
            running=True,
            wait_failures=(
                guarded_attempt.subprocess.TimeoutExpired(["fake-guard"], 10),
            ),
            kill_error=OSError("exact handle kill failed"),
        )
        streams = _ScriptedStreams()
        attempt = guarded_attempt.WindowsGuardedAttempt(config)

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ) as harness:
            terminal, error, exit_code = attempt.run(
                ["child.exe"],
                {},
                **self._callbacks(
                    on_started=lambda _handle: (_ for _ in ()).throw(
                        RuntimeError("force cleanup")
                    )
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "did not retire cleanly"):
                attempt.run(["later-child.exe"], {}, **self._callbacks())

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("guard cleanup failed: OSError", error or "")
        self.assertTrue(attempt._retirement_failed)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(harness.os_probe.kill_calls, [])
        self.assertEqual(len(harness.popen_calls), 1)

    def test_stream_join_exception_still_deregisters_then_poison_refuses_a_later_attempt(self) -> None:
        config = self._config()
        process = _FakeProcess(returncode=0)
        streams = _ScriptedStreams(
            [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":true}')],
            join_error=RuntimeError("reader join failed"),
        )
        attempt = guarded_attempt.WindowsGuardedAttempt(config)
        finished: list[object] = []

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ) as harness:
            terminal, error, exit_code = attempt.run(
                ["child.exe"],
                {},
                **self._callbacks(on_finished=finished.append),
            )
            with self.assertRaisesRegex(RuntimeError, "did not retire cleanly"):
                attempt.run(["later-child.exe"], {}, **self._callbacks())

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("worker stream join failed: RuntimeError", error or "")
        self.assertTrue(attempt._retirement_failed)
        self.assertEqual(streams.join_timeouts, [1])
        self.assertEqual(len(finished), 1)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertEqual(len(harness.popen_calls), 1)

    def test_last_reader_join_failure_deregisters_and_poison_refuses_a_later_attempt(self) -> None:
        config = self._config()
        process = _FakeProcess(returncode=0)
        streams = _ScriptedStreams(
            [StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":true}')],
            join_result=False,
        )
        attempt = guarded_attempt.WindowsGuardedAttempt(config)
        finished: list[object] = []

        with self._patched_runner(
            process_factory=lambda _command, _kwargs: process,
            streams_factory=lambda _stdout, _stderr: streams,
        ) as harness:
            terminal, error, exit_code = attempt.run(
                ["child.exe"],
                {},
                **self._callbacks(on_finished=finished.append),
            )
            with self.assertRaisesRegex(RuntimeError, "did not retire cleanly"):
                attempt.run(["later-child.exe"], {}, **self._callbacks())

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("worker stream readers did not retire", error or "")
        self.assertTrue(attempt._retirement_failed)
        self.assertEqual(streams.join_timeouts, [1])
        self.assertEqual(len(finished), 1)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertEqual(len(harness.popen_calls), 1)


class SourceContractTests(unittest.TestCase):
    def _class_method(self, path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )

    def test_runner_uses_the_existing_video_job_event_parser_without_a_second_json_parser(self) -> None:
        source_path = PRODUCT_ROOT / "jasna" / "gui" / "windows_guarded_attempt.py"
        method = self._class_method(source_path, "WindowsGuardedAttempt", "_run")
        imports = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.ImportFrom)
            and node.module == "jasna.gui.video_job_process"
        ]
        self.assertTrue(
            any(any(alias.name == "parse_event_line" for alias in node.names) for node in imports)
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr == "loads"
                for node in ast.walk(method)
            )
        )

    def test_processor_backend_is_explicit_default_none_and_legacy_runner_remains_after_it(self) -> None:
        source_path = PRODUCT_ROOT / "jasna" / "gui" / "processor.py"
        init = self._class_method(source_path, "Processor", "__init__")
        keyword_defaults = dict(zip(init.args.kwonlyargs, init.args.kw_defaults))
        backend_default = keyword_defaults.get(
            next(
                argument
                for argument in init.args.kwonlyargs
                if argument.arg == "video_job_attempt_backend"
            )
        )
        self.assertIsInstance(backend_default, ast.Constant)
        self.assertIsNone(backend_default.value)

        method = self._class_method(source_path, "Processor", "_run_isolated_video_job_attempt")
        backend_assignment_index = next(
            index
            for index, node in enumerate(method.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "backend" for target in node.targets)
        )
        backend_branch_index = next(
            index
            for index, node in enumerate(
                method.body[backend_assignment_index + 1 :],
                backend_assignment_index + 1,
            )
            if isinstance(node, ast.If)
            and any(
                isinstance(candidate, ast.Name) and candidate.id == "backend"
                for candidate in ast.walk(node.test)
            )
        )
        backend_branch = method.body[backend_branch_index]
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "backend"
                and node.func.attr == "run"
                for node in ast.walk(backend_branch)
            )
        )

        legacy_nodes = method.body[backend_branch_index + 1 :]
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "jasna.gui.video_job_process"
                and any(alias.name == "parse_event_line" for alias in node.names)
                for node in legacy_nodes
            )
        )
        legacy_popen_calls = [
            node
            for node in ast.walk(ast.Module(body=legacy_nodes, type_ignores=[]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(legacy_popen_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in legacy_popen_calls[0].keywords}
        stderr = keywords.get("stderr")
        self.assertIsInstance(stderr, ast.Attribute)
        self.assertIsInstance(stderr.value, ast.Name)
        self.assertEqual(stderr.value.id, "subprocess")
        self.assertEqual(stderr.attr, "STDOUT")


if __name__ == "__main__":
    unittest.main()

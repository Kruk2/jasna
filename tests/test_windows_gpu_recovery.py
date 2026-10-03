"""CPU-only contracts for guarded Windows GPU recovery.

The tests deliberately use synthetic whole-card samples and fake guarded-child
transport.  They never initialize HIP, PDH, media libraries, a GUI, a native
guard, or a real subprocess.
"""

from __future__ import annotations

import ast
from collections import deque
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from jasna.gui import gpu_recovery
from jasna.gui import windows_guarded_attempt as guarded_attempt
from jasna.gui.isolated_worker_streams import StreamRecord
from jasna.windows_global_vram import WindowsGpuIdentity


EVENT_PREFIX = "JASNA_JOB_EVENT\t"
_TOKEN = "a" * 32
_MARKER = "luid_0x00000000_0x0000f7b0_phys_"


class _FakeStopEvent:
    """A no-sleep Event replacement with optionally scheduled cancellation."""

    def __init__(self, *, initially_set: bool = False, set_after_waits: int | None = None):
        self._set = initially_set
        self._set_after_waits = set_after_waits
        self.is_set_calls = 0
        self.wait_calls: list[float] = []

    def is_set(self) -> bool:
        self.is_set_calls += 1
        return self._set

    def wait(self, timeout: float) -> bool:
        self.wait_calls.append(timeout)
        if self._set_after_waits is not None and len(self.wait_calls) >= self._set_after_waits:
            self._set = True
        return self._set


class _SequenceReader:
    def __init__(self, samples):
        self._samples = deque(samples)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self._samples:
            raise AssertionError("recovery reader was called too many times")
        return self._samples.popleft()


class GpuRecoveryPolicyTests(unittest.TestCase):
    def test_requires_two_consecutive_sufficient_samples(self) -> None:
        reader = _SequenceReader(((700, 1000), (650, 1000)))
        event = _FakeStopEvent()
        logs: list[tuple[str, str]] = []

        with patch.object(gpu_recovery.time, "monotonic", side_effect=(0.0, 0.0)):
            recovered = gpu_recovery.wait_for_vram_recovery(
                reader,
                stop_event=event,
                on_log=lambda level, message: logs.append((level, message)),
                min_headroom_bytes=300,
                timeout_seconds=1,
                poll_seconds=.25,
                stable_samples=2,
            )

        self.assertTrue(recovered)
        self.assertEqual(reader.calls, 2)
        self.assertEqual(event.wait_calls, [.25])
        self.assertEqual(logs, [])

    def test_low_sample_resets_stability_before_two_later_high_samples(self) -> None:
        reader = _SequenceReader(
            ((700, 1000), (800, 1000), (700, 1000), (680, 1000))
        )
        event = _FakeStopEvent()

        with patch.object(
            gpu_recovery.time, "monotonic", side_effect=(0.0, 0.0, 0.0, 0.0)
        ):
            recovered = gpu_recovery.wait_for_vram_recovery(
                reader,
                stop_event=event,
                on_log=lambda _level, _message: None,
                min_headroom_bytes=300,
                timeout_seconds=1,
                poll_seconds=.25,
                stable_samples=2,
            )

        self.assertTrue(recovered)
        self.assertEqual(reader.calls, 4)
        self.assertEqual(event.wait_calls, [.25, .25, .25])

    def test_unavailable_telemetry_is_fail_closed_except_the_explicit_legacy_mode(self) -> None:
        for allow_unavailable, expected, expected_logs in (
            (False, False, 1),
            (True, True, 0),
        ):
            with self.subTest(allow_unavailable=allow_unavailable):
                event = _FakeStopEvent()
                logs: list[tuple[str, str]] = []
                with patch.object(gpu_recovery.time, "monotonic", return_value=0.0):
                    result = gpu_recovery.wait_for_vram_recovery(
                        lambda: None,
                        stop_event=event,
                        on_log=lambda level, message: logs.append((level, message)),
                        min_headroom_bytes=1,
                        timeout_seconds=1,
                        poll_seconds=.25,
                        allow_unavailable=allow_unavailable,
                    )

                self.assertIs(result, expected)
                self.assertEqual(len(logs), expected_logs)
                if logs:
                    self.assertEqual(logs[0][0], "ERROR")
                    self.assertIn("Cannot verify whole-card GPU recovery", logs[0][1])
                self.assertEqual(event.wait_calls, [])

    def test_malformed_or_out_of_range_telemetry_fails_closed(self) -> None:
        invalid_samples = (
            ("list rather than tuple", [0, 10]),
            ("boolean byte count", (True, 10)),
            ("wrong arity", (0,)),
            ("zero total", (0, 0)),
            ("negative total", (0, -1)),
            ("negative used", (-1, 10)),
            ("used exceeds total", (11, 10)),
        )

        for name, sample in invalid_samples:
            with self.subTest(sample=name):
                event = _FakeStopEvent()
                logs: list[tuple[str, str]] = []
                with patch.object(gpu_recovery.time, "monotonic", return_value=0.0):
                    result = gpu_recovery.wait_for_vram_recovery(
                        lambda sample=sample: sample,
                        stop_event=event,
                        on_log=lambda level, message: logs.append((level, message)),
                        min_headroom_bytes=1,
                        timeout_seconds=1,
                        poll_seconds=.25,
                    )

                self.assertFalse(result)
                self.assertEqual(event.wait_calls, [])
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0][0], "ERROR")
                self.assertIn("Cannot verify whole-card GPU recovery", logs[0][1])

    def test_reader_exception_fails_closed(self) -> None:
        event = _FakeStopEvent()
        logs: list[tuple[str, str]] = []

        def broken_reader():
            raise RuntimeError("synthetic telemetry failure")

        with patch.object(gpu_recovery.time, "monotonic", return_value=0.0):
            result = gpu_recovery.wait_for_vram_recovery(
                broken_reader,
                stop_event=event,
                on_log=lambda level, message: logs.append((level, message)),
                min_headroom_bytes=1,
                timeout_seconds=1,
                poll_seconds=.25,
            )

        self.assertFalse(result)
        self.assertEqual(event.wait_calls, [])
        self.assertEqual(logs, [("ERROR", "Cannot verify whole-card GPU recovery: synthetic telemetry failure")])

    def test_cancellation_before_or_during_fake_wait_returns_without_sleeping(self) -> None:
        with self.subTest(when="before read"):
            event = _FakeStopEvent(initially_set=True)

            def should_not_run():
                raise AssertionError("cancelled recovery must not read telemetry")

            with patch.object(gpu_recovery.time, "monotonic", return_value=0.0):
                result = gpu_recovery.wait_for_vram_recovery(
                    should_not_run,
                    stop_event=event,
                    on_log=lambda _level, _message: None,
                    min_headroom_bytes=1,
                    timeout_seconds=1,
                    poll_seconds=.25,
                )

            self.assertFalse(result)
            self.assertEqual(event.wait_calls, [])

        with self.subTest(when="during wait"):
            event = _FakeStopEvent(set_after_waits=1)
            reader = _SequenceReader(((1000, 1000),))
            with patch.object(gpu_recovery.time, "monotonic", side_effect=(0.0, 0.0)):
                result = gpu_recovery.wait_for_vram_recovery(
                    reader,
                    stop_event=event,
                    on_log=lambda _level, _message: None,
                    min_headroom_bytes=1,
                    timeout_seconds=1,
                    poll_seconds=.25,
                )

            self.assertFalse(result)
            self.assertEqual(reader.calls, 1)
            self.assertEqual(event.wait_calls, [.25])

    def test_timeout_uses_fake_monotonic_and_never_waits_for_real_time(self) -> None:
        event = _FakeStopEvent()
        reader = _SequenceReader(((1000, 1000),))
        logs: list[tuple[str, str]] = []

        with patch.object(gpu_recovery.time, "monotonic", side_effect=(0.0, 1.0)):
            result = gpu_recovery.wait_for_vram_recovery(
                reader,
                stop_event=event,
                on_log=lambda level, message: logs.append((level, message)),
                min_headroom_bytes=1,
                timeout_seconds=1,
                poll_seconds=.25,
                stable_samples=2,
            )

        self.assertFalse(result)
        self.assertEqual(reader.calls, 1)
        self.assertEqual(event.wait_calls, [])
        self.assertEqual(logs[0][0], "ERROR")
        self.assertIn("GPU memory did not recover", logs[0][1])


class _Digest:
    def hexdigest(self) -> str:
        return guarded_attempt.GUARD_SHA256


class _NoPidKillOs:
    """Minimal Windows stand-in which exposes unexpected PID cleanup."""

    name = "nt"

    def kill(self, *_args: object) -> None:
        raise AssertionError("guard cleanup must use only the owned process handle")


class _FakeBinaryPipe:
    def __init__(self) -> None:
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise ValueError("pipe is closed")
        return len(data)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("pipe is closed")

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    _next_pid = 9100

    def __init__(self, *, returncode: int = 0) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.stdin = _FakeBinaryPipe()
        self.stdout = _FakeBinaryPipe()
        self.stderr = _FakeBinaryPipe()
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class _ScriptedStreams:
    """No-thread stand-in for the already-tested stream-record transport."""

    def __init__(self, records=()) -> None:
        self._records = deque(records)
        self.started = False
        self.join_timeouts: list[float] = []

    def start(self) -> None:
        self.started = True

    def next_record(self, timeout: float) -> StreamRecord | None:
        if self._records:
            return self._records.popleft()
        return None

    @property
    def has_terminal_error(self) -> bool:
        return False

    @property
    def finished(self) -> bool:
        return not self._records

    @property
    def dropped_records(self) -> dict[str, int]:
        return {"stdout": 0, "stderr": 0}

    def join(self, timeout: float) -> bool:
        self.join_timeouts.append(timeout)
        return True


class _RunnerHarness:
    def __init__(self) -> None:
        self.popen_calls: list[tuple[list[str], dict[str, object]]] = []
        self.verification_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []


class _GuardedAttemptTestCase(unittest.TestCase):
    def _config(self) -> guarded_attempt.GuardedAttemptConfig:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        guard_path = root / "GuardedRunner.exe"
        guard_path.write_bytes(b"fake guard")
        return guarded_attempt.GuardedAttemptConfig(
            guard_path=guard_path,
            working_directory=root,
            timeout_seconds=5,
        )

    @contextmanager
    def _patched_runner(
        self,
        *,
        process: _FakeProcess,
        streams: _ScriptedStreams,
        verified_exit: int,
        verifier_error: Exception | None = None,
    ):
        harness = _RunnerHarness()

        def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
            command_copy = list(command)
            harness.popen_calls.append((command_copy, dict(kwargs)))
            report_path = Path(command_copy[command_copy.index("--report") + 1])
            report_path.write_bytes(b'{"report":"fake"}')
            return process

        def fake_verify(*args: object, **kwargs: object) -> int:
            harness.verification_calls.append((args, dict(kwargs)))
            if verifier_error is not None:
                raise verifier_error
            return verified_exit

        def parser(line: str):
            if not line.startswith(EVENT_PREFIX):
                return None
            return json.loads(line[len(EVENT_PREFIX) :])

        protocol_module = type(sys)("jasna.gui.video_job_process")
        protocol_module.parse_event_line = parser
        create_no_window = getattr(guarded_attempt.subprocess, "CREATE_NO_WINDOW", 0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(guarded_attempt, "os", _NoPidKillOs()))
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
                patch.object(guarded_attempt, "IsolatedWorkerStreams", return_value=streams)
            )
            stack.enter_context(
                patch.object(guarded_attempt, "verify_guard_exit", side_effect=fake_verify)
            )
            stack.enter_context(
                patch.dict(sys.modules, {"jasna.gui.video_job_process": protocol_module})
            )
            yield harness

    @staticmethod
    def _callbacks(*, on_event=None, on_started=None, on_finished=None) -> dict[str, object]:
        return {
            "on_event": on_event or (lambda _event: False),
            "on_log": lambda _level, _message: None,
            "on_started": on_started or (lambda _process: None),
            "on_finished": on_finished or (lambda _process: None),
        }

    @staticmethod
    def _identity_payload(*, token: str = _TOKEN, **changes: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "windows_gpu_identity",
            "attempt_token": token,
            "adapter_marker": _MARKER,
            "node_index": 0,
        }
        payload.update(changes)
        return payload

    @classmethod
    def _identity_record(cls, **changes: object) -> StreamRecord:
        return StreamRecord(
            "stdout",
            EVENT_PREFIX + json.dumps(cls._identity_payload(**changes), separators=(",", ":")),
        )

    @staticmethod
    def _guard_record() -> StreamRecord:
        return StreamRecord("stderr", guarded_attempt.GUARD_PREFIX + '{"guard":"fake"}')

    def _commit_identity(self) -> tuple[guarded_attempt.WindowsGuardedAttempt, WindowsGpuIdentity]:
        attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
        expected = WindowsGpuIdentity(_MARKER, 0)
        with (
            self._patched_runner(
                process=_FakeProcess(),
                streams=_ScriptedStreams((self._identity_record(), self._guard_record())),
                verified_exit=0,
            ),
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
        ):
            result = attempt.run(
                ["child.exe"],
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                **self._callbacks(),
            )
        self.assertEqual(result, (None, None, 0))
        self.assertEqual(attempt.last_gpu_identity, expected)
        return attempt, expected


class WindowsGuardedAttemptGpuIdentityTests(_GuardedAttemptTestCase):
    def test_requested_identity_replaces_the_caller_token_in_a_copied_child_environment(self) -> None:
        attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
        source_environment = {
            "JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1",
            "JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN": "stale-token",
            "JASNA_TEST": "preserved",
        }
        original_environment = dict(source_environment)
        process = _FakeProcess()
        streams = _ScriptedStreams((self._identity_record(), self._guard_record()))

        with (
            self._patched_runner(process=process, streams=streams, verified_exit=0) as harness,
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN) as token_hex,
        ):
            result = attempt.run(["child.exe"], source_environment, **self._callbacks())

        self.assertEqual(result, (None, None, 0))
        token_hex.assert_called_once_with(16)
        self.assertEqual(source_environment, original_environment)
        child_environment = harness.popen_calls[0][1]["env"]
        self.assertIsNot(child_environment, source_environment)
        self.assertEqual(
            child_environment,
            {
                "JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1",
                "JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN": _TOKEN,
                "JASNA_TEST": "preserved",
            },
        )

    def test_valid_identity_commits_only_after_finished_callback_and_clean_retirement(self) -> None:
        attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
        process = _FakeProcess()
        streams = _ScriptedStreams((self._identity_record(), self._guard_record()))
        events: list[dict[str, object]] = []
        states_at_finished: list[tuple[object, bool, bool, bool]] = []

        def on_finished(_handle: object) -> None:
            states_at_finished.append(
                (
                    attempt.last_gpu_identity,
                    process.stdin.closed,
                    process.stdout.closed,
                    process.stderr.closed,
                )
            )

        with (
            self._patched_runner(process=process, streams=streams, verified_exit=75),
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
        ):
            result = attempt.run(
                ["child.exe"],
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                **self._callbacks(
                    on_event=lambda event: events.append(event) or False,
                    on_finished=on_finished,
                ),
            )

        self.assertEqual(result, (None, None, 75))
        self.assertEqual(events, [])
        self.assertEqual(states_at_finished, [(None, True, True, True)])
        self.assertEqual(attempt.last_gpu_identity, WindowsGpuIdentity(_MARKER, 0))
        self.assertEqual(streams.join_timeouts, [1])

    def test_forced_or_timeout_verifier_rejection_never_commits_identity(self) -> None:
        for status in ("forced", "timeout"):
            with self.subTest(status=status):
                attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
                with (
                    self._patched_runner(
                        process=_FakeProcess(),
                        streams=_ScriptedStreams(
                            (self._identity_record(), self._guard_record())
                        ),
                        verified_exit=0,
                        verifier_error=ValueError(f"synthetic {status} status"),
                    ) as harness,
                    patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
                ):
                    terminal, error, exit_code = attempt.run(
                        ["child.exe"],
                        {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                        **self._callbacks(),
                    )

                self.assertIsNone(terminal)
                self.assertEqual(exit_code, 1)
                self.assertIn(f"synthetic {status} status", error or "")
                self.assertEqual(len(harness.verification_calls), 1)
                self.assertIsNone(attempt.last_gpu_identity)
                self.assertFalse(attempt._retirement_failed)

    def test_finished_callback_failure_poisoned_attempt_never_commits_identity(self) -> None:
        attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
        process = _FakeProcess()

        def fail_finished(_handle: object) -> None:
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
            raise RuntimeError("synthetic parent deregistration failure")

        with (
            self._patched_runner(
                process=process,
                streams=_ScriptedStreams((self._identity_record(), self._guard_record())),
                verified_exit=0,
            ),
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
        ):
            terminal, error, exit_code = attempt.run(
                ["child.exe"],
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                **self._callbacks(on_finished=fail_finished),
            )

        self.assertIsNone(terminal)
        self.assertEqual(exit_code, 1)
        self.assertIn("parent deregistration failed: RuntimeError", error or "")
        self.assertIsNone(attempt.last_gpu_identity)
        self.assertTrue(attempt._retirement_failed)

    def test_next_attempt_clears_cached_identity_and_allows_normal_failure_without_one(self) -> None:
        attempt, _identity = self._commit_identity()
        identity_at_next_start: list[object] = []
        with (
            self._patched_runner(
                process=_FakeProcess(),
                streams=_ScriptedStreams((self._guard_record(),)),
                verified_exit=1,
            ),
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
        ):
            result = attempt.run(
                ["failure-child.exe"],
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                **self._callbacks(
                    on_started=lambda _handle: identity_at_next_start.append(
                        attempt.last_gpu_identity
                    )
                ),
            )

        self.assertEqual(identity_at_next_start, [None])
        self.assertEqual(result, (None, None, 1))
        self.assertIsNone(attempt.last_gpu_identity)
        self.assertFalse(attempt._retirement_failed)

    def test_invalid_or_unrequested_identity_events_fail_before_guard_result_acceptance(self) -> None:
        cases = (
            (
                "duplicate",
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                (self._identity_record(), self._identity_record()),
                "unexpected or mismatched",
            ),
            (
                "wrong token",
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                (self._identity_record(token="b" * 32),),
                "unexpected or mismatched",
            ),
            (
                "extra field",
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                (self._identity_record(extra="not allowed"),),
                "unexpected or mismatched",
            ),
            (
                "malformed field",
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                (self._identity_record(node_index=True),),
                "node_index must be a plain int",
            ),
            (
                "unrequested",
                {},
                (self._identity_record(),),
                "unexpected or mismatched",
            ),
        )

        for name, environment, identity_records, expected_error in cases:
            with self.subTest(identity=name):
                attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
                streams = _ScriptedStreams((*identity_records, self._guard_record()))
                applied_events: list[dict[str, object]] = []
                with (
                    self._patched_runner(
                        process=_FakeProcess(), streams=streams, verified_exit=0
                    ) as harness,
                    patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
                ):
                    terminal, error, exit_code = attempt.run(
                        ["child.exe"],
                        environment,
                        **self._callbacks(
                            on_event=lambda event: applied_events.append(event) or False
                        ),
                    )

                self.assertIsNone(terminal)
                self.assertEqual(exit_code, 1)
                self.assertIn(expected_error, error or "")
                self.assertEqual(applied_events, [])
                self.assertIsNone(attempt.last_gpu_identity)
                self.assertEqual(harness.verification_calls, [])

    def test_requested_normal_exits_require_identity_but_worker_failure_exit_does_not(self) -> None:
        for verified_exit in (0, 75, 86):
            with self.subTest(verified_exit=verified_exit):
                attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
                with (
                    self._patched_runner(
                        process=_FakeProcess(),
                        streams=_ScriptedStreams((self._guard_record(),)),
                        verified_exit=verified_exit,
                    ) as harness,
                    patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
                ):
                    terminal, error, exit_code = attempt.run(
                        ["child.exe"],
                        {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                        **self._callbacks(),
                    )

                self.assertIsNone(terminal)
                self.assertEqual(exit_code, 1)
                self.assertIn("did not report its selected GPU identity", error or "")
                self.assertIsNone(attempt.last_gpu_identity)
                self.assertEqual(len(harness.verification_calls), 1)

        attempt = guarded_attempt.WindowsGuardedAttempt(self._config())
        with (
            self._patched_runner(
                process=_FakeProcess(),
                streams=_ScriptedStreams((self._guard_record(),)),
                verified_exit=1,
            ),
            patch.object(guarded_attempt.secrets, "token_hex", return_value=_TOKEN),
        ):
            result = attempt.run(
                ["child.exe"],
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY": "1"},
                **self._callbacks(),
            )

        self.assertEqual(result, (None, None, 1))
        self.assertIsNone(attempt.last_gpu_identity)

    def test_recovery_reader_is_single_use_closes_and_holds_the_attempt_lock(self) -> None:
        attempt, identity = self._commit_identity()
        reader = _FakeRecoveryReader()

        with patch.object(
            guarded_attempt.WindowsGlobalVramReader,
            "from_identity",
            return_value=reader,
        ) as from_identity:
            with attempt.open_recovery_reader() as received:
                self.assertIs(received, reader)
                with self.assertRaisesRegex(RuntimeError, "guarded attempt is already active"):
                    attempt.run(["later-child.exe"], {}, **self._callbacks())

        from_identity.assert_called_once_with(identity)
        self.assertEqual(reader.close_calls, 1)
        self.assertIsNone(attempt.last_gpu_identity)
        with self.assertRaisesRegex(RuntimeError, "no clean retired worker GPU identity"):
            with attempt.open_recovery_reader():
                self.fail("a recovery identity must be single-use")

    def test_reader_construction_or_close_failure_poison_later_attempts(self) -> None:
        with self.subTest(failure="from_identity"):
            attempt, _identity = self._commit_identity()
            with patch.object(
                guarded_attempt.WindowsGlobalVramReader,
                "from_identity",
                side_effect=RuntimeError("synthetic reader construction failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic reader construction failure"):
                    with attempt.open_recovery_reader():
                        self.fail("construction failure must not yield a reader")

            self.assertIsNone(attempt.last_gpu_identity)
            self.assertTrue(attempt._retirement_failed)
            with self.assertRaisesRegex(RuntimeError, "did not retire cleanly"):
                attempt.run(["later-child.exe"], {}, **self._callbacks())

        with self.subTest(failure="close"):
            attempt, _identity = self._commit_identity()
            reader = _FakeRecoveryReader(close_error=RuntimeError("synthetic reader close failure"))
            with patch.object(
                guarded_attempt.WindowsGlobalVramReader,
                "from_identity",
                return_value=reader,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic reader close failure"):
                    with attempt.open_recovery_reader() as received:
                        self.assertIs(received, reader)

            self.assertEqual(reader.close_calls, 1)
            self.assertTrue(attempt._retirement_failed)
            with self.assertRaisesRegex(RuntimeError, "did not retire cleanly"):
                attempt.run(["later-child.exe"], {}, **self._callbacks())


class _FakeRecoveryReader:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_calls = 0
        self._close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class ProcessorRecoverySourceTests(unittest.TestCase):
    @staticmethod
    def _wait_method() -> ast.FunctionDef:
        source_path = PRODUCT_ROOT / "jasna" / "gui" / "processor.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        processor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Processor"
        )
        return next(
            node
            for node in processor.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_wait_for_isolated_gpu_recovery"
        )

    def test_windows_branch_uses_a_closed_identity_reader_and_legacy_linux_stays_explicit(self) -> None:
        method = self._wait_method()
        backend_assignment = next(
            node
            for node in method.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "backend" for target in node.targets)
        )
        self.assertIsInstance(backend_assignment.value, ast.Call)
        self.assertIsInstance(backend_assignment.value.func, ast.Name)
        self.assertEqual(backend_assignment.value.func.id, "getattr")
        self.assertIsInstance(backend_assignment.value.args[2], ast.Constant)
        self.assertIsNone(backend_assignment.value.args[2].value)

        backend_if = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and any(
                isinstance(candidate, ast.Name) and candidate.id == "backend"
                for candidate in ast.walk(node.test)
            )
        )
        self.assertIsInstance(backend_if.test, ast.Compare)
        self.assertTrue(any(isinstance(operator, ast.IsNot) for operator in backend_if.test.ops))

        reader_with = next(node for node in ast.walk(backend_if) if isinstance(node, ast.With))
        item = reader_with.items[0]
        self.assertIsInstance(item.context_expr, ast.Call)
        self.assertIsInstance(item.context_expr.func, ast.Attribute)
        self.assertEqual(item.context_expr.func.attr, "open_recovery_reader")
        self.assertIsInstance(item.context_expr.func.value, ast.Name)
        self.assertEqual(item.context_expr.func.value.id, "backend")
        self.assertIsInstance(item.optional_vars, ast.Name)
        self.assertEqual(item.optional_vars.id, "reader")

        windows_wait = next(
            node
            for node in ast.walk(backend_if)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "wait_for_vram_recovery"
        )
        self.assertIsInstance(windows_wait.args[0], ast.Name)
        self.assertEqual(windows_wait.args[0].id, "reader")
        windows_keywords = {keyword.arg: keyword.value for keyword in windows_wait.keywords}
        headroom = windows_keywords["min_headroom_bytes"]
        self.assertIsInstance(headroom, ast.BinOp)
        self.assertIsInstance(headroom.op, ast.Pow)
        self.assertEqual((headroom.left.value, headroom.right.value), (1024, 3))
        self.assertNotIn("allow_unavailable", windows_keywords)
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom) and node.module == "jasna.vram_offloader"
                for node in ast.walk(backend_if)
            )
        )

        backend_index = method.body.index(backend_if)
        legacy_import_index = next(
            index
            for index, node in enumerate(method.body)
            if isinstance(node, ast.ImportFrom) and node.module == "jasna.vram_offloader"
        )
        self.assertGreater(legacy_import_index, backend_index)
        legacy_return = next(
            node
            for node in method.body[legacy_import_index + 1 :]
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "wait_for_vram_recovery"
        )
        legacy_wait = legacy_return.value
        self.assertIsInstance(legacy_wait.args[0], ast.Name)
        self.assertEqual(legacy_wait.args[0].id, "read_linux_amd_system_vram")
        legacy_keywords = {keyword.arg: keyword.value for keyword in legacy_wait.keywords}
        self.assertIsInstance(legacy_keywords["min_headroom_bytes"], ast.Name)
        self.assertEqual(
            legacy_keywords["min_headroom_bytes"].id,
            "AMD_MIN_VRAM_STARTUP_BUDGET",
        )
        self.assertIsInstance(legacy_keywords["allow_unavailable"], ast.Constant)
        self.assertIs(legacy_keywords["allow_unavailable"].value, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

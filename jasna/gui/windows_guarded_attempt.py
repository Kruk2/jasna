"""Opt-in Windows Job backend for the shared GUI video-worker protocol.

No automatic platform selection, media import, retry policy or runtime discovery
lives here. The caller supplies an already configured child command/environment.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import tempfile
import threading
import time

from jasna.gui.isolated_worker_streams import IsolatedWorkerStreams
from jasna.gui.windows_guard_exit_result import MAX_RECORD_BYTES, verify_guard_exit
from jasna.windows_global_vram import WindowsGlobalVramReader, WindowsGpuIdentity

GUARD_SHA256 = "1a9f4a6961dba52bd7b3cc3e853a914907b0cd583fb4c0d503a5479f788f17ad"
GUARD_PREFIX = "JASNA_JOB_RESULT="
MAX_COMMAND_BYTES = 4096
MAX_PENDING_COMMANDS = 8
MAX_DIAGNOSTIC_LINES = 64


@dataclass(frozen=True)
class GuardedAttemptConfig:
    guard_path: Path
    working_directory: Path
    timeout_seconds: int = 180

    def __post_init__(self):
        for value in (self.guard_path, self.working_directory):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError("guard paths must be explicit absolute Path objects")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 180:
            raise ValueError("guarded attempt timeout must be an integer from 1 to 180")


class _TextCommandPipe:
    """Bounded nonblocking GUI-side command enqueue; writer owns the OS pipe."""
    def __init__(self, stream):
        self._stream = stream
        self._queue = queue.Queue(maxsize=MAX_PENDING_COMMANDS)
        self._error = None
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._write_loop, daemon=True,
                                        name="windows-video-command-writer")
        self._thread.start()

    @property
    def error(self):
        with self._lock:
            return self._error

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("worker command must be text")
        encoded = text.encode("utf-8")
        if not 0 < len(encoded) <= MAX_COMMAND_BYTES or not text.endswith("\n") or text.count("\n") != 1:
            raise ValueError("worker command exceeds line contract")
        with self._lock:
            if self._closed or self._error is not None:
                raise OSError("worker command pipe is unavailable")
            try:
                self._queue.put_nowait(encoded)
            except queue.Full:
                self._error = "worker command queue overflow"
                raise OSError(self._error)
        return len(text)

    def flush(self):
        # write() enqueues a complete command; the writer flushes each one.
        if self.error is not None:
            raise OSError(self.error)

    def _write_loop(self):
        try:
            while True:
                data = self._queue.get()
                if data is None:
                    return
                if self._stream.write(data) != len(data):
                    raise OSError("partial worker command write")
                self._stream.flush()
        except BaseException as error:
            with self._lock:
                self._error = f"worker command writer failed: {type(error).__name__}"

    def close(self):
        # Caller must first terminate/wait the exact guard so blocked writes end.
        with self._lock:
            self._closed = True
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)
        self._thread.join(1)
        if self._thread.is_alive():
            raise RuntimeError("worker command writer did not retire")


class _GuardProcessHandle:
    """Only operations already used by Processor's Stop/reaper boundary."""
    def __init__(self, process):
        self._process = process
        self.stdin = _TextCommandPipe(process.stdin)
        self.pid = process.pid

    def poll(self):
        return self._process.poll()

    def wait(self, timeout=None):
        return self._process.wait(timeout=timeout)

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()


class WindowsGuardedAttempt:
    def __init__(self, config: GuardedAttemptConfig, *, worker_runtime=None):
        if not isinstance(config, GuardedAttemptConfig):
            raise TypeError("a GuardedAttemptConfig is required")
        self.config = config
        self.worker_runtime = worker_runtime
        self._run_lock = threading.Lock()
        self.last_guard_report = None
        self.last_guard_command = None
        self.last_outer_exit_code = None
        self._retirement_failed = False
        self.last_gpu_identity = None

    @contextmanager
    def open_recovery_reader(self):
        """Own one HIP-free reader only after its worker fully retired.

        The same lock prevents starting another worker during recovery. A
        handoff identity is single-use and never survives the next attempt.
        """
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("cannot observe recovery while an attempt is active")
        reader = None
        try:
            identity = self.last_gpu_identity
            self.last_gpu_identity = None
            if self._retirement_failed or identity is None:
                raise RuntimeError("no clean retired worker GPU identity is available")
            try:
                reader = WindowsGlobalVramReader.from_identity(identity)
            except BaseException:
                # Construction may fail while cleaning up a rejected native
                # reader. We cannot prove retirement in that case either.
                self._retirement_failed = True
                raise
            yield reader
        finally:
            try:
                if reader is not None:
                    try:
                        reader.close()
                    except BaseException:
                        self._retirement_failed = True
                        raise
            finally:
                self._run_lock.release()

    def prepare_request(self, request_path, environment):
        if self.worker_runtime is None:
            raise RuntimeError("Windows worker runtime was not explicitly configured")
        return self.worker_runtime.prepare(request_path, environment)

    def run(self, command, environment, *, on_event, on_log, on_started, on_finished):
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("a guarded attempt is already active")
        try:
            if self._retirement_failed:
                raise RuntimeError("prior guarded attempt did not retire cleanly; restart required")
            return self._run(command, environment, on_event=on_event, on_log=on_log,
                             on_started=on_started, on_finished=on_finished)
        finally:
            self._run_lock.release()

    def _run(self, command, environment, *, on_event, on_log, on_started, on_finished):
        from jasna.gui.video_job_process import parse_event_line

        self.last_guard_report = self.last_guard_command = self.last_outer_exit_code = None
        self.last_gpu_identity = None
        requested_identity = environment.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") == "1"
        attempt_token = secrets.token_hex(16) if requested_identity else None
        environment = dict(environment)
        environment.pop("JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN", None)
        if requested_identity:
            environment["JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN"] = attempt_token
        candidate_identity = None
        if os.name != "nt":
            raise RuntimeError("Windows Job backend requires Windows")
        if not isinstance(command, list) or not command or any(type(arg) is not str for arg in command):
            raise ValueError("explicit child command list required")
        if len(command) > 256 or sum(len(arg) for arg in command) > 32768 or not command[0]:
            raise ValueError("child command exceeds bounded contract")
        if not self.config.working_directory.is_dir():
            raise ValueError("guard working directory is unavailable")
        with self.config.guard_path.open("rb") as stream:
            observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed_sha != GUARD_SHA256:
            raise ValueError("Windows guard binary identity changed")

        process = handle = streams = None
        terminal = error_message = None
        child_exit = 1
        diagnostics = 0
        guard_record = None
        with tempfile.TemporaryDirectory(prefix="jasna-guard-attempt-") as temporary:
            report_path = Path(temporary) / "guard-result.json"
            guard_command = [str(self.config.guard_path), "--job-memory-mib", "6144",
                "--host-commit-reserve-mib", "10240", "--host-physical-reserve-mib", "8192",
                "--timeout-seconds", str(self.config.timeout_seconds), "--poll-ms", "1000",
                "--active-process-limit", "16", "--working-directory", str(self.config.working_directory),
                "--report", str(report_path), "--", *command]
            self.last_guard_command = tuple(guard_command)
            try:
                process = subprocess.Popen(guard_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, bufsize=0, cwd=self.config.working_directory,
                    env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
                handle = _GuardProcessHandle(process)
                streams = IsolatedWorkerStreams(process.stdout, process.stderr)
                streams.start()
                on_started(handle)
                deadline = time.monotonic() + self.config.timeout_seconds + 15
                while time.monotonic() < deadline:
                    record = streams.next_record(.1)
                    if streams.has_terminal_error or handle.stdin.error is not None:
                        raise RuntimeError("guarded worker communication failed")
                    if record is not None:
                        if record.source == "stderr" and record.line.startswith(GUARD_PREFIX):
                            if guard_record is not None:
                                raise ValueError("duplicate guard result")
                            guard_record = record.line[len(GUARD_PREFIX):].encode("utf-8")
                        elif record.source == "stdout":
                            event = parse_event_line(record.line)
                            if event is not None:
                                if event.get("type") == "windows_gpu_identity":
                                    if (not requested_identity or candidate_identity is not None
                                            or set(event) != {"type", "attempt_token", "adapter_marker", "node_index"}
                                            or event.get("attempt_token") != attempt_token):
                                        raise ValueError("unexpected or mismatched worker GPU identity")
                                    candidate_identity = WindowsGpuIdentity(event["adapter_marker"], event["node_index"])
                                    continue
                                applied = on_event(event)
                                if isinstance(applied, dict):
                                    if terminal is not None:
                                        raise ValueError("isolated worker emitted multiple terminal events")
                                    terminal = applied
                            elif record.line:
                                diagnostics += 1
                                if diagnostics <= MAX_DIAGNOSTIC_LINES:
                                    on_log("WARNING", "[video worker] " + record.line[:2048])
                        elif record.line:
                            diagnostics += 1
                            if diagnostics <= MAX_DIAGNOSTIC_LINES:
                                on_log("WARNING", "[video worker stderr] " + record.line[:2048])
                    if record is None and streams.finished and process.poll() is not None:
                        break
                else:
                    raise RuntimeError("guard did not retire within its bounded attempt deadline")
                outer = process.wait(timeout=1)
                self.last_outer_exit_code = outer
                if streams.has_terminal_error or streams.dropped_records["stdout"]:
                    raise RuntimeError("worker protocol was lost")
                if guard_record is None:
                    raise ValueError("guard did not emit exactly one final report")
                with report_path.open("rb") as stream:
                    raw = stream.read(MAX_RECORD_BYTES + 1)
                self.last_guard_report = raw[:MAX_RECORD_BYTES + 1]
                child_exit = verify_guard_exit(raw, guard_record, expected_command=command,
                    expected_working_directory=str(self.config.working_directory), actual_outer_exit_code=outer)
                if requested_identity and child_exit in (0, 75, 86) and candidate_identity is None:
                    raise ValueError("worker did not report its selected GPU identity")
                if diagnostics > MAX_DIAGNOSTIC_LINES or streams.dropped_records["stderr"]:
                    on_log("WARNING", "Worker diagnostics exceeded the bounded display budget; excess lines were dropped")
            except Exception as error:
                terminal, error_message, child_exit = None, f"{type(error).__name__}: {error}"[:4096], 1
            finally:
                cleanup_errors = []
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.terminate()  # Exact guard handle; Job owns descendants.
                        process.wait(timeout=10)
                    except Exception as error:
                        # Preserve the exact Popen handle; never look up or kill
                        # an integer PID that could already have been reused.
                        try:
                            process.kill()
                            process.wait(timeout=5)
                        except Exception as kill_error:
                            cleanup_errors.append(f"guard cleanup failed: {type(kill_error).__name__}")
                    if handle is not None:
                        try:
                            handle.stdin.close()
                        except Exception as error:
                            cleanup_errors.append(str(error))
                    if streams is not None:
                        try:
                            if not streams.join(1):
                                cleanup_errors.append("worker stream readers did not retire")
                        except Exception as error:
                            cleanup_errors.append(f"worker stream join failed: {type(error).__name__}")
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe is not None:
                            try:
                                pipe.close()
                            except (OSError, ValueError) as error:
                                cleanup_errors.append(str(error))
                if handle is not None:
                    try:
                        on_finished(handle)
                    except Exception as error:
                        cleanup_errors.append(f"parent deregistration failed: {type(error).__name__}")
                if cleanup_errors:
                    self._retirement_failed = True
                    terminal, child_exit = None, 1
                    error_message = "; ".join(cleanup_errors)[:4096]
        if error_message is None and requested_identity and child_exit in (0, 75, 86):
            self.last_gpu_identity = candidate_identity
        return terminal, error_message, child_exit

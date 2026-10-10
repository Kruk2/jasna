"""Bounded binary stdout/stderr transport for isolated GUI workers.

This module deliberately knows nothing about process creation, termination,
return codes, GUI callbacks, or the worker's event schema.  Its caller owns
those concerns and can pass each stdout record to the existing parse_event_line
function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import os
import threading
import time
from typing import Any, Callable, Literal


READ_CHUNK_BYTES = 4 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_RECORDS = 64
MAX_ERRORS = 8
MAX_NEXT_RECORD_TIMEOUT_SECONDS = 1.0
MAX_JOIN_SECONDS = 1.0

StreamSource = Literal["stdout", "stderr"]
ReadFunction = Callable[[int, int], bytes]
ThreadFactory = Callable[..., Any]


class TransportStartError(RuntimeError):
    """A reader could not be started; started readers remain owned by caller."""


@dataclass(frozen=True)
class StreamRecord:
    """One complete UTF-8 line from a worker stream, without its line ending."""

    source: StreamSource
    line: str


@dataclass(frozen=True)
class TransportIssue:
    """A bounded diagnostic about transport integrity or a dropped record."""

    source: str
    code: str
    message: str
    terminal: bool


@dataclass
class _LineState:
    buffer: bytearray = field(default_factory=bytearray)
    discarding_line: bool = False
    discard_all: bool = False


class IsolatedWorkerStreams:
    """Drain two binary pipes without letting consumers backpressure readers.

    Stdout is treated as the worker protocol channel.  Any stdout loss,
    malformed UTF-8, incomplete EOF line, reader error, or queue overflow is a
    terminal transport error.  Stderr is diagnostic: its bounded records may
    be dropped, with counters and nonterminal issues retained for the caller.
    """

    def __init__(
        self,
        stdout: Any,
        stderr: Any,
        *,
        read_function: ReadFunction = os.read,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        self._streams: dict[StreamSource, Any] = {
            "stdout": stdout,
            "stderr": stderr,
        }
        self._read_function = read_function
        self._thread_factory = thread_factory
        self._lock = threading.RLock()
        self._record_ready = threading.Condition(self._lock)
        self._records: deque[StreamRecord] = deque()
        self._states: dict[StreamSource, _LineState] = {
            "stdout": _LineState(),
            "stderr": _LineState(),
        }
        self._dropped_records: dict[StreamSource, int] = {
            "stdout": 0,
            "stderr": 0,
        }
        self._issues: list[TransportIssue] = []
        self._suppressed_issue_count = 0
        self._terminal = False
        self._threads: list[Any] = []
        self._start_called = False
        self._start_finished = False
        self._join_timeout_reported = False

    @property
    def reader_threads(self) -> tuple[Any, ...]:
        """Successfully started readers, including ones surviving start failure."""

        with self._lock:
            return tuple(self._threads)

    @property
    def errors(self) -> tuple[TransportIssue, ...]:
        """At most MAX_ERRORS issues; use suppressed_error_count for the rest."""

        with self._lock:
            return tuple(self._issues)

    @property
    def suppressed_error_count(self) -> int:
        with self._lock:
            return self._suppressed_issue_count

    @property
    def dropped_records(self) -> dict[StreamSource, int]:
        with self._lock:
            return dict(self._dropped_records)

    @property
    def has_terminal_error(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def finished(self) -> bool:
        """Whether startup completed and every successfully started reader ended."""

        with self._lock:
            if not self._start_finished:
                return False
            threads = tuple(self._threads)
        return all(not thread.is_alive() for thread in threads)

    def start(self) -> None:
        """Start exactly two daemon readers, or retain started readers on failure."""

        with self._lock:
            if self._start_called:
                raise RuntimeError("IsolatedWorkerStreams.start() may only be called once")
            self._start_called = True

        start_error: BaseException | None = None
        try:
            for source in ("stdout", "stderr"):
                stream = self._streams[source]
                reader: Any | None = None
                try:
                    reader = self._thread_factory(
                        target=self._reader_main,
                        args=(source, stream),
                        name=f"isolated-worker-{source}",
                        daemon=True,
                    )
                    reader.start()
                except BaseException as error:
                    # A nonstandard factory could start then raise. Retain it
                    # only if it is actually alive; unstarted Thread.join()
                    # would itself be invalid.
                    try:
                        started_despite_error = (
                            reader is not None and reader.is_alive()
                        )
                    except BaseException:
                        started_despite_error = False
                    if started_despite_error:
                        with self._lock:
                            self._threads.append(reader)
                    self._record_issue(
                        source,
                        "reader_start_failure",
                        f"could not start {source} reader: {type(error).__name__}",
                        terminal=True,
                    )
                    start_error = error
                    break
                else:
                    with self._lock:
                        self._threads.append(reader)
        finally:
            with self._lock:
                self._start_finished = True

        if start_error is not None:
            raise TransportStartError("could not start isolated worker readers") from start_error

    def next_record(self, timeout: float = 0.1) -> StreamRecord | None:
        """Return one queued line, or None after a bounded empty wait.

        This method never invokes callbacks and never blocks longer than one
        second.  A None result is not EOF by itself; check finished as well.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
            or timeout > MAX_NEXT_RECORD_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "next_record timeout must be a number from 0 through "
                f"{MAX_NEXT_RECORD_TIMEOUT_SECONDS:.1f} seconds"
            )
        deadline = time.monotonic() + float(timeout)
        with self._record_ready:
            while not self._records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._record_ready.wait(remaining)
            return self._records.popleft()

    def join(self, timeout: float = MAX_JOIN_SECONDS) -> bool:
        """Boundedly join started readers after the caller has cleaned up process IO.

        This transport never closes pipes or changes process state.  A false
        result means at least one reader remains alive and is terminal because
        stdout completeness can no longer be established safely.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
            or timeout > MAX_JOIN_SECONDS
        ):
            raise ValueError(
                "join timeout must be a number from 0 through "
                f"{MAX_JOIN_SECONDS:.1f} seconds"
            )
        with self._lock:
            threads = tuple(self._threads)
        deadline = time.monotonic() + float(timeout)
        for reader in threads:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                reader.join(remaining)
            except BaseException as error:
                self._record_issue(
                    "transport",
                    "reader_join_failure",
                    f"could not join reader: {type(error).__name__}",
                    terminal=True,
                )
        complete = all(not reader.is_alive() for reader in threads)
        if not complete:
            with self._lock:
                already_reported = self._join_timeout_reported
                self._join_timeout_reported = True
            if not already_reported:
                self._record_issue(
                    "transport",
                    "reader_join_timeout",
                    "reader did not stop within the bounded join timeout",
                    terminal=True,
                )
        return complete

    def _reader_main(self, source: StreamSource, stream: Any) -> None:
        try:
            file_descriptor = stream.fileno()
            while True:
                chunk = self._read_function(file_descriptor, READ_CHUNK_BYTES)
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("read function must return bytes")
                if not chunk:
                    self._finish_eof(source)
                    return
                self._consume_chunk(source, bytes(chunk))
        except BaseException as error:
            self._reader_failed(source, error)

    def _consume_chunk(self, source: StreamSource, chunk: bytes) -> None:
        with self._lock:
            state = self._states[source]
            if state.discard_all:
                return
            position = 0
            while position < len(chunk):
                if state.discard_all:
                    return
                newline = chunk.find(b"\n", position)
                if newline < 0:
                    fragment = chunk[position:]
                    complete_line = False
                    position = len(chunk)
                else:
                    fragment = chunk[position:newline]
                    complete_line = True
                    position = newline + 1

                if state.discarding_line:
                    if complete_line:
                        state.discarding_line = False
                    continue

                if len(state.buffer) + len(fragment) > MAX_LINE_BYTES:
                    state.buffer.clear()
                    self._line_overflow(source)
                    state.discarding_line = not complete_line
                    continue

                state.buffer.extend(fragment)
                if complete_line:
                    raw_line = bytes(state.buffer)
                    state.buffer.clear()
                    self._emit_line(source, raw_line)

    def _line_overflow(self, source: StreamSource) -> None:
        self._dropped_records[source] += 1
        if source == "stdout":
            self._states[source].discard_all = True
            self._record_issue(
                source,
                "stdout_line_overflow",
                f"stdout line exceeded {MAX_LINE_BYTES} bytes",
                terminal=True,
            )
            return
        self._record_issue(
            source,
            "stderr_line_overflow",
            f"stderr line exceeded {MAX_LINE_BYTES} bytes and was dropped",
            terminal=False,
        )

    def _emit_line(self, source: StreamSource, raw_line: bytes) -> None:
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if source == "stdout":
            try:
                line = raw_line.decode("utf-8", "strict")
            except UnicodeDecodeError:
                self._dropped_records[source] += 1
                self._states[source].discard_all = True
                self._record_issue(
                    source,
                    "stdout_invalid_utf8",
                    "stdout protocol line was not valid UTF-8",
                    terminal=True,
                )
                return
        else:
            # Diagnostics must never make a reader fail or block on decoding.
            line = raw_line.decode("utf-8", "replace")

        self._enqueue_record(StreamRecord(source=source, line=line))

    def _enqueue_record(self, record: StreamRecord) -> None:
        """Append without blocking, preserving capacity for stdout protocol lines."""

        if len(self._records) < MAX_RECORDS:
            self._records.append(record)
            self._record_ready.notify()
            return
        if record.source == "stderr":
            self._dropped_records["stderr"] += 1
            self._record_issue(
                "stderr",
                "stderr_queue_overflow",
                f"stderr queue exceeded {MAX_RECORDS} records; diagnostic dropped",
                terminal=False,
            )
            return

        # A native stderr flood must not consume the entire shared budget and
        # turn a later stdout protocol event into a false terminal overflow.
        # Evicting one queued diagnostic keeps the total cap at MAX_RECORDS.
        stderr_index = next(
            (
                index
                for index, queued in enumerate(self._records)
                if queued.source == "stderr"
            ),
            None,
        )
        if stderr_index is not None:
            del self._records[stderr_index]
            self._dropped_records["stderr"] += 1
            self._record_issue(
                "stderr",
                "stderr_queue_evicted_for_stdout",
                "queued stderr diagnostic dropped to preserve stdout protocol capacity",
                terminal=False,
            )
            self._records.append(record)
            self._record_ready.notify()
            return

        self._dropped_records["stdout"] += 1
        self._states["stdout"].discard_all = True
        self._record_issue(
            "stdout",
            "stdout_queue_overflow",
            f"stdout queue exceeded {MAX_RECORDS} protocol records",
            terminal=True,
        )

    def _finish_eof(self, source: StreamSource) -> None:
        with self._lock:
            state = self._states[source]
            if state.discard_all:
                state.buffer.clear()
                return
            if state.discarding_line:
                state.buffer.clear()
                state.discarding_line = False
                return
            if not state.buffer:
                return
            raw_line = bytes(state.buffer)
            state.buffer.clear()
            if source == "stdout":
                self._dropped_records[source] += 1
                self._record_issue(
                    source,
                    "stdout_partial_eof",
                    "stdout ended with an incomplete protocol line",
                    terminal=True,
                )
                return
            self._emit_line(source, raw_line)

    def _reader_failed(self, source: StreamSource, error: BaseException) -> None:
        with self._lock:
            state = self._states[source]
            had_buffered_line = bool(state.buffer)
            had_partial_line = had_buffered_line or state.discarding_line
            if had_buffered_line:
                self._dropped_records[source] += 1
            state.buffer.clear()
            state.discarding_line = False
            if source == "stdout":
                state.discard_all = True
            detail = type(error).__name__
            if had_partial_line:
                detail += " while a partial line was buffered"
            self._record_issue(
                source,
                f"{source}_reader_failure",
                f"{source} reader failed: {detail}",
                # A failed stderr reader no longer drains its pipe either.
                # The child can then block on native diagnostics indefinitely;
                # diagnostic truncation is recoverable, a dead reader is not.
                terminal=True,
            )

    def _record_issue(
        self, source: str, code: str, message: str, *, terminal: bool
    ) -> None:
        bounded_message = message[:256]
        with self._lock:
            if terminal:
                self._terminal = True
            issue = TransportIssue(
                source=source,
                code=code,
                message=bounded_message,
                terminal=terminal,
            )
            if len(self._issues) < MAX_ERRORS:
                self._issues.append(issue)
                return
            if terminal and not any(existing.terminal for existing in self._issues):
                # Keep at least one root-cause terminal issue visible even if
                # a stderr flood filled the bounded diagnostic history first.
                self._issues[-1] = issue
                self._suppressed_issue_count += 1
                return
            self._suppressed_issue_count += 1


__all__ = [
    "IsolatedWorkerStreams",
    "MAX_ERRORS",
    "MAX_JOIN_SECONDS",
    "MAX_LINE_BYTES",
    "MAX_NEXT_RECORD_TIMEOUT_SECONDS",
    "MAX_RECORDS",
    "READ_CHUNK_BYTES",
    "StreamRecord",
    "TransportIssue",
    "TransportStartError",
]

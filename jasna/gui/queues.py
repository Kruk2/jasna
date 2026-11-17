"""Command-queue helpers shared by the GUI background workers."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def queued_now(pending: queue.Queue | queue.SimpleQueue) -> Iterator:
    """Take each item queued so far, without waiting for more."""
    while True:
        try:
            yield pending.get_nowait()
        except queue.Empty:
            return


def drain(pending: queue.Queue) -> None:
    """Discard everything currently queued."""
    for _ in queued_now(pending):
        pass


def replace_pending(pending: queue.Queue, command: object) -> None:
    """Replace any queued command with ``command``; the GUI thread is the only producer."""
    drain(pending)
    pending.put_nowait(command)


class Stop:
    """Wakes a closed ``CommandWorker``; never returned by ``_next_command``."""


class Cancel:
    """Supersedes the pending command so a running pass drops its result; never returned by
    ``_next_command``."""


class CommandWorker:
    """One daemon thread running ``_run`` and fed through a one-command slot: a new command
    replaces the pending one, and ``close`` cancels the running pass and stops the thread."""

    def __init__(self, thread_name: str) -> None:
        self._commands: queue.Queue = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._cancel_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)

    def _run(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._cancel_active_pass()
        replace_pending(self._commands, Stop())

    def _send(self, command: object) -> None:
        if not self._closed.is_set():
            replace_pending(self._commands, command)

    def _cancel_active_pass(self) -> None:
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()

    @contextmanager
    def _cancellable_pass(self) -> Iterator[threading.Event]:
        """The cancel event of one pass, set by ``_cancel_active_pass`` while the pass runs."""
        cancel = threading.Event()
        with self._cancel_lock:
            self._active_cancel = cancel
        try:
            yield cancel
        finally:
            with self._cancel_lock:
                self._active_cancel = None

    def _superseded(self) -> bool:
        return not self._commands.empty() or self._closed.is_set()

    def _poll_command(self) -> object | None:
        """The pending command, waiting up to 0.1 s; None when none came."""
        try:
            return self._commands.get(timeout=0.1)
        except queue.Empty:
            return None

    def _next_command(self) -> object | None:
        """Wait for the next command to run; None once the worker is closed."""
        while not self._closed.is_set():
            command = self._poll_command()
            if command is not None and not isinstance(command, (Stop, Cancel)):
                return command
        return None


class MainThreadCalls:
    """Runs callbacks posted by worker threads on the Tk main thread.

    Tk must only be touched from the main thread, so workers ``post`` and the
    main loop polls; call ``close`` before destroying ``widget``.
    """

    def __init__(self, widget, interval_ms: int) -> None:
        self._widget = widget
        self._interval_ms = interval_ms
        self._pending: queue.SimpleQueue = queue.SimpleQueue()
        self._after_id = widget.after(interval_ms, self._run_pending)

    def post(self, callback: Callable[[], object]) -> None:
        self._pending.put(callback)

    def _run_pending(self) -> None:
        self._after_id = self._widget.after(self._interval_ms, self._run_pending)
        for callback in queued_now(self._pending):
            callback()

    def close(self) -> None:
        self._widget.after_cancel(self._after_id)

import queue
import threading
from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.queues import MainThreadCalls, drain, replace_pending


def test_replace_pending_keeps_only_the_newest_command() -> None:
    pending: queue.Queue = queue.Queue(maxsize=1)
    replace_pending(pending, "first")
    replace_pending(pending, "second")

    assert pending.get_nowait() == "second"
    assert pending.empty()


def test_drain_empties_the_queue() -> None:
    pending: queue.Queue = queue.Queue()
    for item in range(3):
        pending.put(item)

    drain(pending)

    assert pending.empty()


def test_main_thread_calls_run_worker_posts_on_the_tk_thread() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        calls = MainThreadCalls(root, 1)
        ran_on: list[threading.Thread] = []
        worker = threading.Thread(target=lambda: calls.post(lambda: ran_on.append(threading.current_thread())))
        worker.start()
        worker.join()
        while not ran_on:
            root.update()
        assert ran_on == [threading.main_thread()]

        calls.close()
        calls.post(lambda: ran_on.append(threading.current_thread()))
        root.after(20, root.quit)
        root.mainloop()
        assert len(ran_on) == 1
    finally:
        root.destroy()

import queue
import threading

from jasna.gui.queues import MainThreadCalls, queued_now, replace_pending


def test_queued_now_takes_what_is_queued_and_stops_without_waiting() -> None:
    pending: queue.Queue = queue.Queue()
    for event in ("loaded", "frame"):
        pending.put(event)

    taken = queued_now(pending)

    assert next(taken) == "loaded"
    pending.put("ended")
    assert list(taken) == ["frame", "ended"]
    assert pending.empty()


def test_replace_pending_keeps_only_the_newest_command() -> None:
    pending: queue.Queue = queue.Queue(maxsize=1)
    replace_pending(pending, "first")
    replace_pending(pending, "second")

    assert pending.get_nowait() == "second"
    assert pending.empty()


def test_main_thread_calls_run_worker_posts_on_the_tk_thread(tk_root) -> None:
    calls = MainThreadCalls(tk_root, 1)
    ran_on: list[threading.Thread] = []
    worker = threading.Thread(target=lambda: calls.post(lambda: ran_on.append(threading.current_thread())))
    worker.start()
    worker.join()
    while not ran_on:
        tk_root.update()
    assert ran_on == [threading.main_thread()]

    scheduled = set(tk_root.tk.splitlist(tk_root.tk.call("after", "info")))
    calls.close()
    still_scheduled = set(tk_root.tk.splitlist(tk_root.tk.call("after", "info")))
    assert still_scheduled < scheduled
    assert len(scheduled - still_scheduled) == 1

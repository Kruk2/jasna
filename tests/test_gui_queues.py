import queue

from jasna.gui.queues import drain, replace_pending


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

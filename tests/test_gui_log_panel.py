from __future__ import annotations

import logging
import threading


from jasna.gui import log_panel as log_panel_module
from jasna.gui.app import GUILogHandler
from jasna.gui.log_panel import LogPanel
from jasna.gui.queues import MainThreadCalls


def test_log_handler_delivers_worker_records_on_the_main_thread(tk_root) -> None:
    panel = LogPanel(tk_root)
    main_thread = MainThreadCalls(tk_root, 50)
    handler = GUILogHandler(main_thread, panel)
    record = logging.LogRecord("jasna", logging.INFO, __file__, 1, "from worker", None, None)

    worker = threading.Thread(target=handler.emit, args=(record,))
    worker.start()
    worker.join()
    assert not panel._entries

    main_thread._run_pending()
    assert [(level, message) for _, level, message in panel._entries] == [("INFO", "from worker")]
    main_thread.close()


def test_log_panel_keeps_only_the_newest_entries(tk_root, monkeypatch) -> None:
    monkeypatch.setattr(log_panel_module, "_MAX_LOG_ENTRIES", 3)
    panel = LogPanel(tk_root)
    panel.add_log("INFO", "first\nsecond line")
    panel.add_log("DEBUG", "hidden")
    panel.add_log("INFO", "kept 1")
    panel.add_log("INFO", "kept 2")
    panel.add_log("INFO", "kept 3")

    assert [message for _, _, message in panel._entries] == ["kept 1", "kept 2", "kept 3"]
    shown = panel._log_text.get("1.0", "end-1c").splitlines()
    assert [line.split("INFO")[-1].strip() for line in shown] == ["kept 1", "kept 2", "kept 3"]


def test_gui_logs_reach_the_panel_and_warnings_also_the_terminal(capsys, monkeypatch) -> None:
    from jasna.gui.app import route_gui_logs

    root, jasna_logger = logging.getLogger(), logging.getLogger("jasna")
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(jasna_logger, "handlers", list(jasna_logger.handlers))
    monkeypatch.setattr(jasna_logger, "propagate", jasna_logger.propagate)
    panel: list[str] = []

    class PanelHandler(logging.Handler):
        def emit(self, record):
            panel.append(record.getMessage())

    route_gui_logs(PanelHandler())
    processor_logger = logging.getLogger("jasna.gui.processor")
    for level, message in ((logging.INFO, "Started processing clip.mp4"),
                           (logging.WARNING, "Failed to process clip.mp4: no internet")):
        processor_logger.handle(processor_logger.makeRecord(processor_logger.name, level, __file__, 1, message, None, None))

    assert panel == ["Started processing clip.mp4", "Failed to process clip.mp4: no internet"]
    assert capsys.readouterr().err == "WARNING jasna.gui.processor: Failed to process clip.mp4: no internet\n"

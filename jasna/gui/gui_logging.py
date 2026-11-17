"""Route Python logging into the GUI log panel."""

import logging

from jasna.gui.log_filter import runtime_log_level_for_filter
from jasna.gui.log_panel import LogPanel
from jasna.gui.queues import MainThreadCalls


class GUILogHandler(logging.Handler):
    """Custom logging handler that forwards logs to the GUI log panel."""

    def __init__(self, main_thread: MainThreadCalls, log_panel: LogPanel):
        super().__init__()
        self._main_thread = main_thread
        self._log_panel = log_panel

    def emit(self, record):
        try:
            msg = self.format(record)
            self._main_thread.post(lambda: self._log_panel.add_log(record.levelname, msg))
        except Exception:
            self.handleError(record)


def route_gui_logs(gui_handler: logging.Handler) -> None:
    """Every log goes to the log panel; warnings and errors also go to the terminal Jasna was started from."""
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().handlers = [gui_handler, console]
    jasna_logger = logging.getLogger("jasna")
    jasna_logger.handlers = [gui_handler, console]
    jasna_logger.propagate = False


def install_gui_logging(main_thread: MainThreadCalls, log_panel: LogPanel) -> None:
    """Send all logs to ``log_panel`` and let its filter set the runtime log level."""
    gui_handler = GUILogHandler(main_thread, log_panel)
    gui_handler.setFormatter(logging.Formatter('%(message)s'))
    route_gui_logs(gui_handler)
    root_logger = logging.getLogger()
    jasna_logger = logging.getLogger("jasna")

    def _apply_runtime_log_level(filter_level: str) -> None:
        runtime_level = runtime_log_level_for_filter(filter_level=filter_level)
        root_logger.setLevel(runtime_level)
        jasna_logger.setLevel(runtime_level)

    log_panel.set_filter_changed_callback(_apply_runtime_log_level)
    _apply_runtime_log_level(log_panel.get_filter_level())

"""Physical-pixel to CustomTkinter-logical-pixel conversion for window and widget sizing.

On Windows, Jasna owns DPI handling: ``run_gui()`` calls :func:`activate_static_dpi` before
any CTk window exists. That disables CustomTkinter's automatic per-monitor rescaling (its
poll loop alpha-blinks the window and re-issues ``geometry()`` mid-drag, which flickered and
hung at high display scaling), makes the process system-DPI aware, and sets one static CTk
scaling factor: the system DPI factor, shrunk just enough that the design-minimum window
plus :data:`SCREEN_MARGIN` fits the primary screen. Windows bitmap-stretches the window on
monitors with a different DPI. ``winfo_*`` stays physical while ``geometry()``/``minsize()``
multiply their width/height arguments by the factor and pass the ``+x+y`` offset through
untouched.

Three rules follow, and every helper here exists to keep them straight:

- ``winfo_*`` values and ``+x+y`` offsets are physical.
- Width/height handed to ``geometry()``/``minsize()`` are logical.
- CTk widget options are logical; raw ``tkinter`` pixel options are physical.

Off Windows the factor is exactly 1 and every function here is the identity.
"""

import logging
import math
import sys

import customtkinter as ctk

logger = logging.getLogger(__name__)

SCREEN_MARGIN = (40, 80)


def activate_static_dpi(design_min: tuple[int, int]) -> None:
    """Fix process DPI awareness and one static CTk scaling factor, before any window exists."""
    if sys.platform != "win32":
        return
    ctk.deactivate_automatic_dpi_awareness()
    _windows_set_dpi_aware()
    dpi = _windows_system_dpi()
    screen = _windows_screen_size()
    factor = _static_scaling(dpi, screen, design_min, SCREEN_MARGIN)
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)
    logger.info(
        "Display %dx%d at %d%% scaling, UI scale factor %.2f",
        screen[0], screen[1], round(dpi * 100), factor,
    )


def _static_scaling(
    dpi: float,
    screen: tuple[int, int],
    design_min: tuple[int, int],
    margin: tuple[int, int],
) -> float:
    return min(
        dpi,
        screen[0] / (design_min[0] + margin[0]),
        screen[1] / (design_min[1] + margin[1]),
    )


def _windows_set_dpi_aware() -> None:
    import ctypes

    # A failure here means a manifest already fixed the awareness (first caller wins);
    # the metric reads below are correct either way.
    ctypes.windll.shcore.SetProcessDpiAwareness(1)


def _windows_system_dpi() -> float:
    import ctypes

    return ctypes.windll.user32.GetDpiForSystem() / 96


def _windows_screen_size() -> tuple[int, int]:
    import ctypes

    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def window_scaling(window) -> float:
    return ctk.ScalingTracker.get_window_scaling(window)


def widget_scaling(widget) -> float:
    return ctk.ScalingTracker.get_widget_scaling(widget)


def fit_size(
    size: tuple[int, int],
    screen: tuple[int, int],
    margin: tuple[int, int],
) -> tuple[int, int]:
    return (
        max(1, min(size[0], screen[0] - margin[0])),
        max(1, min(size[1], screen[1] - margin[1])),
    )


def centered_position(
    size: tuple[int, int],
    area: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    x = area[0] + (area[2] - size[0]) // 2
    y = area[1] + (area[3] - size[1]) // 2
    return (
        max(bounds[0], min(x, bounds[0] + bounds[2] - size[0])),
        max(bounds[1], min(y, bounds[1] + bounds[3] - size[1])),
    )


def to_physical(window, width: int, height: int) -> tuple[int, int]:
    scaling = window_scaling(window)
    return round(width * scaling), round(height * scaling)


def to_logical(window, width: int, height: int) -> tuple[int, int]:
    scaling = window_scaling(window)
    return math.floor(width / scaling), math.floor(height / scaling)


def raw_tk_size(widget, logical_pixels: int) -> int:
    """Pixel size for a raw tkinter option, matching CTk's own widget scaling."""
    return int(logical_pixels * widget_scaling(widget))


def raw_tk_font_size(widget, logical_pixels: int) -> int:
    """Negative (pixel-unit) font size for a raw tkinter widget, matching CTk's font scaling."""
    return -abs(round(logical_pixels * widget_scaling(widget)))


def screen_rect(window) -> tuple[int, int, int, int]:
    """Work area of the monitor holding the window, as a physical ``(x, y, w, h)`` rect."""
    if sys.platform == "win32":
        return _windows_monitor_work_area(window.winfo_id())
    return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def _windows_monitor_work_area(hwnd: int) -> tuple[int, int, int, int]:
    import ctypes
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    MONITOR_DEFAULTTONEAREST = 2
    monitor = ctypes.windll.user32.MonitorFromWindow(
        wintypes.HWND(hwnd), wintypes.DWORD(MONITOR_DEFAULTTONEAREST)
    )
    info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
    ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info))
    work = info.rcWork
    return work.left, work.top, work.right - work.left, work.bottom - work.top


def apply_geometry(window, width: int, height: int, x: int, y: int) -> None:
    logical_width, logical_height = to_logical(window, width, height)
    window.geometry(f"{logical_width}x{logical_height}+{x}+{y}")


def apply_minsize(window, logical_width: int, logical_height: int) -> None:
    """Clamp the design minimum to what fits, so the window stays shrinkable at any DPI."""
    rect = screen_rect(window)
    screen = to_logical(window, rect[2], rect[3])
    window.minsize(*fit_size((logical_width, logical_height), screen, SCREEN_MARGIN))


def place_centered_on_screen(window, width: int, height: int) -> None:
    rect = screen_rect(window)
    size = fit_size((width, height), rect[2:], to_physical(window, *SCREEN_MARGIN))
    x, y = centered_position(size, rect, rect)
    apply_geometry(window, *size, x, y)


def place_centered_on_parent(window, parent, width: int, height: int) -> None:
    rect = screen_rect(parent)
    size = fit_size((width, height), rect[2:], to_physical(window, *SCREEN_MARGIN))
    area = (parent.winfo_x(), parent.winfo_y(), parent.winfo_width(), parent.winfo_height())
    x, y = centered_position(size, area, rect)
    apply_geometry(window, *size, x, y)

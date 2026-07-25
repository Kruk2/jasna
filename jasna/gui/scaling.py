"""Physical-pixel to CustomTkinter-logical-pixel conversion for window and widget sizing.

CustomTkinter makes the process DPI aware on Windows, so every ``winfo_*`` value is in
physical pixels, while ``CTk.geometry()``/``minsize()``/``maxsize()`` multiply their
width/height arguments by the window scaling factor and pass the ``+x+y`` offset through
untouched. CTk widget options are multiplied by the widget scaling factor; raw ``tkinter``
widgets are not scaled at all.

Three rules follow, and every helper here exists to keep them straight:

- ``winfo_*`` values and ``+x+y`` offsets are physical.
- Width/height handed to ``geometry()``/``minsize()`` are logical.
- CTk widget options are logical; raw ``tkinter`` pixel options are physical.

Off Windows and macOS CustomTkinter reports a DPI factor of exactly 1 and Jasna never calls
``set_widget_scaling``/``set_window_scaling``, so every function here is the identity.
"""

import math

import customtkinter as ctk

SCREEN_MARGIN = (40, 80)


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
    screen: tuple[int, int],
) -> tuple[int, int]:
    x = area[0] + (area[2] - size[0]) // 2
    y = area[1] + (area[3] - size[1]) // 2
    return (
        max(0, min(x, screen[0] - size[0])),
        max(0, min(y, screen[1] - size[1])),
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


def screen_size(window) -> tuple[int, int]:
    return window.winfo_screenwidth(), window.winfo_screenheight()


def apply_geometry(window, width: int, height: int, x: int, y: int) -> None:
    logical_width, logical_height = to_logical(window, width, height)
    window.geometry(f"{logical_width}x{logical_height}+{x}+{y}")


def apply_minsize(window, logical_width: int, logical_height: int) -> None:
    """Clamp the design minimum to what fits, so the window stays shrinkable at any DPI."""
    screen = to_logical(window, *screen_size(window))
    window.minsize(*fit_size((logical_width, logical_height), screen, SCREEN_MARGIN))


def place_centered_on_screen(window, width: int, height: int) -> None:
    screen = screen_size(window)
    size = fit_size((width, height), screen, to_physical(window, *SCREEN_MARGIN))
    x, y = centered_position(size, (0, 0, *screen), screen)
    apply_geometry(window, *size, x, y)


def place_centered_on_parent(window, parent, width: int, height: int) -> None:
    screen = screen_size(window)
    size = fit_size((width, height), screen, to_physical(window, *SCREEN_MARGIN))
    area = (parent.winfo_x(), parent.winfo_y(), parent.winfo_width(), parent.winfo_height())
    x, y = centered_position(size, area, screen)
    apply_geometry(window, *size, x, y)

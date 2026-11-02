"""Windows HiDPI regression tests for issue #241.

CustomTkinter multiplies the width/height handed to geometry()/minsize() by the DPI
factor, while winfo_* reports physical pixels. Measuring in physical pixels and feeding
the result back into geometry() therefore double-counts the factor, which produced a
window taller than the screen with an un-shrinkable minimum on Windows.
"""
from __future__ import annotations

from types import SimpleNamespace

import customtkinter as ctk
import pytest

from jasna.gui import scaling
from jasna.gui.icons import CompactSwitch, NativeIconButton
from jasna.gui.settings_sections.widgets import create_slider_value_label
from jasna.gui.theme import Colors, Fonts


def _physical_size(geometry_string: str, factor: float) -> tuple[int, int]:
    width, height = geometry_string.split("+")[0].split("x")
    return round(int(width) * factor), round(int(height) * factor)


def _resize_main_window(app, monkeypatch, *, screen=None, factor=None) -> tuple[str, tuple[int, int]]:
    """Re-run the main window's sizing and return the logical geometry and minimum it requested.

    The window manager only reports a realized geometry once the window is mapped, so the
    requests are captured instead of read back from winfo_*/wm_geometry.
    """
    if screen is not None:
        monkeypatch.setattr(scaling, "screen_rect", lambda _window: (0, 0, *screen))
    if factor is not None:
        monkeypatch.setattr(scaling, "window_scaling", lambda _window: factor)
    requested: list[str] = []
    minimum: list[tuple[int, int]] = []
    monkeypatch.setattr(app, "geometry", requested.append)
    monkeypatch.setattr(app, "minsize", lambda width, height: minimum.append((width, height)))
    app._size_and_center()
    return requested[-1], minimum[-1]


def test_fit_size_caps_an_oversized_window():
    assert scaling.fit_size((1800, 1320), (1920, 1080), (60, 120)) == (1800, 960)


def test_fit_size_leaves_a_window_that_already_fits():
    assert scaling.fit_size((1200, 880), (1920, 1080), (40, 80)) == (1200, 880)


def test_centered_position_keeps_an_oversized_window_on_screen():
    x, y = scaling.centered_position((3000, 2000), (0, 0, 1920, 1080), (0, 0, 1920, 1080))

    assert (x, y) == (0, 0)


def test_centered_position_never_overflows_the_screen():
    x, y = scaling.centered_position((400, 300), (1800, 1000, 400, 300), (0, 0, 1920, 1080))

    assert x + 400 <= 1920
    assert y + 300 <= 1080


def test_centered_position_clamps_into_the_given_monitor_rect():
    bounds = (2560, 0, 2560, 1440)

    assert scaling.centered_position((400, 300), (5000, 1300, 400, 300), bounds) == (4720, 1140)
    assert scaling.centered_position((400, 300), (2400, -100, 200, 100), bounds) == (2560, 0)


def test_screen_rect_falls_back_to_the_full_screen_off_windows(monkeypatch):
    monkeypatch.setattr(scaling.sys, "platform", "linux")
    window = SimpleNamespace(winfo_screenwidth=lambda: 1920, winfo_screenheight=lambda: 1080)

    assert scaling.screen_rect(window) == (0, 0, 1920, 1080)


@pytest.mark.parametrize("dpi", [2.5, 3.0])
def test_static_scaling_shrinks_when_design_minimum_does_not_fit(dpi) -> None:
    factor = scaling._static_scaling(dpi, (2560, 1440), (900, 580), (40, 80))

    assert factor < dpi
    assert (900 + 40) * factor <= 2560
    assert (580 + 80) * factor <= 1440


def test_static_scaling_is_identity_when_design_minimum_fits() -> None:
    assert scaling._static_scaling(2.25, (3840, 2160), (900, 580), (40, 80)) == 2.25
    assert scaling._static_scaling(1.0, (1920, 1080), (900, 580), (40, 80)) == 1.0


@pytest.mark.parametrize("dpi", [2.5, 3.0])
def test_main_window_keeps_design_minimum_at_high_dpi_on_1440p(dpi, jasna_app, monkeypatch) -> None:
    factor = scaling._static_scaling(dpi, (2560, 1440), (900, 580), scaling.SCREEN_MARGIN)

    requested, minimum = _resize_main_window(jasna_app, monkeypatch, screen=(2560, 1440), factor=factor)

    width, height = (int(value) for value in requested.split("+")[0].split("x"))
    # to_logical floors, so the realized size may fall one logical pixel short.
    assert width >= 899 and height >= 579
    assert minimum[0] >= 899 and minimum[1] >= 579


@pytest.mark.parametrize(("raw_xft_dpi", "factor"), [("192", 2.0), ("144", 1.5), ("", 1.0)])
def test_xft_dpi_factor(raw_xft_dpi, factor) -> None:
    assert scaling.xft_dpi_factor(raw_xft_dpi) == factor


def test_activate_static_dpi_is_a_no_op_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(scaling.sys, "platform", "darwin")
    monkeypatch.setattr(ctk.ScalingTracker, "deactivate_automatic_dpi_awareness", False)

    scaling.activate_static_dpi((900, 580))

    assert not ctk.ScalingTracker.deactivate_automatic_dpi_awareness
    assert ctk.ScalingTracker.widget_scaling == 1
    assert ctk.ScalingTracker.window_scaling == 1


def test_activate_static_dpi_windows_path(monkeypatch) -> None:
    monkeypatch.setattr(scaling.sys, "platform", "win32")
    monkeypatch.setattr(scaling, "_windows_set_dpi_aware", lambda: None)
    monkeypatch.setattr(scaling, "_windows_system_dpi", lambda: 2.5)
    monkeypatch.setattr(scaling, "_windows_screen_size", lambda: (2560, 1440))
    monkeypatch.setattr(ctk.ScalingTracker, "deactivate_automatic_dpi_awareness", False)
    try:
        scaling.activate_static_dpi((900, 580))

        expected = scaling._static_scaling(2.5, (2560, 1440), (900, 580), scaling.SCREEN_MARGIN)
        assert ctk.ScalingTracker.deactivate_automatic_dpi_awareness
        assert ctk.ScalingTracker.widget_scaling == expected
        assert ctk.ScalingTracker.window_scaling == expected
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


def test_scaling_helpers_are_identity_without_scaling(tk_root):
    assert scaling.widget_scaling(tk_root) == 1.0
    assert scaling.window_scaling(tk_root) == 1.0
    assert scaling.raw_tk_size(tk_root, 42) == 42
    assert scaling.raw_tk_font_size(tk_root, 15) == -15
    assert scaling.to_physical(tk_root, 1200, 880) == (1200, 880)
    assert scaling.to_logical(tk_root, 1200, 880) == (1200, 880)


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5, 2.0, 3.0], indirect=True)
def test_main_window_fits_screen_at_hidpi(hidpi, jasna_app, monkeypatch) -> None:
    requested, _minimum = _resize_main_window(jasna_app, monkeypatch)

    screen_w, screen_h = jasna_app.winfo_screenwidth(), jasna_app.winfo_screenheight()
    width, height = _physical_size(requested, hidpi)
    assert width <= screen_w and height <= screen_h

    # wm_minsize bypasses CTk's minsize override and reports physical pixels.
    min_w, min_h = jasna_app.wm_minsize()
    assert min_w <= screen_w and min_h <= screen_h
    assert min_w <= width and min_h <= height


@pytest.mark.parametrize("screen", [(1280, 720), (1366, 768), (1920, 1080)])
@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_main_window_fits_a_small_screen(screen, factor, jasna_app, monkeypatch) -> None:
    requested, minimum = _resize_main_window(jasna_app, monkeypatch, screen=screen, factor=factor)

    _, x, y = requested.split("+")
    width, height = _physical_size(requested, factor)
    min_width, min_height = (round(value * factor) for value in minimum)
    assert width <= screen[0] and height <= screen[1]
    assert int(x) + width <= screen[0] and int(y) + height <= screen[1]
    assert int(x) >= 0 and int(y) >= 0
    assert min_width <= width and min_height <= height


def test_control_bar_stays_visible_in_a_short_window(jasna_app) -> None:
    jasna_app.wm_minsize(1, 1)
    jasna_app.geometry("900x360")
    jasna_app.update()

    assert jasna_app._control_bar.winfo_ismapped()
    control_bar_bottom = jasna_app._control_bar.winfo_rooty() + jasna_app._control_bar.winfo_height()
    assert control_bar_bottom <= jasna_app.winfo_rooty() + jasna_app.winfo_height()


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_compact_switch_matches_a_ctk_sibling_at_hidpi(hidpi, tk_root) -> None:
    switch = CompactSwitch(tk_root, lambda: None, Colors.BG_CARD)
    reference = ctk.CTkLabel(tk_root, text="", width=40, height=24)
    tk_root.update()

    assert switch.winfo_reqwidth() == reference.winfo_reqwidth()
    assert switch.winfo_reqheight() == reference.winfo_reqheight()


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_native_icon_button_scales_at_hidpi(hidpi, tk_root) -> None:
    button = NativeIconButton(
        tk_root,
        "save",
        18,
        Colors.TEXT_PRIMARY,
        Colors.BG_PANEL,
        Colors.BG_CARD,
        Colors.BORDER_LIGHT,
        lambda: None,
        32,
        32,
    )
    tk_root.update()

    assert int(button.cget("width")) == int(32 * hidpi)
    assert int(button.cget("height")) == int(32 * hidpi)


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_slider_value_label_font_scales_at_hidpi(hidpi, tk_root) -> None:
    label = create_slider_value_label(tk_root, "90", Colors.BG_PANEL)

    expected_size = round(Fonts.SIZE_NORMAL * hidpi)
    assert tk_root.tk.splitlist(label.cget("font")) == (Fonts.FAMILY, f"-{expected_size}")


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5], indirect=True)
def test_wizard_button_is_on_screen_at_hidpi(hidpi, tk_root, monkeypatch, show_window) -> None:
    from jasna.gui import wizard as wizard_module

    monkeypatch.setattr(wizard_module, "run_system_checks", lambda results: None)
    tk_root.geometry("1000x700+0+0")
    tk_root.update()
    wizard = wizard_module.FirstRunWizard(tk_root)
    show_window(wizard)

    assert wizard.winfo_height() <= wizard.winfo_screenheight()
    assert wizard._continue_btn.winfo_ismapped()
    button_bottom = wizard._continue_btn.winfo_rooty() + wizard._continue_btn.winfo_height()
    assert button_bottom <= wizard.winfo_screenheight()

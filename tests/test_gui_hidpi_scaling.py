"""Windows HiDPI regression tests for issue #241.

CustomTkinter multiplies the width/height handed to geometry()/minsize() by the DPI
factor, while winfo_* reports physical pixels. Measuring in physical pixels and feeding
the result back into geometry() therefore double-counts the factor, which produced a
window taller than the screen with an un-shrinkable minimum on Windows.
"""
from __future__ import annotations

import threading
from tkinter import TclError
from types import SimpleNamespace

import customtkinter as ctk
import pytest

from jasna.gui import scaling
from jasna.gui.app import JasnaApp
from jasna.gui.icons import CompactSwitch, NativeIconButton
from jasna.gui.settings_sections.widgets import create_slider_value_label
from jasna.gui.theme import Colors, Fonts


def _record_geometry(window, monkeypatch) -> list[str]:
    """Capture the logical geometry strings handed to CTk, which scales them by the DPI factor.

    The window manager only reports a realized geometry once the window is mapped, so
    asserting on winfo_*/wm_geometry here would be timing-dependent.
    """
    requested: list[str] = []
    monkeypatch.setattr(window, "geometry", requested.append)
    return requested


def _physical_size(geometry_string: str, factor: float) -> tuple[int, int]:
    width, height = geometry_string.split("+")[0].split("x")
    return round(int(width) * factor), round(int(height) * factor)


def _root() -> ctk.CTk:
    try:
        return ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")


def _stub_main_body_callbacks(root) -> None:
    root._on_jobs_changed = lambda: None
    root._open_interactive_image_restore = lambda: None
    root._processor = None
    root._set_preview_gpu_busy = lambda _busy: None
    root._on_output_changed = lambda *_args: None
    root.TkdndVersion = None


def _stub_footer_callbacks(root) -> None:
    root._on_start = lambda: None
    root._on_stop = lambda: None
    root._toggle_logs = lambda: None
    root._update_start_button_state = lambda: None


def test_fit_size_caps_an_oversized_window():
    assert scaling.fit_size((1800, 1320), (1920, 1080), (60, 120)) == (1800, 960)


def test_fit_size_leaves_a_window_that_already_fits():
    assert scaling.fit_size((1200, 880), (1920, 1080), (40, 80)) == (1200, 880)


def test_centered_position_keeps_an_oversized_window_on_screen():
    x, y = scaling.centered_position((3000, 2000), (0, 0, 1920, 1080), (1920, 1080))

    assert (x, y) == (0, 0)


def test_centered_position_never_overflows_the_screen():
    x, y = scaling.centered_position((400, 300), (1800, 1000, 400, 300), (1920, 1080))

    assert x + 400 <= 1920
    assert y + 300 <= 1080


def test_scaling_helpers_are_identity_without_scaling():
    root = _root()
    try:
        assert scaling.widget_scaling(root) == 1.0
        assert scaling.window_scaling(root) == 1.0
        assert scaling.raw_tk_size(root, 42) == 42
        assert scaling.raw_tk_font_size(root, 15) == -15
        assert scaling.to_physical(root, 1200, 880) == (1200, 880)
        assert scaling.to_logical(root, 1200, 880) == (1200, 880)
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5, 2.0, 3.0], indirect=True)
def test_main_window_fits_screen_at_hidpi(hidpi, monkeypatch) -> None:
    root = _root()
    try:
        requested = _record_geometry(root, monkeypatch)
        JasnaApp._size_and_center(root)

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = _physical_size(requested[-1], hidpi)
        assert width <= screen_w and height <= screen_h

        # wm_minsize bypasses CTk's minsize override and reports physical pixels.
        min_w, min_h = root.wm_minsize()
        assert min_w <= screen_w and min_h <= screen_h
        assert min_w <= width and min_h <= height
    finally:
        root.destroy()


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0, 3.0])
def test_segment_editor_fits_screen_at_hidpi(factor, monkeypatch) -> None:
    from jasna.gui import segment_editor
    from jasna.gui.segment_editor import SegmentEditor

    monkeypatch.setattr(segment_editor.scaling, "window_scaling", lambda _window: factor)
    editor = SimpleNamespace(
        winfo_screenwidth=lambda: 2560,
        winfo_screenheight=lambda: 1440,
        update_idletasks=lambda: None,
        geometry=lambda spec: requested.append(spec),
        minsize=lambda width, height: minimum.append((width, height)),
    )
    requested: list[str] = []
    minimum: list[tuple[int, int]] = []

    SegmentEditor._size_and_center(editor)

    width, height = _physical_size(requested[-1], factor)
    assert width <= 2560 and height <= 1440
    min_w, min_h = (round(value * factor) for value in minimum[-1])
    assert min_w <= width and min_h <= height


def test_build_ui_packs_the_footer_before_the_expanding_body() -> None:
    order: list[str] = []
    recorder = SimpleNamespace(
        _build_header=lambda: order.append("header"),
        _build_footer=lambda: order.append("footer"),
        _build_main_body=lambda: order.append("body"),
    )

    JasnaApp._build_ui(recorder)

    assert order.index("footer") < order.index("body")


def test_control_bar_stays_visible_in_a_short_window() -> None:
    root = _root()
    try:
        root.geometry("900x360")
        _stub_footer_callbacks(root)
        _stub_main_body_callbacks(root)

        JasnaApp._build_footer(root)
        JasnaApp._build_main_body(root)
        root.update()

        assert root._control_bar.winfo_ismapped()
        control_bar_bottom = root._control_bar.winfo_rooty() + root._control_bar.winfo_height()
        assert control_bar_bottom <= root.winfo_rooty() + root.winfo_height()
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_compact_switch_matches_a_ctk_sibling_at_hidpi(hidpi) -> None:
    root = _root()
    try:
        switch = CompactSwitch(root, lambda: None, Colors.BG_CARD)
        reference = ctk.CTkLabel(root, text="", width=40, height=24)
        root.update()

        assert switch.winfo_reqwidth() == reference.winfo_reqwidth()
        assert switch.winfo_reqheight() == reference.winfo_reqheight()
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_native_icon_button_scales_at_hidpi(hidpi) -> None:
    root = _root()
    try:
        button = NativeIconButton(
            root,
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
        root.update()

        assert int(button.cget("width")) == int(32 * hidpi)
        assert int(button.cget("height")) == int(32 * hidpi)
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [1.25, 1.5], indirect=True)
def test_slider_value_label_font_scales_at_hidpi(hidpi) -> None:
    root = _root()
    try:
        label = create_slider_value_label(root, "90", 4, Colors.BG_PANEL)

        expected_size = round(Fonts.SIZE_NORMAL * hidpi)
        assert label.cget("font") == f"{Fonts.FAMILY} -{expected_size}"
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5], indirect=True)
def test_wizard_button_is_on_screen_at_hidpi(hidpi, monkeypatch) -> None:
    from jasna.gui import wizard as wizard_module

    monkeypatch.setattr(
        wizard_module.FirstRunWizard, "_start_checks_in_background", lambda _self: None
    )
    root = _root()
    try:
        root.geometry("1000x700+0+0")
        root.update()
        wizard = wizard_module.FirstRunWizard(root)
        root.update()

        assert wizard.winfo_height() <= wizard.winfo_screenheight()
        assert wizard._continue_btn.winfo_ismapped()
        button_bottom = wizard._continue_btn.winfo_rooty() + wizard._continue_btn.winfo_height()
        assert button_bottom <= wizard.winfo_screenheight()
    finally:
        root.destroy()


def test_wizard_close_button_releases_the_user(monkeypatch) -> None:
    from jasna.gui import wizard as wizard_module

    monkeypatch.setattr(
        wizard_module.FirstRunWizard, "_start_checks_in_background", lambda _self: None
    )
    completed: list[tuple[bool, bool]] = []
    root = _root()
    try:
        root.update()
        wizard = wizard_module.FirstRunWizard(
            root, on_complete=lambda can_continue, all_passed: completed.append(
                (can_continue, all_passed)
            )
        )
        root.update()

        wizard.tk.call(wizard.protocol("WM_DELETE_WINDOW"))
        root.update()

        assert not wizard.winfo_exists()
        assert completed == [(False, False)]
    finally:
        root.destroy()


def test_wizard_stops_waiting_on_a_wedged_check_thread(monkeypatch) -> None:
    from jasna.gui import wizard as wizard_module

    monkeypatch.setattr(
        wizard_module.FirstRunWizard, "_start_checks_in_background", lambda _self: None
    )
    root = _root()
    stop = threading.Event()
    wedged = threading.Thread(target=stop.wait, daemon=True)
    wedged.start()
    try:
        root.update()
        wizard = wizard_module.FirstRunWizard(root)
        root.update()
        assert wizard._continue_btn.cget("state") == "disabled"

        wizard._checks_thread = wedged
        wizard._checks_deadline = 0.0
        wizard._poll_checks_thread()
        root.update()

        assert wizard._continue_btn.cget("state") == "normal"
        assert wizard._has_required_failure
    finally:
        stop.set()
        root.destroy()

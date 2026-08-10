from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import customtkinter as ctk
import pytest
from PIL import Image
from tkinter import TclError

from jasna.gui import video_player as video_player_module
from jasna.gui.app import JasnaApp
from jasna.gui.models import AppSettings
from jasna.gui.theme import Colors
from jasna.gui.video_player import (
    VideoPlayerDialog,
    available_player_secondary_restorations,
    format_player_time,
)


def test_format_player_time_handles_minutes_and_hours() -> None:
    assert format_player_time(65) == "1:05"
    assert format_player_time(3661) == "1:01:01"
    assert format_player_time(-5) == "0:00"


@pytest.mark.parametrize(
    ("previous_deadline", "now", "expected_deadline", "expected_delay"),
    [
        (10.0, 10.012, 10.0 + 1 / 30, 22),
        (10.0, 10.040, 10.0 + 2 / 30, 27),
    ],
)
def test_player_tick_schedule_keeps_absolute_frame_cadence(
    previous_deadline: float,
    now: float,
    expected_deadline: float,
    expected_delay: int,
) -> None:
    deadline, delay = video_player_module.next_player_tick(
        previous_deadline,
        now,
    )

    assert deadline == pytest.approx(expected_deadline)
    assert delay == expected_delay


def test_playing_status_reports_buffer_without_redrawing_every_tick(
    monkeypatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr(video_player_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        video_player_module,
        "t",
        lambda key, **values: f"{key}:{values.get('seconds')}",
    )
    dialog = SimpleNamespace(
        _frame_buffer=SimpleNamespace(buffered_ahead=lambda _seconds: 2.34),
        _next_buffer_status_at=0.0,
        _set_status=MagicMock(),
    )

    VideoPlayerDialog._update_playing_buffer_status(dialog, 4.0)
    now[0] = 10.1
    VideoPlayerDialog._update_playing_buffer_status(dialog, 4.1)

    dialog._set_status.assert_called_once_with(
        "player_playing_buffer:2.3",
        Colors.STATUS_COMPLETED,
    )


def test_time_label_skips_unchanged_text() -> None:
    dialog = SimpleNamespace(
        _metadata=SimpleNamespace(duration=60.0),
        _time_label=MagicMock(),
        _last_time_text=None,
    )

    VideoPlayerDialog._update_time_label(dialog, 1.1)
    VideoPlayerDialog._update_time_label(dialog, 1.2)

    dialog._time_label.configure.assert_called_once_with(text="0:01 / 1:00")


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((1920, 1080), (1920, 1080)),
        ((1000, 1000), (1000, 562)),
        ((1000, 400), (711, 400)),
    ],
)
def test_player_view_size_preserves_sixteen_by_nine(
    bounds: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    assert video_player_module.fit_player_view_size(bounds) == expected


def test_player_view_size_uses_source_aspect_ratio() -> None:
    assert video_player_module.fit_player_view_size(
        (1920, 1080),
        (4, 3),
    ) == (1440, 1080)


def test_player_display_aspect_includes_sample_aspect_ratio() -> None:
    metadata = SimpleNamespace(
        video_width=720,
        video_height=576,
        sample_aspect_ratio=16 / 15,
    )

    assert video_player_module.video_display_aspect(metadata) == (768, 576)


def test_player_dialog_size_maximizes_sixteen_by_nine_view() -> None:
    size = video_player_module.player_dialog_size(
        screen_size=(1920, 1080),
        chrome_size=(32, 260),
        screen_margin=(40, 80),
        minimum_size=(820, 620),
    )

    assert size == (1348, 1000)


def test_player_dialog_centers_large_fixed_chrome_layout(monkeypatch) -> None:
    geometry = []
    minimum = []
    master = object()
    dialog = SimpleNamespace()

    monkeypatch.setattr(
        video_player_module.scaling,
        "screen_rect",
        lambda window: (0, 0, 1332, 1060),
    )
    monkeypatch.setattr(
        video_player_module.scaling,
        "to_physical",
        lambda window, width, height: (width, height),
    )
    monkeypatch.setattr(
        video_player_module.scaling,
        "to_logical",
        lambda window, width, height: (width, height),
    )
    monkeypatch.setattr(
        video_player_module.scaling,
        "apply_geometry",
        lambda window, width, height, x, y: geometry.append((width, height, x, y)),
    )
    monkeypatch.setattr(
        video_player_module.scaling,
        "apply_minsize",
        lambda window, width, height: minimum.append((width, height)),
    )

    VideoPlayerDialog._size_and_center(dialog, master)

    assert geometry == [(1292, 979, 20, 40)]
    assert minimum == [(820, 620)]


def test_player_dialog_reapplies_geometry_after_mapping() -> None:
    events = []
    master = object()
    dialog = SimpleNamespace(
        _size_and_center=lambda owner: events.append(("center", owner)),
        deiconify=lambda: events.append(("deiconify",)),
        wait_visibility=lambda: events.append(("visible",)),
        update_idletasks=lambda: events.append(("update",)),
    )

    VideoPlayerDialog._show_centered(dialog, master)

    assert events == [
        ("center", master),
        ("deiconify",),
        ("visible",),
        ("center", master),
        ("update",),
    ]


def test_player_reuses_tk_image_for_same_sized_frames(monkeypatch) -> None:
    photos = []

    class Photo:
        def __init__(self, image):
            self.image = image
            self.pasted = []
            photos.append(self)

        def paste(self, image):
            self.pasted.append(image)

    monkeypatch.setattr(video_player_module.ImageTk, "PhotoImage", Photo)
    surface = MagicMock()
    dialog = SimpleNamespace(
        _generation=3,
        _photo=None,
        _photo_size=None,
        _last_frame_image=None,
        _video_surface=surface,
        _surface_size=lambda: (640, 360),
    )
    dialog._display_image = lambda image, size=None: (
        VideoPlayerDialog._display_image(dialog, image, size)
    )
    first = SimpleNamespace(generation=3, image=Image.new("RGB", (640, 360), "red"))
    second = SimpleNamespace(generation=3, image=Image.new("RGB", (640, 360), "blue"))

    VideoPlayerDialog._show_frame(dialog, first)
    VideoPlayerDialog._show_frame(dialog, second)

    assert len(photos) == 1
    assert photos[0].pasted == [second.image]
    surface.configure.assert_called_once_with(image=photos[0], text="")
    assert dialog._last_frame_image is second.image


def test_player_resizes_buffered_windowed_frame_for_fullscreen(monkeypatch) -> None:
    photos = []

    class Photo:
        def __init__(self, image):
            self.image = image
            photos.append(self)

    monkeypatch.setattr(video_player_module.ImageTk, "PhotoImage", Photo)
    dialog = SimpleNamespace(
        _generation=1,
        _photo=None,
        _photo_size=None,
        _last_frame_image=None,
        _video_surface=MagicMock(),
        _surface_size=lambda: (1280, 720),
    )
    dialog._display_image = lambda image, size=None: (
        VideoPlayerDialog._display_image(dialog, image, size)
    )
    frame = SimpleNamespace(
        generation=1,
        image=Image.new("RGB", (640, 360), "red"),
    )

    VideoPlayerDialog._show_frame(dialog, frame)

    assert photos[0].image.size == (1280, 720)


def test_player_fullscreen_toggle_hides_configuration_chrome(monkeypatch) -> None:
    events = []

    class Widget:
        def __init__(self, name):
            self.name = name

        def grid_remove(self):
            events.append((self.name, "hide"))

        def grid(self):
            events.append((self.name, "show"))

        def configure(self, **kwargs):
            events.append((self.name, "configure", kwargs))

        def pack_configure(self, **kwargs):
            events.append((self.name, "pack", kwargs))

        def place_forget(self):
            events.append((self.name, "place_forget"))

    geometry = ["1200x800+100+80"]

    def window_geometry(value=None):
        if value is None:
            return geometry[-1]
        geometry.append(value)

    monkeypatch.setattr(video_player_module, "t", lambda key: key)
    dialog = SimpleNamespace(
        _fullscreen=False,
        _windowed_geometry=None,
        _file_row=Widget("file"),
        _settings_card=Widget("settings"),
        _actions=Widget("actions"),
        _bottom_panel=Widget("bottom"),
        _outer=Widget("outer"),
        _fullscreen_btn=Widget("button"),
        _fullscreen_controls_visible=False,
        geometry=window_geometry,
        attributes=lambda *args: events.append(("attributes", *args)),
    )
    dialog._exit_fullscreen = lambda event: VideoPlayerDialog._exit_fullscreen(
        dialog,
        event,
    )
    dialog._hide_fullscreen_controls = lambda: (
        VideoPlayerDialog._hide_fullscreen_controls(dialog)
    )

    VideoPlayerDialog._toggle_fullscreen(dialog)
    VideoPlayerDialog._toggle_fullscreen(dialog)

    assert geometry == ["1200x800+100+80", "1200x800+100+80"]
    assert events == [
        ("file", "hide"),
        ("settings", "hide"),
        ("actions", "hide"),
        ("bottom", "hide"),
        ("outer", "pack", {"padx": 0, "pady": 0}),
        ("attributes", "-fullscreen", True),
        ("button", "configure", {"text": "player_exit_fullscreen"}),
        ("bottom", "place_forget"),
        ("attributes", "-fullscreen", False),
        ("file", "show"),
        ("bottom", "configure", {"fg_color": "transparent"}),
        ("bottom", "show"),
        ("settings", "show"),
        ("actions", "show"),
        ("outer", "pack", {"padx": 16, "pady": 16}),
        ("button", "configure", {"text": "player_fullscreen"}),
    ]


def test_fullscreen_controls_show_at_bottom_edge_and_hide_outside() -> None:
    events = []

    class BottomPanel:
        def configure(self, **kwargs):
            events.append(("configure", kwargs))

        def place(self, **kwargs):
            events.append(("place", kwargs))

        def place_forget(self):
            events.append(("hide",))

        def lift(self):
            events.append(("lift",))

        def winfo_rootx(self):
            return 0

        def winfo_rooty(self):
            return 960

        def winfo_width(self):
            return 1920

        def winfo_height(self):
            return 120

    dialog = SimpleNamespace(
        _fullscreen=True,
        _fullscreen_controls_visible=False,
        _bottom_panel=BottomPanel(),
        winfo_rooty=lambda: 0,
        winfo_height=lambda: 1080,
    )
    dialog._show_fullscreen_controls = lambda: (
        VideoPlayerDialog._show_fullscreen_controls(dialog)
    )
    dialog._hide_fullscreen_controls = lambda: (
        VideoPlayerDialog._hide_fullscreen_controls(dialog)
    )

    VideoPlayerDialog._fullscreen_mouse_moved(
        dialog,
        SimpleNamespace(x_root=100, y_root=1079),
    )
    assert dialog._fullscreen_controls_visible

    VideoPlayerDialog._fullscreen_mouse_moved(
        dialog,
        SimpleNamespace(x_root=100, y_root=500),
    )
    assert not dialog._fullscreen_controls_visible
    assert events[-1] == ("hide",)


def test_player_secondary_options_follow_gpu_and_model_availability(
    monkeypatch,
    tmp_path,
) -> None:
    from jasna import accelerator, engine_paths

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda: True)
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_ENC_PATH", tmp_path / "missing.enc")

    assert available_player_secondary_restorations() == ("none", "rtx-super-res")

    engine_paths.UNET4X_ONNX_PATH.touch()
    assert available_player_secondary_restorations() == (
        "none",
        "rtx-super-res",
        "unet-4x",
    )

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda: False)
    assert available_player_secondary_restorations() == ("none",)


def test_header_player_button_is_disabled_while_gpu_is_busy() -> None:
    configurations: list[dict[str, str]] = []
    app = SimpleNamespace(
        _video_player_btn=SimpleNamespace(
            configure=lambda **kwargs: configurations.append(kwargs)
        ),
        _preview_gpu_busy=False,
        _processor=SimpleNamespace(is_running=lambda: False),
    )

    JasnaApp._update_video_player_button_state(app)
    app._preview_gpu_busy = True
    JasnaApp._update_video_player_button_state(app)

    assert configurations == [
        {
            "state": "normal",
            "fg_color": Colors.PLAYER,
            "border_color": Colors.PLAYER_BORDER,
        },
        {
            "state": "disabled",
            "fg_color": Colors.BG_CARD,
            "border_color": Colors.BORDER_LIGHT,
        },
    ]


def test_video_player_dialog_starts_in_file_picker_state() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    closed: list[bool] = []
    try:
        dialog = VideoPlayerDialog(
            root,
            AppSettings(),
            on_closed=lambda: closed.append(True),
        )
        root.update()

        assert not hasattr(dialog, "_start_btn")
        assert dialog._play_btn.cget("state") == "disabled"
        assert dialog._choose_btn.cget("state") == "normal"
        assert dialog._choose_btn.master is dialog._actions
        assert not hasattr(dialog, "_change_btn")
        assert int(dialog._secondary.grid_info()["row"]) == 0
        assert int(dialog._secondary.grid_info()["column"]) == 3

        dialog.request_close()
        root.update()
        assert closed == [True]
    finally:
        root.destroy()


def test_initial_path_uses_shared_probe_flow_without_autoplay(monkeypatch) -> None:
    path = Path("/tmp/queued.mp4")
    metadata = SimpleNamespace(duration=60.0)
    probed = MagicMock()

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(video_player_module, "get_video_meta_data", lambda value: metadata)
    monkeypatch.setattr(video_player_module.threading, "Thread", ImmediateThread)
    dialog = SimpleNamespace(
        _path=None,
        _metadata=object(),
        _probe_generation=0,
        _file_label=MagicMock(),
        _video_surface=MagicMock(),
        _photo=object(),
        _photo_size=(1, 1),
        _last_frame_image=object(),
        _play_btn=MagicMock(),
        _set_status=MagicMock(),
        _ui_after=lambda callback: callback(),
        _video_probed=probed,
    )

    VideoPlayerDialog._load_path(dialog, path)

    assert dialog._path == path
    dialog._file_label.configure.assert_called_once()
    dialog._play_btn.configure.assert_called_once_with(state="disabled", text="▶")
    probed.assert_called_once_with(1, metadata, "")


def test_queue_player_path_uses_existing_busy_guard(monkeypatch) -> None:
    created = MagicMock()
    monkeypatch.setattr(video_player_module, "VideoPlayerDialog", created)
    settings = MagicMock()
    app = SimpleNamespace(
        _preview_gpu_busy=False,
        _processor=None,
        _set_preview_gpu_busy=MagicMock(),
        _settings_panel=SimpleNamespace(get_settings=settings),
        _video_player_closed=MagicMock(),
    )
    path = Path("/tmp/queued.mp4")

    JasnaApp._open_video_player(app, path)

    created.assert_called_once_with(
        app,
        settings.return_value,
        initial_path=path,
        on_closed=app._video_player_closed,
    )
    app._set_preview_gpu_busy.assert_called_once_with(True)

    app._preview_gpu_busy = True
    JasnaApp._open_video_player(app)
    assert created.call_count == 1

    app._preview_gpu_busy = False
    app._processor = MagicMock()
    app._processor.is_running.return_value = True
    JasnaApp._open_video_player(app, path)
    assert created.call_count == 1


def test_play_button_starts_selected_video_before_worker_exists() -> None:
    start_playback = MagicMock()
    dialog = SimpleNamespace(
        _clock=None,
        _frame_buffer=None,
        _metadata=SimpleNamespace(),
        _start_playback=start_playback,
    )

    VideoPlayerDialog._toggle_play(dialog)

    start_playback.assert_called_once_with()


def test_successful_video_probe_marks_surface_ready(monkeypatch) -> None:
    monkeypatch.setattr(video_player_module, "t", lambda key: key)
    metadata = SimpleNamespace(duration=60.0)
    dialog = SimpleNamespace(
        _probe_generation=2,
        _closed=False,
        _metadata=None,
        _seek=MagicMock(),
        _play_btn=MagicMock(),
        _video_surface=MagicMock(),
        _video_area=SimpleNamespace(
            winfo_width=lambda: 1280,
            winfo_height=lambda: 720,
        ),
        _fit_video_surface=MagicMock(),
        _update_time_label=MagicMock(),
        _set_status=MagicMock(),
    )

    VideoPlayerDialog._video_probed(dialog, 2, metadata, "")

    dialog._video_surface.configure.assert_called_once_with(
        image="",
        text="player_ready_to_play",
    )
    dialog._play_btn.configure.assert_called_once_with(state="normal")


def test_keyboard_controls_toggle_and_seek_thirty_seconds() -> None:
    seek_to = MagicMock()
    toggle_play = MagicMock()
    dialog = SimpleNamespace(
        _worker=object(),
        _metadata=SimpleNamespace(duration=100.0),
        _current_seconds=45.0,
        _seek_to=seek_to,
        _toggle_play=toggle_play,
    )

    assert VideoPlayerDialog._seek_relative(dialog, 30.0) == "break"
    seek_to.assert_called_once_with(75.0)
    assert VideoPlayerDialog._space_pressed(dialog) == "break"
    toggle_play.assert_called_once_with()


def test_choose_video_stops_active_worker_before_opening_picker() -> None:
    begin_stop = MagicMock()
    choose_video = MagicMock()
    dialog = SimpleNamespace(
        _worker=object(),
        _choose_after_stop=False,
        _begin_stop=begin_stop,
        _choose_video=choose_video,
    )

    VideoPlayerDialog._choose_video_action(dialog)

    assert dialog._choose_after_stop
    begin_stop.assert_called_once_with()
    choose_video.assert_not_called()


def test_player_setting_change_reloads_at_current_position() -> None:
    worker = MagicMock()
    worker.reload_from.return_value = 7
    clock = MagicMock()
    settings = AppSettings(secondary_restoration="rtx-super-res")
    dialog = SimpleNamespace(
        _worker=worker,
        _clock=clock,
        _stopping=False,
        _current_seconds=42.5,
        _desired_playing=True,
        _playing=True,
        _buffering=False,
        _eof=True,
        _generation=3,
        _aligned_generation=3,
        _playback_settings=MagicMock(return_value=settings),
        _play_btn=MagicMock(),
        _set_status=MagicMock(),
    )

    VideoPlayerDialog._request_pipeline_reload(dialog)

    worker.reload_from.assert_called_once_with(settings, 42.5)
    clock.pause.assert_called_once_with()
    clock.seek.assert_called_once_with(42.5)
    clock.close.assert_not_called()
    assert dialog._generation == 7
    assert dialog._aligned_generation == -1
    assert not dialog._playing
    assert dialog._buffering
    assert not dialog._eof

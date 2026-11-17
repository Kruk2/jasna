from __future__ import annotations

import queue
import threading
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import pytest
from PIL import Image

from factories import video_metadata
from jasna.gui import raw_player
from jasna.gui import video_player as video_player_module
from jasna.gui import windows_playback
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.raw_player import PlayerFrame
from jasna.gui.player_settings_card import available_player_secondary_restorations
from jasna.gui.video_player import VideoPlayerDialog, format_player_time


def test_format_player_time_handles_minutes_and_hours() -> None:
    assert format_player_time(65) == "1:05"
    assert format_player_time(3661) == "1:01:01"
    assert format_player_time(-5) == "0:00"


@pytest.mark.parametrize(
    ("previous_deadline", "now", "expected_deadline", "expected_delay"),
    [
        (10.0, 10.005, 10.0 + 1 / 60, 12),
        (10.0, 10.020, 10.0 + 2 / 60, 14),
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


def test_windows_player_timer_resolution_is_scoped(monkeypatch) -> None:
    calls = []
    timer = SimpleNamespace(
        timeBeginPeriod=lambda milliseconds: (
            calls.append(("begin", milliseconds)) or 0
        ),
        timeEndPeriod=lambda milliseconds: calls.append(("end", milliseconds)) or 0,
    )
    monkeypatch.setattr(windows_playback.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_playback,
        "_load_windows_multimedia_timer",
        lambda: timer,
    )

    resolution = windows_playback.WindowsTimerResolution()
    resolution.close()
    resolution.close()

    assert calls == [("begin", 1), ("end", 1)]


def test_player_timer_resolution_does_nothing_off_windows(monkeypatch) -> None:
    load_timer = MagicMock()
    monkeypatch.setattr(windows_playback.sys, "platform", "linux")
    monkeypatch.setattr(
        windows_playback,
        "_load_windows_multimedia_timer",
        load_timer,
    )

    resolution = windows_playback.WindowsTimerResolution()
    resolution.close()

    load_timer.assert_not_called()


def test_native_video_renderer_is_windows_only(monkeypatch) -> None:
    monkeypatch.setattr(windows_playback.sys, "platform", "linux")

    assert windows_playback.create_native_video_renderer(object()) is None


class _FakePlayerWorker:
    """Stands in for the decode/restore worker: records the positions it is asked to play from."""

    def __init__(self, path, metadata, settings, frame_buffer, *, max_size, on_stopped):
        self.settings = settings
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.positions: list[float] = []
        self.closed = False
        self._on_stopped = on_stopped

    def start(self) -> None:
        pass

    def play_from(self, seconds: float) -> int:
        self.positions.append(seconds)
        return len(self.positions)

    def reload_from(self, settings: AppSettings, seconds: float) -> int:
        self.settings = settings
        return self.play_from(seconds)

    def set_max_size(self, size) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self._on_stopped()


_METADATA = video_metadata(fps=24, num_frames=2400, video_width=1920, video_height=1080)


@pytest.fixture
def player(tk_root, monkeypatch):
    """A real player dialog whose probe returns ``_METADATA`` and whose worker is a fake."""
    monkeypatch.setattr(raw_player, "get_video_meta_data", lambda _path: _METADATA)
    monkeypatch.setattr(video_player_module, "RawPlayerWorker", _FakePlayerWorker)
    monkeypatch.setattr(raw_player, "source_has_audio", lambda _path: False)
    closed: list[bool] = []

    def open_player(settings: AppSettings = AppSettings(), initial_path: Path | None = None):
        dialog = VideoPlayerDialog(tk_root, settings, initial_path=initial_path, on_closed=lambda: closed.append(True))
        return dialog, closed

    return open_player


def _run_posted_work(dialog) -> None:
    """Finish the probe thread and run what it posted back to the Tk thread."""
    for thread in threading.enumerate():
        if thread.name.startswith("player-probe-"):
            thread.join()
    dialog._main_thread._run_pending()


def _playing_dialog(player, settings: AppSettings = AppSettings()):
    dialog, closed = player(settings, Path("/videos/clip.mp4"))
    _run_posted_work(dialog)
    dialog._play_btn.invoke()
    return dialog, dialog._worker, closed


def test_player_starts_in_file_picker_state_and_closes_once(player) -> None:
    dialog, closed = player()

    assert dialog._play_btn.cget("state") == "disabled"
    assert dialog._choose_btn.cget("state") == "normal"
    assert int(dialog._settings_card._secondary.grid_info()["column"]) == 3

    dialog.request_close()
    dialog.request_close()
    assert closed == [True]


def test_player_window_is_centered_on_screen(player) -> None:
    dialog, _closed = player()

    size, x, y = dialog.geometry().split("+")
    width, height = (int(value) for value in size.split("x"))
    assert abs(int(x) - (dialog.winfo_screenwidth() - width) // 2) <= 1
    assert abs(int(y) - (dialog.winfo_screenheight() - height) // 2) <= 1


def test_initial_path_is_probed_but_not_played(player) -> None:
    dialog, _closed = player(initial_path=Path("/videos/queued.mp4"))

    assert dialog._play_btn.cget("state") == "disabled"
    _run_posted_work(dialog)

    assert dialog._file_label.cget("text") == "queued.mp4"
    assert dialog._video_surface.cget("text") == t("player_ready_to_play")
    assert dialog._play_btn.cget("state") == "normal"
    assert dialog._worker is None


def test_play_starts_the_worker_and_keys_toggle_and_seek(player) -> None:
    dialog, worker, _closed = _playing_dialog(player)

    assert worker.positions == [0.0]
    assert dialog._play_btn.cget("text") == "⏸"
    assert dialog._seek_relative(30.0) == "break"
    assert dialog._space_pressed() == "break"

    assert worker.positions == [0.0, 30.0]
    assert dialog._play_btn.cget("text") == "▶"


def test_setting_change_reloads_at_current_position_with_the_standard_model(player) -> None:
    dialog, worker, _closed = _playing_dialog(player, AppSettings(restoration_model="ltx"))
    dialog._seek_relative(30.0)

    dialog._settings_card._secondary._on_label_selected(t("secondary_none"))

    assert worker.positions[-1] == 30.0
    assert worker.settings.restoration_model == "basicvsrpp"
    assert dialog._status.cget("text") == t("player_loading_models")


def test_choose_video_stops_playback_before_opening_picker(player, monkeypatch) -> None:
    dialog, worker, _closed = _playing_dialog(player)
    picked: list[bool] = []
    monkeypatch.setattr(
        video_player_module.filedialog, "askopenfilename", lambda **_kwargs: picked.append(worker.closed) or ""
    )

    dialog._choose_btn.invoke()
    dialog._main_thread._run_pending()

    assert picked == [True]
    assert dialog._worker is None


def test_closing_while_playing_stops_the_worker_first(player) -> None:
    dialog, worker, closed = _playing_dialog(player)

    dialog.request_close()
    assert worker.closed
    assert closed == []

    dialog._main_thread._run_pending()
    assert closed == [True]


def test_same_sized_frames_reuse_the_scaled_tk_image(player, monkeypatch) -> None:
    monkeypatch.setattr(video_player_module, "create_native_video_renderer", lambda _host: None)
    dialog, _closed = player()
    first = PlayerFrame(0.0, Image.new("RGB", (640, 360), "red"), 0)
    second = PlayerFrame(0.0, Image.new("RGB", (640, 360), "blue"), 0)

    dialog._show_frame(first)
    photo = dialog._photo
    dialog._show_frame(second)

    assert dialog._photo is photo
    assert (photo.width(), photo.height()) == dialog._view_size


def test_fullscreen_hides_chrome_and_shows_controls_at_the_bottom_edge(player) -> None:
    dialog, _closed = player()
    dialog.update()

    dialog._toggle_fullscreen()
    dialog.update()
    assert not dialog._settings_card.winfo_manager()
    assert dialog._fullscreen_btn.cget("text") == t("player_exit_fullscreen")

    bottom = dialog.winfo_rooty() + dialog.winfo_height() - 1
    dialog._fullscreen_mouse_moved(SimpleNamespace(x_root=100, y_root=bottom))
    assert dialog._bottom_panel.winfo_manager() == "place"
    dialog._fullscreen_mouse_moved(SimpleNamespace(x_root=100, y_root=dialog.winfo_rooty()))
    assert not dialog._bottom_panel.winfo_manager()

    dialog._toggle_fullscreen()
    dialog.update()
    assert dialog._settings_card.winfo_manager() == "grid"
    assert dialog._bottom_panel.winfo_manager() == "grid"


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
    metadata = video_metadata(video_width=720, video_height=576, sample_aspect_ratio=Fraction(16, 15))

    assert video_player_module.video_display_aspect(metadata) == (768, 576)


def test_player_dialog_size_maximizes_sixteen_by_nine_view() -> None:
    size = video_player_module.player_dialog_size(
        screen_size=(1920, 1080),
        chrome_size=(32, 260),
        screen_margin=(40, 80),
        minimum_size=(820, 620),
    )

    assert size == (1348, 1000)


def test_player_secondary_options_follow_gpu_and_model_availability(
    monkeypatch,
    tmp_path,
) -> None:
    from jasna import accelerator, engine_paths

    monkeypatch.setattr(accelerator, "vendor_for_device", lambda: accelerator.AcceleratorVendor.NVIDIA)
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_ENC_PATH", tmp_path / "missing.enc")

    assert available_player_secondary_restorations() == ("none", "rtx-super-res")

    engine_paths.UNET4X_ONNX_PATH.touch()
    assert available_player_secondary_restorations() == (
        "none",
        "rtx-super-res",
        "unet-4x",
    )

    monkeypatch.setattr(accelerator, "vendor_for_device", lambda: accelerator.AcceleratorVendor.AMD)
    assert available_player_secondary_restorations() == ("none", "unet-4x")

    monkeypatch.setattr(accelerator, "vendor_for_device", lambda: accelerator.AcceleratorVendor.CPU)
    assert available_player_secondary_restorations() == ("none",)


def test_header_player_button_follows_gpu_use(jasna_app, monkeypatch) -> None:
    button = jasna_app._header_bar._video_player_btn

    jasna_app._set_preview_gpu_busy("preview", True)
    jasna_app._main_thread._run_pending()
    assert button.cget("state") == "disabled"

    jasna_app._set_preview_gpu_busy("preview", False)
    jasna_app._main_thread._run_pending()
    assert button.cget("state") == "normal"

    monkeypatch.setattr(jasna_app._processor, "is_running", lambda: True)
    jasna_app._update_video_player_button_state()
    assert button.cget("state") == "disabled"


def test_queue_player_opens_one_player_and_frees_the_gpu_on_close(jasna_app, monkeypatch) -> None:
    monkeypatch.setattr(raw_player, "get_video_meta_data", lambda _path: _METADATA)

    jasna_app._queue_panel.add_jobs([JobItem(path=Path("/videos/queued.mp4"))])
    jasna_app._queue_panel._job_widgets[0]._handle_play()
    dialog = jasna_app._video_player_dialog
    jasna_app._open_video_player(Path("/videos/other.mp4"))

    assert dialog is not None
    assert jasna_app._video_player_dialog is dialog
    assert jasna_app._preview_gpu_busy

    dialog.request_close()
    assert jasna_app._video_player_dialog is None
    assert not jasna_app._preview_gpu_busy


def test_player_does_not_open_while_processing(jasna_app, monkeypatch) -> None:
    monkeypatch.setattr(jasna_app._processor, "is_running", lambda: True)

    jasna_app._open_video_player(Path("/videos/queued.mp4"))

    assert jasna_app._video_player_dialog is None


def test_player_settings_card_picks_model_threshold_and_falls_back_to_known_choices(tk_root) -> None:
    from jasna.gui.player_settings_card import PlayerSettingsCard
    from jasna.mosaic.detection_registry import detection_model_choices, recommended_score_threshold

    changes: list[bool] = []
    card = PlayerSettingsCard(
        tk_root,
        AppSettings(detection_model="retired-model", secondary_restoration="tvai"),
        lambda: changes.append(True),
    )
    first_model = detection_model_choices()[0]
    assert card.apply_to(AppSettings()).detection_model == first_model
    assert card.apply_to(AppSettings()).secondary_restoration == "none"

    last_model = detection_model_choices()[-1]
    card._model.set(last_model)
    card._model._on_label_selected(last_model)

    settings = card.apply_to(AppSettings())
    assert settings.detection_model == last_model
    assert settings.detection_score_threshold == pytest.approx(recommended_score_threshold(last_model), abs=0.05)
    assert changes == [True]

import pytest

from jasna.gui import tray
from jasna.gui.components import TrayOptionsDialog
from jasna.gui.paths import read_settings, update_settings
from jasna.gui.tray import TraySupport


class FakeTrayIcon:
    """Records what the tray shows; tests click it on their own thread instead of the tray's."""

    def __init__(self, image, title, on_open, on_quit):
        self.title = title
        self.on_open = on_open
        self.on_quit = on_quit
        self.running = False

    def run_detached(self):
        self.running = True

    def stop(self):
        self.running = False


@pytest.fixture
def fake_tray(monkeypatch):
    icons = []
    monkeypatch.setattr(tray, "_create_icon", lambda *args: icons.append(FakeTrayIcon(*args)) or icons[-1])
    monkeypatch.setattr(tray, "tray_support", lambda: TraySupport(available=True, has_menu=True))
    return icons


def test_options_dialog_saves_switches_and_shows_them_again(tk_root, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = TrayOptionsDialog(tk_root, TraySupport(available=True, has_menu=True))
    dialog.switches["minimize_to_tray"]._toggle()
    dialog.destroy()

    assert read_settings() == {"minimize_to_tray": True}
    reopened = TrayOptionsDialog(tk_root, TraySupport(available=True, has_menu=True))
    assert reopened.switches["minimize_to_tray"].get() == 1
    assert reopened.switches["close_to_tray"].get() == 0
    reopened.destroy()


@pytest.mark.parametrize(
    ("support", "minimize_state", "close_state"),
    [
        (TraySupport(available=True, has_menu=True), "normal", "normal"),
        (TraySupport(available=True, has_menu=False), "normal", "disabled"),
        (TraySupport(available=False, has_menu=False), "disabled", "disabled"),
    ],
)
def test_options_dialog_disables_what_the_desktop_cannot_do(tk_root, monkeypatch, tmp_path, support, minimize_state, close_state):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = TrayOptionsDialog(tk_root, support)

    assert dialog.switches["minimize_to_tray"].cget("state") == minimize_state
    assert dialog.switches["close_to_tray"].cget("state") == close_state
    dialog.destroy()


def close_window(app) -> None:
    app.tk.call(app.protocol("WM_DELETE_WINDOW"))


def test_close_hides_to_tray_and_the_icon_brings_the_window_back(jasna_app, fake_tray, show_window):
    update_settings(close_to_tray=True)
    show_window(jasna_app)

    close_window(jasna_app)

    assert jasna_app.state() == "withdrawn"
    [icon] = fake_tray
    assert icon.running

    icon.on_open()
    jasna_app._main_thread._run_pending()

    assert jasna_app.state() == "normal"
    assert not icon.running


def test_quit_from_the_tray_closes_the_app(jasna_app, fake_tray, show_window):
    update_settings(close_to_tray=True)
    show_window(jasna_app)
    close_window(jasna_app)

    [icon] = fake_tray
    icon.on_quit()
    jasna_app._main_thread._run_pending()

    assert not icon.running
    assert not jasna_app.children


@pytest.mark.parametrize(
    ("settings", "support"),
    [
        ({"minimize_to_tray": True}, TraySupport(available=True, has_menu=True)),
        ({"close_to_tray": True}, TraySupport(available=True, has_menu=False)),
        ({"close_to_tray": True}, TraySupport(available=False, has_menu=False)),
    ],
)
def test_close_quits_unless_close_to_tray_can_work(jasna_app, fake_tray, monkeypatch, show_window, settings, support):
    monkeypatch.setattr(tray, "tray_support", lambda: support)
    update_settings(**settings)
    show_window(jasna_app)

    close_window(jasna_app)

    assert fake_tray == []
    assert not jasna_app.children

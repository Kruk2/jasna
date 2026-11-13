import time
import webbrowser

import pytest

from jasna import update_check
from jasna.gui.components import OptionsDialog
from jasna.gui.tray import TraySupport
from jasna.user_settings import read_settings, update_settings

TRAY = TraySupport(available=True, has_menu=True)


def chip_shown(app) -> bool:
    return app._update_chip.winfo_manager() == "pack"


def refuse_to_answer():
    pytest.fail("the update check asked the release service")


def test_a_newer_release_shows_a_header_chip_that_opens_its_release_page(make_jasna_app, settle_activation, monkeypatch):
    monkeypatch.setattr(update_check, "fetch_latest_release", lambda: "99.0.0")
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    app = make_jasna_app()
    settle_activation(app)

    assert chip_shown(app)
    assert "99.0.0" in app._update_chip.cget("text")
    app._update_chip.invoke()
    assert opened == [update_check.release_page_url("99.0.0")]


def test_no_chip_without_a_newer_release(make_jasna_app, settle_activation):
    app = make_jasna_app()
    settle_activation(app)

    assert not chip_shown(app)


def test_the_chip_shows_the_remembered_release_without_asking_again(make_jasna_app, settle_activation, monkeypatch):
    update_settings(update_checked_at=time.time(), latest_release="99.0.0")
    monkeypatch.setattr(update_check, "fetch_latest_release", refuse_to_answer)

    app = make_jasna_app()
    settle_activation(app)

    assert chip_shown(app)


def test_turning_the_check_off_hides_the_chip_and_on_shows_it_again(make_jasna_app, settle_activation, monkeypatch):
    monkeypatch.setattr(update_check, "fetch_latest_release", lambda: "99.0.0")
    app = make_jasna_app()
    settle_activation(app)
    dialog = OptionsDialog(app, TRAY, app._on_update_check_changed)

    dialog.switches["check_for_updates"]._toggle()
    settle_activation(app)
    assert read_settings()["check_for_updates"] is False
    assert not chip_shown(app)

    dialog.switches["check_for_updates"]._toggle()
    settle_activation(app)
    assert read_settings()["check_for_updates"] is True
    assert chip_shown(app)
    dialog.destroy()


def test_the_options_dialog_has_the_update_check_on_by_default(tk_root, user_config_dir):
    changes = []
    dialog = OptionsDialog(tk_root, TRAY, changes.append)

    assert dialog.switches["check_for_updates"].get() == 1
    dialog.switches["check_for_updates"]._toggle()
    dialog.destroy()

    assert changes == [False]
    reopened = OptionsDialog(tk_root, TRAY, changes.append)
    assert reopened.switches["check_for_updates"].get() == 0
    reopened.destroy()

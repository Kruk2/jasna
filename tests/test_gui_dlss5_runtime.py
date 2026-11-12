from __future__ import annotations

from tkinter import messagebox

import pytest

from jasna.gui import dlss5_runtime as dlss5_runtime_module
from jasna.gui.dlss5_runtime import Dlss5Runtime
from jasna.upscale import dlss5


class _NowCalls:
    def post(self, call) -> None:
        call()


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dlss5, "dlss5_runtime_dir", lambda: tmp_path)
    return tmp_path


def _answer(monkeypatch, yes: bool) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(messagebox, "askyesno", lambda _title, message: asked.append(message) or yes)
    return asked


def _fake_download(monkeypatch, runtime_dir, error: str | None) -> list:
    downloads = []

    def start(download, on_percent, on_done):
        downloads.append(download)
        on_percent(40)
        if error is None:
            for name in dlss5.DLSS5_RUNTIME_FILES:
                (runtime_dir / name).touch()
        on_done(error)

    monkeypatch.setattr(dlss5_runtime_module, "start_download", start)
    return downloads


def test_the_runtime_is_downloaded_after_the_user_agrees(monkeypatch, runtime_dir) -> None:
    asked = _answer(monkeypatch, yes=True)
    downloads = _fake_download(monkeypatch, runtime_dir, error=None)
    changes: list[int | None] = []
    runtime = Dlss5Runtime(_NowCalls(), lambda: changes.append(runtime.percent))
    ready: list[bool] = []

    assert not runtime.installed()
    runtime.ensure(on_ready=lambda: ready.append(True))

    assert "467 MB" in asked[0] and dlss5.DLSS5_RELEASE_PAGE in asked[0]
    assert len(downloads) == 1 and changes == [0, 40, None] and ready == [True]
    assert runtime.installed()


def test_nothing_is_downloaded_when_the_user_declines(monkeypatch, runtime_dir) -> None:
    _answer(monkeypatch, yes=False)
    downloads = _fake_download(monkeypatch, runtime_dir, error=None)
    runtime = Dlss5Runtime(_NowCalls(), lambda: None)

    runtime.ensure(on_ready=lambda: pytest.fail("not downloaded"))

    assert downloads == [] and not runtime.downloading


def test_a_failed_download_is_shown_and_the_job_does_not_start(monkeypatch, runtime_dir) -> None:
    _answer(monkeypatch, yes=True)
    _fake_download(monkeypatch, runtime_dir, error="connection reset")
    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda _title, message: errors.append(message))
    runtime = Dlss5Runtime(_NowCalls(), lambda: None)

    runtime.ensure(on_ready=lambda: pytest.fail("download failed"))

    assert len(errors) == 1 and "connection reset" in errors[0]
    assert not runtime.downloading and not runtime.installed()

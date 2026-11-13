from __future__ import annotations

import io
import urllib.error

from PIL import Image
import pytest

from jasna.gui import xianyu_qr


def _png(color: tuple[int, int, int]) -> bytes:
    body = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(body, format="PNG")
    return body.getvalue()


def test_download_returns_the_qr_code_published_on_github(monkeypatch):
    monkeypatch.setattr(xianyu_qr, "urlopen", lambda *args, **kwargs: io.BytesIO(_png((255, 0, 0))))

    assert xianyu_qr.download_xianyu_qr().getpixel((0, 0)) == (255, 0, 0)


def _unreachable(*args, **kwargs):
    raise urllib.error.URLError("no route")


@pytest.mark.parametrize("fake_urlopen", [
    _unreachable,
    lambda *args, **kwargs: io.BytesIO(b"<html>rate limited</html>"),
], ids=["unreachable", "not-an-image"])
def test_download_gives_nothing_when_github_has_no_usable_qr_code(monkeypatch, fake_urlopen):
    monkeypatch.setattr(xianyu_qr, "urlopen", fake_urlopen)

    assert xianyu_qr.download_xianyu_qr() is None


def test_bundled_qr_code_keeps_a_white_quiet_zone():
    image = xianyu_qr.bundled_xianyu_qr()

    assert image.size == (280, 280)
    assert min(min(image.getpixel((x, y))) for x in range(20) for y in range(20)) > 240

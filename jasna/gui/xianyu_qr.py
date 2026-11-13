"""The official Xianyu seller's QR code: the current one on GitHub, or the copy bundled with this build."""

from __future__ import annotations

from collections.abc import Callable
import http.client
import io
import logging
import ssl
import threading
from urllib.request import urlopen

import certifi
from PIL import Image

from jasna.gui.branding import XIANYU_QR_PNG, brand_asset_path

logger = logging.getLogger(__name__)

XIANYU_QR_URL = "https://raw.githubusercontent.com/Kruk2/jasna/main/assets/xianyu-qr.png"
MAX_QR_DOWNLOAD_BYTES = 1 << 20


def bundled_xianyu_qr() -> Image.Image:
    with Image.open(brand_asset_path(XIANYU_QR_PNG)) as source:
        return source.convert("RGB")


def download_xianyu_qr() -> Image.Image | None:
    """The QR code published on GitHub, so a new seller code reaches released builds; None when it can't be fetched."""
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(XIANYU_QR_URL, timeout=10, context=context) as response:
            body = response.read(MAX_QR_DOWNLOAD_BYTES)
        with Image.open(io.BytesIO(body)) as source:
            return source.convert("RGB")
    except (OSError, http.client.HTTPException) as error:
        logger.warning("Showing the bundled Xianyu QR code; the current one could not be downloaded (%s: %s)",
                       type(error).__name__, error)
        return None


def download_xianyu_qr_in_background(on_downloaded: Callable[[Image.Image], object]) -> None:
    """Call ``on_downloaded`` from a worker thread once GitHub's QR code arrives; never when it can't be fetched."""
    def run() -> None:
        image = download_xianyu_qr()
        if image is not None:
            on_downloaded(image)

    threading.Thread(target=run, daemon=True, name="xianyu-qr-download").start()

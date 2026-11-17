from __future__ import annotations

import sys

from PIL import Image


def _load_windows_multimedia_timer():
    import ctypes

    return ctypes.WinDLL("winmm")


class WindowsTimerResolution:
    def __init__(self) -> None:
        self._timer = None
        if sys.platform != "win32":
            return
        timer = _load_windows_multimedia_timer()
        if timer.timeBeginPeriod(1) != 0:
            raise OSError("Windows could not enable 1 ms timer resolution")
        self._timer = timer

    def close(self) -> None:
        if self._timer is None:
            return
        timer = self._timer
        self._timer = None
        if timer.timeEndPeriod(1) != 0:
            raise OSError("Windows could not restore timer resolution")


def _load_windows_video_backend():
    import ctypes

    import cv2
    import numpy as np

    return cv2, np, ctypes.WinDLL("user32", use_last_error=True)


class _WindowsVideoRenderer:
    _SW_HIDE = 0
    _SW_SHOWNA = 8
    _GWL_STYLE = -16
    _WS_CHILD = 0x40000000
    _WS_VISIBLE = 0x10000000

    def __init__(self, host) -> None:
        self._cv2, self._np, self._user32 = _load_windows_video_backend()
        self._title = f"JasnaVideo-{id(self):x}"
        self._cv2.namedWindow(self._title, self._cv2.WINDOW_NORMAL)
        self._hwnd = self._user32.FindWindowW(None, self._title)
        if not self._hwnd:
            self._cv2.destroyWindow(self._title)
            raise OSError("OpenCV did not create a Windows video surface")
        self._user32.ShowWindow(self._hwnd, self._SW_HIDE)
        self._user32.SetParent(self._hwnd, host.winfo_id())
        self._user32.SetWindowLongW(
            self._hwnd,
            self._GWL_STYLE,
            self._WS_CHILD | self._WS_VISIBLE,
        )
        self._user32.EnableWindow(self._hwnd, False)
        self._size: tuple[int, int] | None = None
        self._bgr = None
        self._visible = False

    def display(self, image: Image.Image, size: tuple[int, int]) -> None:
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        rgb = self._np.asarray(image)
        if self._bgr is None or self._bgr.shape != rgb.shape:
            self._bgr = self._np.empty_like(rgb)
        self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR, dst=self._bgr)
        if self._size != size:
            self._user32.MoveWindow(self._hwnd, 0, 0, *size, True)
            self._size = size
        self._cv2.imshow(self._title, self._bgr)
        self._cv2.pollKey()
        if not self._visible:
            self._user32.ShowWindow(self._hwnd, self._SW_SHOWNA)
            self._visible = True

    def hide(self) -> None:
        if self._visible:
            self._user32.ShowWindow(self._hwnd, self._SW_HIDE)
            self._visible = False

    def close(self) -> None:
        if not self._hwnd:
            return
        self._cv2.destroyWindow(self._title)
        self._hwnd = 0


def create_native_video_renderer(host):
    if sys.platform != "win32":
        return None
    return _WindowsVideoRenderer(host)

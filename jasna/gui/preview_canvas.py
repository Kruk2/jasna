"""Image view fitted to its canvas with wheel zoom and drag pan, shared by the segment editor
preview and the mask suggestion dialog."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image, ImageTk

from jasna.gui import scaling
from jasna.gui.theme import Colors, Fonts

PREVIEW_ZOOM_MIN = 1.0
PREVIEW_ZOOM_MAX = 8.0
PREVIEW_ZOOM_STEP = 0.25

Transform = tuple[float, float, float]
"""Image to canvas mapping: ``(scale, offset_x, offset_y)``."""


def fit_scale_and_offset(
    image_size: tuple[int, int], canvas_size: tuple[int, int]
) -> tuple[float, float, float]:
    image_w, image_h = image_size
    canvas_w, canvas_h = canvas_size
    scale = min(canvas_w / image_w, canvas_h / image_h)
    offset_x = (canvas_w - image_w * scale) / 2
    offset_y = (canvas_h - image_h * scale) / 2
    return scale, offset_x, offset_y


def canvas_to_image(
    x: float,
    y: float,
    *,
    view: Transform,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """Map a canvas point to image coords, snapping to the frame edges."""

    scale, offset_x, offset_y = view
    image_x = (x - offset_x) / scale
    image_y = (y - offset_y) / scale
    return (
        min(float(image_size[0]), max(0.0, image_x)),
        min(float(image_size[1]), max(0.0, image_y)),
    )


def image_to_canvas(
    x: float,
    y: float,
    *,
    view: Transform,
) -> tuple[float, float]:
    scale, offset_x, offset_y = view
    return x * scale + offset_x, y * scale + offset_y


def _clamp_center(center: tuple[float, float], zoom: float) -> tuple[float, float]:
    half_visible = 0.5 / zoom
    return (
        min(1.0 - half_visible, max(half_visible, float(center[0]))),
        min(1.0 - half_visible, max(half_visible, float(center[1]))),
    )


@dataclass
class PreviewView:
    """Zoom and pan of a preview; ``center`` is the visible middle as source fractions."""

    zoom: float = PREVIEW_ZOOM_MIN
    center: tuple[float, float] = (0.5, 0.5)

    @property
    def zoomed(self) -> bool:
        return self.zoom > PREVIEW_ZOOM_MIN

    @property
    def is_reset(self) -> bool:
        return not self.zoomed and self.center == (0.5, 0.5)

    def reset(self) -> None:
        self.zoom = PREVIEW_ZOOM_MIN
        self.center = (0.5, 0.5)

    def zoom_to(self, zoom: float, anchor: tuple[float, float] | None) -> bool:
        """Change the zoom, keeping the source point under ``anchor`` (fractions of the shown image) still.

        Returns whether the zoom changed.
        """
        new_zoom = min(PREVIEW_ZOOM_MAX, max(PREVIEW_ZOOM_MIN, float(zoom)))
        if new_zoom == self.zoom:
            return False
        center = _clamp_center(self.center, self.zoom)
        if anchor is not None:
            fx, fy = anchor
            center = (
                center[0] + (fx - 0.5) / self.zoom - (fx - 0.5) / new_zoom,
                center[1] + (fy - 0.5) / self.zoom - (fy - 0.5) / new_zoom,
            )
        self.zoom = new_zoom
        self.center = _clamp_center(center, new_zoom)
        return True

    def pan(self, dx: float, dy: float) -> None:
        """Follow a drag of ``dx``/``dy`` fractions of the shown image."""
        self.center = _clamp_center(
            (self.center[0] - dx / self.zoom, self.center[1] - dy / self.zoom),
            self.zoom,
        )

    def crop_box(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """Source box to show for a ``width`` x ``height`` image, or None when not zoomed."""
        if not self.zoomed:
            self.center = (0.5, 0.5)
            return None
        self.center = _clamp_center(self.center, self.zoom)
        crop_width = max(1, min(width, round(width / self.zoom)))
        crop_height = max(1, min(height, round(height / self.zoom)))
        left = round(self.center[0] * width - crop_width / 2)
        top = round(self.center[1] * height - crop_height / 2)
        left = min(width - crop_width, max(0, left))
        top = min(height - crop_height, max(0, top))
        return left, top, left + crop_width, top + crop_height


def preview_layout(
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    view: PreviewView,
) -> tuple[tuple[int, int, int, int], tuple[int, int], Transform]:
    """Where an image shows on a canvas under ``view``: the source crop box, its display size in
    pixels, and the image to canvas transform of the centered, fitted crop."""
    crop = view.crop_box(*image_size) or (0, 0, *image_size)
    crop_size = (crop[2] - crop[0], crop[3] - crop[1])
    scale, offset_x, offset_y = fit_scale_and_offset(crop_size, canvas_size)
    display_size = (max(2, round(crop_size[0] * scale)), max(2, round(crop_size[1] * scale)))
    return crop, display_size, (scale, offset_x - crop[0] * scale, offset_y - crop[1] * scale)


class PreviewCanvas(tk.Canvas):
    """Shows the image ``source`` returns, fitted and centered; the wheel zooms around the
    pointer and dragging with ``pan_button`` pans while zoomed. ``source`` returning None keeps
    what is shown. ``decorate`` may paint on the display image (given the image to display
    transform) and ``on_drawn`` may add canvas items over it (given the image to canvas transform)."""

    def __init__(
        self,
        master,
        *,
        source: Callable[[], Image.Image | None],
        pan_button: int,
        on_view_changed: Callable[[], None] | None = None,
        decorate: Callable[[Image.Image, Transform], Image.Image] | None = None,
        on_drawn: Callable[[Transform], None] | None = None,
    ) -> None:
        super().__init__(
            master,
            width=1,
            height=1,
            background=Colors.BG_PANEL,
            borderwidth=0,
            highlightthickness=0,
        )
        self.view = PreviewView()
        self._source = source
        self._on_view_changed = on_view_changed
        self._decorate = decorate
        self._on_drawn = on_drawn
        self._photo: ImageTk.PhotoImage | None = None
        self._shown_size: tuple[int, int] | None = None
        self._message: tuple[str, str] | None = None
        self._pan_anchor: tuple[int, int] | None = None
        self._resize_after: str | None = None
        self.bind("<Configure>", self._on_configure)
        self.bind("<MouseWheel>", lambda event: self._on_wheel(event, event.delta))
        self.bind("<Button-4>", lambda event: self._on_wheel(event, 1))
        self.bind("<Button-5>", lambda event: self._on_wheel(event, -1))
        self.bind(f"<ButtonPress-{pan_button}>", self._on_pan_start)
        self.bind(f"<B{pan_button}-Motion>", self._on_pan_drag)
        self.bind(f"<ButtonRelease-{pan_button}>", self._on_pan_end)

    def _canvas_size(self) -> tuple[int, int]:
        return max(2, self.winfo_width()), max(2, self.winfo_height())

    def transform(self) -> Transform:
        """Image to canvas mapping of the shown image."""
        return preview_layout(self._shown_size, self._canvas_size(), self.view)[2]

    def _shown_rect(self) -> tuple[float, float, float, float]:
        """Canvas ``(left, top, width, height)`` of the shown image."""
        crop, _, transform = preview_layout(self._shown_size, self._canvas_size(), self.view)
        left, top = image_to_canvas(crop[0], crop[1], view=transform)
        return left, top, (crop[2] - crop[0]) * transform[0], (crop[3] - crop[1]) * transform[0]

    def redraw(self) -> None:
        self._resize_after = None
        source = self._source()
        if source is None:
            if self._message is not None:
                self._draw_message()
            return
        self._message = None
        self._shown_size = source.size
        crop, display_size, transform = preview_layout(source.size, self._canvas_size(), self.view)
        display = source.crop(crop).resize(display_size, Image.Resampling.LANCZOS)
        if self._decorate is not None:
            scale = transform[0]
            display = self._decorate(display, (scale, -crop[0] * scale, -crop[1] * scale))
        self.delete("all")
        self._photo = ImageTk.PhotoImage(display)
        width, height = self._canvas_size()
        self.create_image((width - display.width) // 2, (height - display.height) // 2, image=self._photo, anchor="nw")
        if self._on_drawn is not None:
            self._on_drawn(transform)

    def show_message(self, text: str, color: str) -> None:
        """Replace the image with ``text`` until the next redraw that has a source."""
        self._message = (text, color)
        self._shown_size = None
        self._draw_message()

    def _draw_message(self) -> None:
        text, color = self._message
        self.delete("all")
        self._photo = None
        width, height = self._canvas_size()
        self.create_text(
            width / 2,
            height / 2,
            text=text,
            fill=color,
            font=(Fonts.FAMILY, scaling.raw_tk_font_size(self, Fonts.SIZE_SMALL)),
        )

    def zoom_to(self, zoom: float, anchor: tuple[float, float] | None = None) -> None:
        """Zoom, keeping the image point under the canvas point ``anchor`` still."""
        anchor_fraction = None
        if anchor is not None and self._shown_size is not None:
            left, top, width, height = self._shown_rect()
            anchor_fraction = (
                min(1.0, max(0.0, (anchor[0] - left) / width)),
                min(1.0, max(0.0, (anchor[1] - top) / height)),
            )
        changed = self.view.zoom_to(zoom, anchor_fraction)
        self._view_changed()
        if changed:
            self.redraw()

    def zoom_by(self, amount: float) -> None:
        self.zoom_to(self.view.zoom + amount)

    def reset_view(self) -> None:
        self.view.reset()
        self._pan_anchor = None
        self._view_changed()
        self.redraw()

    def _view_changed(self) -> None:
        if self._on_view_changed is not None:
            self._on_view_changed()

    def _on_configure(self, _event=None) -> None:
        if self._resize_after is not None:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(60, self.redraw)

    def _on_wheel(self, event, delta: int):
        if not delta:
            return None
        direction = 1 if delta > 0 else -1
        self.zoom_to(self.view.zoom + direction * PREVIEW_ZOOM_STEP, anchor=(float(event.x), float(event.y)))
        return "break"

    def _on_pan_start(self, event):
        if not self.view.zoomed:
            self._pan_anchor = None
            return None
        self._pan_anchor = (int(event.x), int(event.y))
        return "break"

    def _on_pan_drag(self, event):
        anchor = self._pan_anchor
        if anchor is None or self._shown_size is None or not self.view.zoomed:
            return None
        _, _, width, height = self._shown_rect()
        self.view.pan((int(event.x) - anchor[0]) / width, (int(event.y) - anchor[1]) / height)
        self._pan_anchor = (int(event.x), int(event.y))
        self.redraw()
        return "break"

    def _on_pan_end(self, _event=None):
        was_panning = self._pan_anchor is not None
        self._pan_anchor = None
        return "break" if was_panning else None

    def destroy(self) -> None:
        if self._resize_after is not None:
            self.after_cancel(self._resize_after)
            self._resize_after = None
        super().destroy()

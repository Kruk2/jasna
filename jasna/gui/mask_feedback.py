"""Community mask-suggestion feature for the segment editor.

Users draw polygon masks over the paused frame; the frame (full resolution),
the rendered mask, and a minimal anonymous meta JSON are zipped, sealed with
the project public key (X25519 + HKDF-SHA256 + AES-256-GCM, format shared
with cloudflare-jasna/sealbox.py), and uploaded to the feedback endpoint.
The stored blob is opaque to the storage provider; meta never contains
filenames, timestamps, or any user-identifying data.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk

from jasna.gui.locales import t
from jasna.gui.theme import Colors, Fonts

FEEDBACK_ENDPOINT = "https://jasna-mask-feedback.kruk2.workers.dev/submit"
FEEDBACK_TOKEN = "da8ef41e807b62bc3ad146b17c7850bffec618703524ce01"
FEEDBACK_PUBLIC_KEY = "gtUutgR6zVqjq+qxu+8uzLvhvmVlc8RlqJhba+TXyEI="
FEEDBACK_TIMEOUT_SECONDS = 30.0
JPEG_QUALITY = 90
CLOSE_HIT_PX = 10
MIN_POLYGON_POINTS = 3

_SEAL_INFO = b"jasna-mask-feedback-v1"

Polygon = tuple[tuple[float, float], ...]


def fit_scale_and_offset(
    image_size: tuple[int, int], canvas_size: tuple[int, int]
) -> tuple[float, float, float]:
    image_w, image_h = image_size
    canvas_w, canvas_h = canvas_size
    scale = min(canvas_w / image_w, canvas_h / image_h)
    offset_x = (canvas_w - image_w * scale) / 2
    offset_y = (canvas_h - image_h * scale) / 2
    return scale, offset_x, offset_y


def view_transform(
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    zoom: float,
    center: tuple[float, float],
) -> tuple[float, float, float]:
    """Scale and offsets for a zoomed view centered on ``center`` (image coords).

    At zoom 1.0 this equals ``fit_scale_and_offset``. When the scaled image
    exceeds the canvas on an axis, the offset follows ``center`` but is clamped
    so the view never leaves the image."""

    fit_scale, _, _ = fit_scale_and_offset(image_size, canvas_size)
    scale = fit_scale * float(zoom)
    offsets = []
    for axis in (0, 1):
        span = image_size[axis] * scale
        if span <= canvas_size[axis]:
            offsets.append((canvas_size[axis] - span) / 2)
        else:
            offset = canvas_size[axis] / 2 - float(center[axis]) * scale
            offsets.append(min(0.0, max(canvas_size[axis] - span, offset)))
    return scale, offsets[0], offsets[1]


def canvas_to_image(
    x: float,
    y: float,
    *,
    view: tuple[float, float, float],
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
    view: tuple[float, float, float],
) -> tuple[float, float]:
    scale, offset_x, offset_y = view
    return x * scale + offset_x, y * scale + offset_y


def polygons_to_mask(
    polygons: Sequence[Sequence[tuple[float, float]]], size: tuple[int, int]
) -> Image.Image:
    from PIL import ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < MIN_POLYGON_POINTS:
            raise ValueError("polygon needs at least three points")
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    return mask


def build_meta(
    app_version: str, detection_model: str, frame_width: int, frame_height: int
) -> dict:
    return {
        "app_version": str(app_version),
        "detection_model": str(detection_model),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
    }


def encode_submission(frame: Image.Image, mask: Image.Image, meta: dict) -> bytes:
    frame_bytes = io.BytesIO()
    frame.save(frame_bytes, format="JPEG", quality=JPEG_QUALITY)
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG", optimize=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("frame.jpg", frame_bytes.getvalue())
        archive.writestr("mask.png", mask_bytes.getvalue())
        archive.writestr("meta.json", json.dumps(meta))
    return payload.getvalue()


def seal(payload: bytes, recipient_public_key_b64: str) -> bytes:
    import base64
    import os

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    recipient = X25519PublicKey.from_public_bytes(
        base64.b64decode(recipient_public_key_b64)
    )
    ephemeral = X25519PrivateKey.generate()
    key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=_SEAL_INFO
    ).derive(ephemeral.exchange(recipient))
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, payload, None)
    ephemeral_public = ephemeral.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return ephemeral_public + nonce + ciphertext


def _post(url: str, data: bytes, headers: dict, timeout: float) -> None:
    import requests

    response = requests.post(url, data=data, headers=headers, timeout=timeout)
    response.raise_for_status()


@dataclass(frozen=True)
class FeedbackUploadFinished:
    ok: bool
    message: str


class MaskFeedbackWorker:
    """One-shot upload threads; submissions are rare so no persistent thread."""

    def __init__(self) -> None:
        self.events: queue.Queue[FeedbackUploadFinished] = queue.Queue()

    def upload(
        self,
        frame_image: Image.Image,
        polygons: Sequence[Polygon],
        detection_model: str,
        app_version: str,
    ) -> None:
        def _run() -> None:
            try:
                mask = polygons_to_mask(polygons, frame_image.size)
                meta = build_meta(
                    app_version, detection_model, *frame_image.size
                )
                blob = seal(
                    encode_submission(frame_image, mask, meta),
                    FEEDBACK_PUBLIC_KEY,
                )
                _post(
                    FEEDBACK_ENDPOINT,
                    blob,
                    {
                        "x-jasna-token": FEEDBACK_TOKEN,
                        "content-type": "application/octet-stream",
                    },
                    FEEDBACK_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                self.events.put(FeedbackUploadFinished(False, str(exc)))
                return
            self.events.put(FeedbackUploadFinished(True, ""))

        threading.Thread(target=_run, name="mask-feedback-upload", daemon=True).start()


class MaskSuggestDialog(ctk.CTkToplevel):
    """Modal polygon editor over a full-resolution frame.

    Click adds a vertex; clicking near the first vertex, double-clicking, or
    Enter closes the polygon. Multiple polygons supported. Submit hands the
    closed polygons (full-resolution image coordinates) to ``on_submit``.
    """

    def __init__(
        self,
        master,
        frame_image: Image.Image,
        on_submit: Callable[[tuple[Polygon, ...]], None],
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._frame_image = frame_image
        self._on_submit = on_submit
        self._on_closed = on_closed
        self._polygons: list[list[tuple[float, float]]] = []
        self._draft: list[tuple[float, float]] = []
        self._photo = None
        self._resize_after: str | None = None
        self._done = False
        self._zoom = 1.0
        self._center = (frame_image.width / 2, frame_image.height / 2)
        self._shapes_hidden = False
        self._mask_alpha = 0.35
        self._alpha_after: str | None = None
        self._pan_anchor: tuple[float, float, float, float] | None = None

        self.title(t("mask_editor_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(screen_w - 120, max(800, screen_w - 400))
        height = min(screen_h - 160, max(560, screen_h - 260))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        ctk.CTkLabel(
            self,
            text=t("mask_editor_instructions"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=width - 48,
            justify="left",
        ).pack(fill="x", padx=16, pady=(12, 4))

        self._canvas = tk.Canvas(
            self,
            background=Colors.BG_PANEL,
            borderwidth=0,
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill="both", expand=True, padx=16, pady=4)
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", lambda event: self._on_wheel(event, delta=120))
        self._canvas.bind("<Button-5>", lambda event: self._on_wheel(event, delta=-120))
        self._canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self._canvas.bind("<B3-Motion>", self._on_pan_move)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(4, 12))
        info_column = ctk.CTkFrame(footer, fg_color="transparent")
        info_column.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            info_column,
            text=t("mask_editor_quick_help"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=width - 520,
            justify="left",
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            info_column,
            text=t("mask_editor_anonymous_note"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            wraplength=width - 520,
            justify="left",
            anchor="w",
        ).pack(fill="x")
        self._interactive_widgets: list = []
        self._submit_btn = ctk.CTkButton(
            footer,
            text=t("mask_editor_submit"),
            state="disabled",
            command=self._submit,
        )
        self._submit_btn.pack(side="right")
        self._interactive_widgets.append(self._submit_btn)
        cancel_btn = ctk.CTkButton(
            footer,
            text=t("segments_cancel"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._cancel,
        )
        cancel_btn.pack(side="right", padx=8)
        self._interactive_widgets.append(cancel_btn)
        erase_btn = ctk.CTkButton(
            footer,
            text=t("mask_editor_erase_all"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.STATUS_ERROR,
            command=self._erase_all,
        )
        erase_btn.pack(side="right", padx=8)
        self._interactive_widgets.append(erase_btn)
        undo_btn = ctk.CTkButton(
            footer,
            text=t("mask_editor_undo_point"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._undo_point,
        )
        undo_btn.pack(side="right")
        self._interactive_widgets.append(undo_btn)
        self._hide_btn = ctk.CTkButton(
            footer,
            text=t("mask_editor_hide_shapes"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._toggle_shapes,
        )
        self._hide_btn.pack(side="right", padx=8)
        self._interactive_widgets.append(self._hide_btn)
        self._alpha_slider = ctk.CTkSlider(
            footer,
            from_=0.0,
            to=0.9,
            width=110,
            command=self._on_alpha,
        )
        self._alpha_slider.set(self._mask_alpha)
        self._alpha_slider.pack(side="right", padx=(4, 8))
        self._interactive_widgets.append(self._alpha_slider)
        ctk.CTkLabel(
            footer,
            text=t("mask_editor_opacity"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="right")
        for text, command in (
            (t("segments_fit"), self._reset_zoom),
            ("+", lambda: self._zoom_step(1.25)),
            ("−", lambda: self._zoom_step(1 / 1.25)),
        ):
            zoom_btn = ctk.CTkButton(
                footer,
                text=text,
                width=34 if len(text) == 1 else 48,
                fg_color=Colors.BG_CARD,
                hover_color=Colors.BORDER_LIGHT,
                command=command,
            )
            zoom_btn.pack(side="right", padx=(0, 4))
            self._interactive_widgets.append(zoom_btn)

        self.bind("<ButtonPress-1>", self._on_global_press)
        self.bind("<Double-Button-1>", self._on_global_double)
        self.bind("<Return>", lambda _e: self._close_draft())
        self.bind("<BackSpace>", lambda _e: self._undo_point())
        self.bind("<Control-z>", lambda _e: self._undo_point())
        self.bind("<KeyPress-h>", lambda _e: self._toggle_shapes())
        self.bind("<KeyPress-plus>", lambda _e: self._zoom_step(1.25))
        self.bind("<KeyPress-equal>", lambda _e: self._zoom_step(1.25))
        self.bind("<KeyPress-minus>", lambda _e: self._zoom_step(1 / 1.25))
        self.bind("<KeyPress-0>", lambda _e: self._reset_zoom())
        self.bind("<Escape>", lambda _e: self._cancel())

        self.update_idletasks()
        self.wait_visibility()
        self.grab_set()
        self.lift()
        self.focus_force()
        self._redraw()

    def _canvas_size(self) -> tuple[int, int]:
        return max(2, self._canvas.winfo_width()), max(2, self._canvas.winfo_height())

    def _on_resize(self, _event=None) -> None:
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(60, self._redraw)

    def _view(self) -> tuple[float, float, float]:
        return view_transform(
            self._frame_image.size, self._canvas_size(), self._zoom, self._center
        )

    def _is_interactive(self, widget) -> bool:
        current = widget
        while current is not None and current is not self:
            if current in self._interactive_widgets:
                return True
            current = getattr(current, "master", None)
        return False

    def _canvas_event_coords(self, event) -> tuple[int, int]:
        return (
            event.x_root - self._canvas.winfo_rootx(),
            event.y_root - self._canvas.winfo_rooty(),
        )

    def _on_global_press(self, event) -> None:
        if self._is_interactive(event.widget):
            return
        x, y = self._canvas_event_coords(event)
        self._on_press(SimpleNamespace(x=x, y=y))

    def _on_global_double(self, event) -> None:
        if self._is_interactive(event.widget):
            return
        self._close_draft()

    def _on_press(self, event) -> None:
        view = self._view()
        point = canvas_to_image(
            event.x, event.y, view=view, image_size=self._frame_image.size
        )
        if len(self._draft) >= MIN_POLYGON_POINTS:
            first_canvas = image_to_canvas(*self._draft[0], view=view)
            if (
                abs(event.x - first_canvas[0]) <= CLOSE_HIT_PX
                and abs(event.y - first_canvas[1]) <= CLOSE_HIT_PX
            ):
                self._close_draft()
                return
        self._draft.append(point)
        self._redraw()

    def _on_wheel(self, event, *, delta: int | None = None) -> str:
        wheel_delta = delta if delta is not None else int(getattr(event, "delta", 0))
        if wheel_delta:
            factor = 1.25 if wheel_delta > 0 else 1 / 1.25
            self._zoom_step(factor, anchor=(event.x, event.y))
        return "break"

    def _zoom_step(
        self, factor: float, anchor: tuple[int, int] | None = None
    ) -> None:
        old_view = self._view()
        new_zoom = min(12.0, max(1.0, self._zoom * float(factor)))
        if new_zoom == self._zoom:
            return
        canvas_w, canvas_h = self._canvas_size()
        if anchor is None:
            anchor = (canvas_w / 2, canvas_h / 2)
        anchored = canvas_to_image(
            *anchor, view=old_view, image_size=self._frame_image.size
        )
        fit_scale, _, _ = fit_scale_and_offset(
            self._frame_image.size, self._canvas_size()
        )
        new_scale = fit_scale * new_zoom
        self._zoom = new_zoom
        self._center = (
            anchored[0] + (canvas_w / 2 - anchor[0]) / new_scale,
            anchored[1] + (canvas_h / 2 - anchor[1]) / new_scale,
        )
        self._redraw()

    def _reset_zoom(self) -> None:
        self._zoom = 1.0
        self._center = (self._frame_image.width / 2, self._frame_image.height / 2)
        self._redraw()

    def _on_pan_start(self, event) -> None:
        self._pan_anchor = (event.x, event.y, *self._center)

    def _on_pan_move(self, event) -> None:
        if self._pan_anchor is None:
            return
        scale, _, _ = self._view()
        start_x, start_y, center_x, center_y = self._pan_anchor
        self._center = (
            center_x - (event.x - start_x) / scale,
            center_y - (event.y - start_y) / scale,
        )
        self._pan_anchor = (event.x, event.y, *self._center)
        self._redraw()

    def _on_alpha(self, value: float) -> None:
        self._mask_alpha = float(value)
        if self._alpha_after is not None:
            try:
                self.after_cancel(self._alpha_after)
            except tk.TclError:
                pass
        self._alpha_after = self.after(40, self._redraw)

    def _toggle_shapes(self) -> None:
        self._shapes_hidden = not self._shapes_hidden
        self._hide_btn.configure(
            text=t("mask_editor_show_shapes")
            if self._shapes_hidden
            else t("mask_editor_hide_shapes")
        )
        self._redraw()

    def _close_draft(self) -> None:
        if len(self._draft) >= MIN_POLYGON_POINTS:
            self._polygons.append(self._draft)
            self._draft = []
            self._redraw()

    def _undo_point(self) -> None:
        if self._draft:
            self._draft.pop()
        elif self._polygons:
            self._draft = self._polygons.pop()
        self._redraw()

    def _erase_all(self) -> None:
        self._polygons = []
        self._draft = []
        self._redraw()

    def _redraw(self) -> None:
        self._resize_after = None
        canvas = self._canvas
        canvas.delete("all")
        canvas_size = self._canvas_size()
        image_size = self._frame_image.size
        view = self._view()
        scale, offset_x, offset_y = view

        crop_x0 = max(0, int(-offset_x / scale))
        crop_y0 = max(0, int(-offset_y / scale))
        crop_x1 = min(image_size[0], int((canvas_size[0] - offset_x) / scale) + 1)
        crop_y1 = min(image_size[1], int((canvas_size[1] - offset_y) / scale) + 1)
        crop = self._frame_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        target = (
            max(2, round(crop.width * scale)),
            max(2, round(crop.height * scale)),
        )
        display = crop.resize(target, Image.Resampling.LANCZOS)
        draw_origin = (crop_x0 * scale + offset_x, crop_y0 * scale + offset_y)
        if not self._shapes_hidden and self._polygons and self._mask_alpha > 0:
            from PIL import ImageDraw

            overlay = Image.new("RGBA", display.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            primary = Colors.PRIMARY.lstrip("#")
            fill = (
                *(int(primary[i : i + 2], 16) for i in (0, 2, 4)),
                round(self._mask_alpha * 255),
            )
            for polygon in self._polygons:
                overlay_draw.polygon(
                    [
                        (
                            point[0] * scale + offset_x - draw_origin[0],
                            point[1] * scale + offset_y - draw_origin[1],
                        )
                        for point in polygon
                    ],
                    fill=fill,
                )
            display = Image.alpha_composite(
                display.convert("RGBA"), overlay
            ).convert("RGB")
        self._photo = ImageTk.PhotoImage(display)
        canvas.create_image(*draw_origin, image=self._photo, anchor="nw")

        def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
            return image_to_canvas(*point, view=view)

        if self._shapes_hidden:
            self._refresh_submit_state()
            return

        for polygon in self._polygons:
            flat = [coord for point in polygon for coord in to_canvas(point)]
            canvas.create_polygon(
                flat,
                fill="",
                outline=Colors.PRIMARY,
                width=2,
            )
            for point in polygon:
                cx, cy = to_canvas(point)
                canvas.create_oval(
                    cx - 3, cy - 3, cx + 3, cy + 3, fill="#e0e7ff", outline=""
                )

        if self._draft:
            points = [to_canvas(point) for point in self._draft]
            if len(points) > 1:
                canvas.create_line(
                    [coord for point in points for coord in point],
                    fill=Colors.STATUS_PAUSED,
                    width=2,
                )
            for index, (cx, cy) in enumerate(points):
                radius = 5 if index == 0 else 3
                canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=Colors.STATUS_PAUSED if index == 0 else "#e0e7ff",
                    outline="",
                )

        self._refresh_submit_state()

    def _refresh_submit_state(self) -> None:
        self._submit_btn.configure(
            state="normal"
            if self._polygons or len(self._draft) >= MIN_POLYGON_POINTS
            else "disabled"
        )

    def _submit(self) -> None:
        self._close_draft()
        if not self._polygons:
            return
        polygons = tuple(tuple(polygon) for polygon in self._polygons)
        self._finish()
        self._on_submit(polygons)

    def _cancel(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.grab_release()
        except tk.TclError:
            pass
        if self._on_closed is not None:
            self._on_closed()
        self.destroy()

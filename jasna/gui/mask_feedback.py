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

import customtkinter as ctk
from PIL import Image

from jasna.gui.components import grab_modal
from jasna.gui import scaling
from jasna.gui.locales import t
from jasna.gui.preview_canvas import PREVIEW_ZOOM_STEP, PreviewCanvas, Transform, canvas_to_image, image_to_canvas
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
        self._done = False
        self._shapes_hidden = False
        self._mask_alpha = 0.35
        self._alpha_after: str | None = None

        self.title(t("mask_editor_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        screen_w, screen_h = scaling.to_logical(self, *scaling.screen_rect(self)[2:])
        width = min(screen_w - 120, max(800, screen_w - 400))
        height = min(screen_h - 160, max(560, screen_h - 260))
        scaling.place_centered_on_screen(self, *scaling.to_physical(self, width, height))

        ctk.CTkLabel(
            self,
            text=t("mask_editor_instructions"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=width - 48,
            justify="left",
        ).pack(fill="x", padx=16, pady=(12, 4))

        self._canvas = PreviewCanvas(
            self,
            source=lambda: self._frame_image,
            pan_button=3,
            decorate=self._fill_shapes,
            on_drawn=self._draw_shapes,
        )
        self._canvas.configure(cursor="crosshair")
        self._canvas.pack(fill="both", expand=True, padx=16, pady=4)

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
            text=t("btn_cancel"),
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
            (t("segments_fit"), self._canvas.reset_view),
            ("+", lambda: self._canvas.zoom_by(PREVIEW_ZOOM_STEP)),
            ("−", lambda: self._canvas.zoom_by(-PREVIEW_ZOOM_STEP)),
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
        self.bind("<KeyPress-plus>", lambda _e: self._canvas.zoom_by(PREVIEW_ZOOM_STEP))
        self.bind("<KeyPress-equal>", lambda _e: self._canvas.zoom_by(PREVIEW_ZOOM_STEP))
        self.bind("<KeyPress-minus>", lambda _e: self._canvas.zoom_by(-PREVIEW_ZOOM_STEP))
        self.bind("<KeyPress-0>", lambda _e: self._canvas.reset_view())
        self.bind("<Escape>", lambda _e: self._cancel())

        self.update_idletasks()
        grab_modal(self)
        self._canvas.redraw()

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
        view = self._canvas.transform()
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
        self._canvas.redraw()

    def _on_alpha(self, value: float) -> None:
        self._mask_alpha = float(value)
        if self._alpha_after is not None:
            self.after_cancel(self._alpha_after)
        self._alpha_after = self.after(40, self._canvas.redraw)

    def _toggle_shapes(self) -> None:
        self._shapes_hidden = not self._shapes_hidden
        self._hide_btn.configure(
            text=t("mask_editor_show_shapes")
            if self._shapes_hidden
            else t("mask_editor_hide_shapes")
        )
        self._canvas.redraw()

    def _close_draft(self) -> None:
        if len(self._draft) >= MIN_POLYGON_POINTS:
            self._polygons.append(self._draft)
            self._draft = []
            self._canvas.redraw()

    def _undo_point(self) -> None:
        if self._draft:
            self._draft.pop()
        elif self._polygons:
            self._draft = self._polygons.pop()
        self._canvas.redraw()

    def _erase_all(self) -> None:
        self._polygons = []
        self._draft = []
        self._canvas.redraw()

    def _fill_shapes(self, display: Image.Image, view: Transform) -> Image.Image:
        if self._shapes_hidden or not self._polygons or self._mask_alpha <= 0:
            return display
        from PIL import ImageDraw

        overlay = Image.new("RGBA", display.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        primary = Colors.PRIMARY.lstrip("#")
        fill = (
            *(int(primary[i : i + 2], 16) for i in (0, 2, 4)),
            round(self._mask_alpha * 255),
        )
        for polygon in self._polygons:
            overlay_draw.polygon([image_to_canvas(*point, view=view) for point in polygon], fill=fill)
        return Image.alpha_composite(display.convert("RGBA"), overlay).convert("RGB")

    def _draw_shapes(self, view: Transform) -> None:
        canvas = self._canvas

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
                    fill=Colors.STATUS_WARNING,
                    width=2,
                )
            for index, (cx, cy) in enumerate(points):
                radius = 5 if index == 0 else 3
                canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=Colors.STATUS_WARNING if index == 0 else "#e0e7ff",
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
        self.grab_release()
        if self._on_closed is not None:
            self._on_closed()
        self.destroy()

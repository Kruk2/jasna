from __future__ import annotations

import io
import json
import queue
import zipfile
from tkinter import TclError
from types import SimpleNamespace

import customtkinter as ctk
import pytest
from PIL import Image

from jasna.gui import mask_feedback


def test_triangle_mask_fills_interior():
    mask = mask_feedback.polygons_to_mask(
        [((10, 10), (100, 10), (10, 100))], (200, 150)
    )
    assert mask.size == (200, 150)
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((150, 100)) == 0
    assert set(mask.getdata()) <= {0, 255}


def test_multiple_polygons_union():
    mask = mask_feedback.polygons_to_mask(
        [((0, 0), (10, 0), (10, 10), (0, 10)), ((50, 50), (60, 50), (60, 60), (50, 60))],
        (100, 100),
    )
    assert mask.getpixel((5, 5)) == 255
    assert mask.getpixel((55, 55)) == 255
    assert mask.getpixel((30, 30)) == 0


def test_too_few_points_rejected():
    with pytest.raises(ValueError):
        mask_feedback.polygons_to_mask([((0, 0), (10, 10))], (100, 100))


@pytest.mark.parametrize(
    "image_size,canvas_size",
    [((1920, 1080), (800, 600)), ((1080, 1920), (800, 600)), ((640, 480), (1000, 400))],
)
def test_coordinate_round_trip(image_size, canvas_size):
    view = mask_feedback.view_transform(
        image_size, canvas_size, 1.0, (image_size[0] / 2, image_size[1] / 2)
    )
    point = (image_size[0] * 0.25, image_size[1] * 0.75)
    cx, cy = mask_feedback.image_to_canvas(*point, view=view)
    back = mask_feedback.canvas_to_image(
        cx, cy, view=view, image_size=image_size
    )
    assert back == pytest.approx(point)


def test_letterbox_margin_click_snaps_to_edge():
    image_size = (1920, 1080)
    view = mask_feedback.view_transform(image_size, (800, 600), 1.0, (960, 540))
    snapped = mask_feedback.canvas_to_image(400, 10, view=view, image_size=image_size)
    assert snapped[1] == 0.0
    assert 0 <= snapped[0] <= 1920
    below = mask_feedback.canvas_to_image(-50, 700, view=view, image_size=image_size)
    assert below == (0.0, 1080.0)


def test_view_transform_zoom_one_equals_fit():
    image_size, canvas_size = (1920, 1080), (800, 600)
    fit = mask_feedback.fit_scale_and_offset(image_size, canvas_size)
    view = mask_feedback.view_transform(image_size, canvas_size, 1.0, (10, 10))
    assert view == pytest.approx(fit)


def test_view_transform_clamps_at_image_edges():
    image_size, canvas_size = (1000, 1000), (500, 500)
    scale, off_x, off_y = mask_feedback.view_transform(
        image_size, canvas_size, 4.0, (0, 0)
    )
    assert (off_x, off_y) == (0.0, 0.0)
    scale, off_x, off_y = mask_feedback.view_transform(
        image_size, canvas_size, 4.0, (1000, 1000)
    )
    assert off_x == pytest.approx(500 - 1000 * scale)
    assert off_y == pytest.approx(500 - 1000 * scale)


def test_meta_contains_only_allowed_keys():
    meta = mask_feedback.build_meta("0.7.2", "rfdetr-v5", 1920, 1080)
    assert meta == {
        "app_version": "0.7.2",
        "detection_model": "rfdetr-v5",
        "frame_width": 1920,
        "frame_height": 1080,
    }


def test_encode_submission_zip_round_trip():
    frame = Image.new("RGB", (64, 36), (200, 30, 30))
    mask = Image.new("L", (64, 36), 0)
    payload = mask_feedback.encode_submission(
        frame, mask, mask_feedback.build_meta("v", "m", 64, 36)
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert sorted(archive.namelist()) == ["frame.jpg", "mask.png", "meta.json"]
        assert json.loads(archive.read("meta.json"))["frame_width"] == 64
        restored_mask = Image.open(io.BytesIO(archive.read("mask.png")))
        assert restored_mask.size == (64, 36)


def test_seal_produces_opaque_blob():
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    import base64

    private = X25519PrivateKey.generate()
    public_b64 = base64.b64encode(
        private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()

    payload = b"secret payload" * 100
    blob = mask_feedback.seal(payload, public_b64)
    assert payload not in blob
    assert blob != mask_feedback.seal(payload, public_b64)

    ephemeral_public = X25519PublicKey.from_public_bytes(blob[:32])
    key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"jasna-mask-feedback-v1"
    ).derive(private.exchange(ephemeral_public))
    assert AESGCM(key).decrypt(blob[32:44], blob[44:], None) == payload


def test_upload_worker_posts_sealed_blob(monkeypatch):
    recorded = {}

    def fake_post(url, data, headers, timeout):
        recorded.update(url=url, data=data, headers=headers, timeout=timeout)

    monkeypatch.setattr(mask_feedback, "_post", fake_post)
    worker = mask_feedback.MaskFeedbackWorker()
    frame = Image.new("RGB", (32, 32))
    worker.upload(frame, (((1, 1), (20, 1), (20, 20)),), "rfdetr-v5", "0.7.2")

    event = worker.events.get(timeout=10)
    assert isinstance(event, mask_feedback.FeedbackUploadFinished)
    assert event.ok
    assert recorded["url"] == mask_feedback.FEEDBACK_ENDPOINT
    assert recorded["headers"]["x-jasna-token"] == mask_feedback.FEEDBACK_TOKEN
    assert len(recorded["data"]) > 32 + 12 + 16


def test_upload_worker_reports_failure(monkeypatch):
    def failing_post(url, data, headers, timeout):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(mask_feedback, "_post", failing_post)
    worker = mask_feedback.MaskFeedbackWorker()
    worker.upload(Image.new("RGB", (16, 16)), (((0, 0), (8, 0), (8, 8)),), "m", "v")

    event = worker.events.get(timeout=10)
    assert not event.ok
    assert "endpoint down" in event.message


@pytest.fixture
def root():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


def _click_image(dialog, image_x, image_y):
    x, y = mask_feedback.image_to_canvas(image_x, image_y, view=dialog._view())
    dialog._on_press(SimpleNamespace(x=round(x), y=round(y)))


def test_dialog_draws_and_submits_polygons(root):
    root.update()
    submitted = {}
    dialog = mask_feedback.MaskSuggestDialog(
        root,
        Image.new("RGB", (640, 360), (40, 40, 40)),
        on_submit=lambda polygons: submitted.update(polygons=polygons),
    )
    root.update()
    assert dialog._submit_btn.cget("state") == "disabled"

    for point in ((100, 100), (200, 100), (200, 200)):
        _click_image(dialog, *point)
    dialog._close_draft()
    root.update()
    assert len(dialog._polygons) == 1
    assert dialog._submit_btn.cget("state") == "normal"

    for point in ((300, 100), (400, 100), (400, 200), (300, 200)):
        _click_image(dialog, *point)

    dialog._submit()
    assert len(submitted["polygons"]) == 2
    for polygon in submitted["polygons"]:
        for x, y in polygon:
            assert 0 <= x <= 640 and 0 <= y <= 360


def test_dialog_undo_and_erase(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    for point in ((100, 100), (200, 100), (200, 200)):
        _click_image(dialog, *point)
    assert len(dialog._draft) == 3
    dialog._undo_point()
    assert len(dialog._draft) == 2
    dialog._close_draft()
    assert dialog._polygons == [] and len(dialog._draft) == 2

    _click_image(dialog, 240, 240)
    dialog._close_draft()
    dialog._erase_all()
    assert dialog._polygons == [] and dialog._draft == []
    assert dialog._submit_btn.cget("state") == "disabled"
    dialog._cancel()


def test_dialog_close_hits_first_vertex(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    for point in ((100, 100), (200, 100), (200, 200)):
        _click_image(dialog, *point)
    _click_image(dialog, 100.5, 100.5)
    assert len(dialog._polygons) == 1
    assert dialog._draft == []
    dialog._cancel()


def test_dialog_click_outside_frame_snaps_to_edge(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    dialog._on_press(SimpleNamespace(x=0, y=0))
    assert len(dialog._draft) == 1
    x, y = dialog._draft[0]
    assert x == 0.0 or y == 0.0
    dialog._cancel()


def test_dialog_zoom_and_reset(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    fit_scale, _, _ = mask_feedback.fit_scale_and_offset(
        dialog._frame_image.size, dialog._canvas_size()
    )
    dialog._zoom_step(2.0)
    assert dialog._zoom == pytest.approx(2.0)
    assert dialog._view()[0] == pytest.approx(fit_scale * 2.0)
    dialog._zoom_step(1 / 4)
    assert dialog._zoom == 1.0

    dialog._on_wheel(SimpleNamespace(x=10, y=10, delta=120))
    assert dialog._zoom > 1.0
    dialog._reset_zoom()
    assert dialog._zoom == 1.0
    dialog._cancel()


def test_dialog_hide_shapes_toggle(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    for point in ((100, 100), (200, 100), (200, 200)):
        _click_image(dialog, *point)
    dialog._close_draft()
    visible_items = len(dialog._canvas.find_all())

    dialog._toggle_shapes()
    assert dialog._shapes_hidden
    assert len(dialog._canvas.find_all()) < visible_items
    from jasna.gui.locales import t
    assert dialog._hide_btn.cget("text") == t("mask_editor_show_shapes")
    assert dialog._submit_btn.cget("state") == "normal"

    dialog._toggle_shapes()
    assert not dialog._shapes_hidden
    assert len(dialog._canvas.find_all()) == visible_items
    dialog._cancel()


def test_dialog_alpha_slider_updates_and_redraws(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    for point in ((100, 100), (200, 100), (200, 200)):
        _click_image(dialog, *point)
    dialog._close_draft()

    dialog._on_alpha(0.8)
    assert dialog._mask_alpha == 0.8
    dialog._redraw()
    dialog._on_alpha(0.0)
    dialog._redraw()
    assert dialog._polygons
    dialog._cancel()


def test_click_above_canvas_snaps_point_to_top_edge(root):
    root.update()
    dialog = mask_feedback.MaskSuggestDialog(
        root, Image.new("RGB", (640, 360)), on_submit=lambda polygons: None
    )
    root.update()
    canvas_x = dialog._canvas.winfo_rootx()
    canvas_y = dialog._canvas.winfo_rooty()
    view = dialog._view()
    inside_x, _ = mask_feedback.image_to_canvas(320, 0, view=view)

    dialog._on_global_press(
        SimpleNamespace(
            widget=dialog, x_root=canvas_x + round(inside_x), y_root=canvas_y - 40
        )
    )
    assert len(dialog._draft) == 1
    assert dialog._draft[0][1] == 0.0

    dialog._on_global_press(
        SimpleNamespace(
            widget=dialog._submit_btn._canvas,
            x_root=canvas_x + 10,
            y_root=canvas_y + 10,
        )
    )
    assert len(dialog._draft) == 1
    dialog._cancel()

from __future__ import annotations

import pytest
from PIL import Image

from jasna.gui.preview_canvas import (
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_STEP,
    PreviewCanvas,
    PreviewView,
    canvas_to_image,
    fit_scale_and_offset,
    image_to_canvas,
    preview_layout,
)
from factories import shown_message, shown_pixel


@pytest.mark.parametrize(
    "image_size,canvas_size",
    [((1920, 1080), (800, 600)), ((1080, 1920), (800, 600)), ((640, 480), (1000, 400))],
)
@pytest.mark.parametrize("zoom", [1.0, 2.5])
def test_coordinate_round_trip(image_size, canvas_size, zoom):
    view = PreviewView()
    view.zoom_to(zoom, None)
    transform = preview_layout(image_size, canvas_size, view)[2]
    point = (image_size[0] * 0.45, image_size[1] * 0.55)

    cx, cy = image_to_canvas(*point, view=transform)

    assert canvas_to_image(cx, cy, view=transform, image_size=image_size) == pytest.approx(point)


def test_letterbox_margin_click_snaps_to_edge():
    image_size = (1920, 1080)
    transform = preview_layout(image_size, (800, 600), PreviewView())[2]
    snapped = canvas_to_image(400, 10, view=transform, image_size=image_size)
    assert snapped[1] == 0.0
    assert 0 <= snapped[0] <= 1920
    below = canvas_to_image(-50, 700, view=transform, image_size=image_size)
    assert below == (0.0, 1080.0)


def test_layout_fits_the_whole_image_until_zoomed():
    crop, size, transform = preview_layout((1920, 1080), (900, 540), PreviewView())

    assert crop == (0, 0, 1920, 1080)
    assert size == (900, 506)
    assert transform == pytest.approx(fit_scale_and_offset((1920, 1080), (900, 540)))


def test_zoomed_layout_shows_the_crop_at_the_fitted_size():
    view = PreviewView(zoom=2.0, center=(1.0, 0.0))

    crop, size, transform = preview_layout((400, 200), (200, 200), view)

    assert crop == (200, 0, 400, 100)
    assert size == (200, 100)
    assert image_to_canvas(200, 0, view=transform) == pytest.approx((0, 50))
    assert image_to_canvas(400, 100, view=transform) == pytest.approx((200, 150))


def test_preview_crop_uses_zoom_and_clamps_pan_to_source() -> None:
    view = PreviewView(zoom=2.0, center=(0.0, 0.0))

    assert view.crop_box(400, 200) == (0, 0, 200, 100)
    assert view.center == pytest.approx((0.25, 0.25))


def test_preview_view_zoom_limits_and_reset() -> None:
    view = PreviewView()

    assert view.is_reset
    assert not view.zoom_to(0.5, None)
    assert view.zoom_to(1.5, None)
    assert view.zoomed and not view.is_reset
    assert view.zoom_to(100.0, None)
    assert view.zoom == PREVIEW_ZOOM_MAX

    view.center = (0.7, 0.6)
    view.reset()
    assert view.is_reset
    assert view.crop_box(400, 200) is None


def test_preview_zoom_keeps_anchor_point_still() -> None:
    view = PreviewView()

    view.zoom_to(2.0, (1.0, 0.5))

    assert view.center == pytest.approx((0.75, 0.5))


def test_dragging_zoomed_preview_pans_the_source() -> None:
    view = PreviewView(zoom=2.0)

    view.pan(0.1, 0.0)

    assert view.center[0] < 0.5
    assert view.center[1] == pytest.approx(0.5)


@pytest.fixture
def shown_canvas(tk_root):
    """A 400x200 canvas filled by a 400x200 image (red left half, blue right half), so canvas and
    shown image pixels match."""
    image = Image.new("RGB", (400, 200), "red")
    image.paste((0, 0, 255), (200, 0, 400, 200))
    changes: list[float] = []
    canvas = PreviewCanvas(tk_root, source=lambda: image, pan_button=1, on_view_changed=lambda: changes.append(canvas.view.zoom))
    canvas.place(x=0, y=0, width=400, height=200)
    tk_root.update()
    canvas.redraw()
    return canvas, changes


def test_mouse_wheel_zooms_around_the_pointer(shown_canvas, tk_root) -> None:
    canvas, changes = shown_canvas

    canvas.event_generate("<MouseWheel>", delta=120, x=390, y=100)
    canvas.event_generate("<MouseWheel>", delta=120, x=390, y=100)
    tk_root.update()

    assert changes == [1 + PREVIEW_ZOOM_STEP, 1 + 2 * PREVIEW_ZOOM_STEP]
    assert shown_pixel(canvas, 390, 100) == (0, 0, 255)


def test_dragging_pans_only_while_zoomed(shown_canvas, tk_root) -> None:
    canvas, _ = shown_canvas
    canvas.event_generate("<ButtonPress-1>", x=10, y=100)
    canvas.event_generate("<B1-Motion>", x=300, y=100, state=0x100)
    assert canvas.view.is_reset

    canvas.zoom_to(2.0)
    assert shown_pixel(canvas, 390, 100) == (0, 0, 255)
    canvas.event_generate("<ButtonPress-1>", x=10, y=100)
    canvas.event_generate("<B1-Motion>", x=300, y=100, state=0x100)
    canvas.event_generate("<ButtonRelease-1>", x=300, y=100)
    tk_root.update()

    assert shown_pixel(canvas, 390, 100) == (255, 0, 0)
    canvas.reset_view()
    assert canvas.view.is_reset


def test_message_replaces_the_image_until_a_redraw(shown_canvas) -> None:
    canvas, _ = shown_canvas

    canvas.show_message("Restoring…", "#ffffff")

    assert shown_message(canvas) == "Restoring…"
    canvas.redraw()
    assert shown_pixel(canvas, 10, 100) == (255, 0, 0)

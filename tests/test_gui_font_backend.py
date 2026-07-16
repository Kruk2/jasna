import json

from jasna.gui.font_backend import (
    FontBackendStatus,
    font_backend_error,
    font_backend_problem,
    font_backend_status_json,
)
from jasna.gui.theme import _font_families_for_platform


def _status(
    *,
    platform: str = "linux",
    ui_family: str = "Noto Sans",
    mono_family: str = "Noto Sans Mono",
    shape_family: str = "CustomTkinter_shapes_font",
) -> FontBackendStatus:
    return FontBackendStatus(
        platform=platform,
        windowing_system="x11",
        tk_version="8.6.13",
        family_count=300,
        ui_family=ui_family,
        mono_family=mono_family,
        shape_family=shape_family,
    )


def test_linux_theme_uses_fontconfig_generic_families():
    assert _font_families_for_platform("linux") == ("sans-serif", "monospace")


def test_windows_theme_keeps_existing_families():
    assert _font_families_for_platform("win32") == ("Segoe UI", "Consolas")


def test_healthy_linux_font_backend_has_no_problem():
    assert font_backend_problem(_status()) is None


def test_linux_fixed_font_fallback_is_rejected():
    problem = font_backend_problem(_status(ui_family="fixed", mono_family="fixed"))

    assert problem is not None
    assert "bitmap 'fixed'" in problem


def test_linux_missing_customtkinter_shape_font_is_rejected():
    problem = font_backend_problem(_status(shape_family="Noto Sans"))

    assert problem is not None
    assert "CustomTkinter shapes font" in problem


def test_non_linux_backend_is_not_subject_to_linux_xft_check():
    assert font_backend_problem(_status(platform="win32", ui_family="fixed")) is None


def test_source_error_points_to_distribution_python_tk():
    message = font_backend_error(_status(ui_family="fixed"), frozen=False)

    assert "distribution-provided Python" in message
    assert "Xft/fontconfig" in message


def test_frozen_error_identifies_broken_release_bundle():
    message = font_backend_error(_status(ui_family="fixed"), frozen=True)

    assert "release bundle" in message
    assert "packaging error" in message


def test_probe_status_is_machine_readable_json():
    payload = json.loads(font_backend_status_json(_status()))

    assert payload["ok"] is True
    assert payload["ui_family"] == "Noto Sans"
    assert payload["shape_family"] == "CustomTkinter_shapes_font"

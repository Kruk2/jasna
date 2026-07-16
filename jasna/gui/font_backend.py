"""Linux Tk font-backend validation for source and frozen GUI launches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sys
from tkinter import font as tkfont


SHAPE_FONT_FAMILY = "CustomTkinter_shapes_font"


@dataclass(frozen=True)
class FontBackendStatus:
    platform: str
    windowing_system: str
    tk_version: str
    family_count: int
    ui_family: str
    mono_family: str
    shape_family: str


class GuiFontBackendError(RuntimeError):
    pass


def inspect_font_backend(root) -> FontBackendStatus:
    def resolved_family(family: str) -> str:
        return str(tkfont.Font(root=root, family=family, size=12).actual("family"))

    return FontBackendStatus(
        platform=sys.platform,
        windowing_system=str(root.tk.call("tk", "windowingsystem")),
        tk_version=str(root.tk.call("info", "patchlevel")),
        family_count=len(tkfont.families(root)),
        ui_family=resolved_family("sans-serif"),
        mono_family=resolved_family("monospace"),
        shape_family=resolved_family(SHAPE_FONT_FAMILY),
    )


def font_backend_problem(status: FontBackendStatus) -> str | None:
    if not status.platform.startswith("linux"):
        return None

    fixed_families = [
        label
        for label, family in (("UI", status.ui_family), ("monospace", status.mono_family))
        if family.casefold() == "fixed"
    ]
    problems: list[str] = []
    if fixed_families:
        problems.append(f"{', '.join(fixed_families)} text resolved to the bitmap 'fixed' font")
    if status.shape_family.casefold() != SHAPE_FONT_FAMILY.casefold():
        problems.append(
            "the CustomTkinter shapes font resolved to "
            f"'{status.shape_family}' instead of '{SHAPE_FONT_FAMILY}'"
        )
    return "; ".join(problems) or None


def font_backend_error(status: FontBackendStatus, *, frozen: bool) -> str:
    problem = font_backend_problem(status)
    if problem is None:
        return ""
    details = (
        f"{problem}. Tk {status.tk_version} ({status.windowing_system}) reported "
        f"{status.family_count} font families."
    )
    if frozen:
        return (
            f"Jasna cannot start because this Linux release bundle has a broken font backend: "
            f"{details} This is a release packaging error; use a build whose Tk library has "
            "Xft/fontconfig support."
        )
    return (
        f"Jasna cannot start because this Python installation has a broken Tk font backend: "
        f"{details} Create the venv from a distribution-provided Python with its matching Tk "
        "package and Xft/fontconfig support. Standalone uv-managed Python builds may bundle "
        "a no-Xft Tk build."
    )


def font_backend_status_json(status: FontBackendStatus) -> str:
    payload = asdict(status)
    payload["ok"] = font_backend_problem(status) is None
    payload["problem"] = font_backend_problem(status)
    return json.dumps(payload, sort_keys=True)

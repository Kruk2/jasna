from __future__ import annotations

__all__ = ["JasnaApp", "run_gui"]


def run_gui() -> None:
    from jasna.gui.app import run_gui as _run_gui

    _run_gui()


def __getattr__(name: str):
    if name == "JasnaApp":
        from jasna.gui.app import JasnaApp

        return JasnaApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

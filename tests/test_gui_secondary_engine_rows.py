"""The AMD panel must show only the rows the selected engine actually uses.

``amf-sr`` rides FFmpeg's ``sr_amf`` filter (algorithm / timeout / ffmpeg-path,
2x/4x/6x/8x) and has no checkpoint; real-esr / realesrgan run in process (a
checkpoint, 2x/4x only) and have none of the AMF knobs.  The shared sharpness box
swaps its tooltip to match the engine.  These drive ``_refresh_amd_rows`` with
fake widgets so no Tk root is needed.
"""
from __future__ import annotations

from types import SimpleNamespace

from jasna.gui.settings_sections.secondary import SecondarySection
from jasna.gui.settings_sections.widgets import get_tooltip

_ROW_NAMES = ("engine", "model", "scale", "algorithm", "sharpness", "timeout", "path")


class _Row:
    def __init__(self, name: str):
        self.name = name
        self.packed = False

    def pack(self, **kwargs) -> None:
        self.packed = True

    def pack_forget(self) -> None:
        self.packed = False


class _Scale:
    def __init__(self, value: str):
        self._value = value
        self.values: list[str] | None = None

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value

    def configure(self, *, values=None) -> None:
        if values is not None:
            self.values = list(values)


class _Tip:
    def __init__(self) -> None:
        self.text: str | None = None

    def set_text(self, text: str) -> None:
        self.text = text


class _Entry:
    """Just enough of a CTkEntry for the sharpness display normalization."""

    def __init__(self, text: str):
        self._text = text

    def get(self) -> str:
        return self._text

    def delete(self, first: int, last: str | int) -> None:
        self._text = ""

    def insert(self, index: int, text: str) -> None:
        self._text = text


def _section(scale: str = "4x", sharpness: str = "-1"):
    rows = {name: _Row(name) for name in _ROW_NAMES}
    ns = SimpleNamespace(
        _amd_rows=[(rows[name], dict(fill="x")) for name in _ROW_NAMES],
        _amd_amf_only_rows=(rows["algorithm"], rows["timeout"], rows["path"]),
        _amd_net_only_rows=(rows["model"],),
        _widgets={
            "amd_upscale_scale": _Scale(scale),
            "amd_upscale_sharpness": _Entry(sharpness),
        },
        _amd_sharpness_tooltip=_Tip(),
    )
    return ns, rows


def _refresh(ns, engine: str):
    SecondarySection._refresh_amd_rows(ns, engine)


def test_network_engine_shows_checkpoint_and_24_scale_only() -> None:
    ns, rows = _section(scale="4x")
    _refresh(ns, "real-esr")

    assert rows["engine"].packed and rows["model"].packed
    assert rows["scale"].packed and rows["sharpness"].packed
    assert not rows["algorithm"].packed
    assert not rows["timeout"].packed
    assert not rows["path"].packed
    assert ns._widgets["amd_upscale_scale"].values == ["2x", "4x"]
    assert ns._amd_sharpness_tooltip.text == get_tooltip("amd_sharpness_patch")


def test_amf_engine_shows_amf_rows_and_full_scale_and_hides_the_checkpoint() -> None:
    ns, rows = _section(scale="4x")
    _refresh(ns, "amf-sr")

    assert rows["engine"].packed and rows["scale"].packed and rows["sharpness"].packed
    assert rows["algorithm"].packed and rows["timeout"].packed and rows["path"].packed
    assert not rows["model"].packed
    assert ns._widgets["amd_upscale_scale"].values == ["2x", "4x", "6x", "8x"]
    assert ns._amd_sharpness_tooltip.text == get_tooltip("amd_sharpness")


def test_scale_snaps_off_an_invalid_value_when_switching_to_a_network_engine() -> None:
    ns, _rows = _section(scale="8x")
    _refresh(ns, "real-esr")
    assert ns._widgets["amd_upscale_scale"].get() == "4x"


def test_an_already_valid_scale_is_left_alone() -> None:
    ns, _rows = _section(scale="2x")
    _refresh(ns, "real-esr")
    assert ns._widgets["amd_upscale_scale"].get() == "2x"


def test_realesrgan_is_treated_as_a_network_engine() -> None:
    ns, rows = _section(scale="6x")
    _refresh(ns, "realesrgan")

    assert rows["model"].packed
    assert not rows["algorithm"].packed
    assert not rows["timeout"].packed
    assert not rows["path"].packed
    assert ns._widgets["amd_upscale_scale"].get() == "4x"
    assert ns._amd_sharpness_tooltip.text == get_tooltip("amd_sharpness_patch")


def test_negative_sharpness_displays_as_zero_under_a_network_engine() -> None:
    ns, _rows = _section(sharpness="-1")
    _refresh(ns, "real-esr")
    assert ns._widgets["amd_upscale_sharpness"].get() == "0"

    ns, _rows = _section(sharpness="-1")
    _refresh(ns, "realesrgan")
    assert ns._widgets["amd_upscale_sharpness"].get() == "0"


def test_negative_sharpness_stays_minus_one_under_amf() -> None:
    ns, _rows = _section(sharpness="-1")
    _refresh(ns, "amf-sr")
    assert ns._widgets["amd_upscale_sharpness"].get() == "-1"


def test_positive_sharpness_is_left_alone() -> None:
    ns, _rows = _section(sharpness="0.3")
    _refresh(ns, "real-esr")
    assert ns._widgets["amd_upscale_sharpness"].get() == "0.3"


def test_non_numeric_sharpness_is_left_alone() -> None:
    ns, _rows = _section(sharpness="")
    _refresh(ns, "real-esr")
    assert ns._widgets["amd_upscale_sharpness"].get() == ""

from types import SimpleNamespace

from jasna.gui.settings_panel import SettingsPanel


class _Slider:
    def __init__(self, value: int):
        self.value = value
        self.config: dict[str, int] = {}

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)

    def get(self) -> int:
        return self.value

    def set(self, value: int) -> None:
        self.value = value


class _Label:
    def __init__(self):
        self.text = ""

    def configure(self, *, text: str) -> None:
        self.text = text


def test_temporal_filter_sliders_stay_below_max_clip_size() -> None:
    gap = _Slider(10)
    duration = _Slider(10)
    gap_label = _Label()
    duration_label = _Label()
    panel = SimpleNamespace(
        _widgets={
            "max_detection_gap": gap,
            "max_detection_gap_val": gap_label,
            "min_detection_duration": duration,
            "min_detection_duration_val": duration_label,
        }
    )

    SettingsPanel._sync_temporal_filter_limits(panel, max_clip_size=10)

    assert gap.config == {"to": 9, "number_of_steps": 9}
    assert duration.config == {"to": 9, "number_of_steps": 9}
    assert gap.value == 9
    assert duration.value == 9
    assert gap_label.text == "9"
    assert duration_label.text == "9"


def test_temporal_filter_sliders_keep_normal_upper_limit() -> None:
    gap = _Slider(2)
    duration = _Slider(2)
    panel = SimpleNamespace(
        _widgets={
            "max_detection_gap": gap,
            "max_detection_gap_val": _Label(),
            "min_detection_duration": duration,
            "min_detection_duration_val": _Label(),
        }
    )

    SettingsPanel._sync_temporal_filter_limits(panel, max_clip_size=90)

    assert gap.config == {"to": 10, "number_of_steps": 10}
    assert duration.config == {"to": 10, "number_of_steps": 10}
    assert gap.value == 2
    assert duration.value == 2

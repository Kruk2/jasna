"""Old settings.json presets (PyNvVideoCodec-era) must survive loading: unknown
fields dropped, encoder custom args translated to hevc_nvenc names."""
from jasna.gui.models import AppSettings, _migrate_preset_dict
from jasna.media.encoder_settings import parse_encoder_settings


def test_unknown_fields_are_dropped():
    old = {"codec": "hevc", "encoder_cq": 22, "pynv_preset": "P7"}
    migrated = _migrate_preset_dict(old)
    settings = AppSettings(**migrated)
    assert settings.encoder_cq == 22
    assert "pynv_preset" not in migrated


def test_legacy_codec_spellings_normalized():
    for legacy, canonical in [
        ("HEVC", "hevc"), ("h265", "hevc"), ("H.265", "hevc"),
        ("H264", "h264"), ("H.264", "h264"), ("avc", "h264"),
        ("AV1", "av1"), ("av01", "av1"),
    ]:
        migrated = _migrate_preset_dict({"codec": legacy})
        assert migrated["codec"] == canonical, legacy
        assert AppSettings(**migrated).codec == canonical


def test_unknown_codec_falls_back_to_hevc():
    assert _migrate_preset_dict({"codec": "prores"})["codec"] == "hevc"


def test_custom_cq_moves_to_literal_preset_field():
    migrated = _migrate_preset_dict(
        {
            "codec": "h264",
            "encoder_cq": 28,
            "encoder_custom_args": "cq=22,rc-lookahead=32",
        }
    )

    assert migrated["encoder_cq"] == 22
    assert parse_encoder_settings(migrated["encoder_custom_args"]) == {
        "rc-lookahead": 32
    }


def test_amf_quality_alias_moves_to_literal_preset_field():
    migrated = _migrate_preset_dict(
        {
            "codec": "av1",
            "encoder_custom_args": "qvbr_quality_level=31,g=120",
        }
    )

    assert migrated["encoder_cq"] == 31
    assert parse_encoder_settings(migrated["encoder_custom_args"]) == {"g": 120}


def test_gui_codec_label_maps_round_trip():
    from jasna.gui.settings_sections.encoding import (
        CODEC_CANONICAL_TO_LABEL,
        CODEC_LABEL_TO_CANONICAL,
    )

    assert set(CODEC_CANONICAL_TO_LABEL) == {"hevc", "h264", "av1"}
    for canonical, label in CODEC_CANONICAL_TO_LABEL.items():
        assert CODEC_LABEL_TO_CANONICAL[label] == canonical
        # .lower() on a display label must never be used as the canonical value
        if canonical != "av1":
            assert label.lower() != canonical


def test_unreadable_custom_args_are_kept_with_a_warning(caplog):
    migrated = _migrate_preset_dict({"encoder_custom_args": "not==valid,,=x"})

    assert migrated["encoder_custom_args"] == "not==valid,,=x"
    assert "unreadable custom encoder settings" in caplog.text

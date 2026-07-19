"""Old settings.json presets (PyNvVideoCodec-era) must survive loading: unknown
fields dropped, encoder custom args translated to hevc_nvenc names."""
from jasna.gui.models import AppSettings, _migrate_encoder_custom_args, _migrate_preset_dict
from jasna.media import parse_encoder_settings, validate_encoder_settings


def test_unknown_fields_are_dropped():
    old = {"codec": "hevc", "encoder_cq": 22, "pynv_preset": "P7"}
    migrated = _migrate_preset_dict(old)
    settings = AppSettings(**migrated)
    assert settings.encoder_cq == 22
    assert "pynv_preset" not in migrated


def test_old_encoder_args_are_translated():
    old = "lookahead=32,temporalaq=1,aq=8,initqp=17,gop=250,nonrefp=1,tflevel=0,maxbitrate=5000,vbvbufsize=10000"
    migrated = parse_encoder_settings(_migrate_encoder_custom_args(old))
    assert migrated["rc-lookahead"] == 32
    assert migrated["temporal-aq"] == 1
    assert migrated["spatial_aq"] == 1
    assert migrated["aq-strength"] == 8
    assert migrated["init_qpI"] == 17
    assert migrated["init_qpP"] == 17
    assert migrated["init_qpB"] == 17
    assert migrated["g"] == 250
    assert migrated["nonref_p"] == 1
    assert migrated["tf_level"] == 0
    assert migrated["maxrate"] == 5000
    assert migrated["bufsize"] == 10000
    validate_encoder_settings(migrated)


def test_preset_and_tuning_values_translated():
    migrated = parse_encoder_settings(_migrate_encoder_custom_args("preset=P5,tuning_info=high_quality"))
    assert migrated == {"preset": "p5", "tune": "hq"}
    validate_encoder_settings(migrated)


def test_vbvinit_is_dropped():
    assert parse_encoder_settings(_migrate_encoder_custom_args("vbvinit=0,cq=22")) == {"cq": 22}


def test_new_style_args_pass_through():
    new = "cq=22,rc-lookahead=32,b_ref_mode=middle"
    assert parse_encoder_settings(_migrate_encoder_custom_args(new)) == parse_encoder_settings(new)


def test_garbage_args_returned_unchanged():
    assert _migrate_encoder_custom_args("not==valid,,=x") == "not==valid,,=x"


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


def test_gui_codec_label_maps_round_trip():
    from jasna.gui.settings_panel import (
        CODEC_CANONICAL_TO_LABEL,
        CODEC_LABEL_TO_CANONICAL,
        translate_cq_for_codec,
    )

    assert set(CODEC_CANONICAL_TO_LABEL) == {"hevc", "h264", "av1"}
    for canonical, label in CODEC_CANONICAL_TO_LABEL.items():
        assert CODEC_LABEL_TO_CANONICAL[label] == canonical
        # .lower() on a display label must never be used as the canonical value
        if canonical != "av1":
            assert label.lower() != canonical

    assert translate_cq_for_codec(22, "hevc", "av1") == 29
    assert translate_cq_for_codec(29, "av1", "hevc") == 22
    assert translate_cq_for_codec(22, "hevc", "h264") == 22
    assert translate_cq_for_codec(35, "hevc", "av1") == 35

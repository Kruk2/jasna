"""Old settings.json presets (PyNvVideoCodec-era) must survive loading: unknown
fields dropped, encoder custom args translated to hevc_nvenc names."""
from jasna.gui.models import AppSettings, _migrate_encoder_custom_args, _migrate_preset_dict
from jasna.media import parse_encoder_settings, validate_encoder_settings


def test_unknown_fields_are_dropped():
    old = {"codec": "hevc", "encoder_cq": 22, "working_directory": "/tmp/x"}
    migrated = _migrate_preset_dict(old)
    settings = AppSettings(**migrated)
    assert settings.encoder_cq == 22
    assert "working_directory" not in migrated


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

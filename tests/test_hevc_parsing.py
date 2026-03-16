import pytest

from jasna.media.video_encoder import (
    _parse_hevc_nal_units,
    _is_hevc_keyframe,
    _extract_hevc_extradata,
)


def _make_nal(nal_type: int, payload: bytes = b"\x00") -> bytes:
    first_byte = (nal_type << 1) & 0x7E
    return b"\x00\x00\x00\x01" + bytes([first_byte, 0x00]) + payload


class TestParseHevcNalUnits:
    def test_empty_data(self):
        assert _parse_hevc_nal_units(b"") == []

    def test_single_nal_unit(self):
        data = _make_nal(19, b"\xAA\xBB")
        nals = _parse_hevc_nal_units(data)
        assert len(nals) == 1
        assert nals[0][0] == 19  # IDR_W_RADL

    def test_two_nal_units(self):
        data = _make_nal(32, b"\x01\x02") + _make_nal(33, b"\x03\x04")
        nals = _parse_hevc_nal_units(data)
        assert len(nals) == 2
        assert nals[0][0] == 32  # VPS
        assert nals[1][0] == 33  # SPS

    def test_three_byte_start_code(self):
        first_byte = (19 << 1) & 0x7E
        data = b"\x00\x00\x01" + bytes([first_byte, 0x00]) + b"\xAA"
        nals = _parse_hevc_nal_units(data)
        assert len(nals) == 1
        assert nals[0][0] == 19

    def test_nal_types_extracted_correctly(self):
        data = _make_nal(32) + _make_nal(33) + _make_nal(34) + _make_nal(19)
        nals = _parse_hevc_nal_units(data)
        types = [n[0] for n in nals]
        assert types == [32, 33, 34, 19]


class TestIsHevcKeyframe:
    def test_idr_w_radl(self):
        assert _is_hevc_keyframe(_make_nal(19)) is True

    def test_idr_n_lp(self):
        assert _is_hevc_keyframe(_make_nal(20)) is True

    def test_cra_nut(self):
        assert _is_hevc_keyframe(_make_nal(21)) is True

    def test_bla_types(self):
        assert _is_hevc_keyframe(_make_nal(16)) is True
        assert _is_hevc_keyframe(_make_nal(17)) is True
        assert _is_hevc_keyframe(_make_nal(18)) is True

    def test_non_keyframe(self):
        assert _is_hevc_keyframe(_make_nal(1)) is False  # TRAIL_R
        assert _is_hevc_keyframe(_make_nal(0)) is False  # TRAIL_N

    def test_empty_data(self):
        assert _is_hevc_keyframe(b"") is False

    def test_mixed_nals_with_keyframe(self):
        data = _make_nal(32) + _make_nal(33) + _make_nal(34) + _make_nal(19) + _make_nal(1)
        assert _is_hevc_keyframe(data) is True

    def test_mixed_nals_without_keyframe(self):
        data = _make_nal(32) + _make_nal(33) + _make_nal(34) + _make_nal(1)
        assert _is_hevc_keyframe(data) is False


class TestExtractHevcExtradata:
    def test_extracts_vps_sps_pps(self):
        vps_payload = b"\x01\x02\x03"
        sps_payload = b"\x04\x05\x06"
        pps_payload = b"\x07\x08\x09"
        data = _make_nal(32, vps_payload) + _make_nal(33, sps_payload) + _make_nal(34, pps_payload)
        extradata = _extract_hevc_extradata(data)
        assert len(extradata) > 0
        assert b"\x00\x00\x00\x01" in extradata

    def test_ignores_non_param_nals(self):
        data = _make_nal(19, b"\xAA\xBB") + _make_nal(1, b"\xCC\xDD")
        extradata = _extract_hevc_extradata(data)
        assert extradata == b""

    def test_empty_data(self):
        assert _extract_hevc_extradata(b"") == b""

    def test_mixed_param_and_slice_nals(self):
        data = _make_nal(32, b"\x01") + _make_nal(19, b"\x02") + _make_nal(33, b"\x03")
        extradata = _extract_hevc_extradata(data)
        nals = _parse_hevc_nal_units(extradata)
        types = [n[0] for n in nals]
        assert 32 in types
        assert 33 in types
        assert 19 not in types

"""The single-decode look-ahead cap must not shrink the metadata queue below the window
the detect thread needs to reach a clip boundary.

That shrinkage is what deadlocked the 8K VR180 pipeline: blend parks on the frame of a
clip that is still open, detect keeps publishing until the boundary - but the scaled
depth (25 entries against clip 90) filled first, the tracker never reached the boundary,
the clip was never emitted and nothing moved for 14 minutes. The floor is now kept
whole; these tests pin that down.
"""
from jasna import pipeline_threads as pt


def test_metadata_keeps_the_clip_window_while_the_others_shrink(monkeypatch):
    monkeypatch.setenv("JASNA_HOST_YUV_BUDGET_GB", "6")
    clip, secondary, encode, metadata, note = pt._bounded_lookahead_depths(
        max_clip_size=90, secondary_workers=1, height=4096, width=8192, ten_bit=True
    )
    assert metadata >= 180, "the 2x-clip window must survive the cap"
    assert clip < 90, "the other queues do absorb the budget"
    assert note, "the cap must say so"


def test_small_frames_are_left_alone(monkeypatch):
    monkeypatch.setenv("JASNA_HOST_YUV_BUDGET_GB", "6")
    clip, secondary, encode, metadata, note = pt._bounded_lookahead_depths(
        max_clip_size=24, secondary_workers=1, height=1080, width=1920, ten_bit=False
    )
    assert (clip, metadata) == (24, 48)
    assert note == ""


def test_disabling_the_budget_returns_the_configured_depths(monkeypatch):
    monkeypatch.setenv("JASNA_HOST_YUV_BUDGET_GB", "0")
    clip, secondary, encode, metadata, note = pt._bounded_lookahead_depths(
        max_clip_size=90, secondary_workers=2, height=4320, width=7680, ten_bit=True
    )
    assert metadata == 180
    assert clip == 90 and secondary == 180
    assert note == ""

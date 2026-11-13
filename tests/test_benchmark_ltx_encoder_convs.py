import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_encoder_convs as convs

from jasna.models.ltx_vae.video_encoder import ConvLaunch

LAUNCH = ConvLaunch(256, 128, 64, 8, 3, 8)
PROMOTED = ConvLaunch(128, 256, 64, 8, 3, 8, promoted=True)


@pytest.mark.parametrize(("head", "accepted"), [
    ({512: 1.009, 768: 0.98}, True),
    ({512: 1.011, 768: 0.95}, False),
    ({512: 1.004, 768: 1.002}, False),
    ({512: 0.95, 768: 0.97}, True),
])
def test_an_entry_may_move_at_most_one_percent_from_fp32_per_size_and_not_on_average(head, accepted):
    assert convs.acceptable({512: 1.0, 768: 1.0}, head) is accepted


def test_the_1024_channel_convs_are_also_tried_without_and_promoted():
    heads = convs.candidates({(128, 128): LAUNCH, (1024, 1024): LAUNCH}, PROMOTED)

    assert {name: (head.down_blocks, head.triton_convs) for name, head in heads.items()} == {
        "kitchen head": (6, {}),
        "entry": (7, {(128, 128): LAUNCH, (1024, 1024): LAUNCH}),
        "entry without 1024": (6, {(128, 128): LAUNCH}),
        "entry, 1024 promoted": (7, {(128, 128): LAUNCH, (1024, 1024): PROMOTED}),
    }


def test_a_key_shared_by_several_convs_is_decided_by_their_count_weighted_total():
    slow, fast = LAUNCH, ConvLaunch(128, 128, 32, 4, 4, 8)
    shapes = [
        ((256, 256), 12, 17.8, {slow: 18.9, fast: 19.5}),  # residual convs: kitchen wins
        ((256, 256), 1, 18.0, {slow: 16.0, fast: 15.0}),  # one downsample: Triton wins
        ((128, 64), 1, 5.9, {slow: 3.0, fast: 3.5}),
    ]

    assert convs.key_launches(shapes) == {(128, 64): slow}


def test_a_shared_key_takes_the_launch_fastest_over_all_its_convs():
    slow, fast = LAUNCH, ConvLaunch(128, 128, 32, 4, 4, 8)
    shapes = [((256, 256), 12, 20.0, {slow: 10.0, fast: 11.0}), ((256, 256), 1, 20.0, {slow: 15.0, fast: 5.0})]

    assert convs.key_launches(shapes) == {(256, 256): slow}


def test_an_entry_without_1024_channel_convs_is_tried_alone():
    assert set(convs.candidates({(128, 128): LAUNCH}, None)) == {"kitchen head", "entry"}

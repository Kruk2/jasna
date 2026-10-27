from __future__ import annotations

import pytest
import torch

from jasna import accelerator
from jasna.os_utils import env_flag
from jasna.restorer.basicvsrpp_hip_graph import (
    HipGraphReplay,
    graph_lengths,
)


class TestEnvFlag:
    def test_unset_and_empty_use_default(self, monkeypatch):
        monkeypatch.delenv("JASNA_TEST_FLAG", raising=False)
        assert env_flag("JASNA_TEST_FLAG") is False
        assert env_flag("JASNA_TEST_FLAG", default=True) is True
        monkeypatch.setenv("JASNA_TEST_FLAG", "")
        assert env_flag("JASNA_TEST_FLAG", default=True) is True

    def test_common_spellings(self, monkeypatch):
        for value in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("JASNA_TEST_FLAG", value)
            assert env_flag("JASNA_TEST_FLAG") is True, value
        for value in ("0", "false", "NO", "off"):
            monkeypatch.setenv("JASNA_TEST_FLAG", value)
            assert env_flag("JASNA_TEST_FLAG", default=True) is False, value

    def test_tolerates_cmd_exe_trailing_space(self, monkeypatch):
        # ``set FLAG=1 && prog`` binds "1 " (cmd.exe folds the space before &&),
        # which a literal ``== "1"`` read would silently treat as off.
        monkeypatch.setenv("JASNA_TEST_FLAG", "1 ")
        assert env_flag("JASNA_TEST_FLAG") is True
        monkeypatch.setenv("JASNA_TEST_FLAG", "0 ")
        assert env_flag("JASNA_TEST_FLAG", default=True) is False

    def test_unrecognised_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("JASNA_TEST_FLAG", "maybe")
        assert env_flag("JASNA_TEST_FLAG") is False
        assert env_flag("JASNA_TEST_FLAG", default=True) is True


class TestGraphLengths:
    def test_includes_the_clip_size_and_power_of_two_anchors(self):
        lengths = graph_lengths(90, 0)
        assert 90 in lengths
        assert {16, 32, 64}.issubset(set(lengths))
        assert all(1 < n <= 90 for n in lengths)

    def test_overlap_adds_the_second_dominant_length(self):
        assert 74 in graph_lengths(90, 8)

    def test_extra_lengths_from_env(self, monkeypatch):
        monkeypatch.setenv("JASNA_BV_HIP_GRAPH_LENGTHS", "52, 40, bogus, 7")
        lengths = graph_lengths(90, 0)
        assert 52 in lengths and 40 in lengths
        assert 7 in lengths, "an explicit length is honoured even below the anchors"
        assert all(isinstance(n, int) for n in lengths)

    def test_short_clip_size_has_no_anchors(self):
        lengths = graph_lengths(8, 0)
        assert lengths == [8]


class TestPoolGuard:
    def test_guard_is_released_when_the_graphs_are(self, monkeypatch):
        """Dropping the graphs must hand the release back.

        A pool that stays registered makes every later release a no-op, so the
        reserve walks up to the card's ceiling (measured at 8K: 194 offload calls
        that could not free anything).
        """
        monkeypatch.setattr(accelerator, "_graph_pool_owners", set())
        r = HipGraphReplay(torch.device("cpu"))
        assert r._guard_held is False

        r._hold_guard(True)
        assert id(r) in accelerator._graph_pool_owners
        r.close()
        assert id(r) not in accelerator._graph_pool_owners
        assert r._guard_held is False

    def test_release_is_a_no_op_while_a_pool_is_registered(self, monkeypatch):
        monkeypatch.setattr(accelerator, "_graph_pool_owners", set())
        calls = []
        monkeypatch.setattr(accelerator, "synchronize", lambda *a, **k: calls.append("sync"))
        monkeypatch.setattr(accelerator, "empty_cache", lambda *a, **k: calls.append("free"))
        monkeypatch.setattr(accelerator.torch.cuda, "is_available", lambda: True)

        r = HipGraphReplay(torch.device("cpu"))
        r._hold_guard(True)
        before = accelerator.hip_graph_pool_generation()
        accelerator.release_vram_cache()
        assert calls == [], "a release under a live graph pool must not run"
        assert accelerator.hip_graph_pool_generation() == before, (
            "a skipped release must not move the generation, or owners throw away "
            "perfectly good graphs"
        )

        r.close()
        accelerator.release_vram_cache()
        assert calls == ["sync", "free"]
        assert accelerator.hip_graph_pool_generation() == before + 1


class TestReplay:
    def test_no_graphs_means_eager(self):
        r = HipGraphReplay(torch.device("cpu"))
        r.enabled = False
        assert r.replay(torch.zeros(1, 4, 3, 256, 256)) is None
        assert len(r) == 0

    def test_uncaptured_length_means_eager(self):
        r = HipGraphReplay(torch.device("cpu"))
        r._graphs[16] = (None, None, None)  # type: ignore[assignment]
        assert r.replay(torch.zeros(1, 90, 3, 256, 256)) is None

    def test_a_release_outside_the_guard_drops_the_graphs(self, monkeypatch):
        monkeypatch.setattr(accelerator, "_graph_pool_owners", set())
        r = HipGraphReplay(torch.device("cpu"))
        r._graphs[16] = (object(), object(), object())  # type: ignore[assignment]
        r._hold_guard(True)
        r._generation = 0
        monkeypatch.setattr(r, "_drop_if_over_reserve", lambda: False)

        # The generation moving behind the owner's back means a release slipped
        # through; replay must not trust the captured pool afterwards.
        monkeypatch.setattr(accelerator, "_hip_graph_pool_generation", 5)
        assert r.replay(torch.zeros(1, 16, 3, 256, 256)) is None
        assert len(r) == 0
        assert id(r) not in accelerator._graph_pool_owners

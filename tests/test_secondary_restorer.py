"""Tests for SecondaryRestorerAdapter and _IdentitySecondaryRestorer."""
from __future__ import annotations

import torch

from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer, SecondaryRestorerAdapter
from jasna.restorer.restoration_pipeline import _IdentitySecondaryRestorer


class TestIdentitySecondaryRestorer:
    def test_restore_returns_uint8_kept_frames(self):
        restorer = _IdentitySecondaryRestorer()
        frames = torch.rand((5, 3, 256, 256))
        result = restorer.restore(frames, keep_start=1, keep_end=4)
        assert len(result) == 3
        assert all(f.dtype == torch.uint8 for f in result)
        assert all(f.shape == (3, 256, 256) for f in result)

    def test_restore_empty_range(self):
        restorer = _IdentitySecondaryRestorer()
        frames = torch.rand((5, 3, 256, 256))
        assert restorer.restore(frames, keep_start=3, keep_end=3) == []

    def test_name_and_workers(self):
        restorer = _IdentitySecondaryRestorer()
        assert restorer.name == "identity"
        assert restorer.num_workers == 1


class TestSecondaryRestorerAdapter:
    def _make_sync_restorer(self):
        class _FakeSync:
            name = "fake"
            num_workers = 1
            closed = False

            def restore(self, frames_256, *, keep_start, keep_end):
                t = frames_256.shape[0]
                ks = max(0, keep_start)
                ke = min(t, keep_end)
                return [frames_256[i].mul(255).clamp(0, 255).to(torch.uint8) for i in range(ks, ke)]

            def close(self):
                self.closed = True

        return _FakeSync()

    def test_push_pop_cycle(self):
        sync = self._make_sync_restorer()
        adapter = SecondaryRestorerAdapter(sync)

        frames = torch.rand((5, 3, 256, 256))
        seq = adapter.push_clip(frames, 1, 4)
        assert seq == 0

        completed = adapter.pop_completed()
        assert len(completed) == 1
        assert completed[0][0] == 0
        assert len(completed[0][1]) == 3

    def test_pop_completed_clears(self):
        sync = self._make_sync_restorer()
        adapter = SecondaryRestorerAdapter(sync)

        adapter.push_clip(torch.rand((3, 3, 256, 256)), 0, 3)
        assert len(adapter.pop_completed()) == 1
        assert len(adapter.pop_completed()) == 0

    def test_sequence_increments(self):
        sync = self._make_sync_restorer()
        adapter = SecondaryRestorerAdapter(sync)

        s0 = adapter.push_clip(torch.rand((3, 3, 256, 256)), 0, 3)
        s1 = adapter.push_clip(torch.rand((3, 3, 256, 256)), 0, 3)
        assert s0 == 0
        assert s1 == 1
        completed = adapter.pop_completed()
        assert [c[0] for c in completed] == [0, 1]

    def test_flush_all_is_noop(self):
        adapter = SecondaryRestorerAdapter(self._make_sync_restorer())
        adapter.flush_all()

    def test_close_delegates(self):
        sync = self._make_sync_restorer()
        adapter = SecondaryRestorerAdapter(sync)
        adapter.close()
        assert sync.closed is True

    def test_close_no_close_method(self):
        class _NoClose:
            name = "nope"
            num_workers = 1
            def restore(self, frames_256, *, keep_start, keep_end):
                return []

        adapter = SecondaryRestorerAdapter(_NoClose())
        adapter.close()

    def test_name_and_num_workers_delegated(self):
        sync = self._make_sync_restorer()
        adapter = SecondaryRestorerAdapter(sync)
        assert adapter.name == "fake"
        assert adapter.num_workers == 1

    def test_satisfies_async_protocol(self):
        adapter = SecondaryRestorerAdapter(self._make_sync_restorer())
        assert isinstance(adapter, AsyncSecondaryRestorer)

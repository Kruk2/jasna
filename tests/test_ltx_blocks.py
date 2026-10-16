import pytest
import torch

from jasna.ltx.blocks import BlockStore, OpaqueBlockStore

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _blocks(count: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "a.qdata": torch.randint(-127, 128, (64, 48), dtype=torch.int8, generator=generator),
            "a.scales": torch.rand(64, generator=generator),
            "norm.weight": torch.randn(33, generator=generator).to(torch.bfloat16),
        }
        for _ in range(count)
    ]


@needs_cuda
@pytest.mark.parametrize("resident", [0, 2, 5])
def test_streamed_and_resident_blocks_hand_out_the_same_weights(resident):
    blocks = _blocks(5)
    store = BlockStore([dict(b) for b in blocks], torch.device("cuda"), resident=resident)
    for _ in range(3):
        for index, expected in enumerate(blocks):
            weights = store.acquire(index)
            for name, tensor in expected.items():
                assert torch.equal(weights[name].cpu(), tensor)
            store.release(index)
    store.close()


@needs_cuda
@pytest.mark.parametrize("resident, warns", [(5, False), (3, True)])
def test_streaming_blocks_from_ram_is_a_warning(resident, warns, caplog):
    with caplog.at_level("WARNING", logger="jasna.ltx.blocks"):
        BlockStore(_blocks(5), torch.device("cuda"), resident=resident).close()
    assert any("2 of 5 transformer blocks stream from RAM" in r.getMessage() for r in caplog.records) is warns


@needs_cuda
def test_packed_blocks_are_released_from_the_input():
    blocks = _blocks(3)
    BlockStore(blocks, torch.device("cuda"), resident=1).close()
    assert blocks == [{}, {}, {}]


def test_file_order_groups_fill_a_stage_and_read_cpu_tensors_in_place():
    from jasna.ltx.blocks import _file_order_groups

    gpu = torch.empty(300, dtype=torch.uint8, device="meta")
    cpu = torch.empty(10_000, dtype=torch.uint8)
    targets = {"a": gpu, "b": cpu, "c": gpu, "d": gpu}
    assert _file_order_groups(["a", "b", "c", "d"], targets, 1024) == [
        (("a", "b", "c"), {"a": 0, "c": 512}),
        (("d",), {"d": 0}),
    ]
    assert _file_order_groups(["b"], targets, 0) == [(("b",), {})]


@needs_cuda
def test_blocks_are_read_in_file_order(tmp_path):
    from safetensors.torch import save_file

    from jasna.ltx.model_files import open_tensors

    path = tmp_path / "blocks.safetensors"
    tensors = {f"blocks.{index}.{part}": torch.full((4,), index, dtype=dtype)
               for index in range(3) for part, dtype in (("a", torch.int8), ("b", torch.float32))}
    save_file(tensors, str(path))
    names = [{part: f"blocks.{index}.{part}" for part in ("a", "b")} for index in range(3)]
    with open_tensors(path) as reader:
        stored = reader.offset_keys()
        load = reader.read_tensors_into
        operations = []

        def observe(group_names, groups):
            operations.append(group_names)
            load(group_names, groups)

        reader.read_tensors_into = observe
        store = BlockStore.from_reader(reader, names, torch.device("cuda"), resident=3)
    try:
        assert [name for group in operations[0] for name in group] == stored
        for index in range(3):
            assert all(torch.equal(tensor.cpu(), tensors[f"blocks.{index}.{part}"])
                       for part, tensor in store.acquire(index).items())
    finally:
        store.close()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_opaque_load_preserves_native_transaction_failure(error_type, monkeypatch):
    from jasna.ltx.native.aot import BlockDims

    primary = error_type("original native transaction failure")
    operations = []

    class Reader:
        def tensor_specs(self):
            return {"blocks.0.weight": (torch.float32, (2,))}

        def private_load(self, device, resident):
            operations.append((device, resident))
            raise primary

    monkeypatch.setattr(BlockDims, "from_specs", lambda specs: type("Dims", (), {"dim": 8, "heads": 2}))
    with pytest.raises(error_type) as caught:
        OpaqueBlockStore.from_reader(
            Reader(), [{"weight": "blocks.0.weight"}], torch.device("cuda:1"), resident=1,
        )
    assert caught.value is primary
    assert operations == [(1, 1)]


def test_opaque_load_rejects_cpu_before_decryption(monkeypatch):
    from jasna.ltx.native.aot import BlockDims
    from types import SimpleNamespace

    monkeypatch.setattr(BlockDims, "from_specs", lambda specs: SimpleNamespace(dim=8, heads=2))
    reader = SimpleNamespace(tensor_specs=lambda: {"blocks.0.weight": (torch.float32, (2,))},
                             private_load=lambda *args: pytest.fail("native decryption attempted"))
    with pytest.raises(ValueError, match="CUDA device"):
        OpaqueBlockStore.from_reader(
            reader, [{"weight": "blocks.0.weight"}], torch.device("cpu"), resident=1,
        )


@needs_cuda
def test_streamed_blocks_are_page_locked_without_power_of_two_rounding():
    torch.cuda.init()
    before = torch.cuda.memory.host_memory_stats().get("allocated_bytes.current", 0)
    store = BlockStore(_blocks(4), torch.device("cuda"), resident=1)
    assert all(buffer.is_pinned() for buffer in store._host)
    assert torch.cuda.memory.host_memory_stats().get("allocated_bytes.current", 0) == before
    store.close()


@needs_cuda
def test_refused_page_locking_keeps_the_rest_in_regular_ram_without_unlocking(monkeypatch, caplog):
    import jasna.ltx.blocks as blocks_module

    register = blocks_module._register
    calls = []

    def refuse_after_three(buffer):
        calls.append(buffer)
        return register(buffer) if len(calls) <= 3 else 2

    def unlocking_fails_at_the_windows_cap(buffers):
        if buffers:
            raise RuntimeError("cuMemHostUnregister failed: CUDA_ERROR_OUT_OF_MEMORY: out of memory")

    monkeypatch.setattr(blocks_module, "_register", refuse_after_three)
    monkeypatch.setattr(blocks_module, "_unregister", unlocking_fails_at_the_windows_cap)
    blocks = _blocks(6)
    with caplog.at_level("WARNING", logger="jasna.ltx.blocks"):
        store = BlockStore([dict(b) for b in blocks], torch.device("cuda"), resident=0)
    assert [buffer.is_pinned() for buffer in store._host] == [True, True, True, False, False, False]
    assert any("(CUDA error 2); 3 of 6 weight buffers use regular RAM" in r.getMessage() for r in caplog.records)
    for _ in range(2):
        for index, expected in enumerate(blocks):
            weights = store.acquire(index)
            for name, tensor in expected.items():
                assert torch.equal(weights[name].cpu(), tensor)
            store.release(index)
    monkeypatch.undo()
    store.close()
    assert not any(buffer.is_pinned() for buffer in calls[:3])


@needs_cuda
def test_page_locking_stops_at_the_budget_before_the_system_refuses(monkeypatch, caplog):
    import jasna.ltx.blocks as blocks_module

    blocks = _blocks(4)
    block_bytes = BlockStore([dict(b) for b in blocks[:1]], torch.device("cuda"), resident=0)
    one = block_bytes._host[0].nbytes
    block_bytes.close()
    registered = []
    register = blocks_module._register
    monkeypatch.setattr(blocks_module, "_page_lock_budget", lambda: 2 * one + one // 2)
    monkeypatch.setattr(blocks_module, "_register", lambda buffer: registered.append(buffer) or register(buffer))
    with caplog.at_level("WARNING", logger="jasna.ltx.blocks"):
        store = BlockStore([dict(b) for b in blocks], torch.device("cuda"), resident=0)
    assert len(registered) == 2
    assert [buffer.is_pinned() for buffer in store._host] == [True, True, False, False]
    assert any("about half the RAM); 2 of 4 weight buffers" in r.getMessage() for r in caplog.records)
    store.close()


def test_windows_page_lock_budget_stays_below_half_the_ram(monkeypatch):
    import sys
    from types import SimpleNamespace

    import jasna.ltx.blocks as blocks_module

    gib = 1 << 30
    monkeypatch.setattr(blocks_module.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * gib))
    monkeypatch.setattr(sys, "platform", "win32")
    assert blocks_module._page_lock_budget() == 6 * gib
    monkeypatch.setattr(blocks_module.psutil, "virtual_memory", lambda: SimpleNamespace(total=2 * gib))
    assert blocks_module._page_lock_budget() == 0
    monkeypatch.setattr(sys, "platform", "linux")
    assert blocks_module._page_lock_budget() == sys.maxsize


@needs_cuda
def test_failed_load_reports_its_own_error_when_cleanup_also_fails(monkeypatch):
    import jasna.ltx.blocks as blocks_module

    def refuse(_buffers):
        raise RuntimeError("cuMemHostRegister failed")

    def cleanup_fails(_store):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(blocks_module, "_pin_while_allowed", refuse)
    monkeypatch.setattr(BlockStore, "close", cleanup_fails)
    with pytest.raises(RuntimeError, match="cuMemHostRegister failed") as raised:
        BlockStore(_blocks(2), torch.device("cuda"), resident=0)
    assert "out of memory" in str(raised.value.__cause__)

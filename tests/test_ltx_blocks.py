import sys
from types import ModuleType

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


@pytest.mark.parametrize("failure_stage", ["read", "finalize"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_opaque_load_preserves_primary_error_on_cleanup(
    failure_stage, cleanup_fails, error_type, monkeypatch,
):
    monkeypatch.setattr(sys, "platform", "linux")
    primary = error_type("original load failure")
    cleanup = OSError("cleanup failure") if cleanup_fails else None
    operations = []
    destination = object()

    class NativeStore:
        def __init__(self, specs, device, resident):
            assert specs == [{"weight": ("float32", (2,))}]
            assert device == "cpu"
            assert resident == 1

        def destination(self, index, name):
            assert (index, name) == (0, "weight")
            return destination

        def finalize(self):
            operations.append("finalize")
            if failure_stage == "finalize":
                raise primary

        def close(self):
            operations.append("close")
            if cleanup is not None:
                raise cleanup

    class Reader:
        def tensor_specs(self):
            return {"blocks.0.weight": (torch.float32, (2,))}

        def read_tensors_into(self, groups, destinations):
            assert groups == (("blocks.0.weight",),)
            assert list(destinations) == [{"blocks.0.weight": destination}]
            operations.append("read")
            if failure_stage == "read":
                raise primary

    native_module = ModuleType("jasna.protection.opaque_ltx")
    native_module.OpaqueStore = NativeStore
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    with pytest.raises(error_type) as caught:
        OpaqueBlockStore.from_reader(
            Reader(), [{"weight": "blocks.0.weight"}], torch.device("cpu"), resident=1,
        )
    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    expected = ["read", "close"] if failure_stage == "read" else ["read", "finalize", "close"]
    assert operations == expected

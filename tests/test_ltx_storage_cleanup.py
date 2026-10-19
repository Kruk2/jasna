import torch
import pytest
from safetensors.torch import save_file

from jasna.ltx.blocks import BlockStore, OpaqueBlockStore
from jasna.ltx.model_files import PlaceholderTensors, open_tensors
from jasna.ltx import sampler, trial_shapes, transformer as T
from jasna.ltx import blocks as blocks_module
from jasna.models.ltx_vae import loader as vae_loader


def _small_transformer_specs(block_spec):
    """Top tensors of a working 8-wide transformer (adaln tables, patchify, prompt) and
    ``block_spec`` for each of the ``NUM_BLOCKS`` blocks."""
    top = {"patchify_proj.weight": (8, 8), "prompt_context": (1, 4, 8)}
    for prefix, rows in (("adaln_single", 72), ("prompt_adaln_single", 16)):
        for name, shape in (
            ("emb.timestep_embedder.linear_1", (8, 256)),
            ("emb.timestep_embedder.linear_2", (8, 8)),
            ("linear", (rows, 8)),
        ):
            top[f"{prefix}.{name}.weight"] = shape
            top[f"{prefix}.{name}.bias"] = (shape[0],)
    blocks = {f"blocks.{index}.{name}": shape for index in range(T.NUM_BLOCKS) for name, shape in block_spec.items()}
    return top | blocks


def _small_transformer_file(path):
    values = {name: torch.full(shape, 0.01, dtype=torch.bfloat16)
              for name, shape in _small_transformer_specs({"weight": (2, 2)}).items()}
    values["patchify_proj.weight"].fill_(1)
    values["prompt_context"].fill_(1)
    for name, tensor in values.items():
        if name.endswith(".bias") or name.startswith("blocks."):
            tensor.fill_(1)
    save_file(values, str(path), metadata={"sigmas": "[1.0, 0.0]", "stg_scale": "0.0"})
    return path


def _twenty_gib_free(device):
    return 20 << 30


def _without_compiling(monkeypatch):
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: T.block_forward)
    monkeypatch.setattr(T, "require_kernels", lambda device: None)
    monkeypatch.setattr(T, "enable_int8_attention", lambda: None)
    monkeypatch.setattr(sampler, "compile_static", lambda function, **kwargs: function)


class _NativeStore:
    """Stands in for the private native opaque store; records what the sampler asks of it."""

    def __init__(self, specs, device, resident):
        self.specs = specs
        self.events = []

    def activate(self, block):
        self.events.append(("activate", block))

    def begin_step(self, rows, windows, stg):
        self.events.append(("step", rows, windows, stg))

    def block_forward(self, x, block_rows, prompt_rows, context, rope_cos, rope_sin, stg_branch):
        self.events.append(("forward", stg_branch))
        return x

    def close(self):
        self.events.append(("close",))


def _install_native_store(monkeypatch):
    # This CPU sampler fixture substitutes the completed inference owner.
    # Actual CUDA admission and private-load cleanup have separate native tests.
    def from_reader(cls, reader, names, device, *, resident):
        specs = reader.tensor_specs()
        native = _NativeStore([{name: specs[source] for name, source in block.items()} for block in names],
                              device, resident)
        return cls(native, device, len(names), resident)
    monkeypatch.setattr(OpaqueBlockStore, "from_reader", classmethod(from_reader))


def test_close_clears_resident_storage_even_when_a_consumer_keeps_a_view():
    source = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    original = source["weight"]
    store = BlockStore([source], torch.device("cpu"), resident=1)
    retained = store.acquire(0)["weight"]
    assert source == {}
    assert torch.equal(retained, original)

    store.close()
    assert torch.count_nonzero(retained).item() == 0
    assert torch.equal(original, torch.arange(12, dtype=torch.float32).reshape(3, 4))
    store.close()


def test_failed_block_fill_clears_every_owned_destination(tmp_path):
    path = tmp_path / "blocks.safetensors"
    save_file({f"blocks.{i}.weight": torch.ones(4) for i in range(2)}, str(path))
    captured = []
    with open_tensors(path) as reader:
        fill = reader.get_tensor_into

        def failing_fill(name, destination):
            captured.append(destination)
            fill(name, destination)
            if name == "blocks.1.weight":
                raise ValueError("synthetic reader failure")

        reader.get_tensor_into = failing_fill
        with pytest.raises(ValueError, match="synthetic reader failure"):
            BlockStore.from_reader(reader, [{"weight": f"blocks.{i}.weight"} for i in range(2)],
                                   torch.device("cpu"), resident=2)
    assert len(captured) == 2
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in captured)


def test_transformer_close_clears_top_context_conditioning_and_blocks(tmp_path, monkeypatch):
    path = _small_transformer_file(tmp_path / "small-transformer.safetensors")
    original = path.read_bytes()
    _without_compiling(monkeypatch)
    transformer = sampler.LtxTransformer(sampler.TransformerSource(path), torch.device("cpu"), _twenty_gib_free)
    condition = transformer.conditions[0]
    retained = [*transformer.top.values(), transformer.context,
                condition.block_rows, condition.output_rows, condition.prompt_rows,
                *transformer.blocks.acquire(0).values()]
    assert all(torch.count_nonzero(tensor).item() > 0 for tensor in retained)
    transformer.close()
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in retained)
    assert path.read_bytes() == original
    transformer.close()


def test_failed_transformer_load_reports_its_own_error_when_cleanup_also_fails(tmp_path, monkeypatch):
    path = _small_transformer_file(tmp_path / "small-transformer.safetensors")
    _without_compiling(monkeypatch)

    def load_fails(*_args, **_kwargs):
        raise RuntimeError("LTX private load rejected: native block commit")

    def cleanup_fails(_transformer):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(sampler.BlockStore, "from_reader", load_fails)
    monkeypatch.setattr(sampler.LtxTransformer, "close", cleanup_fails)
    with pytest.raises(RuntimeError, match="native block commit") as raised:
        sampler.LtxTransformer(sampler.TransformerSource(path), torch.device("cpu"), _twenty_gib_free)
    assert "out of memory" in str(raised.value.__cause__)


def test_frozen_trial_keeps_its_placeholder_blocks_in_the_plain_tensor_store(monkeypatch):
    spec = _small_transformer_specs({"weight": (2, 2)})
    top = {name: ("BF16", shape) for name, shape in spec.items() if not name.startswith("blocks.")}
    monkeypatch.setitem(trial_shapes.TRANSFORMER, "int8", {"top": top, "block": {"weight": ("BF16", (2, 2))}})
    monkeypatch.setattr(sampler, "is_frozen", lambda: True)
    _install_native_store(monkeypatch)
    _without_compiling(monkeypatch)
    transformer = sampler.LtxTransformer(sampler.TransformerSource(PlaceholderTensors("int8", "distilled")), torch.device("cpu"), _twenty_gib_free)
    try:
        assert isinstance(transformer.blocks, BlockStore)
        assert transformer.blocks.acquire(0)["weight"].shape == (2, 2)
    finally:
        transformer.close()


def test_block_loading_fills_final_storage_without_returning_weight_tensors(tmp_path):
    path = tmp_path / "blocks.safetensors"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"blocks.0.weight": expected}, str(path))
    with open_tensors(path) as reader:
        def reject_copy(_name):
            raise AssertionError("Block loading must fill its final destination")

        reader.get_tensor = reject_copy
        store = BlockStore.from_reader(reader, [{"weight": "blocks.0.weight"}], torch.device("cpu"), resident=1)
    retained = store.acquire(0)["weight"]
    assert torch.equal(retained, expected)
    store.close()
    assert torch.count_nonzero(retained).item() == 0


def test_block_loading_streams_all_destinations_in_one_reader_operation(tmp_path):
    path = tmp_path / "blocks.safetensors"
    expected = {f"blocks.{index}.weight": torch.arange(6, dtype=torch.float32) + index
                for index in range(3)}
    save_file(expected, str(path))
    operations = []
    with open_tensors(path) as reader:
        load = reader.read_tensors_into

        def observe(group_names, groups):
            operations.append(group_names)
            load(group_names, groups)

        reader.read_tensors_into = observe
        store = BlockStore.from_reader(reader, [{"weight": name} for name in expected],
                                       torch.device("cpu"), resident=3)
    try:
        assert operations == [(tuple(expected),)]
        for index, value in enumerate(expected.values()):
            assert torch.equal(store.acquire(index)["weight"], value)
    finally:
        store.close()


def test_opaque_block_store_keeps_complete_blocks_out_of_python_tensors(monkeypatch):
    class Reader:
        def tensor_specs(self):
            return {
                "blocks.0.proj.qdata": (torch.int8, (4, 4)),
                "blocks.0.proj.scales": (torch.float32, (4,)),
                "blocks.0.scale_shift_table": (torch.bfloat16, (2, 4)),
            }

        def read_tensors_into(self, group_names, groups):
            pytest.fail("private block storage requested Python tensor destinations")

    _install_native_store(monkeypatch)
    store = OpaqueBlockStore.from_reader(
        Reader(),
        [{
            "proj.qdata": "blocks.0.proj.qdata",
            "proj.scales": "blocks.0.proj.scales",
            "scale_shift_table": "blocks.0.scale_shift_table",
        }],
        torch.device("cpu"), resident=1,
    )
    native = store._native
    assert set(native.specs[0]) == {"proj.qdata", "proj.scales", "scale_shift_table"}
    store.begin_step(8192, 3, True)
    store.activate(0)
    store.close()
    assert native.events == [("step", 8192, 3, True), ("activate", 0), ("close",)]


def test_opaque_denoise_uses_only_native_complete_block_forward(tmp_path, monkeypatch):
    path = _small_transformer_file(tmp_path / "small-transformer.safetensors")
    monkeypatch.setattr(sampler, "is_frozen", lambda: True)
    monkeypatch.setattr(sampler.T, "NUM_BLOCKS", 2)
    monkeypatch.setattr(sampler, "STG_BLOCK", 1)
    _install_native_store(monkeypatch)
    _without_compiling(monkeypatch)
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: pytest.fail("Python block forward compiled"))
    monkeypatch.setattr(sampler, "compile_static", lambda function, **kwargs: pytest.fail("Python function compiled"))
    monkeypatch.setattr(sampler.T, "embed_tokens", lambda top, x, tokens_per_frame: x)
    monkeypatch.setattr(
        sampler.T, "velocity", lambda top, x, cond: torch.zeros_like(x[:, : x.shape[1] // 2]),
    )
    transformer = sampler.LtxTransformer(sampler.TransformerSource(path), torch.device("cpu"), _twenty_gib_free)
    transformer.stg_scale = 1.0
    native = transformer.blocks._native
    rope = T.Rope(cos=torch.ones(1, 1, 4, 4), sin=torch.zeros(1, 1, 4, 4))
    try:
        result = transformer._denoise([torch.zeros(1, 8, 8)], transformer.conditions[0], rope, 1, torch.tensor(1.0))
        assert len(result) == 1
        assert native.events == [
            ("step", 8, 1, True),
            ("activate", 0), ("forward", False),
            ("activate", 1), ("forward", True), ("forward", False),
        ]
    finally:
        transformer.close()


def test_late_stream_failure_clears_every_committed_destination(tmp_path):
    path = tmp_path / "blocks.safetensors"
    names = [f"blocks.{index}.weight" for index in range(3)]
    save_file({name: torch.ones(4) for name in names}, str(path))
    retained = []
    with open_tensors(path) as reader:
        load = reader.read_tensors_into

        def fail_after_completion(group_names, groups):
            def observe():
                for group in groups:
                    retained.extend(group.values())
                    yield group

            load(group_names, observe())
            raise ValueError("synthetic final verification failure")

        reader.read_tensors_into = fail_after_completion
        with pytest.raises(ValueError, match="final verification"):
            BlockStore.from_reader(reader, [{"weight": name} for name in names],
                                   torch.device("cpu"), resident=3)
    assert len(retained) == 3
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in retained)


def test_top_tensor_loading_uses_one_owned_destination_group(tmp_path):
    path = tmp_path / "top.safetensors"
    expected = {"context": torch.arange(6).reshape(2, 3), "bias": torch.ones(3)}
    save_file(expected, str(path))
    operations = []
    with open_tensors(path) as reader:
        load = reader.read_tensors_into

        def observe(group_names, groups):
            operations.append(group_names)
            load(group_names, groups)

        reader.read_tensors_into = observe
        result = sampler._read_host_tensors(reader, tuple(expected), reader.tensor_specs())
    assert operations == [(tuple(expected),)]
    assert all(torch.equal(result[name], value) for name, value in expected.items())


def test_vae_stream_failure_clears_the_entire_owned_state(tmp_path, monkeypatch):
    from contextlib import contextmanager

    path = tmp_path / "vae.safetensors"
    save_file({"encoder.weight": torch.ones(6), "encoder.bias": torch.ones(3), "decoder.bias": torch.ones(2)}, str(path))
    retained = []

    @contextmanager
    def fail_after_verification(source):
        with open_tensors(source) as reader:
            load = reader.read_tensors_into

            def fail(group_names, groups):
                def observe():
                    for group in groups:
                        retained.extend(group.values())
                        yield group

                load(group_names, observe())
                raise ValueError("synthetic final VAE verification failure")

            reader.read_tensors_into = fail
            yield reader

    monkeypatch.setattr(vae_loader, "open_tensors", fail_after_verification)
    with pytest.raises(ValueError, match="final VAE verification"):
        with vae_loader.open_tensors(path) as reader:
            vae_loader._read_tensors(reader, ("encoder.",))
    assert len(retained) == 2
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in retained)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("resident", (0, 1, 3))
def test_cuda_block_loading_and_close_clear_retained_storage(tmp_path, resident):
    path = tmp_path / "blocks.safetensors"
    expected = {f"blocks.{index}.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) + index + 1
                for index in range(3)}
    save_file(expected, str(path))
    original = path.read_bytes()
    with open_tensors(path) as reader:
        load = reader.read_tensors_into
        staging = []

        def observe(group_names, groups):
            def record():
                for group in groups:
                    staging.append(next(iter(group.values())).data_ptr())
                    yield group

            load(group_names, record())

        reader.read_tensors_into = observe
        store = BlockStore.from_reader(reader, [{"weight": name} for name in expected],
                                       torch.device("cuda"), resident=resident)
    if resident == 3:
        assert staging == [staging[0], staging[1], staging[0]] and staging[0] != staging[1]
    retained = store._resident_buffers + store._host + store._slots
    try:
        for index, name in enumerate(expected):
            assert torch.equal(store.acquire(index)["weight"].cpu(), expected[name])
            store.release(index)
    finally:
        store.close()
    assert all(torch.count_nonzero(buffer).item() == 0 for buffer in retained)
    assert path.read_bytes() == original
    store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_stream_final_failure_clears_gpu_pinned_and_staging_storage(tmp_path, monkeypatch):
    path = tmp_path / "blocks.safetensors"
    names = [f"blocks.{index}.weight" for index in range(3)]
    save_file({name: torch.ones(4) for name in names}, str(path))
    retained = []
    allocate = BlockStore._allocate

    def observe_allocation(store, layouts, device, resident):
        allocate(store, layouts, device, resident)
        retained.extend(store._resident_buffers + store._host + store._slots)

    monkeypatch.setattr(BlockStore, "_allocate", observe_allocation)
    with open_tensors(path) as reader:
        load = reader.read_tensors_into

        def fail_after_copy(group_names, groups):
            def record():
                for group in groups:
                    retained.extend(group.values())
                    yield group

            load(group_names, record())
            raise ValueError("synthetic final GPU verification failure")

        reader.read_tensors_into = fail_after_copy
        with pytest.raises(ValueError, match="final GPU verification"):
            BlockStore.from_reader(reader, [{"weight": name} for name in names],
                                   torch.device("cuda"), resident=1)
    assert retained
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in retained)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_staging_failure_closes_the_store(tmp_path, monkeypatch):
    path = tmp_path / "blocks.safetensors"
    save_file({"blocks.0.weight": torch.ones(4)}, str(path))
    stores = []
    allocate = BlockStore._allocate

    def observe_allocation(store, layouts, device, resident):
        allocate(store, layouts, device, resident)
        stores.append(store)

    def no_pinned_memory(buffers):
        if buffers:
            raise RuntimeError("synthetic pinned allocation failure")
        return []

    monkeypatch.setattr(BlockStore, "_allocate", observe_allocation)
    monkeypatch.setattr(blocks_module, "_pin_while_allowed", no_pinned_memory)
    with open_tensors(path) as reader, pytest.raises(RuntimeError, match="pinned allocation"):
        BlockStore.from_reader(reader, [{"weight": "blocks.0.weight"}], torch.device("cuda"), resident=1)
    assert stores and not stores[0]._resident_buffers

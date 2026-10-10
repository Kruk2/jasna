"""Transformer block weights, resident on the GPU or streamed from pinned host memory.

Each block is packed into one flat byte buffer so it moves in a single copy. As many
leading blocks as the VRAM budget allows stay resident; the rest live in pinned host
memory and are copied on a side stream into one of two GPU slots, one block ahead of
the block being computed. Frozen releases keep every transformer-block tensor and the
complete block forward in the native opaque store; Python sees only block inputs/outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_ALIGN = 256


@dataclass(frozen=True)
class _Entry:
    offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * self.dtype.itemsize


def _layout(specs: dict[str, tuple[torch.dtype, tuple[int, ...]]]) -> tuple[dict[str, _Entry], int]:
    layout: dict[str, _Entry] = {}
    offset = 0
    for name, (dtype, shape) in specs.items():
        entry = _Entry(offset, dtype, shape)
        layout[name] = entry
        offset += -(-entry.nbytes // _ALIGN) * _ALIGN
    return layout, offset


def _pack(tensors: dict[str, torch.Tensor], layout: dict[str, _Entry], buffer: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        entry = layout[name]
        buffer[entry.offset : entry.offset + entry.nbytes].copy_(tensor.contiguous().view(-1).view(torch.uint8))


def _views(buffer: torch.Tensor, layout: dict[str, _Entry]) -> dict[str, torch.Tensor]:
    return {
        name: buffer[entry.offset : entry.offset + entry.nbytes].view(entry.dtype).view(entry.shape)
        for name, entry in layout.items()
    }


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


def _file_order_groups(
    order: list[str], targets: dict[str, torch.Tensor], stage_bytes: int
) -> list[tuple[tuple[str, ...], dict[str, int]]]:
    """Consecutive runs of ``order`` whose GPU tensors fit one stage, each with those
    tensors' stage offsets. CPU tensors are read straight into their final storage."""
    groups, group, offsets, used = [], [], {}, 0
    for name in order:
        size = _aligned(targets[name].nbytes) if targets[name].device.type != "cpu" else 0
        if size and used + size > stage_bytes:
            groups.append((tuple(group), offsets))
            group, offsets, used = [], {}, 0
        if size:
            offsets[name] = used
            used += size
        group.append(name)
    if group:
        groups.append((tuple(group), offsets))
    return groups


def _staged(stage: torch.Tensor, offset: int, target: torch.Tensor) -> torch.Tensor:
    return stage[offset : offset + target.nbytes].view(target.dtype).view(target.shape)


class BlockStore:
    """Hands out each block's weights on ``device`` in a fixed cyclic block order. Empties each
    dict of ``blocks`` once packed, so loading never holds the weights twice in RAM."""

    def __init__(self, blocks: list[dict[str, torch.Tensor]], device: torch.device, *, resident: int) -> None:
        layouts = [_layout({name: (tensor.dtype, tuple(tensor.shape)) for name, tensor in block.items()})
                   for block in blocks]
        self._allocate(layouts, device, resident)
        try:
            for index, tensors in enumerate(blocks):
                _pack(tensors, self._layouts[index], self._buffer(index))
                tensors.clear()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_reader(cls, reader, names: list[dict[str, str]], device: torch.device, *, resident: int) -> BlockStore:
        """Read in file order, so every encrypted chunk is decrypted once. Tensors of GPU-resident
        blocks pass through two pinned stages; one stage's copies run while the other is read."""
        specs = reader.tensor_specs()
        layouts = [_layout({name: specs[source] for name, source in block.items()}) for block in names]
        store = cls.__new__(cls)
        store._allocate(layouts, device, resident)
        try:
            store._read(reader, names)
        except BaseException:
            store.close()
            raise
        return store

    def _read(self, reader, names: list[dict[str, str]]) -> None:
        targets = {sources[name]: view for index, sources in enumerate(names)
                   for name, view in _views(self._buffer(index), self._layouts[index]).items()}
        staged_sizes = [_aligned(target.nbytes) for target in targets.values() if target.device.type != "cpu"]
        stage_bytes = max(_aligned(max(self._buffer(index).numel() for index in range(self.resident)) // 2),
                          *staged_sizes) if staged_sizes else 0
        groups = _file_order_groups([name for name in reader.offset_keys() if name in targets], targets, stage_bytes)
        stages, copied = [], []

        def destinations():
            try:
                if stage_bytes:
                    stages.extend(torch.empty(stage_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2))
                    copied.extend(torch.cuda.Event() for _ in stages)
                for number, (group, offsets) in enumerate(groups):
                    slot = number % 2
                    if offsets:
                        copied[slot].synchronize()
                    views = {name: _staged(stages[slot], offsets[name], targets[name]) if name in offsets else targets[name]
                             for name in group}
                    yield views
                    for name in offsets:
                        targets[name].copy_(views[name], non_blocking=True)
                    if offsets:
                        copied[slot].record()
            finally:
                if stages:
                    torch.cuda.synchronize(self.device)
                    for stage in stages:
                        stage.zero_()

        stream = destinations()
        try:
            reader.read_tensors_into(tuple(group for group, _ in groups), stream)
        finally:
            stream.close()

    def _buffer(self, index: int) -> torch.Tensor:
        return self._resident_buffers[index] if index < self.resident else self._host[index - self.resident]

    def _allocate(self, layouts: list[tuple[dict[str, _Entry], int]], device: torch.device, resident: int) -> None:
        if resident < 0:
            raise ValueError("Resident block count must not be negative")
        self.device = device
        self.count = len(layouts)
        self.resident = min(resident, self.count)
        self._layouts: list[dict[str, _Entry]] = []
        self._resident: list[dict[str, torch.Tensor]] = []
        self._resident_buffers: list[torch.Tensor] = []
        self._host: list[torch.Tensor] = []
        self._slots: list[torch.Tensor] = []
        slot_bytes = 0
        try:
            for index, (layout, nbytes) in enumerate(layouts):
                self._layouts.append(layout)
                if index < self.resident:
                    buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
                    self._resident_buffers.append(buffer)
                    self._resident.append(_views(buffer, layout))
                else:
                    buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                    self._host.append(buffer)
                    slot_bytes = max(slot_bytes, nbytes)
            streamed = self.count - self.resident
            for _ in range(min(2, streamed)):
                self._slots.append(torch.empty(slot_bytes, dtype=torch.uint8, device=device))
            self._slot_block = [-1] * len(self._slots)
            self._copy_done = [torch.cuda.Event() for _ in self._slots]
            self._compute_done = [torch.cuda.Event() for _ in self._slots]
            self._copy_stream = torch.cuda.Stream(device) if self._slots else None
        except BaseException:
            self.close()
            raise
        logger.info("LTX blocks: %d resident, %d streamed", self.resident, streamed)
        if streamed:
            logger.warning(
                "LTX: %d of %d transformer blocks stream from RAM (not enough free VRAM); "
                "closing other programs that use the GPU makes LTX faster",
                streamed,
                self.count,
            )

    def _slot_of(self, index: int) -> int | None:
        return self._slot_block.index(index) if index in self._slot_block else None

    def _prefetch(self, index: int, keep: int | None) -> None:
        if index < self.resident or self._slot_of(index) is not None:
            return
        busy = self._slot_of(keep) if keep is not None else None
        slot = next(s for s in range(len(self._slots)) if s != busy)
        source = self._host[index - self.resident]
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._compute_done[slot])
            self._slots[slot][: source.numel()].copy_(source, non_blocking=True)
            self._copy_done[slot].record(self._copy_stream)
        self._slot_block[slot] = index

    def acquire(self, index: int) -> dict[str, torch.Tensor]:
        """Weights of block ``index``; also starts copying the next streamed block."""
        if index < self.resident:
            weights = self._resident[index]
        else:
            self._prefetch(index, None)
            slot = self._slot_of(index)
            torch.cuda.current_stream(self.device).wait_event(self._copy_done[slot])
            weights = _views(self._slots[slot], self._layouts[index])
        following = (index + 1) % self.count
        if following < self.resident and self.resident < self.count:
            following = self.resident
        if following != index:
            self._prefetch(following, index)
        return weights

    def release(self, index: int) -> None:
        """Mark block ``index``'s slot reusable once the work queued so far is done.
        Call before acquiring the next block, so its prefetch waits for this work."""
        slot = self._slot_of(index) if index >= self.resident else None
        if slot is not None:
            self._compute_done[slot].record(torch.cuda.current_stream(self.device))

    def native_vram_bytes(self) -> int:
        """PyTorch's allocator holds these blocks."""
        return 0

    def close(self) -> None:
        buffers = self._resident_buffers + self._host + self._slots
        if not buffers:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for buffer in buffers:
            buffer.zero_()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._resident_buffers.clear()
        self._resident.clear()
        self._host.clear()
        self._slots.clear()


class OpaqueBlockStore:
    """Keeps complete transformer blocks in the native protection extension."""

    def __init__(self, native, device: torch.device, count: int, resident: int) -> None:
        self._native = native
        self.device = device
        self.count = count
        self.resident = min(resident, count)

    @classmethod
    def from_reader(
        cls,
        reader,
        names: list[dict[str, str]],
        device: torch.device,
        *,
        resident: int,
    ) -> OpaqueBlockStore:
        from jasna.ltx.native.aot import BlockDims
        from jasna.protection.opaque_ltx import OpaqueInferenceStore

        specs = reader.tensor_specs()
        dims = BlockDims.from_specs({name: specs[source] for name, source in names[0].items()})
        if device.type != "cuda":
            raise ValueError("native opaque blocks require a CUDA device")
        index = device.index if device.index is not None else torch.cuda.current_device()
        owner = reader.private_load(index, resident)
        native = OpaqueInferenceStore(owner, head_dim=dims.dim // dims.heads,
                                      count=len(names), resident=resident)
        return cls(native, device, len(names), resident)

    def activate(self, index: int) -> None:
        self._native.activate(index)

    def begin_step(self, rows: int, windows: int, stg: bool) -> None:
        self._native.begin_step(rows, windows, stg)

    def block_forward(self, x, cond, context, rope, *, stg_branch: bool = False):
        return self._native.block_forward(
            x, cond.block_rows, cond.prompt_rows, context, rope.cos, rope.sin, stg_branch,
        )

    def native_vram_bytes(self) -> int:
        """VRAM holding the blocks outside PyTorch's allocator."""
        return self._native.device_bytes()

    def close(self) -> None:
        self._native.close()

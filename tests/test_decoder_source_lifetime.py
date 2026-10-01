"""Pure-stdlib lifetime checks for the actual shared product decoder.

This file parses and executes the staged method AST with fakes.  It intentionally
does not import the staged module or any media/GPU dependency.
"""

from __future__ import annotations

import ast
import copy
import gc
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


STAGED_SOURCE = Path(__file__).resolve().parents[1] / "jasna/media/video_decoder.py"


def _reader_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(STAGED_SOURCE.read_text(encoding="utf-8"), filename=str(STAGED_SOURCE))
    reader = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NvidiaVideoReader"
    )
    return copy.deepcopy(
        next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == name)
    )


def _compiled_method(name: str, namespace: dict[str, object]):
    method = _reader_method(name)
    method.decorator_list = []
    method.returns = None
    method.type_comment = None
    arguments = (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs)
    for argument in arguments:
        argument.annotation = None
    if method.args.vararg is not None:
        method.args.vararg.annotation = None
    if method.args.kwarg is not None:
        method.args.kwarg.annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(STAGED_SOURCE), "exec"), namespace)
    return namespace[name]


class _Vendor:
    AMD = object()
    NVIDIA = object()


class _ColorRange:
    JPEG = object()
    MPEG = object()


class _Av:
    class FFmpegError(Exception):
        pass


class _Component:
    bits = 8


class _Format:
    name = "yuv420p"
    components = (_Component(),)


class _Source:
    def __init__(self, pts: int):
        self.pts = pts
        self.format = _Format()


class _Plane:
    line_size = 2


class _Normalized:
    def __init__(self):
        self.planes = (_Plane(), _Plane())


class _BufferTensor:
    def __init__(self, plane: _Plane):
        self.plane = plane

    def reshape(self, *shape):
        return self

    def __getitem__(self, key):
        return self


class _HostSlice:
    def __init__(self, root):
        self.root = root

    def copy_(self, source, non_blocking: bool = False):
        if non_blocking or not isinstance(source, _BufferTensor):
            raise AssertionError("source-to-pinned copy must be synchronous and use a plane view")
        self.root.state.events.append("cpu-copy")
        # Deliberately retain no source view: this models completion before copy_ returns.
        return self


class _Tensor:
    def __init__(self, state, kind: str):
        self.state = state
        self.kind = kind

    def __getitem__(self, key):
        if self.kind == "pinned":
            return _HostSlice(self)
        return _Tensor(self.state, "device-slice")

    def reshape(self, *shape):
        return self

    def view(self, *shape):
        return self

    def copy_(self, source, non_blocking: bool = False):
        if self.kind not in {"device", "device-slice"}:
            raise AssertionError(f"unexpected copy destination: {self.kind}")
        if not non_blocking or not isinstance(source, _HostSlice):
            raise AssertionError("async H2D must retain pinned storage, not a source plane")
        self.state.events.append("h2d")
        self.state.stream.pending.append(source.root)
        return self


class _Torch:
    uint8 = object()
    uint16 = object()

    def __init__(self, state):
        self.state = state

    def empty(self, shape, *, dtype, pin_memory: bool = False, device=None):
        tensor = _Tensor(self.state, "pinned" if pin_memory else "device")
        if pin_memory:
            self.state.pinned_refs.append(weakref.ref(tensor))
        return tensor

    @staticmethod
    def frombuffer(plane, *, dtype):
        return _BufferTensor(plane)


class _Stream:
    def __init__(self, state):
        self.state = state
        self.pending: list[object] = []

    def synchronize(self):
        self.state.events.append("sync")
        self.pending.clear()


class _State:
    def __init__(self):
        self.events: list[str] = []
        self.pinned_refs: list[weakref.ReferenceType] = []
        self.stream = _Stream(self)


class _Converter:
    def __init__(self, state):
        self.state = state

    def convert_into(self, y, uv, output):
        self.state.events.append("convert")


class _Reformatter:
    def __init__(self, refs):
        self.refs = refs

    def reformat(self, frame, **kwargs):
        normalized = _Normalized()
        for plane in normalized.planes:
            plane.line_size = 4 if kwargs['format'] == 'p010le' else 2
        self.refs.append(weakref.ref(normalized))
        return normalized


class _ReformatterFactory:
    def __init__(self, refs):
        self.refs = refs

    def __call__(self):
        return _Reformatter(self.refs)


@contextmanager
def _stream_context(stream):
    yield


class _Metadata:
    color_space = "bt709"
    is_10bit = False


class _SoftwareReader:
    def __init__(self, state, old_source_ref, normalized_refs, *, batch_size: int):
        self.batch_size = batch_size
        self.height = 2
        self.width = 2
        self.metadata = _Metadata()
        self._full_range = False
        self.vendor = _Vendor.AMD
        self.device = "fake-device"
        self.file = "fake.mkv"
        self.state = state
        self.old_source_ref = old_source_ref
        self.normalized_refs = normalized_refs
        self.next_groups: list[list[_Source]] = []
        self.prefetch_old_source_alive = None
        self.prefetch_normalized_alive = None
        self.prefetch_pinned_alive = None
        self.prefetch_pending_count = None

    def _read_group(self, decoded):
        gc.collect()
        self.prefetch_old_source_alive = self.old_source_ref() is not None
        self.prefetch_normalized_alive = any(ref() is not None for ref in self.normalized_refs)
        self.prefetch_pinned_alive = bool(self.state.pinned_refs and self.state.pinned_refs[0]())
        self.prefetch_pending_count = len(self.state.stream.pending)
        self.state.events.append("prefetch")
        return self.next_groups.pop(0) if self.next_groups else []


def _software_method(state, normalized_refs):
    return _compiled_method(
        "_frames_software",
        {
            "torch": _Torch(state),
            "AcceleratorVendor": _Vendor,
            "AvColorRange": _ColorRange,
            "VideoReformatter": _ReformatterFactory(normalized_refs),
            "YuvToRgbConverter": lambda *args, **kwargs: _Converter(state),
            "current_stream": lambda device: state.stream,
            "new_stream": lambda device: state.stream,
            "stream_context": _stream_context,
            "av": _Av,
        },
    )


class _WeakList(list):
    pass


class _OuterReader:
    def __init__(self, *, empty: bool):
        self.empty = empty
        self.vendor = _Vendor.AMD
        self._amf_interop_enabled = False
        self._windows_resident_enabled = False
        self._software_only = True
        self.backend_calls = 0
        self.initial_group_ref = None

    def _decoded_frames(self, seek_ts):
        return object()

    def _selected_frames(self, decoded_frames):
        return decoded_frames

    def _read_group(self, decoded):
        if self.empty:
            return []
        group = _WeakList([_Source(1)])
        self.initial_group_ref = weakref.ref(group)
        return group

    def _frames_software(self, decoded, group):
        self.backend_calls += 1

        def backend():
            yield "batch", [1]

        return backend()


class _Log:
    @staticmethod
    def warning(*args, **kwargs):
        raise AssertionError("AMD fake route must not issue the NVIDIA fallback warning")


class DecoderOwnerReleaseTests(unittest.TestCase):
    def test_main10_source_release_on_amd_and_single_staging_fallback(self):
        for vendor in (_Vendor.AMD, _Vendor.NVIDIA):
            with self.subTest(vendor=vendor):
                state = _State()
                refs = []
                frame = _Source(31)
                frame.format = SimpleNamespace(name='yuv420p10le', components=[SimpleNamespace(bits=10)])
                old = weakref.ref(frame)
                group = [frame]
                del frame
                reader = _SoftwareReader(state, old, refs, batch_size=1)
                reader.vendor = vendor
                results = list(_software_method(state, refs)(reader, object(), group))
                self.assertEqual(results[0][1],[31])
                self.assertFalse(reader.prefetch_old_source_alive)
                self.assertFalse(reader.prefetch_normalized_alive)
                self.assertTrue(reader.prefetch_pinned_alive)
                self.assertEqual(state.events.count('cpu-copy'),2)
                self.assertEqual(state.events.count('h2d'),1)
                self.assertEqual(state.events.count('sync'),1)

    def test_ast_places_release_between_h2d_and_prefetch(self):
        source = STAGED_SOURCE.read_text(encoding="utf-8")
        method = _reader_method("_frames_software")
        while_node = next(node for node in ast.walk(method) if isinstance(node, ast.While))
        body = while_node.body
        prefetch_index = next(
            index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "next_group"
        )
        release = body[prefetch_index - 2]
        clear = body[prefetch_index - 1]
        self.assertIsInstance(release, ast.Delete)
        self.assertEqual(
            [target.id for target in release.targets if isinstance(target, ast.Name)],
            ["frame", "normalized", "y_plane", "uv_plane", "y", "uv"],
        )
        self.assertIsInstance(clear, ast.Expr)
        self.assertEqual(ast.unparse(clear.value), "group.clear()")
        rendered = ast.unparse(method)
        self.assertIn("pinned[i, :H].copy_(y)", rendered)
        self.assertIn("pinned[i, H:].copy_(uv)", rendered)
        self.assertIn("plane.copy_(pinned[i], non_blocking=True)", rendered)

    def test_b1_releases_old_sources_before_prefetch_but_keeps_pinned_and_next_group(self):
        state = _State()
        normalized_refs: list[weakref.ReferenceType] = []
        initial_group = [_Source(11)]
        old_source_ref = weakref.ref(initial_group[0])
        reader = _SoftwareReader(state, old_source_ref, normalized_refs, batch_size=1)
        prefetched_group = [_Source(12)]
        prefetched_source_ref = weakref.ref(prefetched_group[0])
        reader.next_groups.append(prefetched_group)
        del prefetched_group

        frames = _software_method(state, normalized_refs)(reader, object(), initial_group)
        result = next(frames)

        self.assertEqual(result[1], [11])
        self.assertEqual(initial_group, [])
        self.assertFalse(reader.prefetch_old_source_alive)
        self.assertFalse(reader.prefetch_normalized_alive)
        self.assertTrue(reader.prefetch_pinned_alive)
        self.assertEqual(reader.prefetch_pending_count, 1)
        self.assertLess(state.events.index("cpu-copy"), state.events.index("h2d"))
        self.assertLess(state.events.index("h2d"), state.events.index("prefetch"))
        self.assertLess(state.events.index("prefetch"), state.events.index("sync"))
        self.assertEqual(state.stream.pending, [])
        self.assertIsNotNone(prefetched_source_ref(), "B1 prefetch must remain live across yield")
        self.assertTrue(state.pinned_refs[0](), "generator retains pinned staging across yield")
        del result

        frames.close()
        gc.collect()
        self.assertIsNone(prefetched_source_ref(), "generator close must release the prefetched B1 group")
        self.assertIsNone(state.pinned_refs[0](), "generator close must release pinned staging")

    def test_partial_group_runs_once_and_releases_its_owner_set(self):
        state = _State()
        normalized_refs: list[weakref.ReferenceType] = []
        partial_group = [_Source(21)]
        old_source_ref = weakref.ref(partial_group[0])
        reader = _SoftwareReader(state, old_source_ref, normalized_refs, batch_size=3)

        results = list(_software_method(state, normalized_refs)(reader, object(), partial_group))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], [21])
        self.assertEqual(partial_group, [])
        self.assertFalse(reader.prefetch_old_source_alive)
        self.assertFalse(reader.prefetch_normalized_alive)
        self.assertEqual(state.events.count("sync"), 1)
        gc.collect()
        self.assertIsNone(old_source_ref())

    def test_frames_pyav_deletes_its_initial_group_and_short_circuits_empty_input(self):
        frames_pyav = _compiled_method(
            "_frames_pyav",
            {"AcceleratorVendor": _Vendor, "log": _Log()},
        )
        populated = _OuterReader(empty=False)
        iterator = frames_pyav(populated, None)
        self.assertEqual(next(iterator), ("batch", [1]))
        gc.collect()
        self.assertEqual(populated.backend_calls, 1)
        self.assertIsNone(
            populated.initial_group_ref(),
            "_frames_pyav must delete its outer initial-group alias before yielding backend output",
        )
        iterator.close()

        empty = _OuterReader(empty=True)
        self.assertEqual(list(frames_pyav(empty, None)), [])
        self.assertEqual(empty.backend_calls, 0)


class PacketConsumptionTests(unittest.TestCase):
    def reader(self, packet_groups):
        iterator = iter(packet_groups)
        return SimpleNamespace(
            container=SimpleNamespace(demux=lambda stream: range(len(packet_groups))),
            video_stream=SimpleNamespace(start_time=100, time_base=1),
            metadata=SimpleNamespace(start_pts=100),
            _decoder_ctx=None,
            _decode_packet=lambda packet, errors: (next(iterator), 0),
        )

    def method(self):
        return _compiled_method('_decoded_frames', {'resolve_video_start_pts':lambda a,b:a if a is not None else b})

    def test_packet_list_drops_consumed_reference_before_suspend(self):
        packet_frames = [_Source(1), _Source(2)]
        first, second = weakref.ref(packet_frames[0]), weakref.ref(packet_frames[1])
        reader = self.reader([packet_frames])
        iterator = self.method()(reader, None)
        frame = next(iterator)
        self.assertEqual(frame.pts, 1)
        self.assertEqual([f.pts for f in packet_frames], [2])
        del frame
        gc.collect()
        self.assertIsNone(first())
        self.assertIsNotNone(second())
        frame = next(iterator)
        self.assertEqual(frame.pts, 2)
        self.assertEqual(packet_frames, [])
        del frame
        gc.collect()
        self.assertIsNone(second())
        self.assertEqual(list(iterator), [])

    def test_seek_discards_early_frames_and_flushes_context(self):
        skipped = _Source(101)
        old = weakref.ref(skipped)
        packet_frames = [skipped, _Source(103), _Source(104)]
        del skipped
        reader = self.reader([packet_frames])
        events = []
        reader.container.seek = lambda *a,**k:events.append(('seek',a,k))
        reader._decoder_ctx = SimpleNamespace(flush_buffers=lambda:events.append(('flush',)))
        iterator = self.method()(reader, 3)
        frame = next(iterator)
        self.assertEqual(frame.pts, 103)
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(events[0][1], (103,))
        self.assertEqual(events[1], ('flush',))
        self.assertEqual([f.pts for f in iterator], [104])

    def test_empty_packets_null_pts_and_close(self):
        reader = self.reader([[], [_Source(None), _Source(5)], []])
        events = []
        reader.container.seek = lambda *a,**k:events.append(a)
        iterator = self.method()(reader, 3)
        self.assertIsNone(next(iterator).pts)
        self.assertEqual(next(iterator).pts, 5)
        self.assertEqual(list(iterator), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)

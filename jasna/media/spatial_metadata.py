"""MP4/MOV Spherical Video V2 atom injection.

The box layout and rewrite strategy are derived from Google's Apache-2.0
spatial-media project:
https://github.com/google/spatial-media
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_VIDEO_SAMPLE_ENTRIES = {
    b"avc1",
    b"av01",
    b"hev1",
    b"hvc1",
    b"dvh1",
    b"vp09",
}
_CONTAINER_TYPES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


@dataclass(frozen=True)
class _Box:
    start: int
    size: int
    header_size: int
    box_type: bytes

    @property
    def end(self) -> int:
        return self.start + self.size


def _box_at(data: bytes, start: int, end: int) -> _Box:
    if start + 8 > end:
        raise ValueError("Truncated MP4 box header")
    size32, box_type = struct.unpack_from(">I4s", data, start)
    header_size = 8
    if size32 == 1:
        if start + 16 > end:
            raise ValueError("Truncated extended MP4 box header")
        size = struct.unpack_from(">Q", data, start + 8)[0]
        header_size = 16
    elif size32 == 0:
        size = end - start
    else:
        size = size32
    if size < header_size or start + size > end:
        raise ValueError(
            f"Invalid MP4 box {box_type!r} size {size} at offset {start}"
        )
    return _Box(start, size, header_size, box_type)


def _boxes(data: bytes, start: int, end: int) -> list[_Box]:
    boxes = []
    position = start
    while position < end:
        box = _box_at(data, position, end)
        boxes.append(box)
        position = box.end
    return boxes


def _file_boxes(path: Path) -> list[_Box]:
    file_size = path.stat().st_size
    boxes = []
    with path.open("rb") as file:
        position = 0
        while position < file_size:
            file.seek(position)
            header = file.read(16)
            if len(header) < 8:
                raise ValueError("Truncated top-level MP4 box")
            size32, box_type = struct.unpack_from(">I4s", header)
            header_size = 8
            if size32 == 1:
                if len(header) < 16:
                    raise ValueError("Truncated extended top-level MP4 box")
                size = struct.unpack_from(">Q", header, 8)[0]
                header_size = 16
            elif size32 == 0:
                size = file_size - position
            else:
                size = size32
            if size < header_size or position + size > file_size:
                raise ValueError(
                    f"Invalid top-level MP4 box {box_type!r} at {position}"
                )
            boxes.append(_Box(position, size, header_size, box_type))
            position += size
    return boxes


def _pack_box(box_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    if size >= 2**32:
        return struct.pack(">I4sQ", 1, box_type, 16 + len(payload)) + payload
    return struct.pack(">I4s", size, box_type) + payload


def _payload(raw: bytes) -> tuple[bytes, int]:
    box = _box_at(raw, 0, len(raw))
    return raw[box.header_size : box.size], box.header_size


def _children(raw: bytes, prefix_size: int = 0) -> list[bytes]:
    payload, _ = _payload(raw)
    return [
        payload[box.start : box.end]
        for box in _boxes(payload, prefix_size, len(payload))
    ]


def _find_child(raw: bytes, box_type: bytes, prefix_size: int = 0) -> bytes | None:
    for child in _children(raw, prefix_size):
        if child[4:8] == box_type:
            return child
    return None


def _rewrite_container(raw: bytes, prefix_size: int, rewrite) -> bytes:
    box = _box_at(raw, 0, len(raw))
    payload = raw[box.header_size : box.size]
    prefix = payload[:prefix_size]
    children = _boxes(payload, prefix_size, len(payload))
    rewritten = [prefix]
    for child in children:
        child_raw = payload[child.start : child.end]
        rewritten.append(rewrite(child_raw))
    return _pack_box(box.box_type, b"".join(rewritten))


def _is_video_track(trak: bytes) -> bool:
    mdia = _find_child(trak, b"mdia")
    if mdia is None:
        return False
    hdlr = _find_child(mdia, b"hdlr")
    if hdlr is None:
        return False
    payload, _ = _payload(hdlr)
    return len(payload) >= 12 and payload[8:12] == b"vide"


def _video_sample_entry(moov: bytes) -> bytes:
    for trak in _children(moov):
        if trak[4:8] != b"trak" or not _is_video_track(trak):
            continue
        mdia = _find_child(trak, b"mdia")
        minf = _find_child(mdia, b"minf")
        stbl = _find_child(minf, b"stbl")
        stsd = _find_child(stbl, b"stsd")
        if stsd is None:
            break
        for entry in _children(stsd, 8):
            if entry[4:8] in _VIDEO_SAMPLE_ENTRIES:
                return entry
    raise ValueError("MP4 file has no supported video sample entry")


def _canonical_spatial_atoms() -> tuple[bytes, bytes]:
    st3d = _pack_box(b"st3d", struct.pack(">IB", 0, 2))
    svhd = _pack_box(b"svhd", struct.pack(">I", 0) + b"Jasna\0")
    prhd = _pack_box(b"prhd", struct.pack(">IIII", 0, 0, 0, 0))
    equi = _pack_box(
        b"equi",
        struct.pack(
            ">IIIII",
            0,
            0,
            0,
            0x40000000,
            0x40000000,
        ),
    )
    proj = _pack_box(b"proj", prhd + equi)
    return st3d, _pack_box(b"sv3d", svhd + proj)


def _compatible_spatial_atoms(
    st3d: bytes | None,
    sv3d: bytes | None,
) -> tuple[bytes, bytes]:
    canonical_st3d, canonical_sv3d = _canonical_spatial_atoms()
    if st3d is not None:
        payload, _ = _payload(st3d)
        if len(payload) < 5 or payload[4] != 2:
            st3d = None
    if sv3d is not None:
        proj = _find_child(sv3d, b"proj")
        if proj is None or _find_child(proj, b"equi") is None:
            sv3d = None
    return st3d or canonical_st3d, sv3d or canonical_sv3d


def _read_moov(path: Path) -> tuple[_Box, bytes]:
    moov_box = next(
        (box for box in _file_boxes(path) if box.box_type == b"moov"),
        None,
    )
    if moov_box is None:
        raise ValueError(f"{path.name} has no moov box")
    with path.open("rb") as file:
        file.seek(moov_box.start)
        return moov_box, file.read(moov_box.size)


def _read_spatial_atoms(path: Path) -> tuple[bytes | None, bytes | None]:
    _, moov = _read_moov(Path(path))
    entry = _video_sample_entry(moov)
    st3d = None
    sv3d = None
    for child in _children(entry, 78):
        if child[4:8] == b"st3d":
            st3d = child
        elif child[4:8] == b"sv3d":
            sv3d = child
    return st3d, sv3d


def _inject_into_sample_entry(
    entry: bytes,
    st3d: bytes,
    sv3d: bytes,
) -> bytes:
    box = _box_at(entry, 0, len(entry))
    payload = entry[box.header_size : box.size]
    prefix = payload[:78]
    children = [
        payload[child.start : child.end]
        for child in _boxes(payload, 78, len(payload))
        if child.box_type not in {b"st3d", b"sv3d"}
    ]
    return _pack_box(box.box_type, prefix + b"".join(children) + st3d + sv3d)


def _rewrite_video_track(trak: bytes, st3d: bytes, sv3d: bytes) -> bytes:
    def rewrite_mdia(child: bytes) -> bytes:
        if child[4:8] != b"minf":
            return child

        def rewrite_minf(grandchild: bytes) -> bytes:
            if grandchild[4:8] != b"stbl":
                return grandchild

            def rewrite_stbl(stbl_child: bytes) -> bytes:
                if stbl_child[4:8] != b"stsd":
                    return stbl_child
                replaced = False

                def rewrite_entry(entry: bytes) -> bytes:
                    nonlocal replaced
                    if not replaced and entry[4:8] in _VIDEO_SAMPLE_ENTRIES:
                        replaced = True
                        return _inject_into_sample_entry(entry, st3d, sv3d)
                    return entry

                result = _rewrite_container(stbl_child, 8, rewrite_entry)
                if not replaced:
                    raise ValueError("Video stsd has no supported sample entry")
                return result

            return _rewrite_container(grandchild, 0, rewrite_stbl)

        return _rewrite_container(child, 0, rewrite_minf)

    return _rewrite_container(
        trak,
        0,
        lambda child: (
            _rewrite_container(child, 0, rewrite_mdia)
            if child[4:8] == b"mdia"
            else child
        ),
    )


def _inject_moov(moov: bytes, st3d: bytes, sv3d: bytes) -> bytes:
    replaced = False

    def rewrite(child: bytes) -> bytes:
        nonlocal replaced
        if child[4:8] == b"trak" and not replaced and _is_video_track(child):
            replaced = True
            return _rewrite_video_track(child, st3d, sv3d)
        return child

    result = _rewrite_container(moov, 0, rewrite)
    if not replaced:
        raise ValueError("MP4 file has no video track")
    return result


def _adjust_chunk_offsets(raw: bytes, delta: int) -> bytes:
    box = _box_at(raw, 0, len(raw))
    if box.box_type in {b"stco", b"co64"}:
        payload = bytearray(raw[box.header_size : box.size])
        width = 4 if box.box_type == b"stco" else 8
        mode = ">I" if width == 4 else ">Q"
        if len(payload) < 8:
            raise ValueError(f"Truncated {box.box_type.decode()} box")
        count = struct.unpack_from(">I", payload, 4)[0]
        if 8 + count * width > len(payload):
            raise ValueError(f"Truncated {box.box_type.decode()} offsets")
        for index in range(count):
            offset = 8 + index * width
            value = struct.unpack_from(mode, payload, offset)[0] + delta
            if value < 0 or value >= 1 << (width * 8):
                raise ValueError("MP4 chunk offset overflow")
            struct.pack_into(mode, payload, offset, value)
        return _pack_box(box.box_type, bytes(payload))
    if box.box_type not in _CONTAINER_TYPES:
        return raw
    return _rewrite_container(
        raw,
        0,
        lambda child: _adjust_chunk_offsets(child, delta),
    )


def _copy_range(source, destination, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(64 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("Unexpected end of MP4 file")
        destination.write(chunk)
        remaining -= len(chunk)


def _verify_with_ffprobe(path: Path) -> None:
    from jasna.media import get_video_meta_data

    metadata = get_video_meta_data(str(path))
    stereo = metadata.stereo_layout.lower()
    projection = metadata.spherical_projection.lower()
    if "side by side" not in stereo or "equirect" not in projection:
        raise RuntimeError(
            "ffprobe did not find left-right equirectangular metadata after injection"
        )


def inject_vr180_spatial_metadata(
    source_path: Path,
    output_path: Path,
) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".mp4", ".mov"}:
        log.warning(
            "VR spatial metadata is not injected into %s outputs",
            output_path.suffix or "extensionless",
        )
        return

    source_atoms: tuple[bytes | None, bytes | None] = (None, None)
    if source_path.suffix.lower() in {".mp4", ".mov"}:
        try:
            source_atoms = _read_spatial_atoms(source_path)
        except ValueError as exc:
            log.warning(
                "Could not read source VR spatial metadata from %s; using canonical metadata: %s",
                source_path,
                exc,
            )
    st3d, sv3d = _compatible_spatial_atoms(*source_atoms)
    top_boxes = _file_boxes(output_path)
    moov_box, moov = _read_moov(output_path)
    rewritten_moov = _inject_moov(moov, st3d, sv3d)
    mdat_box = next(
        (box for box in top_boxes if box.box_type == b"mdat"),
        None,
    )
    if mdat_box is None:
        raise ValueError(f"{output_path.name} has no mdat box")
    delta = len(rewritten_moov) - len(moov)
    if moov_box.start < mdat_box.start and delta:
        rewritten_moov = _adjust_chunk_offsets(rewritten_moov, delta)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.spatial-",
            suffix=output_path.suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with output_path.open("rb") as source:
                for box in top_boxes:
                    if box.box_type == b"moov":
                        temporary.write(rewritten_moov)
                    else:
                        source.seek(box.start)
                        _copy_range(source, temporary, box.size)
            temporary.flush()
            os.fsync(temporary.fileno())
        _verify_with_ffprobe(temporary_path)
        shutil.copymode(output_path, temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

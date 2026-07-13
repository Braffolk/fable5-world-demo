"""LAC1 chunk container: one small self-describing binary file per (layer, lod, cx, cz).

Header layout is little-endian, fixed 56 bytes (struct FMT below), followed by the
compressed payload. originE/originN are float64 absolute L-EST97 QA fields — the client
must position chunks from (cx, cz) + the manifest anchor, never from these.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"LAC1"
FMT = "<4sBBBBiiHxxIddffII"  # 56 bytes
HEADER_SIZE = struct.calcsize(FMT)
MAGIC_V2 = b"LAC2"
FMT_V2 = "<4sBbBBiiHxxIddffII"  # same 56 bytes; signed physical LOD
HEADER_SIZE_V2 = struct.calcsize(FMT_V2)
assert HEADER_SIZE_V2 == HEADER_SIZE

LAYER_IDS = {
    "height": 0, "biome": 1, "water": 2, "trees": 3, "soil": 4,
    "understory": 5, "debris": 6, "boulders": 7, "canopy": 8, "watercover": 9,
}
LAYER_NAMES = {v: k for k, v in LAYER_IDS.items()}


@dataclass(frozen=True)
class ChunkMeta:
    layer: str
    lod: int
    enc: int
    cx: int
    cz: int
    res: int  # texels per side (rasters) or 0 (record layers)
    count: int  # record count (record layers) or 0
    origin_e: float
    origin_n: float
    qoffset: float
    qscale: float
    flags: int = 0


def write_chunk(path: Path, meta: ChunkMeta, payload: bytes) -> None:
    header = struct.pack(
        FMT,
        MAGIC,
        LAYER_IDS[meta.layer],
        meta.lod,
        meta.enc,
        meta.flags,
        meta.cx,
        meta.cz,
        meta.res,
        meta.count,
        meta.origin_e,
        meta.origin_n,
        meta.qoffset,
        meta.qscale,
        len(payload),
        zlib.crc32(payload),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(header + payload)
    tmp.replace(path)


def write_chunk_v2(path: Path, meta: ChunkMeta, payload: bytes) -> None:
    """Write format-2 chunk without changing the LAC1 byte contract."""
    if not -8 <= meta.lod <= 55:
        raise ValueError(f"LAC2 LOD {meta.lod} outside -8..55")
    if abs(meta.cx) >= 1 << 20 or abs(meta.cz) >= 1 << 20:
        raise ValueError(f"LAC2 chunk coordinate outside signed key range: {(meta.cx, meta.cz)}")
    header = struct.pack(
        FMT_V2,
        MAGIC_V2,
        LAYER_IDS[meta.layer],
        meta.lod,
        meta.enc,
        meta.flags,
        meta.cx,
        meta.cz,
        meta.res,
        meta.count,
        meta.origin_e,
        meta.origin_n,
        meta.qoffset,
        meta.qscale,
        len(payload),
        zlib.crc32(payload),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(header + payload)
    tmp.replace(path)


def _unpack_meta(path: Path, header: bytes) -> tuple[ChunkMeta, int, int]:
    (
        magic, layer, lod, enc, flags, cx, cz, res, count,
        origin_e, origin_n, qoffset, qscale, plen, crc,
    ) = struct.unpack(FMT, header)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    meta = ChunkMeta(
        layer=LAYER_NAMES[layer], lod=lod, enc=enc, cx=cx, cz=cz, res=res, count=count,
        origin_e=origin_e, origin_n=origin_n, qoffset=qoffset, qscale=qscale, flags=flags,
    )
    return meta, plen, crc


def _unpack_meta_v2(path: Path, header: bytes) -> tuple[ChunkMeta, int, int]:
    if len(header) != HEADER_SIZE_V2:
        raise ValueError(f"{path}: short LAC2 header ({len(header)} bytes)")
    (
        magic, layer, lod, enc, flags, cx, cz, res, count,
        origin_e, origin_n, qoffset, qscale, plen, crc,
    ) = struct.unpack(FMT_V2, header)
    if magic != MAGIC_V2:
        raise ValueError(f"{path}: bad LAC2 magic {magic!r}")
    if layer not in LAYER_NAMES:
        raise ValueError(f"{path}: unknown layer id {layer}")
    if not -8 <= lod <= 55:
        raise ValueError(f"{path}: LAC2 LOD {lod} outside -8..55")
    if abs(cx) >= 1 << 20 or abs(cz) >= 1 << 20:
        raise ValueError(f"{path}: LAC2 coordinate outside signed key range")
    meta = ChunkMeta(
        layer=LAYER_NAMES[layer], lod=lod, enc=enc, cx=cx, cz=cz, res=res, count=count,
        origin_e=origin_e, origin_n=origin_n, qoffset=qoffset, qscale=qscale, flags=flags,
    )
    return meta, plen, crc


def read_header(path: Path) -> ChunkMeta:
    """Decode just the 56-byte header (no payload read/CRC) — cheap idempotence checks."""
    with open(path, "rb") as f:
        return _unpack_meta(path, f.read(HEADER_SIZE))[0]


def read_header_v2(path: Path) -> ChunkMeta:
    with open(path, "rb") as f:
        return _unpack_meta_v2(path, f.read(HEADER_SIZE_V2))[0]


def read_chunk(path: Path) -> tuple[ChunkMeta, bytes]:
    blob = path.read_bytes()
    meta, plen, crc = _unpack_meta(path, blob[:HEADER_SIZE])
    payload = blob[HEADER_SIZE : HEADER_SIZE + plen]
    if len(payload) != plen or zlib.crc32(payload) != crc:
        raise ValueError(f"{path}: payload length/crc mismatch")
    return meta, payload


def read_chunk_v2(path: Path) -> tuple[ChunkMeta, bytes]:
    blob = path.read_bytes()
    meta, plen, crc = _unpack_meta_v2(path, blob[:HEADER_SIZE_V2])
    payload = blob[HEADER_SIZE_V2 : HEADER_SIZE_V2 + plen]
    if len(payload) != plen or len(blob) != HEADER_SIZE_V2 + plen or zlib.crc32(payload) != crc:
        raise ValueError(f"{path}: LAC2 payload length/crc mismatch")
    return meta, payload


def read_chunk_any(path: Path) -> tuple[int, ChunkMeta, bytes]:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == MAGIC:
        meta, payload = read_chunk(path)
        return 1, meta, payload
    if magic == MAGIC_V2:
        meta, payload = read_chunk_v2(path)
        return 2, meta, payload
    raise ValueError(f"{path}: unknown chunk magic {magic!r}")

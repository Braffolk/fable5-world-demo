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

LAYER_IDS = {
    "height": 0, "biome": 1, "water": 2, "trees": 3, "soil": 4,
    "understory": 5, "debris": 6, "boulders": 7,
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


def read_chunk(path: Path) -> tuple[ChunkMeta, bytes]:
    blob = path.read_bytes()
    (
        magic, layer, lod, enc, flags, cx, cz, res, count,
        origin_e, origin_n, qoffset, qscale, plen, crc,
    ) = struct.unpack(FMT, blob[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    payload = blob[HEADER_SIZE : HEADER_SIZE + plen]
    if len(payload) != plen or zlib.crc32(payload) != crc:
        raise ValueError(f"{path}: payload length/crc mismatch")
    meta = ChunkMeta(
        layer=LAYER_NAMES[layer], lod=lod, enc=enc, cx=cx, cz=cz, res=res, count=count,
        origin_e=origin_e, origin_n=origin_n, qoffset=qoffset, qscale=qscale, flags=flags,
    )
    return meta, payload

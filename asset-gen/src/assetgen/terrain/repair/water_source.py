"""Bounded, integrity-checked water reads from a pinned format-1 release."""
from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import ChunkMeta, read_chunk
from ...cook.encode import decode_quant16
from ...grid import ChunkId
from ...release import IndexRecord, audit_base_release, read_v1_index
from .water_halo import DecodedWaterHaloChunk, WaterHaloSource
from .water_pack import DecodedWaterDependency


_ABSENT_DOMAIN = b"laas.structural-water.inherited-absent.v1\0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InheritedWaterChunk:
    chunk: ChunkId
    meta: ChunkMeta
    water_y: np.ndarray
    artifact_sha256: str
    payload_sha256: str


class AuditedFormat1WaterSource:
    """Decode ordinary water chunks with explicit, release-bound absence."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        content_root: Path,
        grid: GridConfig,
        encode: EncodeConfig,
        cache_chunks: int = 4,
        audit: bool = True,
    ) -> None:
        if isinstance(cache_chunks, bool) or not isinstance(cache_chunks, int):
            raise TypeError("cache_chunks must be an integer")
        if cache_chunks < 1:
            raise ValueError("cache_chunks must be positive")
        manifest_path = Path(manifest_path)
        if audit:
            audit_base_release(manifest_path, manifest_sha256, content_root)
        elif _sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("pinned water manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("format") != 1 or manifest.get("codec") != encode.codec:
            raise ValueError("pinned water source requires a codec-compatible format-1 release")
        layer = manifest.get("layers", {}).get("water")
        if not isinstance(layer, dict) or layer.get("enc") != 1:
            raise ValueError("pinned release has no quantized water layer")
        index_path = manifest_path.parent / layer["index"]
        self._records = {record.key: record for record in read_v1_index(index_path)}
        self._content_root = Path(content_root)
        self._grid = grid
        self._encode = encode
        self._capacity = cache_chunks
        self._cache: OrderedDict[ChunkId, InheritedWaterChunk] = OrderedDict()
        self.manifest_sha256 = manifest_sha256

    @property
    def cache_capacity(self) -> int:
        return self._capacity

    @property
    def cached_chunk_count(self) -> int:
        return len(self._cache)

    def contains(self, chunk: ChunkId) -> bool:
        return (chunk.lod, chunk.cx, chunk.cz) in self._records

    def _content_path(self, record: IndexRecord) -> Path:
        hash8 = ((record.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
        return (
            self._content_root
            / "c"
            / "water"
            / str(record.lod)
            / f"{record.cx}_{record.cz}.{hash8}.bin"
        )

    def absent_sha256(self, chunk: ChunkId) -> str:
        digest = hashlib.sha256(_ABSENT_DOMAIN)
        digest.update(bytes.fromhex(self.manifest_sha256))
        digest.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
        return digest.hexdigest()

    def load(self, chunk: ChunkId) -> InheritedWaterChunk | None:
        cached = self._cache.pop(chunk, None)
        if cached is not None:
            self._cache[chunk] = cached
            return cached
        record = self._records.get((chunk.lod, chunk.cx, chunk.cz))
        if record is None:
            return None
        path = self._content_path(record)
        if not path.is_file() or path.stat().st_size != record.size:
            raise ValueError(f"pinned water content size mismatch: {path}")
        artifact_sha = _sha256_file(path)
        if int.from_bytes(bytes.fromhex(artifact_sha)[:8], "big") != record.hash64:
            raise ValueError(f"pinned water content hash mismatch: {path}")
        meta, payload = read_chunk(path)
        expected_res = self._grid.chunk_m // 2 + 1
        if (
            (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc, meta.res, meta.count)
            != ("water", chunk.lod, chunk.cx, chunk.cz, 1, expected_res, 0)
        ):
            raise ValueError(f"pinned water header mismatch: {path}")
        decoded = decode_quant16(
            self._encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        codes = np.rint(
            (decoded.astype(np.float64) - float(meta.qoffset)) / float(meta.qscale)
        )
        if float(codes.min()) < 0.0 or float(codes.max()) > 65535.0:
            raise ValueError(f"pinned water codes are not recoverable: {path}")
        water = decoded.astype(np.float32, copy=True)
        water[codes == 0.0] = np.nan
        water.flags.writeable = False
        result = InheritedWaterChunk(
            chunk=chunk,
            meta=meta,
            water_y=water,
            artifact_sha256=artifact_sha,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
        self._cache[chunk] = result
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return result

    def load_halo(self, chunk: ChunkId) -> DecodedWaterHaloChunk:
        inherited = self.load(chunk)
        if inherited is None:
            return DecodedWaterHaloChunk(
                chunk, WaterHaloSource.ABSENT, None, self.absent_sha256(chunk)
            )
        return DecodedWaterHaloChunk(
            chunk,
            WaterHaloSource.INHERITED,
            inherited.water_y,
            inherited.artifact_sha256,
        )

    def load_dependency(self, chunk: ChunkId) -> DecodedWaterDependency:
        inherited = self.load(chunk)
        if inherited is None:
            return DecodedWaterDependency(None, self.absent_sha256(chunk))
        return DecodedWaterDependency(inherited.water_y, inherited.artifact_sha256)

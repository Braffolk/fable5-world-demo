"""Read exact LOD0 height cells from an explicitly pinned base release."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import EncodeConfig, GridConfig
from ..grid import ChunkId
from ..release import IndexRecord, audit_base_release, read_v1_index
from .chunkio import read_chunk
from .encode import decode_quant16


def assemble_lod0_cell_window(
    grid: GridConfig,
    e_min: int,
    n_max: int,
    width: int,
    height: int,
    load_chunk: Callable[[ChunkId], np.ndarray],
) -> np.ndarray:
    """Assemble integral one-meter cells, rows north-to-south.

    Only each LOD0 chunk's 2048-square core is authoritative.  Its final row and
    column are duplicate neighbor samples and are never used for window assembly.
    """
    if width < 1 or height < 1:
        raise ValueError("LOD0 cell window dimensions must be positive")
    if (grid.chunk_m, grid.chunk_res, grid.lod_step) != (2048, 2048, 4):
        raise ValueError("pinned height reader requires the frozen world grid")
    e_max, n_min = e_min + width, n_max - height
    cx0 = (e_min - grid.anchor_e) // grid.chunk_m
    cx1 = (e_max - 1 - grid.anchor_e) // grid.chunk_m
    cz0 = (grid.anchor_n - n_max) // grid.chunk_m
    cz1 = (grid.anchor_n - n_min - 1) // grid.chunk_m
    result = np.empty((height, width), dtype=np.float32)
    for cz in range(cz0, cz1 + 1):
        origin_n = grid.anchor_n - cz * grid.chunk_m
        chunk_n_min = origin_n - grid.chunk_m
        copy_n_min = max(n_min, chunk_n_min)
        copy_n_max = min(n_max, origin_n)
        dst_y0 = n_max - copy_n_max
        src_y0 = origin_n - copy_n_max
        rows = copy_n_max - copy_n_min
        for cx in range(cx0, cx1 + 1):
            origin_e = grid.anchor_e + cx * grid.chunk_m
            copy_e_min = max(e_min, origin_e)
            copy_e_max = min(e_max, origin_e + grid.chunk_m)
            dst_x0 = copy_e_min - e_min
            src_x0 = copy_e_min - origin_e
            cols = copy_e_max - copy_e_min
            values = np.asarray(load_chunk(ChunkId(cx, cz, 0)))
            expected = (grid.chunk_res + 1, grid.chunk_res + 1)
            if values.dtype != np.float32 or values.shape != expected:
                raise ValueError(f"LOD0 chunk {(cx, cz)} is not float32 {expected}")
            if not np.isfinite(values).all():
                raise ValueError(f"LOD0 chunk {(cx, cz)} contains nonfinite height")
            result[dst_y0 : dst_y0 + rows, dst_x0 : dst_x0 + cols] = values[
                src_y0 : src_y0 + rows, src_x0 : src_x0 + cols
            ]
    return result


class PinnedBaseHeight:
    """Audited base-release LOD0 decoder with a small per-process cache."""

    def __init__(
        self,
        manifest_path: Path,
        manifest_sha256: str,
        out_root: Path,
        encode: EncodeConfig,
        *,
        audit: bool = True,
    ) -> None:
        if audit:
            audit_base_release(manifest_path, manifest_sha256, out_root)
        elif hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
            raise ValueError("base manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_bytes())
        self.grid = GridConfig(
            anchor_e=int(manifest["anchor"]["e"]),
            anchor_n=int(manifest["anchor"]["n"]),
            chunk_m=int(manifest["chunkMeters"]),
            lod_step=int(manifest["lodStep"]),
            lods=tuple(int(v) for v in manifest["layers"]["height"]["lods"]),
            chunk_res=int(manifest["chunkRes"]),
        )
        if manifest.get("format") != 1 or manifest.get("codec") != encode.codec:
            raise ValueError("pinned base must be a codec-compatible format-1 release")
        index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
        self._records = {record.key: record for record in read_v1_index(index_path)}
        self._out_root = out_root
        self._encode = encode
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _content_path(self, record: IndexRecord) -> Path:
        hash8 = ((record.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
        return self._out_root / "c" / "height" / "0" / f"{record.cx}_{record.cz}.{hash8}.bin"

    def read_lod0_chunk(self, chunk: ChunkId) -> np.ndarray:
        if chunk.lod != 0:
            raise ValueError("PinnedBaseHeight decodes LOD0 only")
        cache_key = (chunk.cx, chunk.cz)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            record = self._records[(0, chunk.cx, chunk.cz)]
        except KeyError as exc:
            raise ValueError(f"pinned base lacks LOD0 height chunk {cache_key}") from exc
        path = self._content_path(record)
        if path.stat().st_size != record.size:
            raise ValueError(f"pinned height content size mismatch: {path}")
        meta, payload = read_chunk(path)
        if (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc, meta.res) != (
            "height", 0, chunk.cx, chunk.cz, 1, self.grid.chunk_res + 1
        ):
            raise ValueError(f"pinned height header mismatch: {path}")
        decoded = decode_quant16(
            self._encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        self._cache[cache_key] = decoded
        return decoded

    def read_cells(self, e_min: int, n_max: int, width: int, height: int) -> np.ndarray:
        return assemble_lod0_cell_window(
            self.grid, e_min, n_max, width, height, self.read_lod0_chunk
        )

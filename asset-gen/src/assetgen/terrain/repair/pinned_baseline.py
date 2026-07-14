"""Bounded reconstruction of canonical 0.25 m baseline from a pinned LOD0 release."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import read_chunk
from ...cook.encode import decode_quant16
from ...cook.micro_hierarchy import box_mean_fixed
from ...height_geom import (
    UNITS_PER_METER,
    HeightChunkId,
    chunk_origin_en_units,
    footprint_units,
)
from ...release import IndexRecord, audit_base_release, read_v1_index
from .baseline import BaselineTile
from .prolong import prolong_structural_4x


PINNED_BASELINE_VERSION = "pinned-decoded-keys-bubble-4x/2"
_SOURCE_CORE_RES = 2048
_AUTHORITY_CORE_RES = 512
_SOURCE_CELLS = _AUTHORITY_CORE_RES // 4
_SOURCE_HALO = 2
_SOURCE_WINDOW = _SOURCE_CELLS + 2 * _SOURCE_HALO
_DEPENDENCY_DOMAIN = b"laas.terrain.pinned-baseline.dependencies.v1\0"


@dataclass(frozen=True)
class PinnedHeightDependency:
    chunk: HeightChunkId
    content_relative_path: str
    bytes: int
    sha256: str
    index_hash64: int
    decoded_sha256: str
    qoffset: float
    qscale: float
    flags: int


@dataclass(frozen=True)
class ReconstructedBaselineTile:
    chunk: HeightChunkId
    tile: BaselineTile
    manifest_sha256: str
    dependencies: tuple[PinnedHeightDependency, ...]
    dependency_root_sha256: str
    authority_sha256: str
    maximum_mean_error_m: float
    mean_error_limit_m: float
    reconstruction_version: str = PINNED_BASELINE_VERSION


@dataclass(frozen=True)
class PinnedBaselineReleaseIdentity:
    manifest_sha256: str
    height_index_relative_path: str
    height_index_bytes: int
    height_index_sha256: str
    reconstruction_version: str = PINNED_BASELINE_VERSION


@dataclass(frozen=True)
class _DecodedChunk:
    values: np.ndarray
    dependency: PinnedHeightDependency


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decoded_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _authority_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _dependency_root(dependencies: tuple[PinnedHeightDependency, ...]) -> str:
    rows = []
    for dependency in dependencies:
        row = asdict(dependency)
        row["chunk"] = [
            dependency.chunk.lod,
            dependency.chunk.cx,
            dependency.chunk.cz,
        ]
        rows.append(row)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(_DEPENDENCY_DOMAIN + payload).hexdigest()


class PinnedDecodedBaseline:
    """Audited LOD0 reader with a strict, bounded decoded-chunk LRU."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        content_root: Path,
        encode: EncodeConfig,
        cache_chunks: int = 4,
    ) -> None:
        if isinstance(cache_chunks, bool) or not isinstance(cache_chunks, int):
            raise TypeError("cache_chunks must be an integer")
        if cache_chunks < 1:
            raise ValueError("cache_chunks must be positive")
        manifest_path = Path(manifest_path)
        content_root = Path(content_root)
        audit_base_release(manifest_path, manifest_sha256, content_root)
        manifest = json.loads(manifest_path.read_bytes())
        self.grid = GridConfig(
            anchor_e=int(manifest["anchor"]["e"]),
            anchor_n=int(manifest["anchor"]["n"]),
            chunk_m=int(manifest["chunkMeters"]),
            lod_step=int(manifest["lodStep"]),
            lods=tuple(int(value) for value in manifest["layers"]["height"]["lods"]),
            chunk_res=int(manifest["chunkRes"]),
        )
        if (
            manifest.get("format") != 1
            or manifest.get("codec") != encode.codec
            or (self.grid.chunk_m, self.grid.chunk_res, self.grid.lod_step)
            != (2048, 2048, 4)
        ):
            raise ValueError("pinned baseline requires the canonical format-1 height grid")
        index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
        if not index_path.is_file():
            raise ValueError(f"pinned base height index is missing: {index_path}")
        self._records = {
            record.key: record for record in read_v1_index(index_path) if record.lod == 0
        }
        if not self._records:
            raise ValueError("pinned base has no LOD0 height records")
        self._manifest_sha256 = manifest_sha256
        self._release_identity = PinnedBaselineReleaseIdentity(
            manifest_sha256=manifest_sha256,
            height_index_relative_path=manifest["layers"]["height"]["index"],
            height_index_bytes=index_path.stat().st_size,
            height_index_sha256=_sha256_file(index_path),
        )
        self._content_root = content_root
        self._encode = encode
        self._cache_chunks = cache_chunks
        self._cache: OrderedDict[HeightChunkId, _DecodedChunk] = OrderedDict()

    @property
    def cached_chunk_count(self) -> int:
        return len(self._cache)

    @property
    def release_identity(self) -> PinnedBaselineReleaseIdentity:
        return self._release_identity

    def _content_path(self, record: IndexRecord) -> tuple[Path, str]:
        hash8 = ((record.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
        relative = Path("c") / "height" / "0" / f"{record.cx}_{record.cz}.{hash8}.bin"
        return self._content_root / relative, relative.as_posix()

    def _load_chunk(self, chunk: HeightChunkId) -> _DecodedChunk:
        if chunk.lod != 0:
            raise ValueError("pinned baseline source chunks must be LOD0")
        cached = self._cache.pop(chunk, None)
        if cached is not None:
            self._cache[chunk] = cached
            return cached
        try:
            record = self._records[(0, chunk.cx, chunk.cz)]
        except KeyError as error:
            raise ValueError(f"pinned base lacks required LOD0 chunk {chunk}") from error
        path, relative = self._content_path(record)
        if not path.is_file() or path.stat().st_size != record.size:
            raise ValueError(f"pinned height content size mismatch: {path}")
        artifact_sha256 = _sha256_file(path)
        if int.from_bytes(bytes.fromhex(artifact_sha256)[:8], "big") != record.hash64:
            raise ValueError(f"pinned height content hash mismatch: {path}")
        meta, payload = read_chunk(path)
        expected_origin_e = self.grid.anchor_e + chunk.cx * self.grid.chunk_m
        expected_origin_n = self.grid.anchor_n - chunk.cz * self.grid.chunk_m
        expected_header = (
            "height",
            0,
            chunk.cx,
            chunk.cz,
            1,
            _SOURCE_CORE_RES + 1,
            0,
            float(expected_origin_e),
            float(expected_origin_n),
        )
        actual_header = (
            meta.layer,
            meta.lod,
            meta.cx,
            meta.cz,
            meta.enc,
            meta.res,
            meta.count,
            meta.origin_e,
            meta.origin_n,
        )
        if actual_header != expected_header or not np.isfinite(meta.qscale) or meta.qscale <= 0:
            raise ValueError(f"pinned height header mismatch: {path}")
        values = decode_quant16(
            self._encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        if values.dtype != np.float32 or values.shape != (
            _SOURCE_CORE_RES + 1,
            _SOURCE_CORE_RES + 1,
        ) or not np.isfinite(values).all():
            raise ValueError(f"pinned height decode is invalid: {path}")
        loaded = _DecodedChunk(
            values=values,
            dependency=PinnedHeightDependency(
                chunk=chunk,
                content_relative_path=relative,
                bytes=record.size,
                sha256=artifact_sha256,
                index_hash64=record.hash64,
                decoded_sha256=_decoded_sha256(values),
                qoffset=meta.qoffset,
                qscale=meta.qscale,
                flags=meta.flags,
            ),
        )
        self._cache[chunk] = loaded
        while len(self._cache) > self._cache_chunks:
            self._cache.popitem(last=False)
        return loaded

    def _source_window(
        self, authority_chunk: HeightChunkId, *, parent_halo: int
    ) -> tuple[np.ndarray, tuple[PinnedHeightDependency, ...]]:
        if authority_chunk.lod != -2:
            raise ValueError("0.25 m baseline tiles use LOD-2 128 m alignment")
        footprint = footprint_units(self.grid, authority_chunk.lod)
        if footprint != 128 * UNITS_PER_METER:
            raise ValueError("authority chunk does not have a 128 m footprint")
        origin_e_units, origin_n_units = chunk_origin_en_units(self.grid, authority_chunk)
        if origin_e_units % UNITS_PER_METER or origin_n_units % UNITS_PER_METER:
            raise ValueError("authority tile boundary is not aligned to whole-meter cells")
        west = origin_e_units // UNITS_PER_METER
        north = origin_n_units // UNITS_PER_METER
        source_window = _SOURCE_WINDOW + 2 * parent_halo
        e_min = west - _SOURCE_HALO - parent_halo
        n_max = north + _SOURCE_HALO + parent_halo
        e_max = e_min + source_window
        n_min = n_max - source_window
        cx0 = (e_min - self.grid.anchor_e) // self.grid.chunk_m
        cx1 = (e_max - 1 - self.grid.anchor_e) // self.grid.chunk_m
        cz0 = (self.grid.anchor_n - n_max) // self.grid.chunk_m
        cz1 = (self.grid.anchor_n - n_min - 1) // self.grid.chunk_m
        output = np.empty((source_window, source_window), dtype=np.float32)
        dependencies: dict[HeightChunkId, PinnedHeightDependency] = {}
        for cz in range(cz0, cz1 + 1):
            chunk_origin_n = self.grid.anchor_n - cz * self.grid.chunk_m
            chunk_n_min = chunk_origin_n - self.grid.chunk_m
            copy_n_min = max(n_min, chunk_n_min)
            copy_n_max = min(n_max, chunk_origin_n)
            destination_row = n_max - copy_n_max
            source_row = chunk_origin_n - copy_n_max
            rows = copy_n_max - copy_n_min
            for cx in range(cx0, cx1 + 1):
                chunk_origin_e = self.grid.anchor_e + cx * self.grid.chunk_m
                copy_e_min = max(e_min, chunk_origin_e)
                copy_e_max = min(e_max, chunk_origin_e + self.grid.chunk_m)
                destination_col = copy_e_min - e_min
                source_col = copy_e_min - chunk_origin_e
                cols = copy_e_max - copy_e_min
                source_chunk = HeightChunkId(0, cx, cz)
                loaded = self._load_chunk(source_chunk)
                dependencies[source_chunk] = loaded.dependency
                output[
                    destination_row : destination_row + rows,
                    destination_col : destination_col + cols,
                ] = loaded.values[
                    source_row : source_row + rows,
                    source_col : source_col + cols,
                ]
        ordered = tuple(
            dependencies[chunk]
            for chunk in sorted(dependencies, key=lambda item: (item.lod, item.cz, item.cx))
        )
        return output, ordered

    def reconstruct(
        self, authority_chunk: HeightChunkId, *, halo_samples: int = 0
    ) -> ReconstructedBaselineTile:
        """Reconstruct one canonical tile/window and prove its decoded LOD0 means.

        ``halo_samples`` is measured on the 0.25 m output grid and must describe
        whole 1 m parent cells. It exists for spatial algorithms whose core
        result depends on a real neighboring collar; no edge padding is used.
        """
        if (
            isinstance(halo_samples, bool)
            or not isinstance(halo_samples, int)
            or halo_samples < 0
            or halo_samples % 4
        ):
            raise ValueError("halo_samples must be a nonnegative multiple of four")
        parent_halo = halo_samples // 4
        source, dependencies = self._source_window(
            authority_chunk, parent_halo=parent_halo
        )
        parent_side = _SOURCE_CELLS + 2 * parent_halo
        authority = np.asarray(
            source[
                _SOURCE_HALO : _SOURCE_HALO + parent_side,
                _SOURCE_HALO : _SOURCE_HALO + parent_side,
            ],
            dtype=np.float64,
        )
        fine = prolong_structural_4x(
            source,
            parent_rows=(_SOURCE_HALO, _SOURCE_HALO + parent_side),
            parent_cols=(_SOURCE_HALO, _SOURCE_HALO + parent_side),
        )
        expected_side = _AUTHORITY_CORE_RES + 2 * halo_samples
        if fine.shape != (expected_side, expected_side):
            raise AssertionError("baseline prolongation returned the wrong tile shape")
        reduced = box_mean_fixed(fine)
        maximum_error = float(np.max(np.abs(reduced - authority), initial=0.0))
        peak = float(np.max(np.abs(authority), initial=1.0))
        limit = 64.0 * np.finfo(np.float64).eps * max(1.0, peak)
        if maximum_error > limit:
            raise AssertionError(
                f"pinned baseline mean error {maximum_error} exceeds {limit}"
            )
        valid = np.ones(fine.shape, dtype=bool)
        return ReconstructedBaselineTile(
            chunk=authority_chunk,
            tile=BaselineTile(fine, valid),
            manifest_sha256=self._manifest_sha256,
            dependencies=dependencies,
            dependency_root_sha256=_dependency_root(dependencies),
            authority_sha256=_authority_sha256(authority),
            maximum_mean_error_m=maximum_error,
            mean_error_limit_m=limit,
        )

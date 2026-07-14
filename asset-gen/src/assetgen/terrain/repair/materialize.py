"""Bounded, disk-backed materialization of one structural repair hierarchy.

This module deliberately stops at checked height artifacts.  Corrected-base
apron promotion and immutable release assembly belong to the release layer.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import read_chunk_v2
from ...cook.encode import decode_quant16
from ...height_geom import HeightChunkId
from .authority_inventory import (
    CanonicalBaselineInput,
    StructuralAuthorityPair,
    validated_sha256,
)
from .hierarchy import (
    FINE_QSCALE_M,
    FineReevaluation,
    FineReevaluationResult,
    HierarchyLayout,
    PackedHeightChunk,
    _pack_height,
    assemble_fine_surface_window,
    reduce_decoded_children,
)
from .plan import StructuralRepairPlan


MATERIALIZER_VERSION = "structural-hierarchy-disk-transaction/3"
_FINE_DEPENDENCY_DOMAIN = b"laas.terrain.structural-fine-dependency.v1\0"
_OWNERSHIP_HEADER = struct.Struct("<4sHIIIII")
_OWNERSHIP_MAGIC = b"TOM1"
_OWNERSHIP_FORMAT = 1


def _validated_sha256(value: str, name: str) -> str:
    return validated_sha256(value, name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(chunk: HeightChunkId) -> list[int]:
    return [chunk.lod, chunk.cx, chunk.cz]


PairLoader = Callable[[HeightChunkId], StructuralAuthorityPair]
CanonicalLoader = Callable[[HeightChunkId], CanonicalBaselineInput]


@dataclass(frozen=True)
class FineOwnershipArtifact:
    relative_path: str
    bytes: int
    container_sha256: str
    authority_sha256: str
    abstention_sha256: str
    authority_samples: int
    abstained_samples: int
    closure_sha256: str


@dataclass(frozen=True)
class MaterializedArtifact:
    chunk: HeightChunkId
    relative_path: str
    sidecar_relative_path: str
    bytes: int
    container_sha256: str
    decoded_values_sha256: str
    dependency_merkle_root: str
    ownership: FineOwnershipArtifact | None = None


@dataclass(frozen=True)
class StructuralMaterialization:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    fine: tuple[MaterializedArtifact, ...]
    parents: tuple[MaterializedArtifact, ...]
    corrected_lod0: tuple[MaterializedArtifact, ...]
    published_overlay: tuple[MaterializedArtifact, ...]


class _BoundedCache:
    def __init__(self, load: Callable, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self._load = load
        self._capacity = capacity
        self._items: OrderedDict[HeightChunkId, object] = OrderedDict()

    def get(self, chunk: HeightChunkId):
        value = self._items.pop(chunk, None)
        if value is None:
            value = self._load(chunk)
        self._items[chunk] = value
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return value


def _combined_fine_dependency(
    pair: StructuralAuthorityPair, baseline: CanonicalBaselineInput
) -> str:
    if pair.chunk != baseline.chunk:
        raise ValueError("fine dependency pair has mismatched chunk keys")
    digest = hashlib.sha256(_FINE_DEPENDENCY_DOMAIN)
    digest.update(bytes.fromhex(pair.pair_sha256))
    digest.update(bytes.fromhex(baseline.artifact_sha256))
    return digest.hexdigest()


def _overlap_sha256(
    height: np.ndarray, authority: np.ndarray, abstained: np.ndarray
) -> str:
    digest = hashlib.sha256(b"laas.terrain.fine-world-overlap.v1\0")
    digest.update(np.ascontiguousarray(height, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(authority, dtype="u1").tobytes())
    digest.update(np.ascontiguousarray(abstained, dtype="u1").tobytes())
    return digest.hexdigest()


class _WorldOverlapLedger:
    """Require identical height and ownership on every shared fine sample edge."""

    def __init__(self, chunks: tuple[HeightChunkId, ...]) -> None:
        self._chunks = set(chunks)
        self._edges: dict[tuple[str, int, int, int], str] = {}
        self._matched = 0
        self._expected = sum(
            HeightChunkId(chunk.lod, chunk.cx + 1, chunk.cz) in self._chunks
            for chunk in chunks
        ) + sum(
            HeightChunkId(chunk.lod, chunk.cx, chunk.cz + 1) in self._chunks
            for chunk in chunks
        )

    def _record(self, key: tuple[str, int, int, int], digest: str) -> None:
        prior = self._edges.get(key)
        if prior is None:
            self._edges[key] = digest
            return
        if prior != digest:
            raise ValueError(f"adjacent fine chunks disagree on world overlap {key}")
        self._matched += 1

    def observe(self, chunk: HeightChunkId, result: FineReevaluationResult) -> None:
        self._record(
            ("vertical", chunk.lod, chunk.cx, chunk.cz),
            _overlap_sha256(
                result.height[:, 0], result.authority[:, 0], result.abstained[:, 0]
            ),
        )
        self._record(
            ("vertical", chunk.lod, chunk.cx + 1, chunk.cz),
            _overlap_sha256(
                result.height[:, -1], result.authority[:, -1], result.abstained[:, -1]
            ),
        )
        self._record(
            ("horizontal", chunk.lod, chunk.cx, chunk.cz),
            _overlap_sha256(
                result.height[0, :], result.authority[0, :], result.abstained[0, :]
            ),
        )
        self._record(
            ("horizontal", chunk.lod, chunk.cx, chunk.cz + 1),
            _overlap_sha256(
                result.height[-1, :], result.authority[-1, :], result.abstained[-1, :]
            ),
        )

    def finish(self) -> None:
        if self._matched != self._expected:
            raise AssertionError(
                f"checked {self._matched} fine overlaps, expected {self._expected}"
            )

    @property
    def matched(self) -> int:
        return self._matched


def _write_json_atomic(path: Path, document: object) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable transaction metadata differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _artifact_path(root: Path, chunk: HeightChunkId) -> Path:
    return root / "chunks" / "height" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac2"


def _mask_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="u1").tobytes()).hexdigest()


def decode_fine_ownership(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Strictly decode one TOM1 authority/abstention mask pair."""
    if len(payload) < _OWNERSHIP_HEADER.size:
        raise ValueError("fine ownership payload is truncated")
    magic, version, rows, cols, raw_bytes, compressed_bytes, crc = (
        _OWNERSHIP_HEADER.unpack_from(payload)
    )
    compressed = payload[_OWNERSHIP_HEADER.size :]
    bit_count = rows * cols
    packed_bytes = (bit_count + 7) // 8
    if (
        magic != _OWNERSHIP_MAGIC
        or version != _OWNERSHIP_FORMAT
        or rows < 1
        or cols < 1
        or raw_bytes != 2 * packed_bytes
        or compressed_bytes != len(compressed)
        or zlib.crc32(compressed) != crc
    ):
        raise ValueError("fine ownership header or checksum is invalid")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed) + decoder.flush()
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(raw) != raw_bytes
    ):
        raise ValueError("fine ownership compressed stream is invalid")
    authority_bits = np.unpackbits(
        np.frombuffer(raw[:packed_bytes], dtype=np.uint8), bitorder="little"
    )
    abstained_bits = np.unpackbits(
        np.frombuffer(raw[packed_bytes:], dtype=np.uint8), bitorder="little"
    )
    if np.any(authority_bits[bit_count:]) or np.any(abstained_bits[bit_count:]):
        raise ValueError("fine ownership padding bits must be zero")
    authority = authority_bits[:bit_count].reshape((rows, cols)).astype(np.bool_)
    abstained = abstained_bits[:bit_count].reshape((rows, cols)).astype(np.bool_)
    if np.any(authority & abstained):
        raise ValueError("fine ownership masks overlap")
    return authority, abstained


def _write_fine_ownership(
    root: Path,
    chunk: HeightChunkId,
    authority: np.ndarray,
    abstained: np.ndarray,
    closure_sha256: str,
) -> FineOwnershipArtifact:
    _validated_sha256(closure_sha256, "ownership closure_sha256")
    authority = np.asarray(authority, dtype=np.bool_)
    abstained = np.asarray(abstained, dtype=np.bool_)
    if (
        chunk.lod != -2
        or authority.ndim != 2
        or authority.shape != abstained.shape
        or np.any(authority & abstained)
    ):
        raise ValueError("fine ownership requires disjoint equal-shape LOD-2 masks")
    bit_count = authority.size
    packed_authority = np.packbits(authority.ravel(), bitorder="little")
    packed_abstained = np.packbits(abstained.ravel(), bitorder="little")
    raw = packed_authority.tobytes() + packed_abstained.tobytes()
    compressed = zlib.compress(raw, level=9)
    payload = _OWNERSHIP_HEADER.pack(
        _OWNERSHIP_MAGIC,
        _OWNERSHIP_FORMAT,
        authority.shape[0],
        authority.shape[1],
        len(raw),
        len(compressed),
        zlib.crc32(compressed),
    ) + compressed
    # Reopen the exact wire representation before it becomes verification evidence.
    decoded_authority, decoded_abstained = decode_fine_ownership(payload)
    if not np.array_equal(decoded_authority, authority) or not np.array_equal(
        decoded_abstained, abstained
    ):
        raise AssertionError("fine ownership wire round-trip changed a mask")
    container_sha = hashlib.sha256(payload).hexdigest()
    path = (
        root
        / "ownership"
        / str(chunk.lod)
        / f"{chunk.cx}_{chunk.cz}.{container_sha}.mask"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable fine ownership differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return FineOwnershipArtifact(
        relative_path=path.relative_to(root).as_posix(),
        bytes=len(payload),
        container_sha256=container_sha,
        authority_sha256=_mask_sha256(authority),
        abstention_sha256=_mask_sha256(abstained),
        authority_samples=int(np.count_nonzero(authority)),
        abstained_samples=int(np.count_nonzero(abstained)),
        closure_sha256=closure_sha256,
    )


def _write_packed(
    root: Path,
    packed: PackedHeightChunk,
    *,
    recipe_sha256: str,
    role: str,
    ownership: FineOwnershipArtifact | None = None,
) -> MaterializedArtifact:
    path = _artifact_path(root, packed.chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    packed.write_lac2(temporary)
    if path.exists():
        if _sha256_file(path) != _sha256_file(temporary):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"immutable materialized height differs: {path}")
        temporary.unlink()
    else:
        temporary.replace(path)
    container_sha = _sha256_file(path)
    relative = path.relative_to(root).as_posix()
    sidecar = path.with_suffix(".json")
    document = {
        "format": 1,
        "materializerVersion": MATERIALIZER_VERSION,
        "recipeSha256": recipe_sha256,
        "role": role,
        "key": _key(packed.chunk),
        "path": relative,
        "bytes": path.stat().st_size,
        "containerSha256": container_sha,
        "payloadSha256": packed.payload_sha256,
        "sourceValuesSha256": packed.source_values_sha256,
        "decodedValuesSha256": packed.decoded_values_sha256,
        "maxRoundtripErrorM": packed.max_roundtrip_error_m,
        "roundtripLimitM": packed.roundtrip_limit_m,
        "qoffset": packed.meta.qoffset,
        "qscale": packed.meta.qscale,
        "dependencyMerkleRoot": packed.dependency_merkle_root,
        "dependencies": [
            {"key": _key(chunk), "sha256": sha256}
            for chunk, sha256 in packed.dependencies
        ],
        "ownership": (
            None
            if ownership is None
            else {
                "format": "TOM1/1",
                "path": ownership.relative_path,
                "bytes": ownership.bytes,
                "containerSha256": ownership.container_sha256,
                "authoritySha256": ownership.authority_sha256,
                "abstentionSha256": ownership.abstention_sha256,
                "authoritySamples": ownership.authority_samples,
                "abstainedSamples": ownership.abstained_samples,
                "closureSha256": ownership.closure_sha256,
            }
        ),
    }
    _write_json_atomic(sidecar, document)
    return MaterializedArtifact(
        chunk=packed.chunk,
        relative_path=relative,
        sidecar_relative_path=sidecar.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        container_sha256=container_sha,
        decoded_values_sha256=packed.decoded_values_sha256,
        dependency_merkle_root=packed.dependency_merkle_root,
        ownership=ownership,
    )


class _DecodedArtifactCache:
    def __init__(
        self,
        *,
        root: Path,
        records: dict[HeightChunkId, MaterializedArtifact],
        encode: EncodeConfig,
        capacity: int = 3,
    ) -> None:
        self._root = root
        self._records = records
        self._encode = encode
        self._cache = _BoundedCache(self._read, capacity)

    def _read(self, chunk: HeightChunkId) -> np.ndarray:
        try:
            record = self._records[chunk]
        except KeyError as error:
            raise ValueError(f"transaction lacks decoded child {chunk}") from error
        path = self._root / record.relative_path
        if path.stat().st_size != record.bytes or _sha256_file(path) != record.container_sha256:
            raise ValueError(f"transaction child integrity mismatch: {path}")
        meta, payload = read_chunk_v2(path)
        if (meta.layer, meta.lod, meta.cx, meta.cz) != (
            "height",
            chunk.lod,
            chunk.cx,
            chunk.cz,
        ):
            raise ValueError(f"transaction child header mismatch: {path}")
        values = decode_quant16(self._encode, payload, meta.res, meta.qoffset, meta.qscale)
        digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
        if digest != record.decoded_values_sha256:
            raise ValueError(f"transaction decoded child hash mismatch: {path}")
        return values

    def load(self, chunk: HeightChunkId) -> np.ndarray:
        return self._cache.get(chunk)


def _stage_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != _sha256_file(source):
            raise ValueError(f"immutable staged artifact differs: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    os.link(source, temporary)
    temporary.replace(destination)


def materialize_structural_hierarchy(
    *,
    grid: GridConfig,
    encode: EncodeConfig,
    plan: StructuralRepairPlan,
    recipe_sha256: str,
    authority_manifest_sha256: str,
    canonical_base_manifest_sha256: str,
    load_pair: PairLoader,
    load_canonical_baseline: CanonicalLoader,
    reevaluate_fine_surface: FineReevaluation,
    ownership_closure: dict,
    output_root: Path,
    layout: HierarchyLayout = HierarchyLayout(),
    pair_cache_tiles: int = 12,
    canonical_cache_tiles: int = 12,
) -> StructuralMaterialization:
    """Materialize full reducer closure while retaining a review-only overlay.

    Peak live data is one fine reconstruction, one checked decoder cache of at
    most three chunks, and two small authority LRUs.  No collection stores
    decoded rasters for the complete 777/45/two hierarchy.
    """
    recipe_sha = _validated_sha256(recipe_sha256, "recipe_sha256")
    authority_sha = _validated_sha256(
        authority_manifest_sha256, "authority_manifest_sha256"
    )
    base_sha = _validated_sha256(
        canonical_base_manifest_sha256, "canonical_base_manifest_sha256"
    )
    expected_ownership_keys = {
        "version",
        "authorityManifestSha256",
        "campaignContentSha256",
        "campaignQualificationSha256",
        "scientificClosureSha256",
        "rasterizer",
    }
    if (
        not isinstance(ownership_closure, dict)
        or set(ownership_closure) != expected_ownership_keys
        or ownership_closure.get("authorityManifestSha256") != authority_sha
        or not isinstance(ownership_closure.get("rasterizer"), dict)
        or set(ownership_closure["rasterizer"]) != {"id", "sourceSha256"}
    ):
        raise ValueError("ownership closure identity is incomplete or names another authority")
    for name in (
        "campaignContentSha256",
        "campaignQualificationSha256",
        "scientificClosureSha256",
    ):
        _validated_sha256(ownership_closure.get(name), f"ownership closure {name}")
    _validated_sha256(
        ownership_closure["rasterizer"].get("sourceSha256"),
        "ownership closure rasterizer sourceSha256",
    )
    ownership_closure_sha = hashlib.sha256(
        json.dumps(
            ownership_closure, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
    ).hexdigest()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    pair_cache = _BoundedCache(load_pair, pair_cache_tiles)
    canonical_cache = _BoundedCache(load_canonical_baseline, canonical_cache_tiles)
    validated_inputs: set[HeightChunkId] = set()

    def checked_inputs(
        chunk: HeightChunkId,
    ) -> tuple[StructuralAuthorityPair, CanonicalBaselineInput]:
        pair = pair_cache.get(chunk)
        baseline = canonical_cache.get(chunk)
        if pair.chunk != chunk or baseline.chunk != chunk:
            raise ValueError(f"authority loader returned the wrong key for {chunk}")
        if pair.authority_manifest_sha256 != authority_sha:
            raise ValueError("authority loader is not bound to the declared manifest")
        if baseline.base_manifest_sha256 != base_sha:
            raise ValueError("canonical baseline is not bound to the declared base")
        if chunk not in validated_inputs:
            expected = (layout.authority_core_res,) * 2
            if (
                pair.structural.height.shape != expected
                or baseline.tile.height.shape != expected
            ):
                raise ValueError(f"authority tile {chunk} has the wrong resolution")
            if not pair.structural.valid.all() or not baseline.tile.valid.all():
                raise ValueError(f"authority tile {chunk} contains unresolved samples")
            if (
                not np.array_equal(pair.paired_baseline.height, baseline.tile.height)
                or not np.array_equal(pair.paired_baseline.valid, baseline.tile.valid)
            ):
                raise ValueError(f"paired and canonical baseline differ for {chunk}")
            validated_inputs.add(chunk)
        return pair, baseline

    domain_minimum = np.inf
    for chunk in plan.lod2_reducer:
        pair, baseline = checked_inputs(chunk)
        composed = np.where(
            pair.structural.unknown_bathymetry,
            pair.structural.height,
            baseline.tile.height,
        )
        domain_minimum = min(domain_minimum, float(np.min(composed)))
    fine_qoffset = float(np.floor(domain_minimum - 2.0))

    fine_records: dict[HeightChunkId, MaterializedArtifact] = {}
    overlap_ledger = _WorldOverlapLedger(plan.lod2_reducer)
    for chunk in plan.lod2_reducer:
        halo = tuple(
            HeightChunkId(chunk.lod, chunk.cx + dx, chunk.cz + dz)
            for dz in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
        loaded = {
            dependency: checked_inputs(dependency) for dependency in halo
        }
        pairs = {dependency: loaded[dependency][0] for dependency in halo}
        baselines = {dependency: loaded[dependency][1] for dependency in halo}
        window = assemble_fine_surface_window(
            chunk,
            load_structural_height=lambda dependency: pairs[
                dependency
            ].structural.height,
            load_baseline_height=lambda dependency: baselines[
                dependency
            ].tile.height,
            layout=layout,
        )
        result = reevaluate_fine_surface(window)
        if not isinstance(result, FineReevaluationResult):
            raise TypeError("fine reevaluation must return FineReevaluationResult")
        expected_fine = (layout.fine_core_res + 1,) * 2
        if result.height.shape != expected_fine:
            raise ValueError(f"fine reevaluation returned the wrong shape for {chunk}")
        baseline_core = window.baseline_height[window.core_slice]
        if not np.array_equal(
            result.height[~result.authority], baseline_core[~result.authority]
        ):
            raise ValueError(f"fine reevaluation changed unowned canonical baseline: {chunk}")
        overlap_ledger.observe(chunk, result)
        dependencies = tuple(
            (
                dependency,
                _combined_fine_dependency(pairs[dependency], baselines[dependency]),
            )
            for dependency in sorted(halo)
        )
        packed = _pack_height(
            grid=grid,
            encode=encode,
            chunk=chunk,
            values=result.height,
            qscale=FINE_QSCALE_M,
            qoffset=fine_qoffset,
            dependencies=dependencies,
        )
        ownership = _write_fine_ownership(
            root,
            chunk,
            result.authority,
            result.abstained,
            ownership_closure_sha,
        )
        fine_records[chunk] = _write_packed(
            root,
            packed,
            recipe_sha256=recipe_sha,
            role="transient_fine",
            ownership=ownership,
        )
        del window, baseline_core, result, packed, ownership, pairs, baselines, loaded

    overlap_ledger.finish()

    parent_records: dict[HeightChunkId, MaterializedArtifact] = {}
    fine_decoder = _DecodedArtifactCache(
        root=root, records=fine_records, encode=encode, capacity=3
    )
    for chunk in plan.lod1_reducer:
        reduced = reduce_decoded_children(
            chunk, fine_decoder.load, core_res=layout.fine_core_res
        )
        dependencies = tuple(
            (dependency, fine_records[dependency].decoded_values_sha256)
            for dependency in reduced.dependencies
        )
        packed = _pack_height(
            grid=grid,
            encode=encode,
            chunk=chunk,
            values=reduced.values,
            qscale=0.005,
            qoffset=None,
            dependencies=dependencies,
        )
        parent_records[chunk] = _write_packed(
            root, packed, recipe_sha256=recipe_sha, role="transient_lod1"
        )
        del reduced, packed

    lod0_records: dict[HeightChunkId, MaterializedArtifact] = {}
    parent_decoder = _DecodedArtifactCache(
        root=root, records=parent_records, encode=encode, capacity=3
    )
    for chunk in plan.corrected_lod0:
        reduced = reduce_decoded_children(
            chunk, parent_decoder.load, core_res=layout.fine_core_res
        )
        dependencies = tuple(
            (dependency, parent_records[dependency].decoded_values_sha256)
            for dependency in reduced.dependencies
        )
        packed = _pack_height(
            grid=grid,
            encode=encode,
            chunk=chunk,
            values=reduced.values,
            qscale=encode.height_qscale_for(0),
            qoffset=None,
            dependencies=dependencies,
        )
        lod0_records[chunk] = _write_packed(
            root, packed, recipe_sha256=recipe_sha, role="corrected_lod0_core_candidate"
        )
        del reduced, packed

    published_keys = (*plan.published_lod2, plan.review_parent)
    all_records = {**fine_records, **parent_records, **lod0_records}
    published = tuple(all_records[chunk] for chunk in published_keys)
    for record in published:
        source = root / record.relative_path
        destination = (
            root
            / "overlay"
            / "height"
            / str(record.chunk.lod)
            / source.name
        )
        _stage_link(source, destination)
        _stage_link(
            root / record.sidecar_relative_path,
            destination.with_suffix(".json"),
        )
    for record in lod0_records.values():
        source = root / record.relative_path
        destination = root / "corrected-base" / "height" / "0" / source.name
        _stage_link(source, destination)
        _stage_link(
            root / record.sidecar_relative_path,
            destination.with_suffix(".json"),
        )

    def rows(records: tuple[MaterializedArtifact, ...]) -> list[dict]:
        return [
            {
                "key": _key(record.chunk),
                "path": record.relative_path,
                "sidecar": record.sidecar_relative_path,
                "bytes": record.bytes,
                "containerSha256": record.container_sha256,
                "decodedValuesSha256": record.decoded_values_sha256,
                "dependencyMerkleRoot": record.dependency_merkle_root,
                "ownership": (
                    None
                    if record.ownership is None
                    else {
                        "format": "TOM1/1",
                        "path": record.ownership.relative_path,
                        "bytes": record.ownership.bytes,
                        "containerSha256": record.ownership.container_sha256,
                        "authoritySha256": record.ownership.authority_sha256,
                        "abstentionSha256": record.ownership.abstention_sha256,
                        "authoritySamples": record.ownership.authority_samples,
                        "abstainedSamples": record.ownership.abstained_samples,
                        "closureSha256": record.ownership.closure_sha256,
                    }
                ),
            }
            for record in records
        ]

    fine = tuple(fine_records[chunk] for chunk in plan.lod2_reducer)
    parents = tuple(parent_records[chunk] for chunk in plan.lod1_reducer)
    corrected = tuple(lod0_records[chunk] for chunk in plan.corrected_lod0)
    transaction = {
        "format": 1,
        "materializerVersion": MATERIALIZER_VERSION,
        "recipeSha256": recipe_sha,
        "authorityManifestSha256": authority_sha,
        "canonicalBaseManifestSha256": base_sha,
        "ownershipClosure": ownership_closure,
        "fineSharedQoffset": fine_qoffset,
        "worldOverlap": {
            "version": "exact-height-authority-abstention-edge-sha256/1",
            "matchedEdges": overlap_ledger.matched,
        },
        "boundedMemory": {
            "pairCacheTiles": pair_cache_tiles,
            "canonicalCacheTiles": canonical_cache_tiles,
            "decodedChunkCache": 3,
            "completeHierarchyHeldInMemory": False,
        },
        "fine": rows(fine),
        "parents": rows(parents),
        "correctedLod0": rows(corrected),
        "publishedOverlay": [_key(record.chunk) for record in published],
    }
    manifest_path = root / "transaction.json"
    _write_json_atomic(manifest_path, transaction)
    manifest_sha = _sha256_file(manifest_path)
    return StructuralMaterialization(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        fine=fine,
        parents=parents,
        corrected_lod0=corrected,
        published_overlay=published,
    )

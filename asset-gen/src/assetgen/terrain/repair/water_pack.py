"""Checked LAC1 packing for corrected structural water surfaces."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import FMT, LAYER_IDS, MAGIC, ChunkMeta, write_chunk
from ...cook.encode import decode_quant16, encode_quantized16
from ...grid import ChunkId, chunk_bounds_en
from .water_layer import CorrectedWaterLayer, reduce_water_lod1


_TOMBSTONE_DOMAIN = b"laas.structural-water.absent.v1\0"
_DEPENDENCY_DOMAIN = b"laas.structural-water.dependencies.v1\0"


def _validated_sha256(value: str, name: str) -> str:
    try:
        if value != value.lower() or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.array(values, dtype=np.dtype(dtype), copy=True)
    if np.issubdtype(array.dtype, np.floating):
        array[np.isnan(array)] = np.nan
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _dependency_sha256(dependencies: tuple[tuple[ChunkId, str], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_DEPENDENCY_DOMAIN)
    for chunk, artifact_sha256 in dependencies:
        digest.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
        digest.update(bytes.fromhex(artifact_sha256))
    return digest.hexdigest()


def _ordered_dependencies(
    dependencies: tuple[tuple[ChunkId, str], ...],
) -> tuple[tuple[ChunkId, str], ...]:
    ordered = tuple(sorted(dependencies, key=lambda item: (item[0].lod, item[0].cz, item[0].cx)))
    if len({chunk for chunk, _ in ordered}) != len(ordered):
        raise ValueError("water dependencies contain duplicate chunk keys")
    for _, artifact_sha256 in ordered:
        _validated_sha256(artifact_sha256, "dependency artifact_sha256")
    return ordered


def _container_bytes(meta: ChunkMeta, payload: bytes) -> bytes:
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
    return header + payload


@dataclass(frozen=True)
class DecodedWaterDependency:
    """One decoded LOD0 water artifact, including an explicit absent identity."""

    water_y: np.ndarray | None
    artifact_sha256: str

    def __post_init__(self) -> None:
        _validated_sha256(self.artifact_sha256, "artifact_sha256")
        if self.water_y is None:
            return
        water = np.array(self.water_y, dtype=np.float32, copy=True)
        if water.ndim != 2 or np.isinf(water).any():
            raise ValueError("decoded water dependency must be 2D finite-or-NaN float32")
        water.flags.writeable = False
        object.__setattr__(self, "water_y", water)


@dataclass(frozen=True)
class PackedWaterChunk:
    chunk: ChunkId
    meta: ChunkMeta
    payload: bytes
    artifact_sha256: str
    payload_sha256: str
    source_values_sha256: str
    decoded_values_sha256: str
    evidence_sha256: str
    dependencies: tuple[tuple[ChunkId, str], ...]
    dependency_sha256: str
    wet_samples: int
    dry_samples: int
    max_roundtrip_error_m: float
    roundtrip_limit_m: float
    source_water_y: np.ndarray

    def __post_init__(self) -> None:
        source = np.array(self.source_water_y, dtype=np.float64, copy=True)
        if source.ndim != 2 or np.isinf(source).any():
            raise ValueError("packed water source must be 2D finite-or-NaN float64")
        source.flags.writeable = False
        object.__setattr__(self, "source_water_y", source)

    def write_lac1(self, path: Path) -> None:
        write_chunk(path, self.meta, self.payload)


@dataclass(frozen=True)
class AbsentWaterChunk:
    chunk: ChunkId
    tombstone_sha256: str
    source_values_sha256: str
    evidence_sha256: str
    dependencies: tuple[tuple[ChunkId, str], ...]
    dependency_sha256: str
    dry_samples: int
    reason: str = "all_dry"
    source_water_y: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.source_water_y is None:
            raise ValueError("absent water must retain its checked all-dry source")
        source = np.array(self.source_water_y, dtype=np.float64, copy=True)
        if source.ndim != 2 or np.isfinite(source).any() or np.isinf(source).any():
            raise ValueError("absent water source must be an all-NaN 2D float64 array")
        source.flags.writeable = False
        object.__setattr__(self, "source_water_y", source)


WaterPackResult = PackedWaterChunk | AbsentWaterChunk
DecodedWaterDependencyLoader = Callable[[ChunkId], DecodedWaterDependency]


def _checked_water_codes(
    water_y: np.ndarray,
    *,
    requested_qscale: float,
    inherited_qoffset: float | None,
    inherited_qscale: float | None,
) -> tuple[np.ndarray, float, float]:
    wire_scale = float(np.float32(requested_qscale))
    if not np.isfinite(wire_scale) or wire_scale <= 0.0:
        raise ValueError("water qscale must be positive and finite")
    if inherited_qscale is not None and float(np.float32(inherited_qscale)) != wire_scale:
        raise ValueError("inherited water qscale differs from the configured wire qscale")
    wet = np.isfinite(water_y)
    if not np.any(wet):
        raise ValueError("all-dry water must be represented as absent")
    if inherited_qoffset is None:
        qoffset = float(np.floor(np.min(water_y[wet]) - 1.0))
    else:
        qoffset = float(inherited_qoffset)
    if not np.isfinite(qoffset) or float(np.float32(qoffset)) != qoffset:
        raise ValueError("water qoffset must be finite and exactly representable as float32")

    wet_codes = np.rint((water_y[wet] - qoffset) / wire_scale)
    if wet_codes.min() < 1.0 or wet_codes.max() > 65535.0:
        raise OverflowError("wet water samples do not fit reserved-zero u16 quantization")
    codes = np.zeros(water_y.shape, dtype=np.uint16)
    codes[wet] = wet_codes.astype(np.uint16)
    if np.any(codes[wet] == 0) or np.any(codes[~wet] != 0):
        raise AssertionError("reserved q=0 water semantics were violated")
    return codes, qoffset, wire_scale


def _pack_water_values(
    *,
    water_y: np.ndarray,
    chunk: ChunkId,
    grid: GridConfig,
    encode: EncodeConfig,
    evidence_sha256: str,
    dependencies: tuple[tuple[ChunkId, str], ...],
    inherited_qoffset: float | None,
    inherited_qscale: float | None,
) -> WaterPackResult:
    values = np.asarray(water_y, dtype=np.float64)
    expected_res = grid.chunk_m // 2 + 1
    if values.shape != (expected_res, expected_res) or np.isinf(values).any():
        raise ValueError(f"water {chunk} must be finite-or-NaN {(expected_res, expected_res)}")
    evidence_sha256 = _validated_sha256(evidence_sha256, "evidence_sha256")
    ordered = _ordered_dependencies(dependencies)
    dependency_sha = _dependency_sha256(ordered)
    source_sha = _array_sha256(values, "<f8")
    wet = np.isfinite(values)
    if not np.any(wet):
        identity = hashlib.sha256()
        identity.update(_TOMBSTONE_DOMAIN)
        identity.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
        identity.update(bytes.fromhex(source_sha))
        identity.update(bytes.fromhex(evidence_sha256))
        identity.update(bytes.fromhex(dependency_sha))
        return AbsentWaterChunk(
            chunk=chunk,
            tombstone_sha256=identity.hexdigest(),
            source_values_sha256=source_sha,
            evidence_sha256=evidence_sha256,
            dependencies=ordered,
            dependency_sha256=dependency_sha,
            dry_samples=values.size,
            source_water_y=values,
        )

    codes, qoffset, qscale = _checked_water_codes(
        values,
        requested_qscale=encode.water_qscale,
        inherited_qoffset=inherited_qoffset,
        inherited_qscale=inherited_qscale,
    )
    payload = encode_quantized16(encode, codes)
    decoded = decode_quant16(encode, payload, values.shape[0], qoffset, qscale)
    decoded = decoded.astype(np.float32, copy=False)
    decoded[codes == 0] = np.nan
    error = float(
        np.max(np.abs(decoded[wet].astype(np.float64) - values[wet]), initial=0.0)
    )
    peak = np.float32(np.max(np.abs(decoded[wet]), initial=0.0))
    limit = qscale * 0.5 + 2.0 * abs(float(np.spacing(peak)))
    if error > limit:
        raise AssertionError(f"checked water quantization error {error} exceeds {limit}")
    if not np.array_equal(np.isnan(decoded), ~wet):
        raise AssertionError("decoded reserved-zero water changed wet/dry semantics")

    e_min, _, _, n_max = chunk_bounds_en(grid, chunk)
    meta = ChunkMeta(
        layer="water",
        lod=chunk.lod,
        enc=1,
        cx=chunk.cx,
        cz=chunk.cz,
        res=values.shape[0],
        count=0,
        origin_e=e_min,
        origin_n=n_max,
        qoffset=qoffset,
        qscale=qscale,
    )
    container = _container_bytes(meta, payload)
    return PackedWaterChunk(
        chunk=chunk,
        meta=meta,
        payload=payload,
        artifact_sha256=hashlib.sha256(container).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_values_sha256=source_sha,
        decoded_values_sha256=_array_sha256(decoded, "<f4"),
        evidence_sha256=evidence_sha256,
        dependencies=ordered,
        dependency_sha256=dependency_sha,
        wet_samples=int(np.count_nonzero(wet)),
        dry_samples=int(np.count_nonzero(~wet)),
        max_roundtrip_error_m=error,
        roundtrip_limit_m=limit,
        source_water_y=values,
    )


def _corrected_evidence_sha256(layer: CorrectedWaterLayer) -> str:
    evidence = layer.evidence
    expected = {
        "output_sha256": _array_sha256(layer.water_y, "<f8"),
        "authority_sha256": _array_sha256(layer.authority, "u1"),
        "abstention_sha256": _array_sha256(layer.abstained, "u1"),
    }
    for field, digest in expected.items():
        if getattr(evidence, field) != digest:
            raise ValueError(f"corrected water evidence {field} does not match its arrays")
    counts = {
        "dry_samples": int(np.count_nonzero(~np.isfinite(layer.water_y))),
        "baseline_wet_samples": int(np.count_nonzero(np.isfinite(layer.water_y))),
        "mapped_wet_samples": int(np.count_nonzero(layer.authority | layer.abstained)),
        "authority_samples": int(np.count_nonzero(layer.authority)),
        "abstained_mapped_samples": int(np.count_nonzero(layer.abstained)),
    }
    for field, count in counts.items():
        if getattr(evidence, field) != count:
            raise ValueError(f"corrected water evidence {field} does not match its arrays")
    payload = json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def pack_corrected_water_lod0(
    layer: CorrectedWaterLayer,
    *,
    chunk: ChunkId,
    grid: GridConfig,
    encode: EncodeConfig,
    inherited_artifact_sha256: str,
    inherited_qoffset: float | None,
    inherited_qscale: float | None,
) -> WaterPackResult:
    """Pack one corrected 2 m LOD0 layer against its inherited water identity."""
    if not isinstance(layer, CorrectedWaterLayer) or chunk.lod != 0:
        raise ValueError("corrected water packing requires one LOD0 layer")
    evidence_sha = _corrected_evidence_sha256(layer)
    if np.isfinite(layer.water_y).any() and (
        inherited_qoffset is None or inherited_qscale is None
    ):
        raise ValueError("wet corrected LOD0 water requires its inherited quantizer")
    return _pack_water_values(
        water_y=layer.water_y,
        chunk=chunk,
        grid=grid,
        encode=encode,
        evidence_sha256=evidence_sha,
        dependencies=((chunk, inherited_artifact_sha256),),
        inherited_qoffset=inherited_qoffset,
        inherited_qscale=inherited_qscale,
    )


def pack_reduced_water_lod1(
    *,
    parent: ChunkId,
    grid: GridConfig,
    encode: EncodeConfig,
    load_dependency: DecodedWaterDependencyLoader,
    evidence_sha256: str,
    inherited_qoffset: float | None = None,
    inherited_qscale: float | None = None,
) -> WaterPackResult:
    """Reduce exact decoded LOD0 dependencies and pack their ordinary LOD1 parent."""
    if parent.lod != 1:
        raise ValueError("reduced structural water parent must be LOD1")
    core_res = grid.chunk_m // 2
    loaded: dict[ChunkId, DecodedWaterDependency] = {}

    def decoded(chunk: ChunkId) -> np.ndarray | None:
        dependency = load_dependency(chunk)
        if not isinstance(dependency, DecodedWaterDependency):
            raise TypeError("water dependency loader returned an invalid record")
        loaded[chunk] = dependency
        return dependency.water_y

    reduction = reduce_water_lod1(parent, decoded, core_res=core_res)
    expected = tuple(
        ChunkId(parent.cx * 4 + dx, parent.cz * 4 + dz, 0)
        for dz in range(5)
        for dx in range(5)
    )
    if tuple(loaded) != expected:
        raise AssertionError("water reducer did not inspect its exact fixed dependency closure")
    if reduction is not None:
        present = tuple(chunk for chunk in expected if loaded[chunk].water_y is not None)
        if reduction.dependencies != expected or reduction.present_dependencies != present:
            raise AssertionError("water reducer reported inconsistent child dependencies")
    dependencies = tuple((chunk, loaded[chunk].artifact_sha256) for chunk in expected)
    water_y = (
        np.full((core_res + 1, core_res + 1), np.nan, dtype=np.float32)
        if reduction is None
        else reduction.water_y
    )
    return _pack_water_values(
        water_y=water_y,
        chunk=parent,
        grid=grid,
        encode=encode,
        evidence_sha256=evidence_sha256,
        dependencies=dependencies,
        inherited_qoffset=inherited_qoffset,
        inherited_qscale=inherited_qscale,
    )

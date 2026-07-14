"""Decoded structural-height hierarchy construction.

Only structural repair is handled here.  Fine morphology synthesis belongs to a
later terrain stage and must not be smuggled into hierarchy materialization.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import ChunkMeta, write_chunk_v2
from ...cook.encode import decode_quant16, encode_quant16_checked
from ...cook.micro_hierarchy import box_mean_fixed, dependency_merkle_root
from ...height_geom import (
    UNITS_PER_METER,
    HeightChunkId,
    chunk_origin_en_units,
)
from .baseline import BaselineTile
from .model import StructuralTile
from .plan import StructuralRepairPlan
from .prolong import prolong_structural_4x

FINE_QSCALE_M = 0.002
FINE_SURFACE_HALO_SAMPLES = 16
FINE_SURFACE_WINDOW_VERSION = "real-expanded-structural-baseline-halo-16/1"


@dataclass(frozen=True)
class HierarchyLayout:
    """Resolution contract; the small variants are useful for focused tests."""

    authority_core_res: int = 512
    factor: int = 4

    def __post_init__(self) -> None:
        if self.authority_core_res < 4 or self.factor != 4:
            raise ValueError("structural hierarchy requires core_res >= 4 and factor 4")

    @property
    def fine_core_res(self) -> int:
        return self.authority_core_res * self.factor


@dataclass(frozen=True)
class AuthorityTileInput:
    """One integrity-bound, decoded 0.25 m structural authority artifact."""

    chunk: HeightChunkId
    tile: StructuralTile
    artifact_sha256: str

    def __post_init__(self) -> None:
        _validated_sha256(self.artifact_sha256, "artifact_sha256")


@dataclass(frozen=True)
class BaselineTileInput:
    """One integrity-bound, unmodified 0.25 m baseline authority artifact."""

    chunk: HeightChunkId
    tile: BaselineTile
    artifact_sha256: str

    def __post_init__(self) -> None:
        _validated_sha256(self.artifact_sha256, "artifact_sha256")


@dataclass(frozen=True)
class FineSurfaceWindow:
    """Real fine structural/baseline samples with a one-metre halo.

    Production windows are 2081 by 2081: the 2049-sample packed core is
    ``core_slice == [16:2065, 16:2065]``. Small hierarchy fixtures retain the
    same 16-sample physical contract around their reduced core.
    """

    chunk: HeightChunkId
    structural_height: np.ndarray
    baseline_height: np.ndarray
    halo_samples: int = FINE_SURFACE_HALO_SAMPLES

    def __post_init__(self) -> None:
        structural = np.asarray(self.structural_height, dtype=np.float64)
        baseline = np.asarray(self.baseline_height, dtype=np.float64)
        if self.chunk.lod != -2:
            raise ValueError("fine surface windows require LOD -2 chunks")
        if self.halo_samples != FINE_SURFACE_HALO_SAMPLES:
            raise ValueError("fine surface windows require the frozen 16-sample halo")
        if (
            structural.ndim != 2
            or baseline.shape != structural.shape
            or structural.shape[0] != structural.shape[1]
            or structural.shape[0] <= 2 * self.halo_samples
            or (structural.shape[0] - 2 * self.halo_samples - 1) % 4
            or not np.isfinite(structural).all()
            or not np.isfinite(baseline).all()
        ):
            raise ValueError(
                "fine surface window must be square finite arrays around a 4n+1 core"
            )
        structural.flags.writeable = False
        baseline.flags.writeable = False
        object.__setattr__(self, "structural_height", structural)
        object.__setattr__(self, "baseline_height", baseline)

    @property
    def core_side(self) -> int:
        return self.structural_height.shape[0] - 2 * self.halo_samples

    @property
    def core_slice(self) -> tuple[slice, slice]:
        stop = self.halo_samples + self.core_side
        core = slice(self.halo_samples, stop)
        return core, core


# Temporary import compatibility while all callers move to the explicit window name.
FineReevaluationInput = FineSurfaceWindow


@dataclass(frozen=True)
class FineReevaluationResult:
    """Exact fine ownership, matching corrected-water authority semantics."""

    height: np.ndarray
    authority: np.ndarray
    abstained: np.ndarray

    def __post_init__(self) -> None:
        height = np.asarray(self.height, dtype=np.float64)
        authority = np.asarray(self.authority, dtype=np.bool_)
        abstained = np.asarray(self.abstained, dtype=np.bool_)
        if height.ndim != 2 or authority.shape != height.shape or abstained.shape != height.shape:
            raise ValueError("fine reevaluation result arrays must have matching 2D shapes")
        if not np.isfinite(height).all() or np.any(authority & abstained):
            raise ValueError("fine reevaluation must be finite with disjoint ownership masks")
        for array in (height, authority, abstained):
            array.flags.writeable = False
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "abstained", abstained)


@dataclass(frozen=True)
class PackedHeightChunk:
    """Checked height payload and the exact LAC2 header state used to write it."""

    chunk: HeightChunkId
    meta: ChunkMeta
    payload: bytes
    payload_sha256: str
    source_values_sha256: str
    decoded_values_sha256: str
    max_roundtrip_error_m: float
    roundtrip_limit_m: float
    dependencies: tuple[tuple[HeightChunkId, str], ...]
    dependency_merkle_root: str

    def write_lac2(self, path: Path) -> None:
        write_chunk_v2(path, self.meta, self.payload)


@dataclass(frozen=True)
class ReductionResult:
    """Fixed-order decoded-child reduction plus its complete dependency set."""

    values: np.ndarray
    dependencies: tuple[HeightChunkId, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("reduced height must be one finite 2D float64 array")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class StructuralHierarchyMetrics:
    fine_domain_authority_min_m: float
    fine_shared_qoffset: float
    max_fine_roundtrip_error_m: float
    max_decoded_seam_m: float
    parent_roundtrip_error_m: float
    fine_reevaluation_applied: bool
    authority_dependencies: tuple[tuple[HeightChunkId, str], ...]
    authority_dependency_merkle_root: str
    baseline_dependencies: tuple[tuple[HeightChunkId, str], ...]
    baseline_dependency_merkle_root: str
    fine_authority_samples: int
    fine_abstained_samples: int
    max_pre_reevaluation_unowned_delta_m: float
    max_post_reevaluation_unowned_delta_m: float
    parent_dependencies: tuple[tuple[HeightChunkId, str], ...]
    parent_dependency_merkle_root: str


@dataclass(frozen=True)
class StructuralHierarchy:
    fine: tuple[PackedHeightChunk, ...]
    parent: PackedHeightChunk
    metrics: StructuralHierarchyMetrics


AuthorityLoader = Callable[[HeightChunkId], AuthorityTileInput]
BaselineLoader = Callable[[HeightChunkId], BaselineTileInput]
FineReevaluation = Callable[[FineSurfaceWindow], FineReevaluationResult]
DecodedHeightLoader = Callable[[HeightChunkId], np.ndarray]


def _validated_sha256(value: str, name: str) -> str:
    try:
        if value != value.lower() or len(value) != 64 or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _decoded_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _mask_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.uint8)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _paired_dependency_sha256(structural_sha256: str, baseline_sha256: str) -> str:
    """Bind both artifacts under one chunk key for the existing Merkle primitive."""
    digest = hashlib.sha256()
    digest.update(b"laas.structural-baseline-pair.v1\0")
    digest.update(bytes.fromhex(_validated_sha256(structural_sha256, "structural_sha256")))
    digest.update(bytes.fromhex(_validated_sha256(baseline_sha256, "baseline_sha256")))
    return digest.hexdigest()


def _fine_closure(plan: StructuralRepairPlan) -> tuple[HeightChunkId, ...]:
    closure = tuple(sorted((*plan.published_lod2, *plan.review_lod2_support)))
    if len(closure) != 25 or len(set(closure)) != 25:
        raise ValueError("review fine closure must contain 16 children and nine apron chunks")
    return closure


class _AuthorityCache:
    def __init__(
        self,
        plan: StructuralRepairPlan,
        load: AuthorityLoader,
        layout: HierarchyLayout,
    ) -> None:
        self._allowed = set(plan.authority_support)
        self._load = load
        self._layout = layout
        self._cache: dict[HeightChunkId, AuthorityTileInput] = {}

    def get(self, chunk: HeightChunkId) -> AuthorityTileInput:
        if chunk not in self._allowed:
            raise ValueError(f"fine reconstruction halo leaves structural authority: {chunk}")
        if chunk not in self._cache:
            item = self._load(chunk)
            if not isinstance(item, AuthorityTileInput) or item.chunk != chunk:
                raise ValueError(f"authority loader returned the wrong artifact for {chunk}")
            expected = (self._layout.authority_core_res,) * 2
            if item.tile.height.shape != expected:
                raise ValueError(
                    f"authority {chunk} must be complete {expected}, got {item.tile.height.shape}"
                )
            if not item.tile.valid.all():
                raise ValueError(f"authority {chunk} contains unresolved structural samples")
            self._cache[chunk] = item
        return self._cache[chunk]

    def dependencies(self) -> tuple[tuple[HeightChunkId, str], ...]:
        return tuple(
            (chunk, item.artifact_sha256)
            for chunk, item in sorted(self._cache.items())
        )


class _BaselineCache:
    def __init__(
        self,
        plan: StructuralRepairPlan,
        load: BaselineLoader,
        layout: HierarchyLayout,
    ) -> None:
        self._allowed = set(plan.authority_support)
        self._load = load
        self._layout = layout
        self._cache: dict[HeightChunkId, BaselineTileInput] = {}

    def get(self, chunk: HeightChunkId) -> BaselineTileInput:
        if chunk not in self._allowed:
            raise ValueError(f"fine reconstruction halo leaves baseline authority: {chunk}")
        if chunk not in self._cache:
            item = self._load(chunk)
            if not isinstance(item, BaselineTileInput) or item.chunk != chunk:
                raise ValueError(f"baseline loader returned the wrong artifact for {chunk}")
            expected = (self._layout.authority_core_res,) * 2
            if item.tile.height.shape != expected:
                raise ValueError(
                    f"baseline {chunk} must be complete {expected}, got {item.tile.height.shape}"
                )
            if not item.tile.valid.all():
                raise ValueError(f"baseline {chunk} contains unresolved structural samples")
            self._cache[chunk] = item
        return self._cache[chunk]

    def dependencies(self) -> tuple[tuple[HeightChunkId, str], ...]:
        return tuple(
            (chunk, item.artifact_sha256)
            for chunk, item in sorted(self._cache.items())
        )


def _validate_authority_pair(
    chunk: HeightChunkId,
    authority: _AuthorityCache,
    baseline: _BaselineCache,
) -> None:
    structural = authority.get(chunk).tile
    original = baseline.get(chunk).tile
    unowned = ~structural.unknown_bathymetry
    if not np.array_equal(structural.height[unowned], original.height[unowned]):
        raise ValueError(f"structural authority changed unowned baseline samples in {chunk}")
    if not np.array_equal(structural.valid[unowned], original.valid[unowned]):
        raise ValueError(f"structural authority changed unowned baseline validity in {chunk}")


def _raster_support(
    chunk: HeightChunkId,
    load_height: Callable[[HeightChunkId], np.ndarray],
    layout: HierarchyLayout,
) -> np.ndarray:
    """Assemble real parent centers -6..core+6 for the fine halo."""
    core = layout.authority_core_res
    local = np.arange(-6, core + 7, dtype=np.int64)
    tile_delta = np.floor_divide(local, core)
    tile_index = np.mod(local, core)
    support = np.empty((core + 13, core + 13), dtype=np.float64)
    for dz in np.unique(tile_delta):
        row_positions = np.flatnonzero(tile_delta == dz)
        rows = tile_index[row_positions]
        for dx in np.unique(tile_delta):
            col_positions = np.flatnonzero(tile_delta == dx)
            cols = tile_index[col_positions]
            source = load_height(
                HeightChunkId(chunk.lod, chunk.cx + int(dx), chunk.cz + int(dz))
            )
            support[np.ix_(row_positions, col_positions)] = source[np.ix_(rows, cols)]
    return support


@dataclass(frozen=True)
class _FineReconstruction:
    values: np.ndarray
    authority_sha256: str
    abstention_sha256: str
    authority_samples: int
    abstained_samples: int
    pre_unowned_delta_m: float
    post_unowned_delta_m: float


def _prolong_support(support: np.ndarray, layout: HierarchyLayout) -> np.ndarray:
    """Prolong parent centers -4..core+4 and retain fine -16..core*4+16."""
    core = layout.authority_core_res
    prolonged = prolong_structural_4x(
        support,
        parent_rows=(2, 2 + core + 9),
        parent_cols=(2, 2 + core + 9),
    )
    side = layout.fine_core_res + 2 * FINE_SURFACE_HALO_SAMPLES + 1
    return prolonged[:side, :side]


def assemble_fine_surface_window(
    chunk: HeightChunkId,
    *,
    load_structural_height: Callable[[HeightChunkId], np.ndarray],
    load_baseline_height: Callable[[HeightChunkId], np.ndarray],
    layout: HierarchyLayout = HierarchyLayout(),
) -> FineSurfaceWindow:
    """Build the canonical fine window from real neighboring parent samples."""
    if chunk.lod != -2:
        raise ValueError("fine surface assembly requires one LOD -2 chunk")
    structural = _prolong_support(
        _raster_support(chunk, load_structural_height, layout), layout
    )
    baseline = _prolong_support(
        _raster_support(chunk, load_baseline_height, layout), layout
    )
    expected = layout.fine_core_res + 2 * FINE_SURFACE_HALO_SAMPLES + 1
    if structural.shape != (expected, expected) or baseline.shape != (expected, expected):
        raise AssertionError("fine surface assembly violated the expanded-window contract")
    return FineSurfaceWindow(chunk, structural, baseline)


def _reconstruct_fine(
    chunk: HeightChunkId,
    authority_cache: _AuthorityCache,
    baseline_cache: _BaselineCache,
    layout: HierarchyLayout,
    reevaluate: FineReevaluation,
) -> _FineReconstruction:
    halo = tuple(
        HeightChunkId(chunk.lod, chunk.cx + dx, chunk.cz + dz)
        for dz in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    for dependency in halo:
        _validate_authority_pair(dependency, authority_cache, baseline_cache)
    window = assemble_fine_surface_window(
        chunk,
        load_structural_height=lambda dependency: authority_cache.get(
            dependency
        ).tile.height,
        load_baseline_height=lambda dependency: baseline_cache.get(
            dependency
        ).tile.height,
        layout=layout,
    )
    result = reevaluate(window)
    if not isinstance(result, FineReevaluationResult):
        raise TypeError("fine reevaluation must return FineReevaluationResult")
    expected = (layout.fine_core_res + 1,) * 2
    if result.height.shape != expected:
        raise ValueError(f"fine surface {chunk} must be {expected}")
    baseline_core = window.baseline_height[window.core_slice]
    structural_core = window.structural_height[window.core_slice]
    unowned = ~result.authority
    if not np.array_equal(result.height[unowned], baseline_core[unowned]):
        raise ValueError(
            f"fine reevaluation changed dry or abstained baseline samples in {chunk}"
        )
    pre_delta = float(
        np.max(np.abs(structural_core[unowned] - baseline_core[unowned]), initial=0.0)
    )
    post_delta = float(
        np.max(np.abs(result.height[unowned] - baseline_core[unowned]), initial=0.0)
    )
    return _FineReconstruction(
        values=np.ascontiguousarray(result.height, dtype=np.float64),
        authority_sha256=_mask_array_sha256(result.authority),
        abstention_sha256=_mask_array_sha256(result.abstained),
        authority_samples=int(np.count_nonzero(result.authority)),
        abstained_samples=int(np.count_nonzero(result.abstained)),
        pre_unowned_delta_m=pre_delta,
        post_unowned_delta_m=post_delta,
    )


def _pack_height(
    *,
    grid: GridConfig,
    encode: EncodeConfig,
    chunk: HeightChunkId,
    values: np.ndarray,
    qscale: float,
    qoffset: float | None,
    dependencies: tuple[tuple[HeightChunkId, str], ...],
) -> PackedHeightChunk:
    payload, selected_offset, wire_scale = encode_quant16_checked(
        encode, values, qscale, qoffset
    )
    decoded = decode_quant16(
        encode, payload, values.shape[0], selected_offset, wire_scale
    )
    error = float(np.max(np.abs(decoded.astype(np.float64) - values), initial=0.0))
    peak = np.float32(np.max(np.abs(decoded), initial=0.0))
    allowance = 2.0 * abs(float(np.spacing(peak)))
    limit = wire_scale * 0.5 + allowance
    if error > limit:
        raise AssertionError(f"checked quantization error {error} exceeds {limit} on {chunk}")
    origin_e, origin_n = chunk_origin_en_units(grid, chunk)
    meta = ChunkMeta(
        layer="height",
        lod=chunk.lod,
        enc=1,
        cx=chunk.cx,
        cz=chunk.cz,
        res=values.shape[0],
        count=0,
        origin_e=origin_e / UNITS_PER_METER,
        origin_n=origin_n / UNITS_PER_METER,
        qoffset=selected_offset,
        qscale=wire_scale,
    )
    ordered_dependencies = tuple(sorted(dependencies))
    return PackedHeightChunk(
        chunk=chunk,
        meta=meta,
        payload=payload,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_values_sha256=_array_sha256(values),
        decoded_values_sha256=_decoded_array_sha256(decoded),
        max_roundtrip_error_m=error,
        roundtrip_limit_m=limit,
        dependencies=ordered_dependencies,
        dependency_merkle_root=dependency_merkle_root(ordered_dependencies),
    )


def reduce_decoded_children(
    parent: HeightChunkId,
    load_decoded: DecodedHeightLoader,
    *,
    core_res: int = 2048,
) -> ReductionResult:
    """Derive one 2049-sample parent from its decoded 4x4+apron closure.

    The callback may mix newly replaced chunks with inherited base chunks.  All
    inputs are required to be browser-equivalent float32 decodes.
    """
    if core_res < 4 or core_res % 4:
        raise ValueError("child core_res must be positive and divisible by four")
    child_lod = parent.lod - 1
    cx0, cz0 = parent.cx * 4, parent.cz * 4
    required = tuple(
        HeightChunkId(child_lod, cx0 + dx, cz0 + dz)
        for dz in range(5)
        for dx in range(5)
    )

    def child(chunk: HeightChunkId) -> np.ndarray:
        values = np.asarray(load_decoded(chunk))
        expected = (core_res + 1,) * 2
        if values.dtype != np.float32 or values.shape != expected:
            raise ValueError(f"decoded {chunk} must be float32 {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"decoded {chunk} contains nonfinite height")
        return values

    reduced_core = core_res // 4
    output = np.empty((core_res + 1, core_res + 1), dtype=np.float64)
    for dz in range(4):
        for dx in range(4):
            values = child(HeightChunkId(child_lod, cx0 + dx, cz0 + dz))
            output[
                dz * reduced_core : (dz + 1) * reduced_core,
                dx * reduced_core : (dx + 1) * reduced_core,
            ] = box_mean_fixed(values[:core_res, :core_res])
    for dz in range(4):
        values = child(HeightChunkId(child_lod, cx0 + 4, cz0 + dz))
        output[dz * reduced_core : (dz + 1) * reduced_core, -1:] = box_mean_fixed(
            values[:core_res, :4]
        )
    for dx in range(4):
        values = child(HeightChunkId(child_lod, cx0 + dx, cz0 + 4))
        output[-1:, dx * reduced_core : (dx + 1) * reduced_core] = box_mean_fixed(
            values[:4, :core_res]
        )
    output[-1, -1] = box_mean_fixed(
        child(HeightChunkId(child_lod, cx0 + 4, cz0 + 4))[:4, :4]
    )[0, 0]
    return ReductionResult(values=output, dependencies=required)


def _decode_packed(encode: EncodeConfig, packed: PackedHeightChunk) -> np.ndarray:
    return decode_quant16(
        encode,
        packed.payload,
        packed.meta.res,
        packed.meta.qoffset,
        packed.meta.qscale,
    )


def _max_seam(
    encode: EncodeConfig,
    chunks: dict[HeightChunkId, PackedHeightChunk],
    parent: HeightChunkId,
) -> float:
    cx0, cz0 = parent.cx * 4, parent.cz * 4
    maximum = 0.0
    for dz in range(5):
        previous = _decode_packed(encode, chunks[HeightChunkId(-2, cx0, cz0 + dz)])
        for dx in range(1, 5):
            current = _decode_packed(encode, chunks[HeightChunkId(-2, cx0 + dx, cz0 + dz)])
            maximum = max(maximum, float(np.max(np.abs(previous[:, -1] - current[:, 0]))))
            previous = current
    for dx in range(5):
        previous = _decode_packed(encode, chunks[HeightChunkId(-2, cx0 + dx, cz0)])
        for dz in range(1, 5):
            current = _decode_packed(encode, chunks[HeightChunkId(-2, cx0 + dx, cz0 + dz)])
            maximum = max(maximum, float(np.max(np.abs(previous[-1, :] - current[0, :]))))
            previous = current
    return maximum


def build_structural_hierarchy(
    *,
    grid: GridConfig,
    encode: EncodeConfig,
    plan: StructuralRepairPlan,
    load_authority: AuthorityLoader,
    load_baseline: BaselineLoader,
    reevaluate_fine_surface: FineReevaluation,
    layout: HierarchyLayout = HierarchyLayout(),
    parent_qscale_m: float = 0.005,
) -> StructuralHierarchy:
    """Reconstruct and pack the review 5x5 fine closure and decoded parent.

    The composed structural authority and its unmodified baseline are prolonged
    independently. ``reevaluate_fine_surface`` must classify exact fine-grid
    authority/abstention and restore every unowned sample from that baseline.
    This builder never infers ownership from an interpolated coarse mask.
    """
    fine_closure = _fine_closure(plan)
    authority_cache = _AuthorityCache(plan, load_authority, layout)
    baseline_cache = _BaselineCache(plan, load_baseline, layout)
    domain_authority_min = min(
        float(np.min(authority_cache.get(chunk).tile.height)) for chunk in fine_closure
    )
    shared_offset = float(np.floor(domain_authority_min - 2.0))
    first_pass: dict[HeightChunkId, tuple] = {}
    fine_minimum = np.inf
    fine_maximum = -np.inf
    for chunk in fine_closure:
        reconstruction = _reconstruct_fine(
            chunk,
            authority_cache,
            baseline_cache,
            layout,
            reevaluate_fine_surface,
        )
        first_pass[chunk] = (
            _array_sha256(reconstruction.values),
            reconstruction.authority_sha256,
            reconstruction.abstention_sha256,
            reconstruction.authority_samples,
            reconstruction.abstained_samples,
            reconstruction.pre_unowned_delta_m,
            reconstruction.post_unowned_delta_m,
        )
        fine_minimum = min(fine_minimum, float(reconstruction.values.min()))
        fine_maximum = max(fine_maximum, float(reconstruction.values.max()))
    wire_scale = float(np.float32(FINE_QSCALE_M))
    fine_code_min = float(np.round((fine_minimum - shared_offset) / wire_scale))
    fine_code_max = float(np.round((fine_maximum - shared_offset) / wire_scale))
    if fine_code_min < 0 or fine_code_max > 65535:
        raise OverflowError(
            "shared fine closure cannot be represented at qscale 0.002 m: "
            f"authority_min={domain_authority_min}, qoffset={shared_offset}, "
            f"fine_range={fine_minimum}..{fine_maximum}, "
            f"codes={fine_code_min:.0f}..{fine_code_max:.0f}"
        )

    packed_fine: list[PackedHeightChunk] = []
    fine_authority_samples = 0
    fine_abstained_samples = 0
    max_pre_unowned_delta = 0.0
    max_post_unowned_delta = 0.0
    for chunk in fine_closure:
        reconstruction = _reconstruct_fine(
            chunk,
            authority_cache,
            baseline_cache,
            layout,
            reevaluate_fine_surface,
        )
        signature = (
            _array_sha256(reconstruction.values),
            reconstruction.authority_sha256,
            reconstruction.abstention_sha256,
            reconstruction.authority_samples,
            reconstruction.abstained_samples,
            reconstruction.pre_unowned_delta_m,
            reconstruction.post_unowned_delta_m,
        )
        if signature != first_pass[chunk]:
            raise ValueError(f"fine re-evaluation is not deterministic for {chunk}")
        fine_authority_samples += reconstruction.authority_samples
        fine_abstained_samples += reconstruction.abstained_samples
        max_pre_unowned_delta = max(
            max_pre_unowned_delta, reconstruction.pre_unowned_delta_m
        )
        max_post_unowned_delta = max(
            max_post_unowned_delta, reconstruction.post_unowned_delta_m
        )
        # One domain-separated leaf binds both exact 0.25 m artifacts per halo tile.
        dependencies = tuple(
            (
                dependency,
                _paired_dependency_sha256(
                    authority_cache.get(dependency).artifact_sha256,
                    baseline_cache.get(dependency).artifact_sha256,
                ),
            )
            for dependency in sorted(
                HeightChunkId(chunk.lod, chunk.cx + dx, chunk.cz + dz)
                for dz in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
        )
        packed_fine.append(
            _pack_height(
                grid=grid,
                encode=encode,
                chunk=chunk,
                values=reconstruction.values,
                qscale=FINE_QSCALE_M,
                qoffset=shared_offset,
                dependencies=dependencies,
            )
        )
    by_chunk = {item.chunk: item for item in packed_fine}

    reduction = reduce_decoded_children(
        plan.review_parent,
        lambda chunk: _decode_packed(encode, by_chunk[chunk]),
        core_res=layout.fine_core_res,
    )
    parent_dependencies = tuple(
        (chunk, by_chunk[chunk].decoded_values_sha256)
        for chunk in reduction.dependencies
    )
    parent = _pack_height(
        grid=grid,
        encode=encode,
        chunk=plan.review_parent,
        values=reduction.values,
        qscale=parent_qscale_m,
        qoffset=None,
        dependencies=parent_dependencies,
    )
    seam = _max_seam(encode, by_chunk, plan.review_parent)
    if seam != 0.0:
        raise ValueError(f"decoded fine closure has a {seam} m apron seam")
    authority_dependencies = authority_cache.dependencies()
    baseline_dependencies = baseline_cache.dependencies()
    return StructuralHierarchy(
        fine=tuple(packed_fine),
        parent=parent,
        metrics=StructuralHierarchyMetrics(
            fine_domain_authority_min_m=domain_authority_min,
            fine_shared_qoffset=shared_offset,
            max_fine_roundtrip_error_m=max(
                item.max_roundtrip_error_m for item in packed_fine
            ),
            max_decoded_seam_m=seam,
            parent_roundtrip_error_m=parent.max_roundtrip_error_m,
            fine_reevaluation_applied=True,
            authority_dependencies=authority_dependencies,
            authority_dependency_merkle_root=dependency_merkle_root(authority_dependencies),
            baseline_dependencies=baseline_dependencies,
            baseline_dependency_merkle_root=dependency_merkle_root(baseline_dependencies),
            fine_authority_samples=fine_authority_samples,
            fine_abstained_samples=fine_abstained_samples,
            max_pre_reevaluation_unowned_delta_m=max_pre_unowned_delta,
            max_post_reevaluation_unowned_delta_m=max_post_unowned_delta,
            parent_dependencies=parent_dependencies,
            parent_dependency_merkle_root=dependency_merkle_root(parent_dependencies),
        ),
    )

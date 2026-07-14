"""Fail-closed correction of the existing 2 m flowing-water surface layer."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import binary_dilation

from ...cook.pyramid import assemble_finer, blocks16
from ...grid import ChunkId
from .baseline import SampleGrid
from .model import FlowProfile

WATER_LAYER_REPAIR_VERSION = "flowing-water-layer-repair/1"
WATER_LAYER_COLLAR_SAMPLES = 4
WATER_LAYER_ADJACENCY = "queen-8"

_QUEEN_STRUCTURE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class CorrectedWaterEvidence:
    recipe_sha256: str
    baseline_sha256: str
    output_sha256: str
    authority_sha256: str
    abstention_sha256: str
    dry_samples: int
    baseline_wet_samples: int
    mapped_wet_samples: int
    qualified_wet_samples: int
    profile_supported_samples: int
    authority_samples: int
    abstained_mapped_samples: int
    removed_support_samples: int
    zero_weight_boundary_samples: int
    tapered_samples: int
    full_profile_samples: int
    maximum_abs_correction_m: float
    maximum_boundary_correction_m: float
    collar_samples: int
    adjacency: str


@dataclass(frozen=True)
class CorrectedWaterLayer:
    water_y: np.ndarray
    authority: np.ndarray
    abstained: np.ndarray
    taper_weight: np.ndarray
    evidence: CorrectedWaterEvidence

    def __post_init__(self) -> None:
        water = np.array(self.water_y, dtype=np.float64, copy=True)
        authority = np.array(self.authority, dtype=np.bool_, copy=True)
        abstained = np.array(self.abstained, dtype=np.bool_, copy=True)
        weight = np.array(self.taper_weight, dtype=np.float64, copy=True)
        if water.ndim != 2 or any(
            array.shape != water.shape for array in (authority, abstained, weight)
        ):
            raise ValueError("corrected water arrays must be matching 2D rasters")
        if np.any(authority & abstained):
            raise ValueError("water authority and abstention masks overlap")
        if not np.isfinite(water[authority]).all():
            raise ValueError("water authority contains a nonfinite surface")
        if not np.isfinite(weight).all() or np.any((weight < 0.0) | (weight > 1.0)):
            raise ValueError("water taper weights must be finite within [0,1]")
        for array in (water, authority, abstained, weight):
            array.flags.writeable = False
        object.__setattr__(self, "water_y", water)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "abstained", abstained)
        object.__setattr__(self, "taper_weight", weight)


@dataclass(frozen=True)
class _WaterCorrectionState:
    layer: CorrectedWaterLayer
    baseline: np.ndarray
    mapped_wet: np.ndarray
    qualified_wet: np.ndarray
    profile_supported: np.ndarray
    boundary: np.ndarray


@dataclass(frozen=True)
class WaterReduction:
    water_y: np.ndarray
    dependencies: tuple[ChunkId, ...]
    present_dependencies: tuple[ChunkId, ...]

    def __post_init__(self) -> None:
        water = np.array(self.water_y, dtype=np.float32, copy=True)
        if water.ndim != 2 or np.isinf(water).any():
            raise ValueError("reduced water must be one 2D finite-or-NaN float32 array")
        water.flags.writeable = False
        object.__setattr__(self, "water_y", water)


DecodedWaterLoader = Callable[[ChunkId], np.ndarray | None]


def _canonical_array_bytes(values: np.ndarray, dtype: str) -> bytes:
    array = np.asarray(values, dtype=np.dtype(dtype)).copy()
    if np.issubdtype(array.dtype, np.floating):
        array[np.isnan(array)] = np.nan
    return np.ascontiguousarray(array).tobytes(order="C")


def _canonical_array_sha256(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(_canonical_array_bytes(values, dtype)).hexdigest()


def _validated_geometry(mapped_water_polygon, qualified_water_polygon, centerline) -> None:
    for name, polygon in (
        ("mapped_water_polygon", mapped_water_polygon),
        ("qualified_water_polygon", qualified_water_polygon),
    ):
        if (
            polygon is None
            or polygon.is_empty
            or polygon.geom_type not in {"Polygon", "MultiPolygon"}
            or not polygon.is_valid
            or bool(shapely.has_z(polygon))
        ):
            raise ValueError(f"{name} must be one valid nonempty 2D polygonal geometry")
    if not mapped_water_polygon.covers(qualified_water_polygon):
        raise ValueError("qualified water territory must stay inside mapped water")
    if (
        centerline is None
        or centerline.is_empty
        or centerline.geom_type != "LineString"
        or not centerline.is_simple
        or not centerline.is_valid
        or bool(shapely.has_z(centerline))
    ):
        raise ValueError("centerline must be one valid simple nonempty 2D LineString")


def _taper_weights(
    *,
    authority: np.ndarray,
    mapped_wet: np.ndarray,
    collar_samples: int,
) -> np.ndarray:
    """Close correction to zero inside authority before mapped abstention."""
    weights = np.ones(authority.shape, dtype=np.float64)
    visited = mapped_wet & ~authority
    frontier = visited.copy()
    for distance in range(1, collar_samples + 1):
        frontier = (
            binary_dilation(frontier, structure=_QUEEN_STRUCTURE, mask=mapped_wet)
            & ~visited
        )
        if not np.any(frontier):
            break
        layer = frontier & authority
        phase = (distance - 1.0) / collar_samples
        weights[layer] = phase * phase * (3.0 - 2.0 * phase)
        visited |= frontier
    weights[~authority] = 0.0
    return weights


def _recipe_sha256(
    *,
    grid: SampleGrid,
    baseline_sha256: str,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    collar_samples: int,
) -> str:
    digest = hashlib.sha256()
    fixed = {
        "version": WATER_LAYER_REPAIR_VERSION,
        "grid": {
            "centerE": grid.center_e,
            "centerN": grid.center_n,
            "texelMeters": grid.texel_m,
            "rows": grid.rows,
            "cols": grid.cols,
        },
        "baselineSha256": baseline_sha256,
        "profile": {
            "reachId": profile.reach_id,
            "epochId": profile.epoch_id,
            "orientation": profile.orientation.value,
            "longestMissingSpanMeters": profile.longest_missing_span_m,
            "sourceArtifactSha256": profile.source_artifact_sha256,
        },
        "collarSamples": collar_samples,
        "adjacency": WATER_LAYER_ADJACENCY,
    }
    digest.update(json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode())
    for geometry in (mapped_water_polygon, qualified_water_polygon, centerline):
        payload = shapely.to_wkb(
            geometry, byte_order=1, output_dimension=2, include_srid=True
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    for values, dtype in (
        (profile.station_m, "<f8"),
        (profile.observation_y, "<f8"),
        (profile.water_y, "<f8"),
        (profile.accepted, "u1"),
    ):
        payload = _canonical_array_bytes(values, dtype)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _correct_flowing_water_layer_state(
    *,
    baseline_water_y: np.ndarray,
    grid: SampleGrid,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    collar_samples: int = WATER_LAYER_COLLAR_SAMPLES,
) -> _WaterCorrectionState:
    """Replace only evidence-supported wet samples in an existing LOD0 layer.

    NaN is the caller-decoded dry state. Dry samples and every finite wet sample
    outside final authority retain their baseline value exactly. The correction
    reaches zero on the authority cells adjacent to mapped abstention, and the
    complete smoothstep collar remains inside qualified territory.
    """
    baseline = np.asarray(baseline_water_y)
    if baseline.shape != (grid.rows, grid.cols) or baseline.ndim != 2:
        raise ValueError("baseline water does not match its SampleGrid")
    if not np.issubdtype(baseline.dtype, np.floating) or np.isinf(baseline).any():
        raise ValueError("decoded baseline water must contain only finite values or NaN dry")
    if not np.isclose(grid.texel_m, 2.0, rtol=0.0, atol=1e-12):
        raise ValueError("corrected format-1 water authority requires the existing 2 m grid")
    if not isinstance(collar_samples, int) or isinstance(collar_samples, bool):
        raise TypeError("collar_samples must be an integer")
    if collar_samples < 1:
        raise ValueError("collar_samples must be positive")
    if not isinstance(profile, FlowProfile):
        raise TypeError("profile must be one qualified FlowProfile")
    _validated_geometry(mapped_water_polygon, qualified_water_polygon, centerline)
    if profile.station_m[0] < 0.0 or profile.station_m[-1] > centerline.length + 1e-8:
        raise ValueError("profile station domain leaves the qualified centerline")

    baseline = np.array(baseline, dtype=np.float64, copy=True)
    east = np.tile(grid.eastings(), grid.rows)
    north = np.repeat(grid.northings(), grid.cols)
    points = shapely.points(east, north)
    mapped_wet = np.asarray(
        shapely.covers(mapped_water_polygon, points), dtype=bool
    ).reshape(baseline.shape)
    qualified_wet = np.asarray(
        shapely.covers(qualified_water_polygon, points), dtype=bool
    ).reshape(baseline.shape)
    if np.any(qualified_wet & ~mapped_wet):
        raise ValueError("qualified sample territory leaves mapped water")
    station = np.asarray(
        shapely.line_locate_point(centerline, points), dtype=np.float64
    ).reshape(baseline.shape)
    profile_supported = qualified_wet & (
        (station >= profile.station_m[0]) & (station <= profile.station_m[-1])
    )
    baseline_wet = np.isfinite(baseline)
    authority = profile_supported & baseline_wet
    target = np.full(baseline.shape, np.nan, dtype=np.float64)
    target[authority] = np.interp(
        station[authority], profile.station_m, profile.water_y
    )
    if not np.isfinite(target[authority]).all():
        raise ValueError("qualified water profile produced nonfinite surface authority")

    weights = _taper_weights(
        authority=authority,
        mapped_wet=mapped_wet,
        collar_samples=collar_samples,
    )
    corrected = baseline.copy()
    corrected[authority] = baseline[authority] + weights[authority] * (
        target[authority] - baseline[authority]
    )
    abstained = mapped_wet & ~authority
    if not np.array_equal(corrected[~authority], baseline[~authority], equal_nan=True):
        raise AssertionError("water correction changed dry or abstained baseline")

    mapped_abstention = mapped_wet & ~authority
    boundary = authority & binary_dilation(
        mapped_abstention, structure=_QUEEN_STRUCTURE, mask=mapped_wet
    )
    correction = corrected - baseline
    boundary_error = float(np.max(np.abs(correction[boundary]), initial=0.0))
    if boundary_error != 0.0:
        raise AssertionError("water correction is not C0 at mapped abstention")

    baseline_sha = _canonical_array_sha256(baseline, "<f8")
    output_sha = _canonical_array_sha256(corrected, "<f8")
    authority_sha = _canonical_array_sha256(authority, "u1")
    abstention_sha = _canonical_array_sha256(abstained, "u1")
    tapered = authority & (weights > 0.0) & (weights < 1.0)
    evidence = CorrectedWaterEvidence(
        recipe_sha256=_recipe_sha256(
            grid=grid,
            baseline_sha256=baseline_sha,
            mapped_water_polygon=mapped_water_polygon,
            qualified_water_polygon=qualified_water_polygon,
            centerline=centerline,
            profile=profile,
            collar_samples=collar_samples,
        ),
        baseline_sha256=baseline_sha,
        output_sha256=output_sha,
        authority_sha256=authority_sha,
        abstention_sha256=abstention_sha,
        dry_samples=int(np.count_nonzero(~baseline_wet)),
        baseline_wet_samples=int(np.count_nonzero(baseline_wet)),
        mapped_wet_samples=int(np.count_nonzero(mapped_wet)),
        qualified_wet_samples=int(np.count_nonzero(qualified_wet)),
        profile_supported_samples=int(np.count_nonzero(profile_supported)),
        authority_samples=int(np.count_nonzero(authority)),
        abstained_mapped_samples=int(np.count_nonzero(abstained)),
        removed_support_samples=int(np.count_nonzero(profile_supported & ~authority)),
        zero_weight_boundary_samples=int(np.count_nonzero(boundary)),
        tapered_samples=int(np.count_nonzero(tapered)),
        full_profile_samples=int(np.count_nonzero(authority & (weights == 1.0))),
        maximum_abs_correction_m=float(
            np.max(np.abs(correction[authority]), initial=0.0)
        ),
        maximum_boundary_correction_m=boundary_error,
        collar_samples=collar_samples,
        adjacency=WATER_LAYER_ADJACENCY,
    )
    return _WaterCorrectionState(
        layer=CorrectedWaterLayer(corrected, authority, abstained, weights, evidence),
        baseline=baseline,
        mapped_wet=mapped_wet,
        qualified_wet=qualified_wet,
        profile_supported=profile_supported,
        boundary=boundary,
    )


def correct_flowing_water_layer(
    *,
    baseline_water_y: np.ndarray,
    grid: SampleGrid,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    collar_samples: int = WATER_LAYER_COLLAR_SAMPLES,
) -> CorrectedWaterLayer:
    """Public array primitive; halo-aware chunk correction is a separate adapter."""
    return _correct_flowing_water_layer_state(
        baseline_water_y=baseline_water_y,
        grid=grid,
        mapped_water_polygon=mapped_water_polygon,
        qualified_water_polygon=qualified_water_polygon,
        centerline=centerline,
        profile=profile,
        collar_samples=collar_samples,
    ).layer


def _wet_majority_fixed(blocks: np.ndarray) -> np.ndarray:
    """Current majority-wet semantics with explicit row-major float64 addition."""
    values = np.asarray(blocks, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 16:
        raise ValueError("water reducer requires 16-sample fine blocks")
    wet = np.isfinite(values)
    count = wet.sum(axis=-1, dtype=np.int16)
    total = np.zeros(values.shape[:2], dtype=np.float64)
    for index in range(16):
        total = total + np.where(wet[..., index], values[..., index], 0.0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return np.where((count >= 8) & (count > 0), mean, np.nan).astype(np.float32)


def reduce_water_lod1(
    parent: ChunkId,
    load_decoded: DecodedWaterLoader,
    *,
    core_res: int = 1024,
) -> WaterReduction | None:
    """Reduce decoded LOD0 water through the established majority-wet policy.

    The callback may return corrected replacements, inherited base chunks, or
    ``None`` for the format-1 all-dry/absent state. Apron fallback behavior is
    delegated to the existing assembly primitive so the current contract stays
    unchanged.
    """
    if parent.lod != 1:
        raise ValueError("corrected flowing-water reduction produces an LOD1 parent")
    if core_res < 4 or core_res % 4:
        raise ValueError("water child core_res must be positive and divisible by four")
    dependencies = tuple(
        ChunkId(parent.cx * 4 + dx, parent.cz * 4 + dz, 0)
        for dz in range(5)
        for dx in range(5)
    )
    present: list[ChunkId] = []

    def read(chunk: ChunkId) -> list[np.ndarray] | None:
        values = load_decoded(chunk)
        if values is None:
            return None
        array = np.asarray(values)
        expected = (core_res + 1,) * 2
        if array.dtype != np.float32 or array.shape != expected or np.isinf(array).any():
            raise ValueError(f"decoded water {chunk} must be float32 {expected} or absent")
        present.append(chunk)
        return [array]

    assembled = assemble_finer(read, parent, core_res, fills=[np.nan])
    if assembled is None:
        return None
    reduced = _wet_majority_fixed(blocks16(assembled[0]))
    return WaterReduction(reduced, dependencies, tuple(present))

"""Integrity-bound halo correction for ordinary LOD0 water chunks."""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ...config import GridConfig
from ...grid import ChunkId, chunk_bounds_en
from .baseline import SampleGrid
from .model import FlowProfile
from .water_layer import (
    WATER_LAYER_COLLAR_SAMPLES,
    CorrectedWaterEvidence,
    CorrectedWaterLayer,
    _canonical_array_sha256,
    _correct_flowing_water_layer_state,
)


_DEPENDENCY_DOMAIN = b"laas.structural-water.halo-dependencies.v1\0"
_RECIPE_DOMAIN = b"laas.structural-water.halo-correction.v1\0"


class WaterHaloSource(StrEnum):
    INHERITED = "inherited"
    CORRECTED = "corrected"
    ABSENT = "explicit_absent"


@dataclass(frozen=True)
class DecodedWaterHaloChunk:
    chunk: ChunkId
    source: WaterHaloSource
    water_y: np.ndarray | None
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, WaterHaloSource):
            raise TypeError("halo source must be a WaterHaloSource")
        try:
            if (
                self.artifact_sha256 != self.artifact_sha256.lower()
                or len(bytes.fromhex(self.artifact_sha256)) != 32
            ):
                raise ValueError
        except ValueError as error:
            raise ValueError("halo artifact_sha256 must be a lowercase SHA-256") from error
        if self.source is WaterHaloSource.ABSENT:
            if self.water_y is not None:
                raise ValueError("explicit-absent water must not carry decoded samples")
            return
        if self.water_y is None:
            raise ValueError("present inherited/corrected water requires decoded samples")
        water = np.array(self.water_y, dtype=np.float32, copy=True)
        if water.ndim != 2 or np.isinf(water).any():
            raise ValueError("decoded halo water must be 2D finite-or-NaN float32")
        water.flags.writeable = False
        object.__setattr__(self, "water_y", water)


@dataclass(frozen=True)
class HaloCorrectedWaterLayer:
    layer: CorrectedWaterLayer
    dependencies: tuple[tuple[ChunkId, WaterHaloSource, str], ...]
    dependency_sha256: str


WaterHaloLoader = Callable[[ChunkId], DecodedWaterHaloChunk]


def _dependency_sha256(
    dependencies: tuple[tuple[ChunkId, WaterHaloSource, str], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_DEPENDENCY_DOMAIN)
    for chunk, source, artifact_sha256 in dependencies:
        encoded_source = source.value.encode()
        digest.update(struct.pack("<BiiB", chunk.lod, chunk.cx, chunk.cz, len(encoded_source)))
        digest.update(encoded_source)
        digest.update(bytes.fromhex(artifact_sha256))
    return digest.hexdigest()


def _assemble_halo(
    *,
    chunk: ChunkId,
    core_res: int,
    halo: int,
    load: WaterHaloLoader,
) -> tuple[np.ndarray, tuple[tuple[ChunkId, WaterHaloSource, str], ...]]:
    res = core_res + 1
    records: dict[tuple[int, int], DecodedWaterHaloChunk] = {}
    dependencies: list[tuple[ChunkId, WaterHaloSource, str]] = []
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            expected = ChunkId(chunk.cx + dx, chunk.cz + dz, 0)
            record = load(expected)
            if not isinstance(record, DecodedWaterHaloChunk) or record.chunk != expected:
                raise ValueError(f"halo loader returned the wrong water artifact for {expected}")
            if record.water_y is not None and record.water_y.shape != (res, res):
                raise ValueError(f"decoded halo water {expected} must be {(res, res)}")
            records[dx, dz] = record
            dependencies.append((expected, record.source, record.artifact_sha256))

    indices = np.arange(-halo, res + halo, dtype=np.int64)
    tile_delta = np.floor_divide(indices, core_res)
    local = indices - tile_delta * core_res
    expanded = np.full((res + 2 * halo, res + 2 * halo), np.nan, dtype=np.float32)
    for dz in (-1, 0, 1):
        rows = np.flatnonzero(tile_delta == dz)
        source_rows = local[rows]
        for dx in (-1, 0, 1):
            record = records[dx, dz]
            if record.water_y is None:
                continue
            cols = np.flatnonzero(tile_delta == dx)
            source_cols = local[cols]
            expanded[np.ix_(rows, cols)] = record.water_y[
                np.ix_(source_rows, source_cols)
            ]
    return expanded, tuple(dependencies)


def correct_flowing_water_chunk_halo(
    *,
    chunk: ChunkId,
    grid: GridConfig,
    load_baseline: WaterHaloLoader,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
) -> HaloCorrectedWaterLayer:
    """Correct one 2 m payload using a real four-sample decoded neighborhood."""
    if chunk.lod != 0 or grid.chunk_m % 2:
        raise ValueError("halo water correction requires an ordinary even-sized LOD0 grid")
    core_res = grid.chunk_m // 2
    halo = WATER_LAYER_COLLAR_SAMPLES
    expanded, dependencies = _assemble_halo(
        chunk=chunk,
        core_res=core_res,
        halo=halo,
        load=load_baseline,
    )
    dependency_sha = _dependency_sha256(dependencies)
    e_min, _, _, n_max = chunk_bounds_en(grid, chunk)
    expanded_grid = SampleGrid(
        center_e=e_min + 1.0 - halo * 2.0,
        center_n=n_max - 1.0 + halo * 2.0,
        texel_m=2.0,
        rows=expanded.shape[0],
        cols=expanded.shape[1],
    )
    state = _correct_flowing_water_layer_state(
        baseline_water_y=expanded,
        grid=expanded_grid,
        mapped_water_polygon=mapped_water_polygon,
        qualified_water_polygon=qualified_water_polygon,
        centerline=centerline,
        profile=profile,
        collar_samples=halo,
    )
    res = core_res + 1
    crop = np.s_[halo : halo + res, halo : halo + res]
    baseline = state.baseline[crop]
    water = state.layer.water_y[crop]
    authority = state.layer.authority[crop]
    abstained = state.layer.abstained[crop]
    weight = state.layer.taper_weight[crop]
    mapped = state.mapped_wet[crop]
    qualified = state.qualified_wet[crop]
    profile_supported = state.profile_supported[crop]
    boundary = state.boundary[crop]
    correction = water - baseline

    recipe = hashlib.sha256()
    recipe.update(_RECIPE_DOMAIN)
    recipe.update(bytes.fromhex(state.layer.evidence.recipe_sha256))
    recipe.update(bytes.fromhex(dependency_sha))
    recipe.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
    boundary_error = float(np.max(np.abs(correction[boundary]), initial=0.0))
    evidence = CorrectedWaterEvidence(
        recipe_sha256=recipe.hexdigest(),
        baseline_sha256=_canonical_array_sha256(baseline, "<f8"),
        output_sha256=_canonical_array_sha256(water, "<f8"),
        authority_sha256=_canonical_array_sha256(authority, "u1"),
        abstention_sha256=_canonical_array_sha256(abstained, "u1"),
        dry_samples=int(np.count_nonzero(~np.isfinite(baseline))),
        baseline_wet_samples=int(np.count_nonzero(np.isfinite(baseline))),
        mapped_wet_samples=int(np.count_nonzero(mapped)),
        qualified_wet_samples=int(np.count_nonzero(qualified)),
        profile_supported_samples=int(np.count_nonzero(profile_supported)),
        authority_samples=int(np.count_nonzero(authority)),
        abstained_mapped_samples=int(np.count_nonzero(abstained)),
        removed_support_samples=int(np.count_nonzero(profile_supported & ~authority)),
        zero_weight_boundary_samples=int(np.count_nonzero(boundary)),
        tapered_samples=int(np.count_nonzero(authority & (weight > 0.0) & (weight < 1.0))),
        full_profile_samples=int(np.count_nonzero(authority & (weight == 1.0))),
        maximum_abs_correction_m=float(
            np.max(np.abs(correction[authority]), initial=0.0)
        ),
        maximum_boundary_correction_m=boundary_error,
        collar_samples=halo,
        adjacency=state.layer.evidence.adjacency,
    )
    if boundary_error != 0.0:
        raise AssertionError("halo water correction is not C0 at mapped abstention")
    layer = CorrectedWaterLayer(water, authority, abstained, weight, evidence)
    if not np.array_equal(layer.water_y[~authority], baseline[~authority], equal_nan=True):
        raise AssertionError("halo correction changed dry or abstained baseline")
    return HaloCorrectedWaterLayer(layer, dependencies, dependency_sha)

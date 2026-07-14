"""Fail-closed C0 closure between supported and abstained wet authority."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .baseline import BaselineTile
from .model import StructuralTile, WaterSamples


MIXED_AUTHORITY_CLOSURE_VERSION = "mixed-authority-c0/2"
MIXED_AUTHORITY_COLLAR_SAMPLES = 4
MIXED_AUTHORITY_ADJACENCY = "queen-8"

_STRUCTURE = np.ones((3, 3), dtype=bool)
_CLEARANCE_EPSILON_M = 1e-12


@dataclass(frozen=True)
class MixedAuthorityClosure:
    """Closed structural tile and support removed to preserve its guarantees."""

    tile: StructuralTile
    removed_support: np.ndarray

    def __post_init__(self) -> None:
        removed = np.array(self.removed_support, dtype=np.bool_, copy=True)
        if removed.shape != self.tile.height.shape:
            raise ValueError("removed_support must match the structural tile")
        removed.flags.writeable = False
        object.__setattr__(self, "removed_support", removed)


@dataclass(frozen=True)
class MixedAuthoritySurface:
    """One expanded mixed-authority closure before any storage crop."""

    height: np.ndarray
    authority: np.ndarray
    abstained: np.ndarray
    removed_support: np.ndarray

    def __post_init__(self) -> None:
        height = np.asarray(self.height, dtype=np.float64)
        authority = np.asarray(self.authority, dtype=np.bool_)
        abstained = np.asarray(self.abstained, dtype=np.bool_)
        removed = np.asarray(self.removed_support, dtype=np.bool_)
        if height.ndim != 2 or any(
            array.shape != height.shape
            for array in (authority, abstained, removed)
        ):
            raise ValueError("mixed-authority surface arrays must be matching 2D rasters")
        if np.any(authority & abstained) or np.any(removed & authority):
            raise ValueError("mixed-authority surface ownership is contradictory")
        for array in (height, authority, abstained, removed):
            array.flags.writeable = False
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "abstained", abstained)
        object.__setattr__(self, "removed_support", removed)


def _taper_weights(
    *,
    support: np.ndarray,
    wet: np.ndarray,
    collar_samples: int,
) -> np.ndarray:
    """Return a C0 smoothstep wholly inside current supported territory."""
    weights = np.ones(support.shape, dtype=np.float64)
    visited = wet & ~support
    frontier = visited.copy()
    for distance in range(1, collar_samples + 1):
        frontier = binary_dilation(frontier, structure=_STRUCTURE, mask=wet) & ~visited
        if not np.any(frontier):
            break
        layer = frontier & support
        t = (distance - 1.0) / collar_samples
        weights[layer] = t * t * (3.0 - 2.0 * t)
        visited |= frontier
    return weights


def close_mixed_water_authority(
    baseline: BaselineTile,
    water: WaterSamples,
    *,
    collar_samples: int = MIXED_AUTHORITY_COLLAR_SAMPLES,
) -> MixedAuthorityClosure:
    """Compose water-bed authority without a step into abstained wet terrain.

    The correction is ``bed - baseline``. At every supported cell sharing an
    edge or corner with mapped-wet abstention its multiplier is exactly zero;
    farther collar layers use a deterministic cubic smoothstep and the interior
    targets the bed exactly. Any result above qualified water loses authority,
    becomes abstained wet, and causes the closure to be recomputed until stable.
    """
    if not isinstance(collar_samples, int) or isinstance(collar_samples, bool):
        raise TypeError("collar_samples must be an integer")
    if collar_samples < 1:
        raise ValueError("collar_samples must be positive")
    closed = close_mixed_water_surface(
        baseline.height,
        baseline.valid,
        water,
        collar_samples=collar_samples,
    )
    wet = water.wet.reshape(baseline.height.shape)
    valid = np.array(baseline.valid, dtype=bool, copy=True)
    valid[closed.authority] = True
    tile = StructuralTile(
        height=closed.height,
        valid=valid,
        unknown_bathymetry=closed.authority,
        forbidden_morphology=wet,
    )
    if not np.array_equal(tile.valid[~closed.authority], baseline.valid[~closed.authority]):
        raise AssertionError("mixed-authority closure changed abstained/dry baseline")
    return MixedAuthorityClosure(tile=tile, removed_support=closed.removed_support)


def close_mixed_water_surface(
    baseline_height: np.ndarray,
    baseline_valid: np.ndarray,
    water: WaterSamples,
    *,
    collar_samples: int = MIXED_AUTHORITY_COLLAR_SAMPLES,
) -> MixedAuthoritySurface:
    """Close one expanded surface with a shared coarse/fine physical rule.

    ``bed_y`` is the exact full-weight target, not a clearance ceiling. The
    qualified ``water_y`` is the hard visibility ceiling for both full and
    tapered samples. Unsafe support is converted to mapped-wet abstention and
    the collar is recomputed until stable. Dry and final-abstained samples are
    then restored from the supplied baseline without arithmetic.
    """
    if not isinstance(collar_samples, int) or isinstance(collar_samples, bool):
        raise TypeError("collar_samples must be an integer")
    if collar_samples < 1:
        raise ValueError("collar_samples must be positive")
    baseline = np.asarray(baseline_height, dtype=np.float64)
    valid = np.asarray(baseline_valid, dtype=np.bool_)
    if baseline.ndim != 2 or valid.shape != baseline.shape:
        raise ValueError("baseline height and validity must be matching 2D arrays")
    if not np.isfinite(baseline[valid]).all() or np.isfinite(baseline[~valid]).any():
        raise ValueError("baseline invalid samples must be NaN and valid samples finite")
    if water.wet.size != baseline.size:
        raise ValueError("water samples do not match the baseline surface")

    shape = baseline.shape
    wet = water.wet.reshape(shape)
    initial_support = water.supported.reshape(shape)
    water_y = water.water_y.reshape(shape)
    bed_y = water.bed_y.reshape(shape)
    support = initial_support & valid
    removed = initial_support & ~support
    height = np.array(baseline, dtype=np.float64, copy=True)

    while True:
        weights = _taper_weights(
            support=support,
            wet=wet,
            collar_samples=collar_samples,
        )
        height[...] = baseline
        full_weight = support & (weights == 1.0)
        tapered = support & ~full_weight
        # Assignment, rather than baseline + (bed - baseline), makes the
        # conservative target bit-exact wherever the collar has full weight.
        height[full_weight] = bed_y[full_weight]
        height[tapered] = baseline[tapered] + weights[tapered] * (
            bed_y[tapered] - baseline[tapered]
        )
        unsafe = support & (
            ~np.isfinite(height)
            | ~np.isfinite(water_y)
            | ~np.isfinite(bed_y)
            | (height > water_y + _CLEARANCE_EPSILON_M)
        )
        if not np.any(unsafe):
            break
        support[unsafe] = False
        removed[unsafe] = True

    height[~support] = baseline[~support]
    abstained = wet & ~support
    if not np.array_equal(height[~support], baseline[~support], equal_nan=True):
        raise AssertionError("mixed-authority closure changed abstained/dry baseline")
    if np.any(height[support] > water_y[support] + _CLEARANCE_EPSILON_M):
        raise AssertionError("mixed-authority closure exceeded qualified water")
    if not np.array_equal(height[full_weight & support], bed_y[full_weight & support]):
        raise AssertionError("mixed-authority closure changed its full-weight bed target")
    return MixedAuthoritySurface(height, support, abstained, removed)


def rerasterize_fine_shared_closure_for_verification(
    *, chunk, scientific_manifest_path, authority_manifest_path
):
    """Compatibility entry point; implementation lives with its frozen inputs."""
    from .shared_closure import (
        rerasterize_fine_shared_closure_for_verification as rerasterize,
    )

    return rerasterize(
        chunk=chunk,
        scientific_manifest_path=scientific_manifest_path,
        authority_manifest_path=authority_manifest_path,
    )

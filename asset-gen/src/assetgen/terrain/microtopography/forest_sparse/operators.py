"""Support-aware ideal degradation, patch pairing, and sparse reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class SurfaceArrays:
    source_id: str
    elevation_m: np.ndarray
    measured: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class LowSurface:
    elevation_m: np.ndarray
    support: np.ndarray
    confidence: np.ndarray
    direct_fraction: np.ndarray


@dataclass(frozen=True)
class Atom:
    source_id: str
    source_y: int
    source_x: int
    low_shape: np.ndarray
    low_support: np.ndarray
    low_confidence: np.ndarray
    high_shape: np.ndarray
    high_support: np.ndarray
    high_confidence: np.ndarray


@dataclass(frozen=True)
class Reconstruction:
    elevation_m: np.ndarray
    support: np.ndarray
    confidence: np.ndarray
    atom_index: np.ndarray
    coefficient: np.ndarray
    match_error: np.ndarray
    patch_origin_low: np.ndarray
    atom_source: tuple[str, ...]
    atom_origin_low: np.ndarray


def polynomial_mask(cells: int) -> np.ndarray:
    """Pinned MATLAB build_mask.m polynomial overlap window."""
    if cells <= 1 or cells % 2:
        raise ValueError("mask size must be an even integer greater than one")
    radius = (cells - 1) * 0.5
    axis = (np.arange(cells, dtype=np.float64) - radius) / radius
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    value = np.maximum(0.0, 1.0 - (1.0 - 1.0 / cells) * (xx * xx + yy * yy))
    return value * value


def ideal_area_degrade(
    surface: SurfaceArrays,
    *,
    factor: int,
    min_direct_fraction: float,
    min_support_fraction: float,
    min_mean_confidence: float,
) -> LowSurface:
    """Metric 1 m area observation; unsupported fine cells never enter the mean."""
    z = np.asarray(surface.elevation_m, dtype=np.float64)
    measured = np.asarray(surface.measured, dtype=bool)
    confidence = np.asarray(surface.confidence, dtype=np.float64)
    if z.ndim != 2 or z.shape != measured.shape or z.shape != confidence.shape:
        raise ValueError("surface arrays must have equal two-dimensional shapes")
    if z.shape[0] % factor or z.shape[1] % factor:
        raise ValueError("surface dimensions must be divisible by degradation factor")
    supported = np.isfinite(z) & np.isfinite(confidence) & (confidence > 0)
    if np.any(measured & ~supported):
        raise ValueError("direct-measurement mask includes unsupported cells")
    h, w = z.shape[0] // factor, z.shape[1] // factor
    block = (h, factor, w, factor)
    support_fraction = supported.reshape(block).mean(axis=(1, 3))
    direct_fraction = measured.reshape(block).mean(axis=(1, 3))
    conf_sum = np.where(supported, confidence, 0.0).reshape(block).sum(axis=(1, 3))
    weighted_z = np.where(supported, z * confidence, 0.0).reshape(block).sum(axis=(1, 3))
    low_z = np.divide(weighted_z, conf_sum, out=np.full((h, w), np.nan), where=conf_sum > 0)
    mean_conf = conf_sum / float(factor * factor)
    low_support = (
        (support_fraction >= min_support_fraction)
        & (direct_fraction >= min_direct_fraction)
        & (mean_conf >= min_mean_confidence)
    )
    low_z[~low_support] = np.nan
    return LowSurface(low_z, low_support, np.where(low_support, mean_conf, 0.0), direct_fraction)


def support_preserving_bilinear(low: LowSurface, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear baseline with normalized support; no interpolated cell becomes evidence."""
    values = np.where(low.support, low.elevation_m * low.confidence, 0.0)
    weights = np.where(low.support, low.confidence, 0.0)
    scale = (factor, factor)
    value_hr = zoom(values, scale, order=1, mode="nearest", grid_mode=True)
    weight_hr = zoom(weights, scale, order=1, mode="nearest", grid_mode=True)
    support_hr = zoom(low.support.astype(np.float64), scale, order=1, mode="nearest", grid_mode=True)
    output = np.divide(value_hr, weight_hr, out=np.full_like(value_hr, np.nan), where=weight_hr > 0)
    valid = support_hr >= 1.0 - 1e-12
    output[~valid] = np.nan
    return output, valid


def _patch_origins(cells: int, patch_cells: int, stride_cells: int) -> tuple[int, ...]:
    if patch_cells > cells or (cells - patch_cells) % stride_cells:
        raise ValueError("patch geometry does not tile the low-resolution surface exactly")
    return tuple(range(0, cells - patch_cells + 1, stride_cells))


def build_atoms(
    surface: SurfaceArrays,
    low: LowSurface,
    *,
    factor: int,
    patch_cells: int,
    stride_cells: int,
    min_low_support_fraction: float,
    min_high_direct_fraction: float,
    min_atom_norm_m: float,
) -> list[Atom]:
    atoms: list[Atom] = []
    low_mask = polynomial_mask(patch_cells)
    high_cells = patch_cells * factor
    for y0 in _patch_origins(low.elevation_m.shape[0], patch_cells, stride_cells):
        for x0 in _patch_origins(low.elevation_m.shape[1], patch_cells, stride_cells):
            lys = slice(y0, y0 + patch_cells)
            lxs = slice(x0, x0 + patch_cells)
            low_support = low.support[lys, lxs]
            if np.mean(low_support) < min_low_support_fraction:
                continue
            hy0, hx0 = y0 * factor, x0 * factor
            hys, hxs = slice(hy0, hy0 + high_cells), slice(hx0, hx0 + high_cells)
            high_z = surface.elevation_m[hys, hxs]
            high_direct = surface.measured[hys, hxs]
            high_support = np.isfinite(high_z) & (surface.confidence[hys, hxs] > 0)
            if np.mean(high_direct) < min_high_direct_fraction:
                continue
            # The release removes the arithmetic mean before applying its mask.
            # Restricting that arithmetic mean to supported cells is the unique
            # all-valid-equivalent extension that does not invent void values.
            low_mean = float(np.mean(low.elevation_m[lys, lxs][low_support]))
            low_residual = np.where(low_support, low.elevation_m[lys, lxs] - low_mean, 0.0)
            low_norm = float(np.linalg.norm(low_residual * low_mask * low_support))
            if not np.isfinite(low_norm) or low_norm < min_atom_norm_m:
                continue
            if not high_support.any():
                continue
            high_mean = float(np.mean(high_z[high_support]))
            atoms.append(
                Atom(
                    source_id=surface.source_id,
                    source_y=y0,
                    source_x=x0,
                    low_shape=np.where(low_support, low_residual / low_norm, 0.0),
                    low_support=low_support.copy(),
                    low_confidence=low.confidence[lys, lxs].copy(),
                    # MATLAB divides the corresponding high atom by the LOW atom
                    # norm (terrain_super_resolution.m:203-206), not its own norm.
                    high_shape=np.where(high_support, (high_z - high_mean) / low_norm, 0.0),
                    high_support=high_support.copy(),
                    high_confidence=surface.confidence[hys, hxs].copy(),
                )
            )
    return atoms


def reconstruct_sparse_one(
    low: LowSurface,
    atoms: list[Atom],
    *,
    factor: int,
    patch_cells: int,
    stride_cells: int,
    min_common_support_fraction: float,
    target_high_support: np.ndarray,
    target_high_confidence: np.ndarray,
) -> Reconstruction:
    """Released-code s=1 OMP, generalized to a different mask per target/atom pair."""
    if not atoms:
        raise ValueError("dictionary is empty")
    output_shape = (low.elevation_m.shape[0] * factor, low.elevation_m.shape[1] * factor)
    numerator = np.zeros(output_shape, dtype=np.float64)
    denominator = np.zeros(output_shape, dtype=np.float64)
    confidence_sum = np.zeros(output_shape, dtype=np.float64)
    high_cells = patch_cells * factor
    low_mask = polynomial_mask(patch_cells)
    high_mask = polynomial_mask(high_cells)
    assignments: list[int] = []
    coefficients: list[float] = []
    errors: list[float] = []
    origins: list[tuple[int, int]] = []
    for y0 in _patch_origins(low.elevation_m.shape[0], patch_cells, stride_cells):
        for x0 in _patch_origins(low.elevation_m.shape[1], patch_cells, stride_cells):
            origins.append((y0, x0))
            lys, lxs = slice(y0, y0 + patch_cells), slice(x0, x0 + patch_cells)
            query_support = low.support[lys, lxs]
            if np.mean(query_support) < min_common_support_fraction:
                assignments.append(-1)
                coefficients.append(float("nan"))
                errors.append(float("nan"))
                continue
            query_mean = float(np.mean(low.elevation_m[lys, lxs][query_support]))
            query = np.where(query_support, low.elevation_m[lys, lxs] - query_mean, 0.0)
            best: tuple[float, int, float] | None = None
            for index, atom in enumerate(atoms):
                common = query_support & atom.low_support
                if np.mean(common) < min_common_support_fraction:
                    continue
                # Both the query and atom are pre-multiplied by the release mask,
                # hence Euclidean OMP has mask^2 weight. Confidence determines
                # support admission but does not silently alter that mechanism.
                weight = low_mask * low_mask * common
                denom = float(np.sum(weight * atom.low_shape * atom.low_shape))
                if denom <= 0:
                    continue
                coefficient = float(np.sum(weight * query * atom.low_shape) / denom)
                residual = query - coefficient * atom.low_shape
                error = float(np.sum(weight * residual * residual) / np.sum(weight))
                candidate = (error, index, coefficient)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                assignments.append(-1)
                coefficients.append(float("nan"))
                errors.append(float("nan"))
                continue
            error, index, coefficient = best
            assignments.append(index)
            coefficients.append(coefficient)
            errors.append(error)
            atom = atoms[index]
            hy0, hx0 = y0 * factor, x0 * factor
            hys, hxs = slice(hy0, hy0 + high_cells), slice(hx0, hx0 + high_cells)
            target_patch_support = target_high_support[hys, hxs]
            target_patch_confidence = target_high_confidence[hys, hxs]
            common_high = atom.high_support & target_patch_support
            # This is the release's single high mask factor in overlap-add,
            # restricted to the intersection of exemplar and target evidence.
            weight = high_mask * common_high
            patch = query_mean + coefficient * atom.high_shape
            numerator[hys, hxs] += weight * patch
            denominator[hys, hxs] += weight
            confidence_sum[hys, hxs] += weight * np.minimum(
                atom.high_confidence, target_patch_confidence
            )
    valid = denominator > 0
    output = np.divide(numerator, denominator, out=np.full(output_shape, np.nan), where=valid)
    confidence = np.divide(confidence_sum, denominator, out=np.zeros(output_shape), where=valid)
    return Reconstruction(
        output,
        valid,
        confidence,
        np.asarray(assignments, dtype=np.int32),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        np.asarray(origins, dtype=np.int32),
        tuple(atom.source_id for atom in atoms),
        np.asarray([(atom.source_y, atom.source_x) for atom in atoms], dtype=np.int32),
    )

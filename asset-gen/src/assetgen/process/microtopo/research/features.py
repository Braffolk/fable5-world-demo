"""Common geometry-only descriptors for ForestSemantic and Hovi point sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .contracts import RADII_M


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray
    valid: np.ndarray


def _lower_envelope_height(xyz: np.ndarray, cell_m: float = 1.0) -> np.ndarray:
    """Height over a robust plot-local 1 m lower-envelope tangent plane."""
    origin = np.floor(xyz[:, :2].min(axis=0) / cell_m) * cell_m
    ij = np.floor((xyz[:, :2] - origin) / cell_m).astype(np.int64)
    width, height = ij.max(axis=0) + 1
    linear = ij[:, 1] * width + ij[:, 0]
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    starts = np.flatnonzero(np.r_[True, sorted_linear[1:] != sorted_linear[:-1]])
    ends = np.r_[starts[1:], len(order)]
    slices = {
        int(sorted_linear[begin]): order[begin:end]
        for begin, end in zip(starts, ends)
    }
    relative = np.full(len(xyz), np.nan, dtype=np.float64)
    for cell, target_indices in slices.items():
        cy, cx = divmod(cell, width)
        neighbors = [
            slices[ny * width + nx]
            for ny in range(max(0, cy - 1), min(height, cy + 2))
            for nx in range(max(0, cx - 1), min(width, cx + 2))
            if ny * width + nx in slices
        ]
        local_indices = np.concatenate(neighbors)
        points = xyz[local_indices]
        cutoff = np.quantile(points[:, 2], 0.20, method="nearest")
        lower = points[points[:, 2] <= cutoff]
        center = origin + np.asarray((cx + 0.5, cy + 0.5)) * cell_m
        design = np.column_stack(
            (lower[:, 0] - center[0], lower[:, 1] - center[1], np.ones(len(lower)))
        )
        coefficients, *_ = np.linalg.lstsq(design, lower[:, 2], rcond=None)
        for _ in range(4):
            residual = lower[:, 2] - design @ coefficients
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale <= 1e-6:
                break
            absolute = np.abs(residual)
            weights = np.minimum(1.0, 1.345 * scale / np.maximum(absolute, 1e-12))
            weighted = design * np.sqrt(weights)[:, None]
            coefficients, *_ = np.linalg.lstsq(
                weighted, lower[:, 2] * np.sqrt(weights), rcond=None
            )
        target = xyz[target_indices]
        target_design = np.column_stack(
            (
                target[:, 0] - center[0],
                target[:, 1] - center[1],
                np.ones(len(target)),
            )
        )
        relative[target_indices] = target[:, 2] - target_design @ coefficients
    if not np.isfinite(relative).all():
        raise ValueError("robust lower-envelope plane left undefined points")
    return relative


def geometry_features(
    context_xyz: np.ndarray,
    query_xyz: np.ndarray | None = None,
    *,
    batch_points: int = 32_768,
) -> FeatureBatch:
    """Compute bounded spherical-neighborhood descriptors in query batches.

    Support counts are exact. Covariance uses the nearest 64 observations inside each
    radius so extreme sensor density cannot change the geometry definition or memory.
    """
    context = np.asarray(context_xyz, dtype=np.float64)
    query = context if query_xyz is None else np.asarray(query_xyz, dtype=np.float64)
    if context.ndim != 2 or context.shape[1] != 3 or query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("geometry descriptors require Nx3 point matrices")
    if not np.isfinite(context).all() or not np.isfinite(query).all():
        raise ValueError("geometry descriptors reject non-finite coordinates")
    tree = cKDTree(context, compact_nodes=True, balanced_tree=True)
    features = np.full((len(query), len(RADII_M) * 6 + 1), np.nan, dtype=np.float32)
    valid = np.ones(len(query), dtype=np.bool_)
    for begin in range(0, len(query), batch_points):
        end = min(begin + batch_points, len(query))
        query_batch = query[begin:end]
        distances, indices = tree.query(query_batch, k=min(64, len(context)), workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        nearest_spacing = distances[:, 1] if query_xyz is None else distances[:, 0]
        neighbors = context[indices]
        for radius_index, radius in enumerate(RADII_M):
            offset = radius_index * 6
            mask = distances <= radius
            sampled_count = mask.sum(axis=1)
            exact_count = tree.query_ball_point(
                query_batch, radius, return_length=True, workers=-1
            )
            weights = mask.astype(np.float64)
            denominator = np.maximum(sampled_count, 1)[:, None]
            mean = np.einsum("bn,bnj->bj", weights, neighbors) / denominator
            centered = neighbors - mean[:, None, :]
            covariance = np.einsum(
                "bn,bni,bnj->bij", weights, centered, centered, optimize=True
            ) / denominator[:, :, None]
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0.0)
            l0, l1, l2 = values.T
            usable = (sampled_count >= 6) & (l2 > 1e-12)
            scale = np.where(usable, l2, 1.0)
            rows = slice(begin, end)
            features[rows, offset] = np.where(usable, (l2 - l1) / scale, np.nan)
            features[rows, offset + 1] = np.where(usable, (l1 - l0) / scale, np.nan)
            features[rows, offset + 2] = np.where(usable, l0 / scale, np.nan)
            features[rows, offset + 3] = np.where(usable, np.abs(vectors[:, 2, 0]), np.nan)
            features[rows, offset + 4] = exact_count
            features[rows, offset + 5] = nearest_spacing
            valid[rows] &= usable
    if query_xyz is None:
        features[:, -1] = _lower_envelope_height(context)
    else:
        # Cross-sensor candidate use has no validated common lower-envelope reference.
        valid[:] = False
    valid &= np.isfinite(features).all(axis=1)
    return FeatureBatch(features, valid)

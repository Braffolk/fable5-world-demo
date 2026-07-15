"""Robust multiscale ALS reference and connected structural authority."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .evidence import StructuralInputs


@dataclass(frozen=True)
class LocalFit:
    surface_m: np.ndarray
    confidence: np.ndarray
    count: np.ndarray
    detrended_spread_m: np.ndarray
    geometry_score: np.ndarray
    scan_score: np.ndarray
    last_return_fraction: np.ndarray
    overlap_disagreement_m: np.ndarray


@dataclass(frozen=True)
class StructuralReference:
    fine_fit: LocalFit
    broad_fit: LocalFit
    reference_surface_m: np.ndarray
    residual_target_m: np.ndarray
    confidence: np.ndarray
    physical_domain: np.ndarray
    break_network: np.ndarray
    ridge_network: np.ndarray
    valley_network: np.ndarray
    network_strength: np.ndarray
    guide_normal_x: np.ndarray
    guide_normal_y: np.ndarray
    solve_target_m: np.ndarray
    solve_confidence: np.ndarray
    solve_network_strength: np.ndarray
    solve_guide_normal_x: np.ndarray
    solve_guide_normal_y: np.ndarray
    solve_physical_domain: np.ndarray


def _solve_weighted_quadratic(features: np.ndarray, z: np.ndarray, weight: np.ndarray, ridge: float) -> np.ndarray:
    normal = np.einsum("bki,bk,bkj->bij", features, weight, features, optimize=True)
    rhs = np.einsum("bki,bk,bk->bi", features, weight, z, optimize=True)
    diagonal = np.asarray([1e-7, ridge, ridge, ridge, ridge, ridge], dtype=np.float64)
    normal += np.eye(6, dtype=np.float64)[None, :, :] * diagonal[None, :, None]
    return np.linalg.solve(normal, rhs[..., None])[..., 0]


def _moving_quadratic(
    inputs: StructuralInputs,
    point_selector: np.ndarray,
    *,
    radius_m: float,
    neighbors: int,
    minimum_points: int,
    config: dict,
) -> LocalFit:
    px = inputs.point_x[point_selector]
    py = inputs.point_y[point_selector]
    pz = inputs.point_z[point_selector]
    scan = inputs.point_scan_angle_degrees[point_selector]
    last = inputs.point_last_return[point_selector]
    overlap = inputs.point_overlap[point_selector]
    if len(px) < neighbors:
        raise ValueError("insufficient ALS points for local reference")
    tree = cKDTree(np.column_stack([px, py]))
    xx, yy = np.meshgrid(inputs.reference_x, inputs.reference_y)
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    shape = xx.shape
    surface = np.full(len(centers), np.nan, dtype=np.float64)
    confidence = np.zeros(len(centers), dtype=np.float64)
    count_out = np.zeros(len(centers), dtype=np.int16)
    spread_out = np.full(len(centers), np.inf, dtype=np.float64)
    geometry_out = np.zeros(len(centers), dtype=np.float64)
    scan_out = np.zeros(len(centers), dtype=np.float64)
    last_out = np.zeros(len(centers), dtype=np.float64)
    overlap_out = np.full(len(centers), np.inf, dtype=np.float64)
    cfg = config["local_reference"]
    qcfg = config["als_qualification"]
    block_size = 2048

    for start in range(0, len(centers), block_size):
        stop = min(start + block_size, len(centers))
        center = centers[start:stop]
        distance, index = tree.query(
            center,
            k=neighbors,
            distance_upper_bound=radius_m,
            workers=-1,
        )
        if neighbors == 1:
            distance, index = distance[:, None], index[:, None]
        valid = index < len(px)
        safe_index = np.minimum(index, len(px) - 1)
        dx = (px[safe_index] - center[:, 0, None]) / radius_m
        dy = (py[safe_index] - center[:, 1, None]) / radius_m
        values = pz[safe_index]
        features = np.stack(
            [np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=-1
        )
        spatial = np.exp(-0.5 * (distance / (0.65 * radius_m)) ** 2)
        scan_quality = np.exp(
            -(np.abs(scan[safe_index]) / float(qcfg["scan_angle_soft_limit_degrees"])) ** 2
        )
        return_quality = np.where(last[safe_index], 1.0, 0.62)
        overlap_quality = np.where(overlap[safe_index], 0.92, 1.0)
        base_weight = np.where(valid, spatial * scan_quality * return_quality * overlap_quality, 0.0)
        count = valid.sum(axis=1)
        solvable = count >= minimum_points
        weight = base_weight.copy()
        coefficients = np.zeros((len(center), 6), dtype=np.float64)
        for _ in range(int(cfg["irls_iterations"])):
            coefficients = _solve_weighted_quadratic(
                features, values, weight, float(cfg["ridge"])
            )
            residual = values - np.einsum("bki,bi->bk", features, coefficients)
            residual_nan = np.where(valid, residual, np.nan)
            median = np.nanmedian(residual_nan, axis=1)
            mad = 1.4826 * np.nanmedian(
                np.abs(residual_nan - median[:, None]), axis=1
            )
            delta = np.maximum(float(cfg["huber_floor_m"]), 1.5 * np.nan_to_num(mad, nan=1.0))
            huber = np.minimum(1.0, delta[:, None] / np.maximum(np.abs(residual), 1e-8))
            weight = base_weight * huber

        prediction = np.einsum("bki,bi->bk", features, coefficients)
        residual = np.where(valid, values - prediction, np.nan)
        spread = np.nanpercentile(residual, 90, axis=1) - np.nanpercentile(residual, 10, axis=1)
        total = np.maximum(valid.sum(axis=1), 1)
        mean_x = np.sum(np.where(valid, dx, 0.0), axis=1) / total
        mean_y = np.sum(np.where(valid, dy, 0.0), axis=1) / total
        cov_xx = np.sum(np.where(valid, (dx - mean_x[:, None]) ** 2, 0.0), axis=1) / total
        cov_yy = np.sum(np.where(valid, (dy - mean_y[:, None]) ** 2, 0.0), axis=1) / total
        cov_xy = np.sum(np.where(valid, (dx - mean_x[:, None]) * (dy - mean_y[:, None]), 0.0), axis=1) / total
        trace = cov_xx + cov_yy
        root = np.sqrt(np.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, 0.0))
        eig_min = np.maximum(0.5 * (trace - root), 0.0)
        eig_max = np.maximum(0.5 * (trace + root), 1e-8)
        geometry = np.sqrt(np.clip(eig_min / eig_max, 0.0, 1.0))
        scan_median = np.nanmedian(np.where(valid, np.abs(scan[safe_index]), np.nan), axis=1)
        scan_score = np.exp(
            -(scan_median / float(qcfg["scan_angle_soft_limit_degrees"])) ** 2
        )
        last_fraction = np.sum(np.where(valid, last[safe_index], False), axis=1) / total
        return_floor = float(qcfg["minimum_last_return_fraction"])
        return_score = np.clip(last_fraction / max(return_floor, 1e-6), 0.0, 1.0)
        overlap_residual = np.where(valid & overlap[safe_index], residual, np.nan)
        primary_residual = np.where(valid & ~overlap[safe_index], residual, np.nan)
        overlap_count = np.sum(valid & overlap[safe_index], axis=1)
        primary_count = np.sum(valid & ~overlap[safe_index], axis=1)
        overlap_difference = np.where(
            (overlap_count >= 3) & (primary_count >= 3),
            np.abs(np.nanmedian(overlap_residual, axis=1) - np.nanmedian(primary_residual, axis=1)),
            0.0,
        )
        overlap_score = np.exp(-(overlap_difference / 0.18) ** 2)
        count_score = np.clip(count / max(minimum_points * 1.75, 1.0), 0.0, 1.0)
        spread_score = np.exp(
            -0.5 * (spread / float(qcfg["maximum_detrended_spread_m"])) ** 2
        )
        score = (
            count_score
            * np.sqrt(np.clip(geometry, 0.0, 1.0))
            * spread_score
            * scan_score
            * (0.55 + 0.45 * return_score)
            * overlap_score
        )
        score[~solvable] = 0.0
        coefficients[~solvable, 0] = np.nan
        surface[start:stop] = coefficients[:, 0]
        confidence[start:stop] = score
        count_out[start:stop] = np.minimum(count, np.iinfo(np.int16).max)
        spread_out[start:stop] = spread
        geometry_out[start:stop] = geometry
        scan_out[start:stop] = scan_score
        last_out[start:stop] = last_fraction
        overlap_out[start:stop] = overlap_difference

    safe = inputs.safe_reference.ravel()
    confidence[~safe] = 0.0
    surface[~safe] = np.nan
    return LocalFit(
        surface_m=surface.reshape(shape).astype(np.float32),
        confidence=confidence.reshape(shape).astype(np.float32),
        count=count_out.reshape(shape),
        detrended_spread_m=spread_out.reshape(shape).astype(np.float32),
        geometry_score=geometry_out.reshape(shape).astype(np.float32),
        scan_score=scan_out.reshape(shape).astype(np.float32),
        last_return_fraction=last_out.reshape(shape).astype(np.float32),
        overlap_disagreement_m=overlap_out.reshape(shape).astype(np.float32),
    )


def _keep_connected(mask: np.ndarray, minimum_cells: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    if count == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_cells
    keep[0] = False
    return keep[labels]


def _physical_domain(inputs: StructuralInputs, confidence: np.ndarray, config: dict) -> np.ndarray:
    labels, count = ndimage.label(
        inputs.safe_reference,
        structure=ndimage.generate_binary_structure(2, 1),
    )
    adjacency_cells = int(round(
        float(config["structural_network"]["target_adjacency_m"])
        / inputs.reference_pitch_m
    ))
    adjacent = ndimage.binary_dilation(inputs.mapped_face_reference, iterations=adjacency_cells)
    support = confidence >= float(config["local_reference"]["minimum_confidence"])
    result = np.zeros_like(inputs.safe_reference)
    minimum = int(config["structural_network"]["minimum_component_cells"])
    for identity in range(1, count + 1):
        component = labels == identity
        if np.count_nonzero(component & support) >= minimum and np.any(component & adjacent):
            result |= component
    return result


def _network(
    surface: np.ndarray,
    confidence: np.ndarray,
    domain: np.ndarray,
    inputs: StructuralInputs,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filled = np.where(np.isfinite(surface), surface, inputs.c0_reference_m)
    scales = [float(value) / inputs.reference_pitch_m for value in config["structural_network"]["scales_m"]]
    break_strength = np.zeros_like(filled, dtype=np.float64)
    ridge_strength = np.zeros_like(filled, dtype=np.float64)
    valley_strength = np.zeros_like(filled, dtype=np.float64)
    orientation_hxx = np.zeros_like(filled, dtype=np.float64)
    orientation_hyy = np.zeros_like(filled, dtype=np.float64)
    orientation_hxy = np.zeros_like(filled, dtype=np.float64)
    strongest = np.zeros_like(filled, dtype=np.float64)
    previous_slope = None
    for sigma in scales:
        smooth = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
        gy, gx = np.gradient(smooth, inputs.reference_pitch_m)
        slope = np.hypot(gx, gy)
        hyy, hyx = np.gradient(gy, inputs.reference_pitch_m)
        hxy, hxx = np.gradient(gx, inputs.reference_pitch_m)
        hxy = 0.5 * (hxy + hyx)
        trace = hxx + hyy
        root = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy * hxy, 0.0))
        low = 0.5 * (trace - root)
        high = 0.5 * (trace + root)
        scale_factor = sigma * sigma
        candidate_ridge = np.maximum(-low, 0.0) * scale_factor
        candidate_valley = np.maximum(high, 0.0) * scale_factor
        ridge_strength = np.maximum(ridge_strength, candidate_ridge)
        valley_strength = np.maximum(valley_strength, candidate_valley)
        candidate_curvature = np.maximum(candidate_ridge, candidate_valley)
        update = candidate_curvature > strongest
        orientation_hxx[update] = hxx[update]
        orientation_hyy[update] = hyy[update]
        orientation_hxy[update] = hxy[update]
        strongest[update] = candidate_curvature[update]
        if previous_slope is not None:
            candidate_break = np.abs(slope - previous_slope) * sigma
            update = candidate_break > strongest
            orientation_hxx[update] = hxx[update]
            orientation_hyy[update] = hyy[update]
            orientation_hxy[update] = hxy[update]
            strongest[update] = candidate_break[update]
            break_strength = np.maximum(break_strength, candidate_break)
        previous_slope = slope

    eligible = domain & (confidence >= float(config["local_reference"]["minimum_confidence"]))
    percentile = float(config["structural_network"]["strength_percentile"])
    gap = int(round(float(config["structural_network"]["gap_close_m"]) / inputs.reference_pitch_m))
    minimum = int(config["structural_network"]["minimum_component_cells"])

    def connected(value: np.ndarray) -> np.ndarray:
        threshold = float(np.percentile(value[eligible], percentile)) if np.any(eligible) else np.inf
        mask = eligible & (value >= threshold) & (value > 0.0)
        if gap > 0:
            mask = ndimage.binary_closing(mask, structure=np.ones((2 * gap + 1, 2 * gap + 1)))
        return _keep_connected(mask, minimum)

    break_mask = connected(break_strength)
    ridge_mask = connected(ridge_strength)
    valley_mask = connected(valley_strength)
    combined = np.maximum.reduce([break_strength, ridge_strength, valley_strength])
    active_values = combined[break_mask | ridge_mask | valley_mask]
    scale = float(np.percentile(active_values, 95)) if active_values.size else 1.0
    network_strength = np.clip(combined / max(scale, 1e-8), 0.0, 1.0)
    network_strength *= (break_mask | ridge_mask | valley_mask)
    angle = 0.5 * np.arctan2(2.0 * orientation_hxy, orientation_hxx - orientation_hyy)
    nx, ny = np.cos(angle), np.sin(angle)
    return break_mask, ridge_mask, valley_mask, network_strength, nx, ny


def build_reference(
    inputs: StructuralInputs,
    config: dict,
    *,
    include_holdout: bool,
) -> StructuralReference:
    selector = np.ones(len(inputs.point_x), dtype=bool)
    if not include_holdout:
        selector &= ~inputs.point_holdout
    cfg = config["local_reference"]
    fine = _moving_quadratic(
        inputs,
        selector,
        radius_m=float(cfg["fine_radius_m"]),
        neighbors=int(cfg["fine_neighbors"]),
        minimum_points=int(cfg["fine_minimum_points"]),
        config=config,
    )
    broad = _moving_quadratic(
        inputs,
        selector,
        radius_m=float(cfg["broad_radius_m"]),
        neighbors=int(cfg["broad_neighbors"]),
        minimum_points=int(cfg["broad_minimum_points"]),
        config=config,
    )
    fine_valid = np.isfinite(fine.surface_m) & (fine.confidence > 0.0)
    broad_valid = np.isfinite(broad.surface_m) & (broad.confidence > 0.0)
    # Macro authority comes from the 6 m fit. The 3 m fit qualifies independent
    # scale agreement but does not inject its locally granular residual islands.
    reference = np.where(broad_valid, broad.surface_m, np.nan)
    scale_difference = np.abs(fine.surface_m - broad.surface_m)
    agreement = np.exp(-(np.nan_to_num(scale_difference, nan=1.0) / 0.35) ** 2)
    confidence = np.clip(np.sqrt(fine.confidence * broad.confidence) * agreement, 0.0, 1.0)
    domain = _physical_domain(inputs, confidence, config)
    confidence *= domain
    residual = reference - inputs.c0_reference_m
    high = confidence >= float(cfg["minimum_confidence"])
    if np.any(high):
        order = np.argsort(residual[high])
        values = residual[high][order]
        weights = confidence[high][order]
        datum = float(values[np.searchsorted(np.cumsum(weights), 0.5 * np.sum(weights))])
    else:
        datum = 0.0
    residual -= datum
    fine_delta = fine.surface_m - inputs.c0_reference_m - datum
    broad_delta = broad.surface_m - inputs.c0_reference_m - datum
    depression_unqualified = (
        (residual < -0.08)
        & (
            ~fine_valid
            | ~broad_valid
            | (fine_delta >= 0.0)
            | (broad_delta >= 0.0)
            | (scale_difference > float(cfg["depression_scale_agreement_m"]))
        )
    )
    confidence[depression_unqualified] = 0.0
    residual = np.clip(
        residual,
        -float(config["tgv"]["maximum_residual_m"]),
        float(config["tgv"]["maximum_residual_m"]),
    )
    residual[~np.isfinite(residual)] = 0.0
    reference = inputs.c0_reference_m + residual
    break_mask, ridge_mask, valley_mask, strength, nx, ny = _network(
        reference, confidence, domain, inputs, config
    )
    transition_m = float(config["structural_network"]["confidence_transition_m"])
    transition_sigma = transition_m / inputs.reference_pitch_m
    confidence = ndimage.gaussian_filter(confidence, sigma=transition_sigma, mode="nearest")
    distance_inside_m = ndimage.distance_transform_edt(domain) * inputs.reference_pitch_m
    envelope = np.clip(distance_inside_m / max(transition_m, 1e-8), 0.0, 1.0)
    envelope = envelope * envelope * (3.0 - 2.0 * envelope)
    confidence *= domain * envelope
    confidence = np.clip(confidence + 0.35 * strength * envelope, 0.0, 1.0)
    confidence[inputs.hard_reference] = 0.0

    zoom = inputs.reference_pitch_m / inputs.solve_pitch_m
    solve_shape = inputs.c0_solve_m.shape

    def upsample(value: np.ndarray, order: int) -> np.ndarray:
        result = ndimage.zoom(value, zoom=zoom, order=order, mode="nearest", prefilter=order > 1)
        return result[: solve_shape[0], : solve_shape[1]]

    solve_target = upsample(residual, 1).astype(np.float32)
    solve_confidence = upsample(confidence, 1).astype(np.float32)
    solve_strength = upsample(strength, 1).astype(np.float32)
    solve_nx = upsample(nx, 1).astype(np.float32)
    solve_ny = upsample(ny, 1).astype(np.float32)
    normal_length = np.maximum(np.hypot(solve_nx, solve_ny), 1e-8)
    solve_nx /= normal_length
    solve_ny /= normal_length
    solve_domain = upsample(domain.astype(np.float32), 0) >= 0.5
    solve_confidence[inputs.hard_solve] = 0.0
    solve_target[inputs.hard_solve] = 0.0
    return StructuralReference(
        fine_fit=fine,
        broad_fit=broad,
        reference_surface_m=reference.astype(np.float32),
        residual_target_m=residual.astype(np.float32),
        confidence=confidence.astype(np.float32),
        physical_domain=domain,
        break_network=break_mask,
        ridge_network=ridge_mask,
        valley_network=valley_mask,
        network_strength=strength.astype(np.float32),
        guide_normal_x=nx.astype(np.float32),
        guide_normal_y=ny.astype(np.float32),
        solve_target_m=solve_target,
        solve_confidence=solve_confidence,
        solve_network_strength=solve_strength,
        solve_guide_normal_x=solve_nx,
        solve_guide_normal_y=solve_ny,
        solve_physical_domain=solve_domain,
    )

"""Measured feature graph, screened biharmonic solve, and conservative transport."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg

from .evidence import Evidence, MASTER_PITCH_M, SOLVE_PITCH_M


FEATURE_NAMES = ("crest_shoulder", "upper_face_break", "lower_face_bench", "valley_rill", "toe_apron")


@dataclass(frozen=True)
class SourceStatistics:
    normalized_profile: np.ndarray
    normalized_curvature: np.ndarray
    erosion_band_abs_p50_m: float
    erosion_band_abs_p95_m: float
    erosion_band_abs_p99_m: float
    curvature_abs_p95_per_m: float


@dataclass(frozen=True)
class Result:
    master_m: np.ndarray
    macro_m: np.ndarray
    micro_m: np.ndarray
    solve_absolute_m: np.ndarray
    solve_delta_m: np.ndarray
    feature_labels: np.ndarray
    feature_strength: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    measured_target_m: np.ndarray
    measured_confidence: np.ndarray
    source_statistics: SourceStatistics
    solver_info: int


def _normalize(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros_like(value)
    lo, hi = np.percentile(value[valid], [10.0, 96.0])
    return np.clip((value - lo) / max(float(hi - lo), 1.0e-9), 0.0, 1.0)


def source_statistics(evidence: Evidence) -> SourceStatistics:
    source = evidence.source_reconstruction_m
    support = evidence.source_support
    smooth = ndimage.gaussian_filter(source, 2.5)
    gy, gx = np.gradient(smooth, 0.5)
    slope = np.hypot(gx, gy)
    selected = support & (slope >= np.percentile(slope[support], 82.0))
    rows, cols = np.nonzero(selected)
    if len(rows) > 2048:
        step = max(1, len(rows) // 2048)
        rows, cols = rows[::step], cols[::step]
    nx = gx[rows, cols] / np.maximum(slope[rows, cols], 1.0e-9)
    ny = gy[rows, cols] / np.maximum(slope[rows, cols], 1.0e-9)
    offsets = np.linspace(-4.0, 4.0, 33)
    rr = rows[:, None] + ny[:, None] * offsets[None] / 0.5
    cc = cols[:, None] + nx[:, None] * offsets[None] / 0.5
    profiles = ndimage.map_coordinates(source, [rr, cc], order=1, mode="nearest")
    profiles -= profiles[:, :1]
    span = np.maximum(np.ptp(profiles, axis=1), 0.25)
    profiles /= span[:, None]
    profile = np.median(profiles, axis=0)
    curvature = np.gradient(np.gradient(profile, offsets), offsets)

    # The retained source is sampled at 0.5 m. The named 0.25-1.25 m band is
    # evaluated on a 0.25 m interpolation lattice; no independent sub-0.5 m
    # measurement claim is made from those interpolated samples.
    fine = ndimage.zoom(evidence.source_fine_m, 2.0, order=3)
    band = ndimage.gaussian_filter(fine, 0.45) - ndimage.gaussian_filter(fine, 5.0)
    valid = ndimage.zoom(support.astype(np.uint8), 2.0, order=0) > 0
    absolute = np.abs(band[valid])
    return SourceStatistics(
        normalized_profile=profile.astype(np.float32),
        normalized_curvature=curvature.astype(np.float32),
        erosion_band_abs_p50_m=float(np.percentile(absolute, 50.0)),
        erosion_band_abs_p95_m=float(np.percentile(absolute, 95.0)),
        erosion_band_abs_p99_m=float(np.percentile(absolute, 99.0)),
        curvature_abs_p95_per_m=float(np.percentile(np.abs(curvature), 95.0)),
    )


def _upsample(value: np.ndarray, order: int = 3) -> np.ndarray:
    result = ndimage.zoom(value, 4.0, order=order, mode="nearest", prefilter=order > 1)
    return result[:513, :513]


def _keep_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def build_feature_graph(evidence: Evidence) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c0_reference = evidence.c0_solve_m[::4, ::4]
    measured = np.where(np.isfinite(evidence.reference_m), evidence.reference_m, c0_reference)
    confidence = np.clip(evidence.reference_confidence, 0.0, 1.0)
    measured = ndimage.gaussian_filter(measured, 1.15)
    gy, gx = np.gradient(measured, 1.0)
    slope = np.hypot(gx, gy)
    hyy, hyx = np.gradient(gy, 1.0)
    hxy, hxx = np.gradient(gx, 1.0)
    hxy = 0.5 * (hxy + hyx)
    nx, ny = gx / np.maximum(slope, 1.0e-5), gy / np.maximum(slope, 1.0e-5)
    normal_curvature = nx * nx * hxx + 2.0 * nx * ny * hxy + ny * ny * hyy
    tangent_curvature = ny * ny * hxx - 2.0 * nx * ny * hxy + nx * nx * hyy
    supported = confidence >= 0.08
    face = supported & (slope >= np.percentile(slope[supported], 58.0))
    network = _normalize(evidence.network_strength, supported)
    crest_score = _normalize(np.maximum(-normal_curvature, 0.0), supported) * (0.35 + 0.65 * network)
    toe_score = _normalize(np.maximum(normal_curvature, 0.0), supported) * (0.35 + 0.65 * network)
    valley_score = _normalize(np.maximum(tangent_curvature, 0.0), supported) * network * _normalize(slope, supported)
    break_score = _normalize(np.abs(normal_curvature), supported) * _normalize(slope, supported)

    crest = _keep_components((crest_score >= 0.48) & ndimage.binary_dilation(face), 3)
    toe = _keep_components((toe_score >= 0.46) & ndimage.binary_dilation(face, iterations=2), 3)
    valley = _keep_components((valley_score >= 0.30) & ndimage.binary_dilation(face, iterations=4), 3)
    valley = ndimage.binary_closing(valley, structure=np.ones((3, 3)), iterations=1)
    breaks = _keep_components((break_score >= 0.52) & face & ~(crest | toe), 3)
    # Measured elevation orders the paired face features, rather than assigning
    # the two break families from raster direction or a repeated profile.
    median_height = float(np.median(measured[breaks])) if np.any(breaks) else float(np.median(measured[face]))
    upper_break = breaks & (measured >= median_height)
    lower_break = breaks & ~upper_break

    masks_1m = (crest, upper_break, lower_break, valley, toe)
    labels = np.zeros(evidence.c0_solve_m.shape, dtype=np.uint8)
    strengths = np.zeros_like(evidence.c0_solve_m, dtype=np.float64)
    scores = (crest_score, break_score, break_score, valley_score, toe_score)
    for identity, (mask, score) in enumerate(zip(masks_1m, scores), start=1):
        expanded = _upsample(mask.astype(np.float64), order=0) > 0.5
        expanded = ndimage.binary_dilation(expanded, iterations=2 if identity == 4 else 1)
        candidate = expanded & ~evidence.hard_solve
        replace = candidate & (_upsample(score) >= strengths)
        labels[replace] = identity
        strengths[replace] = _upsample(score)[replace]
    measured_solve = _upsample(measured)
    confidence_solve = _upsample(confidence, order=1)
    return labels, strengths, measured_solve, confidence_solve


def _gradient(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(value)
    dy = np.zeros_like(value)
    dx[:, :-1] = (value[:, 1:] - value[:, :-1]) / SOLVE_PITCH_M
    dy[:-1] = (value[1:] - value[:-1]) / SOLVE_PITCH_M
    return dx, dy


def _adjoint(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    result = np.zeros_like(dx)
    result[:, :-1] -= dx[:, :-1] / SOLVE_PITCH_M
    result[:, 1:] += dx[:, :-1] / SOLVE_PITCH_M
    result[:-1] -= dy[:-1] / SOLVE_PITCH_M
    result[1:] += dy[:-1] / SOLVE_PITCH_M
    return result


def solve_graph_surface(
    evidence: Evidence,
    labels: np.ndarray,
    strength: np.ndarray,
    measured: np.ndarray,
    confidence: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, int]:
    c0 = evidence.c0_solve_m
    active = ~evidence.hard_solve
    graph = labels > 0
    target = np.clip(measured - c0, -float(config["maximum_shift_m"]), float(config["maximum_shift_m"]))
    target = ndimage.gaussian_filter(target, float(config["measured_target_sigma_m"]) / SOLVE_PITCH_M)
    tgx, tgy = _gradient(target)
    cgx, cgy = _gradient(ndimage.gaussian_filter(c0, 1.0 / SOLVE_PITCH_M))
    slope = np.hypot(cgx, cgy)
    nx, ny = cgx / np.maximum(slope, 1.0e-6), cgy / np.maximum(slope, 1.0e-6)
    tx, ty = -ny, nx
    graph_weight = graph * confidence * (0.3 + 0.7 * strength)
    normal_weight = float(config["normal_constraint_weight"]) * graph_weight
    tangent_weight = float(config["tangent_constraint_weight"]) * graph_weight
    wxx = normal_weight * nx * nx + tangent_weight * tx * tx
    wyy = normal_weight * ny * ny + tangent_weight * ty * ty
    wxy = normal_weight * nx * ny + tangent_weight * tx * ty
    height_weight = float(config["height_constraint_weight"]) * graph_weight
    screen = float(config["c0_screen_weight"]) + height_weight
    rhs = height_weight * target
    qx = wxx * tgx + wxy * tgy
    qy = wxy * tgx + wyy * tgy
    rhs += _adjoint(qx, qy)
    biharmonic = float(config["biharmonic_weight_m4"])

    def apply(flat: np.ndarray) -> np.ndarray:
        value = np.zeros_like(c0)
        value[active] = flat
        gx, gy = _gradient(value)
        wx = wxx * gx + wxy * gy
        wy = wxy * gx + wyy * gy
        lap = _adjoint(gx, gy)
        lgx, lgy = _gradient(lap)
        result = screen * value + _adjoint(wx, wy) + biharmonic * _adjoint(lgx, lgy)
        return result[active]

    operator = LinearOperator((int(active.sum()), int(active.sum())), matvec=apply, dtype=np.float64)
    solved, info = cg(
        operator,
        rhs[active],
        x0=np.zeros(int(active.sum()), dtype=np.float64),
        rtol=float(config["cg_relative_tolerance"]),
        atol=0.0,
        maxiter=int(config["cg_iterations"]),
    )
    delta = np.zeros_like(c0)
    delta[active] = solved
    delta = np.clip(delta, -float(config["maximum_shift_m"]), float(config["maximum_shift_m"]))
    delta[evidence.hard_solve] = 0.0
    return delta, int(info)


def conservative_transport(
    evidence: Evidence,
    labels: np.ndarray,
    strength: np.ndarray,
    solved_delta: np.ndarray,
    stats: SourceStatistics,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    active = ~evidence.hard_solve
    valley = labels == 4
    toe = labels == 5
    c0 = ndimage.gaussian_filter(evidence.c0_solve_m, 1.0 / SOLVE_PITCH_M)
    gy, gx = np.gradient(c0, SOLVE_PITCH_M)
    slope = np.hypot(gx, gy)
    slope_n = _normalize(slope, active)
    valley_field = ndimage.gaussian_filter(valley.astype(np.float64) * (0.35 + 0.65 * strength), 1.1 / SOLVE_PITCH_M)
    valley_field /= max(float(np.max(valley_field)), 1.0e-9)
    erosion_depth = min(
        float(config["maximum_erosion_depth_m"]),
        max(0.08, float(config["biala_p95_gain"]) * stats.erosion_band_abs_p95_m),
    )
    erosion = erosion_depth * valley_field**1.35 * (0.25 + 0.75 * slope_n)
    headcut = ndimage.binary_dilation((labels == 1) & ndimage.binary_dilation(valley), iterations=3)
    erosion += 0.35 * erosion_depth * ndimage.gaussian_filter(headcut.astype(np.float64), 0.75 / SOLVE_PITCH_M)
    erosion *= active

    toe_field = ndimage.gaussian_filter(toe.astype(np.float64), 2.2 / SOLVE_PITCH_M)
    # Downhill offsets of the measured rill graph seed the apron, so deposition
    # follows this target's routing rather than a repeated lobe primitive.
    downhill = np.zeros_like(toe_field)
    rows, cols = np.nonzero(valley)
    if len(rows):
        dx = -gx[rows, cols] / np.maximum(slope[rows, cols], 1.0e-6)
        dy = -gy[rows, cols] / np.maximum(slope[rows, cols], 1.0e-6)
        for distance_m, weight in ((1.5, 0.45), (3.0, 0.35), (5.0, 0.20)):
            rr = np.clip(np.rint(rows + dy * distance_m / SOLVE_PITCH_M).astype(int), 0, downhill.shape[0] - 1)
            cc = np.clip(np.rint(cols + dx * distance_m / SOLVE_PITCH_M).astype(int), 0, downhill.shape[1] - 1)
            np.add.at(downhill, (rr, cc), weight * strength[rows, cols])
    apron = (toe_field + ndimage.gaussian_filter(downhill, 1.8 / SOLVE_PITCH_M)) * active
    apron *= ndimage.distance_transform_edt(active) >= 1
    total = float(np.sum(erosion))
    if total <= 0.0 or float(np.sum(apron)) <= 0.0:
        raise ValueError("measured graph produced no conservative erosion/deposition support")
    deposition = apron * (total / float(np.sum(apron)))
    erosion[evidence.hard_solve] = 0.0
    deposition[evidence.hard_solve] = 0.0
    return erosion, deposition


def synthesize(evidence: Evidence, config: dict) -> Result:
    stats = source_statistics(evidence)
    labels, strength, measured, confidence = build_feature_graph(evidence)
    delta, info = solve_graph_surface(evidence, labels, strength, measured, confidence, config["solver"])
    erosion, deposition = conservative_transport(
        evidence, labels, strength, delta, stats, config["erosion"]
    )
    process_delta = delta - erosion + deposition
    process_delta[evidence.hard_solve] = 0.0
    macro = ndimage.zoom(process_delta, 4.0, order=3, mode="nearest")[:2049, :2049]
    graph_fine = ndimage.zoom(strength, 4.0, order=3, mode="nearest")[:2049, :2049]
    continuation = ndimage.gaussian_filter(graph_fine, 0.75) - ndimage.gaussian_filter(graph_fine, 4.0)
    valid = ~evidence.hard_master
    p95 = float(np.percentile(np.abs(continuation[valid]), 95.0))
    ceiling = min(
        float(config["micro_continuation"]["absolute_p95_ceiling_m"]),
        stats.erosion_band_abs_p95_m * float(config["micro_continuation"]["biala_p95_fraction"]),
    )
    micro = continuation * (ceiling / max(p95, 1.0e-9))
    distance = ndimage.distance_transform_edt(valid) * MASTER_PITCH_M
    taper = np.clip(distance / float(config["solver"]["outer_collar_m"]), 0.0, 1.0)
    micro *= taper * taper * (3.0 - 2.0 * taper)
    macro[evidence.hard_master] = 0.0
    micro[evidence.hard_master] = 0.0
    master = evidence.c0_master_m + macro + micro
    master[evidence.hard_master] = evidence.c0_master_m[evidence.hard_master]
    return Result(
        master_m=master.astype(np.float32),
        macro_m=macro.astype(np.float32),
        micro_m=micro.astype(np.float32),
        solve_absolute_m=(evidence.c0_solve_m + process_delta).astype(np.float32),
        solve_delta_m=process_delta.astype(np.float32),
        feature_labels=labels,
        feature_strength=strength.astype(np.float32),
        erosion_m=erosion.astype(np.float32),
        deposition_m=deposition.astype(np.float32),
        measured_target_m=measured.astype(np.float32),
        measured_confidence=confidence.astype(np.float32),
        source_statistics=stats,
        solver_info=info,
    )

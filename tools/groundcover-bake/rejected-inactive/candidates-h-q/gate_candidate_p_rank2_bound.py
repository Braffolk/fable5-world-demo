#!/usr/bin/env python3
"""Candidate-P optimistic affine-rank-2 representability bound.

The actual two-event record drives all eight joint sigma4/sigma16 channels by
the same two transition scalars.  This gate grants each phase vertex and
direction triangle its best arbitrary affine rank-2 subspace, arbitrary
held-out coordinates, signed states, and private edges.  It is therefore a
strict superset of Candidate P's representation.  The numerical search is a
large robust oracle rather than a claim of globally optimal L-infinity fitting;
RED is decisive only when its gap is correspondingly unambiguous, while GREEN
merely authorises the actual canonical chord fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import analyze_candidate_h_angular_continuity as common


VERSION = "candidate-p-rank2-capacity-v4"
TRUTH_VERSION = "candidate-p-filtered-direction-truth-v1"
SOURCE = "src/assets/groundcover/calamagrostis-canescens.gcrp"
SPEC = "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-P-CANONICAL-EVENT-FUNCTION-FIELD.md"
N = 64
TRUTH_N = 128
REFERENCE_Y = 0.49255
SOURCE_SCALES = (4, 16)
PHYSICAL_FILTER_WIDTHS_256 = (9.0, 33.0)
FILTER_WIDTHS_AT_TRUTH_N = tuple(width * TRUTH_N / 256 for width in PHYSICAL_FILTER_WIDTHS_256)
TRANSLATIONS_MM = (1.0, 2.0, 3.0, 4.5)
TRANSLATION_AXES = ((1.0, 0.0), (0.0, 1.0), (2**-.5, 2**-.5), (2**-.5, -(2**-.5)))
PHASE_SITES = (("phase-a", 2 / 3, 1 / 3), ("phase-b", 1 / 3, 2 / 3))
EVALUATION_PHASE_SITES = (("vertices", 0.0, 0.0), *PHASE_SITES)
LIMITS = {"coverageP95": .08, "coverageP99": .20, "rgbP95": .06, "rgbP99": .15, "connected": .01, "translationP95": .06}
TOLERANCE4 = np.asarray((.06, .06, .06, .08), dtype=np.float32)
TOLERANCE8 = np.tile(TOLERANCE4, 2)


@dataclass(frozen=True)
class Quad:
    key: str
    elevation_low: float
    elevation_high: float
    azimuth_low: float
    azimuth_high: float


@dataclass(frozen=True)
class Triangle:
    key: str
    quad: str
    half: int
    vertices: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # slope x,z


QUADS = (
    Quad("grazing-worst", .25, 2.0, 67.5, 90.0),
    Quad("standing-worst", 10.0, 18.0, 247.5, 270.0),
)
TRAIN_BARY = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (2 / 3, 1 / 3, 0.0), (0.0, 2 / 3, 1 / 3), (1 / 3, 0.0, 2 / 3),
    (.60, .20, .20),
)
HELDOUT_BARY = (
    ("centroid", 1 / 3, 1 / 3, 1 / 3),
    ("quarter-v0", .50, .25, .25),
    ("shared-diagonal", .50, 0.0, .50),
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def slope_point(elevation: float, azimuth: float) -> tuple[float, float]:
    magnitude = 1 / math.tan(math.radians(elevation))
    return magnitude * math.cos(math.radians(azimuth)), magnitude * math.sin(math.radians(azimuth))


def from_slope(value: tuple[float, float]) -> tuple[float, float]:
    magnitude = math.hypot(*value)
    elevation = math.degrees(math.atan2(1.0, magnitude))
    azimuth = math.degrees(math.atan2(value[1], value[0])) % 360.0
    return elevation, azimuth


def triangles() -> list[Triangle]:
    result = []
    for quad in QUADS:
        q = (
            slope_point(quad.elevation_low, quad.azimuth_low),
            slope_point(quad.elevation_low, quad.azimuth_high),
            slope_point(quad.elevation_high, quad.azimuth_high),
            slope_point(quad.elevation_high, quad.azimuth_low),
        )
        result.append(Triangle(f"{quad.key}-t0", quad.key, 0, (q[0], q[1], q[2])))
        result.append(Triangle(f"{quad.key}-t1", quad.key, 1, (q[0], q[2], q[3])))
    return result


def bary_direction(triangle: Triangle, bary: tuple[float, float, float]) -> tuple[float, float]:
    value = tuple(sum(bary[i] * triangle.vertices[i][axis] for i in range(3)) for axis in range(2))
    return from_slope(value)


def score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    a = np.abs(prediction[..., 3] - truth[..., 3])
    rgb = np.max(np.abs(prediction[..., :3] - truth[..., :3]), axis=2)
    exceed = (a > LIMITS["coverageP99"]) | (rgb > LIMITS["rgbP99"])
    connected = common.largest_periodic_component(exceed) / exceed.size
    aq = {name: float(np.quantile(a, fraction)) for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99))}
    rq = {name: float(np.quantile(rgb, fraction)) for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99))}
    checks = {"coverageP95": aq["p95"] <= .08, "coverageP99": aq["p99"] <= .20, "rgbP95": rq["p95"] <= .06, "rgbP99": rq["p99"] <= .15, "connected": connected < .01}
    return {"green": all(checks.values()), "coverage": aq, "premulRgb": rq, "largestConnectedFraction": connected, "checks": checks}


def translation_score(base_t: np.ndarray, moved_t: np.ndarray, base_p: np.ndarray, moved_p: np.ndarray) -> dict[str, Any]:
    excess = (moved_p - base_p) - (moved_t - base_t)
    a = np.abs(excess[..., 3])
    rgb = np.max(np.abs(excess[..., :3]), axis=2)
    exceed = (a > .20) | (rgb > .15)
    ap95 = float(np.quantile(a, .95))
    rp95 = float(np.quantile(rgb, .95))
    connected = common.largest_periodic_component(exceed) / exceed.size
    return {"green": max(ap95, rp95) <= .06 and connected < .01, "coverageP95": ap95, "premulRgbP95": rp95, "largestConnectedFraction": connected}


def _pca_plane(samples: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return one affine rank-2 plane for every phase vertex."""
    if weights is None:
        mean = np.mean(samples, axis=0)
        centered = samples - mean[None]
        covariance = np.einsum("shwi,shwj->hwij", centered, centered, optimize=True)
    else:
        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-12)
        mean = np.einsum("shw,shwi->hwi", weights, samples, optimize=True) / weight_sum[..., None]
        centered = samples - mean[None]
        covariance = np.einsum("shw,shwi,shwj->hwij", weights, centered, centered, optimize=True)
    _, vectors = np.linalg.eigh(covariance)
    return mean.astype(np.float32), vectors[..., -2:].astype(np.float32)


def _solve_weighted_coordinates(
    scaled_values: np.ndarray,
    scaled_basis: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    a0 = scaled_basis[..., 0]
    a1 = scaled_basis[..., 1]
    g00 = np.einsum("shwi,hwi->shw", weights, a0 * a0, optimize=True)
    g01 = np.einsum("shwi,hwi->shw", weights, a0 * a1, optimize=True)
    g11 = np.einsum("shwi,hwi->shw", weights, a1 * a1, optimize=True)
    h0 = np.einsum("shwi,hwi,shwi->shw", weights, a0, scaled_values, optimize=True)
    h1 = np.einsum("shwi,hwi,shwi->shw", weights, a1, scaled_values, optimize=True)
    ridge = 1e-7 * (g00 + g11 + 1)
    g00 += ridge
    g11 += ridge
    determinant = np.maximum(g00 * g11 - g01 * g01, 1e-12)
    z0 = (h0 * g11 - h1 * g01) / determinant
    z1 = (h1 * g00 - h0 * g01) / determinant
    return np.stack((z0, z1), axis=3).astype(np.float32)


def _optimise_coordinates(
    samples: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    tolerance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine the two plane coordinates toward weighted L-infinity."""
    scaled_values = (samples - origin[None]) / tolerance
    scaled_basis = basis / tolerance[:, None]
    weights = np.ones_like(scaled_values, dtype=np.float32)
    coordinates = np.zeros((*samples.shape[:3], 2), dtype=np.float32)
    for power in (2, 8, 24):
        coordinates = _solve_weighted_coordinates(scaled_values, scaled_basis, weights)
        scaled_prediction = np.einsum("hwik,shwk->shwi", scaled_basis, coordinates, optimize=True)
        residual = scaled_prediction - scaled_values
        magnitude = np.clip(np.abs(residual) + .02, 1e-3, 8)
        weights = magnitude ** (power - 2)
        weights /= np.maximum(np.max(weights, axis=3, keepdims=True), 1e-12)
    coordinates = _solve_weighted_coordinates(scaled_values, scaled_basis, weights)
    prediction = origin[None] + np.einsum("hwik,shwk->shwi", basis, coordinates, optimize=True)
    residual = prediction - samples
    return prediction.astype(np.float32), residual.astype(np.float32), coordinates


def _normalised_linf(
    samples: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    tolerance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction, residual, coordinates = _optimise_coordinates(samples, origin, basis, tolerance)
    objective = np.max(np.abs(residual) / tolerance, axis=(0, 3))
    return objective.astype(np.float32), residual, coordinates


def _scaled_pca_plane(samples: np.ndarray, tolerance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin, basis = _pca_plane(samples / tolerance)
    return (origin * tolerance).astype(np.float32), (basis * tolerance[:, None]).astype(np.float32)


def _triple_plane(
    samples: np.ndarray,
    i: int,
    j: int,
    k: int,
    tolerance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    working = samples if tolerance is None else samples / tolerance
    origin = working[i]
    u = working[j] - origin
    u /= np.maximum(np.linalg.norm(u, axis=2, keepdims=True), 1e-12)
    v = working[k] - origin
    v -= np.sum(v * u, axis=2, keepdims=True) * u
    v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-12)
    basis = np.stack((u, v), axis=3)
    if tolerance is not None:
        origin = origin * tolerance
        basis = basis * tolerance[:, None]
    return origin.astype(np.float32), basis.astype(np.float32)


def _lift_sigma4_plane(
    samples: np.ndarray,
    origin4: np.ndarray,
    basis4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, coordinates = _optimise_coordinates(samples[..., :4], origin4, basis4, TOLERANCE4)
    ones = np.ones((*coordinates.shape[:3], 1), dtype=np.float64)
    design = np.concatenate((ones, coordinates.astype(np.float64)), axis=3)
    gram = np.einsum("shwi,shwj->hwij", design, design, optimize=True)
    rhs = np.einsum("shwi,shwj->hwij", design, samples[..., 4:].astype(np.float64), optimize=True)
    # Triple-supported candidate planes may expose an arbitrarily scaled or
    # rank-deficient coordinate gauge.  Normalize every local 3x3 system
    # before its relative ridge; a fixed float32 ridge can disappear beside a
    # huge coordinate Gram and incorrectly raise Singular matrix.  This is an
    # auxiliary favourable regression, so the minimum-norm regularized limit
    # is the intended value.
    gram_scale = np.maximum(np.max(np.abs(gram), axis=(2, 3), keepdims=True), 1.0)
    normalized_gram = gram / gram_scale
    normalized_rhs = rhs / gram_scale
    normalized_gram += np.eye(3, dtype=np.float64)[None, None] * 1e-10
    try:
        coefficients = np.linalg.solve(normalized_gram, normalized_rhs)
    except np.linalg.LinAlgError:
        coefficients = np.einsum(
            "hwij,hwjk->hwik",
            np.linalg.pinv(normalized_gram, rcond=1e-12),
            normalized_rhs,
            optimize=True,
        )
    origin16 = coefficients[..., 0, :]
    basis16 = np.transpose(coefficients[..., 1:, :], (0, 1, 3, 2))
    origin = np.concatenate((origin4, origin16), axis=2)
    basis = np.concatenate((basis4, basis16), axis=2)
    return origin.astype(np.float32), basis.astype(np.float32)


def fit_rank2_robust(samples: np.ndarray) -> dict[str, Any]:
    """All-sample robust joint, sigma4, and sigma4-lifted rank-2 oracles."""
    sample_count = samples.shape[0]

    def empty(channel_count: int) -> dict[str, np.ndarray | int]:
        return {
            "objective": np.full(samples.shape[1:3], np.inf, dtype=np.float32),
            "origin": np.zeros((*samples.shape[1:3], channel_count), dtype=np.float32),
            "basis": np.zeros((*samples.shape[1:3], channel_count, 2), dtype=np.float32),
            "candidateCount": 0,
        }

    joint = empty(8)
    sigma4 = empty(4)
    lifted = empty(8)

    def update(target: dict[str, Any], values: np.ndarray, origin: np.ndarray, basis: np.ndarray, tolerance: np.ndarray) -> None:
        objective, _, _ = _normalised_linf(values, origin, basis, tolerance)
        take = objective < target["objective"]
        target["objective"] = np.where(take, objective, target["objective"])
        target["origin"] = np.where(take[..., None], origin, target["origin"])
        target["basis"] = np.where(take[..., None, None], basis, target["basis"])
        target["candidateCount"] += 1

    def consider_joint(origin: np.ndarray, basis: np.ndarray) -> None:
        update(joint, samples, origin, basis, TOLERANCE8)

    def consider_sigma4(origin: np.ndarray, basis: np.ndarray) -> None:
        update(sigma4, samples[..., :4], origin, basis, TOLERANCE4)
        lifted_origin, lifted_basis = _lift_sigma4_plane(samples, origin, basis)
        update(lifted, samples, lifted_origin, lifted_basis, TOLERANCE8)
        consider_joint(lifted_origin, lifted_basis)

    # Raw and threshold-normalized PCA candidates.
    raw_origin, raw_basis = _pca_plane(samples)
    consider_joint(raw_origin, raw_basis)
    normal_origin, normal_basis = _scaled_pca_plane(samples, TOLERANCE8)
    consider_joint(normal_origin, normal_basis)
    raw4_origin, raw4_basis = _pca_plane(samples[..., :4])
    consider_sigma4(raw4_origin, raw4_basis)
    normal4_origin, normal4_basis = _scaled_pca_plane(samples[..., :4], TOLERANCE4)
    consider_sigma4(normal4_origin, normal4_basis)

    # Threshold-aware reweighted candidates.
    for base_origin, base_basis, tolerance, values, consumer in (
        (raw_origin, raw_basis, TOLERANCE8, samples, consider_joint),
        (normal_origin, normal_basis, TOLERANCE8, samples, consider_joint),
        (raw4_origin, raw4_basis, TOLERANCE4, samples[..., :4], consider_sigma4),
        (normal4_origin, normal4_basis, TOLERANCE4, samples[..., :4], consider_sigma4),
    ):
        origin, basis = base_origin, base_basis
        for _ in range(4):
            _, residual, _ = _normalised_linf(values, origin, basis, tolerance)
            per_sample = np.max(np.abs(residual) / tolerance, axis=3)
            weights = np.maximum(per_sample, 1e-4) ** 2
            if np.array_equal(tolerance, TOLERANCE8):
                origin, basis = _pca_plane(values / tolerance, weights)
                origin = origin * tolerance
                basis = basis * tolerance[:, None]
            else:
                origin, basis = _pca_plane(values / tolerance, weights)
                origin = origin * tolerance
                basis = basis * tolerance[:, None]
            consumer(origin.astype(np.float32), basis.astype(np.float32))

    # Leave-one-out candidates in both raw and threshold-normalized metrics.
    for omitted in range(sample_count):
        subset = np.delete(samples, omitted, axis=0)
        consider_joint(*_pca_plane(subset))
        consider_joint(*_scaled_pca_plane(subset, TOLERANCE8))
        subset4 = subset[..., :4]
        consider_sigma4(*_pca_plane(subset4))
        consider_sigma4(*_scaled_pca_plane(subset4, TOLERANCE4))

    # Every raw three-observation plane; threshold-normalized PCA, robust,
    # and leave-one-out candidates above cover the complementary metric.
    for i in range(sample_count - 2):
        for j in range(i + 1, sample_count - 1):
            for k in range(j + 1, sample_count):
                consider_joint(*_triple_plane(samples, i, j, k, None))
                consider_sigma4(*_triple_plane(samples[..., :4], i, j, k, None))

    def diagnostics(target: dict[str, Any]) -> dict[str, Any]:
        objective = target["objective"]
        return {
            "candidatePlanesPerPhaseVertex": target["candidateCount"],
            "normalisedLinfP50": float(np.quantile(objective, .50)),
            "normalisedLinfP95": float(np.quantile(objective, .95)),
            "normalisedLinfP99": float(np.quantile(objective, .99)),
            "normalisedLinfMax": float(np.max(objective)),
        }

    return {
        "joint": (joint["origin"], joint["basis"]),
        "sigma4": (sigma4["origin"], sigma4["basis"]),
        "lifted": (lifted["origin"], lifted["basis"]),
        "diagnostics": {
            "joint": diagnostics(joint),
            "sigma4": diagnostics(sigma4),
            "lifted": diagnostics(lifted),
        },
    }


def fit_rank2_scale(
    samples: np.ndarray,
    seed: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Independent one-footprint rank-2 oracle for ablation B."""
    target: dict[str, Any] = {
        "objective": np.full(samples.shape[1:3], np.inf, dtype=np.float32),
        "origin": np.zeros(samples.shape[1:], dtype=np.float32),
        "basis": np.zeros((*samples.shape[1:], 2), dtype=np.float32),
        "candidateCount": 0,
    }

    def consider(origin: np.ndarray, basis: np.ndarray) -> None:
        objective, _, _ = _normalised_linf(samples, origin, basis, TOLERANCE4)
        take = objective < target["objective"]
        target["objective"] = np.where(take, objective, target["objective"])
        target["origin"] = np.where(take[..., None], origin, target["origin"])
        target["basis"] = np.where(take[..., None, None], basis, target["basis"])
        target["candidateCount"] += 1

    raw_origin, raw_basis = _pca_plane(samples)
    normal_origin, normal_basis = _scaled_pca_plane(samples, TOLERANCE4)
    if seed is not None:
        # Independent per-scale rank two is a strict superset of joint rank
        # two.  Include the joint plane restricted to this scale so numerical
        # candidate search can never violate that containment relation.
        consider(*seed)
    consider(raw_origin, raw_basis)
    consider(normal_origin, normal_basis)
    for base_origin, base_basis in ((raw_origin, raw_basis), (normal_origin, normal_basis)):
        origin, basis = base_origin, base_basis
        for _ in range(4):
            _, residual, _ = _normalised_linf(samples, origin, basis, TOLERANCE4)
            per_sample = np.max(np.abs(residual) / TOLERANCE4, axis=3)
            weights = np.maximum(per_sample, 1e-4) ** 2
            origin, basis = _pca_plane(samples / TOLERANCE4, weights)
            origin = (origin * TOLERANCE4).astype(np.float32)
            basis = (basis * TOLERANCE4[:, None]).astype(np.float32)
            consider(origin, basis)
    for omitted in range(samples.shape[0]):
        subset = np.delete(samples, omitted, axis=0)
        consider(*_pca_plane(subset))
        consider(*_scaled_pca_plane(subset, TOLERANCE4))
    for i in range(samples.shape[0] - 2):
        for j in range(i + 1, samples.shape[0] - 1):
            for k in range(j + 1, samples.shape[0]):
                consider(*_triple_plane(samples, i, j, k, None))

    objective = target["objective"]
    return {
        "origin": target["origin"],
        "basis": target["basis"],
        "diagnostics": {
            "candidatePlanesPerPhaseVertex": target["candidateCount"],
            "normalisedLinfP50": float(np.quantile(objective, .50)),
            "normalisedLinfP95": float(np.quantile(objective, .95)),
            "normalisedLinfP99": float(np.quantile(objective, .99)),
            "normalisedLinfMax": float(np.max(objective)),
        },
    }


def project(value: np.ndarray, origin: np.ndarray, basis: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    prediction, _, _ = _optimise_coordinates(value[None], origin, basis, tolerance)
    return prediction[0]


def phase_interpolate(vertices: np.ndarray, ox: float, oz: float) -> np.ndarray:
    """Piecewise-linear periodic phase FE at grid offset (ox,oz)."""
    y, x = np.indices((N, N))
    qx = x + ox
    qz = y + oz
    x0 = np.floor(qx).astype(np.int64)
    z0 = np.floor(qz).astype(np.int64)
    u = qx - x0
    v = qz - z0
    x1 = (x0 + 1) % N
    z1 = (z0 + 1) % N
    x0 %= N
    z0 %= N
    a = vertices[z0, x0]
    b = vertices[z0, x1]
    c = vertices[z1, x1]
    d = vertices[z1, x0]
    lower = v <= u
    out0 = (1 - u)[..., None] * a + (u - v)[..., None] * b + v[..., None] * c
    out1 = (1 - v)[..., None] * a + u[..., None] * c + (v - u)[..., None] * d
    return np.where(lower[..., None], out0, out1).astype(np.float32)


def fit_phase_coefficients(
    observations: list[np.ndarray],
    offsets: list[tuple[float, float]],
    difference_pairs: list[tuple[int, int, float]],
) -> np.ndarray:
    """Globally optimal periodic triangular-FE coefficients in least squares.

    For each fixed fractional offset the triangular interpolation operator is
    circular and shift-invariant.  Its response to a phase-grid impulse gives
    the exact Fourier multiplier H_m.  Every frequency therefore has the
    scalar normal-equation solution
        C_hat = sum(conj(H_m) Y_m) / sum(abs(H_m)^2).
    Coefficients are deliberately not constrained to equal truth at vertices.
    """
    if len(observations) != len(offsets) or not observations:
        raise ValueError("phase observations and offsets must be non-empty and aligned")
    numerator = np.zeros((N, N, observations[0].shape[2]), dtype=np.complex128)
    denominator = np.zeros((N, N), dtype=np.float64)
    for truth, (ox, oz) in zip(observations, offsets, strict=True):
        transfer = phase_transfer(ox, oz)
        transformed_truth = np.fft.fft2(truth, axes=(0, 1))
        numerator += np.conj(transfer)[..., None] * transformed_truth
        denominator += np.abs(transfer) ** 2
    for base_index, moved_index, weight in difference_pairs:
        transfer = (
            phase_transfer(*offsets[moved_index])
            - phase_transfer(*offsets[base_index])
        )
        target = observations[moved_index] - observations[base_index]
        numerator += weight * np.conj(transfer)[..., None] * np.fft.fft2(target, axes=(0, 1))
        denominator += weight * np.abs(transfer) ** 2
    coefficients = np.fft.ifft2(
        numerator / np.maximum(denominator, 1e-12)[..., None],
        axes=(0, 1),
    ).real
    return coefficients.astype(np.float32)


def phase_transfer(ox: float, oz: float) -> np.ndarray:
    impulse = np.zeros((N, N, 1), dtype=np.float32)
    impulse[0, 0, 0] = 1
    return np.fft.fft2(phase_interpolate(impulse, ox, oz)[..., 0])


def coupled_phase_refinement(
    observations: np.ndarray,
    offsets: list[tuple[float, float]],
    coordinates: np.ndarray,
    initial_origin: np.ndarray,
    initial_basis: np.ndarray,
    difference_pairs: list[tuple[int, int, float]],
    iterations: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One coupled phase×direction linear solve with fixed direction coords.

    observations is [S,M,H,W,C], coordinates is [S,H,W,2].  The unknown at
    every phase vertex is the affine state theta=(origin,basis0,basis1).  The
    periodic phase operators are applied exactly in Fourier space; conjugate
    gradient solves the coupled normal equations without forming a matrix.
    """
    original_coordinates = coordinates.astype(np.float64)
    original_origin = initial_origin.astype(np.float64)
    original_basis = initial_basis.astype(np.float64)

    # The affine rank-2 chart has a free per-vertex coordinate gauge.  Leaving
    # that gauge arbitrary makes the coupled normal equations needlessly ill
    # conditioned.  Whiten the S direction coordinates independently at every
    # phase vertex, while conjugating the affine field exactly:
    #
    #     z = m + z' L
    #     a + B z = (a + B m) + (B L) z'.
    #
    # Therefore this changes no represented coefficient field; it only fixes
    # the solver gauge before PCG.
    coordinate_mean = np.mean(original_coordinates, axis=0)
    centred_coordinates = original_coordinates - coordinate_mean[None]
    covariance = np.einsum(
        "shwi,shwj->hwij",
        centred_coordinates,
        centred_coordinates,
        optimize=True,
    ) / max(original_coordinates.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_floor = np.maximum(np.max(eigenvalues, axis=-1, keepdims=True) * 1e-8, 1e-12)
    safe_eigenvalues = np.maximum(eigenvalues, eigen_floor)
    inverse_sqrt = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        1.0 / np.sqrt(safe_eigenvalues),
        eigenvectors,
        optimize=True,
    )
    coordinate_sqrt = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        np.sqrt(safe_eigenvalues),
        eigenvectors,
        optimize=True,
    )
    coordinates = np.einsum(
        "shwi,hwij->shwj",
        centred_coordinates,
        inverse_sqrt,
        optimize=True,
    )
    initial_origin = (
        original_origin
        + np.einsum("hwck,hwk->hwc", original_basis, coordinate_mean, optimize=True)
    )
    initial_basis = np.einsum(
        "hwck,hwkj->hwcj",
        original_basis,
        coordinate_sqrt,
        optimize=True,
    )

    transfers = np.stack([phase_transfer(*offset) for offset in offsets])
    phase_normal = np.sum(np.abs(transfers) ** 2, axis=0)
    for base_index, moved_index, weight in difference_pairs:
        difference_transfer = transfers[moved_index] - transfers[base_index]
        phase_normal += weight * np.abs(difference_transfer) ** 2
    rhs_per_direction = []
    for direction_index in range(observations.shape[0]):
        transformed = np.fft.fft2(observations[direction_index], axes=(1, 2))
        rhs_hat = np.sum(np.conj(transfers)[..., None] * transformed, axis=0)
        for base_index, moved_index, weight in difference_pairs:
            difference_transfer = transfers[moved_index] - transfers[base_index]
            difference_truth = (
                observations[direction_index, moved_index]
                - observations[direction_index, base_index]
            )
            rhs_hat += (
                weight
                * np.conj(difference_transfer)[..., None]
                * np.fft.fft2(difference_truth, axes=(0, 1))
            )
        rhs_per_direction.append(np.fft.ifft2(rhs_hat, axes=(0, 1)).real.astype(np.float64))

    right = np.zeros((N, N, 3, observations.shape[-1]), dtype=np.float64)
    for direction_index, rhs in enumerate(rhs_per_direction):
        z = coordinates[direction_index]
        right[..., 0, :] += rhs
        right[..., 1, :] += rhs * z[..., 0, None]
        right[..., 2, :] += rhs * z[..., 1, None]

    ridge = 1e-6
    features = np.concatenate((
        np.ones((*coordinates.shape[:3], 1), dtype=np.float64),
        coordinates,
    ), axis=3)
    phase_diagonal = float(np.mean(phase_normal))
    block_diagonal = (
        phase_diagonal * np.einsum("shwi,shwj->hwij", features, features, optimize=True)
        + np.eye(3, dtype=np.float64)[None, None] * ridge
    )

    def precondition(value: np.ndarray) -> np.ndarray:
        return np.linalg.solve(block_diagonal, value)

    def apply(theta: np.ndarray) -> np.ndarray:
        result = ridge * theta
        for direction_index in range(observations.shape[0]):
            z = coordinates[direction_index]
            coefficient = (
                theta[..., 0, :]
                + theta[..., 1, :] * z[..., 0, None]
                + theta[..., 2, :] * z[..., 1, None]
            )
            normal_coefficient = np.fft.ifft2(
                phase_normal[..., None] * np.fft.fft2(coefficient, axes=(0, 1)),
                axes=(0, 1),
            ).real
            result[..., 0, :] += normal_coefficient
            result[..., 1, :] += normal_coefficient * z[..., 0, None]
            result[..., 2, :] += normal_coefficient * z[..., 1, None]
        return result

    theta = np.concatenate(
        (initial_origin[..., None, :], np.transpose(initial_basis, (0, 1, 3, 2))),
        axis=2,
    ).astype(np.float64)

    def actual_rmse(value: np.ndarray) -> float:
        squared = 0.0
        count = 0.0
        for direction_index in range(observations.shape[0]):
            z = coordinates[direction_index]
            coefficient = (
                value[..., 0, :]
                + value[..., 1, :] * z[..., 0, None]
                + value[..., 2, :] * z[..., 1, None]
            )
            predictions = [
                phase_interpolate(coefficient, *offset)
                for offset in offsets
            ]
            for phase_index, prediction in enumerate(predictions):
                difference = prediction - observations[direction_index, phase_index]
                squared += float(np.sum(difference.astype(np.float64) ** 2))
                count += difference.size
            for base_index, moved_index, weight in difference_pairs:
                difference = (
                    (predictions[moved_index] - predictions[base_index])
                    - (
                        observations[direction_index, moved_index]
                        - observations[direction_index, base_index]
                    )
                )
                squared += weight * float(np.sum(difference.astype(np.float64) ** 2))
                count += weight * difference.size
        return math.sqrt(squared / max(count, 1))

    actual_rmse_initial = actual_rmse(theta)
    residual = right - apply(theta)
    preconditioned = precondition(residual)
    direction = preconditioned.copy()
    residual_norm0 = float(np.sqrt(np.sum(residual.astype(np.float64) ** 2)))
    residual_square = float(np.sum(residual.astype(np.float64) ** 2))
    initial_square = residual_square
    residual_preconditioned = float(np.sum(
        residual.astype(np.float64) * preconditioned.astype(np.float64)
    ))
    completed = 0
    for iteration in range(iterations):
        applied = apply(direction)
        denominator = float(np.sum(direction.astype(np.float64) * applied.astype(np.float64)))
        if denominator <= 1e-20:
            break
        alpha = residual_preconditioned / denominator
        theta += alpha * direction
        next_residual = residual - alpha * applied
        next_square = float(np.sum(next_residual.astype(np.float64) ** 2))
        completed = iteration + 1
        if next_square <= max(1e-20, initial_square * 1e-12):
            residual = next_residual
            residual_square = next_square
            break
        next_preconditioned = precondition(next_residual)
        next_residual_preconditioned = float(np.sum(
            next_residual.astype(np.float64) * next_preconditioned.astype(np.float64)
        ))
        beta = next_residual_preconditioned / max(residual_preconditioned, 1e-30)
        direction = next_preconditioned + beta * direction
        residual = next_residual
        residual_square = next_square
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned

    actual_rmse_final = actual_rmse(theta)
    accepted = actual_rmse_final < actual_rmse_initial
    active_theta = theta if accepted else np.concatenate(
        (initial_origin[..., None, :], np.transpose(initial_basis, (0, 1, 3, 2))),
        axis=2,
    )
    refined = []
    for direction_index in range(observations.shape[0]):
        z = coordinates[direction_index]
        refined.append(
            active_theta[..., 0, :]
            + active_theta[..., 1, :] * z[..., 0, None]
            + active_theta[..., 2, :] * z[..., 1, None]
        )
    diagnostics = {
        "iterations": completed,
        "normalResidualL2Initial": residual_norm0,
        "normalResidualL2Final": float(math.sqrt(residual_square)),
        "normalResidualRatio": float(math.sqrt(residual_square) / max(residual_norm0, 1e-30)),
        "actualWeightedRmseInitial": actual_rmse_initial,
        "actualWeightedRmseFinal": actual_rmse_final,
        "acceptedByActualImageLoss": accepted,
        "whitenedCoordinateCovarianceEigenvalueP01": float(np.quantile(eigenvalues, .01)),
        "whitenedCoordinateCovarianceEigenvalueP99": float(np.quantile(eigenvalues, .99)),
    }
    return np.stack(refined).astype(np.float32), diagnostics


def sample_filtered_truth(page: np.ndarray, ox: float, oz: float) -> np.ndarray:
    """Sample a physically filtered truth page on the 64^2 phase lattice.

    The conversion is derived from texel centres, not fitted: array coordinate
    r*(x+0.5)-0.5 corresponds to physical phase (x+0.5)/64 for
    r=TRUTH_N/64. Filtering happens before this phase resampling.
    """
    y, x = np.indices((N, N), dtype=np.float64)
    ratio = TRUTH_N / N
    qx = ratio * (x + 0.5 + ox) - 0.5
    qz = ratio * (y + 0.5 + oz) - 0.5
    x0 = np.floor(qx).astype(np.int64)
    z0 = np.floor(qz).astype(np.int64)
    fx = (qx - x0)[..., None]
    fz = (qz - z0)[..., None]
    x1 = (x0 + 1) % TRUTH_N
    z1 = (z0 + 1) % TRUTH_N
    x0 %= TRUTH_N
    z0 %= TRUTH_N
    a = page[z0, x0]
    b = page[z0, x1]
    c = page[z1, x1]
    d = page[z1, x0]
    return ((1 - fz) * ((1 - fx) * a + fx * b) + fz * ((1 - fx) * d + fx * c)).astype(np.float32)


def periodic_fractional_box_filter(page: np.ndarray, full_width: float) -> np.ndarray:
    """Exact separable integral of the piecewise-constant periodic page.

    `full_width` is measured in truth texels and may be fractional.  Each
    source texel contributes its exact interval overlap with the centred box;
    the two 1-D integrals therefore preserve the physical 9/256 and 33/256
    tile footprints without rounding their support.
    """
    radius = full_width / 2
    maximum_offset = math.ceil(radius + 0.5)
    weights = []
    for offset in range(-maximum_offset, maximum_offset + 1):
        overlap = max(0.0, min(radius, offset + 0.5) - max(-radius, offset - 0.5))
        if overlap > 0:
            weights.append((offset, overlap))
    assert abs(sum(weight for _, weight in weights) - full_width) < 1e-12
    horizontal = sum(weight * np.roll(page, shift=offset, axis=1) for offset, weight in weights) / full_width
    vertical = sum(weight * np.roll(horizontal, shift=offset, axis=0) for offset, weight in weights) / full_width
    return vertical.astype(np.float32)


def write_qa(path: Path, title: str, truth: np.ndarray, prediction: np.ndarray) -> None:
    error = np.max(np.abs(truth - prediction), axis=2)
    panels = []
    for value in (truth, prediction):
        panels.append(np.rint(np.clip(value[..., :3] + .12 * (1 - value[..., 3:4]), 0, 1) * 255).astype(np.uint8))
    heat = np.zeros((*error.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(np.clip(error / .20, 0, 1) * 255).astype(np.uint8)
    heat[..., 1] = np.rint(np.clip(error / .20, 0, 1) * 170).astype(np.uint8)
    panels.append(heat)
    scale = 4
    header = 34
    image = Image.new("RGB", (N * scale * 3, N * scale + header), (16, 16, 16))
    ImageDraw.Draw(image).text((8, 6), title, fill=(255, 255, 255))
    for index, panel in enumerate(panels):
        image.paste(Image.fromarray(panel).resize((N * scale, N * scale), Image.Resampling.NEAREST), (index * N * scale, header))
    image.save(path)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_path = root / SOURCE
    spec_path = root / SPEC
    script_path = Path(__file__).resolve()
    exporter = root / "tools/groundcover-bake/export-candidate-o-truth.ts"
    source_sha = sha(source_path.read_bytes())
    source_size = 0.52
    tri = triangles()

    direction_records = []
    for triangle in tri:
        for index, bary in enumerate(TRAIN_BARY):
            e, a = bary_direction(triangle, bary)
            direction_records.append({"key": f"{triangle.key}-train{index}", "triangle": triangle.key, "kind": "train", "elevation": e, "azimuth": a})
        for name, *coords in HELDOUT_BARY:
            bary = tuple(coords)
            e, a = bary_direction(triangle, bary)  # type: ignore[arg-type]
            direction_records.append({"key": f"{triangle.key}-{name}", "triangle": triangle.key, "kind": "heldout", "site": name, "elevation": e, "azimuth": a})

    phase_offsets: dict[str, tuple[float, float]] = {}
    for phase_name, ox, oz in EVALUATION_PHASE_SITES:
        phase_offsets[phase_name] = (ox, oz)
        for millimetres in TRANSLATIONS_MM:
            for axis_index, axis in enumerate(TRANSLATION_AXES):
                cells = millimetres * .001 / source_size * N
                phase_offsets[f"{phase_name}-t{millimetres:g}mm-d{axis_index}"] = (ox + cells * axis[0], oz + cells * axis[1])

    # Stable algebra check for the Fourier-domain triangular-FE inverse.
    py, px = np.indices((N, N), dtype=np.float32)
    phase_probe = (
        .4
        + .2 * np.sin(2 * np.pi * (3 * px + 5 * py) / N)
        + .1 * np.cos(2 * np.pi * (7 * px - 2 * py) / N)
    )[..., None].astype(np.float32)
    fitted_phase_names = list(phase_offsets)
    probe_offsets = [phase_offsets[name] for name in fitted_phase_names]
    fitted_phase_indices = {name: index for index, name in enumerate(fitted_phase_names)}
    difference_pairs = []
    for phase_name, _, _ in EVALUATION_PHASE_SITES:
        for millimetres in TRANSLATIONS_MM:
            for axis_index in range(len(TRANSLATION_AXES)):
                moved_name = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                difference_pairs.append((
                    fitted_phase_indices[phase_name],
                    fitted_phase_indices[moved_name],
                    1.0,
                ))
    recovered_probe = fit_phase_coefficients(
        [phase_interpolate(phase_probe, *offset) for offset in probe_offsets],
        probe_offsets,
        difference_pairs,
    )
    phase_solver_self_error = float(np.max(np.abs(recovered_probe - phase_probe)))
    if phase_solver_self_error > 2e-5:
        raise RuntimeError(f"phase Fourier solver self-check failed: {phase_solver_self_error}")

    truth_recipe = {
        "version": TRUTH_VERSION,
        "sourceSha256": source_sha,
        "truthExporterSha256": sha(exporter.read_bytes()),
        "resolution": TRUTH_N,
        "referenceY": REFERENCE_Y,
        "physicalFilterFullWidthsAt256": PHYSICAL_FILTER_WIDTHS_256,
        "fractionalFilterFullWidthsAtTruthResolution": FILTER_WIDTHS_AT_TRUTH_N,
        "directions": [
            {"key": value["key"], "elevation": value["elevation"], "azimuth": value["azimuth"]}
            for value in direction_records
        ],
    }
    truth_recipe_sha = canonical_sha(truth_recipe)
    truth_cache = root / "data/work/groundcover-candidate-p-filtered-truth" / source_sha[:16] / truth_recipe_sha[:16]
    truth_manifest_path = truth_cache / "manifest.json"

    recipe = {
        "version": VERSION,
        "sourceSha256": source_sha,
        "analyzerSha256": sha(script_path.read_bytes()),
        "truthExporterSha256": sha(exporter.read_bytes()),
        "filteredTruthRecipeSha256": truth_recipe_sha,
        "specSha256": sha(spec_path.read_bytes()),
        "hemisphereTopology": "240 used triangles: seven 16-sector annuli x2 plus sixteen degenerate 75-pole caps; allocation has sixteen padding records",
        "targetTriangles": [triangle.key for triangle in tri],
        "phaseVertices": [N, N],
        "physicalTruthResolution": [TRUTH_N, TRUTH_N],
        "trainingBarycentrics": TRAIN_BARY,
        "heldoutBarycentrics": HELDOUT_BARY,
        "phaseSites": EVALUATION_PHASE_SITES,
        "translationDifferenceObservations": {"count": len(difference_pairs), "weight": 1.0},
        "translationsMillimetres": TRANSLATIONS_MM,
        "translationAxes": TRANSLATION_AXES,
        "sourceScales": SOURCE_SCALES,
        "physicalFilterFullWidthsAt256": PHYSICAL_FILTER_WIDTHS_256,
        "fractionalFilterFullWidthsAtTruthResolution": FILTER_WIDTHS_AT_TRUTH_N,
        "filterLaw": "separable periodic exact interval-overlap integral of a piecewise-constant 128x128 point page, followed by bilinear sampling at 64x64 phase sites",
        "limits": LIMITS,
        "phaseCoefficientFit": "global periodic triangular-FE least squares over every scored phase offset and translation, solved independently per direction/channel in Fourier space",
        "phaseSolverSelfCheckMaxError": phase_solver_self_error,
        "coordinateFit": "threshold-normalized vectorized high-p IRLS toward L-infinity for every direction's two coordinates",
        "relaxation": "all scored directions and phases are leaked into independent robust affine rank-2 oracles; arbitrary coordinates; signed states; private edges; no chord, ordering, shared-edge, or nonnegativity constraint",
    }
    recipe_sha = canonical_sha(recipe)
    output = root / "data/work/groundcover-candidate-p-rank2-bound" / source_sha[:16] / recipe_sha[:16]
    qa_root = output / "qa"
    output.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)

    config = {
        "source": SOURCE, "resolution": TRUTH_N, "referenceY": REFERENCE_Y, "radii": [0],
        "entries": [{"key": f"entry{index:04d}", "elevationDegrees": value["elevation"], "azimuthDegrees": value["azimuth"], "phaseOffsetX": 0.0, "phaseOffsetZ": 0.0} for index, value in enumerate(direction_records)],
    }
    with tempfile.TemporaryDirectory(prefix="candidate-p-rank2-") as temporary:
        temp = Path(temporary)
        config_path = temp / "config.json"
        truth_root = temp / "truth"
        if not truth_manifest_path.exists():
            config_path.write_text(json.dumps(config))
            subprocess.run(["node", "--import", "tsx", str(exporter), "--config", str(config_path), "--output", str(truth_root)], cwd=root, check=True)
            truth_cache.mkdir(parents=True, exist_ok=True)
            page_records = []
            for index, record in enumerate(direction_records):
                point_page = np.fromfile(
                    truth_root / f"entry{index:04d}-r0.f32",
                    dtype=np.float32,
                ).reshape(TRUTH_N, TRUTH_N, 4)
                for scale, width in zip(SOURCE_SCALES, FILTER_WIDTHS_AT_TRUTH_N, strict=True):
                    filtered = periodic_fractional_box_filter(point_page, width)
                    path = truth_cache / f"entry{index:04d}-sigma{scale}.f32"
                    filtered.tofile(path)
                    page_records.append({
                        "direction": record["key"],
                        "scale": scale,
                        "file": path.name,
                        "sha256": sha(path.read_bytes()),
                        "shape": [TRUTH_N, TRUTH_N, 4],
                    })
            manifest = {
                "schema": TRUTH_VERSION,
                "recipe": truth_recipe,
                "recipeSha256": truth_recipe_sha,
                "pages": page_records,
            }
            truth_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        truth_manifest = json.loads(truth_manifest_path.read_text())
        if truth_manifest["recipeSha256"] != truth_recipe_sha:
            raise RuntimeError("filtered truth cache recipe mismatch")
        source_pages = []
        for index in range(len(direction_records)):
            pages = []
            for scale in SOURCE_SCALES:
                path = truth_cache / f"entry{index:04d}-sigma{scale}.f32"
                data = path.read_bytes()
                expected = next(
                    value["sha256"] for value in truth_manifest["pages"]
                    if value["file"] == path.name
                )
                if sha(data) != expected:
                    raise RuntimeError(f"filtered truth cache hash mismatch: {path}")
                pages.append(np.frombuffer(data, dtype=np.float32).reshape(TRUTH_N, TRUTH_N, 4))
            source_pages.append(tuple(pages))
        direction_indices = {value["key"]: index for index, value in enumerate(direction_records)}
        sampled_cache: dict[tuple[str, str], np.ndarray] = {}

        def joint(key: str, phase: str) -> np.ndarray:
            cache_key = (key, phase)
            if cache_key not in sampled_cache:
                index = direction_indices[key]
                ox, oz = phase_offsets[phase]
                sampled_cache[cache_key] = np.concatenate((
                    sample_filtered_truth(source_pages[index][0], ox, oz),
                    sample_filtered_truth(source_pages[index][1], ox, oz),
                ), axis=2)
            return sampled_cache[cache_key]

        fitted_phase_offsets = [phase_offsets[name] for name in fitted_phase_names]
        phase_coefficients = {
            record["key"]: fit_phase_coefficients(
                [joint(record["key"], name) for name in fitted_phase_names],
                fitted_phase_offsets,
                difference_pairs,
            )
            for record in direction_records
        }
        print(f"[candidate-p-v4] phase coefficients fitted for {len(direction_records)} directions")

        evaluations = []
        baseline_evaluations = []
        phase_only_evaluations = []
        lifted_evaluations = []
        baseline_lifted_evaluations = []
        sigma4_evaluations = []
        baseline_scale_evaluations = []
        qa_candidates = []
        fit_diagnostics = []
        relaxed_predictions: dict[tuple[str, str, str], np.ndarray] = {}
        for triangle in tri:
            print(f"[candidate-p-v4] fitting {triangle.key}")
            direction_keys = [
                *[f"{triangle.key}-train{index}" for index in range(len(TRAIN_BARY))],
                *[f"{triangle.key}-{name}" for name, *_ in HELDOUT_BARY],
            ]
            initial_coefficients = np.stack([phase_coefficients[key] for key in direction_keys])
            # Intentional data leakage: representability, not generalisation,
            # is the question.  Candidate P's cook could see the full angular
            # function, so the impossibility oracle must see every scored
            # direction as well.
            initial_fit = fit_rank2_robust(initial_coefficients)
            initial_mean, initial_basis = initial_fit["joint"]
            initial_lifted_mean, initial_lifted_basis = initial_fit["lifted"]
            _, _, initial_coordinates = _optimise_coordinates(
                initial_coefficients,
                initial_mean,
                initial_basis,
                TOLERANCE8,
            )
            initial_scale_fits = [
                fit_rank2_scale(
                    initial_coefficients[..., scale_index * 4:scale_index * 4 + 4],
                    seed=(
                        initial_mean[..., scale_index * 4:scale_index * 4 + 4],
                        initial_basis[..., scale_index * 4:scale_index * 4 + 4, :],
                    ),
                )
                for scale_index in range(2)
            ]
            initial_scale_coordinates = []
            for scale_index, scale_fit in enumerate(initial_scale_fits):
                span = slice(scale_index * 4, scale_index * 4 + 4)
                _, _, coordinates = _optimise_coordinates(
                    initial_coefficients[..., span],
                    scale_fit["origin"],
                    scale_fit["basis"],
                    TOLERANCE4,
                )
                initial_scale_coordinates.append(coordinates)
            observations = np.stack([
                np.stack([joint(key, phase) for phase in fitted_phase_names])
                for key in direction_keys
            ])
            coupled_coefficients, coupled_diagnostics = coupled_phase_refinement(
                observations,
                fitted_phase_offsets,
                initial_coordinates,
                initial_mean,
                initial_basis,
                difference_pairs,
            )
            coupled_scale_coefficients = []
            coupled_scale_diagnostics = []
            for scale_index, scale_fit in enumerate(initial_scale_fits):
                span = slice(scale_index * 4, scale_index * 4 + 4)
                refined, scale_coupled_diagnostics = coupled_phase_refinement(
                    observations[..., span],
                    fitted_phase_offsets,
                    initial_scale_coordinates[scale_index],
                    scale_fit["origin"],
                    scale_fit["basis"],
                    difference_pairs,
                )
                coupled_scale_coefficients.append(refined)
                coupled_scale_diagnostics.append(scale_coupled_diagnostics)
            fit_result = fit_rank2_robust(coupled_coefficients)
            print(f"[candidate-p-v4] coupled/refit complete {triangle.key}")
            mean, basis = fit_result["joint"]
            lifted_mean, lifted_basis = fit_result["lifted"]
            final_scale_fits = [
                fit_rank2_scale(
                    coupled_scale_coefficients[scale_index],
                    seed=(
                        mean[..., scale_index * 4:scale_index * 4 + 4],
                        basis[..., scale_index * 4:scale_index * 4 + 4, :],
                    ),
                )
                for scale_index in range(2)
            ]
            fit_diagnostics.append({
                "triangle": triangle.key,
                "initial": initial_fit["diagnostics"],
                "coupledPhaseSolve": coupled_diagnostics,
                "refit": fit_result["diagnostics"],
                "independentScales": [
                    {
                        "scale": SOURCE_SCALES[index],
                        "initialFit": initial_scale_fits[index]["diagnostics"],
                        "coupledPhaseSolve": coupled_scale_diagnostics[index],
                        "refit": final_scale_fits[index]["diagnostics"],
                    }
                    for index in range(2)
                ],
            })
            for heldout_index, (name, *_) in enumerate(HELDOUT_BARY):
                key = f"{triangle.key}-{name}"
                held_coefficients = coupled_coefficients[len(TRAIN_BARY) + heldout_index]
                baseline_coefficients = initial_coefficients[len(TRAIN_BARY) + heldout_index]
                phase_only_coefficients = phase_coefficients[key]
                projected_vertices = project(held_coefficients, mean, basis, TOLERANCE8)
                baseline_projected_vertices = project(
                    baseline_coefficients,
                    initial_mean,
                    initial_basis,
                    TOLERANCE8,
                )
                lifted_vertices = project(held_coefficients, lifted_mean, lifted_basis, TOLERANCE8)
                baseline_lifted_vertices = project(
                    baseline_coefficients,
                    initial_lifted_mean,
                    initial_lifted_basis,
                    TOLERANCE8,
                )
                scale_vertices = []
                baseline_scale_vertices = []
                for scale_index, scale_fit in enumerate(final_scale_fits):
                    refined_coefficients = coupled_scale_coefficients[scale_index][len(TRAIN_BARY) + heldout_index]
                    scale_vertices.append(project(
                        refined_coefficients,
                        scale_fit["origin"],
                        scale_fit["basis"],
                        TOLERANCE4,
                    ))
                    span = slice(scale_index * 4, scale_index * 4 + 4)
                    baseline_scale_vertices.append(project(
                        baseline_coefficients[..., span],
                        initial_scale_fits[scale_index]["origin"],
                        initial_scale_fits[scale_index]["basis"],
                        TOLERANCE4,
                    ))
                for phase_name, ox, oz in EVALUATION_PHASE_SITES:
                    truth = joint(key, phase_name)
                    prediction = phase_interpolate(projected_vertices, ox, oz)
                    baseline_prediction = phase_interpolate(baseline_projected_vertices, ox, oz)
                    lifted_prediction = phase_interpolate(lifted_vertices, ox, oz)
                    baseline_lifted_prediction = phase_interpolate(baseline_lifted_vertices, ox, oz)
                    scale_predictions = [
                        phase_interpolate(vertices, ox, oz)
                        for vertices in scale_vertices
                    ]
                    baseline_scale_predictions = [
                        phase_interpolate(vertices, ox, oz)
                        for vertices in baseline_scale_vertices
                    ]
                    phase_only_prediction = phase_interpolate(phase_only_coefficients, ox, oz)
                    relaxed_predictions[(triangle.key, str(name), phase_name)] = prediction
                    scale_scores = []
                    baseline_scale_scores = []
                    phase_only_scale_scores = []
                    lifted_scale_scores = []
                    baseline_lifted_scale_scores = []
                    sigma4_scale_scores = []
                    baseline_independent_scale_scores = []
                    for scale_index, scale in enumerate(SOURCE_SCALES):
                        span = slice(scale_index * 4, scale_index * 4 + 4)
                        result = score(truth[..., span], prediction[..., span])
                        baseline_result = score(truth[..., span], baseline_prediction[..., span])
                        phase_only_result = score(truth[..., span], phase_only_prediction[..., span])
                        lifted_result = score(truth[..., span], lifted_prediction[..., span])
                        baseline_lifted_result = score(truth[..., span], baseline_lifted_prediction[..., span])
                        scale_scores.append({"scale": scale, "score": result})
                        baseline_scale_scores.append({"scale": scale, "score": baseline_result})
                        phase_only_scale_scores.append({"scale": scale, "score": phase_only_result})
                        lifted_scale_scores.append({"scale": scale, "score": lifted_result})
                        baseline_lifted_scale_scores.append({"scale": scale, "score": baseline_lifted_result})
                        sigma4_scale_scores.append({"scale": scale, "score": score(truth[..., span], scale_predictions[scale_index])})
                        baseline_independent_scale_scores.append({"scale": scale, "score": score(truth[..., span], baseline_scale_predictions[scale_index])})
                        normalized = max(result["coverage"]["p95"] / .08, result["premulRgb"]["p95"] / .06, result["largestConnectedFraction"] / .01)
                        qa_candidates.append((normalized, f"{triangle.key}-{name}-{phase_name}-sigma{scale}", truth[..., span], prediction[..., span]))

                    translations = []
                    baseline_translations = []
                    phase_only_translations = []
                    lifted_translations = []
                    baseline_lifted_translations = []
                    sigma4_translations = []
                    baseline_independent_translations = []
                    for millimetres in TRANSLATIONS_MM:
                        for axis_index, axis in enumerate(TRANSLATION_AXES):
                            moved_phase = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                            moved_truth = joint(key, moved_phase)
                            cells = millimetres * .001 / source_size * N
                            moved_prediction = phase_interpolate(projected_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            baseline_moved_prediction = phase_interpolate(baseline_projected_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            lifted_moved_prediction = phase_interpolate(lifted_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            baseline_lifted_moved_prediction = phase_interpolate(baseline_lifted_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            scale_moved_predictions = [
                                phase_interpolate(vertices, ox + cells * axis[0], oz + cells * axis[1])
                                for vertices in scale_vertices
                            ]
                            baseline_scale_moved_predictions = [
                                phase_interpolate(vertices, ox + cells * axis[0], oz + cells * axis[1])
                                for vertices in baseline_scale_vertices
                            ]
                            phase_only_moved_prediction = phase_interpolate(phase_only_coefficients, ox + cells * axis[0], oz + cells * axis[1])
                            for scale_index, scale in enumerate(SOURCE_SCALES):
                                span = slice(scale_index * 4, scale_index * 4 + 4)
                                translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], prediction[..., span], moved_prediction[..., span])})
                                baseline_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], baseline_prediction[..., span], baseline_moved_prediction[..., span])})
                                phase_only_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], phase_only_prediction[..., span], phase_only_moved_prediction[..., span])})
                                lifted_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], lifted_prediction[..., span], lifted_moved_prediction[..., span])})
                                baseline_lifted_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], baseline_lifted_prediction[..., span], baseline_lifted_moved_prediction[..., span])})
                                sigma4_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], scale_predictions[scale_index], scale_moved_predictions[scale_index])})
                                baseline_independent_translations.append({"millimetres": millimetres, "axis": axis_index, "scale": scale, "score": translation_score(truth[..., span], moved_truth[..., span], baseline_scale_predictions[scale_index], baseline_scale_moved_predictions[scale_index])})
                    evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": scale_scores, "translations": translations})
                    baseline_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": baseline_scale_scores, "translations": baseline_translations})
                    phase_only_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": phase_only_scale_scores, "translations": phase_only_translations})
                    lifted_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": lifted_scale_scores, "translations": lifted_translations})
                    baseline_lifted_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": baseline_lifted_scale_scores, "translations": baseline_lifted_translations})
                    sigma4_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": sigma4_scale_scores, "translations": sigma4_translations})
                    baseline_scale_evaluations.append({"triangle": triangle.key, "site": name, "phaseSite": phase_name, "scaleScores": baseline_independent_scale_scores, "translations": baseline_independent_translations})

    static_scores = [value["score"] for evaluation in evaluations for value in evaluation["scaleScores"]]
    translation_scores = [value["score"] for evaluation in evaluations for value in evaluation["translations"]]
    baseline_static_scores = [value["score"] for evaluation in baseline_evaluations for value in evaluation["scaleScores"]]
    baseline_translation_scores = [value["score"] for evaluation in baseline_evaluations for value in evaluation["translations"]]
    phase_only_static_scores = [value["score"] for evaluation in phase_only_evaluations for value in evaluation["scaleScores"]]
    phase_only_translation_scores = [value["score"] for evaluation in phase_only_evaluations for value in evaluation["translations"]]
    lifted_static_scores = [value["score"] for evaluation in lifted_evaluations for value in evaluation["scaleScores"]]
    lifted_translation_scores = [value["score"] for evaluation in lifted_evaluations for value in evaluation["translations"]]
    baseline_lifted_static_scores = [value["score"] for evaluation in baseline_lifted_evaluations for value in evaluation["scaleScores"]]
    baseline_lifted_translation_scores = [value["score"] for evaluation in baseline_lifted_evaluations for value in evaluation["translations"]]
    sigma4_static_scores = [value["score"] for evaluation in sigma4_evaluations for value in evaluation["scaleScores"]]
    sigma4_translation_scores = [value["score"] for evaluation in sigma4_evaluations for value in evaluation["translations"]]
    baseline_scale_static_scores = [value["score"] for evaluation in baseline_scale_evaluations for value in evaluation["scaleScores"]]
    baseline_scale_translation_scores = [value["score"] for evaluation in baseline_scale_evaluations for value in evaluation["translations"]]

    shared_edges = []
    for quad in QUADS:
        left = next(value for value in tri if value.quad == quad.key and value.half == 0)
        right = next(value for value in tri if value.quad == quad.key and value.half == 1)
        for phase_name, _, _ in PHASE_SITES:
            a = relaxed_predictions[(left.key, "shared-diagonal", phase_name)]
            b = relaxed_predictions[(right.key, "shared-diagonal", phase_name)]
            shared_edges.append({"quad": quad.key, "phaseSite": phase_name, "relaxedPrivateEdgeMismatch": score(a, b)})

    # The favourable far limit is the exact heldout phase mean.  It is phase
    # independent, so its translation excess is identically zero.
    far_mean = {"construction": "exact heldout joint-measure phase mean (optimistic lower bound)", "staticError": 0.0, "translationError": 0.0, "green": True}
    qa_candidates.sort(key=lambda value: value[0], reverse=True)
    image_records = []
    for index, (_, key, truth, prediction) in enumerate(qa_candidates[:5], start=1):
        path = qa_root / f"{index:03d}-{key}.png"
        write_qa(path, f"Candidate P rank-2 bound {key}: truth | prediction | error", truth, prediction)
        image_records.append({"file": path.name, "sha256": sha(path.read_bytes()), "dimensions": [768, 290], "interpretation": "exact-BVH heldout mid-phase truth, optimistic arbitrary-coordinate affine-rank-2 prediction, max-channel error"})

    def maximum(values: list[dict[str, Any]], path: tuple[str, ...]) -> float:
        result = []
        for value in values:
            current: Any = value
            for part in path:
                current = current[part]
            result.append(float(current))
        return max(result)

    def metric_summary(static: list[dict[str, Any]], translations: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "green": all(value["green"] for value in static) and all(value["green"] for value in translations),
            "staticCases": len(static), "staticGreen": sum(value["green"] for value in static),
            "translationCases": len(translations), "translationGreen": sum(value["green"] for value in translations),
            "coverageP95Worst": maximum(static, ("coverage", "p95")),
            "coverageP99Worst": maximum(static, ("coverage", "p99")),
            "rgbP95Worst": maximum(static, ("premulRgb", "p95")),
            "rgbP99Worst": maximum(static, ("premulRgb", "p99")),
            "connectedWorst": maximum(static, ("largestConnectedFraction",)),
            "translationCoverageP95Worst": maximum(translations, ("coverageP95",)),
            "translationRgbP95Worst": maximum(translations, ("premulRgbP95",)),
            "translationConnectedWorst": maximum(translations, ("largestConnectedFraction",)),
        }

    metrics = metric_summary(static_scores, translation_scores)
    baseline_metrics = metric_summary(baseline_static_scores, baseline_translation_scores)
    phase_only_metrics = metric_summary(phase_only_static_scores, phase_only_translation_scores)
    lifted_metrics = metric_summary(lifted_static_scores, lifted_translation_scores)
    baseline_lifted_metrics = metric_summary(baseline_lifted_static_scores, baseline_lifted_translation_scores)
    sigma4_metrics = metric_summary(sigma4_static_scores, sigma4_translation_scores)
    baseline_scale_metrics = metric_summary(baseline_scale_static_scores, baseline_scale_translation_scores)
    unlimited_green = metrics["green"] or baseline_metrics["green"]
    baseline_normalised_gap = max(
        baseline_metrics["coverageP95Worst"] / LIMITS["coverageP95"],
        baseline_metrics["coverageP99Worst"] / LIMITS["coverageP99"],
        baseline_metrics["rgbP95Worst"] / LIMITS["rgbP95"],
        baseline_metrics["rgbP99Worst"] / LIMITS["rgbP99"],
        baseline_metrics["connectedWorst"] / LIMITS["connected"],
        baseline_metrics["translationCoverageP95Worst"] / LIMITS["translationP95"],
        baseline_metrics["translationRgbP95Worst"] / LIMITS["translationP95"],
        baseline_metrics["translationConnectedWorst"] / LIMITS["connected"],
    )
    quantisation = {"run": False, "reason": "unlimited-precision necessary bound is RED"} if not unlimited_green else {"run": False, "reason": "rank-2 bound GREEN would require actual canonical chord fitting before packing"}
    bridge = {"run": False, "reason": "binding order permits bridge fitting only after unlimited-precision Candidate P is GREEN"}
    verdict = (
        "GREEN-NECESSARY-BOUND-ONLY"
        if unlimited_green
        else ("INCONCLUSIVE-NARROW-RED" if baseline_normalised_gap < 1.5 else "RED")
    )
    independent_scale_metrics = {}
    baseline_independent_scale_metrics = {}
    for scale in SOURCE_SCALES:
        scale_static = [
            value["score"] for evaluation in sigma4_evaluations
            for value in evaluation["scaleScores"] if value["scale"] == scale
        ]
        scale_translations = [
            value["score"] for evaluation in sigma4_evaluations
            for value in evaluation["translations"] if value["scale"] == scale
        ]
        independent_scale_metrics[str(scale)] = metric_summary(scale_static, scale_translations)
        baseline_scale_static = [
            value["score"] for evaluation in baseline_scale_evaluations
            for value in evaluation["scaleScores"] if value["scale"] == scale
        ]
        baseline_scale_translations = [
            value["score"] for evaluation in baseline_scale_evaluations
            for value in evaluation["translations"] if value["scale"] == scale
        ]
        baseline_independent_scale_metrics[str(scale)] = metric_summary(
            baseline_scale_static,
            baseline_scale_translations,
        )
    report = {
        "schema": VERSION, "verdict": verdict,
        "source": {"path": SOURCE, "sha256": source_sha, "triangleCount": 2171134},
        "recipe": {**recipe, "recipeSha256": recipe_sha},
        "bound": {
            "logic": "Every Candidate-P two-event record lies in an affine subspace of joint two-scale RGBA dimension at most two. This deliberately favourable oracle sees all scored directions and searches PCA, robust reweighted PCA, leave-one-out PCA, and every three-observation affine plane independently at every complete phase vertex, then grants each direction its optimal coordinates. It relaxes chords, canonical edges, topology, ordering, nonnegativity, and packing. A large robust RED gap is a representability failure rather than a train/heldout or least-squares artefact.",
            "completePhaseVertexFunctions": f"all {N*N} phase vertices fitted independently for each of the four targeted triangles",
            "canonicalEdgeStatus": "not enforced: private edges are a favourable superset; measured private-edge mismatch is reported, and canonicalisation cannot improve the per-side best projections",
            "fitDiagnostics": fit_diagnostics,
        },
        "metrics": metrics, "evaluations": evaluations,
        "initialRobustJointRank2": {
            "logic": "Preserved high-p robust joint rank-two fit before the coupled least-squares alternation. It is scored independently so lower L2 normal residual cannot overwrite better frozen tail behaviour.",
            "metrics": baseline_metrics,
            "worstNormalisedFrozenGap": baseline_normalised_gap,
            "evaluations": baseline_evaluations,
        },
        "phaseOnlyOracle": {
            "logic": "Per-direction phase coefficients are globally least-squares-fitted against every scored phase offset and translation, then evaluated with no angular rank restriction. Coefficients are not forced to equal vertex truth.",
            "metrics": phase_only_metrics, "evaluations": phase_only_evaluations,
        },
        "independentPerScaleRank2": {
            "logic": "Sigma4 and sigma16 each receive independent initial rank-two fits, coupled phase solves, angular refits, and high-p direction-coordinate refinement. Combined GREEN requires both scales.",
            "combinedGreen": sigma4_metrics["green"],
            "metricsByScale": independent_scale_metrics,
            "combinedMetrics": sigma4_metrics,
            "evaluations": sigma4_evaluations,
            "initialRobust": {
                "combinedGreen": baseline_scale_metrics["green"],
                "metricsByScale": baseline_independent_scale_metrics,
                "combinedMetrics": baseline_scale_metrics,
                "evaluations": baseline_scale_evaluations,
            },
        },
        "sigma4LiftedJointRank2": {
            "logic": "Planes are fitted on sigma4 first; sigma16 is optimally regressed from the same two coordinates, after which direction coordinates receive the same high-p joint refinement.",
            "crossesEveryFrozenThreshold": lifted_metrics["green"],
            "metrics": lifted_metrics, "evaluations": lifted_evaluations,
            "initialRobust": {
                "crossesEveryFrozenThreshold": baseline_lifted_metrics["green"],
                "metrics": baseline_lifted_metrics,
                "evaluations": baseline_lifted_evaluations,
            },
        },
        "sharedEdgeDiagnostics": shared_edges,
        "farPhaseMean": far_mean, "quantisation": quantisation, "bridge": bridge,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    report_sha = sha(report_text.encode())
    index = {"schema": f"{VERSION}-qa-index", "sourceSha256": source_sha, "recipeSha256": recipe_sha, "reportSha256": report_sha, "images": image_records}
    (qa_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    summary = [
        "# Candidate P optimistic affine-rank-2 necessary bound", "", f"Verdict: **{verdict}**", "",
        f"Source SHA-256: `{source_sha}`", f"Recipe SHA-256: `{recipe_sha}`", f"Report SHA-256: `{report_sha}`", "",
        "This is a deliberately favourable representability oracle for the canonical two-event codec: all scored directions are leaked into a robust per-vertex fit, while arbitrary signed region states, arbitrary coordinates, and private/noncanonical edges are allowed. GREEN would not yet authorise Candidate P; a RED verdict is only called decisive when the robust gap is large enough not to depend on exact L-infinity optimality.", "",
        f"- Static cases GREEN: `{metrics['staticGreen']}/{metrics['staticCases']}`",
        f"- Translation cases GREEN: `{metrics['translationGreen']}/{metrics['translationCases']}`",
        f"- Worst coverage p95/p99: `{metrics['coverageP95Worst']:.6f}/{metrics['coverageP99Worst']:.6f}`",
        f"- Worst RGB p95/p99: `{metrics['rgbP95Worst']:.6f}/{metrics['rgbP99Worst']:.6f}`",
        f"- Worst connected exceedance: `{metrics['connectedWorst']:.6f}`",
        f"- Worst translation A/RGB p95: `{metrics['translationCoverageP95Worst']:.6f}/{metrics['translationRgbP95Worst']:.6f}`",
        f"- Worst translation connected exceedance: `{metrics['translationConnectedWorst']:.6f}`", "",
        "Phase-only globally optimized exact-direction control:",
        f"- Static cases GREEN: `{phase_only_metrics['staticGreen']}/{phase_only_metrics['staticCases']}`",
        f"- Translation cases GREEN: `{phase_only_metrics['translationGreen']}/{phase_only_metrics['translationCases']}`",
        f"- Worst coverage/RGB p95: `{phase_only_metrics['coverageP95Worst']:.6f}/{phase_only_metrics['rgbP95Worst']:.6f}`",
        f"- Worst translation A/RGB p95: `{phase_only_metrics['translationCoverageP95Worst']:.6f}/{phase_only_metrics['translationRgbP95Worst']:.6f}`", "",
        "Independent sigma4 and sigma16 rank-2 controls:",
        f"- Combined GREEN: `{sigma4_metrics['green']}`; static `{sigma4_metrics['staticGreen']}/{sigma4_metrics['staticCases']}`; translation `{sigma4_metrics['translationGreen']}/{sigma4_metrics['translationCases']}`",
        f"- Worst coverage/RGB p95: `{sigma4_metrics['coverageP95Worst']:.6f}/{sigma4_metrics['rgbP95Worst']:.6f}`",
        f"- Worst translation A/RGB p95: `{sigma4_metrics['translationCoverageP95Worst']:.6f}/{sigma4_metrics['translationRgbP95Worst']:.6f}`", "",
        "Sigma4-first, same-coordinate lifted joint rank-2 control:",
        f"- GREEN: `{lifted_metrics['green']}`; static `{lifted_metrics['staticGreen']}/{lifted_metrics['staticCases']}`; translation `{lifted_metrics['translationGreen']}/{lifted_metrics['translationCases']}`",
        f"- Worst coverage/RGB p95: `{lifted_metrics['coverageP95Worst']:.6f}/{lifted_metrics['rgbP95Worst']:.6f}`",
        f"- Worst translation A/RGB p95: `{lifted_metrics['translationCoverageP95Worst']:.6f}/{lifted_metrics['translationRgbP95Worst']:.6f}`", "",
        "Far mean uses the exact heldout phase mean and is identically translation stable. Quantisation and the one-read bridge are not run unless unlimited precision is GREEN.",
        "QA panels are truth | optimistic rank-2 prediction | error; hashes are in `qa/index.json`.", "",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary))
    print(json.dumps({"verdict": verdict, "output": str(output.relative_to(root)), "metrics": metrics, "reportSha256": report_sha}, indent=2))


if __name__ == "__main__":
    main()

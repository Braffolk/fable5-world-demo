"""Final TGV projection of the connected ALS structural reference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import StructuralInputs
from .reference import StructuralReference


@dataclass(frozen=True)
class ProjectionResult:
    residual_m: np.ndarray
    iterations: int
    convergence_delta_m: float


def _gradient(value: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(value)
    dy = np.zeros_like(value)
    dx[:, :-1] = (value[:, 1:] - value[:, :-1]) / pitch
    dy[:-1] = (value[1:] - value[:-1]) / pitch
    return dx, dy


def _gradient_adjoint(dx: np.ndarray, dy: np.ndarray, pitch: float) -> np.ndarray:
    result = np.zeros_like(dx)
    result[:, :-1] -= dx[:, :-1] / pitch
    result[:, 1:] += dx[:, :-1] / pitch
    result[:-1] -= dy[:-1] / pitch
    result[1:] += dy[:-1] / pitch
    return result


def _symmetric_gradient(
    vx: np.ndarray, vy: np.ndarray, pitch: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dvx_dx, dvx_dy = _gradient(vx, pitch)
    dvy_dx, dvy_dy = _gradient(vy, pitch)
    return dvx_dx, dvy_dy, 0.5 * (dvx_dy + dvy_dx)


def _symmetric_adjoint(
    qxx: np.ndarray, qyy: np.ndarray, qxy: np.ndarray, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    vx = _gradient_adjoint(qxx, 0.5 * qxy, pitch)
    vy = _gradient_adjoint(0.5 * qxy, qyy, pitch)
    return vx, vy


def _huber_prox(
    value: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    step: np.ndarray,
    delta: float,
) -> np.ndarray:
    distance = value - target
    threshold = delta + step * weight
    quadratic = np.abs(distance) <= threshold
    error = np.where(
        quadratic,
        distance / (1.0 + step * weight / delta),
        distance - step * weight * np.sign(distance),
    )
    return target + error


def solve_projection(
    inputs: StructuralInputs,
    reference: StructuralReference,
    config: dict,
) -> ProjectionResult:
    cfg = config["tgv"]
    pitch = inputs.solve_pitch_m
    tau = float(cfg["tau"])
    sigma = float(cfg["sigma"])
    theta = float(cfg["theta"])
    alpha1 = float(cfg["first_order_weight"])
    alpha0 = float(cfg["second_order_weight_m"])
    delta = float(cfg["huber_delta_m"])
    maximum = float(cfg["maximum_residual_m"])

    shape = inputs.c0_solve_m.shape
    u = np.zeros(shape, dtype=np.float32)
    ux = np.zeros(shape, dtype=np.float32)
    uy = np.zeros(shape, dtype=np.float32)
    p0 = np.zeros(shape, dtype=np.float32)
    p1 = np.zeros(shape, dtype=np.float32)
    q0 = np.zeros(shape, dtype=np.float32)
    q1 = np.zeros(shape, dtype=np.float32)
    q2 = np.zeros(shape, dtype=np.float32)
    u_bar, ux_bar, uy_bar = u.copy(), ux.copy(), uy.copy()
    u_average = np.zeros(shape, dtype=np.float64)
    average_count = 0

    confidence = reference.solve_confidence
    strength = reference.solve_network_strength
    data_weight = float(cfg["data_weight"]) * confidence
    inside_prior = float(cfg["c0_prior_inside"])
    outside_prior = float(cfg["c0_prior_outside"])
    prior = inside_prior + (outside_prior - inside_prior) * (1.0 - confidence)
    normal_floor = float(cfg["normal_regularization_floor"])
    tangent_floor = float(cfg["tangent_regularization_floor"])
    normal_radius = alpha1 * (1.0 - strength * (1.0 - normal_floor))
    tangent_radius = alpha1 * (1.0 - strength * (1.0 - tangent_floor))
    nx, ny = reference.solve_guide_normal_x, reference.solve_guide_normal_y
    tx, ty = -ny, nx

    convergence = 0.0
    for iteration in range(int(cfg["iterations"])):
        dux, duy = _gradient(u_bar, pitch)
        p0 += sigma * (dux - ux_bar)
        p1 += sigma * (duy - uy_bar)
        pn = p0 * nx + p1 * ny
        pt = p0 * tx + p1 * ty
        length = np.sqrt(
            (pn / np.maximum(normal_radius, 1e-6)) ** 2
            + (pt / np.maximum(tangent_radius, 1e-6)) ** 2
        )
        scale = np.maximum(length, 1.0)
        p0 /= scale
        p1 /= scale

        exx, eyy, exy = _symmetric_gradient(ux_bar, uy_bar, pitch)
        q0 += sigma * exx
        q1 += sigma * eyy
        q2 += sigma * exy
        q_length = np.sqrt(q0 * q0 + q1 * q1 + 2.0 * q2 * q2)
        q_scale = np.maximum(q_length / alpha0, 1.0)
        q0 /= q_scale
        q1 /= q_scale
        q2 /= q_scale

        old_u, old_ux, old_uy = u, ux, uy
        candidate = u - tau * _gradient_adjoint(p0, p1, pitch)
        step = tau / (1.0 + tau * prior)
        candidate /= 1.0 + tau * prior
        u = _huber_prox(
            candidate,
            reference.solve_target_m,
            data_weight,
            step,
            delta,
        )
        u = np.clip(u, -maximum, maximum)
        u[inputs.hard_solve] = 0.0

        eqx, eqy = _symmetric_adjoint(q0, q1, q2, pitch)
        ux = ux - tau * (-p0 + eqx)
        uy = uy - tau * (-p1 + eqy)
        ux[inputs.hard_solve] = 0.0
        uy[inputs.hard_solve] = 0.0

        u_bar = u + theta * (u - old_u)
        ux_bar = ux + theta * (ux - old_ux)
        uy_bar = uy + theta * (uy - old_uy)
        if iteration >= int(cfg["iterations"]) - 10:
            convergence = max(convergence, float(np.max(np.abs(u - old_u))))
        if iteration >= int(cfg["iterations"]) // 2:
            u_average += u
            average_count += 1

    residual = np.asarray(u_average / max(average_count, 1), dtype=np.float32)
    residual[inputs.hard_solve] = 0.0
    return ProjectionResult(
        residual_m=residual,
        iterations=int(cfg["iterations"]),
        convergence_delta_m=convergence,
    )

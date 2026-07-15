"""Primal-dual second-order TGV residual reconstruction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import AlsTgvEvidence


@dataclass(frozen=True)
class TgvResult:
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


def _symmetric_gradient(vx: np.ndarray, vy: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dvx_dx, dvx_dy = _gradient(vx, pitch)
    dvy_dx, dvy_dy = _gradient(vy, pitch)
    return dvx_dx, dvy_dy, 0.5 * (dvx_dy + dvy_dx)


def _symmetric_adjoint(qxx: np.ndarray, qyy: np.ndarray, qxy: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros_like(qxx)
    vx = _gradient_adjoint(qxx, 0.5 * qxy, pitch)
    vy = _gradient_adjoint(0.5 * qxy, qyy, pitch)
    return vx, vy


def _huber_prox(value: np.ndarray, target: np.ndarray, weight: np.ndarray, step: float, delta: float) -> np.ndarray:
    distance = value - target
    threshold = delta + step * weight
    quadratic = np.abs(distance) <= threshold
    error = np.where(
        quadratic,
        distance / (1.0 + step * weight / delta),
        distance - step * weight * np.sign(distance),
    )
    return target + error


def solve_tgv(evidence: AlsTgvEvidence, config: dict) -> TgvResult:
    cfg = config["tgv"]
    pitch = evidence.pitch_m
    tau = float(cfg["tau"])
    sigma = float(cfg["sigma"])
    theta = float(cfg["theta"])
    alpha1 = float(cfg["first_order_weight"])
    alpha0 = float(cfg["second_order_weight_m"])
    beta = float(cfg["c0_prior_weight"])
    delta = float(config["qualification"]["huber_delta_m"])
    data_weight = float(cfg["data_weight"]) * evidence.data_weight
    maximum = float(cfg["maximum_residual_m"])

    shape = evidence.c0_m.shape
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

    confidence = evidence.guide_edge_confidence
    normal_floor = float(config["imagery_guidance"]["normal_regularization_floor"])
    tangent_floor = float(config["imagery_guidance"]["tangent_regularization_floor"])
    normal_radius = alpha1 * (1.0 - confidence * (1.0 - normal_floor))
    tangent_radius = alpha1 * (1.0 - confidence * (1.0 - tangent_floor))
    nx, ny = evidence.guide_normal_x, evidence.guide_normal_y
    tx, ty = -ny, nx

    convergence = 0.0
    for iteration in range(int(cfg["iterations"])):
        dux, duy = _gradient(u_bar, pitch)
        p0 += sigma * (dux - ux_bar)
        p1 += sigma * (duy - uy_bar)
        pn = p0 * nx + p1 * ny
        pt = p0 * tx + p1 * ty
        length = np.sqrt((pn / np.maximum(normal_radius, 1e-6)) ** 2 + (pt / np.maximum(tangent_radius, 1e-6)) ** 2)
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
        u_candidate = u - tau * _gradient_adjoint(p0, p1, pitch)
        tau_data = tau / (1.0 + tau * beta)
        u_candidate = u_candidate / (1.0 + tau * beta)
        u = _huber_prox(u_candidate, evidence.data_target_m, data_weight, tau_data, delta)
        u = np.clip(u, -maximum, maximum)
        u[evidence.hard_zero | ~evidence.active] = 0.0

        eqx, eqy = _symmetric_adjoint(q0, q1, q2, pitch)
        ux = ux - tau * (-p0 + eqx)
        uy = uy - tau * (-p1 + eqy)
        ux[evidence.hard_zero | ~evidence.active] = 0.0
        uy[evidence.hard_zero | ~evidence.active] = 0.0

        u_bar = u + theta * (u - old_u)
        ux_bar = ux + theta * (ux - old_ux)
        uy_bar = uy + theta * (uy - old_uy)
        if iteration >= int(cfg["iterations"]) - 10:
            convergence = max(convergence, float(np.max(np.abs(u - old_u))))
        if iteration >= int(cfg["iterations"]) // 2:
            u_average += u
            average_count += 1

    u = np.asarray(u_average / max(average_count, 1), dtype=np.float32)
    u[evidence.hard_zero | ~evidence.active] = 0.0
    return TgvResult(residual_m=u, iterations=int(cfg["iterations"]), convergence_delta_m=convergence)

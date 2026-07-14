"""Frozen F4/R4/Q4 analysis operators and Moore band materialization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resample import AreaGrid, FinePhase


F4_COEFFICIENTS = np.asarray((1, 3, 6, 10, 12, 12, 10, 6, 3, 1), dtype=np.float64) / 64.0


@dataclass(frozen=True)
class PhaseBands:
    phase: FinePhase
    a1: AreaGrid
    a0: AreaGrid
    o1_q4: AreaGrid
    o0_q4q4: AreaGrid
    b1: AreaGrid
    derived_b2: AreaGrid


def _axis_bounds_from_centers(centers: np.ndarray, cell_m: float) -> np.ndarray:
    if centers.size == 0:
        raise ValueError("empty filtered grid")
    return centers[0] - 0.5 * cell_m + np.arange(centers.size + 1) * cell_m


def filter4(grid: AreaGrid, role: str) -> AreaGrid:
    """Apply the normative float64 east-then-north factor-four filter."""
    ny, nx = grid.values.shape
    out_nx = nx // 4
    out_ny = ny // 4
    px = np.arange(out_nx, dtype=np.int64)
    py = np.arange(out_ny, dtype=np.int64)
    complete_x = (4 * px - 3 >= 0) & (4 * px + 6 < nx)
    complete_y = (4 * py - 3 >= 0) & (4 * py + 6 < ny)
    east = np.zeros((ny, out_nx), dtype=np.float64)
    east_valid = np.zeros((ny, out_nx), dtype=np.bool_)
    good_x = np.flatnonzero(complete_x)
    east_valid[:, good_x] = True
    for coefficient, k in zip(F4_COEFFICIENTS, range(10), strict=True):
        columns = 4 * px[good_x] - 3 + k
        selected_valid = grid.valid[:, columns]
        east[:, good_x] += coefficient * np.where(
            selected_valid, grid.values[:, columns], 0.0
        )
        east_valid[:, good_x] &= selected_valid
    values = np.zeros((out_ny, out_nx), dtype=np.float64)
    valid = np.zeros((out_ny, out_nx), dtype=np.bool_)
    good_y = np.flatnonzero(complete_y)
    valid[good_y] = True
    for coefficient, k in zip(F4_COEFFICIENTS, range(10), strict=True):
        rows = 4 * py[good_y] - 3 + k
        selected_valid = east_valid[rows]
        values[good_y] += coefficient * np.where(selected_valid, east[rows], 0.0)
        valid[good_y] &= selected_valid
    values[~valid] = np.nan
    cell = grid.cell_m * 4.0
    return AreaGrid(
        values=values,
        valid=valid,
        x_bounds_m=grid.x_bounds_m[0] + np.arange(out_nx + 1) * cell,
        y_bounds_m=grid.y_bounds_m[0] + np.arange(out_ny + 1) * cell,
        role=role,
    )


def _keys_weights(t: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            -0.5 * t + t * t - 0.5 * t * t * t,
            1.0 - 2.5 * t * t + 1.5 * t * t * t,
            0.5 * t + 2.0 * t * t - 1.5 * t * t * t,
            -0.5 * t * t + 0.5 * t * t * t,
        ),
        axis=1,
    )


def reconstruct4(coarse: AreaGrid, target: AreaGrid, role: str) -> AreaGrid:
    """Keys a=-0.5 reconstruction at target centers with complete support only."""
    if coarse.values.size == 0:
        values = np.full(target.values.shape, np.nan, dtype=np.float64)
        return AreaGrid(values, np.zeros_like(target.valid), target.x_bounds_m, target.y_bounds_m, role)
    coarse_x = coarse.x_centers_m
    coarse_y = coarse.y_centers_m
    target_x = target.x_centers_m
    target_y = target.y_centers_m
    ux = (target_x - coarse_x[0]) / coarse.cell_m
    uy = (target_y - coarse_y[0]) / coarse.cell_m
    base_x = np.floor(ux + 0.5).astype(np.int64)
    base_y = np.floor(uy + 0.5).astype(np.int64)
    tx = ux - base_x
    ty = uy - base_y
    if np.any((tx < -0.5000000001) | (tx > 0.5000000001)) or np.any(
        (ty < -0.5000000001) | (ty > 0.5000000001)
    ):
        raise AssertionError("R4 target phase is outside the Keys phase convention")
    wx = _keys_weights(tx)
    wy = _keys_weights(ty)
    x_ok = (base_x - 1 >= 0) & (base_x + 2 < coarse.values.shape[1])
    y_ok = (base_y - 1 >= 0) & (base_y + 2 < coarse.values.shape[0])
    east = np.zeros((coarse.values.shape[0], target_x.size), dtype=np.float64)
    east_valid = np.zeros((coarse.values.shape[0], target_x.size), dtype=np.bool_)
    good_x = np.flatnonzero(x_ok)
    if good_x.size:
        east_valid[:, good_x] = True
        for slot, delta in enumerate((-1, 0, 1, 2)):
            indices = base_x[good_x] + delta
            selected_valid = coarse.valid[:, indices]
            east[:, good_x] += wx[good_x, slot] * np.where(
                selected_valid, coarse.values[:, indices], 0.0
            )
            east_valid[:, good_x] &= selected_valid
    values = np.zeros(target.values.shape, dtype=np.float64)
    valid = np.zeros(target.values.shape, dtype=np.bool_)
    good_y = np.flatnonzero(y_ok)
    if good_y.size:
        valid[good_y] = True
        for slot, delta in enumerate((-1, 0, 1, 2)):
            indices = base_y[good_y] + delta
            selected_valid = east_valid[indices]
            values[good_y] += wy[good_y, slot, None] * np.where(
                selected_valid, east[indices], 0.0
            )
            valid[good_y] &= selected_valid
    valid &= target.valid
    values[~valid] = np.nan
    return AreaGrid(values, valid, target.x_bounds_m.copy(), target.y_bounds_m.copy(), role)


def q4(grid: AreaGrid, role: str) -> AreaGrid:
    """Fixed-order east-then-north exact 4x4 decoded-child area mean."""
    ny = grid.values.shape[0] // 4
    nx = grid.values.shape[1] // 4
    if ny == 0 or nx == 0:
        raise ValueError("grid too small for Q4")
    east = np.zeros((grid.values.shape[0], nx), dtype=np.float64)
    east_valid = np.ones((grid.values.shape[0], nx), dtype=np.bool_)
    for k in range(4):
        columns = 4 * np.arange(nx) + k
        east += np.where(grid.valid[:, columns], grid.values[:, columns], 0.0)
        east_valid &= grid.valid[:, columns]
    east *= 0.25
    values = np.zeros((ny, nx), dtype=np.float64)
    valid = np.ones((ny, nx), dtype=np.bool_)
    for k in range(4):
        rows = 4 * np.arange(ny) + k
        values += np.where(east_valid[rows], east[rows], 0.0)
        valid &= east_valid[rows]
    values *= 0.25
    values[~valid] = np.nan
    cell = grid.cell_m * 4.0
    x_centers = grid.x_bounds_m[0] + (4 * np.arange(nx) + 2) * grid.cell_m
    y_centers = grid.y_bounds_m[0] + (4 * np.arange(ny) + 2) * grid.cell_m
    return AreaGrid(
        values, valid, _axis_bounds_from_centers(x_centers, cell),
        _axis_bounds_from_centers(y_centers, cell), role,
    )


def materialize_bands(phase: FinePhase) -> PhaseBands:
    h = phase.height
    a1 = filter4(h, "A1=F4(H), 0.25m")
    a0 = filter4(a1, "A0=F4(A1), 1m")
    r4_a1 = reconstruct4(a1, h, "R4(A1) at 0.0625m")
    r4_a0 = reconstruct4(a0, a1, "R4(A0) at 0.25m")
    b2_valid = h.valid & r4_a1.valid
    b2_values = np.where(b2_valid, h.values - r4_a1.values, np.nan)
    b1_valid = a1.valid & r4_a0.valid
    b1_values = np.where(b1_valid, a1.values - r4_a0.values, np.nan)
    o1 = q4(h, "O1=Q4(H), 0.25m oracle observation")
    o0 = q4(o1, "O0=Q4(Q4(H)), 1m M0 observation")
    return PhaseBands(
        phase=phase,
        a1=a1,
        a0=a0,
        o1_q4=o1,
        o0_q4q4=o0,
        b1=AreaGrid(b1_values, b1_valid, a1.x_bounds_m.copy(), a1.y_bounds_m.copy(), "B1"),
        derived_b2=AreaGrid(
            b2_values, b2_valid, h.x_bounds_m.copy(), h.y_bounds_m.copy(),
            "derived_B2_hypothesis=H-R4(F4(H))",
        ),
    )

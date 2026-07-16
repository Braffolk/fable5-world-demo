"""Build and prove a non-folding harmonic parameterization of the curved bank."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio.features
import shapely
from rasterio.transform import from_origin
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg
from shapely.geometry import Polygon, shape

from ..unconsolidated_sand_curvilinear_strip.evidence import (
    Evidence,
    SOLVE_PITCH_M,
    TARGET_BBOX_EN,
)
from ..unconsolidated_sand_curvilinear_strip.model import (
    CHAIN_NAMES,
    StripModel,
    _orient_normals,
)


@dataclass(frozen=True)
class HarmonicRibbon:
    mask: np.ndarray
    boundary: np.ndarray
    u: np.ndarray
    v: np.ndarray
    chains_xy: np.ndarray
    chain_u: np.ndarray
    chain_v: np.ndarray
    polygon_area_m2: float
    solver_residual: float
    jacobian_min_abs: float
    jacobian_wrong_sign_count: int
    jacobian_interior_zero_count: int
    boundary_degenerate_count: int
    control_order_fraction: float
    corner_error_max_m: float


def _chain_xy(evidence: Evidence, strip: StripModel) -> np.ndarray:
    x, y, nx, ny = _orient_normals(evidence, strip.station_m)
    return np.stack(
        [
            np.column_stack(
                [x + nx * strip.chain_n_m[:, index], y + ny * strip.chain_n_m[:, index]]
            )
            for index in range(len(CHAIN_NAMES))
        ],
        axis=1,
    )


def _topological_mask(chains: np.ndarray, closing_cells: int) -> tuple[np.ndarray, object]:
    apron, crest = chains[:, 0], chains[:, -1]
    quads = []
    for index in range(len(apron) - 1):
        candidate = Polygon([apron[index], apron[index + 1], crest[index + 1], crest[index]])
        quads.append(candidate if candidate.is_valid else candidate.buffer(0.0))
    transform = from_origin(
        TARGET_BBOX_EN[0] - 0.5 * SOLVE_PITCH_M,
        TARGET_BBOX_EN[3] + 0.5 * SOLVE_PITCH_M,
        SOLVE_PITCH_M,
        SOLVE_PITCH_M,
    )
    mask = rasterio.features.rasterize(
        [(geometry, 1) for geometry in quads if not geometry.is_empty],
        out_shape=(513, 513),
        transform=transform,
        fill=0,
        dtype="uint8",
    ) > 0
    mask = ndimage.binary_closing(mask, structure=np.ones((closing_cells, closing_cells), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count < 1:
        raise ValueError("measured cross-strip cells produce no ribbon domain")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    records = [
        shape(geometry)
        for geometry, value in rasterio.features.shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if value == 1
    ]
    polygon = max(records, key=lambda value: value.area)
    if (
        polygon.geom_type != "Polygon"
        or not polygon.is_valid
        or not polygon.exterior.is_simple
        or len(polygon.interiors) != 0
    ):
        raise ValueError("topology-regularized ribbon is not one simple disk")
    return mask, polygon


def _boundary_map(mask: np.ndarray, polygon, chains: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    boundary = mask & ~ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(2, 1)
    )
    rows, cols = np.nonzero(boundary)
    x = TARGET_BBOX_EN[0] + cols * SOLVE_PITCH_M
    y = TARGET_BBOX_EN[3] - rows * SOLVE_PITCH_M
    exterior = polygon.exterior
    position = np.asarray(shapely.line_locate_point(exterior, shapely.points(x, y)))
    apron, crest = chains[:, 0], chains[:, -1]
    corner_xy = (crest[-1], crest[0], apron[0], apron[-1])
    corner = np.asarray(
        [
            float(shapely.line_locate_point(exterior, shapely.Point(point)))
            for point in corner_xy
        ]
    )
    length = float(exterior.length)
    relative = lambda value: np.mod(value - corner[0], length)
    corner_relative = relative(corner)
    if not np.all(np.diff(corner_relative) > 0.0):
        raise ValueError("ribbon corners are not cyclically ordered")
    corner_error = max(float(shapely.Point(point).distance(exterior)) for point in corner_xy)
    t = relative(position)
    ub = np.empty(len(t), dtype=np.float64)
    vb = np.empty(len(t), dtype=np.float64)
    c0, c1, c2, c3 = corner_relative
    selected = t <= c1
    fraction = t[selected] / c1
    ub[selected], vb[selected] = 1.0 - fraction, 1.0
    selected = (t > c1) & (t <= c2)
    fraction = (t[selected] - c1) / (c2 - c1)
    ub[selected], vb[selected] = 0.0, 1.0 - fraction
    selected = (t > c2) & (t <= c3)
    fraction = (t[selected] - c2) / (c3 - c2)
    ub[selected], vb[selected] = fraction, 0.0
    selected = t > c3
    fraction = (t[selected] - c3) / (length - c3)
    ub[selected], vb[selected] = 1.0, fraction
    u = np.full(mask.shape, np.nan, dtype=np.float64)
    v = np.full(mask.shape, np.nan, dtype=np.float64)
    u[rows, cols], v[rows, cols] = ub, vb
    return boundary, u, v, corner_error


def _harmonic_solve(mask: np.ndarray, boundary: np.ndarray, u: np.ndarray, v: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    interior = mask & ~boundary
    rows, cols = np.nonzero(interior)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[rows, cols] = np.arange(len(rows))
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    rhs_u = np.zeros(len(rows), dtype=np.float64)
    rhs_v = np.zeros(len(rows), dtype=np.float64)
    for identity, (row, col) in enumerate(zip(rows, cols)):
        matrix_rows.append(identity)
        matrix_cols.append(identity)
        matrix_data.append(4.0)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            neighbor = index[nr, nc]
            if neighbor >= 0:
                matrix_rows.append(identity)
                matrix_cols.append(int(neighbor))
                matrix_data.append(-1.0)
            else:
                rhs_u[identity] += u[nr, nc]
                rhs_v[identity] += v[nr, nc]
    matrix = sparse.csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)), shape=(len(rows), len(rows))
    )
    solved_u, info_u = cg(
        matrix,
        rhs_u,
        rtol=float(config["solver_relative_tolerance"]),
        atol=0.0,
        maxiter=int(config["solver_iterations"]),
    )
    solved_v, info_v = cg(
        matrix,
        rhs_v,
        rtol=float(config["solver_relative_tolerance"]),
        atol=0.0,
        maxiter=int(config["solver_iterations"]),
    )
    if info_u != 0 or info_v != 0:
        raise ValueError(f"harmonic coordinate solve did not converge: {info_u}/{info_v}")
    u[rows, cols], v[rows, cols] = solved_u, solved_v
    residual = max(
        float(np.max(np.abs(matrix @ solved_u - rhs_u))),
        float(np.max(np.abs(matrix @ solved_v - rhs_v))),
    )
    if residual > float(config["maximum_solver_residual"]):
        raise ValueError(f"harmonic coordinate residual {residual} exceeds the gate")
    return u, v, residual


def _jacobian_gate(mask: np.ndarray, boundary: np.ndarray, u: np.ndarray, v: np.ndarray, config: dict) -> tuple[float, int, int, int]:
    full = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    determinant_a = (
        (u[:-1, 1:] - u[:-1, :-1]) * (v[1:, :-1] - v[:-1, :-1])
        - (u[1:, :-1] - u[:-1, :-1]) * (v[:-1, 1:] - v[:-1, :-1])
    )
    determinant_b = (
        (u[1:, 1:] - u[:-1, 1:]) * (v[1:, :-1] - v[:-1, 1:])
        - (u[1:, :-1] - u[:-1, 1:]) * (v[1:, 1:] - v[:-1, 1:])
    )
    boundary_a = boundary[:-1, :-1] & boundary[:-1, 1:] & boundary[1:, :-1]
    boundary_b = boundary[:-1, 1:] & boundary[1:, :-1] & boundary[1:, 1:]
    values = np.concatenate((determinant_a[full], determinant_b[full]))
    boundary_only = np.concatenate((boundary_a[full], boundary_b[full]))
    sign = -1.0 if float(np.median(values[~boundary_only])) < 0.0 else 1.0
    signed = sign * values
    tolerance = float(config["minimum_interior_jacobian"])
    wrong = int(np.count_nonzero(signed < -tolerance))
    interior_zero = int(np.count_nonzero((signed <= tolerance) & ~boundary_only))
    boundary_zero = int(np.count_nonzero((np.abs(signed) <= tolerance) & boundary_only))
    if wrong or interior_zero:
        raise ValueError(
            f"harmonic ribbon folds or degenerates: wrong={wrong}, interior_zero={interior_zero}"
        )
    return float(np.min(signed[~boundary_only])), wrong, interior_zero, boundary_zero


def _extend_nearest(mask: np.ndarray, value: np.ndarray) -> np.ndarray:
    _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
    result = value.copy()
    result[~mask] = value[indices[0][~mask], indices[1][~mask]]
    return result


def _sample_grid(value: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(
        value,
        [
            (TARGET_BBOX_EN[3] - xy[..., 1]) / SOLVE_PITCH_M,
            (xy[..., 0] - TARGET_BBOX_EN[0]) / SOLVE_PITCH_M,
        ],
        order=1,
        mode="nearest",
    )


def build_harmonic_ribbon(evidence: Evidence, strip: StripModel, config: dict) -> HarmonicRibbon:
    chains = _chain_xy(evidence, strip)
    mask, polygon = _topological_mask(chains, int(config["topology_closing_cells"]))
    minimum_area = float(config["minimum_domain_area_m2"])
    if float(polygon.area) < minimum_area:
        raise ValueError(f"ribbon area {polygon.area} is below {minimum_area}")
    boundary, u, v, corner_error = _boundary_map(mask, polygon, chains)
    if corner_error > float(config["maximum_corner_error_m"]):
        raise ValueError(f"ribbon corner error {corner_error} exceeds the gate")
    u, v, residual = _harmonic_solve(mask, boundary, u, v, config)
    jacobian, wrong, interior_zero, boundary_zero = _jacobian_gate(mask, boundary, u, v, config)
    extended_u, extended_v = _extend_nearest(mask, u), _extend_nearest(mask, v)
    chain_u = _sample_grid(extended_u, chains)
    chain_v = _sample_grid(extended_v, chains)
    ordered = np.all(np.diff(chain_v, axis=1) > float(config["minimum_control_v_separation"]), axis=1)
    order_fraction = float(np.mean(ordered))
    if order_fraction < float(config["minimum_control_order_fraction"]):
        raise ValueError(f"control-chain harmonic order {order_fraction} fails")
    return HarmonicRibbon(
        mask=mask,
        boundary=boundary,
        u=u,
        v=v,
        chains_xy=chains,
        chain_u=chain_u,
        chain_v=chain_v,
        polygon_area_m2=float(polygon.area),
        solver_residual=residual,
        jacobian_min_abs=jacobian,
        jacobian_wrong_sign_count=wrong,
        jacobian_interior_zero_count=interior_zero,
        boundary_degenerate_count=boundary_zero,
        control_order_fraction=order_fraction,
        corner_error_max_m=corner_error,
    )

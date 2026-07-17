"""Two-field activator-inhibitor solve, typed-form extraction, topology.

The scale-dependent ecohydrological feedback (Rietkerk 2004 / Eppinga 2008-2009) is
realised with Gray-Scott activator-inhibitor kinetics: the activator V is the
self-facilitating peat/vascular ridge-former (autocatalysis), the inhibitor U is the
fast-diffusing plant-available water/nutrient substrate it consumes. Anisotropy comes
from enhanced along-flow water/solute diffusion scaled by |grad dome|, which orients the
Turing bands transverse to flow (strings) where the dome is graded and leaves an
isotropic labyrinth/maze on the flat dome. Steady-state V bands become strings; the wet
inter-string troughs become hollows; the wettest ponded troughs become pools.
Deterministic float64; Neumann (zero-flux) solve-domain boundary; no periodicity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .hydrology import ProcessHydrology


@dataclass(frozen=True)
class SolveResult:
    activator: np.ndarray  # V: peat/vascular ridge-former (high on strings)
    inhibitor: np.ndarray  # U: water/nutrient substrate (high in wet troughs)
    anisotropy_strength: np.ndarray  # clip(|grad dome|/slope_ref)
    residual_trace: np.ndarray
    final_residual: float
    iterations: int


def solve(
    hydro: ProcessHydrology,
    nucleation: np.ndarray,
    params: dict,
    *,
    budget: int,
    dt: float,
) -> SolveResult:
    feed = float(params["feed"])
    kill = float(params["kill"])
    d_u = float(params["D_u"])
    d_v = float(params["D_v"])
    aniso = float(params["anisotropy"])
    slope_ref = float(params["slope_ref"])
    seed_quantile = float(params["seed_quantile"])

    strength = np.clip(hydro.slope_mag / slope_ref, 0.0, 1.0)
    flow_row = hydro.flow_row
    flow_col = hydro.flow_col

    inhibitor = np.ones_like(nucleation)
    activator = np.zeros_like(nucleation)
    seed = nucleation >= seed_quantile
    activator[seed] = 0.25
    inhibitor[seed] = 0.5

    trace = np.empty(budget, dtype=np.float64)
    residual = np.inf
    for step in range(budget):
        lap_u = ndimage.laplace(inhibitor, mode="nearest")
        lap_v = ndimage.laplace(activator, mode="nearest")
        # Along-flow directional second derivative of the inhibitor (anisotropic
        # water/solute spreading). pad edge = Neumann zero-flux boundary.
        pad = np.pad(inhibitor, 1, mode="edge")
        u_rr = pad[2:, 1:-1] - 2.0 * inhibitor + pad[:-2, 1:-1]
        u_cc = pad[1:-1, 2:] - 2.0 * inhibitor + pad[1:-1, :-2]
        u_rc = 0.25 * (pad[2:, 2:] - pad[2:, :-2] - pad[:-2, 2:] + pad[:-2, :-2])
        dir2 = flow_row * flow_row * u_rr + flow_col * flow_col * u_cc + 2.0 * flow_row * flow_col * u_rc

        uvv = inhibitor * activator * activator
        du = d_u * lap_u + aniso * strength * d_u * dir2 - uvv + feed * (1.0 - inhibitor)
        dv = d_v * lap_v + uvv - (feed + kill) * activator
        inhibitor = np.clip(inhibitor + dt * du, 0.0, None)
        activator = np.clip(activator + dt * dv, 0.0, None)
        residual = float(np.max(np.abs(dv)))
        trace[step] = residual

    return SolveResult(
        activator=activator,
        inhibitor=inhibitor,
        anisotropy_strength=strength,
        residual_trace=trace,
        final_residual=residual,
        iterations=budget,
    )


# ---- typed-form classification -------------------------------------------------

LAWN, RIDGE, HOLLOW, POOL = 0, 1, 2, 3


def classify(
    activator: np.ndarray,
    inhibitor: np.ndarray,
    region: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """Return a label field (LAWN/RIDGE/HOLLOW/POOL) inside ``region`` (else LAWN).

    Strings/ridges are the emergent high-activator bands. The inter-string zone is
    sub-typed by the water/nutrient inhibitor W: the wettest troughs become pools,
    the driest ridge margins become lawn, the moderate wet flats become hollows. This
    makes pools inter-string troughs (bounded by ridges) by construction.
    """
    labels = np.zeros(activator.shape, dtype=np.uint8)
    values = activator[region]
    if values.size == 0:
        return labels
    ridge_cut = thresholds["ridge_level"] * np.percentile(values, 98.0)
    ridge = region & (activator >= ridge_cut)
    non_ridge = region & ~ridge
    w_values = inhibitor[non_ridge]
    labels[non_ridge] = LAWN
    labels[ridge] = RIDGE
    if w_values.size:
        pool_cut = np.percentile(w_values, thresholds["pool_wet_percentile"])
        hollow_cut = np.percentile(w_values, thresholds["hollow_wet_percentile"])
        labels[non_ridge & (inhibitor >= hollow_cut)] = HOLLOW
        labels[non_ridge & (inhibitor >= pool_cut)] = POOL
    return labels


def compute_cuts(
    activator: np.ndarray, inhibitor: np.ndarray, region: np.ndarray, thresholds: dict
) -> dict:
    """Freeze absolute class-boundary values over ``region`` (the whole-mire authority).

    The same absolute cuts are then applied at every resolution so the process-grid
    gate classification and the 0.25 m carve agree.
    """
    v = activator[region]
    u = inhibitor[region & (activator < thresholds["ridge_level"] * np.percentile(activator[region], 98.0))]
    ridge_cut = thresholds["ridge_level"] * float(np.percentile(v, 98.0))
    if u.size == 0:
        u = inhibitor[region]
    pool_cut = float(np.percentile(u, thresholds["pool_wet_percentile"]))
    hollow_cut = float(np.percentile(u, thresholds["hollow_wet_percentile"]))
    lo, hi = np.percentile(v, [2.0, 98.0])
    return {
        "ridge_cut": ridge_cut,
        "pool_cut": pool_cut,
        "hollow_cut": hollow_cut,
        "norm_lo": float(lo),
        "norm_hi": float(hi),
    }


def classify_from_cuts(
    activator: np.ndarray, inhibitor: np.ndarray, region: np.ndarray, cuts: dict
) -> np.ndarray:
    """Label field from frozen absolute cut values (see :func:`compute_cuts`)."""
    labels = np.zeros(activator.shape, dtype=np.uint8)
    ridge = region & (activator >= cuts["ridge_cut"])
    non_ridge = region & ~ridge
    labels[non_ridge] = LAWN
    labels[ridge] = RIDGE
    labels[non_ridge & (inhibitor >= cuts["hollow_cut"])] = HOLLOW
    labels[non_ridge & (inhibitor >= cuts["pool_cut"])] = POOL
    return labels


def normalized_activator(activator: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Map the activator to [0,1] using its 2nd/98th percentiles inside ``region``."""
    values = activator[region]
    if values.size == 0:
        return np.zeros_like(activator)
    lo, hi = np.percentile(values, [2.0, 98.0])
    return np.clip((activator - lo) / max(hi - lo, 1.0e-9), 0.0, 1.0)


# ---- deterministic Zhang-Suen thinning ----------------------------------------

_NEIGH = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _neighbors(padded: np.ndarray) -> list[np.ndarray]:
    return [padded[1 + dr : padded.shape[0] - 1 + dr, 1 + dc : padded.shape[1] - 1 + dc]
            for dr, dc in _NEIGH]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen morphological thinning to a 1-pixel skeleton (deterministic)."""
    image = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for sub in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p = _neighbors(padded)  # P2..P9
            b = sum(p)
            transitions = np.zeros(image.shape, dtype=np.int32)
            for i in range(8):
                transitions += ((p[i] == 0) & (p[(i + 1) % 8] == 1)).astype(np.int32)
            if sub == 0:
                c1 = p[0] * p[2] * p[4]
                c2 = p[2] * p[4] * p[6]
            else:
                c1 = p[0] * p[2] * p[6]
                c2 = p[0] * p[4] * p[6]
            flag = (
                (image == 1)
                & (b >= 2)
                & (b <= 6)
                & (transitions == 1)
                & (c1 == 0)
                & (c2 == 0)
            )
            if flag.any():
                image[flag] = 0
                changed = True
    return image.astype(bool)


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    return sum(_neighbors(padded)) * skeleton


@dataclass(frozen=True)
class Topology:
    skeleton: np.ndarray
    junctions: np.ndarray
    terminations: np.ndarray
    branch_count: int
    merge_count: int
    termination_count: int
    junction_count: int
    spacing_samples: np.ndarray


def analyze_topology(
    ridge_mask: np.ndarray,
    flow_row: np.ndarray,
    flow_col: np.ndarray,
) -> Topology:
    skeleton = skeletonize(ridge_mask)
    degree = _skeleton_degree(skeleton)
    junction_mask = skeleton & (degree >= 3)
    termination_mask = skeleton & (degree == 1)

    # Branch vs merge: at a degree-3 node, split incident skeleton neighbors into
    # upstream (against downslope flow) and downstream halves. A downslope split
    # (1 upstream, >=2 downstream) is a branch; the reverse is a merge.
    branch = merge = 0
    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    jr, jc = np.nonzero(junction_mask)
    for r, c in zip(jr.tolist(), jc.tolist()):
        downstream = 0
        upstream = 0
        for dr, dc in _NEIGH:
            if padded[r + 1 + dr, c + 1 + dc] == 0:
                continue
            # dot of neighbor offset with downslope flow at the node
            proj = dr * flow_row[r, c] + dc * flow_col[r, c]
            if proj > 0:
                downstream += 1
            elif proj < 0:
                upstream += 1
        if downstream >= 2 and upstream <= 1:
            branch += 1
        elif upstream >= 2 and downstream <= 1:
            merge += 1
        else:
            # ambiguous crossing counts as neither branch nor merge type
            pass

    # Perpendicular string spacing: 2 x local-maximum of the distance transform
    # away from the ridge skeleton, sampled at ridges of that distance field.
    distance = ndimage.distance_transform_edt(~skeleton)
    local_max = ndimage.maximum_filter(distance, size=3)
    peaks = (distance == local_max) & (distance > 0.5)
    spacing = 2.0 * distance[peaks]

    return Topology(
        skeleton=skeleton,
        junctions=junction_mask,
        terminations=termination_mask,
        branch_count=branch,
        merge_count=merge,
        termination_count=int(termination_mask.sum()),
        junction_count=int(junction_mask.sum()),
        spacing_samples=spacing,
    )


def pool_coupling(labels: np.ndarray, radius_cells: float) -> tuple[float, int, int]:
    """Fraction of pool-perimeter cells within a string of the trough-membership radius.

    A pool cell is on the perimeter if it borders a non-pool cell. It is margin-coupled
    if the nearest ridge (string/hummock) cell lies within ``radius_cells`` - i.e. the
    pool sits in an inter-string trough bounded by a string, rather than floating as an
    isolated socket far from any string (the form-graph failure mode). ``radius_cells`` is
    the trough-membership scale (~half the string spacing).
    """
    pool = labels == POOL
    ridge = labels == RIDGE
    if not pool.any():
        return 1.0, 0, 0
    padded = np.pad(pool.astype(np.uint8), 1, mode="constant")
    exterior_neighbors = np.zeros(pool.shape, dtype=bool)
    for dr, dc in _NEIGH:
        shifted = padded[1 + dr : padded.shape[0] - 1 + dr, 1 + dc : padded.shape[1] - 1 + dc]
        exterior_neighbors |= pool & (shifted == 0)
    distance_to_ridge = ndimage.distance_transform_edt(~ridge)
    perimeter = pool & exterior_neighbors
    coupled = perimeter & (distance_to_ridge <= radius_cells)
    perim_n = int(perimeter.sum())
    coupled_n = int(coupled.sum())
    return (coupled_n / perim_n if perim_n else 1.0), coupled_n, perim_n

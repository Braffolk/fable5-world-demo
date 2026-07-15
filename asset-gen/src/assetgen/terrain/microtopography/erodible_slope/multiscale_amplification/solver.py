"""World-aligned multiscale erosion, thermal transfer, and deposition."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..morphodynamics.assemble import BoundDevelopmentA
from .model import AmplificationResult, ScaleLedger


_DIRECTIONS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


@dataclass(frozen=True)
class _StageResult:
    residual_m: np.ndarray
    flow_m3_s: np.ndarray
    eroded_depth_m: np.ndarray
    deposited_depth_m: np.ndarray
    thermal_delta_m: np.ndarray
    ledger: ScaleLedger


def _pair_slices(
    shape: tuple[int, int], dr: int, dc: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    rows, cols = shape
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(cols, cols - dc)
    return (slice(r0, r1), slice(c0, c1)), (
        slice(r0 + dr, r1 + dr),
        slice(c0 + dc, c1 + dc),
    )


def _sample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask.copy()
    supported = ndimage.minimum_filter(
        mask.astype(np.uint8), size=2 * factor + 1, mode="constant", cval=0
    )
    return supported[::factor, ::factor].astype(bool)


def _sample_mean(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values.copy()
    averaged = ndimage.uniform_filter(
        values.astype(np.float64), size=factor, mode="nearest"
    )
    return averaged[::factor, ::factor]


def _prolong(
    residual: np.ndarray,
    target_shape: tuple[int, int],
    old_texel_m: float,
    new_texel_m: float,
    active: np.ndarray,
) -> np.ndarray:
    rows = np.linspace(0.0, residual.shape[0] - 1, target_shape[0])
    cols = np.linspace(0.0, residual.shape[1] - 1, target_shape[1])
    result = np.empty(target_shape, dtype=np.float64)
    for start in range(0, target_shape[0], 256):
        stop = min(start + 256, target_shape[0])
        rr, cc = np.meshgrid(rows[start:stop], cols, indexing="ij")
        result[start:stop] = ndimage.map_coordinates(
            residual, (rr, cc), order=1, mode="nearest", prefilter=False
        )
    result[~active] = 0.0
    old_area = old_texel_m**2
    new_area = new_texel_m**2
    for positive in (True, False):
        old_values = residual[residual > 0.0] if positive else -residual[residual < 0.0]
        new_values = result[result > 0.0] if positive else -result[result < 0.0]
        old_volume = float(np.sum(old_values, dtype=np.float64) * old_area)
        new_volume = float(np.sum(new_values, dtype=np.float64) * new_area)
        if old_volume > 0.0 and new_volume > 0.0:
            selected = result > 0.0 if positive else result < 0.0
            result[selected] *= old_volume / new_volume
    return result


def _receiver_graph(
    height: np.ndarray,
    active: np.ndarray,
    outlet: np.ndarray,
    filled: np.ndarray,
    rank: np.ndarray,
    texel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = height.shape
    size = height.size
    receiver = np.full(shape, -1, dtype=np.int32)
    best_slope = np.zeros(shape, dtype=np.float32)
    columns = np.arange(shape[1], dtype=np.int64)[None, :]
    rows = np.arange(shape[0], dtype=np.int64)[:, None]
    flat_index = rows * shape[1] + columns
    for dr, dc in _DIRECTIONS:
        source, target = _pair_slices(shape, dr, dc)
        potential = (filled[target] < filled[source]) | (
            (filled[target] == filled[source]) & (rank[target] < rank[source])
        )
        distance = texel_m * (np.sqrt(2.0) if dr and dc else 1.0)
        slope = np.maximum((height[source] - height[target]) / distance, 0.0)
        better = (
            active[source]
            & active[target]
            & potential
            & (slope > best_slope[source])
        )
        best_slope[source][better] = slope[better]
        receiver[source][better] = flat_index[target][better].astype(np.int32)

    fallback = active & ~outlet & (receiver < 0)
    best_fill = np.full(shape, np.inf, dtype=np.float32)
    best_rank = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    for dr, dc in _DIRECTIONS:
        source, target = _pair_slices(shape, dr, dc)
        potential = (filled[target] < filled[source]) | (
            (filled[target] == filled[source]) & (rank[target] < rank[source])
        )
        better = fallback[source] & active[target] & potential & (
            (filled[target] < best_fill[source])
            | ((filled[target] == best_fill[source]) & (rank[target] < best_rank[source]))
        )
        best_fill[source][better] = filled[target][better]
        best_rank[source][better] = rank[target][better]
        receiver[source][better] = flat_index[target][better].astype(np.int32)
    receiver[outlet] = -1
    order = np.flatnonzero(active.ravel())
    order = order[
        np.lexsort((rank.ravel()[order], filled.ravel()[order]))[::-1]
    ].astype(np.int32)
    if np.any(receiver.ravel() >= size):
        raise RuntimeError("multiscale receiver leaves its grid")
    return receiver, best_slope, order


def _accumulate_flow(
    source_m3_s: np.ndarray, receiver: np.ndarray, order: np.ndarray
) -> np.ndarray:
    flow = source_m3_s.ravel().astype(np.float64, copy=True)
    target = receiver.ravel()
    for index in order:
        destination = int(target[index])
        if destination >= 0:
            flow[destination] += flow[index]
    return flow.reshape(source_m3_s.shape)


def _thermal_transfer(
    height: np.ndarray,
    active: np.ndarray,
    repose: np.ndarray,
    texel_m: float,
) -> tuple[np.ndarray, float, float]:
    working = height.copy()
    total_moved = 0.0
    area = texel_m**2
    before = float(np.sum(working, dtype=np.float64) * area)
    row_index, col_index = np.indices(working.shape, dtype=np.int32)
    for iteration in range(3):
        directions = ((0, 1), (1, 1), (1, 0), (1, -1))
        if iteration % 2:
            directions = tuple(reversed(directions))
        for dr, dc in directions:
            source, target = _pair_slices(working.shape, dr, dc)
            distance = texel_m * (np.sqrt(2.0) if dr and dc else 1.0)
            parity = row_index[source] if dr else col_index[source]
            for phase in (0, 1):
                eligible = (
                    active[source]
                    & active[target]
                    & ((parity & 1) == phase)
                )
                difference = working[source] - working[target]
                threshold = 0.5 * (repose[source] + repose[target]) * distance
                transfer = np.minimum(
                    0.20 * np.maximum(np.abs(difference) - threshold, 0.0),
                    0.02 * texel_m,
                )
                transfer *= eligible
                signed = np.sign(difference) * transfer
                working[source] -= signed
                working[target] += signed
                total_moved += float(np.sum(transfer, dtype=np.float64) * area)
    after = float(np.sum(working, dtype=np.float64) * area)
    return working - height, total_moved, after - before


def _stage(
    base: np.ndarray,
    residual: np.ndarray,
    active: np.ndarray,
    outlet: np.ndarray,
    filled: np.ndarray,
    rank: np.ndarray,
    material: np.ndarray,
    vegetation: np.ndarray,
    boundary_source: np.ndarray,
    bound: BoundDevelopmentA,
    texel_m: float,
) -> _StageResult:
    height = base + residual
    receiver, slope, order = _receiver_graph(
        height, active, outlet, filled, rank, texel_m
    )
    runoff = np.zeros(height.shape, dtype=np.float64)
    critical = np.full(height.shape, np.inf, dtype=np.float64)
    detachment = np.zeros(height.shape, dtype=np.float64)
    bulk_density = np.ones(height.shape, dtype=np.float64)
    max_depth = np.zeros(height.shape, dtype=np.float64)
    repose = np.full(height.shape, 1.0, dtype=np.float64)
    for index, name in enumerate(bound.authority.material_rule_names):
        selected = material == index
        rule = bound.process.material_rules[name]
        runoff[selected] = rule["runoff_fraction"]
        critical[selected] = rule["critical_shear_pa"]
        detachment[selected] = rule["detachment_kg_m2_s_pa"]
        bulk_density[selected] = rule["bulk_density_kg_m3"]
        max_depth[selected] = rule["max_erodible_depth_m"]
        repose[selected] = (
            bound.recipe.config.repose_sand_gradient
            if name == "sand_mineral"
            else bound.recipe.config.repose_till_gradient
        )
    source = (
        (bound.process.rainfall_depth_m / bound.process.event_duration_s)
        * runoff
        * np.clip(1.0 - 0.78 * vegetation, 0.08, 1.0)
        * texel_m**2
    )
    source += boundary_source
    source[~active] = 0.0
    flow = _accumulate_flow(source, receiver, order)
    unit_flow = flow / texel_m
    hydraulic_slope = np.maximum(slope.astype(np.float64), bound.process.min_slope)
    depth = np.zeros(height.shape, dtype=np.float64)
    wet = active & (unit_flow > 0.0)
    depth[wet] = (
        unit_flow[wet] * bound.process.manning_n_s_m13 / np.sqrt(hydraulic_slope[wet])
    ) ** (3.0 / 5.0)
    shear = (
        bound.process.water_density_kg_m3
        * bound.process.gravity_m_s2
        * depth
        * slope
    )
    raw_erosion = (
        detachment
        * np.maximum(shear - critical, 0.0)
        * bound.process.event_duration_s
        / bulk_density
    )
    raw_erosion = np.minimum(raw_erosion, max_depth * min(1.0, 2.0 * texel_m))
    raw_erosion[~active] = 0.0
    sigma = max(0.5, bound.process.rill_width_min_m / (2.355 * texel_m))
    eroded = ndimage.gaussian_filter(raw_erosion, sigma=sigma, mode="nearest")
    eroded[~active] = 0.0
    raw_volume = float(np.sum(raw_erosion, dtype=np.float64) * texel_m**2)
    smooth_volume = float(np.sum(eroded, dtype=np.float64) * texel_m**2)
    if raw_volume > 0.0 and smooth_volume > 0.0:
        eroded *= raw_volume / smooth_volume
    eroded = np.minimum(eroded, max_depth)

    eroded_height = height - eroded
    thermal_halo_cells = max(1, int(np.ceil(bound.process.rill_width_max_m / texel_m)))
    thermal_active = active & ndimage.binary_dilation(
        eroded > 1e-12, iterations=thermal_halo_cells, border_value=0
    )
    thermal_delta, thermal_moved, thermal_error = _thermal_transfer(
        eroded_height, thermal_active, repose, texel_m
    )
    sediment = (eroded * texel_m**2).ravel()
    deposited_volume = np.zeros(height.shape, dtype=np.float64).ravel()
    target = receiver.ravel()
    flat_slope = slope.ravel().astype(np.float64)
    flat_active = active.ravel()
    exported = 0.0
    trapped = 0.0
    for index in order:
        available = sediment[index]
        fraction = np.clip(
            (bound.process.deposition_slope_ceiling - flat_slope[index])
            / bound.process.deposition_slope_ceiling,
            0.0,
            1.0,
        )
        deposit = available * 0.45 * fraction
        deposited_volume[index] += deposit
        remaining = available - deposit
        destination = int(target[index])
        if destination >= 0:
            sediment[destination] += remaining
        elif outlet.ravel()[index]:
            exported += remaining
        elif flat_active[index]:
            deposited_volume[index] += remaining
            trapped += remaining
    deposited = deposited_volume.reshape(height.shape) / texel_m**2
    deposited[~active] = 0.0
    new_residual = residual - eroded + thermal_delta + deposited
    new_residual[~active] = 0.0
    eroded_volume = float(np.sum(eroded, dtype=np.float64) * texel_m**2)
    deposited_m3 = float(np.sum(deposited_volume, dtype=np.float64))
    sediment_error = eroded_volume - deposited_m3 - exported
    if abs(sediment_error) > max(1e-10, eroded_volume * 1e-9):
        raise RuntimeError(f"multiscale sediment ledger differs by {sediment_error} m3")
    return _StageResult(
        residual_m=new_residual,
        flow_m3_s=flow,
        eroded_depth_m=eroded,
        deposited_depth_m=deposited,
        thermal_delta_m=thermal_delta,
        ledger=ScaleLedger(
            texel_m=texel_m,
            active_nodes=int(active.sum()),
            routed_nodes=int(np.count_nonzero(receiver >= 0)),
            eroded_m3=eroded_volume,
            deposited_m3=deposited_m3,
            exported_m3=float(exported),
            trapped_m3=float(trapped),
            thermal_m3=float(thermal_moved),
            sediment_error_m3=float(sediment_error),
            thermal_error_m3=float(thermal_error),
        ),
    )


def solve(bound: BoundDevelopmentA) -> AmplificationResult:
    """Run one fixed three-scale absolute-master candidate and crop nowhere."""
    authority = bound.authority
    c0 = authority.c0_node_m
    filled_fine = np.pad(
        bound.canonical_organization.filled_level_m, ((0, 1), (0, 1)), mode="edge"
    )
    rank_fine = np.pad(
        bound.canonical_organization.drain_rank, ((0, 1), (0, 1)), mode="edge"
    )
    boundary_fine = np.pad(
        bound.canonical_boundary.water_inflow_m3_s,
        ((0, 1), (0, 1)),
        mode="constant",
    )
    residual: np.ndarray | None = None
    previous_texel = 0.0
    ledgers: list[ScaleLedger] = []
    final: _StageResult | None = None
    for factor in (4, 2, 1):
        texel = authority.recipe.fine_texel_m * factor
        base = c0[::factor, ::factor]
        active = _sample_mask(authority.form_active_node, factor)
        outlet = ndimage.maximum_filter(
            authority.outlet_node.astype(np.uint8), size=max(1, factor)
        )[::factor, ::factor].astype(bool) & active
        filled = filled_fine[::factor, ::factor]
        rank = rank_fine[::factor, ::factor]
        material = authority.material_rule_node[::factor, ::factor]
        vegetation = _sample_mean(authority.vegetation_node, factor)
        boundary_source = _sample_mean(boundary_fine, factor) * factor**2
        if residual is None:
            residual = np.zeros(base.shape, dtype=np.float64)
        else:
            residual = _prolong(
                residual, base.shape, previous_texel, texel, active
            )
        final = _stage(
            base,
            residual,
            active,
            outlet,
            filled,
            rank,
            material,
            vegetation,
            boundary_source,
            bound,
            texel,
        )
        residual = final.residual_m
        previous_texel = texel
        ledgers.append(final.ledger)
    if final is None or residual is None:
        raise AssertionError("multiscale stage sequence did not run")
    active_fine = authority.form_active_node
    residual[~active_fine] = 0.0
    c1 = c0 + residual
    if not np.isfinite(c1[authority.valid_node]).all():
        raise RuntimeError("multiscale absolute master contains nonfinite valid height")
    protected = ~active_fine
    protected_max = float(np.max(np.abs((c1 - c0)[protected]), initial=0.0))
    if protected_max != 0.0:
        raise RuntimeError("multiscale amplification changed protected C0")
    return AmplificationResult(
        c0_node_m=c0,
        c1_node_m=c1,
        active_node=active_fine,
        flow_m3_s=final.flow_m3_s,
        eroded_depth_m=final.eroded_depth_m,
        deposited_depth_m=final.deposited_depth_m,
        thermal_delta_m=final.thermal_delta_m,
        ledgers=tuple(ledgers),
        maximum_abs_relief_m=float(np.max(np.abs(residual), initial=0.0)),
        protected_max_abs_m=protected_max,
        volume_change_m3=float(
            np.sum(residual, dtype=np.float64) * authority.recipe.fine_texel_m**2
        ),
    )

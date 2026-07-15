"""Dimensional detachment and finite-residence sediment transport."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hydrology import HydrologyResult
from .material_field import MaterialFields
from .state import (
    FineAuthority,
    MorphodynamicState,
    MorphodynamicsConfig,
    NestedBoundaryFlux,
    allocate_array,
)


@dataclass(frozen=True)
class SedimentEvent:
    eroded_kg: float
    deposited_kg: float
    exported_kg: float
    imported_kg: float
    suspended_kg: float
    ponded_kg: float
    mass_error_kg: float


_GL8_NODES = np.asarray(
    [
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ],
    dtype=np.float64,
)
_GL8_WEIGHTS = np.asarray(
    [
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ],
    dtype=np.float64,
)


def update_maturity(
    authority: FineAuthority,
    state: MorphodynamicState,
    material: MaterialFields,
    hydrology: HydrologyResult,
    config: MorphodynamicsConfig,
    *,
    workspace: Path | None,
) -> np.ndarray:
    """Advance the linear maturity ODE exactly and integrate its event power."""
    factor = allocate_array(
        workspace,
        "event-maturity-factor",
        authority.cell_shape,
        np.float32,
    )
    duration = config.event_duration_s
    active = authority.form_active_cell
    for start in range(0, authority.cell_shape[0], 128):
        stop = min(authority.cell_shape[0], start + 128)
        safe_tau = np.maximum(material.critical_shear_pa[start:stop], 1e-6)
        drive = np.maximum(hydrology.shear_pa[start:stop] / safe_tau - 1.0, 0.0)
        initial = state.maturity[start:stop].astype(np.float64)
        growth = drive.astype(np.float64) / config.maturity_growth_s
        rate = growth + 1.0 / config.maturity_heal_s
        equilibrium = np.divide(growth, rate, out=np.zeros_like(growth), where=rate > 0.0)
        final = equilibrium + (initial - equilibrium) * np.exp(-rate * duration)
        average_power = np.zeros(initial.shape, dtype=np.float64)
        for node, weight in zip(_GL8_NODES, _GL8_WEIGHTS, strict=True):
            time_s = 0.5 * duration * (node + 1.0)
            value = equilibrium + (initial - equilibrium) * np.exp(-rate * time_s)
            average_power += 0.5 * weight * np.power(
                np.clip(value, 0.0, 1.0), config.maturity_exponent
            )
        factor_block = config.maturity_floor + (
            1.0 - config.maturity_floor
        ) * average_power
        active_block = active[start:stop]
        state.maturity[start:stop] = np.where(
            active_block, np.clip(final, 0.0, 1.0), 0.0
        ).astype(np.float32)
        factor[start:stop] = np.where(active_block, factor_block, 0.0).astype(
            np.float32
        )
    return factor


def _detach(
    authority: FineAuthority,
    state: MorphodynamicState,
    material: MaterialFields,
    hydrology: HydrologyResult,
    config: MorphodynamicsConfig,
    event_detachment_factor: np.ndarray,
) -> float:
    if event_detachment_factor.shape != authority.cell_shape:
        raise ValueError("event maturity integral differs from the control grid")
    cell_area = authority.recipe.fine_texel_m**2
    total = 0.0
    for start in range(0, authority.cell_shape[0], 128):
        stop = min(authority.cell_shape[0], start + 128)
        candidate = (
            material.detachment_kg_m2_s_pa[start:stop].astype(np.float64)
            * event_detachment_factor[start:stop].astype(np.float64)
            * np.maximum(
                hydrology.shear_pa[start:stop]
                - material.critical_shear_pa[start:stop],
                0.0,
            ).astype(np.float64)
            * config.event_duration_s
            * cell_area
        )
        remaining = (
            state.substrate_limit_kg[start:stop]
            - state.eroded_substrate_kg[start:stop]
        )
        eroded = np.minimum(candidate, np.maximum(remaining, 0.0))
        eroded[~authority.form_active_cell[start:stop]] = 0.0
        state.eroded_substrate_kg[start:stop] += eroded
        state.suspended_kg[start:stop] += eroded
        total += float(np.sum(eroded, dtype=np.float64))
    return total


def _branch_fraction(
    discharge_m3_s: float,
    slope: float,
    length_m: float,
    cell_width_m: float,
    config: MorphodynamicsConfig,
) -> float:
    if discharge_m3_s <= 0.0 or length_m <= 0.0:
        return 0.0
    unit_q = discharge_m3_s / cell_width_m
    depth = (
        unit_q * config.manning_n_s_m13 / np.sqrt(max(slope, 1e-6))
    ) ** (3.0 / 5.0)
    velocity = (
        depth ** (2.0 / 3.0)
        * np.sqrt(max(slope, 1e-6))
        / config.manning_n_s_m13
    )
    travel_s = length_m / max(velocity, 1e-12)
    return config.event_duration_s / (config.event_duration_s + travel_s)


def _transport(
    authority: FineAuthority,
    state: MorphodynamicState,
    hydrology: HydrologyResult,
    config: MorphodynamicsConfig,
) -> tuple[float, float]:
    suspended = state.suspended_kg.ravel()
    deposited = state.deposited_colluvium_kg.ravel()
    form_active = authority.form_active_cell.ravel()
    exits = (authority.outlet_cell | authority.interface_outflow_cell).ravel()
    q = hydrology.discharge_m3_s.ravel()
    slope = hydrology.slope.ravel()
    ta, tb = hydrology.target_a.ravel(), hydrology.target_b.ravel()
    wb = hydrology.weight_b.ravel().astype(np.float64)
    drops = (hydrology.drop_a_m.ravel(), hydrology.drop_b_m.ravel())
    lengths = (
        hydrology.branch_length_a_m.ravel(),
        hydrology.branch_length_b_m.ravel(),
    )
    reservoir_owner = hydrology.reservoir_owner.ravel()
    if np.any(reservoir_owner >= state.ponded_sediment_kg.size):
        raise ValueError("hydrology reservoir owner exceeds the solid inventory")
    ponded_water = hydrology.ponded_water_m3.ravel()
    reservoir_count = state.ponded_sediment_kg.size
    release_cell = np.full(reservoir_count, -1, dtype=np.int32)
    ponded_water_by_reservoir = np.bincount(
        reservoir_owner[reservoir_owner >= 0],
        weights=ponded_water[reservoir_owner >= 0],
        minlength=reservoir_count,
    )
    reservoir_outlets: list[list[tuple[int, float]]] = [
        [] for _ in range(reservoir_count)
    ]
    for owner in range(reservoir_count):
        members = np.flatnonzero(reservoir_owner == owner)
        if members.size:
            release_cell[owner] = members[
                np.argmin(hydrology.drain_rank.ravel()[members])
            ]
        for index in members:
            for target, share in (
                (int(ta[index]), 1.0 - float(wb[index])),
                (int(tb[index]), float(wb[index])),
            ):
                if share <= 0.0 or target == -1:
                    continue
                if target >= 0 and reservoir_owner[target] == owner:
                    continue
                volume = max(float(q[index]), 0.0) * share * config.event_duration_s
                if volume > 0.0:
                    reservoir_outlets[owner].append((target, volume))
    deposited_total = 0.0
    exported_total = 0.0
    for index in hydrology.descending_order:
        available = float(suspended[index])
        owner = int(reservoir_owner[index])
        if owner >= 0:
            state.ponded_sediment_kg[owner] += available
            suspended[index] = 0.0
            if index != release_cell[owner]:
                continue
            outlets = reservoir_outlets[owner]
            overflow_water = sum(value for _, value in outlets)
            total_water = overflow_water + ponded_water_by_reservoir[owner]
            overflow_fraction = overflow_water / total_water if total_water > 0.0 else 0.0
            released = state.ponded_sediment_kg[owner] * overflow_fraction
            state.ponded_sediment_kg[owner] -= released
            if released > 0.0 and overflow_water > 0.0:
                for target, volume in outlets:
                    amount = released * volume / overflow_water
                    if target == -2:
                        exported_total += amount
                    else:
                        suspended[target] += amount
            # Reservoir overflow is an advected existing load. The rank-only
            # fallback supplies no stream power and therefore neither erodes nor
            # deposits terrain at the reservoir spill.
            continue
        if available <= 0.0:
            suspended[index] = 0.0
            continue
        if exits[index]:
            exported_total += available
            suspended[index] = 0.0
            continue
        if not form_active[index]:
            # Protected/unknown/water controls may convey an upstream load but
            # may not own a persistent morphology or sediment inventory.
            sent = 0.0
            if ta[index] >= 0:
                amount = available * (1.0 - wb[index])
                suspended[ta[index]] += amount
                sent += amount
            if tb[index] >= 0:
                amount = available * wb[index]
                suspended[tb[index]] += amount
                sent += amount
            exported_total += available - sent
            suspended[index] = 0.0
            continue
        targets = (int(ta[index]), int(tb[index]))
        shares = (1.0 - float(wb[index]), float(wb[index]))
        assigned_total = 0.0
        residence_retained = 0.0
        depositable_excess = 0.0
        nondeposit_excess = 0.0
        reservoir_excess = 0.0
        branch_payloads: list[tuple[int, float]] = []
        for branch, (target, share) in enumerate(zip(targets, shares, strict=True)):
            if target < 0 or share <= 0.0:
                continue
            branch_q = float(q[index]) * share
            # rho*g*Q*deltaH*dt is energy in joules. The recipe coefficient
            # converts only that branch's available work to a sediment mass.
            capacity = (
                config.transport_kg_per_joule
                * config.water_density_kg_m3
                * config.gravity_m_s2
                * branch_q
                * float(drops[branch][index])
                * config.event_duration_s
            )
            assigned = available * share
            assigned_total += assigned
            transportable = min(assigned, capacity)
            capacity_excess = assigned - transportable
            fraction = _branch_fraction(
                branch_q,
                float(slope[index]),
                float(lengths[branch][index]),
                authority.recipe.fine_texel_m,
                config,
            )
            released = transportable * fraction
            residence_retained += transportable - released
            if capacity_excess > 0.0:
                if owner >= 0 and float(drops[branch][index]) <= 0.0:
                    reservoir_excess += capacity_excess
                elif slope[index] <= config.deposition_slope_ceiling:
                    depositable_excess += capacity_excess
                else:
                    nondeposit_excess += capacity_excess
            if released > 0.0:
                branch_payloads.append((target, released))
        unassigned = max(available - assigned_total, 0.0)
        if owner >= 0:
            state.ponded_sediment_kg[owner] += reservoir_excess
        deposited[index] += depositable_excess
        deposited_total += depositable_excess
        suspended[index] = residence_retained + nondeposit_excess + unassigned
        for target, moving in branch_payloads:
            suspended[target] += moving
    return deposited_total, exported_total


def advance_sediment_event(
    authority: FineAuthority,
    state: MorphodynamicState,
    material: MaterialFields,
    hydrology: HydrologyResult,
    boundary: NestedBoundaryFlux,
    config: MorphodynamicsConfig,
    event_detachment_factor: np.ndarray,
) -> SedimentEvent:
    """Advance one authoritative physical event; no level repeats its history."""
    boundary.validate(authority.cell_shape)
    if np.any((boundary.sediment_inflow_kg_s > 0.0) & ~authority.routing_cell):
        raise ValueError("parent sediment enters outside fine routing support")
    imported_total = float(
        np.sum(boundary.sediment_inflow_kg_s, dtype=np.float64)
        * config.event_duration_s
    )
    state.suspended_kg += boundary.sediment_inflow_kg_s * config.event_duration_s
    state.ledger.imported_sediment_kg += imported_total
    eroded = _detach(
        authority,
        state,
        material,
        hydrology,
        config,
        event_detachment_factor,
    )
    deposited, exported = _transport(authority, state, hydrology, config)
    state.ledger.exported_sediment_kg += exported
    residual = state.sediment_residual_kg()
    scale = eroded + state.ledger.imported_sediment_kg
    tolerance = max(1e-8, scale * config.sediment_relative_tolerance)
    if abs(residual) > tolerance:
        raise RuntimeError(f"solid ledger differs by {residual} kg")
    state.ledger.maximum_step_error_kg = max(
        state.ledger.maximum_step_error_kg, abs(residual)
    )
    return SedimentEvent(
        eroded_kg=eroded,
        deposited_kg=deposited,
        exported_kg=exported,
        imported_kg=imported_total,
        suspended_kg=float(np.sum(state.suspended_kg, dtype=np.float64)),
        ponded_kg=float(np.sum(state.ponded_sediment_kg, dtype=np.float64)),
        mass_error_kg=residual,
    )

"""Causal runoff, seepage, detachment, transport, and deposition solve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ProcessConfig, SlopeDomain
from .routing import RoutingResult, route_continuous


@dataclass(frozen=True)
class ProcessResult:
    routing: RoutingResult
    runoff_depth_m: np.ndarray
    seep_depth_m: np.ndarray
    flow_depth_m: np.ndarray
    shear_stress_pa: np.ndarray
    detached_kg: np.ndarray
    deposited_kg: np.ndarray
    transported_kg: np.ndarray
    exported_sediment_kg: float
    trapped_sediment_kg: float
    eroded_depth_m: np.ndarray
    deposited_depth_m: np.ndarray
    generated_sediment_kg: float
    mass_balance_error_kg: float


def _material_field(
    domain: SlopeDomain,
    config: ProcessConfig,
    field: str,
) -> np.ndarray:
    result = np.zeros(domain.height_m.shape, dtype=np.float64)
    for index, name in enumerate(domain.material_rule_names):
        if name not in config.material_rules:
            raise ValueError(f"condition bundle references absent material rule {name}")
        result[domain.material_rule == index] = config.material_rules[name][field]
    return result


def solve_process(domain: SlopeDomain, config: ProcessConfig) -> ProcessResult:
    """Solve one declared event over the complete physical drainage domain."""
    if domain.texel_m != config.process_texel_m:
        raise ValueError("condition grid and process config texel differ")
    active = domain.active
    form_active = domain.form_active
    source_active = domain.hydrologic_source_active
    runoff_fraction = _material_field(domain, config, "runoff_fraction")
    seep_flux = _material_field(domain, config, "seep_flux_m_s")
    vegetation_attenuation = np.clip(1.0 - 0.78 * domain.vegetation_cover, 0.08, 1.0)
    runoff_depth = np.where(
        source_active,
        config.rainfall_depth_m * runoff_fraction * vegetation_attenuation,
        0.0,
    )
    seep_depth = np.where(
        source_active,
        seep_flux * domain.seep_likelihood * config.event_duration_s,
        0.0,
    )
    routing = route_continuous(
        domain,
        local_water_depth_m=runoff_depth + seep_depth,
    )

    discharge_m3_s = routing.water_m3 / config.event_duration_s
    unit_discharge_m2_s = discharge_m3_s / domain.texel_m
    hydraulic_slope = np.maximum(routing.slope, config.min_slope * 0.25)
    flow_depth = np.where(
        active & (unit_discharge_m2_s > 0.0),
        (
            unit_discharge_m2_s
            * config.manning_n_s_m13
            / np.sqrt(hydraulic_slope)
        )
        ** (3.0 / 5.0),
        0.0,
    )
    shear = (
        config.water_density_kg_m3
        * config.gravity_m_s2
        * flow_depth
        * routing.slope
    )
    critical_shear = _material_field(domain, config, "critical_shear_pa")
    detach_coefficient = _material_field(
        domain, config, "detachment_kg_m2_s_pa"
    )
    bulk_density = _material_field(domain, config, "bulk_density_kg_m3")
    max_erodible_depth = _material_field(domain, config, "max_erodible_depth_m")
    area_activation = np.clip(
        np.log2(
            np.maximum(routing.contributing_area_m2, config.rill_area_threshold_m2)
            / config.rill_area_threshold_m2
        )
        / 4.0,
        0.0,
        1.0,
    )
    slope_activation = np.clip(
        (routing.slope - config.min_slope) / max(config.min_slope * 4.0, 1e-12),
        0.0,
        1.0,
    )
    detachment_rate = (
        detach_coefficient
        * np.maximum(shear - critical_shear, 0.0)
        * area_activation
        * slope_activation
    )
    cell_area = domain.texel_m**2
    detached = np.where(
        form_active,
        detachment_rate * config.event_duration_s * cell_area,
        0.0,
    )
    detached = np.minimum(
        detached,
        max_erodible_depth * bulk_density * cell_area,
    )

    target_a = routing.target_a.ravel()
    target_b = routing.target_b.ravel()
    weight_a = routing.weight_a.ravel()
    weight_b = routing.weight_b.ravel()
    sediment = np.zeros(domain.height_m.size, dtype=np.float64)
    deposited = np.zeros_like(sediment)
    transported = np.zeros_like(sediment)
    order = np.flatnonzero(active.ravel())
    order = order[np.argsort(domain.height_m.ravel()[order])[::-1]]
    outlet = domain.outlet.ravel()
    exported = 0.0
    trapped = 0.0
    detached_flat = detached.ravel()
    slope_flat = routing.slope.ravel()
    discharge_flat = discharge_m3_s.ravel()
    for index in order:
        available = sediment[index] + detached_flat[index]
        if outlet[index]:
            exported += available
            transported[index] = available
            continue
        ta = target_a[index]
        tb = target_b[index]
        if ta < 0 and tb < 0:
            deposited[index] += available
            trapped += available
            continue
        stream_power_w = (
            config.water_density_kg_m3
            * config.gravity_m_s2
            * discharge_flat[index]
            * slope_flat[index]
        )
        transport_capacity = (
            config.transport_kg_per_joule
            * stream_power_w
            * config.event_duration_s
        )
        moving = min(available, transport_capacity)
        local_deposit = available - moving
        if slope_flat[index] > config.deposition_slope_ceiling:
            moving = available
            local_deposit = 0.0
        deposited[index] += local_deposit
        transported[index] = moving
        if ta >= 0:
            sediment[ta] += moving * weight_a[index]
        if tb >= 0:
            sediment[tb] += moving * weight_b[index]

    generated = float(np.sum(detached))
    deposited_total = float(np.sum(deposited))
    error = generated - deposited_total - exported
    tolerance = max(1e-12, generated * config.mass_balance_relative_tolerance)
    if abs(error) > tolerance:
        raise RuntimeError(
            "sediment accounting is not conservative: "
            f"generated={generated}, deposited={deposited_total}, "
            f"exported={exported}, error={error} kg"
        )
    safe_bulk = np.where(bulk_density > 0.0, bulk_density, 1.0)
    eroded_depth = detached / (safe_bulk * cell_area)
    deposited_depth = deposited.reshape(domain.height_m.shape) / (
        config.deposit_bulk_density_kg_m3 * cell_area
    )
    return ProcessResult(
        routing=routing,
        runoff_depth_m=runoff_depth,
        seep_depth_m=seep_depth,
        flow_depth_m=flow_depth,
        shear_stress_pa=shear,
        detached_kg=detached,
        deposited_kg=deposited.reshape(domain.height_m.shape),
        transported_kg=transported.reshape(domain.height_m.shape),
        exported_sediment_kg=exported,
        trapped_sediment_kg=trapped,
        eroded_depth_m=eroded_depth,
        deposited_depth_m=deposited_depth,
        generated_sediment_kg=generated,
        mass_balance_error_kg=error,
    )

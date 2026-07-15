"""Input-only typed process opportunity for erodible-slope site qualification."""
from __future__ import annotations

from typing import Any

import numpy as np

from .refine_sites import _material_support

_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _material_field(
    arrays: dict[str, np.ndarray],
    material_supported: np.ndarray,
    config: dict[str, Any],
    field: str,
) -> np.ndarray:
    geology = arrays["geology_lithology_code"]
    result = np.zeros(geology.shape, dtype=np.float64)
    sand = material_supported & (geology == 40)
    till = material_supported & (geology == 50)
    result[sand] = float(config["material_rules"]["sand_mineral"][field])
    result[till] = float(config["material_rules"]["till_mineral"][field])
    if np.any(material_supported & ~sand & ~till):
        raise ValueError("material support contains an unbound C1 material rule")
    return result


def _conservative_multiple_flow(
    height: np.ndarray,
    active: np.ndarray,
    local_water_depth_m: np.ndarray,
    texel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route only to strictly lower neighbors; unresolved storage abstains.

    This is a selection diagnostic, not the synthesis solver. It deliberately
    omits depression spilling, so it cannot invent downstream runoff opportunity.
    """
    rows, columns = height.shape
    flat_active = np.flatnonzero(active)
    flat_height = height.ravel()
    order = flat_active[np.lexsort((flat_active, -flat_height[flat_active]))]
    water = (local_water_depth_m * texel_m**2).ravel().astype(np.float64)
    area = np.where(active, texel_m**2, 0.0).ravel()
    slope = np.zeros(height.size, dtype=np.float64)
    for index in order.tolist():
        row, column = divmod(index, columns)
        targets: list[int] = []
        gradients: list[float] = []
        for drow, dcolumn in _NEIGHBORS:
            next_row = row + drow
            next_column = column + dcolumn
            if not (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and active[next_row, next_column]
                and height[next_row, next_column] < height[row, column]
            ):
                continue
            distance = texel_m * (
                np.sqrt(2.0) if drow != 0 and dcolumn != 0 else 1.0
            )
            targets.append(next_row * columns + next_column)
            gradients.append(
                (height[row, column] - height[next_row, next_column]) / distance
            )
        if not gradients:
            continue
        weights = np.asarray(gradients, dtype=np.float64)
        slope[index] = float(np.max(weights))
        weights /= np.sum(weights)
        for target, weight in zip(targets, weights.tolist(), strict=True):
            water[target] += water[index] * weight
            area[target] += area[index] * weight
    return (
        water.reshape(height.shape),
        area.reshape(height.shape),
        slope.reshape(height.shape),
    )


def typed_process_opportunity(
    arrays: dict[str, np.ndarray],
    soil_document: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Derive runoff and seep opportunity without generating terrain."""
    texel_m = float(config["process_texel_m"])
    if texel_m != 1.0:
        raise ValueError("qualification is defined on the bound 1 m evidence grid")
    material_supported = _material_support(arrays, soil_document)
    valid = arrays["valid"].astype(bool, copy=False)
    solve = arrays["solve_domain"].astype(bool, copy=False)
    upstream = arrays["upstream_domain"].astype(bool, copy=False)
    collar = arrays["collar"].astype(bool, copy=False)
    outlet = arrays["outlet"].astype(bool, copy=False)
    unknown = arrays["unknown"].astype(bool, copy=False)
    water = arrays["water"].astype(bool, copy=False)
    objects = arrays["object"].astype(bool, copy=False)
    non_heightfield = arrays["non_heightfield"].astype(bool, copy=False)
    protected = arrays["protected_structure"].astype(bool, copy=False)
    hard_exclusion = water | objects | non_heightfield | protected
    routing_barrier = objects | non_heightfield | (water & ~outlet)
    routing_active = solve & ~collar & valid & (~routing_barrier | outlet)
    source_active = (
        routing_active & upstream & ~unknown & ~outlet & material_supported
    )
    form_active = source_active & ~hard_exclusion

    runoff_fraction = _material_field(
        arrays, material_supported, config, "runoff_fraction"
    )
    vegetation_attenuation = np.clip(
        1.0 - 0.78 * arrays["vegetation_evidence"].astype(np.float64),
        0.08,
        1.0,
    )
    local_runoff = np.where(
        source_active,
        float(config["rainfall_depth_m"])
        * runoff_fraction
        * vegetation_attenuation,
        0.0,
    )
    water_m3, contributing_area_m2, slope = _conservative_multiple_flow(
        arrays["height"].astype(np.float64),
        routing_active,
        local_runoff,
        texel_m,
    )
    event_duration_s = float(config["event_duration_s"])
    unit_discharge_m2_s = water_m3 / event_duration_s / texel_m
    hydraulic_slope = np.maximum(slope, np.finfo(np.float64).tiny)
    flow_depth_m = np.where(
        unit_discharge_m2_s > 0.0,
        (
            unit_discharge_m2_s
            * float(config["manning_n_s_m13"])
            / np.sqrt(hydraulic_slope)
        )
        ** (3.0 / 5.0),
        0.0,
    )
    shear_pa = (
        float(config["water_density_kg_m3"])
        * float(config["gravity_m_s2"])
        * flow_depth_m
        * slope
    )
    critical_shear_pa = _material_field(
        arrays, material_supported, config, "critical_shear_pa"
    )
    runoff = (
        form_active
        & (contributing_area_m2 >= float(config["rill_area_threshold_m2"]))
        & (slope >= float(config["min_slope"]))
        & (shear_pa > critical_shear_pa)
    )
    seep = form_active & (
        arrays["topographic_seep_support_likelihood"].astype(np.float64) > 0.0
    )
    if np.any((runoff | seep) & (~form_active | hard_exclusion | unknown)):
        raise RuntimeError("typed opportunity escaped form-active hard exclusions")
    return {
        "runoff": runoff,
        "seep": seep,
        "form_active": form_active,
        "routing_active": routing_active,
        "material_supported": material_supported,
        "diagnostics": {
            "routing_active_cells": int(np.count_nonzero(routing_active)),
            "form_active_cells": int(np.count_nonzero(form_active)),
            "runoff_opportunity_cells": int(np.count_nonzero(runoff)),
            "seep_opportunity_cells": int(np.count_nonzero(seep)),
            "maximum_contributing_area_m2": float(
                np.max(contributing_area_m2[form_active], initial=0.0)
            ),
            "maximum_runoff_shear_pa": float(
                np.max(shear_pa[form_active], initial=0.0)
            ),
        },
    }

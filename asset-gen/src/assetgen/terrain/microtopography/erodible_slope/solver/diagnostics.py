"""Focused rotation, partition, and domain-support qualifications."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .forms import FormPlan, render_fine_surface
from .model import ProcessConfig, SlopeDomain
from .process import ProcessResult
from .routing import route_continuous


def partition_error(
    domain: SlopeDomain,
    process: ProcessResult,
    plan: FormPlan,
    config: ProcessConfig,
    bbox_en: tuple[float, float, float, float],
) -> float:
    whole = render_fine_surface(
        domain,
        process,
        plan,
        config,
        bbox_en=bbox_en,
        row_block=None,
    )
    partitioned = render_fine_surface(
        domain,
        process,
        plan,
        config,
        bbox_en=bbox_en,
        row_block=137,
    )
    return float(np.max(np.abs(whole.c1_height_m - partitioned.c1_height_m)))


def rotation_error(config: ProcessConfig) -> dict[str, float]:
    """Qualify scalar routing on a non-cardinal analytic plane under 90-degree rotation."""
    side = 96
    row, col = np.indices((side, side), dtype=np.float64)
    height = 100.0 - 0.031 * col - 0.057 * row
    solve = np.zeros((side, side), dtype=bool)
    solve[2:-2, 2:-2] = True
    upstream = np.zeros_like(solve)
    upstream[2, 2:-2] = True
    upstream[2:-2, 2] = True
    outlet = np.zeros_like(solve)
    outlet[-3, 2:-2] = True
    outlet[2:-2, -3] = True
    valid = np.ones_like(solve)
    zero = np.zeros_like(solve)
    material = np.full((side, side), -1, dtype=np.int16)
    material[solve] = 0
    domain = SlopeDomain(
        site_id="rotation_qualification",
        bbox_en=(0.0, 0.0, float(side), float(side)),
        texel_m=config.process_texel_m,
        height_m=height,
        valid=valid,
        solve_domain=solve,
        upstream_domain=upstream,
        outlet=outlet,
        collar=~solve,
        routing_barrier=zero,
        hard_exclusion=zero,
        unknown=zero,
        vegetation_cover=np.zeros_like(height),
        seep_likelihood=np.zeros_like(height),
        upstream_water_m3=np.zeros_like(height),
        material_rule=material,
        material_rule_names=(next(iter(config.material_rules)),),
        source_identity={"kind": "analytic_non_cardinal_plane"},
    )
    water = np.where(solve, 0.004, 0.0)
    original = route_continuous(domain, local_water_depth_m=water)

    rotate = lambda value: np.rot90(value, k=1).copy()
    rotated = replace(
        domain,
        height_m=rotate(domain.height_m),
        valid=rotate(domain.valid),
        solve_domain=rotate(domain.solve_domain),
        upstream_domain=rotate(domain.upstream_domain),
        outlet=rotate(domain.outlet),
        collar=rotate(domain.collar),
        routing_barrier=rotate(domain.routing_barrier),
        hard_exclusion=rotate(domain.hard_exclusion),
        unknown=rotate(domain.unknown),
        vegetation_cover=rotate(domain.vegetation_cover),
        seep_likelihood=rotate(domain.seep_likelihood),
        upstream_water_m3=rotate(domain.upstream_water_m3),
        material_rule=rotate(domain.material_rule),
    )
    turned = route_continuous(rotated, local_water_depth_m=rotate(water))
    expected_water = rotate(original.water_m3)
    expected_area = rotate(original.contributing_area_m2)
    water_bound = max(float(np.max(expected_water)), 1e-12)
    area_bound = max(float(np.max(expected_area)), 1e-12)
    return {
        "rotation_water_max_abs_m3": float(
            np.max(np.abs(expected_water - turned.water_m3))
        ),
        "rotation_water_max_relative": float(
            np.max(np.abs(expected_water - turned.water_m3)) / water_bound
        ),
        "rotation_area_max_abs_m2": float(
            np.max(np.abs(expected_area - turned.contributing_area_m2))
        ),
        "rotation_area_max_relative": float(
            np.max(np.abs(expected_area - turned.contributing_area_m2)) / area_bound
        ),
    }


def support_radius_m(plan: FormPlan) -> float:
    values = (
        plan.rill_radius_m,
        plan.headcut_radius_m,
        plan.seep_radius_m,
    )
    maxima = [float(np.max(value)) for value in values if value.size]
    return max(maxima, default=0.0)

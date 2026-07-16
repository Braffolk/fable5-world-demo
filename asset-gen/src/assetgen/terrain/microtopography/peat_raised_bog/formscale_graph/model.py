"""Domain-scale strings, flarks, pools, and hummock-hollow organization."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
from scipy import ndimage

from ....repair.prolong import prolong_structural_4x


@dataclass(frozen=True)
class Result:
    measured_parent_m: np.ndarray
    synthesized_parent_m: np.ndarray
    parent_delta_m: np.ndarray
    measured_fine_m: np.ndarray
    final_fine_m: np.ndarray
    fine_delta_m: np.ndarray
    travel_distance_m: np.ndarray
    broad_height_m: np.ndarray
    slope: np.ndarray
    string_strength: np.ndarray
    flark_strength: np.ndarray
    pool_strength: np.ndarray
    pool_margin_strength: np.ndarray
    hummock_strength: np.ndarray
    hollow_strength: np.ndarray
    form_class: np.ndarray


def _nearest_fill(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = ndimage.distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    return value[tuple(indices)]


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    unit = np.clip((value - low) / max(high - low, 1.0e-12), 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def _crest_mask(
    broad: np.ndarray,
    authority: np.ndarray,
    interior: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    method = config["method"]
    crests = np.zeros_like(authority)
    components, count = ndimage.label(authority)
    for component_id in range(1, count + 1):
        component = components == component_id
        eligible = component & (
            interior >= float(method["crest_minimum_interior_m"])
        )
        if not np.any(eligible):
            eligible = component
        threshold = float(
            np.percentile(broad[eligible], method["crest_percentile"])
        )
        high = eligible & (broad >= threshold)
        high = ndimage.binary_closing(high, iterations=3) & component
        if not np.any(high):
            row, col = np.unravel_index(
                np.argmax(np.where(eligible, broad, -np.inf)), broad.shape
            )
            high[row, col] = True
        crests |= high
    return crests


def _travel_distance(
    authority: np.ndarray,
    crest: np.ndarray,
    broad: np.ndarray,
) -> np.ndarray:
    """Eight-neighbor geodesic distance with a mild uphill penalty."""
    height, width = authority.shape
    distance = np.full(authority.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for row, col in np.argwhere(crest):
        distance[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))
    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 2.0**0.5), (-1, 1, 2.0**0.5),
        (1, -1, 2.0**0.5), (1, 1, 2.0**0.5),
    )
    while queue:
        current, row, col = heapq.heappop(queue)
        if current != distance[row, col]:
            continue
        for dr, dc, step in neighbors:
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= height or cc < 0 or cc >= width or not authority[rr, cc]:
                continue
            uphill = max(float(broad[rr, cc] - broad[row, col]), 0.0)
            candidate = current + step * (1.0 + 22.0 * uphill)
            if candidate < distance[rr, cc]:
                distance[rr, cc] = candidate
                heapq.heappush(queue, (candidate, rr, cc))
    if np.any(~np.isfinite(distance[authority])):
        raise ValueError("every supported authority component requires a crest owner")
    distance[~authority] = 0.0
    return distance


def _pool_field(
    filled: np.ndarray,
    authority: np.ndarray,
    interior: np.ndarray,
    slope: np.ndarray,
    method: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = ndimage.gaussian_filter(filled, 4.0) - ndimage.gaussian_filter(filled, 17.0)
    central = authority & (interior >= 55.0) & (slope <= 0.00105)
    threshold = float(np.percentile(local[central], 12.0))
    candidate = central & (local <= threshold)
    candidate = ndimage.binary_opening(candidate, iterations=2)
    candidate = ndimage.binary_closing(candidate, iterations=3)
    labels, count = ndimage.label(candidate)
    pool_mask = np.zeros_like(authority)
    kept = 0
    for label in range(1, count + 1):
        selected = labels == label
        area = int(np.count_nonzero(selected))
        if int(method["pool_minimum_area_m2"]) <= area <= int(method["pool_maximum_area_m2"]):
            pool_mask |= selected
            kept += 1
    if kept == 0:
        return np.zeros_like(filled), np.zeros_like(filled), pool_mask
    inside = ndimage.distance_transform_edt(pool_mask)
    pool = np.where(pool_mask, 1.0 - np.exp(-inside / 2.5), 0.0)
    outside = ndimage.distance_transform_edt(~pool_mask)
    margin = np.where(~pool_mask, np.exp(-0.5 * ((outside - 2.0) / 1.3) ** 2), 0.0)
    margin *= authority
    return pool, margin, pool_mask


def _form_graph(
    height: np.ndarray,
    authority: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    method = config["method"]
    filled = _nearest_fill(height, authority)
    broad = ndimage.gaussian_filter(filled, float(method["broad_sigma_m"]))
    gy, gx = np.gradient(broad)
    slope = np.hypot(gx, gy)
    interior = ndimage.distance_transform_edt(authority)
    crest = _crest_mask(broad, authority, interior, config)
    travel = _travel_distance(authority, crest, broad)

    sigma_small, sigma_large = (float(v) for v in method["medium_sigmas_m"])
    medium = ndimage.gaussian_filter(filled, sigma_small) - ndimage.gaussian_filter(
        filled, sigma_large
    )
    medium_scale = max(float(np.percentile(np.abs(medium[authority]), 95.0)), 1.0e-6)
    medium_unit = np.clip(medium / medium_scale, -2.0, 2.0)

    max_travel = max(float(np.percentile(travel[authority], 98.0)), 1.0)
    radial = np.clip(travel / max_travel, 0.0, 1.0)
    inner = float(method["string_spacing_inner_m"])
    outer = float(method["string_spacing_outer_m"])
    spacing = inner + (outer - inner) * _smoothstep(radial, 0.12, 0.88)
    warped_cycles = travel / spacing + 0.20 * medium_unit
    phase = np.mod(warped_cycles, 1.0)
    ridge_distance = np.minimum(phase, 1.0 - phase)
    flark_distance = np.abs(phase - 0.5)

    slope_gate = _smoothstep(
        slope,
        float(method["string_slope_start"]),
        float(method["string_slope_full"]),
    )
    interior_gate = _smoothstep(
        interior, float(method["minimum_form_interior_m"]), 28.0
    )
    form_gate = authority * slope_gate * interior_gate
    string = np.exp(
        -0.5 * (ridge_distance / float(method["string_half_width_fraction"])) ** 2
    ) * form_gate
    flark = np.exp(
        -0.5 * (flark_distance / float(method["flark_half_width_fraction"])) ** 2
    ) * form_gate

    pool, pool_margin, pool_mask = _pool_field(
        filled, authority, interior, slope, method
    )
    string *= 1.0 - ndimage.gaussian_filter(pool_mask.astype(np.float64), 4.0)
    flark *= 1.0 - pool

    # Measured medium structure breaks long bands into site-conditioned form complexes.
    positive = _smoothstep(medium_unit, -0.15, 0.85)
    negative = _smoothstep(-medium_unit, -0.10, 0.90)
    hummock = string * (0.30 + 0.70 * positive)
    hollow = flark * (0.28 + 0.72 * negative)

    edge_taper = _smoothstep(interior, 2.0, 14.0)
    parent_delta = (
        0.075 * string
        + 0.035 * hummock
        - 0.065 * flark
        - 0.040 * hollow
        - 0.155 * pool
        + 0.050 * pool_margin
    )
    # Preserve broad mass locally without erasing connected form geometry.
    parent_delta -= ndimage.gaussian_filter(parent_delta * authority, 48.0)
    parent_delta *= edge_taper * authority
    limit = float(method["parent_relief_limit_m"])
    maximum = float(np.max(np.abs(parent_delta)))
    if maximum > limit:
        parent_delta *= limit / maximum

    form_class = np.zeros_like(height, dtype=np.uint8)
    form_class[(flark > 0.45) & authority] = 1
    form_class[(string > 0.45) & authority] = 2
    form_class[(hollow > 0.58) & authority] = 3
    form_class[(hummock > 0.58) & authority] = 4
    form_class[pool_margin > 0.35] = 5
    form_class[pool_mask] = 6
    return (
        parent_delta, travel, broad, slope, string, flark, pool, pool_margin,
        hummock, hollow, form_class, medium_unit,
    )


def simulate(
    height: np.ndarray,
    authority: np.ndarray,
    config: dict[str, Any],
) -> Result:
    if height.shape != authority.shape:
        raise ValueError("height and authority must share the whole-mire 1 m grid")
    if not np.isfinite(height).all() or not np.any(authority):
        raise ValueError("invalid measured whole-mire carrier")
    (
        parent_delta, travel, broad, slope, string, flark, pool, pool_margin,
        hummock, hollow, form_class, medium_unit,
    ) = _form_graph(height, authority, config)
    synthesized_parent = height + parent_delta

    measured_support = np.pad(height, 2, mode="reflect")
    synthesized_support = np.pad(synthesized_parent, 2, mode="reflect")
    rows = (2, height.shape[0] + 2)
    cols = (2, height.shape[1] + 2)
    measured_fine = prolong_structural_4x(
        measured_support, parent_rows=rows, parent_cols=cols
    )
    final_fine = prolong_structural_4x(
        synthesized_support, parent_rows=rows, parent_cols=cols
    )

    fine_authority = np.repeat(np.repeat(authority, 4, axis=0), 4, axis=1)
    # Submetre relief follows the global form field, and is small enough not to expose
    # the conservation cells that close it to the synthesized parent.
    fine_carrier = ndimage.zoom(
        0.55 * medium_unit + 0.30 * (hummock - hollow),
        4,
        order=3,
        mode="nearest",
        prefilter=True,
    )
    fine_carrier = ndimage.gaussian_filter(fine_carrier, 0.55)
    micro = float(config["method"]["fine_relief_limit_m"]) * np.tanh(fine_carrier)
    micro *= fine_authority
    blocks = micro.reshape(height.shape[0], 4, height.shape[1], 4)
    blocks -= np.mean(blocks, axis=(1, 3), dtype=np.float64)[:, None, :, None]
    final_fine += micro
    final_fine[~fine_authority] = measured_fine[~fine_authority]

    return Result(
        measured_parent_m=height,
        synthesized_parent_m=synthesized_parent,
        parent_delta_m=parent_delta,
        measured_fine_m=measured_fine,
        final_fine_m=final_fine,
        fine_delta_m=final_fine - measured_fine,
        travel_distance_m=travel,
        broad_height_m=broad,
        slope=slope,
        string_strength=string,
        flark_strength=flark,
        pool_strength=pool,
        pool_margin_strength=pool_margin,
        hummock_strength=hummock,
        hollow_strength=hollow,
        form_class=form_class,
    )

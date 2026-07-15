"""Whole-domain headcut retreat, connected incision, and conservative toe events."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .forms import FineSurface, _fine_samples, _indices_to_en, _nearest_line_relief, _sample
from .model import ProcessConfig, SlopeDomain
from .process import ProcessResult, _material_field

FORM_NONE = 0
FORM_GULLY = 1
FORM_HEADCUT = 2
FORM_TOE = 4


@dataclass(frozen=True)
class HeadcutEventConfig:
    values: dict[str, Any]

    @property
    def physical(self) -> dict[str, float]:
        return self.values["physical_parameters"]

    @property
    def budgets(self) -> dict[str, float | int]:
        return self.values["budgets"]

    @property
    def gates(self) -> dict[str, float | int]:
        return self.values["recognizable_form_gates"]

    @property
    def seed(self) -> int:
        return int(self.values["canonical_seed"])

    @property
    def synthesis_bbox_en(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.values["synthesis_bbox_en"])


@dataclass(frozen=True)
class HeadcutEvent:
    event_id: int
    seed_en: tuple[float, float]
    age_years: float
    path_points_en: np.ndarray
    path_radius_m: np.ndarray
    path_depth_m: np.ndarray
    head_en: tuple[float, float]
    head_downhill_en: tuple[float, float]
    head_half_width_m: float
    head_depth_m: float
    fan_origin_en: tuple[float, float]
    fan_downhill_en: tuple[float, float]
    fan_length_m: float
    fan_half_width_m: float
    fan_amplitude_m: float
    erosion_volume_m3: float
    deposition_volume_m3: float
    retreat_length_m: float
    incision_length_m: float
    opportunity_peak: float
    seep_peak: float


@dataclass(frozen=True)
class HeadcutEventPlan:
    events: tuple[HeadcutEvent, ...]
    opportunity: np.ndarray
    opportunity_mask: np.ndarray
    candidate_components: int
    rejected_components: dict[str, int]
    erosion_volume_m3: float
    deposition_volume_m3: float
    exported_volume_m3: float
    maximum_support_radius_m: float


def load_headcut_event_config(path: Path) -> HeadcutEventConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "bindings",
        "implementation",
        "canonical_seed",
        "synthesis_bbox_en",
        "physical_parameters",
        "budgets",
        "recognizable_form_gates",
        "negative_controls",
        "exclusions",
    }
    if set(document) != required:
        raise ValueError("headcut-event config fields differ from frozen contract")
    if document["schema_version"] != "laas.erodible-slope-headcut-event-config/1":
        raise ValueError("unsupported headcut-event config")
    physical = document["physical_parameters"]
    expected_physical = {
        "candidate_halo_m",
        "minimum_contributing_area_m2",
        "minimum_slope_gradient",
        "maximum_slope_gradient",
        "minimum_shear_ratio",
        "minimum_seep_likelihood",
        "minimum_downstream_slope_break",
        "minimum_failure_opportunity",
        "minimum_failure_patch_area_m2",
        "failure_footprint_sigma_m",
        "event_age_min_years",
        "event_age_max_years",
        "retreat_length_min_m",
        "retreat_length_max_m",
        "downstream_length_min_m",
        "downstream_length_max_m",
        "gully_width_min_m",
        "gully_width_max_m",
        "incision_depth_min_m",
        "incision_depth_max_m",
        "headcut_width_min_m",
        "headcut_width_max_m",
        "headcut_depth_max_m",
        "toe_slope_ceiling",
        "fan_length_min_m",
        "fan_length_max_m",
        "fan_half_width_min_m",
        "fan_half_width_max_m",
        "fan_relief_max_m",
    }
    if set(physical) != expected_physical:
        raise ValueError("headcut-event physical parameters differ from contract")
    numeric = np.asarray(list(physical.values()), dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
        raise ValueError("headcut-event physical parameters must be positive and finite")
    if physical["minimum_slope_gradient"] >= physical["maximum_slope_gradient"]:
        raise ValueError("headcut-event slope interval is empty")
    if physical["retreat_length_min_m"] > physical["retreat_length_max_m"]:
        raise ValueError("headcut retreat interval is reversed")
    if physical["downstream_length_min_m"] > physical["downstream_length_max_m"]:
        raise ValueError("gully length interval is reversed")
    if physical["incision_depth_max_m"] > physical["headcut_depth_max_m"]:
        raise ValueError("gully incision exceeds headcut depth ceiling")
    return HeadcutEventConfig(document)


def _primary_targets(process: ProcessResult) -> np.ndarray:
    a = process.routing.target_a.ravel()
    b = process.routing.target_b.ravel()
    choose_b = process.routing.weight_b.ravel() > process.routing.weight_a.ravel()
    return np.where(choose_b, b, a).astype(np.int64, copy=False)


def _downstream_field(values: np.ndarray, process: ProcessResult) -> np.ndarray:
    flat = values.ravel()
    result = np.zeros(flat.shape, dtype=np.float64)
    for target, weight in (
        (process.routing.target_a.ravel(), process.routing.weight_a.ravel()),
        (process.routing.target_b.ravel(), process.routing.weight_b.ravel()),
    ):
        used = target >= 0
        result[used] += weight[used] * flat[target[used]]
    return result.reshape(values.shape)


def _failure_opportunity(
    domain: SlopeDomain,
    process: ProcessResult,
    process_config: ProcessConfig,
    config: HeadcutEventConfig,
    *,
    hydrology_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    p = config.physical
    if not hydrology_enabled:
        return np.zeros(domain.height_m.shape, dtype=np.float64), np.zeros(
            domain.height_m.shape, dtype=bool
        )
    critical = _material_field(domain, process_config, "critical_shear_pa")
    erodible_depth = _material_field(domain, process_config, "max_erodible_depth_m")
    shear_ratio = np.divide(
        process.shear_stress_pa,
        critical,
        out=np.zeros_like(process.shear_stress_pa),
        where=critical > 0.0,
    )
    downhill_slope = _downstream_field(process.routing.slope, process)
    slope_break = np.maximum(downhill_slope - process.routing.slope, 0.0)
    east = process.routing.downhill_east
    south = process.routing.downhill_south
    magnitude = np.hypot(east, south)
    unit_east = np.divide(east, magnitude, out=np.zeros_like(east), where=magnitude > 0)
    unit_south = np.divide(south, magnitude, out=np.zeros_like(south), where=magnitude > 0)
    divergence = np.gradient(unit_east, domain.texel_m, axis=1) + np.gradient(
        unit_south, domain.texel_m, axis=0
    )
    convergence = np.clip(-divergence * 2.0, 0.0, 1.0)
    shear_drive = np.maximum(shear_ratio - p["minimum_shear_ratio"], 0.0)
    seep_drive = domain.seep_likelihood * np.clip(
        slope_break / p["minimum_downstream_slope_break"], 0.0, 2.0
    )
    material_depth = np.clip(erodible_depth / 0.18, 0.0, 1.0)
    rooted_resistance = np.clip(1.0 - 0.45 * domain.vegetation_cover, 0.35, 1.0)
    raw = material_depth * rooted_resistance * (
        shear_drive * (0.55 + 0.45 * convergence) + 0.75 * seep_drive
    )
    sigma = p["failure_footprint_sigma_m"] / domain.texel_m
    opportunity = ndimage.gaussian_filter(raw, sigma=sigma, mode="nearest")

    e_min, n_min, e_max, n_max = config.synthesis_bbox_en
    halo = p["candidate_halo_m"]
    rows, cols = np.indices(domain.height_m.shape, dtype=np.float64)
    east_m, north_m = _indices_to_en(domain, rows, cols)
    fixed_window = (
        (east_m >= e_min - halo)
        & (east_m < e_max + halo)
        & (north_m > n_min - halo)
        & (north_m <= n_max + halo)
    )
    causal = (
        (shear_ratio >= p["minimum_shear_ratio"])
        | (
            (domain.seep_likelihood >= p["minimum_seep_likelihood"])
            & (slope_break >= p["minimum_downstream_slope_break"])
        )
    )
    eligible = (
        domain.form_active
        & fixed_window
        & causal
        & (process.routing.contributing_area_m2 >= p["minimum_contributing_area_m2"])
        & (process.routing.slope >= p["minimum_slope_gradient"])
        & (process.routing.slope <= p["maximum_slope_gradient"])
        & (opportunity >= p["minimum_failure_opportunity"])
    )
    return opportunity, eligible


def _event_rng(seed: int, east: float, north: float) -> np.random.Generator:
    identity = f"{seed}:{east:.3f}:{north:.3f}".encode("ascii")
    digest = hashlib.sha256(identity).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _trace_downstream(
    start: int,
    primary: np.ndarray,
    domain: SlopeDomain,
    process: ProcessResult,
    *,
    minimum_length_m: float,
    maximum_length_m: float,
    toe_slope_ceiling: float,
) -> list[int]:
    path = [start]
    current = start
    maximum_steps = int(np.ceil(maximum_length_m / domain.texel_m))
    for _ in range(maximum_steps):
        target = int(primary[current])
        if target < 0 or target == current or not domain.form_active.ravel()[target]:
            break
        path.append(target)
        current = target
        length = (len(path) - 1) * domain.texel_m
        if length >= minimum_length_m and process.routing.slope.ravel()[current] <= toe_slope_ceiling:
            break
    return path


def _donor_csr(primary: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources = np.flatnonzero(active.ravel() & (primary >= 0))
    order = np.argsort(primary[sources], kind="stable")
    sources = sources[order]
    targets = primary[sources]
    counts = np.bincount(targets, minlength=primary.size)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return sources, offsets


def _trace_retreat(
    start: int,
    donors: np.ndarray,
    offsets: np.ndarray,
    opportunity: np.ndarray,
    domain: SlopeDomain,
    process: ProcessResult,
    maximum_length_m: float,
) -> list[int]:
    path = [start]
    current = start
    visited = {start}
    maximum_steps = int(np.ceil(maximum_length_m / domain.texel_m))
    flat_opportunity = opportunity.ravel()
    flat_area = process.routing.contributing_area_m2.ravel()
    for _ in range(maximum_steps):
        candidates = donors[offsets[current] : offsets[current + 1]]
        candidates = candidates[
            domain.form_active.ravel()[candidates]
            & ~np.isin(candidates, np.fromiter(visited, dtype=np.int64))
        ]
        if not candidates.size:
            break
        score = flat_opportunity[candidates] * np.sqrt(np.maximum(flat_area[candidates], 1.0))
        next_cell = int(candidates[np.argmax(score)])
        visited.add(next_cell)
        path.append(next_cell)
        current = next_cell
    return path


def _cells_to_points(domain: SlopeDomain, cells: list[int]) -> np.ndarray:
    rows, cols = np.unravel_index(np.asarray(cells, dtype=np.int64), domain.height_m.shape)
    east, north = _indices_to_en(domain, rows.astype(np.float64), cols.astype(np.float64))
    return np.column_stack((east, north))


def _densify(points: np.ndarray, spacing_m: float = 0.25) -> np.ndarray:
    values: list[np.ndarray] = []
    for start, stop in zip(points[:-1], points[1:], strict=True):
        length = float(np.linalg.norm(stop - start))
        count = max(1, int(np.ceil(length / spacing_m)))
        phase = np.arange(count, dtype=np.float64) / count
        values.append(start[None, :] + phase[:, None] * (stop - start)[None, :])
    values.append(points[-1:])
    return np.concatenate(values)


def _headcut_relief(
    event: HeadcutEvent,
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    de = east - event.head_en[0]
    dn = north - event.head_en[1]
    downhill = np.asarray(event.head_downhill_en)
    cross_axis = np.asarray((-downhill[1], downhill[0]))
    along = de * downhill[0] + dn * downhill[1]
    cross = de * cross_axis[0] + dn * cross_axis[1]
    lateral_q = np.abs(cross) / event.head_half_width_m
    lateral = np.where(lateral_q < 1.0, (1.0 - lateral_q**2) ** 2, 0.0)
    lip = np.clip((along + 0.20) / 0.45, 0.0, 1.0)
    lip = lip * lip * (3.0 - 2.0 * lip)
    decay = np.where((along >= -0.20) & (along <= 3.0), np.exp(-np.maximum(along, 0.0) / 2.0), 0.0)
    return event.head_depth_m * lateral * lip * decay


def _fan_raw(event: HeadcutEvent, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    de = east - event.fan_origin_en[0]
    dn = north - event.fan_origin_en[1]
    downhill = np.asarray(event.fan_downhill_en)
    cross_axis = np.asarray((-downhill[1], downhill[0]))
    along = de * downhill[0] + dn * downhill[1]
    cross = de * cross_axis[0] + dn * cross_axis[1]
    t = np.clip(along / event.fan_length_m, 0.0, 1.0)
    half_width = event.path_radius_m[-1] + t * (
        event.fan_half_width_m - event.path_radius_m[-1]
    )
    q = np.abs(cross) / np.maximum(half_width, 1e-9)
    lateral = np.where(q < 1.0, (1.0 - q**2) ** 2, 0.0)
    longitudinal = np.where(
        (along >= 0.0) & (along <= event.fan_length_m),
        4.0 * t * (1.0 - t),
        0.0,
    )
    return lateral * longitudinal


def _event_incisions(event: HeadcutEvent, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    line = _nearest_line_relief(
        event.path_points_en,
        event.path_radius_m,
        event.path_depth_m,
        east,
        north,
    )
    head = _headcut_relief(event, east, north)
    return line, head


def _local_quadrature(
    event: HeadcutEvent,
    domain: SlopeDomain,
    fine_texel_m: float,
) -> tuple[float, float]:
    points = event.path_points_en
    margin = max(
        float(np.max(event.path_radius_m, initial=0.0)),
        event.head_half_width_m,
        event.fan_half_width_m,
    ) + event.fan_length_m + fine_texel_m
    e0 = fine_texel_m * np.floor(
        (min(float(np.min(points[:, 0])), event.fan_origin_en[0]) - margin) / fine_texel_m
    )
    e1 = fine_texel_m * np.ceil(
        (max(float(np.max(points[:, 0])), event.fan_origin_en[0]) + margin) / fine_texel_m
    )
    n0 = fine_texel_m * np.floor(
        (min(float(np.min(points[:, 1])), event.fan_origin_en[1]) - margin) / fine_texel_m
    )
    n1 = fine_texel_m * np.ceil(
        (max(float(np.max(points[:, 1])), event.fan_origin_en[1]) + margin) / fine_texel_m
    )
    cols = int(np.ceil((e1 - e0) / fine_texel_m)) + 1
    rows = int(np.ceil((n1 - n0) / fine_texel_m)) + 1
    east = e0 + np.arange(cols) * fine_texel_m
    north = n1 - np.arange(rows) * fine_texel_m
    east, north = np.meshgrid(east, north)
    line, head = _event_incisions(event, east, north)
    support = (
        (_sample(domain, domain.form_active, east, north, order=0) >= 0.5)
        & (_sample(domain, domain.hard_exclusion, east, north, order=0) < 0.5)
    )
    erosion = float(np.sum(np.where(support, np.maximum(line, head), 0.0)) * fine_texel_m**2)
    fan_raw = float(np.sum(np.where(support, _fan_raw(event, east, north), 0.0)) * fine_texel_m**2)
    return erosion, fan_raw


def build_headcut_event_plan(
    domain: SlopeDomain,
    process: ProcessResult,
    process_config: ProcessConfig,
    config: HeadcutEventConfig,
    *,
    hydrology_enabled: bool = True,
) -> HeadcutEventPlan:
    p = config.physical
    opportunity, candidate = _failure_opportunity(
        domain, process, process_config, config, hydrology_enabled=hydrology_enabled
    )
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    primary = _primary_targets(process)
    donors, offsets = _donor_csr(primary, domain.form_active)
    rejected = {
        "patch_too_small": 0,
        "route_too_short": 0,
        "no_eligible_toe": 0,
        "fan_relief_ceiling": 0,
        "event_budget": 0,
    }
    candidates: list[tuple[float, int, int]] = []
    minimum_cells = int(np.ceil(p["minimum_failure_patch_area_m2"] / domain.texel_m**2))
    for label in range(1, count + 1):
        if sizes[label] < minimum_cells:
            rejected["patch_too_small"] += 1
            continue
        cells = np.flatnonzero(labels.ravel() == label)
        seed_cell = int(cells[np.argmax(opportunity.ravel()[cells])])
        candidates.append((float(opportunity.ravel()[seed_cell]), seed_cell, label))
    candidates.sort(key=lambda value: (-value[0], value[1]))
    if len(candidates) > int(config.budgets["maximum_events"]):
        rejected["event_budget"] = len(candidates) - int(config.budgets["maximum_events"])
        candidates = candidates[: int(config.budgets["maximum_events"])]

    events: list[HeadcutEvent] = []
    for peak, seed_cell, _ in candidates:
        seed_row, seed_col = np.unravel_index(seed_cell, domain.height_m.shape)
        seed_e, seed_n = _indices_to_en(
            domain, np.asarray(float(seed_row)), np.asarray(float(seed_col))
        )
        rng = _event_rng(config.seed, float(seed_e), float(seed_n))
        severity = float(np.clip(peak / 1.5, 0.0, 1.0))
        age = float(rng.uniform(p["event_age_min_years"], p["event_age_max_years"]))
        age_decay = float(np.exp(-0.18 * age))
        retreat_limit = p["retreat_length_min_m"] + severity * (
            p["retreat_length_max_m"] - p["retreat_length_min_m"]
        )
        downstream_limit = p["downstream_length_min_m"] + severity * (
            p["downstream_length_max_m"] - p["downstream_length_min_m"]
        )
        retreat = _trace_retreat(
            seed_cell, donors, offsets, opportunity, domain, process, retreat_limit
        )
        downstream = _trace_downstream(
            seed_cell,
            primary,
            domain,
            process,
            minimum_length_m=p["downstream_length_min_m"],
            maximum_length_m=downstream_limit,
            toe_slope_ceiling=p["toe_slope_ceiling"],
        )
        if len(downstream) * domain.texel_m < p["downstream_length_min_m"]:
            rejected["route_too_short"] += 1
            continue
        toe_cell = downstream[-1]
        if (
            not domain.form_active.ravel()[toe_cell]
            or process.routing.slope.ravel()[toe_cell] > p["toe_slope_ceiling"]
        ):
            rejected["no_eligible_toe"] += 1
            continue
        cells = list(reversed(retreat[1:])) + downstream
        coarse_points = _cells_to_points(domain, cells)
        dense_points = _densify(coarse_points)
        route_area = _sample(
            domain,
            process.routing.contributing_area_m2,
            dense_points[:, 0],
            dense_points[:, 1],
            order=1,
        )
        area_scale = np.clip(
            np.log2(np.maximum(route_area, p["minimum_contributing_area_m2"]) / p["minimum_contributing_area_m2"])
            / 5.0,
            0.0,
            1.0,
        )
        width = p["gully_width_min_m"] + np.sqrt(area_scale) * (
            p["gully_width_max_m"] - p["gully_width_min_m"]
        )
        path_radius = 0.5 * width
        depth = (
            p["incision_depth_min_m"]
            + severity * (p["incision_depth_max_m"] - p["incision_depth_min_m"])
        ) * age_decay
        path_depth = depth * (0.72 + 0.28 * np.sqrt(area_scale))
        head = coarse_points[0]
        downhill_vector = coarse_points[min(1, len(coarse_points) - 1)] - head
        norm = float(np.linalg.norm(downhill_vector))
        if norm <= 0.0:
            rejected["route_too_short"] += 1
            continue
        head_downhill = downhill_vector / norm
        fan_origin = coarse_points[-1]
        fan_vector = coarse_points[-1] - coarse_points[-2]
        fan_norm = float(np.linalg.norm(fan_vector))
        fan_downhill = fan_vector / max(fan_norm, 1e-12)
        head_half_width = 0.5 * (
            p["headcut_width_min_m"]
            + severity * (p["headcut_width_max_m"] - p["headcut_width_min_m"])
        )
        head_depth = min(depth * 1.25, p["headcut_depth_max_m"])
        fan_length = p["fan_length_min_m"] + severity * (
            p["fan_length_max_m"] - p["fan_length_min_m"]
        )
        fan_half_width = p["fan_half_width_min_m"] + severity * (
            p["fan_half_width_max_m"] - p["fan_half_width_min_m"]
        )
        provisional = HeadcutEvent(
            event_id=len(events),
            seed_en=(float(seed_e), float(seed_n)),
            age_years=age,
            path_points_en=dense_points,
            path_radius_m=path_radius,
            path_depth_m=path_depth,
            head_en=(float(head[0]), float(head[1])),
            head_downhill_en=(float(head_downhill[0]), float(head_downhill[1])),
            head_half_width_m=head_half_width,
            head_depth_m=head_depth,
            fan_origin_en=(float(fan_origin[0]), float(fan_origin[1])),
            fan_downhill_en=(float(fan_downhill[0]), float(fan_downhill[1])),
            fan_length_m=fan_length,
            fan_half_width_m=fan_half_width,
            fan_amplitude_m=1.0,
            erosion_volume_m3=0.0,
            deposition_volume_m3=0.0,
            retreat_length_m=(len(retreat) - 1) * domain.texel_m,
            incision_length_m=(len(cells) - 1) * domain.texel_m,
            opportunity_peak=peak,
            seep_peak=float(np.max(domain.seep_likelihood.ravel()[np.asarray(cells)])),
        )
        erosion_volume, fan_raw_volume = _local_quadrature(
            provisional, domain, process_config.fine_texel_m
        )
        if erosion_volume <= 0.0 or fan_raw_volume <= 0.0:
            rejected["no_eligible_toe"] += 1
            continue
        fan_amplitude = erosion_volume / fan_raw_volume
        if fan_amplitude > p["fan_relief_max_m"]:
            rejected["fan_relief_ceiling"] += 1
            continue
        events.append(
            HeadcutEvent(
                **{
                    **provisional.__dict__,
                    "event_id": len(events),
                    "fan_amplitude_m": fan_amplitude,
                    "erosion_volume_m3": erosion_volume,
                    "deposition_volume_m3": erosion_volume,
                }
            )
        )

    erosion = float(sum(event.erosion_volume_m3 for event in events))
    deposition = float(sum(event.deposition_volume_m3 for event in events))
    maximum_support = max(
        max(
            float(
                np.max(
                    np.linalg.norm(
                        event.path_points_en - np.asarray(event.seed_en)[None, :], axis=1
                    ),
                    initial=0.0,
                )
            )
            + max(float(np.max(event.path_radius_m, initial=0.0)), event.head_half_width_m),
            float(np.linalg.norm(np.asarray(event.fan_origin_en) - np.asarray(event.seed_en)))
            + event.fan_length_m
            + event.fan_half_width_m,
        )
        for event in events
    ) if events else 0.0
    return HeadcutEventPlan(
        events=tuple(events),
        opportunity=opportunity,
        opportunity_mask=candidate,
        candidate_components=count,
        rejected_components=rejected,
        erosion_volume_m3=erosion,
        deposition_volume_m3=deposition,
        exported_volume_m3=max(0.0, erosion - deposition),
        maximum_support_radius_m=maximum_support,
    )


def render_headcut_event_surface(
    domain: SlopeDomain,
    plan: HeadcutEventPlan,
    process_config: ProcessConfig,
    *,
    bbox_en: tuple[float, float, float, float],
    row_block: int | None = None,
) -> FineSurface:
    east, north = _fine_samples(bbox_en, process_config.fine_texel_m)
    block = min(256, east.shape[0]) if row_block is None else row_block
    c0 = np.empty(east.shape, dtype=np.float64)
    incision = np.zeros(east.shape, dtype=np.float64)
    deposition = np.zeros(east.shape, dtype=np.float64)
    ownership = np.zeros(east.shape, dtype=np.uint8)
    hard = np.empty(east.shape, dtype=bool)
    material = np.empty(east.shape, dtype=bool)
    for row0 in range(0, east.shape[0], block):
        row1 = min(row0 + block, east.shape[0])
        ee = east[row0:row1]
        nn = north[row0:row1]
        c0[row0:row1] = _sample(domain, domain.height_m, ee, nn, order=3)
        hard_block = _sample(domain, domain.hard_exclusion, ee, nn, order=0) >= 0.5
        material_block = _sample(domain, domain.form_active, ee, nn, order=0) >= 0.5
        hard[row0:row1] = hard_block
        material[row0:row1] = material_block
        for event in plan.events:
            path_min = np.min(event.path_points_en, axis=0)
            path_max = np.max(event.path_points_en, axis=0)
            reach = max(
                float(np.max(event.path_radius_m, initial=0.0)),
                event.head_half_width_m,
                event.fan_half_width_m,
            ) + event.fan_length_m
            if (
                float(np.max(ee)) < path_min[0] - reach
                or float(np.min(ee)) > path_max[0] + reach
                or float(np.max(nn)) < path_min[1] - reach
                or float(np.min(nn)) > path_max[1] + reach
            ):
                continue
            line, head = _event_incisions(event, ee, nn)
            event_incision = np.maximum(line, head)
            event_fan = event.fan_amplitude_m * _fan_raw(event, ee, nn)
            event_incision = np.where(material_block & ~hard_block, event_incision, 0.0)
            event_fan = np.where(material_block & ~hard_block, event_fan, 0.0)
            existing = incision[row0:row1]
            head_owner = head >= line
            changed = event_incision > existing
            ownership_block = ownership[row0:row1]
            ownership_block[changed & ~head_owner] = FORM_GULLY
            ownership_block[changed & head_owner] = FORM_HEADCUT
            toe_changed = event_fan > deposition[row0:row1]
            ownership_block[toe_changed & (event_fan > event_incision)] = FORM_TOE
            incision[row0:row1] += event_incision
            deposition[row0:row1] += event_fan
    residual = deposition - incision
    residual[hard | ~material] = 0.0
    incision[hard | ~material] = 0.0
    deposition[hard | ~material] = 0.0
    ownership[hard | ~material] = FORM_NONE
    return FineSurface(
        bbox_en=bbox_en,
        texel_m=process_config.fine_texel_m,
        c0_height_m=c0,
        c1_height_m=c0 + residual,
        residual_m=residual,
        incision_m=incision,
        deposition_m=deposition,
        ownership=ownership,
        hard_exclusion=hard,
        material_supported=material,
    )

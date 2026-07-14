"""Candidate-independent source B2 phase, spectrum, form, and capacity gate."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal.windows import hann

from .archive import SourceGrid
from .bands import PhaseBands
from .resample import common_overlay, complete_common_cells


@dataclass(frozen=True)
class Form:
    group_order: int
    plot_order: int
    phase: int
    sign: int
    extremum_index: int
    centroid_xy_m: tuple[float, float]
    prominence_m: float
    equivalent_diameter_m: float
    relief_volume_m3: float

    @property
    def canonical_id(self) -> tuple[int, int, int, int, int]:
        return (self.group_order, self.plot_order, self.phase, self.sign, self.extremum_index)


@dataclass(frozen=True)
class PlotMaterialization:
    source: SourceGrid
    phases: tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands]
    overlay_x_bounds_m: np.ndarray
    overlay_y_bounds_m: np.ndarray
    common_support: np.ndarray
    complete_common_cells: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    windows: tuple[dict[str, Any], ...]
    spectral_cores: tuple[tuple[np.ndarray, ...], ...]
    forms: tuple[tuple[Form, ...], ...]
    plot_metrics: dict[str, Any]


def _overlay_metrics(
    bands: tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands],
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    common: np.ndarray,
) -> dict[str, Any]:
    x_mid = 0.5 * (x_bounds[:-1] + x_bounds[1:])
    y_mid = 0.5 * (y_bounds[:-1] + y_bounds[1:])
    areas = np.diff(y_bounds)[:, None] * np.diff(x_bounds)[None, :]
    values: list[np.ndarray] = []
    for item in bands:
        grid = item.derived_b2
        xi = np.searchsorted(grid.x_bounds_m, x_mid, side="right") - 1
        yi = np.searchsorted(grid.y_bounds_m, y_mid, side="right") - 1
        # The union overlay extends beyond some phase grids. Those rectangles
        # are already false in ``common``; clipping only makes their ignored
        # gather index safe and does not grant them area or support.
        clipped_x = np.clip(xi, 0, grid.values.shape[1] - 1)
        clipped_y = np.clip(yi, 0, grid.values.shape[0] - 1)
        values.append(grid.values[clipped_y[:, None], clipped_x[None, :]])
    area = float(np.sum(areas[common], dtype=np.float64))
    energy_integrals = [float(np.sum(value[common] ** 2 * areas[common], dtype=np.float64)) for value in values]
    disagreement_integrals = {
        f"{p}-{q}": float(np.sum((values[p][common] - values[q][common]) ** 2 * areas[common], dtype=np.float64))
        for p, q in itertools.combinations(range(4), 2)
    }
    return {
        "common_area_m2": area,
        "phase_energy_integral_m4": energy_integrals,
        "pair_disagreement_integral_m4": disagreement_integrals,
    }


def _candidate_centers(lo: float, hi: float, anchor: float, stride: float = 0.5) -> np.ndarray:
    first = math.floor((lo - anchor) / stride) - 1
    last = math.ceil((hi - anchor) / stride) + 1
    return anchor + np.arange(first, last + 1, dtype=np.float64) * stride


def _indices_in_half_open(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((centers >= lo - 1e-12) & (centers < hi - 1e-12))


def enumerate_windows(
    source: SourceGrid,
    bands: tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands],
    common_cells: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[np.ndarray, ...], ...]]:
    anchor_x = math.floor(source.source_boundary_origin_xy_m[0] * 32.0) / 32.0
    anchor_y = math.floor(source.source_boundary_origin_xy_m[1] * 32.0) / 32.0
    all_windows: list[dict[str, Any]] = []
    all_cores: list[tuple[np.ndarray, ...]] = []
    for phase_index, (item, common) in enumerate(zip(bands, common_cells, strict=True)):
        grid = item.derived_b2
        xs, ys = grid.x_centers_m, grid.y_centers_m
        centers_x = _candidate_centers(grid.x_bounds_m[0], grid.x_bounds_m[-1], anchor_x)
        centers_y = _candidate_centers(grid.y_bounds_m[0], grid.y_bounds_m[-1], anchor_y)
        phase_cores: list[np.ndarray] = []
        for cy in centers_y:
            support_y = _indices_in_half_open(ys, cy - 1.0, cy + 1.0)
            core_y = _indices_in_half_open(ys, cy - 0.5, cy + 0.5)
            for cx in centers_x:
                support_x = _indices_in_half_open(xs, cx - 1.0, cx + 1.0)
                core_x = _indices_in_half_open(xs, cx - 0.5, cx + 0.5)
                support_shape = (support_y.size, support_x.size)
                core_shape = (core_y.size, core_x.size)
                support_valid = (
                    support_shape == (32, 32)
                    and bool(np.all(grid.valid[np.ix_(support_y, support_x)]))
                )
                core_valid = (
                    core_shape == (16, 16)
                    and bool(np.all(common[np.ix_(core_y, core_x)]))
                )
                accepted = support_valid and core_valid
                # The source PSD operator is defined on every complete 1 m core;
                # model-window acceptance additionally requires its 2 m support.
                if core_valid:
                    phase_cores.append(grid.values[np.ix_(core_y, core_x)].copy())
                all_windows.append(
                    {
                        "phase": phase_index,
                        "center_xy_m": [float(cx), float(cy)],
                        "support_shape_yx": list(support_shape),
                        "core_shape_yx": list(core_shape),
                        "support_valid": support_valid,
                        "core_valid": core_valid,
                        "accepted": accepted,
                    }
                )
        all_cores.append(tuple(phase_cores))
    return tuple(all_windows), tuple(all_cores)


def _spectral_sums(cores: tuple[np.ndarray, ...]) -> tuple[np.ndarray, int]:
    frequency = np.fft.fftfreq(16, d=0.0625)
    radial = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    accumulated = np.zeros((16, 16), dtype=np.float64)
    window_1d = hann(16, sym=False).astype(np.float64)
    window = window_1d[:, None] * window_1d[None, :]
    for core in cores:
        centered = core.astype(np.float64) - float(np.mean(core, dtype=np.float64))
        spectrum = np.fft.fft2(centered * window)
        accumulated += spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    return np.stack((radial, accumulated)), len(cores)


NEIGHBORS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy != 0 or dx != 0
)


def _forms(
    values: np.ndarray,
    valid: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    group_order: int,
    plot_order: int,
    phase: int,
) -> tuple[Form, ...]:
    result: list[Form] = []
    plot_values = values[valid]
    if plot_values.size == 0:
        return ()
    mad = float(np.median(np.abs(plot_values - np.median(plot_values))))
    prominence_min = 2.0 * mad
    ny, nx = values.shape
    for sign in (-1, 1):
        signed = sign * values
        parent = np.full((ny, nx), -1, dtype=np.int64)
        for row, column in zip(*np.nonzero(valid), strict=True):
            current_value = signed[row, column]
            choices: list[tuple[float, int]] = []
            for dy, dx in NEIGHBORS:
                rr, cc = row + dy, column + dx
                if 0 <= rr < ny and 0 <= cc < nx and valid[rr, cc] and signed[rr, cc] > current_value:
                    choices.append((float(signed[rr, cc]), rr * nx + cc))
            parent[row, column] = min(choices, key=lambda pair: (-pair[0], pair[1]))[1] if choices else row * nx + column

        def root(index: int) -> int:
            trail: list[int] = []
            while True:
                row, column = divmod(index, nx)
                next_index = int(parent[row, column])
                if next_index == index:
                    break
                trail.append(index)
                index = next_index
            for old in trail:
                rr, cc = divmod(old, nx)
                parent[rr, cc] = index
            return index

        labels: dict[int, list[int]] = {}
        for index in np.flatnonzero(valid.ravel()):
            labels.setdefault(root(int(index)), []).append(int(index))
        root_for = np.full((ny, nx), -1, dtype=np.int64)
        for basin_root, indices in labels.items():
            root_for.ravel()[indices] = basin_root
        for basin_root, indices in sorted(labels.items()):
            boundary_saddles: list[float] = []
            touches_boundary = False
            for index in indices:
                row, column = divmod(index, nx)
                for dy, dx in NEIGHBORS:
                    rr, cc = row + dy, column + dx
                    if not (0 <= rr < ny and 0 <= cc < nx) or not valid[rr, cc]:
                        touches_boundary = True
                    elif root_for[rr, cc] != basin_root:
                        boundary_saddles.append(min(float(signed[row, column]), float(signed[rr, cc])))
            if touches_boundary or not boundary_saddles:
                continue
            saddle = max(boundary_saddles)
            peak = float(signed.ravel()[basin_root])
            prominence = peak - saddle
            area = len(indices) * 0.0625 * 0.0625
            diameter = 2.0 * math.sqrt(area / math.pi)
            if prominence < prominence_min or not (0.125 <= diameter <= 0.5):
                continue
            relief = np.maximum(signed.ravel()[indices] - saddle, 0.0)
            volume = float(np.sum(relief, dtype=np.float64) * 0.0625 * 0.0625)
            if volume <= 0:
                continue
            rows, columns = np.divmod(np.asarray(indices), nx)
            result.append(
                Form(
                    group_order=group_order,
                    plot_order=plot_order,
                    phase=phase,
                    sign=sign,
                    extremum_index=basin_root,
                    centroid_xy_m=(float(np.mean(x_centers[columns])), float(np.mean(y_centers[rows]))),
                    prominence_m=prominence,
                    equivalent_diameter_m=diameter,
                    relief_volume_m3=volume,
                )
            )
    result.sort(key=lambda item: item.canonical_id)
    return tuple(result)


def materialize_plot(
    source: SourceGrid,
    phases: tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands],
    *,
    group_order: int,
    plot_order: int,
) -> PlotMaterialization:
    b2_grids = tuple(item.derived_b2 for item in phases)
    overlay_x, overlay_y, common, _, _ = common_overlay(b2_grids)
    common_cells = tuple(
        complete_common_cells(grid, overlay_x, overlay_y, common) for grid in b2_grids
    )
    windows, cores = enumerate_windows(source, phases, common_cells)
    forms = tuple(
        _forms(
            item.derived_b2.values, mask, item.derived_b2.x_centers_m,
            item.derived_b2.y_centers_m, group_order=group_order,
            plot_order=plot_order, phase=phase_index,
        )
        for phase_index, (item, mask) in enumerate(zip(phases, common_cells, strict=True))
    )
    return PlotMaterialization(
        source=source,
        phases=phases,
        overlay_x_bounds_m=overlay_x,
        overlay_y_bounds_m=overlay_y,
        common_support=common,
        complete_common_cells=common_cells,
        windows=windows,
        spectral_cores=cores,
        forms=forms,
        plot_metrics=_overlay_metrics(phases, overlay_x, overlay_y, common),
    )


def _distance(a: Form, b: Form) -> float:
    return math.hypot(a.centroid_xy_m[0] - b.centroid_xy_m[0], a.centroid_xy_m[1] - b.centroid_xy_m[1])


def _families(forms: list[Form]) -> list[tuple[Form, ...]]:
    by_phase = {phase: [form for form in forms if form.phase == phase] for phase in range(4)}
    families: list[tuple[Form, ...]] = []
    for count in (3, 4):
        for phases in itertools.combinations(range(4), count):
            for candidate in itertools.product(*(by_phase[phase] for phase in phases)):
                if all(_distance(a, b) <= 0.125 + 1e-12 for a, b in itertools.combinations(candidate, 2)):
                    families.append(tuple(candidate))
    families.sort(key=lambda family: tuple(form.canonical_id for form in family))
    return families


def _select_disjoint(families: list[tuple[Form, ...]]) -> tuple[tuple[Form, ...], ...]:
    if not families:
        return ()
    form_ids = sorted({form.canonical_id for family in families for form in family})
    bit_for = {form_id: 1 << index for index, form_id in enumerate(form_ids)}
    masks = [sum(bit_for[form.canonical_id] for form in family) for family in families]
    weights = [sum(form.relief_volume_m3 for form in family) for family in families]
    canonical = [tuple(form.canonical_id for form in family) for family in families]

    @lru_cache(maxsize=None)
    def solve(index: int, used: int) -> tuple[float, tuple[int, ...]]:
        if index == len(families):
            return 0.0, ()
        without = solve(index + 1, used)
        if masks[index] & used:
            return without
        rest_weight, rest_ids = solve(index + 1, used | masks[index])
        with_value = (weights[index] + rest_weight, (index, *rest_ids))
        if with_value[0] > without[0]:
            return with_value
        if with_value[0] < without[0]:
            return without
        with_key = tuple(canonical[item] for item in with_value[1])
        without_key = tuple(canonical[item] for item in without[1])
        return with_value if with_key < without_key else without

    _, chosen = solve(0, 0)
    return tuple(families[index] for index in chosen)


def evaluate_source_gate(
    plots: list[PlotMaterialization], group_order: list[str], thresholds: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    for group_id in group_order:
        members = [plot for plot in plots if plot.source.group_id == group_id]
        area = sum(plot.plot_metrics["common_area_m2"] for plot in members)
        row: dict[str, Any] = {
            "group_id": group_id,
            "plot_count": len(members),
            "common_area_m2": area,
            "common_b2_cells": sum(int(np.count_nonzero(mask)) for plot in members for mask in plot.complete_common_cells),
            "strict_b1_valid_cells": sum(int(np.count_nonzero(phase.b1.valid)) for plot in members for phase in plot.phases),
        }
        if area <= 0:
            row.update({"usable": False, "state": "failed", "failed_reason": "zero_common_b2_area"})
            group_rows.append(row)
            continue
        e_integrals = np.sum(
            np.asarray([plot.plot_metrics["phase_energy_integral_m4"] for plot in members]), axis=0
        )
        e_gp = e_integrals / area
        e_g = float(np.mean(e_gp))
        d_integrals = {
            key: sum(plot.plot_metrics["pair_disagreement_integral_m4"][key] for plot in members)
            for key in ("0-1", "0-2", "0-3", "1-2", "1-3", "2-3")
        }
        d_g = float(np.mean([value / area for value in d_integrals.values()]))
        r_g = d_g / e_g if e_g > 0 else math.inf
        cv = float(np.sqrt(np.mean((e_gp - e_g) ** 2)) / e_g) if e_g > 0 else math.inf
        spectral_sum = np.zeros((16, 16), dtype=np.float64)
        spectral_count = 0
        radial = None
        for plot in members:
            for cores in plot.spectral_cores:
                packed, count = _spectral_sums(cores)
                radial = packed[0]
                spectral_sum += packed[1]
                spectral_count += count
        if spectral_count:
            spectral_mean = spectral_sum / spectral_count
            edge = float(np.sum(spectral_mean[(radial >= 7.0) & (radial <= 8.0)], dtype=np.float64))
            registered = float(np.sum(spectral_mean[(radial >= 4.0) & (radial <= 8.0)], dtype=np.float64))
            edge_fraction = edge / registered if registered > 0 else None
        else:
            edge_fraction = None
        all_forms = [form for plot in members for phase_forms in plot.forms for form in phase_forms]
        selected: list[tuple[Form, ...]] = []
        for plot in members:
            for sign in (-1, 1):
                plot_forms = [form for phase_forms in plot.forms for form in phase_forms if form.sign == sign]
                selected.extend(_select_disjoint(_families(plot_forms)))
        total_volume = sum(form.relief_volume_m3 for form in all_forms)
        persistent_volume = sum(form.relief_volume_m3 for family in selected for form in family)
        persistence_fraction = persistent_volume / total_volume if total_volume > 0 else None
        row.update(
            {
                "usable": True,
                "state": "evaluated",
                "E_gp_m2": e_gp.tolist(),
                "E_g_m2": e_g,
                "D_g_m2": d_g,
                "R_g": r_g,
                "population_energy_cv_g": cv,
                "b2_rms_m": math.sqrt(e_g),
                "spectral_core_count": spectral_count,
                "nyquist_edge_energy_fraction_available": edge_fraction is not None,
                "nyquist_edge_energy_fraction": edge_fraction,
                "eligible_form_count": len(all_forms),
                "selected_family_count": len(selected),
                "eligible_relief_volume_m3": total_volume,
                "persistent_relief_volume_m3": persistent_volume,
                "persistent_relief_volume_fraction_available": persistence_fraction is not None,
                "persistent_relief_volume_fraction": persistence_fraction,
            }
        )
        group_rows.append(row)
    used = [row for row in group_rows if row.get("usable")]
    spectral_values = [row["nyquist_edge_energy_fraction"] for row in used if row["nyquist_edge_energy_fraction"] is not None]
    persistence_values = [row["persistent_relief_volume_fraction"] for row in used if row["persistent_relief_volume_fraction"] is not None]
    checks = {
        "minimum_usable_geographic_groups": len(used) >= thresholds["minimum_usable_geographic_groups"],
        "R_g_every_used_group_max": bool(used) and max(row["R_g"] for row in used) <= thresholds["R_g_every_used_group_max"],
        "population_energy_cv_group_median_max": bool(used) and float(np.median([row["population_energy_cv_g"] for row in used])) <= thresholds["population_energy_cv_group_median_max"],
        "population_energy_cv_every_used_group_max": bool(used) and max(row["population_energy_cv_g"] for row in used) <= thresholds["population_energy_cv_every_used_group_max"],
        "b2_rms_group_median_min_m": bool(used) and float(np.median([row["b2_rms_m"] for row in used])) >= thresholds["b2_rms_group_median_min_m"],
        "nyquist_edge_energy_fraction_group_median_max": len(spectral_values) == len(used) and bool(used) and float(np.median(spectral_values)) <= thresholds["nyquist_edge_energy_fraction_group_median_max"],
        "nyquist_edge_energy_fraction_every_used_group_max": len(spectral_values) == len(used) and bool(used) and max(spectral_values) <= thresholds["nyquist_edge_energy_fraction_every_used_group_max"],
        "persistent_absolute_relief_volume_fraction_every_used_group_min": len(persistence_values) == len(used) and bool(used) and min(persistence_values) >= thresholds["persistent_absolute_relief_volume_fraction_every_used_group_min"],
    }
    passed = all(checks.values())
    for row in group_rows:
        if row.get("usable"):
            row["state"] = "passed" if passed else "failed"
    result = {
        "passed": passed,
        "state": (
            "source_decimeter_eligible_research_only_pending_hash_bound_candidate_selection_supplement"
            if passed else "target_evidence_insufficient_for_moore_decimeter_research_preview"
        ),
        "training_authorized": False,
        "usable_geographic_groups": len(used),
        "checks": checks,
        "aggregate": {
            "median_R_g": float(np.median([row["R_g"] for row in used])) if used else None,
            "max_R_g": max((row["R_g"] for row in used), default=None),
            "median_energy_cv": float(np.median([row["population_energy_cv_g"] for row in used])) if used else None,
            "max_energy_cv": max((row["population_energy_cv_g"] for row in used), default=None),
            "median_b2_rms_m": float(np.median([row["b2_rms_m"] for row in used])) if used else None,
            "median_nyquist_edge_fraction": float(np.median(spectral_values)) if len(spectral_values) == len(used) and used else None,
            "max_nyquist_edge_fraction": max(spectral_values) if len(spectral_values) == len(used) and used else None,
            "minimum_persistent_relief_volume_fraction": min(persistence_values) if len(persistence_values) == len(used) and used else None,
        },
    }
    return result, group_rows

"""Typed whole-form extraction and joint Moore morphology marks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from .huhola import (
    CLASS_CELL_M,
    CLASS_HOLLOW,
    CLASS_HUMMOCK,
    CLASS_LAWN,
    CLASS_NAMES,
    CONNECTIVITY,
    SOURCE_CELL_M,
    SOURCE_CELLS_PER_CLASS_CELL,
    ClassifiedPlot,
)


TOPOGRAPHIC_SIGN = {
    CLASS_LAWN: 0,
    CLASS_HOLLOW: -1,
    CLASS_HUMMOCK: 1,
}


@dataclass(frozen=True)
class Component:
    class_id: int
    local_id: int
    mask: np.ndarray
    whole: bool


def _components(plot: ClassifiedPlot) -> list[Component]:
    result: list[Component] = []
    for class_id in (CLASS_LAWN, CLASS_HOLLOW, CLASS_HUMMOCK):
        labels, count = label(
            plot.valid & (plot.classes == class_id),
            structure=CONNECTIVITY,
        )
        for local_id in range(1, count + 1):
            mask = labels == local_id
            result.append(
                Component(
                    class_id=class_id,
                    local_id=local_id,
                    mask=mask,
                    whole=bool(np.all(plot.interior_valid[mask])),
                )
            )
    return result


def _geometry_marks(
    mask: np.ndarray,
    x_centers_m: np.ndarray,
    y_centers_m: np.ndarray,
) -> dict[str, float]:
    rows, columns = np.nonzero(mask)
    points = np.column_stack((x_centers_m[columns], y_centers_m[rows]))
    centroid = np.mean(points, axis=0, dtype=np.float64)
    centered = points - centroid
    covariance = centered.T @ centered / points.shape[0]
    # A raster cell is an area, not a point. This term preserves a finite,
    # isotropic second moment for a one-cell form.
    covariance += np.eye(2, dtype=np.float64) * (CLASS_CELL_M**2 / 12.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minor_variance = max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
    major_variance = max(float(eigenvalues[1]), minor_variance)
    ratio = math.sqrt(major_variance / minor_variance)
    orientation_defined = ratio > 1.0 + 1e-12
    direction = eigenvectors[:, 1]
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = -direction
    orientation = (
        math.atan2(float(direction[1]), float(direction[0]))
        if orientation_defined
        else None
    )
    area = float(points.shape[0]) * CLASS_CELL_M**2
    major_axis = 2.0 * math.sqrt(area * ratio / math.pi)
    minor_axis = 2.0 * math.sqrt(area / (math.pi * ratio))
    projected = centered @ direction
    projected_std = float(np.std(projected, dtype=np.float64))
    asymmetry = (
        float(np.mean((projected / projected_std) ** 3, dtype=np.float64))
        if projected_std > 0.0
        else 0.0
    )
    return {
        "centroid_east_m": float(centroid[0]),
        "centroid_north_m": float(centroid[1]),
        "footprint_area_m2": area,
        "equivalent_diameter_m": 2.0 * math.sqrt(area / math.pi),
        "major_axis_m": major_axis,
        "minor_axis_m": minor_axis,
        "anisotropy_ratio": ratio,
        "orientation_defined": orientation_defined,
        "orientation_rad_east_ccw": orientation,
        "major_axis_asymmetry_skewness": asymmetry,
    }


def _native_mask(coarse_mask: np.ndarray, source_shape: tuple[int, int]) -> np.ndarray:
    expanded = np.repeat(
        np.repeat(coarse_mask, SOURCE_CELLS_PER_CLASS_CELL, axis=0),
        SOURCE_CELLS_PER_CLASS_CELL,
        axis=1,
    )
    result = np.zeros(source_shape, dtype=np.bool_)
    result[: expanded.shape[0], : expanded.shape[1]] = expanded
    return result


def _relief_and_shoulder_marks(
    component: Component,
    classified: ClassifiedPlot,
    source_height_m: np.ndarray,
    source_valid: np.ndarray,
) -> dict[str, Any]:
    source_height = np.asarray(source_height_m, dtype=np.float64)
    source_support = np.asarray(source_valid, dtype=np.bool_)
    native = _native_mask(component.mask, source_height.shape) & source_support
    neighbor_coarse = (
        binary_dilation(component.mask, structure=CONNECTIVITY)
        & classified.valid
        & ~component.mask
    )
    neighbor_native = _native_mask(neighbor_coarse, source_height.shape) & source_support
    if not np.any(native) or not np.any(neighbor_native):
        raise ValueError("whole form lacks measured native footprint or shoulder datum")
    datum = float(np.median(source_height[neighbor_native]))
    sign = TOPOGRAPHIC_SIGN[component.class_id]
    if sign:
        relative = sign * (source_height[native] - datum)
    else:
        relative = np.abs(source_height[native] - datum)
    relief = max(float(np.max(relative)), 0.0)

    depth = distance_transform_edt(native, sampling=(SOURCE_CELL_M, SOURCE_CELL_M))[native]
    order = np.argsort(depth, kind="stable")
    quintiles = np.array_split(order, 5)
    if any(indices.size == 0 for indices in quintiles):
        raise ValueError("whole form cannot support five shoulder strata")
    shoulder = [float(np.mean(relative[indices], dtype=np.float64)) for indices in quintiles]
    hhdh = classified.hhdh_m[component.mask]
    return {
        "topographic_sign": sign,
        "relative_relief_m": relief,
        "boundary_datum_m": datum,
        "mean_hhdh_m": float(np.mean(hhdh, dtype=np.float64)),
        "extreme_abs_hhdh_m": float(np.max(np.abs(hhdh))),
        "shoulder_profile_boundary_to_core_quintile_mean_m": shoulder,
        "shoulder_profile_sample_count": int(relative.size),
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(
        float(a["centroid_east_m"]) - float(b["centroid_east_m"]),
        float(a["centroid_north_m"]) - float(b["centroid_north_m"]),
    )


def _nearest_mark(
    form: dict[str, Any],
    others: list[dict[str, Any]],
    *,
    same_class: bool,
) -> dict[str, Any]:
    candidates = [
        other
        for other in others
        if other["form_id"] != form["form_id"]
        and ((other["class"] == form["class"]) == same_class)
    ]
    prefix = "same_class" if same_class else "cross_class"
    if not candidates:
        return {
            f"nearest_{prefix}_observed": False,
            f"nearest_{prefix}_centroid_m": None,
            f"nearest_{prefix}_edge_m": None,
            f"nearest_{prefix}_form_id": None,
        }
    nearest = min(candidates, key=lambda other: (_distance(form, other), other["form_id"]))
    centroid_distance = _distance(form, nearest)
    edge_distance = max(
        0.0,
        centroid_distance
        - 0.5
        * (
            float(form["equivalent_diameter_m"])
            + float(nearest["equivalent_diameter_m"])
        ),
    )
    return {
        f"nearest_{prefix}_observed": True,
        f"nearest_{prefix}_centroid_m": centroid_distance,
        f"nearest_{prefix}_edge_m": edge_distance,
        f"nearest_{prefix}_form_id": nearest["form_id"],
    }


def extract_whole_forms(
    *,
    group_id: str,
    plot_id: str,
    classified: ClassifiedPlot,
    source_height_m: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components = _components(classified)
    x_centers = 0.5 * (classified.x_bounds_m[:-1] + classified.x_bounds_m[1:])
    y_centers = 0.5 * (classified.y_bounds_m[:-1] + classified.y_bounds_m[1:])
    boundary_distance = distance_transform_edt(
        classified.valid,
        sampling=(CLASS_CELL_M, CLASS_CELL_M),
    )
    forms: list[dict[str, Any]] = []
    rejected = {name: 0 for name in CLASS_NAMES.values()}
    for component in components:
        if not component.whole:
            rejected[CLASS_NAMES[component.class_id]] += 1
            continue
        form_id = (
            f"{group_id}/{plot_id}/{CLASS_NAMES[component.class_id]}/"
            f"{component.local_id:04d}"
        )
        form: dict[str, Any] = {
            "form_id": form_id,
            "group_id": group_id,
            "plot_id": plot_id,
            "class_id": component.class_id,
            "class": CLASS_NAMES[component.class_id],
            "component_cell_count": int(np.count_nonzero(component.mask)),
            "whole_after_one_cell_boundary_erosion": True,
            "spacing_censor_radius_m": float(np.min(boundary_distance[component.mask])),
        }
        form.update(_geometry_marks(component.mask, x_centers, y_centers))
        form.update(
            _relief_and_shoulder_marks(
                component,
                classified,
                source_height_m,
                source_valid,
            )
        )
        adjacent_classes: set[str] = set()
        adjacent_forms = 0
        ring = binary_dilation(component.mask, structure=CONNECTIVITY) & ~component.mask
        for other in components:
            if other is component or not np.any(ring & other.mask):
                continue
            adjacent_forms += 1
            adjacent_classes.add(CLASS_NAMES[other.class_id])
        form["adjacent_component_count"] = adjacent_forms
        form["adjacent_classes"] = sorted(adjacent_classes)
        forms.append(form)

    forms.sort(key=lambda item: item["form_id"])
    for form in forms:
        form.update(_nearest_mark(form, forms, same_class=True))
        form.update(_nearest_mark(form, forms, same_class=False))
    summary = {
        "plot_id": plot_id,
        "group_id": group_id,
        "classification_cells": int(np.count_nonzero(classified.valid)),
        "boundary_eroded_cells": int(np.count_nonzero(classified.interior_valid)),
        "all_connected_components": len(components),
        "boundary_rejected_components_by_class": rejected,
        "whole_form_count": len(forms),
        "whole_form_count_by_class": {
            name: sum(form["class"] == name for form in forms)
            for name in CLASS_NAMES.values()
        },
    }
    return forms, summary

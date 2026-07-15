"""Translation-edge-corrected typed statistics over exact HuHoLa cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evidence.huhola import (
    CLASS_HOLLOW,
    CLASS_HUMMOCK,
    CLASS_LAWN,
    FILL_THRESHOLD_M,
    ClassifiedPlot,
)


# Public/preregistered class order differs from the classifier's integer order.
CLASS_IDS = np.asarray((CLASS_HOLLOW, CLASS_LAWN, CLASS_HUMMOCK), dtype=np.int8)
CLASS_NAMES = ("hollow", "lawn", "hummock")
QUANTILE_PROBABILITIES = np.asarray(
    (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0),
    dtype=np.float64,
)
MINIMUM_SUPPORT_GROUPS = 6


@dataclass(frozen=True)
class PlotEvidence:
    group_id: str
    plot_id: str
    classified: ClassifiedPlot


def canonical_orbits() -> tuple[tuple[int, int], ...]:
    return tuple(
        (major, minor)
        for major in range(1, 5)
        for minor in range(0, major + 1)
        if 0 < major * major + minor * minor <= 16
    )


def d4_offsets(major: int, minor: int) -> tuple[tuple[int, int], ...]:
    offsets = {
        (sy * y, sx * x)
        for x, y in ((major, minor), (minor, major))
        for sx in (-1, 1)
        for sy in (-1, 1)
    }
    offsets.discard((0, 0))
    return tuple(sorted(offsets))


def _paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    if abs(delta) >= length:
        return slice(0, 0), slice(0, 0)
    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


def _class_margin(plot: ClassifiedPlot) -> np.ndarray:
    """Positive distance to the exact active HuHoLa decision for the chosen class."""
    hhdh = plot.hhdh_m
    both = plot.valid & (plot.hollow_fill_m > 0.0) & (plot.hummock_fill_m > 0.0)
    result = np.full(hhdh.shape, np.nan, dtype=np.float64)
    lawn = plot.valid & (plot.classes == CLASS_LAWN)
    hollow = plot.valid & (plot.classes == CLASS_HOLLOW)
    hummock = plot.valid & (plot.classes == CLASS_HUMMOCK)
    result[lawn] = FILL_THRESHOLD_M - np.abs(hhdh[lawn])
    result[hollow & ~both] = hhdh[hollow & ~both] - FILL_THRESHOLD_M
    result[hummock & ~both] = -hhdh[hummock & ~both] - FILL_THRESHOLD_M
    result[hollow & both] = (
        plot.hollow_fill_m[hollow & both] - plot.hummock_fill_m[hollow & both]
    )
    result[hummock & both] = (
        plot.hummock_fill_m[hummock & both] - plot.hollow_fill_m[hummock & both]
    )
    if np.any(result[plot.valid] < -1e-12):
        raise AssertionError("HuHoLa class margin is inconsistent with the exact classifier")
    return np.maximum(result, 0.0)


def _quantiles(values: list[np.ndarray]) -> list[float] | None:
    nonempty = [value for value in values if value.size]
    if not nonempty:
        return None
    joined = np.concatenate(nonempty).astype(np.float64, copy=False)
    return [float(value) for value in np.quantile(joined, QUANTILE_PROBABILITIES)]


def estimate(plots: list[PlotEvidence], group_order: tuple[str, ...]) -> dict[str, Any]:
    orbits = canonical_orbits()
    group_index = {group: index for index, group in enumerate(group_order)}
    shape = (len(group_order), len(orbits), len(CLASS_IDS), len(CLASS_IDS))
    pair_counts = np.zeros(shape, dtype=np.int64)
    endpoint_denominators = np.zeros((len(group_order), len(orbits)), dtype=np.int64)
    class_counts = np.zeros((len(group_order), len(CLASS_IDS)), dtype=np.int64)
    valid_counts = np.zeros(len(group_order), dtype=np.int64)
    margins: list[list[list[np.ndarray]]] = [
        [[] for _ in CLASS_IDS] for _ in group_order
    ]
    relative_heights: list[list[list[np.ndarray]]] = [
        [[] for _ in CLASS_IDS] for _ in group_order
    ]

    for item in plots:
        group = group_index[item.group_id]
        plot = item.classified
        valid_counts[group] += int(np.count_nonzero(plot.valid))
        margin = _class_margin(plot)
        plot_center = float(np.median(plot.height_m[plot.valid]))
        for class_position, class_id in enumerate(CLASS_IDS):
            selected = plot.valid & (plot.classes == class_id)
            class_counts[group, class_position] += int(np.count_nonzero(selected))
            margins[group][class_position].append(margin[selected])
            relative_heights[group][class_position].append(
                plot.height_m[selected] - plot_center
            )

        for orbit_index, (major, minor) in enumerate(orbits):
            for dy, dx in d4_offsets(major, minor):
                source_y, target_y = _paired_slices(plot.valid.shape[0], dy)
                source_x, target_x = _paired_slices(plot.valid.shape[1], dx)
                source_valid = plot.valid[source_y, source_x]
                target_valid = plot.valid[target_y, target_x]
                paired = source_valid & target_valid
                endpoint_denominators[group, orbit_index] += int(np.count_nonzero(paired))
                if not np.any(paired):
                    continue
                source_classes = plot.classes[source_y, source_x][paired]
                target_classes = plot.classes[target_y, target_x][paired]
                for source_position, source_id in enumerate(CLASS_IDS):
                    origin = source_classes == source_id
                    for target_position, target_id in enumerate(CLASS_IDS):
                        pair_counts[group, orbit_index, source_position, target_position] += int(
                            np.count_nonzero(origin & (target_classes == target_id))
                        )

    transition_denominators = np.sum(pair_counts, axis=3, keepdims=True)
    transition_denominators = np.repeat(transition_denominators, len(CLASS_IDS), axis=3)
    transitions = np.full(shape, np.nan, dtype=np.float64)
    np.divide(pair_counts, transition_denominators, out=transitions, where=transition_denominators > 0)
    joint = np.full(shape, np.nan, dtype=np.float64)
    np.divide(
        pair_counts,
        endpoint_denominators[:, :, None, None],
        out=joint,
        where=endpoint_denominators[:, :, None, None] > 0,
    )

    # 0.5 E[(I_i(y)-I_i(x))(I_j(y)-I_j(x))], estimated on the
    # same translation-censored endpoint pairs rather than global cell fractions.
    semivariance = np.full(shape, np.nan, dtype=np.float64)
    for group in range(len(group_order)):
        for orbit in range(len(orbits)):
            denominator = endpoint_denominators[group, orbit]
            if denominator == 0:
                continue
            for i in range(len(CLASS_IDS)):
                for j in range(len(CLASS_IDS)):
                    if i == j:
                        origin_fraction = transition_denominators[group, orbit, i, 0] / denominator
                        semivariance[group, orbit, i, i] = float(
                            origin_fraction - joint[group, orbit, i, i]
                        )
                    else:
                        semivariance[group, orbit, i, j] = float(
                            -0.5
                            * (
                                joint[group, orbit, i, j]
                                + joint[group, orbit, j, i]
                            )
                        )

    class_fractions = np.full_like(class_counts, np.nan, dtype=np.float64)
    np.divide(class_counts, valid_counts[:, None], out=class_fractions, where=valid_counts[:, None] > 0)
    support_by_pair = np.count_nonzero(transition_denominators > 0, axis=0)
    retained = np.all(support_by_pair >= MINIMUM_SUPPORT_GROUPS, axis=(1, 2))
    required_indices = [orbits.index((1, 0)), orbits.index((2, 0))]

    folds = []
    for held_out, held_group in enumerate(group_order):
        training = np.arange(len(group_order)) != held_out
        fold_support = np.count_nonzero(transition_denominators[training] > 0, axis=0)
        fold_retained = np.all(fold_support >= MINIMUM_SUPPORT_GROUPS, axis=(1, 2))
        required_pass = all(bool(fold_retained[index]) for index in required_indices)
        folds.append(
            {
                "held_out_group": held_group,
                "training_group_count": int(np.count_nonzero(training)),
                "retained_orbits": [
                    list(orbits[index]) for index in np.flatnonzero(fold_retained)
                ],
                "required_axial_orbits_retained": required_pass,
            }
        )

    margin_rows = []
    for group, group_id in enumerate(group_order):
        for class_position, class_name in enumerate(CLASS_NAMES):
            margin_rows.append(
                {
                    "group_id": group_id,
                    "class": class_name,
                    "cell_count": int(class_counts[group, class_position]),
                    "hhdh_decision_margin_m_quantiles": _quantiles(margins[group][class_position]),
                    "plot_median_centered_height_m_quantiles": _quantiles(
                        relative_heights[group][class_position]
                    ),
                }
            )

    marginal_support = np.count_nonzero(class_counts > 0, axis=0)
    group_balanced_fractions = np.mean(class_fractions, axis=0, dtype=np.float64)
    group_balanced_transitions = np.full(transitions.shape[1:], np.nan, dtype=np.float64)
    group_balanced_semivariance = np.full(semivariance.shape[1:], np.nan, dtype=np.float64)
    for orbit in range(len(orbits)):
        for i in range(len(CLASS_IDS)):
            for j in range(len(CLASS_IDS)):
                if support_by_pair[orbit, i, j] < MINIMUM_SUPPORT_GROUPS:
                    continue
                finite_transition = transitions[:, orbit, i, j]
                finite_transition = finite_transition[np.isfinite(finite_transition)]
                finite_semivariance = semivariance[:, orbit, i, j]
                finite_semivariance = finite_semivariance[np.isfinite(finite_semivariance)]
                group_balanced_transitions[orbit, i, j] = float(
                    np.mean(finite_transition, dtype=np.float64)
                )
                group_balanced_semivariance[orbit, i, j] = float(
                    np.mean(finite_semivariance, dtype=np.float64)
                )

    group_balanced_quantiles = []
    for class_position, class_name in enumerate(CLASS_NAMES):
        rows = [row for row in margin_rows if row["class"] == class_name]
        if marginal_support[class_position] < MINIMUM_SUPPORT_GROUPS:
            margin_quantiles = None
            height_quantiles = None
        else:
            margin_quantiles = np.mean(
                np.asarray(
                    [row["hhdh_decision_margin_m_quantiles"] for row in rows],
                    dtype=np.float64,
                ),
                axis=0,
                dtype=np.float64,
            ).tolist()
            height_quantiles = np.mean(
                np.asarray(
                    [row["plot_median_centered_height_m_quantiles"] for row in rows],
                    dtype=np.float64,
                ),
                axis=0,
                dtype=np.float64,
            ).tolist()
        group_balanced_quantiles.append(
            {
                "class": class_name,
                "support_group_count": int(marginal_support[class_position]),
                "hhdh_decision_margin_m_quantiles": margin_quantiles,
                "plot_median_centered_height_m_quantiles": height_quantiles,
            }
        )
    failures: list[str] = []
    if not all(bool(retained[index]) for index in required_indices):
        failures.append("REQUIRED_0P5M_OR_1P0M_AXIAL_ORBIT_LACKS_SIX_GROUP_TYPED_SUPPORT")
    if np.any(marginal_support < MINIMUM_SUPPORT_GROUPS):
        failures.append("CLASS_CONDITIONAL_MARGINS_LACK_SIX_GROUP_SUPPORT")
    if not all(fold["required_axial_orbits_retained"] for fold in folds):
        failures.append("LEAVE_ONE_GROUP_OUT_REQUIRED_ORBIT_SUPPORT_FAILED")

    return {
        "orbits": orbits,
        "class_counts": class_counts,
        "valid_counts": valid_counts,
        "class_fractions": class_fractions,
        "group_balanced_class_fractions": group_balanced_fractions,
        "pair_counts": pair_counts,
        "endpoint_denominators": endpoint_denominators,
        "transition_denominators": transition_denominators,
        "transitions": transitions,
        "joint": joint,
        "semivariance": semivariance,
        "group_balanced_transitions": group_balanced_transitions,
        "group_balanced_semivariance": group_balanced_semivariance,
        "support_by_pair": support_by_pair,
        "retained": retained,
        "marginal_support": marginal_support,
        "margins": margin_rows,
        "group_balanced_quantiles": group_balanced_quantiles,
        "folds": folds,
        "failures": failures,
        "result": "pass" if not failures else "reject",
    }

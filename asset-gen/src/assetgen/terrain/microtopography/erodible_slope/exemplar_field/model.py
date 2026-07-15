"""Condition-guided 2D macro-form transfer and conservative toe transport."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FieldResult:
    c1_m: np.ndarray
    delta_m: np.ndarray
    warped_exemplar_m: np.ndarray
    ownership: np.ndarray
    opportunity: np.ndarray
    network: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    source_orientation: str
    registration_displacement_p95_m: float


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _normalize(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(value[mask], [4.0, 96.0])
    return np.clip((value - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)


def _resize(value: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    rows = np.linspace(0.0, value.shape[0] - 1.0, shape[0])
    cols = np.linspace(0.0, value.shape[1] - 1.0, shape[1])
    return ndimage.map_coordinates(value, np.meshgrid(rows, cols, indexing="ij"), order=order, mode="nearest")


def _orientations(value: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("identity", value),
        ("flip_east_west", value[:, ::-1]),
        ("flip_north_south", value[::-1, :]),
        ("rotate_180", value[::-1, ::-1]),
        ("transpose", value.T),
        ("transpose_flip_east_west", value.T[:, ::-1]),
        ("transpose_flip_north_south", value.T[::-1, :]),
        ("transpose_rotate_180", value.T[::-1, ::-1]),
    ]


def _apply_orientation(value: np.ndarray, identity: str) -> np.ndarray:
    return dict(_orientations(value))[identity]


def _register_exemplar(
    source_base: np.ndarray,
    source_macro: np.ndarray,
    source_support: np.ndarray,
    target_base: np.ndarray,
    target_mask: np.ndarray,
    pitch_m: float,
    iterations: int,
    smoothing_m: float,
    maximum_step_m: float,
) -> tuple[np.ndarray, str, float]:
    target = _normalize(ndimage.gaussian_filter(target_base, 6.0 / pitch_m), target_mask)
    target_gradient = np.hypot(*np.gradient(target, pitch_m))
    best: tuple[float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    for identity, oriented_base in _orientations(source_base):
        base = _resize(oriented_base, target.shape)
        macro = _resize(_apply_orientation(source_macro, identity), target.shape)
        support = _resize(_apply_orientation(source_support, identity).astype(np.float64), target.shape)
        normalized = _normalize(base, support > 0.7)
        source_gradient = np.hypot(*np.gradient(normalized, pitch_m))
        selected = target_mask & (support > 0.7)
        if np.count_nonzero(selected) < 100:
            continue
        score = float(np.corrcoef(target_gradient[selected], source_gradient[selected])[0, 1])
        if not np.isfinite(score):
            score = -1.0
        if best is None or score > best[0]:
            best = (score, identity, normalized, macro, support)
    if best is None:
        raise ValueError("Biala exemplar has no registered support")

    _, identity, base, macro, support = best
    row, col = np.meshgrid(
        np.arange(target.shape[0], dtype=np.float64),
        np.arange(target.shape[1], dtype=np.float64),
        indexing="ij",
    )
    dr = np.zeros_like(target)
    dc = np.zeros_like(target)
    maximum_step = maximum_step_m / pitch_m
    for _ in range(iterations):
        warped = ndimage.map_coordinates(base, [row + dr, col + dc], order=1, mode="nearest")
        gy, gx = np.gradient(warped)
        difference = target - warped
        denominator = gx * gx + gy * gy + 0.18 * difference * difference + 1.0e-4
        update_r = np.clip(difference * gy / denominator, -maximum_step, maximum_step)
        update_c = np.clip(difference * gx / denominator, -maximum_step, maximum_step)
        update_r[~target_mask] = 0.0
        update_c[~target_mask] = 0.0
        dr += ndimage.gaussian_filter(update_r, smoothing_m / pitch_m)
        dc += ndimage.gaussian_filter(update_c, smoothing_m / pitch_m)
        dr = ndimage.gaussian_filter(dr, 0.5 * smoothing_m / pitch_m)
        dc = ndimage.gaussian_filter(dc, 0.5 * smoothing_m / pitch_m)

    coordinates = [row + dr, col + dc]
    warped_macro = ndimage.map_coordinates(macro, coordinates, order=1, mode="nearest")
    warped_support = ndimage.map_coordinates(support, coordinates, order=1, mode="nearest")
    warped_macro *= _smoothstep((warped_support - 0.45) / 0.45)
    displacement = np.hypot(dr, dc) * pitch_m
    return warped_macro, identity, float(np.percentile(displacement[target_mask], 95.0))


_NEIGHBORS = (
    (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
)


def _shift_zero(value: np.ndarray, dr: int, dc: int) -> np.ndarray:
    result = np.zeros_like(value)
    source_r = slice(max(0, -dr), min(value.shape[0], value.shape[0] - dr))
    source_c = slice(max(0, -dc), min(value.shape[1], value.shape[1] - dc))
    target_r = slice(max(0, dr), min(value.shape[0], value.shape[0] + dr))
    target_c = slice(max(0, dc), min(value.shape[1], value.shape[1] + dc))
    result[target_r, target_c] = value[source_r, source_c]
    return result


def _transport_to_toes(
    eroded_depth: np.ndarray,
    height: np.ndarray,
    eligible: np.ndarray,
    pitch_m: float,
    steps: int,
) -> np.ndarray:
    smooth_height = ndimage.gaussian_filter(height, 2.5 / pitch_m)
    gy, gx = np.gradient(smooth_height, pitch_m)
    slope = np.hypot(gx, gy)
    mobile = eroded_depth.copy()
    deposited = np.zeros_like(eroded_depth)
    for step in range(steps):
        weights: list[np.ndarray] = []
        total = np.zeros_like(height)
        for dr, dc, distance in _NEIGHBORS:
            neighbor = _shift_zero(smooth_height, -dr, -dc)
            neighbor_valid = _shift_zero(eligible.astype(np.float64), -dr, -dc) > 0.5
            drop = np.maximum((smooth_height - neighbor) / (distance * pitch_m), 0.0)
            weight = np.where(eligible & neighbor_valid, drop ** 1.35, 0.0)
            weights.append(weight)
            total += weight
        retained = mobile.copy()
        routed = np.zeros_like(mobile)
        movable = total > 1.0e-10
        retained[movable] = 0.0
        for (dr, dc, _), weight in zip(_NEIGHBORS, weights):
            share = np.divide(mobile * weight, total, out=np.zeros_like(mobile), where=movable)
            routed += _shift_zero(share, dr, dc)
        routed += retained
        settling = 0.015 + 0.16 * (1.0 - _smoothstep((slope - 0.05) / 0.25))
        if step < 4:
            settling *= 0.15
        settle = routed * settling * eligible
        deposited += settle
        mobile = routed - settle
    deposited += mobile * eligible
    deposited = ndimage.gaussian_filter(deposited, 1.25 / pitch_m)
    deposited *= eligible
    return deposited


def synthesize(
    c0_m: np.ndarray,
    hard: np.ndarray,
    network_reference: np.ndarray,
    source_base_m: np.ndarray,
    source_macro_m: np.ndarray,
    source_support: np.ndarray,
    config: dict,
    pitch_m: float,
) -> FieldResult:
    stride = int(round(float(config["model_pitch_m"]) / pitch_m))
    if stride < 1 or abs(stride * pitch_m - float(config["model_pitch_m"])) > 1.0e-8:
        raise ValueError("model pitch is not integral on C0")
    coarse_c0 = c0_m[::stride, ::stride].astype(np.float64)
    coarse_hard = hard[::stride, ::stride]
    model_pitch = stride * pitch_m
    network = _resize(network_reference, coarse_c0.shape)
    eligible = ~coarse_hard
    distance = ndimage.distance_transform_edt(eligible) * model_pitch
    taper = _smoothstep(distance / float(config["hard_transition_m"]))

    warped, orientation, displacement = _register_exemplar(
        source_base_m,
        source_macro_m,
        source_support,
        coarse_c0,
        eligible & (distance >= float(config["hard_transition_m"])),
        model_pitch,
        int(config["registration_iterations"]),
        float(config["registration_smoothing_m"]),
        float(config["registration_maximum_step_m"]),
    )
    warped = ndimage.gaussian_filter(warped, float(config["exemplar_smoothing_m"]) / model_pitch)
    source_scale = np.percentile(np.abs(warped[eligible]), 95.0)
    warped = warped / max(source_scale, 1.0e-8)

    structural = ndimage.gaussian_filter(coarse_c0, 3.0 / model_pitch)
    gy, gx = np.gradient(structural, model_pitch)
    slope = np.hypot(gx, gy)
    slope_window = _smoothstep((slope - float(config["minimum_slope"])) / float(config["slope_ramp"]))
    steep_limit = 1.0 - _smoothstep((slope - float(config["maximum_slope"])) / float(config["steep_ramp"]))
    network_context = ndimage.gaussian_filter(network, float(config["network_context_m"]) / model_pitch)
    opportunity = taper * slope_window * steep_limit * (0.50 + 0.50 * np.clip(network_context, 0.0, 1.0))
    opportunity[~eligible] = 0.0

    source_field = float(config["macro_p95_m"]) * warped * opportunity
    # Preserve the exemplar's paired 2D organization. Only the added target
    # incision is routed into new toes; transporting this entire field erases
    # the connected shoulders, headwalls, and chutes that justify its use.
    source_broad = ndimage.gaussian_filter(source_field, 8.0 / model_pitch)
    source_field = 0.84 * source_field + 0.16 * source_broad
    source_erosion = np.maximum(-source_field, 0.0)
    source_deposition = np.maximum(source_field, 0.0)
    source_eroded_volume = float(np.sum(source_erosion) * model_pitch * model_pitch)
    source_deposited_volume = float(np.sum(source_deposition) * model_pitch * model_pitch)
    source_deposition *= source_eroded_volume / max(source_deposited_volume, 1.0e-12)
    laplacian = ndimage.laplace(ndimage.gaussian_filter(structural, 2.0 / model_pitch)) / (model_pitch * model_pitch)
    valley = np.maximum(laplacian, 0.0)
    valley_scale = np.percentile(valley[eligible], 98.0)
    valley = np.clip(valley / max(valley_scale, 1.0e-8), 0.0, 1.0)
    incision = float(config["incision_depth_m"]) * valley * (0.35 + 0.65 * network_context) * opportunity
    seep = ndimage.gaussian_filter(valley * (1.0 - _smoothstep((slope - 0.08) / 0.22)), 5.0 / model_pitch)
    seep *= float(config["seep_depth_m"]) * opportunity

    event_erosion = incision + seep
    event_erosion = ndimage.gaussian_filter(event_erosion, float(config["erosion_smoothing_m"]) / model_pitch)
    event_erosion *= taper * eligible
    maximum_depth = float(config["maximum_erosion_m"])
    event_erosion = np.minimum(event_erosion, maximum_depth)

    transported_toe = _transport_to_toes(
        event_erosion,
        structural,
        eligible & (taper > 0.0),
        model_pitch,
        int(config["transport_steps"]),
    )
    shoulder = source_deposition * taper
    event_volume = float(np.sum(event_erosion) * model_pitch * model_pitch)
    toe_sum = float(np.sum(transported_toe) * model_pitch * model_pitch)
    transported_toe *= event_volume / max(toe_sum, 1.0e-12)
    erosion = source_erosion + event_erosion
    deposition = shoulder + transported_toe
    delta_coarse = deposition - erosion

    delta = _resize(delta_coarse, c0_m.shape, order=3)
    erosion_fine = _resize(erosion, c0_m.shape)
    shoulder_fine = _resize(shoulder, c0_m.shape)
    toe_fine = _resize(transported_toe, c0_m.shape)
    incision_fine = _resize(incision, c0_m.shape)
    seep_fine = _resize(seep, c0_m.shape)
    warped_fine = _resize(warped, c0_m.shape)
    opportunity_fine = _resize(opportunity, c0_m.shape)
    network_fine = _resize(network, c0_m.shape)
    fine_distance = ndimage.distance_transform_edt(~hard) * pitch_m
    fine_taper = _smoothstep(fine_distance / float(config["hard_transition_m"]))
    delta *= fine_taper
    delta[hard] = 0.0
    negative = np.maximum(-delta, 0.0)
    positive = np.maximum(delta, 0.0)
    negative_volume = float(np.sum(negative) * pitch_m * pitch_m)
    positive_volume = float(np.sum(positive) * pitch_m * pitch_m)
    positive *= negative_volume / max(positive_volume, 1.0e-12)
    delta = positive - negative
    delta[hard] = 0.0

    ownership = np.zeros(c0_m.shape, dtype=np.uint8)
    active = np.abs(delta) >= float(config["ownership_minimum_relief_m"])
    ownership[active & (delta > 0.0) & (shoulder_fine >= toe_fine)] = 1
    ownership[active & (delta < 0.0) & (incision_fine < 0.45 * erosion_fine)] = 2
    ownership[active & (delta < 0.0) & (incision_fine >= 0.45 * erosion_fine)] = 3
    ownership[active & (delta < 0.0) & (seep_fine >= 0.25 * erosion_fine)] = 4
    ownership[active & (delta > 0.0) & (toe_fine > shoulder_fine)] = 5
    ownership[hard] = 0
    return FieldResult(
        c1_m=(c0_m.astype(np.float64) + delta).astype(np.float32),
        delta_m=delta.astype(np.float32),
        warped_exemplar_m=warped_fine.astype(np.float32),
        ownership=ownership,
        opportunity=opportunity_fine.astype(np.float32),
        network=network_fine.astype(np.float32),
        erosion_m=negative.astype(np.float32),
        deposition_m=positive.astype(np.float32),
        source_orientation=orientation,
        registration_displacement_p95_m=displacement,
    )

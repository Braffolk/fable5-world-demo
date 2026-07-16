"""Biala-conditioned sparse amplification on irregular compact supports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


TYPE_NAMES = ("shoulder", "scarp_safe_side", "incision", "toe")
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class AtomBank:
    low: np.ndarray
    high: np.ndarray
    kind: np.ndarray
    source_row: np.ndarray
    source_col: np.ndarray
    transform: np.ndarray


@dataclass(frozen=True)
class SparseResult:
    c1_m: np.ndarray
    delta_m: np.ndarray
    ownership: np.ndarray
    source_attribution: np.ndarray
    site_attribution: np.ndarray
    condition_strength: np.ndarray
    network: np.ndarray
    placements: list[dict[str, float | int | str]]
    atom_counts: dict[str, int]


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _normalize(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(value[valid], [5.0, 95.0])
    return np.clip((value - lo) / max(float(hi - lo), 1.0e-9), 0.0, 1.0)


def _block_mean(value: np.ndarray, factor: int) -> np.ndarray:
    rows = value.shape[0] // factor * factor
    cols = value.shape[1] // factor * factor
    return value[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def _compact_mask(size: int) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float64) - 0.5 * (size - 1)) / max(0.5 * (size - 1), 1.0)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    one_minus = np.maximum(0.0, 1.0 - radius)
    return one_minus**4 * (4.0 * radius + 1.0)


def _transform(value: np.ndarray, identity: int) -> np.ndarray:
    rotated = np.rot90(value, identity & 3)
    return rotated[:, ::-1] if identity & 4 else rotated


def _source_kind(low: np.ndarray, macro: np.ndarray, pitch_m: float) -> int:
    smooth = ndimage.gaussian_filter(low, 1.25 / pitch_m)
    gy, gx = np.gradient(smooth, pitch_m)
    slope = float(np.median(np.hypot(gx, gy)))
    lap = ndimage.laplace(smooth) / (pitch_m * pitch_m)
    center = tuple(slice(low.shape[axis] // 4, 3 * low.shape[axis] // 4) for axis in (0, 1))
    curvature = float(np.mean(lap[center]))
    macro_center = float(np.mean(macro[center]) - np.mean(macro))
    if slope >= 0.33:
        return 1
    if curvature >= 0.025 or macro_center <= -0.08:
        return 2
    if curvature <= -0.018 or macro_center >= 0.08:
        return 0
    return 3


def build_atom_bank(
    source_reconstruction_m: np.ndarray,
    source_macro_m: np.ndarray,
    source_support: np.ndarray,
    config: dict,
) -> AtomBank:
    """Build paired 1 m/0.5 m atoms; 0.25 m samples are interpolation only."""
    source_pitch = float(config["source_pitch_m"])
    low_pitch = float(config["low_pitch_m"])
    factor = int(round(low_pitch / source_pitch))
    if factor != 2:
        raise ValueError("the qualified Biala pairing must remain 1.0 m low / 0.5 m high")
    high_cells = int(round(float(config["patch_size_m"]) / source_pitch))
    low_cells = high_cells // factor
    stride = int(round(float(config["source_stride_m"]) / source_pitch))
    if high_cells % 2 or low_cells % 2 or stride < 1:
        raise ValueError("invalid coupled atom geometry")

    low_source = _block_mean(source_reconstruction_m, factor)
    low_macro = _block_mean(source_macro_m, factor)
    support_low = _block_mean(source_support.astype(np.float64), factor) == 1.0
    low_mask = _compact_mask(low_cells)
    maximum_per_type = int(config["maximum_atoms_per_type"])
    candidates: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    half_high = high_cells // 2
    for row in range(half_high, source_reconstruction_m.shape[0] - half_high + 1, stride):
        for col in range(half_high, source_reconstruction_m.shape[1] - half_high + 1, stride):
            high = source_reconstruction_m[row - half_high : row + half_high, col - half_high : col + half_high]
            support = source_support[row - half_high : row + half_high, col - half_high : col + half_high]
            if high.shape != (high_cells, high_cells) or float(np.mean(support)) < float(config["minimum_patch_support"]):
                continue
            low_row, low_col = row // factor, col // factor
            half_low = low_cells // 2
            low = low_source[low_row - half_low : low_row + half_low, low_col - half_low : low_col + half_low]
            macro = low_macro[low_row - half_low : low_row + half_low, low_col - half_low : low_col + half_low]
            supported = support_low[low_row - half_low : low_row + half_low, low_col - half_low : low_col + half_low]
            if low.shape != (low_cells, low_cells) or not np.all(supported):
                continue
            low_centered = low - float(np.mean(low))
            low_weighted = low_centered * low_mask
            low_norm = float(np.linalg.norm(low_weighted))
            if low_norm < 0.04:
                continue
            high_parent = np.repeat(np.repeat(_block_mean(high, factor), factor, axis=0), factor, axis=1)
            innovation = high - high_parent
            kind = _source_kind(low, macro, low_pitch)
            for identity in range(8):
                candidates.append((kind, row, col, identity, _transform(low_weighted / low_norm, identity), _transform(innovation / low_norm, identity)))

    selected: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for kind in range(len(TYPE_NAMES)):
        typed = [item for item in candidates if item[0] == kind]
        typed.sort(key=lambda item: ((item[1] * 73856093) ^ (item[2] * 19349663) ^ (item[3] * 83492791)) & 0xFFFFFFFF)
        if not typed:
            raise ValueError(f"Biala source produced no {TYPE_NAMES[kind]} atoms")
        selected.extend(typed[:maximum_per_type])
    return AtomBank(
        low=np.stack([item[4] for item in selected]).astype(np.float32),
        high=np.stack([item[5] for item in selected]).astype(np.float32),
        kind=np.asarray([item[0] for item in selected], dtype=np.uint8),
        source_row=np.asarray([item[1] for item in selected], dtype=np.int16),
        source_col=np.asarray([item[2] for item in selected], dtype=np.int16),
        transform=np.asarray([item[3] for item in selected], dtype=np.uint8),
    )


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _sites(shape: tuple[int, int], pitch_m: float, bbox: tuple[float, float, float, float], config: dict) -> list[tuple[int, int, int]]:
    cell = int(round(float(config["candidate_cell_m"]) / pitch_m))
    minimum = float(config["minimum_site_distance_m"]) / pitch_m
    seed = int(config["site_seed"])
    row0 = int(round(-bbox[3] / pitch_m))
    col0 = int(round(bbox[0] / pitch_m))
    candidates: list[tuple[int, int, int]] = []
    for gy in range(-1, shape[0] // cell + 2):
        for gx in range(-1, shape[1] // cell + 2):
            key = _mix64((gx + col0 // cell) * 0xD6E8FEB86659FD93 ^ (gy + row0 // cell) * 0xA5A3564E27F8862F ^ seed)
            row = gy * cell + int(key % cell)
            col = gx * cell + int((key >> 17) % cell)
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                candidates.append((row, col, key))
    candidates.sort(key=lambda item: item[2])
    selected: list[tuple[int, int, int]] = []
    distance2 = minimum * minimum
    for candidate in candidates:
        if all((candidate[0] - row) ** 2 + (candidate[1] - col) ** 2 >= distance2 for row, col, _ in selected):
            selected.append(candidate)
    selected.sort()
    return selected


def _target_conditions(c0: np.ndarray, network_reference: np.ndarray, pitch_m: float, hard: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(c0.astype(np.float64), 2.0 / pitch_m)
    gy, gx = np.gradient(smooth, pitch_m)
    slope = np.hypot(gx, gy)
    lap = ndimage.gaussian_laplace(smooth, 2.0 / pitch_m)
    valid = ~hard
    network = ndimage.zoom(network_reference.astype(np.float64), (c0.shape[0] / network_reference.shape[0], c0.shape[1] / network_reference.shape[1]), order=1)
    network = network[: c0.shape[0], : c0.shape[1]]
    network = _normalize(network, valid)
    positive = _normalize(np.maximum(lap, 0.0), valid)
    negative = _normalize(np.maximum(-lap, 0.0), valid)
    slope_n = _normalize(slope, valid)
    shoulder = (0.25 + 0.75 * network) * negative * (1.0 - 0.45 * slope_n)
    side = (0.20 + 0.80 * network) * _smoothstep((slope_n - 0.25) / 0.55)
    incision = (0.15 + 0.85 * network) * positive * (0.35 + 0.65 * slope_n)
    toe = (0.20 + 0.80 * network) * positive * (1.0 - 0.70 * slope_n)
    conditions = np.stack([shoulder, side, incision, toe])
    distance = ndimage.distance_transform_edt(valid) * pitch_m
    taper = _smoothstep(distance / 4.0)
    conditions *= taper[None]
    conditions[:, hard] = 0.0
    strength = np.max(conditions, axis=0)
    return conditions, strength, network


def _project_one_metre_closure(delta: np.ndarray, hard: np.ndarray, factor: int) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    rows = result.shape[0] // factor * factor
    cols = result.shape[1] // factor * factor
    for row in range(0, rows, factor):
        for col in range(0, cols, factor):
            region = np.s_[row : row + factor, col : col + factor]
            active = ~hard[region]
            if np.any(active):
                block = result[region]
                block[active] -= float(np.mean(block[active]))
                block[~active] = 0.0
    result[hard] = 0.0
    return result


def synthesize(
    c0_m: np.ndarray,
    hard: np.ndarray,
    network_reference: np.ndarray,
    source_reconstruction_m: np.ndarray,
    source_macro_m: np.ndarray,
    source_support: np.ndarray,
    bbox_en: tuple[float, float, float, float],
    config: dict,
    pitch_m: float,
) -> SparseResult:
    bank = build_atom_bank(source_reconstruction_m, source_macro_m, source_support, config)
    conditions, condition_strength, network = _target_conditions(c0_m, network_reference, pitch_m, hard)
    low_factor = int(round(float(config["low_pitch_m"]) / pitch_m))
    low_cells = bank.low.shape[1]
    high_cells = low_cells * low_factor
    half = high_cells // 2
    compact = _compact_mask(high_cells)
    sites = _sites(c0_m.shape, pitch_m, bbox_en, config)
    numerator = np.zeros_like(c0_m, dtype=np.float64)
    denominator = np.zeros_like(c0_m, dtype=np.float64)
    dominant = np.zeros_like(c0_m, dtype=np.float64)
    ownership = np.zeros_like(c0_m, dtype=np.uint8)
    source_attribution = np.full(c0_m.shape, -1, dtype=np.int32)
    site_attribution = np.full(c0_m.shape, -1, dtype=np.int32)
    low_mask = _compact_mask(low_cells)
    placements: list[dict[str, float | int | str]] = []

    for site_index, (row, col, _priority) in enumerate(sites):
        kind = int(np.argmax(conditions[:, row, col]))
        site_strength = float(condition_strength[row, col])
        if site_strength < float(config["minimum_condition_strength"]):
            continue
        offsets = (np.arange(low_cells, dtype=np.float64) - 0.5 * (low_cells - 1)) * low_factor
        rr, cc = np.meshgrid(row + offsets, col + offsets, indexing="ij")
        query = ndimage.map_coordinates(c0_m.astype(np.float64), [rr, cc], order=1, mode="nearest")
        query = (query - float(np.mean(query))) * low_mask
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1.0e-8:
            continue
        indices = np.flatnonzero(bank.kind == kind)
        correlations = bank.low[indices].reshape(len(indices), -1) @ query.ravel()
        selected = int(indices[int(np.argmax(np.abs(correlations)))])
        coefficient = float(correlations[np.argmax(np.abs(correlations))])
        high = ndimage.zoom(bank.high[selected].astype(np.float64), low_factor / 2.0, order=3)
        high = high[:high_cells, :high_cells]
        high *= coefficient * float(config["amplification_gain"])
        p99 = float(np.percentile(np.abs(high), 99.0))
        if p99 > float(config["maximum_atom_p99_m"]):
            high *= float(config["maximum_atom_p99_m"]) / p99

        row_start, row_end = max(0, row - half), min(c0_m.shape[0], row + half)
        col_start, col_end = max(0, col - half), min(c0_m.shape[1], col + half)
        if row_start >= row_end or col_start >= col_end:
            continue
        source_r0 = row_start - (row - half)
        source_c0 = col_start - (col - half)
        source_r1 = source_r0 + row_end - row_start
        source_c1 = source_c0 + col_end - col_start
        region = np.s_[row_start:row_end, col_start:col_end]
        weight = compact[source_r0:source_r1, source_c0:source_c1] * conditions[kind][region]
        patch = high[source_r0:source_r1, source_c0:source_c1]
        numerator[region] += weight * patch
        denominator[region] += weight
        stronger = weight > dominant[region]
        dominant_region = dominant[region]
        dominant_region[stronger] = weight[stronger]
        ownership_region = ownership[region]
        ownership_region[stronger] = kind + 1
        source_region = source_attribution[region]
        source_region[stronger] = int(bank.source_row[selected]) * 1000 + int(bank.source_col[selected])
        site_region = site_attribution[region]
        site_region[stronger] = site_index
        placements.append({
            "site_index": site_index,
            "row": row,
            "col": col,
            "kind": TYPE_NAMES[kind],
            "source_row": int(bank.source_row[selected]),
            "source_col": int(bank.source_col[selected]),
            "transform": int(bank.transform[selected]),
            "omp_coefficient": coefficient,
            "condition_strength": site_strength,
        })

    delta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1.0e-12)
    distance = ndimage.distance_transform_edt(~hard) * pitch_m
    delta *= _smoothstep(distance / float(config["transition_m"]))
    delta = _project_one_metre_closure(delta, hard, low_factor)
    delta = np.clip(delta, -float(config["maximum_relief_m"]), float(config["maximum_relief_m"]))
    delta = _project_one_metre_closure(delta, hard, low_factor)
    ownership[(np.abs(delta) < float(config["ownership_minimum_m"])) | hard] = 0
    source_attribution[ownership == 0] = -1
    site_attribution[ownership == 0] = -1
    return SparseResult(
        c1_m=(c0_m.astype(np.float64) + delta).astype(np.float32),
        delta_m=delta.astype(np.float32),
        ownership=ownership,
        source_attribution=source_attribution,
        site_attribution=site_attribution,
        condition_strength=condition_strength.astype(np.float32),
        network=network.astype(np.float32),
        placements=placements,
        atom_counts={name: int(np.count_nonzero(bank.kind == index)) for index, name in enumerate(TYPE_NAMES)},
    )

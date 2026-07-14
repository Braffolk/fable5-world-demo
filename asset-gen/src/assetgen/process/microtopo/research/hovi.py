"""Frozen semantic-gated disjoint-view HY_SPRUCE4 weak-surface reconstruction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ....evidence.hovi.full_scan.candidate import (
    SpatialObservationReader,
    load_verified_spatial_manifest,
)
from .bands import b1_confidence, capacity
from .bundle import FrozenBundle
from .contracts import ResearchSurfaceEvidence, SemanticResult
from .features import geometry_features
from .surfaces import _partition


@dataclass(frozen=True)
class _Mode:
    height_m: float
    points: int
    representative: int
    score: float


def _scanner_groups(scans) -> tuple[dict[int, int], dict]:
    origins = np.asarray([scan.translation_xyz[:2] for scan in scans], dtype=np.float64)
    center = np.median(origins, axis=0)
    delta = origins - center
    order = sorted(
        range(len(scans)),
        key=lambda ordinal: (
            math.atan2(delta[ordinal, 1], delta[ordinal, 0]),
            float(np.linalg.norm(delta[ordinal])),
            ordinal,
        ),
    )
    groups = {ordinal: index % 2 for index, ordinal in enumerate(order)}
    return groups, {
        "median_xy": center.tolist(),
        "sorted_scan_ordinals": order,
        "group_a": [scan for scan in order if groups[scan] == 0],
        "group_b": [scan for scan in order if groups[scan] == 1],
    }


def _huber_plane(points: np.ndarray, center: tuple[float, float]) -> np.ndarray | None:
    if len(points) < 6:
        return None
    design = np.column_stack(
        (points[:, 0] - center[0], points[:, 1] - center[1], np.ones(len(points)))
    )
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    for _ in range(6):
        residual = points[:, 2] - design @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if scale <= 1e-6:
            break
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], points[:, 2] * root, rcond=None)
    return coefficients


def _reference_planes(
    accepted_by_scan: list[np.ndarray], origin_xy: tuple[float, float], side_m: int = 8
) -> np.ndarray:
    planes = np.full((side_m, side_m, 3), np.nan, dtype=np.float64)
    all_points = np.concatenate(accepted_by_scan) if accepted_by_scan else np.empty((0, 3))
    if not len(all_points):
        return planes
    ij = np.floor(all_points[:, :2] - np.asarray(origin_xy)).astype(np.int64)
    for y in range(side_m):
        for x in range(side_m):
            selected = (ij[:, 0] == x) & (ij[:, 1] == y)
            plane = _huber_plane(
                all_points[selected], (origin_xy[0] + x + 0.5, origin_xy[1] + y + 0.5)
            )
            if plane is not None:
                planes[y, x] = plane
    return planes


def _scan_modes(
    xyz: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    planes: np.ndarray,
    origin_xy: tuple[float, float],
) -> tuple[list[list[_Mode]], np.ndarray]:
    side = planes.shape[0] * 16
    modes: list[list[_Mode]] = [[] for _ in range(side * side)]
    xy_index = np.floor((xyz[:, :2] - np.asarray(origin_xy)) / 0.0625).astype(np.int64)
    inside = np.all((xy_index >= 0) & (xy_index < side), axis=1)
    cell1 = np.floor(xyz[:, :2] - np.asarray(origin_xy)).astype(np.int64)
    plane_valid = inside & np.isfinite(planes[cell1[:, 1].clip(0, side // 16 - 1), cell1[:, 0].clip(0, side // 16 - 1), 0])
    selected_indices = np.flatnonzero(plane_valid)
    if not len(selected_indices):
        return modes, np.zeros((side, side), dtype=np.uint32)
    local = xyz[selected_indices]
    cells = xy_index[selected_indices]
    cell1_local = cell1[selected_indices]
    coefficient = planes[cell1_local[:, 1], cell1_local[:, 0]]
    centers = np.asarray(origin_xy) + cell1_local + 0.5
    residual = local[:, 2] - (
        coefficient[:, 0] * (local[:, 0] - centers[:, 0])
        + coefficient[:, 1] * (local[:, 1] - centers[:, 1])
        + coefficient[:, 2]
    )
    linear = cells[:, 1] * side + cells[:, 0]
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    starts = np.flatnonzero(np.r_[True, sorted_linear[1:] != sorted_linear[:-1]])
    support = np.zeros((side, side), dtype=np.uint32)
    for begin, end in zip(starts, np.r_[starts[1:], len(order)]):
        members = selected_indices[order[begin:end]]
        local_order = order[begin:end]
        cell = int(sorted_linear[begin])
        cy, cx = divmod(cell, side)
        support[cy, cx] = len(members)
        bins = np.floor(residual[local_order] / 0.01).astype(np.int64)
        first, last = int(bins.min()), int(bins.max())
        histogram = np.bincount(bins - first, minlength=last - first + 1).astype(np.float64)
        smooth = np.convolve(histogram, (0.25, 0.5, 0.25), mode="same")
        peaks = np.flatnonzero(
            (smooth >= np.r_[[-np.inf], smooth[:-1]])
            & (smooth > np.r_[smooth[1:], [-np.inf]])
        )
        if not len(peaks):
            continue
        boundaries = np.r_[0, ((peaks[:-1] + peaks[1:]) // 2) + 1, len(histogram)]
        cell_origin = np.asarray(origin_xy) + np.asarray((cx, cy)) * 0.0625
        for peak_index, peak in enumerate(peaks):
            in_basin = (bins - first >= boundaries[peak_index]) & (
                bins - first < boundaries[peak_index + 1]
            )
            basin_members = members[in_basin]
            if len(basin_members) < 6:
                continue
            point_xy = xyz[basin_members, :2]
            quadrants = ((point_xy[:, 0] >= cell_origin[0] + 0.03125).astype(np.uint8)) | (
                (point_xy[:, 1] >= cell_origin[1] + 0.03125).astype(np.uint8) << 1
            )
            if len(np.unique(quadrants)) < 3:
                continue
            z = xyz[basin_members, 2]
            median = float(np.median(z))
            representative = basin_members[np.argmin(np.abs(z - median))]
            modes[cell].append(
                _Mode(median, len(basin_members), int(source_indices[representative]), float(scores[representative]))
            )
    return modes, support


def _consensus(
    modes_by_scan: list[list[list[_Mode]]],
    groups: dict[int, int],
    side: int,
) -> dict[str, np.ndarray]:
    height = np.full((side, side), np.nan, dtype=np.float64)
    height_a = np.full_like(height, np.nan)
    height_b = np.full_like(height, np.nan)
    n_a = np.zeros((side, side), dtype=np.uint8)
    n_b = np.zeros((side, side), dtype=np.uint8)
    source_index = np.zeros((side, side), dtype=np.uint32)
    unique = np.zeros((side, side), dtype=np.bool_)
    forbidden_competitor = np.zeros((side, side), dtype=np.bool_)
    provisional_probability = np.zeros((side, side), dtype=np.float32)
    for cell in range(side * side):
        entries = [
            (mode.height_m, scan, mode)
            for scan, scan_modes in enumerate(modes_by_scan)
            for mode in scan_modes[cell]
        ]
        if not entries:
            continue
        entries.sort(key=lambda row: row[0])
        clusters: list[list[tuple[float, int, _Mode]]] = []
        for entry in entries:
            if clusters and entry[0] - clusters[-1][0][0] <= 0.03:
                clusters[-1].append(entry)
            else:
                clusters.append([entry])
        qualifying = []
        for cluster in clusters:
            scans_a = {scan for _, scan, _ in cluster if groups[scan] == 0}
            scans_b = {scan for _, scan, _ in cluster if groups[scan] == 1}
            if len(scans_a) >= 2 and len(scans_b) >= 2:
                qualifying.append((cluster, scans_a, scans_b))
        if len(qualifying) != 1:
            continue
        candidate, scans_a, scans_b = qualifying[0]
        support = len(scans_a) + len(scans_b)
        competitors = [len({scan for _, scan, _ in cluster}) for cluster in clusters if cluster is not candidate]
        if competitors and max(competitors) * 2 >= support:
            forbidden_competitor.flat[cell] = True
            continue
        values_a = [value for value, scan, _ in candidate if groups[scan] == 0]
        values_b = [value for value, scan, _ in candidate if groups[scan] == 1]
        y, x = divmod(cell, side)
        height_a[y, x] = np.median(values_a)
        height_b[y, x] = np.median(values_b)
        height[y, x] = np.median([value for value, _, _ in candidate])
        n_a[y, x], n_b[y, x] = len(scans_a), len(scans_b)
        chosen = min(candidate, key=lambda row: abs(row[0] - height[y, x]))[2]
        source_index[y, x] = chosen.representative
        provisional_probability[y, x] = chosen.score
        unique[y, x] = True
    return {
        "height": height,
        "height_a": height_a,
        "height_b": height_b,
        "n_a": n_a,
        "n_b": n_b,
        "source_index": source_index,
        "unique": unique,
        "forbidden_competitor": forbidden_competitor,
        "provisional_probability": provisional_probability,
    }


def _context(
    manifest,
    reader: SpatialObservationReader,
    block_x: int,
    block_y: int,
    model,
    threshold: float,
    envelope: np.ndarray,
    groups: dict[int, int],
    *,
    log=print,
) -> dict[str, np.ndarray]:
    shard_keys = [(block_x * 2 + dx, block_y * 2 + dy) for dy in range(2) for dx in range(2)]
    origin = (-6.775 + block_x * 8.0, -6.1 + block_y * 8.0)
    accepted_by_scan: list[np.ndarray] = []
    source_by_scan: list[np.ndarray] = []
    score_by_scan: list[np.ndarray] = []
    raw_group_support = [np.zeros((128, 128), dtype=np.uint32), np.zeros((128, 128), dtype=np.uint32)]
    for scan in range(16):
        batches = []
        sources = []
        for shard_x, shard_y in shard_keys:
            shard = next((value for value in manifest.shards_at(shard_x, shard_y) if value.scan == scan), None)
            if shard is None:
                continue
            for batch in reader.iter_shard_artifact(shard, batch_records=1 << 18):
                batches.append(batch.file_xyz)
                sources.append(batch.source_ordinal)
        if not batches:
            accepted_by_scan.append(np.empty((0, 3)))
            source_by_scan.append(np.empty(0, dtype=np.uint32))
            score_by_scan.append(np.empty(0))
            continue
        xyz = np.concatenate(batches)
        source = np.concatenate(sources)
        fine = np.floor((xyz[:, :2] - np.asarray(origin)) / 0.0625).astype(np.int64)
        inside = np.all((fine >= 0) & (fine < 128), axis=1)
        np.add.at(raw_group_support[groups[scan]], (fine[inside, 1], fine[inside, 0]), 1)
        descriptor = geometry_features(xyz)
        in_domain = descriptor.valid & np.all(
            (descriptor.values >= envelope[:, 0]) & (descriptor.values <= envelope[:, 1]), axis=1
        )
        score = np.full(len(xyz), np.nan)
        score[in_domain] = model.predict_proba(descriptor.values[in_domain])[:, 1]
        accepted = in_domain & (score >= threshold)
        accepted_by_scan.append(xyz[accepted])
        source_by_scan.append(source[accepted])
        score_by_scan.append(score[accepted])
        log(f"Hovi context {block_x},{block_y} scan {scan:02d}: {accepted.sum():,}/{len(xyz):,}")
    planes = _reference_planes(accepted_by_scan, origin)
    modes_by_scan = []
    for xyz, source, scores in zip(accepted_by_scan, source_by_scan, score_by_scan):
        modes, _ = _scan_modes(xyz, scores, source, planes, origin)
        modes_by_scan.append(modes)
    result = _consensus(modes_by_scan, groups, 128)
    result["joint_direct"] = (raw_group_support[0] > 0) & (raw_group_support[1] > 0)
    result["support_a"] = raw_group_support[0]
    result["support_b"] = raw_group_support[1]
    return result


def reconstruct_hovi_surface(
    bundle: FrozenBundle,
    semantic: SemanticResult,
    *,
    log=print,
) -> tuple[ResearchSurfaceEvidence, dict]:
    if not semantic.audit_passed or semantic.threshold is None:
        raise ValueError("Hovi R1 reconstruction requires the passed frozen semantic audit")
    manifest_path = next(path for path, _ in bundle.inputs if "spatial-materialization" in path.as_posix())
    manifest = load_verified_spatial_manifest(manifest_path)
    reader = SpatialObservationReader(manifest)
    groups, group_record = _scanner_groups(manifest.scans)
    model = joblib.load(semantic.model_path)
    # Six frozen non-overlapping 8 m contexts; no data-dependent context search.
    contexts = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    shape = (640, 768)
    height = np.full(shape, np.nan, dtype=np.float64)
    arrays = {
        "height_a": np.full(shape, np.nan),
        "height_b": np.full(shape, np.nan),
        "n_a": np.zeros(shape, dtype=np.uint8),
        "n_b": np.zeros(shape, dtype=np.uint8),
        "source_index": np.zeros(shape, dtype=np.uint32),
        "unique": np.zeros(shape, dtype=np.bool_),
        "joint_direct": np.zeros(shape, dtype=np.bool_),
        "provisional_probability": np.zeros(shape, dtype=np.float32),
    }
    context_summaries = []
    weight = np.zeros(shape, dtype=np.float32)
    band_a = np.full(shape, np.nan, dtype=np.float32)
    band_b = np.full(shape, np.nan, dtype=np.float32)
    local_energy = np.full(shape, np.nan, dtype=np.float32)
    local_confidence = np.zeros(shape, dtype=np.float32)
    view_confidence = np.zeros(shape, dtype=np.float32)
    signal_sum = noise_sum = valid_band_cells = 0.0
    for block_x, block_y in contexts:
        result = _context(
            manifest, reader, block_x, block_y, model, semantic.threshold,
            semantic.train_feature_envelope, groups, log=log
        )
        confidence = b1_confidence(
            result["height_a"], result["height_b"], result["joint_direct"],
            result["unique"], result["n_a"], result["n_b"]
        )
        y0, x0 = block_y * 128, block_x * 128
        target = np.s_[y0 : y0 + 128, x0 : x0 + 128]
        height[target] = result["height"]
        for name in arrays:
            arrays[name][target] = result[name]
        weight[target] = confidence.weight_fine
        band_a[target] = confidence.b_a_fine
        band_b[target] = confidence.b_b_fine
        local_energy[target] = confidence.local_energy_fine
        local_confidence[target] = confidence.local_confidence_fine
        view_confidence[target] = confidence.view_confidence_fine
        cells = int(confidence.valid_fine.sum())
        signal_sum += confidence.signal_m2 * cells
        noise_sum += confidence.noise_m2 * cells
        valid_band_cells += cells
        context_summaries.append(
            {"block_xy": [block_x, block_y], "c_area": confidence.c_area,
             "S_C_m2": confidence.signal_m2, "N_C_m2": confidence.noise_m2,
             "c_band": confidence.c_band, "weighted_area_m2": float(confidence.weight_fine.sum() * 0.0625**2)}
        )
    partition = _partition(
        shape, (-6.775, -6.1), (-6.775, -6.1), 0.0625,
        bundle.contract["contract_id"], analysis_halo_m=1.875, model_halo_m=1.0
    )
    weight[partition == 255] = 0
    capacity_record = capacity(weight, partition)
    b1_eligible = (weight > 0) & capacity_record["passed"]
    direct = np.isfinite(height) & arrays["unique"]
    signal = signal_sum / valid_band_cells if valid_band_cells else 0.0
    noise = noise_sum / valid_band_cells if valid_band_cells else 0.0
    c_band = signal / (signal + noise) if signal + noise > 0 else 0.0
    semantic_probabilities = np.full((*shape, 6), np.nan, dtype=np.float32)
    surface = ResearchSurfaceEvidence(
        tier="R1_weak_surface", source_id="hovi.hyytiala.2019.HY_SPRUCE4",
        cell_m=0.0625, origin_xy_m=(-6.775, -6.1),
        source_index=arrays["source_index"], height_m=np.where(direct, height, np.nan),
        direct_observed=direct, direct_support_distance_m=np.where(direct, 0.0, np.nan).astype(np.float32),
        footprint_m=np.full(shape, np.nan, dtype=np.float32),
        view_count=(arrays["n_a"] + arrays["n_b"]).astype(np.float32),
        interpolation_distance_m=np.where(direct, 0.0, np.nan).astype(np.float32),
        semantic_probabilities=semantic_probabilities,
        semantic_probability_available=np.zeros((*shape, 6), dtype=np.bool_),
        p_semantic_subclass_unknown=np.ones(shape, dtype=np.float32),
        heightfield_valid_probability=weight,
        provisional_surface_probability=arrays["provisional_probability"],
        p_unknown=(1.0 - weight).astype(np.float32),
        dynamic_water_probability=np.zeros(shape, dtype=np.float32),
        forbidden=np.zeros(shape, dtype=np.bool_),
        group_a_support=np.where(direct, arrays["n_a"], np.nan).astype(np.float32),
        group_b_support=np.where(direct, arrays["n_b"], np.nan).astype(np.float32),
        physical_view_count_available=direct.copy(),
        redundancy_group_count=np.zeros(shape, dtype=np.uint8),
        redundancy_group_count_available=np.zeros(shape, dtype=np.bool_),
        disagreement_m=np.abs(arrays["height_a"] - arrays["height_b"]).astype(np.float32),
        effective_support_radius_m=np.full(shape, np.nan, dtype=np.float32),
        epistemic_uncertainty_m=np.full(shape, np.nan, dtype=np.float32),
        repeatability_uncertainty_m=np.abs(arrays["height_a"] - arrays["height_b"]).astype(np.float32),
        joint_direct_support=arrays["joint_direct"], unique_geometry_support=arrays["unique"],
        band_a_m=band_a, band_b_m=band_b, local_disagreement_energy_m2=local_energy,
        local_confidence=local_confidence, view_confidence=view_confidence,
        weak_geometry_weight=weight, area_confidence=float(arrays["unique"].sum() / max(1, arrays["joint_direct"].sum())),
        cross_signal_m2=signal, cross_noise_m2=noise, band_confidence=c_band,
        b1_eligible=b1_eligible, b2_eligible=np.zeros(shape, dtype=np.bool_),
        partition=partition, unknown_reason=np.where(weight > 0, 0, 1).astype(np.uint8),
    )
    surface.validate()
    return surface, {"scanner_groups": group_record, "contexts": context_summaries, "capacity": capacity_record}

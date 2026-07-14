"""Unqualified, scan-balanced vertical-sheet hypotheses from Hovi TLS evidence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

from .accumulators import (
    CellEvidenceLevel,
    MultiscaleShardEvidence,
    VerticalSampleGroup,
)
from .manifest import RESOLUTIONS_M


@dataclass(frozen=True)
class HypothesisConfig:
    """Every numerical judgment used to form, but not accept, sheet modes."""

    vertical_bin_m: float
    smoothing_sigma_m: float
    max_vertical_bins: int
    max_hypotheses_per_cell: int
    group_batch_records: int
    level_block_records: int
    histogram_memory_ceiling_bytes: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.vertical_bin_m) or self.vertical_bin_m <= 0.0:
            raise ValueError("Hovi hypothesis vertical_bin_m must be positive")
        if (
            not math.isfinite(self.smoothing_sigma_m)
            or self.smoothing_sigma_m < self.vertical_bin_m
        ):
            raise ValueError("Hovi smoothing sigma must span at least one vertical bin")
        for name in (
            "max_vertical_bins",
            "max_hypotheses_per_cell",
            "group_batch_records",
            "level_block_records",
            "histogram_memory_ceiling_bytes",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"Hovi hypothesis {name} must be a positive integer")


@dataclass(frozen=True)
class ScanSheetSupport:
    scan: int
    guid: str
    points: int
    median_z_m: float
    mean_range_m: float
    mean_beam_direction_xyz: tuple[float, float, float] | None
    source_consecutive_pairs: int
    same_row_source_consecutive_pairs: int
    unit_column_source_consecutive_pairs: int
    same_column_source_consecutive_pairs: int
    unit_row_source_consecutive_pairs: int
    row_min: int
    row_max: int
    column_min: int
    column_max: int
    scanner_grid_edge_min: int


@dataclass(frozen=True)
class SheetHypothesis:
    hypothesis_index: int
    vertical_basin_min_m: float
    vertical_basin_max_m: float
    histogram_peak_m: float
    scan_balanced_z_m: float
    inter_scan_mad_m: float
    vertical_binning_half_width_m: float
    unique_scan_count: int
    scan_mask: int
    point_count: int
    scan_mean_range_min_m: float
    median_of_scan_mean_range_m: float
    scan_mean_range_max_m: float
    minimum_pairwise_view_angle_deg: float | None
    median_pairwise_view_angle_deg: float | None
    maximum_pairwise_view_angle_deg: float | None
    incidence_state: Literal["unknown_requires_candidate_normal"]
    scan_support: tuple[ScanSheetSupport, ...]


@dataclass(frozen=True)
class CellHypothesisSet:
    method_config: HypothesisConfig
    resolution_m: float
    shard_x: int
    shard_y: int
    cell_x: int
    cell_y: int
    observation_state: Literal["observed", "unobserved", "hypothesis_overflow"]
    sample_count: int
    unique_scan_count: int
    scan_mask: int
    within_shard_nearest_observed_distance_upper_bound_m: float | None
    distance_is_shard_boundary_censored: bool
    interpolation_state: Literal["not_performed"]
    occlusion_state: Literal["unknown_not_inferred_from_absence"]
    visibility_state: Literal["observed_returns_present", "unknown_no_returns"]
    semantic_state: Literal["unknown_unlabeled"]
    local_point_spacing_state: Literal["not_measured"]
    independent_error_state: Literal["unknown_not_estimated"]
    full_vs_thinned_transfer_state: Literal["not_measured"]
    qualification_status: Literal["raw_candidate_unqualified"]
    surface_claim: Literal[False]
    target_truth: Literal[False]
    synthesis_authorized: Literal[False]
    vertical_min_m: float | None
    vertical_max_m: float | None
    raw_mode_count: int | None
    hypotheses: tuple[SheetHypothesis, ...]
    unknown_reasons: tuple[str, ...]


class CandidateSurfaceHypothesisGenerator:
    """Generate competing sheet hypotheses without selecting a canonical surface."""

    def __init__(self, config: HypothesisConfig):
        self.config = config

    def iter_multiscale(
        self,
        evidence: MultiscaleShardEvidence,
    ) -> Iterator[CellHypothesisSet]:
        for resolution_m in (0.025, 0.05, 0.0625, 0.125, 0.25, 1.0):
            yield from self.iter_resolution(evidence, resolution_m)

    def iter_resolution(
        self,
        evidence: MultiscaleShardEvidence,
        resolution_m: float,
    ) -> Iterator[CellHypothesisSet]:
        if resolution_m not in RESOLUTIONS_M:
            raise ValueError(f"Hovi hypothesis resolution must be one of {RESOLUTIONS_M}")
        with evidence.open_level(
            resolution_m,
            block_records=self.config.level_block_records,
        ) as level:
            cells = level.cells_per_shard
            observed = np.zeros((cells, cells), dtype=np.bool_)
            observed.flat[level.cell_linear] = True
            if np.any(observed):
                nearest = distance_transform_edt(~observed) * resolution_m
            else:
                nearest = np.full((cells, cells), np.inf, dtype=np.float64)
            groups = iter(level.iter_vertical_groups())
            current = next(groups, None)
            for linear in range(cells * cells):
                cell_x = linear % cells
                cell_y = linear // cells
                distance = float(nearest[cell_y, cell_x])
                if current is None or current.cell_y * cells + current.cell_x != linear:
                    yield self._unknown_cell(
                        evidence,
                        resolution_m,
                        cell_x,
                        cell_y,
                        nearest_distance=(distance if math.isfinite(distance) else None),
                    )
                    continue
                yield self._observed_cell(
                    evidence,
                    level,
                    current,
                    nearest_distance=distance,
                )
                current = next(groups, None)
            if current is not None or next(groups, None) is not None:
                raise ValueError("Hovi occupied-cell iteration did not conserve groups")

    def _unknown_cell(
        self,
        evidence: MultiscaleShardEvidence,
        resolution_m: float,
        cell_x: int,
        cell_y: int,
        *,
        nearest_distance: float | None,
    ) -> CellHypothesisSet:
        return CellHypothesisSet(
            method_config=self.config,
            resolution_m=resolution_m,
            shard_x=evidence.shard_x,
            shard_y=evidence.shard_y,
            cell_x=cell_x,
            cell_y=cell_y,
            observation_state="unobserved",
            sample_count=0,
            unique_scan_count=0,
            scan_mask=0,
            within_shard_nearest_observed_distance_upper_bound_m=nearest_distance,
            distance_is_shard_boundary_censored=True,
            interpolation_state="not_performed",
            occlusion_state="unknown_not_inferred_from_absence",
            visibility_state="unknown_no_returns",
            semantic_state="unknown_unlabeled",
            local_point_spacing_state="not_measured",
            independent_error_state="unknown_not_estimated",
            full_vs_thinned_transfer_state="not_measured",
            qualification_status="raw_candidate_unqualified",
            surface_claim=False,
            target_truth=False,
            synthesis_authorized=False,
            vertical_min_m=None,
            vertical_max_m=None,
            raw_mode_count=0,
            hypotheses=(),
            unknown_reasons=("no_measured_return_in_cell", "occlusion_not_resolved"),
        )

    def _observed_cell(
        self,
        evidence: MultiscaleShardEvidence,
        level: CellEvidenceLevel,
        group: VerticalSampleGroup,
        *,
        nearest_distance: float,
    ) -> CellHypothesisSet:
        minimum = math.inf
        maximum = -math.inf
        sample_count = 0
        scan_mask = 0
        for batch in group.iter_batches(max_records=self.config.group_batch_records):
            minimum = min(minimum, float(np.min(batch.file_z_m)))
            maximum = max(maximum, float(np.max(batch.file_z_m)))
            sample_count += int(batch.file_z_m.size)
            for scan in np.unique(batch.scan):
                scan_mask |= 1 << int(scan)
        if sample_count != group.sample_count or not math.isfinite(minimum + maximum):
            raise ValueError("Hovi vertical group count/range is inconsistent")
        bin_count = max(1, int(math.floor((maximum - minimum) / self.config.vertical_bin_m)) + 1)
        if bin_count > self.config.max_vertical_bins:
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                nearest_distance,
                raw_mode_count=None,
                reason="vertical_histogram_ceiling_exceeded",
            )
        histogram_bytes = 16 * bin_count * np.dtype(np.uint64).itemsize
        work_bytes = histogram_bytes * 3
        if work_bytes > self.config.histogram_memory_ceiling_bytes:
            raise MemoryError("Hovi sheet histogram exceeds its explicit memory ceiling")
        histogram = np.zeros((16, bin_count), dtype=np.uint64)
        for batch in group.iter_batches(max_records=self.config.group_batch_records):
            bins = np.floor(
                (batch.file_z_m - minimum) / self.config.vertical_bin_m
            ).astype(np.int64)
            np.clip(bins, 0, bin_count - 1, out=bins)
            np.add.at(histogram, (batch.scan.astype(np.int64), bins), 1)
        totals = histogram.sum(axis=1)
        present = totals > 0
        # Equal per-scan mass prevents dense/near scans from winning by point count.
        normalized = histogram[present] / totals[present, None]
        balanced = normalized.mean(axis=0)
        sigma_bins = self.config.smoothing_sigma_m / self.config.vertical_bin_m
        smoothed = gaussian_filter1d(balanced, sigma=sigma_bins, mode="nearest")
        # Preserve every mode; there is deliberately no low/height/prominence filter.
        peaks = _all_plateau_peaks(smoothed)
        if peaks.size == 0:
            peaks = np.asarray([int(np.argmax(smoothed))], dtype=np.int64)
        if peaks.size > self.config.max_hypotheses_per_cell:
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                nearest_distance,
                raw_mode_count=int(peaks.size),
                reason="sheet_hypothesis_ceiling_exceeded",
            )
        basin_edges = _watershed_edges(smoothed, peaks)
        hypotheses = self._measure_hypotheses(
            evidence,
            group,
            histogram,
            minimum,
            maximum,
            peaks,
            basin_edges,
        )
        return CellHypothesisSet(
            method_config=self.config,
            resolution_m=level.resolution_m,
            shard_x=evidence.shard_x,
            shard_y=evidence.shard_y,
            cell_x=group.cell_x,
            cell_y=group.cell_y,
            observation_state="observed",
            sample_count=sample_count,
            unique_scan_count=scan_mask.bit_count(),
            scan_mask=scan_mask,
            within_shard_nearest_observed_distance_upper_bound_m=nearest_distance,
            distance_is_shard_boundary_censored=True,
            interpolation_state="not_performed",
            occlusion_state="unknown_not_inferred_from_absence",
            visibility_state="observed_returns_present",
            semantic_state="unknown_unlabeled",
            local_point_spacing_state="not_measured",
            independent_error_state="unknown_not_estimated",
            full_vs_thinned_transfer_state="not_measured",
            qualification_status="raw_candidate_unqualified",
            surface_claim=False,
            target_truth=False,
            synthesis_authorized=False,
            vertical_min_m=minimum,
            vertical_max_m=maximum,
            raw_mode_count=int(peaks.size),
            hypotheses=hypotheses,
            unknown_reasons=("surface_semantics_unknown", "occlusion_not_resolved"),
        )

    def _measure_hypotheses(
        self,
        evidence: MultiscaleShardEvidence,
        group: VerticalSampleGroup,
        histogram: np.ndarray,
        minimum: float,
        maximum: float,
        peaks: np.ndarray,
        basin_edges: np.ndarray,
    ) -> tuple[SheetHypothesis, ...]:
        count = peaks.size
        point_count = np.zeros((count, 16), dtype=np.uint64)
        range_sum = np.zeros((count, 16), dtype=np.float64)
        beam_sum = np.zeros((count, 16, 3), dtype=np.float64)
        beam_count = np.zeros((count, 16), dtype=np.uint64)
        row_min = np.full((count, 16), np.iinfo(np.uint16).max, dtype=np.uint16)
        row_max = np.zeros((count, 16), dtype=np.uint16)
        column_min = np.full((count, 16), np.iinfo(np.uint16).max, dtype=np.uint16)
        column_max = np.zeros((count, 16), dtype=np.uint16)
        continuity = np.zeros((count * 16, 5), dtype=np.uint64)
        last_source = np.full(count * 16, -1, dtype=np.int64)
        last_row = np.zeros(count * 16, dtype=np.int64)
        last_column = np.zeros(count * 16, dtype=np.int64)
        bin_to_hypothesis = np.searchsorted(
            basin_edges,
            np.arange(histogram.shape[1]),
            side="right",
        )
        for batch in group.iter_batches(max_records=self.config.group_batch_records):
            bins = np.floor(
                (batch.file_z_m - minimum) / self.config.vertical_bin_m
            ).astype(np.int64)
            np.clip(bins, 0, histogram.shape[1] - 1, out=bins)
            hypothesis = bin_to_hypothesis[bins]
            scan = batch.scan.astype(np.int64)
            key = hypothesis * 16 + scan
            np.add.at(point_count, (hypothesis, scan), 1)
            np.add.at(range_sum, (hypothesis, scan), batch.range_m)
            valid_beam = batch.beam_direction_valid
            np.add.at(
                beam_sum,
                (hypothesis[valid_beam], scan[valid_beam]),
                batch.beam_direction_xyz[valid_beam],
            )
            np.add.at(beam_count, (hypothesis[valid_beam], scan[valid_beam]), 1)
            for key_value in np.unique(key):
                selected = key == key_value
                h = int(key_value // 16)
                s = int(key_value % 16)
                row_min[h, s] = min(row_min[h, s], int(np.min(batch.row[selected])))
                row_max[h, s] = max(row_max[h, s], int(np.max(batch.row[selected])))
                column_min[h, s] = min(
                    column_min[h, s],
                    int(np.min(batch.column[selected])),
                )
                column_max[h, s] = max(
                    column_max[h, s],
                    int(np.max(batch.column[selected])),
                )
            _update_continuity(
                continuity,
                last_source,
                last_row,
                last_column,
                key,
                batch.source_ordinal.astype(np.int64),
                batch.row.astype(np.int64),
                batch.column.astype(np.int64),
            )
        result: list[SheetHypothesis] = []
        for hypothesis_index, peak in enumerate(peaks):
            scans = np.flatnonzero(point_count[hypothesis_index])
            scan_support: list[ScanSheetSupport] = []
            scan_medians: list[float] = []
            scan_ranges: list[float] = []
            view_directions: list[np.ndarray] = []
            low_bin = (
                0
                if hypothesis_index == 0
                else int(basin_edges[hypothesis_index - 1])
            )
            high_bin = (
                histogram.shape[1] - 1
                if hypothesis_index == count - 1
                else int(basin_edges[hypothesis_index] - 1)
            )
            for scan in scans:
                scan_histogram = histogram[scan, low_bin : high_bin + 1]
                median_bin = low_bin + _weighted_median_bin(scan_histogram)
                median_z = min(
                    maximum,
                    max(
                        minimum,
                        minimum + (median_bin + 0.5) * self.config.vertical_bin_m,
                    ),
                )
                points = int(point_count[hypothesis_index, scan])
                mean_range = float(range_sum[hypothesis_index, scan] / points)
                direction: tuple[float, float, float] | None = None
                if beam_count[hypothesis_index, scan] > 0:
                    vector = beam_sum[hypothesis_index, scan]
                    length = float(np.linalg.norm(vector))
                    if length > 0.0:
                        vector = vector / length
                        direction = tuple(float(value) for value in vector)
                        view_directions.append(vector)
                edge_min = min(
                    int(row_min[hypothesis_index, scan]),
                    11_003 - int(row_max[hypothesis_index, scan]),
                    int(column_min[hypothesis_index, scan]),
                    27_480 - int(column_max[hypothesis_index, scan]),
                )
                metric = continuity[hypothesis_index * 16 + int(scan)]
                scan_support.append(
                    ScanSheetSupport(
                        scan=int(scan),
                        guid=evidence.manifest.scans[int(scan)].guid,
                        points=points,
                        median_z_m=median_z,
                        mean_range_m=mean_range,
                        mean_beam_direction_xyz=direction,
                        source_consecutive_pairs=int(metric[0]),
                        same_row_source_consecutive_pairs=int(metric[1]),
                        unit_column_source_consecutive_pairs=int(metric[2]),
                        same_column_source_consecutive_pairs=int(metric[3]),
                        unit_row_source_consecutive_pairs=int(metric[4]),
                        row_min=int(row_min[hypothesis_index, scan]),
                        row_max=int(row_max[hypothesis_index, scan]),
                        column_min=int(column_min[hypothesis_index, scan]),
                        column_max=int(column_max[hypothesis_index, scan]),
                        scanner_grid_edge_min=edge_min,
                    )
                )
                scan_medians.append(median_z)
                scan_ranges.append(mean_range)
            medians = np.asarray(scan_medians, dtype=np.float64)
            # One median per scan, then one median across scans: never point weighted.
            balanced_z = float(np.median(medians))
            inter_scan_mad = float(np.median(np.abs(medians - balanced_z)))
            ranges = np.asarray(scan_ranges, dtype=np.float64)
            angle_minimum, angle_median, angle_maximum = _view_angle_summary(
                view_directions
            )
            result.append(
                SheetHypothesis(
                    hypothesis_index=hypothesis_index,
                    vertical_basin_min_m=max(
                        minimum,
                        minimum + low_bin * self.config.vertical_bin_m,
                    ),
                    vertical_basin_max_m=min(
                        maximum,
                        minimum + (high_bin + 1) * self.config.vertical_bin_m,
                    ),
                    histogram_peak_m=min(
                        maximum,
                        max(
                            minimum,
                            minimum
                            + (int(peak) + 0.5) * self.config.vertical_bin_m,
                        ),
                    ),
                    scan_balanced_z_m=balanced_z,
                    inter_scan_mad_m=inter_scan_mad,
                    vertical_binning_half_width_m=self.config.vertical_bin_m / 2.0,
                    unique_scan_count=len(scan_support),
                    scan_mask=sum(1 << support.scan for support in scan_support),
                    point_count=sum(support.points for support in scan_support),
                    scan_mean_range_min_m=float(np.min(ranges)),
                    median_of_scan_mean_range_m=float(np.median(ranges)),
                    scan_mean_range_max_m=float(np.max(ranges)),
                    minimum_pairwise_view_angle_deg=angle_minimum,
                    median_pairwise_view_angle_deg=angle_median,
                    maximum_pairwise_view_angle_deg=angle_maximum,
                    incidence_state="unknown_requires_candidate_normal",
                    scan_support=tuple(scan_support),
                )
            )
        return tuple(result)

    def _overflow_cell(
        self,
        evidence: MultiscaleShardEvidence,
        level: CellEvidenceLevel,
        group: VerticalSampleGroup,
        sample_count: int,
        scan_mask: int,
        minimum: float,
        maximum: float,
        nearest_distance: float,
        *,
        raw_mode_count: int | None,
        reason: str,
    ) -> CellHypothesisSet:
        return CellHypothesisSet(
            method_config=self.config,
            resolution_m=level.resolution_m,
            shard_x=evidence.shard_x,
            shard_y=evidence.shard_y,
            cell_x=group.cell_x,
            cell_y=group.cell_y,
            observation_state="hypothesis_overflow",
            sample_count=sample_count,
            unique_scan_count=scan_mask.bit_count(),
            scan_mask=scan_mask,
            within_shard_nearest_observed_distance_upper_bound_m=nearest_distance,
            distance_is_shard_boundary_censored=True,
            interpolation_state="not_performed",
            occlusion_state="unknown_not_inferred_from_absence",
            visibility_state="observed_returns_present",
            semantic_state="unknown_unlabeled",
            local_point_spacing_state="not_measured",
            independent_error_state="unknown_not_estimated",
            full_vs_thinned_transfer_state="not_measured",
            qualification_status="raw_candidate_unqualified",
            surface_claim=False,
            target_truth=False,
            synthesis_authorized=False,
            vertical_min_m=minimum,
            vertical_max_m=maximum,
            raw_mode_count=raw_mode_count,
            hypotheses=(),
            unknown_reasons=(reason, "surface_semantics_unknown", "occlusion_not_resolved"),
        )


def _all_plateau_peaks(values: np.ndarray) -> np.ndarray:
    peaks: list[int] = []
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[stop] == values[start]:
            stop += 1
        left = values[start - 1] if start > 0 else -math.inf
        right = values[stop] if stop < values.size else -math.inf
        if values[start] > 0.0 and values[start] > left and values[start] > right:
            peaks.append((start + stop - 1) // 2)
        start = stop
    return np.asarray(peaks, dtype=np.int64)


def _watershed_edges(values: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    edges: list[int] = []
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        left_index = int(left)
        right_index = int(right)
        if right_index - left_index <= 1:
            edges.append(right_index)
            continue
        segment = values[left_index + 1 : right_index]
        edges.append(left_index + 1 + int(np.argmin(segment)))
    return np.asarray(edges, dtype=np.int64)


def _weighted_median_bin(counts: np.ndarray) -> int:
    total = int(np.sum(counts))
    if total <= 0:
        raise ValueError("Hovi sheet scan histogram is empty")
    return int(np.searchsorted(np.cumsum(counts), (total + 1) // 2, side="left"))


def _view_angle_summary(
    directions: list[np.ndarray],
) -> tuple[float | None, float | None, float | None]:
    if len(directions) < 2:
        return None, None, None
    matrix = np.asarray(directions, dtype=np.float64)
    cosine = np.clip(matrix @ matrix.T, -1.0, 1.0)
    pair = np.degrees(np.arccos(cosine[np.triu_indices(len(directions), k=1)]))
    return float(np.min(pair)), float(np.median(pair)), float(np.max(pair))


def _update_continuity(
    metrics: np.ndarray,
    last_source: np.ndarray,
    last_row: np.ndarray,
    last_column: np.ndarray,
    key: np.ndarray,
    source: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
) -> None:
    order = np.argsort(key, kind="stable")
    key = key[order]
    source = source[order]
    row = row[order]
    column = column[order]
    same = key[1:] == key[:-1]
    _count_transitions(
        metrics,
        key[1:][same],
        source[:-1][same],
        row[:-1][same],
        column[:-1][same],
        source[1:][same],
        row[1:][same],
        column[1:][same],
    )
    first = np.flatnonzero(np.r_[True, ~same])
    first_key = key[first]
    prior = last_source[first_key] >= 0
    _count_transitions(
        metrics,
        first_key[prior],
        last_source[first_key[prior]],
        last_row[first_key[prior]],
        last_column[first_key[prior]],
        source[first[prior]],
        row[first[prior]],
        column[first[prior]],
    )
    last = np.r_[first[1:] - 1, key.size - 1]
    last_key = key[last]
    last_source[last_key] = source[last]
    last_row[last_key] = row[last]
    last_column[last_key] = column[last]


def _count_transitions(
    metrics: np.ndarray,
    key: np.ndarray,
    source0: np.ndarray,
    row0: np.ndarray,
    column0: np.ndarray,
    source1: np.ndarray,
    row1: np.ndarray,
    column1: np.ndarray,
) -> None:
    consecutive = source1 - source0 == 1
    same_row = row1 == row0
    same_column = column1 == column0
    np.add.at(metrics[:, 0], key[consecutive], 1)
    np.add.at(metrics[:, 1], key[consecutive & same_row], 1)
    np.add.at(
        metrics[:, 2],
        key[consecutive & same_row & (np.abs(column1 - column0) == 1)],
        1,
    )
    np.add.at(metrics[:, 3], key[consecutive & same_column], 1)
    np.add.at(
        metrics[:, 4],
        key[consecutive & same_column & (np.abs(row1 - row0) == 1)],
        1,
    )

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
from .manifest import RESOLUTIONS_M, SHARD_M


@dataclass(frozen=True)
class HypothesisConfig:
    """Every numerical judgment used to form, but not accept, sheet modes."""

    vertical_bin_m: float
    smoothing_sigma_m: float
    mode_union_resolution_m: float
    max_vertical_bins: int
    max_hypotheses_per_cell: int
    group_batch_records: int
    level_block_records: int
    peak_live_memory_ceiling_bytes: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.vertical_bin_m) or self.vertical_bin_m <= 0.0:
            raise ValueError("Hovi hypothesis vertical_bin_m must be positive")
        if (
            not math.isfinite(self.smoothing_sigma_m)
            or self.smoothing_sigma_m < self.vertical_bin_m
        ):
            raise ValueError("Hovi smoothing sigma must span at least one vertical bin")
        if (
            not math.isfinite(self.mode_union_resolution_m)
            or self.mode_union_resolution_m < self.vertical_bin_m
        ):
            raise ValueError("Hovi mode-union resolution must span at least one bin")
        for name in (
            "max_vertical_bins",
            "max_hypotheses_per_cell",
            "group_batch_records",
            "level_block_records",
            "peak_live_memory_ceiling_bytes",
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
    descriptive_inter_scan_mad_m: float
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
    pooled_mode_contributed: bool
    mode_cluster_span_m: float
    mode_union_resolution_m: float
    contributing_scan_modes: tuple[tuple[int, float], ...]
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
    occupied_cell_center_distance_diagnostic_m: float | None
    occupied_cell_center_distance_role: Literal[
        "diagnostic_only_forbidden_for_support_or_interpolation"
    ]
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
    conservative_peak_live_memory_bound_bytes: int
    hypotheses: tuple[SheetHypothesis, ...]
    unknown_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _ModeCluster:
    peak_bin: int
    source_modes: tuple[tuple[int, int], ...]

    @property
    def pooled_mode_contributed(self) -> bool:
        return any(source < 0 for source, _ in self.source_modes)

    @property
    def scan_modes(self) -> tuple[tuple[int, int], ...]:
        return tuple(value for value in self.source_modes if value[0] >= 0)

    @property
    def minimum_bin(self) -> int:
        return min(mode_bin for _, mode_bin in self.source_modes)

    @property
    def maximum_bin(self) -> int:
        return max(mode_bin for _, mode_bin in self.source_modes)


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
        peak_live_bound = _conservative_peak_live_memory_bound_bytes(
            self.config,
            resolution_m,
        )
        if self.config.peak_live_memory_ceiling_bytes > evidence.memory_ceiling_bytes:
            raise MemoryError(
                "Hovi hypothesis memory ceiling exceeds the evidence workspace ceiling"
            )
        if peak_live_bound > self.config.peak_live_memory_ceiling_bytes:
            raise MemoryError(
                "Hovi hypothesis peak-live bound exceeds its explicit memory ceiling"
            )
        with evidence.open_level(
            resolution_m,
            block_records=self.config.level_block_records,
        ) as level:
            cells = level.cells_per_shard
            observed = np.zeros((cells, cells), dtype=np.bool_)
            observed.flat[level.cell_linear] = True
            if np.any(observed):
                cell_center_distance = distance_transform_edt(~observed) * resolution_m
            else:
                cell_center_distance = np.full(
                    (cells, cells), np.inf, dtype=np.float64
                )
            groups = iter(level.iter_vertical_groups())
            current = next(groups, None)
            for linear in range(cells * cells):
                cell_x = linear % cells
                cell_y = linear // cells
                distance = float(cell_center_distance[cell_y, cell_x])
                if current is None or current.cell_y * cells + current.cell_x != linear:
                    yield self._unknown_cell(
                        evidence,
                        resolution_m,
                        cell_x,
                        cell_y,
                        cell_center_distance=(
                            distance if math.isfinite(distance) else None
                        ),
                        peak_live_bound=peak_live_bound,
                    )
                    continue
                yield self._observed_cell(
                    evidence,
                    level,
                    current,
                    cell_center_distance=distance,
                    peak_live_bound=peak_live_bound,
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
        cell_center_distance: float | None,
        peak_live_bound: int,
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
            occupied_cell_center_distance_diagnostic_m=cell_center_distance,
            occupied_cell_center_distance_role=(
                "diagnostic_only_forbidden_for_support_or_interpolation"
            ),
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
            conservative_peak_live_memory_bound_bytes=peak_live_bound,
            hypotheses=(),
            unknown_reasons=(
                "no_measured_return_in_cell",
                "occlusion_not_resolved",
                "occupied_cell_center_distance_not_support",
            ),
        )

    def _observed_cell(
        self,
        evidence: MultiscaleShardEvidence,
        level: CellEvidenceLevel,
        group: VerticalSampleGroup,
        *,
        cell_center_distance: float,
        peak_live_bound: int,
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
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=None,
                reason="vertical_histogram_ceiling_exceeded",
            )
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
        sigma_bins = self.config.smoothing_sigma_m / self.config.vertical_bin_m
        per_scan_smoothed = gaussian_filter1d(
            normalized,
            sigma=sigma_bins,
            axis=1,
            mode="nearest",
        )
        pooled_smoothed = gaussian_filter1d(
            normalized.mean(axis=0),
            sigma=sigma_bins,
            mode="nearest",
        )
        pooled_peaks = _density_modes(pooled_smoothed)
        per_scan_peaks = tuple(
            (int(scan), _density_modes(per_scan_smoothed[index]))
            for index, scan in enumerate(np.flatnonzero(present))
        )
        clusters = _union_mode_clusters(
            pooled_peaks,
            per_scan_peaks,
            vertical_bin_m=self.config.vertical_bin_m,
            union_resolution_m=self.config.mode_union_resolution_m,
        )
        detected_mode_count = int(pooled_peaks.size) + sum(
            int(scan_peaks.size) for _, scan_peaks in per_scan_peaks
        )
        if not _mode_mapping_is_complete(pooled_peaks, per_scan_peaks, clusters):
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=detected_mode_count,
                reason="per_scan_mode_mapping_unresolved",
            )
        if not _mode_clusters_are_source_unique(clusters):
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=detected_mode_count,
                reason="same_source_mode_collision_unresolved",
            )
        if len(clusters) > self.config.max_hypotheses_per_cell:
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=detected_mode_count,
                reason="sheet_hypothesis_ceiling_exceeded",
            )
        peaks = np.asarray([cluster.peak_bin for cluster in clusters], dtype=np.int64)
        if np.any(peaks[1:] <= peaks[:-1]):
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=detected_mode_count,
                reason="candidate_mode_order_unresolved",
            )
        basin_edges = _constrained_basin_edges(pooled_smoothed, clusters)
        if basin_edges is None or not _mode_basin_mapping_is_complete(
            clusters,
            basin_edges,
        ):
            return self._overflow_cell(
                evidence,
                level,
                group,
                sample_count,
                scan_mask,
                minimum,
                maximum,
                cell_center_distance,
                peak_live_bound,
                raw_mode_count=detected_mode_count,
                reason="mode_basin_assignment_unresolved",
            )
        hypotheses = self._measure_hypotheses(
            evidence,
            group,
            histogram,
            minimum,
            maximum,
            peaks,
            basin_edges,
            clusters,
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
            occupied_cell_center_distance_diagnostic_m=cell_center_distance,
            occupied_cell_center_distance_role=(
                "diagnostic_only_forbidden_for_support_or_interpolation"
            ),
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
            raw_mode_count=detected_mode_count,
            conservative_peak_live_memory_bound_bytes=peak_live_bound,
            hypotheses=hypotheses,
            unknown_reasons=(
                "surface_semantics_unknown",
                "occlusion_not_resolved",
                "occupied_cell_center_distance_not_support",
            ),
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
        clusters: tuple[_ModeCluster, ...],
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
            # Descriptive cross-view disagreement only, never an error estimate.
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
                    descriptive_inter_scan_mad_m=inter_scan_mad,
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
                    pooled_mode_contributed=clusters[
                        hypothesis_index
                    ].pooled_mode_contributed,
                    mode_cluster_span_m=(
                        clusters[hypothesis_index].maximum_bin
                        - clusters[hypothesis_index].minimum_bin
                    )
                    * self.config.vertical_bin_m,
                    mode_union_resolution_m=self.config.mode_union_resolution_m,
                    contributing_scan_modes=tuple(
                        (
                            scan,
                            min(
                                maximum,
                                max(
                                    minimum,
                                    minimum
                                    + (mode_bin + 0.5)
                                    * self.config.vertical_bin_m,
                                ),
                            ),
                        )
                        for scan, mode_bin in clusters[hypothesis_index].scan_modes
                    ),
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
        cell_center_distance: float,
        peak_live_bound: int,
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
            occupied_cell_center_distance_diagnostic_m=cell_center_distance,
            occupied_cell_center_distance_role=(
                "diagnostic_only_forbidden_for_support_or_interpolation"
            ),
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
            conservative_peak_live_memory_bound_bytes=peak_live_bound,
            hypotheses=(),
            unknown_reasons=(
                reason,
                "surface_semantics_unknown",
                "occlusion_not_resolved",
                "occupied_cell_center_distance_not_support",
            ),
        )


def _conservative_peak_live_memory_bound_bytes(
    config: HypothesisConfig,
    resolution_m: float,
) -> int:
    """Bound all explicit NumPy/SciPy live work before opening a level."""
    cells = round(SHARD_M / resolution_m)
    cell_count = cells * cells
    # The index itself is disk-backed. This includes its construction batches and
    # expanded selections, not OS-controlled file-cache residency.
    level_arrays_and_index_batches = (
        96 * cell_count + 256 * config.level_block_records
    )
    occupied_grid_and_distance_transform = 64 * cell_count
    expanded_vertical_batches = 384 * config.group_batch_records
    histogram_modes_and_scipy_temporaries = 4096 * config.max_vertical_bins
    hypothesis_accumulators = 4096 * config.max_hypotheses_per_cell
    fixed_interpreter_and_array_overhead = 1 << 20
    return (
        level_arrays_and_index_batches
        + occupied_grid_and_distance_transform
        + expanded_vertical_batches
        + histogram_modes_and_scipy_temporaries
        + hypothesis_accumulators
        + fixed_interpreter_and_array_overhead
    )


def _density_modes(values: np.ndarray) -> np.ndarray:
    modes = _all_plateau_peaks(values)
    if modes.size == 0:
        return np.asarray([int(np.argmax(values))], dtype=np.int64)
    return modes


def _union_mode_clusters(
    pooled_modes: np.ndarray,
    per_scan_modes: tuple[tuple[int, np.ndarray], ...],
    *,
    vertical_bin_m: float,
    union_resolution_m: float,
) -> tuple[_ModeCluster, ...]:
    mode_bins: list[int] = [int(value) for value in pooled_modes]
    mode_scans: list[int] = [-1] * len(mode_bins)
    for scan, modes in per_scan_modes:
        mode_bins.extend(int(value) for value in modes)
        mode_scans.extend([scan] * int(modes.size))
    if not mode_bins:
        return ()
    bins = np.asarray(mode_bins, dtype=np.int64)
    scans = np.asarray(mode_scans, dtype=np.int16)
    order = np.lexsort((scans, bins))
    bins = bins[order]
    scans = scans[order]
    maximum_span_bins = int(
        math.floor(union_resolution_m / vertical_bin_m + 1e-12)
    )
    starts = [0]
    cluster_minimum = int(bins[0])
    for index in range(1, bins.size):
        if int(bins[index]) - cluster_minimum > maximum_span_bins:
            starts.append(index)
            cluster_minimum = int(bins[index])
    stops = starts[1:] + [int(bins.size)]
    result: list[_ModeCluster] = []
    for start, stop in zip(starts, stops, strict=True):
        cluster_bins = bins[start:stop]
        cluster_scans = scans[start:stop]
        center = float(np.mean(cluster_bins))
        actual_bins = np.unique(cluster_bins)
        representative = min(
            (int(value) for value in actual_bins),
            key=lambda value: (
                abs(value - center),
                not np.any((cluster_bins == value) & (cluster_scans < 0)),
                value,
            ),
        )
        result.append(
            _ModeCluster(
                peak_bin=representative,
                source_modes=tuple(
                    (int(source), int(mode_bin))
                    for source, mode_bin in zip(
                        cluster_scans,
                        cluster_bins,
                        strict=True,
                    )
                ),
            )
        )
    return tuple(result)


def _mode_mapping_is_complete(
    pooled_modes: np.ndarray,
    per_scan_modes: tuple[tuple[int, np.ndarray], ...],
    clusters: tuple[_ModeCluster, ...],
) -> bool:
    expected = [(-1, int(mode_bin)) for mode_bin in pooled_modes]
    expected.extend(
        (scan, int(mode_bin))
        for scan, modes in per_scan_modes
        for mode_bin in modes
    )
    mapped = sorted(
        source_mode for cluster in clusters for source_mode in cluster.source_modes
    )
    return bool(clusters) and mapped == sorted(expected)


def _mode_clusters_are_source_unique(clusters: tuple[_ModeCluster, ...]) -> bool:
    return all(
        len({source for source, _ in cluster.source_modes})
        == len(cluster.source_modes)
        for cluster in clusters
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


def _constrained_basin_edges(
    values: np.ndarray,
    clusters: tuple[_ModeCluster, ...],
) -> np.ndarray | None:
    edges: list[int] = []
    for left, right in zip(clusters[:-1], clusters[1:], strict=True):
        first_cut = left.maximum_bin + 1
        last_cut = right.minimum_bin
        if first_cut > last_cut:
            return None
        cuts = np.arange(first_cut, last_cut + 1, dtype=np.int64)
        cut_density = values[cuts - 1] + values[cuts]
        edges.append(int(cuts[int(np.argmin(cut_density))]))
    return np.asarray(edges, dtype=np.int64)


def _mode_basin_mapping_is_complete(
    clusters: tuple[_ModeCluster, ...],
    basin_edges: np.ndarray,
) -> bool:
    if basin_edges.shape != (max(0, len(clusters) - 1),):
        return False
    if np.any(basin_edges[1:] <= basin_edges[:-1]):
        return False
    for index, cluster in enumerate(clusters):
        if cluster.peak_bin not in {mode_bin for _, mode_bin in cluster.source_modes}:
            return False
        if index > 0 and cluster.minimum_bin < int(basin_edges[index - 1]):
            return False
        if index < len(basin_edges) and cluster.maximum_bin >= int(basin_edges[index]):
            return False
        member_bins = np.asarray(
            [mode_bin for _, mode_bin in cluster.source_modes],
            dtype=np.int64,
        )
        assigned = np.searchsorted(basin_edges, member_bins, side="right")
        if np.any(assigned != index):
            return False
    return True


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

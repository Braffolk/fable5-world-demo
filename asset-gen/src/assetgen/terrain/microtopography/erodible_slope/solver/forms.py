"""World-coordinate typed-form construction and absolute fine-surface rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .model import ProcessConfig, SlopeDomain
from .process import ProcessResult

FORM_NONE = 0
FORM_RILL = 1
FORM_HEADCUT = 2
FORM_SEEP = 3
FORM_TOE = 4


@dataclass(frozen=True)
class FormPlan:
    rill_points_en: np.ndarray
    rill_streamline_id: np.ndarray
    rill_radius_m: np.ndarray
    rill_depth_m: np.ndarray
    headcut_points_en: np.ndarray
    headcut_radius_m: np.ndarray
    headcut_depth_m: np.ndarray
    seep_points_en: np.ndarray
    seep_radius_m: np.ndarray
    seep_depth_m: np.ndarray
    negative_scale: float
    toe_process_relief_m: np.ndarray
    represented_erosion_volume_m3: float
    unrepresented_erosion_volume_m3: float
    represented_deposition_volume_m3: float
    unrepresented_deposition_volume_m3: float
    channel_heads: int
    retained_head_candidates: int
    streamlines: int


@dataclass(frozen=True)
class FineSurface:
    bbox_en: tuple[float, float, float, float]
    texel_m: float
    c0_height_m: np.ndarray
    c1_height_m: np.ndarray
    residual_m: np.ndarray
    incision_m: np.ndarray
    deposition_m: np.ndarray
    ownership: np.ndarray
    hard_exclusion: np.ndarray
    material_supported: np.ndarray


def _indices_to_en(
    domain: SlopeDomain,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e_min, _, _, n_max = domain.bbox_en
    east = e_min + (cols + 0.5) * domain.texel_m
    north = n_max - (rows + 0.5) * domain.texel_m
    return east, north


def _en_to_indices(
    domain: SlopeDomain,
    east: np.ndarray,
    north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e_min, _, _, n_max = domain.bbox_en
    cols = (east - e_min) / domain.texel_m - 0.5
    rows = (n_max - north) / domain.texel_m - 0.5
    return rows, cols


def _sample(
    domain: SlopeDomain,
    values: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    rows, cols = _en_to_indices(domain, east, north)
    return ndimage.map_coordinates(
        np.asarray(values),
        (rows, cols),
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _sample_scalar_bilinear(
    domain: SlopeDomain,
    values: np.ndarray,
    east: float,
    north: float,
) -> float:
    e_min, _, _, n_max = domain.bbox_en
    row_value = min(
        max((n_max - north) / domain.texel_m - 0.5, 0.0),
        values.shape[0] - 1.0,
    )
    col_value = min(
        max((east - e_min) / domain.texel_m - 0.5, 0.0),
        values.shape[1] - 1.0,
    )
    row0 = int(np.floor(row_value))
    col0 = int(np.floor(col_value))
    row1 = min(row0 + 1, values.shape[0] - 1)
    col1 = min(col0 + 1, values.shape[1] - 1)
    row_fraction = row_value - row0
    col_fraction = col_value - col0
    north_value = values[row0, col0] + col_fraction * (
        values[row0, col1] - values[row0, col0]
    )
    south_value = values[row1, col0] + col_fraction * (
        values[row1, col1] - values[row1, col0]
    )
    return float(north_value + row_fraction * (south_value - north_value))


def _sample_scalar_nearest(
    domain: SlopeDomain,
    values: np.ndarray,
    east: float,
    north: float,
) -> float:
    e_min, _, _, n_max = domain.bbox_en
    row = (n_max - north) / domain.texel_m - 0.5
    col = (east - e_min) / domain.texel_m - 0.5
    rr = min(max(int(np.floor(row + 0.5)), 0), values.shape[0] - 1)
    cc = min(max(int(np.floor(col + 0.5)), 0), values.shape[1] - 1)
    return float(values[rr, cc])


def _channel_heads(
    domain: SlopeDomain,
    process: ProcessResult,
    config: ProcessConfig,
) -> np.ndarray:
    channel = (
        domain.form_active
        & (process.routing.contributing_area_m2 >= config.rill_area_threshold_m2)
        & (process.routing.slope >= config.min_slope)
        & (process.eroded_depth_m > 0.0)
    )
    donors = np.zeros(domain.height_m.size, dtype=np.int32)
    channel_flat = channel.ravel()
    for target, weight in (
        (process.routing.target_a.ravel(), process.routing.weight_a.ravel()),
        (process.routing.target_b.ravel(), process.routing.weight_b.ravel()),
    ):
        sources = np.flatnonzero(channel_flat & (target >= 0) & (weight > 0.0))
        selected = sources[channel_flat[target[sources]]]
        np.add.at(donors, target[selected], 1)
    return channel & (donors.reshape(channel.shape) == 0)


def _trace_streamline(
    domain: SlopeDomain,
    process: ProcessResult,
    config: ProcessConfig,
    active: np.ndarray,
    start_e: float,
    start_n: float,
) -> np.ndarray:
    points = [(start_e, start_n)]
    e_min, n_min, e_max, n_max = domain.bbox_en
    outlet_mask = domain.outlet
    max_steps = int(np.ceil(config.streamline_max_length_m / config.streamline_step_m))
    for _ in range(max_steps):
        east, north = points[-1]
        ve = _sample_scalar_bilinear(
            domain, process.routing.downhill_east, east, north
        )
        vs = _sample_scalar_bilinear(
            domain, process.routing.downhill_south, east, north
        )
        norm = float(np.hypot(ve, vs))
        if norm < config.min_slope * 0.1:
            break
        de = config.streamline_step_m * ve / norm
        dn = -config.streamline_step_m * vs / norm
        mid_e = east + 0.5 * de
        mid_n = north + 0.5 * dn
        mid_ve = _sample_scalar_bilinear(
            domain, process.routing.downhill_east, mid_e, mid_n
        )
        mid_vs = _sample_scalar_bilinear(
            domain, process.routing.downhill_south, mid_e, mid_n
        )
        mid_norm = float(np.hypot(mid_ve, mid_vs))
        if mid_norm < config.min_slope * 0.1:
            break
        next_e = east + config.streamline_step_m * mid_ve / mid_norm
        next_n = north - config.streamline_step_m * mid_vs / mid_norm
        if not (e_min <= next_e < e_max and n_min < next_n <= n_max):
            break
        inside = _sample_scalar_nearest(domain, active, next_e, next_n)
        if inside < 0.5:
            break
        points.append((next_e, next_n))
        outlet = _sample_scalar_nearest(domain, outlet_mask, next_e, next_n)
        if outlet >= 0.5:
            break
    return np.asarray(points, dtype=np.float64)


def _nearest_line_relief(
    points_en: np.ndarray,
    radius_m: np.ndarray,
    depth_m: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    if points_en.size == 0:
        return np.zeros(east.shape, dtype=np.float64)
    tree = cKDTree(points_en)
    query = np.column_stack((east.ravel(), north.ravel()))
    distance, index = tree.query(query, workers=1)
    radius = radius_m[index]
    q = np.clip(distance / np.maximum(radius, 1e-9), 0.0, 1.0)
    relief = depth_m[index] * (1.0 - q * q) ** 2
    relief[distance >= radius] = 0.0
    return relief.reshape(east.shape)


def _densify_polyline(
    points: np.ndarray,
    radius: np.ndarray,
    depth: np.ndarray,
    *,
    streamline_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Densify every segment below half its local minimum form radius."""
    dense_points: list[np.ndarray] = []
    dense_radius: list[np.ndarray] = []
    dense_depth: list[np.ndarray] = []
    for index in range(len(points) - 1):
        start = points[index]
        stop = points[index + 1]
        length = float(np.linalg.norm(stop - start))
        spacing = max(min(radius[index], radius[index + 1]) * 0.45, 0.01)
        count = max(1, int(np.ceil(length / spacing)))
        phase = np.arange(count, dtype=np.float64) / count
        dense_points.append(start[None, :] + phase[:, None] * (stop - start)[None, :])
        dense_radius.append(radius[index] + phase * (radius[index + 1] - radius[index]))
        dense_depth.append(depth[index] + phase * (depth[index + 1] - depth[index]))
    dense_points.append(points[-1:])
    dense_radius.append(radius[-1:])
    dense_depth.append(depth[-1:])
    merged_points = np.concatenate(dense_points)
    merged_radius = np.concatenate(dense_radius)
    merged_depth = np.concatenate(dense_depth)
    ids = np.full(len(merged_points), streamline_id, dtype=np.int32)
    return merged_points, merged_radius, merged_depth, ids


def _first_owned_point(
    points_en: np.ndarray,
    radius_m: np.ndarray,
    ownership_bins: dict[tuple[int, int], list[tuple[float, float, float]]],
    *,
    bin_m: float,
    maximum_existing_radius_m: float,
    integration_step_m: float,
) -> int | None:
    """Return the first point entering already-owned continuous rill support."""
    for point_index, ((east, north), radius) in enumerate(
        zip(points_en, radius_m, strict=True)
    ):
        e_bin = int(np.floor(east / bin_m))
        n_bin = int(np.floor(north / bin_m))
        reach = radius + maximum_existing_radius_m + 0.5 * integration_step_m
        bin_reach = int(np.ceil(reach / bin_m))
        for de in range(-bin_reach, bin_reach + 1):
            for dn in range(-bin_reach, bin_reach + 1):
                for other_e, other_n, other_radius in ownership_bins.get(
                    (e_bin + de, n_bin + dn), ()
                ):
                    support = radius + other_radius + 0.5 * integration_step_m
                    if (east - other_e) ** 2 + (north - other_n) ** 2 <= support**2:
                        return point_index
    return None


def _suppress_parallel_heads(
    domain: SlopeDomain,
    process: ProcessResult,
    config: ProcessConfig,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Keep the strongest head per parallel sub-catchment neighborhood."""
    if not rows.size:
        return np.empty((0,), dtype=np.int64)
    east, north = _indices_to_en(
        domain, rows.astype(np.float64), cols.astype(np.float64)
    )
    flow_east = process.routing.downhill_east[rows, cols]
    flow_south = process.routing.downhill_south[rows, cols]
    norm = np.hypot(flow_east, flow_south)
    flow_east = np.divide(flow_east, norm, out=np.zeros_like(flow_east), where=norm > 0)
    flow_south = np.divide(
        flow_south, norm, out=np.zeros_like(flow_south), where=norm > 0
    )
    score = (
        process.eroded_depth_m[rows, cols]
        * process.routing.contributing_area_m2[rows, cols]
    )
    normalized_area = np.clip(
        np.log2(
            np.maximum(
                process.routing.contributing_area_m2[rows, cols],
                config.rill_area_threshold_m2,
            )
            / config.rill_area_threshold_m2
        )
        / 5.0,
        0.0,
        1.0,
    )
    radius = 0.5 * (
        config.rill_width_min_m
        + (config.rill_width_max_m - config.rill_width_min_m)
        * np.sqrt(normalized_area)
    )
    order = np.lexsort((cols, rows, -score))
    bin_m = config.rill_width_max_m
    bins: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []
    for index in order:
        e_bin = int(np.floor(east[index] / bin_m))
        n_bin = int(np.floor(north[index] / bin_m))
        redundant = False
        for de in (-1, 0, 1):
            for dn in (-1, 0, 1):
                for other in bins.get((e_bin + de, n_bin + dn), ()):
                    distance2 = (east[index] - east[other]) ** 2 + (
                        north[index] - north[other]
                    ) ** 2
                    support = (
                        radius[index]
                        + radius[other]
                        + 0.5 * config.streamline_step_m
                    )
                    parallel = (
                        flow_east[index] * flow_east[other]
                        + flow_south[index] * flow_south[other]
                    ) >= np.sqrt(0.5)
                    if distance2 <= support**2 and parallel:
                        redundant = True
                        break
                if redundant:
                    break
            if redundant:
                break
        if redundant:
            continue
        accepted.append(int(index))
        bins.setdefault((e_bin, n_bin), []).append(int(index))
    return np.asarray(accepted, dtype=np.int64)


def _add_owned_points(
    points_en: np.ndarray,
    radius_m: np.ndarray,
    ownership_bins: dict[tuple[int, int], list[tuple[float, float, float]]],
    *,
    bin_m: float,
) -> None:
    for (east, north), radius in zip(points_en, radius_m, strict=True):
        key = (int(np.floor(east / bin_m)), int(np.floor(north / bin_m)))
        ownership_bins.setdefault(key, []).append((float(east), float(north), float(radius)))


def _flow_aligned_toe_relief(
    domain: SlopeDomain,
    process: ProcessResult,
    config: ProcessConfig,
) -> tuple[np.ndarray, float, float]:
    """Reconstruct cell-integrated deposition as a compact continuous lobe field."""
    source_support = domain.form_active & (
        process.routing.slope <= config.deposition_slope_ceiling
    )
    reconstruction_support = domain.form_active
    toe = np.where(source_support, process.deposited_depth_m, 0.0)
    process_deposition_volume = float(
        np.sum(process.deposited_depth_m) * domain.texel_m**2
    )
    eligible_toe_volume = float(np.sum(toe) * domain.texel_m**2)
    east = process.routing.downhill_east
    south = process.routing.downhill_south
    magnitude = np.hypot(east, south)
    unit_east = np.divide(east, magnitude, out=np.zeros_like(east), where=magnitude > 0)
    unit_south = np.divide(
        south, magnitude, out=np.zeros_like(south), where=magnitude > 0
    )
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    # Four conservative finite-volume reconstruction passes give compact four-cell
    # support. Directional weights align the footprint with local drainage while
    # retaining a small cross-flow component for a lobe rather than a line.
    for _ in range(4):
        denominator = np.full(toe.shape, 0.5, dtype=np.float64)
        transfers: list[tuple[tuple[slice, slice], tuple[slice, slice], np.ndarray]] = []
        for dr, dc in directions:
            source_rows = slice(max(0, -dr), min(toe.shape[0], toe.shape[0] - dr))
            source_cols = slice(max(0, -dc), min(toe.shape[1], toe.shape[1] - dc))
            target_rows = slice(max(0, dr), min(toe.shape[0], toe.shape[0] + dr))
            target_cols = slice(max(0, dc), min(toe.shape[1], toe.shape[1] + dc))
            direction_norm = np.hypot(dr, dc)
            alignment = np.abs(
                unit_south[source_rows, source_cols] * (dr / direction_norm)
                + unit_east[source_rows, source_cols] * (dc / direction_norm)
            )
            weight = (0.125 + 0.875 * alignment**2) * reconstruction_support[
                target_rows, target_cols
            ]
            denominator[source_rows, source_cols] += weight
            transfers.append(
                ((source_rows, source_cols), (target_rows, target_cols), weight)
            )
        reconstructed = toe * (0.5 / denominator)
        for source_slice, target_slice, weight in transfers:
            reconstructed[target_slice] += (
                toe[source_slice] * weight / denominator[source_slice]
            )
        toe = reconstructed
    toe[~reconstruction_support] = 0.0
    reconstructed_volume = float(np.sum(toe) * domain.texel_m**2)
    if reconstructed_volume > 0.0:
        toe *= eligible_toe_volume / reconstructed_volume
    toe = np.minimum(toe, config.toe_relief_max_m)
    represented = float(np.sum(toe) * domain.texel_m**2)
    return toe, represented, max(0.0, process_deposition_volume - represented)


def build_form_plan(
    domain: SlopeDomain,
    process: ProcessResult,
    config: ProcessConfig,
) -> FormPlan:
    heads = _channel_heads(domain, process, config)
    head_rows, head_cols = np.nonzero(heads)
    retained_heads = _suppress_parallel_heads(
        domain, process, config, head_rows, head_cols
    )
    head_rows = head_rows[retained_heads]
    head_cols = head_cols[retained_heads]
    head_e, head_n = _indices_to_en(
        domain, head_rows.astype(np.float64), head_cols.astype(np.float64)
    )
    line_points: list[np.ndarray] = []
    line_radius: list[np.ndarray] = []
    line_depth: list[np.ndarray] = []
    line_ids: list[np.ndarray] = []
    headcut_points: list[np.ndarray] = []
    headcut_radius: list[np.ndarray] = []
    headcut_depth: list[np.ndarray] = []
    seep_points: list[np.ndarray] = []
    seep_radius: list[np.ndarray] = []
    seep_depth: list[np.ndarray] = []
    routing_active = domain.active
    ownership_bins: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    ownership_bin_m = config.rill_width_max_m

    for streamline_id, (start_e, start_n) in enumerate(
        zip(head_e, head_n, strict=True)
    ):
        points = _trace_streamline(
            domain,
            process,
            config,
            routing_active,
            float(start_e),
            float(start_n),
        )
        if len(points) < 3:
            continue
        east = points[:, 0]
        north = points[:, 1]
        area = _sample(
            domain, process.routing.contributing_area_m2, east, north, order=1
        )
        erosion = _sample(domain, process.eroded_depth_m, east, north, order=1)
        normalized_area = np.clip(
            np.log2(np.maximum(area, config.rill_area_threshold_m2) / config.rill_area_threshold_m2)
            / 5.0,
            0.0,
            1.0,
        )
        width = config.rill_width_min_m + (
            config.rill_width_max_m - config.rill_width_min_m
        ) * np.sqrt(normalized_area)
        radius = width * 0.5
        # A compact quartic cross-section integrates to approximately 0.533*w*d.
        depth = erosion * domain.texel_m / np.maximum(0.533 * width, 1e-9)
        depth = np.clip(depth, 0.0, config.rill_depth_max_m)
        useful = depth > 0.0
        if np.count_nonzero(useful) < 2:
            continue
        useful_indices = np.flatnonzero(useful)
        joined_at = _first_owned_point(
            points[useful],
            radius[useful],
            ownership_bins,
            bin_m=ownership_bin_m,
            maximum_existing_radius_m=0.5 * config.rill_width_max_m,
            integration_step_m=config.streamline_step_m,
        )
        if joined_at == 0:
            continue
        if joined_at is not None:
            stop = int(useful_indices[joined_at]) + 1
            points = points[:stop]
            east = east[:stop]
            north = north[:stop]
            area = area[:stop]
            erosion = erosion[:stop]
            normalized_area = normalized_area[:stop]
            width = width[:stop]
            radius = radius[:stop]
            depth = depth[:stop]
            useful = depth > 0.0
            if np.count_nonzero(useful) < 2:
                continue
        useful_points = points[useful]
        useful_radius = radius[useful]
        useful_depth = depth[useful]
        dense = _densify_polyline(
            useful_points,
            useful_radius,
            useful_depth,
            streamline_id=streamline_id,
        )
        line_points.append(dense[0])
        line_radius.append(dense[1])
        line_depth.append(dense[2])
        line_ids.append(dense[3])
        _add_owned_points(
            useful_points,
            useful_radius,
            ownership_bins,
            bin_m=ownership_bin_m,
        )

        path_slope = _sample(domain, process.routing.slope, east, north, order=1)
        slope_break = np.diff(path_slope)
        if slope_break.size and float(np.max(slope_break)) >= config.headcut_slope_break_min:
            break_index = int(np.argmax(slope_break)) + 1
            half = int(np.ceil(0.5 / config.streamline_step_m))
            start = max(0, break_index - half)
            stop = min(len(points), break_index + half + 1)
            head_depth = np.clip(
                depth[start:stop] * 1.45,
                0.0,
                config.headcut_relief_max_m,
            )
            if np.any(head_depth > 0.0):
                dense_head = _densify_polyline(
                    points[start:stop],
                    np.maximum(radius[start:stop] * 1.25, 0.14),
                    head_depth,
                    streamline_id=streamline_id,
                )
                headcut_points.append(dense_head[0])
                headcut_radius.append(dense_head[1])
                headcut_depth.append(dense_head[2])

        seep_likelihood = _sample(
            domain, domain.seep_likelihood, east, north, order=1
        )
        if float(np.max(seep_likelihood)) >= 0.55:
            center = int(np.argmax(seep_likelihood))
            half = int(np.ceil(1.25 / config.streamline_step_m))
            start = max(0, center - half)
            stop = min(len(points), center + half + 1)
            strength = np.clip(seep_likelihood[start:stop], 0.0, 1.0)
            seep_width = np.maximum(
                radius[start:stop] * (1.8 + strength), 0.35
            )
            seep_relief = np.minimum(
                depth[start:stop] * (0.65 + 0.5 * strength),
                config.seep_relief_max_m,
            )
            if np.any(seep_relief > 0.0):
                dense_seep = _densify_polyline(
                    points[start:stop],
                    seep_width,
                    seep_relief,
                    streamline_id=streamline_id,
                )
                seep_points.append(dense_seep[0])
                seep_radius.append(dense_seep[1])
                seep_depth.append(dense_seep[2])

    def concatenate(values: list[np.ndarray], columns: int | None = None) -> np.ndarray:
        if values:
            return np.concatenate(values, axis=0)
        shape = (0, columns) if columns is not None else (0,)
        return np.empty(shape, dtype=np.float64)

    rill_points = concatenate(line_points, 2)
    rill_radius_values = concatenate(line_radius)
    rill_depth_values = concatenate(line_depth)
    rill_id_values = (
        np.concatenate(line_ids) if line_ids else np.empty((0,), dtype=np.int32)
    )
    head_points = concatenate(headcut_points, 2)
    head_radius_values = concatenate(headcut_radius)
    head_depth_values = concatenate(headcut_depth)
    seep_point_values = concatenate(seep_points, 2)
    seep_radius_values = concatenate(seep_radius)
    seep_depth_values = concatenate(seep_depth)

    rows, cols = np.indices(domain.height_m.shape, dtype=np.float64)
    sample_e, sample_n = _indices_to_en(domain, rows, cols)
    rill_raw = _nearest_line_relief(
        rill_points,
        rill_radius_values,
        rill_depth_values,
        sample_e,
        sample_n,
    )
    head_raw = _nearest_line_relief(
        head_points,
        head_radius_values,
        head_depth_values,
        sample_e,
        sample_n,
    )
    seep_raw = _nearest_line_relief(
        seep_point_values,
        seep_radius_values,
        seep_depth_values,
        sample_e,
        sample_n,
    )
    negative_raw = np.maximum.reduce((rill_raw, head_raw, seep_raw))
    target_eroded_volume = float(np.sum(process.eroded_depth_m) * domain.texel_m**2)
    raw_volume = float(np.sum(negative_raw) * domain.texel_m**2)
    negative_scale = min(1.0, target_eroded_volume / raw_volume) if raw_volume > 0.0 else 0.0
    represented_erosion = raw_volume * negative_scale

    toe, represented_deposition, unrepresented_deposition = (
        _flow_aligned_toe_relief(domain, process, config)
    )
    return FormPlan(
        rill_points_en=rill_points,
        rill_streamline_id=rill_id_values,
        rill_radius_m=rill_radius_values,
        rill_depth_m=rill_depth_values,
        headcut_points_en=head_points,
        headcut_radius_m=head_radius_values,
        headcut_depth_m=head_depth_values,
        seep_points_en=seep_point_values,
        seep_radius_m=seep_radius_values,
        seep_depth_m=seep_depth_values,
        negative_scale=negative_scale,
        toe_process_relief_m=toe,
        represented_erosion_volume_m3=represented_erosion,
        unrepresented_erosion_volume_m3=max(
            0.0, target_eroded_volume - represented_erosion
        ),
        represented_deposition_volume_m3=represented_deposition,
        unrepresented_deposition_volume_m3=unrepresented_deposition,
        channel_heads=int(np.count_nonzero(heads)),
        retained_head_candidates=len(retained_heads),
        streamlines=len(line_points),
    )


def _fine_samples(
    bbox_en: tuple[float, float, float, float],
    texel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    e_min, n_min, e_max, n_max = bbox_en
    cols = int(round((e_max - e_min) / texel_m))
    rows = int(round((n_max - n_min) / texel_m))
    if cols <= 0 or rows <= 0:
        raise ValueError("fine crop has non-positive shape")
    east = e_min + np.arange(cols + 1, dtype=np.float64) * texel_m
    north = n_max - np.arange(rows + 1, dtype=np.float64) * texel_m
    return np.meshgrid(east, north)


def render_fine_surface(
    domain: SlopeDomain,
    process: ProcessResult,
    plan: FormPlan,
    config: ProcessConfig,
    *,
    bbox_en: tuple[float, float, float, float],
    row_block: int | None = None,
) -> FineSurface:
    e_min, n_min, e_max, n_max = bbox_en
    de0, dn0, de1, dn1 = domain.bbox_en
    if not (de0 <= e_min < e_max <= de1 and dn0 <= n_min < n_max <= dn1):
        raise ValueError("fine crop lies outside solved physical domain")
    east, north = _fine_samples(bbox_en, config.fine_texel_m)
    block = min(256, east.shape[0]) if row_block is None else row_block
    c0 = np.empty(east.shape, dtype=np.float64)
    residual = np.empty(east.shape, dtype=np.float64)
    incision = np.empty(east.shape, dtype=np.float64)
    deposition = np.empty(east.shape, dtype=np.float64)
    ownership = np.zeros(east.shape, dtype=np.uint8)
    hard = np.empty(east.shape, dtype=bool)
    material_supported = np.empty(east.shape, dtype=bool)
    for start in range(0, east.shape[0], block):
        stop = min(start + block, east.shape[0])
        ee = east[start:stop]
        nn = north[start:stop]
        base = _sample(domain, domain.height_m, ee, nn, order=3)
        rill = _nearest_line_relief(
            plan.rill_points_en,
            plan.rill_radius_m,
            plan.rill_depth_m,
            ee,
            nn,
        )
        headcut = _nearest_line_relief(
            plan.headcut_points_en,
            plan.headcut_radius_m,
            plan.headcut_depth_m,
            ee,
            nn,
        )
        seep = _nearest_line_relief(
            plan.seep_points_en,
            plan.seep_radius_m,
            plan.seep_depth_m,
            ee,
            nn,
        )
        negative_stack = np.stack((rill, headcut, seep), axis=0)
        negative = np.max(negative_stack, axis=0) * plan.negative_scale
        negative_owner = np.argmax(negative_stack, axis=0).astype(np.uint8) + FORM_RILL
        toe = np.clip(
            _sample(domain, plan.toe_process_relief_m, ee, nn, order=1),
            0.0,
            config.toe_relief_max_m,
        )
        support = _sample(
            domain,
            (domain.material_rule >= 0).astype(np.float32),
            ee,
            nn,
            order=0,
        ) >= 0.5
        exclusion = _sample(
            domain,
            (
                domain.hard_exclusion
                | domain.unknown
                | ~domain.solve_domain
            ).astype(np.float32),
            ee,
            nn,
            order=0,
        ) >= 0.5
        exclusion |= ~support
        value = toe - negative
        value[exclusion] = 0.0
        meaningful_toe = toe >= 0.0001
        toe_owns = meaningful_toe & (toe > negative)
        owner = negative_owner
        owner[toe_owns] = FORM_TOE
        owner[(negative <= 0.0) & ~toe_owns] = FORM_NONE
        owner[exclusion] = FORM_NONE
        c0[start:stop] = base
        residual[start:stop] = value
        incision[start:stop] = negative
        deposition[start:stop] = toe
        ownership[start:stop] = owner
        hard[start:stop] = exclusion
        material_supported[start:stop] = support
    return FineSurface(
        bbox_en=bbox_en,
        texel_m=config.fine_texel_m,
        c0_height_m=c0,
        c1_height_m=c0 + residual,
        residual_m=residual,
        incision_m=incision,
        deposition_m=deposition,
        ownership=ownership,
        hard_exclusion=hard,
        material_supported=material_supported,
    )

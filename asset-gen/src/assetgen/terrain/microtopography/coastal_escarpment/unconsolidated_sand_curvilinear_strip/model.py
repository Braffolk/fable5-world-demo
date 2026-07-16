"""Topology-coupled curvilinear strip and graph-seeded conservative processes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
import shapely

from .evidence import Evidence, MASTER_PITCH_M, SOLVE_PITCH_M, TARGET_BBOX_EN


CHAIN_NAMES = ("apron", "toe", "lower_bench_break", "upper_bench_break", "shoulder", "crest")


@dataclass(frozen=True)
class StripModel:
    station_m: np.ndarray
    chain_n_m: np.ndarray
    chain_height_m: np.ndarray
    valid_station: np.ndarray
    profile_n_m: np.ndarray
    measured_strip_m: np.ndarray
    confidence_strip: np.ndarray
    macro_strip_m: np.ndarray
    rill_strip: np.ndarray


@dataclass(frozen=True)
class Result:
    master_m: np.ndarray
    macro_master_m: np.ndarray
    micro_master_m: np.ndarray
    solve_absolute_m: np.ndarray
    solve_delta_m: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    strip_s_m: np.ndarray
    strip_n_m: np.ndarray
    strip_support: np.ndarray
    chain_distance_m: np.ndarray
    strip: StripModel
    source_band_p95_m: float


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray, pitch: float = 1.0) -> np.ndarray:
    return ndimage.map_coordinates(
        array,
        [(TARGET_BBOX_EN[3] - y) / pitch, (x - TARGET_BBOX_EN[0]) / pitch],
        order=1,
        mode="nearest",
    )


def _line_frame(line, station: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = shapely.line_interpolate_point(line, station)
    before = shapely.line_interpolate_point(line, np.maximum(station - 0.25, 0.0))
    after = shapely.line_interpolate_point(line, np.minimum(station + 0.25, float(line.length)))
    x, y = shapely.get_x(points), shapely.get_y(points)
    tx = shapely.get_x(after) - shapely.get_x(before)
    ty = shapely.get_y(after) - shapely.get_y(before)
    scale = np.maximum(np.hypot(tx, ty), 1.0e-9)
    tx, ty = tx / scale, ty / scale
    return np.asarray(x), np.asarray(y), np.asarray(tx), np.asarray(ty)


def _orient_normals(evidence: Evidence, station: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, tx, ty = _line_frame(evidence.etak_line, station)
    nx, ny = -ty, tx
    plus = _sample(evidence.c0_solve_m, x + 8.0 * nx, y + 8.0 * ny, SOLVE_PITCH_M)
    minus = _sample(evidence.c0_solve_m, x - 8.0 * nx, y - 8.0 * ny, SOLVE_PITCH_M)
    if float(np.median(plus - minus)) < 0.0:
        nx, ny = -nx, -ny
    return x, y, nx, ny


def _fill_short_gaps(value: np.ndarray, valid: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    result = value.copy()
    accepted = valid.copy()
    missing_labels, count = ndimage.label(~valid)
    for identity in range(1, count + 1):
        indices = np.flatnonzero(missing_labels == identity)
        if not len(indices) or len(indices) > maximum or indices[0] == 0 or indices[-1] == len(valid) - 1:
            continue
        left, right = indices[0] - 1, indices[-1] + 1
        result[indices] = np.interp(indices, [left, right], [result[left], result[right]])
        accepted[indices] = True
    return result, accepted


def _pchip_eval(query: np.ndarray, knots: np.ndarray, heights: np.ndarray) -> np.ndarray:
    width = np.maximum(np.diff(knots, axis=-1), 1.0e-4)
    slope = np.diff(heights, axis=-1) / width
    derivative = np.zeros_like(heights)
    derivative[..., 0] = slope[..., 0]
    derivative[..., -1] = slope[..., -1]
    left, right = slope[..., :-1], slope[..., 1:]
    same = left * right > 0.0
    harmonic = 2.0 * left * right / np.maximum(np.abs(left + right), 1.0e-9)
    derivative[..., 1:-1] = np.where(same, harmonic, 0.0)
    result = heights[..., 0].copy()
    for index in range(knots.shape[-1] - 1):
        selected = (query >= knots[..., index]) & (query <= knots[..., index + 1])
        t = np.clip((query - knots[..., index]) / width[..., index], 0.0, 1.0)
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        value = (
            h00 * heights[..., index]
            + h10 * width[..., index] * derivative[..., index]
            + h01 * heights[..., index + 1]
            + h11 * width[..., index] * derivative[..., index + 1]
        )
        result[selected] = value[selected]
    result = np.where(query < knots[..., 0], heights[..., 0], result)
    return np.where(query > knots[..., -1], heights[..., -1], result)


def fit_strip(evidence: Evidence, config: dict) -> StripModel:
    station_pitch = float(config["station_pitch_m"])
    station = np.arange(0.0, float(evidence.etak_line.length) + 0.25 * station_pitch, station_pitch)
    profile_n = np.arange(float(config["profile_min_n_m"]), float(config["profile_max_n_m"]) + 0.01, float(config["profile_pitch_m"]))
    x, y, nx, ny = _orient_normals(evidence, station)
    px = x[:, None] + nx[:, None] * profile_n[None]
    py = y[:, None] + ny[:, None] * profile_n[None]
    c0_reference = evidence.c0_solve_m[::4, ::4]
    measured_reference = np.where(np.isfinite(evidence.reference_m), evidence.reference_m, c0_reference)
    measured = _sample(measured_reference, px, py)
    confidence = _sample(evidence.reference_confidence, px, py)
    measured = ndimage.gaussian_filter1d(measured, float(config["normal_smoothing_m"]) / float(config["profile_pitch_m"]), axis=1)
    derivative = np.gradient(measured, float(config["profile_pitch_m"]), axis=1)
    chain_n = np.full((len(station), len(CHAIN_NAMES)), np.nan, dtype=np.float64)
    chain_h = np.full_like(chain_n, np.nan)
    valid = np.zeros(len(station), dtype=bool)
    zero_index = int(np.argmin(np.abs(profile_n)))

    for row in range(len(station)):
        candidate = (
            (derivative[row] >= float(config["minimum_face_slope"]))
            & (confidence[row] >= float(config["minimum_reference_confidence"]))
        )
        # A real bench is a low-slope interval inside one face, not a topology
        # break. Close only gaps shorter than the admitted bench width before
        # selecting the component that crosses the mapped escarpment chain.
        close_cells = int(round(float(config["maximum_internal_bench_gap_m"]) / float(config["profile_pitch_m"])))
        candidate = ndimage.binary_closing(candidate, structure=np.ones(close_cells + 1, dtype=bool))
        labels, count = ndimage.label(candidate)
        if count == 0:
            continue
        identities = range(1, count + 1)
        # ETAK is a topology scaffold rather than an asserted toe or crest.
        # Select the supported component with the greatest measured rise; using
        # the component nearest n=0 would jump to incidental micro-breaks where
        # the mapped line wanders within the face.
        def component_rise(identity: int) -> float:
            cells = np.flatnonzero(labels == identity)
            return float(measured[row, cells[-1]] - measured[row, cells[0]])

        identity = max(identities, key=component_rise)
        indices = np.flatnonzero(labels == identity)
        toe_n, shoulder_n = float(profile_n[indices[0]]), float(profile_n[indices[-1]])
        width = shoulder_n - toe_n
        rise = float(measured[row, indices[-1]] - measured[row, indices[0]])
        support_fraction = float(np.mean(confidence[row, indices] >= float(config["minimum_reference_confidence"])))
        if not (
            float(config["minimum_face_width_m"]) <= width <= float(config["maximum_face_width_m"])
            and rise >= float(config["minimum_face_rise_m"])
            and support_fraction >= float(config["minimum_face_support_fraction"])
        ):
            continue
        face_height = measured[row, indices]
        cumulative = (face_height - face_height[0]) / max(float(face_height[-1] - face_height[0]), 1.0e-6)
        lower_base = int(indices[np.argmin(np.abs(cumulative - 0.28))])
        upper_base = int(indices[np.argmin(np.abs(cumulative - 0.72))])
        central = np.arange(lower_base, upper_base + 1)
        bench_center = int(central[np.argmin(derivative[row, central])])
        half_bench = max(0.5, min(1.5, 0.12 * width))
        lower_n = max(toe_n + 1.0, float(profile_n[bench_center]) - half_bench)
        upper_n = min(shoulder_n - 1.0, float(profile_n[bench_center]) + half_bench)
        if upper_n - lower_n < 0.75:
            lower_n = float(profile_n[lower_base])
            upper_n = float(profile_n[upper_base])
        knots = np.asarray([toe_n - 2.5, toe_n, lower_n, upper_n, shoulder_n, shoulder_n + 3.0])
        heights = np.interp(knots, profile_n, measured[row])
        # Outer knots are C0 anchors; the strip owns no terrain outside them.
        heights[0] = float(_sample(c0_reference, np.asarray([x[row] + nx[row] * knots[0]]), np.asarray([y[row] + ny[row] * knots[0]]))[0])
        heights[-1] = float(_sample(c0_reference, np.asarray([x[row] + nx[row] * knots[-1]]), np.asarray([y[row] + ny[row] * knots[-1]]))[0])
        chain_n[row] = knots
        chain_h[row] = heights
        valid[row] = True

    minimum_fraction = float(config["minimum_paired_station_fraction"])
    if float(np.mean(valid)) < minimum_fraction:
        raise ValueError(f"paired chain coverage {np.mean(valid):.3f} is below {minimum_fraction:.3f}")
    maximum_gap = int(round(float(config["maximum_graph_gap_m"]) / station_pitch))
    shared_valid = valid.copy()
    for index in range(chain_n.shape[1]):
        chain_n[:, index], accepted_n = _fill_short_gaps(chain_n[:, index], valid, maximum_gap)
        chain_h[:, index], accepted_h = _fill_short_gaps(chain_h[:, index], valid, maximum_gap)
        shared_valid &= accepted_n & accepted_h
    chain_n = np.where(np.isfinite(chain_n), chain_n, 0.0)
    chain_h = np.where(np.isfinite(chain_h), chain_h, 0.0)
    sigma = float(config["along_strip_smoothing_m"]) / station_pitch
    for index in range(chain_n.shape[1]):
        normalized = ndimage.gaussian_filter1d(shared_valid.astype(np.float64), sigma, mode="nearest")
        chain_n[:, index] = ndimage.gaussian_filter1d(chain_n[:, index] * shared_valid, sigma, mode="nearest") / np.maximum(normalized, 1.0e-6)
        chain_h[:, index] = ndimage.gaussian_filter1d(chain_h[:, index] * shared_valid, sigma, mode="nearest") / np.maximum(normalized, 1.0e-6)
    # Reassert topology after joint along-strip smoothing.
    minimum_separation = np.asarray([1.5, 0.9, 0.8, 0.9, 1.5])
    for index in range(1, chain_n.shape[1]):
        chain_n[:, index] = np.maximum(chain_n[:, index], chain_n[:, index - 1] + minimum_separation[index - 1])
    macro_strip = _pchip_eval(
        np.broadcast_to(profile_n[None], measured.shape),
        np.broadcast_to(chain_n[:, None, :], (*measured.shape, chain_n.shape[1])),
        np.broadcast_to(chain_h[:, None, :], (*measured.shape, chain_h.shape[1])),
    )

    residual = measured - macro_strip
    residual = ndimage.gaussian_filter1d(residual, 0.45 / float(config["profile_pitch_m"]), axis=1)
    along = ndimage.gaussian_filter1d(residual, float(config["rill_along_scale_m"]) / station_pitch, axis=0)
    valley = np.maximum(along - residual, 0.0)
    face = (profile_n[None] >= chain_n[:, 1, None]) & (profile_n[None] <= chain_n[:, 4, None])
    valley *= face * (confidence >= float(config["minimum_reference_confidence"])) * shared_valid[:, None]
    positive = valley[valley > 0.0]
    threshold = float(np.percentile(positive, float(config["rill_strength_percentile"]))) if len(positive) else np.inf
    rill = np.where(valley >= threshold, valley, 0.0)
    labels, count = ndimage.label(rill > 0.0, structure=np.ones((3, 3), dtype=np.uint8))
    retained = np.zeros_like(rill, dtype=bool)
    minimum_span = float(config["minimum_rill_normal_span_m"])
    for identity in range(1, count + 1):
        rows, cols = np.nonzero(labels == identity)
        if len(rows) and (float(profile_n[cols].max() - profile_n[cols].min()) >= minimum_span):
            retained[rows, cols] = True
    rill = ndimage.gaussian_filter(rill * retained, sigma=(0.75 / station_pitch, 0.35 / float(config["profile_pitch_m"])))
    if float(np.max(rill)) <= 0.0:
        raise ValueError("measured strip contains no connected rill spanning the face")
    rill /= float(np.max(rill))
    return StripModel(station, chain_n, chain_h, shared_valid, profile_n, measured, confidence, macro_strip, rill)


def _xy_strip_coordinates(evidence: Evidence, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(round(128.0 / pitch)) + 1
    x = TARGET_BBOX_EN[0] + np.arange(count) * pitch
    y = TARGET_BBOX_EN[3] - np.arange(count) * pitch
    xx, yy = np.meshgrid(x, y)
    points = shapely.points(xx.ravel(), yy.ravel())
    station = np.asarray(shapely.line_locate_point(evidence.etak_line, points)).reshape(xx.shape)
    projected = shapely.line_interpolate_point(evidence.etak_line, station.ravel())
    px = np.asarray(shapely.get_x(projected)).reshape(xx.shape)
    py = np.asarray(shapely.get_y(projected)).reshape(xx.shape)
    _, _, tx, ty = _line_frame(evidence.etak_line, station.ravel())
    tx, ty = tx.reshape(xx.shape), ty.reshape(xx.shape)
    nx, ny = -ty, tx
    plus = _sample(evidence.c0_solve_m, px + 8.0 * nx, py + 8.0 * ny, SOLVE_PITCH_M)
    minus = _sample(evidence.c0_solve_m, px - 8.0 * nx, py - 8.0 * ny, SOLVE_PITCH_M)
    if float(np.median(plus - minus)) < 0.0:
        nx, ny = -nx, -ny
    normal = (xx - px) * nx + (yy - py) * ny
    start_x, start_y, start_tx, start_ty = _line_frame(evidence.etak_line, np.asarray([0.0]))
    end_x, end_y, end_tx, end_ty = _line_frame(evidence.etak_line, np.asarray([float(evidence.etak_line.length)]))
    before = (xx - start_x[0]) * start_tx[0] + (yy - start_y[0]) * start_ty[0] < 0.0
    after = (xx - end_x[0]) * end_tx[0] + (yy - end_y[0]) * end_ty[0] > 0.0
    topology = ~(before | after)
    return station, normal, topology


def _interpolate_chains(strip: StripModel, station: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = station.ravel()
    n = np.stack([np.interp(flat, strip.station_m, strip.chain_n_m[:, index]) for index in range(len(CHAIN_NAMES))], axis=-1)
    h = np.stack([np.interp(flat, strip.station_m, strip.chain_height_m[:, index]) for index in range(len(CHAIN_NAMES))], axis=-1)
    valid = np.interp(flat, strip.station_m, strip.valid_station.astype(np.float64))
    return n.reshape((*station.shape, len(CHAIN_NAMES))), h.reshape((*station.shape, len(CHAIN_NAMES))), valid.reshape(station.shape)


def synthesize(evidence: Evidence, config: dict) -> Result:
    strip = fit_strip(evidence, config["strip"])
    station, normal, topology = _xy_strip_coordinates(evidence, SOLVE_PITCH_M)
    knots, heights, valid = _interpolate_chains(strip, station)
    target = _pchip_eval(normal, knots, heights)
    correction = np.clip(target - evidence.c0_solve_m, -float(config["strip"]["maximum_shift_m"]), float(config["strip"]["maximum_shift_m"]))
    cross_taper = _smoothstep((normal - knots[..., 0]) / 1.25) * _smoothstep((knots[..., -1] - normal) / 1.25)
    endpoint = _smoothstep(np.minimum(station, float(evidence.etak_line.length) - station) / float(config["strip"]["endpoint_taper_m"]))
    support = cross_taper * endpoint * _smoothstep((valid - 0.5) / 0.45) * topology
    correction *= support
    correction[evidence.hard_solve] = 0.0

    profile_pitch = float(config["strip"]["profile_pitch_m"])
    station_pitch = float(config["strip"]["station_pitch_m"])
    rill = ndimage.map_coordinates(
        strip.rill_strip,
        [station / station_pitch, (normal - strip.profile_n_m[0]) / profile_pitch],
        order=1,
        mode="constant",
        cval=0.0,
    )
    rill *= support * (~evidence.hard_solve)
    source_fine = ndimage.zoom(evidence.source_fine_m, 2.0, order=3)
    source_valid = ndimage.zoom(evidence.source_support.astype(np.uint8), 2.0, order=0) > 0
    source_band = ndimage.gaussian_filter(source_fine, 0.45) - ndimage.gaussian_filter(source_fine, 5.0)
    source_p95 = float(np.percentile(np.abs(source_band[source_valid]), 95.0))
    erosion_depth = min(float(config["process"]["maximum_erosion_depth_m"]), source_p95 * float(config["process"]["biala_p95_gain"]))
    erosion = erosion_depth * rill**1.25

    toe_center = knots[..., 1] - float(config["process"]["apron_offset_m"])
    toe_width = float(config["process"]["apron_width_m"])
    arrival = ndimage.gaussian_filter(rill * np.exp(-((normal - knots[..., 1]) / 1.0) ** 2), 1.25 / SOLVE_PITCH_M)
    apron = arrival * np.exp(-((normal - toe_center) / toe_width) ** 2) * support * (~evidence.hard_solve)
    if float(np.sum(erosion)) <= 0.0 or float(np.sum(apron)) <= 0.0:
        raise ValueError("rill graph does not reach a conservative toe apron")
    deposition = apron * (float(np.sum(erosion)) / float(np.sum(apron)))
    process = correction - erosion + deposition
    process[evidence.hard_solve] = 0.0

    macro_master = ndimage.zoom(process, 4.0, order=3, mode="nearest")[:2049, :2049]
    feature_master = ndimage.zoom(rill + 0.35 * support, 4.0, order=3, mode="nearest")[:2049, :2049]
    continuation = ndimage.gaussian_filter(feature_master, 0.8) - ndimage.gaussian_filter(feature_master, 4.0)
    valid_master = ~evidence.hard_master
    ceiling = min(float(config["micro"]["absolute_p95_ceiling_m"]), source_p95 * float(config["micro"]["biala_p95_fraction"]))
    micro_p95 = float(np.percentile(np.abs(continuation[valid_master]), 95.0))
    micro = continuation * (ceiling / max(micro_p95, 1.0e-9))
    distance = ndimage.distance_transform_edt(valid_master) * MASTER_PITCH_M
    micro *= _smoothstep(distance / float(config["strip"]["outer_collar_m"]))
    macro_master[evidence.hard_master] = 0.0
    micro[evidence.hard_master] = 0.0
    master = evidence.c0_master_m + macro_master + micro
    master[evidence.hard_master] = evidence.c0_master_m[evidence.hard_master]
    chain_distance = np.min(np.abs(normal[..., None] - knots), axis=-1)
    return Result(
        master_m=master,
        macro_master_m=macro_master,
        micro_master_m=micro,
        solve_absolute_m=evidence.c0_solve_m + process,
        solve_delta_m=process,
        erosion_m=erosion,
        deposition_m=deposition,
        strip_s_m=station,
        strip_n_m=normal,
        strip_support=support,
        chain_distance_m=chain_distance,
        strip=strip,
        source_band_p95_m=source_p95,
    )

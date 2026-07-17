"""Carrier load, typed-form relief carving, parent-deviation, crop-last output.

The raw 1 m ALS carrier is prolonged to 0.25 m (Keys, parent-mean preserving). Calibrated
low-relief typed forms are carved from the resampled steady-state activator: strings rise,
hollows fall, pools and open water stay relief-free, lawn stays flat. Masks are exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from scipy import ndimage

from ....repair.prolong import prolong_structural_4x
from . import network


@dataclass(frozen=True)
class CarveResult:
    carrier_1m: np.ndarray
    fine_c1_025m: np.ndarray
    fine_c0_025m: np.ndarray
    relief_025m: np.ndarray
    deviation_1m: np.ndarray
    labels_1m: np.ndarray
    activator_norm_1m: np.ndarray
    bbox_en: tuple[int, int, int, int]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _resample_process_to_output(
    field: np.ndarray,
    process_bbox: tuple[int, int, int, int],
    process_pitch: float,
    out_bbox: tuple[int, int, int, int],
    out_pitch: float,
    *,
    order: int,
) -> np.ndarray:
    out_h = int(round((out_bbox[3] - out_bbox[1]) / out_pitch))
    out_w = int(round((out_bbox[2] - out_bbox[0]) / out_pitch))
    east = out_bbox[0] + (np.arange(out_w) + 0.5) * out_pitch
    north = out_bbox[3] - (np.arange(out_h) + 0.5) * out_pitch
    src_col = (east[None, :] - process_bbox[0]) / process_pitch - 0.5
    src_row = (process_bbox[3] - north[:, None]) / process_pitch - 0.5
    src_col = np.broadcast_to(src_col, (out_h, out_w))
    src_row = np.broadcast_to(src_row, (out_h, out_w))
    return ndimage.map_coordinates(
        field.astype(np.float64),
        np.stack((src_row.ravel(), src_col.ravel())),
        order=order,
        mode="nearest",
    ).reshape(out_h, out_w)


def read_carrier(dtm_path: Path, halo_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Read the raw 1 m ALS carrier over the halo plus a two-cell prolong support."""
    support = (halo_bbox[0] - 2, halo_bbox[1] - 2, halo_bbox[2] + 2, halo_bbox[3] + 2)
    with rasterio.open(dtm_path) as source:
        if (
            str(source.crs) != "EPSG:3301"
            or source.transform.a != 1.0
            or source.transform.e != -1.0
            or source.nodata is None
        ):
            raise ValueError("carrier DTM grid contract changed")
        window = rasterio.windows.from_bounds(*support, source.transform)
        height = source.read(1, window=window, out_dtype="float64")
        invalid = ~np.isfinite(height) | (height == float(source.nodata))
        if invalid.any():
            raise ValueError("carrier DTM support contains invalid samples")
    expected = (support[3] - support[1], support[2] - support[0])
    if height.shape != expected:
        raise ValueError(f"carrier support shape changed: {height.shape}")
    return height


def carve(
    dtm_path: Path,
    halo_bbox: tuple[int, int, int, int],
    activator: np.ndarray,
    inhibitor: np.ndarray,
    process_bbox: tuple[int, int, int, int],
    process_pitch: float,
    norm_lo: float,
    norm_hi: float,
    labels_cuts: dict,
    halo_masks,
    amplitudes: dict,
    output_pitch: float,
) -> CarveResult:
    support = read_carrier(dtm_path, halo_bbox)
    rows = support.shape[0]
    cols = support.shape[1]
    carrier_1m = support[2:-2, 2:-2]
    fine_c0 = prolong_structural_4x(support, parent_rows=(2, rows - 2), parent_cols=(2, cols - 2))

    # Resample the steady-state fields to the output 0.25 m grid.
    v_out = _resample_process_to_output(
        activator, process_bbox, process_pitch, halo_bbox, output_pitch, order=1
    )
    u_out = _resample_process_to_output(
        inhibitor, process_bbox, process_pitch, halo_bbox, output_pitch, order=1
    )
    v_norm = np.clip((v_out - norm_lo) / max(norm_hi - norm_lo, 1.0e-9), 0.0, 1.0)

    # Output-resolution class labels from the frozen absolute cut values.
    ridge = v_out >= labels_cuts["ridge_cut"]
    non_ridge = ~ridge
    labels = np.zeros(v_out.shape, dtype=np.uint8)
    labels[non_ridge] = network.LAWN
    labels[ridge] = network.RIDGE
    labels[non_ridge & (u_out >= labels_cuts["hollow_cut"])] = network.HOLLOW
    labels[non_ridge & (u_out >= labels_cuts["pool_cut"])] = network.POOL

    # Relief from the activator continuum: strings up, hollows down, flat elsewhere.
    ridge_amp = float(amplitudes["ridge_peak"])
    hollow_amp = float(amplitudes["hollow_floor"])
    up = ridge_amp * _smoothstep((v_norm - 0.5) / 0.5)
    down = -hollow_amp * _smoothstep((0.5 - v_norm) / 0.5)
    relief = np.where(v_norm >= 0.5, up, down)

    # Exact masks: relief only inside authority; pools and open water relief-free.
    def to_output(mask: np.ndarray) -> np.ndarray:
        repeat = int(round(halo_masks.pitch_m / output_pitch))
        return np.repeat(np.repeat(mask, repeat, axis=0), repeat, axis=1)

    authority = to_output(halo_masks.authority)
    open_water = to_output(halo_masks.open_water)
    relief = np.where(authority, relief, 0.0)
    relief[open_water] = 0.0
    relief[labels == network.POOL] = 0.0

    fine_c1 = fine_c0 + relief

    # Parent deviation: block-mean relief per 1 m cell (C1_1m - carrier_1m).
    block = relief.reshape(carrier_1m.shape[0], 4, carrier_1m.shape[1], 4)
    deviation_1m = np.mean(block, axis=(1, 3), dtype=np.float64)

    # 1 m label / normalized activator for QA and gates.
    labels_1m = _resample_process_to_output(
        activator, process_bbox, process_pitch, halo_bbox, halo_masks.pitch_m, order=1
    )
    v_norm_1m = np.clip((labels_1m - norm_lo) / max(norm_hi - norm_lo, 1.0e-9), 0.0, 1.0)

    return CarveResult(
        carrier_1m=carrier_1m,
        fine_c1_025m=fine_c1,
        fine_c0_025m=fine_c0,
        relief_025m=relief,
        deviation_1m=deviation_1m,
        labels_1m=labels,
        activator_norm_1m=v_norm_1m,
        bbox_en=halo_bbox,
    )


def crop(field: np.ndarray, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], pitch: float) -> np.ndarray:
    r0 = int(round((outer[3] - inner[3]) / pitch))
    r1 = int(round((outer[3] - inner[1]) / pitch))
    c0 = int(round((inner[0] - outer[0]) / pitch))
    c1 = int(round((inner[2] - outer[0]) / pitch))
    return field[r0:r1, c0:c1]

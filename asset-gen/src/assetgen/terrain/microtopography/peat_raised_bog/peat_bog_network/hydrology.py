"""Whole-mire dome gradient, routing, downslope flow direction and slope.

Computed from the national 10 m DTM over the connected mire + context, then resampled
onto the 2 m process grid. This is condition context only (not a corrected C0). Rows run
north-to-south; columns west-to-east.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows
from scipy import ndimage

CONTEXT_M = 100.0
DOME_SMOOTH_M = 30.0


@dataclass(frozen=True)
class ProcessHydrology:
    """Downslope advection field on the 2 m process grid (rows north-to-south)."""

    flow_row: np.ndarray  # unit downslope component along +row (south)
    flow_col: np.ndarray  # unit downslope component along +col (east)
    slope_mag: np.ndarray  # |grad dome| (m/m) of the smoothed dome
    dome_m: np.ndarray  # smoothed 10 m dome resampled to the process grid
    bbox_en: tuple[int, int, int, int]
    process_pitch_m: float


def _priority_fill(height: np.ndarray) -> np.ndarray:
    rows, cols = height.shape
    filled = np.asarray(height, dtype=np.float64).copy()
    visited = np.zeros(height.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in (0, cols - 1):
            if not visited[row, col]:
                visited[row, col] = True
                heapq.heappush(heap, (float(filled[row, col]), row, col))
    for col in range(cols):
        for row in (0, rows - 1):
            if not visited[row, col]:
                visited[row, col] = True
                heapq.heappush(heap, (float(filled[row, col]), row, col))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while heap:
        level, row, col = heapq.heappop(heap)
        for drow, dcol in neighbors:
            rr, cc = row + drow, col + dcol
            if not (0 <= rr < rows and 0 <= cc < cols) or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            filled[rr, cc] = max(level, float(filled[rr, cc]))
            heapq.heappush(heap, (float(filled[rr, cc]), rr, cc))
    return filled


def _resample_to_process(
    field: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    src_pitch: float,
    dst_bbox: tuple[int, int, int, int],
    dst_pitch: float,
    *,
    order: int,
) -> np.ndarray:
    dst_h = int(round((dst_bbox[3] - dst_bbox[1]) / dst_pitch))
    dst_w = int(round((dst_bbox[2] - dst_bbox[0]) / dst_pitch))
    east = dst_bbox[0] + (np.arange(dst_w) + 0.5) * dst_pitch
    north = dst_bbox[3] - (np.arange(dst_h) + 0.5) * dst_pitch
    src_col = (east[None, :] - src_bbox[0]) / src_pitch - 0.5
    src_row = (src_bbox[3] - north[:, None]) / src_pitch - 0.5
    src_col = np.broadcast_to(src_col, (dst_h, dst_w))
    src_row = np.broadcast_to(src_row, (dst_h, dst_w))
    return ndimage.map_coordinates(
        field.astype(np.float64),
        np.stack((src_row.ravel(), src_col.ravel())),
        order=order,
        mode="nearest",
    ).reshape(dst_h, dst_w)


def compute(
    country_dtm: Path,
    whole_mire_bbox_en: tuple[float, float, float, float],
    process_bbox_en: tuple[int, int, int, int],
    process_pitch_m: float,
) -> ProcessHydrology:
    """Return the downslope advection field on the process grid.

    ``process_bbox_en`` is the integer solve bbox (whole mire + halo, process-aligned).
    """
    src_pitch = 10.0
    e0 = int(np.floor((whole_mire_bbox_en[0] - CONTEXT_M) / src_pitch) * src_pitch)
    n0 = int(np.floor((whole_mire_bbox_en[1] - CONTEXT_M) / src_pitch) * src_pitch)
    e1 = int(np.ceil((whole_mire_bbox_en[2] + CONTEXT_M) / src_pitch) * src_pitch)
    n1 = int(np.ceil((whole_mire_bbox_en[3] + CONTEXT_M) / src_pitch) * src_pitch)
    src_bbox = (e0, n0, e1, n1)
    with rasterio.open(country_dtm) as source:
        if (
            str(source.crs) != "EPSG:3301"
            or source.transform.a != src_pitch
            or source.transform.e != -src_pitch
        ):
            raise ValueError("country DTM context grid contract changed")
        window = rasterio.windows.from_bounds(*src_bbox, source.transform)
        height = source.read(1, window=window, out_dtype="float64")
        invalid = ~np.isfinite(height) | (height == float(source.nodata))
        if invalid.any():
            raise ValueError("country DTM window contains invalid cells")
    filled = _priority_fill(height)
    dome = ndimage.gaussian_filter(filled, DOME_SMOOTH_M / src_pitch, mode="nearest")
    grad_row, grad_col = np.gradient(dome, src_pitch)
    slope = np.hypot(grad_row, grad_col)
    safe = np.maximum(slope, 1.0e-9)
    down_row = -grad_row / safe  # downslope +row (south) component
    down_col = -grad_col / safe  # downslope +col (east) component

    kwargs = dict(
        src_bbox=src_bbox,
        src_pitch=src_pitch,
        dst_bbox=process_bbox_en,
        dst_pitch=process_pitch_m,
    )
    proc_row = _resample_to_process(down_row, **kwargs, order=1)
    proc_col = _resample_to_process(down_col, **kwargs, order=1)
    proc_slope = _resample_to_process(slope, **kwargs, order=1)
    proc_dome = _resample_to_process(dome, **kwargs, order=1)
    norm = np.maximum(np.hypot(proc_row, proc_col), 1.0e-9)
    return ProcessHydrology(
        flow_row=proc_row / norm,
        flow_col=proc_col / norm,
        slope_mag=proc_slope,
        dome_m=proc_dome,
        bbox_en=process_bbox_en,
        process_pitch_m=process_pitch_m,
    )

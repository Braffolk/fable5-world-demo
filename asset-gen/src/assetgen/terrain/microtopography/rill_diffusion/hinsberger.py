"""Fail-closed Hinsberger DEM ingest and four-phase height-band qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
import zipfile
import zlib

import numpy as np
import rasterio
from rasterio.windows import Window

from ....config import DATA_WORK
from .bands import (
    CommonPhaseIntegral,
    GridIntegral,
    PhaseBands,
    analyze_phase,
    exact_overlap_resample,
    integrate_common_phases,
    integrate_grid,
)
from .contract import (
    ANALYSIS_CORE_M,
    ANALYSIS_HALO_M,
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_MEMBERS,
    ARCHIVE_NAME,
    DATASET_DOI,
    LICENSE_SPDX,
    LICENSE_URL,
    PUBLISHER_FILE_ID,
    SELECTED_SURVEYS,
    SurveyPair,
    member_by_path,
)
from .qa import qa_index, render_height_band_qa, write_json


_SCHEMA = "hinsberger-rill-height-qualification/1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _hash_file(path: Path, *, include_md5: bool = False, include_crc32: bool = False) -> dict[str, Any]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5() if include_md5 else None  # noqa: S324 - publisher identity requires MD5.
    crc = 0
    size = 0
    with path.open("rb") as source:
        while block := source.read(16 << 20):
            sha256.update(block)
            if md5 is not None:
                md5.update(block)
            if include_crc32:
                crc = zlib.crc32(block, crc)
            size += len(block)
    result: dict[str, Any] = {"bytes": size, "sha256": sha256.hexdigest()}
    if md5 is not None:
        result["md5"] = md5.hexdigest()
    if include_crc32:
        result["crc32"] = f"{crc & 0xFFFFFFFF:08x}"
    return result


def _verify_archive(archive_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not archive_path.exists():
        raise FileNotFoundError(
            f"complete Hinsberger archive is absent: {archive_path}; partial files and range segments are not accepted"
        )
    if archive_path.name != ARCHIVE_NAME:
        raise ValueError(f"expected raw archive name {ARCHIVE_NAME!r}, got {archive_path.name!r}")
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(
            f"Hinsberger archive is incomplete: {archive_path.stat().st_size:,} bytes; expected {ARCHIVE_BYTES:,}"
        )
    identity = _hash_file(archive_path, include_md5=True)
    if identity["md5"] != ARCHIVE_MD5:
        raise ValueError(
            f"Hinsberger archive MD5 mismatch: {identity['md5']}; expected publisher MD5 {ARCHIVE_MD5}"
        )

    expected = {member.path: member for member in ARCHIVE_MEMBERS}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
    if len(infos) != len(ARCHIVE_MEMBERS) or len({info.filename for info in infos}) != len(infos):
        raise ValueError("Hinsberger ZIP must contain exactly 28 unique members")
    actual_names = {info.filename for info in infos}
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise ValueError(f"Hinsberger ZIP inventory changed; missing={missing}, extra={extra}")
    inventory = []
    for info in infos:
        contract = expected[info.filename]
        actual = (
            info.compress_size,
            info.file_size,
            info.CRC,
            info.is_dir(),
            info.compress_type,
            info.flag_bits,
        )
        frozen = (
            contract.compressed_bytes,
            contract.uncompressed_bytes,
            contract.crc32,
            contract.is_dir,
            zipfile.ZIP_STORED if contract.is_dir else zipfile.ZIP_DEFLATED,
            0,
        )
        if actual != frozen:
            raise ValueError(f"Hinsberger ZIP member metadata changed: {info.filename}")
        inventory.append(
            {
                "path": info.filename,
                "is_directory": info.is_dir(),
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "compression": info.compress_type,
            }
        )
    survey_directories = [item for item in inventory if item["is_directory"] and item["path"] != "Hinsberger_aerial_survey_data/"]
    files = [item for item in inventory if not item["is_directory"]]
    if len(survey_directories) != 9 or len(files) != 18:
        raise AssertionError("frozen Hinsberger inventory lost its nine paired surveys")
    for directory in survey_directories:
        children = [item["path"] for item in files if item["path"].startswith(directory["path"])]
        if len(children) != 2 or sum(path.lower().endswith("_dem.tif") for path in children) != 1 or sum(
            path.lower().endswith("_orthomosaic.tif") for path in children
        ) != 1:
            raise ValueError(f"survey directory is not a sole DEM/orthomosaic pair: {directory['path']}")
    return identity, inventory


def _verify_extracted_members(extracted_root: Path) -> dict[str, dict[str, Any]]:
    expected = member_by_path()
    selected_paths = [
        member
        for survey in SELECTED_SURVEYS
        for member in (survey.dem_member, survey.orthomosaic_member)
    ]
    missing = [member for member in selected_paths if not (extracted_root / member).is_file()]
    if missing:
        raise FileNotFoundError(
            "selected Hinsberger TIFFs are not fully extracted beneath "
            f"{extracted_root}: {missing}; the qualifier does not read partial files or extract implicitly"
        )
    identities = {}
    for member in selected_paths:
        path = (extracted_root / member).resolve()
        if not path.is_relative_to(extracted_root.resolve()):
            raise ValueError(f"unsafe extracted member path: {member}")
        contract = expected[member]
        if path.stat().st_size != contract.uncompressed_bytes:
            raise ValueError(f"extracted Hinsberger member has wrong size: {member}")
        identity = _hash_file(path, include_crc32=True)
        if identity["crc32"] != f"{contract.crc32:08x}":
            raise ValueError(f"extracted Hinsberger member CRC mismatch: {member}")
        identities[member] = {"path": member, **identity}
    return identities


def _implementation_identity() -> list[dict[str, Any]]:
    return [
        {"path": path.name, **_hash_file(path)}
        for path in sorted(Path(__file__).parent.glob("*.py"))
    ]


def _crs_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    crs = dataset.crs
    if crs is None:
        return {"present": False}
    try:
        linear_units = crs.linear_units
    except Exception:
        linear_units = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    return {
        "present": True,
        "epsg": epsg,
        "projected": bool(crs.is_projected),
        "geographic": bool(crs.is_geographic),
        "linear_units": linear_units,
        "wkt": crs.to_wkt(),
    }


def _dataset_metadata(dataset: rasterio.io.DatasetReader, *, pixels_read: bool) -> dict[str, Any]:
    transform = dataset.transform
    return {
        "driver": dataset.driver,
        "shape": [dataset.height, dataset.width],
        "bands": dataset.count,
        "dtypes": list(dataset.dtypes),
        "nodata": list(dataset.nodatavals),
        "scales": [float(value) for value in dataset.scales],
        "offsets": [float(value) for value in dataset.offsets],
        "units": list(dataset.units),
        "crs": _crs_metadata(dataset),
        "affine": [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],
        "bounds": [float(value) for value in dataset.bounds],
        "gsd_xy_native": [
            float(math.hypot(transform.a, transform.d)),
            float(math.hypot(transform.b, transform.e)),
        ],
        "axis_aligned_north_up": bool(
            transform.a > 0.0
            and transform.e < 0.0
            and abs(transform.b) <= 1e-12
            and abs(transform.d) <= 1e-12
        ),
        "block_shapes": [list(shape) for shape in dataset.block_shapes],
        "pixel_values_read": pixels_read,
    }


def _finite_support(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    finite_valid = 0
    mask_valid_nonfinite = 0
    masked = 0
    minimum: float | None = None
    maximum: float | None = None
    for _, window in dataset.block_windows(1):
        values = dataset.read(1, window=window, masked=False)
        declared = dataset.read_masks(1, window=window) > 0
        finite = np.isfinite(values)
        accepted = declared & finite
        finite_valid += int(np.count_nonzero(accepted))
        mask_valid_nonfinite += int(np.count_nonzero(declared & ~finite))
        masked += int(np.count_nonzero(~declared))
        if np.any(accepted):
            selected = values[accepted]
            block_min = float(np.min(selected))
            block_max = float(np.max(selected))
            minimum = block_min if minimum is None else min(minimum, block_min)
            maximum = block_max if maximum is None else max(maximum, block_max)
    total = dataset.width * dataset.height
    return {
        "source_cells": total,
        "finite_mask_valid_cells": finite_valid,
        "finite_mask_valid_fraction": finite_valid / total,
        "mask_valid_nonfinite_cells": mask_valid_nonfinite,
        "masked_cells": masked,
        "minimum_native_z": minimum,
        "maximum_native_z": maximum,
        "support_semantics": "finite GeoTIFF raster cell only; direct measurement versus interpolation unresolved",
        "invented_support_cells": 0,
    }


def _alignment_metadata(
    dem: rasterio.io.DatasetReader,
    orthomosaic: rasterio.io.DatasetReader,
) -> dict[str, Any]:
    same_crs = dem.crs is not None and dem.crs == orthomosaic.crs
    left = max(dem.bounds.left, orthomosaic.bounds.left)
    bottom = max(dem.bounds.bottom, orthomosaic.bounds.bottom)
    right = min(dem.bounds.right, orthomosaic.bounds.right)
    top = min(dem.bounds.top, orthomosaic.bounds.top)
    intersection_area = max(0.0, right - left) * max(0.0, top - bottom)
    dem_area = (dem.bounds.right - dem.bounds.left) * (dem.bounds.top - dem.bounds.bottom)
    inverse = ~orthomosaic.transform
    origin_col, origin_row = inverse * (dem.transform.c, dem.transform.f)
    x_col, x_row = inverse * (dem.transform.c + dem.transform.a, dem.transform.f + dem.transform.d)
    y_col, y_row = inverse * (dem.transform.c + dem.transform.b, dem.transform.f + dem.transform.e)
    return {
        "same_crs": bool(same_crs),
        "bounds_intersection": [left, bottom, right, top] if intersection_area > 0.0 else None,
        "dem_area_overlap_fraction": intersection_area / dem_area if dem_area > 0.0 else 0.0,
        "dem_origin_in_orthomosaic_pixels": [float(origin_col), float(origin_row)],
        "dem_x_pixel_vector_in_orthomosaic_pixels": [float(x_col - origin_col), float(x_row - origin_row)],
        "dem_y_pixel_vector_in_orthomosaic_pixels": [float(y_col - origin_col), float(y_row - origin_row)],
        "empirical_registration_error_pixels": None,
        "orthomosaic_pixel_values_read": False,
    }


def _structural_abort_reasons(
    dem_meta: dict[str, Any],
    orthomosaic_meta: dict[str, Any],
    alignment: dict[str, Any],
    support: dict[str, Any],
) -> list[str]:
    reasons = []
    crs = dem_meta["crs"]
    linear_units = str(crs.get("linear_units", "")).lower()
    if not crs.get("present"):
        reasons.append("DEM_CRS_MISSING")
    elif not crs.get("projected") or linear_units not in {"metre", "meter", "metres", "meters", "m"}:
        reasons.append("DEM_XY_UNIT_NOT_PROJECTED_METRE")
    if dem_meta["bands"] != 1:
        reasons.append("DEM_NOT_SINGLE_BAND")
    if not dem_meta["axis_aligned_north_up"]:
        reasons.append("DEM_NOT_AXIS_ALIGNED_NORTH_UP")
    gsd_x, gsd_y = dem_meta["gsd_xy_native"]
    if not math.isclose(gsd_x, gsd_y, rel_tol=0.0, abs_tol=1e-9):
        reasons.append("DEM_GSD_NOT_SQUARE")
    if max(gsd_x, gsd_y) > 0.05 + 1e-12:
        reasons.append("DEM_GSD_EXCEEDS_0_05_M")
    z_unit = str(dem_meta["units"][0] if dem_meta["units"] else "").lower()
    if z_unit not in {"metre", "meter", "metres", "meters", "m"}:
        reasons.append("DEM_NATIVE_Z_UNIT_UNRESOLVED")
    if support["finite_mask_valid_cells"] == 0:
        reasons.append("DEM_HAS_NO_FINITE_SUPPORT")
    if orthomosaic_meta["crs"].get("present") is not True or not alignment["same_crs"]:
        reasons.append("DEM_ORTHOMOSAIC_CRS_MISMATCH")
    if alignment["dem_area_overlap_fraction"] <= 0.0:
        reasons.append("DEM_ORTHOMOSAIC_BOUNDS_DO_NOT_OVERLAP")
    return reasons


def _authorization_blockers() -> list[str]:
    return [
        "DIRECT_VERSUS_INTERPOLATED_SUPPORT_UNRESOLVED",
        "EMPIRICAL_DEM_ORTHOMOSAIC_REGISTRATION_NOT_MEASURED",
        "VEGETATION_WATER_OBJECT_MASKS_NOT_FROZEN",
        "EVENT_GROUP_INDEPENDENCE_NOT_ESTABLISHED",
        "TOTAL_SURFACE_ERROR_UNCALIBRATED",
        "ALESSIO_CROSS_SOURCE_AUDIT_NOT_COMPLETED",
    ]


def _read_source_window(
    dataset: rasterio.io.DatasetReader,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = bounds
    dx = float(dataset.transform.a)
    dy = float(-dataset.transform.e)
    left = float(dataset.bounds.left)
    top = float(dataset.bounds.top)
    col0 = max(0, int(math.floor((min_x - left) / dx + 1e-10)))
    col1 = min(dataset.width, int(math.ceil((max_x - left) / dx - 1e-10)))
    row0 = max(0, int(math.floor((top - max_y) / dy + 1e-10)))
    row1 = min(dataset.height, int(math.ceil((top - min_y) / dy - 1e-10)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("empty source window")
    window = Window(col0, row0, col1 - col0, row1 - row0)
    values = dataset.read(1, window=window, masked=False)
    valid = (dataset.read_masks(1, window=window) > 0) & np.isfinite(values)
    values = np.asarray(values[::-1], dtype=np.float64)
    valid = valid[::-1]
    actual_min_x = left + col0 * dx
    actual_max_x = left + col1 * dx
    actual_max_y = top - row0 * dy
    actual_min_y = top - row1 * dy
    x_bounds = actual_min_x + np.arange(col1 - col0 + 1, dtype=np.float64) * dx
    y_bounds = actual_min_y + np.arange(row1 - row0 + 1, dtype=np.float64) * dy
    if not math.isclose(x_bounds[-1], actual_max_x, rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
        y_bounds[-1], actual_max_y, rel_tol=0.0, abs_tol=1e-8
    ):
        raise AssertionError("source window bounds drifted")
    return values, valid, x_bounds, y_bounds


def _tile_bounds(dataset: rasterio.io.DatasetReader) -> Iterable[tuple[int, int, tuple[float, float, float, float]]]:
    columns = math.ceil((dataset.bounds.right - dataset.bounds.left) / ANALYSIS_CORE_M)
    rows = math.ceil((dataset.bounds.top - dataset.bounds.bottom) / ANALYSIS_CORE_M)
    for row in range(rows):
        min_y = dataset.bounds.bottom + row * ANALYSIS_CORE_M
        max_y = min(dataset.bounds.top, min_y + ANALYSIS_CORE_M)
        for column in range(columns):
            min_x = dataset.bounds.left + column * ANALYSIS_CORE_M
            max_x = min(dataset.bounds.right, min_x + ANALYSIS_CORE_M)
            yield row, column, (float(min_x), float(min_y), float(max_x), float(max_y))


def _phase_origins(dataset: rasterio.io.DatasetReader) -> tuple[tuple[float, float], ...]:
    dx = float(dataset.transform.a)
    dy = float(-dataset.transform.e)
    offsets = ((0.0, 0.0), (0.5 * dx, 0.0), (0.0, 0.5 * dy), (0.5 * dx, 0.5 * dy))
    return tuple((dataset.bounds.left + x, dataset.bounds.bottom + y) for x, y in offsets)


def _phase_offsets(dataset: rasterio.io.DatasetReader) -> tuple[tuple[float, float], ...]:
    dx = float(dataset.transform.a)
    dy = float(-dataset.transform.e)
    return ((0.0, 0.0), (0.5 * dx, 0.0), (0.0, 0.5 * dy), (0.5 * dx, 0.5 * dy))


def _analyze_window(
    dataset: rasterio.io.DatasetReader,
    core_bounds: tuple[float, float, float, float],
) -> tuple[tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands], float]:
    expanded = (
        max(dataset.bounds.left, core_bounds[0] - ANALYSIS_HALO_M),
        max(dataset.bounds.bottom, core_bounds[1] - ANALYSIS_HALO_M),
        min(dataset.bounds.right, core_bounds[2] + ANALYSIS_HALO_M),
        min(dataset.bounds.top, core_bounds[3] + ANALYSIS_HALO_M),
    )
    values, valid, x_bounds, y_bounds = _read_source_window(dataset, expanded)
    x_centers = 0.5 * (x_bounds[:-1] + x_bounds[1:])
    y_centers = 0.5 * (y_bounds[:-1] + y_bounds[1:])
    core_source = (
        (x_centers >= core_bounds[0])
        & (x_centers < core_bounds[2])
    )[None, :] & (
        (y_centers >= core_bounds[1])
        & (y_centers < core_bounds[3])
    )[:, None]
    support_fraction = float(np.mean(valid[core_source])) if np.any(core_source) else 0.0
    phases = []
    for origin, offset in zip(_phase_origins(dataset), _phase_offsets(dataset), strict=True):
        h2 = exact_overlap_resample(
            values,
            valid,
            x_bounds,
            y_bounds,
            lattice_origin_xy_m=origin,
        )
        phases.append(analyze_phase(h2, offset))
    return tuple(phases), support_fraction  # type: ignore[return-value]


class _GridAccumulator:
    def __init__(self) -> None:
        self.area_m2 = 0.0
        self.complete_cells = 0
        self.sum_area_value = 0.0
        self.sum_area_squared = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: GridIntegral) -> None:
        self.area_m2 += value.area_m2
        self.complete_cells += value.complete_cells
        self.sum_area_value += value.sum_area_value
        self.sum_area_squared += value.sum_area_squared
        if value.minimum is not None:
            self.minimum = value.minimum if self.minimum is None else min(self.minimum, value.minimum)
        if value.maximum is not None:
            self.maximum = value.maximum if self.maximum is None else max(self.maximum, value.maximum)

    def record(self) -> dict[str, Any]:
        return {
            "valid_area_m2": self.area_m2,
            "complete_valid_cells": self.complete_cells,
            "area_weighted_mean_native_z": self.sum_area_value / self.area_m2 if self.area_m2 else None,
            "area_weighted_rms_native_z": math.sqrt(self.sum_area_squared / self.area_m2) if self.area_m2 else None,
            "minimum_native_z": self.minimum,
            "maximum_native_z": self.maximum,
        }


class _CommonAccumulator:
    def __init__(self) -> None:
        self.area_m2 = 0.0
        self.phase_energy = np.zeros(4, dtype=np.float64)
        self.pair_energy = np.zeros(6, dtype=np.float64)

    def add(self, value: CommonPhaseIntegral) -> None:
        self.area_m2 += value.area_m2
        self.phase_energy += value.phase_sum_area_squared
        self.pair_energy += value.pair_sum_area_squared_difference

    def record(self) -> dict[str, Any]:
        if self.area_m2 <= 0.0:
            return {"common_valid_area_m2": 0.0, "energy_native_z2": None, "phase_disagreement_ratio": None}
        energy = self.phase_energy / self.area_m2
        mean_energy = float(np.mean(energy))
        disagreement = float(np.mean(self.pair_energy / self.area_m2))
        cv = float(np.std(energy) / mean_energy) if mean_energy > 0.0 else None
        return {
            "common_valid_area_m2": self.area_m2,
            "energy_native_z2_by_phase": [float(value) for value in energy],
            "mean_energy_native_z2": mean_energy,
            "mean_pairwise_squared_difference_native_z2": disagreement,
            "phase_disagreement_ratio": disagreement / mean_energy if mean_energy > 0.0 else None,
            "population_energy_cv": cv,
        }


def _band_diagnostics(
    dataset: rasterio.io.DatasetReader,
    qa_path: Path,
    survey_id: str,
) -> dict[str, Any]:
    phase_accumulators = {
        band: [_GridAccumulator() for _ in range(4)] for band in ("B1", "B2")
    }
    common_accumulators = {band: _CommonAccumulator() for band in ("B1", "B2")}
    best: tuple[float, int, int, tuple[float, float, float, float]] | None = None
    tile_count = 0
    for row, column, core in _tile_bounds(dataset):
        phases, support_fraction = _analyze_window(dataset, core)
        tile_count += 1
        score = (support_fraction, -row, -column, core)
        if best is None or score[:3] > best[:3]:
            best = score
        for band in ("B1", "B2"):
            grids = tuple(getattr(phase, band.lower()) for phase in phases)
            for phase_index, grid in enumerate(grids):
                phase_accumulators[band][phase_index].add(integrate_grid(grid, core))
            common_accumulators[band].add(integrate_common_phases(grids, core))
    if best is None:
        raise ValueError(f"{survey_id} has no analysis tiles")
    representative_core = best[3]
    representative_phases, representative_support = _analyze_window(dataset, representative_core)
    render_height_band_qa(qa_path, survey_id, representative_phases)
    return {
        "operator": {
            "H2": "0.0625 m exact source-cell overlap; finite mask-valid source only; 100% finite area required",
            "F4": "c=[1,3,6,10,12,12,10,6,3,1]/64; float64 east then north",
            "R4": "separable Keys cubic a=-0.5; complete support only",
            "B1": "A1-R4(A0), A1=F4(H2), A0=F4(A1)",
            "B2": "H2-R4(A1)",
            "analysis_halo_m": ANALYSIS_HALO_M,
            "padding": "forbidden; raster edges and invalid source footprints remain invalid",
        },
        "source_grid_phases": [list(offset) for offset in _phase_offsets(dataset)],
        "phase_definition": "output-grid boundary origin shifted by zero or half one native source cell per axis",
        "core_tile_m": ANALYSIS_CORE_M,
        "tiles": tile_count,
        "phase_metrics": {
            band: [accumulator.record() for accumulator in phase_accumulators[band]]
            for band in ("B1", "B2")
        },
        "common_phase_metrics": {
            band: common_accumulators[band].record() for band in ("B1", "B2")
        },
        "representative_height_blind_selection": {
            "criterion": "maximum finite mask-valid source-cell fraction; row-major tie break; no height values",
            "core_bounds": list(representative_core),
            "source_support_fraction": representative_support,
        },
    }


def _survey_record(
    survey: SurveyPair,
    extracted_root: Path,
    output_root: Path,
    source_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dem_path = extracted_root / survey.dem_member
    orthomosaic_path = extracted_root / survey.orthomosaic_member
    with rasterio.open(dem_path) as dem, rasterio.open(orthomosaic_path) as orthomosaic:
        dem_metadata = _dataset_metadata(dem, pixels_read=True)
        orthomosaic_metadata = _dataset_metadata(orthomosaic, pixels_read=False)
        support = _finite_support(dem)
        alignment = _alignment_metadata(dem, orthomosaic)
        structural_aborts = _structural_abort_reasons(
            dem_metadata, orthomosaic_metadata, alignment, support
        )
        band_blockers = [
            reason
            for reason in structural_aborts
            if reason
            in {
                "DEM_CRS_MISSING",
                "DEM_XY_UNIT_NOT_PROJECTED_METRE",
                "DEM_NOT_SINGLE_BAND",
                "DEM_NOT_AXIS_ALIGNED_NORTH_UP",
                "DEM_GSD_NOT_SQUARE",
                "DEM_GSD_EXCEEDS_0_05_M",
                "DEM_HAS_NO_FINITE_SUPPORT",
            }
        ]
        diagnostics = None
        if not band_blockers:
            diagnostics = _band_diagnostics(
                dem,
                output_root / "qa" / f"{survey.survey_id.lower()}_height_bands.png",
                survey.survey_id,
            )
    record = {
        "survey_id": survey.survey_id,
        "source_members": {
            "dem": source_identities[survey.dem_member],
            "orthomosaic": source_identities[survey.orthomosaic_member],
        },
        "dem": dem_metadata,
        "orthomosaic": orthomosaic_metadata,
        "alignment_metadata_only": alignment,
        "finite_support": support,
        "analysis_abort_reasons": band_blockers,
        "qualification_abort_reasons": sorted(set(structural_aborts + _authorization_blockers())),
        "height_band_diagnostics": diagnostics,
        "authority": {
            "research_only_weak_height_evidence": True,
            "absolute_height_truth": False,
            "production_target": False,
            "synthesis": False,
            "preview": False,
            "orthomosaic_pixels_as_model_input": False,
            "orthomosaic_pixels_read": False,
        },
    }
    write_json(output_root / f"{survey.survey_id.lower()}-metadata.json", record)
    return record


def qualify_hinsberger(
    archive_path: Path,
    extracted_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Verify selected evidence and emit research-only metadata/height-band QA."""
    archive_path = archive_path.resolve()
    extracted_root = extracted_root.resolve()
    archive_identity, archive_inventory = _verify_archive(archive_path)
    source_identities = _verify_extracted_members(extracted_root)
    recipe = {
        "schema": _SCHEMA,
        "dataset": {
            "doi": DATASET_DOI,
            "publisher_file_id": PUBLISHER_FILE_ID,
            "archive_name": ARCHIVE_NAME,
            "archive_bytes": ARCHIVE_BYTES,
            "publisher_md5": ARCHIVE_MD5,
            "license": {"spdx": LICENSE_SPDX, "url": LICENSE_URL},
        },
        "local_archive": archive_identity,
        "selected_extracted_members": source_identities,
        "selected_surveys": [survey.survey_id for survey in SELECTED_SURVEYS],
        "implementation": _implementation_identity(),
        "policy": {
            "orthomosaic_pixels_read": False,
            "orthomosaic_model_input": False,
            "invented_support": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    build_id = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = output_root or DATA_WORK / "microtopography" / "rill-diffusion" / "hinsberger" / "sha256"
    final = parent / build_id
    manifest_path = final / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    staging = parent / f".{build_id}.tmp-{os.getpid()}"
    if staging.exists() or final.exists():
        raise FileExistsError(f"Hinsberger qualification transaction already exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "recipe.json").write_bytes(_canonical_bytes(recipe))
    write_json(staging / "archive-inventory.json", {"members": archive_inventory})
    records = [
        _survey_record(survey, extracted_root, staging, source_identities)
        for survey in SELECTED_SURVEYS
    ]
    interpretations = {
        f"{survey.survey_id.lower()}_height_bands.png": (
            f"{survey.survey_id} height-only B1/B2 four-phase diagnostic; no optical pixels, padding, "
            "support invention, truth, training, production, or preview claim."
        )
        for survey in SELECTED_SURVEYS
        if (staging / "qa" / f"{survey.survey_id.lower()}_height_bands.png").exists()
    }
    qa = qa_index(staging / "qa", interpretations) if interpretations else {"schema": "hinsberger-height-band-qa/1", "images": []}
    write_json(staging / "qa" / "index.json", qa)
    blockers = sorted(
        {
            reason
            for record in records
            for reason in record["qualification_abort_reasons"]
        }
    )
    manifest = {
        "schema": _SCHEMA,
        "build_id": build_id,
        "status": "research_diagnostic_only" if any(record["height_band_diagnostics"] for record in records) else "aborted_before_band_analysis",
        "result": "not_authorized",
        "recipe_sha256": hashlib.sha256((staging / "recipe.json").read_bytes()).hexdigest(),
        "archive_inventory_sha256": hashlib.sha256((staging / "archive-inventory.json").read_bytes()).hexdigest(),
        "surveys": [
            {
                "survey_id": record["survey_id"],
                "metadata_path": f"{record['survey_id'].lower()}-metadata.json",
                "analysis_abort_reasons": record["analysis_abort_reasons"],
                "qualification_abort_reasons": record["qualification_abort_reasons"],
            }
            for record in records
        ],
        "qualification_abort_reasons": blockers,
        "qa_index": {"path": "qa/index.json", "sha256": hashlib.sha256((staging / "qa" / "index.json").read_bytes()).hexdigest()},
        "authority": {
            "raw_retention": True,
            "metadata_and_height_band_qa": True,
            "weak_height_supervision": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    write_json(staging / "manifest.json", manifest)
    staging.rename(final)
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and qualify the selected Hinsberger DEM/orthomosaic pairs without reading optical pixels."
    )
    parser.add_argument("--archive", type=Path, required=True, help=f"complete {ARCHIVE_NAME}")
    parser.add_argument(
        "--extracted-root",
        type=Path,
        required=True,
        help="root beneath which the six selected member paths are already fully extracted",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = qualify_hinsberger(args.archive, args.extracted_root, args.output_root)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Hinsberger qualification blocked: {error}\n")
    print(manifest)


if __name__ == "__main__":
    _main()

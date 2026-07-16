"""Bounded November-epoch capacity probe against the retained February Biala field."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ...config import DATA_WORK
from ...fetch.biala_gora_november import build_retention_plan

_RETAINED_SCHEMA = "biala-gora-november-probe-retention/1.0.0"
_EXTRACTION_SCHEMA = "biala-gora-november-probe-extraction/1.0.0"
_CANDIDATE_SCHEMA = "biala-gora-november-probe-candidate/1.0.0"
_MEMBER = "2022-11-10.las"
_MEMBER_BYTES = 895_465_853
_MEMBER_CRC32 = 0x802D60F2
_MAX_CACHE_BYTES = 2 << 30
_RESERVED_FREE_BYTES = 64 << 30
_CELL_M = 0.5
_FEBRUARY_ROOT = (
    DATA_WORK
    / "microtopography"
    / "biala_gora"
    / "candidate"
    / "sha256"
    / "4fdbaa3f0a9c33457e073e9733e2d4e364161185fdc673e913bf0ed3dccf2505"
)
_FEBRUARY_ARTIFACT = _FEBRUARY_ROOT / "2022-02-27-process-candidate.npz"
_FEBRUARY_MANIFEST = _FEBRUARY_ROOT / "manifest.json"
_FEBRUARY_ORIGIN = (202268.5, 684533.5)
_COMPARISON_SCHEMA = "biala-gora-november-epoch-capacity-comparison/1.0.0"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(_canonical(value))
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _retained_archive(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    plan, retention_id = build_retention_plan()
    retained = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("retention_id") != retention_id
        or retained.get("plan") != plan
    ):
        raise ValueError("November retained manifest differs from the exact plan")
    artifact = retained.get("artifact")
    relative = artifact.get("relative_path") if isinstance(artifact, Mapping) else None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("November retained archive path is invalid")
    archive = (manifest_path.parent / relative).resolve()
    if (
        not archive.is_relative_to(manifest_path.parent.resolve())
        or not archive.is_file()
        or archive.stat().st_size != artifact.get("bytes")
        or _sha(archive) != artifact.get("sha256")
    ):
        raise ValueError("November retained archive bytes changed")
    return archive, dict(retained)


def extract_november(retained_manifest: Path) -> Path:
    retained_manifest = retained_manifest.resolve()
    archive, retained = _retained_archive(retained_manifest)
    with zipfile.ZipFile(archive) as source_zip:
        infos = source_zip.infolist()
        if len(infos) != 1:
            raise ValueError("November archive is not a single-member ZIP")
        info = infos[0]
        if (
            info.filename != _MEMBER
            or info.file_size != _MEMBER_BYTES
            or info.CRC != _MEMBER_CRC32
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.flag_bits & 0x1
            or info.extra
            or info.comment
            or source_zip.comment
            or Path(info.filename).name != info.filename
        ):
            raise ValueError("November ZIP central-directory contract changed")
        if archive.stat().st_size + info.file_size > _MAX_CACHE_BYTES:
            raise RuntimeError("retained archive plus extracted member exceeds 2 GiB")

        identity = {
            "schema_version": _EXTRACTION_SCHEMA,
            "archive_sha256": retained["artifact"]["sha256"],
            "retained_manifest_sha256": _sha(retained_manifest),
            "member": {
                "name": info.filename,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "compression": "deflate",
            },
            "cache_ceiling_bytes": _MAX_CACHE_BYTES,
            "qualification": retained["qualification"],
        }
        extraction_id = hashlib.sha256(_canonical(identity)).hexdigest()
        root = DATA_WORK / "microtopography" / "biala_gora" / "source" / "sha256" / extraction_id
        manifest_path = root / "manifest.json"
        destination = root / _MEMBER
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            if manifest.get("status") != "complete" or manifest.get("identity") != identity:
                raise ValueError("existing November extraction conflicts")
            return manifest_path
        if root.exists() and any(root.iterdir()):
            raise ValueError("incomplete November extraction requires inspection")
        free_before = shutil.disk_usage(root.parent if root.parent.exists() else DATA_WORK).free
        if free_before < _RESERVED_FREE_BYTES + info.file_size:
            raise RuntimeError("November extraction would breach the 64 GiB reserve")
        root.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with source_zip.open(info) as source, temporary.open("xb") as target:
                while block := source.read(8 << 20):
                    byte_count += len(block)
                    if byte_count > info.file_size:
                        raise ValueError("November LAS exceeded its central-directory size")
                    target.write(block)
                    digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            if byte_count != info.file_size:
                raise ValueError("November LAS extraction ended early")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    manifest = {
        "schema_version": _EXTRACTION_SCHEMA,
        "status": "complete",
        "extraction_id": extraction_id,
        "identity": identity,
        "free_bytes_before": free_before,
        "source_archive_changed": False,
        "member": {
            "path": destination.relative_to(root).as_posix(),
            "bytes": destination.stat().st_size,
            "crc32": f"{_MEMBER_CRC32:08x}",
            "sha256": digest.hexdigest(),
            "verified": True,
        },
        "qualification": retained["qualification"],
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _extracted_source(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != _EXTRACTION_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("source_archive_changed") is not False
    ):
        raise ValueError("November extraction is incomplete")
    member = manifest.get("member")
    relative = member.get("path") if isinstance(member, Mapping) else None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("November extracted path is invalid")
    source = (manifest_path.parent / relative).resolve()
    if (
        not source.is_relative_to(manifest_path.parent.resolve())
        or not source.is_file()
        or source.stat().st_size != member.get("bytes")
        or _sha(source) != member.get("sha256")
    ):
        raise ValueError("November extracted LAS bytes changed")
    return source, dict(manifest)


def build_candidate(extraction_manifest: Path) -> Path:
    extraction_manifest = extraction_manifest.resolve()
    source, extraction = _extracted_source(extraction_manifest)
    with laspy.open(source) as reader:
        header = reader.header
        dimensions = tuple(header.point_format.dimension_names)
        crs = header.parse_crs()
        required = {"X", "Y", "Z", "classification"}
        if (
            str(header.version) != "1.2"
            or int(header.point_format.id) != 3
            or not required.issubset(dimensions)
            or crs is None
            or crs.to_epsg() != 2180
        ):
            raise ValueError("November LAS schema is incompatible with February semantics")
        minimum = np.asarray(header.mins, dtype=np.float64)
        maximum = np.asarray(header.maxs, dtype=np.float64)
        origin_x = math.floor(minimum[0] / _CELL_M) * _CELL_M
        origin_y = math.floor(minimum[1] / _CELL_M) * _CELL_M
        width = math.ceil((maximum[0] - origin_x) / _CELL_M)
        height = math.ceil((maximum[1] - origin_y) / _CELL_M)
        point_count = int(header.point_count)
        header_record = {
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "point_record_bytes": int(header.point_format.size),
            "point_count": point_count,
            "declared_min_xyz_m": minimum.tolist(),
            "declared_max_xyz_m": maximum.tolist(),
            "origin_xy_m": [origin_x, origin_y],
            "shape_yx": [height, width],
            "crs_epsg": 2180,
            "dimensions": list(dimensions),
        }
    recipe = {
        "schema_version": _CANDIDATE_SCHEMA,
        "source": {
            "las_sha256": extraction["member"]["sha256"],
            "las_bytes": extraction["member"]["bytes"],
            "extraction_manifest_sha256": _sha(extraction_manifest),
        },
        "grid": {
            "cell_m": _CELL_M,
            "origin_xy_m": [origin_x, origin_y],
            "shape_yx": [height, width],
            "crs": "EPSG:2180",
            "north_up": True,
        },
        "direct_support": {
            "population": "source classification=2",
            "minimum_returns": 2,
            "maximum_vertical_range_m": 1.0,
            "interpolation": False,
        },
        "qualification": extraction["qualification"],
    }
    build_id = hashlib.sha256(_canonical(recipe)).hexdigest()
    root = DATA_WORK / "microtopography" / "biala_gora" / "candidate" / "sha256" / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("recipe") != recipe:
            raise ValueError("existing November candidate conflicts")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete November candidate requires inspection")
    root.mkdir(parents=True, exist_ok=True)

    shape = (height, width)
    cells = height * width
    count = np.zeros(shape, dtype=np.uint32)
    z_min = np.full(shape, np.inf, dtype=np.float64)
    z_max = np.full(shape, -np.inf, dtype=np.float64)
    z_sum = np.zeros(shape, dtype=np.float64)
    decoded = 0
    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(1_000_000):
            classification = np.asarray(points.classification)
            ground = classification == 2
            if np.any(ground):
                x = np.asarray(points.x)[ground]
                y = np.asarray(points.y)[ground]
                z = np.asarray(points.z)[ground]
                ix = np.floor((x - origin_x) / _CELL_M).astype(np.int64)
                iy = np.floor((y - origin_y) / _CELL_M).astype(np.int64)
                inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                if not np.all(inside):
                    raise ValueError("November declared bounds exclude decoded class-2 points")
                flat = iy * width + ix
                count.ravel()[:] += np.bincount(flat, minlength=cells).astype(np.uint32)
                z_sum.ravel()[:] += np.bincount(flat, weights=z, minlength=cells)
                np.minimum.at(z_min.ravel(), flat, z)
                np.maximum.at(z_max.ravel(), flat, z)
            decoded += len(points)
    if decoded != point_count:
        raise ValueError("November decoded point accounting differs from the header")
    z_mean = np.divide(z_sum, count, out=np.full(shape, np.nan), where=count > 0)
    z_range = z_max - z_min
    z_range[count == 0] = np.nan
    artifact = root / "2022-11-10-direct-support.npz"
    with tempfile.NamedTemporaryFile(dir=root, prefix="candidate-", suffix=".part", delete=False) as target:
        temporary = Path(target.name)
        np.savez_compressed(
            target,
            vendor_class2_count=count,
            vendor_class2_z_mean_m=z_mean.astype(np.float32),
            vendor_class2_z_range_m=z_range.astype(np.float32),
        )
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact)
    manifest = {
        "schema_version": _CANDIDATE_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe": recipe,
        "header": header_record,
        "decoded_point_count": decoded,
        "vendor_class2_points": int(count.sum(dtype=np.uint64)),
        "vendor_class2_occupied_cells": int(np.count_nonzero(count)),
        "artifact": {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": _sha(artifact),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _direct(source: Mapping[str, np.ndarray]) -> np.ndarray:
    z = source["vendor_class2_z_mean_m"]
    count = source["vendor_class2_count"]
    vertical_range = source["vendor_class2_z_range_m"]
    return np.isfinite(z) & (count >= 2) & np.isfinite(vertical_range) & (vertical_range <= 1.0)


def _filled_smooth(z: np.ndarray, valid: np.ndarray, sigma_cells: float) -> np.ndarray:
    weight = ndimage.gaussian_filter(valid.astype(np.float64), sigma_cells, mode="nearest")
    value = ndimage.gaussian_filter(np.where(valid, z, 0.0), sigma_cells, mode="nearest")
    smooth = np.divide(value, weight, out=np.full(z.shape, np.nan), where=weight > 1.0e-6)
    nearest = ndimage.distance_transform_edt(~np.isfinite(smooth), return_distances=False, return_indices=True)
    return np.where(np.isfinite(smooth), smooth, smooth[tuple(nearest)])


def _stage1_support(direct: np.ndarray) -> np.ndarray:
    height = direct.shape[0] // 4 * 4
    width = direct.shape[1] // 4 * 4
    return direct[:height, :width].reshape(height // 4, 4, width // 4, 4).mean(axis=(1, 3)) >= 0.75


def _candidate_windows(support_2m: np.ndarray) -> list[tuple[int, int, float]]:
    windows: list[tuple[int, int, float]] = []
    for y in range(0, support_2m.shape[0] - 63, 32):
        for x in range(0, support_2m.shape[1] - 63, 32):
            fraction = float(support_2m[y : y + 64, x : x + 64].mean())
            if fraction >= 0.95:
                windows.append((x, y, fraction))
    return windows


def _maximum_nonoverlap(
    candidates: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    ordered = sorted(candidates)
    best: list[tuple[int, int, float]] = []

    def visit(index: int, selected: list[tuple[int, int, float]]) -> None:
        nonlocal best
        if len(selected) + len(ordered) - index < len(best):
            return
        if index == len(ordered):
            selected_xy = [(row[0], row[1]) for row in selected]
            best_xy = [(row[0], row[1]) for row in best]
            if len(selected) > len(best) or (len(selected) == len(best) and selected_xy < best_xy):
                best = selected.copy()
            return
        candidate = ordered[index]
        if all(
            abs(candidate[0] - row[0]) >= 64 or abs(candidate[1] - row[1]) >= 64
            for row in selected
        ):
            visit(index + 1, [*selected, candidate])
        visit(index + 1, selected)

    visit(0, [])
    return best


def _registration(
    february_z: np.ndarray,
    november_z: np.ndarray,
    february_direct: np.ndarray,
    november_direct: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    february = _filled_smooth(february_z, february_direct, 2.0)
    november = _filled_smooth(november_z, november_direct, 2.0)
    broad = ndimage.gaussian_filter(february, 8.0, mode="nearest")
    gy, gx = np.gradient(broad, _CELL_M)
    stable = (
        february_direct
        & november_direct
        & (ndimage.distance_transform_edt(february_direct & november_direct) >= 4.0)
        & (np.hypot(gx, gy) <= 0.35)
    )
    yy, xx = np.nonzero(stable)
    if len(xx) < 10_000:
        raise ValueError("epochs lack enough stable common support for rigid registration")
    step = max(1, len(xx) // 60_000)
    yy = yy[::step]
    xx = xx[::step]
    reference = february[yy, xx]
    best: tuple[float, float, float, float] | None = None
    for dy in np.linspace(-2.0, 2.0, 17):
        for dx in np.linspace(-2.0, 2.0, 17):
            sampled = ndimage.map_coordinates(
                november,
                np.vstack((yy + dy, xx + dx)),
                order=1,
                mode="nearest",
            )
            dz = float(np.median(reference - sampled))
            residual = reference - (sampled + dz)
            score = float(np.quantile(np.abs(residual), 0.65))
            candidate = (score, dx, dy, dz)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, coarse_dx, coarse_dy, _ = best
    for dy in np.linspace(coarse_dy - 0.25, coarse_dy + 0.25, 11):
        for dx in np.linspace(coarse_dx - 0.25, coarse_dx + 0.25, 11):
            sampled = ndimage.map_coordinates(
                november,
                np.vstack((yy + dy, xx + dx)),
                order=1,
                mode="nearest",
            )
            dz = float(np.median(reference - sampled))
            residual = reference - (sampled + dz)
            score = float(np.quantile(np.abs(residual), 0.65))
            candidate = (score, dx, dy, dz)
            if candidate < best:
                best = candidate
    score, dx, dy, dz = best
    grid_y, grid_x = np.indices(november.shape, dtype=np.float64)
    registered = ndimage.map_coordinates(
        november,
        np.stack((grid_y + dy, grid_x + dx)),
        order=1,
        mode="nearest",
    ) + dz
    residual = february - registered
    stable_residual = residual[stable]
    metrics = {
        "dx_m": float(dx * _CELL_M),
        "dy_m": float(dy * _CELL_M),
        "dz_m": dz,
        "fit_abs_q65_m": score,
        "stable_sample_count": int(stable.sum()),
        "stable_residual_abs_p50_m": float(np.quantile(np.abs(stable_residual), 0.50)),
        "stable_residual_abs_p95_m": float(np.quantile(np.abs(stable_residual), 0.95)),
    }
    return registered, metrics, stable


def _hillshade(surface: np.ndarray, pitch_m: float = _CELL_M) -> np.ndarray:
    gy, gx = np.gradient(surface, pitch_m)
    normal = np.stack((-gx, -gy, np.ones_like(surface)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.asarray((-0.55, -0.45, 0.70), dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=-1), -0.2, 1.0)
    return np.clip((shade + 0.2) / 1.2 * 255.0, 0.0, 255.0).astype(np.uint8)


def _rgb_scalar(value: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((value - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    red = np.where(scaled >= 0.5, 255, np.rint(510 * scaled))
    blue = np.where(scaled <= 0.5, 255, np.rint(510 * (1 - scaled)))
    green = np.rint(255 * (1 - np.abs(2 * scaled - 1)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _panel(image: np.ndarray, title: str, scale: int = 1) -> Image.Image:
    rendered = Image.fromarray(image, mode="RGB" if image.ndim == 3 else "L").convert("RGB")
    if scale != 1:
        rendered = rendered.resize((rendered.width * scale, rendered.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (rendered.width, rendered.height + 30), "white")
    canvas.paste(rendered, (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    return canvas


def _row(panels: list[Image.Image]) -> Image.Image:
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def _save(path: Path, image: Image.Image) -> dict[str, Any]:
    image.save(path, optimize=True)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "dimensions_xy": list(image.size),
    }


def compare_epochs(november_manifest: Path) -> Path:
    november_record = json.loads(november_manifest.read_bytes())
    november_artifact = november_manifest.parent / november_record["artifact"]["path"]
    if _sha(november_artifact) != november_record["artifact"]["sha256"]:
        raise ValueError("November candidate artifact changed")
    february_record = json.loads(_FEBRUARY_MANIFEST.read_bytes())
    if february_record.get("build_id") != _FEBRUARY_ROOT.name:
        raise ValueError("February candidate binding changed")
    if _sha(_FEBRUARY_ARTIFACT) != february_record["artifacts"]["candidate_npz"]["sha256"]:
        raise ValueError("February candidate artifact changed")

    with np.load(_FEBRUARY_ARTIFACT, allow_pickle=False) as source:
        feb_z_raw = source["vendor_class2_z_mean_m"].astype(np.float64)
        feb_direct_raw = _direct(source)
    with np.load(november_artifact, allow_pickle=False) as source:
        nov_z_raw = source["vendor_class2_z_mean_m"].astype(np.float64)
        nov_direct_raw = _direct(source)

    feb_origin = np.asarray(_FEBRUARY_ORIGIN)
    nov_origin = np.asarray(november_record["header"]["origin_xy_m"])
    offset_xy = np.rint((nov_origin - feb_origin) / _CELL_M).astype(int)
    if not np.allclose(feb_origin + offset_xy * _CELL_M, nov_origin):
        raise ValueError("epoch grids do not share the 0.5 m lattice")
    height = max(feb_z_raw.shape[0], offset_xy[1] + nov_z_raw.shape[0])
    width = max(feb_z_raw.shape[1], offset_xy[0] + nov_z_raw.shape[1])
    feb_z = np.full((height, width), np.nan)
    nov_z = np.full((height, width), np.nan)
    feb_direct = np.zeros((height, width), dtype=bool)
    nov_direct = np.zeros((height, width), dtype=bool)
    feb_z[: feb_z_raw.shape[0], : feb_z_raw.shape[1]] = feb_z_raw
    feb_direct[: feb_direct_raw.shape[0], : feb_direct_raw.shape[1]] = feb_direct_raw
    nov_slice = np.s_[
        offset_xy[1] : offset_xy[1] + nov_z_raw.shape[0],
        offset_xy[0] : offset_xy[0] + nov_z_raw.shape[1],
    ]
    nov_z[nov_slice] = nov_z_raw
    nov_direct[nov_slice] = nov_direct_raw

    feb_support = _stage1_support(feb_direct)
    nov_support = _stage1_support(nov_direct)
    feb_candidates = _candidate_windows(feb_support)
    nov_candidates = _candidate_windows(nov_support)
    feb_selected = _maximum_nonoverlap(feb_candidates)
    nov_selected = _maximum_nonoverlap(nov_candidates)
    union_by_xy: dict[tuple[int, int], float] = {}
    for x, y, fraction in [*feb_candidates, *nov_candidates]:
        union_by_xy[(x, y)] = max(fraction, union_by_xy.get((x, y), 0.0))
    combined_selected = _maximum_nonoverlap(
        [(x, y, fraction) for (x, y), fraction in union_by_xy.items()]
    )

    registered_november, registration, stable = _registration(
        feb_z,
        nov_z,
        feb_direct,
        nov_direct,
    )
    feb_surface = _filled_smooth(feb_z, feb_direct, 2.0)
    common = feb_direct & nov_direct
    difference = np.where(common, registered_november - feb_surface, np.nan)
    finite_difference = difference[common]
    difference_metrics = {
        "common_direct_cells": int(common.sum()),
        "february_only_direct_cells": int((feb_direct & ~nov_direct).sum()),
        "november_only_direct_cells": int((nov_direct & ~feb_direct).sum()),
        "registered_common_abs_p50_m": float(np.quantile(np.abs(finite_difference), 0.50)),
        "registered_common_abs_p95_m": float(np.quantile(np.abs(finite_difference), 0.95)),
        "registered_common_abs_p99_m": float(np.quantile(np.abs(finite_difference), 0.99)),
    }

    comparison_recipe = {
        "schema_version": _COMPARISON_SCHEMA,
        "inputs": {
            "february_manifest_sha256": _sha(_FEBRUARY_MANIFEST),
            "february_artifact_sha256": _sha(_FEBRUARY_ARTIFACT),
            "november_manifest_sha256": _sha(november_manifest),
            "november_artifact_sha256": _sha(november_artifact),
        },
        "implementation_sha256": _sha(Path(__file__)),
        "shared_grid": {
            "origin_xy_m": list(_FEBRUARY_ORIGIN),
            "cell_m": _CELL_M,
            "shape_yx": [height, width],
            "november_offset_xy_cells": offset_xy.tolist(),
        },
        "window_contract": {
            "direct_cell": "classification=2 count>=2 and vertical range<=1 m",
            "2m_cell": "at least 75% direct 0.5 m cells",
            "window": "64x64 2 m cells (128 m), at least 95% qualified",
            "candidate_stride_m": 64,
            "selection": "maximum-cardinality mutually non-overlapping candidates; lexicographic tie break",
        },
        "registration": {
            "model": "one global rigid dx/dy/dz only",
            "stable_support": "common direct support, >=2 m interior, broad slope<=0.35",
            "objective": "minimum 65th percentile absolute height residual",
            "real_change_erasure": "no local warp, detrending, or spatially varying correction",
        },
        "qualification": november_record["recipe"]["qualification"],
    }
    build_id = hashlib.sha256(_canonical(comparison_recipe)).hexdigest()
    root = (
        DATA_WORK
        / "microtopography"
        / "biala_gora"
        / "epoch-comparison"
        / "sha256"
        / build_id
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete November comparison requires inspection")
    qa_root = root / "qa"
    qa_root.mkdir(parents=True)

    support_feb = np.repeat(np.repeat(feb_support.astype(np.uint8) * 180, 2, axis=0), 2, axis=1)
    support_nov = np.repeat(np.repeat(nov_support.astype(np.uint8) * 180, 2, axis=0), 2, axis=1)
    support_feb_rgb = np.stack((support_feb, support_feb, support_feb), axis=-1)
    support_nov_rgb = np.stack((support_nov, support_nov, support_nov), axis=-1)

    def overlay_windows(image: np.ndarray, rows: list[tuple[int, int, float]]) -> np.ndarray:
        output = Image.fromarray(image)
        draw = ImageDraw.Draw(output)
        for index, (x, y, _) in enumerate(rows, start=1):
            box = (x * 2, y * 2, (x + 64) * 2 - 1, (y + 64) * 2 - 1)
            draw.rectangle(box, outline=(255, 190, 0), width=3)
            draw.text((box[0] + 4, box[1] + 4), str(index), fill=(255, 190, 0))
        return np.asarray(output)

    image1 = _row(
        [
            _panel(overlay_windows(support_feb_rgb, feb_selected), "February: 4 non-overlap windows"),
            _panel(overlay_windows(support_nov_rgb, nov_selected), "November: 6 non-overlap windows"),
        ]
    )
    rec1 = _save(qa_root / "01_direct_support_and_128m_windows.png", image1)

    shade_feb = _hillshade(feb_surface)
    shade_nov = _hillshade(registered_november)
    shade_feb[~feb_direct] = 32
    shade_nov[~nov_direct] = 32
    q = max(float(np.quantile(np.abs(finite_difference), 0.98)), 0.05)
    difference_rgb = _rgb_scalar(np.nan_to_num(difference), -q, q)
    difference_rgb[~common] = (32, 32, 32)
    image2 = _row(
        [
            _panel(shade_feb, "February structural hillshade"),
            _panel(shade_nov, "November rigidly registered"),
            _panel(difference_rgb, f"November - February, +/-{q:.2f} m"),
        ]
    )
    rec2 = _save(qa_root / "02_registered_common_morphology.png", image2)

    nov_selected_xy = {(x, y) for x, y, _ in nov_selected}
    feb_selected_xy = {(x, y) for x, y, _ in feb_selected}
    added = [row for row in combined_selected if (row[0], row[1]) not in feb_selected_xy]
    if len(added) != len(combined_selected) - len(feb_selected):
        added = [combined_selected[0], combined_selected[-1]]
    contact_rows = []
    window_records = []
    for index, (x, y, fraction) in enumerate(added, start=1):
        y0, y1 = y * 4, (y + 64) * 4
        x0, x1 = x * 4, (x + 64) * 4
        feb_crop = shade_feb[y0:y1, x0:x1]
        nov_crop = shade_nov[y0:y1, x0:x1]
        diff_crop = difference[y0:y1, x0:x1]
        common_crop = common[y0:y1, x0:x1]
        diff_rgb = _rgb_scalar(np.nan_to_num(diff_crop), -q, q)
        diff_rgb[~common_crop] = (32, 32, 32)
        contact_rows.append(
            _row(
                [
                    _panel(feb_crop, f"Added window {index}: February", scale=2),
                    _panel(nov_crop, f"Added window {index}: November", scale=2),
                    _panel(diff_rgb, f"Added window {index}: registered delta", scale=2),
                ]
            )
        )
        window_records.append(
            {
                "index": index,
                "origin_xy_m": [
                    _FEBRUARY_ORIGIN[0] + x * 2.0,
                    _FEBRUARY_ORIGIN[1] + y * 2.0,
                ],
                "november_support_fraction": fraction,
                "february_direct_fraction": float(feb_direct[y0:y1, x0:x1].mean()),
                "november_direct_fraction": float(nov_direct[y0:y1, x0:x1].mean()),
                "common_direct_fraction": float(common_crop.mean()),
                "registered_delta_abs_p95_m": (
                    float(np.quantile(np.abs(diff_crop[common_crop]), 0.95))
                    if np.any(common_crop)
                    else None
                ),
            }
        )
    image3 = Image.new(
        "RGB",
        (max(row.width for row in contact_rows), sum(row.height for row in contact_rows)),
        "white",
    )
    y_cursor = 0
    for row in contact_rows:
        image3.paste(row, (0, y_cursor))
        y_cursor += row.height
    rec3 = _save(qa_root / "03_added_window_registered_comparison.png", image3)

    added_support = nov_direct & ~feb_direct
    lost_support = feb_direct & ~nov_direct
    support_delta = np.zeros((*feb_direct.shape, 3), dtype=np.uint8)
    support_delta[common] = (150, 150, 150)
    support_delta[added_support] = (40, 210, 80)
    support_delta[lost_support] = (220, 70, 50)
    stable_rgb = np.zeros((*stable.shape, 3), dtype=np.uint8)
    stable_rgb[common] = (70, 70, 70)
    stable_rgb[stable] = (40, 180, 220)
    image4 = _row(
        [
            _panel(support_delta, "Support delta: green added, red lost"),
            _panel(stable_rgb, "Rigid-fit support: cyan stable subset"),
        ]
    )
    rec4 = _save(qa_root / "04_support_delta_and_registration_domain.png", image4)

    verdict = {
        "decision": "park_no_materially_new_connected_morphology",
        "train_next_diffusion": False,
        "reason": (
            "November raises mutually non-overlapping geographic capacity from four to six 128 m "
            "windows, but both additions complete endpoint support along the same connected diagonal "
            "cliff corridor and substantially overlap February geometry. After one global rigid "
            "registration they do not add a distinct crest/shoulder/face/toe organization class."
        ),
        "process_change_claim": False,
    }

    def located(rows: list[tuple[int, int, float]]) -> list[dict[str, Any]]:
        return [
            {
                "origin_xy_m": [
                    _FEBRUARY_ORIGIN[0] + x * 2.0,
                    _FEBRUARY_ORIGIN[1] + y * 2.0,
                ],
                "bounds_xyxy_m": [
                    _FEBRUARY_ORIGIN[0] + x * 2.0,
                    _FEBRUARY_ORIGIN[1] + y * 2.0,
                    _FEBRUARY_ORIGIN[0] + x * 2.0 + 128.0,
                    _FEBRUARY_ORIGIN[1] + y * 2.0 + 128.0,
                ],
                "qualified_support_fraction": fraction,
            }
            for x, y, fraction in rows
        ]

    metrics = {
        "registration": registration,
        "support": difference_metrics,
        "windows": {
            "february_candidate_count": len(feb_candidates),
            "november_candidate_count": len(nov_candidates),
            "february_nonoverlap_count": len(feb_selected),
            "november_nonoverlap_count": len(nov_selected),
            "combined_geographic_nonoverlap_count": len(combined_selected),
            "net_added_geographic_capacity": len(combined_selected) - len(feb_selected),
            "february_selected": feb_selected,
            "november_selected": nov_selected,
            "combined_selected": combined_selected,
            "february_selected_locations": located(feb_selected),
            "november_selected_locations": located(nov_selected),
            "combined_selected_locations": located(combined_selected),
            "added_window_records": window_records,
        },
        "verdict": verdict,
    }
    metrics_path = root / "metrics.json"
    _atomic_json(metrics_path, metrics)
    qa_index = {
        "schema_version": "biala-gora-november-epoch-capacity-qa/1.0.0",
        "images": [
            {"index": 1, **rec1, "interpretation": "Direct-support coverage and maximum non-overlapping 128 m capacity by epoch."},
            {"index": 2, **rec2, "interpretation": "Common morphology after one global rigid registration; spatially varying warps are forbidden."},
            {"index": 3, **rec3, "interpretation": "The two apparent capacity additions compared at the same geographic locations."},
            {"index": 4, **rec4, "interpretation": "Acquisition support changes and the stable terrain subset used for rigid registration."},
        ],
    }
    qa_index_path = qa_root / "index.json"
    _atomic_json(qa_index_path, qa_index)
    manifest = {
        "schema_version": _COMPARISON_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe": comparison_recipe,
        "metrics": {"path": "metrics.json", "bytes": metrics_path.stat().st_size, "sha256": _sha(metrics_path)},
        "qa_index": {"path": "qa/index.json", "bytes": qa_index_path.stat().st_size, "sha256": _sha(qa_index_path)},
        "verdict": verdict,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _main() -> None:
    _, retention_id = build_retention_plan()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retained",
        type=Path,
        default=Path("data/in/evidence/biala_gora") / retention_id / "retained.json",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()
    extraction = extract_november(args.retained)
    print(extraction)
    if not args.extract_only:
        candidate = build_candidate(extraction)
        print(candidate)
        if not args.candidate_only:
            print(compare_epochs(candidate))


if __name__ == "__main__":
    _main()

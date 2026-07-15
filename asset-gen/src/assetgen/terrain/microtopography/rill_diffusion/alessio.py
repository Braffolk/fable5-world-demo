"""Research-only Alessio topology, morphology, and flow-alignment qualification."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile
from xml.etree import ElementTree

import numpy as np
import pyogrio
import pyogrio.raw
from pyproj import CRS
import rasterio
from scipy.stats import rankdata
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union

from ....config import DATA_WORK
from .alessio_contract import (
    ARCHIVE_BYTES,
    ARCHIVE_CONTENT_ID,
    ARCHIVE_FILE_ID,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    BEDROCK_VECTORS,
    DATASET_DOI,
    DATASET_VERSION,
    DEM_DATASETS,
    FLOW_DATASETS,
    FORBIDDEN_MEMBER_SUBSTRINGS,
    GEOLOGY_CONTACTS,
    INVENTORY_MEMBER_COUNT,
    INVENTORY_PATHS_SHA256,
    INVENTORY_VERBOSE_SHA256,
    LICENSE_SPDX,
    LICENSE_URL,
    MORPHOLOGY_SELECTION_BYTES,
    MORPHOLOGY_SELECTION_COUNT,
    PORTABLE_SELECTION_COUNT,
    PORTABLE_SELECTION_SHA256,
    PUBLISHER_NOTICE,
    RILL_VECTORS,
    TRANSECT_LINES_GDB,
    TRANSECT_LOCATIONS,
    VEGETATION_VECTORS,
    WATERSHEDS,
    WORKBOOKS,
    selected_member,
)


_SCHEMA = "alessio-rill-morphology-qualification/1"
_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class VectorLayer:
    path: str
    layer: str | None
    crs: CRS | None
    geometry_type: str
    fields: tuple[str, ...]
    geometries: np.ndarray
    values: tuple[np.ndarray, ...]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(16 << 20):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _parse_verbose_inventory(path: Path) -> list[tuple[str, int]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split(maxsplit=8)
        if len(fields) != 9:
            raise ValueError(f"invalid Alessio verbose inventory line {line_number}")
        try:
            size = int(fields[4])
        except ValueError as error:
            raise ValueError(f"invalid Alessio member size at line {line_number}") from error
        records.append((fields[8], size))
    return records


def _verify_inventory(inventory_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    paths_file = inventory_root / "archive-members.txt"
    verbose_file = inventory_root / "archive-members.verbose.txt"
    portable_file = inventory_root / "portable-selection.txt"
    for path in (paths_file, verbose_file, portable_file):
        if not path.is_file():
            raise FileNotFoundError(f"retained Alessio inventory is absent: {path}")
    expected_hashes = {
        paths_file: INVENTORY_PATHS_SHA256,
        verbose_file: INVENTORY_VERBOSE_SHA256,
        portable_file: PORTABLE_SELECTION_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = _hash_file(path)
        if actual["sha256"] != expected:
            raise ValueError(f"retained Alessio inventory changed: {path.name}")
    paths = paths_file.read_text().splitlines()
    verbose = _parse_verbose_inventory(verbose_file)
    portable = portable_file.read_text().splitlines()
    if len(paths) != INVENTORY_MEMBER_COUNT or len(verbose) != INVENTORY_MEMBER_COUNT:
        raise ValueError("Alessio archive inventory must contain exactly 1,106 members")
    if len(paths) != len(set(paths)) or [name for name, _ in verbose] != paths:
        raise ValueError("Alessio archive path and verbose inventories are not one-to-one")
    if len(portable) != PORTABLE_SELECTION_COUNT or not set(portable).issubset(paths):
        raise ValueError("Alessio portable selection is incomplete or no longer inventory-bound")
    sizes = dict(verbose)
    selected = [(name, sizes[name]) for name in paths if selected_member(name)]
    if len(selected) != MORPHOLOGY_SELECTION_COUNT or sum(size for _, size in selected) != MORPHOLOGY_SELECTION_BYTES:
        raise ValueError("Alessio morphology selection changed")
    forbidden = [
        name
        for name, _ in selected
        if any(token in name.lower() for token in FORBIDDEN_MEMBER_SUBSTRINGS)
    ]
    if forbidden:
        raise ValueError(f"optical/30cm/AVI members leaked into Alessio selection: {forbidden}")
    return (
        [{"path": name, "bytes": size} for name, size in selected],
        sizes,
    )


def _verify_archive(archive: Path) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(
            f"complete Alessio MPK is absent: {archive}; .part and range files are not accepted"
        )
    if archive.name != ARCHIVE_NAME:
        raise ValueError(f"expected archive name {ARCHIVE_NAME!r}, got {archive.name!r}")
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(
            f"Alessio MPK is incomplete: {archive.stat().st_size:,} bytes; expected {ARCHIVE_BYTES:,}"
        )
    identity = _hash_file(archive)
    if identity["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Alessio MPK publisher SHA-256 mismatch")
    return identity


def _verify_extraction(
    extracted_root: Path,
    selection: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = [item["path"] for item in selection if not (extracted_root / item["path"]).is_file()]
    if missing:
        preview = missing[:12]
        raise FileNotFoundError(
            f"Alessio selected extraction is incomplete beneath {extracted_root}: "
            f"{len(missing)} of {len(selection)} members absent; first={preview}"
        )
    identities = {}
    resolved_root = extracted_root.resolve()
    for item in selection:
        path = (extracted_root / item["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"unsafe Alessio extracted path: {item['path']}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Alessio extracted member size mismatch: {item['path']}")
        identities[item["path"]] = {"bytes": item["bytes"], "sha256": _hash_file(path)["sha256"]}
    return identities


def _verify_workbooks(companion_root: Path) -> dict[str, dict[str, Any]]:
    identities = {}
    for workbook in WORKBOOKS:
        path = companion_root / workbook.filename
        if not path.is_file():
            raise FileNotFoundError(f"required Alessio companion workbook is absent: {path}")
        identity = _hash_file(path)
        if identity != {"bytes": workbook.bytes, "sha256": workbook.sha256}:
            raise ValueError(f"Alessio workbook identity mismatch: {workbook.filename}")
        identities[workbook.filename] = {
            **identity,
            "file_id": workbook.file_id,
            "role": workbook.role,
        }
    return identities


def _read_vector(path: Path, layer: str | None = None) -> VectorLayer:
    meta, _, geometry_wkb, values = pyogrio.raw.read(path, layer=layer, read_geometry=True)
    geometries = shapely.from_wkb(geometry_wkb, on_invalid="ignore")
    crs = CRS.from_user_input(meta["crs"]) if meta.get("crs") else None
    return VectorLayer(
        str(path),
        layer,
        crs,
        str(meta.get("geometry_type")),
        tuple(str(value) for value in meta.get("fields", ())),
        geometries,
        tuple(values),
    )


def _vector_metadata(layer: VectorLayer) -> dict[str, Any]:
    finite = layer.geometries[~shapely.is_missing(layer.geometries)]
    nonempty = finite[~shapely.is_empty(finite)]
    bounds = shapely.total_bounds(nonempty) if nonempty.size else np.full(4, np.nan)
    valid = shapely.is_valid(nonempty) if nonempty.size else np.empty(0, dtype=bool)
    return {
        "path": layer.path,
        "layer": layer.layer,
        "crs": layer.crs.to_wkt() if layer.crs else None,
        "projected": bool(layer.crs and layer.crs.is_projected),
        "geometry_type": layer.geometry_type,
        "feature_count": int(layer.geometries.size),
        "nonempty_count": int(nonempty.size),
        "valid_geometry_count": int(np.count_nonzero(valid)),
        "bounds": [float(value) for value in bounds] if np.all(np.isfinite(bounds)) else None,
        "fields": list(layer.fields),
    }


def _line_parts(geometries: Iterable[Any]) -> list[LineString]:
    result: list[LineString] = []

    def add(geometry: Any) -> None:
        if geometry is None or shapely.is_empty(geometry):
            return
        kind = geometry.geom_type
        if kind in {"LineString", "LinearRing"}:
            if geometry.length > 0.0:
                result.append(LineString(geometry.coords))
        elif kind == "Polygon":
            add(geometry.boundary)
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                add(part)

    for geometry in geometries:
        add(geometry)
    return result


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _network_descriptors(lines: list[LineString]) -> dict[str, Any]:
    if not lines:
        return {"line_count": 0}
    united = unary_union(lines)
    segments = _line_parts([united])
    degree: dict[tuple[int, int], int] = {}
    for segment in segments:
        for coordinate in (segment.coords[0], segment.coords[-1]):
            key = (round(float(coordinate[0]) * 1000), round(float(coordinate[1]) * 1000))
            degree[key] = degree.get(key, 0) + 1
    nearest_spacing = []
    for left, line in enumerate(lines):
        distances = [line.distance(other) for right, other in enumerate(lines) if right != left]
        if distances:
            nearest_spacing.append(min(distances))
    return {
        "line_count": len(lines),
        "noded_segment_count": len(segments),
        "total_length_native_xy": float(sum(line.length for line in lines)),
        "line_length_native_xy": _summary(line.length for line in lines),
        "nearest_rill_spacing_native_xy": _summary(nearest_spacing),
        "endpoint_node_count_1mm_identity": len(degree),
        "source_or_terminal_nodes": sum(value == 1 for value in degree.values()),
        "degree_two_nodes": sum(value == 2 for value in degree.values()),
        "branch_nodes_degree_ge_3": sum(value >= 3 for value in degree.values()),
        "maximum_node_degree": max(degree.values(), default=0),
        "branch_degree_distribution": {
            str(value): sum(degree_value == value for degree_value in degree.values())
            for value in sorted(set(degree.values()))
        },
        "topology_note": "exact noding followed by 1 mm endpoint identity; descriptive, not a production snap tolerance",
    }


def _raster_metadata(dataset: rasterio.io.DatasetReader, *, pixel_values_read: bool) -> dict[str, Any]:
    transform = dataset.transform
    crs = dataset.crs
    nodata = dataset.nodata
    if nodata is not None and not math.isfinite(nodata):
        nodata = None
    return {
        "driver": dataset.driver,
        "shape": [dataset.height, dataset.width],
        "bands": dataset.count,
        "dtype": dataset.dtypes[0] if dataset.count else None,
        "nodata": nodata,
        "crs": crs.to_wkt() if crs else None,
        "projected": bool(crs and crs.is_projected),
        "linear_units": crs.linear_units if crs else None,
        "affine": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "gsd_xy_native": [math.hypot(transform.a, transform.d), math.hypot(transform.b, transform.e)],
        "bounds": list(dataset.bounds),
        "units": list(dataset.units),
        "pixel_values_read": pixel_values_read,
    }


def _same_crs(vector: VectorLayer, raster: rasterio.io.DatasetReader) -> bool:
    return vector.crs is not None and raster.crs is not None and vector.crs == CRS.from_user_input(raster.crs)


def _flow_alignment(lines: list[LineString], raster: rasterio.io.DatasetReader) -> dict[str, Any]:
    correlations = []
    monotonic = []
    endpoint_ratios = []
    sampled_lines = 0
    valid_samples = 0
    requested_samples = 0
    fractions = np.linspace(0.0, 1.0, 32)
    for line in lines:
        points = [line.interpolate(float(fraction), normalized=True) for fraction in fractions]
        samples = list(raster.sample([(point.x, point.y) for point in points], indexes=1, masked=True))
        requested_samples += len(samples)
        values = np.asarray(
            [float(sample[0]) if not np.ma.is_masked(sample[0]) else np.nan for sample in samples],
            dtype=np.float64,
        )
        good = np.isfinite(values)
        valid_samples += int(np.count_nonzero(good))
        if np.count_nonzero(good) < 8:
            continue
        values = values[good]
        positions = fractions[good]
        if values[-1] < values[0]:
            values = values[::-1]
            positions = 1.0 - positions[::-1]
        ranks = rankdata(values, method="average")
        correlation = float(np.corrcoef(positions, ranks)[0, 1]) if np.std(ranks) > 0.0 else 0.0
        correlations.append(correlation)
        monotonic.append(float(np.mean(np.diff(values) >= 0.0)))
        endpoint_ratios.append(float((max(values[-1], 0.0) + 1.0) / (max(values[0], 0.0) + 1.0)))
        sampled_lines += 1
    return {
        "kind": "along-rill flow-accumulation progression; orientation chosen toward larger endpoint accumulation",
        "requested_samples": requested_samples,
        "valid_samples": valid_samples,
        "sample_coverage_fraction": valid_samples / requested_samples if requested_samples else 0.0,
        "usable_lines": sampled_lines,
        "spearman_progression": _summary(correlations),
        "nondecreasing_step_fraction": _summary(monotonic),
        "oriented_endpoint_accumulation_ratio": _summary(endpoint_ratios),
        "limitation": "flow accumulation is a conditioning raster, not surveyed height or an axis-specific flow-direction observation",
    }


def _xlsx_column(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_reference}")
    value = 0
    for character in letters.group(0):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _xlsx_rows(path: Path) -> dict[str, list[list[Any]]]:
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{_XLSX_NS}}}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{_XLSX_NS}}}t")))
        relations_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relations_root.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }
        workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheets = {}
        for sheet in workbook_root.findall(f".//{{{_XLSX_NS}}}sheet"):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{_DOC_REL_NS}}}id"]
            target = targets[relation_id].lstrip("/")
            member = target if target.startswith("xl/") else f"xl/{target}"
            root = ElementTree.fromstring(archive.read(member))
            rows = []
            for row in root.findall(f".//{{{_XLSX_NS}}}row"):
                cells: dict[int, Any] = {}
                for cell in row.findall(f"{{{_XLSX_NS}}}c"):
                    index = _xlsx_column(cell.attrib["r"])
                    value_node = cell.find(f"{{{_XLSX_NS}}}v")
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.iter(f"{{{_XLSX_NS}}}t"))
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared[int(value_node.text or 0)]
                    elif cell_type in {"str", "e"}:
                        value = value_node.text
                    else:
                        try:
                            value = float(value_node.text or "nan")
                        except ValueError:
                            value = value_node.text
                    cells[index] = value
                width = max(cells, default=-1) + 1
                rows.append([cells.get(index) for index in range(width)])
            sheets[name] = rows
    return sheets


def _workbook_descriptors(path: Path) -> dict[str, Any]:
    sheets = _xlsx_rows(path)
    keywords = ("width", "depth", "spacing", "length", "slope", "area", "volume", "rill", "transect")
    descriptors = []
    string_tokens: set[str] = set()
    for sheet_name, rows in sheets.items():
        for row in rows:
            for value in row:
                if isinstance(value, str):
                    token = re.sub(r"[^a-z0-9]+", "", value.lower())
                    if token:
                        string_tokens.add(token)
        header_index = None
        header_score = 0
        for index, row in enumerate(rows[:50]):
            score = sum(
                any(keyword in value.lower() for keyword in keywords)
                for value in row
                if isinstance(value, str)
            )
            if score > header_score:
                header_score = score
                header_index = index
        columns = []
        if header_index is not None:
            headers = rows[header_index]
            for column, header in enumerate(headers):
                if not isinstance(header, str) or not any(keyword in header.lower() for keyword in keywords):
                    continue
                values = [
                    float(row[column])
                    for row in rows[header_index + 1 :]
                    if column < len(row) and isinstance(row[column], (int, float)) and math.isfinite(row[column])
                ]
                columns.append({"header": header, "numeric": _summary(values)})
        descriptors.append(
            {
                "sheet": sheet_name,
                "rows": len(rows),
                "header_row_1based": header_index + 1 if header_index is not None else None,
                "recognized_columns": columns,
            }
        )
    return {"sheets": descriptors, "binding_tokens": sorted(string_tokens)}


def _attribute_tokens(layer: VectorLayer) -> set[str]:
    tokens = set()
    for values in layer.values:
        for value in values:
            if value is None:
                continue
            token = re.sub(r"[^a-z0-9]+", "", str(value).lower())
            if token:
                tokens.add(token)
    return tokens


def _bounds_overlap(left: list[float] | None, right: list[float] | None) -> bool:
    if left is None or right is None:
        return False
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _implementation_identity() -> list[dict[str, Any]]:
    return [
        {"path": path.name, **_hash_file(path)}
        for path in sorted(Path(__file__).parent.glob("*.py"))
    ]


def qualify_alessio(
    archive_path: Path,
    inventory_root: Path,
    extracted_root: Path,
    companion_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Emit morphology descriptors without treating Alessio heights as truth."""
    archive_path = archive_path.resolve()
    inventory_root = inventory_root.resolve()
    extracted_root = extracted_root.resolve()
    companion_root = companion_root.resolve()
    archive_identity = _verify_archive(archive_path)
    selection, _ = _verify_inventory(inventory_root)
    extracted_identities = _verify_extraction(extracted_root, selection)
    workbook_identities = _verify_workbooks(companion_root)
    recipe = {
        "schema": _SCHEMA,
        "dataset": {
            "doi": DATASET_DOI,
            "version": DATASET_VERSION,
            "archive_file_id": ARCHIVE_FILE_ID,
            "archive_content_id": ARCHIVE_CONTENT_ID,
            "archive_bytes": ARCHIVE_BYTES,
            "publisher_sha256": ARCHIVE_SHA256,
            "license": {"spdx": LICENSE_SPDX, "url": LICENSE_URL, "notice": PUBLISHER_NOTICE},
        },
        "archive": archive_identity,
        "inventory": {
            "member_count": INVENTORY_MEMBER_COUNT,
            "paths_sha256": INVENTORY_PATHS_SHA256,
            "verbose_sha256": INVENTORY_VERBOSE_SHA256,
            "portable_selection_sha256": PORTABLE_SELECTION_SHA256,
        },
        "selected_members": extracted_identities,
        "workbooks": workbook_identities,
        "implementation": _implementation_identity(),
        "policy": {
            "absolute_height_values_read": False,
            "absolute_height_supervision": False,
            "optical_pixels_read": False,
            "imagery_model_input": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    build_id = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = output_root or DATA_WORK / "microtopography" / "rill-diffusion" / "alessio" / "sha256"
    final = parent / build_id
    manifest_path = final / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    staging = parent / f".{build_id}.tmp-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "recipe.json").write_bytes(_canonical_bytes(recipe))

    blockers: set[str] = {"TOTAL_SURFACE_ERROR_UNCALIBRATED", "ABSOLUTE_HEIGHT_SUPERVISION_FORBIDDEN"}
    rill_layers: dict[str, VectorLayer] = {}
    rill_records = {}
    for site, relative in RILL_VECTORS.items():
        layer = _read_vector(extracted_root / relative)
        rill_layers[site] = layer
        metadata = _vector_metadata(layer)
        lines = _line_parts(layer.geometries)
        if layer.crs is None or not layer.crs.is_projected:
            blockers.add(f"{site.upper()}_RILL_CRS_OR_XY_UNIT_UNRESOLVED")
        if not lines:
            blockers.add(f"{site.upper()}_RILL_GEOMETRY_EMPTY")
        rill_records[site] = {"metadata": metadata, "network": _network_descriptors(lines)}

    raster_records = {"dem_metadata_only": {}, "flow": {}}
    dem_metadata = {}
    for relative in DEM_DATASETS:
        with rasterio.open(extracted_root / relative) as dataset:
            record = _raster_metadata(dataset, pixel_values_read=False)
        dem_metadata[relative] = record
        if record["crs"] is None or record["linear_units"] not in {"metre", "meter", "metres", "meters", "m"}:
            blockers.add(f"DEM_CRS_OR_XY_UNIT_UNRESOLVED:{relative}")
    raster_records["dem_metadata_only"] = dem_metadata

    registration = {}
    for site, relative in FLOW_DATASETS.items():
        with rasterio.open(extracted_root / relative) as dataset:
            record = _raster_metadata(dataset, pixel_values_read=True)
            if site == "regional":
                record["flow_alignment"] = None
            else:
                layer = rill_layers[site]
                lines = _line_parts(layer.geometries)
                crs_match = _same_crs(layer, dataset)
                record["flow_alignment"] = _flow_alignment(lines, dataset) if crs_match else None
                registration[site] = {
                    "rill_flow_crs_equal": crs_match,
                    "rill_bounds_overlap_flow": _bounds_overlap(
                        rill_records[site]["metadata"]["bounds"], record["bounds"]
                    ),
                    "valid_flow_sample_fraction": (
                        record["flow_alignment"]["sample_coverage_fraction"]
                        if record["flow_alignment"] is not None
                        else 0.0
                    ),
                }
                if not crs_match or not registration[site]["rill_bounds_overlap_flow"]:
                    blockers.add(f"{site.upper()}_RILL_FLOW_REGISTRATION_FAILED")
                elif registration[site]["valid_flow_sample_fraction"] < 0.95:
                    blockers.add(f"{site.upper()}_RILL_FLOW_SUPPORT_BELOW_95_PERCENT")
        raster_records["flow"][relative] = record

    mask_records = {"bedrock": {}, "vegetation": {}, "geology_contacts": {}, "watersheds": {}}
    for role, mapping in (("bedrock", BEDROCK_VECTORS), ("vegetation", VEGETATION_VECTORS)):
        for site, relative in mapping.items():
            layer = _read_vector(extracted_root / relative)
            metadata = _vector_metadata(layer)
            mask_records[role][site] = metadata
            rill_meta = rill_records[site]["metadata"]
            if layer.crs != rill_layers[site].crs or not _bounds_overlap(metadata["bounds"], rill_meta["bounds"]):
                blockers.add(f"{site.upper()}_{role.upper()}_MASK_REGISTRATION_FAILED")
    for relative in GEOLOGY_CONTACTS:
        layer = _read_vector(extracted_root / relative)
        mask_records["geology_contacts"][relative] = _vector_metadata(layer)
    for relative in WATERSHEDS:
        layer = _read_vector(extracted_root / relative)
        mask_records["watersheds"][relative] = _vector_metadata(layer)

    transect_layers = [_read_vector(extracted_root / relative) for relative in TRANSECT_LOCATIONS]
    gdb_path = extracted_root / TRANSECT_LINES_GDB
    gdb_layers = [str(row[0]) for row in pyogrio.list_layers(gdb_path)]
    transect_layers.extend(_read_vector(gdb_path, layer=name) for name in gdb_layers)
    transect_records = [_vector_metadata(layer) for layer in transect_layers]
    transect_tokens = set().union(*(_attribute_tokens(layer) for layer in transect_layers))

    workbook_records = {}
    workbook_tokens = set()
    for workbook in WORKBOOKS:
        record = _workbook_descriptors(companion_root / workbook.filename)
        workbook_records[workbook.filename] = record
        workbook_tokens.update(record.pop("binding_tokens"))
    shared_binding_tokens = sorted(transect_tokens & workbook_tokens)
    if not shared_binding_tokens:
        blockers.add("WORKBOOK_TRANSECT_IDENTITIES_NOT_BOUND")

    width_depth_spacing = []
    for filename, workbook in workbook_records.items():
        for sheet in workbook["sheets"]:
            for column in sheet["recognized_columns"]:
                header = column["header"].lower()
                if any(token in header for token in ("width", "depth", "spacing")):
                    width_depth_spacing.append(
                        {
                            "workbook": filename,
                            "sheet": sheet["sheet"],
                            "header": column["header"],
                            "numeric": column["numeric"],
                        }
                    )
    if not width_depth_spacing:
        blockers.add("WIDTH_DEPTH_SPACING_COLUMNS_NOT_MACHINE_RESOLVED")

    descriptor = {
        "schema": _SCHEMA,
        "source_role": "morphology_topology_condition_supervision_and_cross_source_audit_only",
        "rill_networks": rill_records,
        "rasters": raster_records,
        "vector_raster_registration": registration,
        "masks_and_conditions": mask_records,
        "transects": {
            "layers": transect_records,
            "workbook_shared_identity_tokens": shared_binding_tokens,
        },
        "workbooks": workbook_records,
        "width_depth_spacing": width_depth_spacing,
        "abort_reasons": sorted(blockers),
        "limitations": {
            "sfm_dem_mean_lidar_difference_m": 0.045,
            "sfm_dem_mean_lidar_difference_is_error_distribution": False,
            "absolute_dem_or_b2_height_loss": False,
            "optical_or_30cm_context_used": False,
            "portable_topology_is_estonia_transfer_truth": False,
        },
    }
    descriptor_path = staging / "descriptors.json"
    descriptor_path.write_bytes(_canonical_bytes(descriptor))
    manifest = {
        "schema": _SCHEMA,
        "build_id": build_id,
        "status": "research_diagnostic_only",
        "result": "not_authorized",
        "recipe_sha256": hashlib.sha256((staging / "recipe.json").read_bytes()).hexdigest(),
        "descriptors": {"path": "descriptors.json", "sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest()},
        "abort_reasons": sorted(blockers),
        "authority": {
            "morphology_topology_descriptors": True,
            "absolute_height_supervision": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
    staging.rename(final)
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify Alessio V3 rill morphology without reading optical or absolute-height pixels."
    )
    parser.add_argument("--archive", type=Path, required=True, help=f"complete {ARCHIVE_NAME}")
    parser.add_argument("--inventory-root", type=Path, required=True, help="retained exact 1,106-member inventory directory")
    parser.add_argument("--extracted-root", type=Path, required=True, help="root of the complete 644-member morphology extraction")
    parser.add_argument("--companion-root", type=Path, required=True, help="directory containing the four verified publisher workbooks")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = qualify_alessio(
            args.archive,
            args.inventory_root,
            args.extracted_root,
            args.companion_root,
            args.output_root,
        )
    except (FileNotFoundError, ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Alessio qualification blocked: {error}\n")
    print(manifest)


if __name__ == "__main__":
    _main()

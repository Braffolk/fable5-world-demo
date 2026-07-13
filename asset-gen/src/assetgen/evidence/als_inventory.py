"""Deterministic file-level inventory for retained ALS repair evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ..config import DATA_IN

_SELECTION_ID = "taevaskoda-als-444679-stage1"
_STRUCTURAL_REPAIR_ROLE = "calibration_only_structural_repair"
_SHA256_LENGTH = 64
_CATEGORICAL_DIMENSIONS = {
    "return_number",
    "number_of_returns",
    "synthetic",
    "key_point",
    "withheld",
    "overlap",
    "scanner_channel",
    "scan_direction_flag",
    "edge_of_flight_line",
    "classification",
    "point_source_id",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"ALS {label} must be a non-empty relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"ALS {label} escapes or is missing: {value}")
    return path


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ALS {label} is not a lowercase SHA-256")
    return value


def _validated_scientific_role(
    root: Path, retained: dict[str, Any]
) -> tuple[str, str]:
    role = retained.get("scientificRole")
    if role is not None:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("ALS retained scientificRole must be a non-empty string")
        return role, "retained"

    if retained.get("morphologyTarget") is not False:
        raise ValueError("ALS inventory cannot infer a scientific role for morphology evidence")
    snapshot = _selection_path(root, retained.get("selectionSnapshot"), "selection snapshot")
    expected_digest = _validate_sha256(
        retained.get("selectionSnapshotSha256"), "selection snapshot digest"
    )
    if _sha256_file(snapshot) != expected_digest:
        raise ValueError("ALS selection snapshot differs from the retained manifest")
    selection = json.loads(snapshot.read_bytes())
    if (
        not isinstance(selection, dict)
        or selection.get("id") != retained.get("selectionId")
        or selection.get("stage") != "stage1-structural-repair"
        or selection.get("morphology_target") is not False
    ):
        raise ValueError("ALS selection snapshot does not prove a structural-repair role")
    return _STRUCTURAL_REPAIR_ROLE, "validated_selection_fallback"


def _validate_artifact_evidence(root: Path, artifact: dict[str, Any]) -> Path:
    required = {
        "year",
        "type",
        "filename",
        "role",
        "relativePath",
        "bytes",
        "sha256",
        "provenance",
        "provenanceSha256",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("ALS retained artifact is missing required evidence fields")
    filename = artifact["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("ALS retained artifact filename is invalid")
    digest = _validate_sha256(artifact["sha256"], f"artifact digest for {filename}")
    if not isinstance(artifact["bytes"], int) or artifact["bytes"] <= 0:
        raise ValueError(f"ALS retained artifact byte count is invalid: {filename}")
    path = _selection_path(root, artifact["relativePath"], f"artifact {filename}")
    expected = Path("raw") / "sha256" / digest / filename
    if Path(artifact["relativePath"]) != expected:
        raise ValueError(f"ALS retained artifact is not content-addressed: {filename}")
    if path.stat().st_size != artifact["bytes"] or _sha256_file(path) != digest:
        raise ValueError(f"ALS artifact differs from retained manifest: {filename}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"ALS artifact digest sidecar is missing or invalid: {filename}")
    provenance = _selection_path(root, artifact["provenance"], f"provenance for {filename}")
    provenance_digest = _validate_sha256(
        artifact["provenanceSha256"], f"provenance digest for {filename}"
    )
    if _sha256_file(provenance) != provenance_digest:
        raise ValueError(f"ALS artifact provenance differs from retained manifest: {filename}")
    return path


def _counter_update(target: Counter[int], values: np.ndarray) -> None:
    unique, counts = np.unique(values, return_counts=True)
    target.update({int(key): int(count) for key, count in zip(unique, counts, strict=True)})


def _range_update(target: dict[str, Any], values: np.ndarray) -> None:
    array = np.asarray(values)
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else np.ones(
        array.shape, dtype=bool
    )
    valid = array[finite]
    target["nonfinite"] += int(array.size - valid.size)
    if valid.size:
        minimum = valid.min().item()
        maximum = valid.max().item()
        target["min"] = minimum if target["min"] is None else min(target["min"], minimum)
        target["max"] = maximum if target["max"] is None else max(target["max"], maximum)


def _vlr_record(vlr: Any) -> dict[str, Any]:
    payload = vlr.record_data_bytes()
    record = {
        "userId": vlr.user_id,
        "recordId": int(vlr.record_id),
        "description": vlr.description,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if vlr.user_id == "LASF_Projection" and hasattr(vlr, "string"):
        record["text"] = vlr.string
    return record


def _inventory_artifact(root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path = _validate_artifact_evidence(root, artifact)
    digest = artifact["sha256"]

    with laspy.open(path) as reader:
        header = reader.header
        dimensions = tuple(header.point_format.dimension_names)
        counters = {
            name: Counter()
            for name in (
                "classification",
                "return_number",
                "number_of_returns",
                "scanner_channel",
                "point_source_id",
            )
            if name in dimensions
        }
        flags = {
            name: 0
            for name in (
                "synthetic",
                "key_point",
                "withheld",
                "overlap",
                "scan_direction_flag",
                "edge_of_flight_line",
            )
            if name in dimensions
        }
        range_names = tuple(
            name
            for name in dimensions
            if name not in {"X", "Y", "Z"} and name not in _CATEGORICAL_DIMENSIONS
        )
        ranges = {name: {"min": None, "max": None, "nonfinite": 0} for name in range_names}
        decoded_points = 0
        for points in reader.chunk_iterator(1_000_000):
            decoded_points += len(points)
            for name, counter in counters.items():
                _counter_update(counter, np.asarray(points[name]))
            for name in flags:
                flags[name] += int(np.count_nonzero(np.asarray(points[name])))
            for name, state in ranges.items():
                _range_update(state, np.asarray(points[name]))
        if decoded_points != header.point_count:
            raise ValueError(
                f"decoded point count {decoded_points} != header {header.point_count}: {path.name}"
            )
        try:
            crs = header.parse_crs()
            crs_wkt = crs.to_wkt() if crs is not None else None
            crs_parse_error = None
        except Exception as error:  # Invalid source WKT is evidence, not an inventory crash.
            crs_wkt = None
            crs_parse_error = {"type": type(error).__name__, "message": str(error)}
        vlrs = [_vlr_record(vlr) for vlr in header.vlrs]
        evlrs = [_vlr_record(vlr) for vlr in (header.evlrs or [])]
        creation_date = header.creation_date.isoformat() if header.creation_date else None
        header_record = {
            "lasVersion": str(header.version),
            "pointFormat": int(header.point_format.id),
            "pointRecordBytes": int(header.point_format.size),
            "extraBytes": int(header.point_format.num_extra_bytes),
            "pointCount": int(header.point_count),
            "globalEncoding": int(header.global_encoding.value),
            "creationDate": creation_date,
            "systemIdentifier": header.system_identifier,
            "generatingSoftware": header.generating_software,
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "mins": [float(value) for value in header.mins],
            "maxs": [float(value) for value in header.maxs],
            "pointsByReturn": [int(value) for value in header.number_of_points_by_return],
            "dimensions": list(dimensions),
            "crsWkt": crs_wkt,
            "crsParseError": crs_parse_error,
            "vlrs": vlrs,
            "evlrs": evlrs,
        }
    return {
        "year": artifact["year"],
        "type": artifact["type"],
        "filename": artifact["filename"],
        "role": artifact["role"],
        "relativePath": artifact["relativePath"],
        "bytes": artifact["bytes"],
        "sha256": digest,
        "header": header_record,
        "counts": {
            name: {str(key): value for key, value in sorted(counter.items())}
            for name, counter in counters.items()
        },
        "flags": flags,
        "ranges": ranges,
    }


def _inventory_retained_als(
    retained_path: Path,
    *,
    strict_primary: bool,
    include_role_source: bool,
    log,
) -> Path:
    als_root = (DATA_IN / "public" / "als").resolve()
    retained_path = retained_path.resolve()
    root = retained_path.parent
    if (
        retained_path.name != "retained.json"
        or root.parent != als_root
        or not retained_path.is_file()
    ):
        raise ValueError("ALS retained manifest must be a selection-root retained.json")
    retained_bytes = retained_path.read_bytes()
    retained = json.loads(retained_bytes)
    if not isinstance(retained, dict):
        raise ValueError("ALS retention manifest must be a JSON object")
    artifacts = retained.get("artifacts")
    if (
        retained.get("selectionId") != root.name
        or retained.get("complete") is not True
        or retained.get("missingFiles") not in (None, [])
        or not isinstance(artifacts, list)
        or not artifacts
    ):
        raise ValueError("ALS retention manifest is incomplete or belongs elsewhere")
    if strict_primary and (
        retained["selectionId"] != _SELECTION_ID or len(artifacts) != 8
    ):
        raise ValueError("Taevaskoda ALS retention manifest is incomplete or belongs elsewhere")
    morphology_target = retained.get("morphologyTarget")
    if not isinstance(morphology_target, bool):
        raise ValueError("ALS retained morphologyTarget must be explicit")
    scientific_role, role_source = _validated_scientific_role(root, retained)
    identities: list[tuple[str, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("ALS retained artifacts contain an invalid record")
        filename = artifact.get("filename")
        relative_path = artifact.get("relativePath")
        if not isinstance(filename, str) or not isinstance(relative_path, str):
            raise ValueError("ALS retained artifacts contain an invalid identity")
        identities.append((filename, relative_path))
    if len(set(identities)) != len(artifacts):
        raise ValueError("ALS retained artifacts contain duplicate or invalid identities")
    retained_sha256 = hashlib.sha256(retained_bytes).hexdigest()
    records = []
    for index, artifact in enumerate(artifacts, start=1):
        records.append(_inventory_artifact(root, artifact))
        log(f"[{index}/{len(artifacts)}] inventoried {artifact['filename']}")
    inventory = {
        "format": 1,
        "selectionId": retained["selectionId"],
        "scientificRole": scientific_role,
        "morphologyTarget": morphology_target,
        "retainedManifest": retained_path.relative_to(root).as_posix(),
        "retainedManifestSha256": retained_sha256,
        "environment": {
            "laspy": laspy.__version__,
            "lazrs": importlib.metadata.version("lazrs"),
        },
        "artifacts": records,
    }
    if include_role_source:
        inventory["scientificRoleSource"] = role_source
    output = root / "inventory" / retained_sha256 / "inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_suffix(".json.part")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    output.with_suffix(".json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )
    return output


def inventory_taevaskoda_als(retained_path: Path | None = None, log=print) -> Path:
    if retained_path is None:
        retained_path = DATA_IN / "public" / "als" / _SELECTION_ID / "retained.json"
        return _inventory_retained_als(
            retained_path,
            strict_primary=True,
            include_role_source=False,
            log=log,
        )
    return _inventory_retained_als(
        retained_path,
        strict_primary=False,
        include_role_source=True,
        log=log,
    )

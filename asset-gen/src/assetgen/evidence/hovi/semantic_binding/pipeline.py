"""Bind retained Hovi photo semantics at their defensible spatial scope."""

from __future__ import annotations

import csv
import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from ....config import DATA_WORK
from ..records import canonical_json_bytes, sha256_bytes, sha256_file
from .model import PUBLISHER_BINDINGS, TARGET_CLASSES, UNMEASURED_TARGET_CLASSES
from .render import render_semantic_scope_sheet

_SCHEMA = "hovi-semantic-scope-binding/1.0.0"
_RECIPE_SCHEMA = "hovi-semantic-scope-binding-recipe/1.0.0"
_QA_SCHEMA = "hovi-semantic-scope-binding-qa/1.0.0"
_PLOT_ID = "HY_SPRUCE4"
_RETENTION_ID = "45f35e2b747b404f0f7b0e594852c43f6493c6368bab738e8f914dffb4297f7e"
_MATERIALIZATION_ID = "ed02d20970b5d9eedf5b36406c947d3cea87c59682bc8853f1a23fac23f35d18"
_SOURCE_FIELDS = ("vasc", "nonvasc", "lichen", "intactlitt", "decomplitt")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi semantic binding {label} must be an object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _load_retained(path: Path) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    raw = _mapping(json.loads(path.read_bytes()), "retained manifest")
    if (
        raw.get("schema_version") != "hovi-retained-evidence/1.0.0"
        or raw.get("retention_id") != _RETENTION_ID
        or raw.get("authorized_scope") != "shared-and-hy-spruce4-only"
    ):
        raise ValueError("Hovi semantic binding requires the frozen HY_SPRUCE4 retention")
    rows: dict[str, Mapping[str, Any]] = {}
    for item in raw.get("files", ()):
        row = _mapping(item, "retained file")
        source_path = row.get("path")
        if isinstance(source_path, str):
            rows[source_path] = row
    return raw, rows


def _retained_source(
    retained_path: Path,
    rows: Mapping[str, Mapping[str, Any]],
    source_path: str,
) -> tuple[Path, dict[str, Any]]:
    row = _mapping(rows.get(source_path), f"retained row {source_path}")
    relative = Path(str(row.get("relative_path")))
    local = (retained_path.parent / relative).resolve()
    expected_sha = row.get("sha256")
    expected_bytes = row.get("bytes")
    if (
        row.get("status") != "verified"
        or relative.is_absolute()
        or not local.is_relative_to(retained_path.parent.resolve())
        or not local.is_file()
        or local.stat().st_size != expected_bytes
        or sha256_file(local) != expected_sha
    ):
        raise ValueError(f"retained Hovi semantic source failed verification: {source_path}")
    return local, {
        "file_id": row.get("file_id"),
        "source_path": source_path,
        "relative_path": relative.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def _one_plot_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        matches = [row for row in csv.DictReader(source) if row.get("plot_ID") == _PLOT_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {_PLOT_ID} row in {path.name}")
    return matches[0]


def _fraction_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    row = _one_plot_row(path)
    quadrats: list[dict[str, Any]] = []
    for number in range(1, 5):
        source_fractions = {
            field: Decimal(row[f"{field}_q{number}"]) for field in _SOURCE_FIELDS
        }
        if any(value < 0 or value > 1 for value in source_fractions.values()):
            raise ValueError("Hovi publisher cover fraction is outside [0,1]")
        if abs(sum(source_fractions.values()) - Decimal(1)) > Decimal("1e-9"):
            raise ValueError("Hovi publisher quadrat fractions do not form one cover partition")
        target = {
            "living_vegetation": float(source_fractions["vasc"]),
            "moss_or_peat": float(source_fractions["nonvasc"]),
            "litter": float(source_fractions["intactlitt"] + source_fractions["decomplitt"]),
            "unknown": float(source_fractions["lichen"]),
        }
        quadrats.append(
            {
                "quadrat": number,
                "approximate_transect_measurement_point": (1, 5, 9, 13)[number - 1],
                "source_fractions": {key: float(value) for key, value in source_fractions.items()},
                "target_fractions": target,
            }
        )
    means: dict[str, float] = {}
    for field in _SOURCE_FIELDS:
        publisher_mean = Decimal(row[f"{field}_mean"])
        derived_mean = sum(Decimal(str(q["source_fractions"][field])) for q in quadrats) / 4
        if abs(publisher_mean - derived_mean) > Decimal("1e-9"):
            raise ValueError(f"Hovi publisher mean does not match quadrats for {field}")
        means[field] = float(publisher_mean)
    return quadrats, means


def _load_photo_quadrats(
    index_path: Path, retained_path: Path
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    index = _mapping(json.loads(index_path.read_bytes()), "photo QA index")
    if (
        index.get("schema_version") != "hovi-semantic-photo-qa-index/1.0.0"
        or index.get("status") != "complete"
        or index.get("plot_id") != _PLOT_ID
        or index.get("retention_id") != _RETENTION_ID
        or index.get("claims", {}).get("metric_registration") is not False
    ):
        raise ValueError("Hovi semantic binding requires the frozen unregistered photo QA")
    result: dict[int, dict[str, Any]] = {}
    for source in index.get("source_photos", ()):
        row = _mapping(source, "photo source")
        if row.get("group") != "quadrat":
            continue
        label = str(row.get("position_label"))
        number = int(label.removeprefix("quadrat "))
        relative = Path("files") / Path(str(row["path"]).lstrip("/"))
        local = (retained_path.parent / relative).resolve()
        if (
            not local.is_file()
            or local.stat().st_size != row.get("bytes")
            or sha256_file(local) != row.get("sha256")
        ):
            raise ValueError(f"Hovi quadrat {number} photo changed")
        result[number] = {
            "photo_path": str(local),
            "photo_source_path": row["path"],
            "photo_sha256": row["sha256"],
            "photo_bytes": row["bytes"],
            "capture_date": row["date_label"],
        }
    if set(result) != {1, 2, 3, 4}:
        raise ValueError("Hovi semantic binding requires four retained quadrat photos")
    return result, {
        "path": str(index_path.resolve()),
        "bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
        "build_id": index.get("build_id"),
    }


def _load_materialization(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _mapping(json.loads(path.read_bytes()), "spatial materialization")
    scans = manifest.get("scans")
    boundary = _mapping(manifest.get("evidenceBoundary"), "materialization boundary")
    if (
        manifest.get("schemaVersion")
        != "hovi-hy-spruce4-spatial-materialization-manifest/1.0.0"
        or not isinstance(scans, list)
        or len(scans) != 16
        or boundary.get("groundFiltered") is not False
        or boundary.get("surfaceClaim") is not False
        or boundary.get("targetTruth") is not False
        or boundary.get("synthesisAuthorized") is not False
    ):
        raise ValueError("Hovi semantic binding requires the unqualified 16-scan materialization")
    if path.parent.name != _MATERIALIZATION_ID:
        raise ValueError("Hovi semantic binding materialization content address changed")
    scan_rows = [
        {
            "ordinal": scan["ordinal"],
            "guid": scan["guid"],
            "publisher_records": scan["publisherRecords"],
            "valid_in_aoi": scan["validInAoi"],
        }
        for scan in scans
    ]
    return {"scans": scan_rows, "evidence_boundary": dict(boundary)}, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "materialization_id": _MATERIALIZATION_ID,
    }


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.py"))
        if path.name != "__main__.py"
    }


def _verify_existing(root: Path, build_id: str) -> Path | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    recipe = root / "recipe.json"
    if not recipe.is_file() or sha256_file(recipe) != build_id:
        raise ValueError("existing Hovi semantic binding recipe identity changed")
    manifest = _mapping(json.loads(manifest_path.read_bytes()), "existing manifest")
    if manifest.get("build_id") != build_id or manifest.get("status") != "complete":
        raise ValueError("existing Hovi semantic binding manifest conflicts")
    for item in manifest.get("artifacts", ()):
        relative = Path(str(item["path"]))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError("existing Hovi semantic binding artifact failed verification")
    return manifest_path


def build_hovi_semantic_binding(
    *,
    retained_path: Path,
    photo_index_path: Path,
    materialization_manifest_path: Path,
    work_root: Path = DATA_WORK,
) -> Path:
    """Publish scoped semantic evidence; never infer point or cell labels."""
    retained_path = retained_path.resolve()
    _, retained_rows = _load_retained(retained_path)
    sources: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for key, source_path in {
        "fractional_cover": "/Laboratory_and_field_data/Fractional_cover_forest_floor/fractional_cover.csv",
        "overview": "/Laboratory_and_field_data/overview.csv",
        "readme": "/Laboratory_and_field_data/README/README.md",
    }.items():
        source_paths[key], sources[key] = _retained_source(
            retained_path, retained_rows, source_path
        )

    fraction_rows, source_means = _fraction_rows(source_paths["fractional_cover"])
    overview = _one_plot_row(source_paths["overview"])
    tls_date = date.fromisoformat(overview["date_TLS"])
    photo_date = date.fromisoformat(overview["date_fractcover"])
    temporal_offset_days = (photo_date - tls_date).days
    if temporal_offset_days != 22:
        raise ValueError("frozen HY_SPRUCE4 TLS-to-quadrat temporal offset changed")
    photo_rows, photo_identity = _load_photo_quadrats(
        photo_index_path.resolve(), retained_path
    )
    materialization, materialization_identity = _load_materialization(
        materialization_manifest_path.resolve()
    )
    quadrats: list[dict[str, Any]] = []
    for row in fraction_rows:
        quadrats.append({**row, **photo_rows[row["quadrat"]]})

    recipe = {
        "schema_version": _RECIPE_SCHEMA,
        "plot_id": _PLOT_ID,
        "method": "publisher_fractional_cover_scoped_binding_v1",
        "retention_id": _RETENTION_ID,
        "inputs": {
            "retained_manifest": {
                "path": str(retained_path),
                "bytes": retained_path.stat().st_size,
                "sha256": sha256_file(retained_path),
            },
            "retained_sources": sources,
            "photo_qa": photo_identity,
            "full_scan_materialization": materialization_identity,
        },
        "ontology": list(TARGET_CLASSES),
        "publisher_bindings": PUBLISHER_BINDINGS,
        "spatial_policy": {
            "quadrat": "publisher fractions bind to the named photo only",
            "plot": "four-quadrat composition summary only",
            "scan": "plot context only; no return labels",
            "view": "no photo-to-TLS camera registration; no labels",
            "cell": "unknown everywhere; no authorized cell mask",
            "unknown_means_zero": False,
        },
        "implementation_sha256": _implementation_hashes(),
    }
    recipe_bytes = canonical_json_bytes(recipe)
    build_id = sha256_bytes(recipe_bytes)
    build_root = work_root / "microtopography" / "hovi" / "semantic-binding" / build_id
    existing = _verify_existing(build_root, build_id)
    if existing is not None:
        return existing
    build_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(build_root / "recipe.json", recipe_bytes)

    plot_target_means = {
        "living_vegetation": source_means["vasc"],
        "moss_or_peat": source_means["nonvasc"],
        "litter": source_means["intactlitt"] + source_means["decomplitt"],
        "unknown": source_means["lichen"],
    }
    evidence = {
        "schema_version": _SCHEMA,
        "build_id": build_id,
        "plot_id": _PLOT_ID,
        "observed_dates": {
            "tls": tls_date.isoformat(),
            "fractional_cover_photos": photo_date.isoformat(),
            "offset_days": temporal_offset_days,
        },
        "publisher_method_boundary": {
            "quadrat_size_m": [1.0, 1.0],
            "transect": "11 m east-west, 1 m south of plot center",
            "quadrat_reference_points": [1, 5, 9, 13],
            "location_status": "approximate_step_estimate",
            "publisher_intended_use": "average cover fraction per plot",
            "metric_photo_to_tls_registration": False,
        },
        "ontology": list(TARGET_CLASSES),
        "class_evidence": {
            "direct_publisher_fraction": ["litter", "living_vegetation"],
            "component_only": {
                "moss_or_peat": "moss measured; peat not inferred",
            },
            "preserved_as_unknown": ["lichen"],
            "not_measured_not_zero": list(UNMEASURED_TARGET_CLASSES),
        },
        "quadrat_evidence": [
            {
                key: value
                for key, value in quadrat.items()
                if key != "photo_path"
            }
            for quadrat in quadrats
        ],
        "plot_composition": {
            "source_mean_fractions": source_means,
            "target_mean_fractions": plot_target_means,
            "authorized_scope": "descriptive_plot_context_only",
        },
        "scan_view_cell_binding": {
            "materialization_id": _MATERIALIZATION_ID,
            "scan_count": len(materialization["scans"]),
            "scans": materialization["scans"],
            "plot_context_attached": True,
            "per_scan_semantic_labels": None,
            "per_view_semantic_labels": None,
            "per_cell_semantic_class": "unknown",
            "authorized_cell_count": 0,
            "cell_mask_emitted": False,
            "reason": (
                "photos lack a metric camera-to-TLS transform; quadrat locations are "
                "approximate and were published for plot averages; acquisition dates differ"
            ),
        },
        "qualification": {
            "go_no": "no_go_cell_semantics",
            "plot_composition_usable": True,
            "surface_semantic_mask_usable": False,
            "target_truth": False,
            "surface_claim": False,
            "synthesis_authorized": False,
            "resume_condition": (
                "registered pixel/geometry annotations or surveyed quadrat footprints with "
                "a frozen class policy and temporal compatibility"
            ),
        },
    }
    evidence_path = build_root / "evidence.json"
    _atomic_write(evidence_path, canonical_json_bytes(evidence))

    qa_path = build_root / "qa" / "01_quadrat_semantic_scope.png"
    panels = render_semantic_scope_sheet(quadrats, qa_path)
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "status": "complete",
        "outputs": [
            {
                "path": qa_path.relative_to(build_root).as_posix(),
                "bytes": qa_path.stat().st_size,
                "sha256": sha256_file(qa_path),
                "panels": panels,
                "interpretation": (
                    "Publisher class fractions are bound to quadrat photos; image pixels are "
                    "not classified and TLS point/view/cell semantics remain unknown."
                ),
            }
        ],
    }
    qa_index_path = build_root / "qa" / "index.json"
    _atomic_write(qa_index_path, canonical_json_bytes(qa_index))
    _atomic_write(
        qa_index_path.with_suffix(".json.sha256"),
        (sha256_file(qa_index_path) + "\n").encode("ascii"),
    )
    artifacts = []
    for path in (evidence_path, qa_path, qa_index_path):
        artifacts.append(
            {
                "path": path.relative_to(build_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "hovi-semantic-scope-binding-manifest/1.0.0",
        "build_id": build_id,
        "status": "complete",
        "result": "no_go_cell_semantics",
        "plot_composition_usable": True,
        "surface_semantic_mask_usable": False,
        "artifacts": artifacts,
    }
    manifest_path = build_root / "manifest.json"
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    _atomic_write(
        manifest_path.with_suffix(".json.sha256"),
        (sha256_file(manifest_path) + "\n").encode("ascii"),
    )
    return manifest_path

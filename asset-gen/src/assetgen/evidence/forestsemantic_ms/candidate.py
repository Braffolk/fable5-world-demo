"""Inventory ForestSemantic-MS as bounded boreal semantic evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import laspy
import numpy as np
from PIL import Image

from ...config import DATA_WORK
from .qa import render_qa

_RETAINED_SCHEMA = "forestsemantic-ms-retention/1.0.0"
_SCHEMA = "forestsemantic-ms-semantic-evidence/1.0.0"
_CELL_M = 0.25
_CLASSES = ("ground", "low_vegetation", "trunk", "branches", "foliage", "woody_debris")
_FEATURES = ("SWIR", "NIR", "Green", "VI")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _load_source(path: Path) -> tuple[dict[str, Any], str, list[tuple[dict[str, Any], Path]]]:
    encoded = path.read_bytes()
    retained = json.loads(encoded)
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("qualification")
        != {
            "state": "retained_raw_candidate",
            "semantic_audit_authorized": True,
            "absolute_target_height_or_error_authorized": False,
        }
    ):
        raise ValueError("ForestSemantic-MS retention is not an authorized complete source")
    split = retained.get("plan", {}).get("split_contract", {})
    if split.get("train") != ["train1.laz", "train2.laz", "train3.laz", "train4.laz"] or split.get(
        "test"
    ) != ["test1.laz", "test2.laz"]:
        raise ValueError("ForestSemantic-MS publisher split changed")
    artifacts = retained.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 6:
        raise ValueError("ForestSemantic-MS retained artifact inventory changed")
    resolved = []
    for raw in artifacts:
        if not isinstance(raw, Mapping) or raw.get("verified") is not True:
            raise ValueError("ForestSemantic-MS artifact is not verified")
        relative = raw.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("ForestSemantic-MS artifact path is invalid")
        artifact = (path.parent / relative).resolve()
        if (
            not artifact.is_relative_to(path.parent.resolve())
            or not artifact.is_file()
            or artifact.stat().st_size != raw.get("bytes")
            or _sha256_file(artifact) != raw.get("sha256")
        ):
            raise ValueError(f"ForestSemantic-MS retained bytes changed: {relative}")
        resolved.append((dict(raw), artifact))
    return dict(retained), hashlib.sha256(encoded).hexdigest(), resolved


def _header_inventory(header: laspy.LasHeader) -> dict[str, Any]:
    dimensions = []
    for dimension in header.point_format.dimensions:
        dimensions.append(
            {
                "name": dimension.name,
                "kind": str(dimension.kind).split(".")[-1],
                "bits": int(dimension.num_bits),
            }
        )
    return {
        "las_version": str(header.version),
        "point_format": int(header.point_format.id),
        "point_record_bytes": int(header.point_format.size),
        "point_count": int(header.point_count),
        "scales_xyz": [float(value) for value in header.scales],
        "offsets_xyz": [float(value) for value in header.offsets],
        "mins_xyz": [float(value) for value in header.mins],
        "maxs_xyz": [float(value) for value in header.maxs],
        "crs": None if header.parse_crs() is None else str(header.parse_crs()),
        "dimensions": dimensions,
        "vlrs": [
            {
                "user_id": vlr.user_id,
                "record_id": int(vlr.record_id),
                "description": vlr.description,
                "payload_sha256": hashlib.sha256(vlr.record_data_bytes()).hexdigest(),
            }
            for vlr in header.vlrs
        ],
    }


def _plot_inventory(source: Mapping[str, Any], path: Path, output: Path) -> tuple[dict[str, Any], np.ndarray]:
    with laspy.open(path) as reader:
        header = reader.header
        header_record = _header_inventory(header)
        xmin, ymin, _ = header.mins
        xmax, ymax, _ = header.maxs
        width = max(1, int(np.floor((xmax - xmin) / _CELL_M)) + 1)
        height = max(1, int(np.floor((ymax - ymin) / _CELL_M)) + 1)
        counts = np.zeros((len(_CLASSES), height, width), dtype=np.uint32)
        semantic_counts = np.zeros(len(_CLASSES), dtype=np.int64)
        standard_classes: Counter[int] = Counter()
        return_numbers: Counter[int] = Counter()
        number_of_returns: Counter[int] = Counter()
        point_sources: Counter[int] = Counter()
        feature_min = np.full((len(_CLASSES), len(_FEATURES)), np.inf)
        feature_max = np.full((len(_CLASSES), len(_FEATURES)), -np.inf)
        feature_sum = np.zeros((len(_CLASSES), len(_FEATURES)), dtype=np.float64)
        decoded = 0
        semantic_nonfinite = 0
        semantic_nonintegral = 0
        semantic_out_of_range = 0
        for points in reader.chunk_iterator(1_000_000):
            n = len(points)
            decoded += n
            labels_raw = np.asarray(points["semantic_GT"], dtype=np.float64)
            finite = np.isfinite(labels_raw)
            integral = finite & (labels_raw == np.rint(labels_raw))
            labels = np.where(integral, labels_raw, -1).astype(np.int16)
            valid = integral & (labels >= 0) & (labels < len(_CLASSES))
            semantic_nonfinite += int(np.count_nonzero(~finite))
            semantic_nonintegral += int(np.count_nonzero(finite & ~integral))
            semantic_out_of_range += int(np.count_nonzero(integral & ~valid))
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            ix = np.clip(np.floor((x - xmin) / _CELL_M).astype(np.int64), 0, width - 1)
            iy = np.clip(np.floor((y - ymin) / _CELL_M).astype(np.int64), 0, height - 1)
            for class_id in range(len(_CLASSES)):
                selected = valid & (labels == class_id)
                count = int(np.count_nonzero(selected))
                semantic_counts[class_id] += count
                if not count:
                    continue
                np.add.at(counts[class_id], (iy[selected], ix[selected]), 1)
                for feature_id, feature in enumerate(_FEATURES):
                    values = np.asarray(points[feature], dtype=np.float64)[selected]
                    feature_min[class_id, feature_id] = min(
                        feature_min[class_id, feature_id], float(values.min())
                    )
                    feature_max[class_id, feature_id] = max(
                        feature_max[class_id, feature_id], float(values.max())
                    )
                    feature_sum[class_id, feature_id] += float(values.sum(dtype=np.float64))
            for values, counter in (
                (points.classification, standard_classes),
                (points.return_number, return_numbers),
                (points.number_of_returns, number_of_returns),
                (points.point_source_id, point_sources),
            ):
                unique, occurrences = np.unique(np.asarray(values), return_counts=True)
                counter.update({int(key): int(value) for key, value in zip(unique, occurrences)})

    if decoded != header.point_count or semantic_counts.sum() != decoded:
        raise ValueError(f"ForestSemantic-MS labels do not cover every point: {path.name}")
    extent_area = float((xmax - xmin) * (ymax - ymin))
    if extent_area <= 0:
        raise ValueError(f"ForestSemantic-MS file has a degenerate XY extent: {path.name}")
    features: dict[str, Any] = {}
    for class_id, class_name in enumerate(_CLASSES):
        features[class_name] = {}
        for feature_id, feature in enumerate(_FEATURES):
            features[class_name][feature] = {
                "min": float(feature_min[class_id, feature_id]),
                "max": float(feature_max[class_id, feature_id]),
                "mean": float(feature_sum[class_id, feature_id] / semantic_counts[class_id]),
            }
    occupied = counts.sum(axis=0) > 0
    artifact = output / f"{path.stem}-semantic-grid.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(artifact.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(
            target,
            semantic_count_u32=counts,
            origin_xy_m=np.asarray((xmin, ymin), dtype=np.float64),
            cell_m=np.asarray(_CELL_M, dtype=np.float64),
        )
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact)
    densities = semantic_counts.astype(np.float64) / extent_area
    inventory = {
        "filename": source["filename"],
        "split": source["split"],
        "source_sha256": source["sha256"],
        "point_count": decoded,
        "header": header_record,
        "semantic_label_storage": "float32_extra_dimension_with_integral_values",
        "semantic_counts": [int(value) for value in semantic_counts],
        "semantic_count_by_name": {
            name: int(semantic_counts[index]) for index, name in enumerate(_CLASSES)
        },
        "semantic_validation": {
            "nonfinite": semantic_nonfinite,
            "nonintegral": semantic_nonintegral,
            "out_of_range": semantic_out_of_range,
            "all_points_labeled": True,
        },
        "standard_las_classification_counts": dict(sorted(standard_classes.items())),
        "return_number_counts": dict(sorted(return_numbers.items())),
        "number_of_returns_counts": dict(sorted(number_of_returns.items())),
        "point_source_id_counts": dict(sorted(point_sources.items())),
        "xy_extent_m": {
            "width": float(xmax - xmin),
            "height": float(ymax - ymin),
            "axis_aligned_area_m2": extent_area,
        },
        "axis_aligned_density_per_m2": {
            name: float(densities[index]) for index, name in enumerate(_CLASSES)
        },
        "terrain_relevant_density_per_m2": {
            "ground": float(densities[0]),
            "low_vegetation": float(densities[1]),
            "woody_debris": float(densities[5]),
        },
        "grid": {
            "cell_m": _CELL_M,
            "shape_yx": [height, width],
            "observed_cells": int(np.count_nonzero(occupied)),
            "observed_area_m2_if_cells_are_counted_in_full": float(np.count_nonzero(occupied) * _CELL_M**2),
            "interpretation": "descriptive direct-return support only; no interpolation or target surface",
        },
        "feature_statistics": features,
        "artifact": {
            "path": f"plots/{artifact.name}",
            "bytes": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
            "fields": ["semantic_count_u32", "origin_xy_m", "cell_m"],
        },
    }
    return inventory, counts


def build_forestsemantic_evidence(
    retained_manifest: Path, output_root: Path | None = None, *, log=print
) -> Path:
    retained, source_manifest_sha256, sources = _load_source(retained_manifest.resolve())
    implementation = {
        "candidate_sha256": _sha256_file(Path(__file__)),
        "qa_sha256": _sha256_file(Path(__file__).with_name("qa.py")),
        "laspy_version": importlib.metadata.version("laspy"),
        "numpy_version": np.__version__,
        "pillow_version": importlib.metadata.version("pillow"),
    }
    recipe = {
        "schema_version": _SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "source_retention_id": retained["retention_id"],
        "publisher_split": retained["plan"]["split_contract"],
        "classes": list(_CLASSES),
        "grid": {"cell_m": _CELL_M, "interpolation": "none"},
        "implementation": implementation,
        "scope": {
            "authorized": "semantic_classifier_development_and_audit_evidence",
            "model_training_performed": False,
            "absolute_height_target": False,
            "target_scale_error_evidence": False,
            "estonia_transfer": False,
        },
    }
    evidence_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    root = (
        output_root
        or DATA_WORK / "microtopography" / "forestsemantic_ms" / "semantic" / "sha256"
    ) / evidence_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_bytes())
        if existing.get("evidence_id") != evidence_id or existing.get("status") != "complete":
            raise ValueError("existing ForestSemantic-MS evidence conflicts")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete ForestSemantic-MS evidence requires inspection")

    plot_records: list[tuple[dict[str, Any], np.ndarray]] = []
    for source, path in sources:
        inventory, counts = _plot_inventory(source, path, root / "plots")
        plot_records.append((inventory, counts))
        log(f"inventoried ForestSemantic-MS {source['filename']}: {inventory['point_count']:,} points")

    qa_records = render_qa(plot_records, root / "qa")
    qa = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions = list(image.size)
        qa.append(
            {
                "index": index,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "dimensions_xy": dimensions,
                "interpretation": record["interpretation"],
            }
        )

    train_counts = np.sum(
        [np.asarray(item["semantic_counts"], dtype=np.int64) for item, _ in plot_records if item["split"] == "train"],
        axis=0,
    )
    test_counts = np.sum(
        [np.asarray(item["semantic_counts"], dtype=np.int64) for item, _ in plot_records if item["split"] == "test"],
        axis=0,
    )
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "evidence_id": evidence_id,
        "recipe": recipe,
        "inventories": [item for item, _ in plot_records],
        "split_summary": {
            "train_files": [item["filename"] for item, _ in plot_records if item["split"] == "train"],
            "test_files": [item["filename"] for item, _ in plot_records if item["split"] == "test"],
            "train_semantic_counts": {
                name: int(train_counts[index]) for index, name in enumerate(_CLASSES)
            },
            "test_semantic_counts": {
                name: int(test_counts[index]) for index, name in enumerate(_CLASSES)
            },
            "test_not_used_for_threshold_or_method_selection": True,
        },
        "qa": qa,
        "decision": {
            "go_no": "go_bounded_semantic_development_evidence",
            "supports": [
                "supervised ground_vs_low_vegetation_vs_woody_debris development",
                "publisher_split semantic audit",
                "class imbalance and spatial support measurement",
            ],
            "does_not_support": [
                "standalone validation of a general Hovi TLS classifier",
                "absolute or normalized 6.25 cm target height",
                "horizontal vertical or registration error",
                "effective target-band transfer",
                "Estonia production transfer",
            ],
            "blocking_evidence": [
                "single Espoonlahti HeliALS dataset and sensor domain",
                "no CRS VLR or vertical datum in the released LAZ files",
                "standard LAS classification return and point-source fields are all zero",
                "only 1021 woody-debris test points and severe class imbalance",
                "no independent annotation-error or positional-error audit",
            ],
            "park_status": "bounded_evidence_complete_no_model_trained",
            "resume_condition": (
                "A separately authorized classifier track supplies Hovi-like TLS training/audit labels, "
                "sensor-domain validation, and a frozen use of the untouched ForestSemantic-MS test split."
            ),
        },
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    log(f"ForestSemantic-MS semantic evidence: {manifest_path}")
    return manifest_path

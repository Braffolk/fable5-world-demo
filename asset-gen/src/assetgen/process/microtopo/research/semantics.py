"""Frozen ForestSemantic geometry-only exclusion model and one-shot audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import laspy
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from .bundle import FrozenBundle, sha256_file
from .contracts import FEATURE_NAMES, SemanticMethod, SemanticResult
from .features import geometry_features


CLASS_NAMES = ("ground", "low_vegetation", "trunk", "branches", "foliage", "woody_debris")


def _semantic_manifest(bundle: FrozenBundle) -> tuple[Path, dict]:
    path = next(
        path for path, _ in bundle.inputs if "forestsemantic_ms/semantic/sha256" in path.as_posix()
    )
    manifest = json.loads(path.read_bytes())
    if manifest.get("status") != "complete" or manifest.get("evidence_id") != "aab4b9b6812d0230b6a6b9d06d1349f89fbb45d82aed7b8c329e39131b515c36":
        raise ValueError("ForestSemantic evidence identity changed")
    return path, manifest


def _source_paths(bundle: FrozenBundle) -> dict[str, Path]:
    _, manifest = _semantic_manifest(bundle)
    retention_id = manifest["recipe"]["source_retention_id"]
    root = bundle.repository_root / "asset-gen/data/in/evidence/forestsemantic_ms" / retention_id / "files"
    expected = {row["filename"]: row["source_sha256"] for row in manifest["inventories"]}
    paths: dict[str, Path] = {}
    for filename, digest in expected.items():
        split = "train" if filename.startswith("train") else "test"
        path = root / split / filename
        if sha256_file(path) != digest:
            raise ValueError(f"ForestSemantic retained source changed: {filename}")
        paths[filename] = path
    return paths


def _read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = laspy.read(path)
    xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
    raw = np.asarray(points["semantic_GT"], dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
        raise ValueError(f"invalid semantic labels in {path.name}")
    labels = raw.astype(np.int8)
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(f"semantic label outside frozen mapping in {path.name}")
    return xyz, labels


def _block_ids(xy: np.ndarray, block_m: float) -> tuple[np.ndarray, int]:
    origin = np.floor(xy.min(axis=0) / block_m) * block_m
    ij = np.floor((xy - origin) / block_m).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    linear = ij[:, 1] * width + ij[:, 0]
    unique, inverse = np.unique(linear, return_inverse=True)
    return inverse.astype(np.int32), len(unique)


def _bootstrap_upper_ratio(
    ground_by_block: np.ndarray,
    forbidden_by_block: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    blocks = len(ground_by_block)
    if blocks == 0 or ground_by_block.sum() + forbidden_by_block.sum() == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(blocks, np.full(blocks, 1.0 / blocks), size=resamples)
    ground = weights @ ground_by_block
    forbidden = weights @ forbidden_by_block
    denominator = ground + forbidden
    ratio = np.divide(forbidden, denominator, out=np.ones_like(forbidden, dtype=np.float64), where=denominator > 0)
    return float(np.quantile(ratio, 0.95, method="higher"))


def _calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    block_count: int,
    xy: np.ndarray,
    method: SemanticMethod,
) -> dict:
    origin = np.floor(xy.min(axis=0) / 0.0625) * 0.0625
    cell_ij = np.floor((xy - origin) / 0.0625).astype(np.int64)
    cell_width = int(cell_ij[:, 0].max()) + 1
    raw_cell = cell_ij[:, 1] * cell_width + cell_ij[:, 0]
    _, cell_inverse = np.unique(raw_cell, return_inverse=True)
    cell_count = int(cell_inverse.max()) + 1
    accepted_max = np.full(cell_count, -np.inf, dtype=np.float64)
    forbidden_max = np.full(cell_count, -np.inf, dtype=np.float64)
    np.maximum.at(accepted_max, cell_inverse, scores)
    np.maximum.at(forbidden_max, cell_inverse[labels != 0], scores[labels != 0])
    cell_block = np.full(cell_count, -1, dtype=np.int32)
    cell_block[cell_inverse] = blocks
    if np.any(cell_block < 0):
        raise AssertionError("calibration cell lacks an 8 m context")
    thresholds = np.unique(np.r_[accepted_max[np.isfinite(accepted_max)], forbidden_max[np.isfinite(forbidden_max)]])
    accepted_sorted = np.sort(accepted_max)
    forbidden_sorted = np.sort(forbidden_max[np.isfinite(forbidden_max)])
    candidates: list[float] = []
    for threshold in thresholds:
        accepted_cells = cell_count - int(np.searchsorted(accepted_sorted, threshold, side="left"))
        forbidden_cells = len(forbidden_sorted) - int(np.searchsorted(forbidden_sorted, threshold, side="left"))
        if accepted_cells and forbidden_cells / accepted_cells <= method.forbidden_ratio_max:
            candidates.append(float(threshold))
    for threshold in candidates:
        accepted = accepted_max >= threshold
        forbidden = forbidden_max >= threshold
        ground_by_block = np.bincount(
            cell_block[accepted & ~forbidden], minlength=block_count
        )
        forbidden_by_block = np.bincount(
            cell_block[forbidden], minlength=block_count
        )
        upper = _bootstrap_upper_ratio(
            ground_by_block,
            forbidden_by_block,
            resamples=method.bootstrap_resamples,
            seed=method.random_state,
        )
        if upper <= method.forbidden_ratio_max:
            return {
                "status": "passed",
                "threshold": threshold,
                "accepted_points": int(np.count_nonzero(scores >= threshold)),
                "accepted_fraction": float(np.count_nonzero(scores >= threshold) / len(scores)),
                "accepted_cells": int(accepted.sum()),
                "accepted_area_m2": float(accepted.sum() * 0.0625**2),
                "forbidden_cells": int(forbidden.sum()),
                "forbidden_area_m2": float(forbidden.sum() * 0.0625**2),
                "forbidden_ratio_ucb95": upper,
                "blocks": block_count,
                "bootstrap_resamples": method.bootstrap_resamples,
            }
    return {
        "status": "failed_no_nonempty_threshold",
        "threshold": None,
        "accepted_points": 0,
        "accepted_fraction": 0.0,
        "accepted_cells": 0,
        "accepted_area_m2": 0.0,
        "forbidden_cells": 0,
        "forbidden_area_m2": 0.0,
        "forbidden_ratio_ucb95": None,
        "blocks": block_count,
        "bootstrap_resamples": method.bootstrap_resamples,
    }


def _extract(
    path: Path, *, log=print
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz, labels = _read_points(path)
    descriptor = geometry_features(xyz)
    blocks, block_count = _block_ids(xyz[:, :2], 8.0)
    log(
        f"geometry descriptors {path.name}: {descriptor.valid.sum():,}/{len(xyz):,} "
        "complete common-feature points"
    )
    return descriptor.values, descriptor.valid, labels, blocks, xyz[:, :2]


def _audit_file(
    model: HistGradientBoostingClassifier,
    path: Path,
    threshold: float,
    envelope: np.ndarray,
    method: SemanticMethod,
    *,
    log=print,
) -> dict:
    features, valid, labels, blocks, xy = _extract(path, log=log)
    in_domain = valid & np.all(
        (features >= envelope[:, 0]) & (features <= envelope[:, 1]), axis=1
    )
    scores = np.full(len(features), np.nan, dtype=np.float64)
    scores[in_domain] = model.predict_proba(features[in_domain])[:, 1]
    accepted = in_domain & (scores >= threshold)
    block_count = int(blocks.max()) + 1
    origin = np.floor(xy.min(axis=0) / 0.0625) * 0.0625
    ij = np.floor((xy - origin) / 0.0625).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    _, inverse = np.unique(ij[:, 1] * width + ij[:, 0], return_inverse=True)
    cell_count = int(inverse.max()) + 1
    cell_accepted = np.zeros(cell_count, dtype=np.bool_)
    cell_forbidden = np.zeros(cell_count, dtype=np.bool_)
    cell_block = np.full(cell_count, -1, dtype=np.int32)
    cell_accepted[inverse[accepted]] = True
    cell_forbidden[inverse[accepted & (labels != 0)]] = True
    cell_block[inverse] = blocks
    ground_by_block = np.bincount(
        cell_block[cell_accepted & ~cell_forbidden], minlength=block_count
    )
    forbidden_by_block = np.bincount(
        cell_block[cell_forbidden], minlength=block_count
    )
    upper = _bootstrap_upper_ratio(
        ground_by_block,
        forbidden_by_block,
        resamples=method.bootstrap_resamples,
        seed=method.random_state,
    )
    return {
        "file": path.name,
        "points": len(features),
        "complete_feature_points": int(valid.sum()),
        "in_domain_points": int(in_domain.sum()),
        "accepted_points": int(accepted.sum()),
        "accepted_ground": int(np.count_nonzero(accepted & (labels == 0))),
        "accepted_forbidden": int(np.count_nonzero(accepted & (labels != 0))),
        "accepted_area_m2": float(cell_accepted.sum() * 0.0625**2),
        "forbidden_area_m2": float(cell_forbidden.sum() * 0.0625**2),
        "woody_debris_points": int(np.count_nonzero(labels == 5)),
        "accepted_woody_debris": int(np.count_nonzero(accepted & (labels == 5))),
        "forbidden_ratio_ucb95": upper,
        "passed": bool(upper <= method.forbidden_ratio_max and np.any(accepted)),
    }


def fit_and_audit_semantics(
    bundle: FrozenBundle,
    output_root: Path,
    method: SemanticMethod = SemanticMethod(),
    *,
    log=print,
) -> SemanticResult:
    paths = _source_paths(bundle)
    fit_features: list[np.ndarray] = []
    fit_labels: list[np.ndarray] = []
    fit_valid: list[np.ndarray] = []
    for name in ("train1.laz", "train2.laz", "train3.laz"):
        features, valid, labels, _, _ = _extract(paths[name], log=log)
        fit_features.append(features)
        fit_labels.append(labels)
        fit_valid.append(valid)
    x = np.concatenate([values[valid] for values, valid in zip(fit_features, fit_valid)])
    raw_labels = np.concatenate([labels[valid] for labels, valid in zip(fit_labels, fit_valid)])
    y = (raw_labels == 0).astype(np.uint8)
    counts = np.bincount(y, minlength=2)
    weights = np.where(y == 0, len(y) / (2 * counts[0]), len(y) / (2 * counts[1]))
    model = HistGradientBoostingClassifier(
        learning_rate=method.learning_rate,
        max_iter=method.max_iter,
        max_leaf_nodes=method.max_leaf_nodes,
        min_samples_leaf=method.min_samples_leaf,
        l2_regularization=method.l2_regularization,
        early_stopping=False,
        random_state=method.random_state,
    )
    model.fit(x, y, sample_weight=weights)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "forestsemantic-geometry-exclusion.joblib"
    joblib.dump(model, model_path, compress=3)

    (
        calibration_features,
        calibration_valid,
        calibration_labels,
        calibration_blocks,
        calibration_xy,
    ) = _extract(
        paths["train4.laz"], log=log
    )
    drift_population = np.concatenate((x, calibration_features[calibration_valid]))
    envelope = np.column_stack(
        (
            np.quantile(drift_population, 0.005, axis=0, method="nearest"),
            np.quantile(drift_population, 0.995, axis=0, method="nearest"),
        )
    )
    calibration_domain = calibration_valid & np.all(
        (calibration_features >= envelope[:, 0]) & (calibration_features <= envelope[:, 1]), axis=1
    )
    calibration_scores = model.predict_proba(calibration_features[calibration_domain])[:, 1]
    calibration = _calibrate_threshold(
        calibration_scores,
        calibration_labels[calibration_domain],
        calibration_blocks[calibration_domain],
        int(calibration_blocks.max()) + 1,
        calibration_xy[calibration_domain],
        method,
    )
    threshold = calibration["threshold"]
    freeze = {
        "schema_version": "forestsemantic-geometry-exclusion-freeze/1.0.0",
        "bundle_id": bundle.bundle_identity,
        "contract_sha256": bundle.contract_sha256,
        "method": method.__dict__,
        "feature_names": FEATURE_NAMES,
        "fit_files": ["train1.laz", "train2.laz", "train3.laz"],
        "calibration_file": "train4.laz",
        "audit_files": ["test1.laz", "test2.laz"],
        "model_sha256": sha256_file(model_path),
        "threshold": threshold,
        "calibration": calibration,
        "feature_envelope_train1_4_0p5_99p5": envelope.tolist(),
        "implementation": {
            "semantics.py": sha256_file(Path(__file__)),
            "features.py": sha256_file(Path(__file__).with_name("features.py")),
            "contracts.py": sha256_file(Path(__file__).with_name("contracts.py")),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "laspy": laspy.__version__,
        },
        "output_schema": {
            "audit": [
                "points",
                "complete_feature_points",
                "in_domain_points",
                "accepted_points",
                "accepted_ground",
                "accepted_forbidden",
                "woody_debris_points",
                "accepted_woody_debris",
                "forbidden_ratio_ucb95",
                "passed",
            ],
            "interpretation": "exclusion prior only; missing or drifted features are unknown",
        },
    }
    freeze_payload = json.dumps(freeze, sort_keys=True, separators=(",", ":")) + "\n"
    freeze_sha = __import__("hashlib").sha256(freeze_payload.encode()).hexdigest()
    freeze_path = output_root / f"semantic-freeze.{freeze_sha}.json"
    freeze_path.write_text(freeze_payload)
    if threshold is None:
        audits = (
            {"file": "test1.laz", "status": "not_opened_calibration_failed", "passed": False},
            {"file": "test2.laz", "status": "not_opened_calibration_failed", "passed": False},
        )
    else:
        # The untouched files are opened only after method, model, envelope, and threshold freeze.
        audits = tuple(
            _audit_file(model, paths[name], threshold, envelope, method, log=log)
            for name in ("test1.laz", "test2.laz")
        )
        audits = tuple({**row, "semantic_freeze_sha256": freeze_sha} for row in audits)
    result = SemanticResult(
        threshold=threshold,
        audit_passed=bool(threshold is not None and all(row["passed"] for row in audits)),
        train_feature_envelope=envelope,
        model_path=model_path,
        calibration=calibration,
        audits=audits,  # type: ignore[arg-type]
    )
    (output_root / "semantic-result.json").write_text(
        json.dumps(
            {
                "schema_version": "forestsemantic-geometry-exclusion/1.0.0",
                "feature_names": FEATURE_NAMES,
                "method": method.__dict__,
                "model_sha256": sha256_file(model_path),
                "semantic_freeze_sha256": freeze_sha,
                "feature_envelope_0p5_99p5": envelope.tolist(),
                "calibration": calibration,
                "audits": audits,
                "audit_passed": result.audit_passed,
                "interpretation": "exclusion prior only; never canonical surface truth",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result

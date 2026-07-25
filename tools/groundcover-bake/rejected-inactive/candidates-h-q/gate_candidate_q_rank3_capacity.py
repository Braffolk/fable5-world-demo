#!/usr/bin/env python3
"""Candidate-Q favourable independent-endpoint affine-rank-3 capacity gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import gate_candidate_p_adaptive_direction as adaptive


base = adaptive.base
ROOT = Path(__file__).resolve().parents[2]
VERSION = "candidate-q-rank3-capacity-v1"
SPEC = "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-Q-RANK3-SCALE-EVENT-SHEETS.md"
TRACK_STARTED = "2026-07-24T20:52:45+0300"
ORIGINAL_TRUTH = (
    "data/work/groundcover-candidate-p-filtered-truth/e3e0a4175b151b89/"
    "a1ff5b80bf4e007d"
)
ADAPTIVE_TRUTH = (
    "data/work/groundcover-candidate-p-adaptive-truth/e3e0a4175b151b89/"
    "7778f2ef61812962"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def pca_rank3(samples: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        origin = np.mean(samples, axis=0)
        centred = samples - origin[None]
        covariance = np.einsum("shwi,shwj->hwij", centred, centred, optimize=True)
    else:
        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-12)
        origin = np.einsum("shw,shwi->hwi", weights, samples, optimize=True) / weight_sum[..., None]
        centred = samples - origin[None]
        covariance = np.einsum("shw,shwi,shwj->hwij", weights, centred, centred, optimize=True)
    _, vectors = np.linalg.eigh(covariance)
    return origin.astype(np.float32), vectors[..., -3:].astype(np.float32)


def scaled_pca_rank3(samples: np.ndarray, tolerance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin, basis = pca_rank3(samples / tolerance)
    return (origin * tolerance).astype(np.float32), (basis * tolerance[:, None]).astype(np.float32)


def quadruple_plane(
    samples: np.ndarray,
    i: int,
    j: int,
    k: int,
    l: int,
) -> tuple[np.ndarray, np.ndarray]:
    origin = samples[i]
    vectors = []
    for sample_index in (j, k, l):
        value = samples[sample_index] - origin
        for previous in vectors:
            value -= np.sum(value * previous, axis=2, keepdims=True) * previous
        value /= np.maximum(np.linalg.norm(value, axis=2, keepdims=True), 1e-12)
        vectors.append(value)
    return origin.astype(np.float32), np.stack(vectors, axis=3).astype(np.float32)


def solve_coordinates(
    scaled_values: np.ndarray,
    scaled_basis: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    gram = np.einsum(
        "shwc,hwci,hwcj->shwij",
        weights,
        scaled_basis,
        scaled_basis,
        optimize=True,
    ).astype(np.float64)
    rhs = np.einsum(
        "shwc,hwci,shwc->shwi",
        weights,
        scaled_basis,
        scaled_values,
        optimize=True,
    ).astype(np.float64)
    scale = np.maximum(np.max(np.abs(gram), axis=(3, 4), keepdims=True), 1.0)
    gram /= scale
    rhs /= scale[..., 0]
    gram += np.eye(3, dtype=np.float64)[None, None, None] * 1e-9
    try:
        result = np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        result = np.einsum(
            "shwij,shwj->shwi",
            np.linalg.pinv(gram, rcond=1e-12),
            rhs,
            optimize=True,
        )
    return result.astype(np.float32)


def optimise_coordinates(
    samples: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    tolerance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled_values = (samples - origin[None]) / tolerance
    scaled_basis = basis / tolerance[:, None]
    weights = np.ones_like(scaled_values, dtype=np.float32)
    coordinates = np.zeros((*samples.shape[:3], 3), dtype=np.float32)
    for power in (2, 8, 24):
        coordinates = solve_coordinates(scaled_values, scaled_basis, weights)
        prediction = np.einsum("hwck,shwk->shwc", scaled_basis, coordinates, optimize=True)
        residual = prediction - scaled_values
        if float(np.max(np.abs(residual))) <= 2e-5:
            unscaled = origin[None] + np.einsum("hwck,shwk->shwc", basis, coordinates, optimize=True)
            return unscaled.astype(np.float32), (unscaled - samples).astype(np.float32), coordinates
        magnitude = np.clip(np.abs(residual) + .02, 1e-3, 8)
        weights = magnitude ** (power - 2)
        weights /= np.maximum(np.max(weights, axis=3, keepdims=True), 1e-12)
    coordinates = solve_coordinates(scaled_values, scaled_basis, weights)
    prediction = origin[None] + np.einsum("hwck,shwk->shwc", basis, coordinates, optimize=True)
    return prediction.astype(np.float32), (prediction - samples).astype(np.float32), coordinates


def fit_rank3_robust(samples: np.ndarray, tolerance: np.ndarray) -> dict[str, Any]:
    target = {
        "objective": np.full(samples.shape[1:3], np.inf, dtype=np.float32),
        "origin": np.zeros(samples.shape[1:], dtype=np.float32),
        "basis": np.zeros((*samples.shape[1:], 3), dtype=np.float32),
        "candidateCount": 0,
    }

    def consider(origin: np.ndarray, basis: np.ndarray) -> None:
        _, residual, _ = optimise_coordinates(samples, origin, basis, tolerance)
        objective = np.max(np.abs(residual) / tolerance, axis=(0, 3))
        take = objective < target["objective"]
        target["objective"] = np.where(take, objective, target["objective"])
        target["origin"] = np.where(take[..., None], origin, target["origin"])
        target["basis"] = np.where(take[..., None, None], basis, target["basis"])
        target["candidateCount"] += 1

    raw_origin, raw_basis = pca_rank3(samples)
    normalized_origin, normalized_basis = scaled_pca_rank3(samples, tolerance)
    consider(raw_origin, raw_basis)
    consider(normalized_origin, normalized_basis)
    for seed_origin, seed_basis in ((raw_origin, raw_basis), (normalized_origin, normalized_basis)):
        origin, basis = seed_origin, seed_basis
        for _ in range(4):
            _, residual, _ = optimise_coordinates(samples, origin, basis, tolerance)
            per_sample = np.max(np.abs(residual) / tolerance, axis=3)
            weights = np.maximum(per_sample, 1e-4) ** 2
            origin, basis = pca_rank3(samples / tolerance, weights)
            origin = (origin * tolerance).astype(np.float32)
            basis = (basis * tolerance[:, None]).astype(np.float32)
            consider(origin, basis)
    for omitted in range(samples.shape[0]):
        subset = np.delete(samples, omitted, axis=0)
        consider(*pca_rank3(subset))
        consider(*scaled_pca_rank3(subset, tolerance))
    for i in range(samples.shape[0] - 3):
        for j in range(i + 1, samples.shape[0] - 2):
            for k in range(j + 1, samples.shape[0] - 1):
                for l in range(k + 1, samples.shape[0]):
                    consider(*quadruple_plane(samples, i, j, k, l))
    objective = target["objective"]
    return {
        "origin": target["origin"],
        "basis": target["basis"],
        "diagnostics": {
            "candidatePlanesPerPhaseVertex": target["candidateCount"],
            "normalisedLinfP50": float(np.quantile(objective, .50)),
            "normalisedLinfP95": float(np.quantile(objective, .95)),
            "normalisedLinfP99": float(np.quantile(objective, .99)),
            "normalisedLinfMax": float(np.max(objective)),
        },
    }


def project(value: np.ndarray, fit: dict[str, Any], tolerance: np.ndarray) -> np.ndarray:
    prediction, _, _ = optimise_coordinates(
        value[None],
        fit["origin"],
        fit["basis"],
        tolerance,
    )
    return prediction[0]


def phase_setup() -> tuple[dict[str, tuple[float, float]], list[tuple[int, int, float]]]:
    offsets: dict[str, tuple[float, float]] = {}
    for phase_name, ox, oz in base.EVALUATION_PHASE_SITES:
        offsets[phase_name] = (ox, oz)
        for millimetres in base.TRANSLATIONS_MM:
            for axis_index, axis in enumerate(base.TRANSLATION_AXES):
                cells = millimetres * .001 / .52 * base.N
                offsets[f"{phase_name}-t{millimetres:g}mm-d{axis_index}"] = (
                    ox + cells * axis[0], oz + cells * axis[1]
                )
    indices = {name: index for index, name in enumerate(offsets)}
    differences = []
    for phase_name, _, _ in base.EVALUATION_PHASE_SITES:
        for millimetres in base.TRANSLATIONS_MM:
            for axis_index in range(len(base.TRANSLATION_AXES)):
                moved = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                differences.append((indices[phase_name], indices[moved], 1.0))
    return offsets, differences


def load_pages(root: Path, prefix: str, count: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text())
    pages = []
    for index in range(count):
        pair = []
        for scale in base.SOURCE_SCALES:
            path = root / f"{prefix}{index:04d}-sigma{scale}.f32"
            expected = next(value["sha256"] for value in manifest["pages"] if value["file"] == path.name)
            data = path.read_bytes()
            if sha(data) != expected:
                raise RuntimeError(f"truth hash mismatch: {path}")
            pair.append(np.frombuffer(data, dtype=np.float32).reshape(base.TRUTH_N, base.TRUTH_N, 4))
        pages.append((pair[0], pair[1]))
    return pages, manifest


def original_dataset() -> dict[str, Any]:
    root = ROOT / ORIGINAL_TRUTH
    manifest = json.loads((root / "manifest.json").read_text())
    directions = manifest["recipe"]["directions"]
    pages, _ = load_pages(root, "entry", len(directions))
    return {
        "name": "original-slope",
        "triangles": base.triangles(),
        "logicalToUnique": {value["key"]: index for index, value in enumerate(directions)},
        "pages": pages,
        "manifest": manifest,
    }


def adaptive_dataset() -> dict[str, Any]:
    root = ROOT / ADAPTIVE_TRUTH
    manifest = json.loads((root / "manifest.json").read_text())
    triangles = adaptive.target_triangles(2)
    unique, logical = adaptive.direction_records(triangles)
    recipe_directions = manifest["recipe"]["directions"]
    if len(unique) != len(recipe_directions):
        raise RuntimeError("adaptive direction count mismatch")
    for generated, recorded in zip(unique, recipe_directions, strict=True):
        if (
            abs(generated["elevation"] - recorded["elevation"]) > 1e-10
            or abs(generated["azimuth"] - recorded["azimuth"]) > 1e-10
        ):
            raise RuntimeError("adaptive direction chart mismatch")
    pages, _ = load_pages(root, "direction", len(unique))
    return {
        "name": "adaptive-logical-polar-p2",
        "triangles": triangles,
        "logicalToUnique": logical,
        "pages": pages,
        "manifest": manifest,
    }


def maximum(values: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    result = []
    for value in values:
        current: Any = value
        for part in path:
            current = current[part]
        result.append(float(current))
    return max(result, default=0.0)


def metric_summary(static: list[dict[str, Any]], translations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "green": all(value["green"] for value in static) and all(value["green"] for value in translations),
        "staticCases": len(static),
        "staticGreen": sum(value["green"] for value in static),
        "translationCases": len(translations),
        "translationGreen": sum(value["green"] for value in translations),
        "coverageP95Worst": maximum(static, ("coverage", "p95")),
        "coverageP99Worst": maximum(static, ("coverage", "p99")),
        "rgbP95Worst": maximum(static, ("premulRgb", "p95")),
        "rgbP99Worst": maximum(static, ("premulRgb", "p99")),
        "connectedWorst": maximum(static, ("largestConnectedFraction",)),
        "translationCoverageP95Worst": maximum(translations, ("coverageP95",)),
        "translationRgbP95Worst": maximum(translations, ("premulRgbP95",)),
        "translationConnectedWorst": maximum(translations, ("largestConnectedFraction",)),
    }


def run() -> Path:
    run_started = time.monotonic()
    phase_offsets, difference_pairs = phase_setup()
    phase_names = list(phase_offsets)
    offset_values = [phase_offsets[name] for name in phase_names]
    datasets = [original_dataset(), adaptive_dataset()]
    evaluations = []
    joint_evaluations = []
    phase_evaluations = []
    diagnostics = []
    qa = []

    for dataset in datasets:
        sampled: dict[tuple[int, str], np.ndarray] = {}

        def joint(unique_index: int, phase: str) -> np.ndarray:
            key = (unique_index, phase)
            if key not in sampled:
                ox, oz = phase_offsets[phase]
                pages = dataset["pages"][unique_index]
                sampled[key] = np.concatenate((
                    base.sample_filtered_truth(pages[0], ox, oz),
                    base.sample_filtered_truth(pages[1], ox, oz),
                ), axis=2)
            return sampled[key]

        coefficients = [
            base.fit_phase_coefficients(
                [joint(index, phase) for phase in phase_names],
                offset_values,
                difference_pairs,
            )
            for index in range(len(dataset["pages"]))
        ]
        print(f"[candidate-q] {dataset['name']} phase coefficients={len(coefficients)}")

        for triangle_index, triangle in enumerate(dataset["triangles"]):
            direction_keys = [
                *[f"{triangle.key}-train{index}" for index in range(len(base.TRAIN_BARY))],
                *[f"{triangle.key}-{name}" for name, *_ in base.HELDOUT_BARY],
            ]
            indices = [dataset["logicalToUnique"][key] for key in direction_keys]
            samples = np.stack([coefficients[index] for index in indices])
            endpoint_fits = [
                fit_rank3_robust(samples[..., scale_index * 4:scale_index * 4 + 4], base.TOLERANCE4)
                for scale_index in range(2)
            ]
            joint_fit = fit_rank3_robust(samples, base.TOLERANCE8)
            diagnostics.append({
                "dataset": dataset["name"],
                "triangle": triangle.key,
                "independentEndpoints": [value["diagnostics"] for value in endpoint_fits],
                "sharedCoordinateR8": joint_fit["diagnostics"],
            })
            print(f"[candidate-q] {dataset['name']} triangle {triangle_index+1}/{len(dataset['triangles'])}")
            for heldout_index, (site, *_) in enumerate(base.HELDOUT_BARY):
                sample_index = len(base.TRAIN_BARY) + heldout_index
                unique_index = indices[sample_index]
                endpoint_vertices = [
                    project(
                        samples[sample_index, ..., scale_index * 4:scale_index * 4 + 4],
                        endpoint_fits[scale_index],
                        base.TOLERANCE4,
                    )
                    for scale_index in range(2)
                ]
                joint_vertices = project(samples[sample_index], joint_fit, base.TOLERANCE8)
                phase_vertices = coefficients[unique_index]
                for phase_name, ox, oz in base.EVALUATION_PHASE_SITES:
                    truth = joint(unique_index, phase_name)
                    independent_prediction = np.concatenate([
                        base.phase_interpolate(value, ox, oz) for value in endpoint_vertices
                    ], axis=2)
                    joint_prediction = base.phase_interpolate(joint_vertices, ox, oz)
                    phase_prediction = base.phase_interpolate(phase_vertices, ox, oz)
                    independent_static = []
                    joint_static = []
                    phase_static = []
                    independent_translations = []
                    joint_translations = []
                    phase_translations = []
                    for scale_index, scale in enumerate(base.SOURCE_SCALES):
                        span = slice(scale_index * 4, scale_index * 4 + 4)
                        result = base.score(truth[..., span], independent_prediction[..., span])
                        independent_static.append({"scale": scale, "score": result})
                        joint_static.append({"scale": scale, "score": base.score(truth[..., span], joint_prediction[..., span])})
                        phase_static.append({"scale": scale, "score": base.score(truth[..., span], phase_prediction[..., span])})
                        severity = max(
                            result["coverage"]["p95"] / .08,
                            result["premulRgb"]["p95"] / .06,
                            result["largestConnectedFraction"] / .01,
                        )
                        qa.append((severity, f"{dataset['name']}-{triangle.key}-{site}-{phase_name}-sigma{scale}", truth[..., span], independent_prediction[..., span]))
                    for millimetres in base.TRANSLATIONS_MM:
                        for axis_index, axis in enumerate(base.TRANSLATION_AXES):
                            moved_name = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                            moved_truth = joint(unique_index, moved_name)
                            cells = millimetres * .001 / .52 * base.N
                            moved_independent = np.concatenate([
                                base.phase_interpolate(value, ox + cells * axis[0], oz + cells * axis[1])
                                for value in endpoint_vertices
                            ], axis=2)
                            moved_joint = base.phase_interpolate(joint_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            moved_phase = base.phase_interpolate(phase_vertices, ox + cells * axis[0], oz + cells * axis[1])
                            for scale_index, scale in enumerate(base.SOURCE_SCALES):
                                span = slice(scale_index * 4, scale_index * 4 + 4)
                                independent_translations.append({"scale": scale, "millimetres": millimetres, "axis": axis_index, "score": base.translation_score(truth[..., span], moved_truth[..., span], independent_prediction[..., span], moved_independent[..., span])})
                                joint_translations.append({"scale": scale, "millimetres": millimetres, "axis": axis_index, "score": base.translation_score(truth[..., span], moved_truth[..., span], joint_prediction[..., span], moved_joint[..., span])})
                                phase_translations.append({"scale": scale, "millimetres": millimetres, "axis": axis_index, "score": base.translation_score(truth[..., span], moved_truth[..., span], phase_prediction[..., span], moved_phase[..., span])})
                    common = {"dataset": dataset["name"], "triangle": triangle.key, "site": site, "phaseSite": phase_name}
                    evaluations.append({**common, "scaleScores": independent_static, "translations": independent_translations})
                    joint_evaluations.append({**common, "scaleScores": joint_static, "translations": joint_translations})
                    phase_evaluations.append({**common, "scaleScores": phase_static, "translations": phase_translations})

    def summarise(values: list[dict[str, Any]], dataset: str | None = None) -> dict[str, Any]:
        selected = values if dataset is None else [value for value in values if value["dataset"] == dataset]
        static = [item["score"] for value in selected for item in value["scaleScores"]]
        translations = [item["score"] for value in selected for item in value["translations"]]
        return metric_summary(static, translations)

    independent_metrics = summarise(evaluations)
    joint_metrics = summarise(joint_evaluations)
    phase_metrics = summarise(phase_evaluations)
    by_dataset = {
        dataset["name"]: {
            "independentEndpoints": summarise(evaluations, dataset["name"]),
            "sharedCoordinateR8": summarise(joint_evaluations, dataset["name"]),
            "phaseOnly": summarise(phase_evaluations, dataset["name"]),
        }
        for dataset in datasets
    }
    gap = max(
        independent_metrics["coverageP95Worst"] / .08,
        independent_metrics["coverageP99Worst"] / .20,
        independent_metrics["rgbP95Worst"] / .06,
        independent_metrics["rgbP99Worst"] / .15,
        independent_metrics["connectedWorst"] / .01,
        independent_metrics["translationCoverageP95Worst"] / .06,
        independent_metrics["translationRgbP95Worst"] / .06,
        independent_metrics["translationConnectedWorst"] / .01,
    )
    verdict = "GREEN-NECESSARY-BOUND-ONLY" if independent_metrics["green"] else ("INCONCLUSIVE-NARROW-RED" if gap < 1.5 else "RED")
    source_sha = sha((ROOT / base.SOURCE).read_bytes())
    recipe = {
        "version": VERSION,
        "sourceSha256": source_sha,
        "scriptSha256": sha(Path(__file__).read_bytes()),
        "specPath": SPEC,
        "preGateSpecSha256": sha((ROOT / SPEC).read_bytes()),
        "baseHarnessSha256": sha(adaptive.BASE_PATH.read_bytes()),
        "adaptiveHarnessSha256": sha(Path(adaptive.__file__).read_bytes()),
        "truthManifests": {
            dataset["name"]: {
                "recipeSha256": dataset["manifest"]["recipeSha256"],
                "manifestSha256": sha((ROOT / (ORIGINAL_TRUTH if dataset["name"] == "original-slope" else ADAPTIVE_TRUTH) / "manifest.json").read_bytes()),
            }
            for dataset in datasets
        },
        "phaseCoefficientFit": "global periodic triangular-FE least squares over all three exact phase sites and all static/translation difference observations",
        "independentEndpointOracle": "separate threshold-normalized robust affine rank3 in R4 for sigma4 and sigma16; arbitrary per-direction coordinates; all scored directions leaked into fit",
        "sharedDiagnostic": "threshold-normalized robust affine rank3 in joint R8 with shared coordinates",
        "candidateSearch": "raw/scaled PCA, high-p IRLS, raw/scaled leave-one-out, every four-observation affine plane",
        "limits": base.LIMITS,
        "newRayCount": 0,
    }
    recipe_sha = canonical_sha(recipe)
    output = ROOT / "data/work/groundcover-candidate-q-rank3-capacity" / source_sha[:16] / recipe_sha[:16]
    qa_root = output / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    qa.sort(key=lambda value: value[0], reverse=True)
    images = []
    for index, (_, key, truth, prediction) in enumerate(qa[:5], start=1):
        path = qa_root / f"{index:03d}-{key}.png"
        base.write_qa(path, f"Candidate Q rank3: {key}", truth, prediction)
        images.append({"file": path.name, "sha256": sha(path.read_bytes()), "dimensions": [768, 290], "interpretation": "truth | favourable independent-endpoint rank3 | max-channel error"})
    track_seconds = time.time() - time.mktime(time.strptime(TRACK_STARTED, "%Y-%m-%dT%H:%M:%S%z"))
    report = {
        "schema": VERSION,
        "verdict": verdict,
        "recipe": {**recipe, "recipeSha256": recipe_sha},
        "wallTime": {"trackStarted": TRACK_STARTED, "trackSeconds": track_seconds, "runSeconds": time.monotonic() - run_started, "budgetSeconds": 3600, "withinBudget": track_seconds <= 3600},
        "independentEndpointRank3": {"metrics": independent_metrics, "worstNormalisedFrozenGap": gap, "evaluations": evaluations},
        "sharedCoordinateR8Rank3": {"metrics": joint_metrics, "evaluations": joint_evaluations},
        "phaseOnly": {"metrics": phase_metrics, "evaluations": phase_evaluations},
        "metricsByDataset": by_dataset,
        "fitDiagnostics": diagnostics,
        "qa": images,
        "nextStages": {"sheetFit": independent_metrics["green"], "quantisation": False, "runtime": False},
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    (qa_root / "index.json").write_text(json.dumps({"schema": f"{VERSION}-qa-index", "recipeSha256": recipe_sha, "reportSha256": sha(report_text.encode()), "images": images}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "output": str(output.relative_to(ROOT)), "independent": independent_metrics, "sharedR8": joint_metrics, "phaseOnly": phase_metrics, "wallSeconds": time.monotonic() - run_started, "reportSha256": sha(report_text.encode())}, indent=2))
    return output


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Targeted Candidate-P adaptive-direction affine-rank-2 capacity gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).with_name("gate_candidate_p_rank2_bound.py")
MODULE_SPEC = importlib.util.spec_from_file_location("candidate_p_rank2_bound", BASE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = base
MODULE_SPEC.loader.exec_module(base)

VERSION = "candidate-p-adaptive-direction-v1"
TRACK_STARTED = "2026-07-24T20:11:24+0300"
MATH_NOTE = "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-ADAPTIVE-DIRECTION-LATTICE-MATH.md"
PARENT_REPORT = (
    "data/work/groundcover-candidate-p-rank2-bound/e3e0a4175b151b89/"
    "21a11246991582a6/report.json"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def logical_polar_quad(
    key: str,
    elevation_low: float,
    elevation_high: float,
    azimuth_low: float,
    azimuth_high: float,
    subdivision: int,
) -> list[Any]:
    """Triangulate one logical-polar rectangle with one frozen diagonal.

    Vertices are stored as (elevation, unwrapped azimuth), not slope vectors.
    Both barycentric heldouts and the analytic runtime witness therefore live
    in exactly the same chart.  Within every child, the diagonal joins the
    low-elevation/low-azimuth and high-elevation/high-azimuth corners.
    """
    result = []
    for elevation_index in range(subdivision):
        e0 = elevation_low + (elevation_high - elevation_low) * elevation_index / subdivision
        e1 = elevation_low + (elevation_high - elevation_low) * (elevation_index + 1) / subdivision
        for azimuth_index in range(subdivision):
            a0 = azimuth_low + (azimuth_high - azimuth_low) * azimuth_index / subdivision
            a1 = azimuth_low + (azimuth_high - azimuth_low) * (azimuth_index + 1) / subdivision
            q0 = (e0, a0)
            q1 = (e0, a1)
            q2 = (e1, a1)
            q3 = (e1, a0)
            cell = elevation_index * subdivision + azimuth_index
            result.extend((
                base.Triangle(f"{key}-c{cell}-t0", key, 0, (q0, q1, q2)),
                base.Triangle(f"{key}-c{cell}-t1", key, 1, (q0, q2, q3)),
            ))
    return result


def target_triangles(subdivision: int) -> list[Any]:
    return [
        *logical_polar_quad("grazing-control", .25, 2.0, 67.5, 90.0, 1),
        *logical_polar_quad("standing-adaptive", 10.0, 18.0, 247.5, 270.0, subdivision),
    ]


def logical_polar_direction(triangle: Any, bary: tuple[float, float, float]) -> tuple[float, float]:
    elevation = sum(bary[index] * triangle.vertices[index][0] for index in range(3))
    azimuth = sum(bary[index] * triangle.vertices[index][1] for index in range(3)) % 360.0
    return elevation, azimuth


def direction_records(triangles: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logical = []
    unique: list[dict[str, Any]] = []
    unique_by_coordinate: dict[tuple[float, float], int] = {}
    logical_to_unique: dict[str, int] = {}
    for triangle in triangles:
        sites = [
            *[(f"train{index}", tuple(bary), "train") for index, bary in enumerate(base.TRAIN_BARY)],
            *[(name, tuple(coords), "heldout") for name, *coords in base.HELDOUT_BARY],
        ]
        for site, bary, kind in sites:
            elevation, azimuth = logical_polar_direction(triangle, bary)
            key = f"{triangle.key}-{site}"
            coordinate = (round(elevation, 11), round(azimuth % 360, 11))
            if coordinate not in unique_by_coordinate:
                unique_by_coordinate[coordinate] = len(unique)
                unique.append({
                    "key": f"direction{len(unique):04d}",
                    "elevation": elevation,
                    "azimuth": azimuth % 360,
                })
            unique_index = unique_by_coordinate[coordinate]
            logical.append({
                "key": key,
                "triangle": triangle.key,
                "site": site,
                "kind": kind,
                "uniqueIndex": unique_index,
            })
            logical_to_unique[key] = unique_index
    return unique, logical_to_unique


def cached_direction_pages(source_sha: str) -> dict[tuple[float, float], tuple[Path, Path]]:
    result: dict[tuple[float, float], tuple[Path, Path]] = {}
    root = ROOT / "data/work/groundcover-candidate-p-filtered-truth" / source_sha[:16]
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
            recipe = manifest["recipe"]
            if (
                recipe.get("sourceSha256") != source_sha
                or recipe.get("resolution") != base.TRUTH_N
                or recipe.get("referenceY") != base.REFERENCE_Y
                or tuple(recipe.get("physicalFilterFullWidthsAt256", ())) != base.PHYSICAL_FILTER_WIDTHS_256
            ):
                continue
            pages = {(value["direction"], value["scale"]): manifest_path.parent / value["file"] for value in manifest["pages"]}
            for direction in recipe["directions"]:
                coordinate = (
                    round(float(direction["elevation"]), 11),
                    round(float(direction["azimuth"]) % 360, 11),
                )
                pair = (pages.get((direction["key"], 4)), pages.get((direction["key"], 16)))
                if all(path is not None and path.exists() for path in pair):
                    result.setdefault(coordinate, pair)  # type: ignore[arg-type]
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return result


def prepare_truth(unique: list[dict[str, Any]], source_sha: str) -> tuple[Path, dict[str, Any], dict[str, int]]:
    exporter = ROOT / "tools/groundcover-bake/export-candidate-o-truth.ts"
    recipe = {
        "version": f"{VERSION}-filtered-truth-v1",
        "sourceSha256": source_sha,
        "exporterSha256": sha(exporter.read_bytes()),
        "resolution": base.TRUTH_N,
        "referenceY": base.REFERENCE_Y,
        "physicalFilterFullWidthsAt256": base.PHYSICAL_FILTER_WIDTHS_256,
        "fractionalFilterFullWidthsAtTruthResolution": base.FILTER_WIDTHS_AT_TRUTH_N,
        "directions": unique,
    }
    recipe_sha = canonical_sha(recipe)
    cache = (
        ROOT / "data/work/groundcover-candidate-p-adaptive-truth"
        / source_sha[:16] / recipe_sha[:16]
    )
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["recipeSha256"] != recipe_sha:
            raise RuntimeError("adaptive truth recipe mismatch")
        return cache, manifest, {value["key"]: index for index, value in enumerate(unique)}

    cache.mkdir(parents=True, exist_ok=True)
    reusable = cached_direction_pages(source_sha)
    missing = []
    provenance = []
    for index, direction in enumerate(unique):
        coordinate = (round(direction["elevation"], 11), round(direction["azimuth"] % 360, 11))
        pair = reusable.get(coordinate)
        if pair is None:
            missing.append((index, direction))
            continue
        for scale, source in zip(base.SOURCE_SCALES, pair, strict=True):
            target = cache / f"direction{index:04d}-sigma{scale}.f32"
            shutil.copyfile(source, target)
            provenance.append({"direction": direction["key"], "scale": scale, "source": str(source.relative_to(ROOT)), "kind": "exact-direction cache reuse"})

    print(f"[adaptive] directions={len(unique)} reused={len(unique)-len(missing)} exact-export={len(missing)}")
    if missing:
        config = {
            "source": base.SOURCE,
            "resolution": base.TRUTH_N,
            "referenceY": base.REFERENCE_Y,
            "radii": [0],
            "entries": [
                {
                    "key": f"missing{position:04d}",
                    "elevationDegrees": value["elevation"],
                    "azimuthDegrees": value["azimuth"],
                    "phaseOffsetX": 0.0,
                    "phaseOffsetZ": 0.0,
                }
                for position, (_, value) in enumerate(missing)
            ],
        }
        with tempfile.TemporaryDirectory(prefix="candidate-p-adaptive-") as temporary:
            temporary_root = Path(temporary)
            config_path = temporary_root / "config.json"
            truth_root = temporary_root / "truth"
            config_path.write_text(json.dumps(config))
            subprocess.run(
                ["node", "--import", "tsx", str(exporter), "--config", str(config_path), "--output", str(truth_root)],
                cwd=ROOT,
                check=True,
            )
            for position, (index, direction) in enumerate(missing):
                point = np.fromfile(
                    truth_root / f"missing{position:04d}-r0.f32",
                    dtype=np.float32,
                ).reshape(base.TRUTH_N, base.TRUTH_N, 4)
                for scale, width in zip(base.SOURCE_SCALES, base.FILTER_WIDTHS_AT_TRUTH_N, strict=True):
                    target = cache / f"direction{index:04d}-sigma{scale}.f32"
                    base.periodic_fractional_box_filter(point, width).tofile(target)
                    provenance.append({"direction": direction["key"], "scale": scale, "kind": "new exact BVH export"})

    pages = []
    for index, direction in enumerate(unique):
        for scale in base.SOURCE_SCALES:
            path = cache / f"direction{index:04d}-sigma{scale}.f32"
            pages.append({
                "direction": direction["key"],
                "scale": scale,
                "file": path.name,
                "sha256": sha(path.read_bytes()),
                "shape": [base.TRUTH_N, base.TRUTH_N, 4],
            })
    manifest = {
        "schema": f"{VERSION}-truth-manifest",
        "recipe": recipe,
        "recipeSha256": recipe_sha,
        "pages": pages,
        "provenance": provenance,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return cache, manifest, {value["key"]: index for index, value in enumerate(unique)}


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


def run(subdivision: int) -> Path:
    started = time.monotonic()
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    source_path = ROOT / base.SOURCE
    source_sha = sha(source_path.read_bytes())
    triangles = target_triangles(subdivision)
    unique, logical_to_unique = direction_records(triangles)
    truth_root, truth_manifest, _ = prepare_truth(unique, source_sha)

    phase_offsets: dict[str, tuple[float, float]] = {}
    for phase_name, ox, oz in base.EVALUATION_PHASE_SITES:
        phase_offsets[phase_name] = (ox, oz)
        for millimetres in base.TRANSLATIONS_MM:
            for axis_index, axis in enumerate(base.TRANSLATION_AXES):
                cells = millimetres * .001 / .52 * base.N
                phase_offsets[f"{phase_name}-t{millimetres:g}mm-d{axis_index}"] = (
                    ox + cells * axis[0], oz + cells * axis[1]
                )
    phase_names = list(phase_offsets)
    phase_indices = {name: index for index, name in enumerate(phase_names)}
    difference_pairs = []
    for phase_name, _, _ in base.EVALUATION_PHASE_SITES:
        for millimetres in base.TRANSLATIONS_MM:
            for axis_index in range(len(base.TRANSLATION_AXES)):
                moved = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                difference_pairs.append((phase_indices[phase_name], phase_indices[moved], 1.0))

    source_pages = []
    for index, direction in enumerate(unique):
        pair = []
        for scale in base.SOURCE_SCALES:
            path = truth_root / f"direction{index:04d}-sigma{scale}.f32"
            expected = next(value["sha256"] for value in truth_manifest["pages"] if value["file"] == path.name)
            data = path.read_bytes()
            if sha(data) != expected:
                raise RuntimeError(f"truth hash mismatch: {path}")
            pair.append(np.frombuffer(data, dtype=np.float32).reshape(base.TRUTH_N, base.TRUTH_N, 4))
        source_pages.append(tuple(pair))

    sampled: dict[tuple[int, str], np.ndarray] = {}

    def joint(index: int, phase: str) -> np.ndarray:
        key = (index, phase)
        if key not in sampled:
            ox, oz = phase_offsets[phase]
            sampled[key] = np.concatenate((
                base.sample_filtered_truth(source_pages[index][0], ox, oz),
                base.sample_filtered_truth(source_pages[index][1], ox, oz),
            ), axis=2)
        return sampled[key]

    coefficients = []
    offsets = [phase_offsets[name] for name in phase_names]
    for index in range(len(unique)):
        coefficients.append(base.fit_phase_coefficients(
            [joint(index, name) for name in phase_names],
            offsets,
            difference_pairs,
        ))
    print(f"[adaptive] fitted phase coefficients for {len(unique)} unique directions")

    evaluations = []
    phase_only_evaluations = []
    fit_diagnostics = []
    qa = []
    for triangle in triangles:
        direction_keys = [
            *[f"{triangle.key}-train{index}" for index in range(len(base.TRAIN_BARY))],
            *[f"{triangle.key}-{name}" for name, *_ in base.HELDOUT_BARY],
        ]
        direction_indices = [logical_to_unique[key] for key in direction_keys]
        samples = np.stack([coefficients[index] for index in direction_indices])
        fit = base.fit_rank2_robust(samples)
        origin, basis = fit["joint"]
        fit_diagnostics.append({"triangle": triangle.key, "fit": fit["diagnostics"]})
        for heldout_index, (site, *_) in enumerate(base.HELDOUT_BARY):
            sample_index = len(base.TRAIN_BARY) + heldout_index
            unique_index = direction_indices[sample_index]
            projected = base.project(samples[sample_index], origin, basis, base.TOLERANCE8)
            phase_only = coefficients[unique_index]
            for phase_name, ox, oz in base.EVALUATION_PHASE_SITES:
                truth = joint(unique_index, phase_name)
                prediction = base.phase_interpolate(projected, ox, oz)
                phase_prediction = base.phase_interpolate(phase_only, ox, oz)
                static_scores = []
                phase_static_scores = []
                translations = []
                phase_translations = []
                for scale_index, scale in enumerate(base.SOURCE_SCALES):
                    span = slice(scale_index * 4, scale_index * 4 + 4)
                    result = base.score(truth[..., span], prediction[..., span])
                    static_scores.append({"scale": scale, "score": result})
                    phase_static_scores.append({"scale": scale, "score": base.score(truth[..., span], phase_prediction[..., span])})
                    severity = max(
                        result["coverage"]["p95"] / .08,
                        result["premulRgb"]["p95"] / .06,
                        result["largestConnectedFraction"] / .01,
                    )
                    qa.append((severity, f"{triangle.key}-{site}-{phase_name}-sigma{scale}", truth[..., span], prediction[..., span]))
                for millimetres in base.TRANSLATIONS_MM:
                    for axis_index, axis in enumerate(base.TRANSLATION_AXES):
                        moved_name = f"{phase_name}-t{millimetres:g}mm-d{axis_index}"
                        moved_truth = joint(unique_index, moved_name)
                        cells = millimetres * .001 / .52 * base.N
                        moved_prediction = base.phase_interpolate(projected, ox + cells * axis[0], oz + cells * axis[1])
                        phase_moved = base.phase_interpolate(phase_only, ox + cells * axis[0], oz + cells * axis[1])
                        for scale_index, scale in enumerate(base.SOURCE_SCALES):
                            span = slice(scale_index * 4, scale_index * 4 + 4)
                            translations.append({"scale": scale, "millimetres": millimetres, "axis": axis_index, "score": base.translation_score(truth[..., span], moved_truth[..., span], prediction[..., span], moved_prediction[..., span])})
                            phase_translations.append({"scale": scale, "millimetres": millimetres, "axis": axis_index, "score": base.translation_score(truth[..., span], moved_truth[..., span], phase_prediction[..., span], phase_moved[..., span])})
                group = "grazing-control" if triangle.key.startswith("grazing") else "standing-child"
                evaluations.append({"group": group, "triangle": triangle.key, "site": site, "phaseSite": phase_name, "scaleScores": static_scores, "translations": translations})
                phase_only_evaluations.append({"group": group, "triangle": triangle.key, "site": site, "phaseSite": phase_name, "scaleScores": phase_static_scores, "translations": phase_translations})

    def summarise(values: list[dict[str, Any]]) -> dict[str, Any]:
        static = [entry["score"] for value in values for entry in value["scaleScores"]]
        translations = [entry["score"] for value in values for entry in value["translations"]]
        return metric_summary(static, translations)

    groups = {
        name: summarise([value for value in evaluations if value["group"] == name])
        for name in ("standing-child", "grazing-control")
    }
    phase_groups = {
        name: summarise([value for value in phase_only_evaluations if value["group"] == name])
        for name in ("standing-child", "grazing-control")
    }
    standing = groups["standing-child"]
    standing_gap = max(
        standing["coverageP95Worst"] / .08,
        standing["coverageP99Worst"] / .20,
        standing["rgbP95Worst"] / .06,
        standing["rgbP99Worst"] / .15,
        standing["connectedWorst"] / .01,
        standing["translationCoverageP95Worst"] / .06,
        standing["translationRgbP95Worst"] / .06,
        standing["translationConnectedWorst"] / .01,
    )
    verdict = "GREEN-LOCAL-CHILD-BOUND" if standing["green"] else ("NARROW-RED" if standing_gap < 1.5 else "RED")
    elapsed = time.monotonic() - started
    track_started_epoch = dt.datetime.strptime(TRACK_STARTED, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    track_elapsed = time.time() - track_started_epoch
    recipe = {
        "version": VERSION,
        "subdivision": subdivision,
        "sourceSha256": source_sha,
        "scriptSha256": sha(Path(__file__).read_bytes()),
        "baseHarnessSha256": sha(BASE_PATH.read_bytes()),
        "mathNotePath": MATH_NOTE,
        "mathContract": "logical-polar elevation/azimuth triangles; conforming local p-way red/green refinement; exact pole/seam/one-sided-horizon topology",
        "parentReportSha256": sha((ROOT / PARENT_REPORT).read_bytes()),
        "truthRecipeSha256": truth_manifest["recipeSha256"],
        "logicalTriangles": len(triangles),
        "directionChart": "logical polar (elevation, unwrapped azimuth); all child vertices and barycentric heldouts generated in this exact chart",
        "uniqueDirections": len(unique),
        "phaseSites": base.EVALUATION_PHASE_SITES,
        "translationsMillimetres": base.TRANSLATIONS_MM,
        "translationAxes": base.TRANSLATION_AXES,
        "limits": base.LIMITS,
        "oracle": "globally optimized periodic phase coefficients plus preserved threshold-normalized high-p robust affine-rank2 candidate oracle; no coupled L2 alternation",
    }
    recipe_sha = canonical_sha(recipe)
    output = ROOT / "data/work/groundcover-candidate-p-adaptive-direction" / source_sha[:16] / recipe_sha[:16]
    qa_root = output / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    qa.sort(key=lambda value: value[0], reverse=True)
    images = []
    for index, (_, key, truth, prediction) in enumerate(qa[:5], start=1):
        path = qa_root / f"{index:03d}-{key}.png"
        base.write_qa(path, f"Candidate P adaptive p={subdivision}: {key}", truth, prediction)
        images.append({"file": path.name, "sha256": sha(path.read_bytes()), "dimensions": [768, 290], "interpretation": "truth | robust affine-rank2 prediction | max-channel error"})
    report = {
        "schema": VERSION,
        "verdict": verdict,
        "recipe": {**recipe, "recipeSha256": recipe_sha},
        "wallTime": {
            "runStarted": started_wall,
            "runSeconds": elapsed,
            "trackStarted": TRACK_STARTED,
            "trackSeconds": track_elapsed,
            "budgetSeconds": 3600,
            "withinBudget": track_elapsed <= 3600,
        },
        "metricsByGroup": groups,
        "phaseOnlyMetricsByGroup": phase_groups,
        "standingWorstNormalisedGap": standing_gap,
        "fitDiagnostics": fit_diagnostics,
        "evaluations": evaluations,
        "phaseOnlyEvaluations": phase_only_evaluations,
        "qa": images,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    (qa_root / "index.json").write_text(json.dumps({
        "schema": f"{VERSION}-qa-index",
        "recipeSha256": recipe_sha,
        "reportSha256": sha(report_text.encode()),
        "images": images,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "output": str(output.relative_to(ROOT)),
        "standing": standing,
        "grazingControl": groups["grazing-control"],
        "wallSeconds": elapsed,
        "reportSha256": sha(report_text.encode()),
    }, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdivision", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    run(args.subdivision)


if __name__ == "__main__":
    main()

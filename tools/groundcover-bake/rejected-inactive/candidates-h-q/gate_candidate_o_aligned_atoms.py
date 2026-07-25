#!/usr/bin/env python3
"""Candidate-O unlimited-precision aligned-positive-atom gate.

Offline only.  Reuses Candidate H's physically filtered atoms and exact-BVH
positive-measure truth.  It never outputs or blends representative depth.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import analyze_candidate_h_angular_continuity as hgate


VERSION = "candidate-o-aligned-positive-atoms-v1"
SOURCE = "src/assets/groundcover/calamagrostis-canescens.gcrp"
AUDIT = "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-EIGHT-READ-ALIGNED-ATOM-AUDIT.md"
RESOLUTION = 128
REFERENCE_Y = 0.49255
SOURCE_SCALES = (4, 16)
TRUTH_RADII = (2, 8)
HELDOUT_ELEVATIONS = (24.0, 45.0, 65.0)
AZIMUTHS = (0.0, 11.25, 67.5, 78.75, 157.5, 168.75, 247.5, 258.75)
CONTROL_ELEVATIONS = (15.0, 35.0, 55.0, 75.0)
CONTROL_AZIMUTHS = (0.0, 67.5, 157.5, 247.5)
TRANSLATION_METRES = 0.0045
LIMITS = {
    "coverage_p95": 0.08,
    "coverage_p99": 0.20,
    "rgb_p95": 0.06,
    "rgb_p99": 0.15,
    "connected": 0.01,
    "translation_p95": 0.06,
}


@dataclass(frozen=True)
class Entry:
    key: str
    elevation: float
    azimuth: float
    offset_x: float = 0.0
    offset_z: float = 0.0


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def direction(elevation: float, azimuth: float) -> np.ndarray:
    e = math.radians(elevation)
    a = math.radians(azimuth)
    return np.array([math.cos(e) * math.cos(a), -math.sin(e), math.cos(e) * math.sin(a)], dtype=np.float64)


def slope(d: np.ndarray) -> np.ndarray:
    return d[[0, 2]] / -d[1]


def nearest_sample(field: np.ndarray, profile: hgate.Profile, qx: np.ndarray, qz: np.ndarray) -> np.ndarray:
    ux = np.mod((qx - profile.origin_x) / profile.size_x, 1.0)
    uz = np.mod((qz - profile.origin_z) / profile.size_z, 1.0)
    ix = np.floor(ux * profile.interior_width).astype(np.int64) % profile.interior_width
    iy = np.floor((1.0 - uz) * profile.interior_height).astype(np.int64) % profile.interior_height
    return field[iy, ix]


def brackets(elevation: float, azimuth: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    rows = (15.0, 35.0, 55.0, 75.0)
    if not rows[0] <= elevation <= rows[-1]:
        raise ValueError(f"direction {elevation} degrees is outside Candidate-H's four-row domain")
    low = max(index for index, value in enumerate(rows) if value <= elevation)
    high = min(low + 1, len(rows) - 1)
    t = 0.0 if low == high else (elevation - rows[low]) / (rows[high] - rows[low])
    coordinate = (azimuth % 360.0) / 22.5
    az0 = math.floor(coordinate) % 16
    az1 = (az0 + 1) % 16
    a = coordinate - math.floor(coordinate)
    nodes = [(low, az0), (low, az1), (high, az0), (high, az1)]
    weights = np.array([(1 - t) * (1 - a), (1 - t) * a, t * (1 - a), t * a], dtype=np.float64)
    return nodes, weights


def predict(
    profile: hgate.Profile,
    atom: hgate.AtomSet,
    mapping: dict[tuple[int, int], int],
    entry: Entry,
    bridge: bool = False,
) -> np.ndarray:
    y, x = np.indices((RESOLUTION, RESOLUTION))
    qx = profile.origin_x + (x + 0.5 + entry.offset_x) / RESOLUTION * profile.size_x
    qz = profile.origin_z + (z := y + 0.5 + entry.offset_z) / RESOLUTION * profile.size_z
    del z
    live_d = direction(entry.elevation, entry.azimuth)
    live_s = slope(live_d)
    nodes, weights = brackets(entry.elevation, entry.azimuth)
    if bridge:
        winner = int(np.argmax(weights))
        nodes = [nodes[winner]]
        weights = np.array([1.0])
    result = np.zeros((RESOLUTION, RESOLUTION, 4), dtype=np.float32)
    for (row, azimuth_index), weight in zip(nodes, weights, strict=True):
        if weight == 0:
            continue
        slice_index = mapping[(row, azimuth_index)]
        node_d = profile.slices[slice_index].direction
        node_s = slope(node_d)

        # q is the live line's middle-plane intercept.  Convert it to this
        # node's top-plane chart for the seed atom read.
        seed_top_x = qx - (profile.top_h - REFERENCE_Y) * node_s[0]
        seed_top_z = qz - (profile.top_h - REFERENCE_Y) * node_s[1]
        seed_a = nearest_sample(atom.coverage[slice_index], profile, seed_top_x, seed_top_z)
        seed_tau = nearest_sample(atom.depth[slice_index], profile, seed_top_x, seed_top_z)
        valid = (seed_a > 1.0 / 255.0) & np.isfinite(seed_tau)
        height = np.where(valid, profile.top_h + seed_tau * node_d[1], REFERENCE_Y)

        # Exact flat-chart point correspondence.  The representative point on
        # the live line is re-expressed in the same node's chart, then reread.
        corrected_middle_x = qx + (REFERENCE_Y - height) * (live_s[0] - node_s[0])
        corrected_middle_z = qz + (REFERENCE_Y - height) * (live_s[1] - node_s[1])
        corrected_top_x = corrected_middle_x - (profile.top_h - REFERENCE_Y) * node_s[0]
        corrected_top_z = corrected_middle_z - (profile.top_h - REFERENCE_Y) * node_s[1]
        corrected_a = nearest_sample(atom.coverage[slice_index], profile, corrected_top_x, corrected_top_z)
        corrected_p = nearest_sample(atom.premul_rgb[slice_index], profile, corrected_top_x, corrected_top_z)
        result[..., 3] += (weight * np.where(valid, corrected_a, 0.0)).astype(np.float32)
        result[..., :3] += (weight * np.where(valid[..., None], corrected_p, 0.0)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def direct_node_atom(
    profile: hgate.Profile,
    atom: hgate.AtomSet,
    mapping: dict[tuple[int, int], int],
    entry: Entry,
) -> np.ndarray:
    """Stored-node identity control, independent of truth resampling grids."""
    y, x = np.indices((RESOLUTION, RESOLUTION))
    qx = profile.origin_x + (x + 0.5 + entry.offset_x) / RESOLUTION * profile.size_x
    qz = profile.origin_z + (y + 0.5 + entry.offset_z) / RESOLUTION * profile.size_z
    nodes, weights = brackets(entry.elevation, entry.azimuth)
    winner = int(np.argmax(weights))
    slice_index = mapping[nodes[winner]]
    node_s = slope(profile.slices[slice_index].direction)
    top_x = qx - (profile.top_h - REFERENCE_Y) * node_s[0]
    top_z = qz - (profile.top_h - REFERENCE_Y) * node_s[1]
    result = np.empty((RESOLUTION, RESOLUTION, 4), dtype=np.float32)
    result[..., 3] = nearest_sample(atom.coverage[slice_index], profile, top_x, top_z)
    result[..., :3] = nearest_sample(atom.premul_rgb[slice_index], profile, top_x, top_z)
    return result


def score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    coverage_error = np.abs(prediction[..., 3] - truth[..., 3])
    rgb_error = np.max(np.abs(prediction[..., :3] - truth[..., :3]), axis=2)
    exceed = (coverage_error > LIMITS["coverage_p99"]) | (rgb_error > LIMITS["rgb_p99"])
    connected = hgate.largest_periodic_component(exceed) / exceed.size
    coverage = {key: float(np.quantile(coverage_error, q)) for key, q in (("p50", .5), ("p95", .95), ("p99", .99))}
    rgb = {key: float(np.quantile(rgb_error, q)) for key, q in (("p50", .5), ("p95", .95), ("p99", .99))}
    checks = {
        "coverageP95": coverage["p95"] <= LIMITS["coverage_p95"],
        "coverageP99": coverage["p99"] <= LIMITS["coverage_p99"],
        "rgbP95": rgb["p95"] <= LIMITS["rgb_p95"],
        "rgbP99": rgb["p99"] <= LIMITS["rgb_p99"],
        "connected": connected < LIMITS["connected"],
    }
    return {"green": all(checks.values()), "coverage": coverage, "premulRgb": rgb, "largestConnectedFraction": connected, "checks": checks}


def translation_score(base_t: np.ndarray, shifted_t: np.ndarray, base_p: np.ndarray, shifted_p: np.ndarray) -> dict[str, Any]:
    excess = (shifted_p - base_p) - (shifted_t - base_t)
    a = np.abs(excess[..., 3])
    rgb = np.max(np.abs(excess[..., :3]), axis=2)
    exceed = (a > LIMITS["coverage_p99"]) | (rgb > LIMITS["rgb_p99"])
    connected = hgate.largest_periodic_component(exceed) / exceed.size
    ap95 = float(np.quantile(a, .95))
    rp95 = float(np.quantile(rgb, .95))
    return {
        "green": max(ap95, rp95) <= LIMITS["translation_p95"] and connected < LIMITS["connected"],
        "coverageP95": ap95,
        "premulRgbP95": rp95,
        "largestConnectedFraction": connected,
    }


def truth_entries() -> list[Entry]:
    entries: list[Entry] = []
    for e in HELDOUT_ELEVATIONS:
        for a in AZIMUTHS:
            entries.append(Entry(f"held-e{e:g}-a{a:g}", e, a))
            entries.append(Entry(f"shift-e{e:g}-a{a:g}", e, a, offset_x=math.nan))
    for e in CONTROL_ELEVATIONS:
        for a in CONTROL_AZIMUTHS:
            entries.append(Entry(f"control-e{e:g}-a{a:g}", e, a))
    return entries


def write_qa(path: Path, title: str, truth: np.ndarray, prediction: np.ndarray) -> None:
    error = np.max(np.abs(truth - prediction), axis=2)
    panels: list[np.ndarray] = []
    for value in (truth, prediction):
        rgb = value[..., :3] + 0.12 * (1.0 - value[..., 3:4])
        panels.append(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    heat = np.zeros((*error.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(np.clip(error / .20, 0, 1) * 255).astype(np.uint8)
    heat[..., 1] = np.rint(np.clip(error / .20, 0, 1) * 170).astype(np.uint8)
    panels.append(heat)
    scale = 3
    header = 34
    image = Image.new("RGB", (RESOLUTION * scale * 3, RESOLUTION * scale + header), (16, 16, 16))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), title, fill=(255, 255, 255))
    for index, panel in enumerate(panels):
        image.paste(Image.fromarray(panel).resize((RESOLUTION * scale, RESOLUTION * scale), Image.Resampling.NEAREST), (index * RESOLUTION * scale, header))
    image.save(path)


def synthetic_two_height() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n = RESOLUTION
    q = (np.arange(n) + .5) / n
    heights = (.20, .90)
    colours = (np.array([.12, .45, .06]), np.array([.68, .42, .55]))
    slopes = (.2, 1.0)
    live = .6

    def components(value: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0 = np.mod(value - (heights[0] - REFERENCE_Y) * s, 1.0)
        x1 = np.mod(value - (heights[1] - REFERENCE_Y) * s, 1.0)
        a0 = .34 * (.5 + .5 * np.sin(math.tau * 7 * x0))
        a1 = .34 * (.5 + .5 * np.sin(math.tau * 11 * x1 + .7))
        a = a0 + a1
        p = a0[:, None] * colours[0] + a1[:, None] * colours[1]
        hbar = (a0 * heights[0] + a1 * heights[1]) / np.maximum(a, 1e-9)
        return a, p, hbar

    truth_a, truth_p, _ = components(q, live)
    predictions = []
    for s in slopes:
        _, _, hbar = components(q, s)
        qc = np.mod(q + (REFERENCE_Y - hbar) * (live - s), 1.0)
        a, p, _ = components(qc, s)
        predictions.append(np.concatenate((p, a[:, None]), axis=1))
    prediction = .5 * (predictions[0] + predictions[1])
    truth = np.concatenate((truth_p, truth_a[:, None]), axis=1)
    truth2 = np.repeat(truth[None, :, :], n, axis=0).astype(np.float32)
    prediction2 = np.repeat(prediction[None, :, :], n, axis=0).astype(np.float32)
    return score(truth2, prediction2), truth2, prediction2


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_path = root / SOURCE
    script_path = Path(__file__).resolve()
    exporter_path = root / "tools/groundcover-bake/export-candidate-o-truth.ts"
    audit_path = root / AUDIT
    source_sha = sha256(source_path.read_bytes())
    profile = hgate.load_profile(source_path)
    entries = truth_entries()
    shift_pixels = TRANSLATION_METRES / profile.size_x * RESOLUTION
    entries = [Entry(e.key, e.elevation, e.azimuth, shift_pixels if math.isnan(e.offset_x) else e.offset_x, e.offset_z) for e in entries]
    recipe = {
        "version": VERSION,
        "sourceSha256": source_sha,
        "analyzerSha256": sha256(script_path.read_bytes()),
        "truthExporterSha256": sha256(exporter_path.read_bytes()),
        "auditSha256": sha256(audit_path.read_bytes()),
        "resolution": RESOLUTION,
        "referenceY": REFERENCE_Y,
        "sourceScales": SOURCE_SCALES,
        "truthRadii": TRUTH_RADII,
        "heldoutElevations": HELDOUT_ELEVATIONS,
        "azimuths": AZIMUTHS,
        "controlElevations": CONTROL_ELEVATIONS,
        "controlAzimuths": CONTROL_AZIMUTHS,
        "translationMetres": TRANSLATION_METRES,
        "limits": LIMITS,
        "sampling": "nearest existing H atom for each seed/corrected logical read; bilinear only across the four angular nodes",
    }
    recipe_sha = canonical_sha(recipe)
    output = root / "data/work/groundcover-candidate-o-aligned-atoms" / source_sha[:16] / recipe_sha[:16]
    qa_root = output / "qa"
    output.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)

    config = {
        "source": SOURCE,
        "resolution": RESOLUTION,
        "referenceY": REFERENCE_Y,
        "radii": TRUTH_RADII,
        "entries": [
            {"key": e.key, "elevationDegrees": e.elevation, "azimuthDegrees": e.azimuth, "phaseOffsetX": e.offset_x, "phaseOffsetZ": e.offset_z}
            for e in entries
        ],
    }
    with tempfile.TemporaryDirectory(prefix="candidate-o-truth-") as temporary:
        temp = Path(temporary)
        config_path = temp / "config.json"
        truth_root = temp / "truth"
        config_path.write_text(json.dumps(config))
        subprocess.run(["node", "--import", "tsx", str(exporter_path), "--config", str(config_path), "--output", str(truth_root)], cwd=root, check=True)

        print("[candidate-o] rebuilding existing Candidate-H atoms", flush=True)
        atoms, _ = hgate.build_atoms(profile)
        mapping = hgate.lattice_map(profile)
        entry_by_key = {entry.key: entry for entry in entries}

        def load_truth(key: str, radius: int) -> np.ndarray:
            return np.fromfile(truth_root / f"{key}-r{radius}.f32", dtype=np.float32).reshape(RESOLUTION, RESOLUTION, 4)

        evaluations: list[dict[str, Any]] = []
        qa_candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for e in HELDOUT_ELEVATIONS:
            for a in AZIMUTHS:
                base = entry_by_key[f"held-e{e:g}-a{a:g}"]
                shifted = entry_by_key[f"shift-e{e:g}-a{a:g}"]
                for scale, radius in zip(SOURCE_SCALES, TRUTH_RADII, strict=True):
                    truth = load_truth(base.key, radius)
                    truth_shift = load_truth(shifted.key, radius)
                    prediction = predict(profile, atoms[scale], mapping, base)
                    prediction_shift = predict(profile, atoms[scale], mapping, shifted)
                    static = score(truth, prediction)
                    temporal = translation_score(truth, truth_shift, prediction, prediction_shift)
                    bridge = score(truth, predict(profile, atoms[scale], mapping, base, bridge=True))
                    key = f"e{e:g}-a{a:g}-sigma{scale}"
                    evaluations.append({"key": key, "elevation": e, "azimuth": a, "scale": scale, "aligned": static, "translation4p5mm": temporal, "oneReadBridge": bridge})
                    normalized = max(static["coverage"]["p95"] / .08, static["premulRgb"]["p95"] / .06, static["largestConnectedFraction"] / .01)
                    qa_candidates.append((normalized, key, truth, prediction))

        controls: list[dict[str, Any]] = []
        for e in CONTROL_ELEVATIONS:
            for a in CONTROL_AZIMUTHS:
                entry = entry_by_key[f"control-e{e:g}-a{a:g}"]
                for scale, radius in zip(SOURCE_SCALES, TRUTH_RADII, strict=True):
                    direct = direct_node_atom(profile, atoms[scale], mapping, entry)
                    result = score(direct, predict(profile, atoms[scale], mapping, entry))
                    controls.append({"key": f"e{e:g}-a{a:g}-sigma{scale}", "score": result})

    synthetic_score, synthetic_truth, synthetic_prediction = synthetic_two_height()
    qa_candidates.sort(key=lambda item: item[0], reverse=True)
    image_records = []
    for index, (_, key, truth, prediction) in enumerate(qa_candidates[:4], start=1):
        path = qa_root / f"{index:03d}-{key}.png"
        write_qa(path, f"Candidate O {key}: truth | aligned atoms | error", truth, prediction)
        image_records.append({"file": path.name, "sha256": sha256(path.read_bytes()), "dimensions": [1152, 418], "interpretation": "exact-BVH truth, aligned-positive-atom prediction, max-channel error"})
    synthetic_path = qa_root / "005-synthetic-two-height.png"
    write_qa(synthetic_path, "Candidate O synthetic two-height counterexample", synthetic_truth, synthetic_prediction)
    image_records.append({"file": synthetic_path.name, "sha256": sha256(synthetic_path.read_bytes()), "dimensions": [1152, 418], "interpretation": "two separated height strata; one representative-height alignment"})

    heldout_green = all(item["aligned"]["green"] for item in evaluations)
    translation_green = all(item["translation4p5mm"]["green"] for item in evaluations)
    controls_green = all(item["score"]["green"] for item in controls)
    bridge_green = all(item["oneReadBridge"]["green"] for item in evaluations)
    synthetic_green = bool(synthetic_score["green"])
    verdict = "GREEN" if heldout_green and translation_green and controls_green and synthetic_green else "RED"
    report = {
        "schema": VERSION,
        "verdict": verdict,
        "recipe": {**recipe, "recipeSha256": recipe_sha},
        "source": {"path": SOURCE, "sha256": source_sha, "triangles": int(profile.triangles.shape[0]), "topH": profile.top_h},
        "method": {
            "logicalReads": "4 angular nodes x (seed atom + corrected same-node atom) = 8",
            "correction": "exact middle-plane epipolar point correspondence in the flat local-affine chart",
            "output": "bilinear angular blend of corrected coverage and premultiplied RGB only; no depth output/blend",
            "unsupportedExteriorElevations": "below 15 or above 75 degrees have no four-node Candidate-H cell and are not silently clamped",
        },
        "checks": {"heldout": heldout_green, "translation4p5mm": translation_green, "exactNodeControls": controls_green, "syntheticTwoHeight": synthetic_green, "oneReadBridgeSeparate": bridge_green},
        "evaluations": evaluations,
        "exactNodeControls": controls,
        "syntheticTwoHeight": synthetic_score,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    report_sha = sha256(report_text.encode())
    index = {"schema": f"{VERSION}-qa-index", "sourceSha256": source_sha, "recipeSha256": recipe_sha, "reportSha256": report_sha, "images": image_records}
    (qa_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    def worst(path: tuple[str, ...]) -> float:
        values = []
        for item in evaluations:
            value: Any = item
            for part in path:
                value = value[part]
            values.append(float(value))
        return max(values)

    summary = [
        "# Candidate O aligned-positive-atom gate", "", f"Verdict: **{verdict}**", "",
        f"Source SHA-256: `{source_sha}`", f"Recipe SHA-256: `{recipe_sha}`", f"Report SHA-256: `{report_sha}`", "",
        f"- Held-out aligned result: `{'GREEN' if heldout_green else 'RED'}`",
        f"- 4.5 mm translation: `{'GREEN' if translation_green else 'RED'}`",
        f"- Exact-node controls: `{'GREEN' if controls_green else 'RED'}`",
        f"- Synthetic two-height: `{'GREEN' if synthetic_green else 'RED'}`",
        f"- One-read bridge (separate): `{'GREEN' if bridge_green else 'RED'}`", "",
        f"Worst aligned coverage p95: `{worst(('aligned','coverage','p95')):.6f}` (limit 0.08)",
        f"Worst aligned RGB p95: `{worst(('aligned','premulRgb','p95')):.6f}` (limit 0.06)",
        f"Worst aligned connected fraction: `{worst(('aligned','largestConnectedFraction')):.6f}` (limit 0.01)",
        f"Worst translation coverage p95: `{worst(('translation4p5mm','coverageP95')):.6f}`",
        f"Worst translation RGB p95: `{worst(('translation4p5mm','premulRgbP95')):.6f}`", "",
        "The four-row H source has no conforming four-node cell below 15 or above 75 degrees. Those exterior directions are an explicit domain failure, not clamped or hidden.",
        "QA panels are truth | prediction | error. See `qa/index.json` for hashes.", "",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary))
    print(json.dumps({"verdict": verdict, "output": str(output.relative_to(root)), "checks": report["checks"], "worst": summary[14:19]}, indent=2))


if __name__ == "__main__":
    main()

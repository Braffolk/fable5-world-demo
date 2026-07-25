#!/usr/bin/env python3
"""Exact-node gate for Candidate-J shared-vertex angular finite elements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


VERSION = "candidate-j-angular-fe-node-gate-v1"
SCALES = (4, 16)
THRESHOLDS = {
    "coverage_p95": 0.08,
    "coverage_p99": 0.20,
    "rgb_p95": 0.06,
    "rgb_p99": 0.15,
    "depth_p95_m": 0.05,
    "depth_p99_m": 0.10,
    "connected_fraction": 0.01,
}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(finite.size),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def nearest_palette(colour: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = colour.reshape(-1, 3)
    output = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], 65536):
        end = min(flat.shape[0], start + 65536)
        d = np.sum((flat[start:end, None, :] - palette[None, :, :]) ** 2, axis=2)
        output[start:end] = np.argmin(d, axis=1).astype(np.uint8)
    return output.reshape(colour.shape[:-1])


def quantise_near_depth(profile, slice_index: int, depth: np.ndarray) -> np.ndarray:
    data = profile.slices[slice_index]
    span = data.depth_max - data.depth_min
    code = np.rint(np.clip((depth - data.depth_min) / span, 0.0, 1.0) * 4095.0)
    result = data.depth_min + code / 4095.0 * span
    result[~np.isfinite(depth)] = np.nan
    return result.astype(np.float32)


def choose_residual_codec(residual: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = residual[np.isfinite(residual)].astype(np.float64)
    if values.size == 0:
        return 0.0, 0.0, np.zeros(residual.shape, dtype=np.float32)
    best = None
    for trim in (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05):
        lo = float(np.quantile(values, trim))
        hi = float(np.quantile(values, 1.0 - trim))
        if hi <= lo + 1.0e-9:
            decoded = np.full(residual.shape, lo, dtype=np.float32)
        else:
            code = np.rint(np.clip((residual - lo) / (hi - lo), 0.0, 1.0) * 31.0)
            decoded = (lo + code / 31.0 * (hi - lo)).astype(np.float32)
        error = np.abs(decoded[np.isfinite(residual)] - residual[np.isfinite(residual)])
        p95 = float(np.quantile(error, 0.95))
        p99 = float(np.quantile(error, 0.99))
        # Frozen gates are primary; mean only breaks ties between passing ranges.
        score = max(p95 / 0.05, p99 / 0.10) + 1.0e-3 * float(np.mean(error))
        if best is None or score < best[0]:
            best = (score, lo, hi, decoded)
    assert best is not None
    return best[1], best[2], best[3]


def best_palette_fallback(a: np.ndarray, premul: np.ndarray, palette: np.ndarray, mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    predicted = a[mask, None, None] * palette[None, :, :]
    error = predicted - premul[mask, None, :]
    score = np.sum(error * error, axis=(0, 2))
    return int(np.argmin(score))


def fit_nodes(hmod, imod, profile, truth, palette: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    ta, td, tp = truth
    count, height, width = ta.shape
    qa = np.rint(np.clip(ta, 0.0, 1.0) * 7.0).astype(np.float32) / 7.0
    pred_depth = np.full(td.shape, np.nan, dtype=np.float32)
    pred_rgb_oracle = np.zeros(tp.shape, dtype=np.float32)
    pred_rgb_palette = np.zeros(tp.shape, dtype=np.float32)
    miss_bases: list[float] = []
    residual_ranges: list[list[float]] = []
    raw_fallbacks: list[list[float]] = []
    palette_fallbacks: list[int] = []
    missing_predictor_fractions: list[float] = []
    for index in range(count):
        near_hit, near_depth, near_rgb = hmod.decode_slice_base(profile, index)
        near_depth = quantise_near_depth(profile, index, near_depth)
        support = np.isfinite(td[index]) & (ta[index] > 1.0 / 255.0)
        hit = support & near_hit & np.isfinite(near_depth)
        miss = support & ~hit
        base_miss = float(np.median(td[index][miss])) if np.any(miss) else float(
            0.5 * (profile.slices[index].depth_min + profile.slices[index].depth_max)
        )
        base = np.where(hit, near_depth, base_miss).astype(np.float32)
        residual = np.where(support, td[index] - base, np.nan).astype(np.float32)
        lo, hi, decoded = choose_residual_codec(residual)
        pred_depth[index][support] = base[support] + decoded[support]
        miss_bases.append(base_miss)
        residual_ranges.append([lo, hi])
        missing_predictor_fractions.append(float(np.count_nonzero(miss) / max(1, np.count_nonzero(support))))

        # Optimistic raw-RGB fallback: three unrestricted f32 values per node.
        denom = float(np.sum(qa[index][miss] ** 2))
        if denom > 1.0e-12:
            fallback = np.sum(qa[index][miss, None] * tp[index][miss], axis=0) / denom
        else:
            fallback = np.zeros(3, dtype=np.float32)
        fallback = np.clip(fallback, 0.0, 1.0).astype(np.float32)
        raw_fallbacks.append(fallback.tolist())
        oracle_colour = np.where(hit[..., None], near_rgb, fallback[None, None, :])
        pred_rgb_oracle[index] = qa[index, ..., None] * oracle_colour

        near_class = nearest_palette(near_rgb, palette)
        fallback_class = best_palette_fallback(qa[index], tp[index], palette, miss)
        palette_fallbacks.append(fallback_class)
        realistic_class = np.where(hit, near_class, fallback_class)
        pred_rgb_palette[index] = qa[index, ..., None] * palette[realistic_class]

    valid = np.isfinite(td) & (ta > 1.0 / 255.0)
    coverage_error = np.abs(qa - ta)
    depth_error = np.where(valid, np.abs(pred_depth - td), np.nan)
    oracle_rgb_error = np.max(np.abs(pred_rgb_oracle - tp), axis=3)
    palette_rgb_error = np.max(np.abs(pred_rgb_palette - tp), axis=3)

    max_connected = 0.0
    max_connected_direction = -1
    for index in range(count):
        support = valid[index]
        exceed = support & (
            (coverage_error[index] > THRESHOLDS["coverage_p99"])
            | (depth_error[index] > THRESHOLDS["depth_p99_m"])
            | (palette_rgb_error[index] > THRESHOLDS["rgb_p99"])
        )
        fraction = hmod.largest_periodic_component(exceed) / max(1, int(np.count_nonzero(support)))
        if fraction > max_connected:
            max_connected = float(fraction)
            max_connected_direction = index

    cq = quantiles(coverage_error)
    dq = quantiles(depth_error)
    oq = quantiles(oracle_rgb_error)
    pq = quantiles(palette_rgb_error)
    checks = {
        "coverageP95": cq["p95"] <= THRESHOLDS["coverage_p95"],
        "coverageP99": cq["p99"] <= THRESHOLDS["coverage_p99"],
        "depthP95": dq["p95"] <= THRESHOLDS["depth_p95_m"],
        "depthP99": dq["p99"] <= THRESHOLDS["depth_p99_m"],
        "optimisticRawRgbP95": oq["p95"] <= THRESHOLDS["rgb_p95"],
        "optimisticRawRgbP99": oq["p99"] <= THRESHOLDS["rgb_p99"],
        "paletteRgbP95": pq["p95"] <= THRESHOLDS["rgb_p95"],
        "paletteRgbP99": pq["p99"] <= THRESHOLDS["rgb_p99"],
        "connected": max_connected < THRESHOLDS["connected_fraction"],
    }
    result = {
        "green": all(checks.values()),
        "checks": checks,
        "coverageAbsoluteError": cq,
        "depthCorrectionErrorMetres": dq,
        "optimisticRawRgbPremulMaxError": oq,
        "realisticPalettePremulMaxError": pq,
        "r0MissGivenFilteredSupport": quantiles(np.asarray(missing_predictor_fractions, dtype=np.float32)),
        "largestConnectedExceedance": {"fraction": max_connected, "direction": max_connected_direction},
    }
    maps = {
        "coverage": np.max(coverage_error, axis=0),
        "depth": np.nanmax(depth_error, axis=0),
        "rgb": np.max(palette_rgb_error, axis=0),
    }
    tables = {
        "missBaseF32": miss_bases,
        "residualLowHighF32": residual_ranges,
        "optimisticRawRgbFallbackF32": raw_fallbacks,
        "paletteFallbackClass": palette_fallbacks,
    }
    return result, maps, tables


def heat(values: np.ndarray, limit: float) -> np.ndarray:
    t = np.clip(np.nan_to_num(values) / limit, 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(255.0 * t).astype(np.uint8)
    rgb[..., 1] = np.rint(255.0 * np.minimum(1.0, 2.0 * t) * (1.0 - 0.6 * t)).astype(np.uint8)
    rgb[..., 2] = np.rint(40.0 * (1.0 - t)).astype(np.uint8)
    return rgb


def write_qa(path: Path, title: str, maps: dict[str, np.ndarray]) -> None:
    panels = [
        ("max A error / .20", heat(maps["coverage"], 0.20)),
        ("max depth error / .10m", heat(maps["depth"], 0.10)),
        ("max palette RGB / .15", heat(maps["rgb"], 0.15)),
    ]
    scale = 2
    w = panels[0][1].shape[1] * scale
    h = panels[0][1].shape[0] * scale
    image = Image.new("RGB", (3 * w, h + 44), (15, 15, 15))
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), title, fill=(255, 255, 255))
    for index, (label, panel) in enumerate(panels):
        image.paste(Image.fromarray(panel, "RGB").resize((w, h), Image.Resampling.NEAREST), (index * w, 44))
        draw.text((index * w + 8, 23), label, fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / args.source).resolve()
    hmod = import_file("candidate_h_gate_for_j", root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py")
    imod = import_file("candidate_i_gate_for_j", root / "tools/groundcover-bake/gate_candidate_i_vertical_extinction.py")
    profile = hmod.load_profile(source)
    palette = imod.load_palette(root)
    doc = root / "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-J-ANGULAR-FE-CODEC.md"
    recipe = {
        "version": VERSION,
        "sourceSha256": sha256(source),
        "scriptSha256": sha256(Path(__file__)),
        "codecDocSha256": sha256(doc),
        "scales": SCALES,
        "thresholds": THRESHOLDS,
        "depthMetadataBytes": 65 * 2 * 3 * 4,
        "optimisticRawRgbMetadataBytes": 65 * 2 * 3 * 4,
        "scaleAtlasBytes": 64 * 256 * 256 * 8,
    }
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = root / "data/work/groundcover-candidate-j-angular-fe-nodes" / recipe["sourceSha256"][:16] / recipe_sha[:16]
    qa = output / "qa"
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for number, scale in enumerate(SCALES, start=1):
        truth = imod.build_truth(hmod, profile, scale)
        result, maps, scale_tables = fit_nodes(hmod, imod, profile, truth, palette)
        results[str(scale)] = result
        tables[str(scale)] = scale_tables
        image_path = qa / f"{number:03d}-sigma{scale}-node-errors.png"
        write_qa(image_path, f"Candidate J exact nodes sigma={scale}; GREEN={result['green']}", maps)
        artifacts.append({"path": str(image_path.relative_to(root)), "sha256": sha256(image_path)})
    report = {
        "schema": VERSION,
        "verdict": "GREEN" if all(value["green"] for value in results.values()) else "RED",
        "topology": {"interRingQuads": 48, "poleDegenerateQuads": 16, "pages": 64, "c0": "bit-identical shared node tuples"},
        "fidelity": results,
        "tables": tables,
        "recipe": recipe,
        "artifacts": artifacts,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (qa / "index.json").write_text(json.dumps({"report": str(report_path.relative_to(root)), "verdict": report["verdict"], "artifacts": artifacts}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "verdict": report["verdict"], "fidelity": results}, indent=2))


if __name__ == "__main__":
    main()


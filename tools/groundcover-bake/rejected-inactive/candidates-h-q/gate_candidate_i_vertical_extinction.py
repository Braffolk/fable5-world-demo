#!/usr/bin/env python3
"""Fit and gate the 64-bit Candidate-I direction-analytic extinction carrier.

This is cook/offline analysis only.  It imports the accepted GCRP/v4 decoder
from the Candidate-H analyzer, rebuilds both filtered truth scales including
the pole, fits one quantised 32-bit extinction slot per phase texel, and scores
all 65 directions.  It never edits a runtime asset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


VERSION = "candidate-i-vertical-extinction-v1"
SCALES = (4, 16)
THRESHOLDS = {
    "coverage_p95": 0.08,
    "coverage_p99": 0.20,
    "premul_rgb_p95": 0.06,
    "premul_rgb_p99": 0.15,
    "position_p95_m": 0.05,
    "position_p99_m": 0.10,
    "connected_fraction": 0.01,
}
PALETTE_INDEX = Path(
    "data/work/groundcover-candidate-h-palette16/"
    "2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c/"
    "dd4b05a99c5ba516fa2c70d4e35b260ca7c58fdd1714a1e226c04cbcf8b910f4/qa/index.json"
)


def load_h_module(root: Path):
    path = root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py"
    spec = importlib.util.spec_from_file_location("candidate_h_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_palette(root: Path) -> np.ndarray:
    data = json.loads((root / PALETTE_INDEX).read_text())
    values = [entry["linearRgb"] for entry in data["palette"]]
    palette = np.asarray(values, dtype=np.float32)
    if palette.shape != (16, 3):
        raise ValueError("expected the accepted 16-entry Candidate-H palette")
    return palette


def build_truth(module, profile, scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(profile.slices)
    shape = (count, profile.interior_height, profile.interior_width)
    coverage = np.empty(shape, dtype=np.float32)
    depth = np.empty(shape, dtype=np.float32)
    premul = np.empty((*shape, 3), dtype=np.float32)
    size = 2 * scale + 1
    for index in range(count):
        covered, raw_depth, rgb = module.decode_slice_base(profile, index)
        a = ndimage.uniform_filter(covered.astype(np.float32), size=size, mode="wrap")
        coverage[index] = np.rint(np.clip(a, 0.0, 1.0) * 255.0) / 255.0
        for channel in range(3):
            premul[index, ..., channel] = ndimage.uniform_filter(
                rgb[..., channel] * covered,
                size=size,
                mode="wrap",
            )
        median = module.toroidal_nanmedian(raw_depth, scale)
        slice_data = profile.slices[index]
        span = slice_data.depth_max - slice_data.depth_min
        code = np.rint(np.clip((median - slice_data.depth_min) / span, 0.0, 1.0) * 1023.0)
        quantised = slice_data.depth_min + code / 1023.0 * span
        quantised[~np.isfinite(median)] = np.nan
        depth[index] = quantised.astype(np.float32)
        print(f"[candidate-i] sigma={scale} truth {index + 1}/{count}", flush=True)
    return coverage, depth, premul


def stable_f(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return F and dF/dx without cancellation at small/saturated x."""
    x = np.maximum(x, 1.0e-8)
    small = x < 1.0e-3
    large = x >= 16.0
    middle = ~(small | large)
    f = np.empty_like(x, dtype=np.float32)
    fp = np.empty_like(x, dtype=np.float32)
    xs = x[small]
    f[small] = 0.5 - xs / 12.0 + xs * xs * xs / 720.0
    fp[small] = -1.0 / 12.0 + xs * xs / 240.0
    xl = x[large]
    f[large] = 1.0 / xl
    fp[large] = -1.0 / (xl * xl)
    xm = x[middle].astype(np.float64)
    em1 = np.expm1(xm)
    f[middle] = (1.0 / xm - 1.0 / em1).astype(np.float32)
    fp[middle] = (-1.0 / (xm * xm) + (em1 + 1.0) / (em1 * em1)).astype(np.float32)
    return f, fp


def decode_params(raw: np.ndarray, top_h: float) -> tuple[np.ndarray, ...]:
    h = ((raw & 0x7F).astype(np.float32) / 127.0) * top_h
    theta_code = ((raw >> 7) & 0x3F).astype(np.float32)
    theta = np.exp2(-8.0 + theta_code * (13.0 / 63.0))
    r_code = ((raw >> 13) & 0x1F).astype(np.float32)
    ratio = np.exp2(-5.0 + r_code * (5.0 / 31.0))
    ax = -0.75 + ((raw >> 18) & 0x1F).astype(np.float32) * (1.5 / 31.0)
    az = -0.75 + ((raw >> 23) & 0x1F).astype(np.float32) * (1.5 / 31.0)
    colour_class = ((raw >> 28) & 0xF).astype(np.uint8)
    return h, theta, ratio, ax, az, colour_class


def encode_params(
    h: np.ndarray,
    log_theta: np.ndarray,
    log_r: np.ndarray,
    ax: np.ndarray,
    az: np.ndarray,
    colour_class: np.ndarray,
    top_h: float,
) -> np.ndarray:
    hc = np.rint(np.clip(h / top_h, 0.0, 1.0) * 127.0).astype(np.uint32)
    tc = np.rint(np.clip((log_theta / math.log(2.0) + 8.0) / 13.0, 0.0, 1.0) * 63.0).astype(np.uint32)
    rc = np.rint(np.clip((log_r / math.log(2.0) + 5.0) / 5.0, 0.0, 1.0) * 31.0).astype(np.uint32)
    ac = np.rint(np.clip((ax + 0.75) / 1.5, 0.0, 1.0) * 31.0).astype(np.uint32)
    zc = np.rint(np.clip((az + 0.75) / 1.5, 0.0, 1.0) * 31.0).astype(np.uint32)
    cc = colour_class.astype(np.uint32) & np.uint32(0xF)
    return hc | (tc << 7) | (rc << 13) | (ac << 18) | (zc << 23) | (cc << 28)


def evaluate(
    directions: np.ndarray,
    top_h: float,
    h: np.ndarray,
    theta: np.ndarray,
    ratio: np.ndarray,
    ax: np.ndarray,
    az: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.maximum(-directions[:, 1:2], 1.0e-7).astype(np.float32)
    dx = directions[:, 0:1].astype(np.float32)
    dz = directions[:, 2:3].astype(np.float32)
    horizontal2 = np.maximum(0.0, 1.0 - v * v)
    base = np.sqrt(horizontal2 + ratio[None, :] * ratio[None, :] * v * v)
    harmonic = np.exp(np.clip(dx * ax[None, :] + dz * az[None, :], -2.0, 2.0))
    x = theta[None, :] * base * harmonic / v
    a = -np.expm1(-np.minimum(x, 80.0)).astype(np.float32)
    f, fp = stable_f(x)
    s = (top_h - h[None, :] + h[None, :] * f) / v
    return a, s.astype(np.float32), x.astype(np.float32), fp


def initialise(coverage: np.ndarray, depth: np.ndarray, directions: np.ndarray, top_h: float) -> tuple[np.ndarray, ...]:
    j, n = coverage.shape
    v = np.maximum(-directions[:, 1:2], 1.0e-4)
    optical = -np.log1p(-np.clip(coverage, 1.0 / 510.0, 1.0 - 1.0 / 510.0))
    side = np.sqrt(np.maximum(1.0e-6, 1.0 - v * v))
    estimate = np.log(np.maximum(optical * v / side, 2.0 ** -8))
    log_theta = np.median(estimate, axis=0).astype(np.float32)
    ax = np.zeros(n, dtype=np.float32)
    az = np.zeros(n, dtype=np.float32)
    log_r = np.full(n, math.log(0.25), dtype=np.float32)
    h = np.full(n, top_h * 0.7, dtype=np.float32)
    return h, log_theta, log_r, ax, az


def fit(
    coverage: np.ndarray,
    depth: np.ndarray,
    premul: np.ndarray,
    directions: np.ndarray,
    palette: np.ndarray,
    top_h: float,
    steps: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    j, height, width = coverage.shape
    n = height * width
    truth_a = coverage.reshape(j, n)
    truth_s = depth.reshape(j, n)
    truth_p = premul.reshape(j, n, 3)
    valid = np.isfinite(truth_s) & (truth_a > 1.0 / 255.0)
    h, p0, pr, ax, az = initialise(truth_a, truth_s, directions, top_h)
    params = [p0, pr, ax, az]
    moments = [np.zeros_like(value) for value in params]
    variances = [np.zeros_like(value) for value in params]
    beta1, beta2, learning_rate = 0.9, 0.999, 0.035
    v = np.maximum(-directions[:, 1:2], 1.0e-7).astype(np.float32)
    dx = directions[:, 0:1].astype(np.float32)
    dz = directions[:, 2:3].astype(np.float32)
    for iteration in range(steps):
        p0[:] = np.clip(p0, math.log(2.0 ** -8), math.log(2.0 ** 5))
        pr[:] = np.clip(pr, math.log(2.0 ** -5), 0.0)
        ax[:] = np.clip(ax, -0.75, 0.75)
        az[:] = np.clip(az, -0.75, 0.75)
        theta = np.exp(p0)
        ratio = np.exp(pr)
        ahat, shat, x, fp = evaluate(directions, top_h, h, theta, ratio, ax, az)

        # The best height for the current angular extinction is a bounded
        # one-variable least-squares solve: H/v-s = h*(1-F)/v.
        f, _ = stable_f(x)
        z = (1.0 - f) / v
        target = top_h / v - truth_s
        weight = np.where(valid, np.maximum(truth_a, 1.0 / 255.0), 0.0)
        numerator = np.nansum(weight * z * target, axis=0)
        denominator = np.sum(weight * z * z, axis=0)
        h[:] = np.clip(np.divide(numerator, denominator, out=h.copy(), where=denominator > 1.0e-8), 0.0, top_h)
        shat = (top_h - h[None, :] + h[None, :] * f) / v

        ea = np.clip((ahat - truth_a) / 0.08, -4.0, 4.0)
        ep = np.where(valid, np.clip((shat - truth_s) / 0.05, -4.0, 4.0), 0.0)
        common_a = 2.0 * ea * (np.exp(-np.minimum(x, 80.0)) * x) / 0.08
        common_p = 4.0 * ep * (h[None, :] / v * fp * x) / 0.05
        common = (common_a + common_p) / float(j)
        horizontal2 = np.maximum(0.0, 1.0 - v * v)
        ratio2v2 = ratio[None, :] ** 2 * v * v
        base2 = horizontal2 + ratio2v2
        derivs = (
            np.ones_like(x),
            ratio2v2 / np.maximum(base2, 1.0e-8),
            np.broadcast_to(dx, x.shape),
            np.broadcast_to(dz, x.shape),
        )
        gradients = [np.sum(common * derivative, axis=0).astype(np.float32) for derivative in derivs]
        for index, (parameter, gradient) in enumerate(zip(params, gradients)):
            moments[index] = beta1 * moments[index] + (1.0 - beta1) * gradient
            variances[index] = beta2 * variances[index] + (1.0 - beta2) * gradient * gradient
            mh = moments[index] / (1.0 - beta1 ** (iteration + 1))
            vh = variances[index] / (1.0 - beta2 ** (iteration + 1))
            parameter -= learning_rate * mh / (np.sqrt(vh) + 1.0e-6)
        if iteration % 10 == 0 or iteration + 1 == steps:
            cov_p95 = float(np.quantile(np.abs(ahat - truth_a), 0.95))
            pos = np.abs(shat - truth_s)[valid]
            pos_p95 = float(np.quantile(pos, 0.95)) if pos.size else 0.0
            print(f"[candidate-i] fit {iteration + 1}/{steps}: A p95={cov_p95:.4f}, P p95={pos_p95:.4f}m", flush=True)

    theta = np.exp(np.clip(p0, math.log(2.0 ** -8), math.log(2.0 ** 5)))
    ratio = np.exp(np.clip(pr, math.log(2.0 ** -5), 0.0))
    ax = np.clip(ax, -0.75, 0.75)
    az = np.clip(az, -0.75, 0.75)
    ahat, _, _, _ = evaluate(directions, top_h, h, theta, ratio, ax, az)
    saa = np.sum(ahat * ahat, axis=0)
    sap = np.einsum("jn,jnc->nc", ahat, truth_p)
    scores = saa[:, None] * np.sum(palette * palette, axis=1)[None, :] - 2.0 * sap @ palette.T
    colour_class = np.argmin(scores, axis=1).astype(np.uint8)
    packed = encode_params(h, np.log(theta), np.log(ratio), ax, az, colour_class, top_h)
    hd, td, rd, adx, adz, cd = decode_params(packed, top_h)
    ahat, shat, _, _ = evaluate(directions, top_h, hd, td, rd, adx, adz)
    phat = ahat[..., None] * palette[cd][None, :, :]
    return packed.reshape(height, width), {
        "coverage": ahat.reshape(j, height, width),
        "depth": shat.reshape(j, height, width),
        "premul": phat.reshape(j, height, width, 3),
    }


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


def largest_periodic_component(module, mask: np.ndarray) -> int:
    return int(module.largest_periodic_component(mask))


def score(module, truth: tuple[np.ndarray, ...], prediction: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ta, ts, tp = truth
    pa, ps, pp = prediction["coverage"], prediction["depth"], prediction["premul"]
    ea = np.abs(pa - ta)
    ergb = np.max(np.abs(pp - tp), axis=3)
    valid = np.isfinite(ts) & (ta > 1.0 / 255.0)
    epos = np.where(valid, np.abs(ps - ts), np.nan)
    max_connected = 0.0
    max_direction = -1
    for index in range(ta.shape[0]):
        support = valid[index]
        count = int(np.count_nonzero(support))
        exceed = support & (
            (ea[index] > THRESHOLDS["coverage_p99"])
            | (ergb[index] > THRESHOLDS["premul_rgb_p99"])
            | (epos[index] > THRESHOLDS["position_p99_m"])
        )
        fraction = largest_periodic_component(module, exceed) / count if count else 0.0
        if fraction > max_connected:
            max_connected = fraction
            max_direction = index
    qa = quantiles(ea)
    qr = quantiles(ergb)
    qp = quantiles(epos)
    checks = {
        "coverageP95": qa["p95"] <= THRESHOLDS["coverage_p95"],
        "coverageP99": qa["p99"] <= THRESHOLDS["coverage_p99"],
        "rgbP95": qr["p95"] <= THRESHOLDS["premul_rgb_p95"],
        "rgbP99": qr["p99"] <= THRESHOLDS["premul_rgb_p99"],
        "positionP95": qp["p95"] <= THRESHOLDS["position_p95_m"],
        "positionP99": qp["p99"] <= THRESHOLDS["position_p99_m"],
        "connected": max_connected < THRESHOLDS["connected_fraction"],
    }
    per_direction = []
    for index in range(ta.shape[0]):
        per_direction.append({
            "index": index,
            "coverage": quantiles(ea[index]),
            "premulRgb": quantiles(ergb[index]),
            "positionMetres": quantiles(epos[index]),
        })
    return ({
        "green": all(checks.values()),
        "checks": checks,
        "coverageAbsoluteError": qa,
        "premultipliedRgbMaxError": qr,
        "representativePositionMetres": qp,
        "largestConnectedExceedance": {"fraction": max_connected, "direction": max_direction},
        "perDirection": per_direction,
    }, {"coverage": np.max(ea, axis=0), "rgb": np.max(ergb, axis=0), "position": np.nanmax(epos, axis=0)})


def vertical_band_position_oracle(
    coverage: np.ndarray,
    depth: np.ndarray,
    directions: np.ndarray,
    top_h: float,
) -> dict[str, Any]:
    """A fitter-independent necessary-condition test for the band model.

    For a homogeneous ground-to-h band, arbitrary positive extinction can move
    the conditional expected hit height only within [h/2,h].  This oracle gives
    every direction an independent, unquantised extinction and chooses the best
    of all 128 representable h values per texel.  It therefore strictly contains
    Candidate I.  We minimise the number of threshold violations independently
    at 5 cm and 10 cm; failure is a representation proof, not an optimiser result.
    """
    j, height, width = coverage.shape
    n = height * width
    v = np.maximum(-directions[:, 1:2], 1.0e-7).astype(np.float32)
    truth_s = depth.reshape(j, n)
    valid = np.isfinite(truth_s) & (coverage.reshape(j, n) > 1.0 / 255.0)
    truth_y = top_h - v * truth_s
    best_counts = {
        threshold: np.full(n, j + 1, dtype=np.int16)
        for threshold in (THRESHOLDS["position_p95_m"], THRESHOLDS["position_p99_m"])
    }
    for code in range(128):
        h = top_h * code / 127.0
        predicted_y = np.clip(truth_y, 0.5 * h, h)
        error_s = np.abs(predicted_y - truth_y) / v
        for threshold, best in best_counts.items():
            count = np.sum(valid & (error_s > threshold), axis=0).astype(np.int16)
            np.minimum(best, count, out=best)
    total = int(np.count_nonzero(valid))
    at5 = int(np.sum(best_counts[THRESHOLDS["position_p95_m"]]))
    at10 = int(np.sum(best_counts[THRESHOLDS["position_p99_m"]]))
    return {
        "description": "per-direction arbitrary extinction; best UNORM7 h per texel; only [h/2,h] band invariant retained",
        "validRecords": total,
        "minimumPossibleFractionAbove5cm": at5 / total if total else 0.0,
        "maximumPossibleFractionWithin5cm": 1.0 - at5 / total if total else 1.0,
        "minimumPossibleFractionAbove10cm": at10 / total if total else 0.0,
        "maximumPossibleFractionWithin10cm": 1.0 - at10 / total if total else 1.0,
        "p95NecessaryCondition": at5 / total <= 0.05 if total else True,
        "p99NecessaryCondition": at10 / total <= 0.01 if total else True,
    }


def heat(values: np.ndarray, limit: float) -> np.ndarray:
    t = np.clip(np.nan_to_num(values) / limit, 0.0, 1.0)
    result = np.zeros((*values.shape, 3), dtype=np.uint8)
    result[..., 0] = np.rint(255.0 * t).astype(np.uint8)
    result[..., 1] = np.rint(255.0 * np.minimum(1.0, 2.0 * t) * (1.0 - 0.6 * t)).astype(np.uint8)
    result[..., 2] = np.rint(40.0 * (1.0 - t)).astype(np.uint8)
    return result


def write_qa(path: Path, title: str, maps: dict[str, np.ndarray]) -> None:
    panels = [
        ("max |A error| / .20", heat(maps["coverage"], 0.20)),
        ("max premul RGB / .15", heat(maps["rgb"], 0.15)),
        ("max position / .10m", heat(maps["position"], 0.10)),
    ]
    scale = 2
    w = panels[0][1].shape[1] * scale
    h = panels[0][1].shape[0] * scale
    canvas = Image.new("RGB", (w * 3, h + 44), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 255, 255))
    for index, (label, values) in enumerate(panels):
        image = Image.fromarray(values, "RGB").resize((w, h), Image.Resampling.NEAREST)
        canvas.paste(image, (index * w, 44))
        draw.text((index * w + 8, 23), label, fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / args.source).resolve()
    codec_doc = root / "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-I-VERTICAL-EXTINCTION-CODEC.md"
    module = load_h_module(root)
    profile = module.load_profile(source)
    palette = load_palette(root)
    directions = np.stack([item.direction for item in profile.slices]).astype(np.float32)
    recipe = {
        "version": VERSION,
        "sourceSha256": sha256(source),
        "scriptSha256": sha256(Path(__file__)),
        "codecDocSha256": sha256(codec_doc),
        "scales": SCALES,
        "steps": args.steps,
        "thresholds": THRESHOLDS,
    }
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = root / "data/work/groundcover-candidate-i-vertical-extinction" / recipe["sourceSha256"][:16] / recipe_sha[:16]
    qa = output / "qa"
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for number, scale in enumerate(SCALES, start=1):
        truth = build_truth(module, profile, scale)
        oracle = vertical_band_position_oracle(truth[0], truth[1], directions, profile.top_h)
        packed, prediction = fit(*truth, directions, palette, profile.top_h, args.steps)
        result, maps = score(module, truth, prediction)
        result["permissiveVerticalBandPositionOracle"] = oracle
        results[str(scale)] = result
        image_path = qa / f"{number:03d}-sigma{scale}-fidelity-errors.png"
        write_qa(image_path, f"Candidate I sigma={scale}; GREEN={result['green']}", maps)
        artifacts.append({"path": str(image_path.relative_to(root)), "sha256": sha256(image_path), "interpretation": "worst error over all 65 truth directions"})
        packed_path = output / f"sigma{scale}-packed-r32uint.bin"
        packed_path.write_bytes(packed.astype("<u4").tobytes())
    report = {
        "schema": VERSION,
        "verdict": "GREEN" if all(item["green"] for item in results.values()) else "RED",
        "continuity": {
            "green": True,
            "reason": "record and address are direction-independent; closed form is continuous for every downward exterior direction",
        },
        "fidelity": results,
        "recipe": recipe,
        "artifacts": artifacts,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    index_path = qa / "index.json"
    index_path.write_text(json.dumps({"report": str(report_path.relative_to(root)), "verdict": report["verdict"], "artifacts": artifacts}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "verdict": report["verdict"], "fidelity": {key: value["checks"] for key, value in results.items()}}, indent=2))


if __name__ == "__main__":
    main()

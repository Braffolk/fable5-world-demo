#!/usr/bin/env python3
"""Candidate K2 nested one-read measure-pyramid gate.

Offline only.  It evaluates stored angular nodes, continuous footprint levels,
and millimetric world-phase translations.  The runtime/shader path is not
imported or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


VERSION = "candidate-k2-pyramid-stability-v1"
THRESHOLD_DOC = "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K-TRANSLATION-SCALE-LIGHTING-FROZEN.md"
CHAINS = {
    "k2_96_uniform": (96, 48, 24, 12, 6, 3),
    "k2_112_original": (112, 56, 28),
    "k2_120_high_rate": (120, 40, 20, 10, 5),
    "k2_128_direct_nested": (128, 16),
}
RADII = {
    "k2_96_uniform": (4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    "k2_112_original": (4.0, 8.0, 16.0),
    "k2_120_high_rate": (4.0, 8.0, 16.0, 32.0, 64.0),
    "k2_128_direct_nested": (4.0, 16.0),
}
BOUNDARY_RATIOS = (0.0, 0.125, 0.25, 0.5)
TRANSLATION_MM = (1.0, 2.0, 3.0, 4.5)
TRANSLATION_DIRECTIONS = {
    "x": (1.0, 0.0),
    "z": (0.0, 1.0),
    "diag_plus": (math.sqrt(0.5), math.sqrt(0.5)),
    "diag_minus": (math.sqrt(0.5), -math.sqrt(0.5)),
}
STATIC_LIMITS = {"a_p95": 0.08, "a_p99": 0.20, "p_p95": 0.06, "p_p99": 0.15, "connected": 0.01}
TEMPORAL_LIMITS = {"a_p95": 0.04, "a_p99": 0.10, "p_p95": 0.03, "p_p99": 0.08, "connected": 0.01}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_module(root: Path):
    path = root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py"
    spec = importlib.util.spec_from_file_location("candidate_h_for_k2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def area_matrix(output_size: int, input_size: int = 256) -> np.ndarray:
    result = np.zeros((output_size, input_size), dtype=np.float32)
    span = input_size / output_size
    for output in range(output_size):
        low = output * span
        high = (output + 1) * span
        for source in range(int(math.floor(low)), min(input_size, int(math.ceil(high)))):
            result[output, source] = max(0.0, min(high, source + 1.0) - max(low, source)) / span
    if np.max(np.abs(np.sum(result, axis=1) - 1.0)) > 1e-6:
        raise RuntimeError("area reduction is not conservative")
    return result


def rgba4444(value: np.ndarray) -> np.ndarray:
    return (np.rint(np.clip(value, 0.0, 1.0) * 15.0) / 15.0).astype(np.float32)


def reduce_measure(field: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.stack([matrix @ field[..., channel] @ matrix.T for channel in range(4)], axis=-1).astype(np.float32)


def smoothstep(value: np.ndarray | float) -> np.ndarray:
    z = np.clip(value, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def level_radii(name: str) -> np.ndarray:
    return np.asarray([*RADII[name], 127.5], dtype=np.float64)


def width_for_level(name: str, chain: tuple[int, ...], level: float) -> float:
    radii = level_radii(name)
    widths = np.minimum(256.0, 2.0 * radii + 1.0)
    bounded = min(float(len(chain)), max(0.0, level))
    low = min(len(chain) - 1, int(math.floor(bounded)))
    if bounded >= len(chain):
        return 256.0
    high = low + 1
    fraction = bounded - low
    return float(math.exp((1.0 - fraction) * math.log(widths[low]) + fraction * math.log(widths[high])))


def box_kernel_fft(size: int, width: float, shift_pixels: float) -> np.ndarray:
    """DFT of exact overlap weights for a periodic piecewise-constant signal."""
    center = 0.5 + shift_pixels
    low = center - width * 0.5
    high = center + width * 0.5
    weights = np.zeros(size, dtype=np.float64)
    first = int(math.floor(low))
    last = int(math.ceil(high)) - 1
    for unwrapped in range(first, last + 1):
        overlap = max(0.0, min(high, unwrapped + 1.0) - max(low, unwrapped))
        weights[unwrapped % size] += overlap / width
    kernel = np.empty(size, dtype=np.float64)
    for offset in range(size):
        kernel[offset] = weights[(-offset) % size]
    return np.fft.fft(kernel)


def filtered_truth(exact_fft: np.ndarray, width: float, shift_x: float, shift_z: float) -> np.ndarray:
    hx = box_kernel_fft(256, width, shift_x)
    hz = box_kernel_fft(256, width, shift_z)
    transfer = hz[:, None] * hx[None, :]
    result = np.empty((256, 256, 4), dtype=np.float32)
    for channel in range(4):
        result[..., channel] = np.fft.ifft2(exact_fft[..., channel] * transfer).real.astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def phase_coordinates(shift_x: float, shift_z: float) -> tuple[np.ndarray, np.ndarray]:
    qx = ((np.arange(256, dtype=np.float64) + 0.5 + shift_x) / 256.0) % 1.0
    qz = ((np.arange(256, dtype=np.float64) + 0.5 + shift_z) / 256.0) % 1.0
    return np.meshgrid(qx, qz)


def boundary_level(chain: tuple[int, ...], qx: np.ndarray, qz: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0.0:
        return np.zeros_like(qx, dtype=np.float64)
    transition = ratio / chain[0]
    value = np.zeros_like(qx, dtype=np.float64)
    for resolution in chain:
        fx = np.mod(qx * resolution, 1.0)
        fz = np.mod(qz * resolution, 1.0)
        dx = np.minimum(fx, 1.0 - fx) / resolution
        dz = np.minimum(fz, 1.0 - fz) / resolution
        distance = np.minimum(dx, dz)
        value += smoothstep(1.0 - distance / transition)
    return np.clip(value, 0.0, float(len(chain)))


def reconstruct(
    levels: list[np.ndarray],
    chain: tuple[int, ...],
    qx: np.ndarray,
    qz: np.ndarray,
    footprint_level: float,
    ratio: float,
) -> np.ndarray:
    boundary = boundary_level(chain, qx, qz, ratio)
    coordinate = np.maximum(boundary, footprint_level)
    maximum = len(chain)
    index = np.minimum(maximum, np.floor(coordinate).astype(np.int32))
    fraction = smoothstep(coordinate - np.floor(coordinate)).astype(np.float32)
    output = np.empty((*qx.shape, 4), dtype=np.float32)
    global_value = levels[-1][0, 0]
    output[...] = global_value
    for level_index, resolution in enumerate(chain):
        mask = index == level_index
        if not np.any(mask):
            continue
        ix = np.floor(qx * resolution).astype(np.int32) % resolution
        iz = np.floor(qz * resolution).astype(np.int32) % resolution
        low = levels[level_index][iz, ix]
        if level_index + 1 < len(chain):
            parent_resolution = chain[level_index + 1]
            px = np.floor(qx * parent_resolution).astype(np.int32) % parent_resolution
            pz = np.floor(qz * parent_resolution).astype(np.int32) % parent_resolution
            high = levels[level_index + 1][pz, px]
        else:
            high = np.broadcast_to(global_value, low.shape)
        blend = fraction[..., None]
        candidate = low * (1.0 - blend) + high * blend
        output[mask] = candidate[mask]
    return output


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def score_error(module, decoded: np.ndarray, truth: np.ndarray, limits: dict[str, float]) -> dict[str, Any]:
    error_a = np.abs(decoded[..., 3] - truth[..., 3])
    error_p = np.max(np.abs(decoded[..., :3] - truth[..., :3]), axis=2)
    support = (decoded[..., 3] > 1.0 / 255.0) | (truth[..., 3] > 1.0 / 255.0)
    exceed = support & ((error_a > limits["a_p99"]) | (error_p > limits["p_p99"]))
    connected = module.largest_periodic_component(exceed) / max(1, int(np.count_nonzero(support)))
    a = quantiles(error_a)
    p = quantiles(error_p)
    green = (
        a["p95"] <= limits["a_p95"]
        and a["p99"] <= limits["a_p99"]
        and p["p95"] <= limits["p_p95"]
        and p["p99"] <= limits["p_p99"]
        and connected < limits["connected"]
    )
    return {"coverage": a, "premulRgb": p, "connected": connected, "green": green}


@dataclass
class Worst:
    coverage_p95: float = -1.0
    coverage_p99: float = -1.0
    premul_p95: float = -1.0
    premul_p99: float = -1.0
    connected: float = -1.0
    detail: dict[str, Any] | None = None

    def add(self, detail: dict[str, Any]) -> None:
        values = (
            detail["coverage"]["p95"], detail["coverage"]["p99"],
            detail["premulRgb"]["p95"], detail["premulRgb"]["p99"], detail["connected"],
        )
        current = (self.coverage_p95, self.coverage_p99, self.premul_p95, self.premul_p99, self.connected)
        if max(values) > max(current):
            self.detail = detail
        self.coverage_p95 = max(self.coverage_p95, values[0])
        self.coverage_p99 = max(self.coverage_p99, values[1])
        self.premul_p95 = max(self.premul_p95, values[2])
        self.premul_p99 = max(self.premul_p99, values[3])
        self.connected = max(self.connected, values[4])

    def report(self, limits: dict[str, float]) -> dict[str, Any]:
        green = (
            self.coverage_p95 <= limits["a_p95"] and self.coverage_p99 <= limits["a_p99"]
            and self.premul_p95 <= limits["p_p95"] and self.premul_p99 <= limits["p_p99"]
            and self.connected < limits["connected"]
        )
        return {
            "coverageP95": self.coverage_p95, "coverageP99": self.coverage_p99,
            "premulRgbP95": self.premul_p95, "premulRgbP99": self.premul_p99,
            "connected": self.connected, "green": green, "worstDetail": self.detail,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    parser.add_argument("--max-nodes", type=int, default=65)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / args.source).resolve()
    source_raw = source.read_bytes()
    threshold_raw = (root / THRESHOLD_DOC).read_bytes()
    module = load_module(root)
    profile = module.load_profile(source)
    if profile.interior_width != 256 or profile.interior_height != 256 or len(profile.slices) != 65:
        raise ValueError("gate requires accepted 65 x 256-square GCRP")
    node_count = min(65, max(1, args.max_nodes))

    matrices = {resolution: area_matrix(resolution) for chain in CHAINS.values() for resolution in chain}
    configurations = [
        (name, chain, ratio)
        for name, chain in CHAINS.items()
        for ratio in BOUNDARY_RATIOS
    ]
    static_levels = {name: [0.5 * step for step in range(0, 2 * len(chain) + 1)] for name, chain in CHAINS.items()}
    translation_levels = {name: [float(step) for step in range(len(chain) + 1)] for name, chain in CHAINS.items()}
    static_worst = {(name, ratio, level): Worst() for name, chain, ratio in configurations for level in static_levels[name]}
    temporal_worst = {
        (name, ratio, level, mm, direction): Worst()
        for name, chain, ratio in configurations
        for level in translation_levels[name]
        for mm in TRANSLATION_MM
        for direction in TRANSLATION_DIRECTIONS
    }

    base_qx, base_qz = phase_coordinates(0.0, 0.0)
    for node in range(node_count):
        covered, _depth, rgb = module.decode_slice_base(profile, node)
        exact = np.empty((256, 256, 4), dtype=np.float32)
        exact[..., :3] = rgb * covered[..., None]
        exact[..., 3] = covered.astype(np.float32)
        exact_fft = np.stack([np.fft.fft2(exact[..., channel]) for channel in range(4)], axis=-1)
        chain_levels: dict[str, list[np.ndarray]] = {}
        for name, chain in CHAINS.items():
            values = []
            for level_index, resolution in enumerate(chain):
                radius = int(level_radii(name)[level_index])
                filtered = np.stack(
                    [ndimage.uniform_filter(exact[..., channel], size=2 * radius + 1, mode="wrap") for channel in range(4)],
                    axis=-1,
                ).astype(np.float32)
                values.append(rgba4444(reduce_measure(filtered, matrices[resolution])))
            values.append(rgba4444(np.mean(exact, axis=(0, 1), keepdims=True)))
            chain_levels[name] = values

        for name, chain, ratio in configurations:
            values = chain_levels[name]
            for footprint_level in static_levels[name]:
                width = width_for_level(name, chain, footprint_level)
                truth = filtered_truth(exact_fft, width, 0.0, 0.0)
                decoded = reconstruct(values, chain, base_qx, base_qz, footprint_level, ratio)
                detail = score_error(module, decoded, truth, STATIC_LIMITS)
                detail.update({"node": node, "level": footprint_level, "widthSourceTexels": width})
                static_worst[(name, ratio, footprint_level)].add(detail)

            for footprint_level in translation_levels[name]:
                width = width_for_level(name, chain, footprint_level)
                truth0 = filtered_truth(exact_fft, width, 0.0, 0.0)
                decoded0 = reconstruct(values, chain, base_qx, base_qz, footprint_level, ratio)
                error0 = decoded0 - truth0
                for mm in TRANSLATION_MM:
                    for direction_name, direction in TRANSLATION_DIRECTIONS.items():
                        shift_x = mm * 0.001 * direction[0] / profile.size_x * 256.0
                        shift_z = mm * 0.001 * direction[1] / profile.size_z * 256.0
                        qx, qz = phase_coordinates(shift_x, shift_z)
                        truth1 = filtered_truth(exact_fft, width, shift_x, shift_z)
                        decoded1 = reconstruct(values, chain, qx, qz, footprint_level, ratio)
                        excess = np.abs((decoded1 - truth1) - error0)
                        temporal_truth = np.zeros_like(excess)
                        detail = score_error(module, excess, temporal_truth, TEMPORAL_LIMITS)
                        detail.update({
                            "node": node, "level": footprint_level, "widthSourceTexels": width,
                            "translationMm": mm, "translationDirection": direction_name,
                        })
                        temporal_worst[(name, ratio, footprint_level, mm, direction_name)].add(detail)
        print(f"[candidate-k2] node {node + 1}/{node_count}", flush=True)

    config_reports: list[dict[str, Any]] = []
    for name, chain, ratio in configurations:
        static = [static_worst[(name, ratio, level)].report(STATIC_LIMITS) | {"level": level} for level in static_levels[name]]
        temporal = [
            temporal_worst[(name, ratio, level, mm, direction)].report(TEMPORAL_LIMITS)
            | {"level": level, "translationMm": mm, "direction": direction}
            for level in translation_levels[name]
            for mm in TRANSLATION_MM
            for direction in TRANSLATION_DIRECTIONS
        ]
        static_green = all(item["green"] for item in static)
        temporal_green = all(item["green"] for item in temporal)
        scale_mib = 128 * 16 * sum(value * value for value in chain) / 1048576.0
        config_reports.append({
            "chain": name, "resolutions": list(chain), "boundaryRatio": ratio,
            "scaleMiB": scale_mib, "totalMiB": scale_mib + 16.25,
            "staticGreen": static_green, "temporalGreen": temporal_green,
            "green": static_green and temporal_green,
            "worstStatic": max(static, key=lambda item: max(item["coverageP95"], item["premulRgbP95"])),
            "worstTemporal": max(temporal, key=lambda item: max(item["coverageP95"], item["premulRgbP95"])),
            "static": static, "temporal": temporal,
        })

    green_configs = [value for value in config_reports if value["green"] and value["boundaryRatio"] > 0]
    chosen: dict[str, Any] | None = None
    if green_configs:
        best_temporal = min(value["worstTemporal"]["premulRgbP95"] for value in green_configs)
        stable = [value for value in green_configs if value["worstTemporal"]["premulRgbP95"] <= best_temporal + 1e-4]
        best_static = min(value["worstStatic"]["premulRgbP95"] for value in stable)
        near_best = [value for value in stable if value["worstStatic"]["premulRgbP95"] <= best_static + 0.005]
        chosen = min(near_best, key=lambda value: (value["scaleMiB"], value["boundaryRatio"]))

    recipe = {
        "version": VERSION,
        "sourceSha256": sha256(source_raw),
        "scriptSha256": sha256(Path(__file__).read_bytes()),
        "thresholdDocSha256": sha256(threshold_raw),
        "nodeCount": node_count,
        "chains": {key: list(value) for key, value in CHAINS.items()},
        "radii": {key: list(value) for key, value in RADII.items()},
        "boundaryRatios": list(BOUNDARY_RATIOS),
        "translationMm": list(TRANSLATION_MM),
        "translationDirections": TRANSLATION_DIRECTIONS,
        "staticLimits": STATIC_LIMITS,
        "temporalLimits": TEMPORAL_LIMITS,
    }
    recipe_hash = sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    output = root / "data/work/groundcover-candidate-k2-pyramid" / recipe["sourceSha256"][:16] / recipe_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        "green": chosen is not None,
        "verdict": "GREEN" if chosen is not None else "RED",
        "chosen": None if chosen is None else {
            "chain": chosen["chain"], "resolutions": chosen["resolutions"],
            "boundaryRatio": chosen["boundaryRatio"], "scaleMiB": chosen["scaleMiB"], "totalMiB": chosen["totalMiB"],
        },
        "scope": "stored angular nodes; static intermediate footprint and millimetric phase translation; angular held-out and lighting separate",
    }
    report = {"recipe": recipe, "decision": decision, "configurations": config_reports}
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    lines = [
        "# Candidate K2 nested-pyramid stability gate", "",
        f"- Verdict: **{decision['verdict']}**", f"- Chosen: `{decision['chosen']}`", f"- Nodes: {node_count}", "",
        "| chain | boundary band | scale MiB | total MiB | static | translation | worst static P p95 | worst temporal P p95 |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for value in config_reports:
        lines.append(
            f"| {value['chain']} | {value['boundaryRatio']:.3f} | {value['scaleMiB']:.6f} | {value['totalMiB']:.6f} "
            f"| {'GREEN' if value['staticGreen'] else 'RED'} | {'GREEN' if value['temporalGreen'] else 'RED'} "
            f"| {value['worstStatic']['premulRgbP95']:.5f} | {value['worstTemporal']['premulRgbP95']:.5f} |"
        )
    lines.extend(["", "Exact command:", "", "```text", "env UV_CACHE_DIR=/tmp/laas-uv-cache uv run --project asset-gen python tools/groundcover-bake/analyze_candidate_k_pyramid_stability.py", "```", ""])
    (output / "SUMMARY.md").write_text("\n".join(lines))
    index = root / "data/work/groundcover-candidate-k2-pyramid/index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"current": str((output / "report.json").relative_to(root)), "decision": decision}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "output": str(output.relative_to(root))}, indent=2))


if __name__ == "__main__":
    main()

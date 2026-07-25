#!/usr/bin/env python3
"""Gate Candidate K's direct angular finite-element scale carrier.

This is an offline codec test only.  Each angular cell texel is RGBA32Uint:
the four u32 channels are its four angular vertices; the low/high 16-bit
halves carry sigma-4/sigma-16 premultiplied RGBA4444 respectively.  The test
area-downsamples the accepted 256-square filtered truth to each proposed page
resolution, quantises exactly, reconstructs at every source phase centre, and
scores every one of the 65 accepted direction nodes separately.
"""

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
from scipy import ndimage


ANALYSIS_VERSION = "candidate-k-direct-fe-v1"
SCALES = (4, 16)
CONFIGURATIONS = ((64, 180), (80, 160), (96, 144), (128, 128))
THRESHOLDS = {
    "coverageP95": 0.08,
    "coverageP99": 0.20,
    "premultipliedRgbP95": 0.06,
    "premultipliedRgbP99": 0.15,
    "connectedExceedanceFraction": 0.01,
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_gate_module(root: Path):
    path = root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py"
    spec = importlib.util.spec_from_file_location("candidate_h_gate_for_k", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_pole_atoms(module, profile, atoms) -> None:
    covered, depth, rgb = module.decode_slice_base(profile, 64)
    for scale in SCALES:
        size = 2 * scale + 1
        coverage = ndimage.uniform_filter(covered.astype(np.float32), size=size, mode="wrap")
        coverage = np.rint(np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
        premul = np.stack(
            [ndimage.uniform_filter(rgb[..., channel] * covered, size=size, mode="wrap") for channel in range(3)],
            axis=-1,
        ).astype(np.float32)
        median = module.toroidal_nanmedian(depth, scale)
        slice_data = profile.slices[64]
        code = np.rint(
            np.clip(
                (median - slice_data.depth_min) / (slice_data.depth_max - slice_data.depth_min),
                0.0,
                1.0,
            )
            * 1023.0
        )
        quantised = slice_data.depth_min + code / 1023.0 * (slice_data.depth_max - slice_data.depth_min)
        quantised[~np.isfinite(median)] = np.nan
        atom = atoms[scale]
        atom.coverage = np.concatenate((atom.coverage, coverage[None]), axis=0)
        atom.depth = np.concatenate((atom.depth, quantised[None].astype(np.float32)), axis=0)
        atom.premul_rgb = np.concatenate((atom.premul_rgb, premul[None]), axis=0)


def area_matrix(output_size: int, input_size: int = 256) -> np.ndarray:
    """Exact separable box integration from input pixel cells to output cells."""
    result = np.zeros((output_size, input_size), dtype=np.float32)
    span = input_size / output_size
    for output in range(output_size):
        low = output * span
        high = (output + 1) * span
        first = int(np.floor(low))
        last = min(input_size - 1, int(np.ceil(high) - 1))
        for source in range(first, last + 1):
            overlap = max(0.0, min(high, source + 1.0) - max(low, source))
            result[output, source] = overlap / span
    if float(np.max(np.abs(np.sum(result, axis=1) - 1.0))) > 1e-6:
        raise RuntimeError("area matrix is not a partition of unity")
    return result


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def heat(values: np.ndarray, limit: float) -> np.ndarray:
    amount = np.clip(values / max(limit, 1e-9), 0.0, 1.0)
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(255.0 * amount).astype(np.uint8)
    rgb[..., 1] = np.rint(210.0 * np.minimum(2.0 * amount, 1.0) * (1.0 - 0.55 * amount)).astype(np.uint8)
    rgb[..., 2] = np.rint(35.0 * (1.0 - amount)).astype(np.uint8)
    return rgb


def write_diagnostic(path: Path, title: str, maps: list[tuple[str, np.ndarray, float]]) -> None:
    panel_size = 512
    header = 48
    canvas = Image.new("RGB", (panel_size * 2, header + panel_size * 2), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))
    for index, (label, values, limit) in enumerate(maps):
        x = (index % 2) * panel_size
        y = header + (index // 2) * panel_size
        image = Image.fromarray(heat(values, limit), mode="RGB").resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        )
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + 8), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_path = (root / args.source).resolve()
    source_raw = source_path.read_bytes()
    source_hash = sha256(source_raw)
    script_hash = sha256(Path(__file__).read_bytes())
    recipe = {
        "analysisVersion": ANALYSIS_VERSION,
        "sourceSha256": source_hash,
        "scriptSha256": script_hash,
        "scales": list(SCALES),
        "configurations": [{"pages": pages, "resolution": resolution} for pages, resolution in CONFIGURATIONS],
        "spatialReduction": "exact separable area average of coverage and premultiplied linear RGB",
        "reconstruction": "nearest page texel at each accepted 256-square phase centre",
        "packing": "RGBA32Uint: one angular vertex/channel; sigma4 low16 and sigma16 high16; each half premultiplied RGBA4444",
        "thresholds": THRESHOLDS,
    }
    recipe_hash = sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    output = root / "data/work/groundcover-candidate-k-direct-fe" / source_hash[:16] / recipe_hash[:16]
    qa = output / "qa"
    output.mkdir(parents=True, exist_ok=True)

    module = load_gate_module(root)
    profile = module.load_profile(source_path)
    if profile.interior_width != 256 or profile.interior_height != 256 or len(profile.slices) != 65:
        raise ValueError("Candidate K gate requires the accepted 256-square, 65-direction profile")
    atoms, _ = module.build_atoms(profile)
    add_pole_atoms(module, profile, atoms)

    report_configs: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    for image_number, (pages, resolution) in enumerate(CONFIGURATIONS, start=1):
        matrix = area_matrix(resolution)
        sample_index = np.minimum(
            resolution - 1,
            np.floor((np.arange(256, dtype=np.float64) + 0.5) * resolution / 256.0).astype(np.int64),
        )
        scale_reports: dict[str, Any] = {}
        diagnostic_maps: list[tuple[str, np.ndarray, float]] = []
        configuration_green = True
        for scale in SCALES:
            node_reports: list[dict[str, Any]] = []
            worst_a_map: np.ndarray | None = None
            worst_rgb_map: np.ndarray | None = None
            worst_a_p95 = -1.0
            worst_rgb_p95 = -1.0
            for node in range(65):
                coverage = atoms[scale].coverage[node]
                premul = atoms[scale].premul_rgb[node]
                reduced_a = matrix @ coverage @ matrix.T
                reduced_p = np.stack(
                    [matrix @ premul[..., channel] @ matrix.T for channel in range(3)], axis=-1
                )
                encoded_a = np.rint(np.clip(reduced_a, 0.0, 1.0) * 15.0) / 15.0
                encoded_p = np.rint(np.clip(reduced_p, 0.0, 1.0) * 15.0) / 15.0
                decoded_a = encoded_a[sample_index[:, None], sample_index[None, :]]
                decoded_p = encoded_p[sample_index[:, None], sample_index[None, :]]
                error_a = np.abs(decoded_a - coverage)
                error_rgb = np.max(np.abs(decoded_p - premul), axis=2)
                support = (coverage > 1.0 / 255.0) | (decoded_a > 1.0 / 255.0)
                exceed = support & (
                    (error_a > THRESHOLDS["coverageP99"])
                    | (error_rgb > THRESHOLDS["premultipliedRgbP99"])
                )
                connected = module.largest_periodic_component(exceed) / max(1, int(np.count_nonzero(support)))
                metric_a = quantiles(error_a)
                metric_rgb = quantiles(error_rgb)
                green = (
                    metric_a["p95"] <= THRESHOLDS["coverageP95"]
                    and metric_a["p99"] <= THRESHOLDS["coverageP99"]
                    and metric_rgb["p95"] <= THRESHOLDS["premultipliedRgbP95"]
                    and metric_rgb["p99"] <= THRESHOLDS["premultipliedRgbP99"]
                    and connected < THRESHOLDS["connectedExceedanceFraction"]
                )
                node_reports.append(
                    {
                        "node": node,
                        "coverageError": metric_a,
                        "premultipliedRgbMaxChannelError": metric_rgb,
                        "largestConnectedExceedanceFraction": connected,
                        "green": green,
                    }
                )
                if metric_a["p95"] > worst_a_p95:
                    worst_a_p95 = metric_a["p95"]
                    worst_a_map = error_a
                if metric_rgb["p95"] > worst_rgb_p95:
                    worst_rgb_p95 = metric_rgb["p95"]
                    worst_rgb_map = error_rgb

            worst_a = max(node_reports, key=lambda value: value["coverageError"]["p95"])
            worst_rgb = max(node_reports, key=lambda value: value["premultipliedRgbMaxChannelError"]["p95"])
            worst_connected = max(node_reports, key=lambda value: value["largestConnectedExceedanceFraction"])
            scale_green = all(bool(value["green"]) for value in node_reports)
            configuration_green = configuration_green and scale_green
            scale_reports[f"sigma{scale}"] = {
                "green": scale_green,
                "worstCoverageNode": worst_a,
                "worstPremultipliedRgbNode": worst_rgb,
                "worstConnectedNode": worst_connected,
                "nodes": node_reports,
            }
            if worst_a_map is None or worst_rgb_map is None:
                raise RuntimeError("no diagnostic map")
            diagnostic_maps.extend(
                [
                    (f"sigma{scale} worst |delta A| / 0.20", worst_a_map, 0.20),
                    (f"sigma{scale} worst premul RGB / 0.15", worst_rgb_map, 0.15),
                ]
            )

        filename = f"{image_number:03d}-{pages}pages-{resolution}px.png"
        image_path = qa / filename
        write_diagnostic(image_path, f"Candidate K: {pages} pages at {resolution}x{resolution}", diagnostic_maps)
        image_records.append(
            {
                "number": image_number,
                "file": filename,
                "sha256": sha256(image_path.read_bytes()),
                "dimensions": [1024, 1072],
                "interpretation": "Worst per-direction phase errors after exact area reduction and direct premultiplied RGBA4444 quantisation. Colour reaches the frozen p99 threshold at red.",
            }
        )
        report_configs.append(
            {
                "pages": pages,
                "resolution": resolution,
                "scaleBytes": pages * resolution * resolution * 16,
                "scaleMiB": pages * resolution * resolution * 16 / (1024.0 * 1024.0),
                "green": configuration_green,
                "scales": scale_reports,
            }
        )

    green = all(bool(value["green"]) for value in report_configs)
    report = {
        "recipe": {**recipe, "recipeSha256": recipe_hash},
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": source_hash,
            "bytes": len(source_raw),
            "directions": 65,
            "dimensions": [256, 256],
        },
        "decision": {
            "green": green,
            "verdict": "GREEN" if green else "RED",
            "scope": "stored-node spatial reduction and RGBA4444 quantisation only; held-out angular interpolation remains a separate gate",
        },
        "configurations": report_configs,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    report_hash = sha256(report_text.encode())
    qa_index = {
        "schema": "laas-groundcover-candidate-k-direct-fe-qa/v1",
        "sourceSha256": source_hash,
        "recipeSha256": recipe_hash,
        "scriptSha256": script_hash,
        "reportSha256": report_hash,
        "images": image_records,
    }
    qa_text = json.dumps(qa_index, indent=2, sort_keys=True) + "\n"
    (qa / "index.json").write_text(qa_text)

    lines = [
        "# Candidate K direct finite-element node gate",
        "",
        f"- Verdict: **{'GREEN' if green else 'RED'}**",
        f"- Source SHA-256: `{source_hash}`",
        f"- Recipe SHA-256: `{recipe_hash}`",
        f"- Script SHA-256: `{script_hash}`",
        f"- Report SHA-256: `{report_hash}`",
        "- Scope: accepted 65 direction nodes; held-out within-cell angular fidelity is not claimed.",
        "",
        "| Pages | Resolution | Scale MiB | sigma | worst A p95/p99 | worst premul RGB p95/p99 | max connected | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for configuration in report_configs:
        for scale in SCALES:
            value = configuration["scales"][f"sigma{scale}"]
            a = value["worstCoverageNode"]["coverageError"]
            rgb = value["worstPremultipliedRgbNode"]["premultipliedRgbMaxChannelError"]
            connected = value["worstConnectedNode"]["largestConnectedExceedanceFraction"]
            lines.append(
                f"| {configuration['pages']} | {configuration['resolution']} | {configuration['scaleMiB']:.3f} "
                f"| {scale} | {a['p95']:.4f}/{a['p99']:.4f} | {rgb['p95']:.4f}/{rgb['p99']:.4f} "
                f"| {100.0 * connected:.3f}% | {'GREEN' if value['green'] else 'RED'} |"
            )
    lines.extend(
        [
            "",
            "Frozen limits: A p95/p99 <= 0.08/0.20; premultiplied RGB p95/p99 <= 0.06/0.15; largest connected p99 exceedance < 1%.",
            "",
            "Exact command:",
            "",
            "```text",
            "env UV_CACHE_DIR=/tmp/laas-uv-cache uv run --project asset-gen python tools/groundcover-bake/analyze_candidate_k_direct_fe.py",
            "```",
            "",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines))
    root_index = root / "data/work/groundcover-candidate-k-direct-fe/index.json"
    root_index.parent.mkdir(parents=True, exist_ok=True)
    root_index.write_text(
        json.dumps(
            {
                "current": str((output / "report.json").relative_to(root)),
                "sourceSha256": source_hash,
                "recipeSha256": recipe_hash,
                "scriptSha256": script_hash,
                "reportSha256": report_hash,
                "verdict": "GREEN" if green else "RED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": "GREEN" if green else "RED",
                "output": str(output.relative_to(root)),
                "reportSha256": report_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

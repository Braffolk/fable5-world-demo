#!/usr/bin/env python3
"""Fit the first same-read direction-analytic filtered-profile codec.

This is an offline mathematical gate, not runtime code.  One periodic phase
cell owns two four-byte latent vectors (one per spatial scale).  A tiny shared
continuous decoder consumes both latents, live direction, and the continuous
scale coordinate and reconstructs coverage, premultiplied colour, and vertical
representative depth.  No angular chart is selected by the model.
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
import torch
from scipy import ndimage


SCALES = (4, 16)
THRESHOLDS = {
    "coverage_p95": 0.08,
    "coverage_p99": 0.20,
    "premul_rgb_p95": 0.06,
    "premul_rgb_p99": 0.15,
    "position_m_p95": 0.05,
    "position_m_p99": 0.10,
    "connected_fraction": 0.01,
}
SEED = 0xCADA2026


def load_gate_module(root: Path) -> Any:
    path = root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py"
    spec = importlib.util.spec_from_file_location("candidate_h_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_truth(gate: Any, profile: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [scale,direction,phase] A, premul RGB, and vertical depth."""
    count = len(profile.slices)
    height = profile.interior_height
    width = profile.interior_width
    phase_count = height * width
    coverage = np.empty((2, count, phase_count), dtype=np.float32)
    premul = np.empty((2, count, phase_count, 3), dtype=np.float32)
    vertical = np.empty((2, count, phase_count), dtype=np.float32)
    for slice_index in range(count):
        hit, depth, rgb = gate.decode_slice_base(profile, slice_index)
        for scale_index, radius in enumerate(SCALES):
            size = 2 * radius + 1
            a = ndimage.uniform_filter(hit.astype(np.float32), size=size, mode="wrap")
            a = np.rint(np.clip(a, 0.0, 1.0) * 255.0) / 255.0
            p = np.empty((*hit.shape, 3), dtype=np.float32)
            for channel in range(3):
                p[..., channel] = ndimage.uniform_filter(
                    rgb[..., channel] * hit,
                    size=size,
                    mode="wrap",
                )
            median = gate.toroidal_nanmedian(depth, radius)
            slice_data = profile.slices[slice_index]
            normalized = (median - slice_data.depth_min) / (slice_data.depth_max - slice_data.depth_min)
            code = np.rint(np.clip(normalized, 0.0, 1.0) * 1023.0)
            tau = slice_data.depth_min + code / 1023.0 * (slice_data.depth_max - slice_data.depth_min)
            # Downward distance from the common top plane.  Empty atoms carry a
            # finite midpoint as required by H; coverage keeps it out of moments.
            z = np.clip((-tau * float(slice_data.direction[1])) / profile.top_h, 0.0, 1.0)
            z[~np.isfinite(median)] = 0.5
            coverage[scale_index, slice_index] = a.reshape(-1)
            premul[scale_index, slice_index] = p.reshape(-1, 3)
            vertical[scale_index, slice_index] = z.astype(np.float32).reshape(-1)
        print(f"[analytic-latent] truth slice {slice_index + 1}/{count}", flush=True)
    return coverage, premul, vertical


class Decoder(torch.nn.Module):
    """Six-hidden-unit decoder: 102 MACs for the final two-scale result."""

    def __init__(self, phase_count: int) -> None:
        super().__init__()
        self.raw_latent = torch.nn.Parameter(torch.zeros(phase_count, 8))
        torch.nn.init.normal_(self.raw_latent, mean=0.0, std=0.35)
        self.hidden = torch.nn.Linear(12, 6)
        self.output = torch.nn.Linear(6, 5)

    def latent(self, phase: torch.Tensor, quantized: bool) -> torch.Tensor:
        value = torch.sigmoid(self.raw_latent[phase])
        if quantized:
            snapped = torch.round(value * 255.0) / 255.0
            value = value + (snapped - value).detach()
        return value

    def forward(
        self,
        phase: torch.Tensor,
        direction: torch.Tensor,
        scale_mix: torch.Tensor,
        quantized: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.latent(phase, quantized)
        x = torch.cat((code, direction, scale_mix[:, None]), dim=1)
        hidden = torch.nn.functional.silu(self.hidden(x))
        raw = self.output(hidden)
        a = torch.sigmoid(raw[:, 0])
        conditional_rgb = torch.sigmoid(raw[:, 1:4])
        z = torch.sigmoid(raw[:, 4])
        return a, a[:, None] * conditional_rgb, z


def gather_target(
    coverage: torch.Tensor,
    premul: torch.Tensor,
    vertical: torch.Tensor,
    direction_index: torch.Tensor,
    phase: torch.Tensor,
    mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a0 = coverage[0, direction_index, phase]
    a1 = coverage[1, direction_index, phase]
    p0 = premul[0, direction_index, phase]
    p1 = premul[1, direction_index, phase]
    z0 = vertical[0, direction_index, phase]
    z1 = vertical[1, direction_index, phase]
    a = torch.lerp(a0, a1, mix)
    p = torch.lerp(p0, p1, mix[:, None])
    numerator = torch.lerp(a0 * z0, a1 * z1, mix)
    z = torch.where(a > 1.0 / 255.0, numerator / torch.clamp(a, min=1.0e-6), torch.full_like(a, 0.5))
    return a, p, z


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def largest_component(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    tiled = np.tile(mask, (3, 3))
    labels, _ = ndimage.label(tiled, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    height, width = mask.shape
    center = labels[height : 2 * height, width : 2 * width]
    counts = np.bincount(center.ravel())
    return int(np.max(counts[1:])) if counts.size > 1 else 0


def evaluate(
    model: Decoder,
    directions: torch.Tensor,
    coverage: np.ndarray,
    premul: np.ndarray,
    vertical: np.ndarray,
    top_h: float,
    height: int,
    width: int,
) -> dict[str, Any]:
    model.eval()
    phase_count = height * width
    phase = torch.arange(phase_count, dtype=torch.long)
    results: dict[str, Any] = {}
    all_green = True
    with torch.no_grad():
        for scale_index, radius in enumerate(SCALES):
            groups = []
            for direction_index in range(directions.shape[0]):
                pred_a: list[np.ndarray] = []
                pred_p: list[np.ndarray] = []
                pred_z: list[np.ndarray] = []
                for start in range(0, phase_count, 16_384):
                    end = min(phase_count, start + 16_384)
                    q = phase[start:end]
                    d = directions[direction_index].expand(end - start, -1)
                    mix = torch.full((end - start,), float(scale_index))
                    a, p, z = model(q, d, mix, quantized=True)
                    pred_a.append(a.cpu().numpy())
                    pred_p.append(p.cpu().numpy())
                    pred_z.append(z.cpu().numpy())
                a_hat = np.concatenate(pred_a)
                p_hat = np.concatenate(pred_p)
                z_hat = np.concatenate(pred_z)
                a_truth = coverage[scale_index, direction_index]
                p_truth = premul[scale_index, direction_index]
                z_truth = vertical[scale_index, direction_index]
                covered = a_truth > 1.0 / 255.0
                delta_a = np.abs(a_hat - a_truth)
                delta_rgb = np.max(np.abs(p_hat - p_truth), axis=1)
                mu = max(1.0e-6, -float(directions[direction_index, 1]))
                delta_position = top_h * np.abs(z_hat - z_truth) / mu
                exceed = (
                    (delta_a > THRESHOLDS["coverage_p99"])
                    | (delta_rgb > THRESHOLDS["premul_rgb_p99"])
                    | (covered & (delta_position > THRESHOLDS["position_m_p99"]))
                )
                support = max(1, int(np.count_nonzero(covered | (a_hat > 1.0 / 255.0))))
                connected = largest_component(exceed.reshape(height, width)) / support
                groups.append({
                    "direction": direction_index,
                    "coverage": quantiles(delta_a),
                    "premulRgb": quantiles(delta_rgb),
                    "positionMetres": quantiles(delta_position[covered]),
                    "connectedFraction": connected,
                })
            merged = {
                "coverageP95Worst": max(item["coverage"]["p95"] for item in groups),
                "coverageP99Worst": max(item["coverage"]["p99"] for item in groups),
                "rgbP95Worst": max(item["premulRgb"]["p95"] for item in groups),
                "rgbP99Worst": max(item["premulRgb"]["p99"] for item in groups),
                "positionP95Worst": max(item["positionMetres"]["p95"] for item in groups),
                "positionP99Worst": max(item["positionMetres"]["p99"] for item in groups),
                "connectedWorst": max(item["connectedFraction"] for item in groups),
            }
            checks = {
                "coverageP95": merged["coverageP95Worst"] <= THRESHOLDS["coverage_p95"],
                "coverageP99": merged["coverageP99Worst"] <= THRESHOLDS["coverage_p99"],
                "rgbP95": merged["rgbP95Worst"] <= THRESHOLDS["premul_rgb_p95"],
                "rgbP99": merged["rgbP99Worst"] <= THRESHOLDS["premul_rgb_p99"],
                "positionP95": merged["positionP95Worst"] <= THRESHOLDS["position_m_p95"],
                "positionP99": merged["positionP99Worst"] <= THRESHOLDS["position_m_p99"],
                "connected": merged["connectedWorst"] < THRESHOLDS["connected_fraction"],
            }
            green = all(checks.values())
            all_green = all_green and green
            results[f"sigma{radius}"] = {"green": green, "checks": checks, "worst": merged, "directions": groups}
    return {"green": all_green, "scales": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=8192)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    gate = load_gate_module(root)
    profile = gate.load_profile(root / args.source)
    coverage_np, premul_np, vertical_np = build_truth(gate, profile)
    directions_np = np.stack([item.direction for item in profile.slices]).astype(np.float32)
    coverage = torch.from_numpy(coverage_np)
    premul = torch.from_numpy(premul_np)
    vertical = torch.from_numpy(vertical_np)
    directions = torch.from_numpy(directions_np)
    phase_count = profile.interior_width * profile.interior_height
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    model = Decoder(phase_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-6)
    generator = torch.Generator().manual_seed(SEED)
    for step in range(args.steps):
        phase = torch.randint(0, phase_count, (args.batch,), generator=generator)
        direction_index = torch.randint(0, len(profile.slices), (args.batch,), generator=generator)
        mix = torch.rand(args.batch, generator=generator)
        target_a, target_p, target_z = gather_target(coverage, premul, vertical, direction_index, phase, mix)
        d = directions[direction_index]
        predicted_a, predicted_p, predicted_z = model(phase, d, mix, quantized=step >= args.steps // 2)
        coverage_loss = torch.nn.functional.smooth_l1_loss(predicted_a, target_a, beta=0.04) / THRESHOLDS["coverage_p95"]
        rgb_loss = torch.nn.functional.smooth_l1_loss(predicted_p, target_p, beta=0.03) / THRESHOLDS["premul_rgb_p95"]
        covered = target_a > 1.0 / 255.0
        if torch.any(covered):
            mu = torch.clamp(-d[covered, 1], min=1.0e-3)
            position_error = profile.top_h * (predicted_z[covered] - target_z[covered]) / mu
            position_loss = torch.nn.functional.smooth_l1_loss(
                position_error,
                torch.zeros_like(position_error),
                beta=0.025,
            ) / THRESHOLDS["position_m_p95"]
        else:
            position_loss = torch.zeros(())
        loss = coverage_loss + rgb_loss + 1.5 * position_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            print(
                f"[analytic-latent] step={step + 1}/{args.steps} loss={float(loss):.5f} "
                f"A={float(coverage_loss):.5f} RGB={float(rgb_loss):.5f} P={float(position_loss):.5f}",
                flush=True,
            )
    result = evaluate(
        model,
        directions,
        coverage_np,
        premul_np,
        vertical_np,
        profile.top_h,
        profile.interior_height,
        profile.interior_width,
    )
    source_sha = hashlib.sha256((root / args.source).read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema": "laas-direction-analytic-latent-codec/v1",
        "sourceSha256": source_sha,
        "analyzerSha256": script_sha,
        "model": {
            "latent": "two four-component UNORM8 vectors in one RG32Uint texel",
            "decoder": "12 inputs -> 6 SiLU hidden -> 5 outputs; 102 matrix MACs for one final two-scale decode",
            "directionContinuous": True,
            "scaleContinuous": True,
            "profileReads": 1,
            "residentBytesDelta": 0,
        },
        "training": {"steps": args.steps, "batch": args.batch, "seed": SEED},
        "thresholds": THRESHOLDS,
        "fidelity": result,
        "verdict": "GREEN" if result["green"] else "RED",
    }
    output = root / "data/work/groundcover-direction-analytic-latent" / source_sha[:16] / script_sha[:16]
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # The fitted shader constants and per-phase latent codes are emitted only
    # after GREEN; a RED experiment must not create an installable carrier.
    if result["green"]:
        latent = torch.round(torch.sigmoid(model.raw_latent.detach()) * 255.0).to(torch.uint8).cpu().numpy()
        np.save(output / "latent-u8.npy", latent)
        torch.save(model.state_dict(), output / "decoder.pt")
    print(json.dumps({
        "verdict": report["verdict"],
        "output": str(output.relative_to(root)),
        "scales": {
            name: {"green": value["green"], "worst": value["worst"]}
            for name, value in result["scales"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

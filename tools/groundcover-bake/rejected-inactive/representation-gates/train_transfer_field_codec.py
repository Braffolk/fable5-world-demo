#!/usr/bin/env python3
"""Fit/evaluate the rank-4/8 factorized marked transfer-field candidate.

The runtime-shaped model is exactly one periodic/clamped trilinear spatial
coefficient read, one bilinear full-sphere angular coefficient read, a
componentwise product, and one fixed linear output map for rank four.  All
mesh traversal, beam tracing, optimization, metrics, and QA are offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


FIELDS = 31
STRATA = 3
OUTPUTS = 24
MAX_BYTES = 51_121_152
MAX_READS = 4
MAX_FMAS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path, count: int) -> np.memmap:
    values = np.memmap(path, dtype="<f4", mode="r")
    if values.size != count * FIELDS:
        raise ValueError(f"{path} has {values.size} floats; expected {count * FIELDS}")
    return values.reshape(count, FIELDS)


def batches(indices: torch.Tensor, batch_size: int):
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def selected_tensor(source: np.ndarray, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = np.asarray(source[np.asarray(indices.cpu())], dtype=np.float32)
    return torch.from_numpy(selected.copy()).to(device)


def oct_encode_torch(direction: torch.Tensor) -> torch.Tensor:
    value = direction / direction.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z = value[:, 0], value[:, 1], value[:, 2]
    folded_x = (1.0 - y.abs()) * torch.where(x >= 0, 1.0, -1.0)
    folded_y = (1.0 - x.abs()) * torch.where(y >= 0, 1.0, -1.0)
    x = torch.where(z < 0, folded_x, x)
    y = torch.where(z < 0, folded_y, y)
    return torch.stack((x * 0.5 + 0.5, y * 0.5 + 0.5), dim=1)


def oct_encode_numpy(direction: np.ndarray) -> np.ndarray:
    value = direction / np.maximum(np.sum(np.abs(direction), axis=-1, keepdims=True), 1e-12)
    x, y, z = value[..., 0], value[..., 1], value[..., 2]
    old_x = x.copy()
    x = np.where(z < 0, (1 - np.abs(y)) * np.where(old_x >= 0, 1, -1), x)
    y = np.where(z < 0, (1 - np.abs(old_x)) * np.where(y >= 0, 1, -1), y)
    return np.stack((x * 0.5 + 0.5, y * 0.5 + 0.5), axis=-1)


def oct_decode_torch(value: torch.Tensor) -> torch.Tensor:
    x = value[:, :, 0] * 2.0 - 1.0
    y = value[:, :, 1] * 2.0 - 1.0
    z = 1.0 - x.abs() - y.abs()
    old_x = x
    x = torch.where(z < 0, (1.0 - y.abs()) * torch.where(old_x >= 0, 1.0, -1.0), x)
    y = torch.where(z < 0, (1.0 - old_x.abs()) * torch.where(y >= 0, 1.0, -1.0), y)
    return torch.nn.functional.normalize(torch.stack((x, y, z), dim=2), dim=2)


def oct_decode_numpy(value: np.ndarray) -> np.ndarray:
    x = value[..., 0] * 2 - 1
    y = value[..., 1] * 2 - 1
    z = 1 - np.abs(x) - np.abs(y)
    old_x = x.copy()
    x = np.where(z < 0, (1 - np.abs(y)) * np.where(old_x >= 0, 1, -1), x)
    y = np.where(z < 0, (1 - np.abs(old_x)) * np.where(y >= 0, 1, -1), y)
    result = np.stack((x, y, z), axis=-1)
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def distance_u(value: torch.Tensor, tile: float, horizon: float) -> torch.Tensor:
    scale = math.log2(1.0 + horizon / tile)
    return torch.log2(1.0 + value / tile) / scale


def distance_u_numpy(value: np.ndarray, tile: float, horizon: float) -> np.ndarray:
    return np.log2(1.0 + value / tile) / math.log2(1.0 + horizon / tile)


def decode_distance(value: np.ndarray, tile: float, horizon: float) -> np.ndarray:
    return tile * (np.exp2(np.clip(value, 0, 1) * math.log2(1.0 + horizon / tile)) - 1.0)


class FourierEncoding(torch.nn.Module):
    def __init__(self, dimensions: int, powers: int, device: torch.device):
        super().__init__()
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(powers, dtype=torch.float32, device=device), persistent=False
        )
        self.dimensions = dimensions
        self.output_dimensions = dimensions * (1 + powers * 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = value.unsqueeze(2) * self.frequencies.view(1, 1, -1) * (2 * math.pi)
        return torch.cat((value, torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)), dim=1)


class TransferFieldCodec(torch.nn.Module):
    """Continuous offline factor generator; only its sampled U/V tables run live."""

    def __init__(self, rank: int, device: torch.device):
        super().__init__()
        self.rank = rank
        self.spatial_encoding = FourierEncoding(3, 7, device)
        self.spatial = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_encoding.output_dimensions, 64, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(64, rank, device=device),
        )
        self.directional = torch.nn.Sequential(
            torch.nn.Linear(3, 32, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(32, rank, device=device),
        )
        self.output = torch.nn.Linear(rank, OUTPUTS, device=device)
        torch.nn.init.normal_(self.output.weight, 0.0, 0.02)
        torch.nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[22] = 0.55
            self.output.bias[23] = 0.20

    def spatial_factor(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.spatial_encoding(xyz))

    def angular_factor(self, direction: torch.Tensor) -> torch.Tensor:
        return self.directional(direction)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.output(self.spatial_factor(batch[:, :3]) * self.angular_factor(batch[:, 3:6]))


def transformed_output(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    coverage = torch.sigmoid(raw[:, 0])
    bounded = raw + (raw.clamp(0, 1) - raw).detach()
    depth_u = bounded[:, 1:4]
    sigma_u = bounded[:, 4:7]
    colour = bounded[:, 7:16].reshape(-1, STRATA, 3)
    normal_oct = bounded[:, 16:22].reshape(-1, STRATA, 2)
    return coverage, depth_u, sigma_u, colour, normal_oct, oct_decode_torch(normal_oct)


def truth_tensors(batch: torch.Tensor, tile: float, horizon: float) -> tuple[torch.Tensor, ...]:
    coverage = batch[:, 6]
    depth = torch.stack([batch[:, 7], batch[:, 15], batch[:, 23]], dim=1)
    variance = torch.stack([batch[:, 8], batch[:, 16], batch[:, 24]], dim=1)
    colour = torch.stack([batch[:, 9:12], batch[:, 17:20], batch[:, 25:28]], dim=1)
    normal = torch.stack([batch[:, 12:15], batch[:, 20:23], batch[:, 28:31]], dim=1)
    sigma = torch.sqrt(variance.clamp_min(0))
    return (
        coverage,
        distance_u(depth, tile, horizon),
        distance_u(sigma, tile, horizon),
        colour,
        torch.nn.functional.normalize(normal, dim=2),
    )


@torch.no_grad()
def predict_continuous(
    model: TransferFieldCodec,
    source: np.ndarray,
    batch_size: int,
    device: torch.device,
    tile: float,
    horizon: float,
) -> dict[str, np.ndarray]:
    outputs: list[np.ndarray] = []
    for indices in batches(torch.arange(len(source), dtype=torch.int64), batch_size):
        batch = selected_tensor(source, indices, device)
        outputs.append(model(batch).cpu().numpy())
    raw = np.concatenate(outputs)
    bounded = np.clip(raw, 0, 1)
    coverage = 1 / (1 + np.exp(-np.clip(raw[:, 0], -20, 20)))
    depth = decode_distance(bounded[:, 1:4], tile, horizon)
    sigma = decode_distance(bounded[:, 4:7], tile, horizon)
    colour = bounded[:, 7:16].reshape(-1, STRATA, 3)
    normal_oct = bounded[:, 16:22].reshape(-1, STRATA, 2)
    return {
        "raw": raw,
        "coverage": coverage,
        "depth": depth,
        "sigma": sigma,
        "colour": colour,
        "normal": oct_decode_numpy(normal_oct),
    }


@torch.no_grad()
def sample_factor_tables(
    model: TransferFieldCodec,
    dims: tuple[int, int, int],
    angular_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = dims
    spatial_chunks: list[np.ndarray] = []
    total = nx * ny * nz
    for start in range(0, total, batch_size):
        index = np.arange(start, min(total, start + batch_size), dtype=np.int64)
        ix = index % nx
        iz = (index // nx) % nz
        iy = index // (nx * nz)
        xyz = np.stack(((ix + 0.5) / nx, (iy + 0.5) / ny, (iz + 0.5) / nz), axis=1).astype(np.float32)
        spatial_chunks.append(model.spatial_factor(torch.from_numpy(xyz).to(device)).cpu().numpy())
    spatial = np.concatenate(spatial_chunks).reshape(ny, nz, nx, model.rank)
    angular_chunks: list[np.ndarray] = []
    total_angular = angular_size * angular_size
    for start in range(0, total_angular, batch_size):
        index = np.arange(start, min(total_angular, start + batch_size), dtype=np.int64)
        ix = index % angular_size
        iy = index // angular_size
        uv = np.stack(((ix + 0.5) / angular_size, (iy + 0.5) / angular_size), axis=1).astype(np.float32)
        direction = oct_decode_numpy(uv)
        angular_chunks.append(model.angular_factor(torch.from_numpy(direction).to(device)).cpu().numpy())
    angular = np.concatenate(angular_chunks).reshape(angular_size, angular_size, model.rank)
    return spatial, angular


def runtime_factors(
    source: np.ndarray,
    spatial: np.ndarray,
    angular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nz, nx, rank = spatial.shape
    xyz = np.asarray(source[:, :3])
    gx, gy, gz = xyz[:, 0] * nx - 0.5, xyz[:, 1] * ny - 0.5, xyz[:, 2] * nz - 0.5
    x0f, y0f, z0f = np.floor(gx), np.floor(gy), np.floor(gz)
    fx, fy, fz = gx - x0f, gy - y0f, gz - z0f
    x0, z0 = x0f.astype(np.int64) % nx, z0f.astype(np.int64) % nz
    x1, z1 = (x0 + 1) % nx, (z0 + 1) % nz
    y0 = np.clip(y0f.astype(np.int64), 0, ny - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    spatial_result = np.zeros((len(source), rank), dtype=np.float32)
    for iy, wy in ((y0, 1 - fy), (y1, fy)):
        for iz, wz in ((z0, 1 - fz), (z1, fz)):
            for ix, wx in ((x0, 1 - fx), (x1, fx)):
                spatial_result += spatial[iy, iz, ix] * (wx * wy * wz)[:, None]

    uv = oct_encode_numpy(np.asarray(source[:, 3:6]))
    angular_size = angular.shape[0]
    gx, gy = uv[:, 0] * angular_size - 0.5, uv[:, 1] * angular_size - 0.5
    x0f, y0f = np.floor(gx), np.floor(gy)
    fx, fy = gx - x0f, gy - y0f
    x0 = np.clip(x0f.astype(np.int64), 0, angular_size - 1)
    x1 = np.clip(x0 + 1, 0, angular_size - 1)
    y0 = np.clip(y0f.astype(np.int64), 0, angular_size - 1)
    y1 = np.clip(y0 + 1, 0, angular_size - 1)
    angular_result = np.zeros((len(source), rank), dtype=np.float32)
    for iy, wy in ((y0, 1 - fy), (y1, fy)):
        for ix, wx in ((x0, 1 - fx), (x1, fx)):
            angular_result += angular[iy, ix] * (wx * wy)[:, None]
    return spatial_result, angular_result


def runtime_predict(
    source: np.ndarray,
    spatial: np.ndarray,
    angular: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    tile: float,
    horizon: float,
) -> dict[str, np.ndarray]:
    spatial_factor, angular_factor = runtime_factors(source, spatial, angular)
    raw = (spatial_factor * angular_factor) @ weight.T + bias
    bounded = np.clip(raw, 0, 1)
    coverage = 1 / (1 + np.exp(-np.clip(raw[:, 0], -20, 20)))
    normal_oct = bounded[:, 16:22].reshape(-1, STRATA, 2)
    return {
        "raw": raw,
        "coverage": coverage,
        "depth": decode_distance(bounded[:, 1:4], tile, horizon),
        "sigma": decode_distance(bounded[:, 4:7], tile, horizon),
        "colour": bounded[:, 7:16].reshape(-1, STRATA, 3),
        "normal": oct_decode_numpy(normal_oct),
    }


def truth_arrays(source: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "coverage": np.asarray(source[:, 6]),
        "depth": np.asarray(source[:, [7, 15, 23]]),
        "sigma": np.sqrt(np.maximum(np.asarray(source[:, [8, 16, 24]]), 0)),
        "colour": np.stack([source[:, 9:12], source[:, 17:20], source[:, 25:28]], axis=1),
        "normal": np.stack([source[:, 12:15], source[:, 20:23], source[:, 28:31]], axis=1),
    }


def quantiles(value: np.ndarray) -> dict[str, float]:
    if value.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.quantile(value, 0.50)),
        "p95": float(np.quantile(value, 0.95)),
        "p99": float(np.quantile(value, 0.99)),
        "maximum": float(np.max(value)),
    }


def evaluate(source: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    truth = truth_arrays(source)
    active = truth["coverage"] > 0
    coverage_error = np.abs(prediction["coverage"] - truth["coverage"])
    truth_hit = truth["coverage"] >= 0.5
    predicted_hit = prediction["coverage"] >= 0.5
    intersection = int(np.count_nonzero(truth_hit & predicted_hit))
    union = int(np.count_nonzero(truth_hit | predicted_hit))
    depth_error = np.abs(prediction["depth"][active] - truth["depth"][active]).reshape(-1)
    sigma_error = np.abs(prediction["sigma"][active] - truth["sigma"][active]).reshape(-1)
    colour_error = prediction["colour"][active] - truth["colour"][active]
    dot = np.sum(prediction["normal"][active] * truth["normal"][active], axis=2)
    normal_error = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1))).reshape(-1)
    intrinsic_sigma = truth["sigma"][active].reshape(-1)
    return {
        "records": len(source),
        "truthCoverageMean": float(np.mean(truth["coverage"])),
        "predictedCoverageMean": float(np.mean(prediction["coverage"])),
        "coverageAbsoluteError": quantiles(coverage_error),
        "coverageRmse": float(np.sqrt(np.mean(coverage_error ** 2))),
        "coverageThresholdIoU": float(intersection / union) if union else 1.0,
        "conditionalDepthAbsoluteErrorMetres": quantiles(depth_error),
        "conditionalSigmaAbsoluteErrorMetres": quantiles(sigma_error),
        "intrinsicWithinStratumSigmaMetres": quantiles(intrinsic_sigma),
        "conditionalColourRmse": float(np.sqrt(np.mean(colour_error ** 2))) if colour_error.size else 0.0,
        "conditionalNormalAngularErrorDegrees": quantiles(normal_error),
    }


def grouped(source: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    elevation = np.round(np.degrees(np.arcsin(np.clip(np.abs(source[:, 4]), 0, 1))), 4)
    result: dict[str, Any] = {}
    for value in sorted(set(float(item) for item in elevation)):
        selected = np.isclose(elevation, value, atol=5e-4)
        result[str(value)] = evaluate(source[selected], {key: item[selected] for key, item in prediction.items()})
    return result


def coherence_metrics(
    source: np.ndarray,
    prediction: dict[str, np.ndarray],
    shift: float,
) -> dict[str, Any]:
    if len(source) % 2:
        raise ValueError("coherence records must be consecutive pairs")
    truth = truth_arrays(source)
    first = np.arange(0, len(source), 2)
    second = first + 1
    truth_coverage_delta = truth["coverage"][second] - truth["coverage"][first]
    predicted_coverage_delta = prediction["coverage"][second] - prediction["coverage"][first]
    coverage_delta_error = np.abs(predicted_coverage_delta - truth_coverage_delta)
    noncrossing = (
        (truth["coverage"][first] > 0)
        & (truth["coverage"][second] > 0)
        & np.all(truth["depth"][first] > shift, axis=1)
    )
    truth_residual = truth["depth"][second] - (truth["depth"][first] - shift)
    predicted_residual = prediction["depth"][second] - (prediction["depth"][first] - shift)
    residual_error = np.abs(predicted_residual - truth_residual)[noncrossing].reshape(-1)
    return {
        "pairs": len(first),
        "noncrossingPairs": int(np.count_nonzero(noncrossing)),
        "truthCoverageDelta": quantiles(np.abs(truth_coverage_delta)),
        "predictedCoverageDelta": quantiles(np.abs(predicted_coverage_delta)),
        "coverageDeltaAbsoluteError": quantiles(coverage_delta_error),
        "truthRightCensoringDepthResidualMetres": quantiles(np.abs(truth_residual[noncrossing]).reshape(-1)),
        "predictedRightCensoringDepthResidualMetres": quantiles(np.abs(predicted_residual[noncrossing]).reshape(-1)),
        "rightCensoringResidualAbsoluteErrorMetres": quantiles(residual_error),
    }


def downsample_volume(volume: np.ndarray) -> np.ndarray:
    ny, nz, nx, rank = volume.shape
    next_y, next_z, next_x = max(1, ny // 2), max(1, nz // 2), max(1, nx // 2)
    result = np.zeros((next_y, next_z, next_x, rank), dtype=np.float32)
    counts = np.zeros((next_y, next_z, next_x, 1), dtype=np.float32)
    for y in range(ny):
        for z in range(nz):
            for x in range(nx):
                result[min(next_y - 1, y // 2), min(next_z - 1, z // 2), min(next_x - 1, x // 2)] += volume[y, z, x]
                counts[min(next_y - 1, y // 2), min(next_z - 1, z // 2), min(next_x - 1, x // 2)] += 1
    return result / counts


def write_mips(weight: np.ndarray, dims: tuple[int, int, int], output: Path) -> list[dict[str, Any]]:
    nx, ny, nz = dims
    level = weight.reshape(ny, nz, nx, -1).astype(np.float32)
    offset = 0
    levels: list[dict[str, Any]] = []
    with output.open("wb") as target:
        while True:
            packed = level.astype("<f2")
            target.write(packed.tobytes())
            levels.append({
                "level": len(levels), "dimensionsXYZ": [level.shape[2], level.shape[0], level.shape[1]],
                "offsetBytes": offset, "bytes": packed.nbytes,
            })
            offset += packed.nbytes
            if level.shape[0] == level.shape[1] == level.shape[2] == 1:
                break
            level = downsample_volume(level)
    return levels


def rgb(array: np.ndarray) -> np.ndarray:
    return np.clip(np.power(np.clip(array, 0, 1), 1 / 2.2) * 255, 0, 255).astype(np.uint8)


def depth_rgb(depth: np.ndarray, coverage: np.ndarray, horizon: float) -> np.ndarray:
    value = np.log1p(np.minimum(depth, horizon)) / math.log1p(horizon)
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    result[..., 0] = np.where(coverage > 0, 255 * value, 8).astype(np.uint8)
    result[..., 1] = np.where(coverage > 0, 255 * np.sqrt(value), 8).astype(np.uint8)
    result[..., 2] = np.where(coverage > 0, 255 * (1 - value), 8).astype(np.uint8)
    return result


def panel(array: np.ndarray, label: str, scale: int = 4) -> Image.Image:
    image = Image.fromarray(array).resize((array.shape[1] * scale, array.shape[0] * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (image.width, image.height + 26), (24, 24, 24))
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((6, 6), label, fill=(240, 240, 240))
    return canvas


def write_qa(
    source: np.ndarray,
    prediction: dict[str, np.ndarray],
    qa_manifest: dict[str, Any],
    horizon: float,
    output: Path,
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    grid = int(qa_manifest["grid"])
    count = grid * grid
    truth = truth_arrays(source)
    entries = []
    for index, definition in enumerate(qa_manifest["slices"]):
        select = slice(index * count, (index + 1) * count)
        tc = truth["coverage"][select].reshape(grid, grid)
        pc = prediction["coverage"][select].reshape(grid, grid)
        truth_colour = truth["colour"][select].mean(axis=1).reshape(grid, grid, 3)
        predicted_colour = prediction["colour"][select].mean(axis=1).reshape(grid, grid, 3)
        truth_depth = truth["depth"][select, 0].reshape(grid, grid)
        predicted_depth = prediction["depth"][select, 0].reshape(grid, grid)
        coverage_error = np.abs(pc - tc)
        error_rgb = np.zeros((grid, grid, 3), dtype=np.uint8)
        error_rgb[..., 0] = np.clip(coverage_error * 255, 0, 255).astype(np.uint8)
        panels = [
            panel(np.repeat((tc[..., None] * 255).astype(np.uint8), 3, axis=2), "truth beam coverage"),
            panel(np.repeat((pc[..., None] * 255).astype(np.uint8), 3, axis=2), "rank prediction coverage"),
            panel(error_rgb, "coverage absolute error"),
            panel(rgb(truth_colour), "truth conditional colour"),
            panel(rgb(predicted_colour), "predicted conditional colour"),
            panel(depth_rgb(truth_depth, tc, horizon), "truth front-stratum depth"),
            panel(depth_rgb(predicted_depth, pc, horizon), "predicted front-stratum depth"),
        ]
        contact = Image.new("RGB", (sum(item.width for item in panels), max(item.height for item in panels)))
        x = 0
        for item in panels:
            contact.paste(item, (x, 0))
            x += item.width
        path = output / f"{definition['id']}.png"
        contact.save(path)
        subset_prediction = {key: value[select] for key, value in prediction.items()}
        entries.append({
            "path": path.name, "sha256": sha256(path), "definition": definition,
            "metrics": evaluate(source[select], subset_prediction),
            "interpretation": "explicit 16-micro-ray beam truth versus packed fixed-rank transfer decoder; depths are conditional stratum representatives, not exact centre-ray owners",
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", default="192,64,192")
    parser.add_argument("--angular", type=int, default=64)
    parser.add_argument("--rank", type=int, choices=(4, 8), default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--spatial-lr", type=float, default=0.03)
    parser.add_argument("--dense-lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2407224)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    for label in ("train", "validation", "qa", "coherence"):
        path = root / manifest["files"][label]["path"]
        if sha256(path) != manifest["files"][label]["sha256"]:
            raise ValueError(f"{label} hash mismatch")
    train = records(root / manifest["files"]["train"]["path"], manifest["files"]["train"]["records"])
    validation = records(root / manifest["files"]["validation"]["path"], manifest["files"]["validation"]["records"])
    qa = records(root / manifest["files"]["qa"]["path"], manifest["files"]["qa"]["records"])
    coherence = records(
        root / manifest["files"]["coherence"]["path"], manifest["files"]["coherence"]["records"]
    )
    nx, ny, nz = (int(value) for value in args.grid.split(","))
    if min(nx, ny, nz, args.angular, args.rank, args.epochs, args.batch_size) <= 0:
        raise ValueError("all dimensions and counts must be positive")
    reads = math.ceil(args.rank / 4) * 2
    fmas = args.rank * OUTPUTS
    base_spatial_bytes = nx * ny * nz * args.rank * 2
    projected_mip_bytes = sum(
        max(1, nx // (2 ** level)) * max(1, ny // (2 ** level)) * max(1, nz // (2 ** level)) * args.rank * 2
        for level in range(math.ceil(math.log2(max(nx, ny, nz))) + 1)
    )
    angular_bytes = args.angular * args.angular * args.rank * 2
    decoder_bytes = (args.rank * OUTPUTS + OUTPUTS) * 2
    resident = projected_mip_bytes + angular_bytes + decoder_bytes
    if reads > MAX_READS or fmas > MAX_FMAS or resident > MAX_BYTES:
        raise ValueError(f"runtime envelope exceeded: {reads} reads, {fmas} FMAs, {resident} bytes")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[transfer-codec] device={device} rank={args.rank} grid={nx}x{ny}x{nz} angular={args.angular}")
    model = TransferFieldCodec(args.rank, device)
    optimizer = torch.optim.Adam([
        {"params": model.spatial.parameters(), "lr": args.spatial_lr},
        {"params": list(model.directional.parameters()) + list(model.output.parameters()), "lr": args.dense_lr},
    ])
    tile = float(manifest["source"]["tileSize"][0])
    horizon = float(manifest["configuration"]["horizon"])
    with torch.no_grad():
        coverage_mean = float(np.mean(train[:, 6]))
        model.output.bias[0] = math.log(max(1e-5, coverage_mean) / max(1e-5, 1 - coverage_mean))
        active = train[:, 6] > 0
        depth = np.asarray(train[active][:, [7, 15, 23]])
        sigma = np.sqrt(np.maximum(np.asarray(train[active][:, [8, 16, 24]]), 0))
        model.output.bias[1:4] = torch.tensor(np.median(distance_u_numpy(depth, tile, horizon), axis=0), device=device)
        model.output.bias[4:7] = torch.tensor(np.median(distance_u_numpy(sigma, tile, horizon), axis=0), device=device)
        colour = np.stack([train[active, 9:12], train[active, 17:20], train[active, 25:28]], axis=1)
        normal = np.stack([train[active, 12:15], train[active, 20:23], train[active, 28:31]], axis=1)
        model.output.bias[7:16] = torch.tensor(np.median(colour, axis=0).reshape(-1), device=device)
        model.output.bias[16:22] = torch.tensor(np.median(oct_encode_numpy(normal), axis=0).reshape(-1), device=device)

    curve = []
    all_indices = torch.arange(len(train), dtype=torch.int64)
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled = all_indices[torch.randperm(len(train), generator=generator)]
        totals = {"loss": 0.0, "coverage": 0.0, "depth": 0.0, "sigma": 0.0, "colour": 0.0, "normal": 0.0, "records": 0}
        model.train()
        for indices in batches(shuffled, args.batch_size):
            batch = selected_tensor(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(batch)
            coverage, depth_u, sigma_u, colour, _, normal = transformed_output(raw)
            tc, td, ts, tcolour, tnormal = truth_tensors(batch, tile, horizon)
            weight = tc.unsqueeze(1)
            weight_sum = weight.sum().clamp_min(1.0)
            coverage_loss = torch.nn.functional.binary_cross_entropy(coverage, tc)
            depth_loss = (torch.nn.functional.smooth_l1_loss(depth_u, td, reduction="none", beta=0.01) * weight).sum() / (weight_sum * STRATA)
            sigma_loss = (torch.nn.functional.smooth_l1_loss(sigma_u, ts, reduction="none", beta=0.01) * weight).sum() / (weight_sum * STRATA)
            colour_loss = (torch.nn.functional.smooth_l1_loss(colour, tcolour, reduction="none", beta=0.02) * weight.unsqueeze(2)).sum() / (weight_sum * STRATA * 3)
            normal_loss = ((1 - torch.abs((normal * tnormal).sum(dim=2))) * weight).sum() / (weight_sum * STRATA)
            loss = coverage_loss + 4 * depth_loss + sigma_loss + 2 * colour_loss + normal_loss
            loss.backward()
            optimizer.step()
            count = len(indices)
            for key, value in (("loss", loss), ("coverage", coverage_loss), ("depth", depth_loss), ("sigma", sigma_loss), ("colour", colour_loss), ("normal", normal_loss)):
                totals[key] += float(value.detach().cpu()) * count
            totals["records"] += count
        entry = {"epoch": epoch + 1, **{key: totals[key] / totals["records"] for key in totals if key != "records"}}
        curve.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"[transfer-codec] epoch {epoch + 1:03d}: loss={entry['loss']:.6f}")

    float_prediction = predict_continuous(model, validation, args.batch_size, device, tile, horizon)
    float_metrics = evaluate(validation, float_prediction)
    print("[transfer-codec] sampling continuous factors into runtime textures")
    spatial_float, angular_float = sample_factor_tables(
        model, (nx, ny, nz), args.angular, args.batch_size, device
    )
    spatial_packed = spatial_float.astype("<f2").astype(np.float32)
    angular_packed = angular_float.astype("<f2").astype(np.float32)
    weight_packed = model.output.weight.detach().cpu().numpy().astype("<f2").astype(np.float32)
    bias_packed = model.output.bias.detach().cpu().numpy().astype("<f2").astype(np.float32)
    packed_prediction = runtime_predict(
        validation, spatial_packed, angular_packed, weight_packed, bias_packed, tile, horizon
    )
    packed_metrics = evaluate(validation, packed_prediction)
    qa_prediction = runtime_predict(
        qa, spatial_packed, angular_packed, weight_packed, bias_packed, tile, horizon
    )
    coherence_prediction = runtime_predict(
        coherence, spatial_packed, angular_packed, weight_packed, bias_packed, tile, horizon
    )

    args.output.mkdir(parents=True, exist_ok=True)
    spatial_path = args.output / "spatial-coefficients-mips-rgba16f.bin"
    mip_levels = write_mips(spatial_float.reshape(-1, args.rank), (nx, ny, nz), spatial_path)
    angular_path = args.output / "angular-coefficients-rgba16f.bin"
    angular_path.write_bytes(angular_float.astype("<f2").tobytes())
    decoder_path = args.output / "decoder-f16.npz"
    np.savez(
        decoder_path,
        weight=weight_packed.astype("<f2"),
        bias=bias_packed.astype("<f2"),
    )
    qa_entries = write_qa(qa, qa_prediction, manifest["files"]["qa"], horizon, args.output / "qa")

    intrinsic = packed_metrics["intrinsicWithinStratumSigmaMetres"]
    representation_pass = intrinsic["p95"] <= 0.025 and intrinsic["p99"] <= 0.10
    codec_pass = (
        packed_metrics["coverageAbsoluteError"]["p95"] <= 0.10
        and packed_metrics["coverageThresholdIoU"] >= 0.90
        and packed_metrics["conditionalDepthAbsoluteErrorMetres"]["p95"] <= 0.05
        and packed_metrics["conditionalColourRmse"] <= 0.06
        and packed_metrics["conditionalNormalAngularErrorDegrees"]["p95"] <= 20
    )
    decision = "accept" if representation_pass and codec_pass else "reject"
    rank8_can_materially_change = (not codec_pass) and representation_pass and args.rank == 4
    report = {
        "schema": "laas-groundcover-transfer-field-codec-evaluation/v1",
        "truth": {
            "manifest": str(manifest_path), "manifestSha256": sha256(manifest_path),
            "sourceSha256": manifest["source"]["sha256"], "contract": manifest["truthContract"],
        },
        "command": ["uv", "run", "--project", "asset-gen", "--extra", "microtopography-ml", "python", *sys.argv],
        "model": {
            "factorization": "z_o(x,d)=b_o+sum_r C_or U_r(x)V_r(d)",
            "rank": args.rank,
            "spatial": {"dimensionsXYZ": [nx, ny, nz], "mips": mip_levels, "format": "RGBA16F groups"},
            "angular": {"dimensions": [args.angular, args.angular], "mapping": "full-sphere octahedral", "format": "RGBA16F groups"},
            "outputs": ["coverage", "3 conditional depths", "3 conditional sigmas", "3 RGB colours", "3 oct normals", "roughness", "transmission"],
            "runtimeEnvelope": {
                "logicalFilteredReads": reads, "scalarFmaEquivalent": fmas,
                "residentBytesProjected": resident, "residentBytesActualArtifacts": spatial_path.stat().st_size + angular_path.stat().st_size + decoder_path.stat().st_size,
                "maximumResidentBytes": MAX_BYTES, "maximumReads": MAX_READS, "maximumFmas": MAX_FMAS,
                "passes": 0, "runtimeLoopsMarchesTraversalsCandidatesSpeciesQueries": 0,
            },
        },
        "training": {
            "seed": args.seed, "epochs": args.epochs, "batchSize": args.batch_size,
            "spatialLearningRate": args.spatial_lr, "denseLearningRate": args.dense_lr, "curve": curve,
        },
        "validation": {
            "float32BeforePacking": float_metrics,
            "packedFloat16": packed_metrics,
            "packedFloat16ByHeldOutElevationMagnitudeDegrees": grouped(validation, packed_prediction),
            "heldOutOriginShiftCoherence": coherence_metrics(
                coherence, coherence_prediction, float(manifest["files"]["coherence"]["shiftMetres"])
            ),
            "exteriorTopEntryDatumShiftInvariant": manifest["invariants"],
        },
        "gate": {
            "decision": decision,
            "representationPass": representation_pass,
            "codecPass": codec_pass,
            "rank8CanMateriallyChangeFailure": rank8_can_materially_change,
            "thresholds": {
                "intrinsicSigmaP95Metres": 0.025, "intrinsicSigmaP99Metres": 0.10,
                "coverageErrorP95": 0.10, "coverageIoU": 0.90,
                "codecDepthErrorP95Metres": 0.05, "colourRmse": 0.06, "normalP95Degrees": 20,
            },
            "interpretation": (
                "rank cannot repair intrinsic within-stratum spread; do not run rank eight"
                if not representation_pass else
                "rank eight is permitted only when the rank-four decoder, rather than the three-stratum target, is the limiting error"
            ),
        },
        "artifacts": {
            "spatialCoefficients": {"path": spatial_path.name, "bytes": spatial_path.stat().st_size, "sha256": sha256(spatial_path), "levels": mip_levels},
            "angularCoefficients": {"path": angular_path.name, "bytes": angular_path.stat().st_size, "sha256": sha256(angular_path)},
            "decoder": {"path": decoder_path.name, "bytes": decoder_path.stat().st_size, "sha256": sha256(decoder_path)},
            "qa": qa_entries,
        },
        "contract": [
            "training consumes explicit footprint-filtered coverage and three conditional strata, never the discontinuous single centre-ray event",
            "all validation elevations are absent from training; phase, azimuth, height, and beam rotations are disjoint",
            "the decoded event is a stochastic representative of a filtered distribution, not an exact source-triangle hit",
            "no runtime shader or asset is emitted by this offline gate",
        ],
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[transfer-codec] wrote {report_path}")
    print(json.dumps({"gate": report["gate"], "packedValidation": packed_metrics, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

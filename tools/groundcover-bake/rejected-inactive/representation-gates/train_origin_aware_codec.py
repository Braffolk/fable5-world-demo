#!/usr/bin/env python3
"""Train/evaluate a fixed-read origin-aware first-successor approximation.

The model is deliberately shader-shaped: one periodic/clamped trilinear sample
from an RGBA16F 3D feature texture and one fixed two-layer ReLU decoder. It is
an offline representation experiment, not runtime shader code and not a claim
of exact geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


FIELDS = 8
MAX_RESIDENT_BYTES = 51_121_152
MAX_FMAS = 256
MAX_READS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path, count: int) -> np.memmap:
    values = np.memmap(path, dtype="<f4", mode="r")
    if values.size != count * FIELDS:
        raise ValueError(f"{path} has {values.size} floats, expected {count * FIELDS}")
    return values.reshape(count, FIELDS)


class OriginAwareCodec(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int, features: int, hidden: int, device: torch.device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.features = features
        self.hidden = hidden
        self.volume = torch.nn.Embedding(nx * ny * nz, features, sparse=True, device=device)
        torch.nn.init.normal_(self.volume.weight, mean=0.0, std=0.01)
        self.input = torch.nn.Linear(features + 3, hidden, device=device)
        self.output = torch.nn.Linear(hidden, 2, device=device)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.output.bias)

    def spatial_features(self, xyz: torch.Tensor) -> torch.Tensor:
        # Match normalized hardware sampling: x/z repeat and Y clamps.
        gx = xyz[:, 0] * self.nx - 0.5
        gy = xyz[:, 1] * self.ny - 0.5
        gz = xyz[:, 2] * self.nz - 0.5
        x0f = torch.floor(gx)
        y0f = torch.floor(gy)
        z0f = torch.floor(gz)
        fx = gx - x0f
        fy = gy - y0f
        fz = gz - z0f
        x0 = x0f.to(torch.int64).remainder(self.nx)
        x1 = (x0 + 1).remainder(self.nx)
        z0 = z0f.to(torch.int64).remainder(self.nz)
        z1 = (z0 + 1).remainder(self.nz)
        y0 = y0f.to(torch.int64).clamp(0, self.ny - 1)
        y1 = (y0 + 1).clamp(0, self.ny - 1)
        result = torch.zeros((xyz.shape[0], self.features), device=xyz.device, dtype=torch.float32)
        for iy, wy in ((y0, 1.0 - fy), (y1, fy)):
            for iz, wz in ((z0, 1.0 - fz), (z1, fz)):
                for ix, wx in ((x0, 1.0 - fx), (x1, fx)):
                    index = (iy * self.nz + iz) * self.nx + ix
                    result = result + self.volume(index) * (wx * wy * wz).unsqueeze(1)
        return result

    def forward(self, records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.spatial_features(records[:, :3])
        hidden = torch.relu(self.input(torch.cat((features, records[:, 3:6]), dim=1)))
        output = self.output(hidden)
        # Runtime uses saturate. The straight-through derivative avoids the
        # dead-gradient failure of literally differentiating that clamp.
        raw_distance = output[:, 1]
        saturated_distance = raw_distance + (raw_distance.clamp(0.0, 1.0) - raw_distance).detach()
        return output[:, 0], saturated_distance


def batches(indices: torch.Tensor, batch_size: int):
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def records_tensor(records: np.ndarray, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = np.asarray(records[np.asarray(indices.cpu())], dtype=np.float32)
    return torch.from_numpy(selected.copy()).to(device)


def distance_u(distance: torch.Tensor, tile_size: float, horizon: float) -> torch.Tensor:
    scale = math.log2(1.0 + horizon / tile_size)
    return torch.log2(1.0 + distance / tile_size) / scale


def decode_distance(value: torch.Tensor, tile_size: float, horizon: float) -> torch.Tensor:
    scale = math.log2(1.0 + horizon / tile_size)
    return tile_size * (torch.exp2(value.clamp(0.0, 1.0) * scale) - 1.0)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


@torch.no_grad()
def predict(
    model: OriginAwareCodec,
    records: np.ndarray,
    batch_size: int,
    device: torch.device,
    tile_size: float,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    all_indices = torch.arange(len(records), dtype=torch.int64)
    model.eval()
    for index in batches(all_indices, batch_size):
        batch = records_tensor(records, index, device)
        logit, encoded_distance = model(batch)
        logits.append(logit.cpu().numpy())
        distances.append(decode_distance(encoded_distance, tile_size, horizon).cpu().numpy())
    return np.concatenate(logits), np.concatenate(distances)


def metrics(records: np.ndarray, logits: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(records[:, 6] >= 0.5)
    predicted = logits >= 0.0
    true_positive = truth & predicted
    false_positive = ~truth & predicted
    false_negative = truth & ~predicted
    true_negative = ~truth & ~predicted
    tp = int(np.count_nonzero(true_positive))
    fp = int(np.count_nonzero(false_positive))
    fn = int(np.count_nonzero(false_negative))
    tn = int(np.count_nonzero(true_negative))
    truth_distance = np.asarray(records[:, 7], dtype=np.float64)
    hit_error = np.abs(distances[truth] - truth_distance[truth])
    true_positive_error = np.abs(distances[true_positive] - truth_distance[true_positive])
    relative_hit_error = hit_error / np.maximum(truth_distance[truth], 1e-4)
    return {
        "records": len(records),
        "truthHitFraction": float(np.mean(truth)),
        "predictedHitFraction": float(np.mean(predicted)),
        "hitAgreement": float((tp + tn) / len(records)),
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "intersectionOverUnion": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "trueNegative": tn,
        "truthHitDistanceAbsoluteErrorMetres": quantiles(hit_error),
        "truePositiveDistanceAbsoluteErrorMetres": quantiles(true_positive_error),
        "truthHitDistanceRelativeError": quantiles(relative_hit_error),
    }


def grouped_metrics(records: np.ndarray, logits: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
    elevation = np.round(np.degrees(np.arcsin(np.clip(np.abs(records[:, 4]), 0.0, 1.0))), 4)
    result: dict[str, Any] = {}
    for value in sorted(set(float(item) for item in elevation)):
        selected = np.isclose(elevation, value, atol=5e-4)
        result[str(value)] = metrics(records[selected], logits[selected], distances[selected])
    return result


def colour_depth(distance: np.ndarray, hit: np.ndarray, horizon: float) -> np.ndarray:
    normalized = np.log1p(np.minimum(distance, horizon)) / math.log1p(horizon)
    result = np.zeros((*distance.shape, 3), dtype=np.uint8)
    result[..., 0] = np.where(hit, np.clip(255 * normalized, 0, 255), 8).astype(np.uint8)
    result[..., 1] = np.where(hit, np.clip(255 * np.sqrt(normalized), 0, 255), 8).astype(np.uint8)
    result[..., 2] = np.where(hit, np.clip(255 * (1.0 - normalized), 0, 255), 8).astype(np.uint8)
    return result


def labelled_panel(array: np.ndarray, label: str, scale: int = 4) -> Image.Image:
    image = Image.fromarray(array).resize((array.shape[1] * scale, array.shape[0] * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (image.width, image.height + 26), (24, 24, 24))
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((6, 6), label, fill=(240, 240, 240))
    return canvas


def write_qa(
    qa_records: np.ndarray,
    qa_manifest: dict[str, Any],
    logits: np.ndarray,
    distances: np.ndarray,
    horizon: float,
    output: Path,
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    grid = int(qa_manifest["grid"])
    per_slice = grid * grid
    entries: list[dict[str, Any]] = []
    for index, definition in enumerate(qa_manifest["slices"]):
        start = index * per_slice
        end = start + per_slice
        records = qa_records[start:end]
        truth_hit = records[:, 6].reshape(grid, grid) >= 0.5
        predicted_hit = logits[start:end].reshape(grid, grid) >= 0.0
        probability = (1.0 / (1.0 + np.exp(-np.clip(logits[start:end], -16, 16)))).reshape(grid, grid)
        truth_distance = records[:, 7].reshape(grid, grid)
        predicted_distance = distances[start:end].reshape(grid, grid)
        truth_coverage = np.repeat((truth_hit[..., None] * 255).astype(np.uint8), 3, axis=2)
        predicted_coverage = np.zeros((grid, grid, 3), dtype=np.uint8)
        predicted_coverage[..., 1] = np.clip(probability * 255, 0, 255).astype(np.uint8)
        classification = np.zeros((grid, grid, 3), dtype=np.uint8)
        classification[truth_hit & ~predicted_hit] = (255, 0, 255)
        classification[~truth_hit & predicted_hit] = (0, 255, 255)
        classification[truth_hit & predicted_hit] = (40, 120, 40)
        panels = [
            labelled_panel(truth_coverage, "actual-mesh hit"),
            labelled_panel(predicted_coverage, "predicted hit probability"),
            labelled_panel(colour_depth(truth_distance, truth_hit, horizon), "actual-mesh first distance"),
            labelled_panel(colour_depth(predicted_distance, predicted_hit, horizon), "predicted first distance"),
            labelled_panel(classification, "errors: magenta miss / cyan false hit"),
        ]
        contact = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        x = 0
        for panel in panels:
            contact.paste(panel, (x, 0))
            x += panel.width
        path = output / f"{definition['id']}.png"
        contact.save(path)
        entries.append({
            "path": path.name,
            "sha256": sha256(path),
            "definition": definition,
            "metrics": metrics(records, logits[start:end], distances[start:end]),
            "interpretation": "band-limited learned coverage/distance; not categorical owner-exact geometry",
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", default="64,32,64", help="X,Y,Z feature texels")
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--feature-lr", type=float, default=0.02)
    parser.add_argument("--decoder-lr", type=float, default=0.003)
    parser.add_argument("--depth-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=240722)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_dir = manifest_path.parent
    train_path = truth_dir / manifest["files"]["train"]["path"]
    validation_path = truth_dir / manifest["files"]["validation"]["path"]
    qa_path = truth_dir / manifest["files"]["qa"]["path"]
    for label, path in (("train", train_path), ("validation", validation_path), ("qa", qa_path)):
        expected = manifest["files"][label]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} truth hash mismatch: {actual} != {expected}")
    train = read_records(train_path, int(manifest["files"]["train"]["records"]))
    validation = read_records(validation_path, int(manifest["files"]["validation"]["records"]))
    qa = read_records(qa_path, int(manifest["files"]["qa"]["records"]))
    nx, ny, nz = (int(value) for value in args.grid.split(","))
    if min(nx, ny, nz, args.features, args.hidden, args.epochs, args.batch_size) <= 0:
        raise ValueError("grid, feature, hidden, epoch, and batch dimensions must be positive")
    reads = math.ceil(args.features / 4)
    fmas = (args.features + 3) * args.hidden + args.hidden * 2
    decoder_scalars = (args.features + 3) * args.hidden + args.hidden + args.hidden * 2 + 2
    resident_bytes = nx * ny * nz * args.features * 2 + decoder_scalars * 2
    if reads > MAX_READS or fmas > MAX_FMAS or resident_bytes > MAX_RESIDENT_BYTES:
        raise ValueError(
            f"runtime envelope exceeded: {reads} reads, {fmas} FMAs, {resident_bytes} bytes"
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[origin-codec] device={device} grid={nx}x{ny}x{nz} C={args.features} hidden={args.hidden}")
    model = OriginAwareCodec(nx, ny, nz, args.features, args.hidden, device)
    feature_optimizer = torch.optim.SparseAdam(model.volume.parameters(), lr=args.feature_lr)
    decoder_parameters = list(model.input.parameters()) + list(model.output.parameters())
    decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=args.decoder_lr)
    tile_size = float(manifest["source"].get("tileSize", [0.52, 0.52])[0])
    horizon = float(manifest["configuration"]["horizon"])
    hit_distances = np.asarray(train[:, 7][train[:, 6] >= 0.5], dtype=np.float32)
    median_u = math.log2(1.0 + float(np.median(hit_distances)) / tile_size) / math.log2(1.0 + horizon / tile_size)
    with torch.no_grad():
        model.output.bias[1] = median_u
    positive = float(np.count_nonzero(train[:, 6] >= 0.5))
    negative = len(train) - positive
    positive_weight = torch.tensor([negative / positive], device=device)
    curve: list[dict[str, float]] = []
    all_indices = torch.arange(len(train), dtype=torch.int64)
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled = all_indices[torch.randperm(len(train), generator=generator)]
        totals = {"loss": 0.0, "coverage": 0.0, "depth": 0.0, "records": 0}
        model.train()
        for index in batches(shuffled, args.batch_size):
            batch = records_tensor(train, index, device)
            truth_hit = batch[:, 6]
            truth_u = distance_u(batch[:, 7], tile_size, horizon)
            feature_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)
            logit, predicted_u = model(batch)
            coverage_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, truth_hit, pos_weight=positive_weight
            )
            hit_count = torch.clamp(truth_hit.sum(), min=1.0)
            depth_loss = (torch.nn.functional.smooth_l1_loss(
                predicted_u, truth_u, reduction="none", beta=0.01
            ) * truth_hit).sum() / hit_count
            loss = coverage_loss + args.depth_weight * depth_loss
            loss.backward()
            feature_optimizer.step()
            decoder_optimizer.step()
            count = len(index)
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["coverage"] += float(coverage_loss.detach().cpu()) * count
            totals["depth"] += float(depth_loss.detach().cpu()) * count
            totals["records"] += count
        entry = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / totals["records"],
            "coverageLoss": totals["coverage"] / totals["records"],
            "depthLoss": totals["depth"] / totals["records"],
        }
        curve.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"[origin-codec] epoch {epoch + 1:03d}: loss={entry['loss']:.6f}")

    float_logits, float_distances = predict(model, validation, args.batch_size, device, tile_size, horizon)
    float_metrics = metrics(validation, float_logits, float_distances)

    # Runtime qualification uses the actual packed float16 representation.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(parameter.to(torch.float16).to(torch.float32))
    packed_logits, packed_distances = predict(model, validation, args.batch_size, device, tile_size, horizon)
    packed_metrics = metrics(validation, packed_logits, packed_distances)
    qa_logits, qa_distances = predict(model, qa, args.batch_size, device, tile_size, horizon)

    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "feature-volume-rgba16f.bin"
    feature_values = model.volume.weight.detach().cpu().numpy().astype("<f2", copy=False)
    feature_path.write_bytes(feature_values.tobytes())
    weights_path = args.output / "decoder-f16.npz"
    np.savez(
        weights_path,
        input_weight=model.input.weight.detach().cpu().numpy().astype("<f2"),
        input_bias=model.input.bias.detach().cpu().numpy().astype("<f2"),
        output_weight=model.output.weight.detach().cpu().numpy().astype("<f2"),
        output_bias=model.output.bias.detach().cpu().numpy().astype("<f2"),
    )
    qa_entries = write_qa(
        qa,
        manifest["files"]["qa"],
        qa_logits,
        qa_distances,
        horizon,
        args.output / "qa",
    )
    report = {
        "schema": "laas-origin-aware-learned-codec-evaluation/v1",
        "truth": {
            "manifest": str(manifest_path),
            "manifestSha256": sha256(manifest_path),
            "sourceSha256": manifest["source"]["sha256"],
            "contract": manifest["truthContract"],
        },
        "command": ["uv", "run", "--project", "asset-gen", "--extra", "microtopography-ml", "python", *(__import__("sys").argv)],
        "model": {
            "featureVolume": {"dimensionsXYZ": [nx, ny, nz], "channels": args.features, "format": "RGBA16F-compatible"},
            "decoder": {"inputs": args.features + 3, "hiddenReLU": args.hidden, "outputs": ["coverageLogit", "rationalDistanceU"]},
            "distanceTransform": "u=log2(1+t/tileSize)/log2(1+horizon/tileSize), decoded by one fixed exp2 after saturate",
            "runtimeEnvelope": {
                "logicalFeatureTextureReads": reads,
                "scalarFmaEquivalent": fmas,
                "reluMaxOperations": args.hidden,
                "exponentialsBase2": 1,
                "fixedBranchFree": True,
                "residentBytes": resident_bytes,
                "maximumResidentBytes": MAX_RESIDENT_BYTES,
                "passes": 0,
                "loopsMarchesTraversalsCandidates": 0,
            },
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "featureLearningRate": args.feature_lr,
            "decoderLearningRate": args.decoder_lr,
            "depthWeight": args.depth_weight,
            "curve": curve,
        },
        "validation": {
            "float32BeforePacking": float_metrics,
            "packedFloat16": packed_metrics,
            "packedFloat16ByElevationMagnitudeDegrees": grouped_metrics(validation, packed_logits, packed_distances),
        },
        "artifacts": {
            "featureVolume": {"path": feature_path.name, "bytes": feature_path.stat().st_size, "sha256": sha256(feature_path)},
            "decoder": {"path": weights_path.name, "bytes": weights_path.stat().st_size, "sha256": sha256(weights_path)},
            "qa": qa_entries,
        },
        "contract": [
            "the output is a band-limited learned coverage and first-distance approximation, not exact categorical geometry",
            "a mean/regressed distance is never relabelled as a source-mesh event; hit/miss and distance errors are reported separately",
            "training uses actual analytic periodic mesh successor truth; the runtime-shaped decoder performs no source traversal or candidate selection",
            "acceptance requires held-out phase/azimuth/height results, packed-f16 results, the explicit byte/read/FMA budget, and visual QA",
        ],
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[origin-codec] wrote {report_path}")
    print(json.dumps({"packedValidation": packed_metrics, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

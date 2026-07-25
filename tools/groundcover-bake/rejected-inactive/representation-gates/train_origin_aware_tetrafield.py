#!/usr/bin/env python3
"""Gate a four-read origin-aware tetra-factor successor field.

The prospective runtime has four hardware-trilinear RGBA16F reads and one
fixed 19 -> 12 -> 2 ReLU decoder.  Training loops below are offline only; the
shader-shaped query contains no loop, march, traversal, or candidate list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def distance_u(distance: torch.Tensor, tile_size: float, horizon: float) -> torch.Tensor:
    scale = math.log2(1.0 + horizon / tile_size)
    return torch.log2(1.0 + distance / tile_size) / scale


def decode_distance(value: torch.Tensor, tile_size: float, horizon: float) -> torch.Tensor:
    scale = math.log2(1.0 + horizon / tile_size)
    return tile_size * (torch.exp2(value.clamp(0.0, 1.0) * scale) - 1.0)


def batches(indices: torch.Tensor, batch_size: int):
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def records_tensor(records: np.ndarray, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = np.asarray(records[np.asarray(indices.cpu())], dtype=np.float32)
    return torch.from_numpy(selected.copy()).to(device)


class TetraField(torch.nn.Module):
    """Four coupled 3-D factors with the exact prospective runtime shape."""

    def __init__(self, resolution: int, hidden: int, device: torch.device):
        super().__init__()
        self.resolution = resolution
        self.hidden = hidden
        entries = resolution**3
        self.volumes = torch.nn.ModuleList(
            [torch.nn.Embedding(entries, 4, sparse=True, device=device) for _ in range(4)]
        )
        for volume in self.volumes:
            torch.nn.init.normal_(volume.weight, mean=0.0, std=0.01)
        self.input = torch.nn.Linear(19, hidden, device=device)
        self.output = torch.nn.Linear(hidden, 2, device=device)
        torch.nn.init.normal_(self.input.weight, mean=0.0, std=0.03)
        torch.nn.init.zeros_(self.input.bias)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.03)
        torch.nn.init.zeros_(self.output.bias)

    def sample_volume(
        self,
        volume: torch.nn.Embedding,
        coordinates: torch.Tensor,
        repeat: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        """Transcription of one normalized hardware-trilinear texture read."""
        resolution = self.resolution
        grid = coordinates * resolution - 0.5
        base_f = torch.floor(grid)
        fraction = grid - base_f
        base = base_f.to(torch.int64)
        result = torch.zeros((coordinates.shape[0], 4), device=coordinates.device)
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    components = []
                    weight = torch.ones(coordinates.shape[0], device=coordinates.device)
                    for axis, delta in enumerate((dx, dy, dz)):
                        index = base[:, axis] + delta
                        if repeat[axis]:
                            index = index.remainder(resolution)
                        else:
                            index = index.clamp(0, resolution - 1)
                        components.append(index)
                        f = fraction[:, axis]
                        weight = weight * (f if delta else 1.0 - f)
                    linear = (components[2] * resolution + components[1]) * resolution + components[0]
                    result = result + volume(linear) * weight.unsqueeze(1)
        return result

    def forward(self, records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase_x = records[:, 0].remainder(1.0)
        height = records[:, 1].clamp(0.0, 1.0)
        phase_z = records[:, 2].remainder(1.0)
        direction = records[:, 3:6]
        azimuth = torch.atan2(direction[:, 2], direction[:, 0]).div(2 * math.pi).remainder(1.0)
        elevation = direction[:, 1].mul(0.5).add(0.5).clamp(0.0, 1.0)
        # Four triples cover every pair among the five independent pointed-ray
        # coordinates (x, z, y, azimuth, elevation).  Unlike a view-rotated
        # phase, each triple is a well-defined function on T^2 x I x S^2: x/z
        # wrap only in their own lattice axes and azimuth wraps only in its own.
        # This avoids inventing a false map R_theta(T^2) -> T^2, which is not
        # periodic for a general theta and would introduce tile-boundary seams.
        coordinates = (
            (torch.stack((phase_x, phase_z, height), dim=1), (True, True, False)),
            (torch.stack((phase_x, azimuth, elevation), dim=1), (True, True, False)),
            (torch.stack((phase_z, azimuth, elevation), dim=1), (True, True, False)),
            (torch.stack((height, azimuth, elevation), dim=1), (False, True, False)),
        )
        features = torch.cat(
            [self.sample_volume(volume, coord, repeat) for volume, (coord, repeat) in zip(self.volumes, coordinates)],
            dim=1,
        )
        hidden = torch.relu(self.input(torch.cat((features, direction), dim=1)))
        output = self.output(hidden)
        raw_distance = output[:, 1]
        encoded_distance = raw_distance + (
            raw_distance.clamp(0.0, 1.0) - raw_distance
        ).detach()
        return output[:, 0], encoded_distance


def metrics(records: np.ndarray, logits: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(records[:, 6] >= 0.5)
    predicted = logits >= 0.0
    true_positive = truth & predicted
    tp = int(np.count_nonzero(true_positive))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    truth_distance = np.asarray(records[:, 7], dtype=np.float64)
    hit_error = np.abs(distances[truth] - truth_distance[truth])
    true_positive_error = np.abs(distances[true_positive] - truth_distance[true_positive])
    return {
        "records": len(records),
        "hitAgreement": float((tp + tn) / len(records)),
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "intersectionOverUnion": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "truthHitDistanceAbsoluteErrorMetres": quantiles(hit_error),
        "truePositiveDistanceAbsoluteErrorMetres": quantiles(true_positive_error),
    }


@torch.no_grad()
def predict(
    model: TetraField,
    records: np.ndarray,
    batch_size: int,
    device: torch.device,
    tile_size: float,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for indices in batches(torch.arange(len(records), dtype=torch.int64), batch_size):
        batch = records_tensor(records, indices, device)
        logit, encoded = model(batch)
        logits.append(logit.cpu().numpy())
        distances.append(decode_distance(encoded, tile_size, horizon).cpu().numpy())
    return np.concatenate(logits), np.concatenate(distances)


def grouped_metrics(records: np.ndarray, logits: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
    elevation = np.round(np.degrees(np.arcsin(np.clip(np.abs(records[:, 4]), 0.0, 1.0))), 4)
    result: dict[str, Any] = {}
    for value in sorted(set(float(item) for item in elevation)):
        selected = np.isclose(elevation, value, atol=5e-4)
        result[str(value)] = metrics(records[selected], logits[selected], distances[selected])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--feature-lr", type=float, default=0.02)
    parser.add_argument("--decoder-lr", type=float, default=0.003)
    parser.add_argument("--depth-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=250722)
    args = parser.parse_args()

    if args.hidden != 12:
        raise ValueError("the frozen runtime gate requires hidden=12")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_dir = manifest_path.parent
    paths = {
        label: truth_dir / manifest["files"][label]["path"]
        for label in ("train", "validation", "qa")
    }
    for label, path in paths.items():
        actual = sha256(path)
        expected = manifest["files"][label]["sha256"]
        if actual != expected:
            raise ValueError(f"{label} truth hash mismatch: {actual} != {expected}")
    train = read_records(paths["train"], int(manifest["files"]["train"]["records"]))
    validation = read_records(paths["validation"], int(manifest["files"]["validation"]["records"]))
    qa = read_records(paths["qa"], int(manifest["files"]["qa"]["records"]))

    reads = 4
    fmas = 19 * args.hidden + args.hidden * 2
    entries = args.resolution**3
    resident_bytes = 4 * entries * 4 * 2
    decoder_scalars = 19 * args.hidden + args.hidden + args.hidden * 2 + 2
    resident_bytes += decoder_scalars * 2
    if reads > MAX_READS or fmas > MAX_FMAS or resident_bytes > MAX_RESIDENT_BYTES:
        raise ValueError(f"runtime envelope exceeded: {reads} reads, {fmas} FMAs, {resident_bytes} bytes")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TetraField(args.resolution, args.hidden, device)
    feature_optimizer = torch.optim.SparseAdam(model.volumes.parameters(), lr=args.feature_lr)
    decoder_parameters = list(model.input.parameters()) + list(model.output.parameters())
    decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=args.decoder_lr)
    tile_size = float(manifest["source"]["tileSize"][0])
    horizon = float(manifest["configuration"]["horizon"])
    hit_distances = np.asarray(train[:, 7][train[:, 6] >= 0.5], dtype=np.float32)
    median_u = math.log2(1.0 + float(np.median(hit_distances)) / tile_size) / math.log2(
        1.0 + horizon / tile_size
    )
    with torch.no_grad():
        model.output.bias[1] = median_u
    positives = float(np.count_nonzero(train[:, 6] >= 0.5))
    positive_weight = torch.tensor([(len(train) - positives) / positives], device=device)
    all_indices = torch.arange(len(train), dtype=torch.int64)
    curve: list[dict[str, float]] = []
    print(
        f"[tetrafield] device={device} res={args.resolution} reads={reads} "
        f"fmas={fmas} resident={resident_bytes}"
    )
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled = all_indices[torch.randperm(len(train), generator=generator)]
        total_loss = 0.0
        total_records = 0
        model.train()
        for indices in batches(shuffled, args.batch_size):
            batch = records_tensor(train, indices, device)
            truth_hit = batch[:, 6]
            truth_u = distance_u(batch[:, 7], tile_size, horizon)
            feature_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)
            logit, predicted_u = model(batch)
            coverage_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, truth_hit, pos_weight=positive_weight
            )
            hit_count = truth_hit.sum().clamp(min=1.0)
            depth_loss = (
                torch.nn.functional.smooth_l1_loss(
                    predicted_u, truth_u, reduction="none", beta=0.01
                )
                * truth_hit
            ).sum() / hit_count
            loss = coverage_loss + args.depth_weight * depth_loss
            loss.backward()
            feature_optimizer.step()
            decoder_optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            total_records += len(indices)
        entry = {"epoch": epoch + 1, "loss": total_loss / total_records}
        curve.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"[tetrafield] epoch {epoch + 1:03d}: loss={entry['loss']:.6f}")

    validation_logits, validation_distances = predict(
        model, validation, args.batch_size, device, tile_size, horizon
    )
    qa_logits, qa_distances = predict(model, qa, args.batch_size, device, tile_size, horizon)
    report = {
        "schema": "laas-origin-aware-tetrafield-gate/v1",
        "sourceManifest": str(manifest_path),
        "sourceManifestSha256": sha256(manifest_path),
        "runtimeShape": {
            "reads": reads,
            "fmaEquivalent": fmas,
            "residentBytes": resident_bytes,
            "volumeCount": 4,
            "volumeResolution": args.resolution,
            "volumeFormat": "RGBA16F",
            "decoder": "19 -> 12 ReLU -> 2",
            "forbiddenAndAbsent": [
                "runtime loop",
                "ray march",
                "mesh traversal",
                "candidate list",
                "species loop",
                "extra pass",
                "raised terrain shell",
            ],
        },
        "training": {"epochs": args.epochs, "seed": args.seed, "curve": curve},
        "validation": {
            "all": metrics(validation, validation_logits, validation_distances),
            "byElevationDegrees": grouped_metrics(
                validation, validation_logits, validation_distances
            ),
        },
        "qa": {
            "all": metrics(qa, qa_logits, qa_distances),
            "byElevationDegrees": grouped_metrics(qa, qa_logits, qa_distances),
        },
        "scope": (
            "capacity gate for actual-mesh hit and distance only; colour, normal, mark, "
            "coupled camera paths, packing, and runtime integration remain mandatory after a pass"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "sha256": sha256(report_path), **report["validation"]["all"]}, indent=2))


if __name__ == "__main__":
    main()

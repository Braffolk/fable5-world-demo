"""Fit one honest K6-CP12 complementary-pair tensor terminal.

The six signed RGBA pair planes form twelve genuine four-coordinate ranks:
uv*ae, ua*ve, and ue*va, channelwise. A dense linear 12->18 head is the only
output map (216 FMAs); constrained moment decoding follows. No hidden layer,
candidate loop, extra read, or additional runtime storage is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_gbc2_k6_cap import (
    BASE_ASSET,
    CHANNELS,
    COORDINATES,
    HORIZON,
    LAYERS,
    OUTPUTS,
    PAIRS,
    PAIR_NAMES,
    PERIODIC,
    SIZE,
    STABLE_ASSET,
    append_asset,
    canonical,
    decode,
    decode_np,
    digest,
    digest_bytes,
    green,
    metrics,
    records,
)
from train_gbc2_k6_sparse_cap import fit_loss, pair_features

HEAD_BYTES_CP12 = 1024
FEATURES = 12
COMPLEMENTS = ((0, 5), (1, 4), (2, 3))
OUTPUT_NAMES = (
    "front.alpha", "front.m1", "front.m2", "front.r", "front.g", "front.b",
    "front.nx", "front.ny", "front.nz", "back.alpha", "back.m1", "back.m2",
    "back.r", "back.g", "back.b", "back.nx", "back.ny", "back.nz",
)


class CP12(torch.nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.register_buffer("pair_axes", torch.tensor(PAIRS, dtype=torch.long))
        self.register_buffer("identity", torch.eye(LAYERS))
        self.plane = torch.nn.Sequential(
            torch.nn.Linear(34 + LAYERS, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, CHANNELS),
        )
        self.head = torch.nn.Linear(FEATURES, OUTPUTS)
        # Nonzero factors are essential: a zero/zero product has zero gradient.
        torch.nn.init.normal_(self.plane[-1].weight, std=.04)
        torch.nn.init.normal_(self.plane[-1].bias, std=.04)
        torch.nn.init.normal_(self.head.weight, std=.04)
        torch.nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -1
            self.head.bias[9] = -1

    def sampled(self, coordinates: torch.Tensor) -> torch.Tensor:
        pairs = coordinates[:, self.pair_axes]
        features = torch.stack(
            tuple(pair_features(pairs[:, layer], PAIRS[layer]) for layer in range(LAYERS)),
            dim=1,
        )
        identity = self.identity.to(dtype=coordinates.dtype).unsqueeze(0).expand(len(coordinates), -1, -1)
        return torch.tanh(self.plane(torch.cat((features, identity), dim=2)))

    @staticmethod
    def ranks(sampled: torch.Tensor) -> torch.Tensor:
        return torch.cat(tuple(sampled[:, a] * sampled[:, b] for a, b in COMPLEMENTS), dim=1)

    def raw(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.head(self.ranks(self.sampled(coordinates)))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return decode(self.raw(coordinates))


def train_steps(
    model: CP12,
    coordinates: torch.Tensor,
    truth: torch.Tensor,
    steps: int,
    batch: int,
    learning_rate: float,
    seed: int,
) -> float:
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model.train()
    last = 0.
    for step in range(steps):
        index = torch.randint(0, len(coordinates), (batch,), generator=generator).to(coordinates.device)
        loss = fit_loss(model(coordinates[index]), truth[index])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimiser.step()
        last = float(loss.detach().cpu())
        if step == 0 or (step + 1) % 250 == 0:
            print(f"[gbc2-k6-cp12] step {step + 1}/{steps} loss={last:.6f}", flush=True)
    return last


@torch.no_grad()
def sample_tables(model: CP12, device: torch.device, batch_rows: int = 8) -> np.ndarray:
    tables = np.empty((LAYERS, SIZE, SIZE, CHANNELS), dtype=np.float16)
    model.eval()
    for layer, axes in enumerate(PAIRS):
        chunks: list[np.ndarray] = []
        for row_start in range(0, SIZE, batch_rows):
            rows = min(batch_rows, SIZE - row_start)
            s = np.tile((np.arange(SIZE, dtype=np.float32) + .5) / SIZE, rows)
            t = np.repeat((np.arange(row_start, row_start + rows, dtype=np.float32) + .5) / SIZE, SIZE)
            pair = torch.from_numpy(np.stack((s, t), axis=1)).to(device)
            features = pair_features(pair, axes)
            identity = model.identity[layer:layer + 1].to(device=device, dtype=pair.dtype).expand(len(pair), -1)
            chunks.append(torch.tanh(model.plane(torch.cat((features, identity), dim=1))).cpu().numpy())
        tables[layer] = np.concatenate(chunks).reshape(SIZE, SIZE, CHANNELS).astype(np.float16)
        print(f"[gbc2-k6-cp12] sampled layer {layer + 1}/6 {PAIR_NAMES[layer]}", flush=True)
    return tables


def bilinear_samples(tables: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    samples = np.empty((len(coordinates), LAYERS, CHANNELS), dtype=np.float32)
    for layer, (axis_s, axis_t) in enumerate(PAIRS):
        s, t = coordinates[:, axis_s], coordinates[:, axis_t]
        sx, ty = s * SIZE - .5, t * SIZE - .5
        x0, y0 = np.floor(sx).astype(np.int64), np.floor(ty).astype(np.int64)
        fx, fy = (sx - x0).astype(np.float32), (ty - y0).astype(np.float32)

        def indices(base: np.ndarray, axis: int, plus: int) -> np.ndarray:
            value = base + plus
            return np.mod(value, SIZE) if axis in PERIODIC else np.clip(value, 0, SIZE - 1)

        ix0, ix1 = indices(x0, axis_s, 0), indices(x0, axis_s, 1)
        iy0, iy1 = indices(y0, axis_t, 0), indices(y0, axis_t, 1)
        table = tables[layer].astype(np.float32, copy=False)
        a, b = table[iy0, ix0], table[iy0, ix1]
        c, d = table[iy1, ix0], table[iy1, ix1]
        top = a + (b - a) * fx[:, None]
        bottom = c + (d - c) * fx[:, None]
        samples[:, layer] = top + (bottom - top) * fy[:, None]
    return samples


def predict_sampled(model: CP12, tables: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    samples = bilinear_samples(tables, coordinates)
    ranks = np.concatenate(tuple(samples[:, a] * samples[:, b] for a, b in COMPLEMENTS), axis=1)
    weight = model.head.weight.detach().cpu().numpy()
    bias = model.head.bias.detach().cpu().numpy()
    return decode_np(ranks @ weight.T + bias)


@torch.no_grad()
def predict_continuous(model: CP12, coordinates: np.ndarray, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(coordinates), 2048):
        x = torch.from_numpy(np.asarray(coordinates[start:start + 2048], dtype=np.float32)).to(device)
        chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks)


def head_bytes(model: CP12) -> bytes:
    result = bytearray(HEAD_BYTES_CP12)
    result[0:3] = b"K6H"
    struct.pack_into("<IIIIIffIIII", result, 4,
                     2, FEATURES, 0, OUTPUTS, LAYERS, HORIZON, 1e-6,
                     2, 64, 234, 3)
    weights = np.concatenate((
        model.head.weight.detach().cpu().numpy().reshape(-1),
        model.head.bias.detach().cpu().numpy(),
    )).astype("<f4")
    if weights.size != 234:
        raise AssertionError(weights.size)
    result[64:64 + weights.nbytes] = weights.tobytes()
    return bytes(result)


def rank_diagnostics(coordinates: np.ndarray, truth: np.ndarray, bins: int = 6) -> dict[str, Any]:
    """Approximate SVD residuals of a fully binned 4D truth tensor.

    This is post-gate diagnosis only. Empty bins are filled with each output's
    occupied-bin mean and the coverage fraction is reported explicitly.
    """
    indices = np.minimum(bins - 1, np.floor(coordinates * bins).astype(np.int64))
    sums = np.zeros((bins, bins, bins, bins, OUTPUTS), dtype=np.float64)
    counts = np.zeros((bins, bins, bins, bins), dtype=np.int32)
    for index, value in zip(indices, truth):
        key = tuple(index)
        sums[key] += value
        counts[key] += 1
    occupied = counts > 0
    tensor = np.zeros_like(sums)
    tensor[occupied] = sums[occupied] / counts[occupied, None]
    global_mean = np.mean(tensor[occupied], axis=0)
    tensor[~occupied] = global_mean
    partitions = {
        "uv_by_ae": ((0, 1), (2, 3)),
        "ua_by_ve": ((0, 2), (1, 3)),
        "ue_by_va": ((0, 3), (1, 2)),
    }
    report: dict[str, Any] = {
        "binsPerAxis": bins,
        "occupiedFraction": float(np.mean(occupied)),
        "emptyFill": "per-output occupied-bin mean",
        "partitions": {},
    }
    for name, (row_axes, column_axes) in partitions.items():
        per_output: dict[str, Any] = {}
        rank4_values: list[float] = []
        rank12_values: list[float] = []
        permutation = (*row_axes, *column_axes, 4)
        arranged = np.transpose(tensor, permutation).reshape(bins * bins, bins * bins, OUTPUTS)
        for output, output_name in enumerate(OUTPUT_NAMES):
            singular = np.linalg.svd(arranged[:, :, output], compute_uv=False)
            energy = float(np.sum(singular * singular))
            residual = lambda rank: float(math.sqrt(np.sum(singular[rank:] ** 2) / max(energy, 1e-30)))
            r4, r12 = residual(4), residual(12)
            rank4_values.append(r4)
            rank12_values.append(r12)
            per_output[output_name] = {
                "relativeFrobeniusResidualRank4": r4,
                "relativeFrobeniusResidualRank12": r12,
                "leadingSingularValues": singular[:12].tolist(),
            }
        report["partitions"][name] = {
            "rank4ResidualMedian": float(np.median(rank4_values)),
            "rank4ResidualMaximum": float(np.max(rank4_values)),
            "rank12ResidualMedian": float(np.median(rank12_values)),
            "rank12ResidualMaximum": float(np.max(rank12_values)),
            "outputs": per_output,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--seed", type=int, default=23072028)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    started = time.time()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    train_count = int(manifest["recipe"]["records"]["train"])
    validation_count = int(manifest["recipe"]["records"]["validation"])
    train = records(root / "train.f32", train_count)
    validation = records(root / "validation.f32", validation_count)
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = CP12().to(device)
    coordinates = torch.from_numpy(np.asarray(train[:, :COORDINATES], dtype=np.float32)).to(device)
    truth = torch.from_numpy(np.asarray(train[:, COORDINATES:], dtype=np.float32)).to(device)
    print(f"[gbc2-k6-cp12] device={device} train/heldout={train_count}/{validation_count}", flush=True)
    last_loss = train_steps(model, coordinates, truth, args.steps, args.batch, args.learning_rate, args.seed)
    tables = sample_tables(model, device)
    continuous = {
        "train": metrics(train[:, COORDINATES:], predict_continuous(model, train[:, :COORDINATES], device)),
        "validation": metrics(validation[:, COORDINATES:], predict_continuous(model, validation[:, :COORDINATES], device)),
    }
    sampled = {
        "train": metrics(train[:, COORDINATES:], predict_sampled(model, tables, train[:, :COORDINATES])),
        "validation": metrics(validation[:, COORDINATES:], predict_sampled(model, tables, validation[:, :COORDINATES])),
    }
    passed = green(sampled["validation"])
    diagnostic = None if passed else rank_diagnostics(
        np.concatenate((train[:, :COORDINATES], validation[:, :COORDINATES])),
        np.concatenate((train[:, COORDINATES:], validation[:, COORDINATES:])),
    )
    recipe: dict[str, Any] = {
        "schema": "laas-gbc2-k6-cp12-cap-fit-recipe/v1",
        "implementationSha256": digest(Path(__file__)),
        "sharedModuleSha256": {
            "base": digest(Path(__file__).with_name("train_gbc2_k6_cap.py")),
            "sparseUtilities": digest(Path(__file__).with_name("train_gbc2_k6_sparse_cap.py")),
        },
        "truthManifestSha256": digest(args.manifest),
        "sourceSha256": manifest["source"]["sha256"],
        "architecture": {
            "name": "K6-CP12", "pairs": PAIR_NAMES,
            "ranks": ["uv*ae RGBA", "ua*ve RGBA", "ue*va RGBA"],
            "features": FEATURES, "head": "dense linear 12x18 plus bias",
            "runtimeFmas": 216, "reads": 6, "tableBytes": int(tables.nbytes),
            "headBytesAligned": HEAD_BYTES_CP12,
        },
        "training": {
            "steps": args.steps, "batch": args.batch, "learningRate": args.learning_rate,
            "seed": args.seed, "device": str(device), "lastLoss": last_loss,
        },
        "gate": {
            "silhouetteIoU": .97, "coverageAbsoluteP95": .15,
            "premultipliedRgbMaxChannelP95": .15,
            "conditionalMeanDepthP95Metres": .05,
            "conditionalNormalP95Degrees": 20,
        },
    }
    recipe_sha = digest_bytes(canonical(recipe).encode())
    output_root = Path("data/work/groundcover-gbc2-k6-cp12-fit") / manifest["source"]["sha256"][:16] / recipe_sha[:16]
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "k6-cp12-mip0-448-rgba16f.bin"
    head_path = output_root / "k6-head-v2-cp12.bin"
    table_path.write_bytes(np.asarray(tables, dtype="<f2", order="C").tobytes())
    head = head_bytes(model)
    head_path.write_bytes(head)
    asset_path: Path | None = None
    if passed:
        asset = append_asset(BASE_ASSET, tables, head)
        asset_path = output_root / "calamagrostis-canescens.gbc2"
        asset_path.write_bytes(asset)
        STABLE_ASSET.write_bytes(asset)
    report = {
        "schema": "laas-gbc2-k6-cp12-cap-fit/v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe": recipe,
        "truth": {"manifest": str(args.manifest), "train": train_count, "heldOut": validation_count},
        "continuousGenerator": continuous,
        "sampledRgba16fTables": sampled,
        "rankDiagnostics": diagnostic,
        "files": {
            "tables": {"path": str(table_path), "sha256": digest(table_path), "bytes": table_path.stat().st_size},
            "head": {"path": str(head_path), "sha256": digest(head_path), "bytes": head_path.stat().st_size},
            "asset": None if asset_path is None else {"path": str(asset_path), "sha256": digest(asset_path), "bytes": asset_path.stat().st_size},
        },
        "validity": {
            "status": "GREEN_ONE_CAP_LEVEL" if passed else "RED_K6_CP12_FIT",
            "productionAssetMutated": passed, "completeExteriorAsset": False,
            "blocker": None if passed else "K6-CP12 failed the held-out cap-level gate",
        },
        "elapsedSeconds": time.time() - started,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outputRoot": str(output_root), "report": str(report_path),
                      "validity": report["validity"], "continuous": continuous["validation"],
                      "sampled": sampled["validation"],
                      "rankSummary": None if diagnostic is None else {
                          name: {key: value for key, value in partition.items() if key != "outputs"}
                          for name, partition in diagnostic["partitions"].items()
                      }}, indent=2), flush=True)


if __name__ == "__main__":
    main()

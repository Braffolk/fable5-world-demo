"""Single diagnose/fix fit for the non-collapsing GBC2-K6 decoder.

Six RGBA pair-plane reads remain 24 independent scalar features. Eighteen
features have one sparse direct output each; the six remaining features are
assigned by sensitivity on a tuning subset taken only from the training file.
A rank-two residual sees all features. Runtime cost is 24 + 48 + 36 = 108
FMAs plus bias initialization and constrained output transforms.
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
    HEAD_BYTES,
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

PRIMARY_FEATURES = tuple(layer * CHANNELS + channel for channel in range(3) for layer in range(LAYERS))
PRIMARY_ASSIGNMENTS = tuple(range(OUTPUTS))
SPARE_FEATURES = tuple(layer * CHANNELS + 3 for layer in range(LAYERS))
assert len(PRIMARY_FEATURES) == len(PRIMARY_ASSIGNMENTS) == OUTPUTS
assert len(SPARE_FEATURES) == 6


def pair_features(pair: torch.Tensor, axes: tuple[int, int]) -> torch.Tensor:
    frequencies = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), dtype=pair.dtype, device=pair.device)
    angles = pair.unsqueeze(2) * frequencies.reshape(1, 1, -1) * (2 * math.pi)
    raw = pair.unsqueeze(2) * 2 - 1
    periodic = torch.tensor(tuple(axis in PERIODIC for axis in axes), device=pair.device).reshape(1, 2, 1)
    raw = torch.where(periodic, torch.zeros_like(raw), raw)
    return torch.cat((raw, torch.sin(angles), torch.cos(angles)), dim=2).flatten(1)


class SparseK6(torch.nn.Module):
    def __init__(self, width: int = 96) -> None:
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
        self.direct_weight = torch.nn.Parameter(torch.ones(LAYERS * CHANNELS) * .05)
        self.bias = torch.nn.Parameter(torch.zeros(OUTPUTS))
        self.residual_a = torch.nn.Linear(LAYERS * CHANNELS, 2)
        self.residual_b = torch.nn.Linear(2, OUTPUTS, bias=False)
        self.spare_assignments: tuple[int, ...] = ()
        torch.nn.init.zeros_(self.plane[-1].weight)
        torch.nn.init.zeros_(self.plane[-1].bias)
        torch.nn.init.normal_(self.residual_a.weight, std=.02)
        torch.nn.init.zeros_(self.residual_a.bias)
        torch.nn.init.normal_(self.residual_b.weight, std=.02)
        with torch.no_grad():
            # Start the coverage logits below 0.5 instead of making the sparse
            # field learn the background prior through every plane.
            self.bias[0] = -1
            self.bias[9] = -1

    def sampled(self, coordinates: torch.Tensor) -> torch.Tensor:
        pairs = coordinates[:, self.pair_axes]
        features = torch.stack(
            tuple(pair_features(pairs[:, layer], PAIRS[layer]) for layer in range(LAYERS)),
            dim=1,
        )
        identity = self.identity.to(dtype=coordinates.dtype).unsqueeze(0).expand(len(coordinates), -1, -1)
        return torch.sigmoid(self.plane(torch.cat((features, identity), dim=2))).flatten(1)

    def raw_from_sampled(self, sampled: torch.Tensor) -> torch.Tensor:
        centred = sampled * 2 - 1
        raw = self.bias.unsqueeze(0).expand(len(sampled), -1).clone()
        for feature, output in zip(PRIMARY_FEATURES, PRIMARY_ASSIGNMENTS):
            raw[:, output] = raw[:, output] + centred[:, feature] * self.direct_weight[feature]
        for feature, output in zip(SPARE_FEATURES, self.spare_assignments):
            raw[:, output] = raw[:, output] + centred[:, feature] * self.direct_weight[feature]
        hidden = torch.nn.functional.silu(self.residual_a(centred))
        return raw + self.residual_b(hidden)

    def raw(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.raw_from_sampled(self.sampled(coordinates))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return decode(self.raw(coordinates))


def fit_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    total_a_prediction = prediction[:, 0] + prediction[:, 9]
    total_a_truth = truth[:, 0] + truth[:, 9]
    total_rgb_prediction = prediction[:, 3:6] + prediction[:, 12:15]
    total_rgb_truth = truth[:, 3:6] + truth[:, 12:15]
    total_n_prediction = prediction[:, 6:9] + prediction[:, 15:18]
    total_n_truth = truth[:, 6:9] + truth[:, 15:18]
    cover_bce = torch.nn.functional.binary_cross_entropy(
        torch.clamp(total_a_prediction, 1e-5, 1 - 1e-5), total_a_truth,
    )
    loss = (
        4 * cover_bce
        + 6 * torch.mean((total_a_prediction - total_a_truth) ** 2)
        + 2 * torch.mean((prediction[:, [0, 9]] - truth[:, [0, 9]]) ** 2)
        + 12 * torch.mean((total_rgb_prediction - total_rgb_truth) ** 2)
        + 3 * torch.mean((prediction[:, [3, 4, 5, 12, 13, 14]] - truth[:, [3, 4, 5, 12, 13, 14]]) ** 2)
        + 2 * torch.mean((total_n_prediction - total_n_truth) ** 2)
    )
    for base in (0, 9):
        valid = truth[:, base] >= .0625
        if torch.any(valid):
            ta = truth[valid, base]
            pa = torch.clamp(prediction[valid, base], min=1e-5)
            truth_mu = truth[valid, base + 1] / ta
            prediction_mu = prediction[valid, base + 1] / pa
            truth_nu = truth[valid, base + 2] / ta
            prediction_nu = prediction[valid, base + 2] / pa
            loss = loss + 3 * torch.mean((10 * (prediction_mu - truth_mu)) ** 2)
            loss = loss + torch.mean((10 * (prediction_nu - truth_nu)) ** 2)
            truth_normal = truth[valid, base + 6:base + 9] / ta.unsqueeze(1)
            prediction_normal = prediction[valid, base + 6:base + 9] / pa.unsqueeze(1)
            loss = loss + torch.mean((prediction_normal - truth_normal) ** 2)
    return loss


def train_steps(
    model: SparseK6,
    coordinates: torch.Tensor,
    truth: torch.Tensor,
    steps: int,
    batch: int,
    learning_rate: float,
    seed: int,
    label: str,
) -> float:
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model.train()
    last = 0.
    for step in range(steps):
        index = torch.randint(0, len(coordinates), (batch,), generator=generator).to(coordinates.device)
        prediction = model(coordinates[index])
        loss = fit_loss(prediction, truth[index])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimiser.step()
        last = float(loss.detach().cpu())
        if step == 0 or (step + 1) % 250 == 0:
            print(f"[gbc2-k6-sparse] {label} {step + 1}/{steps} loss={last:.6f}", flush=True)
    return last


def choose_spares(
    model: SparseK6,
    coordinates: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Select six distinct output accumulators on a train-only tuning split."""
    model.eval()
    with torch.no_grad():
        sampled = model.sampled(coordinates)
        base_raw = model.raw_from_sampled(sampled)
    probe = torch.zeros((len(SPARE_FEATURES), OUTPUTS), dtype=coordinates.dtype, device=coordinates.device, requires_grad=True)
    centred_spares = sampled[:, SPARE_FEATURES] * 2 - 1
    prediction = decode(base_raw + centred_spares @ probe)
    loss = fit_loss(prediction, truth)
    loss.backward()
    sensitivity = np.abs(probe.grad.detach().cpu().numpy())
    assignments: list[int] = []
    available = set(range(OUTPUTS))
    for spare in range(len(SPARE_FEATURES)):
        output = max(available, key=lambda candidate: sensitivity[spare, candidate])
        assignments.append(output)
        available.remove(output)
    return tuple(assignments), sensitivity


@torch.no_grad()
def sample_tables(model: SparseK6, device: torch.device, batch_rows: int = 8) -> np.ndarray:
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
            chunks.append(torch.sigmoid(model.plane(torch.cat((features, identity), dim=1))).cpu().numpy())
        tables[layer] = np.concatenate(chunks).reshape(SIZE, SIZE, CHANNELS).astype(np.float16)
        print(f"[gbc2-k6-sparse] sampled layer {layer + 1}/6 {PAIR_NAMES[layer]}", flush=True)
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
    return samples.reshape(len(coordinates), -1)


def raw_np(model: SparseK6, sampled: np.ndarray) -> np.ndarray:
    centred = sampled * 2 - 1
    direct_weight = model.direct_weight.detach().cpu().numpy()
    raw = np.broadcast_to(model.bias.detach().cpu().numpy(), (len(sampled), OUTPUTS)).copy()
    for feature, output in zip(PRIMARY_FEATURES, PRIMARY_ASSIGNMENTS):
        raw[:, output] += centred[:, feature] * direct_weight[feature]
    for feature, output in zip(SPARE_FEATURES, model.spare_assignments):
        raw[:, output] += centred[:, feature] * direct_weight[feature]
    a_weight = model.residual_a.weight.detach().cpu().numpy()
    a_bias = model.residual_a.bias.detach().cpu().numpy()
    b_weight = model.residual_b.weight.detach().cpu().numpy()
    hidden_pre = centred @ a_weight.T + a_bias
    hidden = hidden_pre / (1 + np.exp(-np.clip(hidden_pre, -30, 30)))
    return raw + hidden @ b_weight.T


def continuous_prediction(model: SparseK6, coordinates: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(coordinates), 2048):
            x = torch.from_numpy(np.asarray(coordinates[start:start + 2048], dtype=np.float32)).to(device)
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks)


def head_bytes(model: SparseK6) -> bytes:
    result = bytearray(HEAD_BYTES)
    result[0:3] = b"K6H"
    # v2 sparse decoder metadata.
    struct.pack_into("<IIIIIffIIIIII", result, 4,
                     2, 24, 18, 2, 6, HORIZON, 1e-6, 1,
                     64, 24, 96, 128, 2)
    assignments = np.full(24, 255, dtype=np.uint8)
    for feature, output in zip(PRIMARY_FEATURES, PRIMARY_ASSIGNMENTS):
        assignments[feature] = output
    for feature, output in zip(SPARE_FEATURES, model.spare_assignments):
        assignments[feature] = output
    result[64:88] = assignments.tobytes()
    weights = np.concatenate((
        model.direct_weight.detach().cpu().numpy(),
        model.bias.detach().cpu().numpy(),
        model.residual_a.weight.detach().cpu().numpy().reshape(-1),
        model.residual_a.bias.detach().cpu().numpy(),
        model.residual_b.weight.detach().cpu().numpy().reshape(-1),
    )).astype("<f4")
    if weights.size != 128:
        raise AssertionError(weights.size)
    result[96:96 + weights.nbytes] = weights.tobytes()
    return bytes(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--selection-steps", type=int, default=1250)
    parser.add_argument("--final-steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=23072027)
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
    model = SparseK6().to(device)
    coordinates = torch.from_numpy(np.asarray(train[:, :COORDINATES], dtype=np.float32)).to(device)
    truth = torch.from_numpy(np.asarray(train[:, COORDINATES:], dtype=np.float32)).to(device)
    fit_end = train_count - 512
    print(f"[gbc2-k6-sparse] device={device} fit/tune/heldout={fit_end}/512/{validation_count}", flush=True)
    train_steps(model, coordinates[:fit_end], truth[:fit_end], args.selection_steps, args.batch,
                args.learning_rate, args.seed, "selection")
    spare_assignments, sensitivity = choose_spares(model, coordinates[fit_end:], truth[fit_end:])
    model.spare_assignments = spare_assignments
    print(f"[gbc2-k6-sparse] spare assignments={spare_assignments}", flush=True)
    last_loss = train_steps(model, coordinates, truth, args.final_steps, args.batch,
                            args.learning_rate * .75, args.seed + 1, "final")
    tables = sample_tables(model, device)
    continuous = {
        "train": metrics(train[:, COORDINATES:], continuous_prediction(model, train[:, :COORDINATES], device)),
        "validation": metrics(validation[:, COORDINATES:], continuous_prediction(model, validation[:, :COORDINATES], device)),
    }
    sampled = {
        "train": metrics(train[:, COORDINATES:], decode_np(raw_np(model, bilinear_samples(tables, train[:, :COORDINATES])))),
        "validation": metrics(validation[:, COORDINATES:], decode_np(raw_np(model, bilinear_samples(tables, validation[:, :COORDINATES])))),
    }
    passed = green(sampled["validation"])
    recipe: dict[str, Any] = {
        "schema": "laas-gbc2-k6-sparse-cap-fit-recipe/v1",
        "implementationSha256": digest(Path(__file__)),
        "sharedModuleSha256": digest(Path(__file__).with_name("train_gbc2_k6_cap.py")),
        "truthManifestSha256": digest(args.manifest),
        "sourceSha256": manifest["source"]["sha256"],
        "architecture": {
            "samples": "six 448x448 RGBA16F pair planes; 24 independent centred channels",
            "direct": "24 scalar weights, one fixed output assignment per feature",
            "primaryAssignments": list(PRIMARY_ASSIGNMENTS),
            "spareFeatures": list(SPARE_FEATURES),
            "spareAssignments": list(spare_assignments),
            "spareSelection": "absolute loss sensitivity on last 512 training records; unique greedy outputs",
            "residual": "24 -> SiLU rank2 -> 18",
            "runtimeFmas": {"direct": 24, "rank2Input": 48, "rank2Output": 36, "total": 108},
            "analyticZJ": "omitted: this truth level has fixed J and pair planes already receive every z coordinate",
        },
        "training": {
            "selectionSteps": args.selection_steps, "finalSteps": args.final_steps,
            "batch": args.batch, "learningRate": args.learning_rate, "seed": args.seed,
            "device": str(device), "lastLoss": last_loss,
        },
        "gate": {
            "silhouetteIoU": .97, "coverageAbsoluteP95": .15,
            "premultipliedRgbMaxChannelP95": .15,
            "conditionalMeanDepthP95Metres": .05,
            "conditionalNormalP95Degrees": 20,
        },
    }
    recipe_sha = digest_bytes(canonical(recipe).encode())
    output_root = Path("data/work/groundcover-gbc2-k6-sparse-fit") / manifest["source"]["sha256"][:16] / recipe_sha[:16]
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "k6-mip0-448-rgba16f.bin"
    head_path = output_root / "k6-head-v2-sparse.bin"
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
        "schema": "laas-gbc2-k6-sparse-cap-fit/v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe": recipe,
        "truth": {"manifest": str(args.manifest), "train": train_count, "heldOut": validation_count},
        "spareSensitivity": sensitivity.tolist(),
        "continuousGenerator": continuous,
        "sampledRgba16fTables": sampled,
        "files": {
            "tables": {"path": str(table_path), "sha256": digest(table_path), "bytes": table_path.stat().st_size},
            "head": {"path": str(head_path), "sha256": digest(head_path), "bytes": head_path.stat().st_size},
            "asset": None if asset_path is None else {"path": str(asset_path), "sha256": digest(asset_path), "bytes": asset_path.stat().st_size},
        },
        "validity": {
            "status": "GREEN_ONE_CAP_LEVEL" if passed else "RED_K6_SPARSE_FIT",
            "productionAssetMutated": passed,
            "completeExteriorAsset": False,
            "blocker": None if passed else "non-collapsing 108-FMA K6 failed the held-out cap-level gate",
        },
        "elapsedSeconds": time.time() - started,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outputRoot": str(output_root), "report": str(report_path),
                      "validity": report["validity"], "continuous": continuous["validation"],
                      "sampled": sampled["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

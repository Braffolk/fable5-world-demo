"""Fit and gate one production GBC2-K6 exterior cap level.

The neural network exists only to generate six 448^2 RGBA16F tables and the
fixed 14->4->18 head. Validation always samples those quantized tables with
the runtime bilinear law; continuous-generator scores cannot make the gate
green. A failed gate writes evidence but never mutates the production asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

RECORD_FLOATS = 22
COORDINATES = 4
OUTPUTS = 18
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_NAMES = ("uv", "ua", "ue", "va", "ve", "ae")
PERIODIC = frozenset((0, 1, 2))
SIZE = 448
LAYERS = 6
CHANNELS = 4
HORIZON = 155.0
HEAD_BYTES = 768
BASE_ASSET = Path("src/assets/groundcover/calamagrostis-canescens.gbc2")
STABLE_ASSET = BASE_ASSET


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def records(path: Path, count: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size != count * RECORD_FLOATS:
        raise ValueError(f"{path}: {data.size} floats, expected {count * RECORD_FLOATS}")
    return data.reshape(count, RECORD_FLOATS)


def coordinate_features(pair: torch.Tensor, axes: tuple[int, int]) -> torch.Tensor:
    frequencies = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), dtype=pair.dtype, device=pair.device)
    angles = pair.unsqueeze(2) * frequencies.reshape(1, 1, -1) * (2 * math.pi)
    raw = pair.unsqueeze(2) * 2 - 1
    periodic = torch.tensor(tuple(axis in PERIODIC for axis in axes), device=pair.device).reshape(1, 2, 1)
    raw = torch.where(periodic, torch.zeros_like(raw), raw)
    return torch.cat((raw, torch.sin(angles), torch.cos(angles)), dim=2).flatten(1)


class K6(torch.nn.Module):
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
        self.a = torch.nn.Linear(14, 4)
        self.b = torch.nn.Linear(4, OUTPUTS)
        torch.nn.init.zeros_(self.plane[-1].weight)
        torch.nn.init.zeros_(self.plane[-1].bias)
        torch.nn.init.normal_(self.a.weight, std=.05)
        torch.nn.init.zeros_(self.a.bias)
        torch.nn.init.normal_(self.b.weight, std=.05)
        torch.nn.init.zeros_(self.b.bias)

    def factors(self, coordinates: torch.Tensor) -> torch.Tensor:
        pairs = coordinates[:, self.pair_axes]
        features = torch.stack(
            tuple(coordinate_features(pairs[:, layer], PAIRS[layer]) for layer in range(LAYERS)),
            dim=1,
        )
        identity = self.identity.to(dtype=coordinates.dtype).unsqueeze(0).expand(len(coordinates), -1, -1)
        sample = torch.sigmoid(self.plane(torch.cat((features, identity), dim=2)))
        return .5 + sample

    def raw(self, coordinates: torch.Tensor) -> torch.Tensor:
        product = torch.prod(self.factors(coordinates), dim=1)
        pairwise = product[:, (0, 0, 0, 1, 1, 2)] * product[:, (1, 2, 3, 2, 3, 3)]
        phi = torch.cat((product, product * product, pairwise), dim=1)
        return self.b(torch.nn.functional.silu(self.a(phi)))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return decode(self.raw(coordinates))


def decode(raw: torch.Tensor) -> torch.Tensor:
    result: list[torch.Tensor] = []
    alpha0 = torch.sigmoid(raw[:, 0:1])
    alpha1 = (1 - alpha0) * torch.sigmoid(raw[:, 9:10])
    for base, alpha in ((0, alpha0), (9, alpha1)):
        mu = torch.sigmoid(raw[:, base + 1:base + 2])
        nu = mu * mu + torch.sigmoid(raw[:, base + 2:base + 3]) * (1 - mu * mu)
        rgb = alpha * torch.sigmoid(raw[:, base + 3:base + 6])
        normal = alpha * torch.tanh(raw[:, base + 6:base + 9])
        result.append(torch.cat((alpha, alpha * mu, alpha * nu, rgb, normal), dim=1))
    return torch.cat(result, dim=1)


def decode_np(raw: np.ndarray) -> np.ndarray:
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    alpha0 = sigmoid(raw[:, 0:1])
    alpha1 = (1 - alpha0) * sigmoid(raw[:, 9:10])
    result: list[np.ndarray] = []
    for base, alpha in ((0, alpha0), (9, alpha1)):
        mu = sigmoid(raw[:, base + 1:base + 2])
        nu = mu * mu + sigmoid(raw[:, base + 2:base + 3]) * (1 - mu * mu)
        rgb = alpha * sigmoid(raw[:, base + 3:base + 6])
        normal = alpha * np.tanh(raw[:, base + 6:base + 9])
        result.append(np.concatenate((alpha, alpha * mu, alpha * nu, rgb, normal), axis=1))
    return np.concatenate(result, axis=1).astype(np.float32)


def loss_value(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    total_a_prediction = prediction[:, 0] + prediction[:, 9]
    total_a_truth = truth[:, 0] + truth[:, 9]
    total_rgb_prediction = prediction[:, 3:6] + prediction[:, 12:15]
    total_rgb_truth = truth[:, 3:6] + truth[:, 12:15]
    total_n_prediction = prediction[:, 6:9] + prediction[:, 15:18]
    total_n_truth = truth[:, 6:9] + truth[:, 15:18]
    return (
        8 * torch.mean((total_a_prediction - total_a_truth) ** 2)
        + 2 * torch.mean((prediction[:, (0, 9)] - truth[:, (0, 9)]) ** 2)
        + 12 * torch.mean((total_rgb_prediction - total_rgb_truth) ** 2)
        + 3 * torch.mean((prediction[:, (3, 4, 5, 12, 13, 14)] - truth[:, (3, 4, 5, 12, 13, 14)]) ** 2)
        + 3 * torch.mean((10 * (prediction[:, (1, 10)] - truth[:, (1, 10)])) ** 2)
        + torch.mean((10 * (prediction[:, (2, 11)] - truth[:, (2, 11)])) ** 2)
        + 2 * torch.mean((total_n_prediction - total_n_truth) ** 2)
    )


@torch.no_grad()
def sample_tables(model: K6, device: torch.device, batch_rows: int = 8) -> np.ndarray:
    tables = np.empty((LAYERS, SIZE, SIZE, CHANNELS), dtype=np.float16)
    model.eval()
    for layer, axes in enumerate(PAIRS):
        chunks: list[np.ndarray] = []
        for row_start in range(0, SIZE, batch_rows):
            rows = min(batch_rows, SIZE - row_start)
            s = np.tile((np.arange(SIZE, dtype=np.float32) + .5) / SIZE, rows)
            t = np.repeat((np.arange(row_start, row_start + rows, dtype=np.float32) + .5) / SIZE, SIZE)
            pair = torch.from_numpy(np.stack((s, t), axis=1)).to(device)
            features = coordinate_features(pair, axes)
            identity = model.identity[layer:layer + 1].to(device=device, dtype=pair.dtype).expand(len(pair), -1)
            sample = torch.sigmoid(model.plane(torch.cat((features, identity), dim=1)))
            chunks.append(sample.cpu().numpy())
        tables[layer] = np.concatenate(chunks).reshape(SIZE, SIZE, CHANNELS).astype(np.float16)
        print(f"[gbc2-k6-fit] sampled layer {layer + 1}/6 {PAIR_NAMES[layer]}", flush=True)
    return tables


def bilinear_factors(tables: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    factors = np.empty((len(coordinates), LAYERS, CHANNELS), dtype=np.float32)
    for layer, (axis_s, axis_t) in enumerate(PAIRS):
        s = coordinates[:, axis_s]
        t = coordinates[:, axis_t]
        sx = s * SIZE - .5
        ty = t * SIZE - .5
        x0 = np.floor(sx).astype(np.int64)
        y0 = np.floor(ty).astype(np.int64)
        fx = (sx - x0).astype(np.float32)
        fy = (ty - y0).astype(np.float32)

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
        factors[:, layer] = .5 + top + (bottom - top) * fy[:, None]
    return factors


def predict_tables(model: K6, tables: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    product = np.prod(bilinear_factors(tables, coordinates), axis=1)
    pairwise = product[:, (0, 0, 0, 1, 1, 2)] * product[:, (1, 2, 3, 2, 3, 3)]
    phi = np.concatenate((product, product * product, pairwise), axis=1)
    a_weight = model.a.weight.detach().cpu().numpy()
    a_bias = model.a.bias.detach().cpu().numpy()
    b_weight = model.b.weight.detach().cpu().numpy()
    b_bias = model.b.bias.detach().cpu().numpy()
    hidden_pre = phi @ a_weight.T + a_bias
    hidden = hidden_pre / (1 + np.exp(-np.clip(hidden_pre, -30, 30)))
    return decode_np(hidden @ b_weight.T + b_bias)


def quantiles(value: np.ndarray) -> dict[str, float]:
    clean = np.asarray(value, dtype=np.float64).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {"p50": 0., "p95": 0., "p99": 0., "maximum": 0.}
    return {name: float(np.quantile(clean, q)) for name, q in (("p50", .5), ("p95", .95), ("p99", .99), ("maximum", 1.))}


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    total_a = truth[:, 0] + truth[:, 9]
    predicted_a = prediction[:, 0] + prediction[:, 9]
    total_rgb = truth[:, 3:6] + truth[:, 12:15]
    predicted_rgb = prediction[:, 3:6] + prediction[:, 12:15]
    hit_truth = total_a >= .125
    hit_prediction = predicted_a >= .125
    union = np.count_nonzero(hit_truth | hit_prediction)
    depth_errors: list[np.ndarray] = []
    normal_angles: list[np.ndarray] = []
    variance_errors: list[np.ndarray] = []
    for base in (0, 9):
        valid = truth[:, base] >= .125
        if not np.any(valid):
            continue
        ta = truth[valid, base]
        pa = np.maximum(prediction[valid, base], 1e-6)
        tmu = truth[valid, base + 1] / ta
        pmu = prediction[valid, base + 1] / pa
        tnu = truth[valid, base + 2] / ta
        pnu = prediction[valid, base + 2] / pa
        depth_errors.append(HORIZON * np.abs(pmu - tmu))
        variance_errors.append(HORIZON * np.abs(np.sqrt(np.maximum(0, pnu - pmu * pmu)) - np.sqrt(np.maximum(0, tnu - tmu * tmu))))
        tn = truth[valid, base + 6:base + 9] / ta[:, None]
        pn = prediction[valid, base + 6:base + 9] / pa[:, None]
        tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-8)
        pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-8)
        normal_angles.append(np.degrees(np.arccos(np.clip(np.sum(tn * pn, axis=1), -1, 1))))
    return {
        "records": len(truth),
        "truthCoverageMean": float(np.mean(total_a)),
        "coverageAbsolute": quantiles(np.abs(predicted_a - total_a)),
        "silhouetteIoU": float(np.count_nonzero(hit_truth & hit_prediction) / max(1, union)),
        "premultipliedRgbMaxChannel": quantiles(np.max(np.abs(predicted_rgb - total_rgb), axis=1)),
        "conditionalMeanDepthMetres": quantiles(np.concatenate(depth_errors) if depth_errors else np.zeros(0)),
        "conditionalDepthStddevMetres": quantiles(np.concatenate(variance_errors) if variance_errors else np.zeros(0)),
        "conditionalNormalDegrees": quantiles(np.concatenate(normal_angles) if normal_angles else np.zeros(0)),
    }


def green(value: dict[str, Any]) -> bool:
    return bool(
        value["silhouetteIoU"] >= .97
        and value["coverageAbsolute"]["p95"] <= .15
        and value["premultipliedRgbMaxChannel"]["p95"] <= .15
        and value["conditionalMeanDepthMetres"]["p95"] <= .05
        and value["conditionalNormalDegrees"]["p95"] <= 20
    )


def head_bytes(model: K6) -> bytes:
    result = bytearray(HEAD_BYTES)
    result[0:3] = b"K6H"
    struct.pack_into("<IIIIIffIII", result, 4,
                     1, 14, 4, 18, 6, HORIZON, 1e-6, 1, 64, 150)
    weights = np.concatenate((
        model.a.weight.detach().cpu().numpy().reshape(-1),
        model.a.bias.detach().cpu().numpy().reshape(-1),
        model.b.weight.detach().cpu().numpy().reshape(-1),
        model.b.bias.detach().cpu().numpy().reshape(-1),
    )).astype("<f4")
    if weights.size != 150:
        raise AssertionError(weights.size)
    result[64:64 + weights.nbytes] = weights.tobytes()
    return bytes(result)


def align(value: int, alignment: int = 256) -> int:
    return (value + alignment - 1) // alignment * alignment


def append_asset(base_path: Path, tables: np.ndarray, head: bytes) -> bytes:
    base = bytearray(base_path.read_bytes())
    if base[:4] != b"GBC2" or struct.unpack_from("<I", base, 4)[0] != 1:
        raise ValueError("base asset is not GBC2/v1")
    if struct.unpack_from("<I", base, 52)[0] != 0:
        raise ValueError("base asset already contains K6")
    source_sha = bytes(base[128:160]).hex()
    if source_sha != "2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c":
        raise ValueError(f"base asset source identity changed: {source_sha}")
    k6 = np.asarray(tables, dtype="<f2", order="C").tobytes()
    k6_offset = align(len(base))
    head_offset = align(k6_offset + len(k6))
    total = align(head_offset + len(head))
    output = bytearray(total)
    output[:len(base)] = base
    output[k6_offset:k6_offset + len(k6)] = k6
    output[head_offset:head_offset + len(head)] = head
    struct.pack_into("<IIIIIIII", output, 52, 1, k6_offset, len(k6), SIZE, LAYERS, 1, head_offset, len(head))
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=23072026)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    started = time.time()
    manifest = json.loads(args.manifest.read_text())
    train_count = int(manifest["recipe"]["records"]["train"])
    validation_count = int(manifest["recipe"]["records"]["validation"])
    root = args.manifest.parent
    train = records(root / "train.f32", train_count)
    validation = records(root / "validation.f32", validation_count)
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = K6().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    train_coordinates = torch.from_numpy(np.asarray(train[:, :COORDINATES], dtype=np.float32)).to(device)
    train_truth = torch.from_numpy(np.asarray(train[:, COORDINATES:], dtype=np.float32)).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    print(f"[gbc2-k6-fit] device={device} records={train_count}/{validation_count} steps={args.steps}", flush=True)
    model.train()
    last_loss = 0.
    for step in range(args.steps):
        index = torch.randint(0, train_count, (args.batch,), generator=generator).to(device)
        prediction = model(train_coordinates[index])
        loss = loss_value(prediction, train_truth[index])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimiser.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % 250 == 0 or step == 0:
            print(f"[gbc2-k6-fit] step {step + 1}/{args.steps} loss={last_loss:.6f}", flush=True)
    tables = sample_tables(model, device)
    train_prediction = predict_tables(model, tables, train[:, :COORDINATES])
    validation_prediction = predict_tables(model, tables, validation[:, :COORDINATES])
    train_metrics = metrics(train[:, COORDINATES:], train_prediction)
    validation_metrics = metrics(validation[:, COORDINATES:], validation_prediction)
    passed = green(validation_metrics)
    recipe = {
        "schema": "laas-gbc2-k6-cap-fit-recipe/v1",
        "implementationSha256": digest(Path(__file__)),
        "truthManifestSha256": digest(args.manifest),
        "sourceSha256": manifest["source"]["sha256"],
        "architecture": {
            "pairs": PAIR_NAMES, "size": SIZE, "channels": CHANNELS,
            "factor": "0.5 + bilinear(RGBA16F sample)",
            "head": "phi14 -> SiLU rank4 -> raw18; constrained two-stratum decoder",
            "storage": "mip-major then six layers then texel then RGBA16F",
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
    output_root = Path("data/work/groundcover-gbc2-k6-fit") / manifest["source"]["sha256"][:16] / recipe_sha[:16]
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "k6-mip0-448-rgba16f.bin"
    head_path = output_root / "k6-head.bin"
    table_path.write_bytes(np.asarray(tables, dtype="<f2", order="C").tobytes())
    head = head_bytes(model)
    head_path.write_bytes(head)
    asset_path: Path | None = None
    asset_bytes: bytes | None = None
    if passed:
        asset_bytes = append_asset(BASE_ASSET, tables, head)
        asset_path = output_root / "calamagrostis-canescens.gbc2"
        asset_path.write_bytes(asset_bytes)
        STABLE_ASSET.write_bytes(asset_bytes)
    report = {
        "schema": "laas-gbc2-k6-cap-fit/v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe": recipe,
        "truth": {"manifest": str(args.manifest), "trainRecords": train_count, "validationRecords": validation_count},
        "sampledTableValidation": {"train": train_metrics, "validation": validation_metrics},
        "files": {
            "tables": {"path": str(table_path), "sha256": digest(table_path), "bytes": table_path.stat().st_size},
            "head": {"path": str(head_path), "sha256": digest(head_path), "bytes": head_path.stat().st_size},
            "asset": None if asset_path is None else {"path": str(asset_path), "sha256": digest(asset_path), "bytes": asset_path.stat().st_size},
        },
        "validity": {
            "status": "GREEN_ONE_CAP_LEVEL" if passed else "RED_K6_FIT",
            "productionAssetMutated": passed,
            "scope": "one 4 m standoff production-pixel top-cap terminal, elevation 5..90 degrees",
            "completeExteriorAsset": False,
            "blocker": None if passed else "sampled RGBA16F K6 failed at least one held-out quality threshold",
        },
        "elapsedSeconds": time.time() - started,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outputRoot": str(output_root), "report": str(report_path), "validity": report["validity"], "validation": validation_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

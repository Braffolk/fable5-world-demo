"""Fit and falsify the frozen six-coordinate pairwise-plane transfer field.

The neural functions in this file are offline table generators only.  A passing
artifact contains fifteen RGBA16F planes and one fixed quadratic affine head.
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

FIELDS = 16
HORIZON = 155.0
ENDPOINT_ALPHA = 10.0
RESOLUTION = 512
PERIODIC = frozenset((0, 1, 3))
COORD_NAMES = ("u", "v", "originY", "phi", "elevation", "endpointQ")


def ordered_pairs() -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            # A mixed layer always places the periodic coordinate on S/x.
            result.append((j, i) if j in PERIODIC and i not in PERIODIC else (i, j))
    return tuple(result)


PAIRS = ordered_pairs()
assert len(PAIRS) == 15 and len(set(tuple(sorted(pair)) for pair in PAIRS)) == 15


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path, count: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size != count * FIELDS:
        raise ValueError(f"{path}: {data.size} floats, expected {count * FIELDS}")
    return data.reshape(count, FIELDS)


def length_from_q(q: np.ndarray) -> np.ndarray:
    return np.expm1(q * math.log1p(ENDPOINT_ALPHA * HORIZON)) / ENDPOINT_ALPHA


def normalized_coordinates(x: torch.Tensor) -> torch.Tensor:
    result = x[:, :6].clone()
    result[:, 4] = result[:, 4] * .5 + .5
    # Azimuth is a quotient coordinate at the two exact vertical poles.
    pole = torch.abs(x[:, 4]) >= (1 - 1e-7)
    result[:, 3] = torch.where(pole, torch.zeros_like(result[:, 3]), result[:, 3])
    return result


def normalized_coordinates_np(x: np.ndarray) -> np.ndarray:
    result = np.asarray(x[:, :6], dtype=np.float32).copy()
    result[:, 4] = result[:, 4] * .5 + .5
    result[np.abs(x[:, 4]) >= (1 - 1e-7), 3] = 0
    return result


def coordinate_features(x: torch.Tensor) -> torch.Tensor:
    frequencies = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), dtype=x.dtype, device=x.device)
    angles = x.unsqueeze(2) * frequencies.reshape(1, 1, -1) * (2 * math.pi)
    harmonic = torch.cat((torch.sin(angles), torch.cos(angles)), dim=2)
    raw = x.unsqueeze(2)
    periodic_mask = torch.tensor(tuple(i in PERIODIC for i in range(6)), device=x.device).reshape(1, 6, 1)
    # Periodic coordinates have no raw term, so the continuous generator and
    # the sampled repeat texture agree exactly at their seams.
    raw = torch.where(periodic_mask, torch.zeros_like(raw), raw * 2 - 1)
    return torch.cat((raw, harmonic), dim=2)


class PairPlaneGenerator(torch.nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.register_buffer("pair_axes", torch.tensor(PAIRS, dtype=torch.long))
        self.register_buffer("pair_identity", torch.eye(len(PAIRS)))
        self.plane = torch.nn.Sequential(
            torch.nn.Linear(17 * 2 + len(PAIRS), width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 4),
        )
        self.head = torch.nn.Linear(14, 8)
        torch.nn.init.zeros_(self.plane[-1].weight)
        torch.nn.init.zeros_(self.plane[-1].bias)
        torch.nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor((0., .45, .18, .33, .08, 0, .8, 0)))

    def factors_from_normalized(self, x: torch.Tensor) -> torch.Tensor:
        features = coordinate_features(x)
        pair = features[:, self.pair_axes].flatten(2)
        identity = self.pair_identity.to(dtype=x.dtype).unsqueeze(0).expand(len(x), -1, -1)
        raw = self.plane(torch.cat((pair, identity), dim=2))
        # Identity initialization; bounded factors prevent a fifteen-term
        # product from hiding instability behind overflow or underflow.
        return 1 + .5 * torch.tanh(raw)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        factors = self.factors_from_normalized(normalized_coordinates(x))
        z = torch.prod(factors, dim=1)
        lifted = torch.cat((z, z * z, z[:, (0, 0, 0, 1, 1, 2)] * z[:, (1, 2, 3, 2, 3, 3)]), dim=1)
        return self.head(lifted)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return decode(self.forward_raw(x), x[:, 5])


def decode(raw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    def sat(value: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(value, 0, 1)
        # The forward value is the exact runtime saturate.  The straight-through
        # gradient prevents an early background overshoot from permanently
        # killing a factor during offline optimisation.
        return value + (clipped - value).detach() if torch.is_grad_enabled() else clipped
    active = (q > 0).to(raw.dtype).unsqueeze(1)
    a = active * sat(raw[:, 0:1])
    m = a * sat(raw[:, 1:2])
    c = a * sat(raw[:, 2:5])
    n = raw[:, 5:8]
    n = n / torch.maximum(torch.linalg.vector_norm(n, dim=1, keepdim=True), torch.ones_like(a))
    return torch.cat((a, m, c, a * n), dim=1)


def decode_np(raw: np.ndarray, q: np.ndarray) -> np.ndarray:
    active = (q > 0).astype(np.float32)[:, None]
    a = active * np.clip(raw[:, 0:1], 0, 1)
    m = a * np.clip(raw[:, 1:2], 0, 1)
    c = a * np.clip(raw[:, 2:5], 0, 1)
    n = raw[:, 5:8]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1)
    return np.concatenate((a, m, c, a * n), axis=1).astype(np.float32)


def quantiles(value: np.ndarray) -> dict[str, float]:
    clean = np.asarray(value, dtype=np.float64).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {"p50": 0., "p95": 0., "p99": 0., "max": 0.}
    return {key: float(np.quantile(clean, q)) for key, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.))}


def metrics(data: np.ndarray, prediction: np.ndarray, raw: np.ndarray | None = None) -> dict[str, Any]:
    truth = data[:, 6:14]
    length = length_from_q(data[:, 5])
    a, pa = truth[:, 0], prediction[:, 0]
    m, pm = truth[:, 1], prediction[:, 1]
    mass = a >= .25
    M, pM = m * length, pm * length
    mu = np.divide(M, a, out=np.zeros_like(M), where=a > 1e-6)
    pmu = np.divide(pM, pa, out=np.zeros_like(pM), where=pa > 1e-6)
    hit_t, hit_p = a >= .125, pa >= .125
    union = np.count_nonzero(hit_t | hit_p)
    tn, pn = truth[mass, 5:8], prediction[mass, 5:8]
    tn = tn / np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-8)
    pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-8)
    angle = np.degrees(np.arccos(np.clip(np.sum(tn * pn, axis=1), -1, 1))) if np.any(mass) else np.zeros(0)
    plume = data[:, 15] >= .25
    result: dict[str, Any] = {
        "records": len(data), "truthCoverageMean": float(a.mean()),
        "AAbsoluteError": quantiles(np.abs(pa - a)),
        "coverageIoU": float(np.count_nonzero(hit_t & hit_p) / max(1, union)),
        "MAbsoluteErrorMetres": quantiles(np.abs(pM - M)),
        "conditionalDistanceAbsoluteErrorMetres": quantiles(np.abs(pmu[mass] - mu[mass])),
        "premultipliedColourRmse": float(np.sqrt(np.mean((prediction[mass, 2:5] - truth[mass, 2:5]) ** 2))) if np.any(mass) else 0.,
        "conditionalNormalAngularErrorDegrees": quantiles(angle),
        "plumeRecall": float(np.mean(pa[plume] >= .125)) if np.any(plume) else 1.,
        "plumePremultipliedColourRmse": float(np.sqrt(np.mean((prediction[plume, 2:5] - truth[plume, 2:5]) ** 2))) if np.any(plume) else 0.,
    }
    if raw is not None:
        result["rawSaturation"] = {
            "A": int(np.count_nonzero((raw[:, 0] < 0) | (raw[:, 0] > 1))),
            "mConditional": int(np.count_nonzero((raw[:, 1] < 0) | (raw[:, 1] > 1))),
            "colourConditional": int(np.count_nonzero((raw[:, 2:5] < 0) | (raw[:, 2:5] > 1))),
            "normalLengthAbove1": int(np.count_nonzero(np.linalg.norm(raw[:, 5:8], axis=1) > 1)),
        }
    return result


def gate_metric(value: dict[str, Any]) -> bool:
    return (
        value["AAbsoluteError"]["p95"] <= .10 and value["coverageIoU"] >= .90
        and value["MAbsoluteErrorMetres"]["p95"] <= .025
        and value["conditionalDistanceAbsoluteErrorMetres"]["p95"] <= .05
        and value["premultipliedColourRmse"] <= .06
        and value["conditionalNormalAngularErrorDegrees"]["p95"] <= 20
        and value["plumeRecall"] >= .95
    )


@torch.no_grad()
def predict_continuous(model: PairPlaneGenerator, data: np.ndarray, batch: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty((len(data), 8), dtype=np.float32)
    raw_result = np.empty_like(result)
    model.eval()
    for start in range(0, len(data), batch):
        x = torch.from_numpy(np.asarray(data[start:start + batch, :6], dtype=np.float32)).to(device)
        raw = model.forward_raw(x)
        raw_result[start:start + len(x)] = raw.cpu().numpy()
        result[start:start + len(x)] = decode(raw, x[:, 5]).cpu().numpy()
    return result, raw_result


@torch.no_grad()
def sample_level_zero(model: PairPlaneGenerator, device: torch.device, batch: int) -> np.ndarray:
    values = np.empty((len(PAIRS), RESOLUTION, RESOLUTION, 4), dtype=np.float16)
    model.eval()
    for layer, _pair in enumerate(PAIRS):
        identity = model.pair_identity[layer:layer + 1].to(device=device)
        chunks: list[np.ndarray] = []
        for row_start in range(0, RESOLUTION, max(1, batch // RESOLUTION)):
            rows = min(max(1, batch // RESOLUTION), RESOLUTION - row_start)
            s = np.tile((np.arange(RESOLUTION, dtype=np.float32) + .5) / RESOLUTION, rows)
            t = np.repeat((np.arange(row_start, row_start + rows, dtype=np.float32) + .5) / RESOLUTION, RESOLUTION)
            x = torch.from_numpy(np.stack((s, t), axis=1)).to(device)
            # Build precisely the one layer's pair input; no unused coordinate
            # values can influence its table.
            feature = coordinate_features_for_axes(x, PAIRS[layer], device)
            one_hot = identity.to(dtype=x.dtype).expand(len(x), -1)
            raw = model.plane(torch.cat((feature, one_hot), dim=1))
            chunks.append((1 + .5 * torch.tanh(raw)).cpu().numpy())
        values[layer] = np.concatenate(chunks).reshape(RESOLUTION, RESOLUTION, 4).astype(np.float16)
        print(f"[pair-plane-fit] sampled f16 layer {layer + 1:02d}/15 {COORD_NAMES[PAIRS[layer][0]]},{COORD_NAMES[PAIRS[layer][1]]}", flush=True)
    return values


def coordinate_features_for_axes(x: torch.Tensor, axes: tuple[int, int], device: torch.device) -> torch.Tensor:
    frequencies = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), dtype=x.dtype, device=device)
    angles = x.unsqueeze(2) * frequencies.reshape(1, 1, -1) * (2 * math.pi)
    raw = x.unsqueeze(2) * 2 - 1
    for local, axis in enumerate(axes):
        if axis in PERIODIC:
            raw[:, local] = 0
    return torch.cat((raw, torch.sin(angles), torch.cos(angles)), dim=2).flatten(1)


def bilinear_layers(tables: np.ndarray, data: np.ndarray) -> np.ndarray:
    coords = normalized_coordinates_np(data)
    factors = np.empty((len(data), len(PAIRS), 4), dtype=np.float32)
    for layer, (axis_s, axis_t) in enumerate(PAIRS):
        s, t = coords[:, axis_s], coords[:, axis_t]
        sx, ty = s * RESOLUTION - .5, t * RESOLUTION - .5
        x0, y0 = np.floor(sx).astype(np.int64), np.floor(ty).astype(np.int64)
        fx, fy = (sx - x0).astype(np.float32), (ty - y0).astype(np.float32)
        def indices(base: np.ndarray, axis: int, plus: int) -> np.ndarray:
            value = base + plus
            return np.mod(value, RESOLUTION) if axis in PERIODIC else np.clip(value, 0, RESOLUTION - 1)
        ix0, ix1 = indices(x0, axis_s, 0), indices(x0, axis_s, 1)
        iy0, iy1 = indices(y0, axis_t, 0), indices(y0, axis_t, 1)
        table = tables[layer].astype(np.float32, copy=False)
        a, b, c, d = table[iy0, ix0], table[iy0, ix1], table[iy1, ix0], table[iy1, ix1]
        top = a + (b - a) * fx[:, None]
        bottom = c + (d - c) * fx[:, None]
        factors[:, layer] = top + (bottom - top) * fy[:, None]
    return factors


def predict_tables(tables: np.ndarray, head_weight: np.ndarray, head_bias: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.prod(bilinear_layers(tables, data), axis=1)
    lift = np.concatenate((z, z * z, z[:, (0, 0, 0, 1, 1, 2)] * z[:, (1, 2, 3, 2, 3, 3)]), axis=1)
    raw = lift @ head_weight.T + head_bias
    return decode_np(raw, data[:, 5]), raw.astype(np.float32)


def path_records(path: Path) -> np.ndarray:
    source = np.fromfile(path, dtype="<f4").reshape(-1, 27)
    direction = source[:, 6:9]
    result = np.zeros((len(source), FIELDS), dtype=np.float32)
    result[:, 0] = source[:, 3]; result[:, 1] = source[:, 5]; result[:, 2] = source[:, 4]
    result[:, 3] = np.mod(np.arctan2(direction[:, 2], direction[:, 0]) / (2 * math.pi), 1)
    result[:, 4] = np.arcsin(np.clip(direction[:, 1], -1, 1)) / (math.pi * .5)
    result[:, 5] = np.log1p(ENDPOINT_ALPHA * source[:, 14]) / math.log1p(ENDPOINT_ALPHA * HORIZON)
    result[:, 6:14] = source[:, 15:23]
    result[:, 14] = source[:, 25]; result[:, 15] = source[:, 26]
    return result


def image_plane(values: np.ndarray, scale: int = 4) -> Image.Image:
    rgb = np.clip(np.power(np.clip(values, 0, 1), 1 / 2.2) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)


def write_path_qa(data: np.ndarray, continuous: np.ndarray, quantized: np.ndarray | None, output: Path) -> None:
    grid, frames = 64, 17
    selected = (0, frames // 2, frames - 1)
    truth = data[:, 6:14].reshape(frames, grid, grid, 8)
    sources = [("truth", truth), ("continuous", continuous.reshape(frames, grid, grid, 8))]
    if quantized is not None:
        sources.append(("f16 tables + hardware-shaped bilinear", quantized.reshape(frames, grid, grid, 8)))
    rows: list[Image.Image] = []
    for label, source in sources:
        row = Image.new("RGB", (768, 282), (20, 20, 20))
        ImageDraw.Draw(row).text((6, 5), f"18-degree path {label}: first / middle / last", fill=(240, 240, 240))
        for index, frame in enumerate(selected):
            row.paste(image_plane(source[frame, :, :, 2:5]), (index * 256, 26))
        rows.append(row)
    canvas = Image.new("RGB", (768, 282 * len(rows)), (20, 20, 20))
    for index, row in enumerate(rows):
        canvas.paste(row, (0, index * 282))
    canvas.save(output)


def sequence_gates(data: dict[str, np.ndarray], predictions: dict[str, np.ndarray], shifts: np.ndarray) -> dict[str, Any]:
    mono = predictions["monotonic"].reshape(-1, 8, 8)
    mono_truth = data["monotonic"].reshape(-1, 8, FIELDS)
    mono_L = length_from_q(mono_truth[:, :, 5])
    mono_M = mono[:, :, 1] * mono_L
    censor_truth = data["censor"].reshape(-1, 2, FIELDS)
    censor_pred = predictions["censor"].reshape(-1, 2, 8)
    censor_L = length_from_q(censor_truth[:, :, 5])
    vertical_pred = predictions["vertical"].reshape(-1, 2, 8)
    vertical_truth = data["vertical"].reshape(-1, 2, FIELDS)
    return {
        "monotonic": {
            "ADecreaseFractionOver0_01": float(np.mean(np.diff(mono[:, :, 0], axis=1) < -.01)),
            "MDecreaseFractionOver0_005m": float(np.mean(np.diff(mono_M, axis=1) < -.005)),
        },
        "censor": {
            "AChangeError": quantiles(np.abs(np.diff(censor_pred[:, :, 0], axis=1)[:, 0] - np.diff(censor_truth[:, :, 6], axis=1)[:, 0])),
            "MChangeErrorMetres": quantiles(np.abs(np.diff(censor_pred[:, :, 1] * censor_L, axis=1)[:, 0] - np.diff(censor_truth[:, :, 7] * censor_L, axis=1)[:, 0])),
            "shiftMetres": quantiles(shifts),
        },
        "verticalPhi": {
            "truthMax": float(np.max(np.abs(vertical_truth[:, 1, 6:14] - vertical_truth[:, 0, 6:14]))),
            "predictionAbsoluteError": quantiles(np.abs(vertical_pred[:, 1] - vertical_pred[:, 0])),
        },
    }


def all_gates(evaluations: dict[str, Any], sequence: dict[str, Any], path: dict[str, Any]) -> bool:
    checked = ("validation", "holdout-u", "holdout-v", "holdout-y", "holdout-phi", "holdout-elevation", "holdout-endpointQ", "horizontal", "vertical")
    return (
        all(gate_metric(evaluations[name]) for name in checked) and gate_metric(path)
        and sequence["monotonic"]["ADecreaseFractionOver0_01"] <= .005
        and sequence["monotonic"]["MDecreaseFractionOver0_005m"] <= .005
        and sequence["censor"]["AChangeError"]["p95"] <= .05
        and sequence["censor"]["MChangeErrorMetres"]["p95"] <= .025
        and sequence["verticalPhi"]["truthMax"] <= 1e-6
        and sequence["verticalPhi"]["predictionAbsoluteError"]["max"] <= 1e-6
    )


def write_mips(level_zero: np.ndarray, output: Path) -> dict[str, Any]:
    path = output / "pair-planes-512x512x15-rgba16f-mips.bin"
    levels: list[dict[str, Any]] = []
    offset = 0
    with path.open("wb") as stream:
        current = level_zero
        while True:
            payload = np.asarray(current, dtype="<f2").tobytes(order="C")
            stream.write(payload)
            levels.append({"resolution": int(current.shape[1]), "offset": offset, "bytes": len(payload)})
            offset += len(payload)
            if current.shape[1] == 1:
                break
            f32 = current.astype(np.float32)
            current = ((f32[:, 0::2, 0::2] + f32[:, 1::2, 0::2] + f32[:, 0::2, 1::2] + f32[:, 1::2, 1::2]) * .25).astype(np.float16)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "levels": levels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2207262)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    manifest_path = args.manifest.resolve(); manifest = json.loads(manifest_path.read_text()); root = manifest_path.parent
    data: dict[str, np.ndarray] = {}
    for name, definition in manifest["files"].items():
        path = root / definition["path"]
        if digest(path) != definition["sha256"]:
            raise ValueError(f"hash mismatch: {name}")
        data[name] = records(path, definition["records"])
    shift_path = root / manifest["censorShifts"]["path"]
    if digest(shift_path) != manifest["censorShifts"]["sha256"]:
        raise ValueError("censor shifts hash mismatch")
    shifts = np.fromfile(shift_path, dtype="<f4")

    model = PairPlaneGenerator(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    train = data["train"]
    curve: list[dict[str, float]] = []
    print(f"[pair-plane-fit] train={len(train)} device={device} epochs={args.epochs} width={args.width}", flush=True)
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        total = np.zeros(5, dtype=np.float64)
        model.train()
        for start in range(0, len(train), args.batch_size):
            batch_np = train[order[start:start + args.batch_size]]
            x = torch.from_numpy(np.asarray(batch_np[:, :6], dtype=np.float32)).to(device)
            truth = torch.from_numpy(np.asarray(batch_np[:, 6:14], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            hit_weight = 1 + 12 * truth[:, 0]
            a_loss = (torch.nn.functional.smooth_l1_loss(out[:, 0], truth[:, 0], beta=.05, reduction="none") * hit_weight).mean()
            metres = torch.from_numpy(length_from_q(batch_np[:, 5]).astype(np.float32)).to(device)
            m_loss = (torch.nn.functional.smooth_l1_loss(out[:, 1] * metres, truth[:, 1] * metres, beta=.01, reduction="none") * hit_weight).mean()
            c_loss = (torch.nn.functional.smooth_l1_loss(out[:, 2:5], truth[:, 2:5], beta=.02, reduction="none").mean(1) * hit_weight).mean()
            n_loss = (torch.nn.functional.smooth_l1_loss(out[:, 5:8], truth[:, 5:8], beta=.02, reduction="none").mean(1) * hit_weight).mean()
            loss = 3 * a_loss + 5 * m_loss + 2 * c_loss + n_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            count = len(x); total += np.array(tuple(float(item.detach()) for item in (loss, a_loss, m_loss, c_loss, n_loss))) * count
        entry = {"epoch": epoch + 1, **dict(zip(("loss", "A", "M", "C", "N"), (total / len(train)).tolist()))}
        curve.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"[pair-plane-fit] epoch {epoch + 1:03d} loss={entry['loss']:.6f} A={entry['A']:.6f}", flush=True)

    continuous_prediction: dict[str, np.ndarray] = {}
    continuous_raw: dict[str, np.ndarray] = {}
    for name, value in data.items():
        continuous_prediction[name], continuous_raw[name] = predict_continuous(model, value, args.batch_size, device)
    continuous_eval = {name: metrics(data[name], continuous_prediction[name], continuous_raw[name]) for name in data if name not in ("monotonic", "censor")}
    continuous_sequence = sequence_gates(data, continuous_prediction, shifts)
    path_truth = Path("data/work/groundcover-hybrid-transfer/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/63a512106a67f5cc44b90d655b178f09f50e36cd2a28488c9c47d30d5a238554/truth/camera-path.f32").resolve()
    path_data = path_records(path_truth)
    path_continuous, path_raw = predict_continuous(model, path_data, args.batch_size, device)
    path_continuous_metrics = metrics(path_data, path_continuous, path_raw)
    continuous_pass = all_gates(continuous_eval, continuous_sequence, path_continuous_metrics)
    print(f"[pair-plane-fit] continuous pass={continuous_pass} validation IoU={continuous_eval['validation']['coverageIoU']:.4f} A.p95={continuous_eval['validation']['AAbsoluteError']['p95']:.4f} path IoU={path_continuous_metrics['coverageIoU']:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True); qa = args.output / "qa"; qa.mkdir(exist_ok=True)
    model_path = args.output / "offline-continuous-generator.pt"; torch.save(model.state_dict(), model_path)
    table_artifact: dict[str, Any] | None = None
    quantized_eval: dict[str, Any] | None = None
    quantized_sequence: dict[str, Any] | None = None
    path_quantized: np.ndarray | None = None
    path_quantized_metrics: dict[str, Any] | None = None
    quantized_pass = False
    if continuous_pass:
        level_zero = sample_level_zero(model, device, args.batch_size)
        head_weight = model.head.weight.detach().cpu().numpy().astype(np.float32)
        head_bias = model.head.bias.detach().cpu().numpy().astype(np.float32)
        quantized_prediction: dict[str, np.ndarray] = {}
        quantized_raw: dict[str, np.ndarray] = {}
        for name, value in data.items():
            quantized_prediction[name], quantized_raw[name] = predict_tables(level_zero, head_weight, head_bias, value)
        quantized_eval = {name: metrics(data[name], quantized_prediction[name], quantized_raw[name]) for name in data if name not in ("monotonic", "censor")}
        quantized_sequence = sequence_gates(data, quantized_prediction, shifts)
        path_quantized, path_quantized_raw = predict_tables(level_zero, head_weight, head_bias, path_data)
        path_quantized_metrics = metrics(path_data, path_quantized, path_quantized_raw)
        quantized_pass = all_gates(quantized_eval, quantized_sequence, path_quantized_metrics)
        table_artifact = write_mips(level_zero, args.output)
        head_path = args.output / "quadratic-head-14x8-plus-bias-f32.bin"
        np.concatenate((head_weight.reshape(-1), head_bias)).astype("<f4").tofile(head_path)
        table_artifact["head"] = {"path": head_path.name, "bytes": head_path.stat().st_size, "sha256": digest(head_path)}
        print(f"[pair-plane-fit] quantized pass={quantized_pass} validation IoU={quantized_eval['validation']['coverageIoU']:.4f} path IoU={path_quantized_metrics['coverageIoU']:.4f}", flush=True)

    path_qa = qa / "000-18-degree-camera-path-truth-continuous-quantized.png"
    write_path_qa(path_data, path_continuous, path_quantized, path_qa)
    qa_entries = [{"path": path_qa.name, "sha256": digest(path_qa)}]
    decision = "quantized-metric-pass-needs-visual-review" if quantized_pass else ("reject-quantized" if continuous_pass else "reject-continuous")
    report = {
        "schema": "laas-endpoint-pair-plane-fit/v1", "decision": decision,
        "truth": {"manifest": str(manifest_path), "manifestSha256": digest(manifest_path), "cameraPath": str(path_truth), "cameraPathSha256": digest(path_truth)},
        "layout": {
            "pairs": [[COORD_NAMES[a], COORD_NAMES[b]] for a, b in PAIRS], "resolution": RESOLUTION,
            "format": "RGBA16F", "layers": 15, "completeMipBytes": 41943000, "headBytes": 512,
            "residentBytes": 41943512, "filteredSamples": 15, "decoderFmaEquivalentBeforePhysicalDecode": 178,
            "runtimeLoopsMarchesPasses": 0,
        },
        "training": {"epochs": args.epochs, "batchSize": args.batch_size, "width": args.width, "seed": args.seed, "device": str(device), "curve": curve, "offlineGeneratorOnly": True},
        "continuous": {"pass": continuous_pass, "splits": continuous_eval, "sequence": continuous_sequence, "cameraPath18Degrees": path_continuous_metrics},
        "quantizedHardwareShaped": None if quantized_eval is None else {"pass": quantized_pass, "filter": "f16 level-zero WebGPU-normalized-coordinate bilinear with declared wrap/clamp", "splits": quantized_eval, "sequence": quantized_sequence, "cameraPath18Degrees": path_quantized_metrics},
        "thresholds": {"AErrorP95": .10, "coverageIoU": .90, "MErrorP95Metres": .025, "conditionalDistanceP95Metres": .05, "colourRmse": .06, "normalP95Degrees": 20, "plumeRecall": .95},
        "artifacts": {"offlineModel": {"path": model_path.name, "bytes": model_path.stat().st_size, "sha256": digest(model_path)}, "runtimeShaped": table_artifact, "qa": qa_entries},
        "runtimeShaderEdits": 0,
        "visualGate": "metric pass is not acceptance; reject on any widening fan, wrong-view sheet, distance band, or panicle disappearance",
        "sources": [
            {"title": "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance", "url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf", "borrowed": "all d-choose-2 bilinear planes combined by Hadamard product"},
            {"title": "HexPlane: A Fast Representation for Dynamic Scenes", "url": "https://arxiv.org/abs/2301.09632", "borrowed": "adjacent compact plane-factor precedent only"},
        ],
        "novelLocalHypotheses": ["pointed finite-ray six-coordinate transfer", "all-fifteen 512-square RGBA16F array packing", "rank-four product plus quadratic fourteen-term head", "premultiplied botanical moment decode", "endpoint/censor/pole/panicle/oblique falsification gates"],
        "command": ["uv", "run", "--project", "asset-gen", "--extra", "microtopography-ml", "python", *sys.argv],
    }
    report_path = args.output / "report.json"; report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": decision, "report": str(report_path), "continuousValidation": continuous_eval["validation"], "continuousPath": path_continuous_metrics, "quantizedValidation": None if quantized_eval is None else quantized_eval["validation"], "quantizedPath": path_quantized_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

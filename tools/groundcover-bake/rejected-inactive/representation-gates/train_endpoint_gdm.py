"""Train and falsify the frozen rank-8 endpoint-conditioned GDM transfer field."""

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
RANK = 8
HORIZON = 155.0
ENDPOINT_ALPHA = 10.0
W_DIMS = (212, 212, 48)
E_DIMS = (64, 193, 20)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path, count: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size != count * FIELDS:
        raise ValueError(f"{path}: {data.size} floats, expected {count * FIELDS}")
    return data.reshape(count, FIELDS)


def length_from_q(q: np.ndarray) -> np.ndarray:
    return np.expm1(q * math.log1p(ENDPOINT_ALPHA * HORIZON)) / ENDPOINT_ALPHA


def quantiles(value: np.ndarray) -> dict[str, float]:
    clean = np.asarray(value, dtype=np.float64).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {key: float(np.quantile(clean, q)) for key, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.0))}


def fourier(x: torch.Tensor, frequencies: tuple[int, ...]) -> torch.Tensor:
    angles = x.unsqueeze(1) * (2 * math.pi) * torch.tensor(frequencies, device=x.device, dtype=x.dtype).unsqueeze(0)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class EndpointGdm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        w_features = (8 + 8 + 7) * 2
        e_features = 1 + 6 * 2 + 1 + 3 + 1 + 5 * 2
        self.w = torch.nn.Sequential(
            torch.nn.Linear(w_features, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, RANK),
        )
        self.e = torch.nn.Sequential(
            torch.nn.Linear(e_features, 96), torch.nn.SiLU(),
            torch.nn.Linear(96, 96), torch.nn.SiLU(), torch.nn.Linear(96, RANK),
        )
        self.head = torch.nn.Linear(RANK, 8)

    def factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u, v, y, phi, elevation, q = (x[:, i] for i in range(6))
        wf = torch.cat((
            fourier(u, (1, 2, 4, 8, 16, 32, 64, 106)),
            fourier(v, (1, 2, 4, 8, 16, 32, 64, 106)),
            fourier(phi, (1, 2, 4, 8, 12, 16, 24)),
        ), dim=1)
        theta = elevation * (math.pi * .5)
        ef = torch.cat((
            y.unsqueeze(1), fourier(y, (1, 2, 4, 8, 16, 32)),
            elevation.unsqueeze(1), torch.sin(theta).unsqueeze(1), torch.cos(theta).unsqueeze(1),
            torch.sign(elevation).unsqueeze(1), q.unsqueeze(1), fourier(q, (1, 2, 4, 8, 16)),
        ), dim=1)
        return self.w(wf), self.e(ef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, e = self.factors(x)
        return self.head(w * e)


def decode(raw: torch.Tensor) -> torch.Tensor:
    a = torch.sigmoid(raw[:, 0:1])
    m = a * torch.sigmoid(raw[:, 1:2])
    c = a * torch.sigmoid(raw[:, 2:5])
    direction = torch.tanh(raw[:, 5:8])
    direction = direction / torch.maximum(
        torch.linalg.vector_norm(direction, dim=1, keepdim=True),
        torch.ones((len(raw), 1), dtype=raw.dtype, device=raw.device),
    )
    n = a * direction
    return torch.cat((a, m, c, n), dim=1)


@torch.no_grad()
def predict(model: EndpointGdm, data: np.ndarray, batch: int, device: torch.device) -> np.ndarray:
    result = np.empty((len(data), 8), dtype=np.float32)
    model.eval()
    for start in range(0, len(data), batch):
        x = torch.from_numpy(np.asarray(data[start:start + batch, :6], dtype=np.float32)).to(device)
        result[start:start + len(x)] = decode(model(x)).cpu().numpy()
    return result


def metrics(data: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = data[:, 6:14]
    length = length_from_q(data[:, 5])
    a, pa = truth[:, 0], prediction[:, 0]
    m, pm = truth[:, 1], prediction[:, 1]
    mass = a >= .25
    M, pM = m * length, pm * length
    mu = np.divide(M, a, out=np.zeros_like(M), where=a > 1e-6)
    pmu = np.divide(pM, pa, out=np.zeros_like(pM), where=pa > 1e-6)
    hit_t = a >= .125
    hit_p = pa >= .125
    union = np.count_nonzero(hit_t | hit_p)
    c_rmse = float(np.sqrt(np.mean((prediction[mass, 2:5] - truth[mass, 2:5]) ** 2))) if np.any(mass) else 0.0
    tn = truth[mass, 5:8]
    pn = prediction[mass, 5:8]
    tn = tn / np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-8)
    pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-8)
    angle = np.degrees(np.arccos(np.clip(np.sum(tn * pn, axis=1), -1, 1))) if np.any(mass) else np.zeros(0)
    plume = data[:, 15] >= .25
    return {
        "records": len(data), "truthCoverageMean": float(a.mean()),
        "AAbsoluteError": quantiles(np.abs(pa - a)),
        "coverageIoU": float(np.count_nonzero(hit_t & hit_p) / max(1, union)),
        "MAbsoluteErrorMetres": quantiles(np.abs(pM - M)),
        "conditionalDistanceAbsoluteErrorMetres": quantiles(np.abs(pmu[mass] - mu[mass])),
        "premultipliedColourRmse": c_rmse,
        "conditionalNormalAngularErrorDegrees": quantiles(angle),
        "plumeRecall": float(np.mean(pa[plume] >= .125)) if np.any(plume) else 1.0,
        "plumePremultipliedColourRmse": float(np.sqrt(np.mean((prediction[plume, 2:5] - truth[plume, 2:5]) ** 2))) if np.any(plume) else 0.0,
        "physicality": {
            "AOutside01": int(np.count_nonzero((pa < 0) | (pa > 1))),
            "mOutside0A": int(np.count_nonzero((pm < 0) | (pm > pa))),
            "colourOutside0A": int(np.count_nonzero((prediction[:, 2:5] < 0) | (prediction[:, 2:5] > pa[:, None]))),
            "normalMomentMagnitudeAboveA": int(np.count_nonzero(np.linalg.norm(prediction[:, 5:8], axis=1) > pa + 1e-6)),
        },
    }


def gate_metric(value: dict[str, Any]) -> bool:
    return (
        value["AAbsoluteError"]["p95"] <= .10
        and value["coverageIoU"] >= .90
        and value["MAbsoluteErrorMetres"]["p95"] <= .025
        and value["conditionalDistanceAbsoluteErrorMetres"]["p95"] <= .05
        and value["premultipliedColourRmse"] <= .06
        and value["conditionalNormalAngularErrorDegrees"]["p95"] <= 20
        and value["plumeRecall"] >= .95
    )


def image_panel(data: np.ndarray, pred: np.ndarray, title: str, width: int = 64) -> Image.Image:
    count = min(len(data), width * width)
    truth = data[:count, 6:14]
    p = pred[:count]
    height = math.ceil(count / width)
    def plane(values: np.ndarray) -> Image.Image:
        padded = np.zeros((height * width, 3), dtype=np.uint8)
        padded[:count] = np.clip(np.power(np.clip(values, 0, 1), 1 / 2.2) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(padded.reshape(height, width, 3)).resize((width * 4, height * 4), Image.Resampling.NEAREST)
    tc = truth[:, 2:5]
    pc = p[:, 2:5]
    error = np.zeros_like(tc)
    error[:, 0] = np.clip(np.abs(p[:, 0] - truth[:, 0]) * 4, 0, 1)
    error[:, 1] = np.clip(np.abs(p[:, 1] - truth[:, 1]) * 4, 0, 1)
    panels = [(plane(tc), "truth"), (plane(pc), "rank8"), (plane(error), "red=A error, green=m error")]
    canvas = Image.new("RGB", (sum(im.width for im, _ in panels), panels[0][0].height + 28), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in panels:
        canvas.paste(im, (x, 28)); draw.text((x + 5, 5), f"{title}: {label}", fill=(240, 240, 240)); x += im.width
    return canvas


def rank8_oracle(data: np.ndarray, rows: int, columns: int) -> tuple[dict[str, Any], np.ndarray]:
    if len(data) != rows * columns:
        raise ValueError("oracle matrix shape mismatch")
    A = data[:, 6].reshape(rows, columns).astype(np.float64)
    L = length_from_q(data[:, 5]).reshape(rows, columns)
    M = data[:, 7].reshape(rows, columns).astype(np.float64) * L
    reconstructions = []
    singular: dict[str, Any] = {}
    for label, matrix in (("A", A), ("M", M)):
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        reconstructed = (u[:, :RANK] * s[:RANK]) @ vh[:RANK]
        reconstructions.append(reconstructed)
        singular[label] = {
            "first16": s[:16].tolist(),
            "rank8EnergyFraction": float(np.sum(s[:RANK] ** 2) / max(1e-30, np.sum(s ** 2))),
        }
    pA, pM = reconstructions
    truth_hit = A >= .125; predicted_hit = pA >= .125
    union = np.count_nonzero(truth_hit | predicted_hit)
    plume = data[:, 15].reshape(rows, columns) >= .25
    mu = np.divide(M, A, out=np.zeros_like(M), where=A > 1e-8)
    pmu = np.divide(pM, pA, out=np.zeros_like(pM), where=np.abs(pA) > 1e-8)
    mass = A >= .25
    result = {
        "contract": "independent best Frobenius rank-8 SVD for A and M on fixed W rows x E columns; this is optimistic relative to shared W/E factors and one head",
        "shape": [rows, columns], "singularValues": singular,
        "AAbsoluteError": quantiles(np.abs(pA - A)),
        "coverageIoU": float(np.count_nonzero(truth_hit & predicted_hit) / max(1, union)),
        "MAbsoluteErrorMetres": quantiles(np.abs(pM - M)),
        "conditionalDistanceAbsoluteErrorMetres": quantiles(np.abs(pmu[mass] - mu[mass])),
        "plumeRecall": float(np.mean(predicted_hit[plume])) if np.any(plume) else 1.0,
        "rawPhysicality": {
            "AOutside01": int(np.count_nonzero((pA < 0) | (pA > 1))),
            "MNegative": int(np.count_nonzero(pM < 0)),
            "MAboveLA": int(np.count_nonzero(pM > L * pA)),
            "note": "metrics use raw SVD values; no clamp hides rank error",
        },
    }
    prediction = np.zeros((len(data), 8), dtype=np.float32)
    prediction[:, 0] = pA.reshape(-1).astype(np.float32)
    prediction[:, 1] = np.divide(pM, L, out=np.zeros_like(pM), where=L > 1e-8).reshape(-1).astype(np.float32)
    return result, prediction


def oracle_image(data: np.ndarray, prediction: np.ndarray, rows: int, columns: int) -> Image.Image:
    truth = data[:, 6].reshape(rows, columns)
    pred = prediction[:, 0].reshape(rows, columns)
    error = np.abs(pred - truth)
    images = []
    for label, values in (("truth A", truth), ("raw best rank8 A (display-clipped only)", pred), ("absolute error", error)):
        rgb = np.repeat((np.clip(values, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)
        image = Image.fromarray(rgb).resize((columns * 2, rows * 2), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (image.width, image.height + 26), (20, 20, 20)); panel.paste(image, (0, 26)); ImageDraw.Draw(panel).text((5, 5), label, fill=(240, 240, 240)); images.append(panel)
    canvas = Image.new("RGB", (sum(item.width for item in images), images[0].height), (20, 20, 20)); x = 0
    for item in images: canvas.paste(item, (x, 0)); x += item.width
    return canvas


def path_records(path: Path) -> np.ndarray:
    source = np.fromfile(path, dtype="<f4").reshape(-1, 27)
    d = source[:, 6:9]
    result = np.zeros((len(source), FIELDS), dtype=np.float32)
    result[:, 0] = source[:, 3]; result[:, 1] = source[:, 5]; result[:, 2] = source[:, 4]
    result[:, 3] = np.mod(np.arctan2(d[:, 2], d[:, 0]) / (2 * math.pi), 1)
    result[:, 4] = np.arcsin(np.clip(d[:, 1], -1, 1)) / (math.pi * .5)
    result[:, 5] = np.log1p(ENDPOINT_ALPHA * source[:, 14]) / math.log1p(ENDPOINT_ALPHA * HORIZON)
    result[:, 6:14] = source[:, 15:23]
    result[:, 14] = source[:, 25]; result[:, 15] = source[:, 26]
    return result


def write_path_qa(data: np.ndarray, prediction: np.ndarray, output: Path) -> None:
    grid, frames = 64, 17
    selected = [0, frames // 2, frames - 1]
    truth = data[:, 6:14].reshape(frames, grid, grid, 8)
    pred = prediction.reshape(frames, grid, grid, 8)
    rows = []
    for label, source in (("truth", truth), ("rank8", pred)):
        strips = []
        for frame in selected:
            rgb = np.clip(np.power(np.clip(source[frame, :, :, 2:5], 0, 1), 1 / 2.2) * 255, 0, 255).astype(np.uint8)
            strips.append(Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST))
        row = Image.new("RGB", (768, 282), (20, 20, 20)); ImageDraw.Draw(row).text((6, 5), f"18-degree path {label}: first / middle / last", fill=(240, 240, 240))
        for i, im in enumerate(strips): row.paste(im, (i * 256, 26))
        rows.append(row)
    canvas = Image.new("RGB", (768, 564)); canvas.paste(rows[0], (0, 0)); canvas.paste(rows[1], (0, 282)); canvas.save(output)


@torch.no_grad()
def sample_tables(model: EndpointGdm, output: Path, device: torch.device, batch: int) -> dict[str, Any]:
    # Only reached after the continuous gate passes.
    wu, wv, wp = W_DIMS
    w_values = np.empty((wp, wv, wu, RANK), dtype=np.float16)
    model.eval()
    for layer in range(wp):
        uv = np.stack(np.meshgrid((np.arange(wu) + .5) / wu, (np.arange(wv) + .5) / wv), axis=-1).reshape(-1, 2)
        x = np.zeros((len(uv), 6), dtype=np.float32); x[:, :2] = uv; x[:, 3] = layer / wp
        values = []
        for start in range(0, len(x), batch): values.append(model.factors(torch.from_numpy(x[start:start + batch]).to(device))[0].cpu().numpy())
        w_values[layer] = np.concatenate(values).reshape(wv, wu, RANK).astype(np.float16)
    ey, ee, el = E_DIMS
    coords = np.stack(np.meshgrid((np.arange(ey) + .5) / ey, np.linspace(-1, 1, ee), (np.arange(el) + .5) / el, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.zeros((len(coords), 6), dtype=np.float32); x[:, 2] = coords[:, 0]; x[:, 4] = coords[:, 1]; x[:, 5] = coords[:, 2]
    values = []
    for start in range(0, len(x), batch): values.append(model.factors(torch.from_numpy(x[start:start + batch]).to(device))[1].cpu().numpy())
    e_values = np.concatenate(values).reshape(ey, ee, el, RANK).astype(np.float16)
    w_path = output / "W-212x212x48-rank8-f16.bin"; w_values.tofile(w_path)
    e_path = output / "E-64x193x20-rank8-f16.bin"; e_values.tofile(e_path)
    return {"W": {"path": w_path.name, "bytes": w_path.stat().st_size, "sha256": digest(w_path)}, "E": {"path": e_path.name, "bytes": e_path.stat().st_size, "sha256": digest(e_path)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2207261)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    manifest_path = args.manifest.resolve(); manifest = json.loads(manifest_path.read_text()); root = manifest_path.parent
    data: dict[str, np.ndarray] = {}
    for name, definition in manifest["files"].items():
        path = root / definition["path"]
        if digest(path) != definition["sha256"]: raise ValueError(f"hash mismatch: {name}")
        data[name] = records(path, definition["records"])
    shifts_path = root / manifest["censorShifts"]["path"]
    if digest(shifts_path) != manifest["censorShifts"]["sha256"]: raise ValueError("censor shifts hash mismatch")
    shifts = np.fromfile(shifts_path, dtype="<f4")
    oracle_metrics, oracle_prediction = rank8_oracle(
        data["oracle"], int(manifest["oracle"]["WRows"]), int(manifest["oracle"]["EColumns"])
    )
    oracle_pass = (
        oracle_metrics["AAbsoluteError"]["p95"] <= .10
        and oracle_metrics["coverageIoU"] >= .90
        and oracle_metrics["MAbsoluteErrorMetres"]["p95"] <= .025
        and oracle_metrics["conditionalDistanceAbsoluteErrorMetres"]["p95"] <= .05
        and oracle_metrics["plumeRecall"] >= .95
        and all(value == 0 for key, value in oracle_metrics["rawPhysicality"].items() if key != "note")
    )
    print(f"[endpoint-gdm-fit] optimistic independent rank8 SVD oracle pass={oracle_pass} IoU={oracle_metrics['coverageIoU']:.4f} A.p95={oracle_metrics['AAbsoluteError']['p95']:.4f} M.p95={oracle_metrics['MAbsoluteErrorMetres']['p95']:.4f}m")
    device = torch.device("cpu")
    model = EndpointGdm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    train = data["train"]
    curve = []
    print(f"[endpoint-gdm-fit] rank=8 train={len(train)} device={device}")
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        totals = np.zeros(5, dtype=np.float64)
        model.train()
        for start in range(0, len(train), args.batch_size):
            batch_np = train[order[start:start + args.batch_size]]
            x = torch.from_numpy(np.asarray(batch_np[:, :6], dtype=np.float32)).to(device)
            truth = torch.from_numpy(np.asarray(batch_np[:, 6:14], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            out = decode(model(x))
            a_loss = torch.nn.functional.binary_cross_entropy(out[:, 0], truth[:, 0])
            M_scale = torch.from_numpy(length_from_q(batch_np[:, 5]).astype(np.float32)).to(device)
            m_loss = torch.nn.functional.smooth_l1_loss(out[:, 1] * M_scale, truth[:, 1] * M_scale, beta=.01)
            c_loss = torch.nn.functional.smooth_l1_loss(out[:, 2:5], truth[:, 2:5], beta=.02)
            n_loss = torch.nn.functional.smooth_l1_loss(out[:, 5:8], truth[:, 5:8], beta=.02)
            x_long = x.clone(); x_long[:, 5] = torch.clamp(x_long[:, 5] + .04, max=1)
            out_long = decode(model(x_long))
            monotonic = torch.relu(out[:, 0] - out_long[:, 0]).mean()
            loss = a_loss + 5 * m_loss + 2 * c_loss + n_loss + 2 * monotonic
            loss.backward(); optimizer.step()
            count = len(x); totals += np.array([float(loss), float(a_loss), float(m_loss), float(c_loss), float(n_loss)]) * count
        entry = {"epoch": epoch + 1, **dict(zip(("loss", "A", "M", "C", "N"), (totals / len(train)).tolist()))}; curve.append(entry)
        if epoch == 0 or (epoch + 1) % 10 == 0: print(f"[endpoint-gdm-fit] epoch {epoch + 1:03d} loss={entry['loss']:.6f}")

    predictions = {name: predict(model, value, args.batch_size, device) for name, value in data.items()}
    evaluations = {name: metrics(value, predictions[name]) for name, value in data.items() if name not in ("monotonic", "censor", "vertical")}
    mono = predictions["monotonic"].reshape(-1, 8, 8)
    mono_truth = data["monotonic"].reshape(-1, 8, FIELDS)
    mono_L = length_from_q(mono_truth[:, :, 5])
    mono_M = mono[:, :, 1] * mono_L
    monotonic_gate = {
        "ADecreaseFractionOver0_01": float(np.mean(np.diff(mono[:, :, 0], axis=1) < -.01)),
        "MDecreaseFractionOver0_005m": float(np.mean(np.diff(mono_M, axis=1) < -.005)),
        "truthADecreaseFraction": float(np.mean(np.diff(mono_truth[:, :, 6], axis=1) < -1e-6)),
        "truthMDecreaseFraction": float(np.mean(np.diff(mono_truth[:, :, 7] * mono_L, axis=1) < -1e-6)),
    }
    censor_truth = data["censor"].reshape(-1, 2, FIELDS); censor_pred = predictions["censor"].reshape(-1, 2, 8)
    censor_L = length_from_q(censor_truth[:, :, 5]); truth_M = censor_truth[:, :, 7] * censor_L; pred_M = censor_pred[:, :, 1] * censor_L
    censor_gate = {
        "AChangeError": quantiles(np.abs(np.diff(censor_pred[:, :, 0], axis=1)[:, 0] - np.diff(censor_truth[:, :, 6], axis=1)[:, 0])),
        "MChangeErrorMetres": quantiles(np.abs(np.diff(pred_M, axis=1)[:, 0] - np.diff(truth_M, axis=1)[:, 0])),
        "shiftMetres": quantiles(shifts),
    }
    vertical_pred = predictions["vertical"].reshape(-1, 2, 8); vertical_truth = data["vertical"].reshape(-1, 2, FIELDS)
    vertical_gate = {
        "truthPhiInvarianceMax": float(np.max(np.abs(vertical_truth[:, 1, 6:14] - vertical_truth[:, 0, 6:14]))),
        "predictedPhiInvarianceAbsoluteError": quantiles(np.abs(vertical_pred[:, 1] - vertical_pred[:, 0])),
    }
    path_truth_path = Path("data/work/groundcover-hybrid-transfer/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/63a512106a67f5cc44b90d655b178f09f50e36cd2a28488c9c47d30d5a238554/truth/camera-path.f32").resolve()
    path_data = path_records(path_truth_path); path_pred = predict(model, path_data, args.batch_size, device); path_metrics = metrics(path_data, path_pred)
    holdout_names = [name for name in data if name.startswith("holdout-")]
    continuous_pass = (
        oracle_pass
        and
        gate_metric(evaluations["validation"])
        and all(gate_metric(evaluations[name]) for name in holdout_names)
        and gate_metric(evaluations["horizontal"])
        and gate_metric(evaluations["vertical"] if "vertical" in evaluations else metrics(data["vertical"], predictions["vertical"]))
        and gate_metric(path_metrics)
        and monotonic_gate["ADecreaseFractionOver0_01"] <= .005
        and monotonic_gate["MDecreaseFractionOver0_005m"] <= .005
        and censor_gate["AChangeError"]["p95"] <= .05
        and censor_gate["MChangeErrorMetres"]["p95"] <= .025
        and vertical_gate["truthPhiInvarianceMax"] <= 1e-6
        and vertical_gate["predictedPhiInvarianceAbsoluteError"]["max"] <= 1e-3
    )
    args.output.mkdir(parents=True, exist_ok=True); qa = args.output / "qa"; qa.mkdir(exist_ok=True)
    qa_entries = []
    oracle_path = qa / "000-rank8-SVD-oracle-A-truth-vs-reconstruction.png"
    oracle_image(data["oracle"], oracle_prediction, int(manifest["oracle"]["WRows"]), int(manifest["oracle"]["EColumns"])).save(oracle_path)
    qa_entries.append({"path": oracle_path.name, "sha256": digest(oracle_path)})
    for number, name in enumerate(("holdout-u", "holdout-v", "holdout-phi", "holdout-y", "holdout-endpointQ", "horizontal", "vertical"), 1):
        image = image_panel(data[name], predictions[name], name); path = qa / f"{number:03d}-{name}-truth-vs-rank8.png"; image.save(path)
        qa_entries.append({"path": path.name, "sha256": digest(path)})
    path_qa = qa / "008-18-degree-camera-path-truth-vs-rank8.png"; write_path_qa(path_data, path_pred, path_qa); qa_entries.append({"path": path_qa.name, "sha256": digest(path_qa)})
    model_path = args.output / "continuous-rank8-model.pt"; torch.save(model.state_dict(), model_path)
    tables = sample_tables(model, args.output, device, args.batch_size) if continuous_pass else None
    report = {
        "schema": "laas-endpoint-conditioned-gdm-fit/v1", "decision": "continuous-pass" if continuous_pass else "reject-rank8-continuous",
        "truth": {"manifest": str(manifest_path), "manifestSha256": digest(manifest_path), "microFootprint": manifest["microFootprint"], "normalSemantics": manifest["normalSemantics"]},
        "model": {"rank": 8, "factorization": "W(u,v,phi)*E(originY,signedElevation,L), one fixed 8x8 head", "architectureGrowth": 0, "runtimeShaderEdits": 0},
        "training": {"epochs": args.epochs, "batchSize": args.batch_size, "seed": args.seed, "curve": curve},
        "validation": {"optimisticIndependentRank8SvdOracle": oracle_metrics, "splits": evaluations, "monotonicAAndM": monotonic_gate, "rightCensorAAndM": censor_gate, "verticalPhiInvariance": vertical_gate, "cameraPath18Degrees": path_metrics},
        "gate": {"rank8OraclePass": oracle_pass, "continuousPass": continuous_pass, "thresholds": {"AErrorP95": .10, "coverageIoU": .90, "MErrorP95Metres": .025, "conditionalDistanceP95Metres": .05, "colourRmse": .06, "normalP95Degrees": 20, "plumeRecall": .95, "monotonicViolationFraction": .005, "censorAChangeP95": .05, "censorMChangeP95Metres": .025, "verticalPhiMax": 1e-3}, "tableSamplingPermitted": continuous_pass},
        "artifacts": {"continuousModel": {"path": model_path.name, "bytes": model_path.stat().st_size, "sha256": digest(model_path)}, "runtimeShapedTables": tables, "qa": qa_entries},
        "contract": ["outputs are linear-premultiplied A,m,C,N", "A and M=L*m, never m alone, govern endpoint and censor gates", "rank/read/layout are frozen; failure does not authorize architecture growth", "no runtime shader edit occurred"],
        "command": ["uv", "run", "--project", "asset-gen", "--extra", "microtopography-ml", "python", *sys.argv],
    }
    report_path = args.output / "report.json"; report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "report": str(report_path), "validation": evaluations["validation"], "path18": path_metrics, "monotonic": monotonic_gate, "censor": censor_gate, "verticalPhi": vertical_gate}, indent=2))


if __name__ == "__main__":
    main()

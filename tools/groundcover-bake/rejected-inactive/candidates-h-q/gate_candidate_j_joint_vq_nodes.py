#!/usr/bin/env python3
"""Offline exact-node gates for Candidate-J A3+VQ5 and joint-VQ8 codecs."""

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
from PIL import Image, ImageDraw


VERSION = "candidate-j-joint-vq-node-gate-v1"
SCALES = (4, 16)
THRESHOLDS = {"a95": 0.08, "a99": 0.20, "rgb95": 0.06, "rgb99": 0.15, "p95": 0.05, "p99": 0.10, "connected": 0.01}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)].astype(np.float64)
    if not values.size:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {"count": int(values.size), "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": float(np.max(values))}


def nearest_palette(colour: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = colour.reshape(-1, 3)
    result = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], 65536):
        end = min(flat.shape[0], start + 65536)
        d = np.sum((flat[start:end, None] - palette[None]) ** 2, axis=2)
        result[start:end] = np.argmin(d, axis=1)
    return result.reshape(colour.shape[:-1])


def allocate_depth_levels(classes: np.ndarray, depth: np.ndarray, total: int = 32) -> np.ndarray:
    allocation = np.zeros(16, dtype=np.int32)
    present = [c for c in range(16) if np.any(classes == c)]
    if not present:
        allocation[0] = total
        return allocation
    for c in present:
        allocation[c] = 1
    remaining = total - len(present)
    mass = np.zeros(16, dtype=np.float64)
    for c in present:
        values = depth[classes == c]
        mass[c] = max(1.0e-6, math.sqrt(values.size) * float(np.std(values)))
    for _ in range(remaining):
        score = np.where(allocation > 0, mass / allocation, -1.0)
        allocation[int(np.argmax(score))] += 1
    return allocation


def fit_j1_node(
    a: np.ndarray,
    depth: np.ndarray,
    premul: np.ndarray,
    valid: np.ndarray,
    palette: np.ndarray,
    top_h: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aq = np.rint(np.clip(a, 0, 1) * 7.0).astype(np.float32) / 7.0
    conditional = np.zeros((*a.shape, 3), dtype=np.float32)
    conditional[valid] = premul[valid] / np.maximum(a[valid][:, None], 1.0e-8)
    classes = nearest_palette(conditional, palette)
    support_classes = classes[valid]
    support_height = top_h - v * depth[valid]
    allocation = allocate_depth_levels(support_classes, support_height)
    prototype_depth: list[float] = []
    prototype_class: list[int] = []
    class_ranges: dict[int, tuple[int, int]] = {}
    for c in range(16):
        k = int(allocation[c])
        if k == 0:
            continue
        values = support_height[support_classes == c]
        if values.size == 0:
            values = support_height
        centers = np.quantile(values, (np.arange(k) + 0.5) / k).astype(np.float64)
        for _ in range(6):
            labels = np.argmin(np.abs(values[:, None] - centers[None]), axis=1)
            updated = centers.copy()
            for index in range(k):
                selected = values[labels == index]
                if selected.size:
                    updated[index] = float(np.mean(selected))
            centers = updated
        start = len(prototype_depth)
        prototype_depth.extend(centers.tolist())
        prototype_class.extend([c] * k)
        class_ranges[c] = (start, len(prototype_depth))
    while len(prototype_depth) < 32:
        prototype_depth.append(prototype_depth[-1] if prototype_depth else 0.0)
        prototype_class.append(prototype_class[-1] if prototype_class else 0)
    prototype_depth = prototype_depth[:32]
    prototype_class = prototype_class[:32]
    py = np.asarray(prototype_depth, dtype=np.float32)
    py = np.rint(np.clip(py / top_h, 0.0, 1.0) * 4095.0) / 4095.0 * top_h
    pd = (top_h - py) / v
    pc = np.asarray(prototype_class, dtype=np.uint8)
    codes = np.zeros(a.shape, dtype=np.uint8)
    for c, (start, end) in class_ranges.items():
        mask = valid & (classes == c)
        if np.any(mask):
            hit_height = top_h - v * depth[mask]
            codes[mask] = start + np.argmin(np.abs(hit_height[:, None] - py[None, start:end]), axis=1)
    pred_depth = pd[codes]
    pred_rgb = aq[..., None] * palette[pc[codes]]
    return aq, pred_depth, pred_rgb, codes


def quantise_j2_centers(raw: np.ndarray, palette: np.ndarray, top_h: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    aq = np.rint(np.clip(raw[:, 0], 0, 1) * 255.0).astype(np.float32) / 255.0
    yq = np.rint(np.clip(raw[:, 1] / top_h, 0, 1) * 4095.0).astype(np.float32) / 4095.0 * top_h
    conditional = np.zeros((raw.shape[0], 3), dtype=np.float32)
    nonzero = aq > 1.0 / 255.0
    conditional[nonzero] = raw[nonzero, 2:5] / aq[nonzero, None]
    classes = nearest_palette(conditional, palette)
    tauq = (top_h - yq) / v
    predicted = np.column_stack((aq, yq, aq[:, None] * palette[classes])).astype(np.float32)
    packed = (
        np.rint(aq * 255.0).astype(np.uint32)
        | (np.rint(yq / top_h * 4095.0).astype(np.uint32) << 8)
        | (classes.astype(np.uint32) << 20)
        | (np.uint32(0x88) << 24)  # charged normal8; normal gate follows node RGB/depth
    )
    return predicted, packed


def normalized(raw: np.ndarray, v: float) -> np.ndarray:
    scale = np.array([.08, max(v * .05, 1.0e-5), .06, .06, .06], dtype=np.float32)
    return raw / scale[None]


def assign(features: np.ndarray, centers: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(features.shape[0], dtype=np.uint16)
    distances = np.empty(features.shape[0], dtype=np.float32)
    c2 = np.sum(centers * centers, axis=1)
    for start in range(0, features.shape[0], chunk):
        end = min(features.shape[0], start + chunk)
        block = features[start:end]
        d = np.sum(block * block, axis=1)[:, None] + c2[None] - 2.0 * block @ centers.T
        choice = np.argmin(d, axis=1)
        labels[start:end] = choice
        distances[start:end] = d[np.arange(end - start), choice]
    return labels, distances


def fit_j2_row(raw: np.ndarray, palette: np.ndarray, top_h: float, v: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_count = min(65536, raw.shape[0])
    sample_index = rng.choice(raw.shape[0], sample_count, replace=False)
    sample_raw = raw[sample_index]
    initial = sample_raw[rng.choice(sample_count, 256, replace=False)].copy()
    centers_raw = initial
    centers, packed = quantise_j2_centers(centers_raw, palette, top_h, v)
    for _ in range(7):
        sample_features = normalized(sample_raw, v)
        center_features = normalized(centers, v)
        labels, distance = assign(sample_features, center_features)
        updated = centers_raw.copy()
        for component in range(5):
            sums = np.bincount(labels, weights=sample_raw[:, component], minlength=256)
            counts = np.bincount(labels, minlength=256)
            good = counts > 0
            updated[good, component] = sums[good] / counts[good]
        empty = np.flatnonzero(np.bincount(labels, minlength=256) == 0)
        if empty.size:
            far = np.argpartition(distance, -empty.size)[-empty.size:]
            updated[empty] = sample_raw[far]
        centers_raw = updated
        centers, packed = quantise_j2_centers(centers_raw, palette, top_h, v)
    labels, _ = assign(normalized(raw, v), normalized(centers, v))
    return labels.astype(np.uint8), centers, packed


def lattice_rows(hmod, profile) -> list[list[int]]:
    mapping = hmod.lattice_map(profile)
    rows = [[mapping[(row, az)] for az in range(16)] for row in range(4)]
    rows.append([len(profile.slices) - 1])
    return rows


def score(hmod, truth, prediction) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ta, td, tp = truth
    pa, pd, pp = prediction
    valid = np.isfinite(td) & (ta > 1.0 / 255.0)
    ea = np.abs(pa - ta)
    ed = np.where(valid, np.abs(pd - td), np.nan)
    er = np.max(np.abs(pp - tp), axis=3)
    ca, cd, cr = quantiles(ea), quantiles(ed), quantiles(er)
    max_connected, max_direction = 0.0, -1
    for index in range(ta.shape[0]):
        support = valid[index]
        exceed = support & ((ea[index] > .20) | (ed[index] > .10) | (er[index] > .15))
        fraction = hmod.largest_periodic_component(exceed) / max(1, int(np.count_nonzero(support)))
        if fraction > max_connected:
            max_connected, max_direction = float(fraction), index
    checks = {
        "coverageP95": ca["p95"] <= .08, "coverageP99": ca["p99"] <= .20,
        "depthP95": cd["p95"] <= .05, "depthP99": cd["p99"] <= .10,
        "rgbP95": cr["p95"] <= .06, "rgbP99": cr["p99"] <= .15,
        "connected": max_connected < .01,
    }
    return ({"green": all(checks.values()), "checks": checks, "coverage": ca, "depthMetres": cd, "premulRgb": cr, "largestConnected": {"fraction": max_connected, "direction": max_direction}},
            {"coverage": np.max(ea, axis=0), "depth": np.nanmax(ed, axis=0), "rgb": np.max(er, axis=0)})


def heat(values: np.ndarray, limit: float) -> np.ndarray:
    t = np.clip(np.nan_to_num(values) / limit, 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(255 * t).astype(np.uint8)
    rgb[..., 1] = np.rint(255 * np.minimum(1, 2*t) * (1-.6*t)).astype(np.uint8)
    rgb[..., 2] = np.rint(40 * (1-t)).astype(np.uint8)
    return rgb


def write_qa(path: Path, title: str, maps: dict[str, np.ndarray]) -> None:
    panels = [("A/.20", heat(maps["coverage"], .20)), ("depth/.10m", heat(maps["depth"], .10)), ("RGB/.15", heat(maps["rgb"], .15))]
    w, h = panels[0][1].shape[1]*2, panels[0][1].shape[0]*2
    image = Image.new("RGB", (3*w, h+44), (15,15,15)); draw = ImageDraw.Draw(image); draw.text((8,4), title, fill=(255,255,255))
    for i,(label,panel) in enumerate(panels):
        image.paste(Image.fromarray(panel,"RGB").resize((w,h),Image.Resampling.NEAREST),(i*w,44)); draw.text((i*w+8,23),label,fill=(230,230,230))
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def run_scale(hmod, imod, profile, palette: np.ndarray, scale: int, output: Path) -> dict[str, Any]:
    truth = imod.build_truth(hmod, profile, scale)
    ta, td, tp = truth
    count, height, width = ta.shape
    valid = np.isfinite(td) & (ta > 1.0/255.0)

    # J1: per-node A3 plus 32 joint depth/class prototypes.
    j1a = np.empty_like(ta); j1d = np.empty_like(td); j1p = np.empty_like(tp); j1symbols = np.empty_like(ta, dtype=np.uint8)
    for index in range(count):
        node_v = float(-profile.slices[index].direction[1])
        j1a[index], j1d[index], j1p[index], j1symbols[index] = fit_j1_node(
            ta[index], td[index], tp[index], valid[index], palette, profile.top_h, node_v
        )
    j1result, j1maps = score(hmod, truth, (j1a,j1d,j1p))

    # J2: 256 joint prototypes shared by an elevation row.
    j2a = np.empty_like(ta); j2d = np.empty_like(td); j2p = np.empty_like(tp); j2symbols = np.empty_like(ta,dtype=np.uint8)
    codebooks = []
    rows = lattice_rows(hmod, profile)
    for row_index, nodes in enumerate(rows):
        v = float(-profile.slices[nodes[0]].direction[1])
        row_a = ta[nodes].reshape(-1)
        row_d = td[nodes].reshape(-1)
        row_p = tp[nodes].reshape(-1,3)
        row_valid = valid[nodes].reshape(-1)
        y = np.zeros_like(row_a)
        y[row_valid] = profile.top_h - v * row_d[row_valid]
        raw = np.column_stack((row_a, y, row_p)).astype(np.float32)
        symbols, centers, packed = fit_j2_row(raw, palette, profile.top_h, v, seed=0x4A3200 + scale*17 + row_index)
        codebooks.append(packed)
        predicted = centers[symbols]
        shape = (len(nodes),height,width)
        for local,node in enumerate(nodes):
            block = predicted.reshape(*shape,5)[local]
            j2a[node] = block[...,0]
            yhat = block[...,1]
            j2d[node] = (profile.top_h-yhat)/v
            j2p[node] = block[...,2:5]
            j2symbols[node] = symbols.reshape(shape)[local]
        print(f"[candidate-j] sigma={scale} J2 row {row_index+1}/5", flush=True)
    j2result, j2maps = score(hmod, truth, (j2a,j2d,j2p))
    (output/f"sigma{scale}-j1-node-symbols.bin").write_bytes(j1symbols.tobytes())
    (output/f"sigma{scale}-j2-node-symbols.bin").write_bytes(j2symbols.tobytes())
    (output/f"sigma{scale}-j2-codebooks.bin").write_bytes(np.stack(codebooks).astype("<u4").tobytes())
    write_qa(output/"qa"/f"sigma{scale}-j1-errors.png",f"J1 sigma={scale} GREEN={j1result['green']}",j1maps)
    write_qa(output/"qa"/f"sigma{scale}-j2-errors.png",f"J2 sigma={scale} GREEN={j2result['green']}",j2maps)
    return {"J1_A3_VQ5": j1result, "J2_jointVQ8": j2result}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--source",default="src/assets/groundcover/calamagrostis-canescens.gcrp"); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]; source=(root/args.source).resolve()
    hmod=import_file("candidate_h_for_j2",root/"tools/groundcover-bake/analyze_candidate_h_angular_continuity.py")
    imod=import_file("candidate_i_for_j2",root/"tools/groundcover-bake/gate_candidate_i_vertical_extinction.py")
    profile=hmod.load_profile(source); palette=imod.load_palette(root); doc=root/"docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-J-ANGULAR-FE-CODEC.md"
    recipe={"version":VERSION,"sourceSha256":sha256(source),"scriptSha256":sha256(Path(__file__)),"docSha256":sha256(doc),"scales":SCALES,"thresholds":THRESHOLDS}
    recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(); output=root/"data/work/groundcover-candidate-j-joint-vq"/recipe["sourceSha256"][:16]/recipe_sha[:16]; output.mkdir(parents=True,exist_ok=True)
    results={str(scale):run_scale(hmod,imod,profile,palette,scale,output) for scale in SCALES}
    variants={name:all(results[str(scale)][name]["green"] for scale in SCALES) for name in ("J1_A3_VQ5","J2_jointVQ8")}
    report={"schema":VERSION,"verdict":"GREEN" if any(variants.values()) else "RED","variantGreen":variants,"topology":{"pages":64,"C0":"canonical bit-identical shared node symbol"},"results":results,"recipe":recipe}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(output),"verdict":report["verdict"],"variantGreen":variants,"results":results},indent=2))


if __name__=="__main__": main()

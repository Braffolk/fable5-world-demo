"""One-site, non-production rough-till/boulder macroform capacity screen.

This intentionally narrow screen places one native-amplitude whole-form residual
from retained FORWARD evidence at one independently mapped Estonian boulder in
mapped glacial till. It establishes rendering/synthesis capacity only. FORWARD is
not target truth and this screen makes no Estonia-transfer or production claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal.windows import tukey
from shapely import wkb
from shapely.geometry import Point

from ....config import ASSET_GEN_ROOT, DATA_WORK

_CONFIG_SCHEMA = "laas.rough-till-boulder-capacity-screen-config/1"
_MANIFEST_SCHEMA = "laas.rough-till-boulder-capacity-screen/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "rough-till-boulder-capacity-screen" / "sha256"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))


def _identity(path: Path) -> dict[str, Any]:
    return {"path": _repo_path(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _published_identity(path: Path, published_path: Path) -> dict[str, Any]:
    return {"path": _repo_path(published_path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _bound(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if not path.is_file() or path.stat().st_size != row["bytes"] or _sha256_file(path) != row["sha256"]:
        raise ValueError(f"bound input differs: {path}")
    return path


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported rough-till/boulder screen config")
    if config.get("production_authorized") is not False or config.get("estonia_transfer_claim") is not False:
        raise ValueError("capacity screen cannot authorize production or Estonia transfer")
    site = config["site"]
    if site["etak_id"] != 7488269 or site["egt_genesis_code"] != 50 or site["egt_lithology_code"] != 50:
        raise ValueError("selected mapped-boulder/till identity differs")
    if config["output_pitch_m"] != 0.25 or config["source_exemplar"]["patch"] != 4:
        raise ValueError("frozen output/exemplar selection differs")
    return config


def _load_mapped_boulder(path: Path, site: dict[str, Any]) -> Point:
    x, y = map(float, site["point_en"])
    meta, _, geometries, fields = pyogrio.raw.read(
        path,
        layer="E_101_kivi_p",
        bbox=(x - 1.0, y - 1.0, x + 1.0, y + 1.0),
        columns=["etak_id", "tyyp", "korgus"],
    )
    columns = dict(zip(meta["fields"], fields, strict=True))
    hits = np.flatnonzero(columns["etak_id"] == site["etak_id"])
    if len(hits) != 1:
        raise ValueError("mapped boulder is not unique in frozen ETAK window")
    index = int(hits[0])
    if int(columns["tyyp"][index]) != site["etak_type"] or int(columns["korgus"][index]) != site["etak_height_attribute"]:
        raise ValueError("mapped boulder attributes differ")
    point = wkb.loads(geometries[index])
    if point.geom_type != "Point" or point.distance(Point(x, y)) > 0.01:
        raise ValueError("mapped boulder geometry differs")
    return point


def _verify_till(path: Path, point: Point, site: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    covering = []
    for feature in document["features"]:
        geometry = wkb.loads(bytes.fromhex(feature["geometry"]["ogr_wkb_hex"]))
        if geometry.covers(point):
            covering.append(feature)
    matches = [
        feature
        for feature in covering
        if feature["decoded"]["genesis"]["code"] == site["egt_genesis_code"]
        and feature["decoded"]["lithology"]["code"] == site["egt_lithology_code"]
    ]
    if len(matches) != 1:
        raise ValueError("mapped boulder is not uniquely covered by the frozen till feature")
    return matches[0]


def _coords(bbox: tuple[float, float, float, float], pitch: float) -> tuple[np.ndarray, np.ndarray]:
    width = int(round((bbox[2] - bbox[0]) / pitch))
    height = int(round((bbox[3] - bbox[1]) / pitch))
    x = bbox[0] + (np.arange(width, dtype=np.float64) + 0.5) * pitch
    y = bbox[3] - (np.arange(height, dtype=np.float64) + 0.5) * pitch
    return x, y


def _sample(array: np.ndarray, transform: list[float], x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    rows = (transform[5] - y) / -transform[4] - 0.5
    cols = (x - transform[2]) / transform[0] - 0.5
    return ndimage.map_coordinates(array, [rows, cols], order=order, mode="nearest", prefilter=order > 1)


def _hillshade(height: np.ndarray, pitch: float) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    nx, ny, nz = -gx, gy, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    light = np.asarray([np.cos(altitude) * np.sin(azimuth), np.cos(altitude) * np.cos(azimuth), np.sin(altitude)])
    shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], -0.25, 1.0)
    return np.asarray(np.round((shade + 0.25) / 1.25 * 255.0), dtype=np.uint8)


def _map_canvas(array_rgb: np.ndarray, title: str, subtitle: str, marker_rc: tuple[int, int]) -> Image.Image:
    map_image = Image.fromarray(array_rgb, mode="RGB").resize((768, 768), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (768, 860), "white")
    canvas.paste(map_image, (0, 92))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), title, fill="black")
    draw.text((18, 42), subtitle, fill=(70, 70, 70))
    row, col = marker_rc
    px = int((col + 0.5) * 3)
    py = 92 + int((row + 0.5) * 3)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(220, 35, 25), width=3)
    draw.line((24, 828, 144, 828), fill="black", width=5)
    draw.text((24, 806), "10 m", fill="black")
    return canvas


def _residual_rgb(values: np.ndarray, limit: float, allowed: np.ndarray | None = None) -> np.ndarray:
    t = np.clip(values / limit, -1.0, 1.0)
    rgb = np.ones((*values.shape, 3), dtype=np.float64)
    negative = t < 0.0
    positive = ~negative
    rgb[negative, 0] = 1.0 + t[negative] * 0.75
    rgb[negative, 1] = 1.0 + t[negative] * 0.45
    rgb[positive, 1] = 1.0 - t[positive] * 0.75
    rgb[positive, 2] = 1.0 - t[positive] * 0.85
    if allowed is not None:
        rgb[~allowed] = np.asarray([0.16, 0.16, 0.16])
    return np.asarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0), dtype=np.uint8)


def _write_pngs(
    qa: Path,
    before: np.ndarray,
    after: np.ndarray,
    residual: np.ndarray,
    allowed: np.ndarray,
    source_template: np.ndarray,
    marker: tuple[int, int],
    pitch: float,
) -> list[Path]:
    qa.mkdir(parents=True)
    before_rgb = np.repeat(_hillshade(before, pitch)[..., None], 3, axis=2)
    after_rgb = np.repeat(_hillshade(after, pitch)[..., None], 3, axis=2)
    paths = [qa / "01-estonia-c0-before.png", qa / "02-whole-form-after.png", qa / "03-source-and-placed-residual.png"]
    _map_canvas(
        before_rgb,
        "01  ESTONIA C0 BEFORE",
        "64 m mapped-till crop; red circle = ETAK boulder 7488269",
        marker,
    ).save(paths[0])
    _map_canvas(
        after_rgb,
        "02  WHOLE-FORM CAPACITY AFTER",
        "Native-amplitude FORWARD form; no production or Estonia-transfer claim",
        marker,
    ).save(paths[1])

    limit = max(float(np.max(np.abs(source_template))), float(np.max(np.abs(residual))), 0.01)
    source_rgb = _residual_rgb(source_template, limit)
    placed_rgb = _residual_rgb(residual, limit, allowed)
    source_image = Image.fromarray(source_rgb, mode="RGB").resize((720, 720), Image.Resampling.BICUBIC)
    placed_image = Image.fromarray(placed_rgb, mode="RGB").resize((720, 720), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1480, 812), "white")
    canvas.paste(source_image, (16, 76))
    canvas.paste(placed_image, (744, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), "03  SOURCE WHOLE-FORM RESIDUAL", fill="black")
    draw.text((744, 14), "PLACED RESIDUAL + HARD MASKS", fill="black")
    draw.text((16, 42), f"shared diverging scale +/-{limit:.3f} m; FORWARD q0.99 is qualitative/unqualified", fill=(70, 70, 70))
    draw.text((744, 42), "dark = forbidden; red circle = mapped boulder", fill=(70, 70, 70))
    row, col = marker
    px = 744 + int((col + 0.5) * 720 / residual.shape[1])
    py = 76 + int((row + 0.5) * 720 / residual.shape[0])
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(220, 35, 25), width=3)
    canvas.save(paths[2])
    return paths


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    inputs = {name: _bound(row) for name, row in config["inputs"].items()}
    site = config["site"]
    point = _load_mapped_boulder(inputs["etak"], site)
    till_feature = _verify_till(inputs["geology_window"], point, site)

    condition = np.load(inputs["condition_arrays"], allow_pickle=False)
    source = np.load(inputs["forward_arrays"], allow_pickle=False)
    transform = config["condition_transform"]
    bbox = tuple(map(float, config["output_bbox_en"]))
    pitch = float(config["output_pitch_m"])
    x, y = _coords(bbox, pitch)
    xx, yy = np.meshgrid(x, y)
    flat_x, flat_y = xx.ravel(), yy.ravel()
    before = _sample(condition["height"], transform, flat_x, flat_y, 3).reshape(xx.shape)

    masks = {}
    for name in ["valid", "water", "object", "protected_structure", "non_heightfield", "unknown"]:
        masks[name] = _sample(condition[name], transform, flat_x, flat_y, 0).reshape(xx.shape).astype(bool)
    geology_genesis = _sample(condition["geology_genesis_code"], transform, flat_x, flat_y, 0).reshape(xx.shape)
    geology_lithology = _sample(condition["geology_lithology_code"], transform, flat_x, flat_y, 0).reshape(xx.shape)
    soil = _sample(condition["soil_feature_index"], transform, flat_x, flat_y, 0).reshape(xx.shape)
    forbidden = masks["water"] | masks["object"] | masks["protected_structure"] | masks["non_heightfield"] | masks["unknown"]
    allowed = (
        masks["valid"]
        & ~forbidden
        & (geology_genesis == site["egt_genesis_code"])
        & (geology_lithology == site["egt_lithology_code"])
        & (soil >= 0)
    )

    selection = config["source_exemplar"]
    raw = source[f"patch_{selection['patch']:02d}_2m_sigma_residual_m"].astype(np.float64)
    filtered = ndimage.gaussian_filter(raw, float(selection["denoise_sigma_cells"]), mode="reflect")
    center_row, center_col = map(int, selection["center_row_col"])
    size = int(selection["size_cells"])
    half = size // 2
    crop = filtered[center_row - half : center_row + half, center_col - half : center_col + half].copy()
    window = np.outer(tukey(size, float(selection["tukey_alpha"])), tukey(size, float(selection["tukey_alpha"])))
    crop = (crop - float(np.sum(crop * window) / np.sum(window))) * window

    marker_col = int(np.argmin(np.abs(x - point.x)))
    marker_row = int(np.argmin(np.abs(y - point.y)))
    residual = np.zeros_like(before)
    row0, col0 = marker_row - half, marker_col - half
    residual[row0 : row0 + size, col0 : col0 + size] = crop
    safe_distance_m = ndimage.distance_transform_edt(allowed) * pitch
    residual *= np.clip(safe_distance_m / float(config["hard_mask_taper_m"]), 0.0, 1.0)
    residual[~allowed] = 0.0
    after = before + residual

    positive = residual >= float(config["capacity_gate"]["positive_form_threshold_m"])
    labels, _ = ndimage.label(positive, structure=np.ones((3, 3), dtype=np.uint8))
    marker_label = int(labels[marker_row, marker_col])
    positive_area = float(np.count_nonzero(labels == marker_label) * pitch * pitch) if marker_label else 0.0
    metrics = {
        "residual_min_m": float(residual.min()),
        "residual_max_m": float(residual.max()),
        "residual_rms_m": float(np.sqrt(np.mean(residual * residual))),
        "mapped_boulder_residual_m": float(residual[marker_row, marker_col]),
        "mapped_boulder_positive_component_m2": positive_area,
        "forbidden_nonzero_cells": int(np.count_nonzero(residual[~allowed])),
        "outer_border_max_abs_m": float(
            max(np.max(np.abs(residual[[0, -1], :])), np.max(np.abs(residual[:, [0, -1]])))
        ),
        "allowed_cells": int(np.count_nonzero(allowed)),
        "forbidden_cells": int(np.count_nonzero(~allowed)),
    }
    gate = config["capacity_gate"]
    passed = (
        metrics["mapped_boulder_residual_m"] >= gate["minimum_center_relief_m"]
        and gate["positive_component_area_m2"][0] <= positive_area <= gate["positive_component_area_m2"][1]
        and metrics["forbidden_nonzero_cells"] == 0
        and metrics["outer_border_max_abs_m"] == 0.0
        and max(abs(metrics["residual_min_m"]), abs(metrics["residual_max_m"])) <= float(np.max(np.abs(raw))) + 1e-6
    )

    implementation_path = Path(__file__).resolve()
    recipe = {
        "schema_version": _MANIFEST_SCHEMA + ".recipe",
        "config": _identity(config_path),
        "implementation": _identity(implementation_path),
        "inputs": {name: _identity(path) for name, path in inputs.items()},
        "selected_site": {
            **site,
            "verified_point_en": [point.x, point.y],
            "covering_geology_feature": {
                "global_id": till_feature["feature_identity"]["global_id"],
                "source_fid": till_feature["feature_identity"]["source_fid"],
                "decoded": till_feature["decoded"],
            },
        },
        "source_exemplar": selection,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    build_sha = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    output = _ARTIFACT_ROOT / build_sha
    if output.is_dir():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_sha}.", dir=output.parent))
    try:
        (temporary / "recipe.json").write_bytes(_canonical_bytes(recipe) + b"\n")
        np.savez_compressed(
            temporary / "screen.npz",
            before_m=before.astype("<f4"),
            after_m=after.astype("<f4"),
            residual_m=residual.astype("<f4"),
            allowed=allowed.astype("u1"),
        )
        pngs = _write_pngs(temporary / "qa", before, after, residual, allowed, crop, (marker_row, marker_col), pitch)
        qa_index = {
            "schema_version": _MANIFEST_SCHEMA + ".qa/1",
            "build_sha256": build_sha,
            "images": [
                {
                    **_identity(path),
                    "path": f"qa/{path.name}",
                    "dimensions_px": list(Image.open(path).size),
                }
                for path in pngs
            ],
            "interpretation": [
                "01 is the retained Estonian C0 sampled to 0.25 m; it is not a morphology result.",
                "02 differs only by one tapered, native-amplitude whole-form residual placed at the mapped boulder.",
                "03 binds the source residual to its placed residual; dark cells are forced abstentions.",
            ],
            "evidence_boundary": "FORWARD is qualitative, unqualified 0.25 m morphology evidence; not target truth and not evidence of Estonia transfer.",
        }
        (temporary / "qa" / "index.json").write_bytes(_canonical_bytes(qa_index) + b"\n")
        manifest = {
            "schema_version": _MANIFEST_SCHEMA,
            "build_sha256": build_sha,
            "decision": "pass_capacity_only" if passed else "park_capacity_not_shown",
            "metrics": metrics,
            "authority": {
                "production_authorized": False,
                "estonia_transfer_claim": False,
                "target_truth": False,
                "packing_authorized": False,
                "browser_preview_authorized": False,
            },
            "outputs": {
                "recipe": _published_identity(temporary / "recipe.json", output / "recipe.json"),
                "screen": _published_identity(temporary / "screen.npz", output / "screen.npz"),
                "qa_index": _published_identity(temporary / "qa" / "index.json", output / "qa" / "index.json"),
            },
            "limitations": [
                "One mapped Estonia site and one foreign processed residual cannot establish transfer or a regime distribution.",
                "ETAK korgus=0 is an absent/zero height attribute and does not measure boulder relief.",
                "The form demonstrates coherent macro capacity only; it is not a rock-wall, cliff, or sandstone specialist.",
                "No packed, cooked, runtime, or production output was produced.",
            ],
        }
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        if not passed:
            raise RuntimeError(f"rough-till/boulder capacity gate failed: {metrics}")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))

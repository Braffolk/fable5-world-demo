"""Bounded multi-form rough-till research preview from mapped Estonia rocks."""
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
import rasterio
import rasterio.features
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal.windows import tukey
from shapely import from_wkb
from shapely.geometry import Point

from ....config import ASSET_GEN_ROOT, DATA_WORK
from ....process.etak_read import etak_gpkg, read_layer_window
from ....process.micro_masks import rasterize_micro_hard_mask
from .screen import _canonical_bytes, _coords, _hillshade, _identity, _sha256_file

_CONFIG_SCHEMA = "laas.rough-till-boulder-multi-form-config/1"
_MANIFEST_SCHEMA = "laas.rough-till-boulder-multi-form/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "rough-till-boulder-multi-form" / "sha256"


def _bound(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if not path.is_file() or path.stat().st_size != row["bytes"] or _sha256_file(path) != row["sha256"]:
        raise ValueError(f"bound input differs: {path}")
    return path


def _rasterize(geometries: list, shape: tuple[int, int], transform: rasterio.Affine) -> np.ndarray:
    values = [(geometry, 1) for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not values:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        values, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8"
    ).astype(bool)


def _load_json_geometries(path: Path, predicate) -> list:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for feature in document["features"]:
        if predicate(feature):
            geometry = feature["geometry"]
            result.append(from_wkb(bytes.fromhex(geometry.get("ogr_wkb_hex", geometry.get("wkb_hex")))))
    return result


def _read_dtm(path: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float64)
        xx, yy = np.meshgrid(x, y)
        cols, rows = (~source.transform) * (xx, yy)
        height = ndimage.map_coordinates(array, [rows - 0.5, cols - 0.5], order=3, mode="nearest", prefilter=True)
        valid = np.isfinite(height)
        if source.nodata is not None:
            valid &= np.abs(height - float(source.nodata)) > 1e-6
    return height, valid


def _source_template(source: np.lib.npyio.NpzFile, patch: int, size: int, sigma: float) -> np.ndarray:
    raw = source[f"patch_{patch:02d}_2m_sigma_residual_m"].astype(np.float64)
    filtered = ndimage.gaussian_filter(raw, sigma, mode="reflect")
    margin = size // 2
    interior = filtered[margin:-margin, margin:-margin]
    row, col = np.unravel_index(int(np.argmax(interior)), interior.shape)
    row += margin
    col += margin
    half = size // 2
    crop = filtered[row - half : row + half, col - half : col + half].copy()
    window = np.outer(tukey(size, 0.55), tukey(size, 0.55))
    return (crop - float(np.sum(crop * window) / np.sum(window))) * window


def _place(residual: np.ndarray, usage: np.ndarray, template: np.ndarray, row: int, col: int, code: int) -> None:
    half_y, half_x = template.shape[0] // 2, template.shape[1] // 2
    ys, xs = slice(row - half_y, row + half_y), slice(col - half_x, col + half_x)
    residual[ys, xs] += template
    block = usage[ys, xs]
    block[np.abs(template) > 1e-5] = code


def _canvas(rgb: np.ndarray, title: str, subtitle: str, markers: list[tuple[int, int, int]]) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB").resize((768, 768), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (768, 858), "white")
    canvas.paste(image, (0, 90))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill="black")
    draw.text((16, 42), subtitle, fill=(65, 65, 65))
    for row, col, identifier in markers:
        px = int((col + 0.5) * 768 / rgb.shape[1])
        py = 90 + int((row + 0.5) * 768 / rgb.shape[0])
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=(220, 35, 25), width=3)
        draw.text((px + 8, py - 10), str(identifier), fill=(180, 20, 20))
    draw.line((24, 824, 174, 824), fill="black", width=5)
    draw.text((24, 802), "100 m", fill="black")
    return canvas


def _residual_rgb(values: np.ndarray, limit: float, allowed: np.ndarray) -> np.ndarray:
    t = np.clip(values / limit, -1.0, 1.0)
    rgb = np.ones((*values.shape, 3), dtype=np.float64)
    neg = t < 0
    rgb[neg, 0] = 1.0 + t[neg] * 0.75
    rgb[neg, 1] = 1.0 + t[neg] * 0.45
    rgb[~neg, 1] = 1.0 - t[~neg] * 0.75
    rgb[~neg, 2] = 1.0 - t[~neg] * 0.85
    rgb[~allowed] = (0.16, 0.16, 0.16)
    return np.asarray(np.round(np.clip(rgb, 0, 1) * 255), dtype=np.uint8)


def _write_qa(qa: Path, before: np.ndarray, after: np.ndarray, residual: np.ndarray, allowed: np.ndarray, usage: np.ndarray, markers: list[tuple[int, int, int]], pitch: float) -> list[Path]:
    qa.mkdir(parents=True)
    before_rgb = np.repeat(_hillshade(before, pitch)[..., None], 3, axis=2)
    after_rgb = np.repeat(_hillshade(after, pitch)[..., None], 3, axis=2)
    paths = [qa / "01-estonia-c0-before.png", qa / "02-multi-form-after.png", qa / "03-mapped-body-closeups.png", qa / "04-residual-source-usage.png"]
    _canvas(before_rgb, "01  ESTONIA C0 BEFORE", "512 m till window; red labels are retained ETAK single boulders", markers).save(paths[0])
    _canvas(after_rgb, "02  MULTI-FORM TILL AFTER", "Three distinct embedded bodies plus one weak source-derived till-fabric form", markers).save(paths[1])

    close = Image.new("RGB", (1180, 930), "white")
    draw = ImageDraw.Draw(close)
    draw.text((16, 14), "03  MAPPED-BODY CLOSEUPS", fill="black")
    draw.text((16, 40), "left C0 / right C1; shared 48 m crops at native 0.25 m", fill=(65, 65, 65))
    crop = 96
    for index, (row, col, identifier) in enumerate(markers):
        y0, y1, x0, x1 = row - crop, row + crop, col - crop, col + crop
        lhs = Image.fromarray(before_rgb[y0:y1, x0:x1], mode="RGB").resize((360, 260), Image.Resampling.BICUBIC)
        rhs = Image.fromarray(after_rgb[y0:y1, x0:x1], mode="RGB").resize((360, 260), Image.Resampling.BICUBIC)
        top = 78 + index * 278
        close.paste(lhs, (40, top))
        close.paste(rhs, (420, top))
        draw.text((810, top + 18), f"ETAK {identifier}", fill="black")
        draw.text((810, top + 48), f"FORWARD family {index + 2}", fill=(65, 65, 65))
    close.save(paths[2])

    limit = max(float(np.max(np.abs(residual))), 0.01)
    residual_image = Image.fromarray(_residual_rgb(residual, limit, allowed), mode="RGB").resize((720, 720), Image.Resampling.BICUBIC)
    palette = np.asarray([[35, 35, 35], [54, 126, 184], [228, 26, 28], [77, 175, 74], [152, 78, 163]], dtype=np.uint8)
    usage_image = Image.fromarray(palette[np.clip(usage, 0, 4)], mode="RGB").resize((720, 720), Image.Resampling.NEAREST)
    sheet = Image.new("RGB", (1480, 806), "white")
    sheet.paste(residual_image, (16, 72))
    sheet.paste(usage_image, (744, 72))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), "04  SIGNED RESIDUAL", fill="black")
    draw.text((744, 14), "SOURCE USAGE", fill="black")
    draw.text((16, 40), f"shared scale +/-{limit:.3f} m; dark = exact abstention", fill=(65, 65, 65))
    draw.text((744, 40), "blue family 1 fabric; red/green/purple families 2/3/4 bodies", fill=(65, 65, 65))
    sheet.save(paths[3])
    return paths


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported multi-form config")
    if config.get("production_authorized") is not False or config.get("packing_authorized") is not False:
        raise ValueError("research float preview cannot authorize production or packing")
    inputs = {name: _bound(row) for name, row in config["inputs"].items()}
    etak_identity = config["etak_source_identity"]
    etak_path = ASSET_GEN_ROOT.parent / etak_identity["path"]
    if not etak_path.is_file() or etak_path.stat().st_size != etak_identity["bytes"] or etak_path != etak_gpkg():
        raise ValueError("frozen ETAK source path or size differs")
    bbox, pitch = tuple(map(float, config["bbox_en"])), float(config["pitch_m"])
    x, y = _coords(bbox, pitch)
    transform = rasterio.transform.from_origin(bbox[0], bbox[3], pitch, pitch)
    shape = (len(y), len(x))
    before, valid = _read_dtm(inputs["dtm"], x, y)

    soil = _load_json_geometries(inputs["soil_window"], lambda f: f["normalized"]["Loimis1"]["status"] == "parsed_complete_official_grammar" and f["normalized"]["Huumus"]["status"] == "parsed_complete_official_grammar")
    till = _load_json_geometries(inputs["geology_window"], lambda f: f["decoded"]["genesis"]["code"] == 50 and f["decoded"]["lithology"]["code"] == 50)
    soil_supported, till_supported = _rasterize(soil, shape, transform), _rasterize(till, shape, transform)
    hard = rasterize_micro_hard_mask(x, y)

    source_etak = etak_gpkg()
    slope, _ = read_layer_window(source_etak, "E_102_nolv_j", bbox)
    shoreline, _ = read_layer_window(source_etak, "E_204_kaldajoon_j", bbox)
    protected = _rasterize([*slope, *shoreline], shape, transform)
    non_heightfield = _rasterize(list(slope), shape, transform)
    rock_geometries, rock_fields = read_layer_window(source_etak, "E_101_kivi_p", bbox, fields=["etak_id", "tyyp"])
    records = {int(identifier): (geometry, int(kind)) for geometry, identifier, kind in zip(rock_geometries, rock_fields["etak_id"], rock_fields["tyyp"], strict=True)}
    anchors = config["single_boulders"]
    for item in anchors:
        geometry, kind = records[int(item["etak_id"])]
        expected = Point(*map(float, item["point_en"]))
        if kind != 10 or geometry.distance(expected) > 0.01:
            raise ValueError(f"mapped single boulder differs: {item['etak_id']}")
    pile_ids = set(map(int, config["excluded_boulder_piles"]))
    piles = [geometry.buffer(float(config["pile_exclusion_radius_m"])) for identifier, (geometry, kind) in records.items() if identifier in pile_ids and kind == 20]
    if len(piles) != len(pile_ids):
        raise ValueError("mapped boulder-pile exclusions differ")
    pile_mask = _rasterize(piles, shape, transform)
    collar = np.zeros(shape, dtype=bool)
    collar_cells = int(round(float(config["outer_collar_m"]) / pitch))
    collar[:collar_cells] = collar[-collar_cells:] = True
    collar[:, :collar_cells] = True
    collar[:, -collar_cells:] = True
    allowed = valid & soil_supported & till_supported & hard.allowed & ~protected & ~non_heightfield & ~pile_mask & ~collar

    source = np.load(inputs["forward_arrays"], allow_pickle=False)
    residual, usage = np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.uint8)
    markers: list[tuple[int, int, int]] = []
    for item, patch in zip(anchors, (2, 3, 4), strict=True):
        geometry, _ = records[int(item["etak_id"])]
        row, col = int(np.argmin(np.abs(y - geometry.y))), int(np.argmin(np.abs(x - geometry.x)))
        _place(residual, usage, _source_template(source, patch, 64, 4.0), row, col, patch)
        markers.append((row, col, int(item["etak_id"])))
    centroid_e = float(np.mean([records[int(item["etak_id"])][0].x for item in anchors]))
    centroid_n = float(np.mean([records[int(item["etak_id"])][0].y for item in anchors]))
    _place(residual, usage, _source_template(source, 1, 96, 8.0), int(np.argmin(np.abs(y - centroid_n))), int(np.argmin(np.abs(x - centroid_e))), 1)
    safe_distance = ndimage.distance_transform_edt(allowed) * pitch
    residual *= np.clip(safe_distance / float(config["hard_mask_taper_m"]), 0.0, 1.0)
    residual[~allowed] = 0.0
    usage[residual == 0.0] = 0
    after = before + residual

    metrics = {
        "residual_min_m": float(residual.min()), "residual_max_m": float(residual.max()),
        "residual_rms_m": float(np.sqrt(np.mean(residual * residual))), "allowed_fraction": float(np.mean(allowed)),
        "forbidden_nonzero_cells": int(np.count_nonzero(residual[~allowed])),
        "outer_border_max_abs_m": float(max(np.max(np.abs(residual[[0, -1], :])), np.max(np.abs(residual[:, [0, -1]])))),
        "source_usage_cells": {str(code): int(np.count_nonzero(usage == code)) for code in range(1, 5)},
        "anchor_residual_m": {str(identifier): float(residual[row, col]) for row, col, identifier in markers},
        "hard_masks": {"water": int(np.count_nonzero(hard.water)), "building": int(np.count_nonzero(hard.building)), "paved_road": int(np.count_nonzero(hard.paved_road)), "protected": int(np.count_nonzero(protected)), "non_heightfield": int(np.count_nonzero(non_heightfield)), "boulder_pile": int(np.count_nonzero(pile_mask))},
    }
    passed = metrics["forbidden_nonzero_cells"] == 0 and metrics["outer_border_max_abs_m"] == 0.0 and all(v > 0 for v in metrics["source_usage_cells"].values()) and all(v > 0.05 for v in metrics["anchor_residual_m"].values())

    recipe = {"schema_version": _MANIFEST_SCHEMA + ".recipe/1", "config": _identity(config_path), "implementation": _identity(Path(__file__).resolve()), "inputs": {name: _identity(path) for name, path in inputs.items()}, "etak_source_identity": etak_identity, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}}
    build_sha = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    output = _ARTIFACT_ROOT / build_sha
    if output.is_dir():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_sha}.", dir=output.parent))
    try:
        (temporary / "recipe.json").write_bytes(_canonical_bytes(recipe) + b"\n")
        np.savez_compressed(temporary / "preview.npz", before_m=before.astype("<f4"), after_m=after.astype("<f4"), residual_m=residual.astype("<f4"), allowed=allowed.astype("u1"), source_usage=usage)
        pngs = _write_qa(temporary / "qa", before, after, residual, allowed, usage, markers, pitch)
        qa_index = {"schema_version": _MANIFEST_SCHEMA + ".qa/1", "build_sha256": build_sha, "images": [{"path": f"qa/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256_file(path), "dimensions_px": list(Image.open(path).size)} for path in pngs], "interpretation": ["01/02 compare the exact same 512 m Estonia DTM window before and after synthesis.", "03 exposes each distinct mapped body at a readable common scale rather than hiding it in overview metrics.", "04 binds every changed cell to one of four retained FORWARD families; dark cells are exact abstentions."]}
        (temporary / "qa" / "index.json").write_bytes(_canonical_bytes(qa_index) + b"\n")
        manifest = {"schema_version": _MANIFEST_SCHEMA, "build_sha256": build_sha, "decision": "inspect_float_preview" if passed else "park_before_inspection", "metrics": metrics, "authority": {"production": False, "packing": False, "browser": False, "estonia_transfer_truth": False}, "evidence_boundary": "FORWARD is qualitative/unqualified 0.25 m morphology evidence; this preview tests generally-should-transfer capacity only.", "limitations": ["Three mapped single boulders and one 512 m window do not establish a national till distribution.", "Mapped type-20 boulder piles are conservative exclusions, not synthesized natural bodies.", "The weak till-fabric form is source-derived but its local Estonia orientation is not independently measured.", "This is a float/PNG checkpoint with no pack, browser, production, or cliff authority."]}
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        if not passed:
            raise RuntimeError(f"multi-form preview gate failed: {metrics}")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(run(parser.parse_args().config))


if __name__ == "__main__":
    main()

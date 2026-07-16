"""Bounded full-field rough-till challenger from retained FORWARD exemplars."""
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
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal.windows import tukey
from shapely.geometry import Point

from ....config import ASSET_GEN_ROOT, DATA_WORK
from ....process.etak_read import etak_gpkg, read_layer_window
from ....process.landcover import load_rules, rasterize_classes
from ....process.micro_masks import rasterize_micro_hard_mask
from .multi_form import (
    _bound,
    _load_json_geometries,
    _rasterize,
    _read_dtm,
    _residual_rgb,
)
from .screen import _canonical_bytes, _coords, _hillshade, _identity, _sha256_file


_CONFIG_SCHEMA = "laas.rough-till-boulder-exemplar-field-config/1"
_MANIFEST_SCHEMA = "laas.rough-till-boulder-exemplar-field/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "rough-till-boulder-exemplar-field" / "sha256"


def _source_bands(source: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Return coupled full-form and fine bands; neither band is synthesized noise."""
    full, fine = [], []
    yy, xx = np.mgrid[:128, :128]
    design = np.column_stack((np.ones(128 * 128), xx.ravel(), yy.ravel()))
    for index in range(1, 5):
        height = source[f"patch_{index:02d}_height_m"].astype(np.float64)
        plane = np.linalg.lstsq(design, height.ravel(), rcond=None)[0]
        detrended = height - (plane[0] + plane[1] * xx + plane[2] * yy)
        # The 8 m high-pass preserves the measured clast/socket and depositional
        # neighborhood while excluding the exemplar site's regional slope.
        whole = detrended - ndimage.gaussian_filter(detrended, 32.0, mode="reflect")
        high = source[f"patch_{index:02d}_2m_sigma_residual_m"].astype(np.float64)
        full.append(whole)
        fine.append(high)
    return np.stack(full), np.stack(fine)


def _transforms(array: np.ndarray) -> list[np.ndarray]:
    return [array, np.rot90(array, 1), np.rot90(array, 2), np.rot90(array, 3)]


def _candidate_bank(
    whole: np.ndarray, fine: np.ndarray, patch_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = ((0, 0), (0, 16), (0, 32), (16, 0), (16, 16), (16, 32),
               (32, 0), (32, 16), (32, 32))
    candidates, fine_candidates, families = [], [], []
    for family in range(4):
        for row, col in offsets:
            whole_crop = whole[family, row : row + patch_cells, col : col + patch_cells]
            fine_crop = fine[family, row : row + patch_cells, col : col + patch_cells]
            for transformed, fine_transformed in zip(
                _transforms(whole_crop), _transforms(fine_crop), strict=True
            ):
                candidate = transformed.copy()
                candidate -= float(np.mean(candidate))
                high = fine_transformed.copy()
                high -= float(np.mean(high))
                candidates.append(candidate)
                fine_candidates.append(high)
                families.append(family)
    return (
        np.stack(candidates),
        np.stack(fine_candidates),
        np.asarray(families, dtype=np.uint8),
    )


def _irregular_positions(size: int, patch: int, rng: np.random.Generator) -> list[int]:
    positions = [0]
    while positions[-1] + patch < size:
        step = patch - int(rng.integers(20, 33))
        candidate = min(positions[-1] + step, size - patch)
        if candidate == positions[-1]:
            break
        positions.append(candidate)
    return positions


def _quilt_master(
    whole_candidates: np.ndarray,
    fine_candidates: np.ndarray,
    families: np.ndarray,
    target_family: np.ndarray,
    shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonparametric overlap synthesis with coupled band matching."""
    rng = np.random.default_rng(seed)
    patch = whole_candidates.shape[1]
    wy = np.clip(tukey(patch, 0.5), 0.025, 1.0)
    window = np.outer(wy, wy)
    total = np.zeros(shape, dtype=np.float64)
    high_total = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    family_weight = np.zeros((4, *shape), dtype=np.float32)
    y_positions = _irregular_positions(shape[0], patch, rng)
    for row_index, row in enumerate(y_positions):
        # Independent horizontal steps per strip prevent a hidden lattice phase.
        x_positions = _irregular_positions(shape[1], patch, rng)
        if row_index & 1:
            x_positions = sorted(set([0, *[min(x + int(rng.integers(3, 15)), shape[1] - patch) for x in x_positions], shape[1] - patch]))
        for col in x_positions:
            ys, xs = slice(row, row + patch), slice(col, col + patch)
            old_weight = weight[ys, xs]
            known = old_weight > 0.08
            desired = int(np.median(target_family[ys, xs]))
            eligible = np.flatnonzero(families == desired)
            if np.count_nonzero(known) < 64:
                chosen = int(eligible[int(rng.integers(0, len(eligible)))])
            else:
                sampled = known[::2, ::2]
                current = np.divide(
                    total[ys, xs], old_weight, out=np.zeros_like(old_weight), where=known
                )[::2, ::2]
                current_high = np.divide(
                    high_total[ys, xs], old_weight, out=np.zeros_like(old_weight), where=known
                )[::2, ::2]
                difference = whole_candidates[:, ::2, ::2] - current
                high_difference = fine_candidates[:, ::2, ::2] - current_high
                costs = np.mean(difference[:, sampled] ** 2, axis=1)
                costs += 0.35 * np.mean(high_difference[:, sampled] ** 2, axis=1)
                family_costs = costs[eligible]
                shortlist = eligible[np.argpartition(family_costs, 6)[:6]]
                chosen = int(shortlist[int(rng.integers(0, len(shortlist)))])
            candidate = whole_candidates[chosen]
            high = fine_candidates[chosen]
            total[ys, xs] += candidate * window
            high_total[ys, xs] += high * window
            weight[ys, xs] += window
            family_weight[int(families[chosen]), ys, xs] += window.astype(np.float32)
    master = total / weight
    fine_master = high_total / weight
    ownership = np.argmax(family_weight, axis=0).astype(np.uint8) + 1
    return master, fine_master, ownership


def _source_atom(array: np.ndarray, family: int, size: int, rotation: int) -> np.ndarray:
    source = array[family]
    margin = size // 2
    interior = source[margin:-margin, margin:-margin]
    row, col = np.unravel_index(int(np.argmax(interior)), interior.shape)
    row, col = row + margin, col + margin
    atom = source[row - margin : row + margin, col - margin : col + margin].copy()
    atom = np.rot90(atom, rotation)
    window = np.outer(tukey(size, 0.65), tukey(size, 0.65))
    atom -= float(np.sum(atom * window) / np.sum(window))
    return atom


def _replace_anchor(
    master: np.ndarray,
    fine_master: np.ndarray,
    ownership: np.ndarray,
    whole: np.ndarray,
    fine: np.ndarray,
    row: int,
    col: int,
    family: int,
    rotation: int,
) -> None:
    size = 64
    half = size // 2
    ys, xs = slice(row - half, row + half), slice(col - half, col + half)
    window = np.outer(tukey(size, 0.72), tukey(size, 0.72))
    whole_atom = _source_atom(whole, family, size, rotation)
    fine_atom = _source_atom(fine, family, size, rotation)
    master[ys, xs] = master[ys, xs] * (1.0 - window) + whole_atom * window
    fine_master[ys, xs] = fine_master[ys, xs] * (1.0 - window) + fine_atom * window
    ownership[ys, xs][window > 0.55] = family + 1


def _map_panel(array: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(array, mode="RGB").resize((768, 768), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (768, 850), "white")
    canvas.paste(image, (0, 82))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, fill="black")
    draw.text((16, 38), subtitle, fill=(60, 60, 60))
    draw.line((24, 816, 174, 816), fill="black", width=5)
    draw.text((24, 794), "100 m", fill="black")
    return canvas


def _common_light_sheet(before: np.ndarray, after: np.ndarray, pitch: float) -> Image.Image:
    before_rgb = np.repeat(_hillshade(before, pitch)[..., None], 3, axis=2)
    after_rgb = np.repeat(_hillshade(after, pitch)[..., None], 3, axis=2)
    sheet = Image.new("RGB", (1552, 850), "white")
    sheet.paste(_map_panel(before_rgb, "01A  ESTONIA C0", "same 512 m forest/till domain"), (0, 0))
    sheet.paste(_map_panel(after_rgb, "01B  EXAMPLAR FIELD", "shared light; full master, then condition mask"), (784, 0))
    return sheet


def _crop_sheet(
    before: np.ndarray,
    after: np.ndarray,
    selections: list[tuple[int, int, int]],
    pitch: float,
) -> Image.Image:
    before_shade, after_shade = _hillshade(before, pitch), _hillshade(after, pitch)
    sheet = Image.new("RGB", (1300, 980), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "02  COMMON-LIGHT 64 M CROPS", fill="black")
    draw.text((16, 38), "left C0 / right challenger; identical light and scale", fill=(60, 60, 60))
    half = 128
    for index, (row, col, family) in enumerate(selections):
        row = int(np.clip(row, half, before.shape[0] - half))
        col = int(np.clip(col, half, before.shape[1] - half))
        lhs = np.repeat(before_shade[row-half:row+half, col-half:col+half, None], 3, axis=2)
        rhs = np.repeat(after_shade[row-half:row+half, col-half:col+half, None], 3, axis=2)
        lhs_image = Image.fromarray(lhs, mode="RGB").resize((280, 200), Image.Resampling.BICUBIC)
        rhs_image = Image.fromarray(rhs, mode="RGB").resize((280, 200), Image.Resampling.BICUBIC)
        top = 72 + index * 222
        sheet.paste(lhs_image, (30, top))
        sheet.paste(rhs_image, (330, top))
        label = f"forest interior / FORWARD family {family}"
        draw.text((635, top + 18), label, fill="black")
        draw.text((635, top + 46), "64 m crop", fill=(60, 60, 60))
    return sheet


def _target_families(before: np.ndarray, allowed: np.ndarray, pitch: float) -> np.ndarray:
    local = before - ndimage.gaussian_filter(before, 8.0 / pitch, mode="nearest")
    energy = np.sqrt(ndimage.gaussian_filter(local * local, 4.0 / pitch, mode="nearest"))
    thresholds = np.quantile(energy[allowed], (0.25, 0.5, 0.75))
    return np.digitize(energy, thresholds).astype(np.uint8)


def _inspection_centers(
    allowed: np.ndarray, target_family: np.ndarray, pitch: float
) -> list[tuple[int, int, int]]:
    safe = ndimage.distance_transform_edt(allowed) * pitch
    centers = []
    for family in range(4):
        score = np.where((target_family == family) & (safe >= 34.0), safe, -1.0)
        if float(score.max()) < 0.0:
            score = np.where(target_family == family, safe, -1.0)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        centers.append((int(row), int(col), family + 1))
    return centers


def _band_sheet(master: np.ndarray, fine: np.ndarray, allowed: np.ndarray) -> Image.Image:
    limit_whole = max(float(np.quantile(np.abs(master[allowed]), 0.995)), 0.01)
    limit_fine = max(float(np.quantile(np.abs(fine[allowed]), 0.995)), 0.01)
    whole_rgb = _residual_rgb(master, limit_whole, allowed)
    fine_rgb = _residual_rgb(fine, limit_fine, allowed)
    sheet = Image.new("RGB", (1552, 850), "white")
    sheet.paste(_map_panel(whole_rgb, "03A  COUPLED 0.25-8 M FORM", f"shared signed scale +/-{limit_whole:.3f} m"), (0, 0))
    sheet.paste(_map_panel(fine_rgb, "03B  SOURCE 0.25-2 M BAND", f"shared signed scale +/-{limit_fine:.3f} m"), (784, 0))
    return sheet


def _ownership_sheet(
    ownership: np.ndarray,
    allowed: np.ndarray,
    markers: list[tuple[int, int, int]],
) -> Image.Image:
    palette = np.asarray([[38, 38, 38], [50, 120, 190], [220, 80, 60], [70, 165, 95], [155, 90, 180]], dtype=np.uint8)
    rgb = palette[ownership]
    rgb[~allowed] = palette[0]
    panel = _map_panel(rgb, "04  SOURCE-FAMILY OWNERSHIP", "blue/red/green/purple = FORWARD 1/2/3/4; dark = exact abstention")
    draw = ImageDraw.Draw(panel)
    for row, col, identifier in markers:
        px = int((col + 0.5) * 768 / ownership.shape[1])
        py = 82 + int((row + 0.5) * 768 / ownership.shape[0])
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="white", width=2)
        draw.text((px + 8, py - 8), str(identifier), fill="white")
    return panel


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported rough-till exemplar-field config")
    if config.get("production_authorized") is not False or config.get("packing_authorized") is not False:
        raise ValueError("float challenger cannot authorize production or packing")
    inputs = {name: _bound(row) for name, row in config["inputs"].items()}
    etak_identity = config["etak_source_identity"]
    etak_path = ASSET_GEN_ROOT.parent / etak_identity["path"]
    if not etak_path.is_file() or etak_path.stat().st_size != etak_identity["bytes"] or etak_path != etak_gpkg():
        raise ValueError("frozen ETAK source path or size differs")

    bbox, pitch = tuple(map(float, config["bbox_en"])), float(config["pitch_m"])
    x, y = _coords(bbox, pitch)
    shape = (len(y), len(x))
    transform = rasterio.transform.from_origin(bbox[0], bbox[3], pitch, pitch)
    before, valid = _read_dtm(inputs["dtm"], x, y)
    soil = _load_json_geometries(inputs["soil_window"], lambda f: f["normalized"]["Loimis1"]["status"] == "parsed_complete_official_grammar" and f["normalized"]["Huumus"]["status"] == "parsed_complete_official_grammar")
    till = _load_json_geometries(inputs["geology_window"], lambda f: f["decoded"]["genesis"]["code"] == 50 and f["decoded"]["lithology"]["code"] == 50)
    supported = _rasterize(soil, shape, transform) & _rasterize(till, shape, transform)
    forest = rasterize_classes(load_rules(), (*bbox, pitch)) == 1
    hard = rasterize_micro_hard_mask(x, y)
    slope, _ = read_layer_window(etak_gpkg(), "E_102_nolv_j", bbox)
    shoreline, _ = read_layer_window(etak_gpkg(), "E_204_kaldajoon_j", bbox)
    protected = _rasterize([*slope, *shoreline], shape, transform)
    rock_geometries, rock_fields = read_layer_window(etak_gpkg(), "E_101_kivi_p", bbox, fields=["etak_id", "tyyp"])
    records = {int(identifier): (geometry, int(kind)) for geometry, identifier, kind in zip(rock_geometries, rock_fields["etak_id"], rock_fields["tyyp"], strict=True)}
    anchors = config["single_boulders"]
    pile_ids = set(map(int, config["excluded_boulder_piles"]))
    piles = [geometry.buffer(float(config["pile_exclusion_radius_m"])) for identifier, (geometry, kind) in records.items() if identifier in pile_ids and kind == 20]
    if len(piles) != len(pile_ids):
        raise ValueError("mapped boulder-pile exclusions differ")
    pile_mask = _rasterize(piles, shape, transform)
    collar = np.zeros(shape, dtype=bool)
    cells = int(round(float(config["outer_collar_m"]) / pitch))
    collar[:cells] = collar[-cells:] = collar[:, :cells] = collar[:, -cells:] = True
    allowed = valid & supported & forest & hard.allowed & ~protected & ~pile_mask & ~collar

    source = np.load(inputs["forward_arrays"], allow_pickle=False)
    whole, fine = _source_bands(source)
    whole_candidates, fine_candidates, families = _candidate_bank(whole, fine, int(config["quilt_patch_cells"]))
    target_family = _target_families(before, allowed, pitch)
    master, fine_master, ownership = _quilt_master(
        whole_candidates, fine_candidates, families, target_family, shape, int(config["deterministic_seed"])
    )
    markers = []
    for index, item in enumerate(anchors):
        geometry, kind = records[int(item["etak_id"])]
        expected = Point(*map(float, item["point_en"]))
        if kind != 10 or geometry.distance(expected) > 0.01:
            raise ValueError(f"mapped single boulder differs: {item['etak_id']}")
        row, col = int(np.argmin(np.abs(y - geometry.y))), int(np.argmin(np.abs(x - geometry.x)))
        markers.append((row, col, int(item["etak_id"])))

    # The master exists independently of Estonia ownership; masking is the final operation.
    safe_distance = ndimage.distance_transform_edt(allowed) * pitch
    taper = np.clip(safe_distance / float(config["hard_mask_taper_m"]), 0.0, 1.0)
    residual = master * taper
    residual[~allowed] = 0.0
    fine_visible = fine_master * taper
    fine_visible[~allowed] = 0.0
    ownership[~allowed] = 0
    after = before + residual
    selections = _inspection_centers(allowed, target_family, pitch)

    anchor_values = {str(identifier): float(residual[row, col]) for row, col, identifier in markers}
    metrics = {
        "residual_min_m": float(residual.min()),
        "residual_max_m": float(residual.max()),
        "residual_rms_allowed_m": float(np.sqrt(np.mean(residual[allowed] ** 2))),
        "fine_rms_allowed_m": float(np.sqrt(np.mean(fine_visible[allowed] ** 2))),
        "allowed_fraction": float(np.mean(allowed)),
        "forest_fraction": float(np.mean(forest)),
        "forbidden_nonzero_cells": int(np.count_nonzero(residual[~allowed])),
        "outer_border_max_abs_m": float(max(np.max(np.abs(residual[[0, -1], :])), np.max(np.abs(residual[:, [0, -1]])))),
        "source_family_cells": {str(code): int(np.count_nonzero(ownership == code)) for code in range(1, 5)},
        "anchor_residual_m": anchor_values,
        "hard_masks": {"water": int(np.count_nonzero(hard.water)), "building": int(np.count_nonzero(hard.building)), "paved_road": int(np.count_nonzero(hard.paved_road)), "protected": int(np.count_nonzero(protected)), "boulder_pile": int(np.count_nonzero(pile_mask))},
    }
    passed = (
        metrics["forbidden_nonzero_cells"] == 0
        and metrics["outer_border_max_abs_m"] == 0.0
        and all(value > 0 for value in metrics["source_family_cells"].values())
        and all(abs(value) <= 1e-12 for value in anchor_values.values())
    )

    recipe = {"schema_version": _MANIFEST_SCHEMA + ".recipe/1", "config": _identity(config_path), "implementation": _identity(Path(__file__).resolve()), "inputs": {name: _identity(path) for name, path in inputs.items()}, "etak_source_identity": etak_identity, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}}
    build_sha = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    output = _ARTIFACT_ROOT / build_sha
    if output.is_dir():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_sha}.", dir=output.parent))
    try:
        (temporary / "recipe.json").write_bytes(_canonical_bytes(recipe) + b"\n")
        np.savez_compressed(temporary / "preview.npz", before_m=before.astype("<f4"), after_m=after.astype("<f4"), residual_m=residual.astype("<f4"), full_master_m=master.astype("<f4"), fine_master_m=fine_master.astype("<f4"), allowed=allowed.astype("u1"), ownership=ownership, target_family=target_family)
        qa = temporary / "qa"
        qa.mkdir()
        images = [qa / "01-full-domain-common-light.png", qa / "02-common-light-crops.png", qa / "03-coupled-scale-bands.png", qa / "04-source-ownership.png"]
        _common_light_sheet(before, after, pitch).save(images[0])
        _crop_sheet(before, after, selections, pitch).save(images[1])
        _band_sheet(residual, fine_visible, allowed).save(images[2])
        _ownership_sheet(ownership, allowed, markers).save(images[3])
        index = {"schema_version": _MANIFEST_SCHEMA + ".qa/1", "build_sha256": build_sha, "images": [{"path": f"qa/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256_file(path), "dimensions_px": list(Image.open(path).size)} for path in images], "interpretation": ["01 is the mandatory same-light full-domain C0/challenger comparison.", "02 exposes four safe forest interiors spanning measured C0 local-relief states at one common scale.", "03 separates the coupled full-form and source fine bands without claiming truth below 0.25 m.", "04 exposes source-family ownership, unchanged field-class mapped boulders, and exact abstention; it is used to reject visible patch/grid phase rather than excuse it."]}
        (qa / "index.json").write_bytes(_canonical_bytes(index) + b"\n")
        manifest = {"schema_version": _MANIFEST_SCHEMA, "build_sha256": build_sha, "decision": "inspect_float_preview" if passed else "park_before_inspection", "metrics": metrics, "authority": {"production": False, "packing": False, "browser": False, "estonia_transfer_truth": False}, "mechanism": "deterministic nonparametric synthesis of coupled FORWARD 0.25-2 m and 2-8 m measured bands, with source-family state selected by Estonia C0 local-relief quantile inside supported forest/till", "evidence_boundary": "The retained Marrviken product is one unqualified 0.25 m provider DTM without measured-versus-kriged support. This artifact is visual research only.", "limitations": ["One Swedish source site cannot establish Estonia transfer or a national till distribution.", "The three retained mapped boulders are ETAK field-class anchors and remain byte-level unchanged rather than being misused as forest validation.", "The source supports morphology at 0.25 m, not 6.25 cm truth.", "Type-20 mapped boulder piles remain exact exclusions.", "No packing, browser, runtime, cliff, or production authority."]}
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        if not passed:
            raise RuntimeError(f"rough-till exemplar-field gate failed: {metrics}")
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

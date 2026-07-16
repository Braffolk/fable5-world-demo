"""Compose the locked mesic-forest owner with accepted mapped body/socket forms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from assetgen.config import ASSET_GEN_ROOT, load_base
from assetgen.cook.pinned_height import PinnedBaseHeight
from assetgen.process.etak_read import etak_gpkg, read_layer_window
from assetgen.process.micro_masks import rasterize_micro_morphology_mask
from assetgen.terrain.repair.prolong import prolong_structural_4x

from .adjacent import _assemble, _identity, _project, _reduce8, _write_c1
from .generalization import (
    BASE_MANIFEST,
    BASE_MANIFEST_SHA256,
    FACTOR,
    _candidate_capacity,
    _hillshade,
    _load_candidates,
)
from .run import _canonical_json, _sha256


SCHEMA = "forest-mesic-mineral-mapped-boulder-composition/1"
FINE_PITCH_M = 0.0625
PARENT_METRES = 512
FINE_CELLS = 8192
ROCK_FACTOR = 4
OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-boulder-composition/artifact/sha256"
)


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported forest/boulder composition config")
    if config.get("authority") != "research_float_only_no_production_no_pack_no_latest":
        raise ValueError("composition cannot authorize packing or production")
    if config.get("bbox_en") != [683520, 6435840, 684032, 6436352]:
        raise ValueError("composition bbox differs from the accepted mapped-rock parent")
    if config.get("parent") != {"lod": -1, "cx": 615, "cz": 389}:
        raise ValueError("composition parent differs")
    if config.get("texel_m") != FINE_PITCH_M:
        raise ValueError("composition changed the accepted fine lattice")
    accepted_transport = {
        "patch_cells": 192,
        "candidate_step_cells": 16,
        "minimum_patch_support_fraction": 0.78,
        "site_candidate_cell": 24,
        "site_minimum_distance_cells": 64,
        "site_saturation_distance_cells": 48,
        "site_saturation_rounds": 8,
        "support_radius_cells": 88,
        "site_seed": 5494753648857171000,
    }
    if config.get("transport") != accepted_transport:
        raise ValueError("composition retuned the locked forest transport")
    return config


def _bound(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(row["bytes"])
        or _sha256(path) != row["sha256"]
    ):
        raise ValueError(f"bound source differs: {path}")
    return path


def _pile_points(config: dict[str, Any]) -> list[tuple[float, float]]:
    bbox = tuple(config["bbox_en"])
    geometries, fields = read_layer_window(
        etak_gpkg(), "E_101_kivi_p", bbox, fields=["etak_id", "tyyp"]
    )
    records = {
        int(identifier): (geometry, int(kind))
        for geometry, identifier, kind in zip(
            geometries, fields["etak_id"], fields["tyyp"], strict=True
        )
    }
    result = []
    for identifier in map(int, config["excluded_boulder_piles"]):
        geometry, kind = records[identifier]
        if kind != 20:
            raise ValueError(f"mapped pile differs: {identifier}")
        result.append((float(geometry.x), float(geometry.y)))
    return result


def _masks(
    bbox: tuple[int, int, int, int],
    forest_path: Path,
    rock_path: Path,
    pile_path: Path,
    pile_points: list[tuple[float, float]],
    pile_radius_m: float,
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, int]]:
    forest = np.memmap(forest_path, dtype=np.uint8, mode="w+", shape=(FINE_CELLS, FINE_CELLS))
    rock = np.memmap(rock_path, dtype=np.uint8, mode="w+", shape=(FINE_CELLS, FINE_CELLS))
    pile = np.memmap(pile_path, dtype=np.uint8, mode="w+", shape=(FINE_CELLS, FINE_CELLS))
    evidence: dict[str, int] = {}
    e_min, _n_min, _e_max, n_max = bbox
    tile_cells = 2048
    radius2 = pile_radius_m * pile_radius_m
    for tile_row in range(4):
        for tile_col in range(4):
            tile_e = e_min + tile_col * 128
            tile_n = n_max - tile_row * 128
            east = tile_e + (np.arange(tile_cells) + 0.5) * FINE_PITCH_M
            north = tile_n - (np.arange(tile_cells) + 0.5) * FINE_PITCH_M
            mask = rasterize_micro_morphology_mask(east, north)
            local_pile = np.zeros((tile_cells, tile_cells), dtype=bool)
            for point_e, point_n in pile_points:
                local_pile |= (
                    (north[:, None] - point_n) ** 2
                    + (east[None, :] - point_e) ** 2
                    <= radius2
                )
            region = np.s_[
                tile_row * tile_cells : (tile_row + 1) * tile_cells,
                tile_col * tile_cells : (tile_col + 1) * tile_cells,
            ]
            forest[region] = mask.allowed & ~local_pile
            # Rock support is already bounded by the accepted source artifact's
            # complete soil+till gate.  Reapplying the narrower forest-soil
            # taxonomy here would incorrectly delete two accepted anchors.
            rock[region] = (
                ~mask.water
                & ~mask.building
                & ~mask.paved_road
                & ~mask.slope_cliff_context
                & ~local_pile
            )
            pile[region] = local_pile
            for key, value in mask.evidence().items():
                if isinstance(value, int):
                    evidence[key] = evidence.get(key, 0) + value
    forest.flush()
    rock.flush()
    pile.flush()
    evidence["mappedBoulderPileCells"] = int(np.count_nonzero(pile))
    evidence["forestAllowedAfterPileCells"] = int(np.count_nonzero(forest))
    evidence["rockConditionAllowedCells"] = int(np.count_nonzero(rock))
    return forest, rock, pile, evidence


def _compose_rocks(
    *,
    c1: np.memmap,
    residual: np.memmap,
    ownership: np.memmap,
    rock_condition: np.memmap,
    source_path: Path,
    mapped: list[dict[str, Any]],
) -> dict[str, Any]:
    source = np.load(source_path, allow_pickle=False)
    coarse_residual = np.asarray(source["residual_m"], dtype=np.float64)
    usage = np.asarray(source["source_usage"], dtype=np.uint8)
    source_allowed = np.asarray(source["allowed"], dtype=bool)
    if coarse_residual.shape != (2048, 2048) or usage.shape != (2048, 2048):
        raise ValueError("mapped-rock source must be 2048 square at 0.25 m")
    if np.any((usage > 0) & ~np.isin(usage, (1, 2, 3, 4))):
        raise ValueError("mapped-rock source usage is invalid")
    if np.any((usage >= 2) & ~source_allowed):
        raise ValueError("mapped body/socket support escaped its accepted soil+till mask")
    padded = np.pad(coarse_residual, 2, mode="constant")
    metrics: dict[str, Any] = {"families": {}, "generic_till_owned_cells": 0}
    for item in mapped:
        code = int(item["source_family"])
        coarse_owner = usage == code
        rows, cols = np.nonzero(coarse_owner)
        if rows.size == 0:
            raise ValueError(f"mapped source family {code} is empty")
        row0, row1 = int(rows.min()), int(rows.max()) + 1
        col0, col1 = int(cols.min()), int(cols.max()) + 1
        condition = np.asarray(
            rock_condition[
                row0 * ROCK_FACTOR : row1 * ROCK_FACTOR,
                col0 * ROCK_FACTOR : col1 * ROCK_FACTOR,
            ],
            dtype=bool,
        )
        condition_cells = condition.reshape(
            row1 - row0, ROCK_FACTOR, col1 - col0, ROCK_FACTOR
        ).all(axis=(1, 3))
        accepted_coarse = coarse_owner[row0:row1, col0:col1] & condition_cells
        fine_owner = np.repeat(np.repeat(accepted_coarse, ROCK_FACTOR, axis=0), ROCK_FACTOR, axis=1)
        fine_rock = prolong_structural_4x(
            padded,
            parent_rows=(row0 + 2, row1 + 2),
            parent_cols=(col0 + 2, col1 + 2),
        )
        fine_rows = np.s_[row0 * ROCK_FACTOR : row1 * ROCK_FACTOR]
        fine_cols = np.s_[col0 * ROCK_FACTOR : col1 * ROCK_FACTOR]
        current = np.asarray(c1[fine_rows, fine_cols], dtype=np.float64)
        forest_residual = np.asarray(residual[fine_rows, fine_cols], dtype=np.float64)
        baseline = current - forest_residual
        composed_residual = np.where(fine_owner, fine_rock, forest_residual)
        c1[fine_rows, fine_cols] = (baseline + composed_residual).astype(np.float32)
        residual[fine_rows, fine_cols] = composed_residual.astype(np.float32)
        local_ownership = ownership[fine_rows, fine_cols]
        local_ownership[fine_owner] = code
        blocks = composed_residual.reshape(
            row1 - row0, ROCK_FACTOR, col1 - col0, ROCK_FACTOR
        ).mean(axis=(1, 3))
        mean_error = float(
            np.max(
                np.abs(blocks[accepted_coarse] - coarse_residual[row0:row1, col0:col1][accepted_coarse]),
                initial=0.0,
            )
        )
        point_e, point_n = map(float, item["point_en"])
        parent_row = int((6436352.0 - point_n) / 0.25)
        parent_col = int((point_e - 683520.0) / 0.25)
        metrics["families"][str(code)] = {
            "etak_id": int(item["etak_id"]),
            "owned_fine_cells": int(np.count_nonzero(fine_owner)),
            "owned_parent_cells": int(np.count_nonzero(accepted_coarse)),
            "parent_mean_max_error_m": mean_error,
            "anchor_parent_residual_m": float(coarse_residual[parent_row, parent_col]),
            "anchor_owned": bool(usage[parent_row, parent_col] == code and accepted_coarse[parent_row - row0, parent_col - col0]),
            "fine_bbox_rows_cols": [row0 * 4, row1 * 4, col0 * 4, col1 * 4],
        }
    c1.flush()
    residual.flush()
    ownership.flush()
    return metrics


def _resize_rgb(rgb: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


def _shade(values: np.ndarray) -> np.ndarray:
    value = _hillshade(values, 0.5)
    return np.repeat(value[..., None], 3, axis=2)


def _write_qa(
    qa: Path,
    c0: np.ndarray,
    c1: np.ndarray,
    ownership: np.ndarray,
    mapped: list[dict[str, Any]],
) -> list[Path]:
    qa.mkdir(parents=True, exist_ok=True)
    shade0, shade1 = _shade(c0), _shade(c1)
    paths = [qa / "01_full_common_light_composition.png"]
    paths.extend(
        qa / f"0{index + 2}_mapped_rock_{int(item['etak_id'])}_socket_closeup.png"
        for index, item in enumerate(mapped)
    )
    paths.append(qa / "05_ordinary_forest_to_rock_transition.png")

    canvas = Image.new("RGB", (1328, 718), (244, 242, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "01  FULL FOREST + MAPPED-ROCK COMPOSITION", fill=(20, 24, 20))
    draw.text((16, 34), "same 512 m parent and common light; mapped bodies replace forest only on bounded support", fill=(70, 72, 66))
    canvas.paste(_resize_rgb(shade0, 640), (16, 62))
    canvas.paste(_resize_rgb(shade1, 640), (672, 62))
    draw.text((16, 44), "PINNED C0", fill=(20, 24, 20))
    draw.text((672, 44), "COMPOSED C1", fill=(20, 24, 20))
    canvas.save(paths[0], compress_level=9)

    centers: list[tuple[int, int]] = []
    for path, item in zip(paths[1:4], mapped, strict=True):
        point_e, point_n = map(float, item["point_en"])
        col = int((point_e - 683520.0) / 0.5)
        row = int((6436352.0 - point_n) / 0.5)
        centers.append((row, col))
        half = 48
        crop = np.s_[row - half : row + half, col - half : col + half]
        sheet = Image.new("RGB", (1328, 718), (244, 242, 234))
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), f"MAPPED ROCK {int(item['etak_id'])} / FAMILY {int(item['source_family'])}", fill=(20, 24, 20))
        draw.text((16, 34), "48 x 48 m at identical common light; left C0, right composed body/socket in forest", fill=(70, 72, 66))
        sheet.paste(_resize_rgb(shade0[crop], 640), (16, 62))
        sheet.paste(_resize_rgb(shade1[crop], 640), (672, 62))
        sheet.save(path, compress_level=9)

    row, col = centers[1]
    half = 72
    crop = np.s_[row - half : row + half, col - half : col + half]
    palette = np.asarray(
        [[42, 43, 39], [50, 135, 82], [218, 92, 63], [225, 155, 54], [120, 91, 177]],
        dtype=np.uint8,
    )
    transition = Image.new("RGB", (1944, 718), (244, 242, 234))
    draw = ImageDraw.Draw(transition)
    draw.text((16, 12), "05  ORDINARY FOREST-TO-ROCK OWNERSHIP TRANSITION", fill=(20, 24, 20))
    draw.text((16, 34), "72 x 72 m context; green forest, orange mapped family 3, charcoal exact abstention", fill=(70, 72, 66))
    transition.paste(_resize_rgb(shade0[crop], 624), (16, 62))
    transition.paste(_resize_rgb(shade1[crop], 624), (656, 62))
    transition.paste(_resize_rgb(palette[np.clip(ownership[crop], 0, 4)], 624), (1296, 62))
    transition.save(paths[4], compress_level=9)
    return paths


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    bbox = tuple(map(int, config["bbox_en"]))
    source_path = _bound(config["rock_source"])
    recipe = {
        "schema": SCHEMA + ".recipe/1",
        "authority": config["authority"],
        "config": _identity(config_path),
        "implementation": _identity(Path(__file__).resolve()),
        "locked_forest_implementation": _identity(Path(__file__).with_name("adjacent.py")),
        "mapped_rock_source": _identity(source_path),
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "composition": "forest owner replaced only by mapped families 2/3/4 on complete admissible 0.25 m supports",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = OUTPUT_ROOT / build_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates()
    capacity = _candidate_capacity(candidates)
    base = PinnedBaseHeight(
        BASE_MANIFEST,
        BASE_MANIFEST_SHA256,
        ASSET_GEN_ROOT / "data/out",
        load_base().encode,
        audit=False,
    )
    with TemporaryDirectory(prefix=f".{build_id}.", dir=destination.parent, delete=False) as temporary:
        staging = Path(temporary)
        surface, qa, scratch = staging / "surface", staging / "qa", staging / "scratch"
        surface.mkdir()
        qa.mkdir()
        scratch.mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        residual_path, forest_source_path, _dominant, placements, source_counts, _cross = _assemble(
            candidates=candidates, bbox=bbox, scratch=scratch, config=config
        )
        forest_allowed, rock_condition, pile_mask, mask_evidence = _masks(
            bbox,
            scratch / "forest_allowed.u8",
            scratch / "rock_condition.u8",
            scratch / "pile.u8",
            _pile_points(config),
            float(config["pile_exclusion_radius_m"]),
        )
        forest_mean_error = _project(residual_path, forest_allowed, (FINE_CELLS, FINE_CELLS))
        residual = np.memmap(residual_path, dtype=np.float32, mode="r+", shape=(FINE_CELLS, FINE_CELLS))
        c1, authority = _write_c1(
            base=base, bbox=bbox, residual_path=residual_path, destination=surface / "c1_height_f32.npy"
        )
        ownership = np.lib.format.open_memmap(
            surface / "composition_ownership_u8.npy", mode="w+", dtype=np.uint8, shape=(FINE_CELLS, FINE_CELLS)
        )
        ownership[:] = np.asarray(forest_allowed, dtype=np.uint8)
        rock_metrics = _compose_rocks(
            c1=c1,
            residual=residual,
            ownership=ownership,
            rock_condition=rock_condition,
            source_path=source_path,
            mapped=config["mapped_boulders"],
        )
        maximum_abstained = 0.0
        maximum_pile = 0.0
        residual_square_sum = 0.0
        for row0 in range(0, FINE_CELLS, 128):
            row1 = min(FINE_CELLS, row0 + 128)
            values = np.asarray(residual[row0:row1], dtype=np.float64)
            local_owner = np.asarray(ownership[row0:row1])
            local_pile = np.asarray(pile_mask[row0:row1], dtype=bool)
            maximum_abstained = max(
                maximum_abstained,
                float(np.max(np.abs(values[local_owner == 0]), initial=0.0)),
            )
            maximum_pile = max(
                maximum_pile,
                float(np.max(np.abs(values[local_pile]), initial=0.0)),
            )
            residual_square_sum += float(np.sum(values * values, dtype=np.float64))
        forest_only = np.asarray(ownership == 1)
        coarse_means = np.asarray(residual).reshape(512, FACTOR, 512, FACTOR).mean(axis=(1, 3))
        forest_blocks = forest_only.reshape(512, FACTOR, 512, FACTOR).all(axis=(1, 3))
        forest_closure = float(np.max(np.abs(coarse_means[forest_blocks]), initial=0.0))
        c1_half = _reduce8(c1, (FINE_CELLS, FINE_CELLS))
        residual_half = _reduce8(residual, (FINE_CELLS, FINE_CELLS))
        c0_half = c1_half - residual_half
        ownership_half = np.asarray(ownership[4::8, 4::8], dtype=np.uint8)
        pngs = _write_qa(qa, c0_half, c1_half, ownership_half, config["mapped_boulders"])
        acceptance = config["acceptance"]
        failures = []
        if float(np.mean(forest_allowed)) < acceptance["minimum_forest_allowed_fraction"]:
            failures.append("insufficient forest ownership")
        if forest_mean_error > acceptance["maximum_forest_one_metre_mean_error_m"]:
            failures.append("forest projection closure")
        if maximum_abstained > acceptance["maximum_abstained_residual_m"] or maximum_pile > 0.0:
            failures.append("forbidden residual")
        for family in rock_metrics["families"].values():
            if family["owned_fine_cells"] < acceptance["minimum_rock_owned_cells_per_family"]:
                failures.append(f"empty mapped family {family['etak_id']}")
            if family["parent_mean_max_error_m"] > acceptance["maximum_rock_parent_mean_error_m"]:
                failures.append(f"mapped family mean {family['etak_id']}")
            if not family["anchor_owned"]:
                failures.append(f"mapped anchor rejected {family['etak_id']}")
        metrics = {
            "bbox_en": list(bbox),
            "shape": [FINE_CELLS, FINE_CELLS],
            "mask": mask_evidence,
            "forest_allowed_fraction": float(np.mean(forest_allowed)),
            "forest_site_count": len(placements),
            "forest_source_placement_counts": source_counts,
            "candidate_capacity": capacity,
            "forest_projection_maximum_one_metre_mean_error_m": forest_mean_error,
            "forest_only_maximum_one_metre_mean_error_after_composition_m": forest_closure,
            "maximum_abstained_residual_m": maximum_abstained,
            "maximum_boulder_pile_residual_m": maximum_pile,
            "rock": rock_metrics,
            "composed_residual_rms_m": float(np.sqrt(residual_square_sum / residual.size)),
            "composed_residual_min_max_m": [float(np.min(residual)), float(np.max(residual))],
            "authority_p01_p99_m": [float(v) for v in np.percentile(authority, [1, 99])],
        }
        (staging / "metrics.json").write_bytes(_canonical_json(metrics) + b"\n")
        (surface / "placements.json").write_bytes(_canonical_json(placements) + b"\n")
        del c1, residual, ownership, forest_allowed, rock_condition, pile_mask
        shutil.rmtree(scratch)
        qa_index = {
            "schema": SCHEMA + ".qa/1",
            "build_id": build_id,
            "images": [
                {"path": f"qa/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256(path), "dimensions_px": list(Image.open(path).size)}
                for path in pngs
            ],
            "interpretation": [
                "01 compares the complete pinned and composed parent under identical common light.",
                "02-04 are separate 48 m mapped body/socket before/after closeups.",
                "05 shows ordinary forest, bounded mapped-rock ownership, and exact abstention together in one 72 m context.",
            ],
        }
        (qa / "index.json").write_bytes(_canonical_json(qa_index) + b"\n")
        files = {
            str(path.relative_to(staging)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema": SCHEMA + ".artifact/1",
            "build_id": build_id,
            "status": "ready_to_pack" if not failures else "park_before_pack",
            "failures": failures,
            "metrics": metrics,
            "authority": {"production": False, "packing": False, "browser": False, "latest": False, "national": False},
            "ownership": {"0": "exact C0 abstention", "1": "locked mesic-mineral forest", "2": "mapped body/socket 1145271", "3": "mapped body/socket 1145269", "4": "mapped body/socket 1145643"},
            "limitations": [
                "This composes one accepted forest research owner with three bounded mapped-rock forms; it establishes no generic till owner.",
                "The mapped forms retain 0.25 m source authority and use structural reconstruction only below that pitch.",
                "Boulder piles, cliffs/slopes, water, hard surfaces, unsupported soil, and unknown context receive no residual.",
                "No packing, browser, runtime, format, production, latest, or national authority is granted by this float artifact.",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(run(parser.parse_args().config))


if __name__ == "__main__":
    main()

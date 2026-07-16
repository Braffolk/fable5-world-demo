"""Compose accepted mapped body/socket forms with their explicit terrain carrier."""

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
from scipy import ndimage

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
POSITIVE_SCHEMA_V1 = "forest-mesic-mineral-mapped-boulder-composition-positive/1"
POSITIVE_SCHEMA_V2 = "forest-mesic-mineral-mapped-boulder-composition-positive/2"
POSITIVE_SCHEMA = "forest-mesic-mineral-mapped-boulder-composition-positive/3"
ROCK_ONLY_SCHEMA = "mapped-boulder-body-socket-rock-only/1"
FINE_PITCH_M = 0.0625
PARENT_METRES = 512
FINE_CELLS = 8192
ROCK_FACTOR = 4
OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-boulder-composition/artifact/sha256"
)
ROCK_ONLY_OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-rock-only-body-socket/artifact/sha256"
)


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") not in {
        SCHEMA,
        POSITIVE_SCHEMA_V1,
        POSITIVE_SCHEMA_V2,
        POSITIVE_SCHEMA,
        ROCK_ONLY_SCHEMA,
    }:
        raise ValueError("unsupported forest/boulder composition config")
    if config.get("authority") != "research_float_only_no_production_no_pack_no_latest":
        raise ValueError("composition cannot authorize packing or production")
    expected_site = (
        ([683520, 6435328, 684032, 6435840], {"lod": -1, "cx": 615, "cz": 390})
        if config["schema"] == ROCK_ONLY_SCHEMA
        else
        ([680448, 6436352, 680960, 6436864], {"lod": -1, "cx": 609, "cz": 388})
        if config["schema"] in {
            POSITIVE_SCHEMA_V1,
            POSITIVE_SCHEMA_V2,
            POSITIVE_SCHEMA,
        }
        else ([683520, 6435840, 684032, 6436352], {"lod": -1, "cx": 615, "cz": 389})
    )
    if config.get("bbox_en") != expected_site[0] or config.get("parent") != expected_site[1]:
        raise ValueError("composition site differs from its frozen schema")
    if config.get("texel_m") != FINE_PITCH_M:
        raise ValueError("composition changed the accepted fine lattice")
    if config["schema"] == ROCK_ONLY_SCHEMA:
        expected_target = [{
            "etak_id": 1145648,
            "point_en": [683871.66, 6435469.33],
            "source_family": 4,
            "source_point_en": [683888.95, 6436052.36],
        }]
        if config.get("mapped_boulders") != expected_target:
            raise ValueError("rock-only target or unchanged family-4 binding differs")
        if config.get("excluded_boulder_piles") != [1086981, 1145223, 1147667]:
            raise ValueError("rock-only type-20 exclusions differ")
        assignment = config.get("family_assignment", {})
        if (
            assignment.get("selected") != {"1145648": 4}
            or assignment.get("no_rotation_scale_or_amplitude_change") is not True
        ):
            raise ValueError("rock-only family-4 transform changed")
        if "transport" in config:
            raise ValueError("rock-only checkpoint must not invoke forest transport")
        return config
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


def _target_evidence(config: dict[str, Any]) -> dict[str, Any]:
    item = config["mapped_boulders"][0]
    point_e, point_n = map(float, item["point_en"])
    geometries, fields = read_layer_window(
        etak_gpkg(),
        "E_101_kivi_p",
        (point_e - 1.0, point_n - 1.0, point_e + 1.0, point_n + 1.0),
        fields=["etak_id", "tyyp", "korgus"],
    )
    records = {
        int(identifier): (geometry, int(kind), int(height))
        for geometry, identifier, kind, height in zip(
            geometries,
            fields["etak_id"],
            fields["tyyp"],
            fields["korgus"],
            strict=True,
        )
    }
    geometry, kind, height = records[int(item["etak_id"])]
    if (
        kind != 10
        or height != 2
        or abs(float(geometry.x) - point_e) > 1.0e-3
        or abs(float(geometry.y) - point_n) > 1.0e-3
    ):
        raise ValueError("rock-only ETAK target evidence differs")
    return {
        "etak_id": int(item["etak_id"]),
        "type": kind,
        "recorded_height_m": height,
        "point_en": [float(geometry.x), float(geometry.y)],
        "point_z_m": float(geometry.z),
    }


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


def _rock_only_masks(
    bbox: tuple[int, int, int, int],
    rock_path: Path,
    pile_path: Path,
    pile_points: list[tuple[float, float]],
    pile_radius_m: float,
) -> tuple[np.memmap, np.memmap, dict[str, int]]:
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
    rock.flush()
    pile.flush()
    evidence["mappedBoulderPileCells"] = int(np.count_nonzero(pile))
    evidence["rockConditionAllowedCells"] = int(np.count_nonzero(rock))
    return rock, pile, evidence


def _compose_rocks(
    *,
    c1: np.memmap,
    residual: np.memmap,
    ownership: np.memmap,
    rock_condition: np.memmap,
    forest_allowed: np.memmap,
    source_path: Path,
    mapped: list[dict[str, Any]],
    bbox: tuple[int, int, int, int],
    rock_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    metrics: dict[str, Any] = {"families": {}, "generic_till_owned_cells": 0}
    qa_records: list[dict[str, Any]] = []
    for item in mapped:
        code = int(item["source_family"])
        source_point = item.get("source_point_en", item["point_en"])
        source_east = 683520.0 + (np.arange(2048) + 0.5) * 0.25
        source_north = 6436352.0 - (np.arange(2048) + 0.5) * 0.25
        source_center_row = int(np.argmin(np.abs(source_north - float(source_point[1]))))
        source_center_col = int(np.argmin(np.abs(source_east - float(source_point[0]))))
        source_row0, source_row1 = source_center_row - 32, source_center_row + 32
        source_col0, source_col1 = source_center_col - 32, source_center_col + 32
        coarse_owner = usage[source_row0:source_row1, source_col0:source_col1] == code
        rows, cols = np.nonzero(coarse_owner)
        if rows.size == 0:
            raise ValueError(f"mapped source family {code} is empty")
        point_e, point_n = map(float, item["point_en"])
        target_east = bbox[0] + (np.arange(2048) + 0.5) * 0.25
        target_north = bbox[3] - (np.arange(2048) + 0.5) * 0.25
        target_center_row = int(np.argmin(np.abs(target_north - point_n)))
        target_center_col = int(np.argmin(np.abs(target_east - point_e)))
        row0, row1 = target_center_row - 32, target_center_row + 32
        col0, col1 = target_center_col - 32, target_center_col + 32
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
        accepted_coarse = coarse_owner & condition_cells
        fine_owner = np.repeat(np.repeat(accepted_coarse, ROCK_FACTOR, axis=0), ROCK_FACTOR, axis=1)
        source_patch = coarse_residual[source_row0:source_row1, source_col0:source_col1]
        fine_rock = prolong_structural_4x(
            np.pad(source_patch, 2, mode="constant"),
            parent_rows=(2, 66),
            parent_cols=(2, 66),
        )
        fine_rows = np.s_[row0 * ROCK_FACTOR : row1 * ROCK_FACTOR]
        fine_cols = np.s_[col0 * ROCK_FACTOR : col1 * ROCK_FACTOR]
        context_margin = 64  # 4 m around the complete 16 m source window.
        context_row0 = row0 * ROCK_FACTOR - context_margin
        context_row1 = row1 * ROCK_FACTOR + context_margin
        context_col0 = col0 * ROCK_FACTOR - context_margin
        context_col1 = col1 * ROCK_FACTOR + context_margin
        context_rows = np.s_[context_row0:context_row1]
        context_cols = np.s_[context_col0:context_col1]
        context_current = np.asarray(c1[context_rows, context_cols], dtype=np.float64).copy()
        context_forest_residual = np.asarray(
            residual[context_rows, context_cols], dtype=np.float64
        ).copy()
        current = np.asarray(c1[fine_rows, fine_cols], dtype=np.float64)
        forest_residual = np.asarray(residual[fine_rows, fine_cols], dtype=np.float64)
        rock_delta = np.where(fine_owner, fine_rock, 0.0)
        composed_residual = forest_residual + rock_delta
        c1[fine_rows, fine_cols] = (current + rock_delta).astype(np.float32)
        residual[fine_rows, fine_cols] = composed_residual.astype(np.float32)
        local_ownership = ownership[fine_rows, fine_cols]
        local_ownership[fine_owner] = code
        # Rock closure is relative to the already-forested carrier, not to C0.
        blocks = rock_delta.reshape(
            row1 - row0, ROCK_FACTOR, col1 - col0, ROCK_FACTOR
        ).mean(axis=(1, 3))
        mean_error = float(
            np.max(
                np.abs(blocks[accepted_coarse] - source_patch[accepted_coarse]),
                initial=0.0,
            )
        )
        if rock_only:
            forest_overlap = None
            max_forest_distance = None
            anchor_clearance = None
        else:
            fine_forest = np.asarray(forest_allowed[fine_rows, fine_cols], dtype=bool)
            forest_overlap = float(np.mean(fine_forest[fine_owner]))
            forest_distance = ndimage.distance_transform_edt(~fine_forest) * FINE_PITCH_M
            max_forest_distance = float(np.max(forest_distance[fine_owner], initial=0.0))
            anchor_fine_row = target_center_row * 4 + 2
            anchor_fine_col = target_center_col * 4 + 2
            forest_interior = ndimage.distance_transform_edt(fine_forest) * FINE_PITCH_M
            local_anchor_row = anchor_fine_row - row0 * 4
            local_anchor_col = anchor_fine_col - col0 * 4
            anchor_clearance = float(forest_interior[local_anchor_row, local_anchor_col])
        recovered_forest = composed_residual - rock_delta
        carrier_error = float(np.max(np.abs(recovered_forest - forest_residual), initial=0.0))
        carrier_rms_before = float(np.sqrt(np.mean(forest_residual * forest_residual)))
        carrier_rms_recovered = float(np.sqrt(np.mean(recovered_forest * recovered_forest)))
        family_metrics = {
            "etak_id": int(item["etak_id"]),
            "owned_fine_cells": int(np.count_nonzero(fine_owner)),
            "owned_parent_cells": int(np.count_nonzero(accepted_coarse)),
            "parent_mean_max_error_m": mean_error,
            "source_anchor_parent_residual_m": float(source_patch[32, 32]),
            "anchor_owned": bool(accepted_coarse[32, 32]),
            "rock_delta_min_max_m": [
                float(np.min(rock_delta[fine_owner], initial=0.0)),
                float(np.max(rock_delta[fine_owner], initial=0.0)),
            ],
            "fine_bbox_rows_cols": [row0 * 4, row1 * 4, col0 * 4, col1 * 4],
        }
        if rock_only:
            family_metrics.update({
                "whole_window_base_carrier_maximum_error_m": carrier_error,
                "whole_window_base_carrier_rms_before_m": carrier_rms_before,
                "whole_window_base_carrier_rms_recovered_m": carrier_rms_recovered,
            })
        else:
            family_metrics.update({
                "rock_support_forest_overlap_fraction": forest_overlap,
                "maximum_rock_support_to_forest_distance_m": max_forest_distance,
                "anchor_forest_interior_clearance_m": anchor_clearance,
                "whole_window_forest_carrier_maximum_error_m": carrier_error,
                "whole_window_forest_carrier_rms_before_m": carrier_rms_before,
                "whole_window_forest_carrier_rms_recovered_m": carrier_rms_recovered,
            })
        metrics["families"][str(code)] = family_metrics
        context_rock_delta = np.zeros_like(context_current)
        context_rock_delta[
            context_margin:-context_margin, context_margin:-context_margin
        ] = rock_delta
        qa_records.append(
            {
                "etak_id": int(item["etak_id"]),
                "source_family": code,
                "c0": (context_current - context_forest_residual).astype(np.float32),
                "forest_carrier": context_current.astype(np.float32),
                "base_carrier": context_current.astype(np.float32),
                "composed": (context_current + context_rock_delta).astype(np.float32),
                "rock_delta": context_rock_delta.astype(np.float32),
                "recovered_forest": context_current.astype(np.float32),
                "recovered_base": context_current.astype(np.float32),
            }
        )
    c1.flush()
    residual.flush()
    ownership.flush()
    return metrics, qa_records


def _resize_rgb(rgb: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


def _shade(values: np.ndarray) -> np.ndarray:
    value = _hillshade(values, 0.5)
    return np.repeat(value[..., None], 3, axis=2)


def _native_shade(values: np.ndarray) -> np.ndarray:
    value = _hillshade(values, FINE_PITCH_M)
    return np.repeat(value[..., None], 3, axis=2)


def _band_energy(values: np.ndarray) -> np.ndarray:
    band = values - ndimage.gaussian_filter(values, sigma=8.0, mode="reflect")
    return np.sqrt(ndimage.gaussian_filter(band * band, sigma=4.0, mode="reflect"))


def _grayscale(values: np.ndarray, maximum: float) -> np.ndarray:
    scaled = np.clip(values / max(maximum, 1.0e-12), 0.0, 1.0)
    rgb = np.rint(scaled[..., None] * 255.0).astype(np.uint8)
    return np.repeat(rgb, 3, axis=2)


def _write_qa(
    qa: Path,
    c0: np.ndarray,
    c1: np.ndarray,
    ownership: np.ndarray,
    mapped: list[dict[str, Any]],
    rock_qa: list[dict[str, Any]],
    bbox: tuple[int, int, int, int],
) -> list[Path]:
    qa.mkdir(parents=True, exist_ok=True)
    shade0, shade1 = _shade(c0), _shade(c1)
    paths = [qa / "01_full_common_light_composition.png"]
    paths.extend(
        qa / f"0{index + 2}_mapped_rock_{int(item['etak_id'])}_socket_closeup.png"
        for index, item in enumerate(mapped)
    )
    transition_number = len(mapped) + 2
    paths.append(qa / f"0{transition_number}_mapped_rock_ownership_transitions.png")
    paths.append(qa / f"0{transition_number + 1}_ordinary_forest_context.png")

    canvas = Image.new("RGB", (1328, 718), (244, 242, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "01  FULL FOREST + MAPPED-ROCK COMPOSITION", fill=(20, 24, 20))
    draw.text((16, 34), "same 512 m parent and common light; mapped rock deltas add to the preserved forest carrier", fill=(70, 72, 66))
    canvas.paste(_resize_rgb(shade0, 640), (16, 62))
    canvas.paste(_resize_rgb(shade1, 640), (672, 62))
    draw.text((16, 44), "PINNED C0", fill=(20, 24, 20))
    draw.text((672, 44), "COMPOSED C1", fill=(20, 24, 20))
    canvas.save(paths[0], compress_level=9)

    centers: list[tuple[int, int]] = []
    for path, item, native in zip(
        paths[1 : 1 + len(mapped)], mapped, rock_qa, strict=True
    ):
        point_e, point_n = map(float, item["point_en"])
        col = int((point_e - bbox[0]) / 0.5)
        row = int((bbox[3] - point_n) / 0.5)
        centers.append((row, col))
        carrier = np.asarray(native["forest_carrier"], dtype=np.float64)
        composed = np.asarray(native["composed"], dtype=np.float64)
        recovered = np.asarray(native["recovered_forest"], dtype=np.float64)
        carrier_energy = _band_energy(carrier)
        recovered_energy = _band_energy(recovered)
        energy_maximum = float(
            np.percentile(np.concatenate((carrier_energy.ravel(), recovered_energy.ravel())), 99.5)
        )
        carrier_error = np.abs(recovered - carrier)
        error_maximum = max(float(np.max(carrier_error)), 1.0e-7)
        sheet = Image.new("RGB", (1248, 900), (244, 242, 234))
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), f"MAPPED ROCK {int(item['etak_id'])} / FAMILY {int(item['source_family'])}", fill=(20, 24, 20))
        draw.text((16, 34), "native 0.0625 m samples over identical 24 m context; outline is the complete 16 m source window", fill=(70, 72, 66))
        panels = (
            ("PINNED C0", _native_shade(np.asarray(native["c0"], dtype=np.float64))),
            ("FOREST CARRIER BEFORE ROCK", _native_shade(carrier)),
            ("ADDITIVE COMPOSITION", _native_shade(composed)),
            ("FOREST BAND ENERGY BEFORE", _grayscale(carrier_energy, energy_maximum)),
            ("RECOVERED CARRIER BAND ENERGY", _grayscale(recovered_energy, energy_maximum)),
            ("RECOVERED CARRIER ERROR (0.1 um white)", _grayscale(carrier_error, error_maximum)),
        )
        for panel_index, (label, rgb) in enumerate(panels):
            panel_col = panel_index % 3
            panel_row = panel_index // 3
            left = 16 + panel_col * 408
            top = 84 + panel_row * 408
            draw.text((left, top - 18), label, fill=(20, 24, 20))
            panel = _resize_rgb(rgb, 384)
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rectangle((64, 64, 319, 319), outline=(220, 45, 35), width=2)
            sheet.paste(panel, (left, top))
        sheet.save(path, compress_level=9)

    palette = np.asarray(
        [[42, 43, 39], [50, 135, 82], [218, 92, 63], [225, 155, 54], [120, 91, 177]],
        dtype=np.uint8,
    )
    transition = Image.new("RGB", (1944, 718 * len(centers)), (244, 242, 234))
    draw = ImageDraw.Draw(transition)
    for index, (row, col) in enumerate(centers):
        half = 72
        crop = np.s_[row - half : row + half, col - half : col + half]
        top = index * 718
        draw.text((16, top + 12), f"MAPPED ROCK OWNERSHIP TRANSITION {index + 1}", fill=(20, 24, 20))
        draw.text((16, top + 34), "72 x 72 m context; green forest, warm mapped body/socket, charcoal exact abstention", fill=(70, 72, 66))
        transition.paste(_resize_rgb(shade0[crop], 624), (16, top + 62))
        transition.paste(_resize_rgb(shade1[crop], 624), (656, top + 62))
        transition.paste(_resize_rgb(palette[np.clip(ownership[crop], 0, 4)], 624), (1296, top + 62))
    transition.save(paths[-2], compress_level=9)

    # Pick a forest-owned 72 m context farthest from mapped support.
    forest = ownership == 1
    rock = ownership >= 2
    distance = ndimage.distance_transform_edt(~rock)
    score = np.where(forest, distance, -1.0)
    context_row, context_col = np.unravel_index(int(np.argmax(score)), score.shape)
    context_row = int(np.clip(context_row, 72, c1.shape[0] - 72))
    context_col = int(np.clip(context_col, 72, c1.shape[1] - 72))
    crop = np.s_[context_row - 72 : context_row + 72, context_col - 72 : context_col + 72]
    forest_sheet = Image.new("RGB", (1328, 718), (244, 242, 234))
    draw = ImageDraw.Draw(forest_sheet)
    draw.text((16, 12), "ORDINARY MESIC-MINERAL FOREST CONTEXT", fill=(20, 24, 20))
    draw.text((16, 34), "72 x 72 m common-light C0/C1 context away from mapped body/socket supports", fill=(70, 72, 66))
    forest_sheet.paste(_resize_rgb(shade0[crop], 640), (16, 62))
    forest_sheet.paste(_resize_rgb(shade1[crop], 640), (672, 62))
    forest_sheet.save(paths[-1], compress_level=9)
    return paths


def _write_rock_only_qa(
    qa: Path,
    item: dict[str, Any],
    native: dict[str, Any],
) -> list[Path]:
    qa.mkdir(parents=True, exist_ok=True)
    path = qa / f"01_mapped_rock_{int(item['etak_id'])}_socket_closeup.png"
    base = np.asarray(native["base_carrier"], dtype=np.float64)
    composed = np.asarray(native["composed"], dtype=np.float64)
    delta = np.asarray(native["rock_delta"], dtype=np.float64)
    sheet = Image.new("RGB", (1248, 492), (244, 242, 234))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (16, 12),
        f"ROCK-ONLY BODY/SOCKET {int(item['etak_id'])} / FAMILY {int(item['source_family'])}",
        fill=(20, 24, 20),
    )
    draw.text(
        (16, 34),
        "native 0.0625 m samples over identical 24 m context; outline is the unchanged 16 m source window",
        fill=(70, 72, 66),
    )
    panels = (
        ("PINNED C0", _native_shade(base)),
        ("UNCHANGED FAMILY-4 DELTA", _native_shade(delta)),
        ("ADDITIVE C0 + DELTA", _native_shade(composed)),
    )
    for panel_index, (label, rgb) in enumerate(panels):
        left = 16 + panel_index * 408
        top = 84
        draw.text((left, top - 18), label, fill=(20, 24, 20))
        panel = _resize_rgb(rgb, 384)
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((64, 64, 319, 319), outline=(220, 45, 35), width=2)
        sheet.paste(panel, (left, top))
    sheet.save(path, compress_level=9)
    return [path]


def _run_rock_only(config_path: Path, config: dict[str, Any]) -> Path:
    bbox = tuple(map(int, config["bbox_en"]))
    source_path = _bound(config["rock_source"])
    recipe = {
        "schema": config["schema"] + ".recipe/1",
        "authority": config["authority"],
        "config": _identity(config_path),
        "implementation": _identity(Path(__file__).resolve()),
        "mapped_rock_source": _identity(source_path),
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "composition": "exact additive rock-only composition: C1 = pinned C0 + unchanged condition-masked family-4 delta",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = ROCK_ONLY_OUTPUT_ROOT / build_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = PinnedBaseHeight(
        BASE_MANIFEST,
        BASE_MANIFEST_SHA256,
        ASSET_GEN_ROOT / "data/out",
        load_base().encode,
        audit=False,
    )
    with TemporaryDirectory(
        prefix=f".{build_id}.", dir=destination.parent, delete=False
    ) as temporary:
        staging = Path(temporary)
        surface, qa, scratch = staging / "surface", staging / "qa", staging / "scratch"
        surface.mkdir()
        qa.mkdir()
        scratch.mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        evidence = _target_evidence(config)
        rock_condition, pile_mask, mask_evidence = _rock_only_masks(
            bbox,
            scratch / "rock_condition.u8",
            scratch / "pile.u8",
            _pile_points(config),
            float(config["pile_exclusion_radius_m"]),
        )
        residual_path = scratch / "residual.f32"
        residual = np.memmap(
            residual_path,
            dtype=np.float32,
            mode="w+",
            shape=(FINE_CELLS, FINE_CELLS),
        )
        residual[:] = 0.0
        residual.flush()
        c1, authority = _write_c1(
            base=base,
            bbox=bbox,
            residual_path=residual_path,
            destination=surface / "c1_height_f32.npy",
        )
        ownership = np.lib.format.open_memmap(
            surface / "composition_ownership_u8.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(FINE_CELLS, FINE_CELLS),
        )
        ownership[:] = 0
        rock_metrics, rock_qa = _compose_rocks(
            c1=c1,
            residual=residual,
            ownership=ownership,
            rock_condition=rock_condition,
            forest_allowed=rock_condition,
            source_path=source_path,
            mapped=config["mapped_boulders"],
            bbox=bbox,
            rock_only=True,
        )
        maximum_abstained = 0.0
        maximum_pile = 0.0
        residual_square_sum = 0.0
        residual_min = float("inf")
        residual_max = float("-inf")
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
            residual_min = min(residual_min, float(np.min(values)))
            residual_max = max(residual_max, float(np.max(values)))
        pngs = _write_rock_only_qa(qa, config["mapped_boulders"][0], rock_qa[0])
        acceptance = config["acceptance"]
        failures = []
        if maximum_abstained > acceptance["maximum_abstained_residual_m"]:
            failures.append("abstained residual")
        if maximum_pile > acceptance["maximum_boulder_pile_residual_m"]:
            failures.append("boulder-pile residual")
        for family in rock_metrics["families"].values():
            if family["owned_fine_cells"] < acceptance["minimum_rock_owned_cells_per_family"]:
                failures.append(f"empty mapped family {family['etak_id']}")
            if family["parent_mean_max_error_m"] > acceptance["maximum_rock_parent_mean_error_m"]:
                failures.append(f"mapped family mean {family['etak_id']}")
            if not family["anchor_owned"]:
                failures.append(f"mapped anchor rejected {family['etak_id']}")
            if (
                family["whole_window_base_carrier_maximum_error_m"]
                > acceptance["maximum_whole_window_base_carrier_error_m"]
            ):
                failures.append(f"base carrier preservation {family['etak_id']}")
        metrics = {
            "bbox_en": list(bbox),
            "shape": [FINE_CELLS, FINE_CELLS],
            "target_evidence": evidence,
            "mask_context_only_no_forest_authority": mask_evidence,
            "maximum_abstained_residual_m": maximum_abstained,
            "maximum_boulder_pile_residual_m": maximum_pile,
            "rock": rock_metrics,
            "composed_residual_rms_m": float(
                np.sqrt(residual_square_sum / residual.size)
            ),
            "composed_residual_min_max_m": [residual_min, residual_max],
            "authority_p01_p99_m": [
                float(value) for value in np.percentile(authority, [1, 99])
            ],
        }
        (staging / "metrics.json").write_bytes(_canonical_json(metrics) + b"\n")
        del c1, residual, ownership, rock_condition, pile_mask
        shutil.rmtree(scratch)
        qa_index = {
            "schema": config["schema"] + ".qa/1",
            "build_id": build_id,
            "images": [
                {
                    "path": f"qa/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "dimensions_px": list(Image.open(path).size),
                }
                for path in pngs
            ],
            "interpretation": [
                "01 is the only visual gate: native 0.0625 m pinned C0, unchanged family-4 delta, and additive result over the same 24 m context.",
                "The red outline is the complete unchanged 16 m source window; this artifact claims no forest transfer.",
            ],
        }
        (qa / "index.json").write_bytes(_canonical_json(qa_index) + b"\n")
        files = {
            str(path.relative_to(staging)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": config["schema"] + ".artifact/1",
            "build_id": build_id,
            "status": "inspect_float_preview" if not failures else "park_before_inspection",
            "failures": failures,
            "metrics": metrics,
            "authority": {
                "production": False,
                "packing": False,
                "browser": False,
                "latest": False,
                "national": False,
                "forest_transfer": False,
            },
            "ownership": {
                "0": "exact pinned C0 abstention",
                "4": "unchanged mapped family-4 body/socket delta at ETAK 1145648",
            },
            "limitations": [
                "This is one bounded rock-only float checkpoint at one ETAK type-10 stone with recorded height 2 m.",
                "Family 4 retains its source amplitude, scale, orientation, and 0.25 m authority; 0.0625 m samples are structural reconstruction only.",
                "Forest classification is context evidence only: no forest residual is synthesized and no forest transfer gate is weakened or claimed.",
                "Mapped type-20 piles, water, buildings, paved roads, and cliff context receive no residual.",
                "No packing, browser, runtime, format, production, latest, or national authority is granted.",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    return destination


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    if config["schema"] == ROCK_ONLY_SCHEMA:
        return _run_rock_only(config_path, config)
    bbox = tuple(map(int, config["bbox_en"]))
    source_path = _bound(config["rock_source"])
    recipe = {
        "schema": config["schema"] + ".recipe/1",
        "authority": config["authority"],
        "config": _identity(config_path),
        "implementation": _identity(Path(__file__).resolve()),
        "locked_forest_implementation": _identity(Path(__file__).with_name("adjacent.py")),
        "mapped_rock_source": _identity(source_path),
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "composition": "exact additive cross-scale composition: C1 = C0 + locked forest residual + condition-masked mapped-rock delta",
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
        rock_metrics, rock_qa = _compose_rocks(
            c1=c1,
            residual=residual,
            ownership=ownership,
            rock_condition=rock_condition,
            forest_allowed=forest_allowed,
            source_path=source_path,
            mapped=config["mapped_boulders"],
            bbox=bbox,
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
        pngs = _write_qa(
            qa,
            c0_half,
            c1_half,
            ownership_half,
            config["mapped_boulders"],
            rock_qa,
            bbox,
        )
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
            if (
                family["whole_window_forest_carrier_maximum_error_m"]
                > acceptance["maximum_whole_window_forest_carrier_error_m"]
            ):
                failures.append(f"forest carrier preservation {family['etak_id']}")
            if not family["anchor_owned"]:
                failures.append(f"mapped anchor rejected {family['etak_id']}")
            if "minimum_rock_support_forest_overlap_fraction" in acceptance:
                if family["rock_support_forest_overlap_fraction"] < acceptance["minimum_rock_support_forest_overlap_fraction"]:
                    failures.append(f"mapped support forest overlap {family['etak_id']}")
                if family["maximum_rock_support_to_forest_distance_m"] > acceptance["maximum_rock_support_to_forest_distance_m"]:
                    failures.append(f"mapped support forest distance {family['etak_id']}")
                if family["anchor_forest_interior_clearance_m"] < acceptance["minimum_anchor_forest_interior_clearance_m"]:
                    failures.append(f"mapped anchor forest clearance {family['etak_id']}")
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
            "schema": config["schema"] + ".qa/1",
            "build_id": build_id,
            "images": [
                {"path": f"qa/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256(path), "dimensions_px": list(Image.open(path).size)}
                for path in pngs
            ],
            "interpretation": [
                "01 compares the complete pinned and composed parent under identical common light.",
                f"02-{len(config['mapped_boulders']) + 1:02d} are native-pitch 24 m same-position C0/carrier/composed and carrier-band preservation sheets; each outlines the complete 16 m source window.",
                f"{len(config['mapped_boulders']) + 2:02d} shows bounded mapped-rock ownership and the surrounding forest transition for every mapped form.",
                f"{len(config['mapped_boulders']) + 3:02d} is an ordinary forest C0/C1 context away from mapped forms.",
            ],
        }
        (qa / "index.json").write_bytes(_canonical_json(qa_index) + b"\n")
        files = {
            str(path.relative_to(staging)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema": config["schema"] + ".artifact/1",
            "build_id": build_id,
            "status": "ready_to_pack" if not failures else "park_before_pack",
            "failures": failures,
            "metrics": metrics,
            "authority": {"production": False, "packing": False, "browser": False, "latest": False, "national": False},
            "ownership": {
                "0": "exact C0 abstention",
                "1": "locked mesic-mineral forest",
                **{
                    str(int(item["source_family"])): f"mapped body/socket {int(item['etak_id'])}"
                    for item in config["mapped_boulders"]
                },
            },
            "limitations": [
                f"This composes one accepted forest research owner with {len(config['mapped_boulders'])} bounded mapped-rock forms; it establishes no generic till owner.",
                "Mapped rock deltas retain 0.25 m source authority, use structural reconstruction only below that pitch, and add to rather than replace the locked forest carrier.",
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

"""Strict condition and semantics evidence for the unsealed Hovi development plot."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ...config import CONFIG_DIR, DATA_WORK
from .records import canonical_json_bytes, sha256_bytes, sha256_file

_SCHEMA = "hovi-condition-semantics-evidence/1.0.0"
_SELECTION_SCHEMA = "hovi-public-target-selection/1.0.0"
_SELECTION_ID = "hovi-2024-jarvselja-hyytiala-first-conversion-v1"
_RETAINED_SCHEMA = "hovi-retained-evidence/1.0.0"
_PLOT_ID = "HY_SPRUCE4"
_AUTHORIZED_SCOPE = "shared-and-hy-spruce4-only"
_REQUIRED_TRANCHES = ("shared", "hy-spruce4-photos")

_OVERVIEW_HEADER = (
    "plot_ID", "lon", "lat", "x_UTM", "y_UTM", "zone_UTM", "elevation",
    "year", "protocol_forinv", "protocol_TLS", "protocol_canopytransm",
    "date_forinv", "date_hemiphoto", "date_TLS", "date_fractcover",
    "date_forestfloor", "date_canopytransm",
)
_TREE_HEADER = (
    "plot_ID", "tree_ID", "x_local", "y_local", "x_UTM", "y_UTM", "species",
    "visibility", "status_DEAD", "status_STEM_BROKEN", "status_TOP_BROKEN",
    "status_TOP_DEAD", "status_BROKEN_AND_REGROWN", "status_NONVIGOROUS",
    "status_LEANING_OR_BENT", "status_STEM_DAMAGES", "dist", "azim",
    "stem_diameter", "height",
)
_SUMMARY_HEADER = (
    "plot_ID", "basal_area", "stem_diameter", "height", "n_trees",
    "p_broadleaved", "p_AB_AL", "p_AC_CA", "p_AC_PL", "p_AL_GL", "p_AL_IN",
    "p_AL_sp", "p_BE_sp", "p_BE_sp_co", "p_CA_BE", "p_CO_AV", "p_CR_sp",
    "p_FA_SY", "p_FR_AL", "p_FR_EX", "p_FR_EX_co", "p_FR_sp", "p_PI_AB",
    "p_PI_SY", "p_PO_TR", "p_PY_sp", "p_QU_RO", "p_QU_sp", "p_SA_CA",
    "p_SA_sp", "p_SO_AU", "p_TI_CO", "p_TI_CO_co", "p_UL_sp",
)
_FLOOR_HEADER = (
    "plot_ID", "vasc_mean", "nonvasc_mean", "lichen_mean", "intactlitt_mean",
    "decomplitt_mean", "vasc_q1", "vasc_q2", "vasc_q3", "vasc_q4",
    "nonvasc_q1", "nonvasc_q2", "nonvasc_q3", "nonvasc_q4", "lichen_q1",
    "lichen_q2", "lichen_q3", "lichen_q4", "intactlitt_q1", "intactlitt_q2",
    "intactlitt_q3", "intactlitt_q4", "decomplitt_q1", "decomplitt_q2",
    "decomplitt_q3", "decomplitt_q4",
)
_TLS_HEADER = (
    "plot_ID", "t_xy_00", "t_xy_10", "t_xy_20", "t_xy_01", "t_xy_11",
    "t_xy_21", "scan_resolution",
)
_CSV_KINDS = {
    "dataset_overview": _OVERVIEW_HEADER,
    "forest_inventory_plot_data": _TREE_HEADER,
    "forest_inventory_summary": _SUMMARY_HEADER,
    "forest_floor_fractional_cover": _FLOOR_HEADER,
    "tls_campaign_metadata": _TLS_HEADER,
}
_PHOTO_KINDS = {
    "semantic_quadrat_photo",
    "context_overview_photo",
    "semantic_transect_photo",
}
_TREE_INTEGER_FIELDS = {
    "tree_ID", "visibility", "status_DEAD", "status_STEM_BROKEN",
    "status_TOP_BROKEN", "status_TOP_DEAD", "status_BROKEN_AND_REGROWN",
    "status_NONVIGOROUS", "status_LEANING_OR_BENT", "status_STEM_DAMAGES",
}
_TREE_NULLABLE_FIELDS = {"height"}
_STATUS_FIELDS = {
    "status_DEAD", "status_STEM_BROKEN", "status_TOP_BROKEN", "status_TOP_DEAD",
    "status_BROKEN_AND_REGROWN", "status_NONVIGOROUS", "status_LEANING_OR_BENT",
    "status_STEM_DAMAGES",
}
_FLOOR_COMPONENTS = {
    "vascular_plants": "vasc",
    "nonvascular_plants_mosses": "nonvasc",
    "lichen": "lichen",
    "intact_plant_litter": "intactlitt",
    "decomposed_plant_litter": "decomplitt",
}

_UNRESOLVED_FACTORS: dict[str, dict[str, str]] = {
    "substrate_and_vertical_soil_profile": {
        "bedrock_formation_and_lithology": "No bedrock observation or mapped geology is paired to this plot.",
        "surficial_deposit": "No surficial-geology observation is paired to this plot.",
        "geomorphological_landform_and_forming_process": "The retained tables do not classify landform or forming process.",
        "soil_class_and_component_mixture": "No soil-map polygon, field soil class, or component shares are supplied.",
        "vertical_soil_horizons_and_thicknesses": "No soil profile or horizon depths are supplied.",
        "mineral_texture_and_erodibility": "Forest-floor cover does not identify mineral-soil texture or erodibility.",
        "stoniness_and_clast_size_distribution": "No clast or stoniness measurement is supplied.",
        "organic_horizon_thickness_and_peat_depth": "Cover fractions do not measure organic-horizon thickness or peat depth.",
        "exposed_mineral_or_rock_fraction": "The five cover classes do not label exposed mineral soil or rock.",
    },
    "hydrology_and_flow": {
        "soil_moisture_at_each_observation": "No volumetric moisture or qualitative moisture state is supplied.",
        "water_table_and_saturation_state": "No water-table, saturation, or peatland-state observation is supplied.",
        "surface_and_subsurface_flow": "No flow direction, discharge, seep, spring, or channel state is supplied.",
        "drainage_network_and_ditches": "No registered drainage or ditch inventory is paired to the plot.",
        "contributing_area_and_wetness_proxy": "No terrain-conditioned contributing area or wetness field is supplied.",
        "flood_or_inundation_history": "No inundation frequency, water level, or dated flood state is supplied.",
    },
    "multiscale_terrain_and_structure_context": {
        "slope_aspect_curvature_at_1_4_16_64_256m": "No registered DTM-derived multiscale terrain fields are paired to the plot.",
        "topographic_position": "No ridge, hollow, toe, flat, bank, or other position is classified.",
        "distance_and_side_to_water_shore_bank_escarpment": "No typed water or escarpment geometry is paired to the plot.",
        "distance_and_side_to_road_track_ditch_or_object": "No typed infrastructure or object geometry is paired to the plot.",
        "physical_domain_and_transition_halo": "No stand boundary or physically justified inference halo is supplied.",
    },
    "biological_and_organic_state": {
        "stand_age_and_age_structure": "Tree inventory and TLS date do not establish stand age.",
        "understory_species_and_structure": "Vascular cover is not a species or three-dimensional understory inventory.",
        "root_geometry_and_root_heave": "No root semantic labels or root geometry are supplied.",
        "deadwood_log_geometry_and_decay": "Tree status flags do not inventory downed deadwood, log geometry, or decay class.",
        "moss_litter_thickness_and_load": "Areal cover fractions do not measure material thickness or mass.",
        "pit_mound_root_plate_events_and_age": "No windthrow event, pit/mound, root-plate, overlap, or decay-age labels are supplied.",
        "canopy_occlusion_and_ground_visibility": "No per-cell visibility, ray, or occlusion state is supplied by the shared tables.",
    },
    "management_and_disturbance": {
        "forest_management_history": "No thinning, harvest, planting, or silvicultural history is supplied.",
        "operation_and_disturbance_dates": "No dated management or disturbance event is supplied.",
        "machine_traffic_ruts_and_skid_trails": "No machine-operation or trail semantic inventory is supplied.",
        "wind_fire_frost_and_biotic_disturbance": "No causal disturbance type or severity is supplied.",
        "disturbance_age_and_recovery_state": "No event age or recovery trajectory is supplied.",
    },
    "observation_support_error_and_registration": {
        "per_cell_independent_support": "The shared tables contain no per-cell point or independent-view support.",
        "visibility_interpolation_and_confidence_masks": "No registered visibility, interpolation, or confidence masks are supplied.",
        "horizontal_and_vertical_error_distributions": "No independent metric error distributions are supplied.",
        "mtf_or_transfer_function": "Nominal pulse pitch is not a measured terrain transfer function.",
        "per_point_sensor_view_and_incidence_geometry": "The shared affine transform does not retain per-return view or incidence geometry.",
        "weather_and_illumination": "No weather, precipitation, wind, or illumination record is paired to TLS or photos.",
        "measurement_season_state": "Dates are observed, but no phenological, snow, freeze, or leaf state is supplied.",
        "registration_to_dtm_orthophoto_etak_soil_geology_hydrology_management": "No temporally paired condition-source registration is supplied.",
        "human_qa_masks": "No human-reviewed ground, contamination, support, or ambiguity masks are supplied.",
        "full_scan_disagreement_and_thinning_damage": "The selected merged preview cannot expose per-scan disagreement or thinning loss.",
    },
    "surface_semantics_and_heightfield_representability": {
        "per_point_or_per_cell_surface_class": "Plot-level cover fractions are not geometric semantic labels.",
        "mineral_soil_vs_stable_organic_surface": "No geometric ownership boundary separates mineral ground and stable organic surface.",
        "peat_moss_roots_clasts_litter_deadwood_vegetation_water_objects": "Required target classes are not labeled in the TLS or registered to photos.",
        "stable_surface_include_exclude_policy": "No qualified policy decides which roots, litter, deadwood, or clasts belong to terrain.",
        "non_heightfield_and_overlap_masks": "No mask identifies vertical, undercut, detached, or overlapping geometry.",
        "water_and_hard_object_forbidden_masks": "No registered water, building, paved, or discrete-object masks are supplied.",
        "target_scale_ground_surface": "No classified or reconstructed 0.03-0.05 m ground/stable-surface target exists yet.",
    },
    "physical_analogue_and_estonia_ood": {
        "substrate_and_soil_profile_match": "The source condition is unknown, so Estonia substrate/soil equivalence cannot be tested.",
        "climate_and_freeze_thaw_match": "No checked climate or freeze-thaw comparison to the Estonia envelope is supplied.",
        "hydrology_match": "Unknown source hydrology prevents an Estonia hydrology match.",
        "vegetation_and_organic_surface_match": "Tree and cover observations are partial and do not establish full condition equivalence.",
        "land_use_management_and_disturbance_match": "Source management and disturbance states are unknown.",
        "forming_process_and_morphology_scale_match": "No qualified morphology or forming-process comparison is supplied.",
        "source_observation_process_match": "Protocol metadata exists, but support, semantics, and error remain unqualified.",
        "estonia_condition_envelope_and_ood_bounds": "No Estonia condition envelope or condition-stratified OOD bound has been evaluated.",
        "independent_site_and_campaign_release_gate": "One development plot from one site/campaign cannot satisfy the release gate.",
    },
}


def _unknown(reason: str, *, relevance: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "ood_status": "unresolved",
        "eligibility_result": "abstain",
        "inference_allowed": False,
        "relevance": relevance,
        "reason": reason,
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi {label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hovi {label} must be a non-empty string")
    return value


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"Hovi {label} must be numeric, got {value!r}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"Hovi {label} must be finite")
    return parsed


def _integer(value: str, label: str) -> int:
    parsed = _finite_float(value, label)
    if not parsed.is_integer():
        raise ValueError(f"Hovi {label} must be an integer")
    return int(parsed)


def _date_or_none(value: str, label: str) -> str | None:
    if value == "NA":
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Hovi {label} is not an ISO date: {value!r}") from None
    return value


def _read_csv(path: Path, header: tuple[str, ...], label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != header:
            raise ValueError(f"Hovi {label} CSV schema changed")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Hovi {label} CSV contains malformed rows")
    return rows


def _one_plot_row(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    matches = [row for row in rows if row["plot_ID"] == _PLOT_ID]
    if len(matches) != 1:
        raise ValueError(f"Hovi {label} must contain exactly one {_PLOT_ID} row")
    return matches[0]


def _strict_retained_artifacts(
    retained_path: Path,
    selection_raw: Mapping[str, Any],
    selection_sha256: str,
    plot_raw: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    retained_path = retained_path.resolve()
    if retained_path.name != "retained.json" or not retained_path.is_file():
        raise ValueError("Hovi conditions require an existing retained.json")
    retained = _require_mapping(json.loads(retained_path.read_bytes()), "retained manifest")
    retained_selection = _require_mapping(retained.get("selection"), "retained selection")
    plan_identity = _require_mapping(retained.get("plan_identity"), "retained plan_identity")
    retention_id = _required_string(retained.get("retention_id"), "retention_id")
    if (
        len(retention_id) != 64
        or any(character not in "0123456789abcdef" for character in retention_id)
        or retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("plan_sha256") != retention_id
        or retained_path.parent.name != retention_id
        or sha256_bytes(canonical_json_bytes(plan_identity).rstrip(b"\n")) != retention_id
        or retained_selection.get("id") != selection_raw["id"]
        or retained_selection.get("config_sha256") != selection_sha256
        or retained.get("authorized_scope") != _AUTHORIZED_SCOPE
    ):
        raise ValueError("Hovi retained manifest identity changed")
    completed_tranches = retained.get("completed_tranches")
    if (
        not isinstance(completed_tranches, list)
        or completed_tranches[: len(_REQUIRED_TRANCHES)] != list(_REQUIRED_TRANCHES)
    ):
        raise ValueError("Hovi conditions require verified shared and HY_SPRUCE4 photo tranches")

    expected_raw = list(selection_raw["shared_files"]) + [
        item for item in plot_raw["files"] if item["kind"] in _PHOTO_KINDS
    ]
    expected_by_id = {item["file_id"]: item for item in expected_raw}
    if len(expected_by_id) != len(expected_raw):
        raise ValueError("Hovi condition-source selection contains duplicate file IDs")
    files = retained.get("files")
    if not isinstance(files, list):
        raise ValueError("Hovi retained manifest files must be an array")
    root = retained_path.parent.resolve()
    verified: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping) or item.get("status") != "verified":
            continue
        file_id = item.get("file_id")
        expected = expected_by_id.get(file_id)
        if expected is None:
            continue
        expected_plot = None if expected in selection_raw["shared_files"] else _PLOT_ID
        expected_tranche = "shared" if expected_plot is None else "hy-spruce4-photos"
        if (
            item.get("dataset_uuid") != selection_raw["source"]["dataset_uuid"]
            or any(
                item.get(key) != expected[key]
                for key in ("path", "bytes", "sha256", "kind")
            )
        ):
            raise ValueError(f"Hovi retained tuple changed for {file_id}")
        if item.get("plot_id") != expected_plot or item.get("tranche") != expected_tranche:
            raise ValueError(f"Hovi retained source scope changed for {file_id}")
        source_path = PurePosixPath(expected["path"])
        expected_relative = Path("files", *source_path.parts[1:])
        relative = Path(_required_string(item.get("relative_path"), "relative_path"))
        local_path = (root / relative).resolve()
        if (
            relative != expected_relative
            or relative.is_absolute()
            or not local_path.is_relative_to(root)
            or not local_path.is_file()
            or local_path.stat().st_size != expected["bytes"]
            or sha256_file(local_path) != expected["sha256"]
        ):
            raise ValueError(f"Hovi retained source failed verification: {expected['path']}")
        verified[expected["kind"] + ":" + file_id] = {
            "file_id": file_id,
            "path": expected["path"],
            "kind": expected["kind"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
            "local_path": local_path,
        }
    if len(verified) != len(expected_raw):
        missing = sorted(set(expected_by_id) - {item["file_id"] for item in verified.values()})
        raise ValueError(f"Hovi retained condition sources are incomplete: {missing}")
    return retention_id, verified


def _artifact_by_kind(
    artifacts: Mapping[str, dict[str, Any]], kind: str
) -> dict[str, Any]:
    matches = [item for item in artifacts.values() if item["kind"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Hovi conditions require exactly one retained {kind}")
    return matches[0]


def _parse_overview(row: dict[str, str]) -> dict[str, Any]:
    return {
        "plot_ID": row["plot_ID"],
        "lon": _finite_float(row["lon"], "overview lon"),
        "lat": _finite_float(row["lat"], "overview lat"),
        "x_UTM": _finite_float(row["x_UTM"], "overview x_UTM"),
        "y_UTM": _finite_float(row["y_UTM"], "overview y_UTM"),
        "zone_UTM": _required_string(row["zone_UTM"], "overview zone_UTM"),
        "elevation": _finite_float(row["elevation"], "overview elevation"),
        "year": _integer(row["year"], "overview year"),
        "protocol_forinv": _integer(row["protocol_forinv"], "protocol_forinv"),
        "protocol_TLS": _integer(row["protocol_TLS"], "protocol_TLS"),
        "protocol_canopytransm": _integer(
            row["protocol_canopytransm"], "protocol_canopytransm"
        ),
        "date_forinv": _date_or_none(row["date_forinv"], "date_forinv"),
        "date_hemiphoto": _date_or_none(row["date_hemiphoto"], "date_hemiphoto"),
        "date_TLS": _date_or_none(row["date_TLS"], "date_TLS"),
        "date_fractcover": _date_or_none(row["date_fractcover"], "date_fractcover"),
        "date_forestfloor": _date_or_none(row["date_forestfloor"], "date_forestfloor"),
        "date_canopytransm": _date_or_none(row["date_canopytransm"], "date_canopytransm"),
    }


def _parse_tree_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["plot_ID"] == _PLOT_ID]
    if not selected:
        raise ValueError(f"Hovi tree inventory contains no {_PLOT_ID} records")
    parsed: list[dict[str, Any]] = []
    for row in selected:
        record: dict[str, Any] = {"plot_ID": row["plot_ID"]}
        for field in _TREE_HEADER[1:]:
            raw = row[field]
            if field == "species":
                if not raw.strip():
                    raise ValueError("Hovi tree species code must be non-empty")
                record["species"] = raw
                record["species_normalized"] = raw.strip()
            elif raw == "NA":
                if field not in _TREE_NULLABLE_FIELDS:
                    raise ValueError(f"Hovi tree field {field} is unexpectedly NA")
                record[field] = None
            elif field in _TREE_INTEGER_FIELDS:
                record[field] = _integer(raw, f"tree {field}")
            else:
                record[field] = _finite_float(raw, f"tree {field}")
        if record["visibility"] not in {1, 2, 3}:
            raise ValueError("Hovi tree visibility must be 1, 2, or 3")
        if any(record[field] not in {0, 1} for field in _STATUS_FIELDS):
            raise ValueError("Hovi tree status flags must be binary")
        parsed.append(record)
    tree_ids = [item["tree_ID"] for item in parsed]
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("Hovi tree inventory contains duplicate tree_ID values")
    return parsed


def _parse_summary(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {"plot_ID": row["plot_ID"]}
    for field in _SUMMARY_HEADER[1:]:
        parsed[field] = (
            _integer(row[field], f"summary {field}")
            if field == "n_trees"
            else _finite_float(row[field], f"summary {field}")
        )
    return parsed


def _parse_floor(row: dict[str, str]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for semantic_name, prefix in _FLOOR_COMPONENTS.items():
        mean = _finite_float(row[prefix + "_mean"], prefix + "_mean")
        quadrats = [
            _finite_float(row[f"{prefix}_q{index}"], f"{prefix}_q{index}")
            for index in range(1, 5)
        ]
        if any(value < 0.0 or value > 1.0 for value in [mean, *quadrats]):
            raise ValueError(f"Hovi fractional cover {prefix} is outside [0, 1]")
        if abs(mean - sum(quadrats) / 4.0) > 1e-12:
            raise ValueError(f"Hovi fractional cover mean changed for {prefix}")
        components[semantic_name] = {
            "mean_fraction": mean,
            "quadrat_fractions": quadrats,
            "source_prefix": prefix,
        }
    for index in range(4):
        if abs(sum(item["quadrat_fractions"][index] for item in components.values()) - 1.0) > 1e-12:
            raise ValueError(f"Hovi quadrat {index + 1} cover fractions no longer sum to one")
    if abs(sum(item["mean_fraction"] for item in components.values()) - 1.0) > 1e-12:
        raise ValueError("Hovi mean forest-floor cover fractions no longer sum to one")
    return {
        "method_scope": "plot-level areal fractions estimated from four 1m x 1m quadrat photographs",
        "geometric_semantics": False,
        "components": components,
    }


def _parse_tls(row: dict[str, str]) -> dict[str, Any]:
    values = {
        field: _finite_float(row[field], f"TLS {field}") for field in _TLS_HEADER[1:-1]
    }
    resolution = _integer(row["scan_resolution"], "TLS scan_resolution")
    return {
        "published_xy_transform": values,
        "homogeneous_xy_transform": [
            [values["t_xy_00"], values["t_xy_10"], values["t_xy_20"]],
            [values["t_xy_01"], values["t_xy_11"], values["t_xy_21"]],
            [0.0, 0.0, 1.0],
        ],
        "scan_resolution_mm_at_100m": resolution,
    }


def _source_record(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: artifact[key] for key in ("file_id", "path", "kind", "bytes", "sha256")
    }


def _photo_inventory(
    artifacts: Mapping[str, dict[str, Any]], fractional_cover_date: str
) -> list[dict[str, Any]]:
    photos = sorted(
        (item for item in artifacts.values() if item["kind"] in _PHOTO_KINDS),
        key=lambda item: item["path"],
    )
    if len(photos) != 10:
        raise ValueError("Hovi conditions require exactly ten HY_SPRUCE4 photos")
    result: list[dict[str, Any]] = []
    for item in photos:
        if item["kind"] == "semantic_quadrat_photo":
            capture_date: dict[str, Any] = {
                "status": "observed",
                "value": fractional_cover_date,
                "source_field": "overview.csv:date_fractcover",
            }
            interpretation = (
                "source photograph for quadrat fractional-cover estimation; no metric or "
                "per-pixel terrain semantic labels"
            )
        else:
            capture_date = _unknown(
                "No retained source assigns a capture date to this specific overview/transect image.",
                relevance="observation_metadata",
            )
            interpretation = "visual plot context only; no metric registration or terrain labels"
        result.append(
            {
                **_source_record(item),
                "capture_date": capture_date,
                "interpretation": interpretation,
                "inspected_for_condition_inference": False,
            }
        )
    return result


def _cross_check_selection(
    plot: Mapping[str, Any], overview: Mapping[str, Any], floor: Mapping[str, Any], tls: Mapping[str, Any]
) -> None:
    center = _require_mapping(plot.get("center"), "plot center")
    epsg4326 = _require_mapping(center.get("epsg4326"), "EPSG:4326 center")
    epsg25835 = _require_mapping(center.get("epsg25835"), "EPSG:25835 center")
    scan_protocol = _require_mapping(plot.get("scan_protocol"), "scan protocol")
    if (
        plot.get("plot_id") != _PLOT_ID
        or plot.get("role") != "development_primary"
        or plot.get("sealed_until_converter_freeze") is not False
        or plot.get("site_id") != "hovi.hyytiala"
        or plot.get("campaign_id") != "hovi.hyytiala.2019"
        or plot.get("country") != "Finland"
        or overview["lon"] != epsg4326.get("longitude")
        or overview["lat"] != epsg4326.get("latitude")
        or overview["x_UTM"] != epsg25835.get("easting_m")
        or overview["y_UTM"] != epsg25835.get("northing_m")
        or overview["date_TLS"] != plot.get("tls_date")
        or overview["date_fractcover"] != plot.get("forest_floor_photo_date")
        or tls["scan_resolution_mm_at_100m"] != scan_protocol.get("published_pitch_mm_at_100m")
    ):
        raise ValueError("Hovi selection metadata disagrees with retained shared CSVs")
    selected_floor = _require_mapping(plot.get("forest_floor_fraction"), "forest floor fraction")
    source_by_selected_name = {
        "vascular": "vascular_plants",
        "nonvascular": "nonvascular_plants_mosses",
        "lichen": "lichen",
        "intact_litter": "intact_plant_litter",
        "decomposed_litter": "decomposed_plant_litter",
    }
    for selected_name, source_name in source_by_selected_name.items():
        if selected_floor.get(selected_name) != floor["components"][source_name]["mean_fraction"]:
            raise ValueError(f"Hovi selection fractional cover changed for {selected_name}")


def emit_hovi_conditions(
    retained_path: Path,
    plot_id: str,
    *,
    selection_path: Path = CONFIG_DIR / "hovi-public-targets.json",
    work_root: Path = DATA_WORK,
) -> Path:
    """Emit immutable condition evidence for HY_SPRUCE4 without inferring missing state."""
    if plot_id != _PLOT_ID:
        raise ValueError("this slice supports HY_SPRUCE4 only; blind plots must remain uninspected")
    selection_bytes = selection_path.read_bytes()
    selection = _require_mapping(json.loads(selection_bytes), "selection")
    selection_sha256 = sha256_bytes(selection_bytes)
    qualification = _require_mapping(selection.get("qualification"), "qualification")
    source = _require_mapping(selection.get("source"), "source identity")
    license_record = _require_mapping(selection.get("license"), "license identity")
    if (
        selection.get("schema_version") != _SELECTION_SCHEMA
        or selection.get("id") != _SELECTION_ID
        or selection.get("status") != "raw_candidate_unqualified"
        or source.get("dataset_uuid") != "ace2a123-00ff-4944-951e-eddbe209b70c"
        or source.get("dataset_doi")
        != "10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9"
        or license_record.get("spdx") != "CC-BY-4.0"
        or license_record.get("url")
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
        or qualification.get("role") != "raw_candidate"
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("transfer_ceiling") != "none"
        or qualification.get("target_truth") is not False
        or qualification.get("synthesis_authorized") is not False
    ):
        raise ValueError("Hovi condition evidence requires the frozen unqualified selection")
    plots = selection.get("plots")
    if not isinstance(plots, list):
        raise ValueError("Hovi selection plots must be an array")
    matches = [plot for plot in plots if isinstance(plot, Mapping) and plot.get("plot_id") == _PLOT_ID]
    if len(matches) != 1:
        raise ValueError("Hovi selection must contain exactly one HY_SPRUCE4 plot")
    plot = matches[0]

    retention_id, artifacts = _strict_retained_artifacts(
        retained_path, selection, selection_sha256, plot
    )
    csv_rows = {
        kind: _read_csv(item["local_path"], header, kind)
        for kind, header in _CSV_KINDS.items()
        for item in [_artifact_by_kind(artifacts, kind)]
    }
    overview = _parse_overview(_one_plot_row(csv_rows["dataset_overview"], "overview"))
    trees = _parse_tree_rows(csv_rows["forest_inventory_plot_data"])
    summary = _parse_summary(
        _one_plot_row(csv_rows["forest_inventory_summary"], "forest inventory summary")
    )
    floor = _parse_floor(
        _one_plot_row(csv_rows["forest_floor_fractional_cover"], "fractional cover")
    )
    tls = _parse_tls(_one_plot_row(csv_rows["tls_campaign_metadata"], "TLS metadata"))
    _cross_check_selection(plot, overview, floor, tls)

    scan_protocol = _require_mapping(plot.get("scan_protocol"), "scan protocol")
    unknowns = {
        group: {
            field: _unknown(reason, relevance=group)
            for field, reason in fields.items()
        }
        for group, fields in _UNRESOLVED_FACTORS.items()
    }
    record = {
        "schema_version": _SCHEMA,
        "evidence_kind": "condition_and_semantics_raw_candidate",
        "selection": {
            "schema_version": selection["schema_version"],
            "id": selection["id"],
            "config_sha256": selection_sha256,
        },
        "retention": {
            "retention_id": retention_id,
            "required_completed_tranches": list(_REQUIRED_TRANCHES),
        },
        "dataset": {
            "title": source["title"],
            "dataset_doi": source["dataset_doi"],
            "dataset_uuid": source["dataset_uuid"],
            "dataset_version": source["dataset_version"],
            "published_revision": source["published_revision"],
            "data_paper_doi": source["data_paper_doi"],
            "license": dict(license_record),
        },
        "qualification": {
            "role": "raw_candidate",
            "qualification_status": "unqualified",
            "target_truth": False,
            "transfer_ceiling": "none",
            "synthesis_authorized": False,
            "eligibility_result": "abstain",
        },
        "plot_identity": {
            "plot_id": _PLOT_ID,
            "selection_role": plot["role"],
            "site_id": plot["site_id"],
            "campaign_id": plot["campaign_id"],
            "country": plot["country"],
            "coordinates": {
                "selection_epsg4326": dict(plot["center"]["epsg4326"]),
                "selection_epsg25835": dict(plot["center"]["epsg25835"]),
                "published_overview_utm_zone": overview["zone_UTM"],
                "published_elevation_m": overview["elevation"],
            },
        },
        "observation_dates": {
            "campaign_year": overview["year"],
            "forest_inventory": overview["date_forinv"],
            "hemispherical_photo": overview["date_hemiphoto"],
            "tls": overview["date_TLS"],
            "fractional_cover_photography": overview["date_fractcover"],
            "forest_floor_hyperspectral_measurement": overview["date_forestfloor"],
            "canopy_transmission": overview["date_canopytransm"],
        },
        "scan_protocol": {
            "published_protocol_code": overview["protocol_TLS"],
            "scan_count": scan_protocol["scan_count"],
            "nominal_layout_m": list(scan_protocol["nominal_layout_m"]),
            "angular_pitch_mrad": scan_protocol["angular_pitch_mrad"],
            "published_pitch_mm_at_100m": scan_protocol["published_pitch_mm_at_100m"],
            **tls,
        },
        "forest_inventory": {
            "published_protocol_code": overview["protocol_forinv"],
            "published_plot_summary": summary,
            "individual_tree_record_count": len(trees),
            "individual_tree_records": trees,
            "interpretation_limit": (
                "published inventory only; no stand age, management history, root, "
                "deadwood, or terrain-surface semantics are inferred"
            ),
        },
        "forest_floor_semantics": {
            **floor,
            "interpretation_limit": (
                "areal cover composition only; it is not thickness, soil profile, "
                "moisture, or a registered geometric surface classification"
            ),
        },
        "photo_inventory": _photo_inventory(
            artifacts, _required_string(overview["date_fractcover"], "fractional cover date")
        ),
        "source_artifacts": sorted(
            (_source_record(item) for item in artifacts.values()),
            key=lambda item: item["path"],
        ),
        "unresolved_conditions": unknowns,
        "ood_and_transfer": {
            "status": "unresolved_out_of_distribution",
            "physical_analogue_dossier": "absent",
            "estonia_condition_envelope_match": "unknown",
            "condition_stratified_ood_bounds": "unknown",
            "transfer_allowed": False,
            "required_action": "abstain",
        },
        "semantic_policy": {
            "photo_pixels_classified": False,
            "tls_points_classified_as_target_surface": False,
            "surface_include_exclude_policy_frozen": False,
            "unknown_may_default_or_use_nearest_class": False,
            "soil_geology_hydrology_inference_performed": False,
        },
    }
    encoded = canonical_json_bytes(record)
    content_sha256 = sha256_bytes(encoded)
    destination = (
        work_root
        / "microtopography"
        / "hovi"
        / "conditions"
        / "sha256"
        / content_sha256
        / f"{_PLOT_ID}-condition-evidence.json"
    )
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError(f"corrupt Hovi content-addressed condition evidence: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(destination)
    return destination


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit fail-closed HY_SPRUCE4 condition and semantics evidence."
    )
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--plot", choices=[_PLOT_ID], default=_PLOT_ID)
    parser.add_argument(
        "--selection", type=Path, default=CONFIG_DIR / "hovi-public-targets.json"
    )
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    output = emit_hovi_conditions(
        args.retained,
        args.plot,
        selection_path=args.selection,
        work_root=args.work_root,
    )
    print(f"condition evidence: {output}")


if __name__ == "__main__":
    _main()

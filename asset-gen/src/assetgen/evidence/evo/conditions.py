"""Materialize strict condition/selection evidence for Evo plot 1086."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Mapping

from ...config import CONFIG_DIR, DATA_WORK
from .selection import EvoArtifact, EvoSelection, load_evo_selection

_SCHEMA = "evo-condition-selection-evidence/1.0.0"
_RETAINED_SCHEMA = "evo-selector-retention/1.0.0"
_RETAINED_MANIFEST_SHA256 = (
    "e3ea289874c682eb8e33d3844a91e5c292a425c3134690679244d0f57ea013ac"
)
_RETENTION_ID = "1584bd396efe1f07f716be7161516aa068a007f2311182ef0ad824d17504c464"
_PLOT_ID = "1086"
_HEADER = (
    "plot_id",
    "x",
    "y",
    "N",
    "G",
    "V",
    "Dg",
    "Hg",
    "Pine_BA%",
    "Spruce_BA%",
    "Birch_BA%",
)
_COLUMN_SEMANTICS = (
    {
        "column": "plot_id",
        "semantic": "sample_plot_identifier",
        "unit": None,
    },
    {
        "column": "x",
        "semantic": "plot_center_easting",
        "unit": "m",
        "crs": "EPSG:3067",
    },
    {
        "column": "y",
        "semantic": "plot_center_northing",
        "unit": "m",
        "crs": "EPSG:3067",
    },
    {"column": "N", "semantic": "trees_per_ha", "unit": "count/ha"},
    {
        "column": "G",
        "semantic": "basal_area_m2_per_ha",
        "unit": "m2/ha",
    },
    {
        "column": "V",
        "semantic": "stem_volume_m3_per_ha",
        "unit": "m3/ha",
    },
    {
        "column": "Dg",
        "semantic": "basal_area_weighted_mean_diameter_cm",
        "unit": "cm",
    },
    {
        "column": "Hg",
        "semantic": "basal_area_weighted_mean_height_m",
        "unit": "m",
    },
    {
        "column": "Pine_BA%",
        "semantic": "pine_share_of_basal_area",
        "unit": "percent",
    },
    {
        "column": "Spruce_BA%",
        "semantic": "spruce_share_of_basal_area",
        "unit": "percent",
    },
    {
        "column": "Birch_BA%",
        "semantic": "birch_share_of_basal_area",
        "unit": "percent",
    },
)
_ROW_TO_CONFIG = {
    "N": "trees_per_ha",
    "G": "basal_area_m2_per_ha",
    "V": "stem_volume_m3_per_ha",
    "Dg": "basal_area_weighted_mean_diameter_cm",
    "Hg": "basal_area_weighted_mean_height_m",
    "Pine_BA%": "pine_basal_area_pct",
    "Spruce_BA%": "spruce_basal_area_pct",
    # The source header is narrower than the frozen config's normalized label.
    "Birch_BA%": "deciduous_basal_area_pct",
}
_UNKNOWN_CONDITIONS = {
    "soil": (
        "field_soil_class",
        "soil_component_mixture",
        "vertical_horizons_and_thicknesses",
        "mineral_texture",
        "stoniness_and_clast_distribution",
        "erodibility",
    ),
    "hydrology": (
        "soil_moisture",
        "water_table_depth",
        "saturation_state",
        "surface_and_subsurface_flow",
        "drainage_and_ditches",
        "inundation_history",
    ),
    "organic_floor": (
        "organic_horizon_thickness",
        "peat_depth",
        "moss_and_litter_thickness",
        "moss_and_litter_load",
        "root_geometry_and_root_heave",
        "deadwood_geometry_and_decay",
        "understory_structure",
        "stable_surface_semantics",
    ),
    "management_and_disturbance": (
        "forest_management_history",
        "thinning_harvest_and_planting_history",
        "machine_traffic_ruts_and_skid_trails",
        "disturbance_type_and_severity",
        "disturbance_date_or_age",
        "recovery_state",
    ),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evo {label} must be an object")
    return value


def _decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"Evo {label} must be decimal, got {value!r}") from None
    if not parsed.is_finite():
        raise ValueError(f"Evo {label} must be finite")
    return parsed


def _config_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evo frozen {label} must be numeric")
    return _decimal(str(value), f"frozen {label}")


def _verify_retained_selector(
    retained_path: Path,
    selection: EvoSelection,
    selector: EvoArtifact,
) -> tuple[Path, str]:
    retained_path = retained_path.resolve()
    if retained_path.name != "retained.json" or not retained_path.is_file():
        raise ValueError("Evo retained selector must be an existing retained.json")
    encoded = retained_path.read_bytes()
    if _sha256_bytes(encoded) != _RETAINED_MANIFEST_SHA256:
        raise ValueError("Evo retained selector manifest bytes changed")
    retained = _mapping(json.loads(encoded), "retained selector manifest")
    manifest_selection = _mapping(retained.get("selection"), "retained selection")
    artifact = _mapping(retained.get("artifact"), "retained artifact")
    expected_artifact = {
        "file_id": selector.file_id,
        "path": selector.source_path,
        "bytes": selector.bytes,
        "sha256": selector.sha256,
        "kind": selector.kind,
    }
    if (
        retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("retention_id") != _RETENTION_ID
        or retained.get("plan_sha256") != _RETENTION_ID
        or retained.get("authorized_scope")
        != "stand_attribute_selector_index_only"
        or retained.get("point_cloud_fetch_authorized") is not False
        or retained.get("signed_url_persisted") is not False
        or manifest_selection.get("config_sha256") != selection.config_sha256
        or any(artifact.get(key) != value for key, value in expected_artifact.items())
        or artifact.get("retained_bytes") != selector.bytes
        or artifact.get("verified") is not True
    ):
        raise ValueError("Evo retained selector contract changed")
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Evo retained selector relative path is invalid")
    selector_path = (retained_path.parent / relative).resolve()
    if (
        not selector_path.is_relative_to(retained_path.parent)
        or not selector_path.is_file()
        or selector_path.stat().st_size != selector.bytes
        or _sha256_file(selector_path) != selector.sha256
    ):
        raise ValueError("Evo retained selector source bytes changed")
    return selector_path, _sha256_bytes(encoded)


def _read_exact_row(selector_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    encoded = selector_path.read_bytes()
    try:
        text = encoded.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Evo selector CSV is not UTF-8") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != _HEADER:
        raise ValueError("Evo selector CSV header changed")
    rows = list(reader)
    if len(rows) != 55 or any(set(row) != set(_HEADER) for row in rows):
        raise ValueError("Evo selector CSV must contain 55 complete rows")
    plot_ids = [row["plot_id"] for row in rows]
    if len(set(plot_ids)) != len(plot_ids):
        raise ValueError("Evo selector CSV contains duplicate plot IDs")
    matches = [row for row in rows if row["plot_id"] == _PLOT_ID]
    if len(matches) != 1:
        raise ValueError("Evo selector CSV must contain exactly one plot 1086 row")

    raw_lines = encoded.splitlines(keepends=True)
    raw_matches = [
        (line_number, line)
        for line_number, line in enumerate(raw_lines, start=1)
        if line.startswith(b"1086,")
    ]
    if len(raw_matches) != 1:
        raise ValueError("Evo selector CSV raw plot-1086 record is ambiguous")
    physical_line, raw_record = raw_matches[0]
    if physical_line != 34 or not raw_record.endswith(b"\r\n"):
        raise ValueError("Evo selector CSV row position or line ending changed")
    try:
        raw_utf8 = raw_record.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Evo selector row is not UTF-8") from None
    return matches[0], {
        "physical_line_1_based": physical_line,
        "data_row_1_based": physical_line - 1,
        "raw_record_utf8": raw_utf8,
        "raw_record_bytes": len(raw_record),
        "raw_record_sha256": _sha256_bytes(raw_record),
        "line_ending": "CRLF",
        "raw_fields": dict(matches[0]),
    }


def _ratio(numerator: Decimal, denominator: Decimal) -> str:
    if denominator == 0:
        raise ValueError("Evo frozen comparator cannot be zero")
    with localcontext() as context:
        context.prec = 50
        return format(numerator / denominator, "f")


def _predicate(
    predicate_id: str,
    observed: Decimal,
    comparator: Decimal,
    lower: Decimal,
    upper: Decimal | None = None,
) -> dict[str, Any]:
    passed = observed >= comparator * lower
    if upper is not None:
        passed = passed and observed <= comparator * upper
    return {
        "id": predicate_id,
        "observed_decimal": str(observed),
        "comparator_decimal": str(comparator),
        "observed_to_comparator_ratio_decimal": _ratio(observed, comparator),
        "inclusive_ratio_bounds": [
            str(lower),
            str(upper) if upper is not None else None,
        ],
        "passed": passed,
    }


def _build_record(
    selection: EvoSelection,
    config: Mapping[str, Any],
    selector: EvoArtifact,
    retained_sha256: str,
    row: Mapping[str, str],
    row_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _mapping(config.get("selection"), "frozen selection")
    center = _mapping(selected.get("center"), "plot center")
    epsg3067 = _mapping(center.get("epsg3067"), "EPSG:3067 center")
    epsg4326 = _mapping(center.get("epsg4326"), "EPSG:4326 center")
    stand = _mapping(selected.get("stand_attributes"), "stand attributes")
    comparator = _mapping(selected.get("hy_spruce4_reference"), "HY_SPRUCE4 comparator")
    selector_rule = _mapping(selected.get("frozen_selector"), "selector predicate")
    substrate = _mapping(selected.get("coarse_substrate_match"), "coarse substrate")
    acquisition = _mapping(config.get("acquisition"), "acquisition")
    qualification = _mapping(config.get("qualification"), "qualification")

    if row["plot_id"] != _PLOT_ID:
        raise ValueError("Evo selected row identity changed")
    if (
        _decimal(row["x"], "x") != _config_decimal(epsg3067.get("easting_m"), "easting")
        or _decimal(row["y"], "y")
        != _config_decimal(epsg3067.get("northing_m"), "northing")
    ):
        raise ValueError("Evo selector coordinates differ from the frozen selection")
    metric_checks = {}
    for column, config_name in _ROW_TO_CONFIG.items():
        observed = _decimal(row[column], column)
        frozen = _config_decimal(stand.get(config_name), config_name)
        if observed != frozen:
            raise ValueError(f"Evo selector {column} differs from frozen {config_name}")
        metric_checks[column] = {
            "config_field": config_name,
            "observed_decimal": row[column],
            "exact_decimal_match": True,
        }
    species_sum = sum(
        (_decimal(row[column], column) for column in _HEADER[-3:]),
        start=Decimal(0),
    )
    if species_sum != Decimal(100):
        raise ValueError("Evo selector basal-area shares do not sum to 100 percent")

    observed_n = _decimal(row["N"], "N")
    observed_g = _decimal(row["G"], "G")
    observed_dg = _decimal(row["Dg"], "Dg")
    observed_spruce = _decimal(row["Spruce_BA%"], "Spruce_BA%")
    comparator_n = _config_decimal(comparator.get("trees_per_ha"), "comparator trees")
    comparator_g = _config_decimal(
        comparator.get("basal_area_m2_per_ha"), "comparator basal area"
    )
    comparator_dg = _config_decimal(
        comparator.get("basal_area_weighted_mean_diameter_cm"),
        "comparator diameter",
    )
    numeric_predicates = [
        {
            "id": "spruce_basal_area_absolute_minimum",
            "observed_decimal": str(observed_spruce),
            "operator": ">=",
            "threshold_decimal": str(
                _config_decimal(
                    selector_rule.get("spruce_basal_area_min_pct"),
                    "spruce threshold",
                )
            ),
            "unit": "percent",
            "passed": observed_spruce
            >= _config_decimal(
                selector_rule.get("spruce_basal_area_min_pct"),
                "spruce threshold",
            ),
        },
        _predicate(
            "tree_density_relative_to_hy_spruce4",
            observed_n,
            comparator_n,
            _config_decimal(
                selector_rule.get("tree_density_min_fraction_of_hy_spruce4"),
                "density lower bound",
            ),
        ),
        _predicate(
            "basal_area_relative_to_hy_spruce4",
            observed_g,
            comparator_g,
            _config_decimal(selector_rule.get("basal_area_ratio_to_hy_spruce4")[0], "G lower"),
            _config_decimal(selector_rule.get("basal_area_ratio_to_hy_spruce4")[1], "G upper"),
        ),
        _predicate(
            "mean_diameter_relative_to_hy_spruce4",
            observed_dg,
            comparator_dg,
            _config_decimal(
                selector_rule.get("mean_diameter_ratio_to_hy_spruce4")[0],
                "Dg lower",
            ),
            _config_decimal(
                selector_rule.get("mean_diameter_ratio_to_hy_spruce4")[1],
                "Dg upper",
            ),
        ),
    ]
    substrate_pass = (
        selector_rule.get("same_coarse_surface_and_subsurface_code_as_hy_spruce4")
        is True
        and substrate.get("surface_material_code")
        == substrate.get("hy_spruce4_surface_material_code")
        and substrate.get("subsurface_material_code")
        == substrate.get("hy_spruce4_subsurface_material_code")
    )
    protocol_pass = (
        selector_rule.get("require_standard_nine_position_protocol") is True
        and acquisition.get("standard_scan_positions") == 9
        and acquisition.get("plot_1086_extra_positions") == 0
    )
    if not all(item["passed"] for item in numeric_predicates):
        raise ValueError("Evo plot 1086 no longer passes the frozen numeric selector")
    if not substrate_pass or not protocol_pass:
        raise ValueError("Evo plot 1086 no longer passes the frozen context selector")
    qualifying_ids = selector_rule.get("qualifying_plot_ids")
    if qualifying_ids != ["1086", "1065"] or _PLOT_ID not in qualifying_ids:
        raise ValueError("Evo frozen qualifying set changed")
    if (
        qualification.get("role") != "raw_candidate"
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("target_truth") is not False
        or qualification.get("synthesis_authorized") is not False
        or qualification.get("transfer_ceiling") != "none"
    ):
        raise ValueError("Evo qualification boundary changed")

    return {
        "schema_version": _SCHEMA,
        "evidence_kind": "condition_and_selection_raw_candidate",
        "source_binding": {
            "selection_config_sha256": selection.config_sha256,
            "retention_id": _RETENTION_ID,
            "retained_manifest_sha256": retained_sha256,
            "retention_state_semantics": {
                "frozen_selection_retained_in_workspace": (
                    selector.retained_in_workspace
                ),
                "frozen_selection_state_scope": (
                    "historical pre-retention state as of the immutable selection audit; "
                    "not current mutable workspace state and not rewritten after retention"
                ),
                "current_verified_retention": True,
                "current_state_source": "content-addressed retained manifest and source bytes",
                "mutable_retention_state_in_selection_identity": False,
            },
            "selector_artifact": {
                "file_id": selector.file_id,
                "path": selector.source_path,
                "bytes": selector.bytes,
                "sha256": selector.sha256,
            },
        },
        "qualification": {
            "role": "raw_candidate",
            "qualification_status": "unqualified",
            "analogue_qualified": False,
            "target_truth": False,
            "synthesis_authorized": False,
            "transfer_ceiling": "none",
            "eligibility_result": "abstain",
        },
        "column_schema": {
            "exact_header_order": list(_HEADER),
            "semantics": list(_COLUMN_SEMANTICS),
            "semantics_basis": (
                "exact selector header plus the frozen config's normalized field names, "
                "units, horizontal CRS, and acquisition identity"
            ),
            "birch_label_boundary": (
                "Birch_BA% is retained as birch basal-area share. Its equality to the "
                "config field deciduous_basal_area_pct does not broaden the source column "
                "to non-birch deciduous species."
            ),
        },
        "plot": {
            "plot_id": _PLOT_ID,
            "site_id": selection.site_id,
            "campaign_id": selection.campaign_id,
            "plot_size_m": list(selected.get("plot_size_m")),
            "coordinates": {
                "selector_epsg3067": {
                    "easting_m_decimal": row["x"],
                    "northing_m_decimal": row["y"],
                    "exact_match_to_frozen_config": True,
                },
                "selection_config_epsg4326": dict(epsg4326),
                "epsg4326_source_limit": "not present in selector CSV; frozen config only",
                "vertical_coordinate": "unknown_not_present_in_selector_csv",
            },
            "exact_selector_row": dict(row_proof),
            "stand_metrics": {
                "trees_per_ha": float(observed_n),
                "basal_area_m2_per_ha": float(observed_g),
                "stem_volume_m3_per_ha": float(_decimal(row["V"], "V")),
                "basal_area_weighted_mean_diameter_cm": float(observed_dg),
                "basal_area_weighted_mean_height_m": float(_decimal(row["Hg"], "Hg")),
                "pine_basal_area_pct": float(_decimal(row["Pine_BA%"], "Pine_BA%")),
                "spruce_basal_area_pct": float(observed_spruce),
                "birch_basal_area_pct": float(_decimal(row["Birch_BA%"], "Birch_BA%")),
                "species_basal_area_sum_decimal": str(species_sum),
            },
        },
        "hy_spruce4_frozen_comparator": {
            "values": dict(comparator),
            "provenance_limit": comparator.get("derivation"),
            "scope": "frozen selector comparator; not a condition analogue qualification",
        },
        "selector_assessment": {
            "numeric_predicates": numeric_predicates,
            "retained_csv_numeric_predicates_all_passed": True,
            "coarse_substrate_config_assertion": {
                "source": substrate.get("source"),
                "map_scale": substrate.get("map_scale"),
                "frozen_config_evo_surface_material_code": substrate.get(
                    "surface_material_code"
                ),
                "frozen_config_evo_subsurface_material_code": substrate.get(
                    "subsurface_material_code"
                ),
                "frozen_config_hy_spruce4_surface_material_code": substrate.get(
                    "hy_spruce4_surface_material_code"
                ),
                "frozen_config_hy_spruce4_subsurface_material_code": substrate.get(
                    "hy_spruce4_subsurface_material_code"
                ),
                "asserted_match": substrate_pass,
                "evidence_status": "frozen_config_assertion_not_reverified",
                "retained_selector_csv_proves_assertion": False,
                "interpretation_limit": substrate.get("interpretation_limit"),
            },
            "scan_protocol_config_assertion": {
                "required_standard_positions": 9,
                "frozen_config_standard_positions": acquisition.get(
                    "standard_scan_positions"
                ),
                "frozen_config_extra_positions": acquisition.get(
                    "plot_1086_extra_positions"
                ),
                "asserted_standard_protocol": protocol_pass,
                "evidence_status": "frozen_config_assertion_not_reverified",
                "retained_selector_csv_proves_assertion": False,
            },
            "frozen_config_declared_qualifying_plot_ids": list(qualifying_ids),
            "frozen_config_declares_plot_1086_qualifying": True,
            "full_selector_independently_proved": False,
            "full_selector_evidence_limit": (
                "the retained CSV proves only the numeric stand predicates; substrate, "
                "scan protocol, and qualifying-set membership are frozen config assertions"
            ),
            "final_choice_rule": selector_rule.get("selected_rule"),
            "final_choice_rule_status": "frozen_config_assertion_not_recomputed",
            "final_choice_rule_limit": (
                "the retained selector CSV has no LAZ byte counts for plot 1065; only "
                "plot 1086's frozen, fetch-disabled LAZ tuple is present in the config"
            ),
        },
        "condition_unknowns": {
            "policy": "unknown_remains_unknown_and_cannot_default_to_a_nearest_class",
            "inference_allowed": False,
            "fields": {
                group: {field: "unknown" for field in fields}
                for group, fields in _UNKNOWN_CONDITIONS.items()
            },
        },
        "interpretation_boundary": {
            "selector_row_is_geometry": False,
            "point_cloud_downloaded": False,
            "terrain_surface_observed": False,
            "soil_profile_observed": False,
            "hydrology_observed": False,
            "organic_floor_observed": False,
            "management_or_disturbance_observed": False,
            "analogue_transfer_allowed": False,
            "synthesis_use_allowed": False,
        },
        "qa": {
            "selector_bytes_verified": True,
            "header_exact": True,
            "selector_row_count": 55,
            "plot_1086_exact_row_count": 1,
            "row_to_config_decimal_checks": metric_checks,
            "coordinates_exact": True,
            "species_basal_area_sum_exact_100_pct": True,
            "retained_csv_numeric_predicates_passed": True,
            "frozen_config_context_assertions_present": True,
            "full_selector_independently_proved": False,
            "unresolved_condition_inference_count": sum(
                len(fields) for fields in _UNKNOWN_CONDITIONS.values()
            ),
        },
    }


def emit_evo_conditions(
    retained_path: Path,
    *,
    selection_path: Path = CONFIG_DIR / "evidence" / "evo-2024-plot-1086.json",
    work_root: Path = DATA_WORK,
) -> Path:
    """Emit immutable selector evidence without qualifying it for synthesis."""
    selection = load_evo_selection(selection_path)
    selector_artifacts = selection.authorized_retention_artifacts()
    if len(selector_artifacts) != 1:
        raise ValueError("Evo condition evidence requires exactly one selector artifact")
    selector = selector_artifacts[0]
    selector_path, retained_sha256 = _verify_retained_selector(
        retained_path, selection, selector
    )
    row, row_proof = _read_exact_row(selector_path)
    config = _mapping(json.loads(selection.path.read_bytes()), "frozen config")
    record = _build_record(
        selection,
        config,
        selector,
        retained_sha256,
        row,
        row_proof,
    )
    encoded = _canonical_json_bytes(record)
    content_sha256 = _sha256_bytes(encoded)
    destination = (
        work_root
        / "microtopography"
        / "evo"
        / "conditions"
        / "sha256"
        / content_sha256
        / "1086-condition-selection-evidence.json"
    )
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError(f"corrupt Evo content-addressed evidence: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("xb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(destination)
    return destination


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit fail-closed Evo plot-1086 condition/selection evidence."
    )
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=CONFIG_DIR / "evidence" / "evo-2024-plot-1086.json",
    )
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    output = emit_evo_conditions(
        args.retained,
        selection_path=args.selection,
        work_root=args.work_root,
    )
    print(f"Evo condition/selection evidence: {output}")


if __name__ == "__main__":
    _main()

"""Issue the frozen, pre-geometry qualification decision for OPARA-1038."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from ...config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK
from .source import (
    GCP_MEMBER,
    LOG_SUMMARY_MEMBER,
    PARAMETER_MEMBER,
    PROCESSED_README_MEMBER,
    PROTOCOL_README_MEMBER,
    TIMELAPSE_README_MEMBER,
    extract_qualification_records,
    inventory,
)


_CONFIG = ASSET_GEN_ROOT / "config/evidence/opara-1038-plot-probe.json"
_RETENTION_ID = "db8163ca878efc5ff98c0f68474724678f524c7b18ac03422d93005e7be0502e"
_ARCHIVE_MD5 = "605952140081ef2d31d6c2a4a28182c5"
_ARCHIVE_BYTES = 3_307_259_536
_EVENT_DATE = "2021-07-21"
_SCHEMA = "opara-1038-frozen-plot-decision/1.0.0"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _retained_source() -> tuple[Path, dict[str, Any]]:
    root = DATA_IN / "evidence" / "opara_1038" / _RETENTION_ID
    manifest_path = root / "retained.json"
    manifest = json.loads(manifest_path.read_bytes())
    archive_record = manifest.get("archive", {})
    archive = root / str(archive_record.get("path"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("retention_id") != _RETENTION_ID
        or archive_record.get("bytes") != _ARCHIVE_BYTES
        or archive_record.get("md5") != _ARCHIVE_MD5
        or not archive.is_file()
        or archive.stat().st_size != _ARCHIVE_BYTES
    ):
        raise ValueError("retained OPARA source does not match the frozen archive")
    return archive, manifest


def _parameter_row(payload: bytes) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(rows) != 1:
        raise ValueError("expected one OPARA field-parameter row")
    return {str(key): str(value) for key, value in rows[0].items()}


def _gcp_rows(payload: bytes) -> list[list[str]]:
    rows = [
        row
        for row in csv.reader(io.StringIO(payload.decode("utf-8-sig")))
        if any(cell.strip() for cell in row)
    ]
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError("OPARA GCP coordinate record is malformed")
    return rows


def _summary_dates(payload: bytes) -> tuple[str, ...]:
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    dates = set()
    for row in rows:
        match = re.search(r"sfm_timelapse\\([^\\]+)\\timelapse", row["file"])
        if match:
            dates.add(match.group(1))
    return tuple(sorted(dates))


def _numeric_ids(members: tuple[str, ...], pattern: str) -> list[int]:
    expression = re.compile(pattern)
    return sorted(
        int(match.group(1))
        for name in members
        if (match := expression.fullmatch(name)) is not None
    )


def build_decision() -> dict[str, Any]:
    frozen = json.loads(_CONFIG.read_bytes())
    if (
        frozen.get("frozen_before_geometry_inventory") is not True
        or frozen.get("event", {}).get("date") != _EVENT_DATE
        or frozen.get("event", {}).get("event_state")
        != "initial_pre_rainfall_epoch"
    ):
        raise ValueError("OPARA frozen event contract changed")

    archive, retention = _retained_source()
    members = inventory(archive)
    records = extract_qualification_records(archive)
    missing_records = sorted(set(records) - set(members))
    if missing_records:
        raise ValueError(f"qualification members absent from inventory: {missing_records}")

    dense_prefix = (
        "B_sample_plot/III_plot_1_processed/sfm_timelapse/2021-07-21/"
        "timelapse/dense/"
    )
    m3c2_prefix = dense_prefix.removesuffix("dense/") + "m3c2/"
    precision_prefix = dense_prefix.removesuffix("dense/") + "ptPrecision/"
    dense_ids = _numeric_ids(members, re.escape(dense_prefix) + r"dense_(\d+)\.ply")
    m3c2_ids = _numeric_ids(
        members, re.escape(m3c2_prefix) + r"m3c2_0-to-(\d+)\.txt"
    )
    precision_ids = _numeric_ids(
        members, re.escape(precision_prefix) + r"pt_prec_index_(\d+)\.txt"
    )
    raw_log_members = sorted(
        name
        for name in members
        if "/log/" in name.lower() or "log_metashape" in name.lower()
    )
    image_members = sorted(
        name
        for name in members
        if "/slr/2021-07-21/" in name
        and name.lower().endswith((".cr2", ".jpg", ".jpeg"))
    )

    parameters = _parameter_row(records[PARAMETER_MEMBER])
    gcp_rows = _gcp_rows(records[GCP_MEMBER])
    summary_dates = _summary_dates(records[LOG_SUMMARY_MEMBER])
    record_hashes = {
        member: {"bytes": len(payload), "sha256": _sha256(payload)}
        for member, payload in sorted(records.items())
    }
    processed_readme = records[PROCESSED_README_MEMBER].decode("utf-8-sig")
    timelapse_readme = records[TIMELAPSE_README_MEMBER].decode("utf-8-sig")
    protocol_readme = records[PROTOCOL_README_MEMBER].decode("utf-8-sig")

    if "only the first and last" not in timelapse_readme:
        raise ValueError("publisher sample-reduction statement changed")
    if "log: (optional)" not in processed_readme:
        raise ValueError("publisher processed-data description changed")
    if "vegetation and rocks" not in protocol_readme:
        raise ValueError("publisher cover semantics changed")

    failures = [
        {
            "gate": "frozen_initial_epoch_present",
            "result": "fail",
            "evidence": (
                "M3C2 products reference initial epoch 0, but dense_0.ply is absent; "
                "the publisher readme says this reduced sample contains only selected "
                "first/last products."
            ),
        },
        {
            "gate": "same_epoch_raw_image_cloud_identity",
            "result": "fail",
            "evidence": (
                "Raw synchronized images and later dense products are present, but the "
                "sample contains no image-to-dense epoch map for the missing epoch 0."
            ),
        },
        {
            "gate": "independent_control_identity",
            "result": "fail",
            "evidence": (
                "The coordinate file has nine unlabeled points and no control/check role; "
                "no event bundle-adjustment log is retained."
            ),
        },
        {
            "gate": "axis_specific_check_point_error",
            "result": "fail",
            "evidence": (
                "The only retained log summary has aggregate 3D GCP/CP errors, contains "
                f"dates {list(summary_dates)}, and has no {_EVENT_DATE} row or X/Y/Z "
                "check-point residuals."
            ),
        },
        {
            "gate": "defensible_exposed_soil_mask",
            "result": "fail",
            "evidence": (
                f"Field cover is {parameters.get('cover.%')}% and is defined jointly as "
                "vegetation and rocks; the archive provides no qualified epoch-0 mask "
                "that separates excluded vegetation, targets, metal and occlusion."
            ),
        },
    ]

    return {
        "schema_version": _SCHEMA,
        "status": "no_go_pregeometry",
        "method": "frozen_event_archive_and_companion_record_audit",
        "authority": {
            "frozen_probe_config": _CONFIG.relative_to(ASSET_GEN_ROOT).as_posix(),
            "frozen_probe_sha256": _sha256(_CONFIG.read_bytes()),
            "selection_contract": frozen["authority"]["selection_contract"],
            "selection_sha256": frozen["authority"]["selection_sha256"],
        },
        "source": {
            "doi": frozen["authority"]["dataset_doi"],
            "retention_id": _RETENTION_ID,
            "archive_bytes": archive.stat().st_size,
            "archive_md5": retention["archive"]["md5"],
            "archive_sha256": retention["archive"]["sha256"],
            "archive_member_count": len(members),
            "qualification_record_hashes": record_hashes,
        },
        "frozen_claim": frozen["physical_claim"],
        "field_conditions": {
            "slope_percent": parameters.get("slope.%"),
            "cover_percent_joint_vegetation_and_rocks": parameters.get("cover.%"),
            "initial_soil_moisture_vol_percent_run_1": parameters.get(
                "initial.soil.moisture.mean.Vol.-%.1"
            ),
            "soil_organic_carbon_percent": parameters.get("SOC.%"),
            "texture_percent": {
                key: parameters.get(key)
                for key in (
                    "coarse.sand.%",
                    "medium.sand.%",
                    "fine.sand.%",
                    "coarse.silt.%",
                    "medium.silt.%",
                    "fine.silt.%",
                    "clay.%",
                )
            },
            "tillage": parameters.get("tillage"),
            "fruit": parameters.get("fruit"),
        },
        "event_inventory": {
            "raw_image_count": len(image_members),
            "coordinate_point_count": len(gcp_rows),
            "dense_epoch_ids": dense_ids,
            "m3c2_target_epoch_ids": m3c2_ids,
            "point_precision_epoch_ids": precision_ids,
            "raw_bundle_adjustment_log_members": raw_log_members,
            "global_log_summary_dates": list(summary_dates),
            "initial_dense_epoch_0_present": 0 in dense_ids,
        },
        "qualification_failures": failures,
        "support_and_error": {
            "B1_0.25_to_1m": "not_qualified",
            "point_precision_role": (
                "propagated tie-point precision is present for selected later epochs but "
                "is not independent ground truth and cannot replace held-out CP error"
            ),
            "horizontal_cp_error_m": None,
            "vertical_cp_error_m": None,
            "effective_support_m": None,
        },
        "surface_masks": {
            "materialized": False,
            "known_valid_cells": 0,
            "unknown_domain": "entire_frozen_event_surface",
            "reason": "no eligible epoch-0 geometry/image binding survived qualification",
        },
        "geometry": {
            "viewed_or_converted": False,
            "reason": "terminal companion-record failure precedes morphology inspection",
        },
        "qa_images": {
            "emitted": [],
            "reason": "no eligible intermediate geometry exists; a chart would be bookkeeping",
        },
        "unknowns": [
            "which nine coordinate points were controls versus independent checks",
            "separate horizontal and vertical check-point residuals for 2021-07-21",
            "which raw image tuple reconstructs the missing pre-rain epoch 0",
            "epoch-0 visibility, occlusion, boundary, target and vegetation masks",
            "whether the reported 10 percent cover is vegetation, rocks, or both",
        ],
        "decision": {
            "go_no_go": "no_go",
            "authorized_use": "none",
            "synthesis_transfer": False,
            "resume_condition": (
                "publisher supplies the exact 2021-07-21 dense epoch 0, its raw-image "
                "mapping, raw adjustment/control export with independently withheld CP "
                "identity and X/Y/Z residuals, and defensible epoch-0 exclusion masks"
            ),
        },
    }


def write_decision(output_root: Path | None = None) -> Path:
    core = build_decision()
    candidate_id = _sha256(_canonical_json(core))
    payload = {**core, "candidate_id": candidate_id}
    root = output_root or (
        DATA_WORK
        / "microtopography"
        / "opara_1038"
        / "plot_probe"
        / "sha256"
        / candidate_id
    )
    destination = root / "candidate.json"
    encoded = _canonical_json(payload)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError("existing OPARA decision conflicts with deterministic output")
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
    parser = argparse.ArgumentParser(description="Qualify the frozen OPARA plot event.")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(write_decision(args.output_root))


if __name__ == "__main__":
    _main()

"""Bounded national coverage inventory for the official soil-profile grammar."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyogrio.raw

from ....config import DATA_IN, DATA_WORK
from .extract import (
    _canonical_bytes,
    _implementation_sha256,
    _profile_authority_sources,
    _runtime_provenance,
    _source_bundle,
)
from .profile import PROFILE_GRAMMAR_ID, parse_humus_profile, parse_texture_profile
from .schema import LAYER

_SCHEMA = "laas.mullastikukaart-profile-coverage/1"
_FIELDS: tuple[tuple[str, Callable[[Any], dict[str, Any]]], ...] = (
    ("Loimis1", parse_texture_profile),
    ("Loimis2", parse_texture_profile),
    ("Huumus", parse_humus_profile),
)
_RESIDUAL_LIMIT = 64


def _status_inventory(
    values: np.ndarray[Any, Any], parser: Callable[[Any], dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(None if value is None else str(value) for value in values)
    status_features: Counter[str] = Counter()
    status_unique: Counter[str] = Counter()
    residuals: list[dict[str, Any]] = []
    for raw, count in counts.items():
        parsed = parser(raw)
        status = str(parsed["status"])
        status_features[status] += count
        status_unique[status] += 1
        if status == "unparseable_preserved":
            residuals.append(
                {
                    "raw": raw,
                    "feature_count": count,
                    "residuals": parsed["residuals"],
                }
            )
    residuals.sort(key=lambda row: (-row["feature_count"], row["raw"] or ""))
    nonmissing = len(values) - status_features["missing"]
    parsed = status_features["parsed_complete_official_grammar"]
    return {
        "feature_count": len(values),
        "unique_value_count_including_null": len(counts),
        "feature_count_by_status": dict(sorted(status_features.items())),
        "unique_value_count_by_status": dict(sorted(status_unique.items())),
        "parsed_fraction_of_nonmissing_features": parsed / nonmissing if nonmissing else None,
        "residual_ledger_is_capped": len(residuals) > _RESIDUAL_LIMIT,
        "residual_ledger_limit": _RESIDUAL_LIMIT,
        "top_unparseable_values": residuals[:_RESIDUAL_LIMIT],
    }


def inventory_profile_coverage(output_root: Path | None = None) -> Path:
    source = DATA_IN / "soil" / "mullakaart" / "Mullakaart.shp"
    source_bundle = _source_bundle(source)
    authorities = _profile_authority_sources()
    recipe = {
        "schema_version": f"{_SCHEMA}.recipe",
        "source_bundle": source_bundle,
        "profile_semantics_authorities": authorities,
        "profile_grammar": PROFILE_GRAMMAR_ID,
        "fields": [field for field, _ in _FIELDS],
        "read_policy": "attribute_only_full_national_scan_no_geometry",
        "residual_ledger_limit": _RESIDUAL_LIMIT,
        "implementation_sha256": _implementation_sha256(),
        "runtime_provenance": _runtime_provenance(),
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_root)
        if output_root is not None
        else DATA_WORK
        / "terrain"
        / "conditions"
        / "soil"
        / "mullastikukaart"
        / "profile-inventory"
        / "sha256"
    )
    output = parent / recipe_sha256 / "inventory.json"

    metadata, _, _, columns = pyogrio.raw.read(
        source,
        layer=LAYER,
        columns=[field for field, _ in _FIELDS],
        read_geometry=False,
    )
    if list(metadata["fields"]) != [field for field, _ in _FIELDS]:
        raise ValueError("profile inventory reader field order differs")
    if _source_bundle(source) != source_bundle:
        raise RuntimeError("Mullastikukaart source changed during profile inventory")
    if _profile_authority_sources() != authorities:
        raise RuntimeError("soil profile authorities changed during profile inventory")

    inventory = {
        field: _status_inventory(values, parser)
        for (field, parser), values in zip(_FIELDS, columns, strict=True)
    }
    artifact = {
        "schema_version": _SCHEMA,
        "status": "bounded_national_profile_grammar_coverage_inventory",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        "inventory": inventory,
        "interpretation": (
            "Coverage measures exact documented grammar support, not inferred correctness. "
            "Every unsupported source value remains unparseable; the residual ledger is "
            "frequency-ranked and capped without changing aggregate counts."
        ),
    }
    payload = json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"existing profile inventory fails reconstruction: {output}")
        return output
    output.parent.mkdir(parents=True)
    temporary = output.with_suffix(".json.part")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return output

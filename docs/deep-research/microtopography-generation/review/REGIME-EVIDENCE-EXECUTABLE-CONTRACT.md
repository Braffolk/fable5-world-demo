# Executable Regime And Evidence Contract

**Status:** normative regime/evidence schema under the final integrated spec;
research/specification only; no implementation is authorized.

**Contract version:** `microtopography-regime-release/1.0.0`

**Date:** 2026-07-13

## 1. Decisions

1. A morphology regime is a versioned evidence record, not a land-cover label.
2. Every release row contains the 13 groups defined in Section 3. An omitted,
   unknown, unavailable, or not-yet-measured value is represented explicitly and
   blocks release when that value is required by a gate.
3. The initial Estonia registry contains all 26 morphology regimes in Section 5.
   Raised-bog work may be `research` because the Moore archive can support a
   narrow foreign-domain experiment. Every other morphology row is initially
   `unsupported`. Neither state emits production residual geometry.
4. There is no default regime, nearest-regime substitution, generic smooth/noise
   fallback, or broad `forest`/`field`/`bog`/`sand` owner.
5. Corrected structural terrain remains valid when morphology abstains. It does
   not become evidence that the unresolved band is realistically modeled.
6. A national coverage claim passes only when every canonical finest-surface
   sample resolves to a released row, a separately target-evidenced
   effectively-smooth result, or an explicit forbidden/non-heightfield terminal.
7. The Moore archive can support only a plot-local, vegetation-clipped engineering
   screen in its measured foreign peat domain. Final target-error and replication
   audit makes it `target_evidence_insufficient` for owner selection; no Moore
   winner or `research_owner` is authorized by the initial registry.
8. The former v1 `library/sources.json` was not a release-quality evidence
   ledger: its 104 records had no authors, per-record SHA-256, or normalized
   access-result field; only four had a license field; repository revisions were
   stored separately. It has now been migrated to the fail-closed v2 contract in
   Section 8 without filling unknown facts by inference.

## 2. Normative Phenomenon Catalog

The release schema binds to catalog
`estonia-microtopography-phenomena/2026-07-13.1`. Its source snapshot is
`REGIME-PHENOMENON-MATRIX.md`, SHA-256
`6ca03d5bb552f6ad83e818da1d16d3fddf7d4006241990a90f6ffc21ab13ee4d`.
Changing a phenomenon, eligibility rule, required condition, scale claim, or
representation limit creates a new catalog version and invalidates affected
release rows.

The catalog has these representation terminals:

| Terminal | Meaning | Morphology behavior |
|---|---|---|
| `measured_or_reconstructed_structure` | trusted or evidence-corrected single-valued structure | preserve/reconstruct before morphology |
| `forbidden.open_water_surface` | visible water surface, not submerged bed | residual is exactly absent |
| `forbidden.hard_structure_or_object` | building, paved structure, or discrete object owned elsewhere | residual is exactly absent |
| `non_heightfield` | vertical, undercut, overhanging, cave, detached or overlapping geometry | excluded from heightfield claim and reported |
| `unsupported_morphology` | single-valued terrain lacks a released unresolved prior | retain corrected authority; fail claimed detail coverage |
| `target_evidenced_effectively_smooth` | measured target shows no resolvable energy in the claimed band for this condition stratum | emit no residual; counts as covered only under Section 6 |

The mandatory morphology decomposition is:

| ID | Constituent phenomena and physical span | Minimum discriminating state | Hard limit |
|---|---|---|---|
| `peat.raised_bog` | hummock, lawn, hollow, pool margin, collapse; centimetres through metre microforms and longer ridge/string organization | peat/depth, water-table proxy, flow, vegetation, drainage | small foreign plots do not establish mire layout; pools are forbidden water |
| `peat.fen_transitional` | strings, flarks, tussocks, saturated lawns, channels, floating-mat margins | minerotrophy, flow, slope, peat/mineral transition, vegetation | raised-bog priors do not transfer automatically; floating cover is not stable bare ground |
| `peat.drained_cut` | cut/extraction ridges, ditches, cut faces, subsidence, tracks, regrowth | boundary/network, operation and restoration date | temporal engineered state cannot use generic peat detail |
| `forest.pit_mound` | windthrow pit/mound, root-plate footprint, overlap and age/decay | stand/species/age, soil, wetness, wind, slope, management | occlusion and non-heightfield root plates; one repeated primitive is invalid |
| `forest.floor_ordinary` | root heave, hollows, decomposed logs, paths, moss/organic mat, microchannels | stand, soil, wetness, slope, organic/deadwood state | target semantics must distinguish mineral ground, stable organic surface and vegetation |
| `forest.managed_clearcut` | ruts, skid trails, drains, stump/root disturbance, slash and recovery | operation/date, soil bearing capacity, wetness, imagery | rapidly changing; static registry state can be stale |
| `agriculture.ploughed` | ridges/furrows, headlands, tracks, clods, erosion and cross-slope drainage | operation/date/implement/direction, soil texture/moisture, slope | current operation cannot be inferred from land cover alone |
| `agriculture.seedbed_harrowed_rolled` | aggregates, weak rows, compaction, tracks, rain crust | operation sequence, moisture/rain, texture, direction | millimetre aggregates are below the useful height-grid limit |
| `grassland.pasture_meadow` | tussocks, hoof prints, mole/ant mounds, drains, paths, gate erosion | grazing/mowing, livestock, soil, wetness, slope | rare-event density cannot be guessed from grassland class |
| `grassland.yard_turf` | grading, mowing/compaction, paths, drainage and disturbance | subtype, object/road context, imagery | rendering palette `yard = grass` is not morphology evidence |
| `sand.exposed` | representable ripples/deflation, tracks, rain and rill marks | sediment family, moisture, wind/water exposure, use | grains are below grid; aeolian/fluvial/coastal/quarry sand are not one regime |
| `sand.dune_aeolian` | ripple fields, slip-face breaks, blowouts, anchoring and disturbance | dune form, wind/orientation, exposure, vegetation | measured macrorelief must not be regenerated |
| `shore.beach_sand` | swash ridges, runnels, berms, wet/dry state, wrack/ice effects | date, water level/wave exposure, sediment, orientation | transient shoreline and optical water edge are not timeless truth |
| `shore.shingle_gravel_cobble` | sorting, imbrication-scale relief, storm ridges, drift/ice push | size distribution, wave/ice exposure, orientation | blocks and overhangs exceed a pure heightfield |
| `glacial.till_plain` | matrix relief, clasts, frost/rain disturbance, drainage and weak lineation | deposit, soil profile/stoniness, wetness, use | drumlins/eskers belong to measured base; no imported patterned-ground analogue |
| `glacial.gravel_esker_outwash` | sorting, stone patches, channels, extraction and track disturbance | mapped deposit/landform, exposure/use, imagery | coarse geology cannot locate individual stones |
| `carbonate.alvar_thin_soil` | pavement/soil islands, shallow solution forms, vegetation-edged relief | lithology, soil thickness, exposed fraction, joints, wetness | vegetation must not become bedrock height |
| `carbonate.pavement_karst` | joints/grikes, karren, pits, clint edges and fills | exposed rock, lithology, fracture/bedding, drainage | narrow/deep cracks and overhangs alias or violate heightfield |
| `outcrop.sandstone` | bedding ledges, joints, pits, blocks, runoff grooves, colluvium | formation, exposure, bedding/joints, moisture, slope | Suur Taevaskoda wall/caves/undercuts are non-heightfield |
| `outcrop.carbonate` | bedding steps, fractures, karren, frost blocks and soil pockets | lithology, structure, exposure, slope | RGB-to-roughness evidence is not a height target |
| `slope.colluvium_talus` | lobes, rills, creep steps, block fields and toes | source-to-toe context, substrate, soil, drainage | detached/overlapping blocks are non-heightfield |
| `fluvial.floodplain` | silt relief, levees, old channels, flood deposits and disturbance | topology, flood state/frequency, soil, wetness, use | generator may not reroute mapped drainage or invent current flood state |
| `fluvial.active_bar_bed` | ripples/dunes, gravel bars, scour, deposition and wet/dry transition | flow, sediment, water level/date, order | submerged bed is poorly observed; water surface is separate |
| `fluvial.rill_gully_seep_spring` | connected incision, headcut, fan and saturation relief | contributing area/flow, erodibility, slope, mapped drainage | local patches may not break connectivity |
| `coast.wetland` | pans, channels, hummocks, wrack/ice and inundation boundaries | water level, salinity proxy, soil, vegetation | strong-tide marsh analogues may not transfer to Baltic conditions |
| `technogenic.disturbed` | cuts, fills, spoil, compaction, tracks, demolition/extraction residue | typed operation/history, technogenic soil, ETAK, dated imagery | `barren` is not a physical subtype; stale imagery is unsafe |

Every row must preserve these state axes separately: substrate and vertical soil
profile; hydrology and flow; multiscale slope/aspect/curvature/topographic
position/contributing area/distance and side to structures; biological and
organic-layer state; management; disturbance age; observation support and date;
and heightfield representability. A broad class may participate in eligibility,
but it may never supply missing state or select an owner by itself.

## 3. Release-Row Schema

### 3.1 The 13 required groups

The exact top-level groups are:

1. `identity`
2. `physical_definition`
3. `eligibility`
4. `transition`
5. `structural_and_representation_constraints`
6. `conditions`
7. `target_evidence`
8. `geographic_splits`
9. `candidate_bakeoff`
10. `band_evidence`
11. `morphology_and_visual_result`
12. `packing_cook_and_serving_cost`
13. `failure_abstention_and_release`

`null` never means pass. It means explicitly unavailable and is legal only in a
state whose gate does not consume that field, normally `unsupported` or
`research`.

### 3.2 JSON Schema

The production registry must validate each resolved row against the following
JSON Schema. The implementation may split `$defs` into a file, but may not weaken
required fields or `additionalProperties: false`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://laas.invalid/schema/microtopography-regime-release-1.0.0.json",
  "title": "LAAS microtopography regime release row",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version", "identity", "physical_definition", "eligibility",
    "transition", "structural_and_representation_constraints", "conditions",
    "target_evidence", "geographic_splits", "candidate_bakeoff",
    "band_evidence", "morphology_and_visual_result",
    "packing_cook_and_serving_cost", "failure_abstention_and_release"
  ],
  "properties": {
    "schema_version": {"const": "microtopography-regime-release/1.0.0"},
    "identity": {
      "type": "object", "additionalProperties": false,
      "required": ["regime_id", "display_name", "catalog_id", "catalog_sha256", "row_revision"],
      "properties": {
        "regime_id": {"type": "string", "pattern": "^[a-z][a-z0-9_.-]+$"},
        "display_name": {"type": "string", "minLength": 1},
        "catalog_id": {"const": "estonia-microtopography-phenomena/2026-07-13.1"},
        "catalog_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "row_revision": {"type": "integer", "minimum": 1}
      }
    },
    "physical_definition": {
      "type": "object", "additionalProperties": false,
      "required": ["surface_semantics", "phenomena", "claimed_wavelength_m", "excluded_features", "transfer_claim"],
      "properties": {
        "surface_semantics": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "phenomena": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "claimed_wavelength_m": {"$ref": "#/$defs/range"},
        "excluded_features": {"type": "array", "items": {"type": "string"}},
        "transfer_claim": {"type": "string", "minLength": 1}
      }
    },
    "eligibility": {
      "type": "object", "additionalProperties": false,
      "required": ["predicate_language", "predicate", "required_evidence", "mutually_exclusive_with", "unknown_result"],
      "properties": {
        "predicate_language": {"const": "laas-condition-ast/1"},
        "predicate": {"$ref": "#/$defs/predicate"},
        "required_evidence": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "mutually_exclusive_with": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "unknown_result": {"const": "abstain"}
      }
    },
    "transition": {
      "type": "object", "additionalProperties": false,
      "required": ["physical_domain", "support_geometry", "signed_distance_field", "inner_m", "outer_m", "kernel", "compatible_mixture_group", "boundary_policy"],
      "properties": {
        "physical_domain": {"enum": ["watershed", "mire_unit", "stand", "managed_parcel", "shore_reach", "outcrop", "corridor", "disturbance_event", "other_explicit"]},
        "support_geometry": {"type": "array", "items": {"type": "string"}},
        "signed_distance_field": {"type": ["string", "null"]},
        "inner_m": {"type": "number", "minimum": 0},
        "outer_m": {"type": "number", "minimum": 0},
        "kernel": {"enum": ["hard_when_physically_discontinuous", "linear", "smoothstep", "measured"]},
        "compatible_mixture_group": {"type": ["string", "null"]},
        "boundary_policy": {"enum": ["explicit_weighted_mixture", "mutually_exclusive", "abstain"]}
      }
    },
    "structural_and_representation_constraints": {
      "type": "object", "additionalProperties": false,
      "required": ["direct_authorities", "protected_features", "forbidden_residual_masks", "non_heightfield_masks", "reapply_after_each_stage"],
      "properties": {
        "direct_authorities": {"type": "array", "items": {"type": "string"}},
        "protected_features": {"type": "array", "items": {"type": "string"}},
        "forbidden_residual_masks": {"type": "array", "items": {"type": "string"}},
        "non_heightfield_masks": {"type": "array", "items": {"type": "string"}},
        "reapply_after_each_stage": {"const": true}
      }
    },
    "conditions": {
      "type": "object", "additionalProperties": false,
      "required": ["required_fields", "optional_fields", "support_radii_m", "confidence_fields", "missingness_channels", "snapshot_ids"],
      "properties": {
        "required_fields": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": true},
        "optional_fields": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "support_radii_m": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "number", "minimum": 0}}},
        "confidence_fields": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "missingness_channels": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "snapshot_ids": {"type": "object", "additionalProperties": {"type": "string"}}
      }
    },
    "target_evidence": {
      "type": "object", "additionalProperties": false,
      "required": ["target_record_ids", "estonia_site_ids", "foreign_site_ids", "surface_semantics", "effective_resolution_m", "measurement_error_m", "mtf_or_transfer_function", "support_masks", "human_qa_masks", "licenses", "artifact_sha256", "transfer_ceiling", "qualification_status"],
      "properties": {
        "target_record_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "estonia_site_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "foreign_site_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "surface_semantics": {"type": "array", "items": {"type": "string"}},
        "effective_resolution_m": {"type": ["number", "null"], "exclusiveMinimum": 0},
        "measurement_error_m": {"$ref": "#/$defs/error"},
        "mtf_or_transfer_function": {"type": ["string", "null"]},
        "support_masks": {"type": "array", "items": {"type": "string"}},
        "human_qa_masks": {"type": "array", "items": {"type": "string"}},
        "licenses": {"type": "array", "items": {"type": "string"}},
        "artifact_sha256": {"type": "array", "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "uniqueItems": true},
        "transfer_ceiling": {"enum": ["none", "foreign_method_only", "estonia_pilot", "estonia_national"]},
        "qualification_status": {"enum": ["not_started", "failed", "insufficient", "passed"]}
      }
    },
    "geographic_splits": {
      "type": "object", "additionalProperties": false,
      "required": ["grouping_unit", "train", "development", "blind_test", "sensor_holdout", "taevaskoda_excluded", "leakage_audit"],
      "properties": {
        "grouping_unit": {"enum": ["site", "peatland", "campaign"]},
        "train": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "development": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "blind_test": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "sensor_holdout": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "taevaskoda_excluded": {"const": true},
        "leakage_audit": {"enum": ["not_run", "failed", "passed"]}
      }
    },
    "candidate_bakeoff": {
      "type": "object", "additionalProperties": false,
      "required": ["preregistration_id", "corrected_only_baseline", "candidate_ids", "frozen_budget_ids", "required_families", "omission_justifications", "allow_no_winner"],
      "properties": {
        "preregistration_id": {"type": ["string", "null"]},
        "corrected_only_baseline": {"const": true},
        "candidate_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "frozen_budget_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "required_families": {"type": "array", "items": {"enum": ["contextual_exemplar", "conditional_simulation", "process_event", "deterministic_regressor", "conditional_gan", "pixel_diffusion", "hybrid"]}, "uniqueItems": true},
        "omission_justifications": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}},
        "allow_no_winner": {"const": true}
      }
    },
    "band_evidence": {
      "type": "object", "additionalProperties": false,
      "required": ["wavelength_0_25_to_1_m", "wavelength_0_125_to_0_25_m"],
      "properties": {
        "wavelength_0_25_to_1_m": {"$ref": "#/$defs/band"},
        "wavelength_0_125_to_0_25_m": {"$ref": "#/$defs/band"}
      }
    },
    "morphology_and_visual_result": {
      "type": "object", "additionalProperties": false,
      "required": ["preregistered_gate_id", "measured_result_ids", "blind_render_result_ids", "real_vs_real_reference_id", "failure_classes", "decision", "decision_rationale"],
      "properties": {
        "preregistered_gate_id": {"type": ["string", "null"]},
        "measured_result_ids": {"type": "array", "items": {"type": "string"}},
        "blind_render_result_ids": {"type": "array", "items": {"type": "string"}},
        "real_vs_real_reference_id": {"type": ["string", "null"]},
        "failure_classes": {"type": "array", "items": {"type": "string"}},
        "decision": {"enum": ["not_run", "target_evidence_insufficient", "no_candidate_survives", "tie_no_owner", "research_owner", "pilot_owner", "national_owner"]},
        "decision_rationale": {"type": "string"}
      }
    },
    "packing_cook_and_serving_cost": {
      "type": "object", "additionalProperties": false,
      "required": ["measurement_status", "packed_bytes_per_km2", "entropy_bytes_per_chunk", "cook", "training", "inference", "scratch_bytes_per_km2", "retry_rate", "egress_bytes_per_km2", "recook_cadence"],
      "properties": {
        "measurement_status": {"enum": ["not_measured", "measured"]},
        "packed_bytes_per_km2": {"type": ["number", "null"], "minimum": 0},
        "entropy_bytes_per_chunk": {"$ref": "#/$defs/percentiles"},
        "cook": {"$ref": "#/$defs/cost"},
        "training": {"$ref": "#/$defs/cost"},
        "inference": {"$ref": "#/$defs/cost"},
        "scratch_bytes_per_km2": {"type": ["number", "null"], "minimum": 0},
        "retry_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "egress_bytes_per_km2": {"type": ["number", "null"], "minimum": 0},
        "recook_cadence": {"type": ["string", "null"]}
      }
    },
    "failure_abstention_and_release": {
      "type": "object", "additionalProperties": false,
      "required": ["status", "band_disposition", "selected_owner_id", "frozen_evidence_ids", "failure_action", "abstention_codes", "representation_limit_action", "release_extent_id", "approval_id"],
      "properties": {
        "status": {"enum": ["unsupported", "research", "pilot", "national"]},
        "band_disposition": {
          "type": "object", "additionalProperties": false,
          "required": ["wavelength_0_25_to_1_m", "wavelength_0_125_to_0_25_m"],
          "properties": {
            "wavelength_0_25_to_1_m": {"enum": ["unsupported", "research_only", "released", "target_evidenced_effectively_smooth", "forbidden"]},
            "wavelength_0_125_to_0_25_m": {"enum": ["unsupported", "research_only", "released", "target_evidenced_effectively_smooth", "forbidden"]}
          }
        },
        "selected_owner_id": {"type": ["string", "null"]},
        "frozen_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "failure_action": {"const": "retain_corrected_structure_and_fail_claimed_morphology_coverage"},
        "abstention_codes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "representation_limit_action": {"enum": ["mask_and_report", "not_applicable"]},
        "release_extent_id": {"type": ["string", "null"]},
        "approval_id": {"type": ["string", "null"]}
      }
    }
  },
  "$defs": {
    "range": {
      "type": "object", "additionalProperties": false,
      "required": ["minimum", "maximum"],
      "properties": {"minimum": {"type": "number", "minimum": 0}, "maximum": {"type": "number", "exclusiveMinimum": 0}}
    },
    "error": {
      "type": "object", "additionalProperties": false,
      "required": ["horizontal_p95", "vertical_p50", "vertical_p95", "method"],
      "properties": {
        "horizontal_p95": {"type": ["number", "null"], "minimum": 0},
        "vertical_p50": {"type": ["number", "null"], "minimum": 0},
        "vertical_p95": {"type": ["number", "null"], "minimum": 0},
        "method": {"type": ["string", "null"]}
      }
    },
    "predicate": {
      "oneOf": [
        {"type": "object", "additionalProperties": false, "required": ["all"], "properties": {"all": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/predicate"}}}},
        {"type": "object", "additionalProperties": false, "required": ["any"], "properties": {"any": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/predicate"}}}},
        {"type": "object", "additionalProperties": false, "required": ["not"], "properties": {"not": {"$ref": "#/$defs/predicate"}}},
        {"type": "object", "additionalProperties": false, "required": ["field", "op", "value"], "properties": {"field": {"type": "string"}, "op": {"enum": ["eq", "ne", "in", "not_in", "lt", "lte", "gt", "gte", "contains", "intersects", "confidence_gte"]}, "value": {}}}
      ]
    },
    "band": {
      "type": "object", "additionalProperties": false,
      "required": ["status", "signal_rms_m", "measurement_error_rms_m", "signal_to_error_ratio", "confidence_interval", "evidence_ids", "supports_claim"],
      "properties": {
        "status": {"enum": ["not_measured", "indistinguishable", "measurable"]},
        "signal_rms_m": {"type": ["number", "null"], "minimum": 0},
        "measurement_error_rms_m": {"type": ["number", "null"], "minimum": 0},
        "signal_to_error_ratio": {"type": ["number", "null"], "minimum": 0},
        "confidence_interval": {"type": ["string", "null"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "supports_claim": {"type": "boolean"}
      }
    },
    "percentiles": {
      "type": "object", "additionalProperties": false,
      "required": ["p50", "p95", "p99"],
      "properties": {"p50": {"type": ["number", "null"], "minimum": 0}, "p95": {"type": ["number", "null"], "minimum": 0}, "p99": {"type": ["number", "null"], "minimum": 0}}
    },
    "cost": {
      "type": "object", "additionalProperties": false,
      "required": ["hardware", "wall_seconds_per_km2", "accelerator_seconds_per_km2", "peak_memory_bytes", "support_overlap_multiplier"],
      "properties": {
        "hardware": {"type": ["string", "null"]},
        "wall_seconds_per_km2": {"type": ["number", "null"], "minimum": 0},
        "accelerator_seconds_per_km2": {"type": ["number", "null"], "minimum": 0},
        "peak_memory_bytes": {"type": ["number", "null"], "minimum": 0},
        "support_overlap_multiplier": {"type": ["number", "null"], "minimum": 1}
      }
    }
  }
}
```

### 3.3 Materialized normative artifacts

The executable contract is materialized under `review/contracts/`; the embedded
schema and status listing in this document are explanatory mirrors, not a reason
to reconstruct files by hand:

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `contracts/regime-release-registry.schema.json` | Draft 2020-12 schema for the complete 26-row registry; every row retains the 13 strict groups and `additionalProperties: false` | `07a9e6a2486c22e70918392194eea897d9e7810aa12499fad596d67b4629268b` |
| `contracts/regime-release-rows.initial.json` | 26 resolved initial rows with physical eligibility, transition, evidence, bakeoff, band, cost, and release state | `74257a057d7aeaf196d0fdccff374cbcafd1a902efcc7723eea4c81fabac7b48` |
| `contracts/regime-initial-status.json` | compact one-to-one status/band/abstention manifest | `3fa08bfa5d290d72378afcd02904e33c53c8db991b3c98da757a40b7a8d33cda` |

Validate the resolved registry with:

```sh
jsonschema \
  -i docs/deep-research/microtopography-generation/review/contracts/regime-release-rows.initial.json \
  docs/deep-research/microtopography-generation/review/contracts/regime-release-registry.schema.json
```

The initial registry is deliberately fail-closed. Twenty-five rows are
`unsupported`; `peat.raised_bog` is only `research` and binds the foreign Moore
archive without claiming that Track A has qualified its transfer/error model.
No row has Estonia target sites, measurable band evidence, a selected owner,
release extent, user approval, or measured serving cost. Changing any of those
facts creates a new row revision and new artifact hashes.

Schema-level validation is necessary, not sufficient. Cross-field gates are:

- `pilot` requires at least two Estonia development sites, one untouched Estonia
  blind site, a passed qualification and leakage audit, measured band evidence,
  a `pilot_owner` decision, measured cost, a bounded extent and user approval.
- `national` requires the same plus `estonia_national` transfer, a
  `national_owner` decision, national extent, national coverage audit, and
  approval. More sites are required until held-out results stabilize; three is a
  go/no-go floor, not a sufficiency theorem.
- `research` may use foreign evidence and may name a `research_owner`, but every
  band remains `research_only` or `unsupported` and cannot enter release assets.
- `target_evidenced_effectively_smooth` requires Estonia target evidence for the
  exact condition stratum and a 95% upper confidence bound on band signal no
  greater than the 95% upper measurement-error bound. Lack of data, low point
  density, smoothing, occlusion, or a failed acquisition is not smoothness.
- `released` requires `supports_claim: true`, a non-null owner, passed measured
  and blind-render gates, measured p50/p95/p99 packing entropy and complete cost.

## 4. Eligibility And Transition Semantics

`laas-condition-ast/1` is evaluated with three-valued logic: `true`, `false`, and
`unknown`. A missing field, uncovered map, stale source outside its accepted date,
failed confidence threshold, or schema drift returns `unknown`. Every row declares
`unknown_result: abstain`; `unknown` is never coerced to `false` to expose a lower
priority fallback.

The dispatcher evaluates all rows; it does not iterate in priority order.

1. Zero eligible rows yields `unsupported_morphology`.
2. More than one eligible mutually exclusive row is an ambiguity error.
3. Compatible overlap is legal only when all rows name the same non-null
   `compatible_mixture_group` and the transition contract explicitly supplies
   physical-domain weights.
4. Mixture weights come only from the registered signed-distance/confidence fields
   and kernel. They are normalized after excluding failed rows. They may not be
   inverse distance to a class centroid, nearest polygon, or model confidence
   invented by the candidate.
5. A map edge changes probability/support. It does not become height relief.
6. Long-range identity is the physical domain ID: watershed, connected mire unit,
   stand, managed parcel, shore reach, outcrop, corridor or disturbance event.
   Chunk ID is never a physical-domain ID.

## 5. Initial Estonia Registry

The following YAML mirrors the authoritative
`contracts/regime-initial-status.json`. It is joined one-to-one to the full rows
in `contracts/regime-release-rows.initial.json` through `regime_id`; the build
fails if an ID is missing, duplicated, unknown to the pinned phenomenon catalog,
or has no resolved 13-group row. `research` does not authorize packing.

```yaml
schema_version: microtopography-regime-initial-status/1.0.0
catalog_id: estonia-microtopography-phenomena/2026-07-13.1
catalog_sha256: 6ca03d5bb552f6ad83e818da1d16d3fddf7d4006241990a90f6ffc21ab13ee4d
no_default_row: true
rows:
  - {regime_id: peat.raised_bog, status: research, bands: [research_only, research_only], reason: moore_foreign_plot_local_moss_only}
  - {regime_id: peat.fen_transitional, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_estonia_targets_and_moore_does_not_span_fen_states}
  - {regime_id: peat.drained_cut, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_dated_target_sites}
  - {regime_id: forest.pit_mound, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_estonia_ground_targets}
  - {regime_id: forest.floor_ordinary, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_estonia_ground_targets}
  - {regime_id: forest.managed_clearcut, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_dated_operation_targets}
  - {regime_id: agriculture.ploughed, status: unsupported, bands: [unsupported, unsupported], reason: foreign_measurement_evidence_only}
  - {regime_id: agriculture.seedbed_harrowed_rolled, status: unsupported, bands: [unsupported, unsupported], reason: foreign_measurement_evidence_only}
  - {regime_id: grassland.pasture_meadow, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_target_sites}
  - {regime_id: grassland.yard_turf, status: unsupported, bands: [unsupported, unsupported], reason: no_qualified_subtype_targets}
  - {regime_id: sand.exposed, status: unsupported, bands: [unsupported, unsupported], reason: no_estonia_material_state_targets}
  - {regime_id: sand.dune_aeolian, status: unsupported, bands: [unsupported, unsupported], reason: no_estonia_context_and_target_pairs}
  - {regime_id: shore.beach_sand, status: unsupported, bands: [unsupported, unsupported], reason: no_dated_estonia_target_series}
  - {regime_id: shore.shingle_gravel_cobble, status: unsupported, bands: [unsupported, unsupported], reason: no_material_matched_targets_and_heightfield_limits}
  - {regime_id: glacial.till_plain, status: unsupported, bands: [unsupported, unsupported], reason: no_estonia_target_scale_surfaces}
  - {regime_id: glacial.gravel_esker_outwash, status: unsupported, bands: [unsupported, unsupported], reason: no_estonia_target_scale_surfaces}
  - {regime_id: carbonate.alvar_thin_soil, status: unsupported, bands: [unsupported, unsupported], reason: no_estonia_alvar_height_targets}
  - {regime_id: carbonate.pavement_karst, status: unsupported, bands: [unsupported, unsupported], reason: unresolved_measurements_and_heightfield_limits}
  - {regime_id: outcrop.sandstone, status: unsupported, bands: [unsupported, unsupported], reason: no_lithology_matched_estonia_targets_and_wall_is_non_heightfield}
  - {regime_id: outcrop.carbonate, status: unsupported, bands: [unsupported, unsupported], reason: no_lithology_matched_estonia_height_targets}
  - {regime_id: slope.colluvium_talus, status: unsupported, bands: [unsupported, unsupported], reason: no_source_to_toe_context_targets_and_non_heightfield_blocks}
  - {regime_id: fluvial.floodplain, status: unsupported, bands: [unsupported, unsupported], reason: no_dated_estonia_flood_target_series}
  - {regime_id: fluvial.active_bar_bed, status: unsupported, bands: [unsupported, unsupported], reason: no_flow_and_bathymetry_conditioned_targets}
  - {regime_id: fluvial.rill_gully_seep_spring, status: unsupported, bands: [unsupported, unsupported], reason: no_connected_network_target_sites}
  - {regime_id: coast.wetland, status: unsupported, bands: [unsupported, unsupported], reason: no_baltic_inundation_conditioned_targets}
  - {regime_id: technogenic.disturbed, status: unsupported, bands: [unsupported, unsupported], reason: no_subtype_specific_dated_targets}
```

These rows are exhaustive for the pinned catalog, not exhaustive forever. A newly
discovered Estonia phenomenon blocks coverage until the catalog is revised and an
explicit row is added; it is not mapped to the closest existing ID.

## 6. Coverage Audit

### 6.1 Terminal classification

The coverage auditor runs on every unique interior sample center of every
published LOD `-2` chunk using the same world-coordinate oracle, condition
snapshot, masks and dispatcher as synthesis. Aprons are checked separately
against their owning neighbor. It streams rather than retaining a national mask,
but it evaluates every sample.

For each sample and each claimed band, exactly one terminal is emitted:

| Code | Pass condition |
|---|---|
| `R` released morphology | one `pilot`/`national` row covers the release extent, its band is `released`, all row gates pass, and any compatible mixture is explicit |
| `S` target-evidenced smooth | one row's exact condition stratum has `target_evidenced_effectively_smooth` with the cross-field measurement gate above |
| `F` forbidden residual | a registered open-water/hard-structure mask owns the sample; residual is absent after every stage |
| `N` non-heightfield | validated representation mask owns the sample; the point is excluded and separately reported |
| `U` unsupported | no eligible released/smooth row, unknown required evidence, research-only row, or no row |
| `A` ambiguity/error | multiple exclusive rows, invalid mixture, missing catalog row, schema/hash drift, or inconsistent masks |

`U` and `A` fail any pilot/national morphology coverage claim. `F` and `N` do not
inflate morphology success: reports show them separately by area, condition and
source. Corrected structural-only publication can proceed under a structural
claim while morphology coverage fails, but the release must state that scope.

### 6.2 Complete audit algorithm

```text
for each published lod_minus_2_chunk in canonical (z, x) order:
    initialize counts[R,S,F,N,U,A] = 0
    initialize sha256 over ordered terminal bytes and resolved row revisions
    for each unique interior sample center in canonical row-major order:
        load condition values, confidence, missingness, source snapshot IDs
        if any required source schema/hash differs from the row: emit A
        else if validated non_heightfield mask: emit N
        else if validated forbidden residual mask: emit F
        else:
            evaluate every catalog row with three-valued condition logic
            reject exclusive overlap or unregistered mixture as A
            resolve explicit compatible mixture, if any
            for each claimed band:
                if all resolved rows pass release gates and band=released: emit R
                else if exact stratum passes effectively-smooth gate: emit S
                else: emit U
        update counts and hash
    verify east/south apron terminals equal the owning neighbor's interior terminals
    write chunk ID, extent, row revisions, counts, terminal-stream hash, pass/fail
aggregate exact counts and area by terminal, regime, band, release status,
condition stratum and source snapshot; fail on any U or A for a full coverage claim
```

The auditor also fails if summed terminal counts differ from the exact number of
unique interior samples, if release extents have a gap/overlap, or if regenerating
with different worker count/order/AOI changes any terminal hash. A national report
must publish counts and area for every terminal, including unsupported and
non-heightfield territory. Sampling a subset, checking only chunk centers, or
reporting percentage coverage rounded to 100% is not a proof.

## 7. Moore Peat Qualification And No-Winner Gate

### 7.1 Verified scope

The Moore paper and Zenodo archive provide:

- 68 disconnected DEM plots totaling `309.1387 m2` of finite cells;
- 50 plots from one Red Earth Creek campaign and 18 plot-analysis surfaces from
  nine northern peatlands in Canada, the United States and Sweden;
- plots of `3.2-10.1 m2`, mostly only a few metres across;
- a `0.01 m` natural-neighbor grid followed by a `0.03 x 0.03 m` mean filter;
- laboratory x/y/z RMSE below `0.01 m` and field elevation median absolute
  disagreement `0.018 m`;
- moss surfaces exposed by manually clipping vascular vegetation; and
- CC BY 4.0 access, with local archive SHA-256
  `044413bb87171776b409172d29fbde16341b228dc43b694cffe9f02a88640a67`.

The companion study found mostly `1-10 m` site-scale variability from manual
transects at two sites and about `32 m2` from ten randomly located plots to capture
about 95% of variance at one unpatterned site. That is sampling evidence, not a
2D training window or transfer theorem.

### 7.2 Claims Moore may test

After Track A passes, Moore may test only:

1. loading, masking, metric scaling, band analysis and site-grouped split code;
2. plot-local clipped-moss height, slope, normal, curvature and local-relief
   distributions after applying the measured target transfer function;
3. plot-local variograms/spectra and morphology descriptors over wavelengths
   from `0.125 m` through `1.0 m`, but only in plots whose valid-mask geometry and
   per-band signal-to-measurement-error gate support the descriptor;
4. local hummock/lawn/hollow elevation-class distributions and relative relief;
5. generalization from development peatlands to whole held-out foreign peatlands;
6. whether a candidate produces plot-local structure distinguishable from generic
   noise, repetition and the corrected-only baseline.

The lower bound is the packed grid's approximate two-sample wavelength, not a
claim that the `0.03 m` acquisition filter preserves every `0.125 m` feature. The
band must still pass measured MTF/error qualification. The conservative `1.0 m`
upper bound prevents a few-metre plot from being credited with long-range 2D
organization.

Moore may not test or qualify:

- 128 m chunk context, whole-mire ridge/string/pool layout, drainage organization,
  class boundaries or cross-regime transitions;
- untouched vegetation, shrub/tussock surface semantics, open-water geometry,
  drained/cut/restored/wooded peat, or seasonal water-table response;
- Estonian peat, soil, geology, climate, orthophoto or Maa-amet degradation
  transfer;
- national partition invariance or national repetition quality; or
- any mineral, forest, agricultural, rock, shore or technogenic regime.

### 7.3 Track A gate precedes Track C

Track C candidate output must not be viewed, ranked or called a quality result
until Track A freezes and passes all of these:

1. archive and per-artifact hashes; license and attribution;
2. exact site/peatland grouping for all 68 plots, with no plot from a blind
   peatland in training or development;
3. surface semantics, vascular-clipping status, valid/interpolated/reference-
   object masks and manual QA;
4. horizontal/vertical error model and the `0.03 m` filter transfer function;
5. exact valid area, plot geometry and supported descriptor wavelengths;
6. per-band signal-to-measurement-error estimates with confidence intervals;
7. preregistered candidates, budgets, descriptors, hard failures, blind render
   protocol and decision rule; and
8. a written transfer ceiling of `foreign_method_only`.

Failure or insufficient evidence in items 2-6 yields
`target_evidence_insufficient`; Track C makes no quality conclusion. Completing
that diagnosis completes the research stage. It is not a reason to relax a gate.

### 7.4 Candidate decision

Every candidate, including corrected-only, exemplar/statistical/process,
regressor, GAN, diffusion and justified hybrids, receives the same split and
frozen evidence. A candidate survives only if it:

- stays within valid measured masks and the registered surface semantics;
- is non-inferior to held-out real-vs-real variation for every preregistered
  descriptor whose band passed Track A;
- passes repetition, memorization, boundary, seed and measurement-artifact tests;
- passes the blind plot-level geometry/render gate over corrected-only; and
- reports complete research compute and packing cost.

The decision enum is intentionally non-forcing:

- `target_evidence_insufficient`: Track A cannot support the claimed test;
- `no_candidate_survives`: every candidate fails at least one hard gate;
- `tie_no_owner`: survivors are not distinguishable under the preregistered rule;
- `research_owner`: one candidate survives and wins the preregistered rule.

Only `research_owner` names a Moore owner, and that owner's maximum release status
is `research`. It cannot satisfy the three-Estonia-site activation gate, emit a
pilot/national band, or serve as a nearest fallback for Estonia. A later Estonia
peat bakeoff starts a new evidence row and may reject the Moore owner.

## 8. Normalized Source Ledger Contract

### 8.1 Migrated v1 ledger audit

The migrated v1 snapshot contained 104 evidence records plus 20 separately held
repository records. Its preserved SHA-256 is
`e60b9cd80386f61b83192dbd1595bb256b757aad944e19a049be68c7a3ae667f`.
The current v2 file contains 124 unified records and validates against
`library/sources.schema.json`.

| Required fact | Current state |
|---|---|
| ID, title, year, type, URL/status/review summary | present on all 104 records |
| authors | absent on all 104 |
| per-record artifact SHA-256 | absent on all 104; a separate 127-artifact manifest exists |
| access attempt/result/date | absent as a normalized object; partly encoded in status strings |
| license | present on 4 of 104; absent on 100 |
| exact read scope | one coarse status string; no sections/pages or access basis |
| repository officialness/revision/license | 20 revisions are in a separate array and not linked by stable record ID |
| data/checkpoint availability | partly buried in prose/status; not normalized |
| demonstrated input/output scales and degradation/data regime | mostly prose in audit documents, not machine fields |
| evaluation, direct finding, limitation, permitted transfer | collapsed into one `applicability` string |
| invalid HTML, paywall, request-only, code-only, analogue, unresolved | represented inconsistently across status/prose |

The v2 ledger remains evidence input, not automatic release authority. It has 88
records with authors verified from Crossref, arXiv, or a local primary title
page; four paper records retain explicit unknown authors. It has 25 verified
license records, three pinned repositories with a verified absent license file,
and 96 explicit unknown licenses. Eight deeply audited works have split direct
finding/limitation/transfer fields; 96 v1 evidence records remain explicitly
`legacy_summary_not_split` and cannot authorize consequential judgment. This is
deliberate: bulk-filling unknown facts would manufacture provenance.

### 8.2 Required per-record schema

`sources.json` schema version 2 uses one `records` array for papers, datasets,
official services, licenses and repositories and is governed by
`library/sources.schema.json`. A paper and its repository remain separate records
linked through `related_record_ids`. Every record has these required objects;
unknowns are explicit enums, never missing:

```yaml
schema_version: microtopography-source-ledger/2.0.0
record:
  id: stable-lowercase-id
  kind: paper | dataset | repository | official_service | license | thesis | report
  citation:
    authors: [{family: Wang, given: Yinhuai, orcid: null}]
    title: Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model
    year: 2023
    venue: ICLR
    doi: 10.48550/arXiv.2212.00490
  identifiers:
    canonical_url: https://...
    related_record_ids: [repo.ddnm]
  access:
    status: open_full_text | paywalled | request_only | metadata_only | blocked | unresolved
    checked_at: 2026-07-13
    result: exact factual access outcome
    full_text_basis: local_artifact | official_online | author_copy | none
    flags: [analogue] # allowed: invalid_html, code_only, analogue, unresolved, retracted
  review:
    status: not_read | abstract_only | methods_only | methods_results | full_primary
    scope: [sections 3-4, appendix]
    reviewer_record: task/review artifact ID
    consequential_judgment_allowed: true
  artifacts:
    - path: papers/example.pdf
      sha256: 64-lowercase-hex
      media_type: application/pdf
      validation: valid_pdf | valid_archive | invalid_html | not_local
  licensing:
    status: verified | absent | unclear | not_applicable
    name: CC BY 4.0
    spdx: CC-BY-4.0
    url: https://...
    applies_to: paper | data | code | metadata
    training_use: allowed | forbidden | legal_review | unknown
    derivative_distribution: allowed | forbidden | legal_review | unknown
    restrictions: []
  software:
    availability: official | community | none_found | not_applicable
    repository_url: https://...
    official: true | false | null
    revision: 40-hex-commit-or-null
    snapshot_path: repos/example.tar.gz
    snapshot_sha256: 64-hex-or-null
    license_status: verified | absent | unclear | not_applicable
  data_and_checkpoints:
    data_status: open_downloaded | open_not_downloaded | request_only | unavailable | none | unknown
    data_record_ids: []
    checkpoint_status: open_downloaded | open_not_downloaded | unavailable | none | unknown
    checkpoint_artifact_ids: []
  demonstration:
    task: exact task performed
    geography_and_regime: exact demonstrated domain
    input_scale: {sample_pitch_m: null, support_m: null, extent_m: null}
    output_scale: {sample_pitch_m: null, support_m: null, extent_m: null}
    degradation_or_observation: exact synthetic/real operator and sensor regime
    training_data: exact data identity and split unit
    evaluation: exact metrics, visual protocol and held-out design
  evidence:
    direct_finding: claim directly established by paper/code/data
    limitation: direct limitation or untested boundary
    permitted_project_transfer: narrowly allowed LAAS use
    prohibited_inference: claim this record must not be used to support
  mentioned_in: [r1, root-claim-audit]
```

Machine validation requires: non-empty authors for a paper; a SHA-256 for every
local artifact; a 40-hex revision and snapshot hash for a local repository;
explicit license and access enums; demonstrated scales with units or explicit
`null`; separate direct finding/limitation/transfer; and a checkpoint state for
every learned method. `license: absent` and `checkpoint_status: unavailable` are
valid evidence facts but block reuse that requires them.

### 8.3 Migration procedure

1. Join every current `local_file` to `metadata/SHA256SUMS`; fail on zero or
   multiple non-identical matches.
2. Convert each repository entry into a stable record and link papers explicitly.
3. Read title-page/official metadata for authors; never infer a full author list
   from `et al.`.
4. Recheck paper, code, dataset and checkpoint licenses independently. A paper's
   CC license does not license code or data.
5. Convert status prose to enumerated access plus a retained factual result.
6. Record exact sections/pages/code paths actually read. Consequential method
   selection requires relevant full methods/results and official code where it
   exists.
7. Extract demonstrated scales, degradation, data, split and evaluation from the
   primary source; preserve `null` where the source does not state a value.
8. Split `applicability` into direct finding, limitation, permitted transfer and
   prohibited inference; label all project architecture judgments as inference.
9. Validate every hash, local PDF/archive and repository snapshot, then write the
   new ledger atomically. Preserve the v1 file or its hash as migration provenance.

## 9. Compact Architecture Evidence Table

This table is the minimum compact trace the standalone spec should embed. It does
not replace the normalized ledger.

| Primary work | Demonstrated scale/data/degradation and evaluation | Code/data/checkpoint; access/license | Direct finding | Limitation and permitted transfer |
|---|---|---|---|---|
| Eric Guérin, Julie Digne, Eric Galin, Adrien Peytavie (2016), *Sparse Representation of Terrains for Procedural Modeling* | paired low/high terrain patches; outputs to `8192 x 8192`; exemplar and input must contain similar terrain; timings and visual terrain examples | full paper local; official MATLAB repo revision `5b83d65315e9845401df78f6a104c21f9bd473d5`, MIT; no national data/checkpoint | paired coefficients/atoms can transfer exemplar detail while conditioning on low terrain | no material/geology semantics or guaranteed geomorphological/small-scale coherence; use as contextual-exemplar baseline/component, not national owner |
| Oscar Argudo, Albert Chica, Carlos Andujar (2018), *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks* | `15 m` DEM + `1 m` orthophoto -> `2 m` DEM on selected Pyrenees/Tyrol alpine terrain, about `400 m` training tiles; Euclidean height loss and metre-scale reconstruction errors | full paper local; no official code/checkpoint found; paper access does not establish code reuse license | registered optical imagery can improve terrain structural inference | vegetation/shadows create false height and scale is far above 6.25 cm; transfer only the masked orthophoto-conditioning hypothesis |
| Yinhuai Wang, Jiwen Yu, Jian Zhang (2023), *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model* | natural-image inverse problems with explicit linear `A`; SR uses synthetic average pooling/replication on ImageNet/CelebA and evaluates restoration quality | full paper local; official repo revision `00b58eac7843a4c99114fd8fa42da7aa2b6808af`, MIT; pretrained natural-image models, not terrain data | exact range/null-space projection enforces a known linear observation | Maa-amet is not known block averaging and contains correctable error; transfer only soft observation-likelihood reasoning, never exact raw-DTM preservation or its image prior |
| Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel (2023), *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation* | overlapping Stable Diffusion image/latent crops fused by least-squares/weighted pixel consensus; panorama/control evaluation | full paper local; official code revision `69bcdcef437dfdbf48c53624d6bf6f397b5f4894`; no license file in audited snapshot; model/data licenses are separate | per-step overlap consensus can reconcile local denoiser predictions | no guarantee for terrain gradients, normals, curvature, drainage or hierarchy; transfer only as a tiling challenger after terrain-native validation |
| Hugo Schott, Eric Galin, Eric Guérin, Axel Paris, Adrien Peytavie (2024), *Terrain Amplification Using Multi-scale Erosion* | hydraulic/thermal amplification through multiscale refinement to `8192 x 8192`; compares supported terrain examples to selected learned/procedural methods | full author manuscript local; implementation status/license must be normalized before reuse; no Estonia target data/checkpoint | multiscale process simulation can add coherent erosion/deposition and drainage structure | one process family, whole-map dependencies and patch-boundary/cost risk; use as a narrow erodible-slope/process challenger |
| Paul A. Moore, Maxwell C. Lukenbach, Dan K. Thompson, Nick Kettridge, Gustaf Granath, James M. Waddington (2019), *Assessing the Peatland Hummock-Hollow Classification Framework Using High-Resolution Elevation Models* | 68 foreign clipped-moss plots, `3.2-10.1 m2`; `0.01 m` grid with `0.03 m` mean filter; lab RMSE `<0.01 m`, field median absolute difference `0.018 m`; plot/site morphology statistics | full CC BY 4.0 paper and CC BY 4.0 Zenodo data local; archive SHA `044413...40a67`; no generative checkpoint | real peat microforms are multi-class and measurable; local distributions and scale relations can evaluate peat candidates | only `309.1387 m2`, disconnected, foreign and altered vegetation; transfer only plot-local foreign method validation under Section 7 |
| Philip Marzahn, Moritz Seidel, Ralf Ludwig (2012), *Decomposing Dual Scale Soil Surface Roughness for Microwave Remote Sensing Applications* | photogrammetric `2 mm` grids over `6-22 m2` worked agricultural fields; geostatistical decomposition separates seedbed-row and wheel-track scales | full CC BY 3.0 paper local; no production geometry dataset/checkpoint in ledger | agricultural roughness has distinct operation-caused scales, directions and states | not Estonia-wide synthesis evidence; transfer causal decomposition, descriptors and acquisition requirements for agricultural challengers |
| Alexander Goslin (2026), *InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation* | global MERIT/ETOPO/climate at about `90 m`; `512 x 512` 90 m patches, hierarchical diffusion and overlap fusion; FID/latency/visual evaluation; roughly two weeks on RTX 3090 Ti | full preprint local; official terrain repo revision `82a0431281f21a6ec3d691a12ee61525de5b0790`; data/checkpoint/license status must be normalized | coordinate/seed-stable lazy overlapping diffusion is a credible unbounded generation mechanism | demonstrated terrain is tens-of-metres scale and runtime generation conflicts with project law; transfer only deterministic cook-side domain/tiling concepts, not weights or quality claim |

**Architecture judgment, not a paper finding:** pixel-space conditional diffusion is
a high-ceiling challenger for genuinely multimodal target-supported regimes. No
audited paper demonstrates that it is the best Estonia 1 m to 0.0625 m method.
Owner selection remains the preregistered per-regime bakeoff, and no-winner is
valid.

## 10. Spec Integration Requirements

The final standalone spec closes H4, H5 and H9 only if it:

- embeds the 26-row phenomenon decomposition or pins this exact catalog ID and
  hash while reproducing all release-critical meanings;
- requires all 13 release-row groups and the cross-field gates, not a reduced
  prose checklist;
- includes the initial status for all 26 rows and states that research emits no
  release geometry;
- makes the complete every-sample coverage audit a release artifact and reports
  `R/S/F/N/U/A` separately;
- forbids default/nearest substitution and treats missing conditions as abstention;
- runs Moore Track A before viewing/ranking Track C and preserves the exact scope,
  transfer ceiling and valid no-winner outcomes;
- embeds the compact architecture evidence table and labels architecture inference
  as inference;
- treats the delivered source-ledger v2 as a Stage 0 gate: any record whose
  authors, license, demonstrated scales, checkpoint/data status, or split
  finding/limitation/transfer remains unknown cannot supply that fact to
  authorize an owner; and
- requires implementers making difficult synthesis or consequential scientific/
  visual judgments to read the relevant full primary sources and official code and
  use `sol` at high effort when model selection is available, while routine
  ingestion, harness and test work stays at normal effort.

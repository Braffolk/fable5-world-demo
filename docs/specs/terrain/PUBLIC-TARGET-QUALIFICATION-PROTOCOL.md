# Public Target Qualification Protocol

**Status:** operational zero-budget evidence protocol; does not select a synthesis owner or authorize unqualified target conversion
**Date:** 2026-07-13
**Budget:** public data, existing local compute, and existing storage only; no paid capture, proprietary data, contractors, new hardware, cloud compute, or paid storage
**Scope:** decide whether a public observation can become a regime- and band-specific microtopography target or a bounded neighboring-analogue pilot input

## 1. Bound Evidence Snapshot

This draft is derived from the following exact files:

| Input | SHA-256 |
|---|---|
| `docs/specs/terrain/MICROTOPOGRAPHY.md` | `942698692ef58a8aa8b034fc9b8d243a91b63ee11ed7e787befe171d316fca3d` |
| `review/TAEVASKODA-ESTONIA-PUBLIC-SCAN-HUNT.md` | `0c5a777f07bce81ce2901ad91a537ffd303e07bb66de9ca5c46bb96a7c21d7ca` |
| `review/NORDIC-PUBLIC-TARGET-DATA-HUNT.md` | `4a74b7840e9b758539d052cab71e41e941aca0495ca0339558e771752a46a302` |
| `review/BALTIC-PUBLIC-TARGET-DATA-HUNT.md` | `42d92c3b41e208d16bbb9b12dbe9d503f2cb8b3f316ffd7bc1fa909f0e739538` |

Paths beginning with `review/` are relative to
`docs/deep-research/microtopography-generation/`.

This protocol does not assume that raw point clouds, classified points, DSMs,
meshes, or centimeter-spaced rasters are terrain target surfaces. It qualifies the
measurement, the surface interpretation, and the conversion separately. It ends
before synthesis-family bakeoff and must not favor diffusion, exemplars,
dictionaries, processes, statistics, or any other synthesis owner.

## 2. Fixed Scientific Objects

Qualification is per `regime_id`, condition stratum, surface contract, campaign,
and analysis band:

- `B1`: registered wavelengths `0.25-1.0 m`;
- `B2`: registered wavelengths `0.125-0.25 m`;
- canonical target lattice: `0.0625 m`;
- target surface: one metric, heightfield-valid physical surface with explicit
  inclusion and exclusion semantics;
- raw observation: immutable sensor output and its acquisition metadata, never the
  target itself;
- converted surface candidate: an auditable estimate derived from raw observations,
  accompanied by support, visibility, semantics, uncertainty, and invalid masks.

Band decomposition must use the canonical float64 `F4`/`R4` operators in Section
13.2 of the specification. An invalid input anywhere in a band operator's support
invalidates that output. Raster size, vertex count, point count, total-return
density, nominal GSD, and scanner precision are never substitutes for this protocol.

## 3. State Machine

Every candidate has one terminal or advancing state. Advancement is explicit; no
missing field is interpreted as success.

```text
catalogued
  -> probe_eligible
  -> retained_raw_candidate
  -> converted_surface_candidate
  -> qualified_exemplar | qualified_target_B1 | qualified_target_B2
  -> pilot_transfer_eligible

any state -> quarantined | rejected | request_only | calibration_only
```

- `qualified_exemplar` permits research, converter evaluation, or pretraining only.
- `qualified_target_B1/B2` permits the target to enter a method-neutral candidate
  bakeoff for that exact band and condition stratum. It does not select an owner.
- `pilot_transfer_eligible` permits only the bounded release in Section 10.
- No foreign-only result can receive `estonia_national` transfer.
- A candidate may pass `B1` and fail `B2`; failure may not be hidden by a single
  combined result.

## 4. Exact Preregistration Record

Before the first geometry byte is downloaded, check in one JSON record with
`schema="laas-public-target-qualification/0.1.0-draft"` and
`additionalProperties=false`. Every field below is required; use explicit `null`
only where the field definition permits it, and treat required scientific `null`
as failure rather than omission.

### 4.1 Identity and freeze

| Field | Required value |
|---|---|
| `protocol_id`, `revision`, `created_utc` | Stable ID, monotonic revision, ISO-8601 UTC |
| `prepared_by`, `approved_by`, `approval_utc` | Named roles; approval is required before target conversion |
| `canonical_spec_sha256`, `hunt_report_sha256s` | Exact hashes from the bound snapshot |
| `regime_catalog_id`, `regime_catalog_sha256`, `regime_id` | Pinned physical row |
| `claimed_bands` | Non-empty subset of `B1`, `B2` |
| `claim_text` | Narrow physical and geographic claim; no generic "forest" or "peat" claim |
| `forbidden_claims` | Explicit list including national transfer unless later qualified |

### 4.2 Budget and retention

| Field | Required value |
|---|---|
| `zero_purchase_required` | Must be `true` |
| `paid_services_allowed` | Must be `false` |
| `free_bytes_at_freeze`, `minimum_free_bytes`, `max_cache_bytes`, `max_persistent_derived_bytes` | Integer byte ceilings measured before download |
| `max_network_bytes` | Integer ceiling across retries; not an estimate |
| `max_cpu_hours`, `max_gpu_hours`, `max_wall_days` | Existing-machine ceilings |
| `raw_retention_mode` | `local_immutable`, `remote_immutable`, or `both`; remote requires stable DOI/file ID and verified checksum |
| `eviction_policy` | Only verified, re-downloadable raw files may be evicted; licenses, manifests, hashes, conversion recipes, masks, and audit results persist |
| `stop_before_ceiling_fraction` | Must be in `(0,1]`; default for this protocol is `0.90` |

No protocol may set a ceiling from expected availability. It records actual free
space and the user's existing-storage reservation. Exhausting a ceiling produces
`budget_exhausted`, not a request for paid infrastructure.

### 4.3 Dataset, site, campaign, and artifact selection

| Field | Required value |
|---|---|
| `dataset_id`, `canonical_url`, `doi_or_handle`, `dataset_version`, `publisher` | Public record identity |
| `license_id`, `license_url`, `license_text_sha256` | Artifact license, not the paper license |
| `training_right`, `derived_artifact_right`, `model_distribution_right`, `commercial_use_right`, `attribution_obligations` | `allowed`, `forbidden`, or `unresolved`, each with evidence record |
| `country`, `site_id`, `campaign_id`, `sensor_id`, `acquisition_start`, `acquisition_end`, `season` | Stable grouping identity |
| `epsg`, `vertical_datum`, `bounds`, `coordinate_transform_id` | Metric spatial identity; local-only coordinates are explicit |
| `instrument`, `beam_or_gsd`, `view_groups`, `registration_method`, `control_method` | Direct acquisition facts, with unknowns explicit |
| `artifact_selection[]` | Ordered `{file_id,url,byte_size,publisher_checksum,media_type,role,site_id,campaign_id,required_before_next}` records |
| `manifest_sha256`, `expected_total_network_bytes` | Exact selected manifest, not whole-release marketing size |
| `condition_pair_artifacts[]` | Closest DTM, orthophoto, soil, surficial geology, bedrock, hydrology, land use, canopy and management IDs/dates |
| `representation_scope` | `heightfield_valid`, `mixed_with_mask`, or `nonheightfield_only` |

### 4.4 Surface contract and conversion

| Field | Required value |
|---|---|
| `surface_contract_id`, `surface_definition` | The physical surface whose height is estimated |
| `included_semantics[]` | Explicit subset of mineral soil, stable moss/peat, embedded clasts, exposed roots, or other named classes |
| `excluded_semantics[]` | Explicit classes, including water, snow, moving vegetation, objects, and any regime-specific forbidden material |
| `ambiguous_semantics_policy` | Must be `invalidate`; never force to nearest included class |
| `stable_organic_definition` | Dated persistence and attachment rule or `not_applicable` |
| `heightfield_validity_rule` | Slope/occlusion/overhang and multi-valued-surface rule |
| `label_source_ids`, `human_audit_plan`, `double_audit_fraction` | Label evidence and independent review; double-audit fraction must be at least `0.20` |
| `conversion_code_sha256`, `config_sha256`, `environment_sha256` | Frozen converter identity |
| `input_channel_order`, `coordinate_precision`, `internal_float_type` | Exact numerical contract; metric conversion uses float64 |
| `point_acceptance_rule`, `view_group_rule`, `surface_estimator`, `interpolation_rule` | Executable method, parameters, and tie behavior |
| `output_grid`, `sample_convention`, `output_fields[]` | Height plus masks/uncertainties listed in Section 6 |
| `max_conversion_revisions` | Default `3`, counting the baseline; exceeding it ends this protocol revision |

### 4.5 Audit sampling and inference

| Field | Required value |
|---|---|
| `context_size_m` | `8.0` |
| `microplot_size_m` | `2.0` |
| `initial_contexts_per_site`, `context_increment`, `maximum_contexts_per_site` | `6`, `6`, `24` |
| `microplots_per_context` | `3` |
| `ordinary_fraction`, `event_fraction` | Must sum to `1`; default `0.5/0.5` when an event morphology is claimed, otherwise `1/0` |
| `sampling_seed` | BLAKE2b-256 derived from protocol ID before conversion |
| `stratification_fields[]` | Condition fields available before target conversion; never target roughness |
| `minimum_context_separation_m` | At least `8.0`; contexts may not overlap |
| `ci_method` | Nested block bootstrap, site/campaign outer unit and context inner unit, 10,000 deterministic replicates |
| `confidence_level`, `ratio_ci_halfwidth_stop` | One-sided `0.95`; maximum acceptable half-width `0.15` |
| `split_manifest`, `leakage_audit_id` | Frozen grouping and overlap audit |

Event-selected contexts estimate event behavior; they never estimate prevalence.
Area-weighted ordinary contexts estimate ordinary coverage. If the available plot
cannot supply the registered counts without overlap or boundary contamination, it
can remain an exemplar but cannot satisfy the target gate.

### 4.6 Support, error, transfer, OOD, and release

| Field | Required value |
|---|---|
| `independent_error_source_ids[]` | Checkpoints or held-out acquisitions, including how independence from registration/conversion is maintained |
| `horizontal_error_model`, `vertical_error_model`, `registration_error_model`, `semantic_error_model` | Distribution and propagation rules |
| `controlled_recovery_design` | Frozen feature/scale recovery experiment |
| `support_threshold_m` | `B1=0.05`, `B2=0.03` |
| `minimum_transfer_gain` | `0.707` throughout the claimed band |
| `maximum_error_signal_ratio` | `0.50` for one-sided CI gate |
| `physical_analogue_dossier_id`, `physical_analogue_sha256` | Section 8 record |
| `development_site_ids`, `blind_site_ids`, `heldout_campaign_ids`, `heldout_sensor_ids` | Whole groups only |
| `ood_strata[]`, `eligibility_ast_sha256`, `unknown_policy` | `unknown_policy` must be `abstain` |
| `estonia_sentinel_manifest_sha256`, `taevaskoda_excluded` | Exact sentinel set; latter must be `true` |
| `transfer_ceiling`, `pilot_extent`, `pilot_expiry`, `user_approval_id` | `research_only`, `neighbor_analogue_only`, or `estonia_pilot`; never national from this protocol alone |
| `stopping_rules[]`, `abstention_codes[]` | Exact rules from Sections 11 and 12 |

## 5. Raw-Candidate Retention Gate

A catalogued source becomes `retained_raw_candidate` only if all statements below
are true:

1. The exact artifact is downloadable at zero purchase cost, or a written no-fee
   release has been received. Viewers, paper figures, and request promises do not pass.
2. Artifact-specific rights permit the registered use. `unresolved` model or
   derivative rights restrict the artifact to legal review/research and block the
   target path.
3. Site, campaign, sensor, acquisition date, CRS/local frame, file identity, and
   processing history are known enough to prevent split leakage and false pairing.
4. Original observations are present. A DSM/mesh-only release may remain an
   exemplar lead but cannot become a production target under the canonical raw-data
   requirement.
5. Acquisition geometry has a physically possible path to the registered support
   threshold. Nominal total density alone is not evidence; an instrument footprint,
   already-published error lower bound, or spacing worse than the band limit rejects
   that band before a bulk download.
6. At least one heightfield-valid area exists for the named regime. Vertical walls,
   caves, undercuts, water, and vegetation-only observations are masked, not projected.
7. The selected probe tranche fits every budget ceiling with retry allowance.
8. The physical-regime hypothesis is specific enough to build the dossier in
   Section 8. Geographic closeness alone is not a hypothesis.

The retained record stores raw files unchanged and records SHA-256 after download.
Any publisher-checksum mismatch quarantines the file. Thinning, classification,
normalization, gridding, or coordinate stripping creates a derived artifact and
never overwrites the raw observation.

## 6. Surface-Semantics Conversion Gate

The converter estimates a named physical surface; it does not manufacture missing
detail. Lowest-point filtering, `treeid=0`, LAS class 2, a vendor `ground` label,
or an SfM DSM is only input evidence, never the final semantic decision.

Every converted candidate writes these co-registered fields:

```text
height_m_f64
valid_observed
heightfield_valid
semantic_class
semantic_confidence
view_group_count
effective_view_count
nearest_support_m
effective_footprint_m
interpolation_distance_m
occluded
interpolated
water_or_dynamic
human_invalid
sigma_xy_m
sigma_z_m
source_point_ids_or_reconstructable_index
```

Gate procedure:

1. Freeze the surface contract before labeling. Regime examples include mineral
   ground, stable living bog surface, or mineral-ground-plus-embedded-clasts; these
   are different targets and may not be pooled.
2. Label the frozen nested microplots from raw views and paired photographs without
   looking at any downstream synthesis result. Ambiguous cells are invalid.
3. Run the frozen converter. A second review independently audits at least 20% of
   microplots and all classes whose inclusion could move height.
4. Calculate forbidden-class false acceptance by context block. Its one-sided 95%
   upper bound must be at most `0.02` of accepted area, and the upper bound on
   resulting band-height contamination must be at most `0.25` of the target's
   registered total error budget. Otherwise the affected band fails.
5. Preserve roots, embedded stones, pit/mound forms, litter, deadwood, moss, peat,
   crops, and low vegetation according to the frozen surface contract. Do not erase
   them merely because a generic ground classifier does.
6. Dilate invalid masks through the complete conversion and `F4` support. No
   interpolation across invalid semantics, water, occlusion, plot boundaries, or
   non-heightfield surfaces enters a target loss.

Passing this gate yields `converted_surface_candidate`, not target truth.

## 7. Effective Support, Error, And Transfer Gate

### 7.1 Effective support

For every accepted cell compute a conservative effective support radius:

```text
r_eff = max(
  horizontal distance to accepted observations,
  projected beam/photogrammetric footprint,
  surface-estimator kernel radius,
  interpolation distance
)
```

`B1` requires the one-sided 95% upper confidence bound of context-level `p95(r_eff)`
to be at most `0.05 m`; `B2` requires at most `0.03 m`. Support must come from the
registered physical surface. Returns on foliage do not support hidden soil. Every
band metric uses masks eroded by the exact `F4` footprint.

At least two independent view/acquisition groups must contribute to each audit
context, unless an independent calibrated measurement directly establishes the
surface error and transfer. The preregistered `view_group_rule` must prevent scan
lines or photographs from one pose cluster being counted as independent views.

### 7.2 Error

Report, separately:

- local measurement repeatability from held-out views/acquisitions;
- checkpoint horizontal and vertical error not used in registration;
- registration error to paired condition fields and the 1 m parent;
- slope-propagated height error from horizontal uncertainty;
- conversion error against held-out accepted observations;
- semantic-choice error from double-audited ambiguous surfaces;
- interpolation error, which is never zero merely because interpolation is smooth.

If independence between terms is demonstrated, combine variances. Otherwise sum
their one-sided 95% bounds conservatively. A receiver estimate, instrument
specification, registration residual, or repeat-view residual may describe one term
but cannot silently stand in for all terms. Missing independent horizontal or
vertical error blocks `qualified_target`; repeatability-only evidence can support an
exemplar.

### 7.3 Effective transfer

Create disjoint acquisition groups before conversion. Build one all-supported
reference and at least two reconstructions from held-out view groups. Use the
actual support geometry to forward-sample the reference, reconvert it, and measure
amplitude recovery with the canonical band operators. This measures the converter
and observation-support transfer, not an assumed ideal raster. Independently bound
the acquisition response from calibration, beam/footprint/incidence evidence, and
check observations. Controlled reconversion cannot reveal detail that the sensor
systematically attenuated in every view; an unbounded acquisition response blocks
target status even when reconversion is repeatable.

For each band and each context, report transfer gain versus wavelength, phase bias,
cross-view coherence, signal RMS, and total error RMS. A band passes only when:

```text
LCB95(transfer_gain(lambda)) >= 0.707 for every registered wavelength bin
UCB95(total_error_RMS / recoverable_signal_RMS) <= 0.50
CI_halfwidth(error_signal_ratio) <= 0.15
```

`recoverable_signal_RMS` is estimated from independent-view cross-signal after
registered error correction, never from the smoothed converted raster alone. It is
not called ground truth: correlated systematic error must be included through the
independent acquisition/checkpoint bound. Negative or unidentifiable corrected
signal is `target_evidence_insufficient`. High measurement error, smoothing, or
missing support never proves that a stratum is physically smooth.

Start with six contexts per site and add six only if the confidence-width stopping
condition is not met. Stop at 24. Failure to obtain a decisive interval by 24 is
inconclusive and abstains; it is not permission to pool patches as independent sites.

## 8. Physical-Analogue Dossier

Every cross-border claim validates one `physical-analogue-v1` dossier. It contains:

```text
dossier_id, source_site_ids, destination_regime_id, claimed_bands
source_evidence_ids, estonia_evidence_ids, evidence_dates
factor_rows[]
morphology_rows[]
source_observation_comparison
unmatched_factors[], unknown_factors[], ood_rules[]
allowed_transfer, prohibited_transfer, reviewer, review_date, sha256
```

Each `factor_rows[]` entry contains `factor_id`, physical relevance, source value
and uncertainty, Estonia destination range and uncertainty, evidence IDs, comparison
operator, preregistered tolerance, status (`match`, `mismatch`, `unknown`), causal
consequence, and conditioning field available to the cook. Mandatory factors are:

- bedrock/lithology and surficial deposit;
- soil profile, texture, clast distribution, organic depth, and moisture state;
- climate, snow, freeze-thaw, and relevant event history;
- hydrology, water table/inundation, drainage, and terrain position;
- vegetation, root/organic surface, canopy and observation-season state;
- land use, management, age since disturbance, and machinery/process state;
- active forming processes and their spatial/temporal scales;
- morphology scale, sign, anisotropy, connectivity, event density, and tails;
- acquisition modality, footprint/incidence, visibility, and surface semantics;
- overlap with the Estonia condition fields actually available at inference.

`morphology_rows` compare qualified real observations and cannot repair a failed
causal factor. Every causal factor marked mandatory must be `match`; `unknown` and
`mismatch` become OOD unless the claim is narrowed to an independently supported
factor stratum. A global similarity score is forbidden because it can hide a fatal
substrate, hydrology, process, or surface-semantics mismatch.

Examples from the hunts remain bounded: Stordalen palsa/thermokarst cannot set
Estonia bog distributions; Dutchman's Cap and Biała Góra cliffs cannot supervise
heightfield-invalid faces; Baltic proximity cannot turn a moraine cliff into
Burtnieki sandstone; Finnish forest-floor observations require soil, organic layer,
hydrology, disturbance, and vegetation matching rather than a shared "boreal" label.

## 9. Site, Campaign, And Split Contract

- `site_id` is a contiguous physical location sharing geomorphic and management
  history. Patches, plots, tiles, or epochs from one site do not become independent sites.
- `campaign_id` is one acquisition operation with common sensor calibration,
  season, processing and control. Repeat epochs can be campaigns but remain one site.
- All overlapping raw observations, derived surfaces, augmentations, nearby plots,
  and repeat campaigns inherit the strictest split of their site.
- Duplicate geometry is detected using bounds, timestamps, point fingerprints, and
  nearest-surface hashes before assignment.
- Taevaskoda is excluded from training, converter tuning, hyperparameter selection,
  synthesis candidate selection, and morphology threshold setting.

A release candidate requires at least three geographically independent qualified
sites across at least two campaigns. The development pool contains two sites; one
whole site is untouched blind. Sensor/campaign holdout is required where available.

The smallest plausible forest split is preregistered, not presumed to pass:

| Group | Intended role | Constraint |
|---|---|---|
| Hovi Hyytiälä | development/train | Select whole plots from one physical stratum; no Järvselja use |
| Evo 2024 | development and held-out sensor/campaign | Match the same physical stratum before download |
| Hovi Järvselja | untouched Estonia blind site | All 13 plots remain outside converter and synthesis tuning; target audit may label them, but no method/config changes follow blind morphology inspection |

If those are not three physically analogous site groups, or the campaign identities
do not meet the two-campaign rule, the forest path remains research-only. FORWARD's
two Swedish sites, Biała Góra's repeated single site, ForestSemantic-MS's single
Espoonlahti site, Stordalen's single mire, and any one Lithuania/Latvia campaign do
not independently satisfy the release minimum.

## 10. OOD, Estonia Sentinels, And Pilot-Only Transfer

### 10.1 OOD

Every claimed condition stratum has an exact `laas-condition-ast/1` eligibility
predicate. A cell is eligible only if every mandatory field is present above its
confidence threshold, every categorical factor is in the dossier's accepted set,
every continuous factor is inside a preregistered physically supported interval,
and no water, protected, structural, semantic, or representation mask forbids it.

Do not interpolate eligibility through gaps in a multidimensional envelope. Each
released stratum must be represented at both development sites. Missing values,
condition combinations observed at only one development site, and any unmatched
analogue factor evaluate to `unknown` and abstain.

### 10.2 Estonia sentinels

Freeze a sentinel manifest before conversion results are viewed. Each row records
`sentinel_id`, EPSG:3301 bounds, regime/stratum, purpose, condition hashes, expected
eligibility, expected band action, protected structures/layers, and review method.
For each proposed release stratum enumerate:

- 16 deterministic Estonia interior condition cells from at least four separated
  geographic units, used only to test eligibility and condition coverage;
- 16 cells adjacent to one eligibility boundary;
- 16 negative cells with one known analogue mismatch or missing required condition;
- three 128 m chunks from each group for cook/render review only after a pilot owner
  exists; negative chunks must contain no synthesized residual in the failed band.

Selection is BLAKE2b hash-ranked from the frozen condition fabric and regime ID,
not chosen after seeing generated terrain. It is not evidence of morphology quality.

Suur Taevaskoda at game coordinates `x=311123.082`, `z=190723.435` and exact ALS
tile `444679` is an additional fixed Estonia structural/visual sentinel. Its public
ALS epochs may support source repair, but not 6.25 cm morphology truth. The
`reference/suur-taevaskoda1.jpg` image remains visual composition evidence only.

### 10.3 Allowed pilot transfer

A foreign-qualified target may produce `neighbor_analogue_only` pilot geometry only
when all Section 9.3 requirements in the canonical spec pass: two independent
neighboring development sites, a held-out campaign, a qualifying physical dossier,
condition-stratified OOD bounds, Estonia sentinels, site-level blind evidence,
bounded extent, and explicit user approval.

The pilot:

- emits only in the exact passed band and stratum;
- is clipped to the intersection of the approved extent, eligibility predicate,
  target-evidenced support envelope, and representation-safe mask;
- has an expiry tied to dataset, condition, converter, target, and owner hashes;
- does not become a national fallback and cannot fill an unsupported neighbor stratum;
- does not authorize a synthesis owner merely because target data qualified;
- is recooked, re-reviewed, or removed when any bound identity changes.

An Estonia target such as a qualified Järvselja surface strengthens the blind test
but does not prove Estonia-wide prevalence or transfer. National release requires a
separate `estonia_national` evidence decision.

## 11. Abstention

Use these exact codes, per band where applicable:

```text
license_unresolved
raw_unavailable
raw_checksum_failed
provenance_unresolved
site_identity_unresolved
campaign_identity_unresolved
surface_semantics_unresolved
heightfield_invalid
water_or_dynamic_surface
support_insufficient_B1
support_insufficient_B2
error_unbounded
error_exceeds_signal_B1
error_exceeds_signal_B2
transfer_unresolved_B1
transfer_unresolved_B2
physical_analogue_unknown
physical_analogue_mismatch
site_count_insufficient
campaign_holdout_missing
split_leakage
condition_missing
condition_ood
sentinel_failed
budget_exhausted
user_not_approved
```

Abstention action is exact: emit no morphology residual in the failed band and use
the accepted corrected structural authority reconstructed through the canonical
hierarchy. Do not substitute a nearest regime, generic noise, a coarser target
upsampled to fine spacing, or a visually similar foreign surface.

## 12. Stopping Rules

1. **Metadata stop:** unresolved artifact license, file identity, site/campaign
   identity, or raw availability ends the target path before geometry download.
2. **Physical impossibility stop:** if documented footprint, spacing, checkpoint
   error, or acquisition geometry has a lower bound worse than a band's threshold,
   reject that band without downloading more plots from the campaign.
3. **Probe stop:** begin with six frozen contexts at one development site. If no
   valid heightfield surface exists, forbidden semantics cannot be separated, or
   the band has less signal than its conservative error bound, do not expand that
   candidate family for the same claim.
4. **Sequential precision stop:** add contexts in groups of six only while a gate
   could pass and CI half-width exceeds `0.15`. Stop successfully once all bounds
   pass; stop inconclusive at 24 contexts per site.
5. **Revision stop:** allow at most three frozen converter revisions. A revision
   reruns the same audit contexts. After the third failure, reject the candidate for
   this protocol revision; new primary evidence is required to reopen it.
6. **Independence stop:** if three independent sites across two campaigns cannot be
   assembled, retain research/exemplar status and stop before synthesis owner selection.
7. **Analogue stop:** any mandatory mismatch or unknown blocks the affected Estonia
   stratum; do not seek a statistical score that averages it away.
8. **Blind stop:** a blind site, campaign, OOD, or sentinel failure ends pilot
   promotion. Do not tune on the failure and reuse the same holdout.
9. **Budget stop:** reaching 90% of any byte/time ceiling stops before the next
   tranche. Existing verified remote raw may be evicted under the retention policy;
   paid capacity may not be proposed.
10. **Request stop:** send one precise no-fee author request and one reminder after
    21 days. No release after 42 days becomes `request_only`; it cannot block the
    open-data path.

## 13. Smallest Selective Download Order

This is a gate sequence, not a synthesis-method preference. Never fetch a later
tranche until the named earlier gate passes, and never fetch a regime branch that is
not the active qualification claim.

| Order | Selection | Maximum initial action | Gate before next action |
|---:|---|---|---|
| 0 | All candidate records | Metadata, licenses, official file manifests and publisher checksums only | Preregistration and raw-retention gate |
| 1 | ForestSemantic-MS, DOI `10.5281/zenodo.17172162` | All six LAZ files, 131.1 MB total, retaining the published four-train/two-test split | Verify fields/labels and use only to exercise semantic audit; no target claim |
| 2 | Hovi Hyytiälä | One mature 16-scan plot: metadata/photos/cover, thinned LAZ, transformations, then its individual full scans one file at a time | Six-context semantics/support/error feasibility; stop if raw view structure or B1/B2 is impossible |
| 3 | Evo 2024, DOI `10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77` | One 32 x 32 m plot matched to the frozen Hyytiälä physical stratum, not the 39.3 GiB release | Held-out sensor/campaign conversion gate |
| 4 | Hovi Järvselja | One preregistered blind plot's thinned LAZ and matching raw scans only after converter, thresholds, analogue dossier, and splits are frozen | Untouched Estonia target qualification; no retuning after morphology inspection |
| 5 | Hovi/Evo forest expansion | Two additional non-overlapping plots per development site and only the preregistered Järvselja blind plots needed to reach CI stopping | Three-site/two-campaign gate, OOD and sentinels |
| 6a | FORWARD, DOI `10.71540/89rs-s553` | Only if till/disturbance is active: one matched ground/UAV tranche from each of Marrviken and Björsjö, starting with the smallest manifest-complete files | Raw ground support, date alignment, two-site status; remains research without a third site |
| 6b | Biała Góra, DOI `10.18150/BHH1RC` | Only if coastal cliff/moraine is active: the smallest complete epoch archive (`351,166,481` bytes per audited manifest), not all 27.402 GB | Classification, checkpoint/error, support, and heightfield mask; single site cannot release |
| 6c | Stordalen mire | Only if peat is active: 234 MiB DSM plus RGB companion | DSM/raw-support and palsa-transfer audit; normally exemplar only |
| 7 | Evo 2021, remaining Hovi, remaining Evo 2024, remaining FORWARD/Biała | No expansion unless a prior tranche passed and additional independent strata are preregistered | Incremental information/power calculation and budget gate |

Run the no-fee Kohv/Shlykova request for Suur/Väike Taevaskoda, Härma and Selisoo,
and requests for Pilkosios, Dutchman's Cap, Preila and Kaigu in parallel because
they consume no geometry storage. They remain `request_only` until files, checksums,
control/support evidence, and usable rights arrive. Exact-site Taevaskoda ALS epochs
are a separate structural-repair download and must not be counted as target-scale
morphology data.

Before each row, the official manifest is frozen into `artifact_selection[]` with
exact file IDs and byte counts. A selector such as "one matched plot" must resolve
to those IDs before download; it is not permission for an operator to choose the
most attractive surface after viewing geometry.

## 14. Immediate Taevaskoda ALS Retention Record

The exact-site ALS is an immediate application of this protocol's retention gate,
not a target-qualification exception. Create one candidate record with:

```text
dataset_id = maaamet_als_tile_444679
site_id = taevaskoda_tile_444679
role = calibration_only
country = EE
epsg = 3301
tile_bounds = [679000, 6444000, 680000, 6445000]
allowed_claims = [source_repair, water_vegetation_strip_diagnostics, coarse_context]
forbidden_claims = [B1_target, B2_target, morphology_owner, effectively_smooth]
```

Freeze these eight campaign/file selections from the official service before
retention, resolving the live canonical file ID, URL, byte size, and publisher
metadata for every row:

| Order | Campaign type | Year | Tile |
|---:|---|---:|---|
| 1 | normal/spring | 2023 | `444679` |
| 2 | normal/spring | 2019 | `444679` |
| 3 | normal/spring | 2015 | `444679` |
| 4 | normal/spring | 2011 | `444679` |
| 5 | low flight | 2016 | `444679` |
| 6 | forestry/summer | 2024 | `444679` |
| 7 | forestry/summer | 2021 | `444679` |
| 8 | forestry/summer | 2017 | `444679` |

The already audited 2023 selection must resolve to
`444679_2023_tava.laz`, `68,066,160` bytes, SHA-256
`9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7`.
A mismatch stops the sequence and requires a new source-version record; it must not
be silently accepted as the same artifact.

For all eight files, preserve raw LAZ, acquisition/campaign identity, class table,
return order, scan geometry, overlap flags, time/color attributes, bounds, point
counts, header version, local SHA-256, exact download UTC, and the Maa- ja
Ruumiamet license/attribution record. Check each file and budget before fetching the
next. A failure in one epoch does not authorize substitution with a DTM raster.

These epochs are exempt from the target support/transfer test only because they are
permanently `calibration_only`. Any later attempt to use them as a morphology target
creates a new protocol and is expected to fail the target-scale support gate. Their
classes and repeated returns remain fallible evidence: source repair must retain
water, vegetation, overlap, occlusion, and uncertainty masks.

## 15. Required Outputs

Each completed protocol run writes immutable, hash-linked artifacts:

- preregistration JSON and selected-file manifest;
- artifact/license/attribution ledger and raw hashes;
- raw-to-surface conversion recipe and environment identity;
- semantic labels, double-audit differences, and complete output fields from Section 6;
- context/microplot and split manifests;
- per-band support, error, transfer, signal, confidence and invalid-area reports;
- physical-analogue dossier, OOD strata, and Estonia sentinel manifest;
- one machine result with state, passed bands, transfer ceiling, abstention codes,
  prohibited claims, and exact evidence hashes.

Only `qualified_target_B1/B2` records may enter the later target-specific owner
preregistration. All synthesis candidates then receive the same frozen surfaces,
masks, splits, observation operators, and measurement-error propagation. Target
qualification must not be reopened merely because a favored synthesis method loses.

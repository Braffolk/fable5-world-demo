# Cliff/escarpment: measured-anchor + transferred-organization hybrid — design brief (orchestrator, 2026-07-17)

Status: design for a NEW cliff-family attempt (rank-1 regime). Requires its own
machine preregistration before generation. Non-normative until frozen.
Scope: **unconsolidated sand/till escarpments and banks ONLY** (Dev A ETAK
`1826743` glaciolacustrine-sand escarpment; Dev C ETAK `9688719` bank slope).
Sandstone/carbonate outcrops remain separate unsupported owners; the Taevaskoda
sandstone wall is non-heightfield and the Taevaskoda parent `(-1,607,372)`
sentinel stays unopened. This attempt gets ONE run + ONE consolidated correction.

## Park-condition compliance (why this attempt is legal)

The park demands "materially new long-context, material-matched morphology
evidence or a different structural representation." Both now exist:
1. **Estonia measured macro anchors**: accepted development-only ALS/TGV float
   `a5d101c1cca4...` (22,194 qualified 2019 class-2 points at Dev A, withheld
   block-median error improved on both mapped sides). Its packed preview was
   rejected as an owner for SPARSENESS (crest/shoulder macro "terminates
   abruptly, long boring gaps; face too soft") — not wrongness. The anchors are
   real but gappy.
2. **Source-domain-proven connected representation**: final Biała screen
   `9d27b52da864...` demonstrated, in the source domain, connected
   crest/face/toe break, asymmetric bench, slump-headwall sockets/aprons,
   branching chute/gully, ordinary-ground transition, with a bounded band-limited
   `0.5-1.25 m` fine component and WITHOUT the recorded failure modes
   (line-normal ribs, finite-line caps, slabs, roughened ramp).

Neither alone is an owner. The new mechanism combines them: Estonia's measured
anchors set WHERE and HOW MUCH; Biała's proven representation sets HOW FORMS
CONNECT. No prior family did this — the exhausted families were pure synthesis
(feature graph, strips, harmonic ribbon), pure measurement (ALS/TGV), or pure
source transfer (exemplar/dictionary/learned/diffusion). This is
measured-anchor-constrained organization transfer: a different structural
representation, satisfying the park's disjunction.

## Mechanism sketch

1. Along each mapped ETAK escarpment/bank line (within soil/geology-supported
   condition envelope from bundle `10a5ea53b007...` machinery): extract the
   measured cross-profile sequence from the ALS/TGV field where anchor support
   exists (support mask from the TGV confidence field).
2. Fit a low-dimensional along-line profile-state model (crest offset, face
   height/steepness, bench presence, toe extent) to the anchored stations;
   interpolate the state BETWEEN anchors along the line with continuity +
   monotone tapering into mapped line endpoints (no finite-line caps: state
   decays to ordinary slope BEFORE the line ends, per the recorded cap failure).
3. Impose the Biała-derived connective grammar on the interpolated segments:
   asymmetric bench insertion, slump-headwall socket/apron placement at
   concavities, branching chute/gully seeds at flow-convergence crossings
   (from existing D-infinity routing conditions), all as CONstrained deviations
   around the anchored state — organization from Biała, magnitudes bounded by
   the local anchored envelope (p05..p95 of nearby anchored stations; never
   Biała amplitudes directly).
4. Solve one whole-domain surface (existing sparse-solve machinery style:
   float64, hard-mask cut edges, collar to C0) so forms are globally coherent;
   crop last.
5. Add the bounded fine component exactly as the accepted Biała screen did:
   band-limited `0.5-1.25 m` retention conditioned on face/toe type, no raw
   broadband residual.

## Frozen gates (bind concrete numbers in preregistration before generation)

- Exactness: hard/water/protected/collar residuals exactly 0; 1 m closure
  ≤ 1e-9 m outside authorized structural-correction envelope; sealed/sentinel
  untouched.
- Anchor fidelity: withheld ALS block-median error must NOT regress vs the
  accepted ALS/TGV field on either mapped side (the a5d101c1 holdout protocol).
- Connectedness (the reason this attempt exists): fraction of mapped line length
  carrying a coherent crest-face-toe profile ≥ 0.85 (vs the rejected preview's
  sparse islands); no gap in the crest break longer than 15 m where the line is
  mapped and condition-supported.
- Recorded failure modes = automatic rejection: line-normal ribs, finite-line
  endpoint caps, slabs/lobes, mere roughened ramp (assessed on common-light QA
  exactly as prior rejections were).
- Fine band: 0.5-1.25 m component energy within the Biała screen's source-band
  envelope for matching form types; no broadband speckle.
- C0 control + 4-6 numbered QA PNGs (whole-line organization, anchored-vs-
  interpolated state map, junction/socket placements, common-light closeups,
  bands, masks).
- One run + one consolidated correction; any gate fail after that →
  `research_rejected`, park family, record resume condition.

## Tier & delegation

This build is judgment-heavy (profile-state model, grammar constraints, solver
coupling): implementer runs on the SESSION MODEL (Fable, high) per tiering law.
Mechanical stages (raster prep, QA rendering) may be delegated down by the
orchestrator only, not by the implementer.

## Orchestrator checklist

1. [ ] Preregistration frozen (hashes: ALS/TGV artifact, Biała screen artifact,
   condition bundle, ETAK lines, code, seeds, concrete gate numbers).
2. [ ] Implementation + single run at Dev A (Dev C only if A passes mechanics).
3. [ ] Orchestrator PNG gate → (if pass) pack ordinary hierarchy → boot grass=0
   → morning review URL. Sentinel remains unopened regardless.

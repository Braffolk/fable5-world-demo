# Raised-bog v3: self-organized string/pool network — design brief (orchestrator, 2026-07-17)

Status: design for the third materially different `peat.raised_bog` R0 challenger.
Authority: spec §9.10/§9.10.1 lineage; this design requires a new
`bundle-preregistration-v3.json` before any generation. Non-normative until that
preregistration is frozen.

## Why this is materially different (park-condition compliance)

The park condition (TASKLIST rank 3) demands "observed or generated network topology
that can branch, merge, terminate, and couple pool margins to string/flark
complexes." The three rejected/parked families were: Moore whole-form marked
sampling (censored forms), v2 two-field plurigaussian (HuHoLa evidence collapse:
621 lawn cells / 1 hollow / 0 hummocks), and the whole-mire form-scale graph
(single scalar travel coordinate → even contour corrugations, isolated pool
sockets). None *generates* topology; all place or threshold forms.

v3 uses **anisotropic ecohydrological self-organization** (Rietkerk et al. 2004;
Eppinga et al. 2008/2009 maze/string patterning in peatlands): a
reaction-advection-diffusion system on the mire's water-routing field whose
steady-state banding is emergent. Published model behavior — strings transverse
to flow that branch, merge, and terminate; maze patterns where flow is weak;
ponding (flarks/pools) in inter-string troughs — is precisely the demanded
topology class, generated rather than placed. This is a process/event family the
research verdict table already lists as transferable ("causal shape, topology,
orientation in supported processes").

## Mechanism sketch

1. Domain = connected mire unit + ≥64 m halo (whole-domain solve, crop last —
   v2 rule retained).
2. From the corrected 1 m parent: dome gradient field, D∞/priority-flood water
   routing, wetness proxy (existing condition machinery).
3. Solve a two-variable activator–inhibitor system (peat growth vs water
   ponding/nutrient advection per Eppinga) with advection along the routing
   field; anisotropy from |∇dome|; deterministic world-PRF (BLAKE2b
   `laas-micro-prf1`) heterogeneity as nucleation noise (keyed by world coords —
   NOT storage chunks; periodic boundaries forbidden).
4. Steady state field → typed forms: string/hummock ridge where activator high;
   hollow/flark in troughs; POOL where trough cells pond above a depth criterion
   under local routing (pools are therefore *automatically* margin-coupled to
   their bounding strings); lawn elsewhere.
5. Relief carving: typed form profiles with amplitudes calibrated ONLY from the
   accepted Valgesoo low-relief carrier stats plus Moore plot-scale descriptive
   distributions (R0 descriptive use; no pixel supervision claim). Exact 1 m
   parent means preserved; hard/water/unknown masks exact; sealed mire
   `etak-component-0004069025` stays abstention-only and pixel-unopened.

## Primary-source grounding (MANDATORY first implementation stage)

Fetch and read (open-access versions) before coding, record equations +
parameter regimes + demonstrated pattern scales in the preregistration:
- Rietkerk et al. 2004, "A putative mechanism for bog patterning" (Am Nat) /
  self-organized patchiness (Science 2004).
- Eppinga et al. 2008/2009 (maze patterns; nutrient accumulation in string
  formation; Plant Ecol / Am Nat).
- Optionally Swanson 2007 (string spacing) if freely accessible.
Record which model variant is used and why; parameter transfer from the papers
is allowed only for dimensionless regime selection (pattern wavelength ratios,
anisotropy regime), never as literal Estonian amplitudes — amplitudes come from
Valgesoo/Moore stats only.

## Prerequisite B0: corrected-base extension

Dev mire `etak-component-0004069028` core `[536384,6452416,536512,6452544]` lies
in LOD0 `(0,81,89)`, OUTSIDE current corrected format-1 coverage
(cx 148..155, cz 90..97) — recorded blocker in the `8d450c7fe056...` condition
snapshot (zero C1 authority). Fix: run the standard evidence-bounded
corrected-structural-base transaction (recipe kind `corrected-structural-base-v1`,
§14.6 order, no `latest`) over `(0,81,89)` + required halo. A near-zero-correction
outcome is valid and expected (bog interior; water/pool surface conditioning may
fire). Only after that transaction verifies may the v3 condition snapshot be
rebuilt and generation run.

## Frozen gates (to bind in preregistration v3, before generation)

- Exactness: 1 m mean error ≤ 1e-12 m; hard/water/unknown/sealed residual
  exactly 0; halo/crop-last identity (whole-domain solve, per-chunk anything
  forbidden).
- Anti-corrugation (v3's reason to exist, measured on the string skeleton):
  string spacing CV ≥ 0.35 (v3 fails if strings are evenly spaced like the
  form-graph's corrugations); orientation coherence localized, not mire-wide.
- Topology: ≥ N_branch junctions and ≥ N_term terminations per km² of patterned
  area (N set after the primary-source read, from published pattern imagery
  statistics, recorded before generation); both branch AND merge junction types
  present.
- Pool coupling: ≥ 70% of pool-perimeter cells adjacent to string/hummock-typed
  cells; pools must be inter-string troughs, not isolated sockets.
- Amplitude envelopes: typed-form relief within Valgesoo carrier + Moore
  descriptive p01..p99 envelopes; B1-band residual RMS within carrier envelope.
- Control: C0 corrected-only run packaged alongside; common-light before/after
  QA; 4-6 numbered PNGs (whole-mire organization, skeleton+junction overlay,
  pool-coupling map, scale bands, closeups, masks/closure).
- Rejection terminals: corrugation gate fail, topology gate fail, pool-socket
  recurrence, any exactness miss → `research_rejected` for this candidate; NO
  parameter tuning after viewing QA (one consolidated correction cycle allowed
  for harness/plumbing defects only, per efficiency law).

## Implementation constraints

- New isolated package `asset-gen/src/assetgen/terrain/microtopography/
  peat_bog_network/`; do not modify v1/v2/form-graph artifacts or code paths.
- Deterministic float64 solve; fixed iteration budget + convergence criterion
  recorded; no wall-clock/random/order dependence (workers must not change bytes).
- Implementer may NOT: retune the rejected families, relax masks, use FBM or
  primitive stamps, open the sealed mire, or claim production/transfer/target
  truth. Output = `SpecialistOutput` research_only; at most an immutable
  non-`latest` preview after all gates pass.
- Tier: mechanism implementation from this brief = opus xhigh; any deviation
  from the brief, gate design change, or disappointing-result diagnosis =
  escalate back to orchestrator (Fable) — do not self-vary the method.

## Orchestrator checklist

1. [ ] B0 corrected-base transaction for (0,81,89) verified.
2. [ ] Primary sources fetched/read; equations + regime recorded.
3. [ ] `bundle-preregistration-v3.json` frozen (hashes: sources, carrier,
   Moore stats, code, seeds, gates above with concrete numbers).
4. [ ] Generation + verify + QA PNGs.
5. [ ] Orchestrator PNG gate → pack via ordinary hierarchy → boot grass=0 →
   morning review URL.
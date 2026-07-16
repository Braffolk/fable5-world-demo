# Agriculture v2: operation-state roughness with real scale breaks — design brief (orchestrator, 2026-07-17)

Status: queued behind forest-2/dune/bog/cliff; run only if overnight time permits
(user re-tabled agriculture 2026-07-17). Needs its own machine preregistration
(conformance-audit requirement: a dedicated agriculture bundle; may reuse the
height transport; may NOT borrow forest evidence roles or silently promote R0→R1).

## Why v1 failed (binding lessons, artifact `de1f0d4adb49...`)

1. Fixed 0.30 m rows owned 94.998% of samples → visible corduroy.
2. Clods were repeated analytic lattice stamps.
3. Per-tile priority-flood hydrology broke partition invariance (11.3 mm edge).
4. Parcel/mask/soil claims not bound to reconstructable source artifacts.
5. Non-normative B1/B2 QA operators.

## v2 mechanism (each failure inverted)

1. **No fixed row lattice.** Row fields are generated per parcel as a smoothly
   varying direction field (from parcel principal axis; orthophoto row direction
   where a dated masked image supports it) with VARIABLE spacing drawn from the
   Marzahn 2012 dual-scale decomposition (seedbed-row scale vs wheel-track scale
   are distinct, operation-caused, direction-locked — descriptors explicitly
   transferable per the research verdict). Row amplitude/spacing get defects:
   headland turn bands at parcel ends, row curvature following boundary
   geometry, local dropouts. Rows own a bounded minority of variance, not 95%.
2. **Clods = marked point process** (world-PRF keyed), size/density conditioned
   on operation state + Marzahn roughness split; no analytic stamp reuse — each
   clod realized from a small parameterized form family with continuous marks.
3. **Whole-domain solve, crop last** (v1's per-tile hydrology is structurally
   excluded by the standing crop-last law); wheel tracks route as compaction
   lines along the direction field, continuous across chunk boundaries.
4. **Evidence binding:** ETAK arable parcels + Mullastikukaart soil texture
   (parsed contract fields) + operation state. Operation state is honest: where
   the state (ploughed vs seedbed vs rolled) is unknown, use an explicit
   `operation_unknown` mixture policy bound in the preregistration — never a
   silent default to one look. Every scalar claim hash-bound to its source
   artifact (v1 lesson 4).
5. QA uses the normative F4/R4 band operators only.

## Gates (freeze concrete numbers in preregistration)

- Anti-corduroy: row-band variance share ≤ 0.5 of total added variance per
  parcel; spacing CV ≥ 0.25; direction field must vary across parcels; no
  parcel-to-parcel phase continuity.
- Scale break: added-relief spectrum must show the Marzahn dual-scale structure
  (distinct row-scale and track-scale peaks), not a single comb.
- Parent deviation per the 07-17 exactness law: report deviation field;
  form-attributed (rows/tracks/clods inside parcels), near-zero taper outside
  parcels; safety exactness unchanged (water, hard masks, protected, ditches
  preserved as typed structure).
- Partition invariance: byte-identical across AOI/worker splits (v1 lesson 3).
- C0 control + 4-6 numbered QA PNGs incl. parcel-boundary transitions and a
  ground-scale closeup; common-light review.
- One run + one consolidated correction; corduroy/lattice recurrence →
  `research_rejected`.

## Tier & site

Implementer: opus xhigh from this brief (mechanism is specified); escalate
deviations to orchestrator. Site: reuse the v1 development tile's parcel
neighborhood on the accepted corrected base (Estonia LOD -2 tile `(2426,1498)`
area) so v1-vs-v2 is directly comparable; expand to a full parcel set within one
LOD -1 parent if mechanics pass.

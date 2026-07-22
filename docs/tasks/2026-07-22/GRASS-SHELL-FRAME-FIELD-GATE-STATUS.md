# Shell-frame field gate status

Date: 2026-07-22
Owner: Codex, single-owner track
Phase: 1 complete, **RED; model and runtime PARKED**
Runtime/shader authorization: **none**

## Inspectable outcome being pursued

One content-addressed two-community report and numbered QA set which says
GREEN or RED for the shell-frame field under the handoff thresholds. A report,
test, or shell fit is supporting evidence; the actual outcome is whether the
fixed-cost model reconstructs the real exterior appearance and winner geometry.

## Limits

- Maximum focused phase-1 time before park/continue decision: 60 minutes.
- Maximum complete attempts: two valid, in-contract measurements. The two
  reconstruction artifacts emitted before the final run are invalidated by a
  discovered E2 transcription defect (the shell smoother stopped while its
  conditioning inequality was still false), so neither may consume the final
  model verdict. One replacement execution is uniquely required; this exception
  is recorded before running it.
- Attempt 2 is permitted only after one upstream premise audit of parameters,
  metric, anchor, lattice, and cost transcription.
- No runtime or shader file may change before GREEN.
- No loop, march, traversal, candidate list, per-species work, or runtime mesh
  is introduced by the prospective model. Offline truth may traverse.

## Bound inputs

- Specification: `GRASS-SHELL-FRAME-FIELD.md`, including all Section 11 errata.
- No-go lemmas: `GRASS-EXACT-REPRESENTATION-THEORY.md` Sections 3-4.
- Baseline harness: `analyze-direct-radiance-field.ts`, Experiment 024.
- Stress community: checked-in Calamagrostis GCRP/v4 SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
- Dense control: deterministic authored `Sphagnum capillifolium` connected
  carpet (profile 5), bound to its checked-in GCRP/v2 asset and fixture mesh
  hashes. Its connected carpet gives the near-total top-down interception
  Section 12 requires; it is a genuine dense ground-cover control.
- Sparse reconstruction case: deterministic authored `Agrostis capillaris`
  fixture (profile 0), now correctly labelled sparse/open after its measured
  8.45% top-down interception. It remains useful for the fixed-97 empirical
  reconstruction gate but cannot stand in for the dense control.

## Frozen interpretation before attempt 1

### Frame coordinates

For slope-disk node `s_i`, use the mean-shell plane `y=Ebar` as the frame
datum and parameterize its ray by **vertical depth** `lambda`, not unit-ray
distance. A node frame is addressed by the periodic phase `u` where its
orthographic ray crosses that plane:

\[
L_i(u,\lambda)=(u_x,Ebar,u_z)+\lambda(s_{i,x},-1,s_{i,z}).
\]

The stored first-hit parameter is split as

\[
\lambda_i(u)=o_i(u)+r_i(u).
\]

For a live ray of slope `s=d_xz/(-d_y)` crossing the same plane at phase `q`,
the exact phase relation to node `i` is

\[
u=q+\lambda(s-s_i).
\]

The smooth shell root solves `lambda=o_i(q+lambda(s-s_i))`. It is evaluated
by one fixed Newton linearization. In this mean-plane XZ parameterization the
orthogonal-frame conditioning product transforms exactly to
`|grad o_i| |s-s_i|`; the shell is smoothed until its conservative support
bound is `|grad o_i| max_cell|s-s_i| <= 0.5`.

This parameterization is load-bearing. Unit-ray depth scales as
`1/sin(elevation)` and would manufacture a divergent `sigma_res` at grazing;
the Section 5.2 slope law's lever is vertical shell residual.

### Cost variants must remain separate

The specification's stated alignment and its stated 6-7-read cost are not the
same data dependency:

1. **No-realignment ablation (6 taps):** per node, read `o_i`, solve the
   smooth shell, then read one full frame record there. This retains the
   first-order `Delta sigma` error and is not the model.
2. **Winner-only alignment (7 taps):** re-read only the max-weight node after
   residual correction. Winner geometry has the full second-order alignment;
   the two lower-weight radiance contributions retain first-order error. This
   is the principled low-end variant.
3. **Faithful residual realignment (9 taps):** per node, read `o_i`, read the
   preliminary residual, shift phase by `Delta v_i r`, then read the aligned
   full record. This is the literal Section 3.3 operation.
4. **Second residual step (12 taps):** repeat the residual shift and record
   read once more; this is E10 and remains an ablation only.

All four are scored. The strict 9-tap path is the model and is the primary
GREEN gate; the inherited 6-7 count was a specification arithmetic error, not
a quality contract. The 7-tap winner-only path is reported as a possible
low-end knob, never substituted silently for the strict model. The 12-tap
variant remains ablation-only and cannot authorize implementation.

### Coupling and filtering

- Premultiplied RGBA is spatially filtered and then blended between the three
  aligned direction nodes.
- Depth, normal, and mark are never blended between owners or view nodes.
- Winner geometry uses the maximum barycentric direction node and a categorical
  spatial event. Its world point is projected onto the live ray before scoring.
- Frames are cover-only. Misses are transparent; no ground depth enters the
  shell or residual statistics.
- Crisp statistics contain foliage ribbons, culms, rhizomes, and panicle axes.
  Spikelet surfaces, hairs/filaments, and anthers are reported separately as
  plume/fuzz and may not inflate the crisp direction law.

### Direction and view domain

- Attempt 1 measures `sigma_res(s)` on pilot slope rings before emitting the
  lattice required by the Section 5.2 radial and azimuth laws.
- The exact Experiment-024 view set is retained: 90, 18, 10, 5, and 1 degrees.
- Eye-height cases span 1.6 m above the low sward to 0.4 m above the tall sward.
- Translation frames use 0, 1, 2.5, and 4.5 mm lateral offsets.
- The finite lattice ends at the 1-degree ring. The quasi-periodic fringe is
  budgeted and reported separately; it is not allowed to improve the finite
  1-degree gate by replacing its geometry with a silhouette strip.
- E5's terrain-tilt margin is charged as one extra ring. The flat-community
  harness cannot validate upward local-ground directions; that transfer remains
  an explicit implementation prerequisite rather than a hidden passing metric.

### Attempt-1 acceptance

Both communities report all metrics, but the handoff treats a large tall-
community lattice as a measured budget rather than an automatic failure.
The original thresholds remain the reconstruction target, but Section 12 now
separates the communities: the dense connected control measures the shell
premise; sparse Agrostis measures reconstruction at the fixed 97-node
Experiment-024 lattice:

- silhouette IoU >= 0.97;
- composited RGB max-channel p95 <= 0.15;
- largest connected wrong-view region < 1% of a frame;
- unforced class-change fraction < 5%;
- crisp winner live-ray world-position p95 <= 0.05 m.

The report additionally checks measured edge-doubling width against
`2 Delta sigma`, shell conditioning, crisp/plume residual curves, emitted
direction count, and uncompressed/compressed byte estimates for each community.

## Terminal log

- Mandated documents and the complete Experiment-024 harness read in order.
- No runtime/shader edit made.
- Corrected sparse/open Agrostis prerequisite audit completed RED for the
  frozen Section-5.2 law. Section 12 accepts that narrow result but rejects its
  use as the dense-control or reconstruction verdict; the missing measurements
  are now active.
- The prerequisite frozen-law blocker is retained at
  `GRASS-SHELL-FRAME-FIELD-RED-BLOCKER.md`; its broad claim was superseded by
  Section 12 and is not the final reconstruction verdict.
- Premise audit during the first execution: a unit-ray-distance residual made
  `sigma_res` diverge toward grazing (Agrostis p95 reached 11.97 m and emitted
  56.7 million directions). The run stopped before metric output. The exact
  slope-disk algebra above shows that the stored parameter must be vertical
  shell displacement; code and frozen equations were corrected together.
- Reconstruction diagnostic: strict-9 initially emitted RED, and a corrected
  crest-vs-in-band scorer confirmed that the broad failure was not confined to
  the crest. Before accepting that verdict, artifact inspection found the 5°
  and 1° shells at conditioning products `1.37–3.06`, violating E2's `<=0.5`
  gate because the offline smoother had an arbitrary 64-pass stop. Those
  reconstruction artifacts are retained as diagnostics but are not admissible
  model verdicts. The final replacement run removes that stop and uses a
  mathematically valid constant-shell fallback if iterative smoothing does not
  reach the bound.

## Final Section-12 outcome

Dense connected control:

`data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/718dd44345aba66f/`

- 100% top-down interception;
- raw crisp residual p95 about 10.6--12.5 mm through 5°, falling to 3.4--4.0 mm
  at 1°;
- every E2 conditioning product at or below 0.5 after 0--210 ordinary smoothing
  passes; no constant fallback used.

Final fixed-97 sparse reconstruction:

`data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/cca44e1d11aa283e/`

- strict-9 IoU 0.61974, required at least 0.97;
- RGB max-channel p95 0.36095, required at most 0.15;
- largest connected wrong region 44.53%, required below 1%;
- crisp winner geometry p95 12.88498 m, required at most 0.05 m;
- unforced class changes 19.66--52.52% by view, required below 5%;
- outer-crest and in-band strata both independently RED;
- 97 directions and estimated 6.18 MB compressed: cost passes, reconstruction
  does not.

The strict correction does not converge toward truth on sparse multimodal
first-hit content. Aggregate no-realignment IoU/RGB are 0.62839/0.35118,
strict-9 is 0.61974/0.36095, and the second-step ablation is
0.58276/0.37124. Residual correction often moves the address onto a different
owner/successor rather than refining the same event sheet. This falsifies the
claimed second-order local bound for the sparse regime; the missing variable is
categorical stratum identity, not another filter or Newton step.

Final designer-facing blocker and objective resume condition:
`GRASS-SHELL-FRAME-FIELD-RECONSTRUCTION-BLOCKER.md`.

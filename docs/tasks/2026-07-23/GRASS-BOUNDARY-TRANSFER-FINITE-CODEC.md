# Finite codec for the guarded box-boundary transfer

**Date:** 2026-07-23
**Scope:** mathematics and cook/runtime format contract only; no shader or
runtime edit
**Status:** **not validated and not an implementation target.**  The first
actual-source gate proved only that the categorical point field was highly
mixed.  Its filtered-codec follow-ups were invalidated by sensor, chart,
Jacobian, packing, train/test and total-depth setup defects.  Therefore
full-domain rate-distortion remains undecided rather than proved RED.  This
format remains an explicit comparison baseline; no independently justified
encoding currently meets both the physical footprint and 250 MiB contracts.

## 1. What changed relative to the rejected codecs

The common-slab carrier maps an exterior pointed ray `(o,d)` to one canonical
boundary point `q` before the transfer is addressed.  The field being coded is
therefore

\[
        T:\partial(P\mathbin\times I)\times S^2\longrightarrow
        \{\mathrm{MISS},\hbox{marked first hit},\hbox{filtered mixed record}\},
\]

a four-dimensional boundary light field (plus a finite chart id).  This is
not the rejected five-dimensional origin-aware light field: the missing
origin coordinate has been removed by a real symmetry and an exact carrier
first-passage operation.  It is also not a fixed-`K` successor/event/moment
codec.  `T(q,d)` already denotes the first triangle in the whole forward
periodic community; there is no live successor relay.  Finally, categorical
owners from different directions are never blended.

The finite-state limitation has not disappeared.  An arbitrary soup can have
arbitrarily high filtered entropy if no source bandwidth, tolerance, guard,
display footprint, or byte cap is fixed.  The selected format consequently
accepts arbitrary soups as cook inputs, but publishes a community only when
its measured physically-filtered transfer fits the frozen byte and error
limits.

## 2. Domain tiling

Each cap/side chart uses a regular four-dimensional lattice over two boundary
coordinates and two direction coordinates.  The finest lattice is divided
into `4 x 4 x 4 x 4` macrobricks, so one brick contains

\[
        L=4^4=256
\]

microcells.  Let `B_0` be the total number of finest bricks over all charts.
Spatial minification of the two boundary coordinates is cooked from the
oracle separately.  It is not made by averaging categorical depth or owner.
For a full two-dimensional spatial pyramid,

\[
        B_\Sigma \simeq {4\over3}B_0.
\]

Angular filtering can be cooked as separately selected levels if required by
the pixel footprint; it must be charged explicitly rather than hidden in the
`4/3` factor.

## 3. Certified brick modes

Every eight-byte brick header declares exactly one of these modes.

1. **MISS.** The entire brick is certified empty to the finite horizon.
2. **Uniform regular.** One surface chart (triangle, periodic copy, plane,
   normal, material and mark) wins throughout the brick with a positive
   incidence and order margin.  The live ray is intersected with that plane;
   sampled depth is not relifted.
3. **Palette regular/mixed.** A topology codeword maps each of the 256
   microcells to a local slot.  A deduplicated palette set maps that slot to a
   coupled payload token.  A token names either one certified surface chart or
   one physically-filtered mixed atom.  It never names separate independently
   blendable depth, normal and mark records.
4. **Fully filtered.** A compact VQ/codeword describes a brick that is wholly
   below the declared physical resolution.  Its output is premultiplied
   appearance plus a conservative depth interval and normal/material moments,
   explicitly not a fictitious real surface.

The cook must conservatively classify a microcell as mixed whenever a single
winner cannot be certified over its whole ray-domain cell.  Refinement stops
at the maximum physically useful resolution fixed by the guard, display
footprint and error tolerance.  A still-mixed leaf becomes a filtered atom;
it is never forced to choose or average a geometry owner.

The topology codebook contains equality/topology patterns, not raw global
triangle ids.  This is what permits reuse.  Absolute chart ids live in
deduplicated palette sets.  Mixed atoms may be vector-quantised local
appearance functions (for example, a quantised affine local patch plus depth
interval and normal moments), but a cook must measure both approximation and
brick-boundary error.

## 4. Resident-memory equation

Let

- `H=8` be header bytes per brick;
- `C_top` be the number of topology prototypes;
- `b_slot` be bytes per topology slot (`1` for at most 256 local slots);
- `S` be the set of unique deduplicated palette sets;
- `K_s` be the number of payload tokens in palette set `s`;
- `b_tok=4` be a packed payload-token index;
- `N_chart,b_chart` count the reachable categorical chart/copy payloads;
- `C_mix,b_mix` count quantised filtered atoms and their coefficients;
- `M_rho` include the complete categorical carrier first-passage codec;
- `M_aux` include chart tables, occupancy, page tables, alignment and any
  separately cooked angular levels.

Then

\[
\boxed{
M_{\rm resident}=
 HB_\Sigma
 + C_{\rm top}L b_{\rm slot}
 + b_{\rm tok}\sum_{s\in S}K_s
 + N_{\rm chart}b_{\rm chart}
 + C_{\rm mix}b_{\rm mix}
 + M_\rho+M_{\rm aux}.}
\]

For the resource audit's representative aggregate boundary allocation,
`B_0=6,291,456`, hence

\[
 HB_0=50.33\ \mathrm{MB},\qquad
 HB_\Sigma=67.11\ \mathrm{MB}.
\]

A `4096`-prototype byte topology dictionary costs `1.05 MB`.  Even charging
all `2.17 million` source triangles as reachable sixteen-byte charts costs
`34.72 MB`; `65,536` thirty-two-byte mixed atoms cost `2.10 MB`.  Those terms
sum to about `104.98 MB`, leaving about `145.02 MB` of the 250 MB ceiling for
all unique palette sets, the carrier and auxiliary tables:

\[
\boxed{
4\sum_{s\in S}K_s+M_\rho+M_{\rm aux}
\le 145.02\ \mathrm{MB}.}
\]

This is a gate, not a prediction.  If every brick owns a separate palette and
the prior approximate six-owner average persists, palette tokens alone are
about `201 MB` at `B_Sigma`, and the format is red.  It fits only if uniform
bricks, repeated topology, palette-set deduplication, mixed-atom VQ and
physical filtering reduce the measured entropy enough.  No renaming of a
per-brick palette as a codebook changes this arithmetic.

The two golden-angle/affine Tier-1 layers query the same resident community
data under different inverse affine maps, so they double reads but not atlas
bytes.  D4 transforms likewise reuse bytes when they are genuine transforms
of the same compiled community.  A distinct moss community or motion law must
be charged independently unless it is represented by a small analytic medium
or is union-baked with a shared transform.

## 5. Carrier codec is categorical too

`rho_P(q,omega)` is smooth only while one guarded-footprint boundary feature
and periodic copy remain the first entry.  Numerically interpolating distances
across a successor change manufactures a boundary point, exactly the error
class forbidden for the transfer.  The carrier therefore stores the winning
footprint edge/copy chart and reconstructs the live line/edge intersection
analytically.  Its three-dimensional `(q_x,q_z,omega)` label field may use the
same `4^3` macrobrick/palette construction or a measured direct packed-entry
field.  `M_rho` is its complete mip-inclusive cost.

Carrier mixed cells are acceptable only when the chosen entry is certified
not to skip the true first plant hit.  The guard margin supplies room for this
conservative rule.  A carrier approximation which can jump past geometry is
red regardless of image metrics.

## 6. Fixed runtime cost

One lane has this worst-case lookup chain:

1. carrier header/direct entry and optional carrier codeword: `1--2` reads;
2. transfer brick header: `1` read;
3. topology or fully-filtered VQ entry at the selected microcell: `1` read;
4. palette token for a palette brick: `0--1` read;
5. one coupled surface-chart or mixed-atom payload: `1` read.

Thus one lane costs `4--6` reads.  Two independently affine grass layers cost
`8--12`, and the control field raises this to `9--13`.  A compact static moss
medium/direct lane can occupy the remaining one or few reads; a completely
general third arbitrary-soup lane can instead reach roughly `15--19` and must
be surfaced as a cost decision.  The first production gate is green only when
the actually selected modes fit the agreed `<=14` bound.

The ALU is fixed chart/index calculation, one four-dimensional microcell
decode, a few fixed comparisons, and at most one analytic plane intersection
per lane (roughly `35--70` scalar operations per ordinary lane, to be measured
after transcription).  There is no loop, march, candidate list, owner blend,
per-species work, barrier, pass or runtime geometry.

## 7. Locality and temporal continuity

Headers are coherent because neighbouring pixels normally occupy the same or
adjacent boundary/direction bricks.  Codewords must be reordered by their
measured brick-adjacency graph and preferably page-local to angular sectors;
otherwise the palette and payload fetches become dependent random traffic.
Correction/palette occupancy and cache-page working set are binding cook
metrics, not post-hoc performance excuses.

Categorical temporal continuity is protected structurally: a regular cell is
published only after whole-cell winner certification, so an owner change at a
grid boundary is either a real visibility change or is surrounded by the
filtered mixed band.  Filtered atoms and independent VQ bricks do not possess
an automatic `C0` seam theorem.  Their fit therefore includes shared-face and
camera-translation losses, and the cook must measure maximum/p95 seam jumps
plus unforced class-change rate under `1--4.5 mm` camera translations.  A red
temporal result cannot be repaired by blending categorical geometry.

## 8. Mandatory cook gate

Before runtime transcription, the actual Calamagrostis and one dense low
community must report:

- `B_0`, every mip-level brick count and exact bytes;
- mode histogram and regular-cell certification margins;
- topology-codebook size/error;
- unique palette-set count, `K_s` distribution and exact palette bytes;
- reachable chart/copy count and payload bytes;
- mixed-atom count, VQ distortion and mixed-band screen width;
- complete categorical carrier bytes and no-skip proof failures;
- silhouette IoU, RGB p95, crisp position p95 and depth/compositing error;
- shared-face discontinuity and millimetre-translation stability;
- page working set/locality and exact worst-case read schedule.

Green means the complete resident sum is at most 250 MB, the selected runtime
schedule is at most the agreed fixed read ceiling, and all visual/temporal
thresholds pass.  If palette entropy, carrier entropy, mixed VQ distortion or
seams are red after one diagnosis/fix cycle, this codec is parked with that
measured blocker.  Adding traversal, more live candidates or owner blending is
not an allowed repair.

## 9. Verdict

This is a complete fixed-cost finite **compiler target** for the new guarded
boundary-transfer model.  It retains the mathematical dimensional reduction,
couples all resolved attributes to one real surface chart, and assigns the
only approximation to declared physically unresolved cells.  It has a
plausible concrete path into the current 250 MB and roughly 9--14-read
envelope, but it does not and cannot supply a soup-independent fixed-byte
guarantee.  The next fact is empirical and narrow: the palette-set entropy,
carrier size and mixed-brick distortion of the accepted communities.

## 10. Provenance and paper boundary

Primary-source ancestry is deliberately narrow:

- Sannikov's 2019 precomputed-raycast article supplies the periodic
  parallel-extrusion first-passage/lift used for `rho_P`; the local PDF and
  later-author record are bound in
  `docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.
- Levoy and Hanrahan's *Light Field Rendering* and the Lumigraph family supply
  the prior-art fact that free-space rays form a four-dimensional exterior
  field.
- Ai, Wang, and Kou's *Cube Surface Light Field* papers are the closest
  boundary-surface parameterisation precedent, and also document the raw-size
  and filtering cost of a dense geometry-free field.
- Lin and Shum's geometric light-field analysis supplies the prior-art warning
  that approximate depth/view interpolation broadens or doubles geometry.
- Chai et al.'s *Plenoptic Sampling* supplies the spatial/angular filtering
  context, not the categorical codec or its grass guarantee.

The following are LAAS derivations and must not be attributed to those papers:
the guarded periodic common-slab reduction; composition of categorical
`rho_P` with a whole-forward marked boundary transfer; interval-certified
regular microcells; the three-mode `4^4` macrobrick format; topology/palette
deduplication with analytic live-plane reintersection; mixed records confined
to a declared physical pixel kernel; the carrier no-skip condition; and the
complete grass/moss terrain-motion budget and acceptance gate.

The primary-source URLs and negative mechanism comparisons are preserved in
`docs/deep-research/grass/FIXED-READ-NOVEL-VIEW-PRIMARY-SOURCE-NEGATIVE-BOUNDARY.md`
and `docs/deep-research/grass/FUR-PRECOMPUTED-VISIBILITY-TRANSFER-AUDIT.md`.

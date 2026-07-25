# Coupled band-limited three-event-sheet feasibility

Status: **rejected and parked after one actual-asset attempt**, 2026-07-22.
No runtime asset or shader implements this candidate.

## Decision

A compact, fixed-cost coupled record can preserve exact authored event atoms
and the required same-line successor identity, but **three monotone depth
sheets do not form a globally integrable visibility field across even a small
pinhole camera/pixel patch** of the accepted Calamagrostis asset.

The strongest tested record fits the complete resident-memory/read/ALU
envelope. It nevertheless has nontrivial sheet-transport holonomy on
`645 / 963 = 66.978193%` of fully populated camera/pixel plaquettes. Its
optimistic three-medoid reduction also has `0.601579 m` p95 conditional
one-dimensional Wasserstein depth error before depth quantisation, palette
fitting, angular interpolation, inside-origin storage, multi-species overlap,
or moss.

Therefore the candidate is not integrated. Increasing rank, filtering the
result again, or choosing a prettier sheet permutation cannot remove the loop
inconsistency. One of those operations must tear, swap owners, or stretch a
sheet somewhere around the loop. This is the exact family of failure visible
as camera-centred fans and wrong-perspective widening.

## 1. Candidate mathematical record

For one pointed query `q`, one footprint level `m`, and nine persistent sensor
labels `xi_i`, trace the exact marked interactions

\[
  G_m(q,\xi_i)\in\{\bot,E_i\}.
\]

Condition the empirical measure on hits, order only its scalar ray depths, and
split the ordered atoms into three contiguous quantile intervals. Each interval
stores its **actual median event**, not an arithmetic event mean:

\[
  R_m(q)=\left(n_h,\widehat E_0,\widehat E_1,\widehat E_2\right).
\]

The common label `xi` selects miss from `n_h/9`, then one of the three
conditional quantile sheets. Position, colour, normal, material, and botanical
mark always come from the same selected authored event. This construction was
the most favourable tiny inverse-CDF carrier still compatible with the coupled
truth contract; it does not claim that scalar depth order is a correct global
surface label.

The proposed packed record is exactly 64 bits:

- 4-bit conditional hit count `n_h in [0,9]`;
- three 14-bit finite-line depths; and
- one 6-bit joint authored appearance/normal/material/mark palette index per
  event atom.

Across neighbouring directional samples, corresponding quantile ranks are the
proposed coupled sheets. Three directional records define three hit points for
the selected sheet; the live ray intersects their fixed three-point plane.
Categorical appearance comes from one actual atom rather than blending owners.
The proposed live path is three fixed reads and less than 192 FMA-equivalent
operations, with no loop, march, traversal, candidate list, shell, extra pass,
or per-species query.

## 2. Line anchoring and footprint ladder

The gate does not regenerate an unrelated random disk per query. It creates a
labelled `3x3` pinhole sensor grid once. For the same-line test, every member
moves along its own micro-direction,

\[
  A_i(s)=A_i(0)+s d_i,
\]

so its oriented line is exactly invariant. For the camera test, the same nine
sensor labels persist while adjacent columns advance by one real device pixel
and adjacent rows translate the camera laterally by `5 mm`.

The footprint half-angle is derived from a `55 degree / 2160 px` view. Levels
`1,2,4,8` retrace increasingly wide exact micro-line bundles. Coarse levels are
not arithmetic averages of finer records.

## 3. Complete fixed-cost budget

The concrete carrier uses 71 directions and the complete independently baked
periodic footprint pyramid with sizes

```text
258, 129, 65, 33, 17, 9, 5, 3, 2, 1
```

At eight bytes per record this is

\[
  71\left(258^2+129^2+\cdots+1^2\right)8
  =50,511,104\ \text{bytes}.
\]

That leaves `610,048` bytes inside the hard `51,121,152`-byte cap for the
64-entry joint palette, spherical triangulation, gutters/metadata corrections,
and alignment. The cost gate therefore passes. The visibility gate does not.

This layout is additionally optimistic: the actual-source camera-path oracle
constructs the pointed-query record directly. A production all-angle field must
still represent arbitrary inside-origin successor phase. There is no remaining
space for a dense origin coordinate, so the measured failure is a lower bound,
not an implementation complaint.

## 4. Actual-asset structured gate

Immutable source:

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
SHA-256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
```

The bounded attempt traced `22,455` exact periodic mesh rays, including:

- 24 transported same-line sequences at grazing-to-steep downward elevations
  `0.1/1/5/15/35/75` degrees, each with 17 origins;
- two phases at 5 and 35 degrees on `17x17` camera-frame/pixel grids;
- four independently traced footprint levels at the 5-degree stress row; and
- 64 actual-panicle-target records over 16 authored panicle triangles and all
  four footprint levels; and
- only fully populated three-sheet edges/plaquettes in the permutation and
  holonomy result, excluding missing-sheet symmetry.

Results:

| gate | requirement | measured | result |
|---|---:|---:|---:|
| uncrossed same-line event identity | 100% | 100% over 738 comparisons | pass |
| same-line world-event jump p95 | numeric zero | `6.55e-14 m` | pass |
| p95 conditional depth `W1` | `<=0.02 m` | `0.601579 m` | **fail** |
| p95 monotone-vs-local-optimal edge cost | `<=0.02 m` | `0.000948 m` | pass |
| nontrivial plaquette holonomy | 0% | `66.978193%` | **fail** |
| p95 panicle-mass absolute error | `<=1/9` | 0 in this bounded sample | pass |

The edge metric and loop metric are deliberately separate. A locally cheap
permutation exists along most individual edges, but those preferred
permutations do not compose consistently around two-dimensional camera/pixel
loops. At the 5-degree finest footprint, nontrivial holonomy is `75.41%` and
`83.98%` in the two phases. The four-times-wider footprint remains nonintegrable
at `46.77%`; the eight-times-wider footprint is `52.45%`. Filtering visibility
does not turn the field into three coherent global sheets.

The panicle gate is non-vacuous: 60 records contain 540 exact panicle hits, and
the 64 targeted records are `93.75%` panicle by hit mass. Their zero
panicle-mass error is a useful positive observation for this bounded
single-species sample: retaining actual medoids avoids the immediate colour and
mark averaging failure. It does not rescue the geometric coupling, and no
claim is made for an overlapping community or moss.

## 5. Reproduction and immutable artifact

Command:

```text
node --import tsx tools/groundcover-bake/analyze-coupled-quantile-event-sheets.ts
```

Tool SHA-256:

```text
b5cfceb7edfbc5b5797b37276bf1672d084dc9252a56d6099d710e3b4f920fed
```

Artifact root:

```text
data/work/groundcover-coupled-quantile-sheet-gate/
  2ed57f59d86e8376/b70facc20d553407/
```

- configuration SHA-256:
  `b70facc20d55340766788a3886703c4b0934a8f610f70d1501dd282e2833735d`
- `metrics.json` SHA-256:
  `3507b001cdb498f48fc6fe9b3c3e44c4e96ee69f25c7284c6d0205f3d314db44`
- `qa/000-coupled-camera-path-sheets-and-holonomy.png` SHA-256:
  `d82f3fb6d586df27b4b72d347b9aa3f687d09f6a28a7ddedb3e505f4231c99b4`

The QA image has seven rows of camera/footprint cases. Its first three panels
are the botanical class of the front/middle/back authored medoid. Red in the
fourth panel marks a nontrivial local sheet-transport loop.

## 6. Sources and LAAS contribution boundary

- Sannikov supplies the repeating, copy-count-independent precomputed ray-field
  objective and the public warning that baked/live direction disagreement
  stretches geometry. His public work does not specify this three-sheet record,
  the persistent bundle, or the holonomy gate.
- Standard one-dimensional optimal transport supplies the fact that common
  quantile rank is the monotone coupling on the real line. It does **not** imply
  that depth quantiles are integrable surface labels over a multidimensional
  camera/ray field. The present use is an LAAS hypothesis, now rejected on the
  asset.
- View-Dependent Displacement Mapping and Generalized Displacement Maps supply
  the prior-art boundary that direct view conditioning and an origin-aware
  field can be fixed cost. Their scalar interpolation is not taken as an owner
  coherence proof.
- Fur PRT remains a negative boundary. It precomputes lighting on shell
  carriers; it neither provides this first-event coupling nor meets the no-shell
  contract.
- The 64-bit complete-atom packing, line-transported sensor gate, minimum-cost
  edge comparison, and plaquette holonomy test are LAAS constructions. The
  measured negative result is local to the immutable Calamagrostis source and
  declared query domain.

Exact primary links and transfer/rejection decisions are maintained in
`../../deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.

## 7. Park/resume condition

Do not reopen K=3 scalar-depth quantile sheets by adding reads, ranks, filters,
direction slices, or a runtime selector. Resume only if a new representation
has a path-independent global marked-event label (or an equivalent analytic
visibility-sharing law), proves arbitrary inside-origin successors, and fits
the same byte/read/ALU contract before shader work.

Multi-species cover and moss remain one offline-composed marked community
query. Separate live species fields, one winner query per species, or a moss
shell are not a valid resume path.

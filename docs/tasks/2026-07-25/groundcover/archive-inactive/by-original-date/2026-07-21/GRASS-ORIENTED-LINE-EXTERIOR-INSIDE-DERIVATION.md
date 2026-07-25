# Periodic ground-cover visibility: exterior, inside-origin, and overlap derivation

**Status:** pure-mathematics decision record, 2026-07-22. No runtime shader
change follows from this note by itself.

**Inspectable outcome governed by this work:** one artifact-free, free-camera
Calamagrostis view which remains a fixed-cost precomputed-ray query and extends
without a query per species to overlapping marked ground-cover communities.

## 1. Result in one page

There are two different mathematical problems which the renderer had conflated.

1. For a camera outside the botanical band, the ray has a canonical entry
   point on the botanical top (or bottom) plane. That point is exactly invariant
   to which datum on the ray was used. Reconstructing this point is fixed affine
   arithmetic and never requires a translated terrain shell.
2. For a camera inside the botanical band, the desired answer is the successor
   of a **pointed** oriented line. An exterior top-first event does not determine
   that successor. The extra origin coordinate is real; no trigonometric
   rescaling removes it for arbitrary finite branching geometry.

For an arbitrary marked 3D population, the continuum exterior field is four
dimensional and the inside-origin successor field is five dimensional. This is
also the semantic boundary reached independently by Generalized Displacement
Maps. A direct field query remains O(1) in repeated copy count, but O(1) runtime
does not make a dense exact field small.

The cheapest exact mathematical representation is therefore a direct marked
successor oracle in pointed-line space. Under the existing 51,121,152-byte
resident-profile ceiling it is not a feasible uncompressed implementation. The
actual-mesh deep-ray census independently rejects replacing the fifth dimension
with a tiny ordered-event record: exact horizontal rays contain up to 3,588
events in the declared 155 m horizon.

The only structurally correct bounded implementation family left is an
**offline-certified adaptive line-cell atlas**. Pure cells contain one analytic
surface chart. Boundary cells contain a shallow, fixed separator program and
are refined offline until either that fixed program is sufficient or the whole
remaining disagreement is conservatively below the declared screen/depth/
radiance tolerance. The hierarchy is flattened to a fixed number of direct
indirections; runtime never traverses it. Camera-inside data uses the same
construction in pointed-line space. This is a candidate format, not an assumed
success: the current actual-asset separator probe already proves that a tiny
unrefined tail is insufficient, so a byte-and-error bake must pass before any
shader integration.

There is no complete all-angle shader fix available from the present 16 x 4
independent first-hit slices alone. In particular, neither restoring the
exterior top entry nor adding a 5-degree row creates the missing angular owner
field or the inside-origin successor field.

## 2. Coordinates and invariants

Let `J` be the local botanical derivative basis and write the ray in profile
coordinates as

\[
q(t)=q_C+t v,\qquad q_C=J^{-1}(C-b),\qquad v=J^{-1}d.
\]

The marked composite population `G` is periodic in `(q_x,q_z)` with period
`(T_x,T_z)` and is bounded vertically by

\[
y_{min}\le q_y\le H.
\]

Every complete event carries position/depth, orientation, coverage, colour or
material parameters, and a species/material mark. Those values are one event;
unrelated owners are never independently averaged.

Any query address `chi(A,d)` for an ordinary oriented-line field must satisfy

\[
\boxed{\chi(A+\lambda d,d)=\chi(A,d)}.
\]

If this identity fails, range, terrain slope, or the arbitrary choice of scene
datum can change the represented surface. This is precisely why the rejected
query-dependent orthogonal carrier generated camera-centred fans.

## 3. Exact exterior entry without a raised shell

Assume the camera lies above the complete population, `q_{C,y} >= H`, and the
ray enters the band, `v_y < 0`. Its top-plane entry parameter and phase are

\[
\boxed{
t_H=\frac{H-q_{C,y}}{v_y},\qquad
Q_H=q_C+t_Hv.
}
\]

Replacing `C` by another datum `C+lambda d` on the same ray changes `q_C` by
`lambda v` and `t_H` by `-lambda`, hence

\[
\boxed{Q_H(C+\lambda d,d)=Q_H(C,d)}.
\]

Equivalently, if the renderer already owns any datum `A=C+t_A d` on the ray
(for example the actual terrain/scene hit) and its botanical height `q_{A,y}`,
then

\[
\boxed{
t_H^{world}=t_A+\frac{H-q_{A,y}}{v_y},\qquad
Q_H=q_A+v\frac{H-q_{A,y}}{v_y}.
}
\]

For a downward ray from above, this parameter lies before the terrain datum.
It is an algebraic coordinate only; nothing is rasterised or translated to
`H`.

The periodic exterior address is therefore

\[
\left(\operatorname{mod}(Q_{H,x},T_x),
      \operatorname{mod}(Q_{H,z},T_z),\;[d]\right),
\]

where `[d]` denotes two coordinates on the oriented direction sphere. If a
complete exterior oracle returns signed line parameter `s_h` from `Q_H`, the
world hit is simply

\[
\boxed{P=b+J(Q_H+s_hv)}.
\]

This is the correct replacement for a physical `H`-translated carrier. It adds
no geometry, draw, pass, binding, barrier, loop, or march. Different communities
may have different `H` because `H` belongs to the selected offline population,
not to terrain.

The top-plane coordinate has an apparent `1/v_y` grazing pole. That pole does
not imply a missing exterior hit: for `q_{C,y}>H` and `v_y=0`, the ray remains
above `G` and is exactly a miss. For finite near-grazing `v_y`, large travel is
real. Robust implementation must reduce periodic phase before low-precision
conversion; clamping the denominator changes the line and is invalid.

For a completely general exterior origin or a bounded coordinate chart, choose
the signed dominant component `j` of unit direction `d` and one fixed chart
plane `x_j=k_j`. With `a,b` the other axes,

\[
Q=A+d\frac{k_j-A_j}{d_j},\qquad
\ell_j=\left(Q_a,Q_b,\frac{d_a}{d_j},\frac{d_b}{d_j}\right).
\]

Because `|d_j| >= 1/sqrt(3)`, this six-chart representation has no live
direction pole and remains line-origin invariant. Chart overlaps must return
the identical marked event.

### What this exterior identity does not solve

The formula supplies the right ray coordinate. It does not synthesize the
unknown value of the four-dimensional arbitrary-mesh first-owner field between
the existing 16 x 4 direction slices. Using a neighbouring slice and deforming
its event is exact only for the extrusion/single-chart hypotheses already
rejected for Calamagrostis. Thus restoring `Q_H` can remove the moving-datum
error but cannot honestly be called the all-angle solution.

## 4. Camera-inside successor theorem

Parameterise one oriented line `L` by `r_L(s)`. The ordered intersections of
the marked population are

\[
\ldots<s_{-1}<s_0<s_1<\ldots.
\]

For a camera at signed line phase `sigma`, the required result is

\[
\boxed{E^+(L,\sigma)=\min\{s_i:s_i>\sigma\}}.
\]

Two cameras may lie on the same `L` but on opposite sides of an event. They
have the same four oriented-line coordinates and different correct successors.
Therefore no function of `L` alone can answer both. The input is a pointed
oriented line `(L,sigma)`, which is five dimensional in 3D.

This gives a direct impossibility proof for all one-exterior-event proposals.
Suppose a decoder `D(L)` returned the correct successor for every camera on
`L`. Choose `sigma_1<s_k<sigma_2<s_{k+1}`. Correctness requires
`D(L)=s_k` for `sigma_1` and `D(L)=s_{k+1}` for `sigma_2`, a contradiction.
Changing top/middle reference height, normal basis, cosine correction, or wind
offset cannot alter that information deficit.

The pointed dimension may be represented in exactly three ways:

1. a direct five-dimensional successor field;
2. a four-dimensional deep record containing enough ordered events to select
   the successor;
3. live geometric search/traversal.

The third violates the runtime contract. The second fails the measured
155-metre free-camera domain: the actual asset has as many as 3,588 retained
events on one exact-horizontal line, 2,052 at 0.1 degrees downward, and 110 at
5 degrees downward. The first is consequently the cheapest exact fixed-query
semantics, but it requires compression rather than wishful dimension removal.

## 5. Exact storage and fixed-work lower accounting

These byte counts are not information-theoretic lower bounds; they are the
smallest direct layouts comparable to the existing atlas and show why a dense
implementation is not authorised.

At `256 x 256` phase samples, 81 direction nodes, and eight bytes per complete
geometry event, one exterior sample layer is

\[
256^2\cdot81\cdot8=42,467,328\text{ bytes}=40.5\text{ MiB}.
\]

Including the existing one-texel gutters gives 43,133,472 bytes. This is only a
sampled four-dimensional exterior field; the measured direction-lattice audit
shows that nearest sampling of it is nowhere near a continuum-quality answer.

Adding even two directly sampled origin phases doubles the unguttered field to
81 MiB. Eight and sixteen origin phases cost 324 MiB and 648 MiB respectively.
Four-byte records merely halve those totals while omitting payload which must
then come from another read. A literal 3,588-event deep version of the eight-byte
exterior field is about 141.9 GiB before attributes or selection data, and a
fixed predecessor selector would still have 3,588-way work.

The accepted resident ceiling is 51,121,152 bytes (48.753 MiB). Therefore:

- the dense exact pointed-line oracle does not fit;
- a tiny exact deep record is disproved by actual geometry;
- adding directions to the existing nearest categorical lattice is not a
  substitute (the separate audit reaches a 1.583 PiB direction-only knee and
  still retains metre-scale tails);
- any viable all-angle field must exploit certified visibility-cell structure
  or a measured compression mechanism, not arithmetic depth blending.

### 5.1 Why `arbitrary geometry` has no universal tiny exact codec

This is not only a failure of the codecs tried so far. Take `N` pairwise
separated representative rays in general position. Around a private point on
each ray, choose an opaque patch small enough that it intersects that ray and
none of the other `N-1` rays. Including or omitting each patch independently
constructs `2^N` finite meshes with all `2^N` possible hit/miss vectors on the
chosen rays. Any codec exact for every arbitrary mesh on those queries must
distinguish all `2^N` cases and therefore carries at least `N` bits before
depth, normal, colour, mark, filtering, or inside-origin order are represented.

This counting argument does **not** prove that the particular Calamagrostis
asset cannot compress. It proves that no algebra or general-purpose promise of
O(1) runtime can guarantee a tiny exact field for arbitrary input geometry.
The required reduction must come from measured, asset-class structure and must
retain that structure when several species are composed.

## 6. Cheapest structurally correct bounded representation

The finite source mesh induces a finite semialgebraic partition of oriented-line
space. Within one visibility cell, the first owner is one surface chart and its
hit is an analytic ray/plane intersection. Cell boundaries are generated by
silhouette edge predicates and depth-order changes. This suggests the only
bounded format which preserves actual geometry rather than stretching a nearby
view.

### 6.1 Offline cell construction

For exterior queries use the four coordinates `ell_j`; for inside-origin
queries append the signed origin phase `sigma`. Build an adaptive dyadic
partition offline. For each cell:

1. prove that every represented ray has the same marked surface chart, then
   store that chart's analytic plane/attribute record; or
2. store a fixed shallow separator program over exact line predicates, with one
   chart record per leaf; or
3. subdivide; or
4. only at a declared screen-filtering limit, store a conservative filtered
   event measure and its unresolved depth/radiance bounds rather than calling
   it one exact surface.

An unresolved geometric cell is accepted only if its conservative screen error
over the full declared camera/range domain is bounded. For a cell whose possible
hits lie in set `P_cell`, one sufficient bound is

\[
\boxed{
\epsilon_{px}\ge
\sup_{P_1,P_2\in P_{cell}}
\frac{f_{px}\,\|(P_1-P_2)_\perp\|}{z_{min}}
}
\]

together with explicit maximum depth, normal, coverage, and radiance ranges.
Different species marks or hit/miss states are never collapsed merely because
their mean position is close. At an unresolved true silhouette the record is a
coverage/radiance measure at the stated footprint, not an invented opaque
mid-depth sheet.

### 6.2 Fixed runtime form

Flatten the adaptive structure offline into a bounded page table and record
atlas. A viable runtime contract is:

- one direct coarse-cell read;
- at most one direct refinement-page read;
- a compile-time fixed `S` separator evaluation using predicated selects, not a
  triangle list or traversal;
- one selected complete event/chart payload read and one analytic intersection;
- no data-dependent loop, march, candidate election, per-copy geometry, or
  per-species query.

All paths execute the same declared maximum read/ALU envelope. Empty and coarse
cells may share records. The bake fails if the actual asset cannot meet the
byte cap, fixed `S`, locality, and conservative error bounds. The existing
probe (202 owners plus miss in one current-scale cell, and at least six binary
decisions after one uniform subdivision) means `S` cannot simply be assumed
tiny; adaptive storage must be measured before this format is accepted.

This is preferable to a neural mean-distance codec or filtered depth atlas
because every geometric output is either tied to one certified source chart or
explicitly labelled a bounded pixel-filtered measure.

## 7. Overlapping species, moss, and population states

Let species or cover components be marked sets `G_i`. The opaque result for one
ray is

\[
t_*=\min_i t_i,
\]

and the winning event carries mark `i`. Querying every `G_i` at runtime makes
cost grow with species count. The O(1) solution is to bake the ecologically
valid community union

\[
\boxed{G_{community}=\bigcup_i G_i}
\]

before building visibility cells. Tall grasses, low leaves, litter, flowers,
and moss then participate in the same offline owner/depth election. Runtime
performs one community query; repetition count and source species count do not
appear in its cost.

Different community mixtures or seasonal states are separate precomposed
marked fields selected before the query. Supporting arbitrary live subsets is
the successor/subset problem again and can require `2^N` precompositions; it is
not part of the one-field contract.

The community's exterior `H` is the maximum height of that composed state, not
a global `1.176 m` constant. Moss remains low because its winning event is low;
it does not require or create a visible plane at `H`.

## 8. Exact implications for the active implementation

1. The physical raised terrain copy remains deleted. Its mathematically correct
   exterior replacement is the algebraic `Q_H` entry above.
2. The query-dependent orthogonal carrier remains rejected; it violates line
   invariance before sampling error is considered.
3. Restoring an algebraic exterior entry is safe as a controlled exterior
   baseline, but it must not be described as the complete all-angle fix while
   the live value still comes from the independent 16 x 4 direction slices.
4. No 5-degree slice, PCF quantile, depth average, or normal correction recovers
   an absent first owner or camera-inside successor.
5. The full solution requires a new offline representation. The adaptive
   certified line-cell atlas is the strongest surviving candidate. It receives
   one go/no-go bake on the actual Calamagrostis mesh: fixed reads and separator
   depth, <=51,121,152 resident bytes, exact line/chart invariants, and declared
   worst-case image/depth/radiance error. Failure means the simultaneous
   arbitrary-mesh, unrestricted-inside-camera, crisp-near-field, and current
   byte/read constraints are incompatible for this asset; it does not authorise
   another shader-space deformation.

### Runtime observation after restoring the identity

The isolated specialization was changed from terrain-datum entry to the exact
algebraic top entry in Section 3 without restoring any carrier geometry. User
review found no visible improvement: stretching and wrong perspective remain.
This is consistent with the proof rather than a contradiction of it. The entry
identity removes datum/range dependence, but the queried value is still taken
from the incomplete independent `16 x 4` angular field. The correction remains
necessary and the representation remains insufficient.

## 9. Provenance boundary

Source-backed facts used here are recorded in:

- `docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`;
- `docs/deep-research/grass/SANNIKOV-ALGORITHM-AND-RUNTIME-MISMATCH-AUDIT.md`;
- `docs/deep-research/grass/FUR-PRECOMPUTED-VISIBILITY-TRANSFER-AUDIT.md`;
- `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-DIRECTION-LATTICE-FEASIBILITY.md`.

The exterior invariance proof, pointed-line contradiction, byte accounting,
adaptive certified-cell contract, and marked-community composition in this note
are LAAS derivations. They must remain distinguishable from Sannikov's exact
parallel-extrusion theorem and from GDM's dense origin-aware sampled field if
the work is later written up as a paper.

## 10. Actual-asset adaptive-cell gate

The one permitted real feasibility attempt rejects Section 6's adaptive atlas
under the current byte and shallow-work envelope.

Tool:
`tools/groundcover-bake/analyze-adaptive-line-cell-atlas.ts`, SHA-256
`1681c0845a921455673a4b7e49098eb02201052816c2a8cc47039bc8c30d216e`.
It traces analytic periodic successors on the accepted GCRP/v4 mesh; no runtime
asset or shader is emitted.

Artifact:
`data/work/groundcover-adaptive-line-cell-probe/2ed57f59d86e8376/0f4abaf3b1b6eddb/`.
Configuration SHA-256 is
`0f4abaf3b1b6eddbfc8837fd435b010fe0525a5eeff3e6179828228d81d04306`;
`report.json` SHA-256 is
`c51195163febbd3d32576273ccae9f0e08723afbe0a0abddbea49939f4efc69b`;
`index.json` SHA-256 is
`31df6ea2b1432d99d0b9f053edd4a257614158005055decf9a1b9b62f1df9290`.

The probe evaluates `173,502` actual-mesh pointed rays in 30 dense coarse
cells: two deterministic cells for every exact horizontal and signed
`0.1/1/5/15/35/75/90` degree class. Periodic phase X/Z, inside-origin height,
and azimuth vary on a tensor lattice inside each nonvertical cell. It measures
refinement levels 0, 1, and 2 and projects the next uniform page level. Runtime
shape is fixed at two indirections: one coarse descriptor and one direct page
record, followed by a compile-time separator program. There is no traversal,
loop, march, candidate list, copy geometry, or species loop.

Results:

| maximum separator decisions | sampled coarse cells still failing after level 2 | unresolved level-2 subcells | unresolved hit/miss-mixed subcells | projected bytes |
|---:|---:|---:|---:|---:|
| 2 | 27 / 30 | 5,355 | 2,647 | 327,466,880 |
| 4 | 25 / 30 | 2,715 | 917 | 316,689,152 |
| 6 | 13 / 30 | 13 | 4 | 357,457,664 |

The decisive projection is deliberately generous. Every failing page at the
next refinement receives only one 12-byte owner/event per subcell and **zero**
separator bytes. It therefore understates any implementable exact layout. Even
the four-decision result is 6.20 times the `51,121,152`-byte cap. The six-
decision result is worse because the sampled pages which do fit six decisions
already carry many distinct complete owner leaves.

Stopping at the measured level is not a bounded visual approximation. For the
four-decision layout, unresolved sampled cells have up to `70.5875 m` observed
successor-depth spread and include 917 direct hit/miss contradictions. Without
a complete owner arrangement the formal conservative depth error remains the
full `155 m` horizon and categorical radiance error is unbounded. The next
uniform azimuth cell is still `5.625 degrees`, about `203.68` device pixels for
the declared 55-degree/2160-pixel view and `15.229 m` wide at 155 m. It is not
a subpixel filtering tail.

This is a sampled lower bound, not a continuum certificate: unseen owners can
only add required decisions and bytes. The track is therefore parked after one
coherent actual-asset attempt. Reopen only if a new representation shares exact
visibility decisions across pointed-line pages without a dependent traversal
and comes with an analytic reason why this measured owner entropy compresses.

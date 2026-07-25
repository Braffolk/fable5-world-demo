# Candidate P adaptive direction lattice: pure-math capacity test

Date: 2026-07-24  
Status: local 2x capacity decisively RED; parked; no 4x/runtime work authorised

## 1. Question and invariants

Candidate P's corrected v4 gate localised its material affine-rank-two error
to the `10--18 deg x 247.5--270 deg` standing-direction quadrilateral.  The
phase-only `64 x 64` finite element is nearly sufficient.  This note asks one
upstream question only: can that quadrilateral be refined without changing
the representation's total allocation or runtime shape?

The construction must retain:

- `256` allocated angular records per phase vertex, at most `240` real cells;
- `3` record reads per affine layer and the existing `16 MiB` P atlas;
- canonical `C0` angular edges, pole, seam, and one-sided horizon coverage;
- analytic constant-time direction-to-cell mapping using fixed arithmetic,
  comparisons, and selects only--no search, loop, or lookup texture.

This is a topology/cost existence proof, not a claim that narrower cells pass
the image gate.  The latter is the single measurement in section 6.

## 2. Conforming annular count identity

Let two adjacent latitude rings contain `m` and `n` vertices.  A conforming
triangulation of the annular strip, with no interior vertex, has exactly

```text
F = m + n
```

triangles.  Indeed, the annulus has Euler characteristic zero, boundary-edge
count `E_b=m+n`, and `3F=2E_i+E_b`; combining this with
`V-(E_i+E_b)+F=0` gives `F=E_b`.  A pole cap incident to an `m`-vertex ring
has `m` triangles.  Thus rings of sizes `m_1,...,m_k`, ending at the horizon,
use

```text
T = m_1 + sum_(i=1)^(k-1) (m_i+m_(i+1)).
```

This identity also explains why globally doubling a standing annulus is the
wrong upstream move: the transition cost propagates around the whole ring.
Local conforming red/green refinement avoids that propagation.

## 3. One chart, frozen before measurement

The adaptive mesh uses the logical polar cylinder

```text
r   = (90 deg - elevation)/90 deg in [0,1],
psi = azimuth/(22.5 deg) modulo 16.
```

Constant-elevation rings are straight `r=constant` lines in this chart and
azimuth sector boundaries are straight `psi=integer` lines.  Each logical
rectangle is split by its low-elevation/low-azimuth to
high-elevation/high-azimuth diagonal.  This deliberately changes the old
gate's slope-affine triangles: no conclusion may mix truth directions or
heldouts generated in `s=d_xz/|d_y|` with these logical-polar cells.  The
targeted gate generates every vertex and barycentric heldout in `(r,psi)`.

At the seam, `psi=0` and `psi=16` are the same canonical edge.  At `r=0`, all
azimuth coordinates are identified with one pole vertex; every incident cap
triangle duplicates the same pole state.  Thus the quotient of the logical
cylinder is the closed descending hemisphere with one horizon boundary.

## 4. A 176-cell base with the two binding control bands unchanged

Use the unique pole, sixteen canonical azimuth sectors of width `22.5 deg`,
and the six non-pole rings

```text
elevation: 75, 45, 18, 10, 2, 0 degrees.
```

The final ring is the one-sided horizon boundary.  The base count is

```text
pole cap                         16
five 16-to-16 annular strips  5*32
total                            176 triangles.
```

The measured standing quadrilateral is exactly one base quad in the
`10--18 deg` strip.  The measured grazing control remains exactly one base
quad in the `0--2 deg` strip.  Refinement therefore need not alter the
grazing cell at all.  The coarser high-elevation bands are an allocation
witness, not yet a fidelity assertion; a constructive child result must be
followed by the global allocation gate before implementation.

The azimuth seam identifies sector `15`'s last edge with sector `0`'s first
edge.  Every pole incident triangle uses the same canonical pole vertex and
state.  The `0 deg` ring is retained as a boundary, so the descending
hemisphere has complete pole/seam/horizon coverage.

## 5. Local `p x p` refinement and exact count

Refine the one standing quad into `p x p` quads, each split along the frozen
diagonal.  This replaces two triangles by `2p^2`.  Each of its four
face-neighbour quads receives the `p-1` new vertices on the shared edge.  A
quad with those collinear boundary vertices is a convex `(p+3)`-gon and is
triangulated into `p+1` triangles rather than two.  No new point appears on
the neighbour's other three edges, so conformity stops after this one green
ring.  The exact increment is therefore

```text
Delta T = (2p^2-2) + 4(p-1) = 2p^2+4p-6.
```

Consequently:

```text
p=2: T = 176 + 10 = 186 real triangles
p=4: T = 176 + 42 = 218 real triangles.
```

Both are below the frozen `240`-real-cell ceiling.  The atlas remains
physically allocated for all `256` records, hence remains exactly

```text
64^2 phase vertices * 256 records * 16 bytes = 16 MiB.
```

Keeping the affected base slots unused and appending dedicated refined slots
is also safe: at `p=4` the appended target/fan allocation is
`32 + 4*5 = 52`, so the largest physical slot count is `176+52=228 <256`.
No compaction lookup is required.

The existing document assigns `240` real cells plus sixteen padding records,
but no decoder invariant consumes those padding records.  Superficially, a
`p=2` result would permit `240+10=250 <=256`.  That is **not yet a complete
compatibility witness**, because the old topology starts at `0.25 deg` and
does not represent the exact horizon or the `0--0.25 deg` fringe.  Candidate
L already showed that this fringe is not constant at a fixed fine footprint.
Only six records would remain after local refinement, whereas a conforming
sixteen-sector horizon strip costs thirty-two triangles.  Therefore the old
mesh cannot simply be preserved unless a separately derived, physically
gated analytic fringe uses the existing footprint-to-`M_inf` contraction
without clamping or hiding any direction.  The current local gate does not
test or authorise that construction.

The `186`-cell witness has an explicit `0 deg` ring and remains the complete
same-budget topology if `p=2` is GREEN.  At `p=4`, `240+42=282` is impossible;
under the stricter `<=240`-real-cell policy the `218`-cell redistribution
witness is required.  Any global follow-on
must score physically realisable footprints at the exact horizon and inside
`0--0.25 deg`; the current `.25--2 deg` grazing control is not a substitute.

A less destructive reusable `p=4` topology also exists if a future codec
needs it: retain eight radial knots, use eight azimuth sectors on the `75 deg`
and `55 deg` rings, and sixteen on all other rings.  Its conforming count is

```text
8 + (8+8) + (8+16) + 5*(16+16) = 208.
```

The sole unequal annulus is an aligned ratio-two `8->16` transition and has a
fixed floor/parity locator.  Replacing the ten retired target/green base
triangles with the fifty-two `p=4` target/fan triangles gives
`208-10+52=250` real records.  Append-only allocation would incorrectly give
`260`; the ten retired slots must be reused.  This is reusable topology only,
not active work after the RED result below, and its halved high-elevation
azimuth resolution would require an independent fidelity gate.

## 6. Analytic constant-time addressing and `C0`

Let `e` be elevation and `phi` canonical azimuth.  Five fixed comparisons
select one of the six radial regions (pole cap plus five strips), and

```text
j = floor(16*phi/(2*pi)) mod 16
```

selects the base sector.  Fixed equality/select tests identify the target
quad and its four face neighbours.  Unaffected cells retain their ordinary
closed-form base address.

Inside the target, the exact logical-polar local coordinates are

```text
u = clamp((18 deg-e)/(8 deg), 0, 1),
v = clamp(wrap_[247.5,270](phi)-247.5 deg, 0, 22.5 deg)/(22.5 deg).
```

The unrefined diagonal is `u+v=1`.  For `p`-way refinement, let

```text
i   = min(p-1, floor(p*u)),
k   = min(p-1, floor(p*v)),
f_u = fract(p*u),
f_v = fract(p*v),
half = select(0,1,f_u+f_v < 1),
child = 2*(p*i+k) + half.
```

For a green neighbour, label the `p+1` points on its refined edge
`b_0,...,b_p` and use the opposite vertex `c` as a fan centre.  The `p`
triangles `(c,b_l,b_(l+1))` plus the remaining corner triangle triangulate
the neighbour.  At `p=4`, three oriented-line comparisons against
`(c,b_1)`, `(c,b_2)`, and `(c,b_3)`, plus one corner test, select the fan
triangle.  These are fixed selects, not a data-dependent search.  Rotating
this formula handles all four neighbours.

Every inserted boundary vertex is shared by both incident cells; there are
no T-junctions.  Candidate P's canonical one-dimensional edge trace is stored
bit-identically by the incident records, so their restrictions agree.  The
refined construction is therefore direction-`C0`; phase-`C0` remains the
unchanged barycentric sum of complete angular functions.  Reads remain three
per layer because refinement changes which record is addressed, not how many
phase vertices are sampled.

## 7. Decisive measurement

Topology, memory, and addressing are feasible.  Whether cell width is the
failed premise is empirical:

1. Replace the old slope-affine standing cell by the explicitly declared
   logical-polar quad, then split it into eight `p=2` child triangles.  The
   grazing control is generated in the same logical-polar chart.
2. For each child, score its vertices, the existing three interior/edge
   heldouts, all three exact phase sites, and the frozen `1--4.5 mm`
   translations at physical sigma4 and sigma16.
3. Globally optimise phase coefficients, then use only the preserved robust
   affine-rank-two oracle.  The converged coupled-L2 candidate is excluded
   because it was measured to lower mean loss while destroying tail metrics.
4. Score the unchanged grazing parent cells as a control.
5. Only if `p=2` is narrowly RED, repeat once at `p=4`.  Two RED subdivision
   levels park the track: local angular width is then not the failed premise
   under this codec and no quantisation, chord, bridge, runtime, or shader work
   follows.

A fully GREEN child result is constructive but local.  It authorises deriving
and gating the global `<=218`-cell witness above; it does not itself authorise
runtime transcription.

## 8. Exact horizon contract for any future global gate

`e=0` is the one-sided in-cover chart limit, not an exterior top-plane hit.
An exactly horizontal exterior ray fails the finite cover-box slab entry and
emits no cover.  For `e -> 0+`, a descending ray enters arbitrarily far away
and queries the one-sided profile; for every positive camera clearance its
screen footprint tends toward the phase-independent `M_inf` limit.  Cameras
arbitrarily close to the box can still expose the complete `0--2 deg` cell,
so this limit does not permit a `.25 deg` clamp.  A future global gate must
include an explicit logical-polar `0 deg` boundary and exact small-positive-
elevation truth at physically realisable footprints.

## 9. Targeted 2x result and blocker

The gate used the corrected logical-polar chart for cell vertices,
barycentric heldouts, and the direction-to-truth map.  It evaluated eight
standing child triangles and two unchanged-chart grazing controls: `83`
unique directions, of which `8` were exact-coordinate cache reuses and `75`
received new exact `128 x 128` BVH pages.  Phase coefficients were globally
optimised over the three exact phase sites and all frozen translations.  The
rank-two stage used the preserved threshold-normalised high-p robust oracle;
the rejected coupled-L2 tail destroyer was not run.

Evidence:

```text
data/work/groundcover-candidate-p-adaptive-direction/
  e3e0a4175b151b89/73102281018064a4/

report SHA-256:
  26e1cbfbff65cdf824ab057df0a867ab009965999d3dee0583bfbdf04edac031
```

Standing-child robust rank two:

```text
static GREEN                         0 / 144
translation GREEN                1887 / 2304
worst coverage p95                 0.135882  (limit 0.08)
worst premultiplied RGB p95        0.110044  (limit 0.06)
worst translation coverage p95    0.109981  (limit 0.06)
worst translation RGB p95         0.081773
worst connected exceedance        0.001709
worst normalised frozen gap       1.834069
```

The phase-only standing control passed all `144` static cases, but its worst
translation coverage p95 was `0.092438`; phase resolution itself therefore
also has a translation-tail limitation in this exact logical-polar sample.
The grazing robust control retained the pre-existing narrow colour tail
(`31/36` static, RGB p95 `0.071151`) while its phase-only control was entirely
GREEN.

The coherent track began at `20:11:24 +0300` and the final report was emitted
after `1944.88 s` (`32.41 min`), inside the preregistered `3600 s` budget.  The
final cache-only scoring run itself took `133.09 s`; the remainder includes
the math derivation, chart correction, and `75` new exact-BVH exports.

Halving both standing angular dimensions was not a sufficient lever: it left
every standing static case RED.  The `1.834x` gap is not narrow, so the
preregistered rule stops before `p=4`.  This is a bounded engineering result,
not an asymptotic theorem.  A continuous two-dimensional angular response has
local tangent rank at most two as cell diameter tends to zero, so this gate
does not prove that arbitrarily fine rank two cannot converge; it proves that
the authorised `p=2`, same-count attempt does not reach the frozen quality
limits, and that the preregistered evidence does not justify spending the
second subdivision.

Candidate P remains parked; no quantisation, chord, bridge, runtime, shader,
or additional subdivision work is authorised in the current track.  Resume
either with (a) separate authorisation for the documented same-allocation
`p=4`/250-cell witness and its mandatory full-domain controls, or (b) a
different angular function family whose local response dimension exceeds two
without increasing frozen runtime reads or resident memory, followed by a new
pure-math capacity proof.

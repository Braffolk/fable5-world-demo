# Support-bounded canonical plane record gate

Date: 2026-07-22

Decision: **REJECT AND PARK.** A conservative finite source-triangle support
eliminates the widening false-plane fans, but on the accepted Calamagrostis
mesh it eliminates every held-out live hit as well. This is a pure-math and
offline actual-asset result. No runtime or shader file was changed.

## 1. Question and fixed boundary

The rejected infinite-plane reconstruction contained one useful exact identity:
after an atlas record has selected the correct source triangle, intersecting its
plane with the exact live ray is projectively correct. Its visible failure was
that the selected microscopic triangle almost never owned the live line, while
the infinite plane continued indefinitely and produced stretched sheets.

This gate asks the strongest cheap finite version of that idea:

1. sample the four angular lattice corners at the nearest phase;
2. reconstruct the plane of each record's exact GCRP/v4 source triangle;
3. attach a conservative circular support which is guaranteed to remain inside
   that finite triangle for the ideal plane;
4. intersect the exact live ray with all four planes;
5. reject a candidate outside its support or source triangle;
6. elect the minimum positive accepted hit without blending owners.

The prospective shader remains statically fixed O(1): four records and one
winner-colour read, no loop, march, traversal, variable candidate list, shell,
pass, barrier, dispatch, or per-species query. A fixed `2x2` phase footprint is
measured separately at sixteen records and cannot rescue the primary method.

## 2. Exact record geometry

For a canonical texel-centre top origin

\[
O_i=(q_{i,x},H,q_{i,z}),
\]

canonical direction `d_i`, and source triangle vertices `A,B,C`, let

\[
n = \frac{(B-A)\times(C-A)}{\lVert(B-A)\times(C-A)\rVert}.
\]

The exact canonical point on that plane is

\[
\boxed{
P_i=O_i+d_i\frac{n\cdot(A-O_i)}{n\cdot d_i}
}.
\]

The record is invalid if the denominator is a true plane pole, the intersection
is behind the origin, or `P_i` is outside the finite source triangle. This last
condition matters because fixed-function raster ownership can include a small
edge cushion which does not intersect the independently quantised analytic
triangle.

For the exact live top origin `O(q)` and live direction `d`, the represented
plane intersection is

\[
\boxed{
t=\frac{n\cdot(P_i-O(q))}{n\cdot d},
\qquad
P=O(q)+td.
}.
\]

This is exact for the selected plane. It does not assert that the selected
triangle is the live first owner.

## 3. Conservative finite support

Let the barycentric coordinates of `P_i` in `ABC` be
`(lambda_A,lambda_B,lambda_C)`. The perpendicular distances from `P_i` to the
three triangle edges are

\[
r_A=\lambda_A\frac{2\mathcal A}{\lVert C-B\rVert},\quad
r_B=\lambda_B\frac{2\mathcal A}{\lVert C-A\rVert},\quad
r_C=\lambda_C\frac{2\mathcal A}{\lVert B-A\rVert},
\]

where `mathcal A` is triangle area. The certified ideal support is

\[
\boxed{
r_{safe}=\max(0,\min(r_A,r_B,r_C)-20\ \mu\mathrm m).
}.
\]

Every point in the selected triangle plane within that circle remains inside
the triangle. The live candidate therefore needs

\[
t\ge0,
\qquad
y_{min}\le P_y\le H,
\qquad
\lVert P-P_i\rVert\le r_{safe}.
\]

The evaluator additionally tests exact source-triangle containment after packed
plane reconstruction. This is an optimistic offline oracle, not an extra
runtime triangle fetch. It also reports any case where the packed support-only
test would leak outside the triangle.

## 4. Frozen eight-byte record and cost

Each plane/support record is two `u32` words:

| Field | Bits | Meaning |
|---|---:|---|
| hit Y | 16 | UNORM over accepted source Y bounds; `65535` reserved miss |
| geometric oct-normal X | 16 | UNORM16 |
| geometric oct-normal Y | 16 | UNORM16 |
| safe support | 16 | floor-quantised over `[0,64 mm]` |

The canonical point's XZ coordinates follow exactly from the texel-centre
origin, canonical direction, and decoded hit Y. The plane normal and point
define the live intersection. After a fixed four-way minimum, the winner atlas
coordinate supplies one separate premultiplied RGBA8 authored-colour read.
Complete records and owners are never numerically blended.

Primary cost:

- four plane reads, `4 * 8 = 32 B`;
- one winner-colour read, `4 B`;
- `36 B` total source traffic per query;
- four unrolled oct decodes, four ray-plane divisions, four squared support
  tests, and a fixed four-way minimum;
- approximately `120--150` scalar FMA-equivalent operations plus four
  divisions;
- candidate `t/valid` pairs may be reduced immediately, so four complete
  colours/normals do not remain live;
- no new synchronization, workgroup behavior, dispatch, downstream pass, or
  variable control flow.

The separate `2x2` phase variant is sixteen plane reads plus one colour read:
`132 B` and sixteen intersections. It is outside the primary traffic ceiling
and is measured only to show whether nearby spatial support could change the
conclusion.

## 5. Immutable source and test domain

Source:

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
SHA-256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
GCRP/v4
2,049,985 vertices
2,171,134 triangles
0.5199999809 m periodic tile
16 azimuths
15/35/55/75 degree elevation rows
```

The gate decodes the actual packed vertices, owner atlas, triangle records, and
authored colours. Exact periodic first-hit truth uses an analytic BVH over the
same quantised source mesh. Panicle versus structure follows the authored
triangle colour partition used by the existing Calamagrostis diagnostics.

Held-out grids contain `28x28` phases at each of:

- 5-degree grazing / 11.25-degree half-bin azimuth;
- 25-degree exact elevation and azimuth half-bin;
- 45-degree oblique half-bin;
- 65-degree high half-bin;
- 82.5-degree top view / half-bin azimuth.

A 192-frame oblique motion path uses 35-degree elevation and 11.25-degree
half-bin azimuth. A separate exact 0-degree azimuth / 35-degree elevation test
uses literal record-centre phases as a harness and packing control.

## 6. Result

### 6.1 Canonical control passes

The exact record-centre control proves that the plane solve, source decode,
periodic copy handling, support, and packed record are functioning:

| Path | Exact first-owner recall | Predicted / truth hits | False positives | Wrong owners |
|---|---:|---:|---:|---:|
| ideal f64 | 90.339% | 349 / 383 | 0 | 3 |
| packed 8-byte | 90.078% | 348 / 383 | 0 | 3 |

The packed path is within `0.261` percentage points of the f64 ideal. Its
ordinary accepted position error is microscopic: p50 `7.98 micrometres`, p95
`14.53 micrometres`; the three wrong events appear only at the distribution
maximum. The gate's control requirement is at least 85% exact-owner recall and
within one percentage point of ideal, so it passes.

### 6.2 Every held-out hit is rejected

Across the five noncanonical grids:

| Metric | Four packed reads | Four f64 ideal reads | Sixteen packed reads |
|---|---:|---:|---:|
| rays | 3,920 | 3,920 | 3,920 |
| truth hits | 1,979 | 1,979 | 1,979 |
| predicted hits | 0 | 0 | 0 |
| exact first-owner recall | 0% | 0% | 0% |
| panicle exact recall | 0% | 0% | 0% |
| structure exact recall | 0% | 0% | 0% |
| false positives | 0 | 0 | 0 |
| wrong-owner hits | 0 | 0 | 0 |
| false negatives | 1,979 | 1,979 | 1,979 |

This is not a marginal support threshold. Among 6,863 valid canonical charts,
safe radius is only:

- p50 `0.1283 mm`;
- p95 `1.1514 mm`;
- p99 `1.7181 mm`;
- maximum `2.1996 mm`.

The four-read live plane displacement is:

- p50 `0.3246 m`;
- p95 `1.1688 m`;
- p99 `2.4314 m`.

Median displacement is `2,308.7` times the packed support. At 45 degrees alone,
median displacement is `0.2418 m` against `0.1403 mm` median support, a factor
of `1,360`. Four neighbouring phase texels per direction do not help: sixteen
reads retain a `2,309` median displacement/support ratio and still accept zero
hits.

The 192-frame motion path contains 50 truth hits and 23 truth hit/miss
transitions. The reconstruction produces no hit in any frame and therefore no
plant motion or usable silhouette.

## 7. Practical visual verdict

Finite support does exactly one useful thing: it prevents the selected source
plane from widening into the stretched triangular fans seen in the live view.
The connected false-positive fan width is exactly zero on every tested grid;
the packed support-only test also has zero source-triangle leaks.

That safety is bought by complete disappearance:

- grazing false negatives form one connected component spanning the full
  `0.52 m` tile (`0.735 m` grid diagonal);
- the 25-degree grid's largest missing component spans the full `0.52 m`
  width;
- the 45-degree grid's largest missing component is `0.334 m` wide;
- 100% of panicle and structure truth events are omitted.

These false negatives are technically representable as zero premultiplied
coverage and do not relocate geometry. They are not an acceptable filtered
silhouette: they make the grass transparent. Premultiplied coverage can encode
the omission honestly; it cannot synthesize the missing first-event field.

The numbered QA makes the distinction explicit. Each image contains truth,
supported prediction, exact/wrong/missing classification, and false-negative
orange versus false-positive magenta. The prediction panels are empty; the
error panels contain only orange holes and no magenta fan.

## 8. Why this closes the path

The canonical atlas triangle is a microscopic tessellation primitive, not a
view-stable botanical surface chart. Half-bin angular motion carries the live
line centimetres or metres across the plane, while the finite primitive owns at
most millimetres around the baked hit. There is no radius choice which both:

1. remains conservative inside that source triangle; and
2. reaches the held-out live line.

Increasing the support recreates the rejected infinite-plane sheet. Keeping it
correct creates total transparency. A `2x2` phase footprint quadruples traffic
without changing the angular ownership mismatch. Therefore this representation
cannot become the missing PCF-like categorical rule through a different
threshold, pack, or fixed minimum.

## 9. Reproducibility and provenance

Tool:

```text
tools/groundcover-bake/analyze-support-bounded-plane-record.ts
SHA-256 45e020fe04e2e2d5fecac44aa85f150227f9c36561a875cdedc1a5ebb36b62a4
```

Artifact:

```text
data/work/groundcover-support-bounded-plane-record/
  2ed57f59d86e8376/5428b274dc9fed8f/
```

Report SHA-256:

```text
742fbe8a945a519fa6973097177268b5ffb59ff1113a2e122f7372300ae5e64c
```

Primary-source boundary:

- Sannikov supplies the low-cost repeated precomputed-ray objective and the
  evidence that naïve view interpolation needs a later categorical rule. This
  experiment is not attributed to his unpublished PCF-like implementation.
- Lin and Shum, *A Geometric Analysis of Light Field Rendering* (2004), supplies
  the geometry-assisted neighbouring-ray reconstruction and disocclusion
  context.
- The in-triangle radius certificate, eight-byte plane/support record,
  fixed-four election, packed actual-source test, and connected
  fan-versus-transparent-hole gate are LAAS derivations and measurements.

## 10. Decision and resume condition

**Reject and park.** Do not integrate this record, loosen its support, add
spatial taps, or use its zero-fan result as visual progress. No shader was
changed.

Resume only if the precompute can select a genuinely view-stable finite
botanical chart whose certified support covers held-out live lines, or if the
missing published PCF-like categorical rule becomes available. A microscopic
source-triangle plane is not such a chart.

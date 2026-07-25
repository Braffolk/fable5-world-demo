# Four-view canonical hit-height reconstruction gate

Date: 2026-07-22  
Decision: **rejected before shader integration**

This note isolates one narrow question left by the live Calamagrostis path:
can the four existing neighbouring precomputed views be recentered, carry only
their stored first-hit height, and place one of those events on the exact live
ray without the deleted raised-terrain query or the invalid projected-path
relift?  The answer on the actual checked-in GCRP is no.  The final placement
identity is exact, but the selected canonical first event is almost never the
live line's first event or even a point on its attached source triangle.

No renderer, WebGPU state, depth buffer, shader, mesh shell, or runtime source
was used or changed by this gate.

## 1. Frozen fixed-cost proposal

Let the live ray begin on the real botanical top plane:

\[
r_d(t)=O_H+t d,\qquad (O_H)_y=H,\qquad d_y<0.
\]

Choose the genuine botanical middle plane `y=y0`.  Its live line phase is

\[
Q=O_H+d\frac{y_0-H}{d_y}.
\]

For a canonical direction `c_i`, address the existing top-plane atlas at

\[
O_i=Q-c_i\frac{y_0-H}{(c_i)_y}.
\]

The complete stored event gives canonical distance `tau_i`, categorical
hit/miss, owner, normal, and colour.  Its hit height is

\[
y_i=H+(c_i)_y\tau_i.
\]

There is exactly one way to place that height on the exact live ray:

\[
\boxed{t_i=\frac{y_i-H}{d_y}=\frac{H-y_i}{-d_y}},\qquad
\boxed{P_i=O_H+t_i d}.
\]

This is not projected-path relift: no horizontal canonical path length is
treated as though it belonged to `d`.  It is also not the deleted translated
or raised terrain query: `O_H` is the real top-plane origin and `H` is only the
asset's fixed coordinate datum.

The four angular records were frozen to one of these constant-size rules:

- arithmetic height blend, included specifically to measure fabricated sheets;
- frontmost complete record;
- weighted-median complete record;
- maximum-weight complete record;
- an unimplementable oracle that may inspect truth and choose the closest of
  the same four records, included only as an upper bound.

The implementable upper cost is four eight-byte complete-record reads
(`32 B/query`), one winner-colour read, one fixed four-input sorting network of
five compare/select stages, and one division.  It adds no loop, march,
candidate list, pass, binding, barrier, dispatch, or per-species query.  The
oracle is not part of this cost and is not proposed for runtime.

## 2. Why exact height placement is not exact geometry

Write the canonical stored event as

\[
e_i=(\ell_i,\tau_i,m_i),
\]

where `ell_i` is its canonical oriented line and `m_i` is its categorical
owner.  The identity above constructs a point on the live line with the same
height as `e_i`; it does **not** prove any of the following:

\[
P_i\in T_{m_i},\qquad m_i=m_d,\qquad t_i=t_d,
\]

where `T_mi` is the attached source triangle and `(m_d,t_d)` is the live first
surface event.  These properties would require correspondence between first
events on two different oriented lines.  Hit height alone does not contain
that correspondence.

Arithmetic blending is worse.  If selected records have distinct owners, then

\[
\bar y=\frac{\sum_i w_i y_i}{\sum_i w_i}
\]

usually belongs to none of their triangles.  Its live point is a new sheet
between unrelated plants.  A categorical selector avoids inventing an
intermediate owner, but it can only switch between the same unrelated events.
Therefore it cannot create the missing live owner.

The domain also has two independent holes:

1. at exact horizontal view `d_y=0`, neither a finite top-plane phase nor
   `t=(H-y_i)/(-d_y)` exists;
2. after moving a camera beyond an exterior first hit, four top-entry first
   records do not encode the next pointed-line successor.

These are information deficits, not numerical conditioning problems.

## 3. Actual-bake gate

The validator uses the exact checked-in production asset:

- `src/assets/groundcover/calamagrostis-canescens.gcrp`;
- SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`;
- `2,049,985` vertices, `2,171,134` triangles;
- actual `0.52 m` periodic tile and `1.1502 m` decoded botanical height band.

Exact truth is analytic periodic BVH intersection in f64.  The source BVH is
offline truth only; it is not a runtime proposal.

Before any held-out comparison, 1,024 exact canonical texel-centre controls
validate the decoder and coordinate system.  They reach `0.99523` silhouette
IoU, `0.0000331 m` depth p95, `0.98321` exact-owner agreement, and `0.97847`
attached-triangle agreement.  Thus the held-out failure is not a sign error,
wrong depth encoding, periodic-copy mismatch, or broken atlas decoder.

Held-out exterior truth uses 1,024 rays at half-bin azimuths, an `8x8` phase
grid, and `5/18/45/82.5` degree elevations.  Results are:

| rule | silhouette IoU | depth p95 | exact owner among true positives | attached source triangle |
|---|---:|---:|---:|---:|
| height blend | 0.63864 | 8.466 m | 0 | 0 |
| front complete | 0.63864 | 8.312 m | 0.00178 | 0.00120 |
| weighted median complete | 0.63864 | 8.918 m | 0.00178 | 0.00120 |
| maximum weight complete | 0.58127 | 9.188 m | 0.00447 | 0.00330 |
| truth-assisted oracle over the same four | 0.93569 | 5.993 m | 0.00344 | 0.00337 |

The exact live owner exists among the four records for only `0.003279` of true
held-out hits.  The height blend fabricates an intermediate depth and mixes
owners on `0.64543` of all predicted hits; it mixes hit and miss support on
`0.71755`.

At 2 mm camera steps, p95 error in the predicted depth delta is
`6.94/2.78/0.362 m` for the height blend at `5/18/45` degrees.  Even the
truth-assisted oracle is `4.07/1.47/0.0699 m`; the required limit is `0.02 m`.
This is the CPU form of the visible swimming and distance-growing streaks.

For 79 near-inside seeds made by advancing 2 mm past an exact exterior first
hit, 48 have a real forward successor.  Thirteen of those successors remain
after every stored first event is already behind the new origin.  The best
oracle over the four old events has only `0.72917` successor recall and
`2.049 m` depth p95.  At exact horizontal middle-plane origins, 48 of 64 lines
have a successor within 8 m, while the top-height decoder is undefined and has
zero recall.

Acceptance was frozen at silhouette IoU at least `0.95`, depth p95 at most
`0.02 m`, exact-owner and attached-triangle agreement at least `0.95`, motion
delta p95 at most `0.02 m`, and near-inside successor recall at least `0.95`.
Every implementation candidate fails multiple independent gates; the oracle
also fails.

## 4. Decision and implementation boundary

**Reject four-view hit-height reconstruction.**  Do not map this experiment
back into `NaniteGrass.ts`, do not enable the dormant four-view branch, and do
not treat its cheap cost as permission to ship a geometrically false sheet.
Adding a fifth low-elevation row can reduce angular spacing but cannot repair
the missing event/owner correspondence, exact-horizontal domain, or
camera-inside successor coordinate.

The reusable positive result is narrow: recentering at a real middle plane and
placing one known hit height on the exact live line are correct identities.
They become implementation-ready only if a future representation supplies the
correct live event/owner rather than asking four unrelated canonical first
events to stand in for it.

Reproducible artifact:

`data/work/groundcover-four-view-height-gate/2ed57f59d86e8376/8d1465c218cd38a3/report.json`

Validator:

`tools/groundcover-bake/analyze-four-view-height-reconstruction.ts`

Tool SHA-256:
`2af423916e87c61055ce13772b7e895da86294073c145304f0d81a6cf3fd1619`  
Report SHA-256:
`6711f703e6f34cb896b256ea49a451656003079a8f2fa239c88ac08b6195be56`


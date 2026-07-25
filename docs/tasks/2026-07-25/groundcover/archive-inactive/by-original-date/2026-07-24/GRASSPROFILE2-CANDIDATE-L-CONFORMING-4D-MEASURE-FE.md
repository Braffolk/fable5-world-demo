# Candidate L: conforming 4D foreground-measure finite elements

Date: 2026-07-24  
Status: **RED / parked at the slope-rate gate; no runtime authorisation.**
See `GRASSPROFILE2-CANDIDATE-L-SLOPE-RATE-RED-BLOCKER.md`.
The subsequent exact-BVH targeted 4D rate gate independently confirmed the
RED verdict; see `GRASSPROFILE2-CANDIDATE-L-TARGETED-RATE-GATE.md`.

## 1. Why another representation is required

The corrected Candidate-K held-out gate proves that the remaining failure is
not four-bit packing.  A filtered first-hit field is a function of two periodic
phase coordinates and two view-direction coordinates:

\[
M(q,d)=(A(q,d),A(q,d)C(q,d)).
\]

Candidate K interpolated the two direction coordinates but held `q` at one
nearest texel.  That is not a conforming approximation of the four-dimensional
field.  A feature at height `y` traces the epipolar shear

\[
\Delta q=(y-h_{ref})\Delta\left({d_{xz}\over -d_y}\right),
\]

so an angular midpoint generally corresponds to a shifted spatial event, not
to the average of two endpoint colours at the same phase.  The measured result
is consequently exact at stored directions and catastrophically wrong between
them.

Candidate L represents the complete filtered field as one conforming
piecewise-affine function on `T^2 x H^2`.  It uses five vertex reads because a
simplex in four dimensions has five vertices.  This is a replacement for the
far/unresolved query, not five reads added to the existing nine.

## 2. Domain and positive value

The spatial domain is the periodic phase torus `Q=T^2`.  The directional domain
`D` is the descending exterior hemisphere, parameterized primarily by slope

\[
s={d_{xz}\over-d_y}.
\]

Slope is the correct local coordinate because the phase shear above is affine
in `s`.  Elevation-angle-uniform rings are not presumed.  The cook builds a
conforming triangular mesh over a compactified slope disk, including periodic
azimuth seams and a unique vertical pole.  Its refinement/ring law is selected
from the measured angular-convergence curve, not from the old 15/35/55/75
degree rows.

At each spatial footprint level `r_l`, the value at a four-dimensional vertex
is only the positive foreground measure

\[
M_l=(A_l,P_l),\qquad P_l=A_l C_l.
\]

It contains no filtered depth, representative position, normal, owner, or
triangle plane.  Fractional resolve retains the real background sample and
uses `C=P+(1-A)C_b`.

## 3. Product-simplex construction

Triangulate the phase torus into spatial triangles and the slope disk into
direction triangles.  The Cartesian product of one spatial triangle and one
direction triangle is four-dimensional.  Its standard staircase
triangulation contains six conforming 4-simplices; each 4-simplex has exactly
five product vertices `(q_i,d_j)`.

Given live `(q,d)`, fixed comparison/sort ALU selects one of those six
4-simplices and produces five non-negative barycentric weights

\[
b_k\ge0,\qquad \sum_{k=0}^{4}b_k=1.
\]

The reconstructed measure is

\[
M_l(q,d)=\sum_{k=0}^{4}b_k M_l(q_k,d_k).
\]

The selector has a closed fixed form.  Give each local 2-simplex an ordered
vertex basis and write its cumulative barycentric coordinates as

\[
1\ge x_1\ge x_2\ge0,\qquad 1\ge y_1\ge y_2\ge0.
\]

Sort the four scalars `(x1,x2,y1,y2)` with a fixed four-element compare network
while retaining whether each step belongs to the spatial or directional
triangle.  The six legal interleavings are exactly the six staircase
4-simplices.  If the sorted values are
`z0 >= z1 >= z2 >= z3`, their five weights are

\[
(1-z_0),\ (z_0-z_1),\ (z_1-z_2),\ (z_2-z_3),\ z_3.
\]

The five product vertices are the monotone path obtained by advancing the
spatial or directional triangle index recorded by each sorted step.  Hence
point location is a bounded compare/select network, not a search or traversal.
Adjacent product cells must use one globally consistent vertex ordering (a
pulling/staircase triangulation); arbitrary per-cell triangle order would break
the shared-face proof.

Every mesh vertex is cooked and quantized once, and its exact bits are shared
by every incident 4-simplex, spatial tile seam, angular triangle, azimuth seam,
and pole cell.  Standard finite-element assembly therefore gives a globally
`C0` function in phase and direction.  There is no nearest phase cell, maximum
direction, categorical row, or independent cross-owner blend in the filtered
branch.

The interpolation is of a positive footprint measure.  It does not interpolate
five geometric records or claim that unrelated first-hit owners form one
surface.

## 4. Scale without an LOD ring

All scale endpoints use the same spatial/directional vertex complex.  A selected
scale-interval record stores two adjacent endpoint codes:

```text
low 16 bits:  M_l vertex as premultiplied RGBA4444
high 16 bits: M_(l+1) vertex as premultiplied RGBA4444
```

Five `R32Uint` vertex loads therefore return both endpoints.  Interpolate the
five low values, interpolate the five high values, then blend by the continuous
log-footprint fraction.  The high endpoint in interval `l` is copied bit for
bit as the low endpoint of interval `l+1`.  Because both intervals use the same
conforming 4D basis, their limits are identical over the whole domain, not only
at texel centres:

\[
\lim_{\beta\to1}M_l(q,d,\beta)
=M_{l+1}(q,d)
=\lim_{\beta\to0}M_{l+1}(q,d,\beta).
\]

Thus spatial-cell, angular-cell, azimuth, pole, and footprint-level value
continuity are structural.  Candidate K2's incompatible parent-resolution copy
is absent: every footprint field is sampled on the same vertex complex.

## 5. Fixed operation schedule

The analytic footprint is known before profile loads, so the shader selects one
of three compile-time query shapes:

```text
resolved:     Candidate-F 4 R0 + 4 R1 + 1 colour = 9 reads
bridge:       4 R0 geometry + 5 Candidate-L vertices = 9 reads
unresolved:   5 Candidate-L vertices                = 5 reads
```

The bridge exists only over the sub-pixel exact/filtered handoff.  It does not
pretend that the uncorrected R0 surface is exact: its categorical depth is a
bounded transition aid, while appearance comes from the conforming filtered
measure.  The unresolved branch preserves background depth/id.  A hard branch
between these shapes is chosen from the continuous analytic footprint before
any dependent record; it never raises worst-case profile traffic above nine.

There is no loop, march, candidate list, runtime geometry, per-species work,
new pass, dispatch, barrier, or screen buffer.  Direction/spatial simplex
selection is fixed ALU.  Branch coherence and register lifetime still require
a real GPU trace after visual acceptance.

## 6. Allocation law

Let `V_Q` be the number of periodic spatial vertices, `V_D` the number of
direction vertices, and `L` the number of stored footprint intervals.  The
scale allocation is exactly

\[
B_{scale}=4 V_Q V_D L\quad\hbox{bytes}.
\]

The binding ceiling is

```text
B_scale <= 32.5 MiB,
near + scale <= 48.75 MiB.
```

Example capacity points, before choosing the convergence-measured mesh, are

```text
V_D=257,  V_Q=64^2, L=6 -> 24.094 MiB
V_D=513,  V_Q=48^2, L=6 -> 27.053 MiB
V_D=769,  V_Q=44^2, L=5 -> 28.396 MiB
```

These are capacity examples, not approved lattices.  The angular convergence
audit determines whether any allocation under the ceiling can meet the rate.
If not, Candidate L is RED; lowering the phase rate until the plant becomes a
blur is not a pass.

## 7. Exterior horizon and adaptive slope mesh

The slope domain is unbounded as `d_y -> 0-`.  No finite mesh can give a uniform
fine-scale approximation to every arbitrary periodic soup at infinite slope.
The relevant representation is scale coupled:

- progressively shallower resolved views are measured directly until their
  required slope density exceeds the fixed budget;
- coarser footprint fields may converge to their periodic-cell measure and use
  a compactified fringe;
- an exactly horizontal ray above the finite cover box misses and is not
  queried;
- no positive elevation is silently clamped to an unrelated stored view.

The cook's rate law is expressed in slope and reference-plane phase error.  For
vertical support radius

\[
R_y=\max_X|X_y-h_{ref}|,
\]

a sufficient local sampling condition for a filtered field of effective phase
bandwidth `w_l` is

\[
R_y\,\|\Delta s\|\lesssim w_l.
\]

This is a sizing law, not a fidelity proof: visibility births can be narrower
than that bound.  Held-out exact-BVH truth remains binding.

## 8. Terrain, wind, and multi-species semantics

On affine terrain and under the approved global affine-in-height wind shear,
the world oriented ray is conjugated into the authored frame before computing
`q`, `s`, footprint, and simplex coordinates.  The same inverse affine map is
used everywhere.  This is exact.  Curved terrain uses the Candidate-F local
tangent carrier and retains its written second-order curvature/incidence bound;
uphill truth is a separate gate.

The cook raycasts the already-overlapped whole community.  First-hit colour and
coverage already include all species and cover types, so runtime cost does not
depend on species count.  A different community receives its own cooked field
and must pass its own rate gate.  Moss, plumes, stems, and overlapping species
are not queried as runtime candidates.

## 9. Lighting

The filtered value still has no single surface normal.  Candidate K's
view-conditioned orientation-distribution response remains the applicable
lighting object and consumes no profile read.  Exact and bridge geometry may
use geometric face normals only for their categorical depth branch; those
normals never enter the filtered measure or its interpolation.

## 10. Falsifiable gates

Before any runtime edit:

1. Measure midpoint/quarter-site convergence in slope coordinates on the real
   GCRP at multiple fixed physical footprints.  Fit no lattice using held-out
   sites.
2. Cook a budget-valid conforming 4D mesh.  Compare unlimited precision first,
   then RGBA4444, per direction and scale against exact-BVH positive-measure
   truth.
3. Sweep every spatial and angular simplex boundary, azimuth seam, pole, and
   footprint interval `+/-epsilon`; values must be `C0` to quantized equality.
4. Sweep `1--4.5 mm` camera/world translations and report temporal p95/p99 and
   connected regions.  There is no exemption for a world-locked FE edge.
5. Gate the exact/bridge/unresolved handoff with moving camera, TAA, AO, shadow,
   cover edge, plume/stem, standing, uphill, and low-oblique cases.
6. Compare the view-conditioned NDF to filtered rendered truth under frozen
   light directions.

Primary filtered-measure limits remain

```text
coverage absolute error:       p95 <= .08, p99 <= .20
premul RGB max-channel error:   p95 <= .06, p99 <= .15
connected p99 exceedance:       < 1%
```

One diagnose/fix cycle is allowed if the first complete conforming cook is RED.
The premise audit checks slope coordinates, simplex assembly, physical filter
support, and allocation-law compliance.  A second genuine RED parks the track;
it does not authorize more than nine reads or 48.75 MiB.

## 11. Claim boundary

Candidate L solves the structural discontinuities which K/K2 retained: its
filtered field is conforming in both phase and direction, and its scale
transition uses the same vertex basis.  It does not prove that a fixed-rate
mesh approximates every arbitrary soup.  Arbitrarily narrow visibility events
can exceed any finite rate; therefore cook-time held-out fidelity is part of
the representation contract.

The question now is singular and measurable: **does the accepted Calamagrostis
community's filtered 4D measure fit the stated 32.5-MiB vertex budget at the
quality required by the live 2--5 m exterior views?**  No shader work begins
until the answer is GREEN and Fable has reviewed this complete mathematical
boundary.

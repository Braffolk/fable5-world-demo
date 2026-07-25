# Four affine slab extrusions: exact math and actual-source rejection

## Decision

**Reject and park after one attempt.** A fixed union of four clipped parallel
mask extrusions is mathematically exact and genuinely O(1), but it is not a
faithful geometry class for the accepted Calamagrostis. On conservative
high-detail masks it produces filled slabs, horizontal band seams, wrong
owners, and loss of the fluffy panicle silhouette. No runtime or shader file
was changed.

The tested representation is distinct from the rejected primitive-chart token
carrier. It replaces the source by four global analytic sweep families; it
does not select or evaluate source primitives at runtime.

## 1. Exact affine band theorem

For band `i`, let `M_i` be an arbitrarily detailed periodic solid mask in
canonical coordinates `q=(u,v)`, let `h` be the extrusion coordinate, and let

\[
C_i=\{(u,h,v):(u,v)\in M_i,\ a_i\le h\le b_i\}.
\]

An invertible affine chart maps it into world space,

\[
x=F_i(y)=A_i y+c_i,\qquad y=(u,h,v)^T.
\]

For a pointed world ray `x(t)=o+td`, `t>=0`, transform once:

\[
y(t)=y_0+t w,\qquad y_0=A_i^{-1}(o-c_i),\quad w=A_i^{-1}d.
\]

Write `w=(w_q,w_h)` with `w_q=(w_u,w_v)`. Intersect the ray with the
finite axial interval. For `w_h != 0`,

\[
t_a=(a_i-h_0)/w_h,\quad t_b=(b_i-h_0)/w_h,
\]

\[
t_{in}=\max(0,\min(t_a,t_b)),\qquad
t_{out}=\max(t_a,t_b).
\]

The band is missed when `t_out<t_in`. If `w_h=0`, it is missed when
`h_0` is outside `[a_i,b_i]`; otherwise its axial interval is `[0,+infinity)`.
This is clipped slab entry only. No slab surface is rendered by the carrier.

At `t_in`, address the mask at

\[
q_{in}=q_0+t_{in}w_q,
\qquad s=\lVert w_q\rVert,
\qquad \omega=w_q/s.
\]

For `s>0`, let the precomputed directed 2D successor field return the first
mask-boundary distance `rho(q_in,omega)` and one complete authored record. The
live hit is

\[
\boxed{t_*=t_{in}+\rho/s},
\]

accepted only when `t_*<=t_out`. This is Sannikov's projected-path equation in
the honest affine derivative basis. Live elevation is absent from the field
address because membership in `M_i` is invariant along `h`; `w_h` only clips
the admissible axial interval. There is no canonical-elevation substitution.

### Camera-inside successor

The directed mask record must distinguish outside-to-entry from inside-to-exit.
If the camera starts inside `M_i x [a_i,b_i]`, the next event is the earlier of
the directed mask exit and the forward axial end. An exterior-only first-entry
record is insufficient. This is still one fixed record per band, provided the
inside/outside state, successor distance, normal, colour, and mark are stored
together.

Finite **closed** extrusions add end caps over all of `M_i`. Entering such a
cap is exact for that geometry, but those horizontal sheets are not present in
the accepted plant. Omitting the caps avoids sheets but makes an axis-parallel
ray unable to see the finite band's top/bottom silhouette and breaks a closed
camera-inside successor. The actual-source gate measures the closed, more
complete alternative and exposes the artificial caps explicitly.

### Exact limiting directions

- `w_h=0`: the ray is parallel to the band boundaries. If its height is in the
  band, `rho/||w_q||` is exact with no height clamp or pole.
- `w_q=0`: the ray is parallel to the extrusion generator. Mask phase is
  constant. If it is outside `M_i`, the band misses; if it is inside, the next
  event is the appropriate axial cap. A zero-direction occupancy/cap record is
  required; dividing by a fabricated epsilon is not valid.
- As `||w_q||` tends to zero, side distance tends to infinity and the cap wins.
  As `w_h` tends to zero, the finite axial bound recedes and the mask event
  wins. These are geometric limits, not hidden camera exclusions.

For a vertical extrusion axis, ordinary world-horizontal rays are the easy
`w_h=0` case and world-vertical rays are the explicit `w_q=0` cap case. Under a
skew chart the canonical components, not world labels, govern the limits.

### Normal transport

A side-mask normal is the canonical covector `n_y=(n_u,0,n_v)`. A cap normal is
`(0,+/-1,0)`. The exact world normal is

\[
\boxed{n_x=\operatorname{normalize}(A_i^{-T}n_y)}.
\]

Forward-transforming a normal with `A_i` is only valid for an orthogonal chart.

## 2. Why a global projective taper is not periodic ground cover

A projective map does preserve rays. If homogeneous `H_i^{-1}` maps the live
ray to `z(t)=p+t r`, dehomogenisation gives a canonical line with a fractional
linear parameter. Around datum `t_0`, with homogeneous denominator
`W(t)=W_0+(t-t_0)r_w`, choose the canonical derivative at `t_0`. A returned
canonical line parameter `lambda` lifts exactly as

\[
t-t_0=\frac{\lambda W_0}{W_0-\lambda r_w}.
\]

The point-dependent Jacobian also transports normals exactly:

\[
J_F(y)=\frac{B-xc^T}{c^Ty+\delta},\qquad
n_x=\operatorname{normalize}(J_F(y)^{-T}n_y).
\]

That algebra does **not** make a non-affine global taper a valid repeating
ground-cover carrier. A canonical period translation `T_lambda` becomes

\[
H_i T_\lambda H_i^{-1}.
\]

For ordinary Euclidean world tiling this conjugate must itself be the same
world translation at every height and phase. A non-affine projective map moves
the plane at infinity, so the conjugate is generally a projective motion whose
displacement and scale depend on position. Equivalently,
`F(y+lambda)-F(y)` is not constant. A full two-dimensional translation lattice
can remain a Euclidean translation lattice only when `H_i` preserves the plane
at infinity, which reduces the chart to an affine map for this purpose.

Resetting a taper independently in each tile does not repair the theorem: a
world ray crossing tile copies then has a different inverse chart per copy, so
one periodic mask-ray query no longer returns the global first event. The
projective option is therefore rejected before the geometry gate. Affine lean
and skew remain exactly periodic because

\[
F_i(y+(\lambda_u,0,\lambda_v))
=F_i(y)+A_i(\lambda_u,0,\lambda_v),
\]

which is a constant world translation.

## 3. One bounded actual-Calamagrostis gate

Tool: `tools/groundcover-bake/analyze-four-slab-extrusion.ts`, SHA-256
`0438b8b9629b4755b9fad4df31970ceeff92ee50987d6343f76cb1d14c930b29`.

Immutable source: `src/assets/groundcover/calamagrostis-canescens.gcrp`,
SHA-256
`2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`,
with `2,049,985` vertices and `2,171,134` triangles.

Artifact:
`data/work/groundcover-four-slab-extrusion/2ed57f59d86e8376/db4432b7634eb595/`.
Recipe SHA-256 is
`db4432b7634eb595373de8a665255ec6767f7bb42122e254cf6433b61b26ace6`;
`report.json` SHA-256 is
`86c2f847b6f3bd490df0d3099b203c5a851a5157d8e1ac219059ebabd8ef6892`.
The machine index binds the seven numbered PNG hashes.

The source was split at normalized heights `0, .45, .70, .86, 1`; the upper
two bands isolate the panicle-bearing regime. For each band the tool clipped
every source triangle, fit one shared affine lean limited to the authored
botanical cone, and conservatively projected the clipped surfaces into a
`512x512` mask. This is an optimistic preservation gate: it retains thin hairs,
and owner agreement passes when the truth family is merely *among* all
families projected into the winning mask cell.

Each numbered image renders analytic source-triangle truth on the left and the
closed four-slab union on the right. The result is unambiguous:

| elevation | silhouette IoU | candidate overfill | truth loss | optimistic owner recall | panicle retention | artificial cap hits |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 deg | 0.1893 | 80.26% | 17.90% | 63.80% | 48.23% | 0.12% |
| 1 deg | 0.1822 | 80.99% | 18.61% | 65.16% | 47.88% | 0.24% |
| 5 deg | 0.1850 | 80.78% | 16.88% | 64.98% | 53.10% | 1.33% |
| 15 deg | 0.1822 | 80.85% | 21.03% | 62.28% | 64.81% | 5.20% |
| 35 deg | 0.1718 | 81.83% | 24.23% | 62.92% | 71.33% | 12.21% |
| 75 deg | 0.1925 | 80.52% | 5.76% | 70.28% | 98.26% | 25.61% |
| 90 deg | 0.2167 | 78.07% | 5.19% | 62.12% | 99.05% | 31.86% |

Required gates were IoU at least `0.95`, overfill/loss at most `0.03`, owner
recall and panicle retention at least `0.95`, and cap fraction at most `0.01`.
Every elevation fails several gates. Side views turn each detailed panicle
projection into a filled axial plank; top views retain panicle coverage only by
exposing artificial horizontal caps. Fewer, more compact panicle masks would
reduce overfill only by deleting the upward branches and hairs already lost in
the low-elevation retention numbers.

## 4. Complete fixed-cost budget

One `rg32uint` complete record can pack hit/successor distance, oct normal,
direct quantized colour, and botanical/community mark without a second
attribute texture. Four bands, `220x220` interiors, one-texel wrapped gutters,
24 azimuth slices, eight bytes per record, and a full `4/3` footprint pyramid
cost

\[
4\cdot222^2\cdot24\cdot8\cdot4/3
=50,466,816\text{ bytes}.
\]

This leaves only `654,336` bytes under the `51,121,152`-byte cap. Live work is
four predetermined complete-record reads, four affine slab/projected-path
solves, fixed min-selection, and less than the 256-FMA ceiling. One texture
array binding suffices. There is no runtime loop, march, traversal, candidate
tail, barrier, extra pass, or distance-dependent work. Register live ranges are
four decoded candidates plus the current minimum; a branchless validity/min
reduction avoids warp divergence. All overlapping species, moss, litter, and
population marks would have to be unioned into the same four offline masks, so
species count does not multiply runtime cost.

Cost therefore passes only barely; geometry fails by a very large margin.
Adding bands is both near-linearly more expensive and explicitly outside this
gate. It would recreate a shell stack rather than solve the representation.

## 5. Provenance boundary and resume condition

Sannikov supplies the exact parallel-extrusion projected-path theorem and the
copy-count-independent periodic sampler objective. The clipped affine-slab
extension, pointed-line successor/cap analysis, projective-quotient rejection,
conservative actual-source mask gate, and direct complete-record budget are
LAAS derivations and measurements. No claim is made about Sannikov's
unpublished later interpolation.

Resume only if a new analytic geometry class shares detailed marked events
across live rays without filling height intervals, remains a Euclidean periodic
community under its chart action, proves arbitrary camera-inside successors,
and stays within four complete reads, 256 FMAs, and the unchanged resident cap.
Do not add more slab layers, per-species queries, live primitives, candidates,
loops, marches, or a larger memory ceiling.

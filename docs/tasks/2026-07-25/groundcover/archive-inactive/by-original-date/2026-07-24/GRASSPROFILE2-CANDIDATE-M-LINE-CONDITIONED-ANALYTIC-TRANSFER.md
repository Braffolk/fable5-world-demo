# Candidate M: line-conditioned analytic transfer field

Date: 2026-07-24  
Status: **periodic-interior fidelity subgate RED; parked; no runtime authorisation**

## 1. Reason for the change of representation

Candidates H--L tried to store, interpolate, or factor the filtered answer as a
function of two phase and two view-direction coordinates.  Candidate L's two
independent gates close that family under the frozen resource envelope:

- a conforming 4D simplex is perfectly continuous but fails the real source at
  essentially the same magnitude as the old same-phase interpolation;
- even the stronger nine-vertex tensor ceiling fails identically; and
- the measured slope rate is `24--156` times beyond the rate carried by the
  admissible allocations, before the unbounded horizon fringe.

The missing reduction is botanical and geometric.  A ray intersects one
forward interval of a locally affine clipped cover volume.  Its horizontal
azimuth selects one oriented line through the periodic community; its
elevation and actual entry height change the analytic height trajectory along
that same line.  Store a compact positive model of the oriented line, not
independent elevation images.  The earlier top-plane-only statement was
wrong: it excluded side-entry and uphill rays and had no exact horizontal
limit.  Sections 2--4 now use the actual forward entry point instead.

Candidate M replaces only the unresolved/filtered branch.  Candidate F's
same-face plane algebra remains the resolved branch, but its record is extended
with the missing finite support certificate in Section 5.  Candidate H's real-
background fractional overlay remains the resolve law.

## 2. Coordinates and exterior domain

Work in the authored local-affine terrain frame, after applying the same
inverse terrain and affine-wind conjugation to the camera ray and cover box.
Let the finite vertical slab be `0 <= h <= H`.  The quality contract excludes
a camera inside the complete local cover envelope; that case fades before the
query.  Cameras between `0` and `H` but horizontally outside the support remain
in contract and enter through a side.

Let the forward ray be `x(t)=o+t d`, `t>=0`, with unit `d`.  Intersect, in a
fixed number of scalar slab operations,

```text
[0,t_background],
0 <= h(x(t)) <= H,
W_0 + grad(W)_0 dot (x(t).xz-x_0.xz) >= 0.
```

The last inequality is the existing world-cover control field in its local
affine chart.  Their intersection is one closed forward interval `[t_a,t_b]`
or MISS.  Define the actual entry point, its phase, and horizontal arclength by

```text
Y_a = x(t_a),       q = Y_a.xz,
s_p = length(d_xz),
omega = d_xz/s_p,  lambda = -d_y/s_p,          (s_p > 0),
rho = s_p (t-t_a), R = s_p (t_b-t_a),
h(rho) = h_a-lambda rho,  h_a=Y_a.y.
```

Thus `q` is always a finite forward-entry phase.  A top-entering ray, a
side-entering ray, and a slightly uphill ray differ only in the computed
`t_a`, `h_a`, and signed `lambda`.  At the vertical pole (`s_p=0`) Section 4
integrates directly in `t`.  An exactly horizontal ray is ordinary:
`s_p=1`, `lambda=0`, and the same finite side-support interval is used.  No
positive elevation is clamped to a stored direction, and no point behind the
camera contributes.

### 2.1 World support and cover edges

The periodic profile is clipped by the existing world ground-cover control
field.  Evaluating that control only at the background terrain point cannot
produce a side silhouette: a ray may pass through covered ground before ending
on uncovered terrain.  Candidate M instead requires the existing guide's
continuous control scalar `W` and analytic gradient to form the local-affine
support

```text
W(q_g + x) = W_g + grad(W)_g dot x.
```

Substituting the live ray produces one scalar affine inequality in `t`; it is
already included in `[t_a,t_b]` above.  Both a background point inside cover
and a background point just outside whose ray crosses the support edge are
therefore handled by the same forward interval.  This adds no *profile* read.
Before runtime authorisation the full shader read must verify that `W` and its
derivative are already resident in this path; otherwise the claimed binding
and traffic budget is RED.  The cook/live gate measures the local-affine
remainder `0.5 ||Hess W|| |d_xz|^2 (t_b-t_a)^2`.  A promised ray segment with
more than one material support interval is outside this one-interval model and
is RED rather than silently filled.  This limitation concerns the world
distribution of the repeating community, not the arbitrary triangle soup
inside one community tile.

## 3. Positive four-mode line model

For each profile, cook four nonnegative semantic/structural modes.  A mode has
profile-global constants

```text
(b_k, gamma_k, c_k),
b_k >= 0, gamma_k >= 0, c_k in [0,1]^3,
```

and two nonnegative amplitude fields

```text
B_k(q)             vertical-pole/base amplitude,
A_k(q,omega)       oriented-line residual amplitude.
```

`B_k` is cooked bit-identically at every azimuth vertex for a fixed phase.
The profile-global colours
are a nonnegative spectral basis chosen by the cook; species count never enters
the runtime query.  For Calamagrostis the expected basis contains low green
structure, high green structure, violet plume, and pale reproductive detail.

Let the absolute analytic height envelope be

```text
g_k(h) = exp[-gamma_k (H-h)].
```

The world extinction/source surrogate along the live interval is

```text
kappa_k(rho; q,omega,lambda)
  = [B_k(q) + s_p A_k(q,omega) exp(-b_k rho)]
    g_k(h_a) exp(-gamma_k lambda rho).
```

The `g_k(h_a)` factor is load-bearing: without it, a side entry would reset the
height profile as though it had entered at the top.  Signed `lambda` supports
both descending and rising chart rays.  The stable exponential primitive in
Section 4 therefore accepts signed `z`; the finite cover interval keeps every
result finite.

This is not claimed to be the original triangle soup.  It is the positive,
direction-analytic visual surrogate to which that soup is compiled at the
unresolved footprint.  The representation is general at the cook boundary:
any finite triangle soup can supply the line-transfer truth and receive an
explicit fit residual.  Reconstruction of the compiled surrogate is closed
form; visual fidelity to the source is empirical and separately gated.

The `b=0` modes carry the stationary repeated-community tail.  The `b>0`
modes carry phase-conditioned near-line structure.  Distinct `gamma` values
separate stems/leaves from high plume mass without storing an elevation chart.

## 4. Exact closed-form live transfer

Define, for signed finite `z`,

```text
z_k = b_k + gamma_k lambda.
E(z,L) = -expm1(-z L)/z,                z != 0,
E(0,L) = L,
I(z,R) = E(z,R).
```

The optical depth of mode `k` on the actual clipped world ray is

```text
tau_k
  = g_k(h_a) [
      (B_k/s_p) I(gamma_k lambda,R)
      + A_k I(b_k+gamma_k lambda,R)
    ].
```

The `s_p` factor in the oriented residual density cancels the `1/s_p` world-
length conversion.  It is not a fitted epsilon: it is the topological factor
which makes every azimuthal residual vanish at the vertical pole.

The small-`zL` implementation uses the fixed series

```text
E(z,L) = L [1 - zL/2 + (zL)^2/6 - (zL)^3/24 + ...].
```

At the vertical pole, write `R=s_p Delta_t` and take `s_p -> 0` with
`h(t)=h_a+d_y(t-t_a)`:

```text
tau_k -> B_k integral_0^Delta_t g_k(h_a+d_y s) ds.
```

This is the same finite `expm1` quotient for either sign of `d_y`; for
`gamma_k=0` it is `B_k Delta_t`.  The `A_k` contribution tends to zero.  Thus
the pole is finite, preserves the phase-dependent vertical field, and is
azimuth independent.  At grazing incidence the finite forward interval remains
inside `R`; there is no artificial `1/d_y` reconstruction and the exact
horizontal case is finite rather than a special miss.

Let `tau=sum_k tau_k`.  The positive foreground measure is

```text
A = -expm1(-tau),
phi(tau) = -expm1(-tau)/tau,  phi(0)=1,
P = phi(tau) sum_k tau_k c_k.
```

`P` is a positive fitted colour surrogate.  It is exact only when the modal
colour mixture is constant along the line; differently shaped/ordered coloured
modes do not in general obey this quotient.  In particular, violet plume colour
migrating onto green stem support is a binding source-to-surrogate RED metric,
not something the formula is allowed to hide.  Runtime output is only
`(A,P)`.  It contains no representative depth, owner, triangle plane, or fake
filtered normal.  Resolve uses

```text
C_out = P + (1-A) C_background
```

and preserves background id/depth for the fractional branch.  Consequently
the unresolved branch cannot create the old stretched foreground surface.

Filtered lighting uses a profile-global statistical NDF conditioned by the
four `tau_k` weights and live direction.  It never feeds Candidate F's face
plane solve.

## 5. Packed resolved and unresolved fields

### 5.1 Resolved record: the missing support certificate

The active face-plane correction proves a plane hit, not membership in the
finite source triangle or its baked first-visible region.  Extrapolating a
correct plane beyond that region is precisely capable of producing a long
streak.  The compact resolved record is therefore one `RG32Uint` texel:

```text
word 0:
  depth12 | oct face normal 8+8 | profile colour class4
word 1:
  25-bit 5x5 first-visible-owner micro-mask | 7 reserved bits
```

For each ordinary atlas texel the cook subdivides its exact baked-entry phase
square into `5x5` half-open microcells.  A bit is one iff the first-visible
region of the centre texel's exact triangle/copy intersects that microcell;
`mask=0` is MISS.  The live face-plane hit is reprojected into the same baked
slice, converted with the same snap/index convention to the exact atlas texel
and microcell, and is eligible only when both agree with the fetched record and
its bit is one.  The half-open convention is lower-inclusive on both authored
axes before the atlas-v flip; the flip is applied only after the integer site
is fixed.

A positive bit bounds phase dilation, not depth error.  At the current phase
rate the furthest point in one intersecting microcell is
`delta_q=sqrt(2)*2.03125/5 = 0.575 mm` from source support.  Therefore every
node additionally enforces the projected-depth bound

```text
|n dot d_a| >= |n_xz| delta_q / epsilon_t,
epsilon_t = max(epsilon_near, one live pixel's world footprint at t).
```

This makes the worst false-support plane displacement no larger than the
declared visible tolerance instead of merely asserting a millimetre phase
number.  The gate measures owner/colour mistakes separately because a small
geometric displacement can still cross a high-contrast occlusion boundary.
MISS, depth, normal, class, and mask are one categorical record and are never
interpolated.

Every node computes its final live plane intersection, forward-ray predicate,
cover-box predicate, and micro-mask predicate **before** the fixed max-weight
election.  Post-election invalidation is forbidden.  The direction cell is
bracketed around the actual authored-frame direction `d_a`, not the obsolete
predictor direction.  These are fixed ALU changes over already loaded records.

The sixteen-entry arbitrary-RGB profile palette has already passed its
Calamagrostis colour gate.  Packing the class in word 0 removes the separate
winner-colour fetch and the load-time owner/triangle colour reconstruction.

### 5.2 Unresolved four-vertex field

The amplitude field is one periodic 64-bit integer 3D texture:

```text
phase x: 128
phase z: 128
azimuth: 64 modulo-identified vertices (no stored 65th seam copy)
payload: 8 x UNORM8 = (B_0..B_3,A_0..A_3), profile-global log/linear decode
format: RG32Uint (point loads; manual tetrahedral interpolation)
bytes: 128*128*64*8 = 8.000 MiB
```

The field is not hardware-trilinearly filtered.  Triangulate the phase torus
with one globally consistent diagonal and take its product with each periodic
azimuth interval.  The azimuth upper vertex is `(j+1) mod 64`; vertex `0` is
physically shared at the seam and the cook requires bit-identical decoded
values there.

For a phase triangle with ordered product coordinates `(u_1,u_2)` satisfying
`0<=u_1<=u_2<=1` and azimuth fraction `a`, sort the labelled triple
`[(u_1,P1),(u_2,P2),(a,A)]` by value, breaking equality `P1<P2<A`.  Starting
at the product cell's all-lower vertex, flip the corresponding coordinate in
that label order.  The four staircase vertices are the start and the three
successive vertices; their weights are
`[v_1, v_2-v_1, v_3-v_2, 1-v_3]`.  The complementary phase triangle uses its
own globally fixed coordinate map before the same rule.  This is the complete
three-tetrahedron monotone-path construction, with one equality convention on
both sides of every face.

One layer therefore uses four point `textureLoad` operations from shared
vertices.  Each load unpacks eight amplitudes before the four vertex values are
interpolated.  Two globally affine anti-tiling layers use eight physical profile
fetches total, below the frozen nine-fetch ceiling.  The two layers transform
the query coordinates but reuse the same coefficient texture, so bytes do not
multiply.  Shared quantized vertex bits make the amplitude field globally `C0`
across phase triangles, azimuth cells, and the azimuth seam.  Bit-identical
`B_k` at all azimuth vertices plus the analytic `s_p` factor make the vertical
pole independent of azimuth.  There is no second scale sample, direction list, owner election,
dependent read, loop, march, geometry, new binding chain, pass, barrier, or
dispatch.

For `T=65*256^2`, the resolved allocation is `T*8 = 32.5 MiB`.  Together with
the 8 MiB unresolved coefficient field, the profile payload is approximately
`40.5 MiB`, below both the active approximately `49.5 MiB` GPU atlases and the
frozen `48.75 MiB` replacement comparison.  Runtime owner pages, source triangle
tables, generated colour atlas, and the 114 MiB research GCRP are cook inputs,
not shipped runtime payloads.

## 6. Continuous minification without a sampled scale coordinate

The coefficient field is cooked at the finest unresolved footprint.  Let the
cook fit profile-global positive constants `Bbar_k,Abar_k`, and let `sigma_px`
be the analytic world-space pixel footprint in the entry chart.  The decoded live
amplitude is

```text
B_k^f = Bbar_k + w_k(sigma_px) (B_k-Bbar_k),
A_k^f = Abar_k + w_k(sigma_px) (A_k-Abar_k),
w_k(sigma) = exp(-0.5 nu_k^2 max(0,sigma^2-sigma_0^2)).
```

`nu_k` is a cook-fitted effective spatial frequency and `sigma_0` is the bake
footprint.  The eight constants and four frequencies are profile constants,
not sampled texture data; their byte source and uniform packing must be shown
in the implementation diff.  The deliberately azimuth-independent `Abar_k`
is part of the fidelity hypothesis, not an omitted table.  This is a
continuous contraction toward one fitted mode mean, not a cell/LOD selection.
It is only a spectral closure, not an exact arbitrary footprint filter; both
the sigma-4 and sigma-16 truth fields and the azimuth-dependent mean residual
are binding in the gate.  If the constant mean fails, Candidate M is RED; an
unaccounted directional mean texture is not an allowed repair.

Large cover-region holes are retained by the separate world control/support
field and are never replaced by the phase mean.  The contraction applies only
inside an active community support.

## 7. Continuity and stability

Tetrahedral interpolation of shared texture vertices gives a continuous
positive amplitude field in phase and azimuth.  Sections 2--4 are continuous
compositions on every finite exterior ray segment, with explicit vertical and
zero-rate limits.  Section 6 is continuous in pixel footprint.  Therefore the
compiled measure has no elevation row, azimuth Voronoi boundary, LOD ring, or
camera-centred radial election band.

Camera translation changes `q`, `t_a`, `t_b`, and `sigma_px` continuously
inside an affine support chart.  At a change of the active slab/support bound,
the interval endpoint is still the max/min of continuous scalar roots and is
therefore continuous.  It cannot
change a categorical filtered owner because none exists.  This structurally
removes the old million-particle turnover mechanism from the unresolved
branch; any remaining temporal error is a texture/filter/address bug or a
failure of the near/far handoff, not an allowed property of the model.

## 8. Fixed cost rationale

Per unresolved layer:

- memory traffic: four coherent point loads from one `RG32Uint` field,
  thirty-two logical payload bytes before cache-line effects; two layers are
  eight loads / sixty-four payload bytes;
- ALU: eight positive decodes, four paired base/residual `E/tau` evaluations, one optical-depth
  reduction, colour/NDF mixing, and fixed small-limit selects; target below
  `90` FMA-equivalent plus four exponential-family evaluations;
- live state: eight amplitudes, four transient `tau_k` values consumed and
  reduced immediately, one accumulated colour, and the ray scalars;
- divergence: fixed-cost selects inside a query shape; footprint-selected
  resolved/bridge/unresolved shapes may diverge across a wave and the generated
  shader must prove inactive texture-load branches are not executed;
- synchronization: none;
- dispatch/pass/binding topology: unchanged until explicitly authorised.

The expensive-looking part is four exponential evaluations.  They are the
closed-form substitute for walking the line.  The cook must test a shared-rate
factorization (two distinct `z` values with paired colours) that reduces this
to two exponentials without changing the four stored amplitudes.  Failure of
that reduction is a measured ALU result, not permission to march.

### 8.1 Fixed query shapes and continuous handoff

The analytic entry-chart pixel footprint is available before profile reads.  It
uses the local-affine cover interval and ray differentials; it does not
depend on a provisional triangle normal or exact HIT/MISS.  It selects these
compile-time shapes for the current single-layer Calamagrostis gate:

```text
resolved:    4 R0 + 4 corrected R1 resolved records = 8 loads
bridge:      4 R0 resolved records + 4 transfer vertices = 8 loads
unresolved:  4 transfer vertices = 4 loads
```

In the bridge, each R0 record directly solves and micro-mask-validates its
same-face live plane; it is mixed as a positive exact sample with Candidate M's
positive measure.  A cubic weight has continuous scalar endpoints, but it does
not prove equality of the R0, corrected-R1, and analytic functions.  Different
eligibility/election at R0 and R1 and the fractional-background versus opaque
depth/id convention can expose a frontier.  The handoff is therefore an
explicit empirical colour/coverage/depth/id/translation gate, not a `C0`
theorem.  At the unresolved edge the exact weight is exactly zero.  There is
no profile-load interval in which the cost exceeds eight and no camera-centred
categorical LOD election inside either function; passing the handoff gate is
required before making that claim about the composed renderer.

The eventual two-layer query is not silently granted: the unresolved branch is
eight loads, but a two-layer resolved/bridge construction needs a separate
frozen schedule under nine.  The current requested single-species visual gate
must pass before solving that later tier-1 variance schedule.

## 9. Decisive pre-runtime gate

Use the actual Calamagrostis triangle soup and the existing exact-BVH positive
measure truth.  Fit only training directions/phases; freeze held-out samples
before fitting.  Score both source footprints and a continuous eye-height/
slope sweep, with special reports for uphill local-affine frames and the cover
edge.

GREEN requires, separately at the finest unresolved and minified footprints:

```text
coverage abs error:                 p95 <= 0.08, p99 <= 0.20
premul RGB max-channel error:       p95 <= 0.06, p99 <= 0.15
largest connected p99 exceedance:  < 1%
4.5 mm translation delta:          p95 <= 0.06, connected < 1%
old row/az/LOD boundary jump:       numerical zero
```

The gate must also render direct truth versus codec images at standing height,
slightly uphill, ten-metre oblique, top-down, and a cover boundary.  The result
is RED if plume mass migrates into green structures, source gaps become a
sheet, the far field becomes a flat colour wash at 2--5 m, or the fitted model
passes scalar metrics while losing the accepted Calamagrostis silhouette.

Before runtime implementation, additionally require:

1. the resolved/bridge identity and sub-pixel handoff gate with no ring;
2. a micro-mask false-accept/false-reject report in millimetres and by semantic
   class, including thin plume and leaf triangles;
3. the same cook/gate for a deliberately non-axial arbitrary triangle soup and
   for an overlapping multi-colour/multi-species community; and
4. one complete-boundary Fable red-team review of the maths, measurements,
   byte/fetch accounting, and proposed code diff.

If the four-mode transfer is RED, do not add modes, texture reads, memory,
sampled elevation, or a neural decoder.  Park it with the fitted residual and
return to the only remaining honest choice: relax the arbitrary-soup fidelity
claim or the fixed-cost contract.  A runtime experiment is not authorised by a
RED fit.

## 10. Measured periodic-interior result (2026-07-24)

The frozen four-mode transfer is **RED** on the real `2,171,134`-triangle
Calamagrostis source.  This is deliberately only the periodic-interior fidelity
subgate.  A separate mathematical audit found that the document's top-entry,
support, emission-order, minification, and near/far handoff claims are not yet
complete enough to support an overall GREEN even if this fit had passed.

The offline gate fitted only the nine training elevations
`{0.25,2,5,10,18,30,55,75,90}` degrees and phase/azimuth lattice sites.  It
then evaluated ten excluded elevations, the `2.8125`-degree azimuth midpoints
of the proposed 64-slice field, phase-cell centres, and a `4.5 mm` phase
translation at both accepted source footprints.  The fitted model enforced:

- four nonnegative `B/A` mode pairs;
- one `B_k(q)` field shared across all trained azimuths;
- four profile-global colours;
- six preregistered logarithmically spaced rate families, selected only by
  training optical-depth error;
- `sigma16` amplitudes obtained from the stored `sigma4` field by the specified
  contraction toward **stored-field/profile-global phase means**.  No
  independently fitted coarse field was used for the scored prediction; and
- profile-global logarithmic `UNORM8` encode/decode for the eight stored
  amplitudes.

Neither packing nor a direction boundary caused the failure.  All `176`
unquantized cases and all `176` quantized cases were RED, while the old
categorical direction-boundary jump is exactly zero by construction.  The
worst binding values were:

```text
metric                         measured worst       binding limit
coverage p95                  0.8482               0.08
premultiplied RGB p95         0.5176               0.06
connected exceedance         0.9158               0.01
```

Even the exact-phase, exact-trained-azimuth subset was decisively RED
(`coverage p95 0.8102`, RGB p95 `0.5033`, connected `0.9075`).  Therefore the
result cannot be blamed on the diagnostic's 64-square phase lattice, its
held-out phase interpolation, or its held-out azimuth interpolation.  The
four positive base-plus-line-residual elevation curves do not preserve the
source's view-dependent first-hit measure; they turn distinct source gaps and
plume/structure changes into broad coherent errors.  The `UNORM8` result is
numerically indistinguishable from unlimited precision, so more packing bits
cannot rescue it.

Durable artifact:

```text
data/work/groundcover-candidate-m-line-transfer/
  e3e0a4175b151b89/47ceac59d413d74d/
    report.json
    SUMMARY.md
    QA_INDEX.json
    qa/001..005-*.png
```

The QA panels are `truth | quantized Candidate M | max-channel error`.  Per
the frozen stop rule, no mode, read, byte, sampled elevation, or decoder was
added after RED.  Candidate M is parked and no runtime/shader transcription is
authorised.

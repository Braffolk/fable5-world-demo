# Candidate N: height-moment positive transfer

Date: 2026-07-24  
Status: **RED / PARKED after the one permitted fit and premise-audit rerun**;
no runtime or shader authorisation

## 1. Scope and decision boundary

Candidate M is RED because four entry-conditioned exponential curves cannot
represent the real community's phase- and view-dependent positive measure.
Candidate L is RED because sampling live direction leaves the unremoved phase
shear inside the sampled coordinate.  Candidate N removes elevation from the
stored domain in a different way from M: it stores four continuous horizontal
coefficient fields attached to a nonnegative height basis, then evaluates the
height moments and the associated parallax addresses analytically on the live
ray.

Candidate N replaces only the unresolved/physically-filtered branch.  The
resolved Candidate-F same-face query remains, with the finite-support record
described in Section 7.  Candidate-H's real-background premultiplied overlay
remains the resolve law.

The codec accepts any finite triangle soup at cook time.  It is **not** an exact
encoding theorem for arbitrary soups: the soup is compiled into a positive
four-height-moment surrogate, and source fidelity is a measured gate.  Exact
reconstruction of every arbitrary soup under a fixed 36 MiB / nine-load codec
would contradict the already measured phase-direction rate and ordinary
information capacity.  A RED fit parks Candidate N; it is not permission to
add heights, reads, bytes, or a sampled direction coordinate.

## 2. Exterior ray and finite support interval

Work in the same authored local-affine terrain/wind chart as Candidate F.  The
camera-inside-cover-box case fades before this query.  For the exterior ray

```text
x(t) = o + t d,       |d|=1,       t >= 0,
```

intersect, with fixed scalar slab algebra,

```text
0 <= h(x(t)) <= H,
0 <= t <= t_background,
W_0 + grad(W)_0 dot (x(t).xz-x_0.xz) >= 0.
```

The result is one forward interval `[t_a,t_b]` or MISS.  The complete shader
read found that the current profile-2 guide already binds and samples
`field3`, but it stores cover density without its derivative; the nearby
`bguide` derivative is terrain height, not cover support.  Therefore the
isolated profile-2 guide representation is explicitly changed, in its existing
dispatch and existing `RGBA16F field3`, to

```text
field3 = (coverDensity, dDensity/dx, dDensity/dz, curvatureBound).
```

The guide dispatch obtains the two central differences from the same packed
cover field while filling the guide.  The per-pixel query moves its one
existing `field3` sample before the profile query and reuses that value for
the final root/support test.  Thus the ray query gains no texture operation,
binding, texture, pass, or resident byte; the guide producer gains two packed-
control samples per guide site and must be included in the GPU trace.  The
legacy multi-species guide layout is a separate module/path and is not silently
reinterpreted.
Let

```text
L = t_b-t_a,
u(r) = h(x(t_a+L r))/H = u_0 + delta_u r,    0 <= r <= 1.
```

Top entry, side entry, uphill entry, vertical rays, and exactly horizontal
rays are all represented by the same finite interval.  No division by `d_y`
or `|d_xz|` occurs anywhere in Candidate N.

## 3. Nonnegative height basis and exact moments

Use the cubic Bernstein partition of unity

```text
beta_j(u) = choose(3,j) u^j (1-u)^(3-j),    j=0..3.
```

These are nonnegative on the cover box and separate basal, two central, and
high mass components without a categorical height slab.  Along the live
interval each is a cubic in `r`:

```text
beta_j(u(r)) = sum_(ell=0)^3 c_ell r^ell.
```

Define the exact zeroth, first, and centred second moments

```text
M_nj = sum_(ell=0)^3 c_ell / (ell+n+1),    n=0,1,2,
Z_j  = M_0j,
R1_j = M_1j/Z_j,
R2_j = M_2j/Z_j - R1_j^2,

J_j       = L Z_j,
tbar_j    = t_a + L R1_j,
variance_j= L^2 R2_j.
```

For `Z_j=0`, `J_j=0` and the address is selected to the finite interval
midpoint before any load.  The contribution is exactly zero, so that address
choice cannot affect the mathematical output.  No `0/0`, dummy infinity, or
epsilon is part of the model.

The formulas are polynomials in the endpoint heights.  Consequently:

- the vertical limit is exact: all four addresses share the same `xz` phase;
- the horizontal limit is exact: `u` is constant, every nonzero mode uses the
  finite interval midpoint and `J_j=L beta_j(u)`;
- rising and descending rays use the same algebra; and
- every nonzero characteristic address
  `q_j=x(tbar_j).xz` is continuous in camera position and direction.

This is the load-bearing replacement for both sampled elevation rows and M's
single entry-conditioned line curve.  Four separately moving characteristic
addresses carry basal, two central, and high parallax components of the plant.

The source field would contain the line integral

```text
integral a_jm(x(t).xz) beta_j(u(t)) dt.
```

Candidate N replaces it with the one-point moment quadrature

```text
a_jm(q_j) J_j.
```

The characteristic point integrates constants and the first spatial Taylor
term exactly because it is the beta-weighted centroid.  Its leading local
error is

```text
0.5 J_j trace(Hess(a_jm)(xi) *
              [variance_j d_xz d_xz^T]).
```

The footprint contraction in Section 5 reduces, but does not prove away, this
second-order term.  Large curvature or multiple distinct phase events inside
one height moment is therefore a binding source-fidelity failure mode.  The
document never treats `q_j` as an exact arbitrary-field line integral.

## 4. C0 one-load spatial cells

For each height basis `j`, store four nonnegative semantic amplitudes

```text
a_jm(q),      m in {low-green, high-green, violet-plume, pale-detail}.
```

The periodic phase torus has `256 x 256` **cells**.  One `RG32Uint` record per
cell duplicates the four corner values of all four modes:

```text
4 corners * 4 modes * 4-bit companded amplitude = 8 bytes.
```

The four decoded corners of each mode define two affine triangles using one
globally fixed cell diagonal and equality convention.  Adjacent cells are
cooked from the same quantized vertex code, including the periodic seam.  Their
edge polynomials are therefore bit-identical.  One point `textureLoad` plus
fixed unpack/triangle ALU evaluates one globally C0 field; no hardware filtering
and no neighbouring load is hidden in the count.

The four-bit code is frozen before the gate because Candidate K already showed
that direct `RGBA4444` stored-node quantisation is below the accepted positive-
measure error.  Candidate N uses a separately fitted profile-global logarithmic
compander for extinction rather than assuming linear four-bit precision.  The
gate reports unquantized, companded, and incremental packing error separately;
packing failure rejects this format instead of silently widening it.

Four height fields cost

```text
4 * 256^2 * 8 bytes = 2.000 MiB
```

and require exactly four physical loads per affine layer.  The fields are the
four layers of one `RG32Uint` 2D-array binding, not four sampled-texture
bindings.  A globally affine second anti-tiling layer reuses the same array at
transformed world-anchored addresses, so it costs another four loads but no
resident bytes.  Decode precedes triangle interpolation in physical extinction
units; positivity is preserved.

## 5. Continuous physical-footprint closure

For each `j,m`, cook the torus mean `mu_jm`; for each height basis cook one
positive-semidefinite `2 x 2` effective frequency tensor `K_j` shared by its
four semantic modes.  Let `Sigma_px` be the live pixel's
horizontal world-space covariance in the authored chart.  Define

```text
Sigma_j   = Sigma_px + variance_j (d_xz d_xz^T),
x_j       = 0.5 trace(K_j Sigma_j),
w_j       = 1 / (1 + x_j + 0.5 x_j^2),
a^f_jm    = mu_jm + w_j (a_jm(q_j)-mu_jm).
```

The rank-one second term in `Sigma_j` is the exact horizontal covariance of
the live height-basis support about its characteristic point.  Thus a long grazing
segment contracts phase detail continuously toward the cooked mean instead of
sweeping an aliased phase sample across the screen.  The pixel term performs
the same contraction under distance minification.  There is no sampled LOD,
scale cell, row, ring, or max-angular election.

This is a one-frequency-tensor closure, not an exact filter of an arbitrary
field.  The tensor lets an oriented graminoid field decay differently along
and across its dominant direction without storing a sampled view or scale.
The positive rational contraction avoids eight exponential-family evaluations
for two affine layers and has the same `w(0)=1`, monotone, zero-at-infinity
limits as the Gaussian closure it approximates.  The cook must fit `K_j` only
on training footprints, and both sigma-4 and
sigma-16 held-outs are binding.  A second frequency, mip, or coarse texture is
not an allowed repair after RED.

## 6. Positive transfer and real-background overlay

Each semantic mode has profile-global colour `c_m` and a positive directional
cross-section.  With `v(d)=(1,d_x,d_y,d_z)`, define

```text
chi_m(d) = v(d)^T M_m v(d),       M_m positive semidefinite.
```

The augmented constant permits a fitted front/back-odd term while the PSD form
keeps the result nonnegative; it therefore does not silently assume a
two-sided arbitrary soup.  It captures broad erect-versus-horizontal and
orientation response without a sampled direction coordinate.  The mode
optical depths are

```text
tau_m = chi_m(d) sum_j a^f_jm J_j,
tau   = sum_m tau_m,
A     = -expm1(-tau),
P     = [-expm1(-tau)/tau] sum_m tau_m c_m,     limit P=0 at tau=0.
```

`(A,P)` is a positive unresolved foreground measure.  It carries no synthetic
surface depth, triangle owner, face normal, or representative position.
Resolve preserves the real background id and depth and writes

```text
C_out = P + (1-A) C_background.
```

Filtered lighting uses a mode-global statistical NDF mixed by `tau_m`; it
never enters Candidate F's plane solve.  Concretely, each mode cooks a
unit-trace PSD geometric-normal second moment `S_m` and nonnegative ambient/sun
scales.  Its lookup-free diffuse response is

```text
ell_m(l) = ambient_m + sun_m sqrt(max(0, l^T S_m l)),
```

multiplied by the already computed shadow visibility, and `c_m` is replaced by
`ell_m c_m` in `P`.  This is a statistical microflake response, not a claimed
surface normal.  The gate compares premultiplied radiance under at least eight
sun azimuths, three elevations, ambient-only, and hard-shadow translations;
NDF parameters are fitted only on the training lights.  The violet mode has its own high
Bernstein coefficients and cannot colour a low green mode merely because a
categorical view selected the wrong source record.  Whether four globally
coloured modes retain the real plume gradients remains an explicit RGB gate.

For two affine anti-tiling layers, compute each layer's four `tau_m`, multiply
each by `1/2`, add by semantic mode, and apply the transfer formula once.  The
layer's affine linear part is applied to `q_j`, `d_xz`, and `Sigma_px` together;
translation applies only to `q_j`.  The layers therefore vary the community
without a frame mismatch or doubled optical density.

## 7. Resolved support and a complete nine-load handoff

The resolved record is one `RG32Uint` texel per existing direction/phase site:

```text
word 0: depth12 | geometric oct normal 8+8 | colour class4
word 1: 25-bit 5x5 first-visible-owner micro-mask | 7 reserved
```

`mask=0` is the sole MISS encoding; the remaining word is ignored.  A covered
record must have at least one bit.  The unwrapped corrected phase first selects
one half-open authored texel and microcell; only then is the texel index wrapped
and atlas-v flipped for the load.  Fetch and membership therefore use one
integer convention at seams and exact boundaries.

As in Candidate M, every node validates the final face-plane hit, forward ray,
cover box, exact texel, and micro-mask **before** election.  With phase support
radius `delta_q=sqrt(2)*2.03125/5=0.575 mm`, it additionally enforces

```text
|n dot d_a| >= |n_xz| delta_q / epsilon_t,
epsilon_t = max(epsilon_near, one live pixel world footprint at t).
```

Thus a positive mask bit cannot amplify its bounded phase dilation into an
unbounded depth streak.  Four R0 and
four corrected R1 point loads are eight physical loads; the colour class is in
the elected record, so there is no ninth colour fetch.  Its allocation is

```text
65 * 256^2 * 8 bytes = 32.500 MiB.
```

A direct blend of this eight-load query and the four-load filtered model
would violate the ceiling.  Candidate N closes that handoff with a distinct
one-load bridge summary, not an uncounted exception.  The bridge texture is one
`RG32Uint` cell record: four 8-bit-companded corner amplitudes for aggregate
body and four for aggregate plume (`0.500 MiB`).  Both amplitudes are evaluated
at one explicitly shared aggregate characteristic phase
`q_B=x(tbar_B).xz`, where `tbar_B` is the centroid of one fixed positive bridge
height profile.  One load cannot address independent body/plume phases; this
co-located approximation is deliberate and is a binding B1/B2 fidelity gate.
It uses the same analytic interval and footprint contraction as Sections 3--6.

Let `s_texel=2.03125 mm` be the resolved authored texel pitch and define the
continuous resolution coordinate, before any profile load, by

```text
Sigma_query = Sigma_px + (L^2/12) d_xz d_xz^T,
rho = sqrt(lambda_max(Sigma_query)) / s_texel.
```

The second term is the exact covariance of a uniform point on the finite ray
interval and makes a long low-oblique traverse become unresolved even if the
screen pixel itself is small.  The frozen query shapes are:

```text
R:   rho <= 0.5, exact Candidate F                         8 loads
B1:  0.5 < rho < 1, exact F + one bridge summary          9 loads
B2:  1 <= rho < 4, bridge summary + two-layer Candidate N 1+8 = 9 loads
U:   rho >= 4, two-layer Candidate N                       8 loads
```

Use `smoothstep(s)=s^2(3-2s)` on each normalized finite interval.  B1 blends premultiplied exact output into
the one-load positive summary and ends at that summary exactly.  B2 starts at
the same summary exactly and blends into the full two-layer four-height
measure.  Therefore the colour/coverage function is C0 at both query-shape
boundaries even when the two representations disagree internally.  There is
no interval in which more than nine profile texels are fetched, and tier-1's
second affine layer is included rather than deferred.

For fractional B1/B2 pixels, background depth/id is retained.  Crossing from
fully exact R to fractional B1 changes that metadata convention at the exact
frontier even though colour/coverage is continuous; the existing Candidate-H
frontier shadow/AO/TAA gate remains binding.  Interpolated fake depth is
forbidden.

Total resident profile payload is

```text
32.500 MiB resolved + 2.000 MiB N + 0.500 MiB bridge = 35.000 MiB,
```

below both the `48.750 MiB` ceiling and the current approximately `49.5 MiB`
GPU allocation.  The worst payload traffic is B1:
`8*8 + 1*8 = 72` logical bytes before cache effects.  U is
`8*8 = 64` bytes but has no dependent reads and only four coherent cell
addresses per layer.  No pass, barrier, dispatch, geometry, candidate list,
per-species work, or loop is introduced.

## 8. Continuity theorem and limits

Assume: the local support interval is one connected affine interval; duplicated
cell-edge codes are identical; `Sigma_px` is continuous; and the cook provides
finite positive parameters.  Then Candidate N's unresolved colour and coverage
are continuous in exterior camera position, direction, phase, and footprint.

Proof sketch:

1. slab/support endpoint roots and their max/min are continuous while the
   interval is nonempty; its length tends to zero at disappearance;
2. Bernstein moments are polynomial in the endpoints; a zero-mass address is
   multiplied by zero;
3. each nonzero characteristic phase is continuous, and duplicated triangular
   cells agree exactly on shared edges and the torus seam;
4. contraction, PSD cross-sections, positive optical transfer, affine layer
   composition, and cubic handoff weights are continuous compositions; and
5. Sections 2--3 give finite, matching vertical and horizontal limits without
   clamping a direction.

The theorem is about reconstruction of the compiled surrogate.  It does not
claim that four height moments equal an arbitrary triangle soup.

## 9. ALU, register, and topology budget

Per full unresolved layer: four 64-bit point loads.  Each height record contains
sixteen nibbles (four corners times four modes), hence the packed two-layer path
exposes `4*16=64` nibble codes per layer and `128` across both layers before
triangle selection.  A conforming implementation may decode only the three
selected corners, but the record/unpack accounting may not pretend the fourth
corner is absent.  The remaining work is four cubic moment tuples shared across
four modes, triangular evaluations, four augmented PSD quadratic forms, and
sixteen multiply-accumulates into four optical depths per layer.
Across two layers, decode/accumulate each height immediately and retain only
four running `tau_m` values.  Honest worst-case decode accounting is up to
`128` bitfield extracts (each a shift/mask pair) and `128` accesses to a
profile-global sixteen-entry compander table; evaluating logarithmic decode
with per-amplitude exponentials is forbidden.  The conservative generated-code
ceiling is `320` integer-plus-scalar ordinary operations for both layers,
eight positive rational contractions, four final statistical-NDF square roots,
and one final optical exponential.
The generated shader and coherent GPU trace must report the actual scalar ALU,
special-function count, registers, and occupancy; exceeding this stated shape
is RED rather than permission for a larger decoder.

There are no workgroups, barriers, synchronization, or new dispatch shapes.
The implementation review must show that inactive R/B1/B2/U texture-load
branches are not speculatively executed, that the new integer texture does not
exceed fragment sampled-texture limits, and that the old runtime owner/triangle
and separate colour assets are removed rather than retained beside the new
payload.

## 10. Decisive offline gate

Fit the actual accepted Calamagrostis triangle soup to physically filtered
premultiplied coverage/colour truth.  Freeze training phases, directions, and
footprints before looking at held-outs.  Evaluate continuous held-out elevation
and azimuth, sigma-4 and sigma-16, 1--4.5 mm camera translations, exact
horizontal/vertical limits, uphill local-affine frames, the world-support edge,
and the complete R/B1/B2/U handoff.

GREEN requires, separately at both footprint regimes:

```text
coverage absolute error:             p95 <= 0.08, p99 <= 0.20
premul RGB max-channel error:         p95 <= 0.06, p99 <= 0.15
largest connected p99 exceedance:    < 1%
4.5 mm translation output delta:      p95 <= 0.06, connected < 1%
cell/seam/query-shape boundary jump:  numerical zero
```

Visual truth-versus-codec panels are mandatory for standing/flat, standing
uphill, low oblique, ten-metre oblique, top-down, and a cover boundary.  RED if
the result becomes a homogeneous fog/flat wash at 2--5 m, loses the accepted
Calamagrostis side/plume structure, migrates violet mass into low green support,
or passes scalar averages while coherent source gaps are filled.

Before runtime authorisation, repeat compilation on a deliberately non-axial
triangle soup and an overlapping multi-colour/multi-species community.  These
do not have to share Calamagrostis' fitted parameters, but may not change the
record layout, read count, or shader shape.

One coherent fit is authorised.  If RED, perform one premise audit (truth
filter, entry/support interval, units, fit objective, and quantizer).  If the
conforming rerun remains RED, park Candidate N.  Do not add height moments,
sampled views, neural decoding, hidden hardware filtering, bytes, or reads.

## 11. What this candidate predicts

If GREEN, Candidate N should remove the causes rather than cosmetically blur
them:

- no sampled direction or scale means no screen-fixed circular rings or
  azimuth wedges;
- four analytic parallax addresses prevent basal/central/high plant mass from
  sharing one wrong projected angle;
- C0 world-anchored fields and analytic footprint contraction remove the
  directional million-particle crawl;
- real-background fractional resolve cannot stretch a fake filtered surface;
- resolved micro-mask membership bounds plane streaks and missing/detail
  erosion before election; and
- explicit violet/high and green/low modes prevent categorical colour-owner
  migration.

The remaining quality risk is equally explicit: four height moments plus a
single-frequency contraction may still be too low-rank to preserve the real
Calamagrostis silhouette at 2--5 m.  Only the frozen source-fidelity gate can
answer that; continuity and budget proofs cannot.

## 12. Measured gate and blocker (2026-07-24)

The frozen representation is decisively **RED** on the accepted
`2,171,134`-triangle Calamagrostis source.  The gate reused the existing GCRP/v4
exact first-hit pages and their physically filtered sigma-4/sigma-16 truth,
reduced to the same `64 x 64` analysis rate used by Candidate M.  The frozen
split fitted all four elevations at eight `45 degree` azimuths plus the pole
(`33` directions) and held out the interleaved `32` directions.  It separately
scored exact-trained nodes, held-out nodes, half-cell phase, and `4.5 mm`
translation.  The fitted data path included:

- four cubic Bernstein height fields and their analytic characteristic shifts;
- a positive augmented-vector PSD direction response per semantic mode;
- four world-phase semantic fields fitted with projected nonnegative
  optimisation;
- the specified rational sigma contraction fitted from an independently
  reduced sigma-16 field; and
- unlimited precision followed by the frozen four-bit logarithmic compander.

The first run exposed one concrete premise-audit defect: an unconstrained
Fourier inverse had been clipped after solving rather than fitted under the
nonnegative constraint.  The sole authorised rerun replaced it with projected
nonnegative optimisation without changing the model, split, packing, or
thresholds.  The rerun remained RED by very large margins:

```text
case                         coverage p95   premul RGB p95   connected
exact-trained sigma4          0.5198          0.3502          0.7939
held-out sigma4               0.4946          0.3238          0.8337
held-out sigma16              0.3750          0.2607          0.6504

limits                        0.0800          0.0600          0.0100
```

Four-bit packing is not the cause.  On exact-trained sigma4 it changed
coverage p95 only `0.5198 -> 0.5204`, RGB p95 `0.3502 -> 0.3503`, and connected
fraction `0.7939 -> 0.7966`.  The QA panels show the structural failure: the
source's alternating green/violet botanical bands become a broad nearly
constant beige transfer field.  This is already present at fitted directions
and unlimited precision, so neither held-out angular interpolation, phase
translation, nor packing can rescue it.

The `8/9/9/8` handoff load algebra and endpoint identities are exactly GREEN
(`R->B1`, `B1->B2`, and `B2->U` scalar jumps are numerical zero).  The bridge's
single shared `q_B` fidelity was not advanced into a separate fit after the
full four-height field failed: a lower-rank one-address summary cannot repair
the rejected carrier and no runtime path can reach it.  This is an explicit
unmeasured subordinate gate, not a claimed GREEN.

Durable output:

```text
data/work/groundcover-candidate-n-height-moment/
  e3e0a4175b151b89/1dc11b5ce5ef38c6/
    report.json
    SUMMARY.md
    QA_INDEX.json
    qa/001..005-*.png
```

Candidate N is parked.  Resume only if a new representation removes the
one-point height-moment quadrature itself while still proving the same fixed
read/memory/handoff limits.  Adding height moments, weakening the 2--5 m
fidelity gate, widening amplitudes, tuning contraction, or proceeding to the
shader are not resume conditions.

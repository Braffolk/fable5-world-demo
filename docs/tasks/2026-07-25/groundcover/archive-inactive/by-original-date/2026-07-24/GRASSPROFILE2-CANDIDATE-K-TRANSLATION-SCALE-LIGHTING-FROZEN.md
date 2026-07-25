# Candidate K: translation, footprint, and lighting gates — frozen protocol

Date: 2026-07-24  
Status: frozen protocol; K2 nested-parent branch is RED and parked in
`GRASSPROFILE2-CANDIDATE-K2-NESTED-PARENT-RED-BLOCKER.md`; the complete
Candidate-K angular FE is independently RED, so the remaining large gates are
not run

## 1. Scope and unchanged runtime contract

These gates test the remaining mathematical obligations of Candidate K's
filtered foreground measure

\[
M_\sigma(q,d)=(A_\sigma(q,d),P_\sigma(q,d)),\qquad P=A C.
\]

They do not change the categorical near reconstruction, shader, render
pipeline, atlas allocation, or runtime tap count.  The candidate remains
exactly `4 R0 + 4 R1 + 1 scale = 9` profile reads and `48.25 MiB` resident.
No result from this gate authorises a new texture, lookup, pass, candidate,
normal field, depth field, or representative grass plane.

The source is the current accepted Calamagrostis GCRP/v4.  Coverage and colour
are always measured as the positive pair `(A,P)`; conditional colour is never
interpolated directly.

## 2. Millimetric translation stability

### 2.1 Quantity under test

The scale atlas is world anchored.  Let `T(q)` be the canonical 256-square
filtered truth and `K(q)` the decoded Candidate-K 128-square direct-RGBA4444
field.  For a world translation `delta`, define codec error

\[
e_\delta(q)=K(q+\delta)-T(q+\delta)
\]

and the excess temporal change

\[
E_\delta(q)=e_\delta(q)-e_0(q)
=[K(q+\delta)-K(q)]-[T(q+\delta)-T(q)].
\]

This subtracts real variation in the continuous truth.  It therefore measures
only codec-induced crawl: a stationary codec error does not fail, while an
error that turns on and off as the camera moves does.  It is the directly
relevant quantity for the reported one-direction particle stream.

Truth is sampled with periodic bilinear reconstruction of the canonical
filtered field.  The codec uses the binding nearest 128-square texel and
RGBA4444 decode.  Both are addressed from the same unwrapped phase; no
camera-relative origin is introduced.

The frozen translation set is

```text
magnitudes: 1, 2, 3, 4.5 mm
directions: +x, +z, +(x+z)/sqrt(2), +(x-z)/sqrt(2)
directions/scales: all 65 accepted angular nodes, sigma 4 and sigma 16
```

The gate reports the expected nearest-cell transition fraction separately,
but does not mistake a cell transition for a defect.  Only excess decoded
change relative to truth is binding.

### 2.2 Frozen limits

For every translation vector and each scale, pooled across all direction nodes:

```text
|E_A|:                         p95 <= 0.04, p99 <= 0.10
max-channel |E_P|:             p95 <= 0.03, p99 <= 0.08
largest connected joint p99 exceedance region per node: < 1% of phase page
```

The joint mask is `|E_A| > 0.10 OR max(|E_P|) > 0.08`.  Static fidelity must
also remain inside Candidate K's already frozen `0.08/0.20` coverage and
`0.06/0.15` premultiplied-RGB limits at every translated phase.

This gate isolates spatial phase translation at fixed direction.  Angular
camera motion is separately covered by the finite-element gate; mixing both
would make a RED result uninterpretable.

## 3. Continuous exact / sigma-4 / sigma-16 footprint partition

### 3.0 Pre-output supersession: nested one-read measure pyramid

Before any result from this file was produced, the two fixed filtered
endpoints were superseded as the **primary** scale candidate by a nested
positive-measure pyramid.  The fixed sigma-4/sigma-16 blend below remains a
binding ablation because it tests the original Candidate-K claim; it is no
longer the preferred implementation if the nested construction is GREEN.

The first proposed spatial resolutions per angular cell are

```text
level 0: 112 x 112
level 1:  56 x 56
level 2:  28 x 28
level 3: one global measure
```

Every level-`l` texel stores four angular `RGBA4444` child measures `M_l` in
the low 64 bits and the bit-identical four parent measures `M_(l+1)` in the
high 64 bits.  A parent code is copied into every child in its integer nesting
block (four children for a 2x level transition).  The coarsest finite level
stores the angular global measure as its parent.  Therefore both sides of a
level switch decode exactly the same bit pattern at the switch:

\[
M_l(\lambda\to(l+1)^-)=M_{l+1}
=M_{l+1}(\lambda\to(l+1)^+).
\]

The exact allocation is

```text
128 * 16 B * (112^2 + 56^2 + 28^2) = 32.15625 MiB scale
65 * 256^2 * 4 B                     = 16.25000 MiB near
total                                = 48.40625 MiB
```

It remains one physical scale operation and nine profile operations total.

`M_l` means the canonical positive field filtered first at radius
`sigma_l=4*2^l` source texels and only then area-reduced to that level's spatial
page.  Spatial cell averaging is not substituted for the footprint filter.
That ordering is load-bearing: the former makes a smooth band-limited field
cheap to store; treating a nearest cell average of raw binary coverage as the
filter would reintroduce the very particle crawl being removed.  The global
parent is the exact tile mean.

The same parent-copy proof requires integer nesting, not specifically binary
nesting.  A pre-output budget audit found a strictly richer frozen candidate:

```text
120 -> 40 -> 20 -> 10 -> 5 -> global
sum of finite level texels per cell = 16525
scale allocation                    = 32.275390625 MiB
near + scale total                  = 48.525390625 MiB
```

This stays `0.224609375 MiB` below the 48.75-MiB comparison ceiling, retains
one 128-bit load, increases the finest spatial resolution from 112 to 120,
and removes the 28-to-global scale desert.  Both chains are measured.  The
120-chain dominates if it is GREEN and its worst translation metric is no
worse than the 112-chain within `1e-4`; otherwise the reason is reported rather
than inferred from level count.

A lower-rate uniform chain is also frozen before outputs:

```text
96 -> 48 -> 24 -> 12 -> 6 -> 3 -> global
scale allocation   = 23.994140625 MiB
near + scale total = 40.244140625 MiB
```

Its exact 2x nesting gives six finite scale intervals and 8.28125 MiB more
headroom than the 120-chain, at the cost of a coarser finest page.  The choice
is empirical, not "spend the ceiling": all three chains receive identical
stored-node/static-fidelity and translation gates.  Among GREEN chains, first
discard any whose worst translation p95 is more than `1e-4` above the best;
then choose the smallest allocation whose worst static premultiplied-RGB p95
is within `0.005` of the best remaining chain.  This lexicographic rule was
fixed before any chain output.

### 3.0.1 One-cycle premise audit after the one-node smoke

The first one-node diagnostic did not produce a Candidate-K verdict.  It first
found and corrected an invalid oracle premise (spatial cell averaging had been
substituted for positive footprint filtering).  With that corrected, it exposed
the real allocation trade-off: distributing the 32-MiB scale budget over many
pages lowers the finest sigma-4 field below the 128-square rate that already
passed the stored-node gate, while several coarser pages are also below their
measured bandwidth.

The single authorised diagnose-and-fix cycle therefore adds one budget-exact
candidate derived from the already-GREEN direct carrier:

```text
level 0: 128 x 128, low = sigma4,  high = sigma16
level 1:  16 x 16, low = sigma16, high = global
scale:   128 * 16 B * (128^2 + 16^2) = 32.5 MiB
total:   32.5 + 16.25 = 48.75 MiB
```

The sigma16 code in level 0 and level 1 is bit-identical for the corresponding
parent cell.  This is the smallest nested extension which preserves the proven
128-square sigma4/sigma16 page and exactly fills, but does not exceed, the
comparison ceiling.  It is now the primary candidate.  The multi-page chains
remain reported allocation ablations; there is no further hierarchy tuning if
this attempt is RED.

Bit-identical parents prove LOD continuity but do **not** alone prove spatial
translation continuity: a nearest child code still changes at an ordinary
cell boundary.  The primary gate therefore includes a boundary-safe analytic
level coordinate.  Let `d_l(q)` be the world-space distance from `q` to the
nearest grid line of nested level `l`, let `a` be one fixed physical transition
half-width, and define

\[
g_l(q)=H(clamp(1-d_l(q)/a,0,1)),\qquad
\lambda_b(q)=clamp(g_0+g_1+g_2,0,3),
\]

where `H(z)=z^2(3-2z)`.  At a level-`l` boundary, every finer nested grid also
has a boundary, hence `g_0=...=g_l=1` and
`lambda_b >= l+1`.  The discontinuous level-`l` address consequently has zero
weight there.

Let `lambda_f(rho)` be the footprint-selected continuous pyramid level,
linearly measured in `log2` texel pitch and clamped to `[0,3]`.  The single
runtime coordinate is

\[
\lambda(q,\rho)=\max(\lambda_f(\rho),\lambda_b(q)).
\]

With `l=floor(lambda)`, one reads the level-`l` record and blends its low/high
halves using `H(fract(lambda))`.  At every integer level the two neighbouring
records are bit-identical, and at every spatial boundary all representations
whose addresses jump have zero weight.  The construction is C0 in value with
one read; the smoothstep makes each isolated join C1 except at the harmless
`max` equality crease.  Corners require no special case because the summed
nested-grid boundary coordinate promotes to the common ancestor.

The transition widths are a frozen, reported candidate set rather than a
post-result tune:

```text
a / finest-cell-pitch = 1/8, 1/4, 1/2
```

Selection rule: a width is eligible only if both the static-fidelity and
millimetric-translation gates are GREEN.  Among eligible widths choose the one
with the lowest worst premultiplied-RGB p95; ties within `1e-4` choose the
narrower band.  The raw pyramid with `lambda_b=0` and the old fixed
sigma-4/sigma-16 carrier are reported ablations.  This rule was fixed before
any pyramid output was generated.

### 3.1 Analytic reference-plane footprint

For camera `C`, normalized ray direction `d(s)`, and Candidate K's fixed
reference height `h_ref`,

\[
t_R={h_{ref}-C_y\over d_y},\qquad q=C+t_Rd.
\]

For either one-pixel screen coordinate `s_k`,

\[
\partial_kq=t_R\left(\partial_kd-
d{\partial_kd_y\over d_y}\right).
\]

After taking the `xz` components and converting world metres to canonical
source texels, form the 2x2 Jacobian `J`.  The footprint coordinate is

\[
\rho=\sigma_{max}(J).
\]

For a pinhole camera with unnormalised ray `v`, basis derivative `b_k`, and
`d=v/|v|`, the required direction derivative is analytic:

\[
\partial_kd={(I-dd^T)b_k\over |v|}.
\]

Thus `q`, `J`, and `rho` are continuous for every descending exterior ray
`d_y<0`.  There is no camera-distance test, record eligibility test, or fake
grass plane in this coordinate.

### 3.2 Frozen C1 scale weights

The exact, sigma-4, and sigma-16 moments correspond to square-kernel support
widths

\[
s_0=1,\qquad s_1=9,\qquad s_2=33.
\]

Map a continuous footprint radius to support width and log coordinate:

\[
s(\rho)=\max(1,2\rho+1),\qquad x=\log s(\rho).
\]

For `x` between adjacent centres `log s_i` and `log s_{i+1}`, let

\[
z={x-\log s_i\over\log s_{i+1}-\log s_i},\qquad
H(z)=z^2(3-2z).
\]

The only nonzero weights are `w_i=1-H(z)` and `w_{i+1}=H(z)`.  Below the first
centre `w_0=1`; above the last centre `w_2=1`.  The weights are non-negative,
sum exactly to one, reproduce every stored scale at its centre, and are C1 at
both joins.  The reconstructed measure is

\[
M(\rho)=w_0M_0+w_1M_4+w_2M_{16}.
\]

`M_0` is the exact binary-hit/premultiplied-colour measure, including a zero
measure at MISS.  The weights never depend on exact eligibility.  Exact depth
is still written only by the categorical near path; these weights govern only
the colour/coverage measure.

### 3.3 Intermediate-radius truth and frozen limits

Independent toroidal square-filter truth is cooked at radii

```text
0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16 source texels.
```

Radii 0, 4, and 16 are identity checks.  All others are binding interpolation
tests.  The unlimited-precision basis is scored before spatial reduction and
RGBA4444.  Candidate K parks if that basis is RED; packing cannot rescue a bad
scale model.

For every accepted direction and every binding radius:

```text
coverage absolute error:       p95 <= 0.08, p99 <= 0.20
premul RGB max-channel error:   p95 <= 0.06, p99 <= 0.15
largest connected joint p99 exceedance region: < 1% of phase page
```

The gate additionally verifies numerically over fixed cameras, FOVs, viewport
sizes, and exterior descending rays that the analytic `rho` is finite and that
the closed-form Jacobian agrees with centred finite differences to relative
error `<= 1e-4`.  This is an equation-conformance check, not a visual proxy.

## 4. No-extra-data analytic filtered lighting

### 4.1 Rendered truth

The GCRP already carries the deterministic geometric face normal of each exact
hit.  For unit light `l`, exact two-sided Lambert truth is

\[
L(q,d,l)=I_{hit}(q,d)\,C(q,d)\,|n(q,d)\cdot l|.
\]

Filtered rendered truth at radius `sigma` is the toroidal positive filter of
this premultiplied quantity.  It is not the response of an averaged normal.
Both sigma-4 and sigma-16 are tested.

### 4.2 Analytic candidate

Let `u=(0,1,0)` be the authored growth axis before the already-approved affine
wind conjugation.  Define

\[
D_{ring}(l)={2\over\pi}\sqrt{1-(u\cdot l)^2},\qquad
D_{plume}(l)=1/2.
\]

The structure response is

\[
D_s(l;\eta)=\eta|u\cdot l|+(1-\eta)D_{ring}(l),
\qquad 0\le\eta\le1.
\]

The only conditioning available at runtime is the already reconstructed
conditional RGB `c=P/max(A,epsilon)`.  A continuous plume fraction is

\[
p(c)=clamp(\beta_0+\beta_r c_r+\beta_g c_g+\beta_b c_b,0,1).
\]

The final no-data response is

\[
D(c,l)=(1-p(c))D_s(l;\eta)+p(c)D_{plume}(l),\qquad
\widehat L=P\,D(c,l).
\]

It consumes no normal texture, direction page, palette entry, or additional
read.  `eta` and `beta` are cook-fitted global constants.  They may not vary by
view cell, spatial phase, or species at runtime.

### 4.3 Frozen fit split and lights

The fit and score split is deterministic and fixed before fitting:

```text
training direction nodes: even GCRP slice index
held-out direction nodes:  odd GCRP slice index plus the pole
training lights:           elevations 15, 45, 75 deg; azimuths 0, 90, 180, 270 deg
held-out lights:           elevations 5, 30, 60, 85 deg; azimuths 45, 135, 225, 315 deg
```

Fitting minimises squared premultiplied-RGB error over a deterministic spatial
subsample of training nodes/lights and both scales.  The held-out score uses
all phase texels of its nodes.  A second ablation fixes `p=0` to expose whether
RGB conditioning materially carries plume response.

The source primitive semantic sidecar partitions exact hits into plume and
structure using the accepted procedural recipe.  It is used only for separate
truth reporting and never as an input to the fit or runtime formula.

### 4.4 Frozen limits

For every held-out light, scale, and reported semantic population:

```text
premul lit-RGB max-channel error: p95 <= 0.06, p99 <= 0.15
coverage-weighted scalar response error: p95 <= 0.08, p99 <= 0.20
largest connected lit-RGB p99 exceedance region per node: < 1% of phase page
```

The primary all-content population is binding.  Plume and structure subsets
are also binding when each has at least 4096 supported filtered texels;
otherwise they are reported as underpowered rather than silently pooled.

If this analytic family is RED, Candidate K's geometry/colour measure is not
thereby rejected.  The honest blocker is narrower: no proved filtered-lighting
response yet fits the no-extra-read/no-extra-memory contract.  The next action
must return to the orientation-distribution mathematics, not add a sampled
normal or another texture.

## 5. Decision rule

Candidate K may advance from these obligations only if:

1. millimetric excess temporal change is GREEN;
2. the nested pyramid's intermediate-footprint fidelity and boundary-safe
   translation are GREEN (the fixed-endpoint ablation may be RED);
3. the analytic Jacobian agrees with its numerical derivative;
4. the no-extra-data lighting response is GREEN on held-out lights.

Every RED result records the failed equation, population, and objective resume
condition.  No threshold may be altered after outputs are observed.

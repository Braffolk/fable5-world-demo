# Candidate I: direction-analytic vertical-extinction carrier

Date: 2026-07-24  
Status: pure mathematics and offline codec gate; runtime is not authorised

## 1. Scope and fixed budget

Candidate I replaces only Candidate H's rejected *filtered scale carrier*.
Candidate F's categorical exact records, owner election, and face-plane solve do
not change.  One `RG32Uint` texel contains two independent 32-bit scale slots.
The runtime budget remains four R0 reads, four R1 reads, and one scale read.

The carrier is a lossy statistical volume, not an averaged triangle.  At every
world phase texel and scale it describes a finite vertical extinction band.  It
has no angular chart index, so no direction boundary can select a different
record.

## 2. Exact 32-bit scale slot

```text
bits  0.. 6  h       occupied height / H, UNORM7
bits  7..12  theta   log2 horizontal optical thickness, 6 bits on [-8, 5]
bits 13..17  r       log2 vertical/horizontal projected-area ratio, 5 bits on [-5, 0]
bits 18..22  ax      signed x first harmonic, SNORM5 on [-0.75, 0.75]
bits 23..27  az      signed z first harmonic, SNORM5 on [-0.75, 0.75]
bits 28..31  class   accepted community-global RGB/material class
```

Thus two scales occupy exactly 64 bits.  The maximum height quantisation error
is `H/(2*127)` (4.63 mm for `H=1.176 m`).  The worst half-step multiplicative
errors are 7.42% for `theta`, 5.75% for `r`, and
`exp(sqrt(2)*0.75/31)-1 = 3.48%` for the harmonic factor.  These are sensitivity
bounds, not a claim that the model error fits inside them.

## 3. Direction-analytic equations

Let `u=(0,1,0)` be the authored growth axis and let a ray travel in unit
direction `d` from the global top plane `y=H`.  Exterior downward rays have

```text
v  = -dot(d,u) in (0,1]
q  = d - dot(d,u) u
g  = sqrt(dot(q,q) + r^2 v^2) * exp(ax*d.x + az*d.z)
x  = theta * g / v.
```

`g` is strictly positive for `r>0`, continuous at the vertical pole, and has a
first azimuthal harmonic without a chart or `atan2`.  The band is `[0,h]`; its
extinction coefficient is `kappa=theta/h` when `h>0`.  The path length is
`L=h/v`.  Therefore

```text
A(d) = -expm1(-x).
```

For `x>0`, define

```text
F(x) = 1/x - 1/expm1(x).
```

The conditional expected first-hit parameter from the global top plane is

```text
s(d) = (H-h)/v + (h/v) F(x),
P(d) = Q + s(d) d,
```

where `Q` is the continuously computed top-plane phase point.  Conditional
colour and premultiplied colour are

```text
Cg(d) = palette[class],
Cp(d) = A(d) Cg(d).
```

One continuous statistical lighting normal (never an exact face covector) is

```text
nraw = (r + (1-r)v) u - (1-r) q,
nstat = normalize(nraw).
```

It tends to `u` at the vertical pole and toward a view-facing, upward-biased
side normal at grazing incidence.  It may be used only for filtered lighting.

### Stable limits

For small `x`, use the fixed polynomial

```text
F(x) = 1/2 - x/12 + x^3/720 + O(x^5).
```

For saturated `x`, `F(x)=1/x+O(exp(-x))`; at `x>=16`, omitting the remainder
changes `F` by less than `1.13e-7`.  These are fixed selects, not iteration.

At the vertical pole, `q=0`, `g=r`, `x=theta*r`, and every output is finite and
azimuth independent.  As `v -> 0+`, `A -> 1`.  If `h=H`, `s` tends to the finite
horizontal mean free path `H/(theta*exp(ax*d.x+az*d.z))`.  If `h<H`, `s` grows
as `(H-h)/v`: the ray must travel unbounded distance before descending from the
top reference plane to the lower band.  At exactly `v=0`, a ray on `y=H` misses
when `h<H`; when `h=H`, the finite horizontal expression applies.  This is the
geometry's real projective grazing limit, not a numerical singularity to hide
with an elevation clamp.

## 4. Cook-side fit

For each texel and each scale, let the 65 filtered truth directions provide
coverage `Aj`, premultiplied colour `Cpj`, and conditional representative depth
`sj` where covered.  Candidate I fits the *unquantised* parameters by minimising

```text
sum_j rho((A(dj)-Aj)/0.08)
+ 2 sum_{j:Aj>0} rho((s(dj)-sj)/0.05m)
+ sum_j rho(maxChannel(A(dj) Cclass-Cpj)/0.06),
```

where `rho` is a fixed smooth-L1 loss and the class is one of the already
accepted sixteen palette entries.  Depth receives twice the unit weight because
representative position drove Candidate H's largest failure.  The parameters
are quantised exactly as Section 2, decoded, and only the decoded carrier is
scored.  No view is held out: all 65 are the required contract, and the report
must remain split by the four elevation rings, pole, and azimuth families.

The reference fitter uses deterministic bounded coordinate/gradient descent;
that is cook work only.  Runtime evaluation is the closed form in Section 3.

## 5. Continuity proof

The decoded record is independent of direction.  On `v>0`, dot products,
square root over a strictly positive argument, exponential, `expm1`, and the
finite-band expectation are continuous compositions.  `q -> 0` at the pole,
while `g -> r` independently of azimuth, proving a unique vertical limit.  Old
15/25/35/45/55/65/75-degree cells and their azimuth midplanes do not occur in
the address or equations.  Consequently both one-sided evaluations at every
old boundary converge to the same `(A,Cp,P,nstat)`.  Integer quantisation changes
the world-phase field between texels, but cannot introduce an angular boundary.

Continuity at the exact horizontal direction is governed by the extended
projective limit described above.  The finite world representative is not
continuous there when `h<H`, because the first hit escapes to infinity; neither
the real infinitely repeated community nor a correct codec has a finite limit
to preserve.

## 6. Fixed runtime cost

The filtered path performs one 64-bit texture load, integer unpack, roughly
35--45 scalar FMA-equivalent operations, one reciprocal, one square root, and
two exponential-family evaluations (`exp` and `expm1`, implementable from one
`exp2` result plus arithmetic).  A transcription can decode and consume one
scale slot at a time, keeping approximately 8--11 additional scalars live.
There is no loop, direction selection, divergence by owner, new binding, pass,
barrier, dispatch, or runtime geometry.  The physical profile-read count and
48.750 MiB allocation remain unchanged.

## 7. Capability boundary and counterexamples

This codec is exact for a homogeneous, locally axially distributed extinction
band with a first horizontal harmonic.  It is not universal for arbitrary
triangle soups.  Its outputs obey three severe invariants:

1. all 65 conditional colours at one phase lie on one palette colour ray;
2. angular extinction has only an axial term plus one `x/z` harmonic;
3. all representative depths arise from one ground-to-`h` homogeneous band.

Open plume branches can expose violet and green owners in different directions;
horizontal branches can create multi-lobed azimuthal visibility; separated
vertical layers can have the same coverage but different first-hit depths;
overlapping species can change both colour and height ordering with direction.
All violate those invariants.  Therefore direction continuity is proved, but
arbitrary-soup fidelity is an empirical gate and may legitimately be RED.  A
RED result rejects this 32-bit extinction family; it does not authorise hiding
the error, blending exact owners, adding taps, or calling the constant field a
solution.


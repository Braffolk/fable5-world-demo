# Direction-analytic filtered ground-cover carrier: designer brief

Date: 2026-07-24  
Status: **RED and superseded by Candidate K**; no runtime authorisation

Post-gate note: the fitted vertical-extinction construction below failed both
coverage and colour fidelity by large margins and its permissive vertical-band
oracle rejected the representative-position family independently of the
optimizer.  A local homogeneous column cannot reproduce the long horizontal
path integral, plume/stem multimodality, or phase-dependent occlusion of the
accepted community.  It is retained as a rejected derivation, not as an
existence proof.  The active replacement is
`GRASSPROFILE2-CANDIDATE-K-BACKGROUND-MEASURE-FE.md`, which first removes the
unnecessary fractional-surface requirement imposed by this brief.

## 1. What failed, and what did not

Candidate H's background-preserving fractional overlay remains valid.  The
joint sixteen-entry Calamagrostis palette is GREEN (`DeltaE00` overall p95
`0.741`, p99 `1.807`).  The reserved payload bit, background depth/id retention,
nine-read budget, and two-scale 64-bit filtered record remain reusable.

The frozen unit-square filtered carrier selected one independently baked view
categorically.  It is RED at every elevation and azimuth boundary.  At the
critical 45-degree boundary for radius 16 it measures:

```text
abs(delta A) p95/p99           = 0.4706 / 0.6196
premul RGB delta p95/p99       = 0.3012 / 0.4107
transverse error p95/p99 @ 2m = 129.24 / 165.84 pixels
largest connected exceedance  = 91.14%
```

The mechanism is structural.  Spatially filtering two independent first-hit
charts does not make them the same world-space measure.  At the categorical
view switch they disagree coherently about coverage, radiance, and
representative position.  Normalized scale weights do not repair a
single-contributor site because the weight cancels from
`sum(w A P) / sum(w A)`.

## 2. Non-negotiable runtime contract

The replacement applies only to the unresolved/filtered scale carrier.  The
near exact record remains categorical.  Do not angularly blend exact depth,
owner, normal, or mark records; the existing no-common-owner counterexamples
and PCF rejection still apply.

The replacement filtered carrier must retain:

```text
one physical filtered-profile read
64 bits total for both spatial scales
four R0 + four R1 + one scale = nine profile reads total
no additional resident bytes
no loop, march, candidate list, pass, barrier, dispatch, or runtime geometry
no runtime per-species work
exterior-camera contract; the cover-box interior may fade
```

An `RGBA32Uint` scale record is not part of this design.  It is a documented
`+16 MiB` fallback only if the user explicitly reopens memory after a proved
64-bit fidelity failure; the designer may not silently depend on it.

## 3. Two independent gates are mandatory

Continuity alone is vacuous: a direction-independent constant field has no
boundaries and would pass.  Fidelity alone reproduces the failed independent
charts.  A carrier is admissible only if both gates pass.

### 3.1 Continuity gate

Keep the frozen Candidate-H boundary gate and its exact world/epipolar pairing.
Evaluate every 25/45/65-degree elevation boundary and every azimuth midpoint at
both proposed scales.  Equal-address pairing remains forbidden.  The metrics
and frozen limits remain:

```text
abs(delta A):             p95 <= 0.08, p99 <= 0.20
premul RGB max delta:     p95 <= 0.06, p99 <= 0.15
transverse displacement:  p95 <= 0.75 px, p99 <= 1.5 px
connected exceedance:     < 1%
```

For a genuinely direction-analytic world-anchored carrier the limiting jump at
an old lattice boundary should be zero by construction, not merely small after
fitting.  The finite numerical gate exists to catch address, frame, packing,
and quantization violations of that construction.

### 3.2 Per-direction fidelity gate

The truth target is the already computed per-direction filtered atom field for
each of the 65 accepted baked directions, evaluated separately at both spatial
scales.  Those atoms are invalid as a categorical runtime carrier but remain
the correct measurement of what each direction sees.

For every truth direction and periodic phase, compare the analytic carrier at
the identical live direction and world-corresponded phase.  Freeze these limits
before fitting or inspecting the result:

```text
abs(A_hat - A_truth):                    p95 <= 0.08, p99 <= 0.20
premul RGB max-channel error:            p95 <= 0.06, p99 <= 0.15
representative world-position error:     p95 <= 0.05 m, p99 <= 0.10 m
connected support exceeding a p99 limit: < 1%
```

Position is the full Euclidean representative-point error.  At the identical
direction it is predominantly an along-ray depth error, so the transverse-only
boundary metric would be vacuous here.  Report signed depth bias and angular
families separately; do not average a grazing failure into a top-view success.
Weight the fit loss toward representative depth/position because it was the
largest measured boundary failure and drives parallax, compositing, and
translation stability.  Coverage and premultiplied colour remain independently
binding, so a depth-perfect constant cannot pass.

The numeric position limits are a codec-quality contract rather than a claim
of exact arbitrary-soup reconstruction.  If they prove inconsistent with the
visual footprint, revise them only through a new preregistered gate with an
explicit screen-space/parallax derivation--never after inspecting a fit.

## 4. Constructive candidate: fitted vertical extinction profile

Use one world-anchored, direction-independent analytic profile per phase texel
and spatial scale.  Let `u` be the authored growth axis, `h` the occupied band
height, `kappa` its fitted extinction density, and `eta` an orientation/
erectness coefficient.  For an exterior ray `d`, compute a continuous projected
extinction

```text
lambda(d) = kappa * g(dot(d,u), eta),
L(d)      = analytic ray length through the finite band,
A(d)      = 1 - exp(-lambda(d) L(d)).
```

One suitable axially symmetric projected-area family is

```text
g(mu,eta) = sqrt(max(g_min^2, 1 - eta mu^2)),  mu = dot(d,u),
```

but the exact positive family is a fit decision, not frozen by this example.
For homogeneous extinction along the clipped interval, the conditional expected
first-hit distance from its entry is analytic:

```text
E[s | hit] = 1/lambda - L/(exp(lambda L)-1).
```

Use stable small-`lambda L` and saturated limits in the mathematical reference;
the eventual shader transcription must contain fixed branch/select algebra, not
iteration.  The representative point is `P_entry + E[s|hit] d`, hence coverage
and position are continuous in world phase and direction wherever the exterior
ray-band intersection is continuous.  An azimuthal harmonic may be introduced
only if the measured Estonia community residual proves axial symmetry
insufficient and it still fits the same 32 bits per scale.

A provisional 32-bit scale slot may allocate fields for band height, extinction,
erectness/anisotropy, and the accepted four-bit colour class.  Exact bit widths
must be derived from fit sensitivity and quantization error; the example
`height8 + density8 + anisotropy6 + class4 + 6 spare` is a budget witness, not a
quality result.  Two slots form the existing 64-bit scale texel.

The filtered lighting normal is a statistical profile output, not a geometric
face covector.  It may be derived continuously from the growth axis,
anisotropy, and live direction.  It must never feed the Candidate-F exact-stage
face-plane solve or masquerade as an exact triangle normal.

The cook fits each texel/scale profile jointly against all 65 directional truth
atoms.  The fit loss and held-out report are exactly the fidelity gate above;
the old RED analyzer supplies the truth data and epipolar/world utilities.  A
GREEN continuity result without GREEN fidelity is rejected as a constant-field
degeneracy.  A GREEN fidelity result with a categorical chart switch is rejected
as Candidate H repeated under a new name.

This construction proves that direction continuity and the one-read analytic
runtime form are compatible.  It does **not** yet prove that arbitrary triangle
soups achieve the frozen fidelity limits in 32 bits per scale.  Only the joint
fit/gate can establish that empirical codec claim.

## 5. Required deliverable from the paper designer

Return a complete mathematical codec, not shader code:

1. exact per-slot parameters and bit allocation within 32 bits;
2. positive continuous formulas for `A`, representative position, colour, and
   statistical normal for every exterior direction, including vertical and
   grazing limits;
3. the cook-side objective which fits all 65 directional truth fields without
   direction-cell selection;
4. quantization and stability bounds;
5. proof that the address and output are continuous across every old lattice
   boundary;
6. predicted fixed ALU/register cost and confirmation of one physical read;
7. counterexamples likely to defeat the model, especially open plumes,
   horizontal branches, overlapping species, and non-axial triangle soups.

No runtime work begins until this codec is explicit and both preregistered gates
are GREEN.

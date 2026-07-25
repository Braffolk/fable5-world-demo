# Candidate Q: rank-three scale-continuous event sheets

Date: 2026-07-24  
Status: favourable rank-three capacity oracle materially RED; Candidate Q parked; runtime not authorised

## 1. Why this is a new upstream premise

Candidate P made one representation choice look like a resource law:

```text
three regions * two scales * direct RGBA4444 = 96 bits,
leaving only enough topology for two events.
```

The corrected adaptive gate now shows that halving the failed standing cell is
not a sufficient remedy: its favourable joint affine-rank-two oracle remains
RED at `0.135882/0.110044` coverage/RGB p95.  This bounded result does not
exclude asymptotic convergence under arbitrarily fine subdivision; it shows
that one affordable `2x` refinement did not remove the immediate local
rank-two capacity failure.  Four independently changing botanical populations
can have local response dimension three.

The direct-colour premise is not fixed.  Candidate H already measured that a
joint 16-entry palette preserves authored first-hit plume and stem/leaf colour
with `DeltaE00` p95 `0.74134`.  That result does **not** prove that filtered
region chromaticities fit the palette, but it proves that four class bits are a
credible packing hypothesis worth gating.  Candidate Q spends the recovered
bits on a third event rather than on more directions, reads, or bytes.

The contract remains:

- any finite marked triangle soup may be cooked, but a soup outside the fixed
  codec's measured capacity fails rather than silently degrading;
- exterior-of-cover-box views only; exact horizontal exterior rays fail the
  slab entry and the direction field owns the `e -> 0+` in-cover limit;
- one `128`-bit record operation for each of three phase vertices per affine
  layer, hence three reads for one layer and six for two;
- the same `64 x 64 x 256 x 16 B = 16 MiB` P allocation;
- no loop, march, candidate list, dependent lookup, geometry, extra pass,
  extra binding, or resident-memory increase.

## 2. Continuous domain

Let `lambda_i(q)`, `i=0,1,2`, be the barycentrics of the live periodic phase
triangle.  Let `mu_l(d)`, `l=0,1,2`, be the barycentrics of the live canonical
direction triangle.  Let

```text
u(rho) = clamp(log(rho/rho4)/log(rho16/rho4), 0, 1)
```

be the continuous physical-footprint coordinate.  It comes from the analytic
screen differential, never from distance bands, a categorical mip, or a
representative grass depth.

## 3. Three ordered scale-continuous sheets

For event sheet `k in {0,1,2}`, footprint endpoint `j in {0,1}`, and canonical
direction vertex `l`, store one signed scalar `g[k,j,l]`.  Decode

```text
g_k(mu,u) = sum_l mu_l * ((1-u)*g[k,0,l] + u*g[k,1,l]),
s_k       = smoothstep(-1, +1, g_k).
```

The cook enforces at every direction vertex and both footprint endpoints

```text
g_2 <= g_1 <= g_0.
```

Convex interpolation in direction and footprint preserves the inequality.
Monotonicity of `smoothstep` gives `s_2 <= s_1 <= s_0`, so

```text
W_0 = 1-s_0,
W_1 = s_0-s_1,
W_2 = s_1-s_2,
W_3 = s_2,
W_r >= 0,                    sum_r W_r = 1.
```

The decoder therefore cannot produce negative coverage, negative radiance,
or colour outside the convex hull of its four positive region states.  A sheet
may leave a cell by saturating to one sign; no categorical topology word is
required.

## 4. Positive states and exact 128-bit witness

For region `r in {0,1,2,3}` store one colour class shared across the two
footprint endpoints and one five-bit coverage at each endpoint:

```text
A_(r,j) = a_(r,j)/31,
C_(r,j) = palette_j[class_r],
M_(r,j) = (A_(r,j)*C_(r,j), A_(r,j)),
M_r(u)  = (1-u)*M_(r,0) + u*M_(r,1).
```

The two 16-entry palettes are distinct but their class indices are aligned by
the cook.  Thus class `c` may have different endpoint chromaticities while one
four-bit state class remains sufficient.

One record evaluates

```text
F_v(d,u) = sum_(r=0)^3 W_r(d,u) M_r(u),
M_Q(q,d,u) = sum_(i=0)^2 lambda_i(q) F_(v_i)(d,u).
```

An exact packing witness is

```text
4 regions * (class4 | coverage5@rho4 | coverage5@rho16) = 56 bits
3 sheets * 2 scales * 3 direction vertices * signed4    = 72 bits
total                                                    = 128 bits
```

The external asset header supplies the format version.  One admissible signed
four-bit sheet decode is

```text
g(q) = (q-7.5)/2,       q in [0,15],
```

with step `0.5` and range `[-3.75,+3.75]`.  This quantisation is only a packing
witness; it must pass the complete image gate and is not assumed adequate.

Five-bit coverage is binding: one quantisation step is `1/31 = 0.032258`,
below the `0.06` translation limit.  Four-bit coverage has step `1/15`, above
that limit, and is not silently substituted.

## 5. Structural continuity

Each sheet is one P1 scalar field on the canonical direction mesh.  Incident
direction triangles duplicate the sheet values at their shared vertices
bit-identically.  Their **weight** restrictions are therefore identical for
every footprint `u`; the azimuth seam reuses the same vertices, and every
pole-incident record reuses the same canonical pole sheet value.

That fact alone does **not** make the complete decoded measure continuous.
The four positive states in section 4 are cell-local constants, so two cells
with identical edge weights but different states generally decode different
edge measures.  The cook must impose, for every canonical direction edge and
every footprint `u`,

```text
sum_r W_r^left(d,u)  M_r^left(u)
  =
sum_r W_r^right(d,u) M_r^right(u).
```

Bit-identical ordered region states in the two incident records are a simple
sufficient condition, but on a connected active sheet ordering that condition
propagates the same four states beyond one cell and is materially restrictive.
More general state/sheet equivalences are allowed only if the cook proves the
complete quantised edge polynomial identical; sheet continuity may not be
used as a proxy.  The independent rank-three capacity oracle deliberately
ignores this condition and is therefore still a favourable necessary bound,
never a structural-continuity witness.

Every phase triangle reuses the same complete angular function at each shared
phase vertex.  The positive barycentric sum consequently has identical phase-
edge restrictions.  Subject to the complete direction-edge constraint above,
the field is jointly `C0` in phase, direction, and footprint.  This proof does
not depend on an image filter or TAA.

Beyond `rho16`, Candidate P's continuous contraction to the cooked phase mean
`M_inf(d)` remains unchanged.  The endpoint is phase independent, so it cannot
carry the one-direction particle crawl to infinity.

## 6. Capacity relation

At one footprint endpoint and one phase vertex,

```text
x(d) = M_0
     + s_0(d)*(M_1-M_0)
     + s_1(d)*(M_2-M_1)
     + s_2(d)*(M_3-M_2).
```

Every response therefore lies in an affine subspace of dimension at most
three.  Candidate P's rank-two RED is not a lower bound against this family.
Conversely, the best arbitrary affine rank-three approximation at each phase
vertex is a strict favourable superset: it ignores positivity, sheet shape,
ordering, canonical edges, packing, and palette classes.  A material RED in
that oracle rejects Candidate Q before any nonlinear sheet fit.

Because sheets may move with footprint, the strongest cheap prerequisite is
one independent affine rank-three oracle in `R4` for each physical endpoint.
A shared-coordinate affine rank-three oracle in `R8` is also reported; GREEN
there would support the even simpler scale-independent-sheet ablation, while
RED there does not reject the written scale-continuous model.

## 7. One bounded gate

No new rays are required.  Reuse the corrected logical-polar adaptive truth,
the original standing/grazing truth, the three exact phase sites, and every
`1--4.5 mm` translation.

Run in this order:

1. fit the existing `64 x 64` phase coefficients with the same complete
   static-and-translation objective and retain the unrestricted-angular phase
   control;
2. fit favourable independent affine rank-three endpoint oracles per phase
   vertex and score their combined exact output;
3. report the shared-coordinate joint rank-three oracle as a diagnostic;
4. stop immediately if the independent endpoint oracle is materially RED;
5. only if it is GREEN, fit unlimited-precision ordered scale-continuous sheets
   and four positive states with canonical shared edges;
6. only if that is GREEN, fit the two aligned 16-entry filtered-state palettes,
   quantise exactly to the 128-bit witness, and rescore.

Frozen limits remain

```text
coverage p95/p99              <= 0.08 / 0.20
premultiplied RGB p95/p99     <= 0.06 / 0.15
largest connected exceedance  < 1%
translation p95               <= 0.06, connected < 1%.
```

The first capacity run and the actual ordered-sheet run are the track's two
maximum real failures.  A RED independent rank-three result parks immediately.
A GREEN favourable bound is not implementation authority.

## 8. Performance rationale

Relative to Candidate P, Candidate Q changes only fixed ALU after each already
required `RGBA32Uint` read: one additional sheet interpolation, one additional
`smoothstep`, and one additional positive-state accumulation.  It adds no
memory transaction, dependent chain, binding, synchronization, divergence, or
pass.  The record is consumed before loading the next phase vertex, so the
extra live state is one scalar sheet value and one running measure.  Register
count and occupancy still require a real trace after visual acceptance; they
are not assumed from source syntax.

No runtime or shader file may change until every unlimited-precision,
canonical-edge, packing, handoff, and exterior-domain gate above is GREEN.

## 9. Bounded capacity result -- RED (2026-07-24)

The sole authorised capacity run reused both exact truth caches and generated
zero new rays.  It used the same globally fitted periodic phase coefficients,
the same three exact phase sites, and every `1--4.5 mm` translation.  At each
phase vertex it searched `240` threshold-normalised candidate affine planes
(raw/scaled PCA, high-p IRLS, raw/scaled leave-one-out, and every four-sample
affine plane).  A synthetic exactly affine rank-three field reproduced to
`4.77e-7` before the real run.

The favourable independent `R4` rank-three endpoint oracle is materially RED:

```text
static cases                          72 / 252 GREEN
translation cases                  1727 / 4032 GREEN
worst coverage p95 / p99          0.320523 / 0.428595
worst premultiplied-RGB p95 / p99 0.458205 / 3.027951
worst connected exceedance        0.158447
translation coverage p95          0.227268
translation RGB p95               0.688661
translation connected exceedance  0.045410
```

The failure is not a marginal threshold miss.  Against limits
`0.08 / 0.20`, `0.06 / 0.15`, and `<0.01`, the worst static p95 errors are
about `4.0x` and `7.6x` over, the static connected region is `15.8x` over,
and translation RGB p95 is `11.5x` over.  Every grazing static case fits;
the standing cases provide the material rank-capacity counterexamples in both
the original-slope and corrected logical-polar datasets.  Because the oracle
allows arbitrary independent endpoint coordinates and ignores positivity,
ordering, canonical-edge continuity, packing, and palette restrictions, an
ordered three-sheet codec cannot recover the omitted response dimension.

The shared-coordinate joint `R8` rank-three diagnostic is also RED at
`0 / 252` static and `1498 / 4032` translation cases, with worst
coverage/RGB p95 `0.260603 / 0.288520`.  This is supporting evidence only;
the independent endpoint result is the rejecting bound.  The unrestricted
angular phase control remains `252 / 252` GREEN statically and
`3979 / 4032` in translation, localising the failure to angular response
capacity rather than the phase fit.

The run consumed `1098.45 s`; the complete track consumed `1739.84 s`, within
the preregistered `3600 s` budget.  The report is

```text
data/work/groundcover-candidate-q-rank3-capacity/
  e3e0a4175b151b89/2060286bc07c3dc7/report.json
SHA-256 5e5ec2abe57b4118a8744ed2fe1d6aa1309ef6e2dfc0fb1ddf342b576672455e
```

Candidate Q is parked.  Sheet fitting, quantisation, packing, shader, and
runtime work are prohibited.  An objective resume condition is a new fixed-
cost representation whose favourable capacity superset contains the measured
standing response without adding a physical read or resident byte; tuning or
refitting this rank-three family is not such a condition.

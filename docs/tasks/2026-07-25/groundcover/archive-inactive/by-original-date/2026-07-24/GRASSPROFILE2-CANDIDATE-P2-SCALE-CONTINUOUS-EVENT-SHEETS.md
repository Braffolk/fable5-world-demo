# Candidate P2: scale-continuous canonical event sheets

Date: 2026-07-24  
Status: pure-math fallback; no runtime or shader work authorised

## 1. Exact scope

This is the bounded fallback for one specific gate outcome:

```text
A  64x64 phase basis                         GREEN
B  independent angular two-flats per scale  GREEN
C  one shared joint R8 angular two-flat      RED
```

That outcome says that two angular events are adequate at each physical
footprint, but forcing the *same two coordinates* to explain both footprints
is not.  P2 removes only that false coupling.  It retains two events, one
`128`-bit record at each phase-vertex/direction-cell address, three physical
record operations per affine layer, the same atlas allocation, and fixed
straight-line evaluation.

The cook truth may come from any finite marked triangle soup.  Fidelity is
capacity-gated; an input outside the fixed codec fails the cook rather than
silently degrading.  Camera-inside-cover quality remains excluded.  This note
adds no ray directions or truth intersections and does not alter Candidate F,
the bridge, the background-compositing protocol, or the far phase mean.

## 2. Domain and scale coordinate

Phase and direction use Candidate P's periodic `64x64` triangular phase mesh
and canonical descending-hemisphere direction mesh.  Let

```text
lambda_i(q), i=0,1,2  = phase-triangle barycentrics
mu_l(d),     l=0,1,2  = direction-triangle barycentrics
u(rho)                = clamp(log(rho/rho4)/log(rho16/rho4), 0, 1).
```

`rho4` and `rho16` are the two physically integrated footprint endpoints.  No
representative depth selects `u`; it comes from the analytic screen
differential in the single terrain chart already specified by Candidate P.

## 3. Why the record stores sheet logits

Interpolating chord endpoints and a separate width is not by itself a
conforming construction: two incident direction triangles can agree on the
crossing point yet induce different smoothstep slopes on their shared edge.
P2 stores the equivalent but stronger object: the values of each affine sheet
coordinate at the three canonical direction vertices.  A zero set is still a
chord, and `|g|<1` is still its transition band, but both the endpoint and the
edge-restricted width are now consequences of one shared scalar trace.

For event sheet `k in {0,1}`, footprint endpoint `j in {0,1}`, and direction
vertex `l`, store a signed logit `g[k,j,l]`.  At live direction and scale,

```text
g_k(mu,u) = sum_l mu_l * ((1-u)*g[k,0,l] + u*g[k,1,l])
s_k       = smoothstep(-1, +1, g_k).
```

Thus the chord `g_k=0`, its two edge crossings, and its width all vary
continuously with log footprint.  A sheet may enter or leave a cell through a
vertex, disappear by having one sign everywhere, or merge with the other
sheet.  There is no categorical scale topology switch.

The cook enforces, at every direction vertex and both scale endpoints,

```text
g_1 <= g_0.
```

Convex interpolation preserves this inequality throughout the full
direction-triangle by scale interval.  Since `smoothstep` is monotone,
`s_1 <= s_0`, and the ordered non-negative partition is

```text
W_0 = 1-s_0,
W_1 = s_0-s_1,
W_2 = s_1,
W_r >= 0,                 sum_r W_r = 1.
```

## 4. Positive scale-varying states

Each of the three ordered regions has one positive premultiplied state at
each physical footprint endpoint.  Two separate community-global 16-entry
linear-RGB palettes are permitted, one fitted at `rho4` and one at `rho16`.
For region `r` and endpoint `j`, decode

```text
A_(r,j) = a_(r,j)/31,
C_(r,j) = palette_j[class_(r,j)],
M_(r,j) = (A_(r,j)*C_(r,j), A_(r,j)),
M_r(u)  = (1-u)*M_(r,0) + u*M_(r,1).
```

One phase-vertex record evaluates

```text
F_v(d,u) = sum_r W_r(d,u) * M_r(u),
```

and the live phase result is

```text
M_P2(q,d,u) = sum_i lambda_i(q) * F_(v_i)(d,u).
```

For footprints beyond `rho16`, Candidate P's existing smooth contraction is
retained:

```text
M = (1-h(rho))*M_P2(q,d,1) + h(rho)*M_inf(d).
```

`M_inf` is the cooked phase mean and is phase independent at the endpoint.

## 5. Exact 128-bit witness

The sheet logit uses a fixed monotone signed 6-bit decode, shared by the whole
community.  One admissible decode is

```text
gamma(q) = (q-31.5)/4,  q in [0,63],
```

giving a transition-coordinate step of `0.25` and saturating range
`[-7.875,+7.875]`.  The capacity gate, not intuition, decides whether that
quantisation is adequate.

```text
3 regions * 2 scale endpoints * (coverage5 | colourClass4)  = 54 bits
2 sheets  * 2 scale endpoints * 3 direction vertices * 6   = 72 bits
reserved/version, required to decode as zero                =  2 bits
total                                                       = 128 bits
```

Concrete layout:

```text
bits   0..26   three rho4 state words, each A5|class4
bits  27..53   three rho16 state words, each A5|class4
bits  54..89   sheet 0's six signed-logit indices
bits 90..125   sheet 1's six signed-logit indices
bits 126..127  zero
```

No topology word is required.  The global direction mesh supplies vertex
ordering; sign patterns of the affine sheets encode absent chords, crossings,
and saturated regions.  Ordering is a cook constraint on the integer logits,
not a runtime branch.

The P field remains exactly

```text
64^2 phase vertices * 256 direction records * 16 B = 16.000 MiB.
```

The two 16-entry endpoint palettes are at most a few hundred header bytes and
fit inside Candidate P's already-budgeted `<0.125 MiB` constants allowance.
One `RGBA32Uint` operation is consumed for each of the three phase vertices:
three reads for one layer, six for two.  Evaluation is a fixed unrolled set of
lerps, dot products, smoothsteps, and positive accumulations.  There is no
loop, candidate, dependent lookup, march, pass, binding, or resident-byte
increase.

## 6. Continuity, positivity, and ordering proofs

### 6.1 Direction

Each sheet is a continuous P1 scalar field on the direction mesh.  For every
shared direction edge, both incident records duplicate the two endpoint
logits bit-identically at both footprint endpoints.  Their restrictions

```text
(1-t)*g_at_edge_vertex_0 + t*g_at_edge_vertex_1
```

are therefore identical for every `u`.  The seam reuses the same canonical
vertices and all pole-incident cells reuse the same pole values.

State correspondence is part of the canonical edge solve, not a later
reconciliation: for each shared edge and scale endpoint, every region state
whose weight is nonzero anywhere on that edge is one shared variable and is
duplicated bit-identically by both cells.  Consequently both incident
decoders have the same `W_r`, the same active `M_r`, and the same output on
the entire edge.  Interior-only states need not be shared.  This is the exact
edge conformance condition the cook must validate after packing.

### 6.2 Scale

`g_k` and every decoded state are affine in `u`; smoothstep and finite sums
are continuous.  Event birth, death, or merge is therefore continuous even
when the zero-chord topology changes.  The far contraction agrees with P2 at
`h=0` and with the continuous `M_inf` at `h=1`.

### 6.3 Phase

A phase vertex owns one complete direction-by-scale function.  Adjacent phase
triangles reuse the same vertex indices.  Their barycentric restrictions on a
shared edge are identical, so their positive sums agree.  Direction and scale
continuity is preserved by phase interpolation.  The result is jointly `C0`
in `(q,d,rho)`.

### 6.4 Positivity and ordering

The endpoint states obey componentwise

```text
0 <= M.rgb <= M.a <= 1.
```

Scale interpolation, the ordered `W` partition, phase barycentrics, and far
contraction are all convex combinations.  They preserve that inequality and
cannot manufacture negative coverage or colour outside coverage.

The integer endpoint inequalities `g_1<=g_0`, together with monotonic decode,
direction/scale convex interpolation, and monotonic smoothstep, prove
`s_1<=s_0` everywhere.  Hence no event crossing can create a negative middle
weight.

## 7. What this codec does and does not contain

P2 strictly removes the joint-R8 rank restriction: the event sheets and all
three region states have independent values at the two physical footprint
endpoints.  It does *not* guarantee that two independent fits can be joined.
The following are honest capacity failures, not implementation bugs:

- the two endpoint fits require event sheets to cross or exchange order;
- a cell needs more than two ordered transitions, a branching transition, or
  a closed transition island not expressible by a P1 zero-line;
- edge conformance forces incompatible state values between adjacent cells;
- appearance changes materially inside a plateau rather than at its two
  events;
- physical appearance as a function of footprint is not captured by the
  affine-state/affine-logit path between the endpoints;
- the fixed logit quantiser collapses an important narrow event;
- endpoint state chromaticities are not represented by their 16-entry
  palettes; or
- the phase64 basis remains inadequate after its coefficients are jointly
  optimised.

Valid degeneracies are explicit: zero coverage makes the colour class
irrelevant and it is canonicalised to zero; identical sheets give `W_1=0`;
a saturated sheet removes a transition; and an event may enter/leave through
a direction vertex without a discontinuity.  Camera-inside-envelope views
remain faded and outside the quality contract.

## 8. Coverage and palette determination

### 8.1 Four-bit coverage is not certified by the known gates

Nearest 4-bit UNORM coverage has pointwise error at most

```text
1/(2*15) = 0.033333.
```

That is below the static `0.08` p95 limit, but a translation compares two
positions.  Opposite quantisation errors can contribute

```text
2/(2*15) = 1/15 = 0.066667,
```

which exceeds the frozen `0.06` translation limit.  This is attainable, for
example by two true coverages on opposite half-step ties whose rounded change
differs from the true change by one complete 4-bit step.  Spatial and angular
convex weights may make the actual data pass, but no existing result proves
that.  Four-bit coverage is therefore an optional measured ablation, not the
binding codec.

Five-bit coverage costs only six more bits across all states and still fits
the same record.  Its worst quantisation contribution to a translation is

```text
1/31 = 0.032258 < 0.06.
```

P2 consequently freezes A5 unless a complete quantised-output gate later
proves A4 and the user explicitly chooses the eight spare bits it would free.

### 8.2 Candidate H's palette GREEN is useful but not sufficient

The existing Candidate-H result proves that one 16-medoid palette preserves
*individual first-hit authored colours* (`DeltaE00 p95=0.74134`).  P2's state
colour is different: it is the unpremultiplied chromaticity of a filtered
positive measure and may be a convex mixture of plume, stem, and leaf colours.
A palette that quantises every source colour accurately need not contain the
interior of their convex hull.  The H result therefore cannot be cited as a
proof that P2's endpoint states fit four class bits.

The two 16-entry endpoint palettes must be fitted only after the unlimited-
precision conforming P2 field exists.  The population is the six endpoint
states, weighted by their integrated `W_r*A_r` use over every frozen static
and translation case.  Fit in the actual premultiplied linear-RGB output
objective; DeltaE is diagnostic only.  Palette16 is accepted only if the
*complete reconstructed images after A5, class4, and logit6 packing* retain
the unchanged RGB p95/p99 and translation gates.  No current measurement
answers that question.  This is a required finite gate, not a reason to add a
read or enlarge the record.

## 9. Capacity gate using the existing truth

No new camera rays are needed.  Reuse the corrected exact-BVH point pages,
held-out directions, phase interiors, and translation sequences already bound
to Candidate P.  Intermediate physical footprints are obtained by applying
the same exact periodic fractional-box integral to those point pages; this is
filtering existing truth, not generating new intersections.

Run one bounded nested gate:

1. **Prerequisite:** confirm the globally optimised results `A=GREEN`,
   `B=GREEN`, `C=RED`.  P2 is not the response to a sigma4-only or phase-basis
   failure.
2. **Unlimited conforming P2:** jointly optimise phase coefficients, the two
   real-valued ordered sheets, positive RGBA endpoint states, and canonical
   shared-edge variables.  Score endpoints plus deterministic log-footprint
   interiors (at minimum `rho=6,8,12`) produced from the existing point pages.
3. **Joint product:** score mid-phase points at held-out directions and all
   `1--4.5 mm` translations at those same directions and footprint interiors.
   A scale-axis-only or phase-axis-only pass is not evidence.
4. **Quantised record:** impose the exact A5/class4/logit6 record, fit the two
   endpoint palettes, and rescore.  Report A4 as a non-binding ablation only.
5. **Far limit and handoff:** retain Candidate P's phase-mean, bridge, and
   `R/B1/B2/U` fidelity/continuity gates unchanged.

The frozen image limits remain

```text
coverage p95/p99              <= 0.08 / 0.20
premultiplied RGB p95/p99     <= 0.06 / 0.15
largest connected exceedance  < 1%
translation p95               <= 0.06, connected < 1%.
```

Continuity is structural, not sampled.  Fidelity over the continuous scale
interval must nevertheless be certified.  On each fixed phase/direction
sample, the P2 prediction is piecewise polynomial in `u` with breakpoints only
where a sheet enters/leaves smoothstep saturation.  The fractional-box truth
from the retained point page has known overlap breakpoints.  An offline
interval branch-and-bound over their union can bound per-channel error between
the reported interior scales; it adds no runtime work and prevents a narrow
scale failure being hidden between samples.

GREEN means the fully packed P2 output passes every frozen case.  RED at the
unlimited conforming stage parks the two-sheet correspondence premise.  RED
only after packing identifies logit or state quantisation; it does not license
extra reads or bytes.  In either case, do not proceed to runtime merely because
the independent endpoint fits were GREEN.

## 10. Conclusion

If A and B are GREEN while C is RED, P2 is the smallest mathematically direct
repair: footprint is allowed to move the two event sheets and their positive
states, rather than being forced through one shared R8 plane.  It preserves
all frozen runtime currencies.  Its conforming scalar-sheet form gives exact
joint `C0` continuity and an explicit positivity/order proof.

The exact same-budget witness uses A5, not A4.  A4 has a concrete worst-case
translation error above the known limit.  Sixteen colour classes remain
plausible but unproved: Candidate H measured the wrong population for this
question, so the complete packed P2 gate must decide it before any shader or
format transcription.

# Candidate P: canonical event-function field

Date: 2026-07-24  
Status: fixed 240-cell direction mesh RED; adaptive same-count partition audit only; runtime not authorised

## 1. Inspectable outcome and frozen scope

The next inspectable product is not another shader variant.  It is one
content-addressed offline reconstruction of the accepted Calamagrostis
community which either proves or refutes this exact representation.  Only a
GREEN result proceeds to Fable's adversarial review and then runtime
transcription.

Candidate P targets the unresolved positive foreground measure

```text
M = (A*C_r, A*C_g, A*C_b, A)
```

over the real terrain/background.  It never emits a filtered grass depth,
normal, owner, or fictitious surface.  The resolved branch remains Candidate
F's categorical face-plane query.  P is designed specifically to replace the
far sampled-view reconstruction which causes directional stretching, wrong
view sectors, camera crawl, and particle-like minification.

The frozen limits are:

- no loop, march, traversal, candidate list, runtime geometry, new pass,
  barrier, or dispatch;
- at most nine physical profile texel operations in any query shape;
- no resident-memory increase over `48.750 MiB`;
- exterior descending rays, including the pole and the one-sided horizon
  limit; camera-inside-cover quality remains outside the contract;
- one cooked community may contain any finite marked triangle soup and any
  number of species.  The cook must fail rather than silently lower fidelity
  when that community is outside the fixed codec's measured capacity.

## 2. Upstream frame correction retained for the eventual transcription

The current resolved path evaluates the predictor in the tangent frame at the
opaque terrain point, re-evaluates a different tangent at a provisional grass
point for correction, and evaluates a third tangent at the elected grass point
for final lifting.  Even if every local formula is individually correct, these
three frames do not describe one affine conjugation.  On curved terrain their
difference changes with the elected node and camera, which is a direct source
of uphill slicing and residual swim.

The replacement uses one affine terrain chart for the entire query.  The
normative anchor `O` is the unmodified opaque-background point elected for the
current pixel before the ground-cover query starts.  It is computed once and
is never replaced by a provisional or elected grass point.  Let `G(O)` be its
height and

```text
n_g = (-G_x(O), 1, -G_z(O)).
```

The chart and its inverse are

```text
X_a.xz = X_w.xz,
X_a.y  = X_w.y - G(O) - gradG(O) dot (X_w.xz-O.xz),

X_w.xz = X_a.xz,
X_w.y  = G(O) + gradG(O) dot (X_a.xz-O.xz) + X_a.y.
```

Its Jacobian is constant.  The camera ray, all direction addressing, every
Candidate-F seed/corrector plane solve, the final point, and the normal
inverse-transpose use this same chart.  On affine terrain the construction is
exact.  On curved terrain the sole approximation is the explicit tangent
remainder `|delta G| <= kappa*r^2/2`; there is no hidden change of frame inside
one query.  Because `O` follows the current camera ray across continuous
terrain, this trades the old discrete frame-switching slices for a bounded,
smooth camera-dependent curvature bias.  A world-cell anchor would remove
that smooth bias only by reintroducing discrete chart switches and is not this
design.  The live gate must therefore score smooth uphill warp under
translation as a separate criterion.  This is a zero-read, zero-memory
replacement and is orthogonal to P's filtered field.  It is not transcribed
until the complete plan passes review.

## 3. Why the ordinary functional codebook is not the answer

Triangularly interpolating three phase functions whose angular functions are
ordinary triangular finite elements gives

```text
sum_i lambda_i(q) sum_j mu_j(d) M_ij.
```

This is exactly Candidate L's measured nine-vertex tensor element.  Candidate
L is unlimited-precision RED, and an id/codebook indirection changes only the
number of reads.  Candidate P therefore does not store an ordinary angular
finite element, a sampled-view blend, a representative depth, a height moment,
or a cell-private boundary-zero bubble.

## 4. Domains and the three-read factorisation

### 4.1 Phase

The periodic `0.52 m` authored torus is sampled on a `64 x 64` canonical
vertex grid and split into two fixed triangles per square.  Prior direct-node
measurements show that this resolution preserves both accepted filtered
footprints within their frozen node limits.  Each phase vertex is its own
prototype; no clustering or categorical spatial codebook is involved.

The load-bearing scale coincidence is explicit: the phase spacing is
`0.52/64 = 8.125 mm`, approximately the support scale of the `sigma=4`
filtered truth.  For a roughly `0.5 m` plant, event-direction drift across one
phase cell is of order `delta q/h ~= 0.93 degrees`, also comparable to that
truth's intrinsic angular blur.  Phase interpolation therefore cross-fades
static event functions; it does not translate their knots.  This is acceptable
only if the joint mid-phase/mid-direction gate below proves the resulting
ghosting remains inside the already-present physical filter.  It is not an
assumed theorem.

For live phase `q`, let its triangle vertices be `v_0,v_1,v_2` and barycentric
coordinates be `lambda_0,lambda_1,lambda_2`.  The vertex indices are computed
analytically, so no id texture is read.

### 4.2 Direction

Descending exterior directions use a bounded hemisphere chart with a unique
pole and a one-sided horizon boundary.  The chart has `240` real direction
triangles plus `16` padding records in a `256`-record allocation.  Radial knots
are allowed to be nonuniform but are
global and frozen before fitting; azimuth seams and the pole are canonical.

For the current direction cell `c(d)`, load exactly one `128`-bit record for
each of `v_0,v_1,v_2`.  Decode the three complete angular functions at `d` and
phase-interpolate their positive measures:

```text
M_P(q,d,sigma) = sum_i lambda_i(q) F_(v_i,c(d))(d,sigma).
```

This is exactly three profile operations for one affine layer and six for two
runtime affine layers.  The three records are lowered and accumulated
sequentially; only one decoded vertex measure and one running sum need remain
live.

## 5. The canonical two-event angular record

Each direction triangle stores at most two ordered, non-crossing visibility
transitions.  The transitions are chords connecting one selected pair of the
triangle's edges.  For chord `k`, its endpoints define an affine signed
function `ell_k(mu)` in direction-triangle barycentrics.  The cook orients and
orders the chords so that the smooth transition functions obey

```text
0 <= s_1 <= s_0 <= 1,
s_k = smoothstep(-w_k, +w_k, ell_k).
```

The non-negative partition is

```text
W_0 = 1-s_0,
W_1 = s_0-s_1,
W_2 = s_1,
W_0+W_1+W_2 = 1.
```

For each of the two physical filter endpoints, the decoded response is

```text
F(d,sigma_j) = sum_(r=0)^2 W_r(d) M_(r,j).
```

All `M_(r,j)` are non-negative premultiplied measures, so the decoder cannot
create negative coverage, colour outside coverage, or a fake categorical
record.

There is a cheap necessary-condition bound before chord fitting.  Concatenate
the two footprint responses into `x(d) in R^8`.  At one phase vertex,

```text
x(d) = M_0 + s_0(d)*(M_1-M_0) + s_1(d)*(M_2-M_1),
```

so every response in one angular cell lies in an affine subspace of dimension
at most two.  The gate first projects the complete sampled truth onto its best
unlimited-precision affine rank-two subspace while generously ignoring chord
topology, shared-edge constraints, non-negativity, packing, and held-out data
separation.  This oracle is strictly more expressive than Candidate P.  If it
is RED under the image/translation limits, the two-event family is impossible
for the accepted community and fitting stops.  If it is GREEN, it is only a
necessary check; the actual canonical chord construction must still pass.

The exact bit witness is:

```text
3 RGBA4444 region states at sigma 4:              48 bits
3 RGBA4444 region states at sigma 16:             48 bits
edge-pair, orientation, topology:                   3 bits
four canonical 5-bit chord-edge positions:         20 bits
two 3-bit transition widths:                        6 bits
order/reserved:                                     3 bits
total:                                             128 bits
```

### 5.1 Structural angular continuity

A band meeting a shared angular edge is not private state.  The cook solves
one canonical one-dimensional edge trace per `(phase vertex, angular edge)`.
Both incident triangle records duplicate bit-identically:

- every crossing position and width on that edge;
- the ordered region-state bits touching every edge interval; and
- the orientation/order mapping.

The two incident decoders therefore have identical restrictions on their
shared edge.  At the azimuth seam the same canonical edge record is reused.
Every pole incident record reduces to one bit-identical pole state.  This is
direction `C0` by construction and is materially different from the rejected
private bubble, whose correction was forced to zero on the edge.

### 5.2 Structural phase continuity

Each phase vertex owns one complete canonical angular function.  Adjacent
phase triangles use the same vertex index and bit-identical records.  Their
barycentric restrictions therefore agree on every shared phase edge.  A
positive barycentric sum of direction-`C0` functions is jointly `C0` in phase
and direction.

The representation is deliberately bounded.  A cell with three crossing,
branching, or closed event components cannot be hidden in the two-band code;
it must fail the cook/gate.  The gate first uses a unique function per phase
vertex and unlimited state precision, so clustering and quantisation cannot
be blamed for a RED event-family result.

## 6. Footprint and the no-particle far limit

`sigma=4` and `sigma=16` are independently integrated positive-measure truth,
not mips of categorical records.  A continuous log-footprint coordinate blends
the two decoded measures.  For footprints larger than `sigma=16`, a second
smooth interval contracts the phase-varying response to the cook-measured
phase mean `M_inf(d)`:

```text
M(q,d,rho) = (1-g(rho))*M_16(q,d) + g(rho)*M_inf(d).
```

`M_inf` is fitted as a small community-global continuous hemisphere function
stored in ordinary constants/uniform data, not another sampled profile
texture.  It is scored against the exact phase mean at all held-out
directions.  At the endpoint the output is phase independent, so camera
translation cannot produce the long-standing one-direction particle crawl.

No representative depth enters `rho`.  It is computed from the analytic
screen differential of the same middle/root chart used by the cook and query.
No scale is selected categorically.

## 7. One-read C0 bridge and query schedule

The resolved eight-read Candidate-F field cannot overlap a three- or six-read
P query without exceeding the cap.  The handoff therefore uses one direct
bridge operation.

The bridge is a `32 x 32` phase-square by `64` direction-triangle atlas.  One
`RGBA32Uint` record stores the twelve shared positive values of a bilinear
phase square times one angular triangle:

```text
4 phase vertices * 3 angular vertices * (A4 | colour-class4) = 96 bits.
```

The remaining bits are reserved.  Canonical shared vertices make this direct
tensor bridge `C0` in phase and direction.  It is a deliberately low-pass
transition object, not the final far codec, and it receives its own fidelity
gate.

With cubic smoothstep weights, the single-layer isolated-Calamagrostis query
shapes are:

```text
R:   resolved Candidate F                                      8 reads
B1:  resolved F + direct bridge                              8+1 = 9
B2:  direct bridge + Candidate P                             1+3 = 4
U:   Candidate P                                                3
```

For the future two-affine-layer variance tier, `B2/U` use six P reads and
remain below nine.  The current isolated resolved review keeps one exact
community layer; enabling a second independently transformed exact layer
would cost sixteen reads and is not silently claimed.  The admissible product
choice is either (a) begin the second variance layer continuously in B2 when
geometry is unresolved, or (b) jointly cook a same-period composite
community.  This is a visible variance decision for the user after the
single-species correctness gate, not permission to exceed the profile budget.

Fractional bridge/P pixels preserve the real background depth/id and composite

```text
C_out = P + (1-A)*C_background.
```

The exact/fractional metadata frontier remains an explicit live shadow/AO/TAA
gate.  No interpolated grass depth is created.

## 8. Exact bytes, traffic, and shader shape

The intended exact atlas repack is the already-reviewed
`depth12 | oct8,8 face normal | colourClass4` `R32Uint` record:

```text
resolved: 65 * 256^2 * 4 B                       = 16.250 MiB
P field:  64^2 phase vertices * 256 cells * 16 B = 16.000 MiB
bridge:   32^2 phase squares * 64 cells * 16 B    =  1.000 MiB
constants/header/alignment                         < 0.125 MiB
total                                             < 33.375 MiB
```

This is more than `15 MiB` below the frozen ceiling and removes the separate
winner-colour texture.  No extra sampled-texture binding is needed: resolved,
bridge, and P records are regions/layers of one unfilterable integer array;
every operation is an explicit point load.

Worst unresolved two-layer traffic is six coherent-in-direction but
phase-separated `16`-byte records (`96 B`).  Worst B1 traffic is eight
four-byte resolved records plus one sixteen-byte bridge record (`48 B`).
There is no workgroup, barrier, divergence inside a query shape, or dependent
codebook load.  Event decode is fixed compare/select/smoothstep ALU.  The
event record is consumed and accumulated before the next phase vertex is
loaded to bound registers.  A fresh trace must confirm generated register
count, occupancy, and texture locality after visual acceptance.

## 9. One decisive offline gate

The gate uses the actual accepted triangle soup and direct physically filtered
camera-pixel truth.  It is one coherent attempt with one correction only for a
proved harness nonconformance.

Order:

1. verify the `64 x 64` phase reduction at every stored direction for both
   physical footprints;
2. fit a unique unlimited-precision canonical two-event function for every
   phase vertex over the `256`-cell direction mesh;
3. score held-out directions including Candidate K's failed shared-edge
   midpoints, Candidate L's worst cells, pole/seam/horizon limits, uphill local
   frames, and deliberately branching/crossing cells;
4. fit/score the community-global phase-mean limit;
5. quantise exactly to the written `128`-bit record and rescore;
6. fit, quantise, and score the one-read bridge and the complete `R/B1/B2/U`
   handoff; the bridge's `A4` values are scored after packing because one
   coverage step is `1/15 ~= 0.067`, already close to the translation limit;
7. score the phase-by-direction product, not either axis in isolation:
   mid-triangle phase points are evaluated at held-out directions, including
   the edge, pole, seam, horizon, uphill, branching, and crossing cases;
8. score `1--4.5 mm` translations at those same held-out directions as excess
   change over truth, including smooth uphill warp from the moving opaque-point
   terrain-chart anchor, and emit numbered QA.

Frozen positive-measure limits remain:

```text
coverage p95/p99              <= 0.08 / 0.20
premultiplied RGB p95/p99     <= 0.06 / 0.15
largest connected exceedance  < 1%
translation p95               <= 0.06, connected < 1%
```

The far mean is additionally judged in screen space: once the entire source
footprint is subpixel, any remaining error region must be below one pixel and
must not preserve phase-correlated motion.  The bridge is not allowed to pass
by being a constant field; it is scored against its own filtered truth and
through both handoff intervals.

Candidate P owns the unresolved/minified appearance family: far stretching,
wrong-view sectors, translation-correlated particle crawl, and excessive fuzz
beyond the resolved footprint.  A GREEN result does not claim to repair the
resolved Candidate-F event-closure failures: nearby missing leaves/branches,
plume-before-stem dropout, and close boundary fragments remain separate work.
The first live review must judge both groups separately.

If the unique, unlimited-precision two-event family is RED, Candidate P is
parked immediately.  More events, directions, bytes, reads, private bubbles,
clustering, or runtime inspection do not follow.  A GREEN result must still
receive Fable's independent adversarial review before any runtime or shader
file changes.

The RED fork is frozen before measurement.  It means the accepted arbitrary
community exceeds this fixed codec's measured rate-distortion capacity.  The
next move is upstream: either constrain/re-author the cooked community to a
cook-certified codec capacity, or ask the user to reopen the read or resident-
memory limits.  There is no Candidate Q inside the same event-field family.

## 10. Corrected bounded gate result

The gate is **RED / inconclusive capacity boundary**. Candidate P is parked
and no runtime, shader, quantisation, or bridge work is authorised. This is
an empirical no-go for the written implementation, not an impossibility
theorem for every globally optimal two-event fit.

### 10.1 The one harness correction

The first diagnostic formed integer-radius filters on a 64 x 64 point page.
That changed the physical footprints and is not evidence. The corrected run
renders exact-BVH point truth at 128 x 128, then applies the separable
periodic integral of the piecewise-constant point page with full widths:

    sigma-4 truth:   (9/256) tile = 4.5 truth samples
    sigma-16 truth: (33/256) tile = 16.5 truth samples

Every source texel is weighted by its exact interval overlap with the
fractional box. Only after physical filtering is the result periodically
bilinear-sampled at the 64 x 64 phase vertices, the two phase-triangle
interiors, and their 1--4.5 mm translations. The centre conversion is:

    i_truth = (128/64) * (i_phase + 0.5 + phaseOffset) - 0.5.

This is the single allowed correction. A direct 256 x 256 exact-BVH run
was stopped after its measured projection exceeded the track's 60-minute
limit; it made no verdict and produced no retained evidence.

### 10.2 Favourable representation oracle

Before fitting chords, ordering, shared edges, nonnegative states, or packing,
the gate grants each of all 4096 phase vertices in each of four targeted
worst-case direction triangles an arbitrary affine rank-two plane in joint
sigma4-RGBA / sigma16-RGBA space. The oracle intentionally sees all ten
scored directions in its cell. Independently per phase vertex it chooses the
best frozen-threshold-normalised L-infinity candidate among 137 planes:
global PCA, six robust reweighted PCA planes, ten leave-one-out PCA planes,
and all 120 planes supported by three observations. It also permits signed
states, arbitrary direction coordinates, and private/noncanonical edges.
These relaxations can only favour Candidate P's representational premise.

The oracle is deliberately broad but is not a certified global L-infinity
optimizer. Because the corrected failure margin is small, its RED result
must not be promoted into a theorem that no two-event field can pass.

### 10.3 Measurements

Content-addressed evidence:

    data/work/groundcover-candidate-p-rank2-bound/
      e3e0a4175b151b89/1c0d0155b78fc12b/
    report SHA-256:
      c002ce49afed0c3a1f5dfa54cc267f338e8995f5f9f7f76294ea8f9814fbfa7a

The favourable rank-two oracle scored:

    static GREEN                         36 / 48
    translation GREEN                  755 / 768
    worst coverage p95             0.080298  (limit 0.08)
    worst premultiplied RGB p95    0.071511  (limit 0.06)
    worst translation coverage p95 0.084253  (limit 0.06)
    worst translation RGB p95      0.054609  (limit 0.06)
    worst connected exceedance     0.000000  (limit <0.01)

All static failures are the sigma-4 footprint in the 10--18 degree standing
direction cell. All p99 and connected-region checks pass. The failure is
therefore a narrow fine-footprint capacity boundary, not a broad wrong-view
sector.

The phase-only control removes the angular rank restriction entirely: it
phase-interpolates exact held-out-direction vertex truth. It scored:

    static GREEN                         48 / 48
    translation GREEN                  767 / 768
    worst coverage p95             0.051719
    worst premultiplied RGB p95    0.034831
    worst translation coverage p95 0.076126  (one 4.5 mm sigma-4 case)
    worst translation RGB p95      0.046229
    worst connected exceedance     0.000000

Thus the 64 x 64 phase finite element is adequate for every static case but
narrowly misses one exact-direction translation case. The robust rank-two
surrogate adds a second small miss in standing-angle sigma-4 colour.
Neither failure licenses threshold tuning after measurement.

### 10.4 Park and resume condition

Quantisation and the one-read bridge were not run because unlimited precision
was not GREEN. The current runtime remains untouched.

Resume Candidate P only with one of these upstream changes:

1. a globally certified robust fit of the actual canonical two-event family
   which includes the complete phase-by-direction and translation objective
   and passes the frozen limits; or
2. a mathematically revised codec/phase basis which removes the measured
   sigma-4, 10--18 degree, 4.5 mm capacity boundary without adding reads or
   resident memory, followed by a newly authorised gate.

Do not resume by adding directions, events, private bubbles, taps, bytes,
runtime searches, or shader filters. The next design move is mathematical
and upstream.

## 11. Final corrected phase-by-direction capacity result

The final cached-truth rerun is:

```text
data/work/groundcover-candidate-p-rank2-bound/
  e3e0a4175b151b89/21a11246991582a6/
report SHA-256
  51e6b64bece429c73ed3aada3db7afe89ad753e8ba2e04a0e6cd9af406bd34a1
filtered-truth manifest
  data/work/groundcover-candidate-p-filtered-truth/
    e3e0a4175b151b89/a1ff5b80bf4e007d/manifest.json
```

It adds exact phase-vertex scoring, all phase-interior translations, explicit
translation-difference equations, and preserved robust-baseline scoring.  The
coupled normal solve is numerically converged (normal-residual ratios near
`8e-7`) and lowers average image loss, but catastrophically worsens the frozen
tail metrics.  It is retained as optimizer evidence and is not substituted for
the stronger robust baseline.

The unrestricted-angular phase control is nearly sufficient:

```text
static GREEN                         72 / 72
translation GREEN                 1151 / 1152
worst coverage/RGB p95          0.04119 / 0.02665
sole translation coverage miss          0.06874  (limit 0.06)
```

The preserved favourable joint affine-rank-two oracle is materially RED:

```text
static GREEN                         32 / 72
translation GREEN                 1062 / 1152
worst coverage p95                    0.13442  (limit 0.08)
worst premultiplied RGB p95           0.10996  (limit 0.06)
worst translation coverage p95        0.09799  (limit 0.06)
largest connected exceedance          0.00073
```

Failures concentrate in the standing cells: sigma-16 static fidelity and
sigma-4 movement at `4.5 mm`.  Sigma-4-first lifted rank two is also RED.
Independent per-scale fitting did not construct a stronger result; because it
is theoretically a strict superset of the joint plane, a numerically worse fit
is optimizer evidence only and cannot be used as a no-go theorem.

Thus the written Candidate P representation on its fixed, nearly uniform
`240`-triangle hemisphere mesh is parked.  Quantisation, canonical chords, the
bridge, packing, shader work, and runtime work remain unauthorised.  The sole
bounded upstream audit is whether the inherited direction partition spent the
same `256` record slots in the wrong places: a conforming, analytically indexed
nonuniform mesh may redistribute but not increase that count.  Two subdivision
failures park that audit.  This does not reopen uniform direction refinement,
extra events, extra reads, or extra resident bytes.

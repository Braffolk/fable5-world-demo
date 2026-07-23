# Grassprofile 2: standing-height exterior reconstruction

Date: 2026-07-23

Status: implementation contract for the first real standing-height correction.
This note is deliberately renderer-free.  It describes one periodic cover tile in
its authored coordinates and one exterior ray.  The existing medium-height result
is the control that must not regress.

## 1. Observed regime and the current mathematical defect

Let the top of the fixed cover box be the plane `y = H`.  A live unit ray enters
that plane at `p = (p_x, H, p_z)` with direction `d`, `d_y < 0`.  A baked node has
unit direction `c`, `c_y < 0`, and stores the first-hit distance `tau_c(q)` for a
ray beginning at top-plane point `q`.

The current acceptance path does not preserve `tau_c`.  It stores horizontal path

```text
lambda_c(q) = tau_c(q) |c_xz|
```

and reconstructs

```text
tau_live = lambda_c / |d_xz|.
```

For a hit at height `y`, this gives

```text
y_hat = H - (H-y) tan(alpha_live) / tan(alpha_bake).
```

Thus a live ray below the lowest 15-degree row collapses the whole plant toward
the top plane.  With `H = 1.1765 m`, `alpha_live = 9 degrees`, and
`alpha_bake = 15 degrees`, a true ground hit is reconstructed around `0.48 m`
above ground.  This is the standing-height vertical stretching.  A 10 m camera
usually supplies rays at or above 15 degrees, explaining why that regime improved.

No filtering rule can repair this identity.  The stored quantity is wrong for the
claimed camera domain.

## 2. Full-depth epipolar identity

The same world point `X` lies on both rays exactly when

```text
X = p + tau_live d = q + tau_c(q) c.
```

Equality of height gives

```text
tau_live = tau_c(q) c_y / d_y.
```

Equality in the top plane then gives the fixed-point address

```text
q = p + B_c(d) tau_c(q)
B_c(d) = (c_y / d_y) d_xz - c_xz.
```

This is the exact projective relation.  It preserves the baked hit height; it does
not reinterpret horizontal travel as live 3D depth.

For a nearby direction node, use one fixed correction:

```text
tau_0 = tau_c(p)
q_1   = p + B_c(d) tau_0
tau_1 = tau_c(q_1)
X_1   = q_1 + tau_1 c
tau_out = dot(X_1 - p, d).
```

At the exact fixed point, `tau_out = tau_1 c_y / d_y`.  The projected form is
used because it remains the correct live-ray compositing coordinate when the one
step has a residual address error.

## 3. Direction-cell and categorical rules

The four regular-lattice corners around the live direction are evaluated
independently.  For each corner:

1. read its full baked depth at `p`;
2. compute `q_1` from that corner's actual direction;
3. read the coupled depth/coverage/normal record at `q_1`.

Depth, normal, species, and owner are never numerically blended.  The geometry
record is selected categorically from the corrected corner with greatest
directional weight among covered records.  Premultiplied colour and alpha alone
may be directionally filtered across the corrected corners.  A miss remains a
miss; it is not converted into a distant surface.

This removes the proven first-order height collapse and performs one projective
address correction.  It does not claim to synthesize an owner which is invisible
at all four lattice corners.  Such an angular birth/crossing is an unresolved
direction-cell event, not permission to blend geometry.  The visual gate therefore
specifically includes slow camera translations at standing height so a coherent
wrong-owner wedge cannot hide in a still image.

## 4. Fixed cost

The isolated profile currently performs four depth reads, four normal reads, and
four colour reads: 12 texture operations.  The replacement performs:

```text
4 initial coupled-record reads
4 corrected coupled-record reads
4 corrected premultiplied-colour reads
= 12 reads
```

Normal and depth come from the same corrected record, so the correction adds no
texture traffic relative to the current isolated path.  It adds fixed arithmetic
only: four direction reconstructions, four two-component epipolar offsets, and one
four-way categorical election.  There is no loop, march, candidate list, new pass,
barrier, dispatch, binding, or runtime geometry.  Accesses remain within the same
two atlases; the second read is dependent but spatially coherent within each view.

## 5. Binding limits and gates

- Exterior only.  `d_y < 0` and the camera is above the cover box.  Entry into the
  box remains the existing top-plane solve.  The permitted whole-box interior fade
  is separate and must not activate at standing height when the eye is above `H`.
- The 15-degree floor remains a hard sampling deficiency below 15 degrees.  The
  exact identity removes the false height reconstruction; one readdress reduces
  phase error, but the angular cell is still wide.  If the standing gate retains
  coherent owner wedges after this correction, the next required data change is a
  genuine low-elevation row or a full exterior categorical boundary field, not a
  shader-side clamp or a second march step.
- Acceptance requires both: (a) no standing-height vertical stretching or
  camera-distance height band, and (b) no regression in the accepted ~10 m oblique
  view.  Top-down correctness and authored panicle colours must remain intact.

## 6. First implementation result

Implemented in the isolated `grassprofile=2` path on 2026-07-23.  The atlas now
retains full baked depth; the runtime performs four initial reads, four corrected
coupled-record reads, a categorical covered-record election, and four corrected
premultiplied-colour reads.  The old standalone follow-up normal and colour queries
are bypassed, so the fixed count remains 12.

Supporting gates passed:

- TypeScript typecheck;
- 42 focused profile/ray-math tests;
- exact real-WebGPU standing boot at `alt=1.7`, `pitch=-0.28`;
- exact real-WebGPU shallow standing boot at `alt=1.7`, `pitch=-0.04`;
- exact real-WebGPU 10 m control boot at `alt=10`, `pitch=-0.28`;
- no page, TSL, WebGPU validation, bind-group, pipeline, or command-buffer error.

The captured standing frames no longer contain the prior camera-distance height
sheet, and the 10 m oblique control retains upright plant-scale structure and
authored colour.  This remains pending the user's close-flight motion review; that
review is the acceptance gate for residual owner wedges or phase jumps that one
still image cannot certify.

## 7. Independent audit: terrain-following entry and chart (2026-07-23)

### 7.1 What the guide actually represents

The guide pitch is `P = 0.84 m`.  For one guide cell, let `(u,v) in [0,1]^2`
be coordinates between the four guide-sample centres and let their decoded
heights be `g00,g10,g01,g11`.  The height used by `bguide` is exactly

```text
G(u,v) = g00 + A u + B v + C u v
A = g10 - g00
B = g01 - g00
C = g11 - g10 - g01 + g00.
```

However, `gradO` is *not* the derivative of this `G`.  It is a separate
bilinear interpolation of four central-difference gradients baked at the guide
centres.  In general

```text
gradO != grad G(O),
grad G(u,v) = ((A + C v)/P, (B + C u)/P).
```

Thus the current code combines a bilinear height value with a different slope
field even before a ray leaves the cell.

### 7.2 The current `O`-anchored solve is camera-dependent

Write the live ray relative to its ordinary scene-depth terrain point `O` as

```text
x(delta) = xO + d_xz delta
y(delta) = yO + d_y delta,
```

where `delta = 0` at `O` and the cover entry has `delta < 0`.  In the rigid
isolated path the current solve is

```text
hO     = yO - G(xO)
kO     = d_y - gradO dot d_xz
deltaE = (H - hO) / kO.
```

This intersects the ray with the top of the infinite plane

```text
TO(x) = G(xO) + gradO dot (x - xO) + H,
```

not with the terrain-following top `G(x)+H`.  `O` changes when the camera
translates, so a curved or piecewise-bilinear depression selects a different
plane and therefore a different atlas phase and view elevation.  The result is
camera-independent only when the whole relevant interval is one affine terrain
plane and the stored slope equals that plane's gradient.  The reported swimming,
stretching on small depressions, and far-rise sides looking like tops are direct
consequences of violating those conditions.

### 7.3 Exact zero-read replacement inside one already-loaded guide cell

The four heights already loaded by `bguide(O)` are sufficient for an exact
intersection with the bilinear top *provided the entry base coordinate lies in
that same guide cell*.  For a rigid cover chart set

```text
u(delta) = uO + alpha delta,    alpha = d_x/P
v(delta) = vO + beta  delta,    beta  = d_z/P.
```

Then `y(delta)-G(u(delta),v(delta))-H = 0` is the quadratic

```text
c2 delta^2 + c1 delta + c0 = 0
c0 = yO - G(uO,vO) - H
c1 = d_y - A alpha - B beta - C(uO beta + vO alpha)
c2 = -C alpha beta.
```

Use the linear root `-c0/c1` when `|c2|` is negligible; otherwise evaluate both
roots stably and choose the domain-valid exterior-to-interior crossing with the
smallest positive camera-ray parameter.  Its relative parameter is negative and
its `(uE,vE)` must lie in `[0,1]^2`.  This adds fixed ALU only.  It needs no new
guide or atlas read because `g00..g11` and `uO,vO` are already live.

At the accepted root, derive the slope from those same heights,

```text
mE = grad G(uE,vE)
   = ((A + C vE)/P, (B + C uE)/P),
```

and use `mE` consistently for direction and normal transforms.  For the rigid
chart

```text
p' = d_xz
h' = d_y - mE dot d_xz.
```

For the existing horizontal deformation `F(h)` with
`A_h = F'(h)`, the exact local differential at any chart point is

```text
h' = (d_y - m dot d_xz) / (1 - m dot A_h)
p' = d_xz - A_h h'.
```

This is the current non-orthogonal basis identity, but with the gradient of the
same terrain function at the same chart point.  The matching inverse-transpose
normal transform is also fixed-cost.  If `(w_p,w_h)` is an authored-chart
covector, then

```text
kappa   = (w_h - w_p dot A_h) / (1 - m dot A_h)
n_world = (w_p - kappa m, kappa).
```

For the isolated rigid path `A_h=0`, this reduces to
`n_world=(w_p-w_h m,w_h)`.  Using `gradO` instead of `mE` selects the wrong
baked elevation and tilts the recovered normal by a camera-dependent amount.

For a deformed top, the exact entry equation uses the top's *base* coordinate

```text
pE(delta) = xO + d_xz delta - F(H)
yO + d_y delta = G(pE(delta)) + H.
```

`F(H)` is constant during this solve, so it is the same quadratic after shifting
`uO,vO` by `-F(H)/P`.  No tangent-plane `Q(h)` extrapolation is required for the
entry itself.

### 7.4 Hard multi-cell and grazing limit

The same-cell construction is exact but not unconditional.  The horizontal
distance between ground and top intersections is approximately

```text
L ~= H |d_xz| / |d_y - m dot d_xz|,
Ncells ~= L/P.
```

For `H = 1.1765 m` and flat ground this is already about `5.2` guide cells at
15 degrees elevation and `16.0` cells at 5 degrees.  The four corners around
`O` contain no information about any of those intervening cells.  Extrapolating
their bilinear polynomial outside `[0,1]^2` merely replaces the current
camera-selected plane with a camera-selected polynomial; it is not an exact
terrain-following solution.

More generally the guide is piecewise bilinear, so the exact first exterior
entry is the first valid root among a direction-dependent number of cellwise
quadratics.  No fixed formula using only `O`'s four samples can identify that
root for arbitrary guide data: two terrains can share all four loaded values and
differ in an unobserved crossed cell, producing different first entries.  Exact
all-angle multi-cell entry therefore requires either traversal/candidates,
additional precomputed directional entry data, or a different globally
queryable terrain representation.  It cannot be obtained from the current four
values by more algebra.

Even within one bilinear cell, a straight world ray maps to a curved chart ray.
For rigid cover

```text
h'' = -d_xz^T Hessian(G) d_xz
    = -2 C d_x d_z / P^2.
```

The atlas query freezes the entry differential and therefore remains a local
linearization whenever `C != 0`; exact entry alone cannot make an arbitrary
triangle-soup ray query exact over a long curved chart interval.  Its error is
second order in the in-cell travel, whereas the present `O`-plane error is
first-order over the entire ground-to-top separation.

### 7.5 Smallest defensible implementation contract

The smallest zero-read correction is therefore:

1. expose the four already-loaded guide heights and `uO,vO` from `bguide`;
2. solve the same-cell bilinear quadratic for the top entry;
3. accept it only when its base coordinate is in that cell;
4. derive `mE` from the same four heights and use it for query direction and the
   inverse-transpose normal transform.

This removes the proven camera-dependent plane error in its valid domain and
adds no read, loop, pass, binding, or dispatch.  It must not be described as the
general grazing solution.  If the observed failures remain predominantly in
the common multi-cell regime, the premise requiring exact terrain-following
entry from the existing four samples is false; the next decision is an upstream
representation change, not another shader-side trigonometric tweak.

## 8. Independent temporal-stability audit: distant moving grains (2026-07-23)

### 8.1 The world origin is not the continuing defect

For profile 2 the deformation is zero and the periodic address reduces to

```text
q_xz = P_xz P_g + O_world,
```

where `P_g` is guide pitch.  On a guide-origin snap, the local coordinate loses
one pitch while `O_world` gains one pitch.  Their sum, and therefore `q_xz`, is
unchanged in exact arithmetic.  The forced profile also bypasses the root hash,
uses one source layer, and has no time-dependent deformation.  Wind phase,
random root rejection, layer interference, and ordinary guide-origin snapping
are therefore not supported explanations for the isolated shimmer.

The camera ray *is* jittered by the renderer's deterministic Halton phase.  That
changes both the top-entry point and live direction by a sub-pixel amount.  This
is expected sampling, not an unanchored grass animation.  It exposes the defect
below; it does not create it.

### 8.2 The violated invariant is footprint covariance

Let `Omega(C,p)` be the world-space footprint of pixel `p` for camera `C`, and
let `F` be the static periodic first-hit field.  Away from a true silhouette,
the reconstructed premultiplied measure must vary with the symmetric difference
of the two footprints:

```text
|M(Omega') - M(Omega)| = O(area(Omega' triangle Omega)).
```

In particular, when a reprojected footprint moves by millimetres, it must not
select an unrelated full-opacity sample from structure much finer than that
footprint.

The current path violates this invariant twice:

1. Every depth and colour lookup is forced to mip level zero.  Both profile
   textures are created with `generateMipmaps = false`; a hardware-linear read
   integrates only a 2-by-2 texel tent regardless of the projected footprint.
   At distance, one pixel covers many source texels, so a sub-pixel jitter or
   millimetre translation changes the fine phase by enough to produce an
   order-one sample change.  This is deterministic under-minification, matching
   the reported moving-grain field.
2. The categorical node is currently

   ```text
   j* = argmax_j b_j a_j,
   valid = (b_j* a_j*) > 1/255,
   ```

   where `b_j` is directional barycentric weight and `a_j` is spatially filtered
   coverage.  Fractional antialias coverage therefore changes the categorical
   direction order.  Two covered corners can exchange ownership merely because
   their bilinear coverage fractions change, and the resulting partial footprint
   is then emitted as one opaque visibility owner.  The already-computed blended
   colour alpha is divided out and discarded before election.  A continuous
   coverage change is thus amplified into a binary surface birth/death.

Angular chart changes are consequently a real secondary discontinuity, but not
a sufficient explanation for millions of distant grains: direction changes
under a millimetre translation are `O(delta/r)`, whereas level-zero spatial
phase changes occur across the whole under-resolved image.

### 8.3 Fixed-cost correction and its representation requirement

Geometry first needs the categorical complete-record rule from Section 9.3.
With binary eligibility `e_j` (initial and corrected records covered, projected
depth non-negative), the direction election is

```text
j* = argmax_(j : e_j = 1) b_j        (fixed node-index tie break)
```

Only node `j*` provides categorical depth, normal, and owner.  Those records
remain sampled at full resolution and are never mipped or averaged.  This
consumes no read and removes *fractional-coverage-driven* chart switching while
retaining the specified ability to choose the greatest-weight genuinely covered
corner when the dominant direction is a categorical miss.

Radiance and coverage are different quantities.  The existing four corrected
colour reads already provide premultiplied `C_j` and coverage `a_j`.  They must
be evaluated at the footprint LOD and retained as

```text
A = sum_j b_j a_j^(lod)
C = sum_j b_j C_j^(lod)
output = C + (1 - A) background.
```

The LOD is fixed ALU from the periodic-address Jacobian and the pixel footprint,
not a new sample:

```text
lod = clamp(log2(max singular stretch of D_screen(q) in atlas texels), 0, Lmax).
```

The four existing colour reads use that LOD while the four corrected coupled
geometry reads remain level zero.  Texture-read count, direction count, loop
count, and candidate count are unchanged.

This is a zero-extra-read correction, but it is **not implementable as a pure
constant tweak in the current carrier**: the authored colour texture has no mip
chain and the visibility/election wire drops `A`.  A complete repair therefore
requires a cook/load-time premultiplied colour-and-coverage mip chain plus an
alpha-preserving composite path.  Those are representation/wiring changes, not
additional grass queries.  Without them, no stateless formula can recover the
many source texels absent from the level-zero point sample.

### 8.4 Explicitly rejected guesses

- Disabling TAA changes the sampling sequence but leaves camera-motion aliasing;
  it is not a mathematical repair.
- Raising/lowering the `1/255` threshold, deterministic thresholding, or adding
  hysteresis merely moves or temporally stores the discontinuity.
- Blurring or mipping depth/normal/owner violates categorical first-hit data.
- Nearest or one-corner PCF sampling is suitable for categorical geometry, but
  using it *instead of* footprint-filtered premultiplied colour/coverage leaves
  radiance underfiltered and replaces shimmer with aliasing or structured popping.
- Wind, time noise, root hashes, and ordinary guide-origin snaps do not affect
  the isolated path in the equations inspected above.

## 9. Independent conformance audit: full-depth epipolar implementation (2026-07-23)

This audit compares the isolated implementation in
`GroundCoverPeriodicProfile.ts`, `LegacyPeriodicProfileSampler.ts`, and
`LegacyPeriodicRayQuery.ts` against Sections 2--4 above.  It does not evaluate
the legacy multi-profile path.  The central epipolar arithmetic is correct, but
the records supplied to it and the validity/election rules are not yet the
records and rules specified by the mathematics.

### 9.1 Identities that are implemented correctly

For one direction corner the sampler reconstructs

```text
c = (cos(elevation) cos(azimuth), -sin(elevation),
     cos(elevation) sin(azimuth))
B = (c_y/d_y) d_xz - c_xz
q1 = p + B tau0.
```

`LegacyPeriodicProfileSampler.ts:308-325` therefore matches Section 2.  Its
returned depth is

```text
B tau0 dot d_xz + tau1 (c dot d)
= dot((q1 + tau1 c) - p, d),
```

so lines 327--333 also match the required projected compositing coordinate.
`LegacyPeriodicRayQuery.ts:452-492` consistently normalizes the affine chart-ray
derivative before the query and divides the returned chart distance by the same
metric speed.  For an actually affine chart this conversion is sound.  The
isolated normal path also contains the appropriate inverse-transpose shape at
lines 507--517; its remaining terrain-gradient datum error is the separate
Section 7 failure.

The load-time transcode in `GroundCoverPeriodicProfile.ts:285-319` now preserves
the baked full-depth channel instead of converting it to horizontal travel.
That fixes the original height-collapse identity.  None of the remaining
failures below is a reason to restore horizontal depth.

### 9.2 A first-address miss is fabricated into a second-address hit

`decodeDepth(initial, row, true)` maps an uncovered initial record to that row's
`depthMax` (`LegacyPeriodicProfileSampler.ts:286-316`).  The code then forms

```text
q1_fake = p + B depthMax
```

and permits a covered texel at `q1_fake` to win.  This directly contradicts
Section 3's rule that a miss remains a miss.  It also makes the fabricated
address move coherently with camera direction, which can create long moving
streaks rather than isolated noise.

The fixed-cost correction is categorical masking, not another query: retain the
initial coverage bit, perform the already-budgeted corrected read at any dummy
address if a fixed graph requires it, and make the corner ineligible unless both
the initial and corrected records are covered.  It remains four initial plus
four corrected geometry reads.

### 9.3 Geometry is numerically mixed before the categorical election

The profile texture is configured with `LinearFilter` in
`GroundCoverPeriodicProfile.ts:312-313`.  Every `address.tap` in
`LegacyPeriodicProfileSampler.ts:134-159` is therefore a hardware-bilinear
mixture of four spatial texels.  The depth decoder subsequently divides the
mixed premultiplied depth by mixed coverage, and the normal decoder does the
same to octahedral components.  For two different owners A and B this produces

```text
tau_sample = (wA alphaA tauA + wB alphaB tauB)
             / (wA alphaA + wB alphaB),
```

which is generally neither owner's first-hit depth.  Normal decoding similarly
normalizes a fabricated cross-owner octahedral value.  Selecting one of four
such mixtures categorically does not make the selected geometry categorical.
This violates Section 3 before angular election even begins and is a direct
mechanism for fuzzy sheets, false intermediate heights, and terrain-like mosaics.

The 12-read-compatible record contract is: one deterministic point/categorical
spatial record for each initial and corrected geometry lookup, while the
separate premultiplied colour texture remains linearly filterable.  This can be
a nearest complete-record fetch or a deterministic one-corner footprint choice;
it is still one texture operation at each of the eight geometry addresses.  A
manual four-texel geometry filter would exceed the ceiling and would still be
wrong if it averaged owners.

### 9.4 The directional winner is coverage-weighted, not the greatest-weight
covered corner

The implementation scores each corrected corner as

```text
score_i = directionWeight_i * filteredCoverage_i
```

(`LegacyPeriodicProfileSampler.ts:337-363`).  Section 3 instead specifies the
greatest directional weight *among covered records*.  With fractional spatial
coverage, the current rule can let a less relevant direction replace the
greatest-weight covered direction as the camera translates by a fraction of a
texel.  That makes geometry ownership depend on antialias coverage and is another
source of swimming.

After geometry reads become categorical, election must be

```text
eligible_i = initialCovered_i && correctedCovered_i && tauOut_i >= 0
score_i    = eligible_i ? directionWeight_i : -infinity.
```

Coverage may still contribute to premultiplied colour/alpha; it must not change
the categorical geometry order.  This is fixed ALU and adds no read.

### 9.5 Negative projected hits are clamped onto the entry plane

The implementation returns `max(winner.liveDepth, 0)` at
`LegacyPeriodicProfileSampler.ts:372-376`.  A one-step point `X1` with
`dot(X1-p,d) < 0` lies behind the exterior entry and is not a valid first hit.
Clamping it to zero creates a surface exactly on the cover top.  A population of
such failed corners therefore becomes an apparent plane or constant-height band.

The mathematical rule is to mark this corner ineligible.  Do not alter the
projected depth of a valid corner, and do not turn an invalid negative depth into
a zero-depth hit.  This changes only the election mask.

### 9.6 The categorical point `X1` is discarded after projection

Section 2 defines both the categorical corrected point

```text
X1 = q1 + tau1 c
```

and its live-ray compositing coordinate `tauOut`.  The sampler returns only
`tauOut`.  The caller then reconstructs a different point

```text
Xprojected = p + tauOut d
```

to derive `profileTip` and the periodic root
(`LegacyPeriodicRayQuery.ts:487-503`).  After only one correction, `X1` and
`Xprojected` are equal only at the exact fixed point.  Thus geometry colour and
normal are selected from the record at `X1`, while height/root metadata can be
assigned from another point.  This can mislabel sides as tops and can move a
root/copy boundary under camera translation even when the winner record itself
does not change.

The sampler must carry the winning corner's already-computed `X1.xz` and
`X1.y = H + tau1 c_y` alongside `tauOut`.  The compositor still uses `tauOut`;
authored height, root/copy identity, and other categorical metadata use `X1`.
These are register values from existing arithmetic, not texture reads.

### 9.7 Small exact-direction discrepancy

The corrected address is required to use each slice's actual stored direction.
The sampler reconstructs an ideal regular-lattice direction from row and column
instead of selecting `profile.slices[slice].direction`.  The parser permits the
baked vector to differ from that ideal by `2e-4`, so strict equality is not
guaranteed.  This is not a plausible explanation for the large observed sheets,
but it is a literal conformance gap and is avoidable with a fixed select among
the already-loaded direction constants.  It costs ALU, not reads.

### 9.8 Binding rewrite contract

Within the unchanged 12-read ceiling, the isolated implementation must therefore:

1. keep full baked depth and the current `B`, `q1`, and projected-depth identities;
2. use categorical spatial geometry records, with linear filtering confined to
   premultiplied colour/alpha;
3. preserve initial MISS through correction;
4. elect the highest directional-weight eligible corner without multiplying its
   ordering key by coverage;
5. reject negative `tauOut` instead of clamping it to the top plane;
6. carry the winning `X1` for authored height and root/copy metadata while using
   `tauOut` only for ray compositing;
7. use the exact baked corner direction; and
8. replace the camera-selected terrain entry/slope datum according to Section 7.

Items 2--7 need no new texture operation, binding, pass, loop, march, candidate
list, or runtime geometry.  The final budget remains exactly four initial
categorical coupled-record reads, four corrected categorical coupled-record
reads, and four corrected premultiplied-colour reads.  The current implementation
is therefore not conformant to the two-spec mathematics despite containing the
correct central epipolar formula.

## 10. Lateral cover boundary is a root predicate, not an entry-plane predicate

Let `E(r)` be the ecological eligibility field evaluated at the authored root
`r` of a periodic plant copy.  The rendered set is

```text
G_E = union { triangle T : E(root(T)) = 1 }.
```

Testing `E` at the ordinary terrain endpoint `O` instead clips the *ray* before
the plant query:

```text
wrong:  E(O) = 0  => no query
right:  query => X1 => r(X1) => retain iff E(r(X1)) = 1.
```

The wrong rule extrudes the 2D endpoint mask through the entire cover height.
Viewed from beside the mask boundary, that extrusion is necessarily a flat top
cut with no plant sides; it also moves as the camera changes which endpoint lies
behind the same visible plant.  The correct rule obtains the periodic copy from
the categorical corrected point `X1`, derives its stable root, and evaluates the
already-bound control field there.  It adds no atlas or control-field read: the
root control read already exists in the query.

There is one exact representational limit.  If the unconditioned first hit has
an ineligible root, a later eligible owner can exist on the same line.  Recovering
that successor requires eligibility-conditioned precomputation or another event;
it cannot be inferred from the rejected record.  The isolated path therefore
uses the exact predicate for the returned complete record and rejects it when
ineligible.  It must never restore the endpoint clip merely to hide boundary
holes.

## 11. Corrected fixed-cost record equations

For each of the four angular corners `j`, define the canonical regular-lattice
direction `c_j`, directional barycentric weight `b_j`, initial record `R0_j`,
and corrected record `R1_j`.  The acceptance asset's stored float directions
were measured against those canonical directions: the maximum Euclidean
difference is `5.22e-8`, so using the canonical direction is the exact
float-precision contract for this isolated asset.

```text
a0_j     = coverage(R0_j)
tau0_j   = decodeDepth(R0_j)
B_j      = (c_j.y / d.y) d.xz - c_j.xz
q1_j     = p + B_j tau0_j
R1_j     = record_j(q1_j)
a1_j     = coverage(R1_j)
tau1_j   = decodeDepth(R1_j)
X1_j     = (q1_j.x, H, q1_j.z) + tau1_j c_j
t_j      = dot(X1_j - (p.x,H,p.z), d)
eligible_j = (a0_j > 1/65535)
          && (a1_j > 1/65535)
          && (t_j >= 0)
j*       = argmax_(eligible_j) b_j, with lower fixed node index winning ties.
```

The output contract is then

```text
compositing depth = t_j*
normal            = normal(R1_j*)
authored height    = X1_j*.y / H
periodic copy/root = tileContaining(X1_j*.xz)
ecological retain  = E(root(X1_j*))
```

No expression reconstructs metadata from `p + t_j* d`; that projected point is
only the camera-ray depth coordinate.  No negative `t_j` is clamped to zero.  No
miss address may become eligible after the dependent read.  No coverage value
multiplies the categorical ordering key.  Spatial geometry sampling returns one
complete texel; only the separate premultiplied colour carrier is linearly
filterable.

The colour equations remain

```text
C = sum_j b_j C_j(q1_j)
A = sum_j b_j a_j(q1_j).
```

`C/A` is still the present near-field authored-colour output.  Section 8's
footprint-LOD `C + (1-A) background` correction is a separate outstanding
minification change because the current screen visibility wire does not yet
preserve `A`.  It must not be confused with, or used to defer, the categorical
geometry correction above.

## 12. Equation-to-code conformance check (2026-07-23)

The corrected note above is the source of truth.  The isolated implementation
was checked expression by expression after editing:

| Mathematical object | Code expression | Result |
|---|---|---|
| canonical `c_j` | `LegacyPeriodicProfileSampler.ts:316-323` | same regular-lattice elevation/azimuth equation |
| `a0_j`, `tau0_j` | `:326-328`, depth decoder `:298-306` | same threshold and full-depth decode |
| `B_j` | `:329-332` | componentwise identical |
| `q1_j` | `:333-336` | componentwise identical |
| `R1_j`, `tau1_j` | `:337-338` | one corrected complete-record read and same decode |
| `t_j` | `:339-345` | expanded `dot(X1-p,d)` identity |
| eligibility | `:346-350` | initial hit AND corrected hit AND non-negative `t_j` |
| `X1_j` | `:351-355` | exact `(q1,H)+tau1 c_j` |
| categorical score | `:359-361` | `b_j` or `-1`; coverage absent |
| stable tie | `:372-388` | strict `>` keeps the left/lower fixed index |
| output depth/point/normal | `:397-407` | no depth clamp; one winning complete record |
| categorical texture | `GroundCoverPeriodicProfile.ts:313-320` | nearest geometry; no mip |
| height and copy from `X1` | `LegacyPeriodicRayQuery.ts:489-505` | projected depth is not reused for metadata |
| endpoint clip removed | `LegacyPeriodicRayQuery.ts:229-232` | isolated path reaches the root test |
| `E(root(X1))` | `LegacyPeriodicRayQuery.ts:694-734` | existing root control read, no extra tap |

The fixed texture-operation budget is unchanged: four initial geometry records,
four corrected geometry records, and four corrected colour records.  Nearest
geometry reduces hidden spatial filtering work; the change adds fixed predicate
and register ALU only.  It adds no pass, binding, barrier, dispatch, candidate,
loop, march, or runtime geometry.  Register live range grows by the winning
three-component `X1`; divergence and synchronization are unchanged.

Section 7 remains an explicit unresolved carrier limitation, not silently marked
fixed by this conformance pass: the current terrain chart is still the
camera-selected local affine approximation.  The present rewrite removes every
identified mismatch inside the precomputed-ray reconstruction and the lateral
mask.  A remaining coherent slope-only swim after this gate is therefore direct
evidence for replacing that terrain carrier premise, not for adding atlas taps.

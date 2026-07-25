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

## 13. Winner-coherent radiance replaces four-node colour smearing

The visual gate after Section 12 changed the edge silhouette but did not change
the near-to-mid fuzzy/grainy appearance.  This separates the categorical geometry
repair from a remaining radiance error.

For angular node `j`, the corrected query returns a different categorical point
`X1_j` and possibly a different owner.  A colour blend

```text
C_wrong = sum_j b_j C_j(X1_j)
```

is a valid angular interpolation only if every `X1_j` is a correspondence for
the same physical surface point, or if the carrier directly stores a filtered
radiance field on the same live oriented line.  The present arbitrary-triangle
profile proves neither condition.  One epipolar address correction aligns each
node to its own first-hit field; it does not make the four resulting first hits
the same owner.  The formula therefore averages unrelated plant surfaces and
creates a soft veil.  Categorical geometry can change underneath that veil,
which presents as grain or swim rather than a stable antialiased surface.

The complete-record rule extends to near-field radiance:

```text
j*      = argmax_(eligible_j) b_j
geometry = R1_j*
colour   = C_j*(q1_j*)
alpha    = A_j*(q1_j*).
```

The winner already carries `j*` and `q1_j*`; one dependent colour read follows
the eight geometry reads.  This changes the explicit budget from

```text
4 initial geometry + 4 corrected geometry + 4 colour = 12 reads
```

to

```text
4 initial geometry + 4 corrected geometry + 1 winner colour = 9 reads.
```

It also shortens live ranges: three losing colour records and their blend ALU
disappear.  It adds no branch-dependent candidate work—the winning slice and UV
are selected arithmetically before one texture operation.  Angular transitions
remain categorical because geometry itself is categorical; blending colour from
non-winning owners cannot repair that transition and only hides it with ghosting.

This equation does not remove Section 8's minification obligation.  The single
winning colour field still needs a footprint-correct premultiplied mip sample at
distance.  Geometry remains nearest level zero.  Winner coherence fixes
cross-view smearing; footprint filtering fixes sub-pixel grain.  They are
independent and neither requires another read.

## 14. Camera-independent curved-terrain carrier (renderer-free)

The post-Section-12 flight gate is decisive: categorical records improved the
lateral edge, but slope stretch and motion swim did not change. The remaining
failure is therefore upstream of the profile record. The isolated path still
maps a world ray through the plane tangent at the ray's *terrain endpoint* `O`:

```text
h_O(x,y) = y - G(O) - m_O dot (x-O),       m_O = storedGradient(O).
```

Changing the camera changes `O` and hence changes the entire carrier in which
the same authored triangle is queried. This is not a small slope error. It is a
camera-dependent definition of the rendered object.

### 14.1 The strongest affine carrier that curved terrain permits

Let `S` be the arbitrary periodic authored triangle soup in a flat box
`0 <= y <= H`, and let `r(T)` be the stable world-XZ root of triangle `T` (or of
the rigid community cell containing it). The terrain height at that root is
`g_r = G(r)`. Define

```text
Phi_r(X) = X + g_r e_y,                    X in T.
```

This is a rigid vertical translation per root. It is an affine conjugation, so
it preserves triangle shape, first-hit order inside the same motion group,
normals, species overlap, and straight rays. Wind may be an authored-frame
affine shear before `Phi_r`; it must not be folded into `G`.

Continuous terrain draping,

```text
Phi_drape(x,h) = (x, G(x)+h),
```

does not have these properties: the inverse of a straight world ray is a curved
chart ray whenever `G` is non-affine. Section 7 proves that no fixed formula
using the endpoint guide samples can query that curved ray across arbitrarily
many cells. Consequently the root-rigid carrier is not an approximation to be
hidden under filtering; it is the strongest camera-independent carrier
representable by one ordinary straight-ray profile query.

For a *known* root `r`, the live ray `R(t)=o+t d` in authored coordinates is

```text
R_r(t) = o - g_r e_y + t d.
```

Its top-box entry and top-plane XZ address are exactly

```text
t_E(r) = (g_r + H - o_y) / d_y,
p_r    = o_xz + t_E(r) d_xz.              (d_y < 0)
```

There is no terrain gradient in either expression. For baked direction `c_j`,
the existing epipolar fixed-point equation remains unchanged except for this
root-stable entry:

```text
B_j(d) = (c_j.y / d_y) d_xz - c_j.xz,
q_j    = p_r + B_j(d) tau_j(q_j).
```

Thus curved terrain changes one scalar datum, `g_r`; it does not shear the
plant, tilt the ray, or tilt the recovered normal.

### 14.2 One-root, one-correction construction using the reads already present

The root is categorical and is not known before the first profile record. The
strongest construction within the existing dependent-read depth is:

1. Use the real terrain endpoint only as a numerical seed. Let
   `g_s=G(O)`, intersect the *horizontal* seed plane `y=g_s+H`, and call its XZ
   point `p_s`. Do not use `gradO`.
2. Read the four existing initial records at `p_s`. Elect the greatest-weight
   initially covered node with the existing fixed tie rule. Its categorical
   point

   ```text
   Z0_j = (p_s.x,H,p_s.z) + tau0_j c_j
   ```

   supplies the provisional periodic root `r0=tileContaining(Z0_j.xz)`.
3. Move the already-existing root-guide lookup before the corrected profile
   reads and obtain `g_0=G(r0)`. This is the same root-guide load currently
   performed after the hit, not an additional load.
4. Form the root-stable top entry `p_0=p_r0` from the equation above and use the
   already-budgeted corrected reads:

   ```text
   q1_j   = p_0 + B_j(d) tau0_j,
   R1_j   = record_j(q1_j),
   X1_j   = (q1_j.x,H,q1_j.z) + tau1_j c_j.
   ```

5. A corrected record is eligible only when all Section-11 predicates hold and
   `tileContaining(X1_j.xz) == r0`. This posterior equality prevents one
   root's height from being applied to another owner's surface.

For the winning record `j*`, a same-triangle residual can be removed with one
fixed plane solve. Let `n` be the categorical authored normal from `R1_j*` and
`P0=(p_0.x,H,p_0.z)`. Then

```text
s_plane = dot(n, X1_j* - P0) / dot(n,d),
t_world = t_E(r0) + s_plane.
```

This is the exact live-ray intersection with the plane of the returned,
root-translated triangle. It is better founded than `dot(X1-P0,d)`, which is
merely an orthogonal projection of an incompletely aligned point onto the ray.
The normal emitted to world space is the authored normal (after only the global
layer rotation/scale and authored affine wind); the isolated path must not apply
`(-n_y gradO.x, 0, -n_y gradO.z)`.

When `dot(n,d)=0`, an isolated ray-plane intersection does not exist (the
coplanar case is measure zero and not a unique first hit). The record is
ineligible; the denominator is never clamped. This is a surface-incidence
condition, not a hidden camera-direction clamp.

### 14.3 Exactness domain and a computable error bound

The terrain part of this construction is exact and camera independent whenever
the provisional and corrected records identify the same root and triangle *and
that triangle remains the first event after translation*. To
state the remaining bound, let `q*` be the exact fixed-point address for that
root, let `tau_j` be `L_tau`-Lipschitz on the returned triangle chart, and let
`q0=p_s`. The one existing correction obeys

```text
|q1_j-q*| <= |B_j| L_tau |q0-q*|.
```

The residual plane solve makes the world intersection exact if `q1_j` remains
on the same triangle. If `mu_T` is the in-plane distance from the exact hit to
the nearest triangle edge and `epsilon_q` is the address residual above, a
sufficient ownership condition is

```text
K_T epsilon_q < mu_T,
```

where `K_T` is the local address-to-plane displacement norm. Neither `mu_T` nor
the successor visibility gap is stored in the current record, so the runtime
cannot certify this condition. The posterior root equality catches root changes,
but not a change between two triangles assigned to the same root.

This limitation is fundamental, not an invitation to add a filter. Two curved
terrains can agree at the seed/root sample already read and differ at another
raised root crossed by the same grazing ray; the raised copy can become the true
first hit without changing any current input record. Likewise two soups can
share the provisional event and have different successors. No deterministic
fixed-ALU function of the current single event can distinguish either pair.
Unconditional exactness for arbitrary triangle soup on arbitrary curved terrain
would require an additional root/event hypothesis, traversal, or root-conditioned
precompute, all outside the frozen contract.

There is a second authoring boundary: the current record derives
`r=tileContaining(X.xz)` and stores no per-triangle root. Root-rigid translation
is therefore exact only when a triangle does not cross a carrier-cell seam (or
when all seam-sharing fragments use the same root height). Generic soup needs a
categorical root offset/owner mark baked into the complete record to remove this
ambiguity; arithmetic cannot recover it from `X1` alone.

The displayed top-entry chart also has an explicit directional boundary. At
`d_y=0`, `t_E(r)` is undefined because a horizontal line never intersects the
top reference plane. This is not repaired by an epsilon or by clamping the
elevation. Exact horizontal exterior views require a second analytic side chart
(addressed by height and the horizontal phase perpendicular to the ray), or an
equivalent horizontal-row carrier baked with that chart. The current top-plane
GCRP contains neither datum. For `d_y` merely small but nonzero the equations
remain exact, although the root hypothesis becomes increasingly sensitive as
`|p_r-p_s|=|G(r)-G(O)| |d_xz/d_y|` grows. The implementation must report this
as a representation boundary, never hide it by flattening, clamping, or fading
an exterior direction.

### 14.4 Exact code substitutions and cost contract

For the isolated `grassprofile=2` path only, the mathematical transcription is:

```text
REMOVE: kGround = d_y - gradO dot d_xz
USE:    kSeed   = d_y

REMOVE: tangent-plane top entry through O
USE:    tSeed = (G(O)+H-o_y)/d_y

REMOVE: profile direction (d_x, d_y-gradO dot d_xz, d_z)
USE:    profile direction d

MOVE:   the existing bguide(root) height lookup between R0 and R1
USE:    tE0 = (G(r0)+H-o_y)/d_y, p0=o_xz+tE0 d_xz

REPLACE: q1_j = pSeed + B_j tau0_j
WITH:    q1_j = p0    + B_j tau0_j

ADD ELIGIBILITY: tileContaining(X1_j.xz) == r0

REPLACE: t = tE + dot(X1-P,d)
WITH:    t = tE0 + dot(n,X1-P0)/dot(n,d)

REMOVE: worldNormal.xz -= authoredNormal.y * gradO
USE:    root-rigid authored normal (plus only authored affine transforms)
```

The profile budget remains four initial categorical records plus four corrected
categorical records. With Section 13 it is followed by one winner-colour read,
so the total is nine, not more than the current path. The root guide/control
loads are moved and reused, not duplicated. The rewrite removes the tangent
gradient basis ALU and adds one fixed initial election, one root-stable entry,
four root-equality predicates, and one winner plane dot/divide. It adds no
texture, binding, pass, barrier, dispatch, loop, march, runtime geometry, or
per-species work. Register pressure grows only by the provisional root/height
and winner plane numerator; removing the tangent/deformation intermediates in
the isolated branch should more than offset that live range.

This carrier addresses the coherent slope stretch/swim. It does not address the
independent distant moving-grain failure: Section 8's premultiplied footprint
mip law remains required after the carrier is stable.

## 15. One-read band-limited winner radiance carrier

This section closes the representation half of Section 13.  It does not alter
the categorical first-hit equations.  In particular, depth, normal, owner,
coverage eligibility, `X1`, and the node election remain nearest, level-zero,
complete records.

### 15.1 Why cross-node colour filtering is not defined here

Premultiplied angular interpolation is only defined when every term is a
measurement of the same surface event.  If `pi_j(X)` were a proved
correspondence from one live point `X` into every view chart, then

```text
sum_j b_j [alpha_j(pi_j(X)) L_j(pi_j(X)), alpha_j(pi_j(X))]
```

would be a meaningful filtered view-dependent signal.  The current one-step
queries instead return four independent first hits `X1_j`.  They neither compare
owner identity nor prove `X1_j = X1_k`.  In an arbitrary triangle soup two
corners commonly see different leaves, stems, or background.  Cross-node
premultiplied blending therefore remains invalid even though premultiplication
itself is correct: it is a convex combination of unrelated surfaces.

The only carrier consistent with the complete-record geometry is the selected
winner's carrier:

```text
j* = argmax_(eligible_j) b_j
P  = P_j*(q1_j*, footprint),       P = (C_premul, A).
```

The angular transition is deliberately the same categorical transition as the
geometry.  Under camera translation its direction-coordinate motion is
`O(delta/r)`, so a node boundary moves as one coherent boundary.  Four unrelated
colour fields cannot remove that transition; they turn it into a spatially soft
ghost which changes under the categorical surface.

### 15.2 Periodic, slice-isolated mip codec

The base signal for direction slice `j` is linear-space authored colour `L_j`
and binary first-hit coverage `alpha_j`:

```text
P_j^0(u,v) = (alpha_j(u,v) L_j(u,v), alpha_j(u,v)).
```

Every coarser texel is formed cook-side in floating point by a periodic box
average and is quantized only after the average:

```text
P_j^(ell+1)(x,y)
  = 1/4 sum_(a,b in {0,1})
      P_j^ell((2x+a) mod N_x^ell, (2y+b) mod N_y^ell).
```

This is an energy-preserving premultiplied measure: empty space contributes
zero, never a fabricated miss colour.  For an opaque categorical hit the
band-limited conditional authored colour is

```text
L_bar = C_premul / max(A, 1/255).
```

`A <= 1/255` is below one 8-bit coverage code and may use the existing bounded
fallback colour.  The selected geometry still decides whether the screen sample
exists; `A` does not elect a different owner.

The carrier must be a **2D texture array with one layer per direction slice**,
not the current 8-by-8 atlas.  Each layer contains the 256-by-256 periodic
interior and its own complete mip chain.  Array-layer isolation prevents a mip
footprint from ever crossing into another direction.  A one-texel base-level
atlas gutter cannot do this: after the first reduction the gutter disappears,
and every later atlas mip blends unrelated slices.  Periodic convolution also
removes the need for stored per-slice gutters.

For the present 64-direction, 256-square, RGBA8 carrier, exact uncompressed GPU
storage is

```text
64 * 256^2 * 4 * (1 + 1/4 + 1/16 + ...) = 21.33 MiB.
```

The current guttered base atlas is about 16.25 MiB, so the complete chain adds
about 5.08 MiB, not another copy of the 113 MiB source asset.  A platform-native
four-channel block-compressed linear format can reduce the resident chain, but
compression is an encoding choice, not part of the equations.  The 64 layers
are below WebGPU's minimum guaranteed array-layer limit.  Cook/load code must
release transient base and mip staging once upload completes.

### 15.3 Exact local screen-to-address Jacobian

The compute path has no implicit fragment derivatives, but all quantities needed
for the derivative already exist.  Let `s=(s_x,s_y)` be pixel coordinates,
`M^-1` the current inverse view-projection matrix, and `o` the camera origin.  At
the far unprojection plane define

```text
h(s) = M^-1 (2(s_x+1/2)/W - 1,
             2(s_y+1/2)/H - 1, 1, 1)
Y(s) = h.xyz / h.w
v(s) = Y(s) - o
d(s) = v / |v|.
```

No neighbouring texture query is required.  With `m_x,m_y` denoting columns
zero and one of `M^-1`, the one-pixel homogeneous derivatives are

```text
h_x = (2/W) m_x,                 h_y = (2/H) m_y
Y_k = (h_k.xyz h.w - h.xyz h_k.w) / h.w^2
d_k = (I - d d^T) Y_k / |v|,     k in {x,y}.
```

Let the selected compositing point be `X=o+t d` and let `n` be the selected
categorical world normal.  Holding its local tangent plane fixed, the exact
first differential of the neighbouring ray/plane intersection is

```text
X_k = t [d_k - d (n dot d_k)/(n dot d)].
```

Let `K` be the same world-to-authored affine differential already used to turn
the live world ray into the profile ray.  Under Section 14's root-rigid isolated
carrier this contains the layer rotation and authored height/metric scales but
**no terrain-gradient shear**; any authored affine wind is included exactly
once.  Then

```text
U_k = K X_k.
```

For the winning baked direction `c`, a surface point `U` maps back to its baked
top-plane ray origin by

```text
q(U) = U.xz - (c.xz/c.y)(U.y-H).
```

Consequently its screen differential is

```text
q_k = U_k.xz - (c.xz/c.y) U_k.y.
```

This last shear is essential.  Using only the projected world-pixel size, or
only `t/W`, ignores the baked ray-origin displacement and underfilters shallow
views.  It would reproduce angle-dependent grain.

Convert the two columns to texels with

```text
D = diag(N_x/S_x, N_y/S_y)
a = D q_x
b = D q_y
g00 = dot(a,a),  g01 = dot(a,b),  g11 = dot(b,b)
lambda_max = 1/2 [g00 + g11
                  + sqrt((g00-g11)^2 + 4 g01^2)]
rho = sqrt(lambda_max).
```

`S_x,S_y` are tile metres and `N_x,N_y` are base-level interior texels.  The
strict one-colour-operation level is

```text
ell = clamp(ceil(log2(max(rho,1))), 0, ell_max).
```

The colour lookup is bilinear **within integer mip `ell`** and names layer
`j*`; it never blends mip levels or array layers.  Choosing `ceil` makes the
isotropic mip kernel at least as wide as the largest singular footprint.
Choosing a smaller statistic such as `sqrt(det)` or the geometric mean preserves
more apparent sharpness but necessarily leaves the major footprint direction
aliased, so it is not the moving-grain fix.

The derivative uses the current inverse projection, so temporal projection
jitter is already represented.  It uses the categorical local plane only for
footprint size; it does not average or mip the selected depth, owner, or normal.
At a true silhouette `|n dot d|` tends to zero, `rho` tends to the coarsest mip,
which is the correct band-limit rather than a numerical epsilon pretending the
footprint stayed small.

### 15.4 Opaque-resolve alpha limit

True filtered compositing would be

```text
out = C_premul + (1-A) B,
```

where `B` is the shaded visibility sample behind the grass.  The existing opaque
election retains only the winning grass visibility; after that election `B` is
not available.  Packing `A` into the winning ID or ray-normal target cannot
reconstruct an unknown `B`.  Therefore exact fractional alpha cannot reach the
current opaque resolve without retaining a second visibility layer or adding a
composite operation.

A stochastic alpha test before election would preserve `B` in expectation,
but it replaces the reported moving grains with Monte-Carlo visibility grains
and is explicitly rejected for this repair.  Multiplying the final colour by
`A` would incorrectly composite against black.  Mipping or averaging an owner,
depth, or normal to manufacture a fractional hit is also forbidden.

The no-new-pass contract therefore has one honest interpretation: `A` is used
inside the single colour sample to recover the coverage-conditioned
band-limited colour `C_premul/A`, while the level-zero categorical geometry
continues to provide binary screen ownership.  This can remove cross-view colour
veiling and sub-pixel chromatic shimmer in dense covered regions.  It cannot
mathematically antialias sparse silhouettes or cover/background occupancy in a
single-layer opaque visibility buffer.  If visible grain remains specifically
on those binary silhouettes after this carrier, the missing background layer is
the blocker; more colour taps or a different LOD formula are not a solution.

The same hard limit applies to nonlinear dynamic lighting from an under-resolved
categorical normal.  The normal remains the exact winner record as required.
One RGBA colour/coverage sample has no capacity for a filtered normal-distribution
model as well; doing that generally would require a wider/moment carrier or a
separate fixed read and must be justified by a post-colour-gate trace rather than
silently added here.

### 15.5 Fixed runtime cost and implementation contract

The complete isolated query is now:

```text
4 nearest level-zero initial geometry records
+ 4 nearest level-zero corrected geometry records
+ 1 winner-layer premultiplied colour/coverage record at integer footprint LOD
= 9 explicit texture operations.
```

Relative to the current twelve-operation path this removes three bilinear colour
operations and the four-node colour blend.  The Jacobian adds fixed arithmetic:
two homogeneous ray derivatives, two local plane differentials, the existing
chart differential applied twice, one 2-by-2 maximum singular value, and one
`log2`.  It adds no loop, march, candidate, branch-dependent query, pass,
barrier, dispatch, or runtime geometry.  Register pressure also falls because
only the winning slice, corrected address, point, normal, and one colour record
survive election.

The required binding transcription is exact:

1. geometry keeps the existing nearest, level-zero complete-record carrier;
2. colour becomes a periodic RGBA8 2D-array mip carrier, one layer per angular
   slice;
3. election returns `j*`, `q1_j*`, `c_j*`, `X1_j*`, and the categorical normal;
4. compute `rho` from the equations above using the same chart differential as
   the ray query;
5. issue one winner-layer colour sample at integer `ell`;
6. write `C/A` through the existing authored RGB packing; do not claim or fake
   fractional opacity in the opaque resolve.

The first visual gate must separate three phenomena: (a) cross-view veil/fuzz,
(b) chromatic minification shimmer on continuously covered regions, and (c)
binary silhouette/normal-lighting shimmer.  This model is expected to remove
(a) and (b).  Category (c) is a stated one-layer/wide-moment representation
limit, not permission to blend categorical records or add unbudgeted samples.

## 16A. Superseding event-completeness audit: a point miss cannot be a relay

The first root-rigid runtime gate made the logical defect visible: valid grass
is randomly absent and, at standing height, most coverage survives only for the
nearest roughly one or two metres.  The fixed graph is presently

```text
four point records R0_j at seed phase p_s
  -> provisional root r0 from one covered R0_j
  -> four point records R1_j at root-shifted corrected phases
  -> require covered(R0_j) && covered(R1_j) && root(R1_j)==r0.
```

That is not a coverage-complete first-event query.  It computes an intersection
of two sampled hit sets and then intersects it with a provisional-root label.
Sections 11 and 14 required those predicates as safeguards against fabricated
records, but the new gate proves that the safeguards expose a missing
categorical relay rather than supplying one.  This section supersedes their
claim that the current point records can form a complete root relay.

### 16A.1 Minimal initial-MISS counterexample

It is enough to work in the `x-y` plane and extrude the construction by an
arbitrarily small positive width in `z`.  Let the top be `H=1`, the baked ray be

```text
c = (0,-1),
```

and the live ray be

```text
d = (1,-1)/sqrt(2),    p_s=0.
```

The epipolar coefficient is

```text
B = (c_y/d_y)d_x-c_x = 1.
```

Place one small opaque horizontal patch around

```text
X = (1/2,1/2).
```

The live ray from `(0,1)` hits `X`.  The canonical baked ray through that same
point starts at `q*=1/2`, has `tau_c=1/2`, and satisfies the exact fixed-point
identity

```text
q* = p_s + B tau_c = 1/2.
```

But the initial baked ray at `p_s=0` misses the patch.  Therefore

```text
R0(p_s)=MISS
```

while the live ray has a true first hit.  Requiring `covered(R0)` deletes it.
This is angular parallax, not a numerical edge case, and it persists on an open
set after giving the patch finite width.

No dummy depth can repair a point MISS.  If the implementation assigns a
constant `tau_bar` to MISS and reads at `p_s+B tau_bar`, move the patch to any
`(a,1-a)` with `a != B tau_bar` inside the supported box.  The live ray still
hits, the point record remains MISS, and the dependent read goes elsewhere.
MISS has no depth from which the fixed-point address can be recovered.

### 16A.2 Minimal same-root counterexample

Let periodic root cells have unit width: `r0=[0,1)` and `r1=[1,2)`.  Retain
`H=1`, `c=(0,-1)`, `d=(1,-1)/sqrt(2)`, and set `p_s=0.25`.  Put two small
horizontal patches at the same height:

```text
A=(0.25,0.20), owned by r0,
B=(1.05,0.20), owned by r1.
```

The initial vertical ray hits `A`, so `tau0=0.80` and the provisional root is
`r0`.  Because the epipolar coefficient is one,

```text
q1 = p_s + B tau0 = 1.05.
```

The corrected vertical record is exactly patch `B`.  The live oblique ray also
passes exactly through `B`; it never passes through `A`, so `B` is its true
first hit.  Choose equal terrain heights for the two roots, making the
provisional root's entry plane numerically identical to `B`'s.  Even in this
most favourable case,

```text
root(R1)=r1 != r0,
```

and the same-root predicate deletes the true event.

If unequal terrain heights are allowed, merely removing the predicate is also
wrong: the corrected record may have been addressed using `G(r0)` although its
owner needs `G(r1)`.  Thus accept and reject are both unsound.  The final root
is needed before the root-specific address, but the current graph discovers it
only after that address.  This is a categorical circular dependency.

`tileContaining(X1.xz)` is additionally not an authored-root proof.  A leaf,
rhizome, litter fragment, or arbitrary input triangle may cross a periodic cell
edge while retaining the root which authored it.  GCRP/v4's offline owner token
contains primitive and periodic-copy identity, but the runtime coupled record
does not.  Inferring root from the hit coordinate can therefore reject even a
same-triangle correction.

### 16A.3 Impossibility with the current point record

The failure is information-theoretic.  A fixed graph observes finitely many
canonical point rays.  Once those values are fixed, place a sufficiently small
triangle on the live ray at an unsampled fixed-point phase, avoiding every
observed point ray.  The soup with that triangle and the soup without it produce
identical `R0` and `R1` records, while one live query is HIT and the other MISS.
No arithmetic rule over those records can return both correct answers.

The four angular corners do not change the proof: a thin surface may be visible
strictly inside their direction cell while all four corner rays miss, or a new
owner may become first strictly inside the cell.  Uploading the existing v4
owner token also does not change it.  That token names the winner of one sampled
ray; it contains no owner for a live-only event, no certified hit/miss closure
over the query cell, and no ordering after per-root terrain translations.

Therefore no coverage-complete rule exists using the current GCRP point record
under the frozen constraints.  In particular, none of the following is valid:

- dropping `initialCovered` and treating MISS depth as a correction seed;
- keeping `initialCovered` and accepting the eroded hit-set intersection;
- dropping `sameRoot` after addressing with the provisional root's height;
- deriving authored root from `tileContaining(X1)`;
- blending, mipping, dilating, or thresholding hit bits, depths, normals, or
  owners.

All trade a deterministic deletion for a fabricated surface or another
camera-dependent discontinuity.

### 16A.4 Minimal categorical representation which closes the logic

The minimal semantic change is not another scalar in the current hit record.
The first page must cease being a point first-hit sample and become a **total,
cell-certified boundary-event relay**.  Address it by the actual exterior box
entry `q`, the live direction `d`, and any declared physical footprint class.
Every addressed leaf is exactly one of:

```text
CERTIFIED_MISS:
  every supported ray in this query cell misses to the finite horizon;

CERTIFIED_HIT:
  one categorical owner/root/chart is first throughout the cell, with
  a proved positive incidence and order/domain margin;

FILTERED_ATOM:
  the cell is wholly below the declared physical resolution and stores one
  explicitly filtered appearance/depth-interval atom, not a fake owner.
```

A categorical HIT leaf must name at least

```text
authored primitive/chart id,
periodic copy and authored root id,
analytic plane/chart payload key,
material/species mark,
direct payload address or exact local coefficients,
domain/order certificate class.
```

The root is then known *before* its terrain height and affine translation are
applied.  There is no `initialCovered` predicate, because a relay MISS means the
whole cell is certified empty rather than one seed ray happened to miss.  There
is no posterior `sameRoot` predicate, because the selected leaf already owns its
root and payload.  A second page may contain the complete payload, but it is
addressed by the one relay leaf; it is not a list of candidate owners.

This can retain the nine-operation ceiling by replacing, not supplementing, the
current reads.  Direct fixed-depth addressing may use the four texture
operations currently allocated to the `R0` stage; fetching/decoding the one
selected complete payload may use the four operations currently allocated to
the `R1` stage; the ninth operation remains winner-coherent filtered colour.
Those operations are fields/pages of one selected leaf, not four independently
elected geometric candidates.  A regular direct-address table or fixed-depth
page layout preserves O(1); a variable decision walk would violate the contract.

The cook requirement is binding.  A cell containing a categorical hit/miss or
owner-order boundary must be subdivided until it is certified, or converted to
a physically justified filtered atom, or make compilation fail.  An arbitrary
triangle soup can require unbounded subdivision at resolved distance, so this
is a complete **compiler/rate-distortion contract**, not a proof that every
soup fits finite memory.  That is the finite-state boundary already proved in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-COMPLETE-MODEL-BOUNDARY.md`.

Per-root terrain translation adds one further non-negotiable condition.  The
first owner after independently translating roots depends on their terrain
heights.  A generic flat-community relay cannot certify that ordering from one
root token.  The relay must therefore be either:

1. cooked against the actual bounded terrain/community state for the streamed
   patch; or
2. addressed by a bounded, explicitly charged terrain-state class and certified
   for that whole class.

Otherwise exact selection requires trying multiple roots, which is precisely
the forbidden candidate problem.  The previously derived common guarded-box
boundary transfer is the appropriate upstream domain: an exterior ray has one
camera-independent boundary entry, and the cooked field directly returns the
whole-community first event.  The ordinary flat GCRP point atlas is not such a
field.

### 16A.5 Why this matches the observed artifact family

The current acceptance mask is approximately

```text
H_live_approx = H_seed intersect H_corrected intersect SameRoot.
```

That single erosion mechanism explains the new and older observations without
inventing separate filters:

- **nearest one-to-two metres only:** close rays have smaller seed-to-fixed-point
  displacement, so the two sampled hit sets overlap more often; shallow/far
  rays separate them and disappear;
- **near detail looks half missing:** thin blades and panicle branches are the
  most likely to be a hit at only one of the two addresses;
- **flat or fuzzy side silhouettes:** silhouettes are hit/miss boundaries, so
  intersecting two shifted masks erodes and doubles them; temporal accumulation
  turns the flipping binary boundary into fuzz;
- **perfect circular/elevation zones:** the phase displacement grows primarily
  with view elevation and range, while provisional-root changes occur on a
  periodic grid; their screen projection produces coherent radial/elevation
  bands rather than botanical structure;
- **distant moving grain:** sub-pixel camera motion moves high-frequency
  `initialCovered`, `correctedCovered`, and root-equality boundaries.  Their
  binary accept/reject flips are accumulated as grains.  Section 15's colour
  under-minification remains an independent possible contributor, but it cannot
  restore geometry which this graph deleted.

The next runtime implementation must therefore not tune these three predicates.
It must either bind a certified whole-cell boundary-event relay, or restore the
last non-root-rigid visual checkpoint while that cook-side representation is
produced.  More epipolar ALU, colour mips, or permissive MISS handling cannot
make the current point atlas coverage-complete.

This conclusion is independent of the terrain carrier.  The counterexample in
Section 16A.1 uses flat terrain, so replacing per-root placement by a continuous
drape can remove root seams but cannot make an `initialCovered` point sample a
complete relay.  Any later carrier derivation which retains that veto remains
coverage-incomplete until it binds the certified event representation above.

## 16. Implemented root-rigid conformance table

This table binds the implementation to Section 14 after transcription.  It
supersedes the old terrain-tangent conformance claims in Section 12 for the
isolated `grassprofile=2` path.  The row numbers below are source-line anchors,
not a second specification: the equation in the left column remains binding.

| Mathematical quantity / invariant | Runtime transcription | Source |
| --- | --- | --- |
| Seed carrier derivative `k_seed=d_y` | The isolated branch selects `rd.y`; the terrain-gradient derivative is retained only by the legacy multi-profile branch. | `LegacyPeriodicRayQuery.ts:297-302` |
| Horizontal seed-top entry `t_seed=(G(O)+H-o_y)/d_y` | `qTop=H`, `qO=o_y+t_scene d_y-G(O)`, and `dtE=(qTop-qO)/rd.y`; therefore `tE=t_scene+dtE` is the stated plane intersection. | `LegacyPeriodicRayQuery.ts:303-318` |
| Live direction is the world direction `d` | Isolated `dpx,dpy,dpz` are exactly `rd.x,rd.y,rd.z`; the normalisation is identity up to floating-point roundoff because `rd` is already unit length. | `LegacyPeriodicRayQuery.ts:456-470` |
| Initial categorical records `R0_j` and depths `tau0_j` | Four initial nearest records are supplied by the angular address, coverage stays categorical, and depth is decoded per elevation row. | `LegacyPeriodicProfileSampler.ts:308-359` |
| Provisional periodic owner/root `r0` | Each initial event reconstructs `q_seed+tau0_j c_j`; `tileRoot` classifies that event, then the highest eligible barycentric weight elects one root without averaging roots. | `LegacyPeriodicProfileSampler.ts:327-379` |
| Root-rigid terrain carrier `Phi_r(X)=X+G(r)e_y` | The elected root is inverse-mapped into the guide frame, the existing bilinear `bguide(root)` read supplies `G(r)`, and that same guide record is retained for the later ecological test. | `LegacyPeriodicRayQuery.ts:493-520`, `732-759` |
| Root-stable top entry `t_E(r)=(G(r)+H-o_y)/d_y`, `p_r=o_xz+t_Ed_xz` | `rootEntryT` is solved directly from `rootGround+topH`; `phAt(rootEntryT)` supplies the live top-plane point, which is transformed once into authored address space. | `LegacyPeriodicRayQuery.ts:504-526` |
| One-step epipolar address `B_j=(c_j.y/d.y)d.xz-c_j.xz`, `q1_j=p_r+B_j tau0_j` | `verticalRatio`, `bx/bz`, and `correctedAddress` are the equation term for term. | `LegacyPeriodicProfileSampler.ts:392-403` |
| Complete corrected categorical record `R1_j` | Each node performs one corrected geometry read and decodes depth, coverage, and normal from that same record; none are blended across nodes. | `LegacyPeriodicProfileSampler.ts:403-412` |
| Corrected owner point `X1_j=(q1_j,H)+tau1_j c_j` | `ownerPoint` is constructed directly in the same root-rigid authored frame. | `LegacyPeriodicProfileSampler.ts:408-412` |
| Exact same-plane live intersection `s_j=n_j dot (X1_j-P_r)/(n_j dot d)`, `X_j=P_r+s_jd` | `incidence`, `numerator`, `liveDepth`, and `point` implement the plane equation; this replaces the old orthogonal projection onto `d`. | `LegacyPeriodicProfileSampler.ts:413-420` |
| Root closure and exterior-forward eligibility | A node survives only if both records are covered, `tileContaining(X1_j)=r0`, `|n dot d|>10^-6`, and `s_j>=0`.  These tests reject a failed hypothesis; they do not clamp or alter an accepted intersection. | `LegacyPeriodicProfileSampler.ts:421-431` |
| One complete winning surface | The greatest eligible barycentric weight selects depth, live point, owner point, normal, address, azimuth, and elevation together. | `LegacyPeriodicProfileSampler.ts:448-467` |
| Winner-coherent radiance and nine-read budget | Only the winning address/slice performs a colour read: four initial geometry + four corrected geometry + one colour.  No cross-owner colour blend remains. | `LegacyPeriodicProfileSampler.ts:468-497` |
| World depth `t_world=t_E(r)+s` | The isolated layer returns `dt=(t_E(r)-t_seed)+s`, and the outer query adds `t_seed`, cancelling it exactly. | `LegacyPeriodicRayQuery.ts:530-546`, `LegacyPeriodicRayQuery.ts:704-712` |
| Root-rigid geometric normal | The isolated authored normal receives only the authored layer rotation and normalization; terrain-gradient inverse-transpose shear is absent.  Shading then uses that authored normal with zero terrain-normal pull. | `LegacyPeriodicRayQuery.ts:562-575`, `LegacyPeriodicGroundCoverShade.ts:293-301` |

The implementation therefore adds no sample, loop, candidate list, pass,
barrier, dispatch, runtime carrier mesh, or per-species work.  Its explicit
profile traffic is nine operations per queried pixel.  The remaining theorem
boundaries are unchanged: the current record does not certify same-triangle
margin or encode a side chart for exactly horizontal rays, and initial-miss
birth events cannot be recovered by inventing a depth from an uncovered record.

## 17. Adversarial review: no seamless curved-terrain conjugation from one flat bake

### 17.1 Verdict and theorem

Sections 14 and 16 make a root-rigid tile camera independent, but they do not
make an infinite set of such tiles a continuous curved-terrain attachment.  The
reported rectangular repetitions and short visible range are predicted by that
distinction.

**No-free-conjugation theorem.**  Let `S` be an arbitrary periodic triangle soup
in a flat authored box and let `G` range over continuous non-affine heightfields.
There is no camera-independent pointwise map `Phi_G` which simultaneously:

1. attaches `S` continuously to `G` across every periodic tile boundary;
2. maps every exterior world ray to an ordinary straight authored ray, so one
   flat precomputed first-hit field remains exact; and
3. can be inverted for first visibility with a fixed finite number of point
   queries for every `S` and `G`.

A continuous nondegenerate map which sends all lines to lines is projective.
Global periodic translation equivariance and the absence of a finite projective
horizon force its denominator to be constant, hence the map is affine.  An
affine map sends the authored ground plane to another plane, not to arbitrary
`G`.  If line preservation is relaxed, the inverse of a world ray is curved and
its first intersection is an implicit visibility problem.  A fixed number of
samples cannot solve that problem for arbitrary `G`: two smooth terrains can
agree at every sampled point while an unsampled narrow bump moves a periodic
copy in front of the reported hit.

This does not say curved-terrain ground cover is impossible.  It says the four
properties “one flat bake, arbitrary soup, arbitrary curved terrain, exact fixed
pointwise query” cannot all be retained.

### 17.2 Minimal counterexample and the implicit root

In a two-dimensional cross-section, drape even the authored horizontal segment
`h=H` pointwise:

```text
Phi_drape(x,H) = (x, G(x)+H).
```

For world ray `y=o_y+m(x-o_x)`, its hit satisfies

```text
G(x) + H = o_y + m(x-o_x).
```

For quadratic `G` this is already a quadratic with zero, one, or two roots; a
smooth arbitrary `G` can produce arbitrarily many ordered roots.  Equivalently,
when parameterized by ray time,

```text
x = o_xz + [(G(x)+h(x)-o_y)/d_y] d_xz.
```

Thus evaluating terrain at the reconstructed surface point is the correct
camera-independent attachment, but the surface point appears on both sides of
the equation.  It is not a post-hit vertical adjustment.

Rigid botanical components replace `x` by a stable authored root `r_T`:

```text
Phi_T(X) = X + G(r_T)e_y.
```

This preserves each plant, but `T` must be the first triangle *after every
component has received its different root translation*.  Therefore

```text
T* = firstHit({Phi_T(T)}),       r* = r_(T*)
```

is a categorical implicit equation.  A flat-bake provisional event does not
determine `T*`: raising a later root can reorder it in front, and an oblique ray
can cross arbitrarily many repeated components.  One correction is a bounded
hypothesis, not an unconditional solution for arbitrary soup.

Using one height `G(c_k)` for the whole periodic tile avoids bending only by
creating the step

```text
Delta_k = G(c_(k+1)) - G(c_k)
```

at its boundary.  In general `Delta_k != 0`, so plants and density expose
rectangular slabs.  Interpolating adjacent tile translations removes the step
but makes `Phi` non-affine; the curved implicit-ray equation immediately
returns.  There is no blend that preserves both continuity and the flat bake's
straight-ray conjugacy.

### 17.3 Why the present failure collapses with range and grazing angle

The Section-14 seed and root entries differ by

```text
p_r - p_s = [G(r)-G(O)] (d_xz/d_y).
```

The mismatch therefore grows with terrain-height separation and with
`|d_xz/d_y|`.  Once it crosses a periodic root boundary, the posterior
`tileContaining(X1)==r0` test rejects the event.  The acceptance set is made from
tile predicates around a camera-derived seed, so its visible boundary naturally
appears as grid-aligned chunks inside roughly radial or sector-shaped range
zones.  This single premise explains, together:

1. **root-hypothesis/whole-tile attachment** — rectangular cells, circular or
   radial zones, the roughly 1--2 metre visible range, slope swim/stretch,
   missing events, and grazing-angle weakness;
2. **incomplete exterior/event data** — weak side silhouettes and further
   missing geometry (no exact horizontal chart, no unseen angular birth, no
   successor/root stack);
3. **underfiltered one-layer visibility** — distant fuzz/grains and disappearing
   fine heads, as already proved in Sections 8 and 15; it cannot explain the
   rectangular range collapse;
4. **single categorical normal plus opaque resolve** — flattened or unstable
   sub-pixel lighting; it cannot repair attachment or firstness.

Per-symptom thresholds, colour tweaks, or more epipolar algebra do not address
cause 1.

### 17.4 Fixed-cost representation ranking

| Representation | Exactness on arbitrary curved terrain / soup | Runtime profile traffic | Required bake data and consequence |
| --- | --- | ---: | --- |
| **Terrain-conditioned world atlas** | Attachment and firstness are exact at baked direction samples because the already-draped world community is baked as one object; angular interpolation remains its stated approximation. Shared world coordinates/gutters remove tile seams. | Existing `4+4+1 = 9` pattern, or fewer with a directly addressable codec. | Directional complete records per streamed terrain area, with world-consistent overlap. Data scales with terrain area and directions; it gives up one-bake infinite reuse but is the strongest fixed-tap arbitrary-soup option. |
| **Per-component rigid roots with a `K`-event stack** | Exact only if authoring proves that at most `K` translated components can reorder on every ray. No finite `K` is unconditional for arbitrary periodic soup. | `O(K)` fixed reads; under the current two-stage four-node chart, approximately `8K+1` profile reads plus root-height data. | Every triangle needs a stable component/root ID and every texel/direction needs ordered categorical successor events. Useful only with a measured hard depth/reorder bound. |
| **Known single affine terrain patch** | Exact while the complete event and ray remain inside the known patch; not exact across an unbounded number of curved-terrain patches. | Approximately 9 plus patch/frame lookup. | Geometry clipped/owned by patch and transition records. Piecewise patches need candidates at boundaries; categorical blending between patches is invalid. |
| **Finite slope/curvature-class bakes** | Approximate. Class switching is itself a seam/swim source, and blending classes blends unrelated first hits. | About 9 after class selection. | One atlas per terrain class and a stable world class field. More classes reduce local error but cannot prove arbitrary-terrain exactness. |
| **One flat bake with pointwise drape** | Seamless geometric map, but exact visibility requires solving the curved implicit ray and potentially unbounded first-event order. | Variable traversal/march; not O(1). | One bake plus queryable `G`; violates the runtime contract rather than the visual contract. |
| **Whole-tile/root-rigid translation (Sections 14/16)** | Affine-exact only inside a selected tile whose event order remains unchanged. Discontinuous between unequal tile heights and hypothesis-dependent before the hit. | 9 plus reused terrain/root reads. | Current records plus a root mark. Cheapest, but mathematically matches the observed chunk/range failure and is not a general curved-terrain carrier. |
| **Runtime triangles/meshlets or exact traversal** | Exact with stable per-component roots and ordinary world visibility. | Not a fixed texture-tap query. | Runtime geometry/acceleration data; explicitly outside the project contract. |

The practical fork is therefore upstream: constrain the authored community so a
small `K` root-event bound is true, or bake terrain-conditioned world visibility.
Continuing to infer one root from one flat-bake event cannot become unconditional
for arbitrary triangle soup through another fixed correction.

## 18. Superseding carrier: continuous drape, not per-copy terrain steps

The live gate of the Section-14 transcription falsified its key authoring
premise.  The result split into rectangular periodic chunks and standing-height
visibility collapsed to roughly one or two metres.  The equations below explain
that result exactly and supersede the **per-root height placement** in Sections
14 and 16.  The same-plane intersection derived in Section 14 remains valid and
is retained, but it must be applied to a continuous world carrier.

### 18.1 Why the root-rigid carrier necessarily makes a grid

The implemented root is

```text
r(x) = tileOrigin + (floor((x-tileOrigin)/S)+1/2) S,
S = (0.52 m, 0.52 m).
```

Therefore the Section-14 vertical datum

```text
g_r(x) = G(r(x))
```

is piecewise constant.  At a periodic seam `x_s`, its two limits are

```text
lim_(x->x_s-) g_r(x) = G(r_-),
lim_(x->x_s+) g_r(x) = G(r_+).
```

On curved terrain these differ generically.  Two authored copies which meet
perfectly at the periodic seam are consequently separated in world Y by

```text
Delta_g = G(r_+) - G(r_-).
```

No ray equation or normal correction can remove that discontinuity because it
is the definition of `Phi_r`.  The visible rectangular chunks are its exact
image, not a shader plumbing defect.

The range collapse follows from the same quantisation.  At elevation `alpha`,
changing the root datum by `Delta_g` changes the top address by

```text
|Delta p| = |Delta_g| cot(alpha).
```

Even a 5 cm root-height difference shifts the query by 0.57 m at 5 degrees,
already wider than the 0.52 m periodic cell.  The corrected hit then commonly
belongs to the adjacent copy, and the required
`tileContaining(X1)=provisionalRoot` test rejects it.  Root changes under camera
translation also switch `p_r` by this finite amount.  Thus the same equations
force the grid, the short range, and a new source of swim.

The deeper representation error is the use of a categorical visibility owner
as the coordinate chart of the terrain.  Owner/root is legitimate metadata for
species, colour, ecological eligibility, and wind group.  It is not a continuous
geometric datum.  Terrain placement must be evaluated at the continuous authored
base coordinate, never at `floor`-selected periodic ownership.

This is the same role-confusion behind the earlier artifact family.  The
endpoint tangent made a camera-ray datum define the world carrier (slope
stretch/swim); the endpoint/root mask made ecological metadata clip geometry
(flat/fuzzy edges); orthogonal projection made an owner sample pretend to be a
live surface point (concentric elevation zones); cross-node colour blending made
unrelated categorical events pretend to be one radiance sample (near-detail
veil); and root height made owner metadata define geometry (rectangular chunks
and range collapse).  The corrected model keeps three objects distinct:

```text
continuous world carrier Phi,
categorical authored owner point U,
actual live-ray surface point Y.
```

Section 15 adds the fourth distinct object, a footprint-filtered radiance
measure.  None may be substituted for another merely because all are available
at the same pixel.

### 18.2 One coherent terrain function from the guide records already loaded

Let `G` be a camera-independent `C1` terrain height field.  The carrier for an
authored point `U=(u_x,h,u_z)` is the vertical drape

```text
Phi(U) = (u_x, G(u_x,u_z)+h, u_z).
```

This map is continuous across every periodic seam because the unwrapped world
coordinate `u` is continuous and `G` is not periodic or owner-quantised.  It
also supports overlapping species: every authored triangle uses the same map,
while categorical owner data remains attached to the triangle.

The guide currently loads, at each of four corners, one height and its two
stored derivatives.  Those twelve scalars are sufficient to define a coherent
tensor-product bicubic Hermite patch with the four mixed derivatives set to the
same global convention `G_xz=0`.  For `t in [0,1]`, define

```text
V0(t)= 2t^3-3t^2+1,       D0(t)= t^3-2t^2+t,
V1(t)=-2t^3+3t^2,         D1(t)= t^3-t^2.
```

For cell pitch `P`, corner heights `g_ij`, and world derivatives
`gx_ij,gz_ij`, use

```text
G_H(u,v) = sum_(i,j in {0,1}) [
    g_ij       Vi(u) Vj(v)
  + P gx_ij    Di(u) Vj(v)
  + P gz_ij    Vi(u) Dj(v)
],
```

with analytic derivatives obtained by differentiating the same polynomial and
dividing normalised-coordinate derivatives by `P`.  Adjacent cells share the
same endpoint height, both endpoint derivatives, and the same zero mixed
derivative.  Their edge value and both first-derivative cubics are therefore
identical.  Hence `G_H` is `C1` across guide-cell edges.  This replaces the
mathematically inconsistent pair “bilinear height plus independently bilinear
gradient” without another guide record.

The exact carrier Jacobian and inverse-transpose normal map are

```text
J(u) = [ 1      0      0
         G_x    1      G_z
         0      0      1 ],

n_world = J(u)^(-T) n_authored
        = (n_x-n_y G_x, n_y, n_z-n_y G_z).
```

Position, ray differential, and normal now all use one terrain function at one
world point.  No endpoint gradient and no root height is allowed to leak into
the final surface.

### 18.3 Fixed two-stage query using the existing dependent read

The inverse image of a straight world ray under `Phi` is curved, so one
straight-ray profile lookup cannot be globally exact on arbitrary curved
terrain.  The strongest fixed construction already compatible with the two
geometry stages is a predictor/corrector on that curve.

For any world-XZ anchor `a`, let `g_a=G_H(a)` and `m_a=grad G_H(a)`.  The affine
tangent carrier is

```text
G_a(x) = g_a + m_a dot (x-a).
```

For world ray `R(t)=o+t d`, its authored top entry and chart derivative in this
carrier are

```text
k_a  = d_y - m_a dot d_xz,
tE_a = [g_a + H + m_a dot (o_xz-a) - o_y] / k_a,
p_a  = o_xz + tE_a d_xz,
v_a  = (d_x, k_a, d_z),
d_a  = v_a / |v_a|.
```

There is no clamping of `k_a`.  `k_a=0` means the world ray is parallel to that
local top plane and this chart is singular; a side chart is the already-recorded
missing datum for the exact horizontal limit.

Use the terrain endpoint `O` only as the continuous predictor anchor.  The four
existing initial records produce a categorical provisional authored point `U0`.
Evaluate the already-budgeted intermediate guide lookup at `a=U0.xz` -- **not**
at `tileContaining(U0)` -- and form `p_a,d_a` above.  For each already selected
baked direction `c_j`, the existing corrected read becomes

```text
B_j(d_a) = (c_j.y/d_a.y) d_a.xz - c_j.xz,
q1_j     = p_a + B_j(d_a) tau0_j,
R1_j     = record_j(q1_j),
U1_j     = (q1_j.x,H,q1_j.z) + tau1_j c_j.
```

The greatest-weight eligible node supplies the owner point `U1`, authored
normal `n`, colour address, and species metadata.  No periodic-root equality
participates in geometry eligibility.

At `u1=U1.xz`, evaluate the already-existing final surface guide lookup and set

```text
Y1 = Phi(U1) = (u1.x, G_H(u1)+U1.y, u1.z),
N1 = (n.x-n.y G_x(u1), n.y, n.z-n.y G_z(u1)).
```

The actual live-ray point is not `U1` and is not an orthogonal projection of
`U1`.  It is the ray intersection with this returned surface tangent plane:

```text
t_plane = dot(N1, Y1-o) / dot(N1,d),
Y       = o + t_plane d.
```

`U1` remains the categorical **owner point** used for authored height, colour,
normal source, periodic copy identity, and ecological root.  `Y` is the
continuous **surface point** used for world depth and compositing.  Conflating
those two points recreates the phase and metadata swim diagnosed in Section 9.

### 18.4 Why the plane solve removes the concentric elevation zones

The former output used

```text
s_proj = dot(U1-P,d),
```

which is an orthogonal projection, not a ray/surface intersection.  Decompose
`v=U1-P` into

```text
v = s_proj d + e,       dot(e,d)=0.
```

For the triangle plane through `U1` with normal `n`, the exact ray-plane
parameter is

```text
s_plane = dot(n,v)/dot(n,d)
        = s_proj + dot(n,e)/dot(n,d).
```

The omitted term depends on the transverse error `e`.  `e` changes abruptly
when the categorical elevation row changes, so the omission creates concentric
constant-row depth zones, apparent stretch, and row-boundary swim.  If adjacent
rows return the same triangle plane, their `U1,n` pairs define the same plane and
the plane solve returns the same intersection: the artificial zone disappears
exactly.  If they return different owners, the discontinuity is a visibility
correspondence problem and cannot be repaired by arithmetic blending.

The denominator is not clamped.  `dot(N1,d)=0` is a tangent/coplanar degeneracy,
not permission to fabricate a top-plane hit.  A finite-precision eligibility
test may reject the measure-zero event; it must not alter an accepted depth.

### 18.5 Continuity proof and curvature error bound

Assume for one open ray neighbourhood that initial and corrected records remain
on the same authored triangle, the visibility order does not change, and
`|k_a|` and `|N1 dot d|` are bounded away from zero.  Then:

1. `G_H` and its gradient are continuous across guide cells by Section 18.2.
2. The authored profile is periodic, so unwrapped `U0,U1` have equal limits at
   the two sides of a periodic texture seam.
3. `p_a,d_a,q1,Y1,N1,t_plane` are compositions of continuous operations with
   nonzero denominators.

Therefore the **carrier and exact same-owner reconstruction** are continuous
across periodic seams, guide-cell boundaries, terrain slopes, and continuous
camera motion in that neighbourhood.  The finite implementation still selects
nearest complete geometry texels and categorical angular nodes.  Without stable
same-primitive correspondence, those discrete records can jump at a texel or
node boundary even when the underlying carrier is continuous.  Such jumps are
bounded by atlas discretisation and do not align into 0.52 m root rectangles or
terrain-guide cells, but strict end-to-end continuity cannot be claimed from the
current record.  The missing correspondence named in Section 18.6 is what would
turn this conditional proof into an implementation proof.

For an authored triangle plane, define the exact draped implicit surface

```text
F(x,y,z) = n_x(x-u1.x) + n_z(z-u1.z)
         + n_y[(y-G_H(x,z))-U1.y].
```

If `||Hess G_H|| <= K` over horizontal radius `rho` from `u1`, the local tangent
remainder is at most `K rho^2/2`.  If
`|grad F(R(t)) dot d| >= sigma > 0` over the correction interval, the plane
solution's ray-parameter error is bounded by

```text
|t_plane-t_exact| <= |n_y| K rho^2 / (2 sigma).
```

The error is second order in local horizontal displacement.  The previous
endpoint tangent accumulated the same curvature over the full ground-to-top
grazing traverse; the corrected anchor moves it to the returned surface and
therefore reduces `rho` to the one-step correspondence residual.

### 18.6 What cannot be proved from the current record

The construction above proves carrier continuity and exactness on affine terrain
for a stable triangle.  It cannot prove unconditional exact first visibility for
an arbitrary soup, because the current complete record does not contain a stable
primitive ID, triangle support/edge margins, or the angular boundary at which a
successor becomes first.  Two soups can have identical records at the sampled
nodes and different live-direction winners between them.  The smallest useful
additional precompute is therefore a **categorical visibility correspondence**:

```text
(stable primitive/owner id,
 plane anchor or barycentric point,
 in-triangle support/edge data,
 direction-cell firstness boundary or successor id).
```

Owner ID alone detects a mismatch but cannot recover the missing successor;
normal plus point defines an infinite plane but cannot prove the intersection is
inside the finite triangle.  Exact arbitrary curved-terrain intersection can
also cross an unbounded number of guide cells at grazing.  Closing that case
without traversal requires a terrain-conditioned first-hit carrier (or an exact
side/event chart) precomputed for the world terrain.  No fixed ALU expression of
the currently returned event can infer data which was never stored.

These are explicit quality boundaries, not reasons to restore per-copy terrain
steps, blend owners, clamp directions, or add runtime candidates.

### 18.7 Superseding implementation equations and cost

For isolated `grassprofile=2`, replace the Section-14/16 root geometry with:

```text
DELETE FROM GEOMETRY:
  G(tileContaining(U0)), root-stable top entry, sameRoot eligibility,
  root-rigid normal, and rootEntry depth offset.

KEEP:
  categorical R0/R1 records, full depth, one epipolar correction,
  winner-coherent colour, and the same-plane `n dot v / n dot d` solve.

USE:
  one coherent G_H and grad G_H from each loaded guide record;
  predictor tangent at O only for R0;
  corrector tangent at continuous provisional U0.xz for R1;
  final Y1=Phi(U1), N1=J(U1)^(-T)n;
  t=dot(N1,Y1-o)/dot(N1,d).

ROOT AFTER GEOMETRY ONLY:
  tileContaining(U1.xz) may select ecology/species metadata,
  but it never changes terrain height, entry address, depth, or normal.
```

Profile traffic remains exactly four initial geometry records, four corrected
geometry records, and one winning colour record.  Existing intermediate/final
guide lookups are retargeted from categorical root to continuous points; no
texture tap, binding, pass, barrier, dispatch, loop, march, candidate, runtime
mesh, or per-species query is added.  Bicubic basis/gradient evaluation and one
winner plane dot/divide are fixed ALU.  Root-equality predicates and root-entry
arithmetic disappear.  Register lifetime is comparable: continuous anchor,
height, and gradient replace the provisional root, root height, and root entry.

Section 15's footprint radiance carrier remains the independent fix for
under-minification grain.  It must use this section's continuous world normal
and surface point in its Jacobian; it must not use the discarded root-rigid
normal or per-copy entry.

## 19. Continuous-drape transcription and failed birth ablation

Section 16 is no longer the active implementation table.  Its root-rigid
transcription produced the exact failure predicted by Sections 17--18: visible
`0.52 m` rectangular copies, shallow-angle disappearance beyond roughly
`1--2 m`, and camera-coupled sector changes.  The root-equality predicate and
root-height entry have been removed from runtime geometry.

The active isolated path now binds to Section 18 as follows:

| Equation / invariant | Active code |
| --- | --- |
| One `C1` `G_H` and analytic `grad G_H` from the four guide records | `LegacyPeriodicRayQuery.ts:165-264`; tensor Hermite shares corner value/first derivative and zero mixed derivatives across guide cells. |
| Predictor tangent at `O` | `LegacyPeriodicRayQuery.ts:351-365`; `k_O=d_y-grad G_H(O) dot d_xz`, with no isolated-path clamp. |
| Initial categorical point `U0` | `LegacyPeriodicProfileSampler.ts:329-376`; four initial records elect one complete covered point, never an averaged point. |
| Corrector tangent at continuous `U0.xz` | `LegacyPeriodicRayQuery.ts:535-595`; `U0` is inverse-mapped continuously, `G_H(U0)` and its gradient form `tE_a,p_a,d_a`, and periodic tile identity is absent. |
| Existing four corrected reads | `LegacyPeriodicProfileSampler.ts:386-425`; `q1_j=p_a+B_j(d_a)tau0_j`, followed by one complete nearest record per node. |
| Categorical complete-surface election and one colour read | `LegacyPeriodicProfileSampler.ts:426-470`; point, normal, address, direction row, and colour stay winner-coherent.  Explicit profile traffic remains `4+4+1=9`. |
| Final continuous drape `Y1=(u1,G_H(u1)+h1)` | `LegacyPeriodicRayQuery.ts:597-614`; terrain is evaluated at the winning authored surface point, not a tile/root centre. |
| Inverse-transpose normal `N1=(nx-ny Gx,ny,nz-ny Gz)` | `LegacyPeriodicRayQuery.ts:601-611`; the same final `G_H` supplies the derivative. |
| Live depth `t=dot(N1,Y1-o)/dot(N1,d)` | `LegacyPeriodicRayQuery.ts:615-624`; the compositing point is the live ray/plane intersection and remains distinct from categorical owner point `U1`. |
| Roots affect ecology only | `LegacyPeriodicRayQuery.ts:625-650`, `849-875`; periodic root coordinates are computed after geometry and never change height, entry, depth, or normal. |
| Authored lighting normal remains authored | `LegacyPeriodicGroundCoverShade.ts:293-301`; no later terrain-normal pull destroys the draped geometric normal. |

The first private real-WebGPU gate passed and removed the rectangular patchwork
and standing-radius collapse.  It is not yet user acceptance.

One tempting current-record ablation was also tested and rejected before user
exposure.  When a node's `R0` missed, the ablation reused the covered
provisional neighbour's categorical height to address that node's `R1`, then
allowed the born node to win by angular weight.  This used no dummy miss depth
and no extra read, but a settled frame developed a large coherent diagonal
visibility wedge: the newly born record can name an unrelated first surface
and displace the valid covered candidate without a primitive/support/firstness
certificate.  The ablation is not active; `R0` and `R1` coverage are both still
required at `LegacyPeriodicProfileSampler.ts:386-411`.

Therefore the carrier regression is fixed independently, while the half-missing
detail remains the exact Section-16A record-closure blocker.  The current point
record cannot be made coverage-complete by another eligibility predicate.  The
resume condition is a cook-side total relay/categorical chart which supplies a
real first event (or certified miss/filtered atom) for the live cell inside the
same fixed `4` address + `4` payload + `1` colour budget; no more shared-depth,
root, support-radius, or relaxed-predicate variants are allowed.

## 20. User-accepted active defect ledger after the continuous-drape gate

The continuous carrier removed the explicit `0.52 m` grid chunks and the
standing-view `1--2 m` radius collapse.  It did **not** materially improve the
main quality target.  The following defects remain active, using the user's
clarified descriptions rather than inferred renderer terminology:

1. Uphill/sloped views fold, smear, or stretch the plants and contain sharp
   slicing/discontinuities.  Rising terrain often shows projected plant tops
   where plant sides should be visible.
2. Ground-level and low-oblique exterior views remain wrong even though a view
   near `10 m` above the field can look plausible.
3. Plants stretch along the viewing direction into streaks and widening
   wedges/triangles, increasingly with distance.
4. Oblique/distant views often present the wrong apparent plant angle, like a
   flattened top-view texture or stripes instead of a side silhouette.
5. Stretching still changes during camera translation, although the swimming
   has lessened.
6. A persistent one-direction stream of tiny moving particles remains,
   especially outside the immediate foreground.
7. Quality becomes unacceptably fuzzy, blurry, and grainy after roughly `2 m`,
   including distances such as `5 m` where the plants should remain legible.
8. Nearby leaves, branches, stems, and plume details appear randomly skipped or
   only partly reconstructed.
9. Violet plume heads can disappear before their green stems, especially near
   cover boundaries.
10. Cover boundaries read as a floating plane or detached point cloud over
    exposed terrain rather than rooted plants with coherent sides.
11. Side silhouettes are fuzzy and fragmentary rather than coherent volumes
    from root to plume.
12. Three perfect camera-centred circular quality/reconstruction bands remain
    unresolved.
13. Very-close top-down views can expose radial/triangular sectors cutting
    through each other.
14. Normals and lighting remain suspect because the plants do not consistently
    read as correctly oriented three-dimensional surfaces.
15. Across exterior angles the accepted source is not preserved well enough:
    the result often resembles particles or a textured field rather than
    recognisable Calamagrostis geometry.  Perfect preservation is not required.
16. The isolated single-species path remains the acceptance path; the legacy
    multi-species view stays disabled until it is correct.

Colour, the accepted source plume shape, removal of the old floating `1.176 m`
carrier, removal of grid chunks, and removal of the standing-radius collapse are
not active defects.

## 21. Review candidate A: restore predictor/corrector differential identity

This section is a proposal for independent review.  It is not yet implemented.

### 21.1 The necessary identity

Let `P=GUIDE_PITCH=0.84 m`, let `x` be world horizontal position in metres,
`X=x/P` the guide coordinate, `s` the authored layer scale, and `R` the layer
rotation.  The standalone authored address is

```text
q(x) = R[s P X] + phase = R[s x] + phase.
```

For a world ray `x(t)=x0+t d_xz`, its authored horizontal differential is

```text
dq/dt = R[s d_xz].
```

This derivative is independent of `P`: the conversion from metres to guide
coordinates contributes `1/P` and the authored-address conversion contributes
`P`, so the two factors cancel.  Predictor and corrector must use this same
horizontal differential on flat terrain.  If their terrain gradients are also
equal, they must produce exactly the same normalised authored ray direction.

### 21.2 Current code mismatch

The predictor at `LegacyPeriodicRayQuery.ts:508-517` uses

```text
profileDerivativeScale = s P / P = s
d_initial = normalize(R[s d_xz], k_O),
```

which is dimensionally correct.  The corrector at lines `586-592` instead uses

```text
d_corrector_actual = normalize(R[s P d_xz], k_a).
```

The guide pitch has been multiplied in a second time.  Even for flat terrain,
where `k_a=k_O`, the corrector therefore disagrees with the predictor:

```text
expected horizontal factor = s,
actual horizontal factor   = 0.84 s.
```

The sampler then uses this wrong direction in

```text
q1 = p_a + [(c_y/d_a.y) d_a.xz - c_xz] tau0.
```

Because the ratio `d_a.xz/d_a.y` is what matters, normalising the vector does
not cancel the error.  Before rotation, the address error is

```text
Delta q = (P-1) (c_y/k_a) d_xz tau0.
```

If `|c_y tau0|` spans the `1.176 m` profile height, its magnitude is about

```text
0.70 m at 15 degrees elevation,
1.07 m at 10 degrees,
2.15 m at  5 degrees.
```

The repeating tile is only `0.52 m`, so a corrected read can query the wrong
periodic copy by several whole plants.  This is a direct mathematical mechanism
for shallow-angle stretching, wrong apparent view, camera-motion swim,
predictor/corrector coverage erosion, and coherent phase shimmer.

### 21.3 Proposed code transcription

Change only the corrector's two horizontal derivative expressions:

```text
qdx = R_x[s d_xz]
qdz = R_z[s d_xz]
```

by reusing the already-computed `profileDerivativeScale`, exactly as the initial
stage does.  Equivalently, divide the current `s*coordinateScale` factor by
`GUIDE_PITCH`.  Reuse is preferred because it makes predictor/corrector identity
structural rather than duplicating the conversion.

No texture read, texture, binding, pass, barrier, dispatch, branch, candidate,
loop, march, register-class object, memory allocation, or synchronisation is
added.  The profile remains `4 R0 + 4 R1 + 1 colour = 9` reads.  ALU does not
increase; two multiplications use an existing scalar with the correct units.

### 21.4 Claims deliberately not made

This correction restores a necessary identity and must be tested before judging
the representation.  It is expected to affect defects 1--5, 8, 12--13, and 15,
because all can be amplified by querying the wrong periodic copy.  It may reduce
part of defect 6, but it cannot prove that defects 6--11 disappear.

Two independent limitations remain in the current data contract:

1. `initialCovered && correctedCovered` is an intersection of sampled hit sets.
   A live hit born between sampled directions has no valid initial point record;
   changing the predicate cannot reconstruct the missing owner/support event.
   This can leave incomplete geometry, early plume loss, and broken edges.
2. Winner colour/coverage is sampled from level zero rather than from a
   premultiplied projected-footprint carrier.  A sub-pixel field therefore
   aliases during camera translation and can retain distant moving grain.

Neither limitation authorises an extra runtime candidate, a relaxed miss depth,
cross-owner blending, more geometry taps, or another spatial filter in candidate
A.

### 21.5 Gate order and stop conditions

1. Independent red-team review checks the derivation, units, bake/runtime
   coordinate agreement, and whether any existing transform already supplies
   the supposedly missing `1/P` factor.
2. If accepted, implement only the two-factor correction.  Do not combine it
   with coverage, filtering, normal, density, or policy changes.
3. Run typecheck, focused stable-contract tests, and exact real-WebGPU boots for
   standing flat, standing uphill, boundary, very-close top-down, and the `10 m`
   control.  No URL is exposed until the exact boots are clean.
4. The user judges the dynamic scene, with emphasis on defects 1--5 and 12--13.
5. If the principal stretch/swim is not materially improved, candidate A is
   rejected despite being dimensionally correct; no adjacent shader constants
   are tuned.  The next work is the separately reviewed record-completeness and
   footprint-radiance representations.
6. If candidate A is accepted, freeze the corrected geometry identity.  Address
   missing geometry/edges and distant shimmer as two separate bake/data changes,
   each required to reuse the existing fixed geometry and colour read budgets.

### 21.6 Independent review verdict and implementation authorization

The independent review approved candidate A exactly as scoped.  It reproduced
the address identity from `guideToAuthored`, confirmed that no downstream
conversion cancels the extra pitch, reproduced the three numerical error
magnitudes, and confirmed that `profileDerivativeScale` is the correct in-scope
quantity for both horizontal corrector components.  Runtime now transcribes
that reviewed change and no other behavioural change.

The visual gate must distinguish a corrected chart from the independent event
closure defect.  Below the `15 degree` baked elevation floor, the old anisotropy
could accidentally shrink the epipolar displacement toward the initial seed.
Restoring the true displacement may therefore expose more
`initialCovered && correctedCovered` erosion at grazing angles even while
stretch, wrong apparent angle, and swim improve.  That is not evidence against
the unit correction; it is evidence for the already-isolated Section-16A data
blocker.  Candidate A is judged principally on defects 3--5.  Uphill folding may
improve less, while categorical row/azimuth boundaries mean defects 12--13 may
soften without disappearing.

The review also identified four items explicitly held out of this patch:

1. per-node election does not currently reject a corrected event behind its
   entry before choosing the winner;
2. the standalone `qInRange` test consumes an already-clamped tip and is
   therefore vacuous;
3. angular bracketing/election weights are chosen from the predictor direction
   rather than the anchor-corrected direction required by Section 18.3;
4. the superseded Section-16 line table contains historical claims, including
   pre-election non-negative depth, which no longer describe active code.

These are audit findings for later mathematical review, not permission to fold
more edits into candidate A.

### 21.7 Live visual verdict

The user tested candidate A dynamically after a clean real-WebGPU boot:

> no regression. checked. nothing improved much

Candidate A is therefore **visually neutral**.  Keep the two-factor correction
because it restores the exact predictor/corrector differential identity, but do
not credit it with solving any item in Section 20 and do not continue tuning
adjacent chart constants.  It is a prerequisite correctness repair, not the
visual remedy.

Three follow-up mathematical audits also close tempting but incorrect local
paths:

1. A division-free forward/envelope predicate is exact in the local affine
   corrector chart, but not after the active curved-terrain drape.  Before
   election the sampler does not know each candidate's final `G(U)` and
   `grad G(U)`.  Adding the local predicate as an exact rule could create more
   uphill slicing.  The clamped `qInRange` check is dead but only cleanup.
2. A stable plant-root affine transform is exact only **after** the true motion
   owner is known.  Different roots can reorder firstness under terrain, while
   the flat first-hit record does not reveal the newly first hidden owner.  It
   cannot be the selector or general terrain carrier.
3. A deterministic mid-height terrain tangent is exact for flat/affine terrain
   and is a better bounded curved-terrain linearisation.  It can target uphill
   slicing later, but it is identical to the current construction on the flat
   field where the circular zones, near-to-mid quality collapse, shimmer, and
   missing events remain.  It is not the next general remedy.

## 22. Measured lattice fingerprints after candidate A

The accepted v4 asset contains exactly four regular elevation rows,

```text
15, 35, 55, 75 degrees,
```

with sixteen azimuths per row and no azimuth-independent vertical record.  The
runtime chooses one complete node categorically by maximum bilinear weight.
Ignoring coverage ties, elevation-row ownership therefore changes at

```text
25, 45, 65 degrees.
```

For a camera `h` metres above flat terrain, an elevation boundary `beta`
intersects the field at

```text
r(beta) = h / tan(beta).
```

At `h = 1.7 m` this gives

```text
beta       65 deg     45 deg     25 deg
r(beta)    0.79 m     1.70 m     3.65 m.
```

This quantitatively predicts the user's three perfect camera-centred circles
inside roughly five metres and places the strongest transition at the reported
roughly two-metre quality boundary.  It is not a distance LOD: it is the radial
image of a categorical angular lattice under a standing camera.

The user's later observation that these circles retain the same screen size as
the camera rises is a direct confirmation.  Since `r(beta)=h/tan(beta)`, their
world radius grows linearly with camera height while their angular radius in
the camera remains constant.  A world-distance LOD would have the opposite
signature.

The lowest `15 degree` row begins only around
`1.7/tan(15 deg) = 6.34 m`.  A new `5 degree` row is relevant to genuinely
shallow views beyond that radius, but cannot remove the measured circles or the
two-metre transition.  More regular rows merely create more, narrower
categorical zones unless the election topology changes.

At the other pole, live elevation above `75 degrees` clamps to the tilted
`75 degree` row while azimuth is still computed from a horizontal vector whose
magnitude tends to zero.  Sixteen tilted views therefore meet as radial wedges
under the camera even though vertical view has no meaningful azimuth.  This is
an exact data-domain explanation for the very-close top-down triangular
sectors.  The minimal missing datum is one azimuth-independent `90 degree`
record, not sixteen redundant vertical records.

These findings divide the next work into independent mathematical causes:

- circular zones and the two-metre transition: categorical angular election;
- top-down radial sectors: missing vertical singleton;
- dense distant moving grain: level-zero projected-footprint aliasing;
- incomplete plumes and broken edges: missing first-hit event closure;
- uphill-specific slicing: curved-terrain chart linearisation.

They must not be blurred into one shader tweak.  Candidate B will address only
the first two lattice-topology causes, will reuse the existing four node
records, and will be independently reviewed before runtime code changes.

## 23. Review candidate B: close the vertical pole, reject unkeyed PCF

This section is a proposal for independent review.  It is not implemented.

### 23.1 Why categorical PCF is not candidate B

Let a live direction have angular weights `w=(w_0,...,w_3)`.  A PCF-like
categorical interpolation would draw one complete record with probabilities
proportional to `w`.  To be temporally stable, its inverse-CDF scalar `xi` must
be attached to the same physical surface event as the camera moves.  The
current record contains only depth, octahedral normal, and hit coverage.  It
contains no event, owner, chart, or cross-view correspondence key.

The top-plane address is not such a key.  For a fixed surface point `X` below
the profile top `H`, its top-plane intercept under direction `d` is

```text
p_X(d) = X.xz - ((X.y-H)/d.y) d.xz.
```

It changes with direction.  Hashing the top-plane address, ground intercept,
screen pixel, or any other live-ray plane intercept therefore slides the
selector over a stationary plant.  After the four corrected reads there are
four independent first-hit events.  Hashing their corrected addresses or owner
points supplies four different keys and can refresh whenever nearest texels or
visible owners change.  There is no common, direction-independent event in the
present data from which to derive `xi`.

This is not repaired by calling the noise world-space.  The world-space point
needed for the key is the unknown true hit which election is trying to choose.
A stable owner seed per record would make candidate-specific random priorities
possible; a shared inverse-CDF seed requires a stronger transition-cell or
cross-node correspondence identifier.  Runtime currently receives neither.

Even with such metadata, stochastic categorical interpolation has a necessary
change rate.  If the normalized categorical distribution changes from `w` to
`w'`, every coupling changes at least

```text
TV(w,w') = 0.5 sum_i |w_i-w'_i|
```

of selectors.  It trades one coherent boundary for spatial geometry changes.
Winner-radiance mips can filter colour inside a slice but cannot filter the
selected depth, normal, or silhouette.  Unkeyed PCF would therefore be a direct
way to turn the three circles into more of defect 6's moving grain.  It is
rejected, not implemented.

### 23.2 Exact vertical-cap domain

Keep the existing regular `16 x 4` lattice as a strict prefix and append one
slice

```text
c_P = (0,-1,0).
```

GCRP/v4 already stores an arbitrary slice count, per-slice directions, and
atlas dimensions, so the container version and header do not need to change.
The lattice metadata gains one optional `poleSliceIndex`.  Parsing accepts only

```text
[strict validated regular azimuth/elevation prefix] + [one final vertical pole]
```

and rejects every other irregular layout.  Sixty-five stored slices fit an
exact `13 x 5` atlas, so the pole costs one tile rather than the eight blank
tiles of a `9 x 8` atlas.

Use Cartesian slope coordinates

```text
s(d) = d.xz / (-d.y).
```

On the cap between the `75 degree` ring and the pole, let

```text
r = |s(d)|,
R = cot(75 degrees),
t = clamp(r/R, 0, 1),
a = fractional azimuth position between az0 and az1.
```

The cap triangle weights are

```text
b_0 = t (1-a),
b_1 = t a,
b_P = 1-t,
b_D = 0.
```

Its nodes are `az0@75`, `az1@75`, the pole, and one forced-ineligible dummy.
At `75 degrees`, `t=1`, so the weights exactly equal the existing two-node ring
interpolation.  At `90 degrees`, `t=0`, so the pole weight is exactly one and
azimuth is irrelevant.  The central singularity and its sixteen radial wedges
therefore disappear by construction.

At finite precision, force azimuth index zero whenever `r` is below one small
fixed representational threshold.  This does not change the weights or result
when the pole record is covered; it prevents an implementation-defined
`atan(0,0)` address from allowing a zero-weight ring node to win if the pole
record is a MISS.

The address/election structure must carry explicit slice indices and per-slice
depth bounds on this cap.  The pole canonical direction is exactly `c_P`.
The four-component regular-row depth bounds do not implicitly cover the pole:
the cap path must select the pole's own fifth `depthMin/depthMax` pair through a
fixed scalar select.
The dummy may address the pole to keep four syntactic loads, but its eligibility
and weight remain exactly zero.  Thus the dynamic path remains

```text
4 initial complete records
+ 4 corrected complete records
+ 1 winner colour record
= 9 profile operations.
```

No binding, pass, barrier, dispatch, loop, march, candidate expansion,
per-species work, or runtime geometry is added.  Fixed ALU adds one slope ratio
and a few multiplies/selects; the expected extra live state is roughly one or
two scalars.  The meaningful asset/runtime cost of one `258 x 258` pole slice is
approximately

```text
0.508 MiB GPU geometry at 8 bytes/texel,
0.254 MiB current GPU colour at 4 bytes/texel,
0.762 MiB GCRP geometry+owner payload at 12 bytes/texel,
```

with the existing mesh tables unchanged.  Under the later full RGBA8 mip
carrier, the pole's colour chain is approximately `0.333 MiB`.

### 23.3 Exact scope and gate

Candidate B directly fixes defect 13's undefined top-down radial/triangular
sector structure.  It may improve only the innermost part of defects 12 and 14.
It does **not** claim to fix the `25/45/65 degree` standing rings, slope folding,
view-direction stretch, missing events, broken edges/plumes, or distance
shimmer.  Maximum categorical election leaves a small ordinary pole/ring
transition near the top cap; it no longer lets sixteen tilted views meet at an
undefined centre.

The pole wins when `1-t > t max(a,1-a)`, so this ordinary transition lies at
approximately `79.9--82.4 degrees` depending on azimuth.  At a `1.7 m` eye it
is a small scalloped ring of roughly `0.23--0.30 m` ground radius, not another
field-scale transition.

The independent review must check the cap barycentrics, continuity at
`75 degrees`, exact azimuth independence at the pole, GCRP prefix parsing,
slice/depth indexing, and byte/read accounting.  If accepted, candidate B is
implemented alone and judged in very-close and ordinary top-down flight.  A
neutral result outside that declared scope is not evidence against it; any
remaining central wedge or new ring/grid artifact rejects it.

The next independent radiance candidate remains Section 15's one-read
premultiplied per-slice mip carrier.  The next geometry-transition candidate
requires an explicit cook-stable cross-view event/chart identity; it cannot be
invented by hashing the current point record.

### 23.4 Implemented pole-cap checkpoint (2026-07-24)

Candidate B is now transcribed in the isolated `grassprofile=2` lane.  The
regular `16 x 4 = 64` direction prefix remains ordered as before; one exact
vertical direction `(0,-1,0)` is appended at slice `64`, and only this profile
moves from an `8 x 8` to a `13 x 5` atlas.  The installed deterministic GCRP/v4
hash is
`59cebe81bbf7b7a70a1ed2aeb99d9d20f8405dff40cdb5a8936679cb2f6c6c1e`.
Its parsed contract is `65` slices, pole index `64`, stored tiles `258 x 258`,
and pole depth interval `[0,1.176407814] m`.

The runtime cap uses the Section-23.2 slope barycentrics.  Azimuth is forced to
zero as horizontal ray length tends to zero, the duplicate fourth cap node has
zero weight **and is explicitly ineligible**, and the pole carries its own
fifth depth interval.  The regular multi-profile lattice advertises no pole and
retains its existing path.

This changes no profile-read count, binding, pass, dispatch, barrier, loop,
march, candidate count, or runtime geometry.  It adds one fixed slope ratio,
selects, and approximately one to two live scalars in the isolated cap path;
the source/GPU atlas cost is the single slice accounted in Section 23.2.

Typecheck and the ten focused profile/surrogate contracts pass.  The exact
Estonia `grassprofile=2` URL completed ninety settled real-WebGPU frames with
no page, TSL, shader, bind-group, command-buffer, or uncaptured WebGPU error.
Visual acceptance remains the user's flight gate; the implementation is not
claimed to fix the separately measured `25/45/65 degree` standing rings.

## 24. Review candidate C: exact chart-consensus election

This is the first non-stochastic candidate which can remove the measured
standing rings rather than merely redistribute them.  It is conditional on one
cheap cook-side measurement and is not implemented.

### 24.1 Correspondence theorem

Suppose every eligible angular node `j` carries an exact key

```text
K_j = (affine surface-chart id, global periodic copy X, global periodic copy Z).
```

For every distinct key represented among the four already-read nodes, define

```text
S(K) = sum_{j eligible and K_j=K} b_j.
```

Elect the key with greatest `S(K)`, then elect the greatest-`b_j` complete
record inside that key's group.  No depth, normal, owner, or point is averaged.

If adjacent elevation nodes see the same chart `K`, their two-row contribution
is

```text
S(K) = (1-t) + t = 1.
```

The arbitrary maximum-weight switch at `t=1/2` therefore disappears.  If the
chart is one affine plane with one geometric face normal, every representative
inside that group defines the same live ray/plane intersection on affine
terrain.  Under the active curved drape, representatives evaluate `G_H` and its
gradient at slightly different authored points; the disagreement is the
Section-18.5 second-order remainder in their centimetre-scale spread, rather
than the current field-scale first-order row switch.
Unlike PCF, this election is deterministic and contains no spatial random
selector, so it does not introduce stochastic geometry grain.

Consensus may deliberately elect a farther surface when two lower individual
weights agree on one exact chart against a single larger-weight record from a
different chart.  That is the declared plurality semantics, not an accidental
depth blend.  Equal group sums and equal representative weights use the lowest
fixed node index, matching the existing deterministic tie convention.

This grouping must govern **both** the initial provisional election and the
corrected final election.  Grouping only corrected records leaves the
provisional corrector frame switching coherently at the original row midpoint.

### 24.2 What is and is not an exact chart identity

- A connected-component ID is too coarse.  One connected grass plant or
  rhizome may contain many unrelated leaves and depth sheets.
- A procedural-primitive ID is sufficient only when the cooker certifies that
  the primitive is one affine visibility chart.
- The source triangle ID is the smallest generally sound chart.  Coplanar
  triangles may be cook-merged only with an explicit plane certificate.
- Periodic copy is mandatory.  Different repeats of the same triangle must not
  reinforce one another.  The global copy is

```text
floor((record address - tile origin)/tile size) + stored relative copy.
```

  Comparing raw triangle IDs or raw relative-copy tokens is incorrect.
- A ten-bit hash is not exact.  Four uniform keys have roughly
  `6/1024 = 0.59%` pair-collision probability per stage, and ten bits cannot
  retain exact copy identity.  Such collisions would be visible false
  reinforcement, so the alpha-mantissa hash idea is rejected for production.

### 24.3 Fixed-width carrier and cost

GCRP/v4 already contains one exact 32-bit owner token per texel:

```text
22-bit triangle id + signed 5-bit copy X + signed 5-bit copy Z.
```

The runtime currently discards it after load-time colour transcoding.  It can
instead be repacked with the coupled nearest geometry into one 64-bit integer
texel:

```text
owner token       32 bits
normalised depth  12 bits
oct face normal   10 + 10 bits
total             64 bits.
```

`owner=0xffffffff` remains MISS.  The 12-bit depth step is per-slice span
divided by `4096`: approximately `0.29 mm` at the pole and `1.11 mm` on the
long `15 degree` row.  The latter is still better than the current float16
conversion's roughly `2.2 mm` spacing near normalised depth one; the resulting
`|B| Delta tau` address error stays at or below roughly `2 mm`, one `2.03 mm`
profile texel.  Ten-bit oct components match the current half-float carrier's
practical precision.  The transcode is load-time use of data already in the
file, so disk format and asset bytes do not increase.

The texture changes from one 64-bit nearest float record to one unfilterable
`RG32Uint` record.  Taps therefore become integer texel loads, which is a real
sampler-code rewrite but is semantically identical to the current NEAREST
reads.  Runtime traffic remains

```text
4 R0 + 4 R1 + 1 colour = 9 reads,
8 bytes per geometry texel.
```

There is no new texture, binding, pass, barrier, dispatch, loop, march, or
candidate expansion.  The price is fixed integer work: pairwise key equality,
four group sums, and two categorical max reductions at each stage.  The
estimated additional live state is eight to twelve scalar values per stage.
That register/occupancy consequence is real and must be checked in the next
coherent GPU trace; it is not assumed free merely because the read count is
unchanged.

The geometry normal packed here must be the affine chart's geometric face
normal, not an interpolated shading normal, because the theorem requires one
plane per exact chart.  Shading implications must be reviewed explicitly; a
separate shading normal read is not authorised.

### 24.4 Required measurement and bounded claim

Before runtime implementation, use the existing v4 owner atlas to measure the
fraction of visible `25/45/65 degree` boundary samples whose adjacent nodes
share the exact `(triangle, global copy)` key.  This is not another surrogate
search.  It answers the one load-bearing question in the theorem: how much of
the observed rings is arbitrary switching between two records of the same
surface chart?

The records must be paired by epipolar reprojection, never equal texel address.
For a covered row-A event at exact point `X_A`, compute the row-B intercept

```text
p_X(c_B) = X_A.xz - ((X_A.y-H)/c_B.y) c_B.xz,
```

fetch B there, normalise both stored tokens to global copies, and compare.
Equal-address pairing is displaced by tens of centimetres between rows versus
roughly `2 mm` texels and would falsely report almost no correspondence.

Before inspecting the result, the GREEN threshold is fixed as follows:

```text
45-degree boundary: >= 60% exact-key agreement among bidirectionally covered
                     epipolar pairs;
25/45/65 aggregate: >= 50% exact-key agreement;
each outer boundary: >= 35% exact-key agreement;
forward/reverse disagreement: <= 10 percentage points.
```

The `45 degree` boundary is primary because it projects to the reported
roughly two-metre ring.  Falling below any threshold means exact triangle
consensus cannot remove enough of the coherent artifact to justify its shader
and register cost; thresholds are not revised after seeing the measurements.

If shared exact-chart support is substantial, implement the integer transcode
and consensus election exactly as above.  If it is rare, park candidate C
without a shader experiment: it cannot materially fix the rings, and the next
representation must carry a certified coarser affine chart ID or the total
event relay of Section 16A.

Candidate C can remove categorical row/azimuth switches only where
correspondence truly exists.  It cannot create a hit missed by every node,
recover a hidden successor, merge different occluders, make a curved
multi-triangle surface one plane, close cover boundaries, add a side chart, or
filter distant radiance.  Those remain distinct defects rather than excuses to
weaken key equality.

### 24.5 Measured result: RED, candidate parked

The exact epipolar measurement was run against accepted GCRP/v4 source
`2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
It covered both directions, all sixteen azimuths, all three elevation
boundaries, both elevation diagonals, and adjacent-azimuth pairs.  Runtime-like
nearest addressing landed within `1.224 mm` p95 and `1.436 mm` maximum of the
exact reprojected target.  Exact source-plane versus stored-depth error was
`0.032 mm` p95.  Same-triangle/wrong-global-copy matches were zero, confirming
that the copy normalisation is operating as intended.

The frozen-gate aligned-row results are:

```text
boundary   conditional exact agreement   shared-key mass   gate
25 deg             30.48%                    29.26%         RED (<35%)
45 deg             39.90%                    36.98%         RED (<60%)
65 deg             42.39%                    38.43%         GREEN (>=35%)
aggregate          35.69%                    33.51%         RED (<50%)
```

Forward/reverse gaps remained within the ten-point guard, but that does not
rescue the failed support thresholds.  Even assigning every roughly `1.07%`
invalid source record a favourable exact match raises the critical `45 deg`
result only to `40.54%` and the aggregate only to `36.42%`.  The result is
therefore robustly RED, not a raster-edge ambiguity.

Candidate C is parked for this carrier.  No integer repack or shader consensus
path is authorised: the exact chart identity is too sparse to remove enough of
the visible standing-height ring, and its fixed ALU/register cost would not buy
the promised result.  Reopen only if a future cook supplies a certified,
coarser affine chart identity whose *measured epipolar* agreement clears the
same thresholds.  The independent vertical-pole candidate B is unaffected.

Measurement artifacts:

```text
data/work/groundcover-chart-correspondence/
  2ed57f59d86e8376/63d23cd20770d8dc/SUMMARY.md
  2ed57f59d86e8376/63d23cd20770d8dc/report.json
```

## 24A. Independent frozen-budget minification result: exact limit and a same-grid scale codec

This section addresses the dominant `2--5 m` fuzz, grain, blur, and moving-particle
artifact under a stricter budget than Section 15:

```text
exactly the existing 4 R0 + 4 R1 + 1 radiance operations;
no extra physical texel contributions hidden inside filtering;
no increase in disk, transient CPU, persistent CPU, or GPU bytes;
no loop, march, candidate list, pass, barrier, runtime geometry, or extra owner;
no discrete camera-distance or camera-angle LOD boundary.
```

The result has two parts.  First, there is a short impossibility boundary for
arbitrary triangle soups under the present opaque one-hit resolve.  Second,
there is an implementable same-grid scale carrier which can remove the
*radiance* part of the minification artifact with fewer bytes and less memory
traffic.  It deliberately does not claim to solve sub-pixel binary silhouettes
or categorical normal-lighting sparkle; those are on the proved side of the
boundary.

### 24A.1 The measured-looking `2--5 m` threshold follows from the bake exactly

The interior tile pitch is

```text
Delta = 0.52 m / 256 = 0.00203125 m = 2.03125 mm.
```

At `1440` vertical pixels and a `55 degree` vertical field of view, the focal
length in pixels is

```text
f = 1440 / (2 tan(55 degrees / 2)) = 1383.107 px.
```

A fronto-parallel bake texel at range `r` therefore projects to

```text
p_texel(r) = f Delta / r = 2.80944 / r  pixels,
```

and one pixel covers

```text
w_pixel(r) = r/f = 0.00072301 r metres.
```

Consequently:

```text
range       texel projection       pixel footprint
2 m             1.405 px              1.446 mm
2.809 m         1.000 px              2.031 mm
5 m             0.562 px              3.615 mm
```

The level-zero signal necessarily becomes minified at `2.81 m` even in the
best-oriented case.  The actual chart footprint is larger.  With `J` denoting
the Section-15.3 screen-to-baked-address Jacobian, its base-texel footprint is

```text
rho = sigma_max(diag(256/0.52, 256/0.52) J).
```

The ray/plane factor in `J` contains `1/(n dot d)`, and the baked top-plane
shear contains `c.xz/c.y`.  A shallow live incidence, a shallow baked row, or
both therefore push `rho=1` closer than `2.81 m`.  This predicts precisely a
soft `2--5 m` failure interval rather than one world-distance cutoff.

The current level-zero bilinear colour lookup does not integrate this growing
footprint.  Its fixed triangular reconstruction kernel remains about one bake
texel wide while the pixel footprint grows through several texels.  Camera
translation moves the query across that under-filtered `2.03 mm` grid, so
unresolved energy aliases as a directional stream of tiny particles.  Nearest
categorical coverage and normals alias at the same threshold.  What looks like
distance blur is therefore a mixture of unresolved high frequencies, changing
binary ownership, and changing face-normal lighting; it is not evidence that
the source suddenly lost geometric detail at two metres.

### 24A.2 What nine *physical* reads can and cannot reconstruct

Three limits are unconditional for arbitrary input radiance and geometry.

**Unknown-background theorem.**  The correct filtered pixel at a grass
silhouette is

```text
C = A C_grass + (1-A) B,
```

where `B` is the visibility sample behind the grass.  Consider two scenes with
identical grass records `(A,C_grass)` and different backgrounds `B1 != B2`.
Any current one-hit opaque resolve receives the same grass record in both
scenes, so it must emit the same value, while the correct values differ by
`(1-A)(B1-B2)`.  Thus no grass-only codec, regardless of packing or bake
quality, can exactly antialias fractional silhouette coverage without retaining
the background visibility.  A deterministic binary threshold aliases; a
stochastic threshold turns the error into moving particles.

**Categorical-cell theorem.**  A coarse footprint may contain two unrelated
first-hit owners with different depths and normals.  One complete categorical
record can return at most one of them.  Averaging the records fabricates a
surface; choosing one discards the other.  Exact arbitrary-soup depth, owner,
and normal minification therefore needs either more than one visibility layer
or a changed output semantics.  Neither exists in this frozen contract.

**Scale-coordinate theorem.**  An arbitrary filtered 2D signal is a function
of `(q_x,q_y,rho)`.  One ordinary 2D sample has only two address coordinates.
Without a low-dimensional signal model, it cannot represent an arbitrary
continuous scale coordinate as well as both spatial coordinates.  Integer mip
selection drops the scale dimension but creates coherent boundaries at
`rho=2^k`; trilinear mip filtering restores it by reading two levels.  Calling
that one source-level read does not change the physical work: one bilinear
level uses four texel contributions and a trilinear sample uses eight.

These theorems reject four tempting fixes under the present freeze:

1. no mip chain may be called free merely because the shader contains one
   texture-sample instruction;
2. integer mip selection would replace the current angular circles with new
   screen-size minification circles;
3. categorical depth/normal/owner mips cannot be numeric averages;
4. coverage dithering cannot be the cure for a moving-particle complaint.

There is therefore no complete, exact, temporally continuous, arbitrary-soup
solution for all three of radiance, silhouettes, and categorical lighting under
the one-hit opaque output plus nine-physical-read freeze.  This is a
representation boundary, not permission to add taps or bytes.

### 24A.3 Candidate S0: same-grid continuous scale stack

The strongest admissible replacement targets the part which *is* solvable:
band-limited conditional grass radiance in continuously covered pixels.  It
also removes the deployed v4 source mesh and owner atlas, which exist only so
the loader can reconstruct colour and are not sampled by the runtime.

Its footprint `rho` is valid only after Section 26 restores the complete
record's sampled texel-centre point and geometric face normal.  Feeding the
current interpolated shading normal to the `1/(n dot d)` plane differential can
mis-estimate the footprint without bound; S0 must not be implemented ahead of
that correction or used to conceal its stretch/swim defect.

Use two periodic, unfilterable integer textures.  Wrap the interior address in
integer arithmetic before each `textureLoad`; because no spatial hardware
filtering occurs, the stored `258 x 258` gutters disappear.

The first texture is one `R32Uint` complete categorical geometry record:

```text
bits  0..15   depth code, 65535 = MISS
bits 16..23   octahedral geometric-normal X, 8 bits
bits 24..31   octahedral geometric-normal Y, 8 bits
```

Depth uses codes `0..65534` over the slice's existing `[depthMin,depthMax]`.
On the longest approximately `4.55 m` `15 degree` slice its step is about
`0.069 mm`; `cot(15 degrees)` epipolar amplification makes that about
`0.26 mm`.  Oct8's conservative angular error is about one degree and must be
included in the visual normal gate.  MISS remains categorical.  No owner,
normal, depth, or coverage value is interpolated.

The second texture is one `RG32Uint` scale record at every *base-grid* texel.
Its 64 bits preserve the current near colour exactly at eight bits/channel and
carry two filtered colours:

```text
word 0 bits  0..23   Q0(q), RGB888 winner-event authored colour
word 0 bits 24..31   A2(q), 8-bit broad-filter coverage/confidence
word 1 bits  0..15   Q1(q), RGB565 convolution at scale sigma1
word 1 bits 16..31   Q2(q), RGB565 convolution at scale sigma2
```

For `ell>0`, cook-side fields are

```text
A_ell(q) = K_sigma_ell * alpha
P_ell(q) = K_sigma_ell * (alpha L)
Q_ell(q) = P_ell(q) / max(A_ell(q), epsilon),
```

using a positive normalized periodic kernel.  They are evaluated and stored at
all `256 x 256` base-grid locations rather than downsampled into spatial mip
cells.  Hence every scale has the same world-anchored address and there is no
cell-size or atlas-offset switch.  `Q0` preserves the current RGBA8 carrier's
RGB precision exactly; alpha is redundant at a categorical HIT because the
opaque resolve does not composite it.  `A2` is confidence only: it may prevent
an unstable low-coverage conditional average from replacing `Q0`, but it is
never treated as opacity.  A safe first scale schedule for the measured failure
range is `sigma={0,2,16}` base texels; the final two values must be fitted
cook-side against the exact Section-15.3 footprint response, not tuned in the
shader.

Let `lambda=log2(max(rho,1))`.  Decode all four words from the one record and
blend only in ALU with a piecewise cubic partition over
`log2(sigma_ell)`.  Each adjacent transition uses `smoothstep`, and its endpoint
equals the next interval's start exactly.  Therefore colour is C0 continuous
in `rho`; there is no integer level election and no camera-centred LOD ring.
At `rho<=1`, `Q0` is returned exactly.  Decode the two `RGB565` filtered values
before blending; bit fields themselves are never numerically filtered.  As
`rho` grows, the spatial variation
of the selected field is bounded by its convolution scale instead of the raw
two-millimetre signal.  Nearest base-grid loading is then deliberate: the
screen-resolved regime uses `Q0`, while the minified regime reads fields whose
adjacent-grid difference has already been bounded cook-side.

This uses exactly

```text
4 R0 R32Uint textureLoad
+ 4 R1 R32Uint textureLoad
+ 1 RG32Uint scale textureLoad
= 9 physical texel loads.
```

It performs no hidden bilinear or trilinear contributions.  Per queried pixel,
uncached payload traffic falls from the current `8*8 + 4*4 = 80` bytes
(`8` RGBA16F nearest records plus the four contributions of one bilinear RGBA8
sample) to `8*4 + 1*8 = 40` bytes.  The fixed integer unpack and two colour
lerps add no divergence, synchronization, or register-resident candidate list.

Every hit record retains one cook-certified source-owner provenance even though
the runtime, as today, does not consume a numeric owner ID; no value in `Q1` or
`Q2` participates in geometry election.  The categorical geometry record still
governs screen ownership.  The scale record supplies the conditional grass
colour only; it does not misuse its
implicit coverage as opacity and does not pretend to know the background.
Consequently candidate S0 can remove chromatic minification grain and the
radiance part of the fuzzy veil.  It cannot guarantee removal of particles
whose source is binary silhouette turnover or categorical face-normal
lighting.  Geometry normal remains exact for reconstruction and shading; a
filtered shading-normal moment would be a separate approximation and is not
smuggled into this carrier.

### 24A.4 Exact byte accounting

For the current `65` slices, the stored base-grid population is

```text
T = 65 * 256^2 = 4,259,840 texels.
```

Candidate S0 stores `4 + 8 = 12` bytes per texel:

```text
atlas payload = 51,118,080 bytes = 48.750 MiB.
metadata      = 128 + 65*64 = 4,288 bytes before ordinary alignment
```

The current two GPU atlases occupy

```text
65 * 258^2 * (8 geometry + 4 colour)
= 51,919,920 bytes = 49.515 MiB.
```

Thus GPU bytes decrease by `801,840` bytes; they do not merely remain equal.
The larger saving is deployment and CPU memory.  GCRP/v4 currently ships the
8-byte geometry atlas, a 4-byte owner atlas, and approximately `67.5 MB` of
quantised source vertices/triangles so load-time code can regenerate colour.
Candidate S0 bakes its scale record directly and ships none of the owner or mesh
tables.  The installed artifact falls from `119,462,112` bytes to approximately
`51.12 MB`, a reduction of about `57%`.  The research/source mesh may remain as
an offline provenance artifact; it is not part of the deployed profile.

A direct little-endian integer upload requires no half-float or load-time colour
transcode.  One fetch buffer plus one detached upload copy therefore peaks near
`102.3 MB` and can be lower with ordinary section streaming.  The current path
simultaneously materialises the approximately `119.5 MB` fetch, geometry,
owner, mesh tables, half-float geometry, and generated colour.  Persistent CPU
texture backing becomes the same `51.12 MB` payload rather than those decoded
and regenerated intermediates.

### 24A.5 Authorization boundary and decisive gate

Candidate S0 is a proved implementable same-or-lower-byte representation, not a
proof that every reported particle is radiance aliasing.  Before transcription,
one existing debug build must classify the moving-particle energy without
changing read count or data:

```text
authored colour held constant + categorical normal held constant
```

If the particle stream substantially remains, it is binary ownership and the
unknown-background/categorical-cell theorems apply; candidate S0 must not be
sold as its fix.  If it collapses when colour is held constant but remains when
only normal is held constant, candidate S0 targets the dominant source.  If it
collapses only when the normal is held constant, the next admissible question
is a shading-normal moment packed by sacrificing one `RGB565` scale word; the
geometric plane normal stays categorical.  No result authorises a tenth read,
a mip byte increase, stochastic coverage, or a background-colour fabrication.

## 25. Rejected candidate D: same-byte projected-footprint radiance

**Post-audit verdict under the user's strict tap freeze:** the footprint
diagnosis and byte-repacking result below are retained, but this candidate is
not admissible for implementation.  A continuous trilinear mip operation
touches two bilinear mip levels instead of the current one bilinear base level:
eight underlying colour texels rather than four.  Calling both cases "one
explicit texture operation" does not satisfy **no extra taps, period**.  An
integer mip level keeps the physical tap count but creates the same fixed-angle
screen-relative transition circles already rejected.  No runtime change is
authorised from this section unless a later codec supplies a continuous
footprint integral with no more physical texel accesses than the current path.

This candidate addresses the dominant whole-field fuzz, grain, and coherent
moving-particle stream before any more local geometry work.  It supersedes
Section 15's memory accounting, but not its categorical-geometry rule.  The
runtime remains exactly

```text
4 level-zero initial geometry records
+ 4 level-zero corrected geometry records
+ 1 winner radiance record
= 9 explicit profile texture operations.
```

There is no new runtime texture operation, binding, pass, barrier, dispatch,
loop, march, candidate, species term, or geometry.  The complete radiance mip
chain is paid for by a smaller equivalent encoding of the existing categorical
geometry, so total resident bytes decrease rather than increase.

### 25.1 The current sampling law and the two different kinds of fuzz

The current isolated path has the following exact spatial filters:

- coupled depth/coverage/normal is `RGBA16F`, nearest, level zero;
- winner authored colour/coverage is `RGBA8`, bilinear, level zero;
- neither texture has mip levels;
- the winner colour address is the corrected address of the categorical
  geometry winner; no cross-direction colour blend remains.

Thus the colour operation integrates at most one `2 x 2` level-zero tent even
when a screen pixel covers tens of source texels.  The geometry operation is a
single categorical point sample.  The bilinear colour lookup can average
adjacent, unrelated fine first-hit colours and produce a local veil, while its
fixed two-texel support is simultaneously much too narrow under minification.
Those facts are not contradictory: the current carrier can be locally soft and
temporally aliased at the same time.  Temporal accumulation of changing,
under-resolved point samples is perceived as a fuzzy grain cloud rather than
as useful crisp detail.

The forced profile has zero authored wind, one source layer, no root hash
rejection, and a world-stationary periodic address.  The camera's mirrored TAA
jitter and ordinary translation move the screen footprint over this static
field.  They expose the alias; they are not a time-dependent grass deformation.

### 25.2 Exact screen-to-baked-address footprint for the active drape

Let `s=(s_x,s_y)` be pixel coordinates, `M^-1` the current *jittered* inverse
view-projection, and `C` the camera position.  At the far unprojection plane,

```text
h(s) = M^-1 (2(s_x+1/2)/W - 1,
             2(s_y+1/2)/H - 1, 1, 1)
Y = h.xyz/h.w
v = Y-C
d = v/|v|.
```

For `k in {x,y}`, with `h_k=(2/W)M^-1_col0` or
`h_k=(2/H)M^-1_col1`, respectively,

```text
Y_k = (h_k.xyz h.w - h.xyz h_k.w)/h.w^2
d_k = (I-dd^T)Y_k/|v|.
```

Let the categorical hit be `X=C+t d` with its categorical world **geometric
face** normal `n`.  Holding that one affine face fixed, the neighbouring
pixel's exact first differential is

```text
X_k = t [d_k - d (n dot d_k)/(n dot d)].
```

This does not blend faces.  It asks only how the already elected face projects
onto one screen pixel.

The current primary record contains an interpolated shading normal, which is
not a valid plane covector.  Candidate D's footprint therefore depends on the
complete-record correction recorded in the following section (sampled
texel-centre ray plus geometric face normal).  It must not silently feed the
current shading normal into this Jacobian; near a false-normal grazing angle
that would overestimate or underestimate the footprint without bound.

For the active isolated drape, authored residual height is

```text
U_y = X_y - G(X_xz),       U_xz = X_xz,
```

where `G` and its analytic gradient are the same tensor-Hermite terrain carrier
used by position and normal reconstruction.  Therefore

```text
(U_k)_y  = (X_k)_y - grad(G) dot (X_k)_xz,
(U_k)_xz = (X_k)_xz.
```

For the winner's baked direction `c`, its top-plane address is

```text
q(U) = U_xz - (c_xz/c_y)(U_y-H_profile),
```

so the exact local address differential is

```text
q_k = [I + (c_xz/c_y) grad(G)^T] (X_k)_xz
      - (c_xz/c_y) (X_k)_y.
```

This terrain-gradient term is required.  Dropping it would use a different
chart from the active position/normal drape and would reintroduce
slope-dependent footprint error.

Convert the two address columns to base texels using

```text
D = diag(N_x/S_x, N_y/S_y),
J = [D q_x, D q_y].
```

The major footprint radius is the largest singular value

```text
g00 = dot(J_x,J_x),  g01 = dot(J_x,J_y),  g11 = dot(J_y,J_y)
rho^2 = 1/2 [g00+g11 + sqrt((g00-g11)^2 + 4g01^2)].
```

The observed onset follows without a fitted constant.  At the image centre,
for vertical field of view `F_y` and effective render height `H_px`, the
unamplified angular pixel width is approximately

```text
delta = 2 tan(F_y/2)/H_px.
```

Here `S=0.52 m`, `N=256`, hence `N/S=492.31 texels/m`.  At `F_y=55 degrees`
and `H_px=1080`,

```text
rho_transverse ~= t delta N/S ~= 0.474 t,
rho=1 at t ~= 2.11 m,
rho~=2.37 at t=5 m.
```

This already predicts the user's `2--5 m` failure interval.  The factors
`1/|n dot d|` in the ray/face intersection and `c_xz/c_y` in the baked chart
make the footprint longer on oblique blade and plume faces.  Its major singular
vector commonly aligns with projected range/epipolar direction.  Translation
or sub-pixel jitter then sweeps fine phase coherently along that vector, which
is the reported one-direction stream of millions of particles.

### 25.3 One-operation band limit, including temporal and pole limits (tap RED)

For each direction slice, bake a periodic premultiplied radiance measure

```text
P^0(q) = (alpha(q) L(q), alpha(q)),       alpha in {0,1},
```

and a complete toroidal mip chain

```text
P^(ell+1)(x,y) = 1/4 sum_(a,b in {0,1})
  P^ell((2x+a) mod N_ell, (2y+b) mod N_ell).
```

Every direction is one independent array layer.  Atlas mips are forbidden:
they eventually mix different directions.  Stored gutters are unnecessary in
an independently repeating array layer.

The continuous level is

```text
lambda = clamp(log2(max(rho,1)), 0, ell_max).
```

One explicit winner-layer sample at `(q,lambda)` returns `(C_premul,A)` and the
authored colour is

```text
L_bar = C_premul/max(A,1/255).
```

Fractional mip interpolation is necessary to keep `lambda` continuous and
avoid replacing the present grain with new screen-relative mip circles.  It is
one explicit texture operation, as the current winner colour is; hardware may
touch the two adjacent mip levels internally.  That extra internal locality is
more than offset by the geometry-byte reduction in Section 25.4.  A stochastic
level selector is forbidden because it would recreate the moving particles.

The current jittered inverse projection participates in `J`, so jitter changes
the footprint and address coherently rather than selecting an unrelated fine
sample.  On periodic seams, toroidal mips make both the signal and its scale
space continuous.  At the vertical pole, `c_xz=0`, so the baked-address shear
vanishes and `q_k=(X_k)_xz`; there is no azimuth singularity.  As
`|n dot d| -> 0` at a true face silhouette, `rho -> infinity` and the lookup
converges to the coarsest mip.  This is the correct extended-real limit.  The
implementation may saturate directly to `ell_max`; it must not use an epsilon
to pretend the footprint stayed narrow.  Grazing live views have the same
bounded coarsest-level limit.

With one isotropic lookup, `rho=sigma_max(J)` is forced by the anti-aliasing
contract.  An area/geometric-mean level leaves the major direction
underfiltered and preserves the directional stream.  Conversely, an isotropic
kernel wide enough for `sigma_max` necessarily filters the minor direction too.
Avoiding that anisotropic loss requires either hardware anisotropic work or a
larger directional/ripmap representation; neither is hidden inside this
candidate.  At `5 m` the unamplified footprint is only about `2.4` source
texels, so the correct filter does not justify the current wholesale loss of
plant legibility there.

### 25.4 No-memory-increase codec: compact categorical geometry pays for mips

The existing categorical record contains binary coverage, one 16-bit
normalised depth, and two 16-bit oct-normal components in an eight-byte
`RGBA16F` GPU texel.  Binary coverage is redundant because the source format
already reserves depth code `65535` for MISS.  Transcode each nearest geometry
record to one universally supported four-byte `RGBA8` texel:

```text
R,G       16-bit depth code, 65535 = MISS, valid range 0..65534
B,A       oct-normal x,y, 8 bits each.
```

Nearest UNORM8 decoding reconstructs the two depth bytes exactly in float32.
On the longest roughly `4.55 m` slice, 16-bit depth spacing is about
`0.069 mm`, substantially better than the current load-time half-float's
millimetre-scale spacing near depth one.  Oct8 coordinate error is at most
half a `2/255` step; because the octahedral pre-normal vector has length at
least `1/sqrt(3)`, the conservative angular error is about one degree.  This is
small enough for the current face-lighting carrier and must still be checked in
the visual normal gate.

For the installed `65`-slice profile, current GPU profile storage is

```text
65 * 258^2 * (8 geometry + 4 colour) = 49.515 MiB.
```

The proposed independent `256 x 256` layers require

```text
geometry: 65 * 256^2 * 4                         = 16.250 MiB
radiance: 65 * (256^2+128^2+...+1) * 4           = 21.667 MiB
total                                                   37.917 MiB.
```

That is `11.598 MiB` (`23.4%`) less resident GPU memory while retaining the
full `256`-square colour base level and adding its complete mip chain.  Eight
geometry operations also move half as many record bytes.  Even if the one
radiance operation performs trilinear filtering across two coherent levels,
worst-case uncached profile bytes per pixel fall rather than rise.

The fixed arithmetic cost is two analytic ray differentials, two ray/plane
differentials, the terrain/chart shear applied twice, one symmetric `2 x 2`
largest-eigenvalue expression, and one `log2`.  It introduces no data-dependent
control flow.  The two columns can be evaluated serially while retaining only
the resulting `vec2`s; the expected additional live state is roughly eight to
twelve scalars, not another candidate record.  That register/occupancy risk is
real and requires one coherent GPU trace after the visual gate.  Replacing the
Jacobian by `t/H_px` would be cheaper but mathematically drops both
`1/|n dot d|` and the baked-address shear—the two factors which create the
reported directional alias—so it is not an accepted equivalent.

The shipped runtime asset must likewise stop carrying the owner atlas and the
2.17-million-triangle source tables solely to reconstruct colour at load time.
The cooker uses those tables once to emit the compact geometry layers and
premultiplied radiance chain, then keeps the analysis/source artifact under
`data/work`.  The production runtime payload is approximately the `37.9 MiB`
above plus small metadata, rather than the present roughly `114 MiB` GCRP.
This is an encoding replacement, not an additional copy or a lowered source
mesh.

### 25.5 Exact claim boundary and visual order

Candidate D is expected to remove the colour/radiance component of defect 6's
directional stream and defect 7's noisy temporal veil on continuously covered
regions.  It should make coherent stems and plumes easier to perceive at
`2--5 m` even though truly sub-pixel detail is correctly band-limited.  It may
preserve the violet plume's integrated colour farther away when a categorical
plume hit exists, addressing part of defect 9.

It cannot create a geometry event which all four nodes missed.  Therefore it
cannot by itself fix plume-before-stem disappearance caused by missing event
closure, defect 11's broken side volume, detached cover boundaries, or wrong
view reconstruction.  It also does not filter the categorical normal or binary
visibility.  If the moving particles remain under an existing constant-colour,
constant-normal resolve, that residue is definitively geometry/visibility
aliasing and no radiance LOD can remove it.  Exact fractional silhouette
coverage still needs a background-aware resolve; dividing colour by alpha does
not manufacture that missing layer.

The originally proposed implementation gate was:

1. first restore the complete record's sampled texel-centre point and geometric
   face normal; then transcribe the exact Jacobian including the current
   terrain-gradient shear;
2. bake the `256`-square periodic per-slice radiance chain and compact geometry
   as one replacement runtime payload, never as extra resident textures;
3. keep both categorical stages level-zero and keep the winner colour operation
   singular;
4. reject any new screen-relative LOD ring, loss of near authored colour,
   increased resident bytes, operation count above nine, or normal error visible
   at ordinary standing distance;
5. judge defects 6 and 7 first.  If they improve but binary edge/plume particles
   remain, record that exact one-layer blocker and move next to event closure;
   do not add colour reads or blur geometry.

That gate is superseded by the strict physical-tap rejection at the start of
this section.  The useful surviving results are the measured `2.11 m`
minification onset, the exact Jacobian, and the geometry-packing opportunity;
none independently authorises a filter implementation.

## 26. Review candidate E: restore the complete record's sampled ray and plane

This candidate is a proved transcription correction, not a new filter or a
larger representation.  It targets the field-wide stretch, view-direction
wedges, and camera-motion swim before any radiance work.  Runtime code is not
changed by this section.

### 26.1 A nearest record belongs to the texel-centre ray, not the requested ray

For one periodic slice, let the authored tile have origin `o`, size `S`, and
interior resolution `N`.  A continuous requested top-plane address `q` is read
with nearest spatial sampling.  The returned record was baked at the unwrapped
texel-centre address

```text
k       = floor((q-o)/S)
f       = (q-o)/S-k
i       = floor(N f)
Q(q)    = o + k S + (i+1/2) S/N.          (componentwise)
```

The wrapped gutter changes only which duplicate texel supplies a seam sample;
it does not change this infinite periodic centre lattice.  A deterministic
half-open convention at an exact seam selects one of two equivalent periodic
copies.

The integer spatial index is load-bearing.  The runtime must implement this as
**snap, then fetch**:

```text
i       = floor(N f)                         (computed once)
Q(q)    = q + ((i+1/2)/N-f) S               (unwrapped)
record  = atlasFetch(slice,i)                (the same i)
```

It must not issue a hardware-nearest fetch at continuous `q` and then derive
`Q(q)` independently in floating point.  WebGPU's filtered coordinate path and
the shader's `floor` path are not required to use the same boundary precision;
the atlas' `z` flip also reverses the apparent half-open convention.  Near a
texel boundary those independent decisions can differ by one texel and restore
the moving sawtooth on one side of the boundary.  Fetching the exact atlas
centre selected by `i` makes both axes and the flipped atlas row unambiguous.
The unwrapped formula for `Q` retains the periodic copy `k`; only the physical
atlas address wraps.

For baked unit direction `c`, the decoded depth is therefore
`tau_c(Q(q))`, and its categorical point is

```text
U_hat(q,c) = (Q(q).x,H,Q(q).z) + tau_c(Q(q)) c.
```

The active sampler instead constructs

```text
U_active(q,c) = (q.x,H,q.z) + tau_c(Q(q)) c.
```

so it translates the stored point by `(q-Q(q),0)` while leaving its depth and
normal unchanged.  `U_active` is generally not on the stored triangle.  The
same error exists in both stages: the provisional terrain anchor uses the
continuous initial request with the nearest initial depth, and the corrected
owner point uses the continuous corrected request with the nearest corrected
depth.

This is not merely the ordinary finite resolution of the atlas.  If the true
authored triangle plane is `n dot X = h`, the exact live-ray/plane solve through
the active point has parameter error

```text
Delta t = n_xz dot (q-Q(q)) / (n dot d).
```

The numerator is a spatial sawtooth under continuous camera motion.  Its
amplitude is at most half a texel per axis, but the denominator makes the error
unbounded as the live ray approaches the triangle plane.  The present tile has
`0.52/256 = 2.03125 mm` texels; a roughly `1.44 mm` half-diagonal becomes
`14.4 cm` at `|n dot d|=0.01` and `1.44 m` at `0.001`.  Thin vertical blades and
panicle faces routinely admit those incidences.  This is a direct mechanism
for long wedges, false depth streaks, and a moving fine-grain field even on
perfectly flat terrain.

At a live direction equal to a baked direction, the defect has an especially
simple interpretation.  Within one nearest texel the active reconstruction
returns the same baked depth for every parallel live ray and moves the inferred
plane point with that ray.  It therefore renders a piecewise-constant depth
sample, not the stationary source triangle plane.  Reconstructing `U_hat` and
intersecting its stationary plane analytically dequantises that plane exactly
throughout its valid triangle support.

### 26.2 The stored normal is not the plane normal used by the solve

The periodic baker writes the interpolated authored vertex normal to the
primary geometry record:

```text
candidate = normalize(interpolated vertex normal).
```

The runtime then treats that value as the covector of the affine triangle plane
when it evaluates `dot(N,Y-o)/dot(N,d)`.  Those are different mathematical
objects.  Blades interpolate normals between curved segments and tubes use
smooth radial vertex normals across planar facets, so equality is not an
authoring invariant.

Let `n` be the geometric face normal, `s` the stored shading normal, and write a
returned point relative to the live line as

```text
U-P = t0 d + e,          e perpendicular to d.
```

The true and active plane corrections are

```text
t_n = t0 + (n dot e)/(n dot d),
t_s = t0 + (s dot e)/(s dot d).
```

Thus

```text
t_s-t_n = (s dot e)/(s dot d) - (n dot e)/(n dot d).
```

It is unbounded when the interpolated shading normal approaches perpendicular
to the ray even if the real face has safe incidence.  In a two-dimensional
counterexample, take a horizontal true plane, live direction
`d=(1,-1)/sqrt(2)`, a returned point displaced horizontally by `epsilon`, and
`s=(sin theta,cos theta)`.  The false-plane depth error is

```text
sqrt(2) epsilon sin(theta)/(sin(theta)-cos(theta)),
```

which diverges at `theta=45 degrees`; the geometric face normal returns the
exact horizontal-plane hit independent of that horizontal residual.  Because
the interpolated normal also varies over a triangle, the false plane itself
changes when the selected texel changes, producing swim without any change of
source triangle.

The bake already computes the geometric `faceFacing` normal, and GCRP/v4
already stores an exact owner token plus its quantised source triangle table.
The primary record can therefore carry the geometric face normal in the same
two octahedral channels.  It can be written directly during the next bake, or
computed from the already-loaded v4 triangle table during the existing
load-time transcode.  No resident texture or source section is added.  Plane
intersection is sign invariant; for two-sided grass lighting the decoded face
normal may be face-forwarded against the live ray with fixed ALU after the
intersection.

### 26.3 Exact correction and invariants

For every existing initial and corrected geometry operation:

```text
record address       q
sampled ray origin   Q(q)
decoded depth        tau
categorical point    U_hat=(Q(q),H)+tau c
plane covector       geometric face normal n
```

The operation order is exact: derive the unwrapped tile and integer interior
texel index from `q`; form `Q(q)` from that index; fetch the geometry record at
the corresponding exact atlas texel centre (including the flipped atlas row);
then decode depth and form `U_hat`.  The winning colour operation reuses the
winner's selected spatial index and slice and fetches their exact atlas centre.
It does not resnap a selected floating-point address.

Use the initial `U_hat` as the continuous-drape corrector anchor.  Use the
corrected `U_hat` for the final drape, owner metadata, and live ray/plane solve.
The one winner-colour operation must use the winning `Q(q1)`, not continuous
`q1`: the colour texel was authored for that same categorical owner at that
same sampled ray.  Since `Q(q1)` maps to the exact texel centre, the existing
linear colour sampler returns that record without blending neighbouring owners.
This restores complete-record coherence without another operation.  It does
not solve distant footprint minification; that independent rate problem must
not be hidden by mixing owners at the geometry address.

The following properties then hold:

1. **Flat or affine terrain, exact baked direction:** for a record on the same
   triangle, the live ray intersects the stationary source plane exactly,
   independent of sub-texel camera phase.
2. **Flat or affine terrain, nearby direction:** if the corrected record remains
   on the same triangle and the resulting plane hit remains inside it, nearest
   address quantisation no longer moves or tilts that plane.  Adjacent texels
   of the same triangle reconstruct the same plane up to stored depth/normal
   quantisation.
3. **Camera translation:** while the categorical triangle stays fixed, moving
   `q` within or across its texels changes only the live ray; it does not move
   the reconstructed plane with the camera.
4. **Curved terrain:** `U_hat` is a real authored-surface point and
   `J(U_hat)^(-T)n` is its correct drape tangent covector.  Section 18.5's
   curvature error returns to second order in the corrector residual instead
   of containing the first-order `(q-Q(q))` false-point term.
5. **Periodic seams:** the unwrapped `k` in `Q(q)` preserves global phase and
   copy continuity; reducing `Q` to a canonical fractional tile before forming
   `U_hat` would be incorrect.

The correction is fixed `floor/fract` ALU plus a replacement of data already in
the same channels.  It remains exactly

```text
4 R0 + 4 R1 + 1 winner colour at Q(q1) = 9 reads,
8 bytes per resident geometry texel,
no additional source or GPU allocation.
```

It adds no binding, pass, dispatch, barrier, loop, march, candidate, filter,
runtime geometry, or per-species work.  Load-time face-normal derivation can
reuse the v4 owner/mesh arrays already needed by the authored-colour transcode
and releases them afterward as today.

### 26.4 Bounded claim and remaining slope discontinuity

This correction restores the meaning of one complete affine record.  It does
not make an infinite triangle plane equal to its finite triangle support, and
it cannot recover a live first event absent from the four point records.  A
same-plane intersection outside the source triangle remains the Section-18.6 /
16A support-and-successor blocker; no relaxed predicate is authorised here.

On non-affine terrain there is also a separate direction-cell drift.  Angular
bracketing and barycentric weights are selected from the predictor direction

```text
d_O = normalize(R[s d_xz], d_y-grad G(O) dot d_xz),
```

while corrected addresses use

```text
d_a = normalize(R[s d_xz], d_y-grad G(U0) dot d_xz).
```

If terrain curvature moves `d_a` into another lattice cell, the required new
slice is not among the four records selected for `d_O`; no exact re-election is
possible from the current reads.  Even inside the same cell, using `d_O`'s
weights to elect records queried with `d_a` is a literal code/spec mismatch and
can create slope-only row slicing.  Recomputing weights for `d_a` is exact only
while it remains in the originally loaded cell; rejecting or clamping a
cross-cell case is not a general fix.  Candidate E must therefore be judged
first on flat/affine stretch and translation swim, then on how much of the
uphill failure remains.  It makes no claim to close this missing-slice case.

### 26.5 Runtime transcription and review delta (2026-07-24)

Candidate E is now transcribed into the dedicated isolated-profile path.  The
active call graph is

```text
GroundCoverFrameBinding(calamagrostis-preview)
  -> buildCalamagrostisPrecomputedRayField
  -> CalamagrostisPrecomputedRayQuery
  -> CalamagrostisPrecomputedRayProfileSampler
  -> GroundCoverResolveShade
```

No module in that graph imports or executes a `LegacyPeriodic*` module.  The
deprecated GCAR multi-species path retains its own multi-only sampler/query;
its former standalone, pole, Hermite, authored-colour, and forced-profile
branches were removed.

`CalamagrostisPrecomputedRayProfileSampler.ts` implements the exact operation
order from Section 26.3:

1. `sampleSite(q)` computes `i=floor(N fract((q-o)/S))` once;
2. it forms the unwrapped `Q=q+((i+1/2)/N-f)S` from that same `i`;
3. it maps that same `i` to an exact atlas texel centre, including the flipped
   atlas row, before the texture operation;
4. all four R0 points use the shared R0 `Q`; each R1 node computes and uses its
   own corrected `Q`; and the winner-colour operation reuses the elected R1
   texel centre rather than resnapping a floating address.

The sampler has exactly four initial geometry operations, four corrected
geometry operations, and one winner-colour operation.  No loop, pass, binding,
dispatch, barrier, candidate list, runtime geometry, or allocation was added.
The only new runtime work is branch-free `floor/fract` centre arithmetic; the
old dummy standalone reads of the wind/deformation guide fields were removed.

The bake now computes one geometric face normal directly from each indexed
triangle's three positions and passes it flat to every covered fragment.  A
first full-mesh attempt exposed that fragment-derivative face normals could
quantise differently between complete submissions; moving the cross product
to the per-primitive vertex result made the construction deterministic by
definition.  Two full Apple Metal-3 submissions then matched exactly:

```text
asset SHA-256  e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be
asset bytes    119462112 (identical to the prior asset)
slices         65 (64 regular + one vertical pole)
atlas          13 x 5, 256 x 256 interiors, unchanged format
```

Typecheck, 43 focused profile/math contracts, bake diff checks, and the
deterministic full cook pass.  The exact Estonia `grassprofile=2` URL also
completed 126 rendered real-WebGPU frames without a page, TSL, shader,
bind-group, command-buffer, or uncaptured WebGPU error.  The user flight gate
remains pending at this record boundary.  The visual gate must
first compare flat/affine standing stretch and translation swim; residual
uphill row slicing and distant radiance minification are explicitly separate
known mechanisms, not reasons to hide a regression in the corrected record.

The live flight gate subsequently reported a slight improvement but no major
closure.  Candidate E is retained because its record/ray and plane identities
are exact, but dominant stretch, wrong apparent view, swim, early fuzz/grain,
incomplete geometry, and incoherent boundaries remain.  The angle dependence
is quantitative rather than categorical: the same failure is visible at both
low and higher oblique views, with stretch more conspicuous and worse at low
angles.  This result returns the track to the reconstruction equations; it
does not authorise more reads, more bytes, or a filter.

## 27. Review candidate F: replace the horizontal-height predictor by the R0 face plane

**Verdict:** approved for transcription.  Candidate E made every R0 record a
coherent sampled point and geometric face covector, but the dependent R1
address still ignores that plane.  The current address is the exact inverse of
a horizontal constant-height surface, not of the recorded grass face.  This is
a root error for near-vertical blades and plume facets.  Candidate F replaces
that predictor algebra using information already returned by the same four R0
operations.  It adds **zero texture operations and zero resident bytes**: the
complete path remains exactly `4 R0 + 4 R1 + 1 winner colour = 9`.

### 27.1 The current correction silently substitutes a horizontal plane

Work in the affine authored chart used by the corrector.  Let

```text
P_a = (p_a.x,H,p_a.z)       live top-plane entry,
d_a                          unit live direction in this chart,
c                            unit baked direction,
Q0                           snapped R0 top-plane origin,
tau0                         decoded R0 depth,
X0 = (Q0.x,H,Q0.z) + tau0 c coherent R0 point,
n                            its geometric face covector.
```

The active R1 address can be written

```text
t_h = tau0 c_y/d_a.y,
q_h = p_a + t_h [d_a.xz - (d_a.y/c_y)c_xz].
```

`t_h` is obtained only by asking when the live ray reaches the **height**
`H+tau0 c_y`.  It neither uses the face covector nor the lateral difference
`Q0-p_a`.  It is therefore exact for the horizontal plane through `X0` and,
as a general surface rule, only for that plane.  For a tilted or vertical face
the proper live intersection time depends on the whole plane.

The omitted error is explicit.  The true affine-plane parameter is

```text
t_0 = n dot (X0-P_a) / (n dot d_a)
    = [n_xz dot (Q0-p_a) + tau0 (n dot c)] / (n dot d_a).
```

Thus the current address error is

```text
q_h-q_* = [d_a.xz-(d_a.y/c_y)c_xz] (t_h-t_0).
```

Both terms which Candidate E made authoritative are absent from `t_h`: the
snapped baked-ray origin `Q0` and the geometric face orientation `n`.  At a
near-parallel live/face incidence, `n dot d_a` makes the phase error large;
at low oblique views that wrong R1 phase selects unrelated owners and becomes
the reported stretch, wrong apparent angle, and translation-dependent swim.
When `d_a=c`, the bracket is zero and both constructions return `q=p_a`, so
the correction cannot disturb an exact baked-direction lock.

### 27.2 Exact face-covariant inverse address

Intersect the live line with the already-recorded R0 face plane:

```text
t_0 = n dot (X0-P_a) / (n dot d_a),
Y_0 = P_a + t_0 d_a.
```

The baked `c` ray which passes through `Y_0` has parameter

```text
tau_* = t_0 d_a.y/c_y
```

because both rays start on `y=H`.  Its top-plane origin is therefore

```text
q_* = Y_0.xz - tau_* c_xz
    = p_a + t_0 [d_a.xz-(d_a.y/c_y)c_xz].
```

This proves the replacement; there is no fitted coefficient or interpolation
rule.  If R1 at `q_*` returns the same finite face and that face remains first,
Candidate E's snapped R1 point lies on the same stationary plane and the final
live ray/plane solve is exact on flat or affine terrain for **arbitrary face
orientation**.  R1 remains necessary because the first-hit chart can change
owner between the seed and corrected address; Candidate F does not numerically
blend or assume equality of those categorical records.

The face covector may be oct-decoded without its final normalisation for this
ratio: multiplying `n` by any nonzero scalar cancels between numerator and
denominator.  The R1 normal keeps its existing normalisation for lighting and
the final world covector.

### 27.3 Terrain equivalence and exact claim boundary

For affine terrain `G(u)=G_a+m dot (u-a)`, draping is an affine map.  The world
ray pulls back to direction

```text
d_tilde = (d_x, d_y-m dot d_xz, d_z),
d_a = d_tilde/|d_tilde|,
```

and the draped face covector is `J^-T n`.  Consequently

```text
(J^-T n) dot d = n dot d_tilde,
```

with the same identity in the plane numerator.  Section 27.2 is therefore not
a flat-terrain heuristic: it is exactly the ordinary world ray/draped-plane
intersection for every affine terrain slope represented by the corrector.

For curved terrain the inverse image of a world ray is a curve.  Along the ray,

```text
h''(t) = -d_xz^T Hess(G) d_xz,
```

so one tangent chart retains a second-order curvature residual.  Candidate F
removes the zeroth/first-order wrong-plane term but cannot make a single affine
corrector exact for arbitrary `Hess(G)`.  It also cannot load a direction slice
which predictor direction `d_O` omitted when corrected direction `d_a` crosses
a lattice cell.  Those are bounded residual mechanisms, not reasons to keep
the horizontal-height predictor.

Nor does this construction manufacture finite support or event closure.  If
the infinite R0 face intersects the live ray outside its source triangle, or
if a live-only first event is absent from all R0/R1 point samples, the current
record cannot decide that fact.  Candidate F makes the inverse step exact
under the stated same-face condition; it does not relax coverage, extend the
plane's claimed support, blur owners, filter radiance, or cure distant
minification.

### 27.4 Eligibility and degenerate limits

For each node, use Candidate F only when

```text
initialCovered,
d_a.y < 0,
abs(n dot d_a) > incidence_epsilon,
t_0 >= 0,
y_min <= H+t_0 d_a.y <= H.
```

Every installed baked direction has `c_y<0`, including the vertical pole, so
the `d_a.y/c_y` conversion is finite for the supported exterior domain.  At
`n dot d_a=0` the live line is parallel to the face plane: there is either no
unique intersection or the line is coplanar, neither of which this point
record can order.  Reusing the final plane solve's existing incidence threshold
is a numerical validity rule, not a hidden angle clamp.  `t_0<0` is a
behind-entry hypothesis and must be ineligible before categorical election;
letting it win and rejecting the whole pixel afterward incorrectly vetoes a
valid lower-weight node.

The last predicate is not a quality clamp.  `Y_0.y=H+t_0 d_a.y` is the seed
face's claimed live intersection, and every source triangle is contained in
the finite authored cover slab `y_min<=y<=H`, where `y_min` comes from the v4
source bounds rather than being assumed to be zero.  An intersection below its
bottom or above its top is therefore provably outside that finite face.  Letting
such an infinite-plane extrapolation generate `q_*` can send the dependent
address many tiles away at near-parallel incidence and recreate a stretched
sheet.  The equivalent baked-ray statement is that
`tau_*=t_0 d_a.y/c_y` lies between top entry and vertical slab exit.  This known
box test does not pretend to supply the unavailable in-triangle support test.

The existing tangent-entry pole `d_a.y -> 0` and a camera inside the cover box
remain outside this equation's promise.  The product contract already fades
inside the plant envelope; Candidate F does not expand that exterior domain.

### 27.5 Zero-read transcription and register lifetime

Each R0 value is already resident when its seed is formed.  For that node:

1. oct-unfold its R0 geometric face covector without normalising it;
2. immediately form the two dot products, `t_0`, and `q_*`;
3. discard the R0 covector before issuing the node's existing R1 operation;
4. retain the current R1 snapped point, R1 face normal, categorical election,
   and singular winner-colour operation.

This replaces the old `verticalRatio*tau0` address arithmetic.  It does not add
an operation, binding, texture, mip, pass, dispatch, barrier, loop, march,
candidate, species term, or persistent record.  The four nodes should be
lowered and reduced sequentially so no array of four R0 normals extends across
the dependent R1 reads.  The incremental work is one unnormalised oct unfold,
two dot products, and one division per existing node; the old vertical division
and address arithmetic disappear.  Texture traffic and locality are identical.

### 27.6 Visual review gate

The binding comparison is ordinary standing-height, low-oblique motion over
flat or gently affine terrain.  Candidate F should reduce:

- viewing-direction stretch and widening wedges;
- the false top-view/side-view orientation on non-horizontal plant faces;
- camera-motion swim caused by R1 phase moving between unrelated owners;
- the face-address component of uphill slicing.

Horizontal ground-like faces and exact baked directions should be visually
unchanged by identity.  Missing plume/stem events, finite-edge fragmentation,
angular cell rings, and unresolved `2--5 m` projected-footprint grain remain
separate known limits and must not be credited to or repaired inside this
change.  Any implementation which increases the nine physical profile reads
or resident profile bytes is not Candidate F and is rejected.  If oblique
stretch/swim remains visually unchanged after a correct transcription, the
remaining dominant cause is point-chart event/support closure rather than
another epipolar coefficient to tune.

### 27.7 Runtime transcription (2026-07-24)

Candidate F is transcribed only in the dedicated
`CalamagrostisPrecomputedRayProfileSampler`.  Each node oct-unfolds its already
loaded R0 face covector, evaluates `t_0`, rejects parallel, behind-entry, or
outside-`[sourceBounds.minY,H]` hypotheses, and forms `q_*` from the exact
face-plane equation.  Invalid hypotheses receive a finite dummy address before
the unavoidable uniform R1 operation and remain categorically ineligible; no
NaN address and no divergent read is introduced.  The R0 covector is consumed
inside the node rather than stored in `Seed`, limiting its live range.

The source still contains exactly one geometry texture operation in the shared
tap function instantiated as four R0 plus four R1 operations, followed by one
winner-colour operation.  There is no asset rebake or byte change after
Candidate E.  Typecheck, all 43 focused profile/ray-math contracts, and diff
checks pass.  The exact Estonia `grassprofile=2` URL completed 125 rendered
real-WebGPU frames without page, TSL, shader, bind-group, command-buffer, or
uncaptured WebGPU errors.  User flight review is pending.

The live gate found the predicted bounded improvement: nearby plume identities
are more stable and individually trackable during camera motion.  Distant
fuzz/stretch beyond a few metres and the stronger uphill warped/sliced view are
unchanged.  Candidate F is therefore retained as the exact local face-address
correction, while the remaining dominant mechanisms are confirmed to be
projected-footprint minification and curved-terrain direction-chart drift.  No
third local epipolar coefficient variant is authorised.

The subsequent existing `grassdbg=flatres` gate held both grass colour and
shading normal constant.  The user still observed the same coherent stream of
millions of tiny moving particles.  This falsifies conditional radiance and
normal-lighting alias as the dominant cause of that complaint.  Candidate S0's
colour scale stack must not be implemented as its cure; the binding mechanism
is categorical hit/miss and owner/support turnover.

Disabling TRAA while retaining `flatres` initially appeared to reduce that
stream, and the user recognised a similar moving-grain effect in AO beneath
branches.  The later AO-only comparison falsified the causal reading: removing
AO made the terrain and constant green grass too similar for the crawl to be
seen.  When an invalid 25-texture full-shadow diagnostic accidentally made the
terrain grey, the same crawl was obvious even though AO remained disabled.
Therefore neither AO nor shadow is the generator; both are contrast modulation
which reveals categorical grass position/visibility turnover.  The failed
binding is not a valid rendered result.  No profile or post-process change is
authorised from this confounded ablation.

## 28. Candidate G for review: same-byte two-scale filtered geometry atoms

**Status:** pure-math/code-design proposal only.  No bake, loader, shader, or
runtime file has been changed for this candidate.  Candidate E (sampled centre
plus face plane) and Candidate F (face-covariant dependent address) remain the
active near-field reconstruction.  The live gates found a bounded near-field
stability improvement from F, no material improvement in the `2--5 m` fuzz or
uphill slicing, and proved with `flatres` that the moving-grain complaint
survives constant authored colour and constant shading normal.  The later AO
comparison was contrast-confounded and authorises no AO/shadow fix.

### 28.1 Why the next carrier must change visibility, not radiance

The exact base-grid pitch is

```text
Delta = 0.52/256 = 2.03125 mm.
```

At the review camera (`1440 px`, approximately `55 degree` vertical field of
view), a fronto-parallel texel reaches one screen pixel at approximately
`2.81 m`; the face-incidence and baked-ray shear factors move that onset closer
for oblique blades.  The active eight geometry operations always return one
level-zero binary point record.  Hence camera translation and projection jitter
change HIT/MISS and owner at precisely the observed distance.  Holding colour
and normal constant cannot remove that signal because its source is screen
ownership and reconstructed position.

Section 24A proves that exact fractional silhouette compositing needs the
unknown background.  Candidate G does not contradict that theorem.  It changes
semantics only once a source footprint is unresolved: the output becomes one
explicit `FILTERED_ATOM`, followed by a deterministic coverage-preserving alpha
test.  It is an approximation to a sub-pixel bundle, not a fabricated exact
triangle owner.  At resolved scale the existing exact categorical record is
returned bit-for-bit.

### 28.2 Cook-side definition, valid for an arbitrary triangle soup

For direction slice `j`, let the ordinary periodic first-hit field be

```text
H_j(q) in {0,1},
tau_j(q), n_j(q), L_j(q) defined where H_j(q)=1.
```

Cook two positive periodic footprint kernels at source-grid scales `sigma_1`
and `sigma_2` (initial values `4` and `16` texels; the final pair is selected
from the measured `rho` distribution, never from camera distance).  For each
base-grid texel centre `q`, each scale stores:

```text
A_k(q)       = K_sigma_k * H_j,
tau_hat_k(q) = weighted median of covered first-hit depth,
class_k(q)   = nearest medoid in a per-slice joint (normal, premultiplied colour)
               codebook, evaluated over the same covered footprint.
```

The convolutions are toroidal and remain on the original `256 x 256` grid, so
all scales share one world-anchored spatial phase.  The median prevents a small
far surface from pulling the representative point behind the dominant front
bundle.  Each of the at most sixteen per-slice classes stores one unit geometric
representative normal and one linear RGB representative colour in metadata.
The input may be any finite triangle soup and may already contain overlapping
species: ray firstness and the footprint statistics are computed offline over
the complete community, so runtime cost is independent of primitive and species
count.

The filtered point for a baked direction `c_j` is the plane atom

```text
P_hat_k(q) = (q_x, H, q_z) + tau_hat_k(q) c_j,
Pi_hat_k   = { X : n_hat_k dot (X-P_hat_k)=0 }.
```

The live chart ray intersects this plane with the same exact line/plane identity
as Candidate F.  This retains a side-facing atom for vertical blades; it does
not substitute a constant-height sheet.  Because the atom is a filtered bundle,
the plane is representative rather than an asserted source triangle, and it is
eligible only in the unresolved-footprint regime.

### 28.3 Exact same-byte packing

Replace the current `RGBA16F` geometry plus `RGBA8` colour pair by:

```text
near RG32Uint (64 bits per texel)
  16 bits  exact categorical depth, 65535 = MISS
  20 bits  oct10 geometric face normal
  24 bits  RGB888 authored colour
   4 bits  spare/source class

scale R32Uint (32 bits per texel)
  atom 1: 10-bit depth, 3-bit coverage, 3-bit class
  atom 2: 10-bit depth, 3-bit coverage, 3-bit class
```

The three-bit coverages are not used as literal opacity.  Per-slice/per-scale
thresholds are fitted cook-side so the quantised field preserves the source hit
area under deterministic alpha test.  Ten-bit filtered depth has at worst about
`4.45 mm` spacing on the longest current slice; it is used only after the source
footprint is already several texels wide.  Near depth, plane normal, and colour
are not degraded.  Each scale has eight per-slice joint normal/colour medoids;
the external review must treat that finite codebook and the nearest-loaded
three-bit coverage field as the candidate's highest visual risks, not as settled
implementation details.

For `T=65*256^2=4,259,840` texels:

```text
new:     T*(8+4)                         = 51,118,080 B = 48.750 MiB
current: 65*258^2*(8 geometry+4 colour) = 51,919,920 B = 49.515 MiB
```

The runtime atlas therefore shrinks by `801,840 B`.  GCRP source triangles,
owner pages, and load-time colour reconstruction remain research/cook inputs
and leave the shipped payload, reducing the installed `119,462,112 B` asset to
approximately the packed atlases plus small metadata.  No old atlas is retained
beside the replacement.

### 28.4 Fixed nine-read runtime

The exact screen-to-address footprint remains Section 25.2's largest singular
value `rho`; it includes face incidence, baked-address shear, terrain gradient,
and the current jittered projection.  Define continuous log-scale weights with
partition of unity over the resolved point, `sigma_1`, and `sigma_2`.  Because
all three fields live at the same base-grid phase, changing `rho` changes only
ALU weights; it never changes a mip cell or texture address and cannot create a
camera-centred LOD circle.

Runtime traffic is exactly:

```text
4 R0 near textureLoad
+ 4 R1 near textureLoad
+ 1 scale textureLoad at the elected corrected site;
  if no near node is eligible, the same operation uses the max-angular-weight
  slice at the continuous top-entry site
= 9 physical texel loads.
```

The current winner-colour operation disappears because exact near RGB rides the
near complete record.  The active path currently performs eight nearest geometry
fetches plus four physical contributors for its bilinear winner colour, so the
replacement decreases worst-case physical texel contributions from twelve to
nine and decreases uncached bytes.  It adds fixed unpack, the already-derived
footprint Jacobian, two line/plane atom solves, and ALU interpolation only: no
loop, march, candidate list, pass, barrier, dispatch, binding-count growth,
runtime geometry, or species term.

At `rho<=1`, an eligible near record wins exactly.  Across the first transition,
near depth/normal/colour converge continuously to atom 1.  Across the second,
atom 1 converges continuously to atom 2.  Coverage follows the same partition,
then one deterministic area-preserving threshold decides whether the filtered
bundle owns the pixel.  A near MISS may therefore become a filtered HIT only as
its physical footprint becomes unresolved; no dummy point depth is used in the
resolved regime.  Atom position, normal, and colour always come from one scale
state or its continuous filtered transition, never from arithmetic over exact
triangle owners.

### 28.5 Claim boundary and gate

Candidate G targets the dominant field-wide failure first:

- directional moving particles and changing binary holes after `2--3 m`;
- fuzzy/grainy loss of coherent plume and stem masses through `2--5 m`;
- distance-triggered plume-before-stem disappearance when the plume exists in
  the local footprint but the exact point ray misses it;
- reduced lighting amplification through stable filtered representative normals.

It does not claim exact fractional compositing at isolated cover boundaries, a
new near-field event absent at `rho<=1`, or exact arbitrary-soup visibility from
one atom.  It also does not cure the separate curved-terrain direction-chart
drift responsible for uphill slicing.  If the filtered result becomes a veil or
textured sheet rather than coherent botanical masses, or if the constant-colour
particle stream is not materially reduced, the atom semantics are RED and must
be parked; neither threshold tuning nor another scale is then authorised.

The implementation order, only after independent review, is one coherent
end-to-end cycle: cook the replacement payload from the accepted source; update
the dedicated loader/sampler; verify near `rho<=1` parity and the exact nine-load
graph; boot real WebGPU; then gate ordinary colour and `flatres` in the same
walking path.  The unchanged uphill defect is measured separately and cannot be
used to obscure the minification verdict.

### 28.6 Adversarial verdict: RED before implementation

Candidate G is rejected as written.  A deterministic alpha threshold after a
filtered coverage field does not supply the missing background term.  On a
checkerboard hit field whose filtered coverage is approximately one half, a
constant threshold makes an opaque sheet or an empty field; a world-keyed
threshold makes a stipple whose nearest-cell turnover is the original moving
grain; a screen-keyed threshold swims.  Positive convolution likewise turns an
isolated filament into dilation, detached particles, or erosion.  More coverage
bits do not change that trilemma.

The review also found that fallback scale selection was circular when no exact
near record supplied the range/normal needed by `rho`, and that the class
codebook access was omitted from the cost model.  The packing byte arithmetic
was correct, but the visibility semantics were not.  No Candidate-G runtime or
asset change was made.  Its reusable results are the same-byte packing envelope
and the requirement for filtered geometry—not authorisation for thresholded
opaque atoms.  The next derivation must go up one level and preserve the actual
background sample.

## 29. Candidate H for external review: background-preserving filtered overlay

**Status:** mathematics and dataflow only; no code or asset change.  This
candidate replaces Section 28 rather than modifying its threshold.

### 29.1 Required output semantics

For filtered grass coverage `A`, conditional grass radiance `C_g`, and the
already visible background radiance `C_b`, the only correct colour equation is

```text
C = A C_g + (1-A) C_b.
```

Candidate H preserves both visibility samples until resolve.  Resolved exact
near grass has `A=1` and is opaque in this equation.  Only the unresolved
footprint uses fractional `A`.  No threshold, stochastic test, dither,
black-background multiplication, or averaged triangle owner appears.

The first H draft proposed replacing the main winner with grass and saving the
background id/depth.  That dataflow is rejected before external review.  The
query currently precedes the voxel raster, so it could save terrain while a
closer voxel later became the true background; worse, the stripped grass/voxel
resolve cannot shade an arbitrary terrain/mesh/voxel id without rebuilding the
forbidden all-material binding union.

The corrected dataflow instead leaves the existing background in place for
every fractional pixel.  Reserve bit 7 of `payloadV`'s low eight tie-break bits;
ordinary solid writers use the remaining seven.  This is a protocol rule for
**every** payload writer, not only meshes and voxels: mesh SW/HW, Mid, splat,
voxel, and exact grass must all write `tie & 0x7f`.  In particular, the current
exact-grass byte `((B & 0xf) << 4) | profileId` must be masked before election;
otherwise an opaque grass pixel can be misread as fractional.  The high 24
depth bits and the full id in `visBV` are unchanged.

```text
A = 1 (opaque exact):
  payloadV high24 = foreground grass depth
  visBV            = existing grass namespace id
  ray RG32Uint     = grass normal/height/coverage/class

0 < A < 1 (filtered):
  payloadV high24 = unchanged background depth
  payloadV bit 7  = fractional-grass marker
  visBV            = unchanged background full id
  ray RG32Uint     = grass depth/normal/height/coverage/class

A = 0:
  background payload/id are completely unchanged; marker is clear.
```

All solid writers, including voxels, finish before the grass query.  The query
then runs before HZB:

```text
mesh SW/HW -> voxel election -> guide/profile query -> HZB -> shadow-half.
```

The guide/query kernels are folded into the existing voxel tail before the HZB
kernels; this is a reorder of existing kernels, not another dispatch or submit.
One profile invocation owns one pixel after all background writers, so its
opaque election or fractional marker write is race-free.  No `depthV` reuse,
clear, read, or diagnostic semantic change is required.

The empty-background case is deliberately unchanged.  The query retains its
current `elect == 0` early-out and never ORs the fractional bit into an empty
payload.  Candidate H therefore does not add grass against sky at a ridge; an
isolated `0x80` payload would be decoded downstream as a far-plane receiver with
a garbage id and is forbidden.  The existing ridge/sky gap is outside this
candidate and may not be repaired by violating the background-preservation
protocol.

The existing resolve partition supplies `C_b`.  In the active split path, the
terrain pass also owns voxel backgrounds (terrain+voxel needs the two voxel
storage buffers but no explicit-vertex buffers), the mesh pass remains the
explicit-mesh background, and the already existing third grass pass becomes a
small grass-only overlay.  The order is

```text
terrain+voxel background -> explicit-mesh background -> grass overlay.
```

The overlay emits premultiplied `A*C_g, A` with fixed `ONE,
ONE_MINUS_SRC_ALPHA` blending.  It stays in the opaque render list with custom
blending so `renderOrder` remains load-bearing; it keeps `depthTest=false` and
`depthWrite=true`.  Fractional depth is the unchanged background depth and exact
depth is the foreground grass depth.  Therefore the actual framebuffer supplies
`C_b`; the grass pass never binds the all-material resource union.

The no-GI merged `both` configuration currently has no third pass.  It must not
gain one.  Its already-bound grass ray is instead used after the background has
been shaded inside that same material: fractional pixels evaluate the equation
internally, reusing the already-computed visibility and ambient terms and issuing
no second shadow/GI read.  These are compile-time resolve shapes, not a runtime
branch between pass topologies.  Exact grass remains the current one-surface
path and never evaluates a background.  The implementation must use a dedicated
profile-2 resolve helper; it must not re-import the legacy multi-profile material
graph into the background shaders.

There is no mathematically unique scalar depth for a fractional pixel.  H uses
the background depth for `0<A<1` and the foreground grass depth only at `A=1`.
This is the conservative raster/minification convention: fractional grass does
not dilate into an HZB/AO sheet, while fully covered grass remains an occluder.
Colour is continuous as `A -> 1`; depth is an explicitly gated approximation.
The `A=1` frontier is consequently a required moving-camera visual gate: walk
through the 2--3 m exact/filtered transition and inspect TAA, AO, and shadow
edges for a depth seam.  If a visible seam survives, the only same-carrier
fallback allowed for this attempt is to switch foreground depth at a lower
near-exact threshold such as `A >= 254/255`; interpolated grass/background depth
is forbidden because it creates a false occluding sheet.
For fractional pixels, grass direct lighting uses its own normal and the
already-evaluated visibility/ambient terms of the surviving background pass.
It may not issue a second shadow, GI, terrain, or canopy sample.  Exact `A=1`
grass retains the ordinary foreground lighting point.  The external review must
judge this bounded minification-lighting approximation rather than call it exact
multisample depth or lighting.

### 29.2 Same-byte arbitrary-soup source carrier

The profile payload is replaced, never duplicated:

```text
near R32Uint (32 bits/base texel)
  12-bit categorical depth, all-ones = MISS
  16-bit oct geometric face normal (8+8)
   4-bit colour class

scale RG32Uint (64 bits/base texel)
  atom 1: 10-bit depth, 8-bit coverage, 10-bit oct normal, 4-bit colour class
  atom 2: 10-bit depth, 8-bit coverage, 10-bit oct normal, 4-bit colour class
```

Twelve-bit depth preserves the accepted per-slice range to roughly
`0.29--1.11 mm`; the 8+8 oct normal is required because the normal drives both
the Candidate-F face-plane solve and shading.  The former amplifies angular
error near grazing incidence, so the rejected 7+7 split is not an acceptable
way to recover two depth bits.

The sixteen colour classes are one profile/community-global arbitrary-RGB plus
material-response palette held in fixed uniform state; they are not restricted
to green and cost no texture read.  This is deliberately lossy and must be
gated against the accepted near plume/stem colours; it is not described as
arbitrary-RGB exactness.  The cook chooses each filtered atom from the same
compact positive periodic footprint measure as Section 28, but fractional
coverage now reaches the actual resolved background instead of a binary
threshold.  The representative plane is explicitly an approximation after
minification.  Near records remain categorical and never blend owners.  The
carrier is produced by raycasting the whole input community, so arbitrary finite
triangle soups and species already overlapped offline require no runtime
primitive or per-species work.  It does not claim arbitrary runtime overlap of
independently selected atlases.

The scale operation must not depend on exact HIT/MISS eligibility.  It always
uses the maximum-angular-weight slice and Candidate-E's snapped top-entry texel
centre.  Atom 1 universally defines the finite provisional plane and footprint
coordinate `rho`, even when the exact record is valid; an empty atom has a
defined finite plane but `A=0`.  Each representative atom plane is intersected
analytically with the live ray.  Switching between an exact corrected site and
a top site at an eligibility boundary is forbidden because it recreates the
moving discontinuity.

Let the continuous exact/scale weights be `w_k`, the atom coverages `A_k`,
conditional colours `C_k`, representative points `P_k`, and normals `n_k`.
Every filtered moment is coverage-premultiplied:

```text
A   = sum_k w_k A_k
C_g = sum_k w_k A_k C_k / max(A, epsilon)
P   = sum_k w_k A_k P_k / max(A, epsilon)
n   = normalize(sum_k w_k A_k n_k).
```

Plain depth/normal/colour interpolation is forbidden: otherwise an empty or
nearly empty atom moves the visible representative.  Continuous log-footprint
weights form a partition over exact, atom 1, and atom 2.  At `rho<=1`, only the
categorical exact state participates and `A` is zero or one; as it becomes
sub-pixel, the exact weight decays continuously into the two filtered scales.
The positive filters have compact support and exact zero coverage outside it.
The one maximum-slice scale read remains a categorical angular approximation;
H targets spatial minification/particle turnover and does not claim that this
single read removes every direction-cell boundary.

For `T=65*256^2`, storage is

```text
T*(4+8) = 51,118,080 B = 48.750 MiB,
```

`0.765 MiB` below the active atlases.  This count requires integer-addressed
interior-only pages; retaining the old one-texel gutters would invalidate it.
Runtime source triangles, owner pages, and load-time colour reconstruction leave
the shipped payload.  Four R0 near loads plus four R1 near loads plus one scale
load are **nine physical profile texels**; the active path currently touches
eight nearest geometry texels plus four bilinear colour contributors.

The screen ray remains one 64-bit texture and `depthV` remains untouched.  Its
two 32-bit words have mode-dependent meaning:

```text
fractional: grass depth24 | coverage8 | oct-normal12 | height4 | RGB565
exact:      oct-normal16 | height8 | coverage=255 | unused/preserved bits
```

Exact RGB888 remains in the existing grass `visBV` payload.  The query expands
the atom palette, performs the coverage-premultiplied mix, then stores final
fractional RGB565; storing a four-bit *final* class would turn continuous colour
back into class pops.  One grass-contributing pixel performs one screen-ray read
in exactly one surviving resolve path, as it does now.  The split overlay owns
that read; the merged material already binds it.  The default exact configuration
must remain at or below the adapter's 24 sampled-texture limit.
The already-invalid `shalfres=0` permutation (`25 > 24`) is not evidence for H
and may not be made legal by requesting the adapter's optional 48-texture limit.

The new terrain+voxel/overlay pass builders must compute their generated
sampled-texture and storage-buffer counts and fail the build if they exceed the
project's default adapter ceilings.  This is a build-time conformance assertion,
not permission to raise a device limit.  Static accounting predicts eight
storage buffers for terrain+voxel, or nine where its existing GI probe is
present, under the ten-buffer ceiling; the real-WebGPU boot remains binding.

The added runtime work is fixed integer unpack, the footprint Jacobian, two
representative line/plane solves, reserved-bit tests, and fractional colour ALU.
There is no new loop, march, candidate list, pass, barrier, dispatch, screen
buffer, texture operation per contributing pixel, runtime geometry, or
per-species term.  Static sampler bindings, register liveness, divergent branch
cost, and the seven-bit tie-break change are real and require the default
real-WebGPU binding gate plus one coherent GPU trace.  Opaque dense grass must
take the existing one-surface path.

### 29.3 Frozen cook-side gates before runtime transcription

The two representation risks are measured before runtime code changes.  Their
thresholds are frozen here before inspecting the results.

**Sixteen-class joint palette gate.**  Decode the accepted source's linear RGB,
convert to D65 Lab, and fit one deterministic coverage-weighted 16-medoid palette
jointly across violet plume and green stem/leaf material semantics.  Source
primitive/material semantics define the two subsets before clustering; post-hoc
colour thresholds are not allowed.  The gate is GREEN only when all of these
hold:

```text
overall CIEDE2000 p95 <= 2.5 and p99 <= 5.0
violet-plume CIEDE2000 p95 <= 3.0
green stem/leaf CIEDE2000 p95 <= 3.0
connected covered support with CIEDE2000 > 5 occupies < 1%
```

Emit an original/quantized palette mosaic plus numbered error diagnostics and a
machine index.  A RED result blocks H; it does not authorize reducing coverage
precision or inventing another runtime read.

**Maximum-angular scale-carrier continuity gate.**  Pair adjacent direction
records by epipolar/world correspondence, never by equal texel address, and
measure every 25/45/65-degree elevation boundary and every azimuth midpoint
separately at both proposed filter scales.  At exterior review distances of
2 m, 5 m, and 10 m, H is GREEN only when every boundary satisfies:

```text
absolute coverage delta:            p95 <= 0.08, p99 <= 0.20
premultiplied linear-RGB max delta:  p95 <= 0.06, p99 <= 0.15
transverse representative error:    p95 <= 0.75 px, p99 <= 1.5 px
connected support exceeding a primary threshold: < 1%
```

Record representative depth delta and colour-class changes as secondary
diagnostics; neither may replace the transverse screen-space metric.  Emit
numbered boundary maps and a machine index.  Pooling boundaries or averaging a
RED boundary into GREEN is forbidden.  A RED primary metric means the declared
one-read angular carrier is not continuous enough and blocks runtime work.

The cook must also reject non-finite atom planes, including `A=0` atoms.  Their
coverage is empty but atom 1 remains the eligibility-independent address/scale
carrier, so NaN sentinels would make Section 29.2 circular again.

### 29.4 External-review resolution and one-attempt gate

External review must answer before code:

1. Is the reserved-bit fractional protocol race-free after moving the query
   behind every solid/voxel writer, and does any payload consumer require all
   eight old tie-break bits?
2. Is background depth for `0<A<1` and foreground depth for `A=1` the least-bad
   scalar convention for HZB/TAA/AO, or does its exact-opacity transition create
   a worse temporal failure?
3. Are twelve-bit near depth, 8+8 oct normal, and sixteen arbitrary RGB classes a
   tolerable near-source encoding, or is there a same-byte repack which retains
   more near fidelity without weakening filtered coverage?
4. Is the eligibility-independent max-angular/top-entry carrier plus universal
   atom-1 footprint genuinely non-circular and stable at all-near-MISS pixels?
5. Does terrain+voxel remain within the storage ceiling, does the existing split
   grass pass support opaque-list premultiplied blending in its load-bearing
   order, and can the merged path composite internally without another
   shadow/GI/terrain read or register collapse?
6. Does fractional background compositing remove the checkerboard, isolated
   filament, sparse plume, dense sward, and overlapping-species counterexamples
   which killed Candidate G?
7. Is the remaining categorical maximum-direction scale read acceptable for one
   attempt, or does it mathematically guarantee the same field-wide crawl/rings
   and therefore require a different same-read angular carrier first?

The external review approved the architecture subject to the protocol repairs
and the two frozen pre-measurements above.  If both measurements are GREEN,
this gets one coherent cook-to-live attempt.  The binding gates are
the normal-colour and `flatres` moving-grain walk at `2--5 m`, connected false
support at isolated filaments/cover edges, near Calamagrostis colour/plane
fidelity, the moving `A=1` depth-frontier walk, and a real GPU trace after visual
acceptance.  A veil, detached
particles, near palette damage, new binding-limit failure, or insignificant
particle reduction parks the carrier.  No threshold tuning, third scale, extra
tap, or extra screen buffer follows a RED result.

### 29.5 Frozen-gate result: palette GREEN, categorical angular carrier RED

Both premeasurements used the accepted Calamagrostis GCRP/v4 source rather
than a proxy.  The joint sixteen-entry palette is GREEN with large margin:
overall CIEDE2000 is `0.741` at p95 and `1.807` at p99, plume p95 is `0`,
stem/leaf p95 is `0.782`, and the largest connected `DeltaE00 > 5` region is
`0.0271%`.  The proposed four-bit community palette is therefore not the
blocker.

The maximum-angular scale carrier is decisively RED at every required
elevation and azimuth boundary and at both proposed spatial scales.  At the
critical 45-degree row boundary, `sigma=16` gives

```text
abs(delta A) p95/p99           = 0.4706 / 0.6196
premul RGB delta p95/p99       = 0.3012 / 0.4107
transverse error p95/p99 @ 2m = 129.24 / 165.84 pixels
largest connected exceedance  = 91.14%
```

Even the least-bad **raw carrier** family, the 75-degree azimuth switch at
`sigma=16`, remains `5.67` pixels p95 when expressed through the 10 m pixel
footprint, versus the frozen `0.75`-pixel limit; its coverage and
premultiplied-radiance errors also
remain roughly four times over their p95 limits.  This is not a near-threshold
or quantisation result.

A false-RED audit confirms that the measurement is the declared H
construction: the same toroidal square kernels, UNORM8 coverage,
coverage-premultiplied colour, covered-sample median and ten-bit per-slice
representative depth are used; each source representative point is reprojected
to the target chart by the exact top-plane epipolar identity; target reads use
the corresponding unwrapped nearest texel centre; both directions are tested;
and equal-address pairing is absent.  The raw atom difference is binding here
because the frozen Section 29.3 gate explicitly requires each proposed
one-read scale carrier to satisfy the boundary limits.  This measurement is
not falsely presented as a simulation of the final rho-weighted composite:
rho can change on both sides, and representative position is the nonlinear
ratio `sum(w A P)/sum(w A)`.  In particular, when one filtered atom is the only
covered contributor, its weight cancels and any nonzero weight retains the raw
position jump.  The least-bad raw mismatch remains `5.67` pixels p95 under a
10 m footprint; hiding the critical 45-degree mismatch would require an
effective linear weight below about `0.029` even at 10 m, incompatible with
that scale being useful in a dominant band.  Thus the 2 m normalization is not
manufacturing a threshold-edge verdict.

This result rejects the frozen unit-square radius-4/radius-16 atom carrier.  It
does not claim a theorem against every compact positive kernel; a differently
defined kernel or world-support-matched cook is a different carrier and must be
specified and premeasured rather than retuned post-result.

Therefore Candidate H's background-preserving fractional overlay remains a
sound reusable dataflow result, but its **one categorical maximum-direction
scale read is parked and may not be transcribed to runtime**.  Independently
baked first-hit charts do not become the same world-space measure under spatial
filtering: at a view-cell switch they coherently disagree about coverage,
radiance, and representative surface position.  That is precisely the
field-wide ring/crawl mechanism the carrier was meant to remove.

Resume only with a same-read, same-byte angular carrier whose continuity is
structural--for example a record defined directly over the live angular cell or
a direction-analytic world-anchored filtered measure--and run that carrier
through both the unchanged frozen boundary gate and a preregistered fidelity
gate against all 65 per-direction filtered truth fields before any shader work.
Continuity alone is not sufficient: a constant direction-independent field
would pass the boundary gate while discarding the plant.  The fidelity gate
must bind coverage, premultiplied colour, and full representative world-position
error independently.  The exact numeric contract and constructive
direction-analytic extinction candidate are recorded in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-DIRECTION-ANALYTIC-CARRIER-DESIGNER-BRIEF.md`.

Extra reads, more resident bytes, wider filters, a third scale, threshold noise, or a
runtime experiment are not authorised by this result.  The durable measurement
and immutable thresholds are recorded in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-GATE.md`
and
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-THRESHOLDS-FROZEN.md`.

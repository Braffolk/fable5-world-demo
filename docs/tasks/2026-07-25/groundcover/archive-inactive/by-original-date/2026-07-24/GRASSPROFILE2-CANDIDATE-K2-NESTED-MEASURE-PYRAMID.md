# Candidate K2: one-read nested measure pyramid

Date: 2026-07-24  
Status: **RED / parked**.  The LOD-threshold identity remains useful, but the
nested carrier fails base fidelity/translation and inherits Candidate K's
decisive arbitrary-angle RED; no runtime authorisation.

## 1. Inspectable outcome

The next inspectable product outcome is a `grassprofile=2` field in which the
Calamagrostis community remains botanical at standing/uphill views and loses
detail smoothly with screen footprint, without the 2--5 m particle crawl,
camera-centred LOD rings, new texture operations, or more resident memory.

Candidate K established the correct filtered quantity: unresolved cover owns
only the positive foreground measure

\[
M=(A,P),\qquad P=A C_g,
\]

and composites it over the real background as

\[
C=P+(1-A)C_b.
\]

It does not invent a filtered grass depth, owner, plane, or normal.  K2 changes
only the spatial scale representation.  It replaces the two isolated blur
endpoints with a nested positive-measure pyramid addressable by one physical
texture operation.

## 2. Nested positive measures

Let the periodic authored tile have side `S`, and let the finest filtered page
have `N_0=96` cells per side.  Define the nested resolutions

```text
N_l = 96, 48, 24, 12, 6, 3;  M_6 = one global periodic-cell measure.
```

For every stored angular vertex `v`, `M_l^v(i)` is a positive, toroidally
filtered foreground measure over level-`l` cell `i`.  Parent construction is
measure preserving:

\[
M_{l+1}^v(p)=\frac{1}{4}\sum_{i\in children(p)}M_l^v(i)
\]

through the `6 -> 3` level, and `M_6` is the mean of all nine `3 x 3` cells.
Coverage and premultiplied colour are averaged together.  Conditional colour
is never averaged independently of coverage.

The cook may replace the square child support with the live-direction
conditioned reference-plane footprint described in Section 6, but nesting and
positive normalization remain binding.  A non-nested collection of convenient
blur radii does not receive the continuity proof below.

## 3. One-read record

One `RGBA32Uint` texel at level `l` contains

```text
low 64 bits: four angular-corner RGBA4444 codes for M_l(i_l(q))
high 64 bits: four angular-corner RGBA4444 codes for
              M_(l+1)(parent(i_l(q)))
```

At level `5`, every high half contains the same quantized global measure `M_6`.
Each global angular vertex/level/cell is quantized exactly once.  Parent bits
are copied into every child record; they are not requantized in the child.

For live angular cell coordinates `(a,t)`, apply the ordinary four non-negative
bilinear weights to the four decoded low measures and separately to the four
decoded high measures.  Let the continuous footprint coordinate be

\[
\lambda=\log_2(\rho/\rho_0),\qquad
l=\mathrm{clamp}(\lfloor\lambda\rfloor,0,5),\qquad
\beta=\mathrm{clamp}(\lambda-l,0,1).
\]

The filtered result is

\[
M(q,d,\rho)=(1-\beta)M_l(q,d)+\beta M_{l+1}(q,d).
\]

The mip level is chosen before one integer `textureLoad`; both endpoints arrive
in that load.  The fixed decode and FE arithmetic contains no loop, march,
candidate list, divergent fetch set, dependent codebook access, or stochastic
selection.

## 4. Exact angular and footprint-level continuity

Angular continuity is inherited from Candidate K: incident angular pages copy
the exact same quantized global vertex bits.  Therefore their bilinear traces
are identical on every elevation boundary, azimuth seam, and pole edge.

At a footprint threshold `lambda -> l+1`, the old record tends to its copied
high endpoint

\[
\lim_{\beta\to1}M=(M_{l+1})_{copied}.
\]

The next level begins with the very same quantized bits as its low endpoint:

\[
\lim_{\beta\to0}M=(M_{l+1})_{native}=(M_{l+1})_{copied}.
\]

Thus every footprint-level transition is exactly `C0`, including `3 x 3` to
the global mean.  A perfect camera-centred colour/coverage ring cannot be
caused by the LOD selector under these premises.  This proves value continuity,
not derivative continuity.

This theorem does **not** claim continuity at an ordinary nearest spatial-cell
boundary inside one level.  There the low child may change.  Its jump is
coverage-premultiplied and is progressively suppressed as `beta` approaches
the shared parent, but it is not algebraically zero.  Millimetric translation
stability is consequently a binding gate.  A boundary-safe level-selection
variant may be used only after it has its own exact switch proof; it may not be
asserted from the LOD theorem.

## 5. Exact memory and operation count

The complete scale allocation is

```text
128 angular cells * 16 B *
  (96^2 + 48^2 + 24^2 + 12^2 + 6^2 + 3^2)
= 25,159,680 B
= 23.994140625 MiB.
```

With the 65-page, 256-square, 32-bit categorical near atlas:

```text
near   = 16.250000000 MiB
scale  = 23.994140625 MiB
total  = 40.244140625 MiB.
```

This is lower than Candidate H (`48.750 MiB`), Candidate K's initial two-scale
layout (`48.250 MiB`), and the active pre-H profile allocation.  The profile
operation count remains exactly

```text
4 R0 + 4 R1 + 1 scale = 9.
```

One scale operation transfers sixteen bytes.  No binding, screen buffer, pass,
dispatch, barrier, or runtime geometry is added.

## 6. Direction-conditioned footprint

An isotropic square blur is not the physical footprint of an oblique ray on the
botanical reference plane.  In the local authored affine frame, let the camera
be `C`, the normalized descending ray be `d(s)`, and the reference plane be
`y=h_ref`.  Then

\[
t_R={h_{ref}-C_y\over d_y},\qquad q=C+t_Rd,
\]

and for either screen coordinate `s_k`

\[
\partial_kq=t_R\left(\partial_kd-d{\partial_kd_y\over d_y}\right).
\]

The two horizontal columns form a rank-two footprint matrix `J_q`.  Its major
axis aligns with the grazing projection of the view and grows faster than its
cross-view axis.  The cook therefore builds `M_l^v` with the normalized
direction-conditioned ellipse/kernel of angular vertex `v`; runtime factors
the live matrix into this stored shape plus the scalar level coordinate
`rho`.  Adjacent angular vertices interpolate positive measures, not normals or
surfaces.

This factorization is exact for the horizontal authored reference plane under
an isotropic infinitesimal angular pixel.  Pixel aspect, projection jitter, and
the local affine terrain map enter `rho` and the two-column metric before level
selection.  Non-affine terrain curvature remains the same second-order local
carrier error already isolated by Candidate F; it is not repaired by widening
the footprint.

## 7. Exterior horizon

The deterministic periodic first-hit field need not possess a fine-scale
pointwise limit as elevation tends to zero: top-entry phase can wind around the
periodic tile arbitrarily many times.  Candidate K's `0.25 degree` node may not
be called an exact one-sided limit by assumption.

K2 instead makes two separate claims, both gated:

1. At resolved scales, progressively shallower directions must pass held-out
   fidelity; no positive-angle clamp is hidden.
2. At footprints which select coarser pyramid levels, the positive measures
   may converge to the global periodic mean.  This is a measured scale-dependent
   limit, not evidence that the fine field converges.

An exactly horizontal ray outside and above the finite cover box misses and is
not queried.  Every descending exterior ray which intersects the box remains in
scope.

## 8. Exact/filtered frontier

The categorical near record remains Candidate E/F's exact first-surface query.
For a resolved eligible hit it owns foreground depth and face normal.  As the
pixel footprint becomes unresolved, its binary foreground measure is blended
continuously into `M_0`; after that, K2's hierarchy governs appearance.

The scale query is eligibility independent.  A near MISS does not suppress the
scale record.  At a footprint where the exact query is still requested, the
two cases are

```text
eligible exact event: exact surface/radiance is the resolved endpoint;
no eligible event:    M_0 is the bounded positive-measure fallback.
```

This is not permission to fabricate an exact surface from `M_0`: the fallback
keeps background depth and contributes only `(A,P)`.  It addresses the known
event-closure erosion (a live event absent from the four exact point records)
without converting an averaged measure into an owner.  The eligible/MISS
frontier is not proved continuous, because the categorical exact record and
the filtered field need not agree.  It is therefore scored explicitly under
millimetric camera motion and thin plume/blade events.  If that frontier pops,
the representation is RED; stochastic eligibility and extra candidates are
not permitted.

The renderer's one-payload visibility architecture cannot retain both exact
grass depth/id and the independently shaded background id for one pixel.
Therefore a scalar depth-mode frontier remains unavoidable: exact mode writes
grass depth, while every fractional mode preserves background depth and
composites the positive foreground measure.  The switch is placed only after
the near event is sub-pixel and is a binding moving-camera/TAA/shadow gate.
Interpolating grass depth with background depth is forbidden because it creates
a false occluding sheet.

## 9. Required offline gates

No runtime transcription is authorised until all of these pass:

1. **Base codec:** compare unquantized and RGBA4444 `96 x 96` `M_0` against the
   accepted 256-square positive-measure truth at every stored direction.
2. **Angular held-outs:** compare every level separately against fresh exact-BVH
   truth at cell interiors, boundaries, pole, and `0.05--89.75` degree exterior
   directions.  Preserve physical filter support when test resolution changes.
3. **Hierarchy:** verify bit identity for every copied parent and both sides of
   every LOD threshold.
4. **Intermediate footprint:** compare the child-parent interpolation against
   independently integrated truth at quarter/mid/three-quarter footprint
   scales.  Endpoint success alone is insufficient.
5. **Translation:** sweep `1--4.5 mm` world/camera translations through ordinary
   spatial-cell boundaries and nested boundaries; report max-channel temporal
   p95/p99 and connected changed regions.  Include exact-HIT/scale-fallback
   eligibility changes as a separate stratum.
6. **Anisotropy:** compare the direction-conditioned kernel with exact projected
   pixel footprints at standing, uphill, low-oblique, and top-down views.
7. **Lighting:** gate the analytic unresolved-cover orientation distribution
   against filtered rendered truth over frozen sun directions and plume/stem
   subsets without adding stored normals or reads.

Primary per-direction appearance limits remain Candidate K's frozen limits:

```text
coverage absolute error:       p95 <= .08, p99 <= .20
premul RGB max-channel error:   p95 <= .06, p99 <= .15
connected p99 exceedance:       < 1%
```

The gate must additionally publish the error attributable to the spatial
nearest-cell boundary.  A large world-locked jump cannot be waved away because
the LOD selector itself is continuous.

## 10. Claim boundary

K2 is a general cook for a finite arbitrary triangle soup and already-overlapped
multi-species community.  Runtime cost is independent of triangle count,
instance count, and species count.  It is not a theorem that the fixed 128-cell,
96-square rate fits every possible soup: angular events narrower than a cell or
spatial variation above the stored rate can fail the offline fidelity gate.

The proved parts are positive-measure semantics, angular seam continuity,
footprint-level continuity, fixed operation count, and exact allocation.  The
unproved parts are held-out rate sufficiency, ordinary spatial-cell temporal
stability, extreme-grazing fidelity, and the filtered lighting fit.  Those are
the only questions the offline attempt is allowed to answer before Fable's
complete-boundary review.

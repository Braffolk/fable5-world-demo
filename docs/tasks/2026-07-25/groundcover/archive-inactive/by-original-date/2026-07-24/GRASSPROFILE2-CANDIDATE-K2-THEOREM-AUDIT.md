# Candidate K2 theorem audit

Date: 2026-07-24  
Status: **K2 as a complete carrier is RED; its copied-parent LOD seam theorem is retained**

## 1. Verdict

Candidate K2 contains one useful exact result: if the parent bits embedded in
level `l` are the same bits read as the current value at level `l+1`, its
footprint-level switch is exactly `C0`.  The byte count and the one-read decode
also check out.

That theorem does not address the active failure.  K2 spends the eight
`RGBA4444` values in its 128-bit record on the two angular coordinates and one
footprint coordinate.  Both periodic world-phase coordinates remain nearest
sampled.  Their discontinuities survive ordinary camera translation and are a
direct mathematical source for the reported one-direction particle crawl.

The corrected held-out result independently parks the other premise: the
unlimited-precision four-corner angular FE is RED in `212/224`
direction-by-scale cases, with worst p95 coverage error `.667`, p95
premultiplied-RGB error `.490`, and connected error `.701`.  Exact stored-edge
directions together with failed sector midpoints are the fingerprint of
angular under-resolution, not codec quantisation.

The machine report used for those figures is
`data/work/groundcover-candidate-k-heldout-fe/e3e0a4175b151b89/60a8e06b5f69e390/report.json`.

Therefore the nested pyramid must not be implemented as the replacement for
`grassprofile=2`.  Its copied-parent layout remains reusable inside a later
carrier which has already solved spatial and angular reconstruction.

## 2. What is actually proved

For phase `f` and adjacent power-of-two levels,

\[
i_l=\lfloor N_l f\rfloor,\qquad
i_{l+1}=\lfloor N_{l+1}f\rfloor
       =\left\lfloor {i_l\over2}\right\rfloor .
\]

If one canonical quantised parent code is copied into every child record, then

\[
\lim_{\beta\to1^-}\big((1-\beta)M_l+\beta M_{l+1}\big)
=M_{l+1}
=\lim_{\beta\to0^+}\big((1-\beta)M_{l+1}+\beta M_{l+2}\big).
\]

This proves value continuity at an LOD page switch.  Shared angular vertex bits
likewise prove value continuity on angular page boundaries.  Positive-measure
semantics, premultiplied compositing, the `40.244140625 MiB` accounting, and the
fixed `4 R0 + 4 R1 + 1 scale` operation count are internally consistent.

These are seam and accounting theorems.  They are not reconstruction or
temporal-stability theorems.

## 3. Fatal spatial-cell discontinuity

Hold the selected level and footprint fraction fixed.  Across an ordinary
phase-cell boundary K2 returns

\[
M^-=(1-\beta)C^-+\beta P^-,\qquad
M^+=(1-\beta)C^++\beta P^+,
\]

so

\[
\Delta M=(1-\beta)\Delta C+\beta\Delta P.
\]

Inside one parent cell, `P^-=P^+`, hence

\[
\Delta M=(1-\beta)\Delta C.
\]

At the fine end of every interval `beta` is near zero, so the complete child
jump survives.  At a parent boundary the parent jump survives as well.  Only
the isolated LOD threshold is `C0`; the field between thresholds is a moving
staircase in both world-phase dimensions.

For a tile of side `S`, the level-zero crossing pitch is `S/96`.  A
`1--4.5 mm` translation is intentionally of the same order as that pitch for
the accepted community.  RGBA4444 additionally turns a non-zero component
jump into a multiple of `1/15`.  The temporal gate can measure this defect, but
cannot make it pass by changing a threshold.  It would have to find the
community nearly constant at every crossed boundary, which is contrary to the
botanical signal K2 is meant to preserve.

This is the most important conclusion of the audit: **the particle crawl is
structurally retained, not merely untested.**

### 3.1 Record-capacity interpretation

One record contains eight independent 16-bit positive-measure samples.  A
generic tensor-product linear FE over `m` binary interpolation coordinates
needs `2^m` corner samples.  K2 uses exactly

```text
2 azimuth * 2 elevation * 2 footprint = 8 samples.
```

Adding both spatial coordinates without a proved factorisation would require

```text
2^5 = 32 RGBA4444 samples = 512 bits.
```

This is not a universal codec lower bound--simplicial, low-rank, or
model-based fields can use other rate structures--but it is a strict no-go for
silently extending K2's present tensor FE.  The missing spatial neighbours are
not recoverable from its current 128 bits.

## 4. Parent construction and anisotropic footprints conflict

K2 states both

\[
M_{l+1}(p)={1\over4}\sum_{c\in children(p)}M_l(c)
\]

and that each level may be independently integrated with a
direction-conditioned elliptical footprint.  These are not generally the same
field.  Averaging four translated child kernels produces the child kernel
convolved with an axis-aligned four-point box.  It is not, in general, the
next homothetic view-aligned ellipse.

The seam proof does not require child averaging.  The mathematically clean
choice is:

1. integrate each physical parent footprint independently in unquantised
   positive measure;
2. quantise that parent once; and
3. copy those exact bits into all incident children.

That preserves LOD `C0` but abandons the claimed exact child-average identity.
K2 must choose this interpretation if its LOD subcodec is reused.

More generally, the rank-two projected pixel footprint is not described by
one scalar radius.  Up to scale it retains aspect and orientation.  A fixed
kernel per angular vertex is exact only for its frozen circular-angular-pixel,
fixed-roll, affine-frame premise.  A square pixel, camera roll, non-conformal
terrain frame, or wind shear changes the shape.  Blending two vertex-filtered
measures produces a mixture of their kernels, not the exact live ellipse.
Largest-singular-value selection is conservative but overblurs the narrow
axis, reproducing the reported 2--5 m fuzziness.

## 5. Fine-scale horizon no-go for a fixed angular lattice

Use slope coordinates

\[
s={d_{xz}\over |d_y|}.
\]

For a hit of height `y` charted on reference height `h`, changing the view from
stored slope `s_j` to live slope `s` moves its phase by exactly

\[
\delta q=(y-h)(s-s_j).
\]

Near elevation `e=0`, `|s|=cot(e)` and

\[
\left|{d s\over d e}\right|=\csc^2(e)\sim {1\over e^2}.
\]

Thus any fixed positive angular spacing causes unbounded fine-scale phase
misregistration toward grazing.  A `0.25 degree` vertex is not a uniform
one-sided approximation for `0<e<0.25 degree`.  The corrected held-out result
shows this in practice: the shallow field is nearly opaque while its
conditional premultiplied colour remains badly wrong.

Coarse positive measures may converge to a periodic mean, but that requires an
explicit **scale-angle coupling**.  If a height family has residual half-range
`R` and the angular cell has slope diameter `Delta_s`, its filter support must
obey, at minimum,

\[
r_l \gtrsim R\,\Delta_s
\]

before interpolation can hide the phase uncertainty.  K2 selects scale from
screen footprint alone and therefore lacks this necessary anti-aliasing term.
Replacing `rho` by a continuous bound such as

\[
\rho_{eff}=\max(\rho_{screen},R\,\Delta_s)
\]

is a structurally stronger rule with no new read or byte.  It necessarily
loses fine detail toward grazing; that is information-theoretic filtering, not
a quality knob.  It still does not repair the nearest spatial staircase.

## 6. Audit of the proposed three-height-slab replacement

### 6.1 What is sound

Let first-hit measure be partitioned by hit height into disjoint slabs:

\[
M=\sum_{k=1}^3 M_k.
\]

This is exact because every ray-footprint sample assigns its first hit to one
slab.  The contributions must be summed; multiplying independent slab
transmittances would invent visibility and is correctly rejected.

Three spatial-triangle fetches per slab give an ordinary `C0` spatial FE if
all incident triangles copy the exact same packed vertex codes.  Packing four
angular corners and two scale endpoints in each fetched 128-bit record also
preserves the existing angular and scale seam proofs.  Nine fixed loads, with
no loop or march, satisfy the raw operation count.

### 6.2 Why three slabs do not solve angular fidelity

For slab reference height `h_k`, every contributing surface satisfies

\[
\|\delta q_k\|
=|y-h_k|\,\|s-s_j\|
\le R_k\,\Delta_s.
\]

Three uniform slabs reduce `R` by only a factor of three.  They do not change
the `1/e^2` horizon divergence, nor the azimuthal error
`Delta_s approximately |s| Delta_phi`.  At shallow elevation, even a
millimetric residual height can traverse many periodic tiles.  Consequently
`K=3` cannot turn the failed fixed-degree angular FE into a horizon-complete
field.  It merely moves the same failure to a smaller constant.

For the accepted approximately `1.176 m` cover, three equal slabs have
`R approximately 0.196 m`.  The slope gap from `0.25` to `2` degrees then
permits about `39 m` of residual phase displacement; extrapolating the
`0.25`-degree node to `0.05` degrees permits about `180 m`.  These are not
codec-scale residuals.

Visibility-conditioned slab membership introduces another cancellation
requirement.  Exact slab measures sum to the total, but the three runtime
fields use different reference-plane addresses and generally different
spatial triangles.  Therefore

\[
\sum_k FE_k(M_k(q_k))
\ne FE\!\left(\sum_k M_k\right)
\]

away from stored nodes.  A surface changing height class with view can leave
double or missing interpolated energy unless the held-out sum, rather than
each slab alone, is gated.

Independent RGBA4444 slab quantisation also has worst-case component error

\[
3\cdot {1\over30}=0.1,
\]

already above the frozen p95 premultiplied-RGB limit `.06`.  Coverage can sum
above one; clamping it is nonlinear and destroys the exact sum identity.
This does not prove measured failure, but it rules out carrying K's one-field
quantisation bound over unchanged.

The proposed analytic far mean is not free generality either.  Spatial phase
averaging removes `q`, but the resulting periodic-cell mean remains a function
of view direction and can retain occlusion-driven colour changes.  Its claimed
analytic fit needs its own directional fidelity gate; a constant mean would
only game continuity.

### 6.3 It consumes the complete query budget but has no near geometry

The nine slab fetches return positive appearance measures only.  They contain
no exact first-hit depth, owner, face plane, or normal.  If they replace the
existing `4 R0 + 4 R1 + 1 scale` query, dense near cover retains background
depth everywhere and cannot produce correct object occlusion, side depth,
near normals, or the accepted stable plume identities.  If the Candidate-F
near query is retained, the total becomes seventeen profile operations, which
violates the contract.

Memory does not rescue this conflict.  Three slab atlases with `N x N`
spatial vertices cost approximately

\[
128\cdot3\cdot N^2\cdot16\ bytes.
\]

The `32.5 MiB` scale allowance limits `N` to about `74`; `N=64` costs
`24 MiB`.  This can fit alongside the near atlas in bytes, but not alongside
its eight near operations.  Dropping the near atlas saves operations and bytes
by dropping a required visual capability.

**Verdict on the proposed replacement: RED as a complete carrier.**  Height
partitioning is a useful coordinate reduction, not a substitute for the exact
branch and not a cure for fixed-angle horizon sampling.

## 7. Exact/filtered frontier

Positive foreground blending can make colour continuous if the exact colour's
weight reaches zero before filtered-only mode.  It cannot make the visibility
payload continuous: the one-payload architecture switches from grass depth/id
to background depth/id.  No interpolation of those depths is legal because it
creates a false occluding sheet.

Therefore the frontier is an admitted representation discontinuity.  It may
be visually harmless only after the exact event is sub-pixel and its colour
weight is zero.  K2's base phase field must agree with that limiting colour at
the same address; a finite nearest `96 x 96` field does not establish this.
TAA, shadow, AO, and moving-camera checks at the frontier remain hard gates.

## 8. Terrain, wind, and dynamic lighting limits

### 8.1 Curved terrain

The tangent-plane drape is exact on affine terrain only.  For curvature
`kappa`, travel `r`, and incidence `n_g dot d`, the documented error

\[
|\delta t|\lesssim {\kappa r^2\over2|n_g\cdot d|}
\]

has no uniform exterior-angle bound because the denominator tends to zero at
grazing/uphill tangency.  Two terrains can share the same point and gradient
but have different curvature and hence different visibility while presenting
the same local K2 inputs.  Exact arbitrary curved-terrain reconstruction needs
a terrain-specific prewarp or more terrain information; K2 can only gate a
bounded-curvature approximation.

### 8.2 Affine wind

Inverse affine conjugation is exact for the central oriented line.  It is not
automatically exact for K2's filtered footprint.  A shear is non-conformal, so
the inverse map changes footprint aspect and orientation; that result is not,
in general, a scalar multiple of the direction-node kernel.  The scalar LOD
factorisation must therefore be gated over the full allowed shear range.

Likewise an axisymmetric normal distribution does not remain axisymmetric by
merely replacing `u` with `normalize(Bu)`.  Face normals transform by
`B^{-T}` with orientation-dependent area weights.  The proposed wind lighting
is a bounded approximation, not an affine-conjugation theorem.

### 8.3 Dynamic lighting

`(A,A*C)` contains no orientation or self-visibility statistic.  There exist
two coloured triangle soups with identical stored positive appearance measure
but different normal distributions and therefore different radiance under a
new light.  No analytic function of `(A,A*C),u,v,l` can distinguish them.

The closed-form ribbon/axial/isotropic integrals themselves are sound, but
community-global fitted weights cannot reproduce arbitrary spatially varying
orientation, correlated first-hit occlusion, self-shadow, or plume/stem
transfer.  Reusing background shadow visibility is especially not a grass
self-shadow solution.  Dynamic lighting is consequently an explicitly fitted
appearance approximation and needs per-light, per-view, plume/stem, and
shadowed-region gates.  It is not part of K2's proved generality.

## 9. Stronger direction for the next carrier

The next design should not add another blur level to K2.  It must satisfy all
three of these structural conditions before codec tuning:

1. **spatial continuity:** use a shared-vertex spatial FE (or an equally exact
   partition of unity), not nearest phase cells;
2. **shear-matched angular sampling:** parameterise by slope `s`, partition
   height, and enforce `R_k Delta_s <= r_l` at every stored
   angle/scale cell, with a measured global fringe when the periodic field has
   genuinely mixed; and
3. **near ownership:** retain an exact depth/plane/normal path until its event
   is sub-pixel, or prove a replacement that carries equivalent visibility.

The three-slab/three-spatial-sample idea satisfies condition 1 and provides a
useful bound for condition 2, but fails the bound near the horizon and spends
all nine operations before condition 3.  A better fixed-dimensional
decomposition must share information between near geometry and height families
rather than treating them as two independent nine- and eight-read systems.

One practical premise audit should precede any new packing work: build a
rate table in **slope coordinates** for `R_k Delta_s / r_l`, using the actual
hit-height residual distribution at every spatial scale.  That table gives the
minimum height-family and angular-cell counts.  If it exceeds the nine-read or
`48.75 MiB` envelope, the proposed representation is disproved before another
shader or atlas is built.

## 10. Claim boundary

The following survive this audit:

- background-preserving positive foreground compositing;
- copied-parent one-read LOD seam continuity;
- shared angular-bit seam continuity;
- fixed-cost offline-overlapped community semantics; and
- the exact memory arithmetic.

The following do not survive:

- K/K2 four-corner angular fidelity;
- a claim that the nested hierarchy removes phase crawl;
- exact anisotropic footprint factorisation by one scalar;
- horizon completeness from a finite degree lattice;
- exact filtered affine-wind lighting;
- general dynamic lighting from `(A,A*C)` alone; and
- the three-height-slab proposal as a complete nine-read replacement.

This is a mathematical RED, not a request for more taps or bytes.  The resume
condition is a carrier whose spatial field is `C0`, whose slope/height/scale
rate obeys the explicit residual-parallax bound, and whose nine operations
still include resolved first-surface ownership.

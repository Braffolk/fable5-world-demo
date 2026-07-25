# Candidate K nested one-read LOD hierarchy — mathematical audit

Date: 2026-07-24  
Status: **mathematically approved with binding footprint and temporal gates;
no runtime authorisation**

## 1. Proposed replacement

Replace Candidate K's two isolated filtered scales with one true nested
positive-measure hierarchy per angular cell:

```text
phase resolutions: 96, 48, 24, 12, 6, 3, then one global parent
angular cells:      128
physical read:      one RGBA32Uint texel
```

For a selected level `ell`, the loaded texel contains eight 16-bit symbols:

```text
low 16 bits of four u32 channels:  four angular RGBA4444 vertices of M_ell
high 16 bits:                       bit-identical parent vertices M_(ell+1)
```

The shader bilinearly combines the four current angular vertices, bilinearly
combines the four parent vertices, then blends those two measures by the
continuous footprint fraction.  This is the same decode/interpolation shape as
the previous two-scale Candidate K, but the two endpoints now belong to the
selected adjacent LODs.

## 2. Exact memory and read accounting

No separate `1x1` page is needed: the global parent is duplicated into the
high halves of the `3x3` current level.  Therefore

\[
\begin{aligned}
B_{scale}
 &=128\cdot16\cdot
   (96^2+48^2+24^2+12^2+6^2+3^2)\\
 &=25,159,680\ {\rm B}
  =23.994140625\ {\rm MiB}.
\end{aligned}
\]

With the unchanged near allocation,

```text
near:   65 * 256^2 * 4 B = 16.250000 MiB
scale:                       23.994141 MiB
total:                       40.244141 MiB
```

This is `8.005859 MiB` below Candidate K's `128@128` allocation.  Runtime
still performs exactly one sixteen-byte scale texture operation and the total
profile count remains `4 R0 + 4 R1 + 1 scale = 9`.  There is no new texture,
binding, pass, dispatch, barrier, loop, march, candidate list, or per-species
work.

## 3. C0 proof across LOD thresholds

Let the periodic phase coordinate in one axis be `f in [0,1)`.  For adjacent
power-of-two levels `N_ell=2N_(ell+1)`, define the half-open indices

\[
i_\ell=\lfloor N_\ell f\rfloor,\qquad
i_{\ell+1}=\lfloor N_{\ell+1}f\rfloor.
\]

Then

\[
\left\lfloor{i_\ell\over2}\right\rfloor=i_{\ell+1}.
\]

The parent code embedded in child texel `i_ell` is therefore exactly the code
which the next level reads as its current value at the same phase.  If `s` is
the local footprint fraction, the output is

\[
M(f,\lambda)=(1-s)M_\ell[i_\ell]+sM_{\ell+1}[i_{\ell+1}].
\]

As the footprint crosses the level boundary, the lower interval tends to
`M_(ell+1)` and the upper interval starts from the identical bits.  The value
is exactly C0 even though the selected page and integer address change.  The
same proof holds in two phase dimensions.  For `3x3 -> global`, every `3x3`
texel embeds the same global parent, so equality is immediate.

This proof requires all of the following cook invariants:

1. each global angular/spatial parent vertex is quantised exactly once;
2. those exact 16 bits are copied into every child and incident angular page;
3. current and parent use the same half-open periodic phase convention;
4. the global parent is one canonical value, not nine independently rounded
   copies; and
5. the LOD fraction reaches exactly zero/one at the page switch (or uses an
   algebraically equivalent continuous partition).

Independently quantising the embedded parent, or using hardware-nearest
rounding for one endpoint and integer floor for the other, would break the C0
guarantee.  Recomputing a parent from already quantised children can remain C0
when the resulting bits are copied identically, but accumulates avoidable
fidelity bias and is rejected by the independent-parent truth gate.

## 4. What the proof does not solve

### 4.1 Spatial texel crossings

Within a fixed LOD, integer phase sampling is still piecewise constant.  At a
child-texel boundary the jump is

\[
\Delta M=(1-s)\Delta M_\ell+s\Delta M_{\ell+1}.
\]

The hierarchy reduces the jump as the footprint grows, but does not make that
boundary C0.  Consequently the old one-direction particle crawl cannot be
declared solved from the LOD proof alone.  One-to-4.5-mm translated-camera
sequences remain a binding gate at every level and at parent-cell boundaries.

### 4.2 Angular interpolation

The hierarchy changes only spatial filtering.  It inherits Candidate K's
shared-angular-vertex continuity proof and its still-pending arbitrary-angle
fidelity obligation.  A smooth LOD cannot repair an angular basis which shows
the wrong plant view between stored directions.

### 4.3 Exact geometry

All hierarchy values are unresolved positive moments `(A,A*C)`.  They own no
grass depth, point, normal, owner, or side surface.  Exact near records remain
categorical.  Numeric `A=1` in a filtered LOD remains fractional mode with
background depth.

## 5. A true hierarchy is a footprint hierarchy, not image resizing

Let `J(q,d)` be the rank-two screen-to-reference-plane Jacobian from Candidate
K.  Its columns define the physical pixel parallelogram.  At low obliquity its
singular values can differ greatly; selecting a generic isotropic square by
the largest singular value overblurs the narrow direction and directly risks
the reported two-to-five-metre fuzziness.

The cook must therefore define, for each angular vertex `d_j`, a base positive
kernel `K_(j,0)` from the canonical rank-two reference-plane footprint and a
nested family

\[
K_{j,\ell+1}=K_{j,\ell}*B_{j,\ell},
\]

where `B` is a normalised positive two-times dilation/aggregation kernel in the
same direction-conditioned frame.  It then integrates the original hit
measure:

\[
M_{j,\ell}=K_{j,\ell} * (H_j, H_j C_j).
\]

Parents are formed from unquantised positive moments and quantised once.  A
repeated resize of conditional RGB, a square-kernel hierarchy unrelated to
`J`, or an average of already quantised children is not this construction.

For fixed camera roll and affine ground, angular direction determines the
footprint orientation and anisotropy while distance supplies its scalar size;
the hierarchy is then a one-dimensional scale family.  Terrain curvature,
arbitrary camera roll, and direction interpolation perturb that shape.  These
are measured held-out errors, not silently collapsed into the scalar LOD.

## 6. LOD coordinate

Let `r_ell` be the physical support radius (or another frozen monotone footprint
measure) of level `ell`, derived from the actual kernels rather than inferred
only from texture resolution.  Choose the unique interval

\[
r_\ell\le\rho<r_{\ell+1}
\]

from the live Jacobian and use

\[
s={\log\rho-\log r_\ell\over
       \log r_{\ell+1}-\log r_\ell}.
\]

This handles the final `3 -> global` interval honestly even though its spatial
ratio is three rather than two.  The page index is a fixed select from this
scalar; no candidate search is introduced.  Exact-to-level-zero blending uses
the same partition discipline and is separately gated because its endpoints
are categorical and filtered representations, respectively.

## 7. Binding offline gate

The nested hierarchy supersedes the old two-scale spatial gate only if all of
the following pass before runtime work:

1. **Base fidelity:** `96x96` level zero, unlimited precision first and then
   RGBA4444, against truth integrated at its actual base footprint.
2. **Every parent:** each unquantised and packed level against independently
   integrated truth at that physical footprint; never only against resized
   child data.
3. **Intermediate scale:** quarter, midpoint, and three-quarter `s` values
   against independently integrated intermediate-footprint truth.
4. **LOD boundaries:** `+/-epsilon` on both sides of every transition, requiring
   bit-exact endpoint equality and no connected visual ring.
5. **Temporal phase:** one-to-4.5-mm translations within child cells, across
   child boundaries, across parent boundaries, and while crossing an LOD
   threshold.
6. **Anisotropy:** standing, low-oblique, uphill, and top-down rank-two
   footprints, scored separately; a pooled top-down success cannot hide
   oblique blur.
7. **Angular heldout:** the frozen Candidate-K arbitrary-direction/grazing gate
   at every required spatial level.
8. **Macro edge and lighting:** footprint-matched macro multiplication and the
   analytic NDF response from the primary Candidate-K audit.

Coverage, premultiplied-RGB, connected-region, and live visual limits remain
unchanged.  No RED gate authorises another level, tap, byte, stochastic sample,
or runtime threshold tweak.

## 8. Verdict

The nested-parent codec is a strict architectural improvement over two
unrelated scales: it lowers resident memory and proves exact continuity at
camera-centred LOD transitions without another read.  It should replace the
two-scale layout **if** its independently integrated anisotropic hierarchy
passes the gates above.  It is not, by itself, a proof against nearest spatial
shimmer, angular underfit, or oblique overblur, and it does not authorise shader
transcription before those measurements.

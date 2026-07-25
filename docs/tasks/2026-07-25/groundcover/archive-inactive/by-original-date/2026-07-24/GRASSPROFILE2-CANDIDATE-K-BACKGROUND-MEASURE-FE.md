# Candidate K: background-measure angular finite elements

Date: 2026-07-24  
Status: **RED / parked**.  The stored-node codec gate is GREEN, but the
corrected arbitrary-angle gate rejects the four-corner angular FE in unlimited
precision; runtime not authorised.  Binding measurements and the objective
resume condition are in
`GRASSPROFILE2-CANDIDATE-K-HELDOUT-RED-BLOCKER.md`.

## 1. Upstream correction

Candidate H already chose the real scene background depth and id for every
fractional ground-cover pixel.  A filtered grass record therefore does **not**
own a surface point.  Asking it to encode a representative depth and plane was
an unnecessary and damaging constraint: a spatial footprint can contain a
plume, several blades, empty space, and background at distinct depths, for
which no single grass surface is geometrically meaningful.

Candidate K keeps the roles disjoint:

```text
categorical near record: exact first surface, face plane, material class
filtered scale record:   unresolved foreground measure (A, A*C) only
background payload:      the actual visible background depth/id/radiance
```

Thus a fractional pixel resolves as

\[
C=A\,C_g+(1-A)C_b,
\]

while retaining the background depth.  No averaged grass depth is written,
used for HZB, or fed back into the near reconstruction.

Mode is determined by the categorical exact election, never by the numeric
filtered coverage.  An RGBA4 filtered record which decodes to `A=1` remains a
fractional/background-depth sample; only an eligible exact record may write
foreground grass depth.

## 2. Continuous angular finite-element field

Let `q` be the live ray's intersection with one botanical **middle reference
plane** `y=h_ref` in the authored growth frame and `d` its descending
direction.  At spatial footprint scale `sigma`, the cook
defines the positive filtered measure

\[
M_\sigma(q,d)=\big(A_\sigma(q,d),
                       P_\sigma(q,d)\big),\qquad
P_\sigma=A_\sigma C_\sigma.
\]

Colour is stored and interpolated premultiplied.  Conditional RGB is formed
only after the complete angular and footprint interpolation:

\[
C_g=P/\max(A,\epsilon).
\]

The angular domain uses sixteen periodic azimuth sectors and eight radial
elevation bands.  There are therefore exactly `16*8=128` cell pages.  The
preregistered non-pole node elevations are
`0.25, 2, 5, 10, 18, 30, 55, 75` degrees and the vertical pole is a singleton.
These knots deliberately spend most angular resolution in the grazing and
standing domain.  This first held-out gate measures that frozen lattice; a
future knot fit is authorised only if the unquantised basis is RED and must use
disjoint training and held-out directions.  A post-result hand-tuned lattice
is not accepted as evidence.

For an ordinary cell with local coordinates `(a,t)`, the four non-negative
bilinear weights are

\[
b_{00}=(1-a)(1-t),\quad b_{10}=a(1-t),\quad
b_{01}=(1-a)t,\quad b_{11}=at.
\]

The cap duplicates its pole value and reduces exactly to

\[
(1-t)((1-a)V_j+aV_{j+1})+tV_p.
\]

Every global angular vertex is quantised once.  Its exact bits are copied to
all incident pages.  A spatial phase index is also computed once by a
cell-independent rule.  Consequently adjacent cells have identical traces on
their shared boundary, the azimuth seam is exact, and the pole limit is unique.
Categorical angular selection has disappeared from the filtered path.  The
preregistered reference height is the source-bounds midpoint

\[
h_{ref}={y_{min}+y_{max}\over2}=0.49255296619984323\ {\rm m}.
\]

It is a geometric minimax coordinate, not a visibility-fitted statistic.  For
a surface point `X` observed along direction `d`, its field address is

\[
q_{xz}=X_{xz}-{X_y-h_{ref}\over d_y}d_{xz}.
\]

Consequently features at `h_ref` have zero angular parallax and the remaining
address shear is proportional to `X_y-h_ref`, rather than to the full drop from
the cover top.  If only the vertical support interval is known, this midpoint
uniquely minimises

\[
\max_{X_y\in[y_{min},y_{max}]}|X_y-h|,
\]

halving the worst lever arm relative to either bounding plane.  For an angular
slope change `Delta s`, the induced phase displacement is exactly
`|(X_y-h_ref) Delta s|`.  A visibility-weighted mean or median could improve an
average metric but would surrender this minimax guarantee; it is future fit
work, not a premise of this gate.  The old top plane `h_ref=topH` is measured as
an ablation under otherwise identical truth, filters, directions, and scores.

An exact horizontal ray from a camera above the cover box misses the box and
does not query the profile.  The stored horizon vertex is the one-sided
`elevation -> 0+` top-entry limit needed by arbitrarily shallow intersecting
rays; it is not derived from the old 15-degree row.

## 3. One-read direct codec

The filtered atlas uses `128x128` phase texels per angular cell and
`RGBA32Uint`.  One 128-bit texel contains eight shared-vertex codes:

```text
bits   0..63    sigma-1: four RGBA4444 premultiplied vertices
bits  64..127   sigma-2: four RGBA4444 premultiplied vertices

per vertex:
  R,G,B = four-bit UNORM premultiplied linear RGB
  A     = four-bit UNORM coverage
```

This is a direct code, not a palette or VQ index.  It has no dependent
codebook reads and naturally represents violet/green mixtures.  Component
quantisation error is at most `1/30`; interpolation cannot amplify that bound.

Exact resident accounting is

```text
near:  65 * 256^2 * 4 B       = 16.250 MiB
scale: 128 * 128^2 * 16 B     = 32.000 MiB
total                           = 48.250 MiB
```

This is `0.500 MiB` below Candidate H.  The runtime still performs

```text
4 R0 + 4 R1 + 1 scale = 9 profile texture operations.
```

The scale operation moves one coherent sixteen-byte texel.  Compared with the
active pre-H path, eight near reads become four bytes each and the old
four-contributor colour operation is removed, so worst-case profile bytes fall
rather than rise.  No binding, pass, dispatch, barrier, loop, march, candidate
list, runtime geometry, or per-species query is introduced.

## 4. Footprint without a fake grass plane

The two stored spatial measures are blended by a continuous footprint
coordinate.  Because fractional depth is the background depth, this coordinate
must not depend on a representative grass surface.  It is obtained from the
analytic screen differential of the live ray's fixed middle-reference address
`q`.

For screen coordinate `s`, camera `C`, normalized authored direction `d(s)`,
and reference plane `y=h_ref`,

\[
t_R=(h_{ref}-C_y)/d_y,\qquad q=C+t_Rd,
\]

and

\[
\partial_k q=t_R\left(\partial_kd-
 d\,{\partial_kd_y\over d_y}\right).
\]

The largest singular value of the two columns after conversion to source
texels is the conservative footprint radius `rho`.  It is continuous for every
descending exterior ray, grows naturally toward grazing, and is defined even
when all four near records MISS.  It has no eligibility switch and no
camera-distance threshold.  Continuous log-footprint partition weights blend
the exact state, `sigma-1`, and `sigma-2` moments.  Independent truth at
intermediate radii is a binding gate; two endpoints alone do not prove the
scale interpolation.

The existing macroscopic community/control weight `m` is evaluated at the
surviving background/root surface and participates as another premultiplied
measure:

\[
A_{final}=mA,\qquad P_{final}=mP.
\]

This is required at finite cover boundaries.  Applying only a binary atlas
selection after reconstruction would turn a correctly filtered community into
a clipped floating sheet.  Candidate K adds no control-field read; it reuses
the already required community weight.

Affine wind is applied by conjugation, not by deforming the reconstructed
answer afterward.  For authored deformation `W(X)=B X+b`, transform the live
oriented line by `W^{-1}`, query the static field in that frame, and transform
the exact near result by `W`.  The filtered measure itself is unchanged while
its analytic orientation axis becomes `normalize(Bu)`.  Height-proportional
gust shear is affine and therefore fits this identity exactly; per-blade
flutter remains outside the approved wind tier.

## 5. Statistical lighting without a sampled fake normal

The fractional field is an unresolved distribution, not one face.  It is
therefore shaded by a fixed analytic orientation distribution rather than by a
representative plane normal.  Let `u` be the wind-sheared authored growth axis.

For randomly oriented ribbon/cylinder faces around `u` and a unit light vector
`l`, the double-sided Lambert expectation is

\[
D_{ring}(l)={2\over\pi}\sqrt{1-(u\cdot l)^2}.
\]

For approximately isotropic plume filaments the corresponding double-sided
spherical expectation is

\[
D_{plume}(l)=1/2.
\]

A compact axisymmetric orientation family is

\[
D(l)=\eta\,|u\cdot l|+(1-\eta)D_{ring}(l),
\]

with community-global or continuously colour-conditioned `eta`.  A continuous
plume fraction may be inferred from conditional linear RGB by a frozen
cook-derived linear separator; this is a fitted lighting response, never a
geometry normal.  The existing shadow visibility, sun radiance, and ambient
terms are reused exactly once.  This response is world-oriented,
direction-continuous, and cannot cancel like interpolated view-facing normals.
It never enters Candidate F's exact face-plane solve.  The response is gated
against rendered filtered truth over fixed light directions and plume/stem
subsets before implementation.

## 6. Binding offline gates

The gate order separates codec, angular basis, and runtime concerns:

1. At every stored angular vertex and both scales, compare the unquantised
   `128x128` area-moment field, then RGBA4444, against the accepted `256x256`
   filtered truth.  Report every direction separately.
2. Raycast untouched directions at every cell centre, quarter/Gauss sites,
   dense random sites, `+/-epsilon` around every boundary, grazing
   `0.25/1/2/5/10/15` degrees, and the pole.  First test unlimited-precision
   shared vertices; quantisation cannot rescue a failed basis.
3. Test intermediate footprint radii and continuous camera-distance sweeps.
4. Test one-to-4.5-mm camera translations, cover edges, uphill views, sparse
   plume/stem subsets, and angular events wholly inside a cell.
5. Gate direct-light response over fixed sun directions and plume/stem subsets.

Primary limits remain

```text
coverage absolute error:       p95 <= .08, p99 <= .20
premul RGB max-channel error:   p95 <= .06, p99 <= .15
connected p99 exceedance:       < 1%
```

The old representative-position limit is deliberately absent: Candidate K
does not reconstruct, store, or consume a fractional grass position.  Near
exact geometry retains its existing world-position gates.  The live visual
gate remains the user's standing/uphill/edge flight review, with defects
3--11 and especially the 2--5 m particle stream evaluated before secondary
work.

## 7. Claim boundary

The topology, continuity, packing, memory, and fixed runtime cost are proved.
No finite angular FE can guarantee fidelity for every possible triangle soup:
an arbitrarily narrow visibility event may lie wholly inside one angular cell.
Candidate K accepts an arbitrary finite triangle soup as cook input and makes
no species-specific runtime assumption, but each cooked community must pass
the held-out angular/scale gate.  A RED unquantised FE result parks this route;
it does not authorize more taps, more bytes, stochastic selection, or a hidden
angle clamp.

## 8. Stored-node codec result

The direct `RGBA4444` codec passed the preregistered stored-direction gate for
every existing directional truth field and both spatial scales.  This is a
codec result, not yet an angular-interpolation result.  The authoritative
complete report and machine-indexed QA are

```text
data/work/groundcover-candidate-k-direct-fe/
  e3e0a4175b151b89/6e8ba84eda853dab/
    report.json
    qa/index.json
```

All four page/resolution allocations were GREEN.  The full 256-square
downsample-plus-reconstruction **worst per-direction** values were:

```text
allocation  MiB      sigma   |dA| p95/p99       max|dP| p95/p99
64 @ 180    31.6406     4     .0667 / .0863      .0495 / .0604
                        16     .0353 / .0431      .0357 / .0390
80 @ 160    31.2500     4     .0706 / .0902      .0511 / .0627
                        16     .0353 / .0431      .0360 / .0398
96 @ 144    30.3750     4     .0745 / .0980      .0540 / .0709
                        16     .0392 / .0471      .0367 / .0409
128 @ 128   32.0000     4     .0706 / .0902      .0543 / .0682
                        16     .0392 / .0431      .0368 / .0397
```

Every allocation also passed p99, connected-region, and every-direction
checks.  Therefore four-bit direct premultiplied appearance is sufficient at
the stored nodes; a palette, VQ table, filtered grass depth, or extra read is
not justified.

The held-out attempt uses `128 @ 128`: eight angular bands at exactly
`32.000 MiB`.  The other three GREEN layouts are retained as a measured
spatial/angular Pareto table, not silently substituted after the held-out
result.  A lower-page layout can be reconsidered only through its own frozen
arbitrary-angle gate.

## 9. Horizon-complete angular domain

The hemisphere topology must include the one-sided grazing limit.  Merely
clamping all elevations below a positive first ring would violate the exterior
view contract while appearing numerically continuous.

Let `e in (0,pi/2]` be downward elevation above the horizon.  The angular FE
has `R` circular vertices plus one pole.  Its first circular vertex is the
finite operational grazing sample at `0.25` degrees.  A literal horizontal ray
from a camera above the cover box misses the box and never queries this field.
For intersecting rays in `0 < e < 0.25` degrees, the first vertex is used as a
one-sided limit only if the progressively shallower truth fields demonstrate
that coverage and premultiplied radiance have converged within the frozen
fidelity tolerances.  Otherwise the basis is RED; the old 15-degree clamp may
not return under a different name.

This limit is an empirical field limit, not an assumed transparent value.  In
an infinitely periodic open community, interception can tend to one as
`e -> 0+` even though the exactly horizontal ray never enters the finite cover
slab.  The cook estimates the limit from progressively shallower truth
directions and must report convergence independently for coverage and
premultiplied radiance.

## 10. Frozen arbitrary-angle and ring-selection gate

The second gate prevents both trivial constant-field continuity and
stored-node overfitting.  It compares the unquantised FE and then the packed FE
against freshly raycast truth at directions which are not angular vertices.

The truth set is frozen before fitting:

```text
grazing elevations:  0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5 degrees
interior elevations: every candidate-band midpoint and its 1/4, 3/4 sites
pole elevations:     80, 84, 87, 89, 89.75 degrees
azimuth offsets:     1/4, 1/2, 3/4 of every 22.5-degree sector
adversarial sites:   +/-epsilon around every eventual ring and seam
```

Training and held-out directions are disjoint.  Ring elevations and the
reference-plane height are fitted only on the training subset.  The complete
held-out set is evaluated once after those parameters are frozen.  The same
coverage, premultiplied-RGB, connected-region, temporal-translation, and
plume/stem subset limits from section 6 remain binding.  Results are reported
per elevation family, so a pole success cannot average away a grazing failure.

The frozen attempt uses the explicit eight knots in section 2.  Quantisation is
applied only after the basis passes; it cannot conceal angular underfit.  No
failed attempt may gain another ring, read, texture, or byte.  A RED
unquantised result first triggers the premise audit (coordinates, truth, and
top-plane ablation); if those are sound, the fixed-knot basis is parked.  A
later minimax knot fit must be preregistered as a distinct attempt and cannot
reuse the held-out directions as training data.

## 11. Macro cover mask: the only admissible multiplication

Let `m(r) in [0,1]` be the ecological cover field evaluated at a botanical
root coordinate and let `H(p,d)` be the binary micro-cover hit for a ray in a
normalised pixel footprint `K`.  The exact masked unresolved measure is

\[
M_m=\int K(p)H(p,d)m(r(p,d))\,(1,C(p,d))\,dp.
\]

It does **not** generally factor into `m(q) M`: a hard macro boundary may be
correlated with the roots which supply the visible micro hits.  Candidate K
therefore uses two different, explicit semantics:

- the exact branch applies the already-defined categorical predicate at the
  elected plant root;
- the unresolved branch multiplies both stored moments by the same
  footprint-filtered macro value `m_bar`:

\[
(A',P')=(\bar m A,\bar m P).
\]

This preserves `0 <= P'_c <= A'`, leaves conditional grass colour unchanged,
and cannot create an opaque sheet.  Multiplying conditional colour after the
division, or evaluating a binary mask only at the terrain endpoint, is
forbidden.  The latter is the old vertically extruded flat-edge defect.

The unresolved factorisation is a scale-separation approximation.  If `m` is
`L_m`-Lipschitz over the footprint and every contributing root lies within
radius `r_K` of its centre, then, componentwise (authored colour is in
`[0,1]`),

\[
|M_m-\bar m M| \le 2L_m r_K.
\]

The factor two is the diameter bound for a footprint-mean `m_bar`; using the
centre value instead gives `L_m r_K` but is less well antialiased.  Thus the
macro field must be filtered with the same live footprint and its
transition width must be large relative to unresolved `r_K`.  A hard boundary
has no such bound.  At hard cover edges, resolved root-categorical geometry is
the only source of a crisp botanical side silhouette; the unresolved branch
must fade rather than pretend its statistical measure owns a side surface.
The live edge review remains binding.

Overlapping species inside one cooked community are already part of `H` and
cost no per-species runtime work.  Arbitrarily mixing two independently cooked
communities at a hard runtime control boundary is a different problem: one
scale read cannot, in general, return both community measures.  It requires a
precompiled mixed community or a separately authorised multi-carrier scheme;
Candidate K must not claim that ordinary scalar mask multiplication solves it.

## 12. Analytic orientation-distribution lighting

The unresolved measure has no single face normal.  Its lighting object is an
orientation distribution conditioned on the viewing direction.  Let `v` be
the unit direction from the cover toward the camera, `l` the unit direction to
the light, and `u` the unit growth axis.  Use a non-negative mixture of axial
faces (`w_a`), azimuthally uniform ribbon-side faces (`w_r`), and isotropic
plume fibres (`w_p`).

Define

\[
F(c)=\sqrt{1-c^2}+c\arcsin c,\qquad 0\le c\le1,
\]

\[
a_v=\sqrt{1-(u\cdot v)^2},\quad
a_l=\sqrt{1-(u\cdot l)^2},\quad c_\gamma=|v\cdot l|,
\]

and, when `a_v a_l > 0`,

\[
c_\delta={|(v-(u\cdot v)u)\cdot(l-(u\cdot l)u)|\over a_v a_l};
\]

otherwise set the ribbon numerator below to zero.  The projected-area
denominators and double-sided Lambert numerators are

\[
Z_a=|u\cdot v|,\qquad N_a=|u\cdot v|\,|u\cdot l|,
\]

\[
Z_r={2a_v\over\pi},\qquad
N_r={a_v a_l\over\pi}F(c_\delta),
\]

\[
Z_p={1\over2},\qquad
N_p={2\over3\pi}F(c_\gamma).
\]

These are closed-form integrals of `|n dot v|` and
`|n dot v||n dot l|` over the three canonical distributions.  The visible
unresolved direct-light response is therefore

\[
D(v,l)={w_aN_a+w_rN_r+w_pN_p\over
              w_aZ_a+w_rZ_r+w_pZ_p}.
\]

This replaces the earlier unconditioned `N dot L` surrogate.  It is continuous
at the pole when evaluated by the limiting formulas and remains finite if the
fitted distribution gives non-zero measure to at least one visible family.
It returns a scalar expected response, not a fake normal; it must never enter
the exact face-plane solve.

The cook fits community-global base weights and, only if the semantic gate
requires it, a continuous plume fraction from conditional authored colour.
Plume and stem subsets plus fixed camera/light pairs gate that fit.  The exact
`F` is the mathematical reference.  A runtime transcription may replace it
with one fixed minimax polynomial only after reporting its maximum response
error; this changes ALU but no read, binding, or resident byte.  Shadow
visibility and ambient terms remain the already computed background terms, so
the filtered-lighting result is explicitly an unresolved approximation rather
than a claim of multisample shadow exactness.

A two-million-sample independent sphere/ring quadrature checked the closed
forms at relative angles `0`, `0.3`, `0.9`, and `pi/2`; the largest absolute
Monte-Carlo discrepancy was `2.12e-4` for the ring and `1.53e-4` for the
isotropic family, consistent with sampling noise.  This validates the written
integrals, not the eventual community weights or polynomial approximation.

## 13. Affine wind conjugation

The permitted wind is one spatially constant, time-varying shear per global
layer.  In authored coordinates with root plane `y=y_0`,

\[
S_\beta(x,y,z)=(x+\beta_x(t)(y-y_0),\ y,\
                 z+\beta_z(t)(y-y_0)).
\]

Its matrix is invertible with determinant one and fixes every root-plane
point.  Including the layer's fixed rotation/reflection `R` and translation
`r`, map authored points to world by

\[
X_w=r+R S_\beta(X_a).
\]

A world ray maps back to the authored straight ray

\[
o_a=S_\beta^{-1}R^{-1}(o_w-r),\qquad
d_a=S_\beta^{-1}R^{-1}d_w.
\]

Candidate K computes angular coordinates, the middle-plane address, and both
screen differentials from this same `(o_a,d_a)`.  Therefore central-ray
visibility is exactly conjugate under the shear and the footprint cannot lag
behind the wind.  Exact-branch geometric normals use the usual
inverse-transpose.  The unresolved NDF uses the wind-sheared growth axis and is
separately lighting-gated; it is not promoted to an exact transformed face
distribution.

Any `beta(x,z,t)` with non-zero spatial gradient is **not** this map: the
inverse of a straight ray becomes curved.  It is forbidden in the exact tier,
irrespective of smoothness.  Spatial richness comes from the two global layers
having different fixed transforms and time phases, not from a spatially
varying shear hidden inside the query.  Likewise smooth world-keyed tint may
remain an attribute modulation, but spatially varying height/vigor is not an
affine conjugation; it needs precompiled discrete states or its own bounded
visual approximation.

## 14. Terrain and the middle reference surface

Sections 2 and 13 are exact in one authored affine ground frame.  On an affine
terrain patch `G(x,z)=g_0+m dot (x,z)`, the world middle surface is
`y=G(x,z)+h_ref`.  With upward covector `n_g=(-m_x,1,-m_z)`, a known terrain
intersection `O` on the live ray gives the exact middle-surface parameter

\[
t_R=t_G+{h_{ref}\over n_g\cdot d},
\]

and its XZ point supplies the periodic phase.  This is the same reference in
the cook and query; no top-plane substitution is allowed.

For curved terrain, replacing `G` by its tangent at `O` has height remainder
at most `kappa r^2/2` over horizontal travel `r`, and corresponding ray error

\[
|\delta t|\lesssim {\kappa r^2\over2|n_g\cdot d|}.
\]

The bound exposes the real uphill/grazing risk instead of calling the tangent
construction exact.  Candidate K adds no terrain march or iterative solve.
Its held-out and live gates must therefore include rising terrain and shallow
incidence; a RED result there returns to the terrain/reference representation,
not to extra grass taps or a camera-relative top plane.

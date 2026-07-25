# Adversarial audit of the line-invariant transfer-field truth contract

**Date:** 2026-07-22  
**Scope:** pure mathematics and offline truth only; no runtime or shader change  
**Decision:** **blocked before codec fitting.** The dominant-axis line
coordinates are mathematically sound, but the current beam truth stores only
independent marginal distributions. It does not contain the coupled invariants
needed to exclude the observed camera-centred widening sectors, distance
stretch, wrong-perspective sheets, or shimmer.

Audited inputs:

- `GRASS-LINE-INVARIANT-TRANSFER-FIELD.md`, SHA-256
  `00be9e5043723e69ad1504accca3373eb20bad7c92fc59c44eb38eda8af5ff19`;
- beam manifest's bound version of
  `tools/groundcover-bake/generate-transfer-field-beam-truth.ts`, SHA-256
  `76b643e7651026de4ab0545ee9716e3f6b3d22179f130323783dd105df417b24`;
- current generator with subsequently added coherence records, SHA-256
  `b9ee08a7150a1d2d051d37295bf02ced2243b2044455d5a24ba013d8051c4dfe`;
- `tools/groundcover-bake/train_transfer_field_codec.py`, SHA-256
  `0906ac7c9ecbf759c9d4f843f2714ce08e1c7a179cd9c1291c6bdae07d6862eb`;
- beam-truth manifest recipe
  `81f45e1c6282e07e37a4dac637e6b5571bb4a3a7b79f86bce911a0d1b6c46fd3`,
  manifest SHA-256
  `ac86eddb28dba015fe57b1fc14ac9dcc10dd25481617460c40d41efa4a2c0d94`.

## 1. Mathematics which passes this audit

### 1.1 Line coordinate

For one signed dominant-axis chart, define

\[
Q=A+d\frac{k_j-A_j}{d_j},\qquad
r=(d_a/d_j,d_b/d_j),\qquad
\eta=A_j-k_j.
\]

Then `Q` and `r` are unchanged under the mere reparameterisation
`A -> A+lambda d`. Since `|d_j| >= 1/sqrt(3)`, no direction denominator tends
to zero inside the selected chart. The stated sheared action of an XZ lattice
translation on `Q` is also correct. This is a valid exterior oriented-line
coordinate.

`eta` is intentionally **not** invariant when the physical origin moves. It is
the missing successor coordinate. Consequently the direct suffix field
`F(A mod Lambda, A_y, d)` has the correct dimensionality for arbitrary
camera-inside starts.

### 1.2 Horizontal endpoint

The endpoint rule treats `d_y=0` as a finite `L_max=155 m` query. The analytic
truth tracer likewise leaves an exact horizontal ray in the slab for the
declared horizon. No five-degree clamp or live `1/d_y` pole appears in this
part of the proposal.

### 1.3 Offline opaque overlap

Tracing the actual union of all marked species before filtering is the correct
fixed-runtime-cost overlap rule. If the union and marks are truly present in
the bake, repeated copy count and species count do not enter runtime cost.

These three results do not validate the current beam record or decoder gate.

## 2. Decisive blocker: marginals do not determine a coherent image field

`encodeBeam` sorts the hits of each beam independently and stores coverage plus
three depth/colour/normal marginal strata. It discards:

- micro-ray identity;
- source owner/species identity;
- correspondence between neighbouring beam centres;
- correspondence between the same line at two origin phases;
- correspondence across dominant-axis chart seams and camera frames.

Therefore the truth cannot test pinhole or camera-path integrability. A simple
counterexample is two neighbouring beam centres with the identical marginal
distribution

\[
\tfrac12\delta_{(t_1,red)}+\tfrac12\delta_{(t_2,green)}.
\]

One underlying field may connect red-to-red and green-to-green across the two
pixels; another may swap the owners. Their current truth records are identical,
but a fixed marginal quantile produces different surface connectivity and can
alternate colours/depths. No per-record coverage, depth, colour, normal, or
variance metric distinguishes them.

For an opaque visibility buffer the required decoded object is not merely a
marginal distribution `nu_ell`. It is a **coupled transport map**

\[
\boxed{G(\ell,\eta,\xi)=\text{one marked event or miss}}
\]

whose fixed sample label `xi` follows one coherent micro-ray/event sheet across
line, origin, chart, and camera changes. Three independently depth-sorted means
do not establish such a map. A codec can reproduce the present truth exactly
and still create the reported widening triangles, wrong-view sheets, or a
million-particle shimmer.

## 3. Same-line invariance and right-censoring remain untested by the gate

The design correctly demands that, for fixed `(ell,xi)`, advancing the camera
without crossing the selected interaction preserves the same world hit, and
crossing it advances only to a later event. The preserved manifest contains no
paired queries of that form. Every train/validation centre is an independent
hash sample at one height. It stores neither `ell`, `eta`, owner, world hit, nor
a pairing id.

The current generator was changed after that manifest to add 512 central-line
origin-shift pairs and an f64 algebraic top-entry identity check. Those are
useful additions but do not close the contract:

1. each paired beam is still independently depth-sorted and reduced to marginal
   means; micro-ray/event identity is absent;
2. the shifted beam is rebuilt around the shifted **central** phase, rather
   than transporting every perturbed micro-line (Section 4);
3. `coherence_metrics` compares the codec residual to the already changed truth
   marginal residual. It does not require the same marked world event;
4. the `noncrossing` mask checks three mean depths against the shift, not whether
   every selected micro-event is un-crossed;
5. neither coherence metric nor the algebraic entry metric participates in
   `representation_pass`, `codec_pass`, or the final accept/reject expression.

Thus even the amended fitter can report `accept` while failing its coherence
diagnostics. The f64 entry check proves the already-derived coordinate identity,
not the fitted field or stochastic selector.

Thus all of the following wrong decoders can pass the active aggregate gate:

- one which shifts every depth smoothly with origin height;
- one which changes the selected owner at every height row;
- one which returns an earlier, now-behind-camera event;
- one whose hash is accidentally based on `(A_x mod P_x,A_z mod P_z)` and
  therefore changes while the camera advances along one line.

The executable test must contain ordered origins `eta_0<...<eta_n` on the same
oriented line, a fixed sample label, and exact marked world events. Before the
event is crossed,

\[
P(\ell,\eta_i,\xi)=P(\ell,\eta_{i+1},\xi).
\]

After crossing, event order must increase. A mere comparison of local distance
is insufficient because the correct local distance changes as
`t_{i+1}=t_i-lambda` while the world point remains fixed.

Exterior datum invariance is also absent. All generated origins have height
fractions in `[0.01,0.99]`; there are no above-top/below-bottom origins and no
two exterior datums re-anchored to the same algebraic slab entry.

## 4. The beam kernel itself is not transported along an oriented line

At each independent origin the generator adds an XZ phase disk at fixed height
and an angular tangent disk. Consider one micro-ray `(A_i,d_i)`. Moving the
central origin by a vertical/profile increment `Delta y` while preserving the
same micro-line requires

\[
A_i'=A_i+d_i\frac{\Delta y}{d_{i,y}}
\]

when `d_{i,y}` is nonzero, or the corresponding dominant-axis chart transport
at grazing. Reusing the same XZ phase offset at the new height does not preserve
that micro-line. Its carrier phase must shear by the micro-direction, not the
central direction.

The current truth has no paired origin construction at all, and `microRay`
would reconstruct a fresh disk if called at another height. Consequently the
filtered footprint may change with camera origin/range even if a fitted codec
has zero per-record error. This is exactly the class of error which can appear
as distance-dependent stretching.

The declared half-texel phase radius and fixed `0.05 degree` angular radius are
also one arbitrary footprint. No actual pinhole Jacobian relates them to range,
surface slope, DPR, or mip. One footprint cannot test widening with distance or
mip continuity.

## 5. Full-sphere and chart-seam acceptance is not implemented

Section 10 requires held-out exact `0/0.1/1/5/15/35/75/90` degree directions.
The generator instead places every one of those rows in **training**. Validation
contains only `0.05/0.5/2/10/25/55/82.5` degrees. Exact horizontal and exact
vertical are therefore absent from held-out metrics; QA images are not a
held-out numerical gate.

There is no equivalent-line pair encoded through two dominant-axis charts and
no sheared-periodic overlap pair. The convenient `(A mod Lambda,y,d)` record
does not by itself exercise the chart/hash rule claimed in the math.

There is an additional finite-quadrature seam in `tangentFrame`: its helper
switches discontinuously at `abs(d_y)=0.9` (about `64.16 degrees`). A continuous
isotropic disk would be independent of tangent-frame rotation, but a deterministic
16-point disk is not exactly rotationally symmetric. Neither training nor
validation brackets this artificial seam densely. A codec can learn a ring
discontinuity there without failing the manifest.

The horizontal 155 m cap is mathematically bounded, but horizon continuity is
not tested. A hit just beyond 155 m becomes a miss discontinuously; any change
to world grass range needs a separately baked field as the document says.

## 6. Three stored strata are not actually equal-mass

The runtime proposal chooses each of three conditional strata with probability
`1/3`, but `encodeBeam` partitions a finite number `h` of hits with

\[
q(i)=\left\lfloor\frac{3i}{h}\right\rfloor.
\]

Unless `h` is divisible by three, the groups do not have equal mass. Empty
groups are filled by duplicating another event. Examples:

- `h=2`: the two real hits have probability `1/2` each, while the stored
  equal-third decoder gives one hit `1/3` and duplicates the other to `2/3`;
- `h=4`: group sizes are `2,1,1`, but the decoder assigns `1/3` to each group,
  changing per-event weights from quarters to `1/6,1/6,1/3,1/3`.

The decoder must either store the actual conditional stratum masses or define
three fixed inverse-CDF sample locations/intervals whose runtime probabilities
match the bake. Until then, even a perfect codec is biased relative to its own
micro-ray distribution.

## 7. Marks, moss, and transfer composition are not in this truth

The active source contains only Calamagrostis. `SurfaceEvent` stores depth,
colour, and normal, not triangle owner, botanical class, species/material mark,
roughness, transmission, or an exact micro-sample id. Within a stratum,
`encodeBeam` arithmetically averages colours and normals from unrelated owners.
It therefore does not emit the document's claimed “one selected visibility
event”; it can invent a colour/material between a violet panicle, green blade,
flower, and moss event. This can specifically hide the fuzzy coloured heads
while aggregate RGB error remains modest.

The tracer also returns only the nearest opaque polygon event. It does not
compose an ordered extinction/transfer record, so the associative monoid and
translucent moss/hair case in Sections 4--5 remain mathematical proposals, not
properties of this truth artifact.

Offline opaque union remains compatible in principle. It is not tested until a
marked multi-species/moss community is traced and the selected output retains
one mark/material event or an explicitly validated marked transfer mixture.

## 8. Other acceptance items missing from the manifest

The document itself calls for these gates, but the beam artifact does not
contain them:

- scene-depth-straddle measurements;
- multiple footprint/mip levels and continuity between them;
- camera-path sequences;
- connected angular-sector/fan width as a function of range;
- terrain-slope variants;
- a no-shimmer sequence using the actual `xi` construction;
- exact world-hit/owner agreement before and after origin shifts;
- chart-overlap equality.

Random held-out ray aggregates cannot substitute for these structured tests.
In particular, a narrow but connected wrong-angular sector can have a small
global error rate while expanding to width

\[
w(r)=2r\tan(\Delta\theta/2)
\]

and dominate the image at range—the exact observed failure.

## 9. The proposed spatial field is more than 91.9% unsupervised

The rank-4 default allocates a `192 x 64 x 192` sparse spatial embedding:

\[
N_v=2,359,296\text{ spatial texels},\qquad4N_v=9,437,184
\text{ learned spatial coefficients}.
\]

There are only 23,744 train beams. One trilinear beam touches at most eight
spatial texels, so even with no overlap the training set can update at most

\[
8\cdot23,744=189,952\text{ texels}=8.051\%.
\]

At least `2,169,344` texels (`91.949%`) therefore remain at random
initialisation. The actual unseen fraction is larger because training supports
overlap. SparseAdam updates only looked-up rows; it supplies no smoothness or
field equation which could determine the unobserved rows.

The same fact appears as a parameter count: rank four has 9.44 million spatial
unknowns while all 24 outputs of all train beams provide only 569,856 scalar
observations, an optimistic 16.56-to-1 unknown/observation ratio before the
directional factors and decoder are counted. Nonlinearity does not manufacture
missing constraints.

QA is especially misleading here. Its 8,192 beam centres touch at most 65,536
spatial texels, most of which need not have appeared in training. Validation
may reject random unseen rows, but if the shared bias/angular factors happen to
clear aggregate thresholds, the report still labels an overwhelmingly
unidentified 43 MB field as accepted. The gate must record exact trained-texel
support and reject any runtime/mip texel outside a certified interpolation
support. A much coarser trained field or a true continuous regularised basis is
required before allocating the 192-grid quality ceiling.

This also invalidates the emitted mip chain. `write_mips` blindly averages the
mostly untrained coefficient volume. More fundamentally, a linear average of
`U_r` followed by sigmoid/clamp/normalisation is not the transfer distribution
baked for a larger footprint. It contradicts the math document's requirement
that every mip be baked for its actual micro-ray kernel, and can create a
distance-locked level or stretch even if level zero fitted.

## 10. Octahedral angular edges are clamped instead of identified

The full sphere is mapped to a square by octahedral encoding. The square
boundary is a quotient seam: appropriate boundary edges represent adjacent or
identical spherical directions with reversed/folded edge coordinates. The
fitter instead clamps bilinear indices independently to `[0,angular-1]`.
Coefficients on identified sides are unrelated and no seam loss, shared gutter,
or edge-reversal constraint is applied.

Consequently a continuous camera direction crossing an octahedral boundary can
sample unrelated angular coefficients. This is another direct route to a
camera-centred sector or seam despite excellent random held-out averages.
Clamping is a boundary condition, not the required spherical topology.

The angular field needs either a seam-safe spherical basis or explicit
octahedral edge/corner identification in both sampling and training, plus
paired directions immediately across every identified edge. The current
`tangentFrame` seam discussed in Section 5 is independent and also remains.

## 11. Required truth contract before codec attempt A

The beam artifact is useful as marginal transfer supervision, but it is not an
acceptance truth. Before fitting the rank-4 codec, add a small decisive coupled
contract:

1. **same-line origin sequences:** fixed `ell`, fixed micro-sample labels,
   ordered `eta`, exact marked world hits, and right-censoring assertions;
2. **pinhole bundles:** regular neighbouring rays at 4, 20, and 155 m over flat
   and sloped root charts, preserving micro-sample identity across pixels and
   camera frames;
3. **angular sweeps:** dense directions across 0 degrees, the dominant-chart
   seams, `abs(d_y)=0.9`, and 90 degrees, with equivalent chart encodings;
4. **true held-out anchors:** exact 0/0.1/1/5/15/35/75/90 rows absent from the
   fitting split;
5. **footprint ladder:** several screen-derived kernels/mips and their
   transitions, not one fixed phase/angular disk;
6. **marked community truth:** at least Calamagrostis plus one differently
   coloured/height-distributed cover and one moss geometry/material, retaining
   selected mark and micro-event identity;
7. **correct stratum probabilities** and the executable frame-invariant,
   line-anchored `xi` mapping;
8. **scene-depth probes** and connected fan/stretch/shimmer metrics;
9. **coefficient-support accounting:** no runtime or mip texel may remain at
   random initialisation or outside a declared interpolation/regularisation
   proof;
10. **spherical seam constraints:** identified octahedral edges/corners and
    paired across-seam directions must decode the same field.

Only a decoder which reproduces this coupled map can claim to address the
current geometric artifacts. If the representation remains a collection of
independent marginal moments, its defensible role is a radiance-like distant
appearance field, not crisp opaque near-field ground-cover geometry.

# Box-boundary transfer mathematics for arbitrary triangle-soup ground cover

**Date:** 2026-07-23
**Scope:** pure mathematics only; no runtime, shader, renderer, or test change
**Status:** **the common-slab carrier plus boundary-transfer factorization is
mathematically complete as a continuous oracle for arbitrary opaque triangle
soup after declared physical filtering, but no finite production encoding has
passed.** The carrier must be the exact parallel-extrusion field
`U=P x I`, not an unpointed line lookup and not the original collection of
finite-height AABBs. Its 3D first-passage field retains the gap origin's
horizontal phase and skips arbitrarily many footprint gaps in one query. At
the first carrier boundary, the rest is one 4D whole-community transfer.

The construction is exact as a continuous oracle. A finite representation
exists for the **physically filtered output contract** defined in Section 4,
but fitting that representation under the fixed 250 MB/read envelope is an
actual-community cook gate, not a soup-independent theorem. It is not bit-exact
first-triangle visibility for an unrestricted continuum soup. Deliberately
enlarged guarded fade boxes make the filtered bandwidth finite. Resolved
visibility cells remain categorical and reconstruct their true hit plane;
only final sub-pixel mixed cells become filtered appearance records.  The
2026-07-23 actual-source gate confirmed the periodic identity but rejected the
proposed finite macrobrick realization; see
`GRASS-BOX-BOUNDARY-TRANSFER-CPU-GATE.md` and Section 11.

## 1. Corrected product domain

Let `Lambda` be the horizontal period lattice. In one fundamental tile let

\[
B_1,\ldots,B_A\subset\mathbb R^3
\]

be tight plant bounding boxes, and repeat them by `lambda in Lambda`. Boxes
may belong to different species and may overlap. These are first enlarged and
assigned to a fixed number of common-slab carrier lanes (Section 3). Let `U`
denote the resulting periodic **carrier union**, not the original tight union.
For one lane it is

\[
U=P\times I.
\]

The marked triangle soup `G` is arbitrary except for
the authoring containment condition `G subset U`. Its marks include colour,
normal, material/species identity, and every attribute which must follow the
winning surface categorically.

The corrected high-quality camera domain is

\[
\boxed{\mathcal Q_{\rm box-ext}
 =\{(o,d):o\notin U,\ d\in S^2\}.}
\]

Thus a lane may fade whenever the camera is in its deliberately enlarged
carrier box/union, including empty space between leaves. An origin in a
genuine horizontal gap of `P` or outside `I` remains exterior and must render
correctly. Section 3.3 states exactly how much broader this is than tight
per-plant boxes; that broader fade domain is the price of the invariant.

For a finite horizon `R`, the desired answer is the first marked triangle
event

\[
V_G(o,d)=\min\{t\in(0,R]:o+td\in G\},
\]

or categorical `MISS`. Ties use one fixed total order on
`(distance, layer, box, triangle, periodic-copy-offset)`; this makes owner and
mark deterministic without blending them.

## 2. The exact factorization that the relaxation enables

For `o notin U`, define the first future box-entry distance

\[
t_U(o,d)=\inf\{t>0:o+td\in U\}.
\]

If the set is empty within `R`, every triangle is also missed because
`G subset U`. Otherwise let

\[
q=o+t_U(o,d)d\in\partial U.
\]

Define the boundary-conditioned whole-community transfer

\[
T(q,d)=\min\{\tau\ge 0:q+\tau d\in G\}.
\]

Crucially, `T` is not restricted to the box whose boundary contains `q`. It
continues through a miss in that box and returns the first actual triangle in
any later periodic box. Consequently

\[
\boxed{
V_G(o,d)=
\begin{cases}
 t_U(o,d)+T(q,d),&t_U+T\le R,\\
 {\rm MISS},&\text{otherwise.}
\end{cases}}
\]

This identity is exact. No successor rank is needed *inside `T`*: `q` is the
canonical start of the forward half-line. The remaining-horizon dependence
does not add a texture coordinate; compare the returned total distance with
`R` live.

Modulo `Lambda`, a regular point of `partial U` has two continuous boundary
coordinates. Direction contributes two. Hence `T` is a four-dimensional
piecewise-analytic field (plus finite face/component charts). This is a real
dimensional reduction relative to the five-dimensional arbitrary pointed-ray
field.

### 2.1 Gaps between boxes are handled semantically

Two origins in different gaps generally obtain different boundary points
`q`. Once those points are known, the same transfer definition is valid for
both. If the first box contains no triangle on the ray, `T` already sees
through it. Therefore the proposed *semantics* do not make the common mistake
of returning `MISS` merely because the first box was empty.

The problem lies in obtaining `q` under the frozen resource contract.

## 3. Common-slab extrusion removes the box-entry obstruction

The carrier is not the literal union of finite-height AABBs from Section 1.
For one carrier lane, enlarge every assigned plant box vertically to one
canonical interval

\[
I=[h^-,h^+]
\]

and union their guarded horizontal footprints into a periodic mask `P`. The
lane carrier is

\[
\boxed{U_P=P\times I.}
\]

This deliberate authoring change supplies the exact parallel-extrusion
invariant. It is the necessary part of the proposal, not an implementation
detail.

Clip the live ray `r(t)=o+td` to the slab `I`, obtaining the forward interval
`[t_a,t_b]`. If it is empty, the lane misses. Put

\[
s=\|d_{xz}\|,
\qquad q_a=o_{xz}+t_a d_{xz}.
\]

For `s>0`, let `omega=d_xz/s` and define the exact periodic first-passage
field

\[
\rho_P(q,\omega)
=\inf\{\ell\ge0:q+\ell\omega\in P\}.
\]

At an oriented outgoing boundary, membership is half-open: the just-exited
component is treated as outside, and `rho_P^+` denotes the first **strictly
later** entry. This prevents an exact boundary query from returning the same
component at distance zero. The cook and decoder use the same deterministic
normal/axis tie rule.

One field query gives

\[
\boxed{t_U=t_a+\rho_P(q_a,\omega)/s,}
\]

accepted only when `t_U<=t_b`; otherwise the lane misses. This is the exact
Sannikov/class-E extrusion lift. It jumps across an arbitrary number of empty
periodic footprint gaps because that first-passage work is in `rho_P`, not
enumerated live.

The field address has three dimensions `(q_x,q_z,omega)`. It is pointed in
the 2D mask plane: the gap origin's phase is retained by `q_a`. Axial
translation inside the common slab contributes no independent information,
which is exactly the symmetry that removes the other two generic pointed-ray
coordinates.

### 3.1 Why the former two-box counterexample does not apply

For footprints `[1,2]` and `[3,4]` on a horizontal line, origins at `x=0` and
`x=2.5` produce different `q_a`. Therefore they read different points of
`rho_P` and correctly return entries `x=1` and `x=3`. They share an oriented
line but **not** a carrier address. The counterexample defeats an unpointed
line lookup; it does not defeat the specified pointed 2D extrusion field.

This correction is material. The first-box stage is an exact 3D oracle, not a
5D visibility field and not a candidate list. Its finite storage is not free:
an arbitrary footprint needs a categorical first-passage atlas (including
boundary component/copy information) or an equally complete authored analytic
invariant. Those bytes and any correction structure are charged in Section 8.

### 3.2 Categorical pole and cap cases

If `s=0`, the ray is vertical. Query occupancy at `q_a`:

- `q_a in P`: the first carrier boundary is the appropriate slab cap at
  `t_a`;
- `q_a notin P`: the lane misses.

No epsilon or division by horizontal speed is used. If `q_a in P` for the
generic case, `rho=0` and the entry is likewise the slab cap. Otherwise a
positive `rho` denotes a side-boundary entry. Exact horizontal rays have
`s=1`; when their height lies in `I`, they are the best-conditioned generic
case.

### 3.3 Visual cost of the common slab

The carrier is larger than the tight per-plant boxes. Its excluded/fade
domain is every `(x,z,y)` with `(x,z) in P` and `y in I`, even where a short
plant's tight box contains no geometry. This buys the invariant at a visible
cost:

- a camera above a short plant but below `h+` fades that lane while its
  horizontal position lies in the footprint;
- overlapping guarded footprints form one carrier occupancy region;
- if dense footprints nearly cover the tile, most eye-height positions inside
  `I` are excluded even though literal leaf matter is sparse.

The horizontal footprint and both ends of `I` should include a guard margin
`delta_g` around the triangle soup. Then every supported exterior origin is
at least `delta_g` from lane geometry. That margin is what gives the final filtered
visibility field a finite maximum useful spatial/angular bandwidth. Without a
positive margin, an exterior camera can approach a triangle arbitrarily
closely and no finite max-resolution table has a uniform quality bound.

One common interval is the fewest-read option and the broadest fade region.
Using `L` fixed height/shape lanes `U_l=P_l x I_l` reduces false fade volume
but costs `L` independent carrier/transfer queries followed by a fixed min.
This is the explicit quality/read trade, not a hidden adaptive path.

## 4. Complete 4D transfer and its finite filtered realization

Inside an open visibility cell of `(q,d)` space, suppose triangle `j` is the
winner and lies in plane

\[
n_j\mathbin\cdot x=c_j.
\]

Then its live intersection is reconstructed exactly by

\[
\tau_j(q,d)=
\frac{c_j-n_j\mathbin\cdot q}{n_j\mathbin\cdot d},
\]

with the categorical triangle/copy token supplying the finite chart and
attributes. A dominant-axis atlas handles `n_j dot d=0` categorically: that
triangle is parallel and cannot be a regular first crossing. Within a valid
cell this is direct hit-plane reconstruction, not marching.

The hard data are the cell labels. Silhouette, triangle-edge incidence, and
depth-order changes partition the four-dimensional boundary/direction domain.
The finite model uses two record classes:

1. **Regular cell.** One owner is first throughout the cell, with a positive
   incidence/order margin. Store its categorical triangle/copy token and
   plane. The live result is the analytic equation above; depth, normal,
   colour and mark remain coupled to one real surface.
2. **Final mixed cell.** Subdivide discontinuity cells to the declared maximum
   physically useful resolution. If a cell is still mixed, integrate the
   triangle oracle through the same smooth spatial/angular pixel kernel used
   by the target renderer and store premultiplied colour/coverage plus a
   conservative depth interval and normal/material moments. This is explicitly
   a filtered appearance record, not a fictitious categorical surface.

A coarse regular atlas plus a fixed perfect-hash correction table provides
direct point location: compute the coarse and maximum-resolution keys
analytically, perform a fixed correction lookup, then fetch exactly one of the
two record classes. There is no adaptive walk. The hash contains every
non-default maximum-resolution child of mixed coarse cells; failure to fit is
a cook failure, never a live candidate search.

### 4.1 Why physical filtering can close a finite contract

Let the supported output be the ray-visibility function convolved with a
declared normalized `C1` four-dimensional footprint kernel `K_epsilon`, whose
smallest spatial support is no smaller than that induced at the guarded
carrier boundary by the closest supported camera/pixel configuration. Colour
and opacity are bounded. Convolution gives derivative bounds

\[
\|\partial_j(F*K_\epsilon)\|_\infty
\le \|F\|_\infty\,\|\partial_jK_\epsilon\|_1,
\]

independent of triangle count. Therefore a finite regular grid chosen from
those derivative bounds approximates the **filtered colour/coverage target**
to any declared tolerance. Arbitrary triangle soup complexity is absorbed by
offline integration rather than runtime complexity. This is an existence and
error statement, not a 250 MB bound: the required grid/codebook rank and the
number of categorical corrections still depend on the actual community.

The positive guard `delta_g`, finite horizon, bounded radiance, fixed maximum
display resolution, and fixed error tolerance are load-bearing. With no guard
or with a demand for unfiltered categorical output at every continuum ray,
the old finite-state counterexample returns: arbitrarily many resolvable
patches require arbitrarily many bits.

Depth/owner filtering has an honest boundary. In regular cells it is exact and
categorical. In final mixed cells no single depth/owner exists for the pixel
footprint. The record must remain coverage plus depth interval/moments, and
the renderer's acceptance contract must judge it as sub-pixel antialiasing.
It must never numerically blend those fields and claim the result is a real
surface. This is the small, explicit quality relaxation that makes the model
general at finite memory; it does not lower resolved near-field geometry.

### 4.2 Selected finite codec: certified 4D macrobricks

The implementation target is a direct `4 x 4 x 4 x 4` macrobrick code over
each boundary-position/direction chart. It is not a sampled-view blend, a
fixed event tail, or a low-rank moment field. Every brick is exactly one of:

1. **uniform regular/MISS:** one categorical chart token governs the brick;
2. **regular topology brick:** a lossless topology codeword maps its 256
   microcells to local categorical slots, and a deduplicated palette set maps
   each slot to one coupled plane/copy/material payload; or
3. **physically filtered mixed brick:** a VQ codeword stores premultiplied
   appearance, coverage, conservative depth interval, and normal/material
   moments at the declared pixel footprint. It is never decoded as a real
   blended surface.

Regular microcells are certified offline by interval/semialgebraic visibility
predicates: the same triangle or canonical analytic surface chart must be
first throughout the cell with positive support and order margin. Merely
sampling corners is not a certificate. Any uncertified owner switch becomes a
mixed cell. The live ray re-intersects the selected plane, so continuous
direction is used analytically after the categorical selection. Consequently
a regular cell cannot create the old `r Delta theta` wrong-owner wedge.

Mips are separately baked from the oracle at their larger physical footprint.
Depth, owner, normal, or mark are never averaged to manufacture a mip. Regular
coarse cells still require one certified owner; otherwise they are filtered
mixed records.

Let `B_Sigma` be the total number of resident bricks over required LODs,
`C_top` the number of topology codewords, `b_slot` the bytes per microcell
slot, `K_s` the palette size of palette-set `s`, `N_chart` the number of
coupled analytic payloads, and `C_mix` the filtered codebook size. The charged
transfer storage is

```text
M_T = 8 B_Sigma
    + 256 C_top b_slot
    + 4 sum_s K_s
    + N_chart b_chart
    + C_mix b_mix
    + M_mip_mixed.
```

At the current illustrative `256^2` boundary by `64^2` direction allocation,
`B_0=6,291,456`; eight-byte headers are `50.33 MB` base or about `67.11 MB`
with the contemplated mips. `4,096` one-byte topology prototypes cost
`1.05 MB`; even all `2.17M` source triangles as 16-byte charts cost
`34.7 MB`; `65,536` 32-byte filtered records cost `2.10 MB`. This leaves about
`145 MB` of the 250 MB ceiling for palette sets, the carrier, and auxiliaries.
The binding inequality is therefore

```text
4 sum_s K_s + M_rho + M_aux <= approximately 145 MB.
```

A distinct palette per brick is already known not to fit at the previously
measured roughly-six-owner scale. Palette-set deduplication or direct whole-
brick visual VQ must pass on the actual boundary field. Lossy replacement of a
certified regular owner is forbidden; only mixed filtered records may be
rate-distortion coded. This is the exact cook gate which replaces the earlier
unspecified phrase “compress the 4D field.”

## 5. Poles, horizon, copies, misses, and ties

None of these edge cases invalidates the common-slab factorization.

### 5.1 Horizontal and vertical rays

Use six dominant-direction charts. In chart `k`, `|d_k|>=1/sqrt(3)`, so line
coordinates never divide by a vanishing live component.

- Exact horizontal rays are ordinary `rho_P` queries with `s=1`; all repeated
  footprint gaps are already included in first passage.
- Exact vertical rays use the categorical occupancy/cap case of Section 3.2.

### 5.2 Finite horizon

The box carrier returns `MISS` if `t_U>R`. The transfer returns a relative
`tau`; the composed query accepts it only when `t_U+tau<=R`. This comparison
is exact and needs no additional sampled dimension.

### 5.3 Periodic copy offsets

Store positions modulo `Lambda` and return a signed periodic-copy offset with
the categorical winner. Add the live origin tile only after winner selection.
The required offset range is finite under `R`; a too-small packed range is a
format error, not a clamped mathematical case.

### 5.4 Face and owner ties

At a box edge/corner, choose the face by a fixed axis order after evaluating
the exact equal distances. Coincident triangle hits use the total owner order
declared in Section 1. Both adjacent boundary charts then denote the same
world ray and must return the same coupled result. Never blend their depth,
normal, mark, or periodic-copy identity.

## 6. Starting inside one or more boxes

The lane semantics are: if `o in U_P`, fade that **carrier lane** at the
camera. This is deliberately broader than identifying and fading one original
tight AABB. It makes overlapping footprints one union and removes owner-chain
traversal.

If later plants in the same lane should remain visible, bake the complementary
first-exit field

\[
\eta_P(q,\omega)
=\inf\{\ell>0:q+\ell\omega\notin P\}.
\]

For an origin in `P x I`, compare the lifted footprint-exit distance
`eta_P/s` with the analytic slab-exit distance. If the slab exit wins, the
lane has no further event. If the footprint exit wins, advance just to that
categorical boundary and query `rho_P` once for the next entry. The resulting
boundary point feeds `T`. This is two fixed mask-field reads, not an overlap
chain; `eta_P` is first passage against the **union mask**, so any number of
overlapping original boxes is already absorbed offline.

At `s=0`, a camera inside the carrier remains inside the same footprint until
it exits a slab cap, so the lane fades and then has no later lane event in
that ray direction.

This construction is complete for the declared lane-union fade semantics. It
does not implement the stronger rule “fade only one identified original plant
while another overlapping plant stays exact.” That stronger rule reintroduces
per-owner queries. The practical cost is intentional: overlapping/enlarged
plant boxes in one lane fade together while the camera occupies their union.

## 7. Two Tier-1 layers, wind, species, and moss

Suppose the two global Tier-1 populations have independent invertible affine
maps `A_0(t),A_1(t)` (golden-angle placement plus affine-in-height wind). For
each layer, transform the world ray to its baked coordinates:

\[
o_i=A_i^{-1}o,\qquad
d_i=\frac{A_i^{-1}d}{\|A_i^{-1}d\|}.
\]

The box-entry/transfer identity is preserved by this affine conjugation. A
world AABB may become a parallelepiped, but the baked local query remains the
same. Querying both layers independently and selecting the nearer categorical
opaque event is fixed work. Therefore two layers do not create a new
dimensional obstruction; they approximately double whatever box-entry and
transfer costs the one-layer representation actually has.

Species and moss can be unioned before baking **when they share the same layer
transform**. The winner token carries species/material categorically, so there
is no per-species runtime loop. Contributions with different motion laws
(wind-driven grass versus static moss) require independent affine queries and
composition; they cannot be pre-unioned into one static transfer without
changing the geometry.

For alpha-tested opaque surfaces, one first hit per independent layer is
sufficient before a nearest-depth selection. For genuinely transmissive fuzz,
one first hit is not a complete composition record. A separately authored
nonnegative medium may be integrated analytically to the nearest opaque
cutoff, but arbitrary interleaved transparent triangle soup again requires an
ordered event sequence.

Allowing a few reads above nine is useful when it funds a genuinely
independent affine/motion lane or the fixed correction lookup. It is not spent
on multiple candidate directions or successor ranks.

## 8. Hidden work audit

Under the corrected common-slab construction there is no hidden live
successor or candidate list:

1. `rho_P` is the complete 2D periodic first passage and skips all footprint
   gaps in one direct query.
2. `T(q,d)` is baked over the complete forward periodic community, so a miss
   in the first entered footprint does not cause a live retry.
3. An inside-lane origin uses one direct `eta_P` exit and one `rho_P` re-entry,
   not an overlap-chain walk.
4. A macrobrick lookup is fixed: brick header, topology/filtered codeword,
   optional palette token, and one coupled payload. A cook which cannot fit
   the frozen table fails rather than adding traversal.

The only semantic fork is recorded, not hidden: resolved cells return one
categorical real surface; final mixed cells return physically filtered
appearance and a depth interval/moments, not a made-up blended surface.

### 8.1 Fixed read envelope

For an exterior origin, one lane needs conceptually:

- one direct carrier first-passage record, or an index plus payload when the
  arbitrary footprint carrier itself is compressed;
- one transfer brick header;
- one topology or filtered-codeword access;
- an optional palette-token access; and
- one coupled chart/filtered payload.

That is approximately `4--6` reads per lane depending on whether the carrier
and transfer use direct or indexed payloads. Starting inside a lane adds one
exit-field read. Two independently affine Tier-1 lanes therefore occupy
roughly `8--12` reads, plus one control/mixture record. A separately moving
static-moss lane can exceed `14`; it is accepted only if its dense carrier and
short profile permit a cheaper specialization or the measured complete path
still satisfies the low/mid-range budget.

This is why nine is a useful target but not a theorem. Extra reads are justified
only when they buy an independent motion/carrier lane or the one fixed
discontinuity correction—not more view candidates. Register state is a small
fixed set of ray/chart coordinates and winner records; all ordinary accesses
are spatially coherent except the sparse correction/payload, so correction
occupancy is also a binding measured budget.

## 9. Exact verdict

The corrected box-fade rule plus common-slab enlargement produces a complete
mathematical model:

> **Exact boundary-transfer factorization.** Conditional on knowing the first
> future entry `q` into the periodic box union, arbitrary forward triangle-soup
> visibility from any gap origin reduces exactly to one boundary-conditioned
> four-dimensional transfer query, even if the ray misses geometry in the
> first box.

For `U=P x I`, `q` is itself obtained exactly by the 3D first-passage field
`rho_P`; the former gap counterexample is answered because the gap phase is
part of `q_a`. Thus the composed **continuous oracle** is complete for every
opaque marked triangle soup contained in the carrier, every exterior
direction, exact horizontal/vertical rays, periodic copies, first-box misses,
and finite horizon. Two fixed affine Tier-1 layers are the min/composition of
two such oracles.

Its finite realization exists relative to the declared physically filtered
output contract: categorical hit-plane reconstruction in regular cells,
smooth kernel-integrated appearance in unresolved final cells, and an explicit
carrier guard/fade margin that bounds the smallest supported footprint. A
particular compiled community must still pass the fixed byte/read gate. It
cannot simultaneously promise bit-exact unfiltered continuum visibility for
arbitrary soup; that stronger statement still violates the finite-state bound.
“Works with arbitrary triangle soup” therefore means the compiler accepts the
soup and targets its filtered visible bandwidth, with an explicit failure if
the error/byte budget cannot encode it—not that unlimited source bits consume
no memory.

## 10. Binding construction contract

Implementation is mathematically authorized only with all of the following
stated rather than hidden:

1. the finite actual community, horizon, closest supported exterior distance,
   display footprint, and colour/depth tolerances are frozen;
2. each lane uses guarded common-slab carrier `P_l x I_l`; origins inside it
   fade that lane (including overlapping plants), and `eta_P` handles later
   components only if required;
3. a compiled 3D carrier first-passage field and 4D boundary transfer are
   permitted to fail their byte/read gate rather than silently degrade;
4. regular cells return direct categorical hit-plane tokens, while final
   discontinuity cells are explicitly scored as filtered appearance rather
   than exact geometry; and
5. two Tier-1/motion classes are budgeted as two independent queries, with all
   same-motion species and moss unioned offline.

Curved terrain, rooted affine stiffness, smooth height/vigor, ecological
control, and full-screen invocation are governed by the companion
`GRASS-BOX-TERRAIN-MOTION-CLOSURE.md`. Curved attachment is never mislabeled
an affine conjugation: the winning root/plane is corrected exactly and its
continued firstness is certified against every later event under the complete
terrain/deformation error bound.

If it passed the finite cook gate, Section 4.2's certified macrobrick decode
would be `O(1)`:
direct carrier address, direct brick header, one topology/filtered-codeword
lookup, at most one palette lookup, one coupled payload, and a fixed lane min.
A sparse max-resolution hash remains only the comparison codec. At `256^2`
boundary samples and `64^2` directions its illustrative `4^4` block table is
already about `67 MB` with mips, while an optimistic R32 carrier is another
`22 MB`; the remaining correction occupancy would have to be extremely small
to remain under 250 MB. The actual gate instead measured 50.66% modal
corrections and 100% mixed cells at every tested tier. Macrobrick palette
entropy was therefore a measured rejection, not an assumed sparse-correction
win.
There is no runtime grass geometry, march, loop, candidate list, per-species
query, or direction-Voronoi interpolation. The first CPU gate must separately
report (a) fade-volume fraction caused by common-slab/guard enlargement, (b)
regular versus final-filtered cell fraction, (c) union-mask perimeter/chart
area plus carrier bytes, (d) codec/index/payload bytes and occupancy, and (e)
resolved-cell categorical geometry error versus final-filtered
colour/coverage error. Those measurements choose the smallest lane count and
codec that meet visual quality; they do not repair unresolved math.

## 11. Actual-source implementation verdict

The 2026-07-23 gate separates three statements which must remain distinct:

1. **continuous periodic factorization: green;** direct truth and composed
   carrier/transfer truth agreed on all 192 sampled rays;
2. **finite ecological-patch entry: incomplete;** the active footprint filled
   the periodic tile, so the gate exercised cap entry but no outer patch side;
3. **selected finite macrobrick codec: red.** The tested cells were wholly
   mixed and remained far larger than the declared closest-camera pixel
   footprint, so they could not legally terminate as filtered atoms.

The gate's point-centre/four-corner RGB values are interpolation diagnostics,
not a valid final filtered-image rate-distortion measurement. A successor
codec must integrate exact pinhole subrays sharing one camera origin; its
boundary point and direction therefore vary together under the rank-two pixel
Jacobian. It must train an actual fixed-cost compressor, decode held-out
camera pixels, and compare them with independently converged integrals through
the same pixel kernel. Raw point-owner mixing cannot be extrapolated into a
per-cell 12-byte lower bound, but neither can it be renamed a filtered green.

No runtime transcription is authorized by this document. The continuous
oracle is an upstream identity available to a future codec; it is not by
itself an implementation plan under the byte and exterior-fidelity contract.

## 12. Filtering reintroduces standoff unless the finest field is stored

The four-dimensional reduction applies to a **central pointed ray**. Physical
pixel filtering has an additional dependency which cannot be hidden. Let a
carrier face be the plane `n dot x = c`, let `o` be the pinhole origin, and let
`d(xi)` be the normalized ray through sub-pixel coordinate `xi`. Then

\[
 t(\xi)={c-n\cdot o\over n\cdot d(\xi)},\qquad
 q(\xi)=o+t(\xi)d(\xi).
\]

For a differential ray perturbation,

\[
 \delta q
 =t\left[I-d{n^T\over n\cdot d}\right]\delta d.
\]

Thus one camera pixel maps to a correlated rank-two footprint in the four
boundary/direction coordinates. The scale contains `t`, the camera-to-face
standoff. Two cameras translated along the same exterior oriented line have
the same central `(q,d)` and therefore the same point-ray answer, but different
`t` and different filtered pixel answers. A single filtered value indexed only
by `(q,d)` cannot represent both.

There are only three honest closures:

1. store the 4D point-ray/finest-filter field at the smallest supported
   footprint and apply the live correlated filter from derivatives/mips;
2. store a finite standoff/covariance family and charge its added state and
   quantisation error; or
3. restrict the supported camera/filter domain so one conservative footprint
   is visually acceptable.

Option 3 is not the current exterior-fidelity contract. Option 1 is finite
because the guard gives a positive source separation, but at the active
2.5 cm guard and frozen 60-degree/1920 projection the minimum footprint is
about 13.6 micrometres. Option 2 is a discretised extra dimension rather than
free 4D filtering. A valid successor gate must choose one of the first two,
charge it, and integrate exact shared-origin subrays; holding `q` fixed while
perturbing `d` is not a pinhole footprint.

### 12.1 No exact identity removes the filter state for arbitrary soup

Use any two-plane light-field chart `L(u,v)`, with `u,v in R^2`. Cameras
translated along one exterior central ray retain the same `(u0,v0)`, while
their sub-pixel rays form a standoff-dependent affine/projective two-plane

\[
 (u(\xi),v(\xi))=(u_0,v_0)+A_s\xi .
\]

The pixel is therefore the restricted Radon transform

\[
 F_L(s)=\int K(\xi)L((u_0,v_0)+A_s\xi)\,d\xi,
\]

not a value or finite jet of the central transfer. To prove that no further
exact identity removes `s`, embed a one-dimensional cross-section and extrude
it. For arbitrary `M`, finite guarded microtubes/occluders can form `M`
disjoint coloured ray bundles away from the central ray, each crossed by the
pixel's sheared segment on a separate exterior standoff interval. Switching
those bundles independently produces `2^M` finite triangle soups with the
same central `(q,d)` value but `M` independently different filtered values.
Any exact state answering all standoffs needs at least `M` bits; `M` is
unbounded for arbitrary finite soup.

The construction keeps all cameras outside the guarded carrier and all
microstructure a positive guard distance inside, so box fading does not remove
the counterexample. A 4D prefix sum also does not help: it integrates
axis-aligned four-volumes, whereas the pixel is an oblique measure-zero
two-plane whose shear is the missing state.

Accordingly the strongest implementable product is necessarily a measured
screen-space approximation: freeze FOV/resolution/horizon/guard, quantise the
live Jacobian to a finite filter family, and require held-out premultiplied
RGB/coverage/depth/temporal tolerances under the shared byte/read cap. This is
not a reduction in supported exterior angles. It is the minimum honest
relaxation from exact physical-pixel equality.

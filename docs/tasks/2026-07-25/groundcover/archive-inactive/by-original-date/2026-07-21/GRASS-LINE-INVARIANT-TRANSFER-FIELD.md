# Line-invariant marked transfer field for periodic ground cover

Status: **blocked before codec fitting**, 2026-07-22. No runtime shader
implements this design. The independent beam marginals proposed here do not
preserve micro-line or marked-event identity and therefore cannot test the
active widening-fan, wrong-perspective, or shimmer defects. The replacement
truth contract is the coupled marked transport
`G_m(q,xi)` derived in `GRASS-COUPLED-MARKED-TRANSPORT-MATH.md`. Sections 7--10
below are preserved as the rejected marginal hypothesis and cost envelope, not
as authority to fit or integrate a codec.

## 1. Outcome and non-negotiable contract

The intended outcome is one precomputed field which can render a repeating,
offline-composed community containing tall grass, shorter species, flowers,
and moss with all of the following properties:

- full-sphere view directions, including exact horizontal and vertical rays;
- exterior and camera-inside starts;
- work independent of repeated plant count;
- no runtime ray march, traversal, candidate list, shell stack, or species loop;
- one selected visibility event for the existing opaque visibility buffer;
- a fixed one-to-four-read decoder under the existing 51,121,152-byte resident
  profile cap;
- explicit uncertainty whenever one opaque event is not a faithful reduction of
  the filtered distribution.

The proposal does **not** claim exact source-triangle depth after filtering.
That object is the discontinuous four-dimensional exterior first-owner field
and five-dimensional pointed-line successor field already shown to defeat the
cheap tested codecs. The proposed object is the image-space signal actually
needed after a pixel footprint has integrated many subpixel stems, blades, and
panicle branches.

## 2. Evidence boundary

This design transfers only the following primary-source results:

- Sannikov: a repeating precomputed ray field makes cost independent of copy
  count; the published elevation reduction is exact only for its parallel
  extrusion model, and later direction mismatch is known to stretch geometry.
- VDM: direct view-conditioned displacement can be fixed cost, while a separate
  nearest silhouette field is evidence that hit/miss boundaries are not safely
  low-rank averaged.
- GDM: the generic non-height-field query is origin-aware,
  `d(x,y,z,theta,phi)`, covers the full sphere, and may include periodic
  neighbours during baking.
- The audited fur/PRT methods precompute lighting or use shells/traversal; none
  supplies the missing first-visibility field.

The marked transfer algebra, line-chart formulation, filtered-distribution
contract, and proposed factorized codec below are LAAS derivations. They are not
attributed to those sources. Source links and exact transfer decisions are in
`../../deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.

## 3. Domain: a pointed oriented line, not a camera-centred carrier

Let a community be periodic under the horizontal lattice

\[
\Lambda=\{(nP_x,0,mP_z): n,m\in\mathbb Z\}
\]

inside the conservative botanical slab `0 <= y <= H_*`. `H_*` is a coordinate
boundary, not rendered geometry and not the height of every species.

For a local ray datum `A` and unit forward direction `d`, choose the signed
dominant axis `j`, so its signed component is positive and

\[
|d_j|\ge 1/\sqrt 3.
\]

Let `a,b` be the other axes, intersect the oriented line with the fixed chart
plane `x_j=k_j`, and define

\[
Q=A+d\frac{k_j-A_j}{d_j},\qquad
r=(d_a/d_j,d_b/d_j),\qquad
\eta=A_j-k_j.
\]

Then

\[
A=Q+\eta(e_j+r_a e_a+r_b e_b).
\]

The exterior line coordinate

\[
\ell_j=(Q_a,Q_b,r_a,r_b)
\]

is unchanged by replacing `A` by any other datum on the same oriented line.
The fifth coordinate `eta` identifies the physical origin on that line and is
therefore allowed to change only for a genuine camera-inside move. This cleanly
separates the four-dimensional exterior query from the five-dimensional
successor query.

Periodicity is a quotient, not a new view-dependent carrier. For a lattice
translation $L\in\Lambda$, re-intersection with the same chart plane gives

\[
(Q_a,Q_b)\sim
(Q_a+L_a-r_aL_j,\;Q_b+L_b-r_bL_j).
\]

This sheared lattice action is the correct wrap rule in a horizontal-dominant
chart. In the vertical chart it reduces to ordinary XZ wrapping. The six chart
overlaps describe the same pointed line and must decode the same transfer
record. There is no elevation clamp or nearest angular cone.

An implementation may encode the equivalent bounded coordinates
`(A_x mod P_x,A_y,A_z mod P_z,d)` because they are convenient for a spatial
volume times angular basis. The oriented-line form above remains the reference
for invariance and chart-seam tests.

## 4. Exact per-ray marked transfer

Parameterise a forward ray segment by physical distance `s`. Let
`kappa_m(s)` be extinction belonging to mark `m` (species/material/cover type),
`kappa=sum_m kappa_m`, and

\[
T(s)=\exp\left(-\int_0^s\kappa(u)\,du\right).
\]

For opaque polygonal microgeometry this notation is understood as a singular
first-event measure: `T` jumps from one to zero at the first hit. It therefore
also covers the exact mesh bake without first voxelising it.

The transfer record of one segment of length `L` is

\[
\mathcal R=
\left(
T_L,\ C,\ M_1,\ M_2,\ N,\ W
\right),
\]

where

\[
\begin{aligned}
A &= 1-T_L,\\
C &= \int_0^L T(s)\kappa(s)c(s)\,ds,\\
M_1 &= \int_0^L sT(s)\kappa(s)\,ds,\\
M_2 &= \int_0^L s^2T(s)\kappa(s)\,ds,\\
N &= \int_0^L T(s)\kappa(s)n(s)\,ds,\\
W_m &= \int_0^L T(s)\kappa_m(s)\,ds.
\end{aligned}
\]

`C` is a transfer-weighted authored base-colour/material quantity rather than
necessarily final lit radiance. `N` is a linear normal moment, not yet a unit
normal. The mark masses satisfy `sum_m W_m=A` for the simple extinction model.
Opaque geometry produces a delta event; translucent moss or hair-like cover may
produce a distributed event without changing the algebra.

## 5. Segment composition and the camera-inside inversion limit

Let adjacent front and back segments have records `R_1`, `R_2`, with local
depths in the second segment measured from its own start and first-segment
length `L_1`. Exact front-to-back composition is

\[
\begin{aligned}
T_{12} &= T_1T_2,\\
C_{12} &= C_1+T_1C_2,\\
N_{12} &= N_1+T_1N_2,\\
W_{12} &= W_1+T_1W_2,\\
M_{1,12} &= M_{1,1}+T_1(M_{1,2}+L_1A_2),\\
M_{2,12} &= M_{2,1}+T_1(M_{2,2}+2L_1M_{1,2}+L_1^2A_2).
\end{aligned}
\]

Thus marked transfer is an associative fixed-size monoid. Grass, panicles,
flowers, litter, and moss must be composed in actual ray order **during the
bake**, before the field is filtered.

If a total unfiltered ray record factors as
$\mathcal R=\mathcal R_P\circ\mathcal R_S$, suffix values can formally be
recovered from a prefix whenever `T_P>0`:

\[
\begin{aligned}
T_S &= T/T_P,\\
C_S &= (C-C_P)/T_P,\\
N_S &= (N-N_P)/T_P,\\
W_S &= (W-W_P)/T_P,\\
A_S &= (A-A_P)/T_P,\\
M_{1,S} &= (M_1-M_{1,P})/T_P-L_PA_S,\\
M_{2,S} &= (M_2-M_{2,P})/T_P-2L_PM_{1,S}-L_P^2A_S.
\end{aligned}
\]

This is **not** an acceptable runtime camera-inside solution:

1. an opaque prefix has `T_P=0`, so its hidden suffix is information-theoretically
   absent from the total record;
2. near opacity makes the division catastrophically sensitive to quantisation;
3. footprint averaging does not commute with composition or inversion, because
   `E[T_1T_2] != E[T_1]E[T_2]` for correlated subrays.

Therefore the field must directly bake the forward suffix from every declared
origin height. Camera-inside support is a genuine origin-aware fifth coordinate,
as in GDM semantics, but its target is a filtered transfer distribution rather
than a discontinuous exact distance. Composition and reverse scans remain
offline tools and consistency checks; runtime performs neither a prefix inverse
nor a successor search.

## 6. Full-sphere endpoint rule

For a start `x` inside the canonical slab, the forward segment ends at

\[
L(x,d)=\min(L_{max},L_{slab}(x,d)),
\]

with

\[
L_{slab}=\begin{cases}
(H_*-x_y)/d_y,&d_y>0,\\
-x_y/d_y,&d_y<0,\\
+\infty,&d_y=0.
\end{cases}
\]

`L_max=155 m` is the currently declared world grass horizon. Exact horizontal
rays are consequently ordinary finite queries, not a clamped five-degree view.
Exterior downward rays first intersect the algebraic top boundary and query at
that point; exterior rays which point away from the slab miss. Inside upward,
downward, horizontal, and vertical rays query the same suffix field at the
physical origin.

The final representative depth is still tested against the existing scene
visibility depth. If the filtered distribution materially straddles an opaque
scene surface, one ground-cover event cannot reproduce the joint visibility;
that is an explicit failure metric, not a reason to average through the scene.

## 7. Filtering changes what correctness means

For footprint level `m`, let `K_m` be a declared distribution of subpixel
phases and directions. Offline, every micro-ray first traces the complete marked
community and composes its exact ordered transfer. Only then are those complete
records accumulated into a first-interaction measure

\[
\nu_m(dt,da)=
\mathbb E_{r\sim K_m}[\text{interaction measure of }r].
\]

Miss mass is `T=1-nu_m(total)`. Colour, normal, depth, and marks are moments of
the same measure. This makes averaging meaningful: it is a mixture of rays in
one pixel footprint, not an invented surface between unrelated owners.

The field coefficients must be prefiltered for their actual reconstruction
kernel. Spatial trilinear interpolation is legitimate only when the stored
coefficients define that piecewise-trilinear band-limited field; it is not
permission to linearly blend exact first depths. Each mip is baked from the
micro-ray distribution at that footprint rather than formed by blindly
averaging nonlinear conditional attributes.

The finest accepted footprint sets the quality ceiling. If its conditional
depth or normal spread remains broad, this representation is not a crisp
near-field solution even if colour looks plausible.

## 8. One visibility-buffer event

The deterministic moment reduction

\[
\bar t=M_1/A,\qquad
\sigma_t^2=M_2/A-\bar t^2
\]

is useful for diagnostics but is not a general visible surface. The previous
event-measure experiment measured exactly why: a multimodal distribution can
have a mean in empty space and appear as a flat or stretched sheet.

The proposed opaque-vis-buffer reduction is instead a stable one-sample
estimator of the conditional first-interaction distribution:

1. use a frame-invariant low-discrepancy scalar `xi` derived from the canonical
   oriented-line coordinate `ell_j`, community-supertile phase, and mip, never
   from time, camera range, or the inside-origin coordinate `eta`;
2. emit miss when `xi >= A`;
3. otherwise set `p=xi/A`, select one of `K_q=3` equal-mass depth strata, and
   emit that stratum's conditional depth, authored colour, normal, and material;
4. write exactly that one depth to the visibility buffer.

The bake stores, for each stratum, its mean depth and variance. Equal-mass
strata need no runtime probability vector. This fixed three-stratum reduction
cannot reproduce an arbitrary continuous distribution exactly, but unlike one
global mean it preserves separated front/middle/back mass statistically. Its
error is measurable: within-stratum `sigma_t`, colour error, normal angular
dispersion, and temporal stability under a continuous camera path.

The line-anchored sample must pass a no-shimmer sequence test. A per-frame hash,
screen-space random particle field, or camera-radius threshold is forbidden.
For a fixed `ell_j` and `xi`, moving the origin forward without crossing the
selected interaction must reconstruct the same world interaction point; after
crossing it, the result may jump only to a later event. This right-censoring
identity is the stochastic camera-inside analogue of exterior line invariance.
Independent fits at each origin height are invalid if they violate it, even
when their per-height images look plausible.
If three strata cannot meet the near-field gate, increasing strata is not an
automatic next step because decoder ALU and output bandwidth rise linearly.

## 9. Offline-composed species, overlap, and moss

One community bake contains the actual simultaneous population:

\[
\mathcal G=\mathcal G_{grass,1}\cup\cdots\cup
\mathcal G_{flower}\cup\mathcal G_{moss}\cup\mathcal G_{litter}.
\]

Exact micro-rays trace that union. A nearer moss tuft can obscure a blade; a
panicle can appear through an opening; non-green flowers and violet panicles
retain their authored colour because colour follows the selected transfer
stratum. No runtime `min_i`, species loop, or independent opacity blend exists.

Species/material marks are retained in the offline truth and source ledger even
if the runtime record reduces them to authored colour, normal, roughness, and
transmission. This keeps the bake auditable and permits later marked outputs.
Moss is not treated as short grass: its geometry/volume, orientation
distribution, material response, and low height participate directly.

A finite catalogue of ecologically valid community states is permitted. Each
state is one precomposed field. Arbitrary runtime Boolean mixtures would require
one query per field or exponentially many precompositions and are outside the
constant-one-field contract. Anti-tiling variants should be baked into a larger
marked supertile rather than evaluated as extra live layers.

## 10. Candidate fixed-cost codec and hard gate

The direct suffix field is five-dimensional in convenient coordinates:

\[
F_m(x,d),\qquad
x=(x\bmod\Lambda,y),\ d\in S^2.
\]

Exact categorical depth made the earlier small origin-aware decoder fail. The
new hypothesis is narrower and independently falsifiable: after footprint
filtering and three-stratum reduction, the bounded transfer outputs have a low
separation rank between origin and direction.

Use the continuous factorisation

\[
z_o(x,d)=b_o+\sum_{r=1}^{R}C_{or}U_r(x)V_r(d),
\]

followed by bounded output transforms which enforce coverage in `[0,1]`, ordered
depth strata, nonnegative variance, valid colour/material ranges, and normal
normalisation. `U_r` are periodic-XZ/clamped-Y 3D coefficient volumes and
`V_r` are continuous full-sphere angular basis fields. Both sides of every
dominant-axis chart seam train against the same truth record.

The proposed runtime-shaped record has 24 decoded scalars:

- one coverage;
- three ordered conditional depths and three log variances;
- three RGB conditional authored colours;
- three two-component conditional normals;
- one shared roughness and one shared transmission term.

Two complete attempts are allowed:

| attempt | rank | logical filtered reads | dense decoder | coefficient bytes including all spatial mips | purpose |
|---|---:|---:|---:|---:|---|
| A | 4 | 1 RGBA16F 3D + 1 RGBA16F angular | 96 FMAs plus bounded transforms | about 21.6 MB | preferred low/mid path |
| B | 8 | 2 RGBA16F 3D + 2 RGBA16F angular | 192 FMAs plus bounded transforms | 43,141,408 spatial + 65,536 angular bytes | hard maximum |

The byte count uses a `192x192x64` spatial grid. Its complete mip chain has
2,696,338 voxels; rank eight at 16 bytes per voxel is 43,141,408 bytes. A
`64x64x8` half-float angular basis is 65,536 bytes. Decoder matrices and
metadata are negligible relative to the remaining approximately 7.9 MB under
the 51,121,152-byte cap. One high-fidelity community state is resident at this
ceiling; multiple simultaneous full-resolution states are not silently assumed.

There is no binding, pass, dispatch, barrier, synchronization scope, data-
dependent branch, or downstream species election in this cost. A future shader
integration must still measure register live ranges, sampler traffic/cache
locality, generated ALU, occupancy, and the visibility-buffer write. Passing
this mathematical gate would authorize that measurement; it would not waive it.

The actual-asset acceptance set is:

1. full-sphere held-out directions including exact 0, 0.1, 1, 5, 15, 35, 75,
   and 90-degree elevations in both vertical senses;
2. exterior and inside-origin height strata through `0..H_*`;
3. held-out phase, direction, height, and community seeds;
4. coverage IoU, conditional colour error, normal angular error, per-stratum
   depth bias and p95/p99 spread, scene-depth-straddle rate, and mip continuity;
5. exterior line-datum invariance and fixed-sample inside-origin
   right-censoring before any aggregate image metric;
6. a camera-path image sequence testing chart seams, radial fans, distance
   stretch, planes, panicle disappearance, and shimmer;
7. numbered QA PNGs and a machine manifest under a content-addressed
   `data/work/groundcover-transfer-field/<source-hash>/<recipe-hash>/qa/` root.

Attempt B is parked if it still shows the current named artifacts or if its
generated fixed cost exceeds four reads, 256 FMA-equivalent operations, or the
resident byte cap. The fallback is the last visually accepted exterior field
while this research track remains explicitly incomplete; it is not more taps,
more strata, a larger network, a march, or a shell.

## 11. Known failure modes and falsifiers

- **Not exact geometry.** One realization is a stochastic sample of a filtered
  distribution. It must never be described as an exact source-mesh hit.
- **Multimodal residue.** Large within-stratum depth variance recreates floating
  planes, elongation, or wrong colour/depth pairing. This is the primary gate.
- **Angular underfit.** Low separation rank can still smear silhouettes or make
  chart-aligned fans. Continuous bases remove nearest cones but do not guarantee
  sufficient bandwidth.
- **Temporal instability.** An unstable coverage sample becomes the old
  million-particle shimmer. The sampling coordinate must be line anchored and
  frame invariant.
- **Scene-depth straddling.** One grass event cannot exactly combine a filtered
  distribution whose mass lies on both sides of an opaque scene hit.
- **Terrain chart error.** The proof is exact in one affine/root chart. A
  continuously draped non-affine terrain can curve a world ray in profile
  coordinates; local-root transforms or a certified terrain-family bake are
  required rather than a frozen arbitrary tangent.
- **Horizontal horizon.** Exact horizontal queries use the declared 155 m cap.
  Changing that cap changes the field and requires a new bake.
- **Community boundaries.** A ray crossing independently selected community
  tiles is not represented by one homogeneous-state lookup unless that boundary
  is part of the baked world/community state.
- **Wind.** A global affine deformation may transform the ray into the canonical
  field exactly. General spatially varying wind changes the pointed-line field
  and needs baked wind states or a separately proved deformation conjugacy.
- **Material reduction.** Shared roughness/transmission can be insufficient when
  different depth strata have strongly different materials; the error must be
  measured before adding outputs.

## 12. Paper-grade claim boundary

If the actual-asset gate passes, the defensible contribution is not “exact
raytraced grass in one texture.” It is:

> a line-consistent, origin-aware, marked first-interaction transfer field for
> periodic ground-cover communities, with associative offline composition,
> direct camera-inside suffixes, a fixed-cost factorized decoder, and a stable
> single-event estimator for an opaque visibility buffer.

The negative results are part of that claim: exact categorical angular
interpolation, tiny deep lists, direct learned first-distance regression,
event-mean depth, literal PCF quantiles, and independent species composition all
have measured or proved failure boundaries. A paper must publish those gates,
the immutable source hashes, full bake recipes, direction/phase splits, codec
rank and bytes, image sequences, and the distinction between statistical
image fidelity and exact source-surface depth.

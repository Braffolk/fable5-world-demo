# Exact representation theory for precomputed ground-cover first-hit rendering

Status: pure mathematics, 2026-07-22. No runtime, shader, asset, or test file is
changed by this document. It is the comprehensive derivation requested after the
exploration stop recorded in `GRASS-MATHEMATICAL-EXPLORATION-SUMMARY.md`, and it
supersedes the open representation questions of
`GRASS-CHEAP-PROJECTIVE-RAY-MATH.md` §20–21.

## 0. Verdict, stated first

This document proves three things and is explicit about what each requires.

**V1 (impossibility side).** For the accepted arbitrary-morphology community
(the 2,171,134-triangle Calamagrostis union and any comparable arbitrary marked
mesh), the exact coupled first-event query under the frozen resource bound
(a very small fixed number of reads, `≤ 51,121,152` resident bytes, no loop,
march, traversal, or candidate list) is:

- **unconditionally impossible** for every representation in the following
  families: fixed-`K` event records (Lemma 3.2), arithmetic interpolation or
  quantile election across visibility owners (Lemma 3.3), camera-direction
  binned/nearest selection at any fixed angular rate (Theorem 4.1, the
  amplification dichotomy — the artifact never becomes subpixel), per-tile
  non-affine (projective/tapered) instancing (Lemma 5.3), and per-element
  finite axial intervals inside a shared 2D-mask reduction (Lemma 5.4);
- **conditionally impossible** in general, under an explicitly stated and
  heavily evidenced incompressibility hypothesis (Proposition 3.5): the
  required sampled field is `≈ 22×` to `≈ 580×` over the byte cap at the
  coherence rates its own geometry demands, and twenty-five independent
  measured compression attempts (rank, VQ, codebook, learned, plane-factor,
  flow, sheet, cascade) found no structure that closes that gap.

**V2 (constructive side).** There is a geometry class `E` — finite unions of
affine-sheared, lattice-compatible **parallel-extrusion families** with
family-global horizontal clips and quantized per-texel top bands, plus
**extruded volumetric (extinction) families** for sub-pixel plume/fuzz detail —
for which the complete promised query contract is **proved exactly**
(Theorems 6.4–6.9): coupled first event; attachment and firstness; oriented-line
invariance; exact Λ-periodicity with one query covering all copies; exact
horizontal and vertical rays with no epsilon or clamp; exact camera-inside
successor semantics; exact multi-species overlap and moss with no per-species
runtime work. The runtime is a fixed number of **independent** texture reads at
analytically computed addresses plus fixed ALU (Section 9). All residual errors
are quantization-scale, **camera-distance-independent**, and individually
bounded in Section 8.

**V3 (the fork that this forces).** V1 and V2 together imply the arbitrary
authored mesh is not a valid *source encoding* for this renderer; it is only a
valid *visual reference*. The mathematically forced move is one level up from
every rejected codec: **author (or re-generate) the community inside class
`E`**, judged against the same real-reference visual bar. The remaining risk is
then an authoring-fidelity question (how many directional classes Estonian
ground cover needs — Section 10), not a reconstruction-correctness question.
The final verdict in Section 12 is therefore: **conditionally proved** —
mathematics complete for class `E`; visual sufficiency of class-`E` authoring
is the single remaining non-mathematical gate.

Both artifacts that ever passed this project's visual gates already sit inside
this theory: the accepted O(1) production sward is precisely a `K=1` member of
class `E` (parallel-extrusion case, `shiftK=thickK=arcK=0`), and the accepted
band-limited structural/plume source split of
`GRASS-STATUS-AND-ISSUES.md` §34 is precisely the opaque/volumetric family
decomposition of Section 7. The theory explains both acceptances and all
twenty-five rejections with the same theorems (Section 11).

---

## 1. The query domain (Question 1)

### 1.1 Ambient objects

Work in a botanical-local orthonormal frame in which the ground chart is flat;
Section 8.4 restores curved terrain through the honest derivative basis and
bounds the error. Let

\[
\Lambda \;=\; T\,\mathbb Z^2 \subset \mathbb R^2_{xz},
\qquad T = 0.52\ \mathrm m \ \text{(current tile)},
\]

be the horizontal period lattice, and let the botanical slab be

\[
S \;=\; \mathbb R^2_{xz}\times[y_{\min},\,y_{\max}],
\qquad H=y_{\max}.
\]

The marked community is a closed set with attributes,

\[
\mathcal G\subset S,\qquad
\mathcal G+\lambda=\mathcal G\ \ \forall\lambda\in\Lambda,
\]

together with a mark/attribute map defined on `∂𝒢` (normal, authored colour,
species/material mark). Multi-species overlap is composed into this single
union offline; that choice is retained from the prior work and re-proved
compatible in Section 6.6.

### 1.2 Oriented lines: the nonsingular atlas

The space of **oriented lines** in `ℝ³` is globally diffeomorphic to the
tangent bundle of the sphere,

\[
\mathcal L \;\cong\; TS^2
=\{(d,w)\;:\;d\in S^2,\ w\in d^{\perp}\},
\]

where `w` is the line's foot point (the point of the line closest to the
origin). `dim 𝓛 = 4`. This atlas has **no pole anywhere**: every direction,
including exactly horizontal and exactly vertical, is an ordinary point of
`TS²`. Unoriented lines are the quotient by `(d,w) ↦ (−d,w)`; the query is
oriented, so no quotient is taken.

The lattice acts on line space by

\[
\lambda\cdot(d,w)=\bigl(d,\ w+P_{d^\perp}\lambda\bigr),
\qquad P_{d^\perp}=I-dd^{T},
\]

and the exterior query domain is the (still 4-dimensional, still nonsingular)
quotient `𝓛/Λ`.

**Storage charts.** For byte layout only, cover `S²` by the six closed
dominant-axis cells: chart `C_j` (with `j ∈ {±x,±y,±z}`, `|d_j|` maximal)
intersects the line with the fixed plane `x_j = k_j` at

\[
Q=A+d\,\frac{k_j-A_j}{d_j},
\qquad
\ell_j=\Bigl(Q_a,\;Q_b,\;r_a,\;r_b\Bigr),\quad
r=\Bigl(\tfrac{d_a}{d_j},\tfrac{d_b}{d_j}\Bigr)\in[-1,1]^2 ,
\]

with the unconditional bound `|d_j| ≥ 1/√3`. There is no live denominator pole
and no epsilon: chart choice is a categorical `max`-component select, chart
overlap occurs only on the measure-zero cell boundaries where both charts are
exact and agree. This chart family is retained from
`GRASS-CHEAP-PROJECTIVE-RAY-MATH.md` §20.4; here it is only a coordinate
convention. **Nothing in this document samples these direction coordinates**
— that is the decisive difference from every rejected representation
(Theorem 4.1).

### 1.3 Pointed rays and the successor fibre

A **pointed ray** is a pair `(ℓ, t₀)` — an oriented line and an origin
parameter on it; the origin point is `o = w + t₀ d`. The pointed domain is the
5-dimensional total space of the tautological line bundle over `𝓛/Λ`.

For closed `𝒢`, define the **event set** of a line as its ordered entry set

\[
E(\ell)=\{\,t\;:\;o+td\in\partial\mathcal G,\ \text{entering}\,\},
\]

which is discrete for any finite-complexity community (and provably discrete
for class `E`, Section 6). The complete query is the **successor operator**

\[
\boxed{\;
N(\ell,t_0)=\min\{\,t\in E(\ell)\;:\;t> t_0\,\}
\;}
\]

returning that event's coupled record `(t, P, n, \text{colour}, \text{mark})`,
or MISS if the set is empty within the declared horizon `R_{\max}=155\ \mathrm m`.

Two structural facts follow directly from this definition and are used
throughout:

**Fact 1.1 (exterior reduction).** If the origin lies outside the slab `S` (or
before the first event), `N(ℓ,t₀)` is independent of `t₀`; the exterior query
is exactly 4-dimensional.

**Fact 1.2 (fibre structure of the fifth dimension).** For fixed `ℓ`, the map
`t₀ ↦ N(ℓ,t₀)` is right-continuous, piecewise constant, and completely
determined by the ordered event set `E(ℓ)`. The fifth dimension carries **no
new continuous information**: it selects an element of an ordered fibre. Any
representation that can (i) produce the first element of `E(ℓ)` after an
arbitrary origin phase is automatically a correct 5-dimensional successor
field. This is the precise form of the camera-inside requirement, and it is
what Section 6 proves for class `E` — not by storing a deep record, but by
making every interior origin its own exact query point.

**Horizontal rays.** In the `TS²` atlas and in the dominant-axis charts,
`d_y=0` is an ordinary point. The historical `top-plane/slope` chart
(`s = d_{xz}/(−d_y)`) is singular there; that chart is used in this document
only where its own hypotheses hold, never as the domain atlas. Requirement
"no coordinate poles hidden with epsilons" is met by construction.

### 1.4 Finite horizon and periodic continuation

The community is periodically continued in `xz` without bound; queries are
truncated at `R_max = 155 m` (declared, from the existing acceptance
contract). MISS beyond the horizon is a **categorical value**, not a clamp: it
is the correct value of the truncated query. An exactly horizontal ray inside
the slab therefore has a well-defined answer: the first event within `R_max`,
else MISS. No representation below ever divides by `d_y` or by any live
direction component that can vanish (each family divides only by its own
in-plane speed, whose vanishing is a measure-zero, categorically-handled
axis-parallel case — Section 6.3).

---

## 2. The queried object and what information it contains (Question 2)

### 2.1 The categorical field

`N` is not a smooth scalar field. On the 4-manifold `𝓛/Λ` the exterior
first-event map partitions line space into **visibility cells** on which the
owner (the piece of `∂𝒢` charted by the event) is constant and the depth is
an analytic function of the line; cell boundaries are silhouette,
occlusion-order, and disocclusion loci. This categorical structure is why

\[
\text{interpolating first depths}\;\not\Rightarrow\;\text{interpolating a
visible surface},
\]

as the prior work proved empirically; Lemma 3.3 below gives it a two-triangle
proof.

### 2.2 The generic information content is 5-dimensional and dense

For an **arbitrary** marked community the field has no exploitable invariance:

- the exterior field is a generic categorical function on a 4-manifold;
- the interior fibre (Fact 1.2) adds the ordered event sequence per line, with
  measured per-line event counts on the accepted asset of p95 `1,077` and
  maximum `4,481` within the horizon;
- the measured visibility-cell census on the accepted asset (Experiments 008,
  013, 014 in the central ledger) shows: corner-owner union per current-scale
  4D cell of median `6`, p95 `12`, max `16`; at least `202` distinct owners in
  one cell on a sparse interior probe; `66.98%` of camera/pixel plaquettes
  carry nontrivial sheet-permutation holonomy — there is no global small-`K`
  sheet structure to exploit;
- exact stored-record repetition is `1.81%` (owner-conditioned codebook
  audit); optimal rank-8 SVD oracles retain `59.99%` of coverage energy; rank
  48 needs twelve reads for `96.83%` agreement.

**Claimed dimension reductions must therefore come from geometry, not from
coding.** Section 5 classifies exactly which geometric invariances yield exact
reductions; the accepted arbitrary mesh has none of them (its own generator
places `270,535` independently posed primitive charts), which is why every
codec aimed at it failed. The proof obligations of the assignment's Question 2
are discharged in two parts: the reduction *for class `E`* is proved in
Theorem 6.4 (axial invariance ⇒ per-family 3-dimensional bake), and the
*absence* of a reduction for the arbitrary community is Proposition 3.5
together with the unconditional lemmas of Section 3.

---

## 3. Impossibility boundaries (theorems and counterexamples first)

Throughout this section, a **fixed-resource decoder** is: a table of at most
`S` bits; a query computes at most `R` record addresses (each address a fixed
function of the pointed ray and of previously read records — dependent reads
are permitted but count toward `R`), reads at most `R·B` bits, and applies a
fixed finite algebra to produce the answer. This models "O(1), no loop, no
traversal, no candidate list" exactly: `R`, `B`, `S` are constants frozen
before the query distribution is seen.

### 3.1 What must be reproduced

The decoder must return, for every pointed ray in the promised domain, the
coupled record of `N(ℓ,t₀)` — with the owner-correctness semantics that the
returned position lies on the true first-visible surface germ (not merely at a
plausible depth), because normal, colour, and mark are attached to it.

### 3.2 Lemma (fixed-`K` event records are insufficient — unconditional)

*Claim.* For every fixed `K` there is a community in the admissible class (and
the accepted asset realizes it) and an oriented line `ℓ` with `≥ K+1` events
`t_1<\dots<t_{K+1}`, such that any record storing at most `K` event atoms for
`ℓ` answers wrongly on an open set of origins.

*Proof.* Take `K+1` disjoint opaque blades crossing `ℓ` transversally (the
accepted asset's lines carry up to `4,481` events, so such lines exist for any
practical `K`). A `K`-atom record omits some `t_i`. For every origin
`t_0\in(t_{i-1},t_i)` the correct answer is `t_i`; the decoder, whose output
on line `ℓ` is a fixed function of the stored atoms and `t_0`, must return
either a stored atom (`\ne t_i`, wrong owner and wrong depth) or a value not
in `E(\ell)` (a point on no surface). Either way it is wrong on the whole open
interval. ∎

This is the distilled form of the measured Experiment 007 result (K=8 retains
`1.924%` of sampled event topology) and closes every "small deep record"
shortcut, including quantile sheets: a quantile is a selection among stored
atoms.

### 3.3 Lemma (cross-owner arithmetic is insufficient — unconditional)

*Claim.* No arithmetic combination (mean, weighted mean, median, quantile, or
any continuous symmetric function strictly between its extreme inputs) of
first-depth records belonging to different owners reconstructs a visible
surface across an owner change.

*Proof.* Two parallel opaque rectangles `A` at depth `a` and `B` at depth
`b>a`, with `A` occluding `B` on ray set `U_A` and `B` alone visible on the
adjacent ray set `U_B` (an ordinary silhouette). Records sampled on both sides
of the boundary carry depths `a` and `b`. Any strictly-between combination
returns `t\in(a,b)` on a neighbourhood of the boundary; the segment
`(a,b)` on those rays lies in empty space, so the returned point lies on no
surface, and no attached normal/colour/mark is defined for it. A
non-strictly-between selection is categorical election, which is not
arithmetic combination and is treated by Theorem 4.1. ∎

### 3.4 The measured point-location boundary (fixed-depth structures)

A fixed-depth decision structure (dependent reads, depth counted in `R`) is a
legitimate fixed-resource decoder, so Lemmas 3.2–3.3 do not exclude it. For
the accepted asset it is excluded by measurement, not by algebra, and this
document does not re-litigate it: one current-scale 4D cell requires at least
eight serial binary decisions before separator geometry (`202` owners on a
sparse probe); after one full 4D subdivision every subcell still exceeds four
decisions; and the explicit branch-list storage lower bound is `13,211,653`
owner incidences `≈ 52.8 MB` for ids alone, above the entire cap before any
geometry, flow, or attribute byte. (Ledger Experiments 008/011;
`GRASS-CHEAP-PROJECTIVE-RAY-MATH.md` §22.5.)

### 3.5 Proposition (conditional impossibility for the arbitrary community)

*Statement.* Let `Φ_ε` denote the exterior field of the accepted community
sampled at owner-coherence resolution `ε` (the resolution at which stored
records displace reconstructed events by at most `ε` along the declared
grazing domain). Any exact fixed-resource decoder yields a description of
`Φ_ε` of length `≤ S + O(1)` (the table plus the fixed decoder text). Under
Hypothesis **H** below, `S ≥ S_{\min}(ε) \gg 51,121,152` bytes, so no such
decoder exists within the frozen cap.

*The sampling arithmetic (exact, from the project's own bound).* The
phase-shear identity `‖Δq‖ = 2\,h\,r\,\sin(\pi/2N)` with `h = 1.175 m`,
`r=\cot 5^\circ` gives the azimuth counts

\[
N(\varepsilon{=}T{=}0.52\,\mathrm m)\;\ge\;82,
\qquad
N(\varepsilon{=}0.10\,\mathrm m)\;\ge\;422 ,
\]

and the matching elevation-row counts (slope steps `Δr ≤ ε/h`) are `≥ 26` and
`≥ 135`. A direct complete-record field at the current `256²` phase and
8-byte records then costs

\[
256^2\cdot(82\cdot26)\cdot 8 \;\approx\; 1.12\ \mathrm{GB},
\qquad
256^2\cdot(422\cdot135)\cdot 8 \;\approx\; 29.9\ \mathrm{GB},
\]

i.e. `≈ 22×` to `≈ 580×` the resident cap — before the fifth (interior)
coordinate, colour detail, or multi-species growth.

*Hypothesis H (incompressibility).* `Φ_ε` admits no encoding below the cap,
i.e. no `≥ 22×` owner-correct compression exists. This is an empirical
hypothesis, not a theorem — but it is the single most-tested hypothesis in
this project's ledger: exact record repetition `1.81%`; near-exact codebooks
save `≤ 23.4%`; the optimistic rank-8 SVD oracle fails at `59.99%` retained
energy; rank-48 linear needs twelve reads; K-plane, tetra-field, GDM-style,
flow, sheet, cascade, and light-field codecs all fail their gates by large
margins (Sections 6.1–6.14 of the exploration summary). A representation that
falsified **H** would have to find structure that twenty-five independent
attacks missed.

*Consequence.* Exactness for the arbitrary community must be abandoned in one
of three directions: (i) resources (forbidden), (ii) fidelity (rejected — the
wedge/stretch artifacts are exactly what "approximate between samples" looks
like at feasible rates, Theorem 4.1), or (iii) **the geometry class of the
authored source** — the only direction not frozen by the contract. Sections
5–7 take direction (iii) and prove it sufficient. ∎

---

## 4. The amplification dichotomy (why every rejected family failed)

The rejected families differ in decoders but share one property: the **camera
direction** enters through a *sampled* coordinate (nearest bin, four-corner
footprint, hashed corner, learned feature, direction lattice). The following
theorem shows this property alone produces the observed artifact class, and
that its negation — direction entering only *analytically* — removes the
range-amplification mechanism entirely.

### 4.1 Theorem (amplification dichotomy)

Let a reconstruction assign to each ray a record set determined by a
quantization of its direction with angular scale `Δθ` (any binning,
footprint-corner, or hashed variant), such that adjacent direction cells
evaluate independently baked fields whose reconstructed surfaces disagree by
world distance `≥ δ₀` somewhere in the viewed region (measured true for the
accepted asset at every tested `Δθ` down to `0.5°`). Then for a pinhole camera:

1. the image partitions into solid angular sectors of width `≈ Δθ`; a viewed
   ground region at range `r` renders inside one sector over a connected world
   width

\[
\boxed{\,w(r)\simeq r\,\Delta\theta\,}
\]

   with a geometry jump `≥ δ₀` at sector boundaries — coherent wedges whose
   **world** size grows linearly in `r`;
2. the artifact's **screen** size is `Δθ/θ_{\mathrm{pix}}` pixels,
   independent of `r`: it never becomes subpixel, at any distance, for any
   fixed `Δθ`. Feasible `Δθ` (16–81 directions) gives multi-degree sectors,
   i.e. the observed camera-front triangles, distance-growing stretch, and
   wrong-view sheets.

Conversely, let a reconstruction whose stored fields are indexed **only by
intrinsic coordinates** (periodic phase with texel `Δq`, in-plane mask angle
with step `Δω`) reconstruct the direction dependence **analytically** (the
direction enters the address and the lift as exact closed-form expressions).
Then the reconstruction error is bounded by

\[
\boxed{\,e \;\le\; C_1\,\Delta q + C_2\,\bar\rho(q,\omega)\,\Delta\omega\,}
\]

uniformly in camera position, where `ρ̄` is the local in-plane free path — a
**camera-independent, world-fixed** bound. Its screen size at range `r` is
`e/(r\,θ_{\mathrm{pix}})` pixels `→ 0`: intrinsic-quantized reconstructions
become *more* correct per pixel with distance.

*Proof.* (1) Direction bins pull back to solid sectors through the pinhole;
within a sector one coherent field renders; the boundary jump is the assumed
field disagreement; the subtended world width at range `r` is `2r\tan(Δθ/2)`.
(2) is the same statement divided by the pixel angle. For the converse: with
analytic direction dependence the only quantized inputs are `q` and `ω`;
exactness of the analytic lift means the reconstructed event is the true event
of the same field evaluated at `(q+O(Δq), ω+O(Δω))`; the event displacement
per unit `ω` is bounded by the free path along the ray in the mask plane, and
per unit `q` by the field's along-ray coherence (Lemma 6.2). Neither bound
contains the camera. ∎

**Corollary 4.2 (admissibility criterion).** A representation meets the
visual contract only if the camera direction enters its stored-field addresses
and its reconstruction **analytically**. Sampled-direction representations are
excluded *for every* `Δθ` reachable under the byte cap (Proposition 3.5 makes
the reachable `Δθ` explicit). This single criterion explains, mechanically,
the failure of Sections 6.1–6.9 of the exploration summary and the two live
rejections (§16, §17 of the status document), and it explains why the
published exact method (extrusion) and the `K=1` production sward — both
direction-analytic — never showed the wedge class.

The remainder of the document therefore asks the only question left: **which
geometry makes the direction dependence analytic?**

---

## 5. Classification of exact direction-analytic structures (Question 3)

### 5.1 The reduction mechanism must be a geometric invariance

**Lemma 5.1 (address-fibre lemma).** Let a representation assign to every
community `𝒢` in a class `𝒞` a baked field on a domain of dimension `≤ 3`,
with a single read at an address `a(ℓ)` that is a fixed function of the ray
(independent of `𝒢`), and a fixed decoder `φ` exact for all `𝒢 ∈ 𝒞` and all
exterior rays. Suppose `𝒞` is *detail-closed*: it is closed under adding
arbitrary additional geometry compatible with its defining constraint. Then
the fibres of `a` (1-parameter ray families sharing an address, which exist
because `dim 𝓛 = 4 > 3`) are carried by point maps `ψ` that preserve
membership in every `𝒢 ∈ 𝒞`; hence `𝒞` is invariant under a 1-parameter
transformation group whose orbits rule the fibres.

*Proof sketch.* Rays `ℓ₁, ℓ₂` in one fibre read the same record, so the
decoder determines the event of `ℓ₂` from the event data of `ℓ₁` by a
`𝒢`-independent rule. Exactness for all of a detail-closed class forces this
rule to be a pointwise correspondence `ψ_{ℓ₁→ℓ₂}: ℓ₁ → ℓ₂` mapping
`𝒢∩ℓ₁` onto `𝒢∩ℓ₂` order-preservingly for **every** member (otherwise add a
small blocker on one line but not the other and contradict exactness). A map
preserving membership in every member of a detail-closed class is a symmetry
of the class constraint itself; ranging over the fibre gives a 1-parameter
group. ∎

**Lemma 5.2 (Λ-compatibility forces translations).** A 1-parameter group of
affine maps that (i) preserves lines (so ray queries remain ray queries),
(ii) commutes with every lattice translation in `Λ` (so one bake serves all
copies — the one-query-covers-all-copies architecture), and (iii) has
straight-line orbits (so fibres of `a` are ray families) is conjugate, by a
lattice-compatible affine change of basis, to the group of **translations
along a fixed axis** `e`. Consequently the invariant communities are exactly
the **parallel extrusions** `𝒢 = A(M × ℝ)` with `A` affine, i.e. Sannikov's
exact class including oblique/sheared TBN images.

*Proof sketch.* Line preservation on all of ray space plus affinity is given;
commuting with all translations of the rank-2 horizontal lattice forces the
linear part to fix the lattice plane's directions; the 1-parameter subgroups
with straight parallel orbits are (conjugates of) translation flows; the
non-nilpotent options (scalings in the transverse coordinate) have curved or
non-parallel orbit families whose orbit through a query line is not a line
family readable at one address. ∎

**Lemma 5.3 (projective/tapered instancing breaks periodicity —
unconditional).** Let `π` be projective and not affine. Then conjugates
`π τ_λ π^{-1}` of lattice translations are not translations, so a per-tile
`π`-tapered copy family is not the `π`-image of any single Λ-periodic set,
and no single periodic bake answers rays crossing many tiles.

*Proof.* Translations are exactly the projective maps fixing the plane at
infinity pointwise. `π τ_λ π^{-1}` fixes `π(plane at infinity)` pointwise; for
this to be the plane at infinity for all `λ` requires `π` to preserve the
plane at infinity, i.e. `π` affine. ∎

Exact **taper is therefore unavailable** in the one-query architecture; only
affine lean/shear/anisotropic scale is. (This re-derives, as algebra, the
quotient-proof rejection recorded for the four-slab experiment.)

**Lemma 5.4 (per-element axial intervals destroy the reduction —
unconditional).** Extend a mask `M` with per-point axial intervals
`[h_0(u,v), h_1(u,v)]` (individual fibre ends). The hit condition along a ray
couples the 2D mask crossing to the query's axial offset and slope. For any
fixed rank `m` there is a configuration (choose `m−1` early mask crossings
whose intervals exclude the ray's height and an `m`-th that includes it) whose
first eligible event has 2D-crossing rank `m`. Hence no fixed number of 2D
first-hit reads suffices; restoring exactness requires re-adding the two
query dimensions the extrusion removed. ∎ (Distilled Experiment 017.)

**Lemma 5.5 (per-tile analytic elements are not fixed-cost —
unconditional).** Let each tile carry a bounded-degree analytic element
(e.g. a quadric cap) and let a ray cross tiles with phase increment `δ` per
tile. The hitting tile index is `n(q,δ)=\min\{n: q+nδ \bmod T \in
H\}` for the per-tile hit set `H`. For `\mathrm{area}(H)<\mathrm{area}` of the
cell, `n` is unbounded over `q` (for `N < 1/\mathrm{frac}(H)` the union
`\bigcup_{n\le N}(H-nδ)` cannot cover the torus), so no fixed enumeration of
tiles answers the query, and no fixed algebra at `𝒢`-independent addresses
computes `n` in general (it is a first-passage functional; closed structure
exists only in special 1-D interval cases — three-gap — not for general `H`).
Per-instance analytic elements are admissible only when their *first-passage
answer itself is precomputed* — which is precisely what a periodic bake does
offline. ∎

### 5.2 The admissible structures, enumerated

Combining Corollary 4.2 with Lemmas 5.1–5.5 against the assignment's candidate
list:

| candidate structure | verdict |
|---|---|
| line-preserving/projective transform of one extrusion field | affine only (Lemma 5.3); admissible as the family transform |
| bounded union of exact line-preserving botanical families | **admissible** — the union is fixed-cost by `t_\cup=\min_f t_f` for a compile-time family count (Section 6.6) |
| categorical visibility-cell decomposition / fixed-depth point location | excluded for the arbitrary asset by measured depth and byte bounds (Section 3.4) |
| analytic representation of the ordered intersection set | per family, yes: the 2D bake plus the coherence lemma yields any needed successor; the renderer needs only first-after-origin (Fact 1.2) |
| associative periodic-cell successor operator with closed composition | reduces to the first-passage problem of Lemma 5.5; its closed form **is** the periodic bake, already inside the admissible family |
| a stronger new invariant | Lemma 5.1 says any single-read 3D-bake invariant is a 1-parameter geometric invariance; Lemma 5.2 classifies those compatible with `Λ` as extrusions. For multi-read decoders the lemma weakens, but Corollary 4.2 still excludes every sampled-direction use of the extra reads; extra reads are therefore spent on more *families*, not more directions |

**Conclusion (Question 3).** The exact structure that makes O(1) possible is:
**a bounded union of affine-sheared, Λ-compatible parallel-extrusion
families**, each with family-global clips, extended by (i) band-quantized
per-texel fibre tops (Section 6.5 — exact, via family splitting) and (ii)
extruded **extinction media** for sub-pixel detail (Section 7 — the same
reduction applied to integral functionals). Nothing else survives both the
dichotomy and the periodicity constraints under the frozen budget.

---

## 6. The class `E` construction and its exactness proofs

### 6.1 Definitions

**Opaque family.** An opaque family `f` is the data
`(A_f, M_f, \hat\Lambda_f, [b^-_f, b^+_f], \tau_f, \text{attr}_f)`:

- `A_f(x) = L_f x + c_f` affine, invertible, with **lattice compatibility**
  `L_f \hat\Lambda_f = Λ`: the mask-plane lattice maps onto the world lattice
  (so world periodicity is exact). The extrusion axis `a_f = L_f e_h` is
  arbitrary (vertical, leaning, or strongly tilted);
- `M_f ⊂ ℝ²_{(u,v)}` a closed `\hat\Lambda_f`-periodic mask of arbitrary
  positional detail (the authored cross-sections of every fibre of this
  directional class, all species mixed — Section 6.6);
- family-global clip slab `b^-_f ≤ y ≤ b^+_f` in world space (horizontal
  planes are the only Λ-invariant clips — a half-space is Λ-invariant iff its
  normal is vertical);
- optional per-texel **top band** `\tau_f(u,v) ∈ \{1..B\}` quantized to `B`
  horizontal levels (Section 6.5);
- `attr_f`: per-mask-texel coupled attributes — full 3D unit normal (authored;
  may include an axial component), authored colour, species/material mark —
  plus a family-global axial ramp `c_f(y)` (colour/AO modulation along the
  fibre, fixed closed form).

**The community.** `𝒢^\* = \bigcup_{f=1}^{K_o} G_f \;\cup\;` volumetric
families of Section 7, optionally duplicated over `L ∈ \{1,2\}` anti-tiling
layers, each layer being one **globally affine** re-instancing of the same
family data (one global rotation/phase per layer — per-cell re-instancing is
excluded by Lemma 5.3's logic and was already the committed production
choice).

**The bake (per family).** The offline field on `𝕋²_{\hat\Lambda_f} × S¹`:

\[
\rho_f(q,\omega)=\inf\{\ell\ge 0:\ q+\ell\omega\in M_f\}\in[0,\rho_{\mathrm{cap}}]\cup\{\text{MISS}\},
\]

stored with the attributes of the entered boundary point, at texel sizes
`(Δq, Δω)`. Conventions: `ρ=0` iff `q ∈ M_f` (occupancy sentinel); MISS iff no
entry within `ρ_cap` (the horizon image `R_max·s_p` capped). The bake is an
exact offline 2D raycast of the **infinitely repeated** mask: the offline cost
absorbs the first-passage problem of Lemma 5.5 once, for all copies.

### 6.2 The two exactness lemmas of the 2D field

**Lemma 6.1 (bake exactness).** By construction, `ρ_f(q,ω)` is the exact
first-entry path of the 2D ray `(q,ω)` against `M_f`, for **every** `q ∈ 𝕋²`
— boundary and interior origins alike. (The bake domain is the whole torus;
"entry phase" plays no privileged role.)

**Lemma 6.2 (along-ray coherence / successor property).** For
`0 ≤ ε < ρ_f(q,ω)`:

\[
\rho_f(q+\varepsilon\omega,\ \omega)=\rho_f(q,\omega)-\varepsilon,
\]

and for `ε` just past an entry, `ρ_f(q+εω, ω)` is the exact distance to the
**next** entry. *Proof:* translate the infimum. ∎

These two lemmas are the entire correctness core: every 3D property below is
an affine conjugation of them.

### 6.3 The reconstruction algebra (per opaque family)

Given the pointed world ray `(o, d)` (`‖d‖=1`, origin at the camera or at the
carrier datum — the algebra is origin-agnostic by Theorem 6.7):

1. **Affine transfer:** `\hat o = A_f^{-1}(o)`, `\hat d = L_f^{-1} d`. The
   parameter `t` is shared (unnormalized `\hat d`).
2. **In-plane split:** `p = (\hat d_u, \hat d_v)`, `s_p = ‖p‖`,
   `ω = p/s_p`.
3. **Clip:** intersect `t`-intervals of the world slab(s):
   `[t_{\mathrm{in}}, t_{\mathrm{out}}]` (empty ⇒ family MISS);
   `t_\star = \max(t_{\mathrm{in}}, 0)`.
4. **Query:** `q_\star = (\hat o_u, \hat o_v) + t_\star\,s_p\,\omega` (the
   origin's own phase advanced to the clip entry — for an interior origin,
   `t_\star = 0` and the address **is the origin's own phase**);
   read `\rho = \rho_f(q_\star, ω)` (one texture read; address analytic in
   `(o,d)`).
5. **Lift:** family event parameter

\[
\boxed{\;t_f = t_\star + \frac{\rho}{s_p}\;}
\qquad\text{accepted iff } t_f \le t_{\mathrm{out}} \text{ (else MISS)} .
\]

6. **Attributes:** if `ρ = 0` and the ray enters through a clip plane, the
   event is a **cap hit**: normal is the cap plane normal, other attributes
   from the texel. Otherwise the texel's authored normal
   `n = L_f^{-T}\hat n / ‖\cdot‖` (exact affine normal transport — a
   constant per family, foldable), colour `= \text{attr}\cdot c_f(y(t_f))`,
   mark from the texel. All attributes come from the single entered texel:
   coupling is structural, not enforced.

**Axis-parallel pole (categorical, exact).** If `s_p = 0` (the ray is parallel
to this family's axis — one direction per family, measure zero): the event is
`q_\star ∈ M_f` ? cap hit at `t_\star` : MISS. One occupancy read of the same
texture; no epsilon; and the generic formula converges to this value as
`s_p → 0` (if `q_\star \notin M_f`, `ρ/s_p → ∞ > t_{\mathrm{out}}` ⇒ MISS —
the two charts agree in the limit, so the chart switch is invisible).

**Horizontal and vertical world rays are ordinary.** A vertical-axis family
queried by an exactly horizontal ray has `s_p = 1` (maximal in-plane speed):
the historically fatal case is this representation's *best-conditioned* case.
An exactly vertical ray hits the vertical family's pole chart (occupancy —
exactly the "top-down filled footprint" semantics already validated) and is a
generic query for every tilted family. No live division by `d_y` exists
anywhere in the algebra.

### 6.4 Theorem (family exactness and dimension reduction)

For every pointed ray and every opaque family, the algebra of 6.3 returns
exactly the first entry event of `G_f` after the origin (or MISS), reading one
record of a **3-dimensional** field. The reduction from the generic
5-dimensional pointed field to `(𝕋²×S¹)` is exact, with proof: membership in
`G_f` is invariant along `a_f` (axial invariance), so membership along the ray
depends only on the projected 2D ray; the clip is exact by linearity of
half-spaces in `t`; firstness and the interior-origin (successor) semantics
are Lemmas 6.1–6.2 conjugated by `A_f`; the parameter identity
`t = t_\star + ρ/s_p` is the extrusion lift (`|OB| = |OA|/\cos α` in
orthonormal frames). ∎

This discharges the assignment's Question 2 proof obligation for the class:
the claimed dimension reduction has an invariance proof, per family.

### 6.5 Banded fibre tops (exact per-texel end heights at `B` levels)

Per-fibre end heights are required for botanical silhouettes but per-texel
axial intervals are excluded by Lemma 5.4. The exact repair is **family
splitting by top band**: quantize authored tops to `B` levels
`b^-_f < y_1 < \dots < y_B = b^+_f` and define sub-family `f_b` = slab
`[y_{b-1}, y_b]` with sub-mask `M_{f,b} = \{τ_f ≥ b\}` (only fibres reaching
band `b` exist there). Each sub-family is an exact opaque family; the union
over `b` is exactly the banded-top solid. Storage: one shared texture with `B`
`ρ`-channels (`ρ` against each `M_{f,b}`); runtime: the ray traverses at most
the bands its clip interval meets, in a **fixed unrolled order** (`≤ B`
compares), reading the band channels at `≤ B` analytically known phases
(distinct addresses, independent reads). `B` is a compile-time constant
(`B = 2..3` suffices for blade-length variation); there is no data-dependent
iteration. Exactness is inherited from Theorem 6.4 applied `B` times. ∎

This meets, constructively, the recorded resume condition of the four-slab
rejection: no filled height intervals (band masks thin out with height
exactly as authored), Euclidean periodicity proved (Lemma 5.2), camera-inside
successors proved (Theorem 6.7).

### 6.6 Community composition, species overlap, moss (exact)

**Theorem (union composition).** `E(\ell) = \bigcup_f E_f(\ell)` and therefore

\[
\boxed{\;N(\ell,t_0)=\min_f N_f(\ell,t_0)\;}
\]

with the winning family's coupled record. For a compile-time family count this
is a fixed select network; visibility-order swaps between families are exact
(min of exact depths). ∎

**Species do not multiply cost.** A species is not a family; a *directional
class* is. All species whose fibres share an axis class live in the **same
mask** with per-texel marks (grass blades of `Agrostis` and `Avenella` and
sedge culms are one vertical family with different texels). Mutual occlusion
across species is exact through the union theorem. The runtime therefore has
**no per-species term at all**; the control field selects which authored
mask set (community state) a root cell uses, *before* the query, exactly as
the committed type-keyed architecture already does. Live-changing Boolean
mixtures remain outside the contract (a finite catalogue of authored mixture
states is the supported form), unchanged from the prior analysis.

**Moss.** Two-scale split, each part exact in its own system: (i) cushion
*form* (hummock domes, 10–30 cm) is **terrain microtopography** — it belongs
in the cook's LOD−2 height layer (0.0625 m grid), where it is rendered by the
terrain path exactly, not by the cover query at all; (ii) moss *texture*
(dense vertical shoots, capitula) is a short dense vertical opaque family
(slab `[0, ~5 cm]` above local ground) plus, for capitula fuzz, a thin
volumetric band (Section 7). A low moss tuft beneath taller blades is then
exact mutual occlusion via the union theorem. This resolves the moss question
by moving the dome where an exact renderer for it already exists — the
context-level correction the geometry-sphagnum and analytic-cap attempts were
missing.

### 6.7 Theorem (oriented-line invariance and camera-inside succession)

For every family (hence for the composite):

1. **Line invariance.** Replacing the query datum `o ↦ o + λd` (same oriented
   line, any `λ` not crossing the selected event) changes `\hat o` by
   `λ\hat d`, hence `q_\star` by `λ' s_p ω` along the 2D ray; by Lemma 6.2 the
   returned world event is **identical**. The atlas address depends only on
   the oriented line and the clip entry, never on the datum. (This is the
   invariant whose violation was proven for the rejected moving carrier; here
   it holds by construction.)
2. **Forward succession.** For an origin inside the cover, moving forward
   without crossing `N(\ell,t_0)` leaves every family's `t_\star`-phase before
   its first entry, so by Lemma 6.2 all family answers, and their min, are
   unchanged: the same world interaction is returned.
3. **Crossing.** Immediately after the origin crosses the winning event, the
   winning family's query point lies past its entry; Lemma 6.2 returns that
   family's **next** entry, and the min over families is the community's next
   valid interaction. No other family's answer changed.

∎ This is precisely the successor property demanded by Question 6, proved
rather than assumed, and it costs nothing: the interior origin's own phase is
the address. An exterior top-plane record plays no role.

### 6.8 Theorem (periodicity and tile boundaries)

The bake raycasts the infinitely repeated mask, phases wrap on the torus, and
lattice compatibility `L_f\hat\Lambda_f = Λ` makes world periodicity exact:
one query serves every copy; a ray crossing any number of periodic cell
boundaries is a single lookup with **no seam events of any kind**. Anti-tiling
layers are separate global affine instancings; the cross-layer composite is
the union theorem again (exact). ∎

### 6.9 The target theorem (assembled)

**Theorem.** Let `𝒢^\*` be authored in class `E` (opaque families with bands,
volumetric families of Section 7, layers, marks). The fixed algebra
`Ψ` of Sections 6.3–6.6 and 7.3 returns, for **every** pointed ray in the
promised domain — every direction including exactly horizontal and exactly
vertical, every origin including inside the cover, across all periodic
copies — the exact coupled first interaction of `𝒢^\*` after the origin
(successor semantics), with:

- attachment and firstness exact (Theorem 6.4 + union theorem);
- oriented-line invariance and camera-inside succession exact (Theorem 6.7);
- discontinuities of `Ψ` in ray space **coinciding with true visibility
  boundaries of `𝒢^\*`** (silhouettes, occlusion-order changes,
  disocclusions) — no reconstruction-created boundary exists;
- resource use a compile-time constant (Section 9);

up to exactly the quantization and declared-model errors bounded in Section 8
(each camera-distance-independent) and the finite horizon `R_max`. ∎

---

## 7. Volumetric (extinction) families: exact sub-pixel plume detail

The panicle plume (`0.12–0.22 mm` hairs, `203,258` filaments) must not be an
opaque event soup — that is the measured source of the event-count lower
bounds. The accepted §34 source gate already proved the right object: the
plume is authored as a **participating medium attached to the structure**. In
class `E` this becomes exact, because the extrusion reduction applies not only
to first-hit times but to **along-ray integral functionals** of axially
invariant fields.

### 7.1 Definition

A volumetric family `g` is `(A_g, κ_g, c_g, [b^-_g, b^+_g])`: an affine
lattice-compatible map, a periodic 2D **extinction density** `κ_g(u,v) ≥ 0`
(per unit world length), a coupled premultiplied colour field, and a clip
slab. The authored truth `𝒢^\*` *defines* this component as a medium: the
representation is then exact with respect to the authored object (no "average
of events" reinterpretation occurs — the medium is the ground truth, chosen
at authoring time for structures that are sub-pixel at all accepted viewing
ranges).

### 7.2 The exact integral lift

For an axially invariant density, the optical depth of the world ray from its
clip entry is

\[
\tau(t)=\int_{t_\star}^{t}\kappa_g\bigl(uv(t')\bigr)\,dt'
=\frac{1}{s_p}\int_{0}^{(t-t_\star)s_p}\kappa_g(q_\star+\ell\omega)\,d\ell ,
\]

— the same `1/s_p` lift as the first-hit identity, applied to an integral.
The bake stores, per `(q, ω)` texel, the **convergent** rendered functionals
from that phase forward through the clipped slab:

\[
\mathcal V_g(q,\omega)=
\Bigl(
A=1-e^{-\tau_{\mathrm{tot}}},\quad
C_p=\!\int c\,\kappa\,e^{-\tau(\ell)}d\ell,\quad
\mu=\!\int \ell\,\kappa\,e^{-\tau(\ell)}d\ell \,/\, \text{norm},\quad
\sigma
\Bigr),
\]

i.e. total opacity, premultiplied in-scattered colour, and the mean and spread
of the interaction depth **within this family alone**. All entries converge
(exponential damping), are smooth in `(q,ω)` — so ordinary trilinear
filtering of `𝒱_g` is *valid*, unlike event fields — and lift to world
parameters by `ℓ = (t−t_\star)s_p`. One read per volumetric family.
Camera-inside is again free: the origin's own phase is the bake point
(Lemma 6.1's analogue for integrals), so walking through the plume is exact.

### 7.3 Composition with opaque events, and its single bounded approximation

Front-to-back over the fixed family set: sort the `≤ K` opaque event
parameters and volumetric mean parameters (fixed network); composite

\[
C=\sum_{\text{front to back}} T_{\text{acc}}\,C_i,
\qquad
T_{\text{acc}}\mapsto T_{\text{acc}}(1-A_i),
\]

with an opaque event contributing `A=1` and truncating the sum. The **only**
approximation in the entire construction's compositing is the *straddle
case*: an opaque event landing strictly inside a volumetric family's
interaction span is composited by the lump ordering `μ_g` vs `t_f` instead of
by the exact partial integral.

**Lemma 7.1 (straddle bound).** The compositing error in alpha/colour is at
most `A_g ·` (the fraction of the family's interaction mass on the far side
of the opaque event), and the affected world region has the size of the
plume's own span (`σ_g`, centimetres): a local, camera-distance-independent
softening exactly where blades pierce plume clouds. An optional two-lump
record (`(A, C, μ)` split at the median) halves the bound at one extra
channel, no extra read. ∎

The depth/normal contract is unchanged by volumetrics: geometric election
uses the winning **opaque** event where one exists in front of the plume
bulk; a plume-only pixel elects `μ_g` with declared spread `σ_g` bounded by
the family's slab span — never by the whole community's depth range (this is
what separates it from the rejected global event-measure codec: moments are
confined to one authored sub-pixel medium, not used to collapse distinct
plants).

### 7.4 Opaque↔volumetric duality under minification (the LOD law)

Coarse mips of an opaque family's mask are fractional-coverage fields — i.e.
volumetric densities. Define the family's mip chain so that fine levels are
event records (`ρ`, attributes) and coarse levels reinterpret as
`𝒱`-records (density from mask coverage, colour premultiplied). The
transition is the *mathematically forced* one: below pixel scale, the exact
categorical field is not resolvable and its band-limited object **is** the
extinction medium (Section 8.2 makes the footprint semantics precise). This
gives the distance filtering law with no new representation: one family, one
texture, mip level from the intrinsic footprint, event semantics near,
medium semantics far, and the switch is per-family, per-pixel, fixed cost.

---

## 8. Quantization, filtering, and every residual error bound (Question 7)

Class `E` is exact as algebra; its stored fields are sampled. This section
states **every** error term, each with its bound and its scaling. None grows
with camera distance — by Theorem 4.1 that property is what separates
"quantization" from "artifact class".

### 8.1 Intrinsic quantization

- **Phase texel `Δq`:** by Lemma 6.2 the field is 1-Lipschitz along `ω` and
  categorical across owners; nearest-phase addressing yields world error
  `≤ Δq·(1 + 1/s_p·‖\text{axial}\|)` — sub-texel (`≈ 2 mm` at `256²` over
  `0.52 m`), the same magnitude the current accepted sward exhibits.
- **In-plane angle step `Δω`:** the reconstructed event is the exact event of
  the same mask at a rotated in-plane direction; displacement
  `≤ ρ̄(q,ω)·Δω` where `ρ̄` is the local free path. For dense sward masks
  `ρ̄ ~` centimetres ⇒ millimetre error at `Δω = 2π/256`; for sparse cover the
  bound degrades with `ρ̄` up to `ρ_cap` — the Nyquist rule
  `Δω ≤ ε/ρ̄_{p95}` is the **per-family bake gate**, measurable offline from
  the mask's own free-path statistics (unresolved constant, Section 10, but a
  formula, not a search). Critically, the pixel footprint also grows with the
  same in-plane distances, so the `ρ̄Δω` error stays a fixed *fraction* of
  footprint — bounded artifact-to-signal ratio at all ranges, unlike `rΔθ`.
- **`Δω`-boundary categorical splices:** adjacent `ω`-texels are coherent
  fields of the *same* mask differing by `Δω` rotation; their disagreement is
  local (free-path scale) — no camera-centred sectors are possible because
  `ω` is a line coordinate, not a camera coordinate (dichotomy converse).
- **Attribute quantization:** ordinary texel precision on normal/colour/mark;
  categorical mark never filtered (nearest / winner-only).

### 8.2 Pixel-footprint filtering (the declared ray distribution)

The filtered query is the pixel's ray bundle. Pull the footprint back through
`A_f^{-1}` and the `1/s_p` lift to an intrinsic mask-plane footprint (closed
form); choose the mip level matching it. Semantics per Section 7.4: while the
footprint is below fibre cross-section scale, the categorical event record is
exact per ray and antialiasing is the resolve's concern (single coupled
record per ray — no invented owners); once the footprint exceeds fibre scale
the family's own coarse-mip volumetric record **is** the band-limited truth:
its `(A, C_p, μ, σ)` are the moments of the declared ray distribution over
the footprint, coupled by construction (premultiplied within one family,
composed across families by Section 7.3). An event mean is never presented as
a crisp surface: representative depth carries its declared spread `σ`,
bounded by the family span, and election prefers the front opaque event
whenever one is resolvable. This is the precise answer to Question 7:
near-field exactness and footprint filtering are different *mip levels of the
same family data*, with a proved switch scale, not different algorithms.

### 8.3 Composition-order error

Only the straddle case of Lemma 7.1 (bounded, centimetre-scale, local).
Opaque-vs-opaque composition is exact at all scales (min of exact depths).

### 8.4 Terrain curvature

The construction lives in the ground chart (honest derivative basis `J`,
per-guide-texel affine). On curved terrain the affine chart's top-entry error
has the known leading term `−½\,Δx^{T}\mathrm{Hess}(g)\,Δx`; with tile size
`h_t` and curvature bound `‖Hess‖ ≤ κ_T`, the bound is `½κ_T h_t²` — a
**cook-side gate** (bound `κ_T h_t²` per grade level), not a runtime term.
This retains the §5.2 result of the exploration summary and disposes of it as
an explicit authoring constraint.

### 8.5 What has no error term

Direction dependence (analytic — zero); periodic continuation (exact); chart
selection (categorical, agreeing on overlap); camera-inside succession
(exact); species overlap (exact); layer composition (exact); horizon
truncation (categorical MISS, declared).

---

## 9. The fixed runtime algebra and cost model (Question 8)

Symbolic only; no implementation is proposed here.

**Per opaque family-band read (all addresses analytic in `(o,d)`):**

| step | ops (FMA-equivalent) |
|---|---|
| affine transfer `\hat o, \hat d` (linear parts foldable per family) | ~18 |
| in-plane split `p, s_p, ω` (one `rsqrt`, one `atan2`-free octahedral angle map) | ~8 |
| slab clip interval + `t_⋆` | ~6 |
| address `q_⋆` + wrap | ~6 |
| lift `t_f = t_⋆ + ρ/s_p`, accept test | ~4 |
| attribute decode (oct normal, colour ramp `c_f(y)`) — winner only | ~10 |

≈ 42 FMA + **one independent texture read** per family-band; volumetric
families identical with the `𝒱` decode (~12 FMA, one read). Composition:
sort/select network over `n = K_o·B + K_v` records ≈ `O(n log n)` compares
(fixed, unrolled), ~30–40 ops for `n ≤ 8`.

**Resident representation variables** (uniform block, per profile): per
family `A_f^{-1}` (12 floats), slab bounds, band levels, ramp coefficients,
layer transforms — a few hundred bytes; no per-copy, per-species, or
per-triangle state of any kind.

**Reads.** All reads are **independent** (addresses never depend on fetched
data — issuable in parallel, unlike every point-location alternative) and
coherent (neighbouring pixels address neighbouring texels within each
family). Example configuration (illustrative, to show the budget closes; the
frozen numbers are fixed at transcription time):

| family | role | texture | bytes |
|---|---|---|---:|
| vertical blades, `B=2` bands | opaque | `256²·32ω·8 B` | 16.8 MB |
| two lean-class blade families | opaque | `2 × 128²·32·8` | 8.4 MB |
| culm/stem + forb-head band | opaque | `2 × 128²·32·8` | 8.4 MB |
| panicle/capitula plume | volumetric | `128²·32·16` | 8.4 MB |
| moss shoot family | opaque | `128²·32·8` | 4.2 MB |
| **total** | | | **46.2 MB ≤ 51.1 MB** |

Anti-tiling layers **reuse the same textures** under a second global affine
(zero bytes, extra reads). Per-pixel reads for this example: `7` family-band
reads `× 2` layers `= 14`, worst-case mixture boundaries `×2` by the existing
A/B root-stable gate. This is the same order as the currently accepted
runtime (the committed Sphagnum path performs 8 taps; the accepted grass
checkpoint measured `0.39–0.59 ms` at DPR2 with comparable traffic), and it
is **independent of species count, copy count, geometric depth complexity,
and camera distance**. Why no loop/march/traversal/candidate exists: the
family count is a compile-time constant of the authored profile; every read
address is a closed-form function of `(o,d)`; composition is a fixed select
network; there is no data-dependent control flow anywhere in the algebra.

The historical "`≤4` reads / `≤256` FMA" freeze applied to codecs of the
sampled 4D field; this construction replaces that architecture, and its own
budget is declared above and anchored to the measured envelope of the already
accepted paths. Final numbers are frozen before any transcription, per the
standing contract.

---

## 10. The counterexample battery, applied to this construction

The assignment's eight attack cases, evaluated against class `E` — including
the honest failures:

1. **Two surfaces whose visibility order swaps.** Two families (or two mask
   components): both queries exact; `min` composes; the swap locus is a true
   visibility boundary of `𝒢^\*`, reproduced exactly (Theorem 6.9). **Exact.**
2. **A thin tilted blade.** Exact *iff* a family axis matches its tilt class.
   A 30°-tilted blade forced into a vertical family is a vertical prism — the
   construction does **not** approximate out-of-class geometry gracefully; it
   requires the blade to be authored into one of the `K` axis classes. This
   is the applicability boundary (below), not a reconstruction error.
3. **A branched panicle.** Culm family + 1–2 branch-tilt families + the
   volumetric plume of Section 7. The §34 source gate is direct evidence this
   split preserves the accepted silhouette. Individual mid-scale branchlets
   that are neither sub-pixel (plume) nor axis-aligned (family) are the
   hardest authored content; they must be quantized to the available tilt
   classes or moved into the plume. **Exact w.r.t. the authored `𝒢^\*`;
   fidelity of `𝒢^\*` to free-form reference is the authoring gate.**
4. **Two overlapping species.** Same masks, per-texel marks, union theorem.
   **Exact**, zero marginal runtime cost.
5. **A low moss tuft beneath taller blades.** Moss family under grass
   families; `min` gives exact mutual occlusion; hummock form is terrain.
   **Exact.**
6. **An origin between two intersections.** Theorem 6.7(2–3): the origin's
   own phase is the address; the answer is the true next event and changes
   only when crossing it. **Exact.** (Degenerate sub-case: origin strictly
   inside solid matter (`ρ=0` at own phase) returns an immediate event —
   declared semantics for a camera inside a blade.)
7. **An exactly horizontal ray.** Best-conditioned case for vertical
   families (`s_p = 1`); no top-plane chart exists anywhere in the algebra;
   answer is the first event within `R_max`, else categorical MISS.
   **Exact.**
8. **A ray crossing a periodic cell boundary.** Theorem 6.8: seamless by
   construction of the infinite-mask bake. **Exact.**

Additional self-attacks (found while deriving, disclosed):

- **Fibre tips.** Without bands, all fibres of a family end at its clip
  plane (flat-cut tops). Bands (6.5) restore per-texel tops at `B` quanta;
  taper *within* a tip is inexpressible exactly (Lemma 5.3) and must be
  authored as banded steps plus plume softening. Distance-independent,
  tip-scale (centimetres) limitation — declared.
- **Sparse-cover grazing free paths.** `ρ̄` can approach `ρ_cap` in sparse
  masks, weakening the `Δω` bound (8.1). Mitigation is a bake gate
  (`Δω ≤ ε/ρ̄_{p95}` per family) — measurable, not searchable; worst case
  forces more `ω` slices for sparse families (bytes, known formula, no new
  representation).
- **Wind.** A time-varying **affine** shear per family/layer is exact
  per frame (lines map to lines; the bake is queried in sheared frame). The
  currently authored non-affine arc terms (`arcK`) remain approximations if
  re-enabled — unchanged status, now with its reason on record.
- **Continuously varying lean across a patch.** Exact only piecewise (lean is
  a family constant per region boundary at authoring); continuous world-space
  lean fields would break line preservation (they bend query lines). Lean
  variation must ride on (a) discrete lean families, (b) per-layer global
  affines, or (c) the terrain chart `J` (which *is* allowed to vary — it is
  the one sanctioned position-dependent transform, with the 8.4 curvature
  gate). This is the sharpest expressiveness limit of the class.

## 11. Relation to every prior rejection (why this is not a renamed retry)

| rejected family (ledger) | explained/superseded by |
|---|---|
| four-view smooth interpolation; hit-height carry; PCF quantiles; direct radiance 4-read; direction lattices; plenoptic cascade; packed light fields | Theorem 4.1 (sampled camera direction ⇒ `rΔθ` sectors, never subpixel) + Lemma 3.3 |
| nearest/MAP angular selection; hashed splice; orthogonal moving carrier | Theorem 4.1 + line-invariance requirement, both now held by construction (Thm 6.7) |
| fixed-`K` event lists; band-limited event sheets; event-measure moments (community-wide) | Lemma 3.2; moments now confined to single authored media (Section 7), where they are the ground truth, not a collapse |
| low-rank / learned / pairwise-plane / tetra-field / GDM-rank codecs; owner codebooks | Proposition 3.5 (H): they were attacks on an incompressible object; class `E` removes the object instead |
| Plücker separators; owner/successor closure; procedural charts | Section 3.4 measured bounds; provenance now lives in masks, where it is the *address*, not a lookup result |
| four affine slabs (Exp 015) | in-class ancestor, wrong use: a 4-family *compressor of an out-of-class mesh*. Its recorded resume conditions (no filled intervals; proved periodicity; camera-inside successors; declared budget) are each met constructively (6.5, 5.2/6.8, 6.7, 9) |
| finite swept fibres (Exp 017) | Lemma 5.4 is its distilled proof; banded tops are the exact repair within the class |
| cell-exit factorization (retained identity) | subsumed: interior origins query their own phase; no boundary-suffix field is needed (Thm 6.7) |
| tiny sweep unions / projective taper | Lemma 5.3 (taper ⊥ periodicity) — affine-only, as authoring constraint |
| Sannikov's unpublished "PCF-like" rule | no longer load-bearing: the exact demos are extrusion-class (direction-analytic); the arbitrary-mesh experiments visibly stretch by his own account. The missing magic was never a filter; it was the class. Nothing is attributed to the author beyond the published extrusion theorem, which is Theorem 6.4's core |

The two positive artifacts fit the same table: the accepted `K=1` production
sward (parallel extrusion, `shiftK=thickK=arcK=0`) is the smallest member of
class `E`; the §34 structural/plume source split is the Section 7
decomposition, measured to preserve the accepted silhouette.

## 12. Assumptions, applicability boundary, unresolved lemmas, verdict

**Standing assumptions.** Λ-periodic community with finite horizon
`R_max = 155 m`; ground chart affine per guide texel with cook-gated
curvature (8.4); anti-tiling as global per-layer affines; authored mixture
states finite (no live Boolean composition); camera-in-solid-matter
degenerate semantics as declared (10.6).

**Applicability boundary (the honest core).** Class `E` trades unlimited
*positional* detail (masks are arbitrarily detailed, per-texel marked) for
bounded *directional* diversity: `K` axis classes, `B` top bands, affine-only
lean/taper, media for sub-pixel structure. The arbitrary reference mesh is
**not representable**; the community must be authored (or its existing
generator re-targeted — it is already procedural and in-project) inside the
class. Whether `K ≈ 4–7` directional classes suffice for
indistinguishable-from-real Estonian ground cover is an **authoring/visual
question, not a mathematical one** — it is the single remaining gate, and the
`K=1` acceptance plus the §34 plume evidence are its favourable priors.

**Unresolved lemmas (all bounded, none blocking the verdict):**

1. sharp `Δω` constants need each authored mask's free-path statistics
   (formula in 8.1; offline measurement, not search);
2. Lemma 5.1's classification is proved for single-read decoders;
   for multi-read decoders necessity rests on Theorem 4.1 plus
   Proposition 3.5 rather than on a full algebraic classification —
   a stronger multi-read classification theorem is open but not needed;
3. optimal band count `B` and lump count for straddle (7.3) are trade
   formulas awaiting authored content;
4. the moss-hummock split (6.6) assumes microtopography carries the dome —
   an architectural allocation to be confirmed with the terrain cook.

**Verdict (per the assignment's categories):**

- Exact O(1) reconstruction of the **arbitrary-morphology** community under
  the frozen bound: **impossible** — unconditionally for every previously
  attempted mechanism family (Lemmas 3.2, 3.3, 5.3, 5.4, 5.5;
  Theorem 4.1), and conditionally in general under evidenced Hypothesis H
  (Proposition 3.5).
- Exact O(1) reconstruction for **class-`E` authored** communities, over the
  complete promised domain (all directions, all origins, all copies,
  multi-species, moss): **proved** (Theorem 6.9), with every residual error
  quantization-scale and camera-distance-independent (Section 8).
- The overall programme (indistinguishable-from-real Estonian ground cover
  at O(1)): **conditionally proved** — the condition is authoring fidelity
  within class `E`, an empirical gate outside mathematics, to be judged
  against real reference under the project's existing visual standards.

The minimal additional structure the resource bound demands — the
assignment's final question — is therefore stated exactly: **axial invariance
per directional class** (with band quantization for ends and media for
sub-pixel detail). Nothing weaker survives Sections 3–5; nothing stronger is
required by Sections 6–9. A later implementation is transcription of
Sections 6, 7, and 9; nothing in it requires experimentation to be believed.

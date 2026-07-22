# The shell-frame field: a new exterior ground-cover rendering model

Status: pure mathematics, 2026-07-22. Invented under the **re-stated user
contract** of this date, which supersedes the inherited constraint frame of
the previous agents:

- **Goal:** a NEW rendering model for ground cover — inspired by Sannikov's
  precomputed-ray idea, not a rebuild of it — that renders the *actual
  authored geometry* (arbitrary morphology, no re-authoring requirement) with
  high fidelity **from every exterior viewing direction**, runs fantastically
  on low/mid-end devices, has **low memory**, and is O(1) per pixel.
- **The one permitted limitation:** the camera may not go *inside* the cover
  volume and expect high quality. Exterior views only.
- Discarded inherited constraints: the `51,121,152`-byte freeze, the
  "`≤4` reads" freeze, the requirement to reproduce the mesh from interior
  origins, and the requirement that the representation equal the source
  continuum-exactly. Retained from measurement (because they are physics, not
  frame): the no-go lemmas of `GRASS-EXACT-REPRESENTATION-THEORY.md` §3–4 —
  cross-owner depth arithmetic invents surfaces; unbounded-lever direction
  sampling makes camera-anchored wedges.

Verdict up front: **a complete new model is constructed below** — the
**shell-frame field** — with its error theory, direction-lattice design law,
cost model, and one decisive falsifiable offline gate. It requires no
re-authoring of the source geometry: the bake renders the real mesh.

---

## 1. Why exterior-only changes the mathematics

Two structural consequences, each worth more than any memory increase:

**1.1 The dimension collapses.** With every ray origin outside the cover, the
query is a function of the oriented **line** only: the field is
4-dimensional, and the entire successor/pointed-ray apparatus (the fifth
coordinate, the ordered event fibre, the deep records) disappears.

**1.2 The lever collapses — this is the decisive one.** Every measured
failure of direction-sampled reconstruction obeyed one law: angular error
`Δθ` displaces correspondence by `L·Δθ`, where the lever `L` is the distance
from the *addressing surface* to the *content*. The prior attempts anchored
on the top or middle plane, so `L ≈ h/2 ≈ 0.6 m` — and at grazing the
interior skim made `L` saturate at the 155 m horizon, giving the
`ε(M) ∝ 1/√M` wall (`≈ 1.8 TB` for crispness).

Exterior viewing changes what is *visible*: from outside, sight lines
terminate in a **shallow shell** around the canopy envelope — the deeper
interior is occluded by the cover itself (dense sward), or the envelope dips
to the ground where cover is sparse (then content hugs the ground). Visible
content therefore concentrates within a **residual spread `σ`** of a smooth
envelope surface, with `σ` of order centimetres — measured per asset, and
*smallest exactly where cover is densest or the view most grazing* (grazing
sight lines see only the envelope crest).

If the representation can be anchored so that its lever is `σ` instead of
`h/2` or the horizon, the direction-sampling cost drops by
`(h/2σ)² ≈ 15–150×` before any other idea, and the residual misalignment
falls to centimetre scale — the world-anchored error class that ensemble
content tolerates. The shell-frame field is the construction that achieves
lever `= σ` **by definition**, plus one correction step that makes the
surviving first-order term second-order.

**1.3 Grazing stops being special.** For an exterior camera at height
`m₀ > 0` above the envelope, a sight line at elevation `α` meets the shell at
distance `D ≈ m₀/α`. Every quantity that makes grazing *hard* (long lever,
dense angular structure) grows with `D` — but the **screen-space tolerance
grows with `D` at the same rate** (a pixel at distance `D` spans
`D·θ_pix` of world; the visible shell band beyond ~100 m subtends 1–2 pixels
total). Section 5 turns this cancellation into a finite direction-lattice
design law. The pathological regime of the old analysis — unbounded-lever
grazing — was precisely the *interior* skim, which the new contract excludes.

---

## 2. The model

### 2.1 Objects

Let the community `𝒢` be the **actual authored geometry** (the accepted
Calamagrostis union or any successor), Λ-periodic per tile, in the
terrain-following ground chart. No structural assumption about `𝒢` is made —
no extrusion, no axis classes, no mask decomposition. Fix:

- a **direction lattice** `𝒟 = {d_i} ⊂ S²_-` (upper-hemisphere viewing
  directions; design in Section 5), triangulated;
- the **envelope** `Σ: y = E(x,z)` — the smooth (low-pass), Λ-periodic
  surface minimizing the mean squared depth of exterior-visible geometry
  below it (a shrink-wrap over the cover, dipping to ground in gaps). `Σ` is
  a bake-side object; the runtime never intersects it.

### 2.2 The frames (the stored representation)

For each `d_i`, the **frame** `F_i` is the *orthographic first-hit image of
the infinite periodic community along `d_i`*: for each point `u` of the
plane `Π_i ⊥ d_i`, cast the ray `u + τ d_i` against the infinitely repeated
`𝒢` and store the coupled first event. Because `𝒢` is Λ-periodic, `F_i` is
periodic in `u` under the projected lattice `P_i Λ` — one finite texture
tiles the world, and **occlusion between periodic copies is baked in**: a
frame texel may record content several tiles deep, resolved offline. This
generalizes Sannikov's infinite-mask bake from his one analytic family to an
arbitrary mesh and an arbitrary direction set: each frame *is* a "Sannikov
bake" of the real geometry from one direction.

Stored per frame texel (coupled, one event):

- premultiplied RGBA (authored colour × coverage; sub-texel geometry becomes
  fractional alpha in the bake's supersampling — the band-limited object);
- octahedral normal;
- **residual depth** `r = τ_hit − o_i(u)` (see 2.3), fine quantization;
- species/material mark.

### 2.3 The two-frequency depth split (the load-bearing invention)

Each frame also stores a **low-resolution smooth offset field** `o_i(u)`
(e.g. `32²` per tile, bilinear): the local mean of `τ_hit` over the frame —
view `i`'s own apparent shell. The full depth is

\[
\tau_i(u) = o_i(u) + r_i(u),
\qquad
\boxed{\;\sigma_i(u) = \text{local spread of } r_i \;=\; \text{the lever}\;}
\]

By construction `o_i` absorbs everything smooth — canopy undulation, gaps
dipping to ground, the grazing fringe surface, terrain following — so
`σ` is only the *categorical* within-shell scatter (which blade at this
texel), measured centimetres for dense cover and **decreasing** toward
grazing (crest-only visibility). Splitting depth this way is what converts
the impostor trick (parallax correction about a plane) into a field method
(parallax correction about each view's own curved shell), and it is the
single element no prior attempt in the ledger had: every tested variant
anchored on a *global flat plane*, whose lever `|y−y₀| ≤ h/2` is 5–20× the
shell residual — and they measurably failed in proportion to exactly that
lever.

---

## 3. The runtime algebra (fixed, loop-free, geometry-free)

Per pixel with camera `C` (exterior) and unit ray `d`:

1. **Node selection.** Find the direction-lattice triangle containing `d`;
   barycentric weights `(w_1,w_2,w_3)` over nodes `d_{i_1},d_{i_2},d_{i_3}`.
   Fixed arithmetic (octahedral map), no search.
2. **Frame addressing (line-invariant by construction).** For each selected
   node `i`: the ray's orthographic projection onto `Π_i` is an **affine
   function of the oriented line** — `u_i^{(0)} = P_i(C) + t^{*}\,P_i(d)`
   evaluated at any datum; moving the datum along the ray moves the address
   *along the ray's epipolar line in the frame*, which step 3 resolves.
   There is no carrier plane, no heightfield solve, no pole, and no
   query-datum dependence — the defect class of every "moving carrier" is
   structurally impossible here.
3. **One-step shell alignment (the parallax correction).** The ray sweeps
   the epipolar line `u_i(τ) = u_i^{(0)} + τ\,P_i(d)` in frame `i`; the
   content it can hit lies where the frame's depth agrees with the ray's
   depth. Solve first against the smooth shell: `τ_0` from
   `⟨d,d_i⟩`-corrected intersection with `o_i(u)` (one bilinear read, one
   division — `o_i` is smooth, so this is well-conditioned); then apply one
   fixed correction using the residual at that address:
   `u_i^{*} = u_i(τ_0 + r_i(u_i(τ_0)))`. This is relief-mapping algebra with
   a **provably small** step: the correction distance is `‖Δs_i‖·σ_i`, where
   `Δs_i` is the transverse slope difference between `d` and `d_i` (bounded
   by the lattice cell). One step suffices because the remaining error is
   second-order (Section 4). Read the full record at `u_i^{*}`.
4. **Combination — respecting the no-go lemmas.**
   - **Radiance:** premultiplied RGBA blends barycentrically across the
     three *aligned* records. Post-alignment, the three views describe the
     same world content to within millimetres, so this is band-limited
     filtering of one signal — not the forbidden blending of unrelated
     owners.
   - **Geometry (depth, normal, mark):** taken **categorically from the
     max-weight node only**. No cross-view depth arithmetic ever occurs
     (Lemma 3.3 of the theory doc is respected verbatim). The world hit is
     `X = ` frame-plane point `+ τ^{*} d_i`-reconstruction mapped to the
     live ray; its scene-depth is used for compositing against terrain,
     trees, and water.
   - Alpha composites the cover over the terrain beneath; a miss texel is
     transparent (ground shows through gaps with correct parallax, because
     gap texels' `o_i` dips to the ground surface).

**Cost:** `3` nodes × (`1` low-res `o_i` read + `1` record read) `+` optional
`1` mark/detail read `= 6–7` texture reads, `< 100` FMA, no loops, no
dependent chain longer than the single alignment step, no per-copy or
per-species work, no compute passes, no geometry. All reads are small,
tiled, block-compressible textures with high spatial coherence — the profile
of a technique that runs on integrated GPUs.

---

## 4. Error theory

Let `Δs_i = P_i(d)/⟨d,d_i⟩`-style transverse slope difference (bounded by
the direction-cell radius `Δ`), and `σ` the local residual spread.

**4.1 Interior (smooth-content) error.** For content whose residual depth is
locally Lipschitz, the one-step alignment leaves misregistration

\[
\boxed{\;e_{\text{int}} \;=\; O\!\bigl(\Delta^2\,\sigma\bigr)\;}
\]

— second order in the lattice spacing, millimetres at `Δ ≈ 0.1–0.2`,
`σ ≈ 5–10 cm`. This is what the three-view blend actually mixes: one signal,
aligned to well below blade width. Crispness of the dominant view (which is a
literal supersampled render of the true mesh — the reference screenshot's
quality *is* one such frame) survives the blend.

**4.2 Edge (categorical) error.** At depth discontinuities inside the shell
(blade silhouettes), the residual is not Lipschitz and alignment for the
affected texels errs by up to `Δ·J`, `J ≤` the local depth jump `≤ 2σ`. The
consequence is **edge doubling/softening of width `≤ 2Δσ`** — with designed
values, `2×0.15×0.06 ≈ 1.8 cm` worst-case at silhouettes, typically far
less, and confined to a 1-D locus. This is the model's honest signature
artifact (identical in kind to the acknowledged impostor silhouette
artifact), it is **world-anchored** (attached to the blades, not the
camera), and it shrinks linearly with lattice density near vertical, where
close-range viewing concentrates.

**4.3 Why the old wedge law is satisfied, not violated.** Camera-anchored
switch surfaces still exist (direction-cell boundaries, dominant-node
flips), but the jump across them is the misregistration `e`, not the old
`Δs·h/2`: millimetres-to-centimetre, i.e. below or at the world-anchored
tolerance of ensemble content, versus the metres that made wedges. The
amplification dichotomy is not repealed — it is *paid*: the lever was
engineered down to `σ` before any direction was sampled. In the notation of
the theory doc's §13.1 law, the constant `4πhR` collapses to
`≈ 4π·σ·L_tol` with `L_tol` bounded by the exterior-tolerance cancellation
of Section 5 — this is why a small lattice suffices where 1.8 TB did not.

**4.4 Depth-output correctness.** Scene compositing uses the winner node's
reconstructed depth: within a dominant region it is the exact baked depth of
real geometry (to quantization); across dominant flips it steps by `e` —
centimetres — irrelevant at terrain/tree compositing scale.

**4.5 What the model does *not* claim.** It does not reproduce the exact
micro-owner of every ray (measured impossible under any small budget); it
reproduces a world-anchored image of the true geometry in which any residual
error is a fixed, sub-blade-scale perturbation. Per the acceptance analysis
already agreed: for ensemble botanical content, that is the visually
correct target — view-consistency exact, positions statistically exact.

---

## 5. The direction lattice: a design law, not a guess

Three regimes, one budget rule.

**5.1 Core (elevation ≳ 25°).** Hemi-octahedral grid, `8×8` to `10×10`
(64–100 nodes). Lever `σ` is smallest here relative to tolerance (close-range
top-down viewing), and `Δ ≈ 0.12–0.2` gives interior error `≪` blade width,
edge term `≤ 1–2 cm` at silhouettes.

**5.2 Oblique-to-grazing rows (`α` from ~25° down to `α_min`).** Rows
uniform in **slope** `s = cot α`, with per-row spacing set by the
tolerance-cancellation law: content viewed at slope `s` sits at distance
`D ≈ m₀ s` (camera height `m₀` above the shell), so the transverse world
tolerance is `D·θ_pix = m₀ θ_pix s`, while misregistration is
`Δs·σ_res(s)`. The admissible spacing is

\[
\boxed{\;\Delta s(s) \;=\; \frac{m_0\,\theta_{\mathrm{pix}}\,s}{\sigma_{\mathrm{res}}(s)}\;}
\qquad\Rightarrow\qquad
N_{\text{rows}}=\int_{s_0}^{s_{\max}}\frac{\sigma_{\mathrm{res}}(s)}{m_0\theta_{\mathrm{pix}}\,s}\,ds ,
\]

and the integral **converges fast** because `σ_res(s)` falls toward grazing
(crest-only visibility) while the denominator grows linearly. `σ_res(s)` is
a measurable curve of the actual asset (one offline pass), not a tunable.
Sannikov's own grass — dense, combed — got away with *four* angular slices;
that is the empirical floor of this integral for dense content.

**5.3 The fringe row (`α < α_min`, e.g. below ~2–3°).** Beyond the last
ring, the visible shell band subtends ~1–2 pixels (Section 1.3): store one
**fringe frame** — the near-horizontal orthographic image of the periodic
canopy silhouette (transverse position × height, alpha against background),
a tiny texture. It renders the sward horizon and distant skim exactly as a
1–2-pixel band should be rendered: as its correct band-limited silhouette.
No angle is hidden, no flattening is authored; the domain simply runs out of
pixels before it runs out of rows.

**5.4 Anisotropic frame resolution.** A frame needs resolution only
transverse to `d_i`; grazing frames foreshorten to thin high-alpha-sparsity
strips and compress accordingly. Vertical frames carry the full `2–3 mm`
phase detail (the single-blade-crispness budget — this is where the
screenshot's fidelity lives, and it is cheap: it is one image).

---

## 6. Memory and cost model

Symbolic: `M = Σ_i A_i·b`, `A_i` frame texels (anisotropic), `b ≈ 10 B`
uncompressed (RGBA8 premult + oct-normal16 + residual16 + mark8 + spare),
plus `32²` `o_i` maps (negligible), block-compressed `×3–4`.

Example (one community type, `0.52 m` tile, `2.7 mm` transverse texels ⇒
`192²` vertical-frame): core `10×10 = 100` frames avg `0.55·192²` texels
(foreshortening), rows `4×16 = 64` frames avg `0.35·192²`, one fringe frame:

\[
M_{\text{raw}} \approx (100\cdot0.55 + 64\cdot0.35)\cdot192^2\cdot10\ \mathrm B
\approx 28.6\ \mathrm{MB}
\qquad\Rightarrow\qquad
\boxed{\;M_{\text{compressed}} \approx 7\text{–}10\ \mathrm{MB}\;}
\]

per resident community type — an order of magnitude under the old cap, in
"low memory" territory, with quality knobs (lattice density, texel pitch,
type count) that degrade *fidelity* gracefully, never *consistency*: coarser
lattices increase the world-anchored `e`, they cannot re-create
camera-anchored wedges because the lever stays `σ`.

Runtime: `6–7` reads, `<100` FMA, one shallow dependent step, fragment-shader
only. This is *lighter* than the currently accepted 0.39 ms path.

---

## 7. Integration semantics (unchanged architecture, new payload)

- **Control field / species patches:** per-type frame atlases selected
  before the query by the cooked cover field; mixtures via the existing
  root-stable A/B two-candidate rule. Species *within* a community are marks
  in the frames — zero per-species runtime work, exact baked mutual
  occlusion (they were rendered together into the frames).
- **Anti-tiling:** two world-anchored layers with incommensurate global
  transforms of the same atlas (the committed pattern) — reads ×2, bytes ×1.
- **Wind:** global/patch-coherent affine shear of the frame addressing
  (exact for coherent lean, the impostor-standard limit), optionally 2–4
  baked gust phases (memory ×phases). Per-blade independent motion is a
  declared non-goal of this model.
- **Interior fallback (the accepted limitation):** below the shell (camera
  descending into the cover), fade to the existing cheap near-representation
  over the last `~0.5 m` of approach. The contract explicitly does not
  promise interior fidelity; the model must only *fail gracefully*, and a
  fade of a world-anchored representation does (no pops, no wedges — detail
  softens).

---

## 8. Relation to prior art and to the ledger (what is actually new)

| ingredient | source | status here |
|---|---|---|
| precomputed first-hit textures of periodic cover | Sannikov 2019 | inherited as *bake concept*; his analytic-extrusion runtime is **not** used — frames are renders of the true mesh, no structural assumption |
| octahedral view lattice, 3-view blend, parallax-corrected per-view UVs | octahedral impostors (Brucks; hemi-octa follow-ups) | inherited as *runtime concept*; impostors require a bounded object, exterior view, distance ≫ radius, LOD handoff to meshes — none hold for a walk-over field |
| **infinite-periodic frames** (copy occlusion baked in, one query serves all copies at any extent) | new | removes the bounded-object requirement |
| **two-frequency depth split** `τ = o_i(u) + r`; lever = residual spread `σ` by construction | new | removes the distance ≫ radius requirement; this is the quantitative difference from every ledger attempt, all of which anchored on flat global planes (lever `h/2`) with no per-texel correction |
| **exterior tolerance-cancellation law** (`Δs(s) ∝ s/σ_res(s)`, convergent row integral, fringe closure) | new | replaces both the unbounded grazing lattice and Sannikov's flatten-the-grass workaround |
| categorical-geometry / blended-aligned-radiance split | forced by the proven no-go lemmas | keeps the model inside the algebra that measurement allows |

Every rejected ledger family is either subsumed (the §43 radiance gate is
this model with `o_i ≡ y_0` and no residual correction — i.e. with a 5–20×
larger lever, and it failed in proportion) or irrelevant under the new
contract (all successor/pointed-ray machinery).

## 9. The decisive falsifiable gate (one offline experiment, before any code)

Rerun the existing §43 radiance-gate harness on the actual asset with three
changes only: slope-aware rows per 5.2, per-view `o_i` shells, one-step
residual alignment. Predictions this model stakes its life on:

1. silhouette IoU `0.852 → ≥ 0.97` and RGB max-channel p95
   `0.547 → ≤ 0.15` at **equal or lower** direction count and bytes;
2. connected wrong-view regions (was 25–59% of frame) collapse below `1%`
   with residual errors world-anchored (verify by camera-translation
   sequences: predicted unforced class-change rate `< 5%`, was 21–45%);
3. measured `σ_res(s)` decreasing in `s`, and the 5.2 integral yielding
   `≤ 16` rows for the accepted asset.

If prediction 1 fails, the model is wrong about its central lever claim and
is parked without a shader attempt. If it holds, implementation is
transcription: the bake is "render the real mesh from N directions" (any
renderer, arbitrarily expensive, embarrassingly parallel), and the runtime
is Section 3's seven reads.

## 10. Verdict

Under the corrected contract (exterior-only, arbitrary morphology, low
memory, low-end O(1)), the shell-frame field is a complete new model:
line-invariant by construction, wedge-free by lever design, crisp by
storing literal supersampled images of the true geometry, `~7–10 MB` per
community type, `6–7` reads per pixel, with one honest signature artifact
(sub-2 cm world-anchored silhouette softening) and one declared domain edge
(interior fade). Its correctness does not rest on replicating Sannikov's
extrusion identity anywhere; his method survives only as the limiting case
`σ → 0`, `N_rows → 4` that dense combed grass allows — which is precisely
why his video looks the way it does.

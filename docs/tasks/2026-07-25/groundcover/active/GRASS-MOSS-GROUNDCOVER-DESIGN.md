# Ground-Cover Reset: A Type-Keyed Precomputed-Raycast System on One Cook-Side Control Field

> Design for approval — RESEARCH phase output (Fable-5/xhigh workflow wf_5fdab918-fad, 2026-07-20).
> NO code until the user approves the direction + the scope decisions at the end.

## USER DECISIONS (2026-07-20 — approved, locked)
1. **Direction APPROVED** — build the type-keyed raycast ground-cover system on one cook-side control field via the phased plan. Start Phase 0 (fresh trace).
2. **Tall hero grass — RAYCAST ONLY, accept the ceiling.** No GPU-blade accent path, ever. Single mechanism. Push the raycast mitigations (mesh-baked silhouettes + variance-preserving detile + more angular slices) hard; stay honest about where tall side-lit waving grass lands. (Estonia cover is mostly sedge/cotton-grass/short-mid/moss, so the weak regime is a small slice.)
3. **Perf target — "comfortably under today (~4 ms), richer far cover."** NOT a hard ≤2 ms ceiling; the distance-LOD ladder does NOT push maximally if it costs look. Look (BAR 1) wins ties over the exact 2 ms number.
4. **Species pipeline — build the GPU baker** (CPU reference baker ~3 days/mesh is too slow for ≥10 species); **derive the palette from cooked biome/wetness/classId layers**, grounding silhouettes in public Baltic/Nordic botanical reference (zero external budget). **HARD CONSTRAINT: Estonia-NATIVE species ONLY** — no introduced/non-native ground cover. Palette drawn only from species genuinely native to Estonian mires/forests/meadows (native bent-grasses/*Agrostis*, sedges/*Carex*, *Eriophorum* cotton-grasses, *Sphagnum* spp., native lichens *Cladonia* etc., native forbs). Same native-only law that flagged the Labrador-tea→sookail issue. Verify nativeness per species before baking.
5. (Deferred defaults, flag if wrong) Interactivity (GoW-style displacement) = later, additive. Cook-side control-field planes = assumed in-budget; start as a 2-type mix in the free guide words, add the overlay-palette plane only if rich co-located overlay is needed for v1.
6. **D2 CLARIFIED (2026-07-20, user, with screenshots) — CORRECTS §3's framing.** D2 is a **SINGLE-FRAME** rendering-quality defect, NOT temporal: sparse/non-dense grass blades render as fuzzy, unrecognizable vertical smears; the dark AO/occupancy speckle is MORE visible than the blades; only DENSE batches read as grass; diagonal LOD/tile banding. It is NOT frame-to-frame jitter, must NOT be patched with TAA/postprocessing, and must NOT add perf cost. The temporal interpretation in §3 (unjitter ray / motion vectors / resolver polish) was a MISFIRE — implemented then REVERTED (it even removed the sub-pixel supersampling that helps thin blades). The real D2 = per-blade single-frame legibility: candidates are sub-pixel blade coverage without analytic AA (hit/miss speckle), over-strong baked AO/occlusion, too-low bake spatial/angular resolution (phantom geometry), sparse blades as isolated sub-pixel columns, weak blade albedo vs AO, and per-tile-constant fields (grid-lesson banding). Root-cause diagnosis in progress; the fix must be single-frame, no postprocess, zero/bounded perf.

## EXECUTION ADDENDUM (2026-07-20) — grass legibility root-caused; legibility fix = the O(1) restoration
Live controls-first A/B (Fable/xhigh, repro pose `?scene=world&dpr=2&shot=6`) PROVED the "grass doesn't
look like grass" defect:
- **Cause 1 (structure):** the lane runs an 11 mm fixed-width blade to 155 m that never widens with
  distance + a binary hit/miss election with NO coverage/AA → beyond ~15 m every blade is sub-pixel →
  Bernoulli dot-speckle. Proven: `grassbakw` ×3.6 reconnects strokes (cover 27→37%).
- **Cause 2 (darkness):** blade albedo is 6.7× darker than ground × tip-AO × ±0.8 rad twist → dark flecks
  beat the blades. Proven: `grassdbg=flatres` kills the flecks. (No baked AO exists — it's shading variance.)
- Q=2 quad = amplifier only. Diagonal banding = **terrain-LOD, NOT grass** (grass=0-proven → #106/#115).
- Refuted as causes: angle slices, bake res, taper bands. LUT interpolation subordinate (keep only the
  cheap R-miss fix).

**THE FOLDED FIX (user-approved, implementing):** the illegibility and the missing O(1) are the SAME
machinery — the reject/refetch loop (full-density bake vs distance-thinned world mask → 5–14 rejects/hit
in the far band = O(texels×rejects), not O(1)). ONE rework fixes both: **distance-banded, width-conserving
bakes** (band bakes halfW×k with fibers/cell ÷k, coverage mass conserved, sized so a blade ≥~1 px at the
band's far edge; band picked by distT per texel) + **band-local accept** (near = exact per-cell mask;
mid/far = root-cell hash-ratio compensated keep) which collapses rejects ~1/p→~1 = the article's
one-fetch-per-step **O(1)**. Zero-cost adjuncts: distance shading-narrowing (fade AO/twist/albedo→sward
mean over 30–90 m) for the darkness; a 3-line R-miss encode fix (0 not 102) to kill phantom hits.
**So "when do we get O(1)?" → it IS this step, folded with the legibility fix — the march is reworked
once, not twice.** Far field >110 m = per-blade banded march for now; user decides a coverage-matched
ground-tint fold after eyeballing. Perf: +~260 KB VRAM + boot-bake ×bands; runtime neutral-to-faster
(reject collapse) but NO ms claim without a fresh trace (before via HEAD/git). Supersedes §3's temporal
D2 framing (that was the misread, reverted).

## Headline recommendation

Keep the precomputed-raycast direction, but correct three things the brief filed under "given": our lane is NOT the O(1) Sannikov technique (it is a 256-iteration compute DDA march — NaniteGrass.ts:846 — which is the whole perf problem), D1 is a missing cook-side control field (not a rendering limit), and D2 is a frame-integration bug (jittered geometry ray + no motion vectors), both orthogonal to perf. Build a single type-keyed ground-cover system driven by one streamed control field: restore true one-fetch-per-column for moss/short/mid cover (its strong regime, where photoreal is reachable), and treat tall side-lit waving hero grass as the method's honest weak regime that may need a GPU-blade accent path on the same field — a scope call to surface, not to decide silently.

# Ground-Cover Reset — Design for Approval

**Status:** design only, no code. **Scope:** all low ground cover — moss, grass, sedge/cotton-grass, lichen, herb/forb, bare — as ONE generalized system. **Bars:** (1) photoreal — indistinguishable from real; (2) significantly faster than today. **Prior defects to solve:** D1 (only-blades, no species/patches, fake density) and D2 (jitter/deformation).

---

## 0. The premise-audit result (why the frame changes before we build)

The brief says "optimize + generalize the Sannikov raycast lane." Applying *go-up-a-level* to that task — and verifying against the code — surfaces three things filed under "given" that are actually the *generators* of the failures. The frame is ~70% right and the raycast direction is sound, but all three defects are mis-attributed to the lane's internals. Each dissolves one level up.

**(1) Our lane is NOT the O(1) Sannikov technique — this is the whole perf story.** Verified: `kRay` is `loopUN('gro', 0..256, ...)` — a compute DDA march of up to 256 texel iterations per ray (`NaniteGrass.ts:846`), paying a fetch-chain per 0.84 m texel crossed **plus** refetches per root-cell rejection. The article's speed comes from rasterizing a hull and doing **one** fetch per fragment. We march because world-space density (`densityAt`, a single scalar collapse at `292-324`) and per-cell swirl-bombing rotation (`573`) are evaluated *inside* the inner loop, changing the effective tile every texel. So O(1)/fragment became O(texels + rejects)/ray. **"Optimize the march" stays inside the wrong problem.** The lever is architectural: lift selection out of the loop, restore one-fetch-per-column.

**(2) D1 is a missing COOK-SIDE control field, not a rendering limit.** `densityAt` collapses biome/moisture/slope/canopy to one scalar, then an i.i.d. white-noise cell-hash accept places clumps with zero spatial correlation. No species channel exists end-to-end (words 6-7 written 0, verified `349`). Patches, species, and structured density were never *representable*. The user's own hard-requirement control field IS the dissolution.

**(3) D2 is a FRAME-INTEGRATION bug, not a raycast-method flaw.** `kRay` builds its geometry ray from the jitter-mirrored VP (`cam.update(jitteredCamera())`, `NaniteFrame.ts:515`), so the analytic surface physically crawls sub-pixel every frame, AND emits zero motion vectors on a wind-animated surface (documented, `2026-07-03-grass-arc.md:539-541`). Both are upstream setup choices that generate most of D2 before any resolver tuning.

**The one place the frame is genuinely wrong:** the implicit "one monolithic raycast lane, indistinguishable from real, for ALL cover including tall side-lit waving hero grass." That regime is the method's honest weak spot (§7).

---

## 1. Recommended direction

**Build a single TYPE-KEYED ground-cover system driven by one streamed control field, restoring the raycast lane to true one-fetch-per-column — and treat tall hero grass as a disclosed hybrid fallback, not a guaranteed win on the raycast body.**

Concretely:

- **Keep** the raycast/precomputed-parallax approach — the research and the code both confirm it is the correct bet over scattered geometry (rejected) and over shell texturing (overdraw-bound, grazing-angle gaps, no true depth for the vis-buffer).
- **Restore O(1)** by moving density/species/patch *selection* out of the inner march into the control field (sampled once per column) and baking bombing into LUT variants — collapsing O(texels+rejects) back to skip-to-sward-top → one tap.
- **Generalize** via the control field + a mesh-baked multi-species LUT array + a baked density/height ladder — all essentially free at runtime on a correct single fetch.
- **Split by regime:** raycast/parallax-carpet for moss + short + mid cover (the large majority of ground, where photoreal is reachable); a disclosed GPU-blade accent path — driven by the *same* control field — held in reserve for tall hero grass if its bar proves unreachable on the raycast body.

**Why this and not "just tune the current lane":** the current lane's slowness, its only-blades look, and its shimmer are three *different* upstream causes. Tuning the march addresses none of them and spends the budget the generalization needs. The control field solves D1 for free; the frame-setup fix solves D2 cheaply; the O(1) restoration buys the headroom for the added photoreal shading.

---

## 2. Technique menu (what it buys / perf / fit)

| Technique | What it buys | Perf | Fit to our lane |
|---|---|---|---|
| **Restore O(1) single-fetch-per-column** (Sannikov core, rasterized-hull spirit) | The perf bar. Density-independent, cache-bound not loop-bound (~1 ms-class @1080p on a GTX750) | Trades O(texels+rejects) for skip-then-one-tap | **Core.** Real re-architecture of `kRay` DDA (846) → skip-to-sward-top (reuse L2 coarse + terrain-Lipschitz skip, 883-993) then one trilinear tap. Highest-value, non-trivial |
| **Mesh-baked multi-species LUT array** | ≥10 real species silhouettes; kills the author's own "curls/seaweed" look | One fetch regardless of N species; VRAM only (tiles ≤128²) | Re-key the existing multi-volume select (`fetchBand`, 1123-1155) from height-band to species. Needs a **GPU baker** (see risks) |
| **Baked density/height LUT ladder** | REAL density variation; the forbidden "sparser blades" trick becomes structurally impossible | One fetch (index+lerp); small VRAM | Density field selects the rung; blue-noise/IGN-dither + 2-rung lerp for continuity |
| **Cook-side plurigaussian facies + domain-warped Worley** (D1 patch generator) | Irregular patches, varying patchiness, species adjacency rules | Cook-side; zero runtime cost | **Machinery already in repo** (BOG-V6). Streamed plane, read free via existing plumbing |
| **Voronoi clump field (GoT)** | Coherent tufts (shared facing/height/color/lean); the "looks natural now" lever | A few in-shader hashes, or cook-side | Cheap runtime complement; clump-id seeds detile hash → free patch outlines |
| **Practical hex-tiling detile + variance preservation** (Mikkelsen/Heitz; = tsVGRd) | Kills repeating-stamp look AND contrast-swim (a D2 cause); free patch outlines from the hash | 3 SampleGrad taps + ALU; mip-safe | Replaces the ad-hoc swirl-bomb (1035-1049), which lacks variance preservation |
| **Laplacian texture blending (Wronski 2025)** | Same, but the only blend that behaves on the NORMAL map our LUT stores in GBA | A few low-mip taps; no precompute | Prototype head-to-head with hex-tile for the normal channel |
| **Frostbite wrap-translucency + baked thickness** | Back-lit grass/moss glow — the biggest "feels alive" cue | A dot+pow+mul + one thickness tap | Pure resolve add; lane already has the 3D hit+normal and a tip-SSS stub (`NaniteResolve.ts:1447-1448`) |
| **Baked horizon/occupancy AO (self-shadow into the sward)** | Dark-at-the-roots volume cue the current lane lacks | One extra baked channel/LUT unpack; ~free | Bake AO-along-height; apply at the N·L gate (1331-1336) |
| **Parallax/relief cushion carpet for MOSS** | Moss micro-relief, cushion parallax, real depth/self-occlusion | Bounded taps or one cushion-LUT tap — the CHEAPEST case | New in-sward branch at `bakedTexel()` entry (1012), reusing walk/emit/resolve |
| **Motion vectors from the raycast hit** (D2 root fix) | Temporal stability — the single biggest D2 win | One RT write + one mat·vec per hit | Emit velocity from reconstructed hit B through prev-frame VP; the lane HAS B and both VPs |
| **Unjitter the geometry ray** (D2 root fix) | Stops the surface crawling; jitter becomes pure supersampling | Zero cost | Split `NaniteFrame.ts:515` into an unjittered geometry VP + a sample-jitter for the resolver |
| **More angular slices + angle-aware interp** (D2 geometric warp) | Kills normal/depth swim as the camera turns | Larger LUT (watch ≤128² locality); still one fetch | Raise `BAKE_ANG` from 8; fix wrap interp; stay near the exact extruded case |
| **Toksvig/Tokuyoshi normal-variance roughness + Persson phone-wire coverage** | No specular sparkle on wet cover; coherent far-field sward | A few ALU + one baked variance channel | TAA cannot fix sub-pixel normal variance — prefilter must; complements MVs |
| **Nearest-depth OVERLAY union** | Moss under grass under forbs, honest occlusion | **Cost-dangerous:** N marches = +11.4 ms ungated; must be ONE bounded 2-3-tap union | The L2 golden-layer machinery (1437-1601) is the structural template |
| **Shared low-res world wind field + per-type stiffness** (GoT + Crysis) | Coherent gusts across the whole scene; per-type/absent wind | One bilinear tap at bake; per-type gain is one byte | Drive the oblique-TBN bend (1050-1104); moss gain=0 |
| *Shell texturing* | *(pressure-test)* cheap fuzz | *O(shells) overdraw; grazing gaps; no true depth* | **Rejected as mechanism** — loses the vis-buffer depth the raycast gives. Keep only its "density = coverage threshold" idea |
| *GPU-blade accent (mesh-shader, GPUOpen)* | Tall hero-grass photoreal where raycast's ceiling is lower | Amplification-bound (opposite profile) | **Disclosed fallback only**, driven by the same control field. Note the user rejected scattered *discrete* geometry |

---

## 3. How this solves D1 and D2

### D1 — ≥10 species in irregular varying-patchiness patches, with REAL density (not sparse blades)

D1 is three sub-requirements, all solved cook-side or as a free single-fetch re-key — **none re-introduces the per-pixel march**, so D1 photorealism is essentially free on top of a correct single-fetch lane.

1. **≥10 species.** A mesh-baked LUT texture-array, one slice per species from real reference (grass, sedge, rush, cotton-grass, lichen, moss cushion, forb rosette). The fragment picks the slice from the control field's dominant/subdominant ids and lerps — **one fetch regardless of N**. Silhouette accuracy from real meshes is the direct cure for the "only generic blades" look and the author's own "curls/seaweed" failure.

2. **Irregular patches with varying patchiness.** Cook-side truncated-plurigaussian facies (reuse BOG-V6): the variogram/correlation-length tunes cluster scale per region ("some regions clustered, some random"), the truncation rule tunes species adjacency. Domain-warped Worley bends outlines organic; spatially modulating the warp gain varies the patchiness itself. This replaces the i.i.d. white-noise cell-hash — the exact anti-pattern that produced D1. The hex-detile hash seeded from the clump-id snaps sub-grid patch edges to hex cells for free.

3. **Real density, not sparse blades.** A baked density/height LUT ladder (short-sparse→tall-dense, fiber count varied IN THE BAKE) indexed by a continuous cook coverage field. Because the runtime output is a continuous extruded BODY with real depth+self-occlusion, "less dense" renders as a genuinely shorter/thinner/patchier sward. **The forbidden trick is structurally impossible — there are no runtime blades to thin.** This matches the AAA gold standard exactly (GoT/Outerra/HZD all vary the *sward*, never the object count except as width-compensated LOD).

### D2 — jitter / deformation: root cause + fix

D2 is TWO artifact families needing DIFFERENT fixes — do not conflate them.

**TEMPORAL SHIMMER / SMEAR (dominant, a correctness bug):**
- *Root cause A:* the geometry ray is built from the jitter-mirrored VP (`515`), so the analytic surface physically translates sub-pixel every frame — antialiasing by moving the geometry, which forces mis-reprojection and defeats clamping. **Fix:** split into an unjittered geometry VP feeding `kRay` + apply the Halton offset only as the resolver's sample position. Zero cost.
- *Root cause B:* TRAA runs with NO motion vectors on a wind-animated surface. **Fix:** emit velocity by reprojecting the reconstructed world hit B through the prev-frame VP into a velocity target the resolve/TAA reads (the lane already computes B and holds both VPs). This is the textbook WPO-foliage case (UE ships `r.Velocity.EnableVertexDeformation` for exactly this).
- *Root cause C:* the ragged-top hash (`colH`/`col2`, 1316-1323/1539-1545) is sampled in the WIND-SHEARED, bombed tile coordinate, so a fixed blade's hit crosses 13 mm column boundaries every frame → tips pop in/out under wind. **Fix:** anchor the hash to the wind-INVARIANT fiber/root coordinate (available pre-shear) and scale-discretize (Wyman hashed-alpha).
- *Resolver polish:* YCoCg variance-clip (replace hard min/max clamp), 1/(1+luma) anti-flicker weight, Catmull-Rom history, closest-depth velocity dilation, FSR2-style reactive mask for self-occlusion churn.

**GEOMETRIC WARP (morph/chunkiness):**
- *Root cause:* only 8 angular LUT slices over 2π (45°/slice) trilerped across angle → two intersection fields blend as the camera turns; plus trilinear reconstruction with no proper mip chain. **Fix:** raise `BAKE_ANG` to 16/32, angle-aware (shortest-arc) interpolation, honor ~1 texel/pixel with mips, and stay near the exact extruded case (limit the shift/thicken/varying-inclination heuristics that warp at grazing angles).
- *Specular sparkle* on wet cover is NOT TAA-fixable — Toksvig/Tokuyoshi roughness widening from the fetched-normal derivatives or a baked normal-variance mip; Persson phone-wire coverage clamp for sub-pixel-width far blades.

**Note (honest, from GoT):** the photoreal benchmark *also* has no grass motion vectors. But it makes the surface temporally stable AT THE SOURCE (clump-normal spec-AA, rounded normals, glancing-angle adjust). We should do both: emit MVs where we cheaply can (we have B), and stabilize at the source (angular slices, fiber-anchored hash) so we do not *lean* on TAA.

---

## 4. Photoreal moss, multi-layer overlay, per-type wind — one system

### Moss — the method's STRONGEST case (inverts the geometry-moss failure)

Sphagnum cushions are short + dense + rounded → tiny LUT tile → tightest cache locality → the cheapest single tap in the whole system. Represent moss NOT as fibers but as a baked height/parallax CUSHION carpet (POM/relief lineage, or a Sannikov cushion-LUT with a low rounded cross-section) tested against the existing per-texel ground plane `gP` (1086-1088) + `topOut` — a new in-sward branch at `bakedTexel()` (1012), reusing the vertical-skip, emit, and rayNrmTex/resolve contracts unchanged. The cues that make it read as *real wet sphagnum* rather than the rejected flat floating brown pancakes:

- Real depth + self-occlusion + micro-parallax from the baked cushion heightfield (the geometry attempt had none of this).
- Frostbite wrap-translucency with a baked local-thickness channel → the vivid green→yellow→ochre→rust translucent glow when back-lit.
- Baked horizon/occupancy AO darkening the crevices (the volume cue).
- Biplanar albedo (2 taps, UV-free) at the honest 3D hit for the speckled wet-substrate look.
- Height/curvature/**moisture**-driven color from the cook wetness field (water-darkening where wet).
- Wind amp = 0 (moss is rigid — an honest per-type stiffness gain, not a hack).

### Multi-layer overlay (moss under grass under forbs) — the ONE cost-dangerous seam

Honest via a NEAREST-DEPTH union (min-over-d) of LUT fetches: the depth-correct `gl_FragDepth` output is exactly what makes correct inter-layer occlusion possible with no sort. **The trap:** N independent marches blow the budget (the golden-angle L2 overlay measured +11.4 ms ungated). The union MUST be answered by ONE bounded fetch set (2-3 taps max, hard per-pixel budget), with the stacked types **pre-selected by the control field** before the fetch — never one march per layer. The L2 golden-layer machinery (1437-1601) is the structural template (swap its same-tile-rotated fetch for a different-species tile). This is the highest-risk piece and is sequenced **last**.

### Per-type wind — one shared field, per-type stiffness

The universal AAA pattern: a single low-res world wind vector field (GoT), sampled by grass/trees/particles so gusts sweep *coherently* across the whole scene. Per-type response is a STIFFNESS GAIN on that shared field (Crysis: main-bend scaled by height² + detail flutter, phase-desynced), NOT a separate wind per type. Fold the gain into the existing per-texel wind-amp block (533-538, 581-609): moss→0, grass→moderate, sedge/cotton-grass→whippy. Drive the oblique-TBN inclination bend (1050-1104) — real geometric bending, not surface noise. Coherent gusts via Lagrangian advection of the detile coords (tsVGRd scrolling variant).

**Critical coupling:** this animated shear is a prime D2 source. It MUST ship together with motion vectors AND the fiber-anchored ragged-top hash, or it re-creates the jitter. Wind and the D2 fix are not separable.

---

## 5. Perf path — from the header's +14.5 ms toward the ≤2 ms law

**Ground truth first (mandatory before any perf claim ships).** The +14.5 ms in `NaniteGrass.ts:97-98` is a **stale** 2026-07-04 look-pass number. The same-day perf pass shipped lever A (field-bake 23.1→12.9 @dpr2) and lever B (quad-march 12.9→3.7), leaving the last **committed** measurement at **+4.0 ms @dpr2 / +3.3 ms quad @dpr1.5**. So "faster than +14.5" is nominally already met — **that is a trap.** The real bar is ≤2 ms, AND the generalization (species/overlay/SSS/detile/motion-vectors) ADDS cost on top of +4.0. **A fresh `.gputrace` (capture → manual Xcode export → `run_all.sh`) is required to ground the optimization** per the standing rule; do not claim ≤2 ms without it.

**The path is STRUCTURAL, not tuning:**

| Lever | Mechanism | Expected direction |
|---|---|---|
| **A. Restore O(1)** | Lift density/species/patch selection out of the inner march into the control field (once/column); bake bombing into LUT variants so the effective tile stops changing mid-ray | The dominant win — trades O(texels+rejects) for a single tap; targets the ~1 ms cache-bound class |
| **B. Skip-then-single-tap** | Use the existing L2 coarse + terrain-Lipschitz vertical skip (883-993) to jump to sward-top, then ONE trilinear tap answering the column | Removes the per-texel fetch chain |
| **C. Delete reject/refetch** | Validate against the cook field BEFORE the fetch; never march to *discover* occupancy. The pre-fix 0.03 m stepping alone measured +6.0 ms | Removes the O(rejects) term |
| **D. LUT hygiene** | Tiles ≤128² RGBA8, ~1 texel/pixel, mips for distance LOD; anti-tiling ladder (IQ 2-tap far → hex 3-tap near) | Keeps cache locality; distance LOD bounds far cost |
| **E. Overlay budget** | Single bounded min-depth union (2-3 taps), hard-gated — never N marches | Keeps the cost-dangerous seam inside budget |

**Confidence:** HIGH that this beats the current march substantially. MODERATE that ≤2 ms holds across ALL cover + the added photoreal shading at retina without a real distance/detile LOD ladder doing work. The single-fetch form is a real cache-locality wall (~1 ms-class), not a blocker — but retina + rich resolve are the pressure points, so the LOD ladder (D) is load-bearing, not optional.

**Honest architectural caveat:** our renderer is a compute vis-buffer, not a raster-hull fragment shader, so "rasterize a hull, one fetch" maps to "skip-then-single-tap in `kRay`" — real re-architecture of the DDA loop, not a knob.

---

## 6. Risks (surfaced, not buried)

1. **Fresh trace not yet run.** Every perf claim rests on the stale-vs-committed discrepancy. The O(1) restoration must land the headroom BEFORE the generalization's added cost, and only a fresh trace proves it.
2. **Overlay seam** is the one piece needing real redesign. If the min-depth union can't fold into 2-3 bounded taps, multi-type overlay breaks BAR 2. Do it last, hard-budgeted.
3. **Tall hero-grass ceiling** on the raycast body is genuinely lower (tiling only hideable; non-parallel/variable-section fibers warp at grazing angles — exactly where large wind deflection lives). I will not rubber-stamp "indistinguishable from real" for that slice. May force the disclosed GPU-blade accent — a scope call.
4. **O(1) restoration is real re-architecture** of `kRay` (DDA → skip-then-single-tap), and the bombing-into-LUT-variants move must be dithered correctly or it re-trips the grid lesson.
5. **Mesh baker does not exist.** The CPU reference baker is ~3 days/mesh; a GPU baker is a prerequisite to author ≥10 species at acceptable velocity. Without it, species authoring stalls.
6. **Grid-lesson re-trip.** Every discrete LUT index from the continuous field must be blue-noise/IGN-dithered + 2-variant-lerped. Two residuals are already documented (per-texel-constant ctx floats; wind-sheared ragged-top hash) and must be fixed in the same pass.
7. **Motion-vector emit for a compute-vis-buffer raycast surface** is the one real design piece of the D2 fix. If the vis-buffer can't cleanly carry a grass velocity channel, wind shimmer persists and no resolver tuning hides it.
8. **Control-field composition** may overflow the 64 spare bits for rich co-located overlay (moss+grass+sedge+forb each with own density) — needs a 2nd streamed plane; modest VRAM + plumbing.
9. **Cook-side cost + determinism.** New species/patch/density/moisture planes in asset-gen must preserve the gen-determinism baselines and the streaming budget — real cook + VRAM work, sequenced behind the runtime seam being proven.

---

## 7. The honest verdict on the raycast ceiling

Raycast is the RIGHT foundation for the **large majority** of ground cover — moss, short, and mid cover reach the photoreal bar on this lane once you stack mesh-baked silhouettes + variance-preserving detile + Frostbite SSS + baked AO + per-patch color. Moss is the method's *strongest* case.

The genuine exception is **tall, side-lit, strongly-waving hero grass**: the method is analytically exact only for extruded parallel fibers of constant cross-section; tall wind-deflected grass is neither, and it lives exactly where the article's stated limits (tiling, grazing-angle warp) bite hardest. AAA does not use precomputed-raycast for hero grass for this reason. The honest resolution is a **hybrid keyed off the SAME control field**: (A) raycast/parallax-carpet for moss+short+mid — full photoreal reachable; (B) for the minority of tall grass, either (b1) accept a mitigated-but-slightly-lower ceiling on the raycast body, or (b2) a GPU-generated-blade accent path driven by the same cook clump/type/density fields. Since the user rejected *scattered discrete geometry*, GPU-amplified blades sit in a gray zone — **default to (b1), disclose (b2) as the fallback IF the tall bar proves unreachable after the raycast fixes.** That is a scope call to surface, not to decide silently.

---

## 8. Phased plan

Sequenced so each phase produces an inspectable artifact and de-risks the next. D2 first because it is cheap, orthogonal to perf, and a jittering surface can never read as photoreal (it gates BAR 1).

**Phase 0 — Ground truth.** Fresh `.gputrace` of the current lane (capture → manual Xcode export → `run_all.sh`). Establishes the real starting number and the per-shader ranking. *Artifact: a results folder + the honest current ms.*

**Phase 1 — D2 setup fix (cheap, unblocks photoreal).** Unjitter the geometry ray; emit velocity from B through prev-frame VP; anchor the ragged-top hash to the fiber; raise `BAKE_ANG` with angle-aware interp; add YCoCg variance-clip + 1/(1+luma) + reactive mask + normal-variance roughness. *Artifact: a stable, non-shimmering current-look grass on a live URL (`grass=0` link for the mesh + an all-layer link for surface agreement).*

**Phase 2 — O(1) restoration (buys the headroom).** Re-architect `kRay` from the DDA to skip-then-single-tap; lift selection to the field; bake bombing into LUT variants; delete reject/refetch; LUT hygiene + mip LOD ladder. Re-trace. *Artifact: same look, materially lower ms — the BAR-2 proof.*

**Phase 3 — Control field + species/density (D1).** Cook-side plurigaussian facies + domain-warped clumpiness + continuous coverage/height + moisture planes, derived from canopy/wetness/soil/biome/classId/slope; pack the mix into guide words 6-7. Multi-species mesh-baked LUT array (+ write the GPU baker) + density/height ladder, blue-noise-dithered selection. *Artifact: multi-species patchy cover with real density gradients on a live URL.*

**Phase 4 — Moss branch + shading (BAR 1).** Parallax cushion branch at `bakedTexel()`; Frostbite wrap-SSS + baked thickness; baked horizon/occupancy AO; biplanar albedo; moisture/curvature/height color ramps. *Artifact: photoreal wet moss + the rich resolve — the BAR-1 review gate.*

**Phase 5 — Wind + overlay (highest risk, last).** Shared low-res world wind field → oblique-TBN bend × per-type stiffness (moss 0), coupled to the Phase-1 MVs. Single bounded min-depth overlay union (2-3 taps, hard-gated). Re-trace to confirm the overlay stays in budget. *Artifact: coherent per-type wind + layered moss-under-grass on a live URL.*

At each live-URL phase: a batched visual verdict from the user (not per-change), never during announced AFK.


---

## Appendix A — The ground-cover CONTROL FIELD (user hard-requirement, full spec)

THE GROUND-COVER CONTROL FIELD (single authority for "what grows here, how much, how it moves") — an EVOLUTION of the existing 32B/texel guide record (NaniteGrass.ts:342-354), not a new subsystem.

LAYOUT — two tiers, both cheap to read:
(1) IN-LINE COMPOSITION (rides the existing per-frame guide record for FREE). The record already reserves words 6-7 written 0 (line 349), read inside the existing ctx uvec4 fetch with zero added bandwidth on the skip path (the confirmed one-16B-load constraint, lines 350-352). Pack a 2-type MIX there: dominant species id (8b) + subdominant species id (8b) + blend weight (8b) + per-type coverage/height/vigor (8b) in word 6; clump-id (16b) + moisture (8b) + canopy-proximity (8b) in word 7. This is what kRay reads once per column to pick the LUT-array slice, the density-ladder rung, the wind stiffness, and the resolve tint.
(2) STREAMED COOK PLANES (the source of truth, terrain-tile-attached, sampled exactly like biomeAt/fieldsAt and 4-corner-bilinear-blended through the existing plumbing at 462-501 that already guarantees C0 continuity): a SPECIES/FACIES plane (categorical, from truncated plurigaussian — reuse BOG-V6 machinery), a CLUMP-ID / patch plane (domain-warped Worley), a CONTINUOUS COVERAGE+HEIGHT (fluffiness/vigor) plane, a MOISTURE/WETNESS plane, and — only if rich co-located overlay is needed — a small OVERLAY-PALETTE plane naming up to 3-4 stacked types (moss+grass+sedge+forb) each with its own density byte. kGuideBake samples these once/texel/frame and bakes the compact mix into words 6-7 so the per-frame cost stays the measured 0.26 ms.

ENCODING OF EACH REQUIREMENT:
- COMPOSITION AS A MIX (not a single value): dominant+subdominant species ids + blend weight → the shader lerps two LUT-array slices; the overlay-palette plane extends this to a true stack.
- DENSITY / FLUFFINESS / HEIGHT / VIGOR: the continuous coverage plane selects a rung on a BAKED density/height LUT ladder (short-sparse→tall-dense, varying fiber count IN THE BAKE) and lerps the two nearest rungs. Because the runtime output is a continuous extruded BODY with real depth+self-occlusion, "less dense" renders as a genuinely shorter/thinner/patchier sward — the forbidden "sparser blades" mechanism is STRUCTURALLY unreachable (there are no runtime blades to thin). This replaces densityAt's single scalar (verified single-scalar collapse, 292-324).
- PATCH STRUCTURE with VARYING patchiness: cook-side. Truncated-plurigaussian facies gives categorical species with a variogram/correlation-length that TUNES cluster scale per region (short range=near-random speckle, long range=big clumps — "some regions clustered, some random") and a truncation rule that TUNES species adjacency (sedge only next to wet-hollow sphagnum). Domain-warped Worley bends patch OUTLINES organic; modulating the warp gain spatially varies the patchiness itself. Patch EDGES finer than the 0.84 m guide grid are realized by seeding the hex-detile node hash from the clump-id (a hex cell = a clump cell) so boundaries snap to hex cells + blue-noise dither, not to the guide grid.
- CONTEXT-DRIVEN derivation (cook-side, from REAL cooked layers): canopy height → less+shorter grass, more moss under canopy (field.canopyAt already wired, 307); wetness/hydrology → bog palette (sedge+sphagnum+cotton-grass); soil/fertility+biome+classId → lush meadow vs barren; elevation/slope/aspect modulate coverage and species (slope+moisture already feed densityAt, 299-321) — now they feed a MIX, not one scalar.
- OVERLAY LAYERS: the overlay-palette plane names the stacked types per texel; the shader composites them by a single bounded nearest-DEPTH union (2-3 taps, hard-gated), the depth-correct gl_FragDepth output making honest inter-layer occlusion possible without a sort.
- PER-TYPE WIND: a per-type STIFFNESS GAIN byte (moss→0, grass→moderate, sedge/cotton-grass→whippy) folded into the existing per-texel wind-amp block (533-538, 581-609); the shared low-res world wind vector field drives the oblique-TBN inclination bend, scaled by this gain.

SHADER READ COST: on the above-sward SKIP path, nothing extra (words 6-7 ride the one 16B ctx load). On the descend path, the mix is already in-hand from the same fetch. Streamed planes are sampled ONCE per column in kGuideBake, never per march step. Extra VRAM: words 6-7 are already allocated (currently zero); +1 small streamed plane only for rich overlay.

EXTENSIBILITY: a new cover type = one new LUT-array slice + one new facies id + (optionally) one palette entry. Zero rearchitecture. GRID-LESSON GUARD (mandatory): every discrete index derived from the continuous field MUST be blue-noise/IGN-dithered at cell grain and lerp the two nearest variants, or patch boundaries band to the 0.84 m grid — the documented residual.


---

## Appendix B — Fit evaluation (premise-audit + per-requirement verdicts)

**Premise-audit:** Applying GO-UP-A-LEVEL to the task ("optimize + generalize the Sannikov raycast lane") surfaces THREE things filed under "given" that are actually the generators of the failures, so the frame is ~70% right but mis-attributes all three defects to the lane's internals. (1) FALSE PREMISE, biggest: "our lane IS the Sannikov O(1) technique." It is NOT. Verified in code — kRay (NaniteGrass.ts:742-1642) is a compute DDA that pays a fetch-CHAIN per 0.84 m texel crossed PLUS refetches per root-cell rejection, because world-space density + swirl-bombing are evaluated INSIDE the inner march, changing the effective tile every texel. The article's O(1) is a rasterized hull doing ONE fetch per fragment. So the perf problem is generated by WHERE density/bombing is evaluated (inner loop), not by the fetch or the method — "optimize the march" stays inside the wrong problem; the lever is restoring one-fetch-per-fragment. (2) D1 ("only blades, no species/patches") is NOT generated by "the tiling/single-tile structure" — tiling is a separate repetition defect fixed by detiling. D1 is generated one level up by a MISSING COOK-SIDE CONTROL FIELD: densityAt (292-325) collapses biome/moisture/slope/canopy to ONE scalar, then an i.i.d. white-noise cellHash accept per 10.5 cm cell (zero spatial correlation) — no species channel exists anywhere (spare words 6-7 written 0, confirmed). Patches, species and structured density were never REPRESENTABLE, not merely untuned. The user's own new HARD-REQUIREMENT control field IS the dissolution. (3) D2 (jitter) is generated by the FRAME-INTEGRATION SETUP, not the raycast method: kRay builds its GEOMETRY ray from the jitter-mirrored VP (NaniteFrame.ts:515), so the analytic surface physically crawls sub-pixel every frame, AND emits zero motion vectors on a wind-animated surface (documented in the 07-03 ledger). Both are upstream setup choices (which matrix feeds kRay + no velocity target) that GENERATE most of D2 before any resolver tuning. Secondary generators: 8 angular LUT slices (bake structure) and the ragged-top hash anchored to the wind-SHEARED coordinate (a hash-anchoring choice). Net: raycast is the right FRAME for the majority of cover, but the perf/D1/D2 fixes all live one level up (the march-vs-fetch architecture, the cook field, and the frame setup) — three problems that dissolve when you change the context instead of grinding inside the lane. The one place the frame is genuinely WRONG is the implicit "ONE monolithic raycast lane, indistinguishable from real, for ALL cover including tall hero grass" — see alternativeIfRaycastWrong.


**Fit — speed:** YES — significantly faster is realistic, and the memory-locality wall is ~1 ms-class, not a blocker. FIRST, ground truth: +14.5 ms is a STALE 07-04 look-pass number still in the header; the last COMMITTED measurement after shipped levers A (field-bake 23.1→12.9) and B (quad-march 12.9→3.7) is +4.0 ms @dpr2 / +3.3 ms quad @dpr1.5. So "faster than +14.5" is nominally already met — that is a trap, because the real bar is ≤2 ms AND the generalization (species/overlay/SSS/detiling/motion-vectors) ADDS cost on top of +4.0. A FRESH .gputrace is MANDATORY (standing rule) before any perf claim ships. The path to ≤2 ms is STRUCTURAL, not tuning: (a) collapse O(texels+rejects) back to true O(1) by lifting density/species/patch SELECTION out of the inner march into the cook-side field sampled once per fragment/column, and baking the bombing rotation into LUT variants — so the effective tile no longer changes mid-ray; (b) skip-to-sward-top via the existing L2 coarse + terrain-Lipschitz vertical skip, then ONE trilinear tap answering the whole column instead of a per-texel fetch chain; (c) eliminate the reject/refetch structure (the pre-fix 0.03 m stepping measured +6.0 ms) by validating against the cook field BEFORE the fetch, never marching to discover; (d) tiles ≤128² RGBA8, ~1 texel/pixel, mips for distance LOD. The O(1) cost is genuinely bounded by texture-cache locality of the single tap across neighbouring fragments — that IS a real wall — but the article hit ~1 ms at 1080p on a GTX750, so on our GPUs the single-fetch form is plausibly in the ≤2 ms class even at retina (the quad-march already exploits dpr≥1.75). HONEST LIMIT: our renderer is a compute vis-buffer, not a raster-hull fragment shader, so "rasterize a hull, one fetch" maps to "skip-then-single-tap in kRay" — real re-architecture of kRay from a DDA loop, not a knob. Confidence: HIGH that it beats the current march substantially; MODERATE that ≤2 ms holds across ALL cover + the added photoreal shading at retina without a distance/detiling LOD ladder doing real work.


**Fit — photoreal grass (D1):** PARTIAL-YES, split by grass regime — and the split is the honest core of the answer. The MECHANISMS for every D1 sub-requirement exist and are buildable: (1) ≥10 species = a multi-species LUT texture-array, each tile MESH-BAKED from real reference (kills the author's own 'curls/seaweed' failure); runtime picks the layer per-fragment from the cook field — ONE fetch regardless of N species (blade-count independent). (2) Irregular patches with VARYING patchiness = a cook-side truncated-PLURIGAUSSIAN facies field (the machinery ALREADY exists in repo: BOG-V6-PLURIGAUSSIAN-DESIGN.md) — the variogram/correlation-length tunes cluster scale ('some regions clustered, some random'), the truncation rule tunes species adjacency; layered with domain-warped Worley for organic outlines. Free at runtime (streamed plane). (3) REAL density = a baked density/height LUT LADDER indexed by a continuous coverage field; because the output is a continuous extruded BODY with real depth/self-occlusion, 'less dense' renders as a genuinely shorter/thinner/patchier sward — the forbidden 'sparser blades' trick is STRUCTURALLY impossible (no runtime blades to thin). INDISTINGUISHABLE-FROM-REAL verdict by regime: for SHORT-TO-MID grass, sedge, cotton-grass, lichen, herb rosettes — YES, plausibly, once you stack mesh-baked silhouettes + variance-preserving hex-detiling (Mikkelsen/Heitz) + Frostbite SSS + baked AO/self-shadow + per-patch color from moisture/curvature fields. For TALL, side-lit, strongly-WAVING hero grass — this is the method's genuine weak regime and I will NOT rubber-stamp 'indistinguishable': the article's own stated limits (tiling only hideable-not-removable; non-parallel + variable-cross-section fibers only APPROXIMATE and warp at grazing angles) bite hardest exactly where tall grass lives (large wind deflection = non-parallel + variable section + grazing silhouettes). AAA does not use precomputed-raycast for hero tall grass for this reason. See alternativeIfRaycastWrong for the honest tall-grass path.


**Fit — moss:** STRONG YES — moss is the method's single strongest case, and it directly inverts the geometry-moss failure. Sphagnum cushions are SHORT + dense + rounded → a tiny LUT tile → tightest cache locality → the CHEAPEST single tap of the whole system. Represent it NOT as fibers but as a baked height/parallax CUSHION carpet (POM/relief lineage, or folded into a Sannikov cushion-LUT with a low rounded cross-section) tested against the existing per-texel ground plane gP (1086-1088) + topOut — reusing the vertical-skip test (986-993), emit path (1356-1390) and rayNrmTex/resolve contract UNCHANGED (a new in-sward branch at bakedTexel() entry, 1012, keyed by per-texel type=moss). The photoreal cues that make it read as real, wet sphagnum rather than the rejected flat floating brown pancakes: (a) real depth + self-occlusion + micro-parallax from the baked cushion heightfield; (b) Frostbite wrap-translucency (Barré-Brisebois: I_back = pow(saturate(V·−normalize(L+N·δ)),p)·s·thickness) with a baked local-thickness channel → the vivid-green→yellow→ochre→rust translucent glow when back-lit; (c) baked horizon/occupancy AO darkening the crevices (grass is dark at the roots — the volume cue the current lane lacks); (d) biplanar albedo (2 taps, UV-free) at the honest 3D hit for the speckled wet-substrate look; (e) height/curvature/MOISTURE-driven color from the cook wetness field (water-darkening where wet). (f) wind amp = 0 (moss is rigid — a per-type stiffness gain, honest absence). The geometry-moss attempt failed because it was scattered discrete geometry with no depth/translucency/self-shadow; a baked-depth cushion carpet is the exact opposite and plays to every strength of the lane.


**Fit — overlay / density / wind:** YES for all three, with ONE cost-dangerous seam flagged honestly. DENSITY: covered above — baked LUT ladder indexed by a continuous cook field; real shorter/thinner sward, forbidden trick structurally unreachable. Not the 'sparser blades' knob. WIND: per-type response via a SHARED low-res world wind vector field (GoT pattern — one field sampled by grass/trees/particles so gusts sweep coherently across the whole scene) driving the oblique-TBN inclination shear (real fiber BENDING, geometry not noise) with a per-TYPE STIFFNESS GAIN folded into the existing per-texel wind-amp block (581-609): moss→0, grass→moderate, sedge/cotton-grass→whippy (Crysis main-bend+detail-bend hierarchy, height²-scaled, phase-desynced per cell). Coherent gusts via Lagrangian advection of the detile sample coords (tsVGRd scrolling variant). CRITICAL COUPLING: this animated shear is a prime D2 shimmer source — it MUST ship WITH motion vectors (reproject the reconstructed world hit B through the prev-frame VP into a velocity target — the lane already computes B and holds both VPs) AND with the ragged-top hash re-anchored to the wind-INVARIANT fiber coord, or it re-creates the jitter. OVERLAY (moss under grass under forbs): honest via NEAREST-DEPTH union (min-over-d) of LUT fetches — the depth-correct gl_FragDepth output is exactly what makes correct inter-layer occlusion possible with no sort. THIS is the one seam needing real redesign, not a plug-in: N independent marches blow budget (the current golden-angle L2 overlay measured +11.4 ms UNGATED, ~0-1 ms with the 3/px gate). The union MUST be answered by ONE bounded fetch set (2-3 taps max, hard per-pixel budget), NOT one march per layer — species pre-SELECTED by the cook field before the fetch, layers composited by a single min-depth pass. Buildable (the L2 golden-layer machinery at 1437-1601 is the structural template — swap same-tile-rotated fetch for a different-species tile) but it is the highest-risk piece.


**Fit — control field:** YES — this is the strongest STRUCTURAL fit of the whole design; the field attaches to the EXISTING guide/ctx machinery almost for free. HOW IT ATTACHES: the guide record is already a 32 B/texel line-aligned struct (confirmed at NaniteGrass.ts:349) with words 6-7 SPARE and written zero, read FREE inside the existing mask uvec4 fetch (words 4-7 = one cache line, touched only on the descend branch — so the skip path stays 1×16 B load). The election body bits 0-5 are also free (cellSlot<<6) to carry a type-id to the resolve. CHANNELS/LAYOUT (an EVOLUTION of ctx = ground/grad/topOff/gustAmp + density mask): pack a COMPOSITION MIX into the spare 64 bits — e.g. dominant species id (8b) + subdominant id (8b) + blend weight (8b) + per-type density/height/vigor (8b) + clump-id (16b) + moisture (8b) + canopy-proximity (8b) for a 2-type mix; a richer co-located mix (moss+grass+sedge+forb each with own density) needs a SECOND small streamed plane (palette-index scheme). DERIVATION (cook-side, asset-gen): the categorical species/patch field from truncated plurigaussian facies (variogram = cluster scale, truncation rule = adjacency: sedge only next to wet-hollow sphagnum, etc.) over the REAL cooked layers — canopy height (less+shorter grass, more moss UNDER canopy), wetness/hydrology (bog→sedge+sphagnum+cotton-grass), soil/fertility (dry/fertile→lush meadow), biome, classId, elevation/slope/aspect; continuous density from a coverage/height-mask (Outerra-style, shapes bare gaps + tall/short as sward SHAPE); patchiness varied by modulating the warp gain / correlation length spatially. STREAMING: terrain-tile-attached planes exactly like biome/landcover, sampled through the existing 4-corner-BILINEAR guide plumbing (470-501) that already guarantees C0 continuity across texels (grid-lesson-safe). COST OF EXTRA SAMPLES: ~free on the skip path (spare words ride the existing fetch); +1 small plane tap only if the richer overlay palette is needed; composition is read ONCE per fragment/column, never per march step. HONEST LIMITS: (a) the discrete LUT-INDEX derived from the continuous field MUST be blue-noise/IGN dithered at cell grain + lerp the two nearest variants, or patch boundaries snap to the 0.84 m guide grid (the documented grid lesson); (b) fine SUB-texel species interleave inside one 0.84 m texel needs a 2nd overlay population or an id-repack (the LUT A-channel is spent on root-cell id, so species is chosen per-texel not per-fiber); (c) patch EDGES finer than 0.84 m rely on the detiling hex-hash seeded from the field (a hex cell = a clump cell) + dither, not the field's own resolution; (d) it is a control field, not a sim — no dynamic ecological change without a re-cook. EXTENSIBILITY: new cover types = new LUT-array slice + new facies id; zero rearchitecture.


**Honest verdict:** Raycast is the RIGHT foundation for most of this, at moderate-to-high confidence — but only after the frame is corrected on three points the task filed under "given." (1) Our lane is NOT the Sannikov O(1) technique; it's a compute DDA march, so the perf win is architectural (restore one-fetch-per-fragment by lifting density/species/patch selection out of the inner loop and baking bombing into LUT variants), not march-tuning — and it plausibly clears the ≤2 ms law because the O(1) form is ~1 ms-class cache-bound, though a fresh trace is mandatory and retina + added shading make ≤2 ms across ALL cover a moderate-confidence bet. (2) D1 ("only blades, no patches/species") is a missing COOK-SIDE control field, not a rendering limit — fully solved cook-side (plurigaussian facies + clump/density/coverage fields, machinery already in repo) + a multi-species mesh-baked LUT array + a density ladder, all essentially FREE on top of a correct single-fetch lane and attaching to the existing guide record's zeroed spare words. (3) D2 is a frame-integration bug (jittered geometry ray + no motion vectors), cheap to fix and orthogonal to perf. Moss is the method's STRONGEST case (cheapest tile, inverts the geometry-moss failure into a baked cushion carpet with real depth + SSS). Overlay is the one cost-dangerous seam (must be a single bounded min-depth union, not N marches). The one honest caveat I will NOT paper over: tall, side-lit, waving HERO grass sits in the raycast method's genuine weak regime (tiling + non-parallel/variable-section warp), so "indistinguishable from real" for that specific slice may require a GPU-blade accent path on the same control field — a disclosed hybrid fallback, not an impossibility. Net: build the type-keyed hybrid on one control field; short/mid/moss cover reaches the photoreal bar on the raycast lane, the perf bar is reachable via O(1) restoration on a fresh trace, and the tall-hero-grass ceiling is the item to surface to the user rather than silently accept.


---

## Open questions for the user (genuine scope decisions)

1. TALL HERO GRASS — the one honest ceiling: are you willing to accept a slightly-lower photoreal ceiling for tall, side-lit, strongly-waving grass on the raycast body (mitigated with mesh-baked silhouettes + detile + more angular slices), OR do you want the disclosed GPU-blade accent path for that minority slice? Note the accent path uses GPU-amplified blades driven by the SAME control field — arguably within the letter of your 'no scattered discrete geometry' rejection, but a gray zone I will not decide silently. Everything short/mid/moss reaches the bar on the raycast lane regardless.

2. PERF BAR precision: the standing law is ≤2 ms, but 'significantly faster than today' is also stated as the bar. If ≤2 ms across ALL cover + the added photoreal shading at retina proves to need aggressive distance-LOD compromises, do you want (a) hard ≤2 ms even if the far field simplifies more, or (b) 'as fast as the O(1) form honestly allows, comfortably under today' with richer far cover? This changes how hard the LOD ladder pushes.

3. SPECIES SET + REAL REFERENCE: to mesh-bake ≥10 species I need the actual Estonia cover palette and real reference (photos/forestry growth-forms) per species — bent-grass, sedge, cotton-grass, rush, sphagnum, lichen, herb rosettes, etc. Do you have a preferred species list / reference source, or should I derive the palette from the cooked biome/wetness/classId layers and ground silhouettes in public Baltic/Nordic botanical reference (zero external budget)?

4. MESH BAKER investment: the CPU reference baker is ~3 days/mesh; a GPU baker is a prerequisite to iterate ≥10 species at acceptable velocity. Do you approve building the GPU baker as part of Phase 3 (real up-front cost, but unblocks all species authoring), or do you want to start with a smaller hand-authored species set to prove the pipeline before investing in the baker?

5. INTERACTIVITY scope: the wind design uses a shared ambient world wind field. Do you want the God-of-War-style interactive displacement (grass parts as entities move, with spring-back) now, later, or never? It is a purely additive channel on the same wind plug point but needs an entity-splat pass — deferrable without rearchitecting.

6. COOK-SIDE BUDGET: the control field adds several streamed planes (species/facies, clump-id, coverage/height, moisture, optional overlay-palette) to asset-gen, with cook time + streaming VRAM cost and a re-cook to change anything. Confirm this is in-budget, and confirm whether rich co-located overlay (moss+grass+sedge+forb each with own density, needing the 2nd palette plane) is required for v1 or can start as a 2-type mix in the free spare words.

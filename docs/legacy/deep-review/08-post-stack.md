# 08 — Post stack: the ~6ms pool (AO / contact / bounce / TRAA / bloom — and what is NOT in it)

Reader 08. Sources read in full: src/render/PostStack.ts (688), src/render/Gtao.ts (362),
src/render/HalfResMrt.ts (138), src/sky/Clouds.ts (353), src/sky/Atmosphere.ts (429),
src/gpu/passes/Froxels.ts (257), src/gpu/passes/ProbeGI.ts (391),
node_modules/three/examples/jsm/tsl/display/TRAANode.js (767), BloomNode.js (534),
plus ForestScene/TerrainScene/NaniteFrame wiring. Numbers from the 2026-07-02 table and
scratchpad JSONs (`fresh-ablate-post3.json`, `fresh-bead-v2-base.json`, `fresh-noleaves-now.json`).

## 1. TL;DR

- **The canonical scene has NO clouds, NO froxels, NO ProbeGI.** ForestScene builds
  `new PostStack(engine, sunSky.atmosphere, bootTod)` — clouds/froxels default null
  (ForestScene.ts:454); Clouds/Froxels/ProbeGI are instantiated only in TerrainScene
  (TerrainScene.ts:335/368/113). The measured 6.7/5.2/~0 pool = **AO + contact shadows +
  SS bounce + TRAA + bloom + ~17 passes of overhead**. `ablate=clouds` is a no-op in forest.
- Biggest predicted single item: the **full-res 12-step contact-shadow march** (PostStack.ts:424–470),
  ~2.5–3.5ms at eye, ~1–1.5 at oblique. Lever: half-res it into the existing merged MRT (Class R, gated).
- Class-I hygiene (TRAA history ping-pong, kill 2 full-res copies, AO attachment rg16f, bloom pass
  fusion): ~−0.8..−1.5ms GPU all poses + ~1ms cpu.submit.
- "Post ≈ 0 at aerial" is NOT explained by code — AO/contact/TAA/bloom all engage at aerial.
  Leading hypothesis: aerial is partially submission-bound (bubbles hide post GPU work) + bimodal
  medians. Probe requested.
- Honest ceiling: post can contribute ~**−2..−2.6 oblique / −2.5..−3.8 eye**. It cannot close the
  −16 oblique gap alone; its unique value is pose-independent fixed-cost cuts that also help live CPU.

## 2. How it works today (mechanism walk)

### 2.1 The chain in the canonical forest scene

`Engine.renderStep` → `post.meter()` + `post.render()` (Engine.ts:157–158); in the nanite frame,
`post.render()` runs after cull/raster/HZB/shadow (NaniteFrame.ts:494). `PostStack.render()` syncs
camera uniforms then `this.post.render()` (PostStack.ts:683–686). The RenderPipeline executes, in order:

1. **Scene pass** — `pass(scene, camera)` (PostStack.ts:133), MRT = `{ output }` only
   (PostStack.ts:161; the header comment at PostStack.ts:3 claiming color/normal/velocity/depth is
   STALE — velocity is only attached under `?skyveldbg`). Full-res rgba16f + depth (three PassNode
   default HalfFloatType, node_modules/.../PassNode.js:259). Contains the nanite resolve quad + sky.
2. **HalfResMrt pass** (PostStack.ts:247, HalfResMrt.ts:100–118) at 0.5× = 1134×736: two rgba16f
   attachments — `ao` (GTAO, Gtao.ts:102–305) and `bounce` (8-tap depth-gated gather,
   PostStack.ts:214–241). In the world scene a third `clouds` attachment marches Clouds.ts:282–352.
3. **Composite RTT** — `traa(withBounce, …)` wraps its input in `convertToTexture`
   (TRAANode.js:767), so the whole composite `aerial haze × aoFaded × contact + bounce`
   (PostStack.ts:264–491) rasters into an EXTRA full-res rgba16f target. This pass is where the
   AO joint-bilateral upsample (PostStack.ts:373–419), the contact march (424–470), the froxel/haze
   application (279–283, froxels null in forest) and the bounce composite (484–491) execute per pixel.
4. **TRAA resolve** (TRAANode.js:404–408): full-res rgba16f. Per pixel: 3×3 depth loads
   (TRAANode.js:505–534), 1 velocity load = our analytic `velReproject` (PostStack.ts:520–536, one
   depth load + 3 mat4 muls), prev-depth sample + reprojection (537–548), beauty sample + 8 neighbor
   loads for variance clipping (580–601), clip + flicker math.
5. **Two full-res copies**: resolve→history rgba16f ≈ 26.7MB (TRAANode.js:412) and scene-depth→
   history-depth float32 ≈ 13.4MB (TRAANode.js:423–428). Every frame.
6. **Bloom** = 13 passes (BloomNode.js:299–335): bright pass at half-res, then 5 mips × 2 separable
   blurs (kernels 6/10/14/18/22, BloomNode.js:373), then composite. All rgba16f.
7. **Final output pass**: `graded` (PostStack.ts:612–628) — samples TRAA output + bloom, exposure
   multiply, WB/split-tone/saturation/contrast/vignette/grain, AgX via toneMapping.
8. **autoExposure compute** (PostStack.ts:560–594): a SINGLE thread samples a 12×12 grid (144 taps)
   of the beauty, EMA 0.07 into a 2-float storage buffer. `?lockexp=1` freezes it (PostStack.ts:670).

Total: ~17–19 encoder passes for post alone. Consistent with cpu.submit: `fresh-ablate-post3.json`
med cpuSubmit 1.8/2.2/2.9 vs baseline 5.3/5.5/5.2 — ablating post removed ~3ms of CPU encode
(caveat: cpuSubmit also fell to ~1.4 in the voxbocc run, so it partly tracks GPU backpressure, not
pure encode count — do not double-book this 3ms).

### 2.2 What each stage costs — code-derived predictions (P5 must verify)

Measured pool (med gpuWall): eye 36.0→29.3 = **6.7**; oblique 43.1→37.9 = **5.2**; aerial
16.8→16.9 = **~0** (§3.6). Per-item predictions, eye/oblique/aerial ms:

| item | scales with | prediction e/o/a | mechanism |
|---|---|---|---|
| contact march | full-res px with dist<240 (PostStack.ts:431) × ≤12 steps ×(2 mat4 + 1 depth tap) | 2.5–3.5 / 1.0–1.5 / 1.0–2.0 | eye: nearly all px <240m; oblique: ~⅓; aerial (cam 150m): most px in 150–240m band. Early-exit (443–463) only helps HIT px |
| GTAO march | half-res px with dist<1800 (Gtao.ts:118) × 12 taps + normal-from-depth | 0.8–1.4 / 0.8–1.4 / 0.8–1.2 | samples=6 ⇒ DIRECTIONS=3, STEPS=2 (Gtao.ts:143–144); everything in-forest is inside the 700m fade start ⇒ fully engaged at all poses |
| AO bilateral | full-res px with k<0.995 (PostStack.ts:416) × 4 rg-taps | 0.3–0.5 each | early-out only fires beyond 1800m |
| bounce | half-res non-sky px × 16 taps (8 depth + 8 beauty) | 0.4–0.8 each | PostStack.ts:225–237 |
| TRAA resolve + 2 copies | full-res, fixed | 1.2–2.0 each | ~20 loads/px + 40MB of copies |
| bloom (13 passes) | half-res mip chain, fixed | 0.4–0.7 each | ~25M taps + pass overhead |
| composite RTT write | full-res rgba16f write + aerial haze math | 0.2–0.4 each | structurally required by TRAA's variance-clipping neighborhood (§4 L-REJ-1) |
| exposure kernel | 1 thread × 144 dependent taps | ~0.05–0.1 | latency-bound single thread + a pass boundary |

Sums: eye ≈ 6.5–8.0 (measured 6.7 ✓), oblique ≈ 4.9–6.5 (measured 5.2 ✓), aerial ≈ 4–6
(measured ~0 ✗ — the doc's sharpest open question, §3.6/§6).

### 2.3 World-scene-only subsystems (NOT in the canonical pool — do not book savings here)

- **Cloud march** (Clouds.ts:282–352): 32 steps × (3 density taps + 3×2 sun-occlusion taps when
  dens>0.002) at half-res; sky rays cross the slab up to 26km (tExit clamp Clouds.ts:292). At an
  aerial/downward ray the slab test fails (tExit<tEnter, camera below CLOUD_BOTTOM=1250 looking
  down ⇒ both t negative, Clouds.ts:286–294) and the whole march skips — in the WORLD scene this,
  plus zero sky fraction, is why clouds cost ~nothing top-down. No transmittance early-exit exists
  (the Loop at 319–349 always runs all 32 steps even at trans≈0).
- **Froxels** (Froxels.ts): 160×90×64 = 921k-thread scatter + integrate EVERY frame
  (Froxels.ts:195, 234–240), each froxel doing ~7 heightfield fetches + cloud shadow.
- **ProbeGI** (ProbeGI.ts:329–334): 3072 probes/frame × 16 dirs × 16 march steps.
  **Consequence for the premise audit's P3:** the "GI probe cycle" bimodality suspect CANNOT apply
  to the canonical forest scene — ProbeGI is never constructed there (only TerrainScene.ts:113).
  The oblique bimodality lives in a scene with no ProbeGI, no clouds, no froxels. P3 as specified
  would measure a mechanism that does not run. Replacement probe in §6.
- **Caustics** (src/render/Caustics.ts): water scenes only; forest has no water surface.

## 3. Waste map

1. **Contact-shadow march at full res** (PostStack.ts:424–470). Output is a smooth scalar cue
   (floor 0.4, fade 140–240m, PostStack.ts:464–467) yet it pays full-res march cost — the single
   largest predicted post item at eye (~3ms). AO and clouds already ship at half-res with a
   joint-bilateral guide; contact does not use that machinery. Cost model: px(dist<240) × 12.
2. **Two full-res copies per frame for TRAA history** (TRAANode.js:412, 423–428): ~40MB/frame of
   pure bandwidth + 2 encoder ops, replaceable by ping-pong targets. Fixed ~0.3–0.6ms + CPU.
3. **AO attachment is rgba16f but carries 2 channels** (vec4(ao, viewZ, 0, 1), Gtao.ts:304;
   HalfResMrt.ts:60–74 already supports per-entry format overrides). rg16f halves its write +
   4-tap-read bandwidth. Fixed ~0.1–0.3ms.
4. **Bloom = 13 encoder passes** (BloomNode.js:299–335) for a subtle 0.28-strength effect: bright
   pass is a separate raster that could fold into the first blur; the whole chain is a candidate
   for a fused compute downsample. GPU small (~0.2–0.4) but ~13 passes of CPU encode.
5. **Composite RTT** is an extra full-res rgba16f surface that exists only to feed TRAA a texture.
   Structurally justified (§4 L-REJ-1) but its FORMAT (and TRAA's history/resolve, TRAANode.js:145,
   154) could drop to rg11b10 — Class R precision trade, common in shipped TAA.
6. **Sky pixels**: already cheap — AO (depth≥1 ⇒ 1, Gtao.ts:118), bounce (isSky skip,
   PostStack.ts:217–218), contact (isSky skip, 428–431) all early-out; TRAA intentionally runs on
   sky (silhouette AA). No lever here in the forest scene.
7. **AO on pixels the fade zeroes**: already handled BOTH ends — the march skips past
   maxDist=aoFadeFar (Gtao.ts:69, 118) and the bilateral skips at k≥0.995 (PostStack.ts:416).
   The remaining "waste" is that aoFadeNear=700m means effectively no forest pixel ever fades —
   but tightening the fade is a visible AO change = Class R, red-flagged (§5).
8. **Exposure meter**: 1 GPU thread, 144 serial taps + a dedicated compute pass every frame
   (PostStack.ts:560–594). Micro (~0.1ms + pass), fold into bloom's bright pass someday. Low value.
9. **Auto-exposure feedback is a frame-coupled loop**: exposure EMA (0.07) → graded luminance →
   next frame's meter; bloom's high-pass coverage and the bilateral's `directK` branch
   (PostStack.ts:382) both depend on absolute luminance. A slow oscillation of the meter would
   oscillate GPU cost with a multi-frame period — a candidate mechanism for the oblique/aerial
   run-of-4 bimodality that lives in MY slice and is testable for free with `?lockexp=1` (§6 P-C).

### 3.6 The aerial contradiction

Code says AO (fully), contact (mostly), TRAA, bloom all engage at aerial [0,150,0]; predicted post
≈ 4–6ms. Measured Δmed ≈ 0 (16.8→16.9), Δp25 = 1.4 (16.0→14.6), distributions overlapping
(`fresh-ablate-post3.json` aerial min 6.5 / max 19.7). Two candidate explanations, both testable:
(a) bimodal 32-frame medians (premise audit §6.6) hide a real few-ms effect; (b) aerial is partly
**submission-bound** — cpuSubmit ~5ms across many submits means the GPU drains cheap aerial passes
faster than the CPU feeds it, and post GPU work hides in those bubbles (consistent with the
pixel-law aerial fixed term ~5ms and with post ablation cutting cpuSubmit 5.2→2.9 while gpuWall
stood still). If (b), post levers buy ~nothing at aerial but still pay at eye/oblique/live.

## 4. Levers (ranked)

**L1 — Half-res contact shadows inside the merged MRT pass.**
Mechanism: move the 12-step SSCS march (PostStack.ts:424–470) into the HalfResMrt pass, packing
occlusion into the AO attachment's unused .z (Gtao.ts:304); upsample with the SAME joint-bilateral
that AO already uses (viewZ guide in .y, PostStack.ts:388–410). Quarter the march cost, zero new
attachments, zero new passes (respects the 10-storage-buffer cliff — this is all textures).
Files: src/render/Gtao.ts (or a sibling layer fn), src/render/PostStack.ts.
Expected: eye −1.5..−2.3, oblique −0.6..−1.1, aerial −0.8..−1.5 *if* aerial post is real (§3.6),
live: helps every frame with near geometry.
Quality: **RISK** (resolution cut on a smooth cue; precedent: AO and clouds are already half-res
and shipped; the existing per-pixel jitter PostStack.ts:434 + TRAA absorb). Gate: shotdiff at 3
canonical + 2 stress poses, jitter index pinned, then user sign-off. Effort: M.
UE5: contact shadows are a short per-light screen trace with ~8 steps and aggressive early-out;
consoles commonly run them ≤half-res.

**L2 — Class-I hygiene bundle: fork TRAANode for ping-pong history + rg16f AO + bloom bright-fold.**
Mechanism: (a) two history RTs alternating write/read kill the 26.7MB resolve→history copy
(TRAANode.js:412) — resolve IS the new history; keep the prev-depth copy (scene pass owns its depth).
We already inject velocity through a seam (PostStack.ts:534–536) and mirror `_jitterIndex`
(NaniteFrame reads it, PostStack.ts:60–62,543) — the fork must preserve that field and the
setViewOffset flow (TRAANode.js:289–339). (b) AO attachment RGFormat/HalfFloat via the existing
HalfResMrt override hook (HalfResMrt.ts:66–74). (c) fold bloom's bright pass into its first H-blur
(threshold per tap is ALU-cheap; saves 1 raster + RT round-trip), keep kernels/weights identical.
Expected: −0.5..−1.0 GPU each pose (copies + bandwidth + 2–3 passes) and −0.5..−1.0 cpu.submit.
Quality: **IDENTICAL** (bit-equal: same math, same fp16 storage; gate with shotdiff maxDiff=0
anyway). Effort: M (TRAA fork is the bulk).

**L3 — RT format diet: rg11b10 for TRAA history/resolve + composite RTT.**
Mechanism: HalfFloatType → R11G11B10 (renderable) on TRAANode RTs (TRAANode.js:145,154) and the
convertToTexture target. Halves ~80MB/frame of full-res traffic to ~40MB.
Expected: −0.3..−0.6 each pose. Quality: **RISK** (mantissa loss in dark HDR values can band under
AgX; shipped TAAs do it, but this project's doctrine says gate it). Gate: shotdiff + dark-scene
crop review. Effort: S–M. UE5: TSR history is 10–11-bit packed in several configs.

**L4 — Bloom pass-count restructure (compute downsample chain).**
Only after L2(c): replace the 5×2 separable rasters with a fused compute chain (shared-memory
gaussian, same coefficients). Expected GPU −0.2..−0.4, CPU −0.4..−0.8 (10 fewer passes).
Quality: IDENTICAL if coefficients/edge handling match exactly, else R. Effort: M. UE5's bloom is
exactly this shape (fused downsample/upsample chain).

**L5 — (world scene only, zero canonical value now) cloud-march transmittance early-exit +
froxel cadence.** Add `If(trans < 0.005) break`-equivalent to Clouds.ts:319–349 (Class R, bounded
<0.5% radiance) and quarter-rate froxel scatter with temporal reuse (Class R). Book ZERO ms against
the forest targets; file for the world-scene integration milestone.

Post-pool honest total vs the oblique −16 mission: L1+L2+L3 ≈ **oblique −1.4..−2.6, eye −2.5..−3.8,
live cpu.submit −1..−2**. Post is a supporting theater; the master plan must source the bulk from
foliage/base (per premise audit §2.4).

## 5. Refuted / rejected paths for this stage

- **`ablate=clouds` as a forest measurement** — no-op; clouds is null in ForestScene
  (ForestScene.ts:454). Any past/future "clouds cost X in forest" claim is invalid.
- **ProbeGI as the forest bimodality suspect (premise audit P3, as written)** — refuted by code:
  ProbeGI only exists in TerrainScene (TerrainScene.ts:113,120). P3 must be re-aimed (§6 P-C).
- **AO sample/radius trims** — the look is already minimum-tuned (6 samples, "over-sampled at 8",
  PostStack.ts:84–87); further trims are visible = Class R with low expected ms. Not proposed.
- **aoFadeNear tightening (700→closer)** — visibly removes mid-field AO. Red-listed here.
- **dpr / internal-res scaling of the beauty** — banned by doctrine; not re-derived here.
- **Folding the composite into TRAA's resolve (skip the RTT)** (L-REJ-1) — rejected structurally:
  variance clipping loads 8 neighbors of the COMPOSITED image (TRAANode.js:580–601); inlining would
  evaluate the composite (incl. contact march) 9×/px. Clipping against pre-composite stats instead
  changes TAA behavior = R with real ghosting risk. Keep the RTT; diet its format (L3).
- **`ablate=taa` as a clean TAA cost probe** — biased LOW: with taa ablated, `taaed = withBounce`
  (PostStack.ts:537–539) and bloom's bright pass then inlines the ENTIRE composite at half-res
  (BloomNode.js:353–362 evaluates `inputNode`), double-evaluating aerial+AO+contact. Measure TAA as
  Δ(`ablate=taa,bloom`) − Δ(`ablate=bloom`).

## 6. Open questions + serial GPU probes requested

- **P-A (itemization, replaces premise-audit P5's list):** interleaved singles at eye+oblique:
  `EXTRA=ablate=contact` / `ablate=ao` / `ablate=bounce` / `ablate=bloom` / `ablate=taa,bloom`.
  Notes: skip `clouds` (no-op); `ablate=ao` ALSO drops contact (PostStack.ts:474–477), so
  AO-alone = Δ(ao) − Δ(contact). Decision: any item >1.5ms at oblique becomes a named lever with
  its own quality class; contact >2ms at eye confirms L1 as the post flagship.
- **P-B (aerial reality check, §3.6):** `EXTRA=ablate=ao,bounce,taa,bloom` at aerial, 128 frames,
  report p25/p50/p75 + mode structure. Decision: Δp25 ≥ 2ms ⇒ bimodal medians hid real post cost
  (aerial pool reopens); Δ≈0 at all quartiles ⇒ aerial is submission-bound, post levers deprioritized
  for aerial and the "aerial fixed ≈ 5ms" term goes to the CPU/pipelining reader.
- **P-C (bimodality, replaces P3):** oblique isolated 128 frames with `?lockexp=1`
  (PostStack.ts:670) vs default, same session. Decision: modes collapse ⇒ the auto-exposure feedback
  loop (§3 item 9) owns the run-of-4 oscillation and the fix is cadence/clamp on the meter; modes
  persist ⇒ exposure struck, AND record that ProbeGI/clouds/froxels are already struck by code
  (§2.3), escalating to voxOccPyr/HZB + Dawn/Metal pipelining as the only survivors.
- **P-D (L1 gate):** contact-half-res prototype vs baseline: shotdiff at 3 canonical + 2 stress
  poses (jitter pinned, settle()), plus Δms at eye. Ship gate: user sign-off on crops (Class R) and
  Δ ≥ 1ms at eye; otherwise revert and keep full-res.
- **P-E (L2 gate):** Class-I bundle A/B interleaved in one session; gate = shotdiff maxDiff=0 at
  all poses AND cpuSubmit/gpuWall deltas reported per pose (expect −0.5..−1.0 GPU, −0.5..−1.0 CPU).

Contradictions with prior docs, stated: (1) the 2026-07-02 table's "whole post stack ~6ms" implies
a clouds/AO/TAA stack — in the canonical scene it is an AO/contact/TAA/bloom stack; clouds
contribute 0 by construction (this doc §2.3 vs the table's ablate label). (2) Premise audit P3's
strongest bimodality suspect (ProbeGI) does not run in the measured scene (§2.3, §5). (3) The
PostStack header comment (PostStack.ts:2–10) does not match the shipped MRT set (PostStack.ts:161).

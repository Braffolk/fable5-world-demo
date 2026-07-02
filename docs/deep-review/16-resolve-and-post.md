# resolve + post deep review (2026-07-02)

Owner area: `src/nanite/NaniteResolve.ts`, `src/render/PostStack.ts`, `src/render/Gtao.ts`,
`src/render/HalfResMrt.ts`, `src/render/ColorScript.ts` (+ `src/sky/Atmosphere.ts::aerial`,
three r184 `TRAANode`/`BloomNode` as consumed).

Verdict up front: **there is no big quality-identical lever in this area.** Post is ~6.8/5.1/≈0 ms
(eye/oblique/aerial) and the resolve is confirmed ~free — but partly for a reason the fact pack
missed (see premise audit). Realistic identical-quality recovery here: **~0.5–2 ms at eye,
~0.5–1.5 ms at oblique**, dominated by one speculative occupancy lever. The −11..13 ms oblique gap
does not close here; it closes in raster/coverage.

## Premise audit

**P1 — The forest scene has NO clouds and NO froxels.** `ForestScene.ts:454` constructs
`new PostStack(engine, sunSky.atmosphere, bootTod)` — the `clouds` and `froxels` params default to
`null` (`PostStack.ts:69-75`). The cloud march (`PostStack.ts:176-189`) and froxel apply
(`PostStack.ts:279-282`) are **never built** in any canonical measurement. The fact pack's
"POST … TOTAL for clouds+AO+bounce+bloom+TRAA" is wrong on membership: the measured 6/5/0 is
**GTAO + SS-bounce + screen-space contact shadows + TRAA + bloom + aerial haze + exposure + grade**.
`ablate=clouds` in `fresh-ablate-post3` was a no-op. "Does the merged half-res pass re-march clouds
when static?" — in forest it never marches; in the world scene yes, every frame with a fresh
temporal jitter (`PostStack.ts:183-185`), which is correct-by-construction there (clouds drift).

**P2 — The forest scene has NO shadow system and NO GI.** `ForestScene.ts:456-459` passes
`gi: null, csm: null` (comment at 441: "no GI bounce, no shadows"). `NaniteFrame.ts:244`
(`shadowOn = … && world.csm !== null`) → `shadow = null`, `shadowHalf = null`; in the resolve,
`shadowsOn = world.csm !== null && …` (`NaniteResolve.ts:237`) is **false** → no PCSS sample, no
half-res shadow upsample, no `keep` CSM node, no `receivedShadowPositionNode`. `world.gi = null`
→ no probe irradiance (`NaniteResolve.ts:973-984`) and **no ProbeGI updates at all** ("3072
probes/frame on a 128-frame cycle" does not exist in forest). `SunSky.ts:36` sets
`sun.castShadow = false`.
- Consequence A: the attribution verdicts "lighting ≈ FREE, shadows ≈ FREE" are structurally
  *trivially* true in forest — most of that machinery is absent. `nanshadow=0` (32.6/37.0/15.6 vs
  31.2/38.8/14.6) toggled a system that was already off; the ±1.5 ms deltas are pure noise, which
  is itself a useful noise-floor calibration. These verdicts say **nothing** about world-scene cost.
- Consequence B: "GI probe updates" as a bimodality suspect
  (`2026-07-02-attribution-and-waves.md` §1/§2) is **structurally refuted for the forest scene** —
  there is no ProbeGI to fire periodically. Remaining suspects: voxOccPyr/HZB interactions, GPU
  pipelining sawtooth.
- Consequence C (**one level up — surfacing per standing rule, not parking it**): the canonical
  60 fps scene is missing quality systems (sun shadows, GI, clouds, froxels) that the world scene
  runs. If forest is ever brought to world visual parity, the frame gains an unmeasured +2–6 ms
  class of cost that the current 24–26 ms isolated budget does not include. The locked-60 verdict
  is being validated on a shadowless/GI-less/cloudless frame. This needs a user-visible decision at
  plan level (either "forest look is final as-is" or "budget must reserve headroom").

**P3 — the 6/5/0 asymmetry is NOT sky-fraction/early-outs (there are no clouds). It is
gather-radius-in-pixels ∝ 1/distance.** All three content-adaptive effects use *world-space* radii:
GTAO 1.6 m (`PostStack.ts:205`), bounce 0.55 m capped at 7 % of screen (`PostStack.ts:222`),
contact 1.7 m within 240 m (`PostStack.ts:431,437`). Projected pixel radius ≈ radius/dist ×
(H/2)/tan(fov/2). At eye (surfaces 2–30 m away) that is tens-to-hundreds of pixels → taps scatter
across cache lines, plus high depth variance breaks the AO/contact march coherence. At aerial
(everything ≈ 150 m) every radius is 2–13 px → taps land in-cache, marches are coherent → the whole
stack costs ≈ noise. This is why post is a **near-field** cost. It also predicts: post cost during
live motion tracks how much of the frame is close-range foliage, i.e. eye-like poses set live p95.

**P4 — measurement sanity.** Post totals: eye 36.1→29.3 = **6.8**, oblique 43.2→38.1 = **5.1**,
aerial 16.8→17.1 = **−0.3** (i.e. ≤ noise; ablate ran later = thermally biased against, so
aerial post ≤ ~1 ms). The ablate-oblique p95 of 70 ms is the known ~4-frame spike mode, present
with TAA off — not post. All full-res pose-independent passes (TRAA + bloom + haze + grade + final)
are bounded by the aerial delta: **≤ ~1 ms combined at 3.34 Mpx.**

## How it works today

Pass chain per frame (forest config, dpr 1.5 = 2268×1473 = 3.34 Mpx; half-res 1134×737 = 0.836 Mpx):

1. **Scene pass** `pass(scene, camera)` with MRT `{output}` only — the velocity attachment is
   built only under `?skyveldbg` (`PostStack.ts:157-165`), so no dead rg16f write. Inside it:
2. **Resolve tri pass** (fullscreen clip-space triangle, renderOrder −1000,
   `NaniteResolve.ts:1085-1090`). Per pixel: load `visBV` + `payloadV` (`:359-360`), discard if
   uncovered (`:361`), reconstruct depth+wp from the 24-bit election key (`:365-371`), discard
   voxel-winner pixels (`:380-385`), decode `(instId,ci)` from `qRasterRO` (`:399-403`), matClass
   from clusters/meshes (`:414-421`), then the per-class branches: terrain (`:446-472`, gated
   `pass==='tri'` but **still built** — see waste W2), rock (`:482-508`), bark/deadwood
   (`:517-681`, moss fbm + normal-map gated `dist < resfar=60` at `:634-660`), leaf (`:692-758`,
   full path with `makeCtx` gust textures + 3×`fetchWorldVert` + 3×`readVertex` only within
   `resfar·0.6 = 36 m`; cheap single-vertex path beyond — measured ceiling < 1 ms via
   `leafcheap=all`). Lighting: sun lambert + ambient floor + backlight only (`:893-1020`) — in
   forest `shadowsOn=false`, `gi=null`, so the PCSS/keep/GI blocks (`:922-959`, `:973-984`) are
   not built. `depthNode` re-loads `payloadV` and writes real depth (`:1071-1078`).
3. **Resolve vox pass** (second fullscreen triangle, renderOrder −999, `:1097-1105`): same
   prelude, discards non-voxel pixels (`:386-390`), decodes winning brick via payload bits 21–27
   (`:788-791`), brick normal + bead blend (`:793-833`), brick albedo + world-anchored jitter
   (`:838-868`), wrap lighting + ambient up-bias (`:908-919`, `:997-999`).
4. **Merged half-res MRT pass** (`HalfResMrt.ts:100-118`, one quad → RT with **2** attachments in
   forest, both rgba16f `HalfResMrt.ts:60-64`): `ao` = GTAO port, 6 samples → 3 dirs × 2 steps ×
   2 sides = 12 marched depth taps + `getNormalFromDepth` + noise (`Gtao.ts:102-305`), packs
   view-z in `.y` for the bilateral guide (`Gtao.ts:304`); `bounce` = 8-tap golden-angle gather of
   beauty+depth within ~0.55 m projected (`PostStack.ts:214-241`).
5. **TRAA input RTT** (full-res rgba16f — `traa()` wraps its input in `convertToTexture`,
   `TRAANode.js:767`): evaluates per pixel the whole composite chain: aerial haze
   (`PostStack.ts:264-358`; `Atmosphere.ts:377-403` = pure ALU + one sky-view LUT tap), × AO
   joint-bilateral 4-tap upsample with distance fade + `k<0.995` skip (`PostStack.ts:373-419`),
   × contact 12-step full-res depth march for `dist<240` with first-hit-wins early-out
   (`PostStack.ts:425-470`), + bounce composite (`PostStack.ts:484-491`).
6. **TRAA resolve** (full-res): 3×3 current depth (9 taps) + 1 velocity (our analytic reprojection
   = 1 depth load + matrix math, `PostStack.ts:520-536`) + 1 previous depth + 1 current color +
   1 history color + 3×3 variance-clip neighborhood (9 taps) ≈ **22 taps/px**, then **two full-res
   `copyTextureToTexture`**: resolve→history color and scene-depth→history depth
   (`TRAANode.js:404-427`).
7. **Bloom** mip chain on the TRAA output (threshold + up/down passes, ~⅓ extra full-res-equiv px).
8. **Final quad**: exposure multiply + white balance + split-tone + saturation/contrast + vignette
   + grain (`PostStack.ts:612-628`); ColorScript is CPU-side keyframe lerp → uniforms (zero GPU).
9. **Auto-exposure**: 1-thread compute, 144 fixed taps of the scene output (`PostStack.ts:560-594`)
   — microseconds; `meter()` is `renderer.compute`, no readback (`Engine.ts:157`).

## Work model

- **Pose-independent full-res floor** (haze + TRAA(22 taps + 2 copies) + bloom + final + exposure):
  measured ≤ ~1 ms total (aerial ablate delta −0.3). Traffic estimate agrees: TRAA neighborhoods
  are cache-local (~3 surface-reads-equiv ≈ 80 MB), copies ≈ 2×27 MB rgba16f + 13 MB depth,
  ~400 GB/s M1 Max → ~0.5 ms.
- **Pose-dependent gathers**: cost ≈ Σ_effects [pixels_in_range × taps × miss_rate(radius_px)],
  radius_px ∝ 1/dist. Eye ≈ 6.8 ms decomposes plausibly as AO ~2 (0.836 M × ~17 scattered taps),
  contact ~2–3 (3.34 M × ≤12 full-res taps, nearly all pixels < 240 m), bounce ~1 (0.836 M × 16
  taps, rPx up to 79 px at half-res), TRAA+bloom+haze ~1. **Singles were never measured** — probe
  order below. Oblique 5.1 = same effects at mid distances. Aerial ≈ 0 = all radii ≤ 13 px.
- **Resolve**: 2 fullscreen passes × 3.34 Mpx × (2–3 buffer loads + discard) + winner-pixel decode.
  Decode ceiling measured < 1 ms (`leafcheap=all` −0.9); lighting delta measured −1.3
  (`nandbg=flat`) consistent with lambert+ambient+backlight-only ALU.
- Explained measurements: post 6.8/5.1/−0.3; `nanshadow=0` no-op (system absent); the pixel-law
  per-Mpx slopes (eye 6.0 vs aerial 3.9 ms/Mpx) partly ARE this near-field post asymmetry.

## Waste inventory

- **W1 — terrain+rock subgraphs are dead weight in the forest tri shader.** Forest registers no
  terrain and no rock clusters, so `isT`/`isR` are never true, yet both branches are built
  (`NaniteResolve.ts:446`, `:482`): ~14 implicit-derivative texture samples + fbm/valueNoise ALU
  chains. The identical subgraph, stripped from the *vox* pass, was "measured as the dominant
  driver of the close-up voxel r.scene cliff (37.5 ms inside a crown)" via occupancy collapse
  (`NaniteResolve.ts:439-445`). The tri pass shades every leaf/bark pixel with that register
  pressure. Unquantified; could be 0 (if bark's own texture samples already dominate register
  allocation) or the single largest item in this area. Side waste: the forest boots a full
  `Heightfield.generate` solely to make the never-sampled bindings valid (`ForestScene.ts:437-446`).
- **W2 — haze evaluated on sky pixels then discarded by `select`** (`PostStack.ts:283-285`; both
  select operands always execute): sky-fraction × (~25 ALU + 1 LUT tap) at full res. ~0.1–0.3 ms eye.
- **W3 — AO attachment is rgba16f but uses 2 channels** (`HalfResMrt.ts:60-64` type applies to all;
  `Gtao.ts:304` returns (ao, viewZ, 0, 1)): 2× the needed write + upsample-read bandwidth on a
  0.836 Mpx surface. ~0.05–0.15 ms.
- **W4 — vox pass reconstructs wp before its isV discard** (`NaniteResolve.ts:365-371` emitted as
  statements via `.toVar()` before the `:386-390` discard): the ~60–90 % non-vox pixels pay
  ~15 ALU + the reconstruction chain. ~0.1 ms (compiler may already sink some of it).
- **W5 — the two-pass resolve partition costs one extra fullscreen pass** of payloadV+visBV loads +
  discard over all pixels (~53 MB traffic + ROP/raster overhead) ≈ 0.2–0.5 ms. It exists for the
  Metal ≤10-storage-buffer cliff; in the *forest* config (gi null → no probe buffer) the union set
  is exactly 10: payloadV, visBV, qRasterRO, qVoxRasterRO, clusters, meshes, verts, indices,
  instances, voxelBricks.
- **W6 — TRAA does two full-res copies per frame** (color resolve→history, depth→history,
  `TRAANode.js:412,426`) ≈ 0.2–0.3 ms; avoidable only by forking TRAANode to ping-pong.
- **W7 (world-scene only) — the `keep` CSM full-screen sample** (≡1 against empty maps) + the
  second per-pixel wp reconstruction that exists only to feed it (`NaniteResolve.ts:277-291`,
  `:330-350`, `:938-954`). Default ON (`reskeep !== '0'`). Zero cost in forest (not built, csm
  null); pure waste wherever the nanite shadow is active with a csm reference.
- **W8 (world-scene only) — clouds re-march every frame when static.** Correct for drifting clouds;
  a temporal-reprojection cache is quality-trading (rejected below).

## Levers

All quality classes per the ABSOLUTE constraint (identical = only conservative-cull correctness or
sub-noise numerical change; anything A/B-visible is not identical).

### L1 — `resolve-post:tri-class-specialization` (the only possibly-non-small item)
Build the tri resolve material's per-class branches only for matClasses actually present in the
registry (forest: drop terrain `:446` and rock `:482` subgraphs entirely; also drop the then-unused
hf bindings and skip the boot `Heightfield.generate`). **Quality: identical** — those classes have
zero clusters, the branches can never execute; output is bit-identical. Mechanism: register
pressure / occupancy on Apple GPUs is set by the worst branch; precedent is the measured vox-pass
cliff (`NaniteResolve.ts:439-445`). Expected: unknown-but-bounded; best guess eye ~1 ms,
oblique ~0.8, aerial ~0.3 (could be ~0). Effort M (plumb a class-inventory from WorldRegistry into
`buildNaniteResolve`; keep a `?resclasses=all` revert flag). Risks: TSL graph edit; must not
change world-scene builds (classes present there). Discriminator: flag A/B gpuWall + shotdiff=0.

### L2 — `resolve-post:single-pass-resolve-when-gi-null`
Merge tri+vox resolve into ONE fullscreen pass in configs whose fragment storage-buffer union is
≤10 (forest: exactly 10; world keeps two-pass). Removes W5 (one full-screen pass of loads +
discard + depth write). Quality: identical (same partition, same math). Expected ~0.3/0.3/0.25 ms.
Effort M. Risks: sits EXACTLY at the 10-buffer cliff — the 11th binding silently kills the
pipeline (the `?vcompact` empty-scene lesson); must gate on config and verify with
`tools/vcdebug.mjs` + screenshots; combining with L1 (which frees nothing binding-wise; hf maps
are textures) is safe, but any future 11th buffer breaks it silently.

### L3 — `resolve-post:skip-haze-on-sky`
Wrap `atmosphere.aerial` in `If(isSky.not())` instead of compute-then-select
(`PostStack.ts:283-285`). Quality: identical (select semantics preserved; no side effects in the
discarded branch; sky pixels keep `col`). Expected ~0.2/0.1/0.0 ms (sky fraction × haze ALU).
Effort S. Risk: none beyond TSL branch mechanics.

### L4 — `resolve-post:ao-attachment-rg16f`
Add an `rg?: true` option to `HalfResEntry` (the `red` flag exists but AO needs .y for the packed
view-z guide) and set the AO attachment to RG16F. Quality: identical (channels .z/.w never read).
Expected ~0.1/0.1/0.05 ms. Effort S. Risk: format support is universal for rg16f render targets.

### L5 — `resolve-post:vox-pass-early-discard-order`
Move the vox pass's isV discard (`:386-390`) before the wp reconstruction (`:365-371`) (tri pass
likewise for its bit31 discard). Quality: identical (pure reorder of side-effect-free statements).
Expected ~0.1/0.1/0.05 ms; may be ~0 if Metal already sinks it. Effort S.

### L6 — `resolve-post:reskeep-corner-default` (regression guard, world-scene value)
Flip the default of the `keep` gate to corner-only (`keepFullU=0`, `NaniteResolve.ts:286-291`) —
the code already documents it as bit-identical for real pixels (keep≡1 vs empty maps) with the
cascade fit kept alive by the corner pixel. Quality: identical. Expected 0 in forest (not built);
~0.3–1 ms whenever a scene runs nanite-shadow + csm reference. Effort S. Risk: the cascade-fit
caveat (`:283-285`) — verify the fit still runs (NaniteFrame drives it explicitly).

### L7 — `resolve-post:traa-history-pingpong`
Fork three's TRAANode to ping-pong resolve/history RTs and double-buffer depth instead of two
full-res `copyTextureToTexture` per frame (`TRAANode.js:404-427`). Quality: identical (same
texels, no copy). Expected ~0.2–0.4 ms all poses. Effort M. Risk: fork maintenance across three
upgrades; binding-group churn when swapping RTs each frame.

### L8 — `resolve-post:halfmrt-shared-center-prelude`
AO and bounce each fetch/unproject their own center depth in the same merged fragment
(`Gtao.ts:108-113` vs `PostStack.ts:216-221`, different uv node objects defeat CSE). Hoist one
shared center depth + view-pos. Quality: identical. Expected ~0.05 ms. Effort S.

### REJECTED-BY-POLICY (quality-trading; listed for completeness only)
- GTAO sample/radius/thickness cuts, or tightening `aoFadeNear/Far` (700/1800 never bind in forest
  anyway) — AO already self-describes as "a near-flat ~0.8 cue", but any change is A/B-visible.
- Contact-shadow march at half res + bilateral upsample (the AO trick) — visible on 1–2 px contact
  detail; would need a proven-invisible gate.
- Bounce gather at quarter res or fewer taps — visible chroma bleed change.
- Cloud temporal reprojection cache (world scene) — ghosting under drift/camera motion.
- Any internal-resolution change (dpr) — BANNED by user ruling.
- Exposure/grade simplifications — visible.

## What UE5/prior art does here

- UE5 Nanite resolve: decode-once via **shading bins** (material classify + per-bin dispatch);
  our measured decode ceiling (<1 ms) says binning would buy nothing here — matches the
  attribution verdict that killed shade-binning.
- UE5 GTAO/SSAO: half-res with spatial + **temporal** filtering (we run spatial-only, 6 samples,
  no temporal — adding temporal would be a quality *change*).
- UE5 contact shadows: full-res screen-space march like ours, but gated per-light and typically
  8 steps; consoles accept the near-field cache behavior as inherent.
- Clouds (Frostbite/Decima/UE): quarter-res + temporal reprojection + checkerboard — all
  quality-trading tricks under our constraint; note we already pay zero in forest (no clouds).
- TSR/DLSS-style upscale is the industry answer to "post scales with Mpx" — explicitly the
  sanctioned last resort per the user, not this area's call.

## Open questions + proposed serial probes

The singles were never measured; eye is the money pose for post (6.8 ms). Thermal order: baseline
first, candidates after (bias against candidates). All `TREES=200000 TICKS=0 COOLDOWN_S=45`.

1. `CONFIG=default LABEL=post2-base npx tsx tools/probe-fresh-stutter.ts`
2. `CONFIG=default EXTRA=ablate=contact LABEL=post2-contact npx tsx tools/probe-fresh-stutter.ts`
   — contact alone (the only single that isolates cleanly; my model says ~2–3 ms eye).
3. `CONFIG=default EXTRA=ablate=ao LABEL=post2-ao npx tsx tools/probe-fresh-stutter.ts`
   — NOTE semantics: `ablate=ao` drops AO **and** contact (`PostStack.ts:472-477`); AO alone =
   (#3 − #2 deltas).
4. `CONFIG=default EXTRA=ablate=bounce LABEL=post2-bounce npx tsx tools/probe-fresh-stutter.ts`
5. `CONFIG=default EXTRA=ablate=taa LABEL=post2-taa npx tsx tools/probe-fresh-stutter.ts`
   — with `ablate=taa` the composite chain still evaluates (moves from the TRAA input RTT to the
   final quad), so this isolates TRAA resolve + copies + input-RTT, not the chain.
6. `CONFIG=default EXTRA=ablate=bloom LABEL=post2-bloom npx tsx tools/probe-fresh-stutter.ts`
7. Haze has no ablate flag — add `ablate.has('haze') → hazed:=col` (one line at
   `PostStack.ts:283`) then `EXTRA=ablate=haze LABEL=post2-haze`.
8. L1 discriminator: add `?resclasses=` (auto|all) then
   `CONFIG=default EXTRA=resclasses=auto LABEL=post2-clsspec` + shotdiff vs #1 (expect 0 diff).
9. L2 discriminator (after L2 built): `EXTRA=respass=1 LABEL=post2-onepass` + `tools/vcdebug.mjs`
   check for silent pipeline death + shotdiff=0.

Open questions:
- Does W1 (dead terrain/rock branches) actually move occupancy on the tri pass? (Probe #8 decides;
  precedent says plausible, bark textures may already dominate.)
- Is contact or AO the bigger eye-pose chunk? (#2/#3 decide; informs whether any future
  *quality-improving* AO/contact spend has headroom.)
- One level up, for the user: does the locked-60 plan need headroom reserved for shadows/GI/clouds
  in forest-parity scenes (P2, Consequence C)? That decision changes what "gap closed" means.

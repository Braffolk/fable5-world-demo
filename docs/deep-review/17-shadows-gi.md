# shadows-gi deep review (2026-07-02)

Area: NaniteShadow.ts, NaniteShadowClip.ts, NaniteShadowHalf.ts, src/render/ShadowSetup.ts,
src/render/CsmCached.ts, src/gpu/passes/ProbeGI.ts — sun shadows (nanite clipmap + legacy CSM)
and the irradiance-probe GI, against the oblique −11..13 ms mission.

## Premise audit

**The fact pack's framing of this area is wrong one level up, and the wrongness IS the finding.**

1. **Shadows and GI are not "measured ~FREE" in the canonical scene — they are ABSENT.**
   The canonical config (`?scene=forest&nanite=1`, the scene every fresh-*.json measures) wires
   the frame with `gi: null, canopyTex: null, csm: null` — `src/debug/ForestScene.ts:457-459`,
   with the explicit comment at `ForestScene.ts:441` ("GI/CSM/canopy = null (no GI bounce, no
   shadows …)"). Downstream:
   - `src/nanite/NaniteFrame.ts:244` — `shadowOn = params.get('nanshadow') !== '0' && world.csm !== null`.
     With `csm === null` the ENTIRE nanite shadow system (clipmap raster, cascade path, shadow-half)
     is never built: `shadow = null` (NaniteFrame.ts:250-253), `shadowHalf = null` (NaniteFrame.ts:268).
   - `src/nanite/NaniteResolve.ts:237` — `shadowsOn = world.csm !== null && …` ⇒ the resolve
     compiles NO shadow sample at all (the PCSS/upsample block at NaniteResolve.ts:922-955 is
     behind `if (shadowsOn && world.naniteShadow)`).
   - `src/nanite/NaniteResolve.ts:973` — `if (world.gi)` ⇒ no probe irradiance is compiled either;
     ambient comes from the `ambFloor` max at NaniteResolve.ts:~1001.
   - `ProbeGI.tick` is wired in exactly one place: `src/debug/TerrainScene.ts:120`
     (`engine.onUpdate(() => gi.tick(...))`). The forest scene never constructs a ProbeGI.
     `new ProbeGI` exists only at TerrainScene.ts:113.

2. **The `nanshadow=0` attribution A/B was a no-op vs no-op.** `fresh-attr-noshadow.json` carries
   `extra = {…, nanshadow:'0'}` on `config=default` — but the forest scene had already forced the
   system off via `csm:null`. Its deltas vs the rested baseline (32.6/37.0/15.6 vs 31.2/38.8/14.6)
   are pure same-session noise (±1.5 ms), not evidence about shadow cost. The verdict "shadows ≈
   FREE (0-2ms)" should be re-labeled: **"shadow flag does nothing in this scene"**. Same for GI
   inside `nandbg=flat`: flat removed lighting math, but ProbeGI cost was never in the frame to
   begin with.

3. **Consequence for the mission target.** Every number in the fact pack (18.9/37.2/16.5, the
   pixel law, live p50 16.7) is for a frame with ZERO shadow raster, ZERO shadow sampling, ZERO
   GI dispatch, and ZERO CSM overhead. If the product is the WORLD scene (TerrainScene wires
   `gi`, `csm: shadowRig.csm` — TerrainScene.ts:409-411), locked-60 on the forest scene excludes
   an estimated **~2-6 ms (moving) of shadow work + ~0.3-1 ms GI/CSM overhead** that will come
   back. This is an upstream scene-setup decision the master plan must surface, not an
   implementation problem in this area.

4. **Metric check on the bimodality question:** the isolated per-pose arrays are 32 frames long —
   a 128-frame GI cycle cannot even complete once inside a sample window, so "cycle boundary every
   N frames matching runs-of-4" was never a testable hypothesis on these JSONs. It is refuted
   structurally instead (see Work model / §bimodality below).

## How it works today

### Nanite clipmap shadows (default producer when csm ≠ null): `NaniteShadowClip.ts`
- 6 concentric sun-aligned ortho levels, half-extents E_k = 12·2^k m = {12,24,48,96,192,384},
  each 1024² r32f (`readClipParams`, NaniteShadowClip.ts:128-141; defaults levels=6, base=12,
  res=1024, minPx=0).
- Per frame, `fitLevels` (NaniteShadowClip.ts:271-339) snaps each level's center to its own texel
  grid (texel_k = 2E_k/1024 = {0.0234, 0.0469, 0.0938, 0.1875, 0.375, 0.75} m) and re-rasters a
  level ONLY when the snapped VP differs bit-exactly from the cached one (`vp.equals(lv.lastVP)`,
  NaniteShadowClip.ts:315). Static camera ⇒ all levels cached ⇒ zero GPU work.
- On any re-raster: ONE shared hierarchical cut sized to the largest re-rastering level
  (`fitCut`, NaniteShadowClip.ts:343-378; `buildClipCull` at :233), then per level:
  `runLevelFilter → clearVis → depth1 → hwDepth → kCopy` (rasterLevels, NaniteShadowClip.ts:382-392),
  serialized over ONE shared vis buffer (NaniteShadowClip.ts:159-160).
- LOD is by the MAIN camera (`cam.camPos.value.copy(cp)`, :322) so caster LOD matches the lit
  surface. Sampling: finest covering level + PCSS (6 blocker + 9 PCF taps), NaniteShadowClip.ts:437-516.
- `cullPrepass` (item 4 overlap, NaniteShadowClip.ts:398-407) can fold the shared cut into the
  camera cull submit — opt-in via `?culloverlap=1`, default OFF (NaniteFrame.ts:260).

### Cascade fallback (`?shadowclip=0`): `NaniteShadow.ts`
4 fixed CSM cascades at 2048², re-rastered on VP change (R1 exact-equality gate,
NaniteShadow.ts:279); VPs come from three's CSM fit, frozen between refreshes by CsmCached.
Each cascade owns a full 2048² vis buffer set (NaniteShadow.ts:202). Non-default; kept as A/B.

### Half-res PCSS eval: `NaniteShadowHalf.ts`
When shadows exist, `kHalf` runs EVERY frame (NaniteFrame.ts:493) over ceil(W/2)×ceil(H/2)
threads — at canonical res 1134×737 = 835,758 threads — each doing 3 world-pos reconstructions
(payload load + inv-VP mul each, NaniteShadowHalf.ts:94-109), a level-select of ≤6 mat4 muls, and
6+9 PCSS texture taps (via `shadow.shadowFactor`, :134). The resolve then does a 4-tap bilateral
upsample per full-res pixel (NaniteShadowHalf.ts:149-190).

### Legacy CSM: `ShadowSetup.ts` + `CsmCached.ts`
World scene only. `CachedCsmShadowNode` re-fits/renders cascade i every PERIODS=[1,2,3,6] frames
(CsmCached.ts:44) with phase stagger + drift/sun-change forced refresh (CsmCached.ts:294-312).
With nanite active its maps are EMPTY (migrated casters suppressed; ShadowProxy only under
`?oldgeo`) — the resolve keeps the node alive by sampling one corner pixel
(NaniteResolve.ts:939-952 "black slate → keep == 1 → folds out"). So the world scene renders
~2 empty 2048² shadow maps per frame on average (Σ 1/PERIODS ≈ 2.0) purely to drive a fit whose
output the clip path doesn't even read (`run()`'s csm arg is ignored on the clip path,
NaniteShadowClip.ts:35).

### ProbeGI: `src/gpu/passes/ProbeGI.ts`
256×256×6 = 393,216 probes (ProbeGI.ts:55-60); `tick` dispatches gather+publish of EXACTLY
PROBES_PER_FRAME = 3072 probes (ProbeGI.ts:61, :329-339) ⇒ full refresh every 393216/3072 = 128
frames exactly (no partial batch at the wrap: `frameBase` advances by 3072 mod TOTAL,
ProbeGI.ts:333). Gather: 16 fixed-count fibonacci dirs × 16 fixed-count march steps
(ProbeGI.ts:62-63, :236-283) — the march loop has NO early-out (hit only latches `hitT`,
:247-253). Publish: 3072 textureStores ×3 (ProbeGI.ts:295-309).

## Work model

- **Canonical forest scene: this entire area contributes exactly 0 ms to every measured number.**
  Nothing here scales with anything; there is no shadow or GI dispatch in the frame. The
  attribution rows "shadows ≈ FREE" and "lighting/GI ≈ FREE" are wiring identities, not
  performance results.

- **World-scene shadow cost model (moving camera), from code structure:**
  Level k re-rasters when the light-plane camera projection crosses a texel_k boundary on either
  axis. Per-frame travel d = v/fps in light-XY; P(re-raster_k) ≈ 1−(1−min(1, d/texel_k))².
  - Walking 1.5 m/s @60: d=0.025 m ⇒ P ≈ {1, .78, .47, .25, .13, .066} ⇒ **E[levels/frame] ≈ 2.7**
  - Fly 10 m/s @60: d=0.167 m ⇒ P ≈ {1, 1, 1, .99, .69, .39} ⇒ **E[levels/frame] ≈ 5.1**
  - The texel grids are NESTED (0.75/0.0234 = 32, powers of two): crossing a level-5 boundary
    re-rasters ALL SIX levels in the same frame AND sizes the shared cut to the full 384 m disc
    (fitCut takes maxHalf over re-rastering levels, NaniteShadowClip.ts:349). At 10 m/s that
    aligned "storm frame" recurs every 0.75 m ≈ every 4-5 frames — a built-in periodic spike
    generator (world scene only).
  - Per re-rastered level: fixed ≈2.1M threads of clearVis+kCopy (2×1024²) + level filter +
    depth-only SW raster. Levels 0-2 (≤48 m) contain the MESH leaf-crown band (voxnear=45) at
    camera LOD — sun-view canopy is coverage-heavy; levels 3-5 raster trunks/bark only, because
    **voxel clusters never write shadow depth** (the shared raster skips class-7:
    `NaniteRaster.ts:541-549`; NaniteShadow/Clip never dispatch the voxel raster).
    Best-effort estimate: 0.3-1.5 ms/level ⇒ **~1-3 ms walking, ~2-6 ms fly-speed**, plus
    ~0.7-1.5 ms shadowHalf + ~0.3-0.7 ms bilateral upsample every frame regardless of motion.
    (Anchors: the deleted HW caster measured 14-31 ms/cascade; the 4-cascade CSM carried ~38× the
    camera's clusters, NaniteShadowClip.ts:15; the clipmap was built to shed exactly that.)

- **ProbeGI work model: constant by construction.** Every frame dispatches 3072 gather threads
  (16 dirs × (16 heightfield samples + 6 sun-vis samples + canopy/sky evals) ≈ ~400 texture ops
  per thread) + 3072 publish threads. No frame does more or less than any other: fixed loop
  bounds, no early-outs, exact 128-frame wrap, uniform-driven offset. Estimated ~0.1-0.5 ms
  (3072 threads is a tiny, latency-bound dispatch on M1 Max).

### Bimodality verdict (focus question b)
The GI-cycle theory is **doubly refuted**:
1. Wiring: ProbeGI never dispatches in the measured scene (gi: null, ForestScene.ts:457; tick
   only in TerrainScene.ts:120).
2. Structure: even where it runs, per-frame probe work is bit-for-bit constant (above) — a
   128-frame cycle with zero per-frame variance cannot produce runs-of-4 alternation.
3. Numerically: run-length + autocorrelation analysis of the 32-frame isolated arrays
   (fresh-final-rested oblique: runs 4,3,1,1,5,4,3,3,3,4,1; autocorr peak at lag 6-7 ≈ +0.4,
   trough at lag 3 ≈ −0.54) indicates a ~6-8 frame quasi-period. 128 is not a candidate; nor is
   any cadence in this area (the only frame-counters here are CsmCached PERIODS [1,2,3,6] — not
   built in this scene — and the GI 128-cycle — not built either). The bimodality hunt belongs to
   voxOccPyr/HZB/submit-batching, not shadows-gi.

### Shadow-storm verdict (focus question a)
In the canonical scene: **no storm exists — there is no shadow system**. The moving-vs-static live
delta measured in the milestones contains 0 ms of shadow re-raster by construction. The storm is
real in the world scene per the model above, with a periodic all-levels-aligned spike every 0.75 m
of travel; see levers.

## Waste inventory

All world-scene (canonical scene wastes nothing here because nothing runs):

1. **~2 empty 2048² CSM cascade renders/frame** (Σ1/PERIODS) whose maps are known-black; kept
   only so three's fit runs, and the clip path ignores that fit anyway (NaniteShadowClip.ts:35).
   Pure waste: est. 0.1-0.4 ms + 2 render passes of CPU/submit overhead per frame.
2. **Voxel clusters in shadow cull queues**: the shared cut walks the same DAG and emits class-7
   clusters that the depth raster then discards per-thread (NaniteRaster.ts:541-549) — wasted
   cull emit + queue traffic + raster launch occupancy on every shadow re-raster. (Flip side:
   far crowns cast no shadows at all — a quality GAP, see levers.)
3. **All-levels-aligned storm frames**: nested snap grids concentrate up to 6 level re-rasters +
   a 384 m-disc cut into single frames instead of spreading them (p95 spikes, not mean cost).
4. **shadowHalf full PCSS on sky pixels is already gated** (empty → lit, NaniteShadowHalf.ts:120)
   — checked, not waste. The 3-reconstruct normal estimate is 3× payload loads/pixel; a
   depth-derivative cache could halve it but it's small.
5. **ProbeGI steady-state churn**: with sun/ToD static, the field is converged after ≤128 frames;
   every subsequent gather re-estimates the same integral with a rotated jitter (EMA blend 0.22,
   ProbeGI.ts:289-291) — sub-noise output wobble for ~0.1-0.5 ms/frame, forever.

## Levers

The honest headline: **this area holds ZERO ms of the oblique −11..13 gap.** Every lever below is
world-scene-only (0.0 ms on all three canonical poses) or a measurement-context fix. Listed
because the master plan needs them when shadows/GI return.

1. **forest-scene-shadow-parity (measurement instrument, the most important item).**
   Mechanism: wire optional `setupSunShadows` + ProbeGI into ForestScene behind `?forestshadow=1`
   (pass csm/gi through to buildNaniteFrame instead of hard null). Makes the canonical scene able
   to measure the real shadow+GI bill at 200k trees and makes "locked 60" mean something for the
   world scene. Quality: identical (flag default-off). Effort M (scene wiring + boot order for
   canopy/ProbeGI init). Risk: forest boot time +~10 s (GI warm at ProbeGI.ts:314-325); none at
   default. Expected canonical delta: 0/0/0 (instrument, not optimization).
2. **empty-csm-cascade-skip.** Mechanism: in CachedCsmShadowNode, when the nanite shadow system
   owns all casters (no castShadow objects in the CSM render list), stop setting
   `shadow.needsUpdate = true` (CsmCached.ts:308) or clamp mapSize to a token 16² — the maps are
   already known-empty and `keep ≡ 1` folds out (NaniteResolve.ts:939-952). Quality: identical
   (output already black; keep≡1 preserved). Effort S. Expected: 0/0/0 canonical; ~0.1-0.4 ms +
   2 passes/frame world. Risk: `?oldgeo` A/B path must keep real maps — gate on caster count.
3. **shadow-cut-voxel-emit-skip.** Mechanism: in the shadow cull/filter (buildClipCull /
   buildNaniteCull shadow instances), drop class-7 (voxel) clusters at EMIT time instead of
   letting the depth raster discard them per-thread (NaniteRaster.ts:541). Quality: identical —
   those clusters already contribute zero shadow texels. Effort S. Expected: 0/0/0 canonical;
   ~0.1-0.3 ms per re-raster frame world. Risk: none (pure dead-work removal); keep bark/trunk
   descent intact (only leaf/crown vox heads are class-7).
4. **clip-storm-budget (stagger the nested-grid alignment).** Mechanism: cap re-rasters at K
   (e.g. 3) levels/frame, finest-first; a deferred coarse level keeps its frozen VP+map (the
   exact consistency contract the cache already uses — NaniteShadowClip.ts:113-116). Deferral
   error ≤ camera travel < texel_k/2 at any speed that didn't already force it — sub-texel.
   Quality: identical at walking/fly speeds (sub-texel latency; the drift is smaller than the
   level's own quantization). Effort M. Expected: 0/0/0 canonical; world p95 smoothing
   (~1-3 ms off storm frames), mean ~0. Risk: teleports must force-flush (existing `ran` gate
   covers first frame).
5. **culloverlap-default-on.** Mechanism: `?culloverlap=1` already folds the shadow shared cut
   into the camera cull submit (NaniteFrame.ts:453-460, NaniteShadowClip.ts:398-407) so Dawn can
   overlap two disjoint culls; it ships default-off. Measure in the world scene, flip default if
   ≥0. Quality: identical (ordering only; cut is camera-disjoint). Effort S (it's built).
   Expected: 0/0/0 canonical; ~0.2-0.5 ms on world re-raster frames. Risk: submit-order
   regressions on other platforms — keep the flag.
6. **gi-sleep-when-converged.** Mechanism: in ProbeGI.tick, count frames since the last
   sunDir/ToD/canopy change; after ≥128+boost frames (one full converged cycle), skip the
   gather+publish dispatches entirely until `invalidate()` or a sun-delta fires. Steady-state
   output differs only by the jitter-EMA wobble the sleep removes — sub-noise numerical change
   (constraint 3). Effort S. Expected: 0/0/0 canonical; ~0.1-0.5 ms world. Risk: anything that
   changes the integrand without calling invalidate() (canopy edits) — hook the same paths that
   rebuild canopyTex.
7. **vox-shadow-splat (QUALITY-IMPROVING, costs perf).** Far crowns (>45 m, and ALL far-tile
   heads) cast no shadows today — vox clusters are skipped in every shadow raster. Add a brick →
   atomicMin depth splat pass for shadow levels ≥2 (coarse bricks, 1 quad/brick). Direction
   matches constraint 2 (quality should improve; funds come from elsewhere). Effort L. Expected:
   NEGATIVE (adds ~0.5-2 ms world); canonical 0/0/0. Risk: peter-panning at brick scale; needs
   per-level brick-size bias.

### REJECTED-BY-POLICY (quality-trading; listed for completeness only)
- `shadowclipres` < 1024, `shadowcliplevels` < 6, `shadowclipbase` < 12 — coarser shadow texels,
  visible softening/crawl.
- `shadowtau` / `shadowminpx` increases (NaniteShadow.ts:189-199) — caster LOD/dropout visible in
  contact shadows.
- Freezing/slowing PROBES_PER_FRAME during ToD animation — visible GI lag on time-of-day sweeps
  (fine when static, that case is lever 6, which is identical-class).

## What UE5/prior art does here

- UE5 Nanite shadows = **Virtual Shadow Maps**: one 16k virtual clipmap per directional light,
  128² pages, cached per page with per-page invalidation on caster OR receiver change; only dirty
  pages re-raster (nanite geometry rastered directly into pages). Our per-LEVEL VP-equality cache
  is the same idea at page-granularity ∞ (whole level = one "page"); the storm-budget lever moves
  us one step toward per-page granularity without the page-table machinery.
- UE5 drops small casters from VSM and recovers them with **screen-space contact shadows**; our
  `shadowminpx` (default 0) + the post stack's contact term mirror this — Epic ships the trade
  we currently keep OFF for quality.
- UE5 caster LOD in shadow = the same DAG at a shadow-specific error target (like our
  camera-relative LOD with per-cascade τ, NaniteShadow.ts:182-199).
- DDGI (RTXGI) sleeps probes via hysteresis when converged and updates a budgeted subset/frame —
  lever 6 is the standard practice there; our fixed 3072/frame slice is already the "budgeted
  subset" half.
- The empty-CSM-keepalive pattern has no prior-art equivalent — engines drive the light-matrix
  fit without rendering the map; that's lever 2.

## Open questions + proposed serial probes

1. **Does the product ship the forest scene or the world scene?** If world: the master plan must
   budget ~2-6 ms (moving) shadows + ~0.5-1.5 ms half-res sampling + ~0.3-1 ms GI/CSM on top of
   today's 18.9/37.2/16.5, or fund it from the same quality-neutral pool. Needs a user ruling
   before "locked 60" is declared on forest numbers.
2. **True world-scene shadow bill today** (blocked on harness: probe-fresh-stutter hardcodes
   `scene:'forest'`, tools/probe-fresh-stutter.ts:72). After lever 1 lands, run serially:
   - `CONFIG=default LABEL=shadow-off TREES=200000 TICKS=600 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   - `CONFIG=default EXTRA=forestshadow=1 LABEL=shadow-on TREES=200000 TICKS=600 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   - `CONFIG=default EXTRA=forestshadow=1,shalfres=0 LABEL=shadow-fullres TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   (live moving p95 diff = the storm; isolated diff = the static sampling cost; shalfres isolates
   the eval-vs-raster split.)
3. **Bimodality ownership**: with shadows/GI eliminated (this doc), the ~6-8-frame alternation
   suspects are voxOccPyr/HZB rebuild interactions and Dawn submit batching. Proposed probe
   (serial, no code change): `CONFIG=default EXTRA=voxbocc=0 LABEL=bimodal-nobocc TICKS=0 FRAMES=128 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   with FRAMES≥128 per pose so period analysis has ≥4 cycles of any ≤32-frame period (32-frame
   arrays are too short — see Premise audit §4).
4. **Is `?oldgeo` still a supported A/B?** Lever 2's gating needs to know whether real CSM maps
   must survive anywhere but debug.

## Reconciliation & verification (2026-07-02, post-limit continuation)

The adversarial verify pass died mid-run; this section is the completed verification for the
shadows-gi area. Every load-bearing file:line below was re-read in code; every measurement number
was recomputed from the scratchpad JSONs.

### Verified (mechanism lens) — the central premise HOLDS in code

- `ForestScene.ts:456-459` — `buildNaniteFrame(…, { gi: null, canopyTex: null, csm: null })`,
  comment at :441 ("no GI bounce, no shadows — forest trees are castShadows:false"). CONFIRMED.
- `NaniteFrame.ts:244` `shadowOn = params.get('nanshadow') !== '0' && world.csm !== null` →
  false in forest; `shadow = null` (:250-254), `shadowHalf = null` (:266-268),
  `culloverlap` gate (:260), `shadow.run` gate (:489), `shadowHalf.run` every frame when built
  (:493). ALL CONFIRMED.
- `NaniteResolve.ts:237` `shadowsOn = world.csm !== null && …`; `receivedShadowPositionNode`
  behind `if (shadowsOn)` (:330-350); PCSS/upsample block behind
  `if (shadowsOn && world.naniteShadow)` (:922); `keep` fold (:938-952); `if (world.gi)` (:973);
  `ambFloor` (:1000-1001). ALL CONFIRMED — with csm null NONE of these compile.
- `new ProbeGI` exists ONLY at `TerrainScene.ts:113`; `gi.tick` ONLY at `TerrainScene.ts:120`
  (repo-wide grep). CONFIRMED: ProbeGI never dispatches in any canonical forest measurement.
- ProbeGI structure: `PROBES_PER_FRAME=3072` (:61), TOTAL=256·256·6=393,216 (:55-60), tick
  advances `frameBase` by 3072 mod TOTAL (:329-334) ⇒ exact 128-frame wrap; gather = 16 dirs ×
  16 march steps with NO early-out (`hitT` latch only, :247-253); EMA blend 0.22 (:95, :289-291);
  publish = 3×3072 textureStores (:295-309); warm loop (:312-325). ALL CONFIRMED.
- Clipmap: defaults levels=6/base=12/res=1024/minPx=0 (`readClipParams`, NaniteShadowClip.ts:128-141
  ⇒ E_k={12,24,48,96,192,384}, texel_k={0.0234…0.75} m); ONE shared vis buffer (:159-160); shared
  cut `buildClipCull` (:233); snap+VP-equality cache (`fitLevels` :271-339, gate :315); LOD by
  main camera (:322); `fitCut` maxHalf (:343-378, :348-351); `rasterLevels` = runLevelFilter →
  clearVis → depth1 → hwDepth → kCopy (:382-392); `cullPrepass` (:398-407); PCSS 6 blocker +
  9 PCF taps (BLOCKER_TAPS/PCF_TAPS at :84-85, sample :437-516). ALL CONFIRMED.
- Cascade fallback: SHADOW_CASCADES=4, SHADOW_MAP=2048 (NaniteShadow.ts:78-79); per-cascade vis
  buffers (:203); VP exact-equality gate (:279); shadowtau/shadowminpx (:189-199). CONFIRMED.
- ShadowHalf: ceil(W/2)×ceil(H/2) threads (1134×737=835,758 at canonical res); sky pixels gated
  (empty → lit, :119-121); COVERED pixels do 3 wp reconstructions (:118, :124-125 — sky does 1,
  not 3: minor precision fix to §How-it-works); PCSS via shadow.shadowFactor (:134); 4-tap
  bilateral upsample (:149-190). CONFIRMED (with the 3-vs-1 precision note).
- CsmCached: PERIODS=[1,2,3,6] (:44) ⇒ Σ1/P = 2.0 empty 2048² renders/frame avg; cadence +
  drift + `shadow.needsUpdate = true` (:294-312, :308). CONFIRMED.
- Voxel casters: class-7 skip is INSIDE `rasterKernel(mode)` unconditionally — applies to the
  depth-only shadow kernels too (NaniteRaster.ts:541-555; the `returnIf(mcVox.equal(uint(7)))`
  is at :554, not :549 — cite range corrected). Shadow paths never dispatch the voxel raster
  (rasterLevels above). CONFIRMED: far crowns + far-tile heads cast no shadows.
- Measurement claims: `fresh-attr-noshadow.json` extra `{nanshadow:'0'}` on config=default,
  medians 32.6/37.0/15.6 vs `fresh-final-rested.json` 31.2/38.8/14.6; both 32-frame isolated
  arrays; rested-oblique run-lengths 4,3,1,1,5,4,3,3,3,4,1; autocorr lag-3 −0.54, lag-6/7
  +0.41/+0.40. ALL RECOMPUTED AND CONFIRMED.
- `tools/probe-fresh-stutter.ts:72` hardcodes `scene:'forest'`. CONFIRMED.

### Corrections to THIS doc

1. **Storm mechanism corrected (consequence intact).** The claim "the texel grids are NESTED …
   crossing a level-5 boundary re-rasters ALL SIX levels in the same frame" is WRONG as stated:
   with `Math.round` snapping (:293-294), level k's snap flips at ODD multiples of texel_k/2,
   and an odd multiple of texel_k/2 is an EVEN multiple of texel_{k-1}/2 — flip boundaries of
   adjacent levels are exactly INTERLEAVED, never coincident. What actually happens: at speed,
   fine levels re-raster ~every frame anyway (P≈1 for k≤3 at 10 m/s), so any frame where level 5
   flips (~every 0.75 m of light-plane travel ≈ every 4-5 frames at 10 m/s) still pays ~5-6
   levels AND `fitCut` sizes the shared cut to the full 384 m disc (maxHalf, :348-351). The
   periodic spike generator is real; the "aligned grids" explanation is not. Lever 4 unaffected.
2. **Lever 4 (clip-storm-budget) quality class corrected: identical → RISK.** A deferred coarse
   level serves a one-snap-stale (VP-consistent) map — bounded by texel_k/2 (≤0.375 m at level
   5) for 1-2 frames during fast motion, but NOT bit-equal and not conservative-cull-equal.
   Gate: fly-speed moving shotdiff on far shadow edges + teleport force-flush test + storm-frame
   p95 histogram. (The same-magnitude staleness already exists between snaps, which is why the
   risk is small — but per the quality law it is a RISK class, not identical.)
3. **ShadowHalf precision**: "each doing 3 world-pos reconstructions" → covered pixels do 3,
   sky pixels do 1 (the initial reconstruct at :118 runs before the empty gate).
4. **Cite fixes**: NaniteRaster class-7 skip = :541-555 (returnIf at :554); NaniteShadow
   per-cascade vis buffer = :203.

### Contradictions with sibling docs — resolved

- **Doc 07 (resolve-lighting, fleet B) §1/§3: "the whole shadow system is 0–2ms (nanshadow=0)"**
  — RESOLVED AGAINST 07, per code: with `csm === null` the flag gates a system that was never
  built (NaniteFrame.ts:244); the 32.6/37.0/15.6 vs 31.2/38.8/14.6 delta is same-session noise.
  Correct statement: "the shadow flag does nothing in this scene; shadow cost in forest is
  structurally 0 because the system is absent." Doc 07's own W1 row already half-knew this
  ("~0 forest").
- **Doc 07 TL;DR: reskeep "expected ~0–1ms forest — flip default after A/B"** — RESOLVED AGAINST
  07: the `keep` sample compiles only under `shadowsOn` (NaniteResolve.ts:922-956, :330), which
  is FALSE in forest. Forest pool for W1+W2 is EXACTLY 0 — no A/B needed there. `reskeep=0` is a
  world-scene-only lever (−3.9 ms prior measurement was a general-vista/world context).
- **Doc 90 (premise audit) §7-P3: "ProbeGI 3072/frame, 128-frame cycle — the strongest surviving
  suspect" for oblique/aerial bimodality** — KILLED: ProbeGI is never constructed or ticked in
  the forest scene (TerrainScene.ts:113/:120 are the only sites); even where it runs, per-frame
  work is constant by construction (fixed dispatch sizes, fixed loop bounds, no early-outs); and
  the verified autocorrelation shows a ~6-7-frame quasi-period, not 128. P3's probe budget should
  go to the voxOccPyr/HZB + submit-batching suspects instead (doc 10/11 territory).
- **Doc 90 §5: "resolve lighting+shadows measured ~FREE (nandbg=flat, nanshadow=0 ≈ baseline)"**
  — PARTIALLY KILLED: the `nandbg=flat` half stands (real ablation of lighting math); the
  `nanshadow=0` half is a no-op-vs-no-op and is NOT evidence of shadow freeness.
- **Docs 10 and 16** independently reached the same gi:null/csm:null finding — AGREE with this
  doc; no conflict.

### Surviving lever table (all 0.0 ms on canonical poses — this area holds none of the oblique gap)

| # | Lever | Mechanism | ms eye/obl/aer (canonical) | ms world-scene est | Quality | Probe | Effort | Conf |
|---|---|---|---|---|---|---|---|---|
| 1 | forest-scene-shadow-parity | wire optional setupSunShadows+ProbeGI into ForestScene behind `?forestshadow=1` (csm/gi pass-through instead of hard null) | 0/0/0 (instrument) | measures the real +2-6 ms bill | IDENTICAL (default-off) | shadow-off vs `forestshadow=1` vs `+shalfres=0`, 600-tick live + isolated (§Open-questions 2) | M | High |
| 2 | empty-csm-cascade-skip | stop `needsUpdate=true` (CsmCached.ts:308) / clamp mapSize when nanite owns all casters — maps already black, keep≡1 folds out | 0/0/0 | ~0.1-0.4 + 2 passes/frame | IDENTICAL (gate on caster count for `?oldgeo`) | world-scene A/B after lever 1 | S | High |
| 3 | shadow-cut-voxel-emit-skip | drop class-7 at shadow-cull EMIT instead of per-thread raster discard (NaniteRaster.ts:554) | 0/0/0 | ~0.1-0.3 per re-raster frame | IDENTICAL (vox clusters already write zero shadow texels) | world-scene re-raster frame delta | S | High |
| 4 | clip-storm-budget | cap re-rasters at K levels/frame finest-first; deferred coarse level keeps frozen VP+map | 0/0/0 | p95 smoothing ~1-3 on storm frames, mean ~0 | RISK — gate: fly-speed shotdiff on far shadow edges + teleport flush | world-scene p95 histogram | M | Med |
| 5 | culloverlap-default-on | `?culloverlap=1` folds shadow shared cut into camera-cull submit (built; NaniteFrame.ts:453-461, disjoint buffers, no HZB dep) | 0/0/0 | ~0.2-0.5 on re-raster frames | IDENTICAL (ordering only) | world-scene A/B; keep flag | S | Med |
| 6 | gi-sleep-when-converged | skip gather/publish after ≥128+boost unchanged frames until invalidate()/sun-delta | 0/0/0 | ~0.1-0.5 | IDENTICAL (removes only sub-noise jitter wobble) — gate: invalidate() coverage audit on canopy/ToD edit paths | world-scene steady-state A/B | S | High |
| 7 | vox-shadow-splat | brick→atomicMin depth splat for shadow levels ≥2 so far crowns/far tiles cast shadows (they cast NONE today) | 0/0/0 | NEGATIVE (−0.5-2, spends budget) | IMPROVING — gate: peter-panning shotdiff + user sign-off on the spend | world-scene shot + cost | L | Med |
| 8 | reskeep-default-flip (owned by doc 07/16, reconciled here) | corner-only CSM keep sample; bit-identical for real pixels (keep≡1) | 0/0/0 (NOT compiled in forest) | ~0.3-1 where nanite-shadow + csm coexist | IDENTICAL | world-scene within-boot setKeepFull A/B | S | High |

### Killed claims (one-line reasons)

- "Shadows ≈ FREE (0-2 ms), measured via nanshadow=0" (doc 07 §1/§3, doc 90 §5, fact pack) —
  no-op vs no-op: csm===null means the flag gated an absent system; delta is session noise.
- "reskeep=0 worth ~0-1 ms in forest, flip after A/B" (doc 07 TL;DR) — the keep block never
  compiles in forest (shadowsOn=false); forest value is exactly 0; world-scene lever only.
- "ProbeGI 128-frame cycle is the strongest surviving bimodality suspect" (doc 90 P3) — ProbeGI
  never runs in forest; its per-frame work is constant; observed period is ~6-7 frames.
- "GI cycle boundary matching runs-of-4 was testable on the isolated JSONs" — 32-frame windows
  cannot contain one 128-frame cycle; structurally untestable (this doc §Premise-audit 4, upheld).
- "Nested texel grids align ⇒ level-5 crossing re-rasters ALL SIX levels" (this doc §Work-model)
  — round-snap flip boundaries interleave exactly; storm period 0.75 m survives via
  always-ticking fine levels + maxHalf cut sizing, not grid alignment.

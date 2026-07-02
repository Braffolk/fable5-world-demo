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

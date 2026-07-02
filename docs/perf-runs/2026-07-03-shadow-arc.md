# 2026-07-03 — SHADOW ARC (arc 2 of the post-beauty roadmap)

User directive: shadows need HEAVY work; "their entire impl is dogshit… they use some
cache that only makes perf okay when standing completely still; slight movement and fps
drops fast." Grounding: `docs/deep-review/17-shadows-gi.md` (twice-verified).
Constraints: QUALITY BAR ABSOLUTE (perf buys detail, no noise tricks); mid-field
(60-280 m voxel band) casting is an ARC GOAL, not an option; far terrain ("a massive
hill far away feels like it should cast") is a design question to answer WITH the
refactor; terrain-LOD shadow flicker is a verification gate.

## Baseline (measured this session — probe-fresh-stutter with NEW `SCENE=world` support)

World scene, 4k-tree default config, dpr1.5, canonical 2268×1473, MF_COOLDOWN=5.

| run | live moving (600 ticks, 8 m/s eye glide) | isolated eye/obl/aerial (med gpuWall) |
|---|---|---|
| wsh-on (shadows default) | avg 31.6 p50 33.3 p95 41.7 — 596/600 >16.7, 128/600 >33 | 21.9 / 17.6 / 15.3 |
| wsh-off (`nanshadow=0`) | avg 14.0 p50 16.5 p95 24.0 — 121/600 >16.7, 0 >33 | 22.5 / 15.9 / 14.0 |

**Decomposition: the moving shadow bill is ≈ +17.6 ms avg (+16.8 p50) — shadows DOUBLE
the moving frame time.** The static/isolated bill is only ~1-2 ms (obl +1.7, aerial
+1.3, eye −0.6 = session noise; that ~1-2 ms = shadowHalf PCSS + bilateral upsample +
empty-CSM keepalive + reskeep sample). So ~90% of the entire problem is the every-frame
6-level re-raster — the cz-snap bug above, then whatever the restored cadence still
costs (fine levels + periodic coarse flips).

**THE SMOKING GUN: `nanite.shRaster` popcount histogram = 598/600 frames re-raster
ALL SIX levels.** avg 5.98 levels/frame vs the deep-review's modeled 2.7 (walking) /
5.1 (fly). The per-level VP-equality cache is fully degenerate under ANY camera motion.

### Root cause (code-level, NaniteShadowClip.ts fitLevels)

The level centre snaps the light-plane coordinates to the level's texel grid:
`scx = round(cx/texel)*texel, scy = round(cy/texel)*texel` — but the third component,
**`cz = cp.dot(forward)` (camera depth along the sun axis), is used RAW/unsnapped**
(`center.addScaledVector(forward, cz)`). Any motion with a nonzero component along the
sun axis changes `cz` continuously → `eye`/`center` move → `matrixWorldInverse` differs
→ the snapped VP is never bit-equal → every level re-rasters every frame, AND `fitCut`
sizes the shared cut to the full 384 m disc every frame (maxHalf = the largest
re-rastering level = level 5, always). The cache only ever hits when motion is exactly
in the light-XY plane — in practice, standing still. This SUPERSEDES the deep-review's
per-frame model (which assumed only light-XY texel crossings mattered) and confirms the
user's report 1:1.

Consequence: the whole "storm" phenomenology (levels/frame, fitCut disc) is currently
the EVERY-FRAME behavior, not a periodic spike. The R1 cadence design was never
functioning while moving.

## Fix ladder (design summary — see §Design)

1. **cz-snap (S, immediate):** quantize `cz` to the level texel grid like scx/scy.
   Restores the designed cadence (fine levels tick, coarse levels cache). The generous
   depth half-span (dHalf = half/sinElev + 100) absorbs the ≤texel eye shift; sampling
   uses the stored `levelVP` so receiver/caster stay consistent.
2. **Toroidal incremental clipmap (L, the structural fix):** on a snap flip, only the
   newly-exposed texel strip needs rasterizing — the rest of the level is a pure
   translation of valid content. Store depth as ABSOLUTE distance along the sun axis
   (`dot(p, fwd) + C` in r32f) so stored texels are translation-invariant (no depth
   re-bias, no full-level re-raster ever for translation; sun-direction change = the
   only full invalidation, cadenced/staggered). Raster cost ∝ strip area (~1-5%/level
   /frame at walking speed) instead of full 1024² levels. This is the classic clipmap
   toroidal update; UE5 VSM = the same idea at page granularity.
3. **vox-shadow-splat (ARC GOAL):** class-7 (voxel) clusters currently skipped in every
   shadow raster (NaniteRaster rasterKernel returnIf mcVox==7) ⇒ the 60-280 m voxel
   band + all fartiles cast NOTHING. Add a depth-only brick scatter (atomicMin into the
   shared shadow vis buffer, exactly like NaniteVoxelRaster's kVoxScatter but
   depth-only + ortho, no NDC-explode pathology since ortho w≡1) consuming the voxel
   clusters the shadow cut already emits (they're in the queue and discarded today —
   lever 3's waste becomes lever 7's input).
4. **Far-field horizon term:** clipmap max reach is 384 m; a massive hill beyond can
   never cast onto anything regardless of what writes depth. Standard fix: heightfield
   horizon/sun-occlusion term (ProbeGI already sun-marches the heightfield + canopy
   slab — same machinery, baked to a map on sun change, 1 tap/pixel in the resolve,
   applied at ALL distances multiplicatively — also fixes "hill 500 m away should
   shade a receiver at 100 m", which the camera-centred clipmap structurally misses).
5. **Dead-work levers:** empty-CSM keepalive skip (2 empty 2048² renders/frame,
   CsmCached), `reskeep` default flip in world (full-screen ≈1 CSM sample/pixel),
   GI sleep-when-converged.

## Design (task order; each step measured before the next)

**P1 — cz-snap** (LANDED; kept — correct and free). Measured wsh-czsnap: E[levels]
5.98→5.46 only (at 8 m/s with 24-40 ms frames the per-frame travel flips even coarse
texels), so the cadence design cannot cache at speed — full-level re-raster is the
disease, P5 is the cure. ⚠️ The wsh-czsnap ABSOLUTE numbers (live 42.1, iso eye 34.0)
are machine-state contaminated (3rd back-to-back GPU-heavy run): identical work
(6-level frames) cost 31.6 in run 1 vs 44.0 in run 3, and STATIC isolated (+12 ms)
cannot be affected by this edit (static = cache hit both ways). Within-session mask
grouping is the honest instrument:

- per-level marginal (same session): L5 ≈ 2.6 ms, L4 ≈ 4-7 ms; frames re-rastering
  only L0-L3 still cost ~37 ms vs 8.3 ms at mask 0 ⇒ **the fine levels L0-L3 (which
  tick every frame; they hold the dense mesh-leaf band at main-cam LOD) carry the
  bulk, ~5-7 ms/level** at run-3 thermal state.
- ⇒ P5 REVISED: toroidal strips for ALL levels (not just coarse). Wind-sway freeze in
  persisted texels is quality-CONSISTENT with today: the VP cache already freezes
  sway in shadows whenever the camera is static — sway-in-shadow was never live
  except while moving (when nobody can track it).

**P2 — sever the legacy CSM entirely (levers 2+8 combined + cloud-gate move).**
The empty-CSM keepalive exists ONLY to drive a fit the clip path ignores, and the
resolve's full-screen `keep` sample exists ONLY to keep that node alive — but `keep`
is ALSO the cloud-shadow carrier (setupSunShadows wraps pcssFilter × clouds.shadowAt
into the CSM filterNode). Plan: pass `cloudShadow(wxz)` into buildNaniteFrame's world
arg, multiply it into `direct` in the resolve, then csm:null the world's
buildNaniteFrame call + skip CachedCsmShadowNode renders when the caster list is empty
(ShadowProxy is inside !DISABLE_OLD_GEOMETRY, so nanite mode has zero CSM casters —
verified TerrainScene.ts:163-169). Quality: cloud gate moves from a 2048² empty-map
node sample to a direct full-res eval — identical or better. Kills ~2 empty 2048²
renders/frame + the per-pixel keep sample + reskeep flag becomes moot.
⚠️ keep `?oldgeo` working: it needs real CSM (gate on caster count / oldgeo flag).

**P3 — vox-shadow-splat (ARC GOAL: mid-field 60-280 m casts).** The shadow shared cut
already emits class-7 voxel clusters into the level queues; the depth raster discards
them per-thread (NaniteRaster.ts:554). Add per-level after the tri depth raster: a
small fan-out (vox items → vox queue + indirect args, mirroring the camera path's
runVoxFanout) + a depth-only brick scatter kernel — per brick: world AABB → level VP
(ortho, w≡1, no NDC-explode class) → texel bbox → atomicMin f32-bits into vis.depthV
(same election the tri depth1 uses; kCopy then publishes both). Peter-panning control:
bias by the brick world half-extent along the sun (project the brick's NEAR face, not
centre). Levels ≥2 only (mesh band owns <48 m; bricks only exist ≥60 m). Fartile
bricks (class fartile heads) included ⇒ the >280 m canopy also darkens the ground at
grazing sun. Verification gates: oblique/aerial shot — crown shadows appear on ground
beyond 60 m; no peter-panning at eye level; cost measured (arc budget, negotiate range
down only if it truly can't carry it).

**P4 — far-field horizon term (the "massive hill" answer: YES it should cast).**
Camera-centred clipmap can NEVER cast from >384 m casters (nor hill→valley at any
distance beyond the ring). Fix: a sun-visibility map baked from the heightfield (+
canopy top = terrain + tree height where canopyTex covers) on sun change — horizon
ray-march like ProbeGI's 6-step sun test but into a world-space r8 map (e.g. 1024² over
4096 m = 4 m texels; far shadows are penumbra-soft anyway), sampled at 1 tap in the
resolve and multiplied into `direct` at ALL distances. Gives: distant mountains shade
valleys, forested hillsides' aggregate darkening, sunset long shadows — all effectively
free per frame. Blend: sharpen near (clipmap owns detail), the horizon term is the
low-frequency envelope.

**P5 — toroidal incremental clipmap (structural; decide scope after wsh-czsnap).**
Store per-level depth as absolute sun-axis distance (dot(p,fwd)+C, r32f) — translation-
invariant texels; on snap flip raster ONLY the exposed strips (outer L + hollow-reveal
strips), kCopy strip-scoped with toroidal addressing; PCSS taps wrap. L0-L1 stay
full-re-raster (small boxes, preserves animated wind sway in contact shadows); L2-L5
toroidal (the expensive rings; sway invisible there). Also likely FIXES the user's
terrain-LOD flicker symptom: LOD changes enter via thin strips gradually instead of
whole-level re-rasters. Verification gates: fly-move shotdiff on far shadow edges,
teleport force-flush, the flicker fly-over gate.

**P6 — GI sleep-when-converged (lever 6, S)** + `?voxalpha` deletion rides any cleanup.

## Harness

- `tools/probe-fresh-stutter.ts` now takes `SCENE=world` (poses ground-relative via
  heightAtCpu; live glide already ground-follows via groundProbe) and records
  `nanite.shRaster` per tick → levels-per-frame histogram in the summary.
- World boots cold in the probe (~105 s; probes stay cold by policy).

# NANITE WORKSTREAM — LOG (journal + ledger, branch `nanite-raster`)

> The dated journal. **Read rule: recent-first** — the newest PROGRESS LOG entries +
> the PERF LEDGER are what matter on rehydration; skim older. Durable design/decisions
> live in `NANITE-SPEC.md` (read fully); the live plan lives in `NANITE-ROADMAP.md`
> (read first). The `## NEXT ACTIONS` section near the bottom of this file is the OLD
> prose plan — **SUPERSEDED by NANITE-ROADMAP.md**, kept only for its detailed reasoning.
> Append new dated entries at the top of the PROGRESS LOG; never edit old ones.

## PROGRESS LOG (append-only, newest first)

- 2026-06-14 (az): **PERF-3 ANALYSIS — the depth rasterizer DECOMPOSED to its sub-stages (no guessing): it is
  per-TRIANGLE bound, and the #1 cost is the 3× `fetchWorldVert` vertex fetch+transform (54%).** (Opus 4.8
  1M.) Three converging measurements at the worst "long alley" view (cam −4.2,303.1,−1.4 yaw 67.5° T11,
  `?pure`, 2592×1676, 82k visClusters):
  (A) **RESOLUTION SCALING** (non-invasive): 16× fewer pixels (4.34M→0.27M) cut `nanRasterDepth` only **1.43×**
  (2.82→1.97 ms) ⇒ ~70% per-triangle (pixel-INDEPENDENT), ~30% per-pixel fill (~0.85 ms @4K).
  (B) **IN-KERNEL ABLATION** (`?rdbg`, BUILD-TIME gate in NaniteRaster — a `returnIf`-true early-out after a
  sink `atomicMin` that consumes the stage's live vars so the compiler can't sink the work past the branch;
  `rdbg=0` emits the kernel byte-UNCHANGED): rdbg=1 (stop after the 3 vertex fetch+transforms) = **1.51 ms**;
  rdbg=2 (stop after edge setup) = **1.64 ms**; rdbg=0 (full) = **2.82 ms** ⇒ **vertex fetch+transform 1.51 ms
  (54%)**, edge/near/backface/snap/bbox **0.13 ms (5%)**, per-pixel loop (coverage+depth interp+atomicMin)
  **1.18 ms (42%)**. Reconciles with (A): pixel-independent total = 1.51 + 0.13 + ~0.33 (tiny-tri loop entry)
  ≈ 1.97 ms = the res floor.
  (C) **atomicMin is NOT a dominator:** depth (atomicMin) 2.82 ≈ payload (plain equality-store, no atomic)
  2.95 — if contention ruled, depth would dwarf payload.
  CONCLUSION (the lever, proven not guessed): the per-triangle **VERTEX fetch+transform**, not the fill or the
  atomic. A 128-tri cluster has ~70 unique verts but the kernel calls `fetchWorldVert` 3×128 = 384 (~5.5×
  redundant), and ONE WORKGROUP already == ONE cluster (128 threads = 128 tris) ⇒ PERF-3 fix = per-cluster
  **VERTEX TRANSFORM CACHING in workgroup shared memory** (cooperatively transform the cluster's unique verts
  ONCE → `workgroupBarrier` → each tri reads 3 from shared mem). Canonical Nanite; also pays back the payload
  pass + the HW vertex stage + every shadow raster (all call `fetchWorldVert`). Est. ~1.0–1.2 ms off the SW
  depth pass (and proportional on payload/HW/shadow). `?rdbg` is build-time gated so production is pristine;
  tsc clean. NEXT: implement the cache.

- 2026-06-14 (ay): **PERF-1 — per-pass measurement made TRUSTWORTHY + the PURE nanite renderer ISOLATED
  (?pure) + the user's WORST view decomposed with COOL numbers. Two integrity defects and one methodology
  error found and fixed; the SW depth+payload raster confirmed the #1 pure-nanite cost.** (Opus 4.8 1M.)
  User directive: "track perf per line, or as close as possible — no guessing which subsystem is slow." The
  infra ALREADY had per-PASS GPU timing (GpuProfiler taps three's per-compute/per-render timestamp pairs →
  `stats.gpuPasses` as r.<label>/c.<label>, every frame; every nanite kernel `setName`'d) — but the first
  real full-res dump exposed:
  (1) **THE HARNESS WAS LYING.** The dead three-CSM keep-alive maps (`r.shadow.c*`, still rendered though the
  D-N29 clipmap owns shadows now) resolve GARBAGE NEGATIVE timestamps (≈ −69 ms) → `render` total = −97 ms →
  `shoot.ts --gpusample` (gate render+compute>0) dropped EVERY sample ("median 0.0 ms over 0 samples"). FIX:
  `GpuProfiler.collect` rejects non-finite/negative durations from the total (key stays visible at 0 + a
  `__garbage` count, never hidden) → median works (45.9 ms / 24 samples). A poisoned aggregate silently
  invalidates everything downstream — integrity must not depend on no pass ever misbehaving.
  (2) **ANONYMOUS COST.** A 6 ms full-res pass `r.rt#16` (= the post aerial/atmosphere composite) and a 0 ms
  `c.compute?` (the postmin exposure noop). Named the noop (`autoExposureNoop`); rt#16 is post — stripped by
  ?pure, attributable by ablation, so deferred.
  (3) **METHODOLOGY (user caught it).** I'd measured bm7 = FOREST INTERIOR (alt 4 m, max occlusion) = the
  CHEAPEST view in the set (`nanRasterDepth` 1.25 ms). Raster cost lives on WIDE deep sightlines. Built
  `probe-worstpos.ts`: boots the user's reported worst pos (cam −4.2, 303.1, −1.4 @ T=11; their in-browser
  43 fps / GPU compute 17.17 ms), SWEEPS yaw to find the heaviest orientation ("looking down the long
  alley"), dumps the full sorted ledger + a screenshot. Worst = yaw 67.5° — a bare-winter-tree hillside vista
  (thin branches = many tiny tris + heavy depth overdraw + foreground trunks = big-tri HW), **82k visClusters
  / 130k hwTris, reproducing the user's ~17 ms compute.**
  Built **?pure** (`main.ts expandPureAblation`): a MASTER that composes `nanite=1 + postmin=1 + nanshadow=0 +
  nandbg=flat` via ONE `history.replaceState` (every existing read-site just works; explicit flags override,
  e.g. `?pure=1&nandbg=albedo`). Strips ALL beauty, KEEPS geometry (`veg` deliberately NOT ablated — it gates
  nanite registration). Fixes the user's "?pure = zero terrain" — it was simply never wired. Renders
  flat-albedo terrain/rock/bark on clear sky.
  **FINDINGS (cool, trustworthy):** at the worst view PURE nanite = SW raster depth **2.82** + payload
  **2.95** = **5.77 ms** + HW pass **2.62** + flat resolve **2.10 ms**; **fps 35→95 just by stripping post.**
  The post chain doesn't only cost itself — it THERMALLY THROTTLES the GPU: the SAME 82k-cluster raster is
  `nanRasterDepth` **2.82 ms COOL (pure) vs 5.96 ms HOT (full beauty) ≈ 2.1×** inflation (post passes bloom/
  TRAA/rt#16/half.mrt also read ~2.4× their bm3 values purely from heat). So the SW depth+payload raster
  (5.77 ms cool) is the #1 pure-nanite cost = the PERF-3 lever, and un-throttling/cutting post pays the nanite
  side back too. tsc clean. Files: GpuProfiler.ts (integrity guard), main.ts (?pure), PostStack.ts (noop
  name), probe-worstpos.ts (NEW). **NEXT (PERF-3 precursor): IN-KERNEL decomposition of `nanRasterDepth`** —
  the truest "per-line" a GPU allows (toggle transform / edge-setup / scanline-fill / atomicMin blocks,
  difference timed runs) to find WHICH part of the 5.77 ms to cut, plus a COOLED (idle-between) batch to lock
  absolute numbers vs the thermal envelope.

- 2026-06-14 (ax): **N5 shadow S3 — the SCREEN-DENSITY SHADOW CLIPMAP (D-N29 point 1) is BUILT,
  correct + beautiful + drops CSM; MEASURED at perf PARITY with the 4 cascades (the cluster win is
  offset by per-level fixed overhead) — the honest moving-fps WIN needs S4-on-the-headroom next.**
  (Opus 4.8 1M.) NaniteShadowClip.ts: replaces the 4 fixed CSM cascades with ONE camera-centred
  clipmap — L concentric ORTHO levels along the sun, doubling half-extent (E_k=E_0·2^k), each
  texel-SNAPPED to its own grid (anti-crawl + free cadence: a coarse level's huge texel rarely moves
  ⇒ the R1 exact-VP gate caches it). Own ortho VPs from sunU.dir.value + the camera (three's CSM
  DROPPED for shadow geometry; world.csm survives only as the resolve's cloud-gate carrier).
  `OrthographicCamera.coordinateSystem = WebGPUCoordinateSystem` (a standalone three cam defaults to
  WebGL z∈[-1,1] — the raster/sample want z∈[0,1]; the #1 correctness trap). HOLLOW (NaniteCull new
  opt `innerReject`=1/E): a level rasters only its RING — clusters whose light-clip bbox lies entirely
  in [±0.5] (the finer box, extents double) are dropped → each caster rasters into EXACTLY ONE level.
  Gap-free because shadows project along the sun ⇒ caster + its shadow share light-XY (the finer level
  covering the caster covers everything it shadows); the radius margin keeps straddlers in both. ONE
  shared vis (raster→copy→reuse) ⇒ less memory than 4 cascades. Same NaniteShadow interface ⇒ resolve
  unchanged; ?shadowclip=0 A/Bs back to the cascades. Knobs: ?shadowcliplevels (6) / shadowclipbase
  (12 m ⇒ 384 m coverage, > the 300 m floor) / shadowclipres (1024) / shadowclipminpx (0). MEASURED
  (probe-shadowclip.ts, NEW; bm7, 2592×1676, moving @0.12/frame): clipmap rasters **×0.32 the clusters**
  (250k vs 786k — the hollow + screen-density + minPx) and is **VISUALLY ≥ the cascades** (smoother
  vista, less foreground banding, soft attached near contact; T1024 ≈ T2048 at distance, clean near).
  BUT fps is PARITY: clipmap 49.2 moving vs cascades 49.9 (static 72 vs 73). WHY (the key finding): the
  cluster win does NOT convert to fps because the cost is per-level FIXED overhead (clear+copy+162k
  instance-cull × MORE maps) + fill, not the cluster raster — the cascades' [1,2,3,6] cadence re-rasters
  only ~1.8 maps/frame, while the clipmap's fine levels (small texels) re-raster ~5/frame. Uniform
  screen-density inherently uses more maps than the cascades' coarse-far. Sweeps: fewer levels (7→5:
  45→48 fps) + lower T (2048→768: 49→53, fill scales T²) each help ~marginally; the floor is the
  T-independent per-level cull/clear/copy. GROUNDING: the NO-shadow ceiling here is **83.5 fps** (not
  100+ — the user's 115-120 was a lighter scene), so the realistic target is "approach 83.5", i.e. make
  shadows ~free. NEXT (the real perf win, recorded not yet built): (1) SHARE the instance cull — the
  levels are concentric + all LOD off the SAME camera, so ONE inst-cull vs the coarsest box feeds all
  levels' cluster culls (kills the ×N 162k-cull + LOD-walk); (2) S4 DAG caster-coarsening on the ×0.32
  cluster HEADROOM (the clipmap's far levels are coarse ⇒ minPx + DAG bite hard where the fine cascades
  made them useless — the lever S4/S2-OCCL lacked on the cascade base); (3) variable-T (sharp near /
  cheap far). Default ON (honours "drop CSM", beauty ≥, perf parity = no regression); coverage drops
  from the cascades' 3200 m to 384 m (a recorded tradeoff, > the 300 m floor; a cheap cached far-backstop
  level closes it). tsc clean. Tested bm3 + bm7 static/moving; broader bookmark + walk-mode + low-sun
  sweep still pending.

- 2026-06-14 (aw): **N5 shadow S2-OCCL — per-cascade light-HZB occlusion cull WIRED but MEASURED WEAK
  (default off); the moving-shadow lever is NOT occlusion either.** (Opus 4.8 1M.) Built the headline
  candidate from log av: a per-cascade light HZB (buildNaniteHzb on each cascade's vis depth) + a NEW ORTHO
  sphereOccluded (NaniteHzb.makeOrthoOccluded — nearest-to-sun point vs the footprint max; no perspective
  foreshortening; span/mapRes level-pick; w==1) + a two-phase cull per cascade (clearVis→phase1→depth1→
  hzb.build→phase2→depth2→copy; prevVp = the cascade's last-raster VP). Behind ?shadowoccl=1. WIRING PROVEN
  — a temp always-true test culled 99% (869k→6k). But the real nearestZ>maxZ math culls ~0 under the moving
  probe (probe-shadowoccl.ts, new). THREE root causes, measured/reasoned: (1) phase2 tests vs the FRESH HZB
  that ALREADY holds the cluster's own phase-1 depth → self-occlusion → never culls; the cull must come from
  phase1 vs the PREV-frame HZB. (2) but CsmCached/R1 freezes cascades on the [1,2,3,6] cadence → on cached
  frames the prev HZB is STALE → phase1 occlusion is weak (only c0, cadence 1, has a 1-frame-fresh prev HZB).
  (3) a high/medium SUN has inherently weak inter-caster occlusion — it sees the canopy TOPS, so few casters
  sit fully behind a closer one (a diagnostic "HZB has ANY near surface" test culled only 15%; the proper
  conservative subset ~0). The camera's 65× comes from looking INTO the forest (trunks behind trunks), which
  the sun does not. CONCLUSION: the 38× shadow/camera cluster disparity (log av) is dominated by WIDE ORTHO
  COVERAGE × 4 CASCADES × fine geometry, NOT by occludable geometry — so neither caster-LOD (S4) NOR occlusion
  (this) is the big moving-raster lever. Kept ?shadowoccl OFF (the two-phase overhead isn't worth ~0 cull) as
  documented WIP — the ortho HZB test is correct, reusable infra once the prev-HZB-staleness is solved
  (decouple the HZB refresh from the CsmCached raster cache). tsc clean; default behaviour unchanged.
  STRATEGIC FORK (flagged for user direction): the moving-fps win needs a more radical lever — collapse the 4
  cascades into ONE screen-density shadow CLIPMAP (S3) and/or far-caster PROXIES/impostors — a bigger change
  with a real quality/perf tradeoff (far-shadow fidelity), so worth a check-in before building.
- 2026-06-14 (av): **N5 shadow S4 (DAG-decoupled caster LOD) — implemented + MEASURED MINOR for this world;
  the real lever is OCCLUSION (redirect).** (Opus 4.8 1M.) The DAG cut predicate in kClusterCull is
  τ-driven (NaniteCull:403), and the shadow culls passed NO τ (default 1 px) — NaniteCull's own comment said
  "proper DAG-decoupled caster LOD is S4." Worse, projK_shadow = uH·½ = 1024 (cotHalfFov 1, 2048 map) vs the
  perspective cam ~468, so shadows rendered ~2× FINER than the camera — pure waste. Wired a coarser
  per-cascade τ (`?shadowtau`, default 4, ×CASCADE_MUL [1,1.7,2.7,4]) into each cascade cull. RESULT: shTotal
  (shadow visible-cluster count, the moving-raster proxy) moved only **807,644 → 805,995 (0.3%)** across
  τ=1→8. ROOT CAUSE (the valuable finding): the dominant shadow casters here are NOT DAG — terrain is a
  discrete CLIPMAP and vegetation is explicit discrete-LOD; only rock/bark are true DAG (**dagClusters ≈ 6%**
  of the view). So the DAG τ cut only trims that minority. Added the cross-class min-screen-size cull as a
  complement (`?shadowminpx`, NaniteCull minPx "applies to all classes", ×CASCADE_MUL): minPx=4 → shTotal
  **806k → 711k (−12%, far cascade c3 −26%)**, beauty-safe at bm7 but it TRADES a little shadow completeness
  (drops small casters — Epic ships this for foliage). **THE BIG FINDING:** the shadow cascades carry **805k
  clusters vs the camera's 21k (38×)** — because they have **NO occlusion culling** (sphereOccluded=null) +
  wide ortho coverage + 4 cascades, where the camera's HZB occlusion cut its clusters ~65× (1.36M→21k). So
  the headline shadow-raster lever is OCCLUSION-CULLING the casters (ZERO quality loss — only drops casters
  hidden from the light), NOT LOD coarsening. That needs a per-cascade light HZB (buildNaniteHzb works on any
  cam's vis buffer) + an ORTHO sphereOccluded variant (the existing one is perspective-only: finite camPos,
  cotHalfFov level-pick, w-divide) + per-cascade two-phase cull. Defaults kept conservative (τ=4 removes the
  over-detail ~free; minPx=0 = no quality loss; both tuning flags). tsc clean; bm7 A/B beauty-neutral. NEXT:
  the shadow OCCLUSION cull (per-cascade ortho HZB) — the actual 2–4× moving win.
- 2026-06-14 (au): **N5 shadow S0 (D-N29) — HALF-RES PCSS eval + depth-aware BILATERAL upsample.** (Opus 4.8
  1M.) First chunk of the shadow perf engine, now that the N8 DAG default unblocks the stack. The fixed
  per-pixel PCSS sample in shadowFactor (6-tap blocker search + 9-tap penumbra PCF ≈ 15 cascade-texture taps)
  is the STATIC shadow cost — paid every frame whether moving or not (R1 caches the per-cascade RASTER, not
  the sample). NEW NaniteShadowHalf.ts: a compute pass evaluates shadowFactor at HALF res into an rg32f
  StorageTexture (R = factor, G = camDist), reconstructing wp EXACTLY like the resolve (cam.invVp · ndc,
  bottom-up fy — the verbatim nanProbe expression; NO new uniform) and using a cheap depth-derivative
  geometric normal oriented toward the camera (shadowFactor uses the normal ONLY for the receiver bias
  offset). The resolve replaces the per-pixel shadowFactor with a depth-aware BILATERAL upsample (bilinear
  footprint over the 4 nearest half-texels, each spatial weight × a camera-distance gaussian) — keeps the
  contact band sharp at depth discontinuities where a naive bilinear would bleed shadow across silhouettes
  (trunk vs background terrain). shadowFactor/pcss gained an optional `pix` param (the IGN sample-rotation
  coord): screenCoordinate/fragCoord is UNDEFINED in a compute stage, so the half pass passes its own pixel
  coord (the fragCoord-in-compute bug — caught at first boot, fixed). Binding-friendly: the resolve no longer
  calls shadowFactor → drops 4 cascade textures, gains 1 half tex (net −3 sampled textures); storage-BUFFER
  count unchanged (still under the Metal 10/stage cap). ?shalfres=0 restores the full-res per-pixel path for
  A/B. MEASURED (probe-shadowcost.ts, bm7 static, median, bypasses shoot.ts's render+compute>0 gate which
  the CSM keep-alive's garbage NEGATIVE r.shadow.c0 timestamp poisons): the resolve-side SAMPLE roughly
  HALVES — full-res r.scene 1.18 ms → half-res r.scene+halfpass 0.98 ms (Δ 0.197 = 17% of resolve) @720p;
  2.36 → 1.90 ms (Δ 0.459 = 19%) @1080p. ~Linear in pixel count (bilateral + 3×-reconstruct half-pass
  overhead eats the theoretical 4× to ~2×), so the win GROWS at the user's higher res (~1.8 ms @4K-retina)
  and helps MOVING too (the sample is every-frame). BEAUTY-NEUTRAL: shalfres on/off A/B at bm7
  indistinguishable (same shadow placement, penumbra softness, contact band). Validated: tsc clean; boots
  clean (no shader error post-fix); ?nandbg=shadow full-res ground-truth unchanged. Files: NaniteShadowHalf.ts
  (new), NaniteShadow.ts (pix param), NaniteResolve.ts (upsample wire), NaniteFrame.ts (build + dispatch
  before post.render), probe-shadowcost.ts (new). NEXT: S1 (shadow-pass WPO-freeze + static/dynamic split) —
  begins on the MOVING raster cost (the user's 30-fps pain).
- 2026-06-14 (at): **Clipmap cross-level OVERLAP fix — root-cause of the user's heavy z-fighting + sharp
  ground shelves (USER CHECKPOINT feedback on the 2e default DAG terrain).** (Opus 4.8 1M.) User returned,
  verified 2e (no gray wash, robust), and flagged 3 issues: (3) shadows popping in/out on cluster change —
  confirmed CSM via `?nanshadow=0` (gone), DEFERRED to S4. (1) heavy z-fighting "in fan patterns," worst
  from altitude; (2) sharp flat shelves/cuts in the ground. BOTH = one root cause: the clipmap had
  near-TOTAL cross-level overlap. `clipmapTiles` snapped each level to ITS OWN tile grid (`snapDown(·,Tk)`),
  so consecutive levels misaligned; the finer extent is only M/2 = 2 coarse-tiles wide (M=4), so when
  misaligned it fully contained NO coarse tile ⇒ the hollow dropped nothing ⇒ the finer level OVERLAID 2–3
  whole coarse tiles. Both LODs rasterized the same ground at different heights ⇒ z-fighting (1) + a coarse
  flat tile poking up among the fine detail (2's shelves). New gate probe-clipoverlap.ts swept 12321 camera
  positions: **12272 (99.6%) had cross-level overlap, worst a full 1024² tile.** (The 2d "thin ring hidden
  by skirts" note was wrong — skirts seal cracks, not overlap, and it was whole tiles not a thin ring.)
  FIX: snap each level to 2×Tk ROUNDED ⇒ the finer extent lands on coarse tile boundaries ⇒ every coarse
  tile is fully-inside (dropped) or fully-outside (kept), never a straddle ⇒ **0/12321 overlap.** Bonus:
  removes the ~2× terrain double-draw at the (vast majority) misaligned positions — the 2e perf-ledger's
  hwTris ×1.77 was overlap-inflated. TRADEOFF: the finest level now re-centers every 2×Tk (256 m, was
  128 m) — coarser snap, masked by the continuous LOD; watch for finest-detail re-center jumps when walking
  (mitigation if needed: larger M or smaller gridN). Validated: tsc; probe-clipoverlap PASS (0 overlap);
  probe-stream + probe-streammove (coverage/residency unaffected — no holes, no leak, bounded). User to
  re-verify (1)+(2) live (static probes can't show the z-fight flicker; the node gate is the proof).
- 2026-06-14 (as): **N8-D2 Stage 2e (part 2/2) — STRIDE-1 terrain vertex buffer (−174 MB) + the
  caustic-wash root-cause it exposed.** (Opus 4.8 1M.) Terrain-DAG verts used only word0 (a packed texel
  coord) of the shared 6-word `verts` record — 5/6 wasted. Moved them to a DEDICATED stride-1 buffer
  (`gpu.hfVerts`, 1 u32/vert): new LateBudget.hfVerts + hfVertsArr/hfVertsAttr/hfVertCursor/hfCap, the
  pool region claimed in hf-space (tilePoolBase.vert), both attach paths (registerHeightDag +
  attachHeightDagTile) write stride-1 + rebase indices into hf-space, NaniteFetch's isHF&&isDAG branch
  reads `hfVerts[vi]` (was `verts[vi·6]`), reserveTilePool/WorldRegistry reserve hfVerts. Explicit meshes
  keep the 6-word `verts`; the window grid uses neither. **MEASURED: terrain verts 208.8 MB → 34.8 MB
  (−174 MB, exactly the 5/6).** probe-tilepool/stream/streammove green (cursors show the shared vertCursor
  bottoms out at ~0 — terrain left it); a GPU READBACK (probe-hfdiag, getArrayBufferAsync) confirms the
  pool's hf bytes are byte-identical CPU↔GPU.
  **THE BUG (cost the session): the clip DEFAULT boot washed flat gray — terrain+trees gone.** Bisected
  it: geometry is CORRECT (probe-seams renders the detailed valley; readback byte-exact; probe-dterrain
  non-pool clean) ⇒ NOT the verts. View-independent gray, present only with caustics ON (ablate=caustics
  → perfect) ⇒ the caustic term in NaniteResolve. ROOT CAUSE: the shared `makeFetch` references
  `gpu.hfVerts`, so wiring it into the RESOLVE fragment stage pushed its storage-buffer count past the
  device ceiling (`maxStorageBuffersPerShaderStage = 10` on this Metal adapter) → the caustic `waterY`
  storage-buffer read (Heightfield.sampleWaterY) silently mis-bound → `causticDepth = waterY − wp.y` blew
  up → `terrainCol ·= (caust·1.7+1)` washed the whole frame. FIX: `makeFetch(gpu,…,bindHfVerts=true)`;
  the RESOLVE passes `false` — it reconstructs terrain world-pos from the DEPTH buffer and only fetches
  rock/bark (the explicit-mesh else branch), so it never needs the terrain buffer ⇒ one fewer storage
  buffer ⇒ caustics correct. The raster keeps `bindHfVerts=true`. LESSON: a new registry storage buffer
  is +1 binding in EVERY pass that calls makeFetch (raster+shadow+hzb+resolve); the resolve sits at the
  10-buffer Metal ceiling, so only bind what a pass actually reads. Validated: tsc clean; probe-flip +
  probe-groundview (default boot renders, caustics ON); probe-dterrain (rock/bark/terrain resolve intact);
  probe-seams (clip vista); probe-tilepool/stream/streammove (pool + residency, no hole/leak). 2e COMPLETE.
- 2026-06-14 (ar): **N8-D2 Stage 2e (part 1/2) — THE FLIP: full-res clip-streamed DAG terrain is the
  DEFAULT, no window fallback (the "boot only to dag, no fucking fallback EVER" mandate).** (Opus 4.8 1M.)
  Holes (ao) + slow-pop (ap) + seams (aq) were the three visible-correctness blockers; all sealed ⇒ the
  flip is now pure plumbing. Bare `?nanite=1` (no terrain flags) now defaults to `dagTerrainGridN=128 +
  dagTerrainClip=true` ⇒ the camera-following 1 m clipmap streamer renders the terrain from frame 1. The
  legacy implicit window grid is retired to the explicit opt-out `?nanitedterrain=0` (tooling / A-B only);
  an explicit `?nanitedterrain=<gridN>` still selects that grid and stays one-shot uniform unless
  `?nanitedclip=1` (per-flag tool semantics preserved — probe-dterrain/probe-dagcache unaffected). ONE
  edit (TerrainScene param defaulting); WorldRegistry already routes `gridN>0 && clip` → streamer, so no
  engine change. MEASURED (new gate tools/probe-flip.ts, walk spawn): default boots 52 resident tiles /
  34230 cl / 4.14 M tris (CLIPMAP 5L M4, gridN 128, skirts on, 2.24 s build, 0 cached cold), renders
  terrain at the walk spawn (not bare sky); `?nanitedterrain=0` → window grid, no streamer. POOL
  96 slots × (v95016 / t140364 / c1204) ⇒ **terrain verts 208.8 MB** (each vert wastes 5/6 words — only
  word0 used) + indices 154.2 MB. That 208.8 MB is exactly what part 2/2 (stride-1 vertex buffer) cuts to
  ~34.8 MB (−174 MB). tsc clean; probe-flip PASS. NEXT → 2e part 2: stride-1 terrain vertex buffer.
- 2026-06-14 (aq): **N8-D2 Stage 2d — crack-free inter-level SKIRTS (the last visible-correctness
  blocker before default-on).** (Opus 4.8 1M.) Clipmap levels ABUT at 2× stride (the hollow drops a
  coarse tile only when FULLY inside the finer extent), so a fine edge has 2× the coarse edge's verts
  on the same line ⇒ T-junction cracks that show SKY. Sealed with a SELF-CONTAINED always-on perimeter
  skirt per tile — no buildDag change (it already pins tile borders at base stride at every LOD, so the
  surface edge passes through every perimeter vert at every cut; the residual crack is purely the
  inter-level mismatch). Skirt = own TOP verts on the perimeter (render on the surface edge) + own
  BOTTOM verts dropped below it; emitted as ALWAYS-ON clusters (level 0, ownError 0, parentError +∞ ⇒
  drawn at every cut, frustum/occlusion-culled normally; cone-cull is off for heightfields anyway —
  NaniteCull:411 `isHF.not()`). The drop is encoded as a 3-bit depth-level CODE in bits 13-15 of word0
  (free: a tile texel coord ≤4095 uses bits 0-11), decoded at the ONE GPU fetch site (NaniteFetch
  `fetchWorldVert`, shared by raster+shadow+hzb+resolve) → `h.sub(skirtDrop)`. DOUBLE-SIDED (4 tris/
  segment): the SW raster backface-culls by signed area (NaniteRaster:241) and the camera grazes a
  boundary from either side, so the area test renders whichever face points at the camera (≈0 raster
  cost — the back pair always culls). **DEPTH CALIBRATION (the real work):** a geometric proof
  (tools/probe-skirtgap.ts — samples the REAL field via __laas.groundProbe, 256 scanlines/axis, GPU/
  tree/jitter-free) measured the actual cracks at 18→22→25→48 m (1500 m-relief cliffs, NOT the few-m I
  first assumed) and that they SATURATE with stride — so depth ∝ 2^k is the wrong model (starves the
  fine transitions, wildly over-covers the coarse → skirt walls). Switched to LINEAR
  `depth = 24 + 12·level` m → the proof PASSES every transition with worst margin +6.1 m. MEASURED
  (probe-seams, gridN 128, ablate=grass/water/shell/particles/caustics — NOT veg, which gates the
  whole nanite path): boot +832 cl / +106 k tris (= 52 tiles × 16 skirt cl / 2048 tris, exact), pool
  cap AUTO-GREW c1180→1204 (the boot measure absorbs the skirt add — no manual bump), +7 MB verts, boot
  2203→2219 ms (negligible). Skirt-OFF vs ON A/B (`?nanitedskirt=0/1`, default ON): the off↔on diff is
  pure tree-jitter noise (two boots, different jitter phase) — NO concentrated gridded walls ⇒ the
  deeper fine skirts hang hidden, no wall artifact. Headless build check: skirt verts/tris/clusters
  exact (8·gridN/16·gridN/⌈16·gridN/128⌉), all cluster invariants clean, indices in range, OFF path
  BYTE-IDENTICAL (zero code bits). tsc clean; probe-stream + probe-streammove (217 loaded / 0 skipped,
  no transient hole) + probe-seams + probe-skirtgap all green. NEXT → 2e: flip full-res DAG terrain
  default ON + stride-1 vertex buffer (the "boot only to dag" mandate).
- 2026-06-14 (ap): **N8-D2 #32 — DagWorker POOL: concurrent tile bakes (the "baking is slow" the
  user named).** (Opus 4.8 1M.) Builds were serialized on ONE persistent DagWorker (~170 ms/tile
  cache-miss) → a camera move needing K fresh tiles stalled ~K×170 ms of coarse→fine pop (no hole —
  lazy-evict (ao) keeps the old LOD up — just a slow climb to detail). Fix is two-part, both needed:
  (1) NEW `DagWorkerPool` (DagWorkerClient.ts) — N persistent DagBuildWorkers behind the same
  `DagBuilder` interface (buildHeight/dispose), LEAST-LOADED dispatch (a fresh bake goes to the
  most-idle thread, not a round-robin victim stuck on a slow tile); size = clamp(cores−2, 2..4);
  throws if no Worker (headless node) → caller falls back to sync. (2) runDiff now BATCHES arrivals
  (≤ MAX_LOADS_PER_DIFF=4, matched to pool size) and bakes them CONCURRENTLY (Promise.all) — the
  batch costs ~one bake, not N — then ATTACHES synchronously (NO await between allocTileSlot and
  attach/resident.set ⇒ the slot pool can't be raced; single-flight still bars re-entry).
  buildBootSet likewise bakes the whole boot ring at once (deterministic: collect in set order, not
  completion order). Build-before-alloc + cap-pre-check + lazy-evict invariants all preserved.
  MEASURED (probe-streammove, cold = 0 cached/52 built, gridN 128): cold BOOT 7717 ms serial
  (bake×1) → 2211 ms pool (bake×4) = **3.49×**; and in the SAME probe wall-time the pool streamed
  219 tiles loaded vs 80 serial, lifting the worst MID-BAKE floor 482 → 729 cl (more detail bakes
  within the settle(8) window, still hole-free). probe-stream (headless sync path, worker=null)
  still green — the parallelization is transparent to it. tsc clean. NEXT (toward full-res default
  ON): 2d skirts (crack-free inter-level seams) — required before flipping default — then 2e
  stride-1 verts + flip.
- 2026-06-14 (ao): **N8-D2 Stage 2b-3 fix — LAZY eviction kills the "LOD vanishes before the new
  one bakes in" hole (user report).** (Opus 4.8 1M.) Root cause: runDiff EVICTED departed tiles
  eagerly (synchronously, first) then started the SLOW async loads — so through the bake window the
  old detailed tile was already gone but its replacement wasn't in ⇒ a transient void (worse when
  the coarse backstop churned too). Fix: the old LOD now SURVIVES until replaced. runDiff no longer
  evicts up front; a departed tile keeps rendering until either (a) the whole desired set is resident
  again (cleanup drops the now-redundant stragglers — lean memory when the camera settles), or (b)
  the pool runs dry and a slot must be reclaimed → evict the departed tile FARTHEST from the camera
  by FOOTPRINT distance (footprintDist2 = 0 if the camera is inside it ⇒ a coarse tile still covering
  the camera is reclaimed LAST — it is the backstop; far small fine tiles go first). Pool headroom
  raised to 1.5×clipmapMaxTiles so stragglers can linger through the bake. probe-stream still green
  (resident≡desired after settle via cleanup, bounded, no-leak). probe-streammove upgraded to read
  dag MID-BAKE (one settle right after each hop, before convergence): worst mid-bake over moderate
  hops 654–768 cl (>0 = NO transient hole; was 0 with eager evict), and even a hard TELEPORT to the
  rim holds 281 cl mid-bake (the field-spanning coarse ring lingers + covers the destination). The
  mid-bake pattern is `654 → 1384` settled = coarse-shows-then-fine-pops-in (graceful LOD, never a
  void) — confirmed by a mid-bake SCREENSHOT (full relief fills the frame during the bake). This
  also largely SUBSUMES the 2b-4 teleport-hole concern from log (an): lazy+footprint-evict keeps the
  old coarse covering through most teleports; an always-resident base is now polish for the extreme
  corner→corner jump (beyond the old coarse ring's reach) + the eventual suppression-free endgame.
  KNOWN/next perf: builds are still SERIALIZED on one DagWorker (~170 ms/tile cache-miss) → detail
  pops in slowly on a big jump (coarse lingers meanwhile, no hole) — a DagWorker POOL (concurrent
  bakes) shortens the coarse→fine window. tsc clean; probe-tilepool + probe-clipmap green.
- 2026-06-14 (an): **N8-D2 Stage 2b-3 — the terrain clipmap FOLLOWS the camera (per-frame async
  streamer); detail re-centers on the live camera, hole-free at steady state (D-N39).** (Opus 4.8
  1M.) New TerrainStreamer.ts owns the residency loop: the field's buildTileGlobal moved into a
  shared `buildTerrainTile(deps,…)` helper (WorldRegistry's uniform path + the streamer both call
  it), plus a PERSISTENT DagWorker (clip mode no longer disposes it at boot). WorldRegistry clip
  path now: construct the streamer, `await buildBootSet(res/2,res/2)` PRE-build (spawn-centered ring
  set + measures caps), reserveTilePool sized for `clipmapMaxTiles+8` slots × boot-worst×1.5 caps,
  `attachBootSet()` POST-build (frame-1 terrain, no fallback), return it. TerrainScene drives
  `engine.onUpdate(()=>streamer.update(camX,camZ))`. update() → camTexel → clipmapTiles → diff vs
  resident Map(tileKey→slot): EVICT departed immediately (sync, cheap — bounds memory+vista every
  frame), LOAD arrived asynchronously (build→cap-check→alloc→attach, ≤4/frame; single-flight via a
  busy-guard + pendingCam coalescing). KEY SAFETY: build-BEFORE-alloc (an in-flight build never
  holds/leaks a slot), cap-pre-check BEFORE alloc (oversized region SKIPPED, coarser ring backstops,
  never throws inside attach), and a `skipped` set so an over-cap tile is built ONCE not every frame.
  Headless probe-stream.ts (res512/gridN64, 6-pose motion path + a skip-storm): resident≡desired at
  every pose, 76 evictions, finest-L0-at-camera resident, cursors BOUNDED (v921312/t1464000/c15168
  identical across the whole churn — the pool reuses slots in place, the memory bound), NO leak
  (free+resident==slots always), deterministic re-settle, and 29 over-cap tiles skipped once with no
  throw/leak/rebuild. GPU probe-streammove.ts (?nanitedclip=1 128, 6 vista poses incl. ±1500 hops):
  with convergence settling every pose reaches dag>0 (no permanent hole), back-spawn re-centers to
  EXACTLY the 52-tile boot set, loaded 184 / evicted 132 / skipped 0; screenshots show full relief
  filling the frame at every pose. KNOWN LIMITATION (documented, next increment): a FAST TELEPORT
  into never-cached terrain shows a TRANSIENT void while ~125 serialized cache-miss builds complete
  (~170 ms/tile first-visit; instant on cached revisit) — because ALL clipmap levels re-center, the
  coarse backstop churns too. Steady-state + continuous motion are hole-free; an always-resident
  non-churning coarse base (revived minimal suppression or a depth-underlay) eliminates even the
  teleport transient. tsc clean; probe-tilepool + probe-clipmap still green. NEXT: the always-
  resident base backstop (teleport no-hole), then 2d skirts, then 2e stride-1 verts + default-on.
- 2026-06-14 (am): **N8-D2 Stage 2b-2 (GPU) — terrain CLIPMAP RENDERS through the pool; bounded
  full-field at true 1 m (D-N39).** (Opus 4.8 1M.) Wired clipmapTiles into WorldRegistry behind
  `?nanitedclip=1` (implies the pool): a shared buildTileGlobal(tx0,tz0,stride,suffix) helper
  (extracted from the uniform loop, now clamps off-field samples both ends) builds each clipmap tile
  into a pool slot; levels auto-sized so the coarsest ring spans the field (no holes). Validated
  (probe-dterrain CLIP=1 128): CLIPMAP 5L M4 → 52 tiles, 33 398 cl, 4.0 M tris, offGrid 0, POOL
  52×(v80848/c1011) ≈ 100 MB terrain verts (vs the ~1.96 GB all-resident full-res wall) — bounded +
  ⊥ world size; cut sheds 1332 cl near → 986 vista (8.9/8.2 ms). Screenshots: lit vista CRACK-FREE
  continuous to the horizon (no sky holes), lit near = TRUE 1 m at the field center (finest level,
  stride 1 = 1 m cells; finer than Stage-1's 4 m subsample) with correct decode (planted trees/rock/
  grass), cluster-tint full coverage. Field-centered (static) for now; the inter-level seams are
  backstopped by the hollow-skip overlap (no cracks visible) — proper snapping + skirts is 2d. NEXT:
  2b-3 per-frame streaming (re-center rings on the live camera via DagWorker async load/evict).
- 2026-06-14 (al): **N8-D2 Stage 2b-2 (math) — terrain CLIPMAP residency, hole-free + bounded +
  world-size-⊥ (D-N39).** (Opus 4.8 1M.) TerrainClipmap.ts: clipmapTiles(camX,camZ,cfg) = L
  concentric levels of same-gridN tiles at DOUBLING stride (level k cell = baseStride·2^k texels),
  each an M×M block centered on the camera, the inner block HOLLOW (covered by the finer level) ⇒
  resident tiles form rings (finest full block at the center, coarser rings out; coarsest spans the
  field = backstop). Same gridN every level ⇒ ONE uniform 2a-pool slot cap. Hollow skip is
  fully-inside-only ⇒ never gaps (skipped area always backed by a finer resident level). Headless
  probe-clipmap.ts over 7 in-field poses (center, corners, sub-tile snap offsets): COVERAGE
  hole-free (every field point under ≥1 tile), FINEST-at-camera (level-0 tile over the cam),
  BOUNDED ≤ clipmapMaxTiles (cfg res4096/gridN128/L5/M4 → ≤64; live 16–52 as the small field clips
  outer rings), WORLD-SIZE ⊥ (16k≡32k = 64 tiles — the resident set does NOT grow with world size),
  deterministic, no intra-level dup. tsc clean. NEXT: 2b-2 GPU wiring (build/load the clipmap set
  into the pool at boot, ?nanitedclip=1) then 2b-3 per-frame streaming.
- 2026-06-14 (ak): **N8-D2 Stage 2b-1 — terrain tiles routed through the streaming POOL; GPU
  render parity (D-N39).** (Opus 4.8 1M.) Resolved the D-N39 2b fork → GEOMETRY CLIPMAP (same-gridN
  tiles at doubling stride, hollow rings, uniform 2a pool, NO suppression — levels don't overlap,
  2c deleted; skirts for inter-level seams). First sub-step lands the pool→GPU path: `?nanitedpool=1`
  makes WorldRegistry collect the terrain tiles and load them into the 2a pool (reserveTilePool +
  allocTileSlot + attachHeightDagTile) instead of per-tile registerHeightDag+attachHeightDag. probe-
  dterrain POOL=1 256 ×4: build BIT-IDENTICAL to the per-tile path (48 840 cl, 5 846 152 tris,
  offGrid 0), cut sheds 1729 cl near → 1375 vista (8.7/7.4 ms), lit vista CRACK-FREE (no sky holes),
  cluster-tint shows the adaptive structure + tile regions, near decode correct (planted trees/rock/
  grass). The instance-driven cull picks up pool tiles with zero cull changes (per D-N39). KNOWN: the
  pool sizes every slot to the WORST tile (DAG soup ~319 k verts/tile → v415195/c4665 ×16 = generous
  reservation, ~50 % over the per-tile sum) — expected + temporary: 2b-2/2b-3 hold FAR fewer slots
  (clipmap rings, not all 256 full-res tiles), so absolute memory drops well below all-resident.
  Default off (opt-in flag). tsc clean. Next: 2b-2 static clipmap at boot.
- 2026-06-14 (aj): **N8-D2 Stage 2a — registry streaming tile POOL + eviction (D-N39).** (Opus
  4.8 1M.) Re-read the cull pipeline first and found the design-reshaping fact: the cull is
  INSTANCE-DRIVEN (kInstCull frustum-culls per instance → only visible instances enqueue their
  cluster range), so off-frustum tiles already cost ~nothing and Stage-1 tiling ALREADY fixed the
  ground-camera cull wall; streaming's real job is MEMORY + the vista (recorded as D-N39, which
  supersedes the D-N38 Stage-2 framing). EVICTION is therefore trivial: clusterCount=0 ⇒
  lodSelectAndPush enqueues ceil(0/64)=0 chunks ⇒ the tile vanishes with NO tombstoning. Built a
  fixed-capacity tile-slot pool in GeometryRegistry (the Nanite "page" model): reserveTilePool
  (pre-build — S slots × cap, one heightfield handle + identity instance each, region claimed once
  at build()), allocTileSlot/tileFreeSlotCount (O(1) free-stack), attachHeightDagTile (reusable
  in-place write into poolBase+slot*cap — mirrors attachHeightDag's pack, rebases indices, repoints
  the mesh record), evictHeightDagTile (clusterCount→0 + sphere parked off-world TILE_EVICTED_FAR +
  free slot; idempotent). Headless probe (probe-tilepool.ts): load/evict/reload across 4 slots —
  mesh repoint + cluster/DAG/vert/index pack correct, evicted draw zeroed, a freed slot reloads a
  DIFFERENT region in place, neighbour slots + a pre-build sentinel UNCORRUPTED, over-cap throws,
  and THE invariant — **cursors stay IDENTICAL (v6924/t12802/c141) across the whole churn**
  (streaming never grows the buffers). tsc clean; probe-dagpack/registry/heightdag still green.
  Next: 2b base DAG + TerrainStreamer (per-frame residency); decide the 2b policy fork (D-N39).
- 2026-06-14 (ai): **N8-D2 — TILED terrain DAG (Stage 1): crack-free seams for free; full-res
  path unblocked (D-N38).** (Opus 4.8 1M.) Measured the single-DAG wall (4096² ≈ 1.1 GB + 640k
  cull clusters → needs tile+stream, not one DAG; gridN=1024/4 m viable now at 8.3 ms). KEY
  FINDING: buildDag already locks mesh-boundary verts, so a tile's perimeter auto-locks at full
  res ⇒ adjacent tiles share identical edges ⇒ crack-free seams with ZERO builder changes. Built
  Stage 1: `?nanitedterrain=<gridN>&nanitedtiles=T` → T×T independent tile DAGs (per-tile texel
  subregion, build at the tile's world origin, remap to global texel coords, one
  registerHeightDag+attachHeightDag + cache key per tile). Validated (probe-dterrain 256 ×4): 16
  tiles, 48 840 cl, 5.85 M tris, offGrid 0, lit vista CRACK-FREE, tint shows 16 tile regions yet
  seamless, 6.6–9.8 ms. tsc clean. Next: Stage 2 distance streaming (load near / evict far →
  bounded memory+cull) + stride-1 vertex buffer + flip the full-res default.
- 2026-06-14 (ai): **N8-D1d — terrain DAG built off-thread + CACHED; DAG-only, no window
  fallback (USER DIRECTIVE; D-N37).** (Opus 4.8 1M.) Built a Vite module Worker
  (DagWorker.worker.ts + DagWorkerClient.ts + DagWorkerTypes.ts — three-free chain, prod bundles
  it as a standalone 19.6 kB chunk) and a persistent IndexedDB cache (DagCache.ts, key
  seed+gridN+version). FIRST landed a window-placeholder + background-swap (off boot path) but the
  user REJECTED it outright — terrain must boot DIRECTLY to the DAG, zero window/default fallback
  ever — so REVERTED it. Correct shape: DAG mode registers ONLY the lean DAG, the build is AWAITED
  (boot's first terrain frame IS the DAG), and the cache makes that fast: MISS builds off-thread +
  persists, HIT loads instantly. Validated (probe-dagcache, two boots one context): boot1 [worker]
  911 ms → boot2 [cache] 21 ms, byte-identical 320-cl cut. Window path survives only as the
  separate DAG-OFF mode (?nanitedterrain=0). tsc clean; prod build clean. Remaining for full-res
  default: stride-1 terrain vertex buffer + gridN=4096 + flip the default.
- 2026-06-14 (ai): **N8-D2b — terrain DAG GPU WIRING: terrain now renders through the adaptive
  LOD DAG + the shared flat cut, headlessly validated. (D-N36.)** (Opus 4.8 1M.) Wired the D2a
  builder to the GPU behind `?nanitedterrain=<gridN>` (default 0 = window path, byte-identical).
  (1) DECODE: third heightfield branch in NaniteFetch keyed on `isDAG` — explicit indexed
  topology, vertex word0 = packed TEXEL coord → texLoadR height + world XZ, shares the window
  micro-disp. (2) PACK: lean `registerHeightDag` (hf entry, zero clusters — no orphaned-window
  waste) + `attachHeightDag` (attachDag's cluster+DAG records, grid coords in word0,
  HEIGHTFIELD|DAG flags). (3) RECONCILE: gridN² power-of-two subsample build → remap ×stride to
  texel coords clamped to res-1 (texLoadR doesn't clamp); texel cell/origin so it lands on the
  placed objects. VALIDATED (probe-dterrain.ts @256 & 512): decode A/B near-identical to the
  window path, crack-free vista, `?nandbg=cluster` shows ADAPTIVE cluster sizes (coarse far/flat,
  fine on cliffs) vs the window's uniform grid, cut live (dagClusters 320→245 @256 / 1058→857
  @512 near→vista), ~8 ms, no boot error. tsc clean. Still gridN-subsampled — full-res 4096²
  (near-camera parity, ~5 min sync) is blocked on the D1d Worker before it can be the default.
- 2026-06-14 (ai): **N8-D1e min-screen-size cull primitive (gated) + the unbounded-envelope
  RE-VALIDATION (D-N35); explicit-class DAG confirmed live; D2b GPU-wiring design mapped.**
  (Opus 4.8 1M.) Pivoted from terrain-speed (banked) toward the D1e milestone per the user's
  "reach d1e". (1) CONFIRMED the explicit-class rollout already works: `?nanitedag=all` DAGs
  rock+bark+deadwood (TerrainScene → buildWorldRegistry, `dag` set excludes terrain), envelope
  probe green (183679/175085 cl). The D1c "one hero mesh" is really a full flag-gated path. (2)
  BUILT the min-screen-size cull (`?nanitemin=<px>`, default 0): per-cluster sub-pixel drop in
  kClusterCull + `minPx` uniform in NaniteFrame. (3) PROTOTYPED + REVERTED the unbounded envelope
  → D-N35: 3.75M cl / 100 ms even occl-ON; it needs HIERARCHICAL instance culling, not a size
  flag. Default path byte-identical (gated). tsc clean; probe-minpx.ts added. (4) MAPPED the D2b
  terrain GPU wiring for the next session (2 Explore passes): the heightfield decode ALREADY
  exists (NaniteFetch `isHF` branch, CLUSTER_FLAG_HEIGHTFIELD, window-procedural — `texLoadR(
  heightTex,sx,sz)` + world XZ + micro-disp); terrain is registered via WorldRegistry.ts:372
  (`kind:'heightfield'`, uniform windows) — THE path D2b replaces. The DAG terrain needs a NEW
  decode variant: grid-coord-INDEXED (store packed `gx|gz<<16` per vertex in the mega-buffer,
  read it by vertex index, unpack→textureLoad→world XZ) distinguished by (isHF && isDAG); a
  terrain attachDag that packs grid coords not float positions (waste words 1-5 or pack tighter)
  + sets CLUSTER_FLAG_HEIGHTFIELD|CLUSTER_FLAG_DAG + the DAG cut records; and the 2^k+1
  reconciliation (HEIGHT_RES 4096 texels = 4095 quads ≠ 2^k; use gridN=4096 sampling texel
  clamp(gx,4095), cell=WORLD_SIZE/gridN, origin must match the window path's `cell/2-WORLD_HALF`
  EXACTLY or terrain shifts off the placed objects). Plus: live 4096² build is ~5 min ⇒ needs the
  Worker (D1d) before it can boot. Key files: NaniteFetch.ts (decode), GeometryRegistry.ts
  (attachDag/packHeightfield/MESH+CLUSTER layouts), NaniteCull.ts (the cut), WorldRegistry.ts:372.
- 2026-06-14 (ai): **N8-D2b (part 2) — terrain DAG build SPEED, single-thread pass
  (user picked "more single-thread opt, no Worker/tiling").** (Opus 4.8 1M.) Three
  more bit-identical structural wins on top of part 1, each profiled-then-fixed
  (terrain probe still 612 cl / 5 roots / 11.2 m; rock byte-identical): (1) TYPED-
  ARRAY pool for `poolVerts/poolIdx` — was `number[].push` over millions of floats +
  a final `Float32Array.from` over a ~400 MB JS array at 33.5M tris (capacity-doubling
  append + exact slice; this is the MEMORY fix that lets 4096² run at all, plus it
  trims the >1M-tri super-linear creep); (2) gridEndpoint keep-star SKIP — the collapse
  target IS the keep vertex's own position, so keep doesn't move and only the DROP star
  can flip/degenerate ⇒ `wouldFlip`/`triDegenerates` check one star not two (halves the
  #2 cost; bit-identical by the no-degenerate-tri invariant); (3) welding Map → open-
  addressing TYPED hash + typed soup scratch (ids stay first-encounter order ⇒ identical;
  kills the group's heaviest GC source) + the final compaction to typed scratch/subarray
  views (no `number[].push`). Clean profiler-free throughput 0.055 → 0.143 Mtri/s @256²
  (cumulative ~2.6×); 4096² ≈ 4.7 min clean single-thread (from ~10 min). Remaining minor
  levers (NOT done — diminishing, <8%): `edgeUse`/`seen` → typed hashes, `buildDag`
  per-cluster object churn. NOTE for wiring: the real 4096² build is still minutes, so it
  must run OFF the main thread (a single background Worker or frame-time-slicing — NOT
  multi-worker tiling) + a deterministic-seed cache; wiring itself is developed on small
  fields (256² ≈ 1 s) so this doesn't block it. tsc clean.
- 2026-06-14 (ai): **N8-D2b (part 1) — terrain DAG build SPEED: bit-identical
  constant-factor pass + the `__name` measurement-artifact finding.** (Opus 4.8 1M.)
  The D2a "0.015 Mtri/s / ~37 min for 4096²" alarm was profiled and largely DEBUNKED:
  a CPU profile (esbuild bundle, V8 --cpu-prof) showed **~70% of the dev/tsx wall-time
  was esbuild/tsx's `__name` keepNames wrapper** (every closure wrapped + a
  `defineProperty` — surfaced as `set natives` 31% + `__name` 34%), an artifact ABSENT
  from a production/minified bundle. Profiler-free clean (bundled, no keepNames) the
  ORIGINAL was 2371 ms @256² = 0.055 Mtri/s (~10 min for 4096², not 37). The TRUE
  algorithmic hot spot was `wouldFlip` (the orientation-flip guard) at ~50% — because
  `rawCross` defined a nested `gx` CLOSURE re-allocated every call (16.6% self-time
  alone), and `triDegenerates` did the same with `pos`. Four bit-identical fixes (the
  terrain probe is unchanged — 612 cl / 5 roots / maxErr 11.2 m / 3.6× cliffs; rock
  byte-identical, cut-sweeps exact): (1) terrainFast — skip the entire QEM quadric
  machinery in `gridErrAt` mode (it computed Q but NEVER read it for cost/target);
  (2) Set→version-stamp (`Int32Array`+gen counter) in `linkOk` + the reseed neighbour
  dedup (kills per-call Set alloc); (3) INLINE `wouldFlip` — no `rawCross`/`gx` closure,
  fully unrolled cross-products; (4) INLINE `triDegenerates` likewise. Result: clean
  1316 ms @256² = 0.100 Mtri/s = **1.80×** (profiler-free, production-representative),
  bit-identical. Clean scaling ~flat 0.08–0.10 Mtri/s to 2M tris, with MILD super-linear
  creep at 1024² (the `poolVerts: number[]` growing-array + GC). probe-heightdag-scale.ts
  added (clean scaling regression). REMAINING for 4096²-viable (the speed prereq is
  reduced, not closed — ~5.6 min clean single-thread): a TYPED-ARRAY pool to replace
  `poolVerts/poolIdx` Array.push (kills the creep + the ~400 MB JS-array memory blowup at
  33.5M tris), then the D-N30 Worker (off boot path) + a deterministic-seed build CACHE;
  Map elimination (weld-by-grid-id, edgeUse typed hash) is a further ~lever. tsc clean.
- 2026-06-14 (ai): **N8-D2a — terrain heightfield-native adaptive DAG builder
  (BuildHeightDag.ts) + node probe; CRACK-FREE, on-grid (F4), adaptive; D-N32 OPEN
  questions settled; D-N34 recorded.** (Opus 4.8 1M.) The terrain DAG rides the SAME
  flat kClusterCull cut as rock (D-N31) — unified runtime, specialised build — via a
  synthesis I had to derive: pure martini/RTIN is crack-free only under per-FRAME
  bintree traversal (ROAM); baked into the flat per-cluster cut, independent per-band
  meshes T-junction at the frontier. So the build feeds the martini VERTICAL-error
  pyramid into BuildDag's PROVEN locked-boundary scaffolding (siblings share the parent
  pair exactly ⇒ cut falls between groups where verts were locked ⇒ crack-free), with
  three flag-guarded BuildDag additions (all gated by `gridEndpoint`, default false ⇒
  rock path BYTE-IDENTICAL, probe-dag re-green-confirmed):
  • `gridEndpoint` — interior edge-collapse targets a grid ENDPOINT, never the off-grid
    QEM-optimal point → every survivor stays on the heightfield grid (F4: store packed
    grid coords `gx|gz<<16`, GPU fetches height; no baked floats).
  • `gridErrAt(x,z)` (martini pyramid, O(n) bottom-up) — the collapse DROPS the lower-
    error vertex and its cost IS that vertical error in metres (the cut's own unit), not
    the area-weighted QEM scalar.
  • `levelBudget(ℓ)=e₀·2^ℓ` — ERROR-BOUNDED, not ratio: a level runs only collapses ≤
    the band; costlier ones DEFER to the next (higher-budget) level (stuck cliffs earn
    intermediate LODs instead of freezing at LOD0 → smooth ±1 cut). LOD0 = full grid
    (ownError 0 ⇒ no holes up close, even on cliffs).
  • MANIFOLD-SAFETY (the crack hunt — adversarial flat|ramp|ridge field, probe W check
    keyed by CANONICAL grid-id catches T-junctions AND overlaps): ratio→error-bound took
    cracks 20→4; +link-condition (reject collapses where a,b share a non-apex neighbour
    → 3-tris-on-edge) →2; +degeneracy (reject near-COLLINEAR result, sin²θ<1e-8 — a flat
    boundary row was folding zero-area slivers that both groups re-made) →1; +shared-
    boundary (reject a result triangle with all-3 verts LOCKED — a seam triangle the
    neighbour group re-creates identically) →**0**. Endpoint-collapse on a REGULAR grid
    trips manifold cases QEM-optimal on irregular rock never does; rock didn't need these.
  • BuildHeightDag.ts wraps it: build LOD0 grid mesh (stride-3, winding matched to the
    GPU heightfield path), martini errs, call buildDag, recover grid coords from the on-
    grid survivor positions → pack `gridVerts`. Reuses clusterize() + ALL DAG metadata.
  GATES (tools/probe-heightdag.ts, node, all green; tsc clean): M/C/E/O/A (shared cut
  invariants) + **W watertight** (every interior edge of the selected cut used exactly 2×
  at τ∈{0,.5,2,8,∞}) + **G on-grid/F4** (offGrid 0, residual 0 cells) + **D deterministic**
  + **adaptivity** (τ=1 cut, cam 100 m up: flat/ramp 3.5k tris vs cliffs 12.8k = 3.6×
  denser cliffs, 46% culled — the "plains SIGNIFICANTLY fewer tris" mandate, MEASURED).
  CAVEAT for D2b: build throughput is 0.01 Mtri/s (32 768-tri probe in 2.4 s — the
  iterative QEM heap + per-collapse manifold guards); the martini pyramid is O(n) but the
  REMOVAL is heap-driven → for 4096² (~33M tris) this needs a martini-DIRECT removal
  and/or the D-N30 Worker before it's viable. D2a proves CORRECTNESS; speed is D2b.

- 2026-06-14 (ah): **N8-D1 user re-test → DAG'd trees VANISHED at ~26 m; root-caused
  to the attachDag draw-envelope + fixed (+ a cut error-scale bug found mid-trace).**
  (Opus 4.8 1M.) USER (cluster-debug walk, ?nanitedag=all): "trees disappear after a
  few tree-sizes; rocks coarsen correctly then also vanish, but LATER; bushes stick
  around the LONGEST." Three behaviours = ONE cause: attachDag retired the LOD chain
  (lodNext=NONE) but LEFT the head's chain-SWITCH distance as lodDist, so the instance
  envelope rule `lodNext==NONE && lodDist>0 && dist>lodDist` dropped the WHOLE instance
  past it — trees at R0_FAR=26 m, rocks/deadwood at EX_R1_FAR=120 m, shrubs at switch 0
  = NEVER (the "stick around longest" tell). The cut was innocent: the root is pinned
  (parentErr=1e30, can NEVER be cut) so a DAG cannot vanish via the cut → the drop is
  provably instance-level. Nanite full-frame draws NO impostors (TerrainScene: old solid
  paths OFF) so the drop reads as a clean vanish, not a billboard pop.
  • FIX 1 (envelope, GeometryRegistry.attachDag): inherit the chain's MAX draw distance
    (walk head→tail BEFORE collapsing — trees TREE_GEO_FAR 496 m, rocks clsMaxDist), NOT
    the head switch and NOT unbounded. I TRIED unbounded (lodDist=0) first and MEASURED:
    3.70M clusters / 92 ms (~11 fps) even with occlusion ON — open vista, little HZB
    occlusion, and the pinned root means every sub-pixel far object still draws ≥1
    cluster. Intended-envelope: 0.14M cl / 16.3 ms near / 8.4 ms far (~60 fps). Unbounded
    is the N8 end state but needs a min-screen-size cull first → DEFERRED to D1e (D-N33).
  • FIX 2 (cut error scale, NaniteCull): pOwn/pPar now multiply rec.ownError/parentError
    by A.w (instance scale) — the own/parent spheres ride the instance transform (×A.w)
    but the error was raw LOCAL metres, so non-unit-scale instances (big trees) picked the
    wrong LOD band. Root sentinel 1e30·A.w still ≫ τ (roots stay pinned). No-op at A.w≈1.
  • GATES (all green, tsc clean): NEW tools/probe-envelope.ts (end-to-end — pull camera
    −300 m back so every object is past the old 26/120 m bug envelope; DAG still draws
    183k cl, no collapse) + probe-envperf.ts (the 92→16 ms occl on/off measurement).
    probe-dagpack.ts gained a 2-LOD-chain envelope assertion (lodDist == tail max 496, NOT
    head switch 26), RED/GREEN proven by toggling the fix. probe-zoom.ts: median-of-9
    reads (kill TAA frustum-edge jitter the now-smaller finite-envelope counts exposed) →
    τ-sweep cleanly monotonic 1026→1104, smooth zoom. probe-dag still green.
  STILL OPEN: trees vanish at 496 m (no impostor past it = pre-existing nanite behaviour);
  truly-unbounded geometry + the min-screen cull = D1e (D-N33).

- 2026-06-13 (ag): **N8-D1a/b/c — GPU continuous-LOD cut LIVE on rock; the N8
  LOGICAL POINT is reached.** (Opus 4.8 1M.) Three commits:
  • D1a (48a1ccd) — registry DAG sidecar: a parallel DAG_WORDS=10 f32 buffer
    (ownErr+ownSphere4 + parentErr+parentSphere4) on RegistryGpu, indexed by the
    same global clusterId (the 8-word cluster rec is full). `attachDag(handle,
    DagBuild)` appends the DAG's full self-contained geometry (all levels incl a
    LOD0 copy — D1 trades duplication for zero index-rebase), re-packs verts
    (explicitToDagVerts stride-12 → VERT_WORDS), writes 8-word recs (CLUSTER_FLAG_DAG)
    + the 10-float cut recs, REPOINTS the mesh at its DAG range + clears lodNext
    (MESH_FLAG_HASDAG). Roots' +inf → DAG_ROOT_PARENT_ERR sentinel (1e30),
    parentSphere←ownSphere. Validated node-only (tools/probe-dagpack.ts): pack
    round-trips f32-faithful; hero rock-d7 τ=0→327680 tris (=LOD0), far→23 cl (=roots).
  • D1b/c (b635b05) — the cut on GPU: kClusterCull tests
    `project(own)≤τ AND project(parent)>τ` (projK=(screenH/2)·cot(fovY/2), mirrors
    probe-dag.project; +1 storage = 8/10 F9 ok). kInstCull UNTOUCHED (repointed
    range makes lodSelectAndPush push the whole DAG). τ uniform (?loderr, live
    __laasNanite.setTau). NaniteShadow reuses buildNaniteCull → 4 cascades inherit
    the cut free (casters track lit surface; decoupled caster-LOD = S4). slot-5
    nanite.dagClusters isolates the rock cut. addLate() reserves the post-build
    append budget; WorldRegistry DAGs ?nanitedag=rock|bark|deadwood|all SYNC at
    boot (D1d Workerizes); TerrainScene threads it.
  • GATE (tools/probe-zoom.ts, bm4 boulder, occl off): τ-sweep 32→0.25 →
    dagClusters 63→63→64→70→74→77→86→124 (MONOTONIC refine, no collapse); zoom
    dolly smooth (≤7% step, no pop); WATERTIGHT at both τ extremes (no cracks,
    eyeballed); τ=1 PIXEL-MATCHES discrete LOD; shadows-ON correct. Rock DAG = 20
    meshes / 192 ms / 94k tris; registry boot 766 ms (rock-only — F15 fine).
    probe-dag + probe-dagpack + probe-registry green; tsc clean. See D-N31 (flat
    per-cluster cut, NOT the hierarchical traversal — semantics vs optimisation).
  REMAINING to fully close N8-D1: D1d (Worker build, D-N30) + D1e (bark/deadwood
  DAG + perf ledger + ?clusterdbg=lod heatmap + USER CHECKPOINT). Then N8-D2 +
  shadows resume at S4.

- 2026-06-13 (af): **N8-D0 — hand-rolled QEM LOD DAG build LANDED + validated
  headless; F15 build-cost trigger FIRED (measured).** (Opus 4.8 1M.) NEW
  src/nanite/BuildDag.ts (`buildDag(verts, vertStride, indices, opts, lod0?)`):
  per level k→k+1 — spatial-median group partition (≤24 clusters; a deterministic
  METIS substitute — crack-freeness is partitioner-INDEPENDENT, only reduction
  efficiency isn't) → position-WELD the group soup (spatial-hash buckets, no
  string keys) → LOCK the group boundary (soup edges used by ≠2 tris = shared-
  with-another-group ∪ open-mesh-boundary) → area-weighted Garland-Heckbert QEM
  edge-collapse to ~50% (binary-heap of collapses, lazy-versioned stale skip,
  3×3 Cramer optimal placement w/ endpoint/midpoint fallback, raw-cross normal-
  FLIP guard) → re-clusterize survivors via clusterize() into 4–16 parents →
  (ownError,ownSphere)/(parentError,parentSphere) pairs with containment (sphere-
  fold), max-monotonicity (+ strict ε bump), and EXACT sibling-pair equality
  (child.parent pair === group pair === parent.own pair, bit-for-bit, so the cut
  boundary always falls between groups). Stuck-fallback: a group reducing <15% →
  its inputs become ROOTS (parentError=∞); multiple roots legal. Attributes ride
  along by linear interp on collapse (normal renormalised via opts.normalOffset);
  RNG-FREE → deterministic by construction. NEW tools/probe-dag.ts (node, no GPU):
  builds on rock (closed, 3 detail levels) + bark-beech (open tube) + deadwood-
  snag, asserts M (parentErr≥ownErr) / C (parentSphere⊇ownSphere) / E (bit-exact
  sibling pairs) / O (no orphans, LOD0 ownErr=0, roots vs grouped) / A (τ-sweep
  cut antichain: no group ever has input+parent both selected; τ=0 cut == LOD0
  tris, τ=∞ == root tris, tris monotone-decreasing in τ) + a 2× determinism check.
  ALL GREEN. Clean ~50% per-level reduction; rock-hero 327,680 tris → 10 levels,
  5,407 clusters, 23 roots; bark 6 roots / snag 4 roots (open/branchy topology
  refuses to over-simplify — the predicted multiple-roots behaviour). **F15
  MEASURED (the boot-budget trigger): pure-TS build = 0.16 Mtri/s (hero 2.0 s /
  327k tris) → the registry's 1.52M explicit tris ≈ 9.3 s, 4M ≈ 24.5 s — OVER the
  ~2 s soft target.** Decision = D-N30 (background/progressive per-pool build, off
  the boot critical path — the spec's pre-planned mitigation, endorsed by the
  user's "i dont care how long it … wont be visible before finished" directive;
  compute-kernel QEM is the documented floor if a background Worker still starves).
  Easy TS opts already applied (weld-hash, sqrt-free flip): 2540→2004 ms hero. The
  builder is correct + the cost is a wiring concern for D1, not a builder defect.
  tsc clean; probe-clusterize still green (unchanged). NEXT: N8-D1 (GPU pack +
  hierarchical runtime cut) — see NEXT ACTIONS.

- 2026-06-13 (ae): **SHADOW/LIGHTING PERF RETHINK — research complete, direction
  chosen (pending sign-off).** (Opus 4.8 1M.) Ran 4 parallel cited research threads
  (production foliage-shadow practice; VSM deep-dive; SDF/voxel/SW-RT; static/dynamic +
  unified GI). All converged: wind casters can't be cached (Epic VSM docs: WPO "always
  invalidates cached pages every frame"), so the production answer is a DISTANCE-REGIME
  SPLIT of the caster representation (near full-sway+modest-coarsen / mid hard-coarsen+
  WPO-freeze / far rigid-cached-impostor — SpeedTree's per-cascade scheme), NOT a
  cleverer cache. VSM rejected (cache dies to wind, Metal disables StaticSeparate);
  unified shadow+GI rejected as primary sun shadow (UE Lumen keeps a separate VSM; SVO
  "too soft"); SW-RT of deformed tris rejected (per-frame BVH rebuild over 162k). Net:
  KEEP SW-raster-into-CSM, replace R2/R3/R4 with R2′/R3′/R3″/R5/R6 (enhanced superset of
  the already-planned coarse-far-LOD + static/dynamic split + two NEW levers: aggressive
  shadow-pass WPO-freeze distance, contact-shadows for small foliage). Capsule-proxy SDF
  staged as a mid/far escalation if the proven path's ceiling is hit. Full findings +
  staged plan in NEXT ACTIONS; promotes to D-N29 on sign-off. No code changed this
  session (research only); tsc not re-run (no edits).

- 2026-06-14 (ad): **Nanite shadows DEFAULT-ON + C1 dead-end cleanup** (user
  directive: "shadows on by default, the new decently fast shadows; get rid of the
  previous super slow shadow stuff"). (Opus 4.8 1M.) Flipped the producer from
  opt-in `?nanshadow2=1` to DEFAULT-ON and UNIFIED it with the resolve's receive
  flag → ONE `?nanshadow` (default on; `?nanshadow=0` disables producer + receive
  together). The C1 HW vertex-pulling caster was already deleted in the R0 rewrite;
  this removes its remnant comments/flags (NaniteFrame header + run-site, NaniteShadow
  header, the nanshadow2 name) and renames shadow2On→shadowOn. The old Forests
  per-pool caster siblings + ShadowProxy are RETAINED (gated behind
  !DISABLE_OLD_GEOMETRY = ?oldgeo only — the A/B reference path; their wholesale
  deletion is N6/N9 pool-migration scope, not this cleanup — flagged for the user).
  VERIFIED: default `?nanite=1` (no shadow flag) @bm7 → shadows present + attached
  (trunk-on-grass, bark self-shade), shRaster=0 (R1 cache), nanRasterDepth 1.38 ms
  == nanRasterPayload (shadow raster zeroed static), fps 117@720p; `?nanshadow=0` →
  producer absent, clean no-shadow A/B, fps 119; bm3 vista (heaviest, shTotal 2.26M)
  → no queue overflow, cached, fps 103. tsc clean. HONEST NOTES: default-on always
  allocates the 4× buildNaniteRaster (~200 MB — R2's depthOnly option trims it), and
  three still renders 4 EMPTY CSM cascade maps per refresh (the keep-alive fit
  side-effect — cheap, empty draws). MOVING cost is still the R1 residual (~35% of
  R0 raster, ~11 ms); R2 (coarse far LOD + depthOnly) + R3 (static/dynamic wind)
  bring it to the 1–2 ms target. NEXT: R2.

- 2026-06-14 (ac): **N5-R1 — CADENCE: per-cascade shadow raster gated on VP change.
  Static camera ⇒ ~0 shadow cost; moving ⇒ only changed cascades.** (Opus 4.8 1M.)
  R0 rastered all 4 cascades EVERY frame (35 ms). R1 gates the per-cascade re-raster
  (clearVis→cull→depth1→hwDepth→kCopy) on whether the cascade's light VP changed
  since its last raster: `cascM = proj·viewInv`, exact `Matrix4.equals(lastVP)` →
  skip. ROBUST with no epsilon because CsmCached FREEZES the light pose between
  refreshes (CsmCached.ts:294 — a cached cascade `continue`s without moving lwLight),
  so a frozen cascade's recomputed VP is BIT-IDENTICAL frame to frame. The depthTex
  StorageTexture persists across frames, so a skip retains last refresh's depth;
  cascVP[c]/cascParam[c] are left untouched in the SAME skip → raster/sample
  LOCKSTEP holds automatically (D-N28's #1 correctness item). New `rasteredMask()`
  (bit c = cascade c re-rastered this run) → stats `nanite.shRaster` for crisp
  non-timing validation. GATES (both halves, bm7):
  • STATIC (shoot, settle 40): nanite.shRaster=0 (all cached); c.nanRasterDepth
    35→**1.77 ms** — and IDENTICAL to c.nanRasterPayload (which is camera-only:
    shadows are depth-only / no payload by D-N28), so the cascades contributed
    ZERO to the depth raster this frame = structural proof the ~33 ms shadow raster
    is gone. fps 22→63. Shadows still present + ATTACHED (cached depth → PCSS):
    trunk casts on grass, downed logs cast, bark self-shades — no peter-pan, no
    visual regression vs R0.
  • MOVING (NEW tools/probe-shadowcadence.ts — world frozen, step camera 1.5 m/fr,
    read shRaster each frame): clean period-6 pattern `1111 0001 0011 0101 0011
    0001`…; c0 90% (period 1, tracks camera), c1 47% (/2), c2 30% (/3), c3 17%
    (/6) = exactly CsmCached PERIODS [1,2,3,6]. 1.83 cascade-rasters/frame vs R0's
    4.00 always; cluster-weighted (shC0..3 = 170k/238k/358k/553k — the cascade that
    rasters most is the CHEAPEST) ≈ 464k/frame = **35% of R0's 1.319M/frame** →
    ~65% less moving-camera raster work, ~100% less static. Probe `--static`
    control = 0.00/frame (validates the tool). tsc clean (incl. the new tool).
  STILL ABOVE the 1–2 ms target while MOVING (the 35% residual ≈ 11 ms shadow
  raster): R2 (coarse far-cascade LOD + a depthOnly buildNaniteRaster option that
  drops the unused payload/HW-resolve build, ~64 MB/cascade) and R3 (static/dynamic
  wind split — cache static depth, re-atomicMin only trunk-channel clusters) close
  it. ?nanshadow2=1 still default-OFF (R4 flips). NEXT: R2.

- 2026-06-14 (ab): **N5-R0 — depth-only SW shadow raster: ARCHITECTURE PROVEN,
  shadows CORRECT (perf is R1–R3).** (Opus 4.8 1M.) Pivoted off the HW caster
  (D-N28, research-grounded). Per cascade: reuse buildNaniteRaster DEPTH-ONLY
  (clearVis→cull.runPhase1→depth1→hwDepth, cam.vp = the cascade LIGHT VP, NO
  payload), copy vis-depth (u32 f32-bits; 0xffffffff→far 1.0) into an own r32f
  StorageTexture; the resolve's shadow factor (NaniteResolve:499) is now OUR PCSS
  over those textures (cascade-select by coverage + world-metric penumbra + Vogel
  PCF), replacing nodeObject(world.csm).x. Shadows CORRECT at bm7: trunk + terrain
  self-shadow, on/off mean|diff| 25.7, 46.9% px darker; cascade-coverage debug
  shows proper near→far red/green/blue. THREE BUGS found+fixed (all non-obvious,
  recorded as GOTCHAs/here): (1) readback of an un-run cascade's GPU buffer →
  FATAL copyBufferToBuffer; fixed with a per-cascade `ran` guard in readCounts.
  (2) THE CSM NEVER FIT ITS CASCADES — removing the HW casters meant three never
  rendered the sun shadow, AND the resolve no longer referenced nodeObject(csm),
  so CsmCached.updateBefore (the per-frame cascade FIT) ran in NO graph →
  csm.lights empty → cull skipped → no shadows. Calling updateBefore() manually
  CRASHED (node never setup() → null frustums). FIX: keep the resolve REFERENCING
  nodeObject(world.csm) as a ×1 keep-alive (its map is EMPTY in the black slate →
  factor 1 → folds out) so three runs its setup + per-frame fit; we read the
  fitted csm.lights[c].shadow.camera VPs. (3) NO SHADOWS until the bias fix: the
  cascade near/far span the full lightMargin+maxFar (~7740 m) so ALL geometry
  compresses to z≈1.0; a constant [0,1] DEPTH_BIAS=4e-4 was ≈3 m of world slop and
  ate every shadow. FIX: WORLD-METRIC bias = DEPTH_BIAS_M(0.35)/depthRange per
  cascade. Diagnosed via ?nandbg=shadow/shadowc/shadowd (factor/cascade/depth
  debug views — KEPT, gated). PERF (the whole point, NOT yet met): nanRasterDepth
  = **35 ms** (all 1.44M cascade clusters rastered EVERY frame, no cache/LOD) →
  22 fps. This is the uncached R0 baseline; the 1–2 ms target is R1+R2+R3:
  • R1 CADENCE — re-raster a cascade only when its VP changes (cache the r32
    texture between; CsmCached already freezes far cascades [1,2,3,6]). Cascade 0
    (period 1) still every frame.
  • R2 COARSE LOD for far-cascade casters (cut cluster counts) + a depthOnly
    buildNaniteRaster option (the per-cascade rasters currently build unused
    payload/resolve/audit — ~64 MB + build waste).
  • R3 STATIC/DYNAMIC split — cache static cascade depth, atomicMin only the
    trunk-channel (wind) clusters each frame on top. Brings cascade 0 to ~0.3 ms.
  Estimated R1+R2+R3 ≈ 1–2 ms (cache the static bulk, re-raster only what moved).
  ?nanshadow2=1 still default-OFF. tsc clean. Memory note: 4× full buildNaniteRaster
  (vis depth+payload + hwRT + hwQueue per cascade) ≈ 200 MB — trim with depthOnly
  in R2. NEXT: R1 (cadence) on a FRESH context (this one is diluted).



- 2026-06-14 (z): **N5-C1 — HW vertex-pulling CASTER landed; black-slate nanite
  geometry now SELF-SHADOWS (the C1 gate is GREEN).** (Opus 4.8, 1M ctx.) Per
  cascade: a vertex-pulling NodeMaterial Mesh (layer 2+c, castShadow=true,
  frustumCulled=false, identity matrixWorld) added to engine.scene; three's CSM
  renders ONLY layer 2+c into cascade c. A per-cascade kCasterArgs kernel turns
  cull.qRasterRO[0].x (visible count) into a non-indexed indirect DRAW
  (vertexCount = count·128·3 over-draw; padding tris localTri≥triCount collapse to
  vec3(0)). buildNaniteShadow now takes (heightTex, disp, wind) — built with the
  SAME makeFetch args as the CAMERA raster so caster geometry is bit-identical
  (no peter-pan). THE INTEGRATION MECHANISM (verified vs three 0.184 source, the
  whole reason this chunk was "risky"): the shadow pass IGNORES a mesh's
  vertexNode — it swaps in a shared depth OVERRIDE material and reads ONLY
  material.castShadowPositionNode (Renderer._getShadowNodes → positionLocal,
  three.webgpu.js:61042+). That node returns LOCAL space; three applies
  modelViewProjection = cameraProj·cameraView·modelWorld, and during cascade c's
  shadow render the active camera IS the cascade ortho light, so cameraView/Proj
  ARE the cascade light VP. Forcing matrixWorld=identity ⇒ the world pos
  fetchWorldVert returns lands straight in light clip space (no hand-rolled vp).
  side=DoubleSide so the override's shadow side stays DoubleSide (the default
  FrontSide→BackSide shadow flip would CULL terrain's single up-faces ⇒ no terrain
  shadow). THE BUG THAT ATE THE SESSION (D-N27 + GOTCHA): a base NodeMaterial
  leaves `map` UNDEFINED, but _getShadowNodes gates on `material.map !== null`
  (NOT truthiness) → undefined passes → three does reference('map','texture',
  material) on a missing texture → "texture() expects a valid Texture" ×4 (one per
  cascade) → the shadow override fails to build → NO shadows, while the caster's
  18 ms shadow.c0 (pure vertex over-draw) masqueraded as "it's rendering". Root-
  caused by patching three.webgpu.js (the BUILT bundle — Vite serves
  build/three.webgpu.js, NOT src/, and pre-bundles it; needed
  optimizeDeps.esbuildOptions.target=esnext + --force to re-bundle the patch) to
  dump the TextureNode.setup stack → ReferenceNode→TextureNode → the map line. FIX
  = one line: `mat.map = null` (MeshStandardNodeMaterial sets it; base NodeMaterial
  does not). DIAGNOSIS LADDER that proved the geometry was fine all along:
  ?nancasterdbg=1 (KEPT — renders cascade 0's vertex-pulled caster in the MAIN
  pass, emissive, depthTest off) showed correct green trunks+terrain ⇒ decode +
  makeFetch + worldPos all correct; the failure was purely three's shadow-override
  build. RESULT (1280×720, framealign-free settle-40, casters on vs off): bm7
  mean|diff| 0.79→14.9, 41% darker; bm3 vista 25.9 mean, 71.8% darker — coherent
  cast shadows (amplified diff = trunk + downslope ridge shadows, NOT speckle),
  ridge trees shade the whole bm3 foreground slope. PERF (the C3 problem, logged
  loud): the caster DRAW is heavy — shadow.c0 18 ms@bm7 / 36 ms@bm3 (720p), fps
  121→42 / 109→29. cascade 0 (CsmCached PERIOD 1) redraws EVERY frame so cadence
  won't save it; C3 = SW depth-only atomicMin raster (D-N5/D-N26) or exact-tri
  compaction to kill the 128-stride over-draw. SCOPE NOTE: the cull still runs
  every frame for all cascades (C0 unchanged); three only DRAWS a cascade on its
  CsmCached refresh tick, so gating the CULL to the tick is folded into C3.
  Robustness: no texture errors / overflow at bm1/3/4/7; default (no ?nanshadow2)
  path untouched. ALSO landed (dev-env, unrelated): vite.config
  optimizeDeps.esbuildOptions.target=esnext — three's capabilities/WebGPU.js has a
  top-level await the dep optimizer's default target rejects; the old setup only
  worked off a warm .vite cache (a cold clear broke `npm run dev`). tsc clean.
  NEXT: N5-C2 (retire ShadowProxy + Forests caster siblings, shadow parity, flip
  ?nanshadow2 default-on) then N5-C3 (the perf pass above).

- 2026-06-13 (y): **N5-C0 — per-cascade nanite shadow CULL landed** (numeric
  milestone, no raster yet). (Opus 4.8.) NEW src/nanite/NaniteShadow.ts:
  SHADOW_CASCADES (4) cull chains, each = buildNaniteCull with the cascade's
  ORTHO frustum planes (refreshed one frame stale from csm.lights[c].shadow.camera
  — the Forests.planesCsmU hook), sphereOccluded=null (casters never HZB-occluded,
  F5), coneCull=false (NEW buildNaniteCull opt — the camera-relative cone backface
  is wrong for a light view), camPos = MAIN camera (LOD matches the lit geometry,
  no peter-pan). Pins each cascade camera to layer 2+c (cam.layers.enable) for
  C1's caster siblings. Wired into NaniteFrame behind ?nanshadow2=1 (default off):
  shadow.update runs before post.render reading last-frame cascade fits; per-
  cascade visClusters → HUD nanite.shC0..3 + shTotal. VALIDATED at 2 framings:
  bm7 181k/269k/427k/565k (total 1.44M), bm3 vista 384k/496k/625k/756k (total
  2.26M) — MONOTONIC by cascade index (each ortho box covers a geometrically
  larger area), every cascade < the 2M QRASTER_CAP, no overflow, no errors; the
  near cascade exceeds the camera's 53k visClusters (no occlusion + wider box, as
  expected). Cost: +~2.5 ms (nanInstCull 0.7→3.2, aggregated across the now-5
  instance culls) — C3 will gate re-cull on each cascade's CsmCached refresh tick
  + drop the unused reject buffers (~144 MB across 4 cascades). tsc clean. NEXT:
  N5-C1 — the HW vertex-pulling caster mesh per cascade (the risky integration
  chunk): nanite self-shadows with ?nanshadow2=1 and NO ?oldgeo.

- 2026-06-13 (x): **N4-C4 close — N4 COMPLETE. Full verification battery green +
  bark/deadwood shadow-receive gate + first full-beauty perf row.** (Opus 4.8.)
  The C3 material+wind+limit-raise+matParam changes are confirmed NON-regressive
  to the cull/raster/registry pipeline: registry (matParam word-7 dual-use intact
  — H/M checks pass), registry-gpu (decode exact), nanitedbg flat+cluster (0
  errors, no overflow), pan (0 holes/10 frames, phase-2 live 42k appends). The N3
  SILHOUETTE gates re-pass BIT-IDENTICAL to N3 once the wind confound is removed:
  parity silhouette 4/0/102/39/13 px (≤0.0111%, gate 0.05%) + flips ≤0.0122%;
  horizon graze + nearfield shimmer/holes/orphans 0, silhouette 0 px. KEY METHOD:
  C3 put trunk wind in the SHARED fetchWorldVert (raster geometry), but NaniteHwRef
  (the parity reference) is rigid — so the gates run `?nanwind=0` (rigid raster ≡
  rigid hwref); the wind branch compiles OUT cleanly with wind off, proving zero
  raster drift. Added an EXTRA env forward to probe-parity (PARITY_EXTRA) +
  probe-horizon-nanite (HORIZON_EXTRA), mirroring pan's PAN_EXTRA.
  SHADOW-RECEIVE — the hard-won part (NEW tools/probe-barkshadow.ts + GOTCHA):
  the FIRST cut ran in black slate and "passed" — FALSELY. Black slate has an
  EMPTY CSM (ShadowProxy + Forests per-cascade casters are BOTH gated behind
  !DISABLE_OLD_GEOMETRY → nothing casts until N5), so ?nanshadow on/off there
  differs ONLY by cross-boot TRAA jitter on busy bark texture (10.9% of px,
  salt-and-pepper ON the fissures — diffed to a red overlay to SEE it, the
  decisive check). The probe was measuring jitter, not shadows. FIX: run the gate
  with `?oldgeo=1` (restores the casters; migrated tree CAMERA draws stay hidden
  via suppressMigrated so nanite bark still owns its pixels and RECEIVES the old
  casters' shadows — the N4 hybrid path, identical to how terrain's receive was
  proven in (s)) + `wind=0 lockexp=1 framealign 0` (deterministic; beauty−noshadow
  diff = the PURE shadow term). Result: coherent cast shadow falls across the
  trunk (not speckle — verified by the framealigned diff image), bm7 50,050
  shadowed-sunlit bark px. NO-BLACK metric redesigned twice: an absolute luma
  floor is the WRONG instrument (fights both the tonemap toe AND the bark's own
  cavity-AO fissure crevices — a zoomed crop showed 21% of DARK px pure-black but
  57% brown = correct detailed dark bark, not a void). The honest signal is
  WARM-ALBEDO RETENTION: a zero-ambient bug zeroes albedo→flat grey-black
  (chroma→0); correct dim shadow keeps the warm tint. bm7 shadowed-sunlit bark =
  100% warm-albedo, 0% void → no-black ✓. Deadwood shares bark's isBD branch
  VERIFIED (one class-agnostic lighting block, NaniteResolve 488–527; only albedo
  differs per isD.select, deadwood's is DIMMER = conservative) — bark's proof
  covers it (its thin ground logs give 6–16k eroded px but 0 sunlit-then-shadowed,
  so not gated on the subset directly). Wind shimmer glance: live (nofreeze)
  ?wind=1 bm7 settled 40 frames renders bark CRISP — no crawl/shimmer (the
  bit-identical raster/resolve windy-position reconstruction holds under TRAA).
  PERF (ledger row, 2592×1676, gpusample-24): FIRST full-beauty nanite row —
  cpu.submit 11–15→**1.4 ms**, draws 548–905→**21**, fps bm1/3/4/7 = 99/60/61/71
  (frameMs 8.3/17.5/15.9/17.5); nanite GPU 3–10 ms, frame floor is post+CSM (F11).
  NOT beauty-comparable to main (D-N21 black slate has no grass/cards/water/leaf).
  tsc clean. NEXT: N5 — per-cascade cluster shadow re-culls (make the cluster
  pipeline DRIVE the casters; until then the resolve correctly SAMPLES the CSM but
  only the old ShadowProxy/Forests casters fill it, i.e. only under ?oldgeo=1).

- 2026-06-13 (w): **N4-C3 trunk WIND channel landed — N4-C3 now COMPLETE.**
  (Opus 4.8.) The 'trunk' transform channel (Wind.vegWindOffset assembly, leaf
  flutter omitted — trunks have low flex) ported into the SHARED fetchWorldVert,
  so the raster (geometry) and the resolve (barycentric corners) reconstruct
  bit-identical windy positions by construction. Per-instance wind profile
  (tree/snag/shrub → k/freq/h0) packed into matParam's HIGH byte (D-N24 low byte
  = bark slice); the 4 gust FIELD texture samples (exposure/gust/lag) are
  precomputed ONCE per instance in makeCtx (gated on the trunk channel so
  terrain/rock pay nothing), leaving only the per-vertex prof/flex ALU in
  fetchWorldVert. Reuses the Wind module's gustAt/gustLagAt/windExposure + windU
  + TSL time (gustLagAt + WIND_LAG_M newly exported); slotHash re-derived so the
  per-instance phase matches the old path's bake. VERIFIED: bark still renders
  correct (no cracks/barycentric corruption ⇒ raster/resolve agree), ?wind=2.5
  visibly bends background trunks downwind while ?nanwind=0 keeps them straight
  (A/B), perf 116.5 vs 118.9 fps (~2%, the precompute pays off), tsc clean. The
  --wind 0 / ?nanwind=0 no-op safety holds (strength 0 zeroes the offset).
  Deadwood stays RIGID (channel 'rigid', as the old path). NEXT: N4-C4 close
  (shadow-receive + no-black-shadows verification, full probe battery, perf
  ledger row, USER CHECKPOINT), then N5.

- 2026-06-13 (v): **N4-C3 BARK + DEADWOOD material landed — the nanite world's
  first TEXTURED, UV-mapped, normal-mapped class (trunks + snags).** (Opus 4.8.)
  PORTED_CLASSES += bark + deadwood (~684k tree/shrub + 25k deadwood instances
  move into the resolve). New capabilities over rock (C2): per-vertex UV
  interpolation, a sampled bark texture-ARRAY (D-N24 — one texture_2d_array per
  map, slice == species bark layer, threaded as the registry's per-mesh matParam
  in mesh word 7), a world-space TBN from the triangle edges driving the tangent
  normal map (three normalMap math: n=tex·2−1, z=1), and an analytic isotropic
  mip LOD. THREE root-causes chained to get there: (1) 17 sampled textures > the
  spec-default 16 → invalid pipeline → black frame; FIXED by raising the device
  limit to 24 (adapter supports 48 — D-N25). (2) every mip>0 of the array was
  BLACK — three never regenerates storage-texture mips after the compute write;
  FIXED by an explicit backend.generateMipmaps after bake (GOTCHA). (3) the
  anisotropic ray-plane .grad() NaNs on very near trunks (black); REPLACED as
  the default by the NaN-proof analytic LOD (grad kept behind ?nanbark=grad).
  Bisected the whole chain with ?nanbark=const|lN|uv|tex0 (flag scaffolding,
  trimmed to const|lN|grad). VERIFIED: forest-interior + gorge shots render
  detailed lit bark (fissures, species tint, cavity AO) with no black/errors/
  overflow; 112–117 fps @ ~1MP, drawCalls 21, nanite.inst 955k. Deadwood shares
  slice 5 + moss/rot. Diffuse-only (roughness unused, like terrain/rock — a
  spec term is future work). NEXT: N4-C3 second commit — trunk WIND channel
  (Wind.ts 'trunk' math into fetchWorldVert, shared by raster+resolve so they
  stay bit-identical; gate runs --wind 0, then a living-wind eyeball).

- 2026-06-13 (u): **N4-C2 ROCK material landed — first real OBJECTS in the
  nanite world + the resolve's first per-vertex attribute interpolation.** (Opus
  4.8.) PORTED_CLASSES += 'rock', so the cull→raster pipeline processes a second
  material class (~122k boulder/slab/stone instances move from deferred into the
  registry). The big new capability: the resolve now RE-FETCHES the cluster
  triangle and barycentric-interpolates per-vertex attributes (terrain shaded
  from world position alone and never needed this). Path: payload→(instId via
  item.x, ci, localTri = pRaw&127)→makeCtx→fetchWorldVert×3 for the world
  corners→3D barycentric of the reconstructed wp (perspective-correct for free)
  →readVertex×3 for vdata (4×u8 unorm unpack, WorldRegistry pack format) + oct
  normals (instance-yaw'd via instRotateDir), interpolate→ported rockMaterial
  (strata banding vdata.y, lichen vdata.z, dust/streak by upness, moss, AO
  vdata.w→aoNode on the indirect term). Gated on isR so heightfield/terrain
  clusters never enter the explicit-mesh fetch. F9 WALL HIT + FIXED (D-N23): the
  added geometry mega-buffers pushed the fragment stage to 11 storage buffers
  (cap 10); fixed by reading the GI ground height from heightTex (TEXTURE) not
  the height storage buffer — NOT by per-class buffers (the mega-buffers stay
  shared per D-N13). VERIFIED: ?nandbg=cls (NEW matClass tint — terrain green /
  rock red / other blue) shows rocks rastering as coherent scattered red shapes,
  correctly classified; rock pixels read pale grey-brown (145-169) vs terrain
  (86,78,58) — sane rock material, no garbage; drawCalls 21, no WebGPU errors,
  tsc clean. NEXT: N4-C3 (BARK + DEADWOOD — needs per-pool texA/texB textures +
  uv/grad derivatives + the trunk wind channel).

- 2026-06-13 (t): **N4-C1 lighting CLOSED on the energy-correct model;
  pixel-parity with old terrain ABANDONED (user directive, D-N22).** (Opus
  4.8.) After CSM-receive (s) the terrain was still ~broadly off vs ?nanite=0.
  Investigated with a 4-probe workflow + GPU luma/diff: exposure ruled out
  (lockexp pins both to 1.0), normals match, and the real gap is that the OLD
  terrain adds a full env-IBL skylight (scene.environment intensity 1.0) the
  nanite lacks. Tried the "energy-correct" global fix (env-IBL=0 in
  dimAmbientForGI): it drove the old golden hour near-black (luma 28→11), proving
  the env term is load-bearing skylight, not a clean double-count — so I reverted
  that shared-path change. USER CALL: stop chasing equivalence; the nanite uses
  the clean probe-only energy-correct model (sun×CSM + probe-GI + caustics, exact
  three BRDF albedo/π on both terms), accepted as dimmer-than-old by design.
  Net code state: the energy-correct resolve (committed bdb24c7: removed the
  spurious hemisphere ambient, added the missing 1/π on sun+probe) STANDS; SunSky
  is back to original (env-IBL untouched on the shared path); the resolve comment
  now states the deliberate divergence. The ≤0.2% terrain-lighting gate is
  retired (D-N22) — terrain lighting is judged on absolute quality, not diffed.
  ?oldgeo=1 / ?nanshadow=0 remain as scaffolding. Measured signposts kept for the
  record: golden-vista nanite-vs-old mean delta 36→18 (π fix) → diff 33% after
  the env experiment; midday nanite 70 vs old 80 luma (nanite dimmer = missing
  env skylight, by design). NEXT: per the user, move forward — N4-C2 (ROCK
  material + register the rock class so the black slate gains real nanite
  objects), not more lighting diffing.

- 2026-06-13 (s): **N4-C1 CSM sun-shadow receive landed on the nanite
  terrain.** (Opus 4.8.) The dominant ~39% over-bright term (D-N20: "almost
  all of it the missing canopy shadow") is now received. MECHANISM (D-N17 as
  written, no deviation): the resolve references the SAME proven
  CachedCsmShadowNode as a multiplicative factor — exactly how AnalyticLightNode
  consumes it (`colorNode.mul(shadowNode)`, three src line 230) — so OUR
  pcssFilter (blocker search → world-metric penumbra → Vogel PCF) and the
  cloud-shadow gate ride along unchanged. The cascade-select reads
  `shadowPositionWorld`, which ShadowBaseNode sources from
  `material.receivedShadowPositionNode`; the resolve sets that to the per-pixel
  RECONSTRUCTED world position (self-contained Fn mirroring depthNode — NOT a
  fragment closure var, so it builds inside the shadow subgraph cleanly). This
  is why the CsmCached.setup() override (shadowPositionWorld-based linearDepth,
  already 0.02% A/B on the old path) was needed: the fullscreen triangle's
  positionView is a near-plane point that would pin every pixel to cascade 0.
  Threaded shadowRig.csm → buildNaniteFrame → buildNaniteResolve (ResolveWorld.csm,
  base CSMShadowNode type; runtime is the Cached subtype). The resolve is now
  the ONLY consumer keeping the CSM alive in black-slate mode (its updateBefore
  refits cascades + the shadow pass renders — r.shadow.* reappears). FLAGS:
  ?nanshadow=0 (A/B the term), and DISABLE_OLD_GEOMETRY is now ?oldgeo=1
  -overridable (default still fully disabled — a gate harness, not a fallback:
  restores casters + the nanite=0 reference for the lighting gate). VALIDATED:
  black slate renders with no fallback (drawCalls 21, tris 2002, no compile
  errors); with casters (oldgeo=1) the nanshadow ON/OFF diff is 29.65% of px,
  spatially shadow-SHAPED not noise; grass-ablated A/B shows the bare nanite
  terrain gaining clear tree + self shadows under the tod-19 low sun. tsc clean.
  STILL OPEN (finishes N4-C1): ambient/IBL parity (the hand-rolled hemisphere
  ambient vs three's environment IBL diffuse) — the smaller residual; then the
  `--framealign N --wind 0 --lockexp 1` ≤0.2% gate vs ?nanite=0 (oldgeo=1) at
  terrain-dominant framings, accounting for the known honest deltas (CDLOD
  morph at far ridges, skirts).

- 2026-06-13 (r): **Old geometry hard-disabled (D-N21) — black-slate nanite
  build.** (Opus 4.8.) Context: across two sessions the user kept reading
  `?nanite=1` as "nanite isn't active" because the hybrid was STILL drawing the
  old-path trees/grass/water with full wind + lighting (correct by D-N19 — only
  `terrain` is migrated — but indistinguishable to the eye from "old pipeline").
  Two non-actions first, to stop the thrash: (1) REVERTED the uncommitted
  `nanlit=three` resolve rewrite back to 001a5ed (the manual-lit resolve D-N20
  mandates — the three-lighting MeshStandardNodeMaterial path is the exact
  fallback-to-transparent trap D-N20 documents). (2) PROVED the terrain already
  IS nanite, two ways before touching anything: console
  `nanite full-frame: classes [terrain]; suppressed … + terrain tiles/far
  shell` + `visClusters 30628`; and `?nanite=1&nandbg=cov&postmin=1` painting
  every nanite-covered pixel red — the ground went red, the swaying trees did
  NOT (old path, on top). Then per the user's directive: DISABLE_OLD_GEOMETRY
  switch in TerrainScene.ts gates off every old solid-geometry add (tiles/
  shadowproxy/forests/groundring/canopyshell/water/particles), keeping the
  registry producers + environment. RESULT at walk spawn 1280×720: draws
  724→21, tris 12.03M→2,002, fps 31→119; the frame is now ONLY the nanite
  terrain + sky (verified visually — bare slopes, distant peaks, no veg). tsc
  clean. NEXT: with the slate clean, resume N4-C1 proper — CSM shadow-receive
  + ambient/IBL parity on the nanite terrain, then the framealign ≤0.2% gate
  (the gate now compares nanite-only vs a temporarily-restored old frame, or
  shifts to absolute-quality eyeballing since there is no live old path to
  diff against frame-for-frame).

- 2026-06-13 (q): **N4-C1 transparency bug fixed; shading parity still open.**
  (Opus 4.8 continuing Fable's work.) User reported near-camera terrain
  "transparent / dithered out" in the full-frame mode (?nanite=1). Root cause
  (D-N20): Fable's C1 resolve (camera-glued near-plane triangle +
  MeshPhysicalNodeMaterial) failed to compile its lighting and three fell back
  to a node-ignoring material → resolve drew nothing → sky showed through. FIX:
  rewrote NaniteResolve as the proven C0 architecture — clip-space fullscreen
  triangle + plain NodeMaterial + fragmentNode; shading reconstructed in the
  fragment (buildTerrainShading on the depth-reconstructed wp + manual sun
  lambert + sky ambient + probe GI). Verified: coverage map fills the whole
  ground near AND far (bm4 + walk spawn, ~97% covered, orphans 0); terrain
  opaque + forest-coloured; full-frame still active (suppression confirmed:
  39% diff vs ?nanite=0, not 0%). REMOVED Fable's camera-glued geometry +
  syncCamera + the unused nanprobe-resolve-write plumbing + NaniteCommon.invVp
  is now only used by the nanprobe kernel. STILL OPEN (next): CSM shadow
  receive (the ~39% gap is the missing canopy shadow → nanite terrain too
  bright), then ambient/IBL parity, then the framealign ≤0.2% gate; then the
  THREE.TSL "normal not found" warning (cosmetic — buildTerrainShading's
  unused normalNode path references a geometry normal; harmless on the plain
  NodeMaterial). Commits: 6b83289 (Fable WIP checkpoint, preserved per user),
  001a5ed (the fix). Two MISFRAMINGS cost time and are recorded so the next
  context doesn't repeat them: (1) the resolve's debug-red looked "black" in
  the FULL pipeline because post (exposure/aerial/tonemap) darkens it — always
  use ?postmin=1 for coverage/paint probes; (2) the fallback material made
  every paint mode show "lit terrain" (actually old-path trees + sky), which
  read as a shading bug for far too long.

- 2026-06-13 (p): **N4-C0 landed — full-frame integration + an N3-era
  depth-bias bug found and fixed.** `?nanite=1` (no nanitedbg) now renders
  the migrated classes INTO the real frame: NaniteFrame.ts (compute
  scheduling in an engine.post wrapper before postStack.render; TRAA
  jitter MIRROR via TRAANode._jitterIndex + halton(i+1,2/3) on a scratch
  camera), NaniteResolve.ts (fullscreen-tri mesh IN engine.scene,
  renderOrder −1000, fragmentNode flat palette for C0, depthNode = vis
  f32 verbatim, Discard keeps cleared depth), NaniteFetch.ts (makeCtx/
  fetchWorldVert extracted verbatim from the raster — one decode for
  raster+resolve), WorldRegistry class filter (D-N19, ?naniteclasses,
  PORTED_CLASSES=[terrain], migratedMatClass exported), suppression:
  terrain tiles+farShell hidden (+CDLOD update skipped), Forests
  opaque parts[0] tagged at addDraw and hidden per class — casters are
  separate cascade-layer meshes, shadows untouched. probe-nanite-boot
  pins N1 semantics via ?naniteframe=0. THE BUG (gotcha above): water
  z-failed against resolve-written depth everywhere; forensics chain
  (postmin bisect → waterdbg 5/7/8 rungs → nandepth=half liveness →
  storage-buffer probe kernels: scene-depth==vis bit-exact, payload
  triangles healthy+covering, min-keyed writer identity → live weights
  dump: cw=(1376,0,595) vs area2=1972) convicted the biased-weight
  interpolation; fix = UNBIASED weights for cz (coverage keeps the
  biases). Re-proven battery: nanitedbg ✓ pan 0 holes ✓ parity
  silhouettes IDENTICAL (4/0/102/39/13 px) with intersection flips
  ↓~10× (47/10/63/107/112 vs 134–1,099 — SW depth now lands on HW
  depth at intersections) ✓ horizon-nanite: graze shimmer 3 px content
  (15-px raw was ±1-LSB byte noise — probe metric now ignores ≤1 LSB),
  holes/orphans/silhouette 0, graze flips 4,866→480 ✓ boot 289 ms ✓.
  C0 gates: bm1/3/4/7+spawn+aerial boot ✓; water/lake/river/grass/
  crowns depth-compose against nanite terrain ✓ (the lake was THE
  canary); audit orphans 0 in frame mode (~900k covered px) ✓;
  cross-boot framealign+lockexp+wind0 diff 0.06% (floor) ✓. Bisects
  kept (URL-gated): ?nanhw=0, ?nandepth=0|half, ?nandbg=dist,
  ?nanprobe=1 (exact-number kernel), waterdbg 7/8.

- 2026-06-13 (o): **N3 COMPLETE** — C3 landed: grazing-horizon (c) +
  walk-mode near-field (d) gates PASS via NEW tools/probe-horizon-nanite.ts
  (uses main's __laas groundProbe/setPose/settle hooks). graze4km (corner
  +40 m eye down the 5.4 km diagonal, ~0.5° grazing): frame shimmer 0 px
  (bit-stable — NO z-race with full-f32 depth), holes 0, orphans 0,
  silhouette parity vs HW 0 px; flips 4,866 (0.53% — tie-ownership blooms
  at grazing where surfaces run near-coplanar for long stretches; same
  class→invisible, cross-class→counted; silhouette exactness is the gate).
  nearfield (eye ground+0.05 m, terrain through the near plane): shimmer 0,
  holes 0, orphans 0, silhouette 0 px, flips 41, hwTris 12,473 — F10c
  near-crossing→HW verified underfoot IN THE WORLD. Two probe-infra bugs
  found by the gates failing (GOTCHAS): the `< 24` background-sum trap
  (probe-pan's N2 hole gate was VACUOUS — fixed to ≤24 and RE-RUN: 0 holes
  honest), and hwref's freeze latch surviving teleports. probe-parity
  re-run with the honest classifier: silhouette 4/0/102/39/13 px
  (0.0004–0.0111%, gate 0.05%) + flips 134–1,099 (≤0.12%) at
  spawn/bm1/bm3/bm4/bm7 — real nonzero numbers proving the classifier
  measures. Two-phase tie oscillation quantified: 1 px/frame at the
  grazing pose (frame0==frame2 — alternation, not drift); stability gate
  allows ≤8 px. N3 GATES ALL MET: watertight fixed-point raster (C1),
  silhouette ≤0.05% with no structural breaks (C2: 0 px), no z-artifacts
  at 4 km grazing (C3), near-crossing→HW underfoot (C3). Perf: bm3
  2592×1676 depth 2.29 + payload 2.69 ms (ledger row).

- 2026-06-13 (n): N3-C2 landed — SILHOUETTE PARITY GATE PASSED: **0 px
  silhouette diff at ALL 5 framings** (spawn/bm1/bm3/bm4/bm7, 1280×720,
  tol 3/255) — the fixed-point raster is EXACTLY silhouette-equivalent to
  the Metal HW rasterizer, far inside the ≤0.05% gate. Interior
  intersection flips 0.014–0.121% (≤0.2% backstop; D-N15 decomposition).
  NEW src/nanite/NaniteHwRef.ts (`?nanitedbg=hwref`): the registry content
  drawn as classic instanced three draws — CPU mirror of kInstCull's chain
  walk + D-N14 envelope, exact contract transform as Matrix4
  (T·Shear·RotY·S), terrain windows rebuilt from hf.cpuHeights with
  fetchWorldVert's corner tables, CPU-frustum-selected per pose (spawn:
  59,322 windows / 5.8M tris; 110 draws total, ~136 ms build); partition
  tracks the settling walk camera and freezes at rest. `?shade=0` on both
  views = the machine-gate mode (pure class colors). tools/probe-parity.ts
  shoots both views per framing, classifies diff px silhouette-vs-flip,
  writes red/yellow overlay PNGs. DEBUG TRAIL (for the next reference
  build): three findings en route — (1) hwref derivative normals break at
  silhouettes + sub-pixel tris → shading excluded from the machine gate
  (D-N15); (2) scene.background black ≠ the flat view's TRANSPARENT
  discard over the #06080a page background = 12.6k phantom "structural"
  diff px; (3) the always-on DOM fps chip pollutes screenshots (probe
  hides it). bm3/bm4 lambert-mode shots eyeballed clean (human half of
  F12). tsc clean.

- 2026-06-13 (m): N3-C1 landed — FIXED-POINT integer edge functions (N3a).
  SW scanline now snaps verts to a 1/256-px grid (8 subpixel bits, D3D HW
  convention) and decides coverage in exact i32 math: integer edge terms +
  integer top-left rule (bias −1 turns ≥0 into >0 on unowned edges — the
  float −1e-5 bias is gone), depth from exact integer weights, per-pixel
  (kills the float core's 16-step incremental z drift). SW/HW routing moved
  to the UNCLAMPED bbox extent — bounds every edge term < 2^26 (i32-safe by
  construction; before, a screen-spanning tri with a small on-screen sliver
  ran SW with unbounded float terms) — border-clipped giants now route HW
  (hwTris +~3–6%). Snapped-degenerate slivers (area2 ≤ 0) are skipped, which
  also guards the reciprocal the float core divided by blindly. NEW ?audit=1
  oracle (kAudit + nanite.orphans/covered HUD + probe gate + resolve
  discards orphans to background so black-px probes see them): counts
  covered pixels whose payload never matched pass-1 depth — THE consistency
  symptom for any SW/HW pass disagreement. MEASURED (720p spawn/bm3/bm7):
  orphans 0 / coverage within ±4 px of float core (909,027→909,023 etc. —
  subpixel boundary ownership); probe-pan 0 holes; HW ±64-ulp equality
  window RETIRED — 0 orphans at EXACT equality on real HW load (25k tris
  underfoot at walk-mode spawn + 79k at bm3 2592); audit stays as the live
  tripwire for future driver/three divergence. PERF (bm3 2592×1676):
  nanRasterDepth 2.29 ms (float 2.5), nanRasterPayload 2.69 (3.1) — integer
  inner loop ~0.6 ms FASTER. tsc clean; registry/clusterize/registry-gpu/
  boot(293 ms)/nanitedbg×3/pan all green. Tsl grew minI/maxI/toI (toI
  documents WGSL f32→i32 saturation — the off-screen-vert containment).

- 2026-06-13 (l): **N2 COMPLETE** — C4 gate measured (ledger row "N2
  close"). Counts at 2592×1676 (two-phase, real boot): bm1 4,161 / bm3
  94,282 / bm4 89,620 / bm7 65,975 visClusters; steady-state p2 appends
  0.2–21k (clusters occluded only by other phase-2 geometry oscillate
  between phases — bounded, known two-phase property, costs a little
  re-raster). GATE READING: cpu.submit 11–15 → 0.6–0.9 ms on the real
  world (THE binding constraint, collapsed); nanite GPU 2.5–7.4 ms total
  at 4.34 Mpx incl. both raster passes. "visible counts match old path
  ±LOD policy" is met as: geometry SET identical within envelopes
  (verified visually all framings), ring/envelope constants mirror the
  old path by construction (D-N14), occlusion removes only hidden
  geometry (image-identical occl on/off at 5 framings); a tris-vs-tris
  number vs old stats is NOT comparable (old total is card/impostor/
  grass-heavy: 12.6–16.5M). probe-pan zero holes (F13) ✓. 5× stress: caps
  hold, F14 flags fire, self-heals ✓. bm3 old-path slow twice in a row
  (58.3, 66.8) — treat as real bm3 content cost on this branch, not a
  spike; investigate only if it blocks an N-gate. Old-path compute hooks
  (grassRing/vegCull/probeGather) intentionally still tick under the
  debug view (+1–2 ms) — they die at N6/N10, not before.

- 2026-06-12 (k): N2-C3 landed — TWO-PHASE occlusion complete. Phase 1
  records occlusion-only rejects at both levels (rejInst/rejClust, counters
  2/3); phase 2 re-tests vs the fresh HZB with CURRENT matrices (instance
  rejects re-expand through full cluster cull; cluster rejects re-test
  occlusion only) and APPENDS to qRaster — payload indices stable,
  qRaster[0]=(total, phase2base); raster split into depth1/depth2(slot
  base)/hwDepth/payload-full passes; final HZB rebuild closes the frame.
  ?cullfreeze=1 (verified: 180° turn → black void), ?nanitedbg=hzb +
  &hzblevel=N (verified clean mip), ?phase2=0 A/B. tools/probe-pan.ts
  (F13): per-frame hard STRAFE (rotation has no parallax — disocclusion
  needs translation), sky-guarded black-pixel gate. PASS 0 holes 8/8
  frames at 1 m/frame; phase 2 recovered up to 24,815 clusters/frame
  (74,893 sampled total) — the loop is LIVE. FINDING: the phase2=0
  negative control ALSO shows 0 visible holes at this occluder scale —
  HZB footprint max is conservative around narrow trunks and revealed
  slivers stay covered by boundary clusters; two-phase is cheap insurance
  that becomes load-bearing when occluders tighten (N9 foliage walls).
  nanite.p2 HUD counter = appends. ?stress=5 (C4): 4.78M instances bound
  (227 MB registry), pipeline intact at spawn+bm3, F14 verified live
  (rejInst 3.8M > cap → flag fired, image intact, victims self-heal next
  frame — phase 1 re-tests everything); reject caps raised to 1M.
  Gotchas re-fired: same-scope ×2 (HZB reduction ro+rw; kRasterArgs2),
  meter readback before first dispatch. main.ts exposes __laasFly (probe
  camera control; rig overwrites camera.rotation — setPose is the API).

- 2026-06-12 (j): N2-C2 landed (HZB + phase-1 occlusion). NaniteHzb.ts:
  example pyramid on the Option C depth buffer (F19) — half-res L0, packed
  f32 mip chain, per-level kernels; init far-fill = frame-0 pass-through;
  rebuilt after each raster. sphereOccluded prev-VP/prev-camPos (NaniteCam
  prev snapshot + cotHalfFov) with TWO example departures: (1) reduction
  reads the prev level through the SAME rw view (a second ro view of one
  buffer = same-scope violation — N0 gotcha re-fired live); (2) NO Y flip
  in the footprint lookup — the example sources a top-down TEXTURE, our
  depth buffer rows are bottom-up; the mirrored lookup over-culled valley
  framings against the near wall (bm1 131k→179 clusters, sawtooth holes;
  bisected via ?occl=0). makeVisBuffers hoisted from the raster builder
  (hzb→cull→raster cycle). MEASURED cuts (720p, image-identical at all 5
  framings): spawn 153k→108k (1.4×), bm1 131k→3.8k (35×, canyon), bm3
  396k→80k (4.9×), bm4 229k→99k (2.3×), bm7 216k→66k (3.3×); bm1 fps chip
  70→121. Single-phase still (pan disocclusion = C3). USER Q&A: param is
  ?nanitedbg=… on world (clusterdbg = spike) — checkpoint table fixed.

- 2026-06-12 (i): N2-C1 landed (86a12a9): the registry-fed cull chain + the
  spike raster ported onto registry buffers — `?nanite=1&nanitedbg=flat|
  cluster` renders the whole migrated world through cull→Option-C raster→
  flat resolve, replacing the frame render via the engine post slot (old
  pipeline boots/updates untouched; `cluster` = the deferred N1 checkpoint,
  meshlet colors on the real world — VERIFIED visually at spawn). NEW:
  mesh record widened to 16 words (12–15 = mesh-local bounding sphere;
  explicit = cluster-sphere union, heightfield = world grid box — probe
  asserts geometry containment, not sphere-in-sphere: window spheres bulge
  outside the global box by design). Cull = kInstCull (frustum on instance
  world sphere incl. lean operator-norm + swayPad; LOD chain walk; HYBRID
  DRAW ENVELOPE) → 64-cluster chunk expansion (bulk atomicAdd, packed
  uvec2: base 26b | count−1 6b) → kClusterCull (frustum + cone backface,
  yaw-rotated axis, sin-slack 0.25 rad for lean/wind) → qRaster. swayPad
  sourced from Wind.ts term bounds (F6): trees 3.8 m / snags 1.7 / shrubs
  2.4 at strength 1. Raster: makeCtx/fetchWorldVert decode registry blobs;
  heightfield verts from hf.heightTex textureLoad (partial windows, window-
  index×winQuads vertex base); HW big/near-tri passes + resolve ported
  unchanged. COUNTS (1280×720, frustum+cone+envelope, NO occlusion yet):
  spawn 153k / bm1 131k / bm3 396k / bm4 229k / bm7 216k visClusters;
  chunks ≤ 13k; hwTris ≤ 25k. The envelope (chain-tail lodDist = old ring
  edge: trees 496 m = R2_FAR+BAND2, pools clsMaxDist 90–700 m, terrain
  unlimited; D-N14) cut bm3 from 18.6M — without it every r2 ring ran to
  4 km. F16 answered: 25-bit payload itemIdx holds 80× the worst measured
  count. qRaster cap 2M ×8 B = 16 MB (terrain binds LAST → its pushes died
  past the old 524k cap — the N0 overflow lesson re-fired before the
  envelope landed). Probes: registry/registry-gpu/nanite-boot PASS + NEW
  tools/probe-nanitedbg.ts (boot+counters+screenshot+fail-on-overflow).
  tsc clean. Tsl.ts grew localX/wgLinear/texLoadR/uniformMat4/uniformV3/
  uniformF/uniformU/uniformArrV4/dispatchIndirect.

- 2026-06-12 (h): **N1 COMPLETE** — C4 landed (a9cc381): ?nanite=1 builds the
  GeometryRegistry from ALL opaque world pools (src/nanite/WorldRegistry.ts;
  TerrainScene hook inside the veg block — nanite=1 with ablate=veg is a
  no-op by design). GATE PASS (tools/probe-nanite-boot.ts, real boot):
  133 meshes, 355,795 clusters, 1.52M explicit + 33.5M implicit terrain
  tris, 1.07M verts, 955,053 instances bound (207,274 leafy deferred);
  89.9 MB (verts 25.8 + idx 18.3 + clusters 11.4 + inst 34.4); boot add-on
  554–570 ms = readback 6–8 + idF partition 70 + terrain minMax 74 +
  clusterize 308–336 (< 2000 gate) + build 16–18. Pool policy: tree barks
  r0→r1→r2 chains (26/150 m), shrub barks, logs/stumps/branches, boulders/
  slabs/stones r1→r2 (120 m); ferns/flowers + card/leaf parts = 3.10M tris
  deferred to N9 (counted per part); GroundRing clipmap stays bespoke until
  N6/N10 (audit unchanged). Terrain: HeightfieldSource generalized to
  total-quad counts + PARTIAL edge windows (res−1 need not divide; mesh
  record word 10 = quadsX|quadsZ now) — 585² windows at winQuads 7, 98 tris
  avg, 100% full. Scatter readback once at boot (placements static),
  CPU partition by idF preserves buffer order (variation law intact).
  N1 USER CHECKPOINT (?clusterdbg=1 world) DEFERRED to N3: it needs cluster
  colors on world geometry, which only the nanite raster can draw without
  touching the old pipeline (untouchability law wins; cluster quality is
  verified numerically + visually on the spike instead). Hero/R0 bark is
  boot-built in VegLibrary (not background) — the LATE path stays exercised
  by probes, real consumer arrives with background hero refinement at N6.
  swayPad left 0 on trunk channels — N2 must source real amplitudes from
  Wind.ts before occlusion tests (F6).

- 2026-06-12 (g): N1 C3 landed (81dbae0): src/nanite/GeometryRegistry.ts —
  the content-contract entry point (registerMesh explicit|heightfield,
  registerLod chain, bindInstances CPU|GPU-scatter, build()/flush() with a
  late-registration budget; capacity overflow THROWS pre-mutation, F14).
  Packed per D-N13; GPU instance streams land via per-stream copy kernels;
  late uploads via addUpdateRange (WebGPUAttributeUtils honors updateRanges —
  verified in source). TSL readVertex/readCluster/readMesh + oct/f16 codecs
  with exact CPU mirrors. VERIFIED: tools/probe-registry.ts (node — pos/
  vdata/sphere/cos BITS exact, oct16 normal 5.4e-5, uv f16 2.4e-4, hf sphere
  containment 4.9e-7, mesh table/LOD/instances exact, late flush + overflow
  throw); tools/probe-registry-gpu.ts (headless, ?scene=rasterspike&regtest=1
  — TSL decode vs CPU mirrors maxErr 2.6e-7, exactFails 0, copy kernel +
  instanceMesh exact). Tsl.ts grew bcF2U/bcU2F/unpackHalfU/unpackSnormU/
  elemU/elemUW/sVec4Views/dispatch/readBuffer. tsc clean; spike scene
  untouched without &regtest.

- 2026-06-12 (f): N1 C1+C2 landed. C1 (46689b4): src/nanite/Tsl.ts typed
  helper layer (one documented cast per @types gap: sUvec2/sUvec4RO/uv2/
  minU/maxU/aLoadU/toF/loopU/loopI/returnIf/sU32Views); SpikeRaster
  refactored onto it, casts 166→99 (bool-comparison and .select casts were
  pure noise — gone; remainder is repo-standard TSLTypes narrowing).
  C2 (64184ac): src/nanite/Clusterize.ts — greedy adjacency clusterizer
  (typed-array hash adjacency, centroid-priority heap, SEED CONTINUITY from
  the previous frontier + underfull refill = the fill-quality fix: avg
  126.9/128, 100% full on hero rock, no fragments) + tools/probe-clusterize.ts
  (node-only invariant checks: permutation/coverage/sphere/cone — all hold).
  Throughput 4.1 Mtri/s; REAL all-pools source ≈ 3–4M tris (the 10–20M doc
  figure counted instance multiplicity) → ~1 s boot, inside the <2 s gate.

- 2026-06-12 (e): USER-REPORTED: ?packing=a failed BindGroupLayout creation —
  11 storage buffers in one compute stage (the F9 10/stage adapter limit,
  first real bite): the overflow fix's counters binding pushed Option A's
  single kernel (both vis atomics + hwQueue) over. Fix: workQueue entry 0 is
  now the RESERVED count slot (items at [1..n]) — the guard reads the queue
  that every kernel already binds, zero extra bindings. Budgets now: A-raster
  10/10, C-depth 9, C-payload 9. Lesson for N1's packed layout: counters that
  gate a queue belong IN the queue buffer. Also user-confirmed expectation:
  spike has NO LOD (full-detail clusters at any distance; stable hash colors
  are the correctness signal) — discrete ring sets arrive with pool
  migration (N1–N6), continuous DAG refinement + ?clusterdbg=lod at N8.
- 2026-06-12 (d): USER-REPORTED (live fly-out): terrain vanished + rock holes
  beyond the standing framing — the work queue overflowed AGAIN at full-field
  visibility (78,464 items vs the 65,535 single-dim dispatch cap; content had
  been sized to the standing pose's partial frustum, which was a dodge).
  REAL FIX: indirect dispatch 2D-splits at 65535 workgroups/dim, kernels
  linearize via workgroupId.y·65535+workgroupId.x + localId.x, partial-last-
  row guard via TSL Return(); WORK_CAP now memory-bound (262144 ≈ 2 MB).
  Two new gotchas hit en route: (1) binding the SAME buffer as atomic AND
  read-only views in ONE dispatch is a WebGPU same-scope usage violation
  (kArgs now uses the atomic view alone); (2) a bare JS `return` inside an
  If() closure builds nothing — TSL Return() is required for a WGSL return.
  Verified: far view complete at 78,464 items / ~105 fps; standing view
  unchanged; gate-viewport medians 6.2 ms (was 6.0 — noise). sw=0 unaffected
  throughout (no queue), which is what localized it.
- 2026-06-12 (c): **N0 COMPLETE — GO.** Branch baselines captured (ledger; bm3
  outlier flagged). Spike shipped: `?scene=rasterspike` (`&sw=0/1`,
  `&packing=a|c`, `&clusterdbg=1`) — src/nanite/SpikeContent.ts (3 rock
  variants + 256² heightfield tile, 128-tri clusters, IMPLICIT terrain
  clusters per F4), src/nanite/SpikeRaster.ts (clear → per-instance cluster
  cull → indirect 1-dim dispatch → Option C two-pass SW raster → HW big/near-
  tri queue rendered as TWO vis-buffer-writing fragment passes → fullscreen
  resolve w/ face normals + cluster tint), ThreePatches.installFragmentStorageWrites
  (opt-in defeat of three's read-only-storage-outside-compute, BOTH the WGSL
  access and the bind-group-layout sides — markFragmentWritable(attr)).
  GATE NUMBERS (table above): cpu.submit 0.2–0.4 ms ≈ dispatch overhead ✓
  (vs 10.5–14.8 ms world pipeline — THE binding constraint, proven
  addressable); Option C full-precision tax = ~1.0 ms over Option A (+20%)
  at full viewport ✓ C stays primary (D-N5 confirmed with data); fragment
  storage writes verified live ✓ (HW path writes the same vis buffer — one
  resolve, one convention). vs the IDEAL 5-draw instanced HW reference the
  SW path reads 6.0 vs 2.8 ms — expected: spike tris are 2–30 px (HW comfort
  zone) with perfect vertex-cache reuse and zero overdraw; the real
  replacement target (905 draws, alpha overdraw, 12 ms submit) is what
  N3–N6 measure. Known spike debts → N1/N2: WORK_CAP 65535 single-dim
  indirect dispatch (overflow DROPS silently — hit it: 65,678 items at the
  first framing made terrain vanish; HUD spike.work + console warn added;
  proper queue scaling is N2), terrain-instance serial cluster loop in cull
  (~0.5 ms tail — two-level cull at N2), HW payload equality needs ±64-ulp
  tolerance (cross-RENDER-pipeline FMA divergence; SW compute passes are
  exact — N3 fixed-point kills the class), `as unknown` casts to be
  consolidated into typed TSL helpers at N1 (user note 2026-06-12).
- 2026-06-12 (b): ADVERSARIAL REVIEW PASS done (fresh context): example source
  re-read line-by-line (3 misreadings corrected), Karis/Epic/community literature
  verified (two-phase, DAG cut machinery, foliage, material binning), adapter
  probed (10 storage buffers/stage; 4 GiB buffers; subgroups present; no 64-bit
  atomics), payload math redone from STATUS scene numbers. 19 findings (F1–F19),
  4 blockers, all folded into the design sections; decisions D-N5..D-N10 added.
  Verdict: GO for N0 with amendments. No implementation yet.
- 2026-06-12 (a): Branch + this plan created. No implementation yet.

## NEXT ACTIONS

### ⏯ ACTIVE: N8 LOD DAG — PULLED FORWARD (user directive 2026-06-13: "pull N8 forward to a logical point")

WHY: the D-N29 shadow perf-engine's biggest lever (S4 — cast shadows from a caster LOD
COARSER than the camera view) NEEDS continuous cluster LOD = the N8 DAG. Rather than build
shadow layers S0–S3 on today's discrete-LOD geometry and redo them for the DAG, build the
DAG foundation FIRST, then drop the perf engine on top. The dominant nanite shadow caster
TODAY is the TRUNK (bark, trunk-wind channel) + terrain/rock — trunks are EXPLICIT meshes,
so an explicit-mesh DAG cheapens the CURRENT shadow cost directly AND sets up N9 foliage.

LOGICAL POINT (the stopping milestone): continuous-LOD DAG WORKING on the currently-
migrated EXPLICIT opaque meshes (rock, bark, deadwood) — boundary-locked QEM build +
hierarchical runtime cut + crack-free + no-pop; continuous-zoom probe green; `?clusterdbg=
lod` heatmap; boot-budget measured (the N8 gate per the spec). DEFERRED past this point:
foliage AGGREGATES (N9 leaf-removal area-preservation).
**TERRAIN-DAG IS COMMITTED, NOT OPTIONAL** (user directive 2026-06-13: "terrain will ALSO
be going through the dags … the terrain itself … various repr" — and the PATH UNIFICATION
audit always said so: "Terrain CDLOD … far shell folds into coarse DAG levels"). The old
"fold in IF clean / DEFER" wording was a wrong hedge — terrain gets the SAME continuous cut
as the explicit classes. Its construction is heightfield-NATIVE (F4: no baked verts), built
in N8-D2/D3 — see the terrain-DAG note in NEXT ACTIONS for the approach fork.

SEQUENCING: N6 (migrate remaining opaque pools) + N7 (hybrid close) STAY after this — the
DAG applies to whatever is registered, so N6's later pools get DAG'd when registered. After
the DAG logical point → resume the shadow stack: S4 (DAG-decoupled caster LOD) becomes
buildable; S0 (half-res sample) / S1 (wind-freeze+split) / S2 (tighter cull+stagger) / S3
(clipmap resolution) layer on; S5 (capsule-SDF+contact) is the beauty ceiling. The shadow
research + D-N29 STAND; only the ORDER changed (S0–S5 follow the DAG, not precede it).

IMPLEMENTATION: follow "### DAG (N8) — implementation-ready spec" (Technical design notes)
+ the phase-table N8 row. Build-cost budget is first-class (F15: time-slice if > ~2 s).
Do NOT re-plan from scratch.

CHUNK PLAN (to the logical point):
- ~~N8-D0~~ **DONE** (log af; D-N30) — BuildDag.ts + tools/probe-dag.ts; all invariants
  green on rock (closed) / bark-beech (open tube) / deadwood-snag; deterministic; F15
  FIRED (0.16 Mtri/s → 1.52M ≈ 9.3 s, over the ~2 s target) → D-N30 background build.
  Original spec for the record:
  DAG BUILD (CPU, new module e.g. BuildDag.ts): given LOD0 BuiltClusters
  (Clusterize.ts), build levels k→k+1: cluster adjacency (shared-edge) → graph-partition
  into groups of 8–32 (recursive boundary-min bisection — METIS substitute, zeux) → weld +
  merge group tris → LOCK group-boundary verts → hand-rolled QEM edge-collapse simplify
  interior to ~50% → re-clusterize the soup into 4–16 parents (reuse clusterize()) → own/
  parent (error, sphere) pairs with containment + max-monotonicity + sibling-shared parent
  pair (EXACT equality); stuck-fallback (<~15% reduction → stop that mesh; multiple roots
  legal, parentError=∞). Output: extended per-cluster set across LOD levels + cut metadata.
  GATE: builds on rock/bark/deadwood; per-level stats (tri reduction, stuck count); boot-
  budget measured (F15); deterministic by seed. Validate via node probe tools/probe-dag.ts
  (no GPU yet — verify monotonicity, containment, sibling-pair equality, no orphan errors).
- ~~N8-D1a/b/c~~ **DONE** (log ag; 48a1ccd + b635b05; D-N31) — parallel 10-f32 DAG buffer
  + attachDag (repoint mesh at DAG range, retire chain) + the per-cluster screen-error cut
  in kClusterCull (FLAT, not the hierarchical traversal — D-N31: semantics vs pruning) +
  ?loderr/setTau + ?nanitedag=rock|bark|deadwood|all SYNC boot wiring. GATE green on rock
  (tools/probe-zoom.ts: τ-sweep monotonic 63→124, smooth zoom, watertight, τ=1 pixel-match,
  shadows-ON correct). +1 storage = 8/10 (F9). nanite.dagClusters counter for HUD/gate.
- N8-D1d/e — CLOSE N8-D1: (PRE-REQ DONE — the envelope + cut error-scale bug from the user
  re-test is FIXED, log ah / D-N33; bark + deadwood DAG confirmed working live under
  ?nanitedag=all.) ~~(d) move the DAG build to a Worker (D-N30)~~ **DONE for terrain** (D-N37):
  DagWorker.worker.ts (three-free, prod bundles a standalone 19.6 kB chunk) + DagCache.ts
  (IndexedDB, seed+gridN key). Terrain is DAG-ONLY — boot renders the DAG from frame 1, the build
  AWAITED (no window fallback per the user directive), and a persistent cache makes that instant
  after the first build ([worker] 911 ms → [cache] 21 ms @256). Explicit pools still build sync
  (<1 s). NOTE: the "progressive / swap after boot" idea was tried + REJECTED — see D-N37.
  (e) bark is the heavy class, 162k trees → WATCH
  the flat-cut cull-dispatch volume (?nanitedag=all near a forest = 0.14M cl / 16 ms occl-on
  measured; if cull dispatch is the bottleneck, D-N31's hierarchical-traversal pruning layer
  is the lever). + MIN-SCREEN-SIZE cull primitive **BUILT** (gated `?nanitemin=<px>`, default
  0; per-cluster sub-pixel drop in kClusterCull) — but the UNBOUNDED envelope it was meant to
  unlock is **NOT a size flag** (D-N35, re-validated 3.75M cl / 100 ms occl-ON): retiring the
  impostor far-field needs HIERARCHICAL instance culling (cull spatial GROUPS of distant
  instances → O(regions) not O(instances)) — its own milestone; impostors STAY for now. STILL
  TODO for D1e: + perf ledger row (cull dispatch, qRaster live, boot budget) vs pre-DAG; +
  ?clusterdbg=lod heatmap (tint by ownErr coarseness — needs gpu.dag in the resolve OR a
  cull-side level write; check resolve storage budget first); + USER CHECKPOINT (continuous
  zoom in Chrome). (d) the Worker is NOT urgent for explicit classes (small meshes build sync
  in <1 s) — it's really a TERRAIN prereq (4096² ≈ 5 min, log D2b); do it with D2b.
- N8-D2 — TERRAIN DAG (COMMITTED, D-N32): a heightfield-NATIVE adaptive builder (RTIN /
  restricted right-triangle quadtree, Mapbox `martini`-class — NOT BuildDag's iterative QEM,
  wrong tool for a grid). O(n) bottom-up vertical-error pyramid; aggressive flat decimation
  (user: plains SIGNIFICANTLY fewer tris — flat → a few big right-triangles, cliffs dense);
  vertices stay on-grid → positions reconstruct from the heights buffer (F4 preserved, store
  compact connectivity); emits the SAME cut metadata (own/parent error+sphere) → the parallel
  buffer → kClusterCull (unified runtime, specialized build). Crack-free via the restricted-
  quadtree forced-split rule (+ cross-tile for the 4 km field). Node-test the error pyramid +
  cut like probe-dag/probe-dagpack BEFORE GPU. Steps:
  - ~~(D2a) BuildHeightDag builder + node probe~~ **DONE** (log ai; D-N34; martini error
    pyramid × BuildDag locked-cluster-DAG via gridEndpoint/gridErrAt/levelBudget + 3 manifold
    guards). probe-heightdag green: crack-free cut (W), on-grid F4 (G), adaptive (3.6× cliffs),
    deterministic; rock byte-identical (gridEndpoint default false). Construction approach NOTE
    corrected vs the line above: it is martini-error-metric × BuildDag-scaffolding, NOT pure
    martini getMesh — the flat per-cluster cut (D-N31) forces locked-cluster boundaries; see
    D-N34 for why pure RTIN (ROAM per-frame traversal) can't ride the flat cut crack-free.
  - ~~(D2b) GPU WIRING — register terrain as DAG'd + the indexed decode~~ **DONE** (log/D-N36;
    `?nanitedterrain=<gridN>`). The grid-coord-INDEXED decode (NaniteFetch isDAG branch: word0 =
    packed texel coord → texLoadR + world XZ), lean registerHeightDag + attachHeightDag, and the
    2^k subsample→texel-coord remap (clamp res-1; texel cell/origin) all landed + validated
    headlessly (probe-dterrain @256/512: decode A/B, crack-free, adaptive tint vs uniform window,
    cut live). BUILD SPEED — parts 1+2 DONE (bit-identical 2.6×, 0.055→0.143 Mtri/s; the typed-
    array `poolVerts/poolIdx` pool + the "37 min = __name artifact" debunk are in those logs).
    REMAINING before DAG terrain is the DEFAULT (not just a flag): (i) the D-N30 **Worker** — the
    full-res 4096² build (1 m cells = near-camera parity with the window grid) is ~5 min sync, so
    it MUST run off the boot path (single background Worker + a deterministic-seed cache); gridN-
    subsample is validation-only / a fallback. (ii) optional memory: a stride-1 terrain vertex
    buffer (the DAG verts waste 5/6 words today). This is now THE same blocker as D1d.
  - (D2c) perf ledger row vs pre-DAG (pre-DAG terrain = 33M-tri uniform windows → measure the
    adaptive draw-tri reduction; the `?nandbg=cluster` A/B already shows the qualitative win —
    DAG sheds clusters near→far while the window grid is uniform-dense everywhere). Plus the
    carried D1e items: perf ledger, ?clusterdbg=lod heatmap, USER CHECKPOINT (continuous zoom on
    hero rock/tree + terrain). THEN shadows S4.

  STAGE-2 STREAMING STATUS (D-N39, the path to full-res-default — supersedes the D2b "(i) Worker
  / (ii) stride-1" remaining-list above with a concrete increment chain):
  - ~~2a pool+evict~~ DONE (aj) · ~~2b-1 pool→GPU~~ DONE (ak) · ~~2b-2 static clipmap~~ DONE
    (al math / am GPU) · ~~2b-3 camera-following streamer~~ DONE (an) · ~~2b-3 fix lazy-evict~~
    DONE (ao) · ~~#32 DagWorker POOL (concurrent bakes)~~ **DONE (ap)** — `?nanitedclip=1` streams
    the 1 m detail with the live camera; resident≡desired, bounded, no-leak, re-centers exactly;
    bakes 4-wide so cold boot 7717→2211 ms (3.49×) + detail climbs ~pool× faster (probe-stream +
    probe-streammove green).
  - ~~#29 2d SKIRTS (crack-free inter-level seams)~~ **DONE (aq)** — self-contained always-on
    perimeter skirt per tile (TOP verts on the surface edge + BOTTOM verts dropped, 3-bit depth-level
    code in word0 bits 13-15, decoded at the one NaniteFetch fetch site, DOUBLE-SIDED for the
    winding-based SW-raster cull). Depth model is LINEAR `24 + 12·level` m (NOT ∝2^k — the cracks
    SATURATE at 18→48 m on these 1500 m cliffs; a geometric proof tools/probe-skirtgap.ts samples the
    real field via groundProbe and confirms every transition seals, worst margin +6.1 m). `?nanitedskirt`
    default ON; boot +832 cl/+106 k tris (cap auto-grows), no wall artifacts, all probes green.
  - ~~2e part 1 — THE FLIP (full-res clip-DAG terrain DEFAULT ON, window → `?nanitedterrain=0`)~~
    **DONE (ar)** — bare `?nanite=1` boots the camera-following 1 m clipmap (52 tiles / 34230 cl / 4.14 M
    tris); window grid retired to the opt-out; probe-flip PASS. Exposed the real cost the stride-1 buffer
    targets: **terrain verts 208.8 MB** (96 pool slots × v95016 × 6 words, only word0 used).
  - ~~2e part 2: stride-1 terrain vertex buffer~~ **DONE (as)** — dedicated `gpu.hfVerts` (1 u32/vert);
    terrain verts 208.8 → 34.8 MB (−174 MB). Exposed + fixed a caustic-wash regression (adding hfVerts to
    the resolve breached the 10-buffer Metal ceiling → mis-bound caustic waterY; fix = `bindHfVerts=false`
    in the resolve, which never needs the terrain buffer). All probes green. **STAGE 2e COMPLETE — the
    "boot only to dag" mandate is met: bare `?nanite=1` boots the full-res clip-streamed DAG terrain by
    default, lean stride-1 verts, no window fallback (window survives only as `?nanitedterrain=0`).**
  - **D2c perf-ledger — DONE (probe-perfledger.ts), with an HONEST correction.** Default clip-DAG vs
    `?nanitedterrain=0` window grid at the spawn (veg+rock identical ⇒ counter delta = terrain): DAG
    visClusters ×0.83 (fewer — better cluster culling), hwTris ×1.77 (MORE near big-tris), regTrisK ×9.84
    (DAG stores adaptive geo; window is implicit). **The naive "DAG = fewer terrain tris" was WRONG:** both
    paths are ~1 m full-res, so the DAG is NOT a same-fidelity draw-cost reduction — it trades a bit more
    near big-tri raster for continuous NO-POP LOD + adaptive flat-decimation + BOUNDED streaming memory
    (34.8 MB fixed vs the window grid's implicit-but-unbounded) + the shadow caster-LOD foundation (S4).
    The real fps win is S4, not terrain draw count. (probe-perfledger's vista row is stale under `?freeze`
    — the per-frame cull doesn't re-run; the qualitative far-shed is the `?nandbg=cluster` A/B.)
    BLOCKED: `?clusterdbg=lod` heatmap via the resolve is OFF THE TABLE — it needs gpu.dag (or a level
    write) in the resolve, but the resolve already sits at the 10-buffer Metal storage ceiling (see log as;
    that ceiling is exactly what the stride-1 hfVerts add breached). A lod heatmap must ride a cull-side
    write or a separate debug pass, not the resolve.
  - **NEXT**: **shadows** — the D-N29 perf engine. Measurement reshaped it (see the plan block below).
    S0 DONE (log au — half-res PCSS + bilateral, sample ~halved, real win). S4 + S2-OCCL TRIED but BOTH
    MEASURED WEAK (logs av/aw — LOD coarsening ≤12% since casters are mostly non-DAG; occlusion ~0 since a
    high sun barely self-occludes + CsmCached starves the prev-HZB; ?shadowoccl default off). Two findings
    prove the 38× shadow/camera disparity is WIDE COVERAGE × 4 CASCADES × fine geo, not LOD/occlusion. NOW
    (NEEDS USER DIRECTION before building — a real far-shadow quality/perf tradeoff): **S3** — collapse the
    4 cascades into ONE screen-density shadow CLIPMAP + far-caster PROXIES/impostors. Also open: S1 (WPO-
    freeze / static-dynamic — fixes stale static wind shadows). Also pending: the carried D1e USER CHECKPOINT
    (continuous zoom in Chrome on hero rock/tree + terrain — USER-PRESENT). Optional polish:
    2b-4 always-resident coarse base (teleport no-hole; DOWNGRADED). The paramless-`?nanite=1` default-on
    (retire the `?nanite=0` opt-out) is a SEPARATE later endgame flip — leave for a user-present session
    (it changes the bare-URL boot for every debug scene). OLD locked 2d spec (kept for the record).
    **AS-BUILT DELTAS from the locked spec (refined during execution — see aq):** (1) encoding uses a
    3-BIT depth-level CODE in bits 13-15 (not the single bit-15 flag) so the depth can vary per level;
    (2) depth is LINEAR `24 + 12·level` m, NOT `SKIRT_DEPTH·2^level` — the geometric proof showed the
    cracks SATURATE (18→48 m over levels), so ∝2^k starves the fine transitions and over-covers the
    coarse (walls); (3) skirts are DOUBLE-SIDED (the SW raster culls by winding); (4) NO manual cap
    bump was needed — the boot pool-cap measure already absorbs the skirt add. Original text follows:
    Adjacent clipmap levels ABUT at 2× stride (the hollow drops a coarse tile only when FULLY
    inside the finer extent) ⇒ the fine edge has 2× the coarse edge's verts on the SAME line ⇒
    T-junction cracks that show SKY (coarse terrain there was hollowed). Holes + slow-pop are
    solved (ao + ap), so seams are the last visible-correctness blocker. **LOCKED, DE-RISKED design
    (investigated 06-14, ready to execute):**
      · buildDag ALREADY pins every tile perimeter at base stride at ALL LODs (border edges are used
        by 1 tri ⇒ "boundary" ⇒ locked; a border vert is always the collapse TARGET, never removed —
        BuildDag.ts:557). So the surface edge passes through every base-stride perimeter vert at every
        cut → no buildDag change needed; the residual crack is purely the inter-level 2× mismatch.
      · GPU position is derived PURELY from word0's packed (gx,gz) → height fetch (NaniteFetch.ts
        `fetchWorldVert`, the isHF&&isDAG branch, ~L214-244, the ONE decode site for raster+shadow+hzb).
        ⇒ a skirt cluster can be FULLY SELF-CONTAINED: its own TOP verts at the perimeter (gx,gz)
        render exactly on the surface edge; its own BOTTOM verts (same gx,gz, flagged) drop by
        skirtDepth. No threading a flag through buildDag's QEM.
      · ENCODING: word0 = (gx&0xffff)|(gz<<16); gx,gz ≤4096 use 13 bits ⇒ bit 15 FREE = SKIRT flag.
        Decode: `isSkirt=(w>>15)&1; sx=w&0x7fff; sz=(w>>16)&0xffff; h -= f32(isSkirt)*SKIRT_DEPTH`
        (subtract AFTER micro-disp so the curtain sits below the final surface). buildTerrainTile's
        global remap must EXTRACT bit15, mask it off before `localGx=p&0x7fff`, then re-OR it onto the
        global packed coord (texX&0x7fff | flag | texZ<<16).
      · BUILD (buildHeightDag, opt-in `skirtDepth?:number` in HeightDagOpts; 0/undef = OFF so the
        headless probe + uniform T×T path stay unchanged): enumerate the base-stride perimeter loop
        (4·gridN local coords 0..gridN around the ring), emit n TOP + n BOTTOM verts + n curtain quads
        (2 tris each, OUTWARD winding), split into ceil(2n_tris/128) ALWAYS-ON clusters: level 0,
        ownError 0, parentError +∞ (→ DAG_ROOT_PARENT_ERR), groupAsInput/Parent −1, sphere = the
        perimeter-arc+drop bound, cone set to NEVER backface-cull (verify the ccos convention in
        NaniteCull first — a vertical curtain must not be cone-culled). ownError0+parent∞ ⇒ drawn at
        every cut, frustum/occlusion-culled normally.
      · CAPS: bump the streamer pool vert/tri/cluster cap basis by the skirt's add (~4·gridN verts,
        ~8·gridN tris, ~⌈8·gridN/128⌉ clusters per tile) so attach never overflows.
      · TOGGLE `?nanitedskirt=0/1` (default ON) for a same-pose A/B; validate with tools/probe-seams.ts
        (already written — boots ablated* clip mode, grazing vistas, screenshots; *NOTE the nanite
        engine is built INSIDE `!ablate.has('veg')` so do NOT ablate veg — ablate grass/water/shell/
        particles/caustics only). Skirt OFF = sky cracks along ring boundaries; ON = sealed.
      · COHERENT change-set (cannot split — the flag bit corrupts the old decode until NaniteFetch
        masks it): buildHeightDag (gen) + TerrainStreamer.buildTerrainTile (remap preserves flag) +
        NaniteFetch (decode mask + drop) + WorldRegistry (cap bump + toggle plumb) + probe. tsc +
        probe-stream + probe-streammove + probe-seams green, then commit with before/after shots.
  - THEN 2e stride-1 terrain vertex buffer (6×→1× vert mem, ~100 MB→~17 MB) + FLIP the full-res DAG
    terrain default ON (retire `?nanitedterrain=0` window to the explicit opt-out) = the "boot only
    to dag" mandate.
  - POLISH (after default-on): 2b-4 always-resident coarse BASE — lazy+footprint-evict (ao) already
    keeps the old coarse covering through the common teleport; only the EXTREME corner→corner jump
    (beyond the lingering ring's reach) still needs a pinned non-churning base + a revived minimal
    suppression (kClusterCull skips base clusters a finer resident ring covers — the deleted 2c,
    now justified by a FIXED base under the camera-centered finer levels). See D-N39 tail (a).

CURRENT INFRA (read 2026-06-13): Clusterize.ts → BuiltClusters {indices (permuted, cluster
tris contiguous), sphere 4f32/cluster, cone 4f32, triStart, triCount} — greedy ≤128-tri,
shared-edge adjacency growth, pure CPU typed-array (node-runnable for probes). GeometryRegistry
packs CLUSTER_WORDS=8 per cluster (sphere 4 + coneOct 1 + coneCos 1 + triStart 1 + (triCount
u8 | flags u8 | meshId u16) 1) + a MESH table of 12 words incl. lodNext/lodDist = TODAY'S
DISCRETE per-mesh LOD (rings as discrete cluster sets — the DAG's continuous per-cluster cut
SUPERSEDES this). DAG metadata won't fit the full 8-word cluster rec → parallel buffer (D1).

DONE: N8-D0 (log af). DagBuild output per cluster = geometric sphere/cone (cull, from
clusterize) + (ownError, ownSphere) + (parentError, parentSphere) + level + groupAsInput/
groupAsParent linkage; DagBuild also carries the grown vertex pool (verts/vertStride/indices)
and groups[] (inputs/parents/error/sphere per group). D1 packs ownErr f32 + ownSphere 4f32 +
parentErr f32 + parentSphere 4f32 (=10 words) into the parallel per-cluster buffer + appends
the DAG's higher-LOD verts/indices/cluster-recs to the registry mega-buffers, then drives the
hierarchical cut in NaniteCull. NOTE the build cost (D-N30): D1 must invoke buildDag OFF the
boot critical path (background Worker preferred — buildDag is three-free, typed-arrays in/out).
NOW BUILDING: N8-D1d/e (Worker DAG build per D-N30 + bark/deadwood DAG + perf ledger +
?clusterdbg=lod + USER CHECKPOINT). D1a/b/c DONE (log ag): the GPU continuous-LOD cut is
LIVE + gate-green on rock — the N8 logical point is reached for the rock class; D1d/e
generalise it to all explicit classes off the boot critical path, then N8-D2 + shadows S4.

---

### SHADOW/LIGHTING PERF RETHINK (research COMPLETE 2026-06-13 → D-N29; resumes after the DAG)

> USER DIRECTIVE (verbatim intent): R1-cached nanite shadows are STILL
> unacceptable — static ~70 fps, MOVING drops to **30 fps**. Baseline WITHOUT
> shadows on this machine = **115–120 fps**. GOAL: **fantastic-looking shadows at
> 100+ fps** (so a ≤~15–20 fps hit). "Complete rethink — explore all sorts of
> nanite-style shadow/lighting solutions via internet search. Is CSM our best
> option? Is it even the right option? What alternatives have a great balance of
> quality and performance?" Research FIRST, then decide (new D-N entry), then build.

MEASURED BOTTLENECK (established R0/R1 — do NOT re-measure): the moving cost is the
per-frame SHADOW GEOMETRY RASTER. R1 caches per cascade so a STATIC camera re-rasters
nothing (nanRasterDepth == nanRasterPayload, camera-only → ~0 shadow cost, 70 fps).
A MOVING camera re-rasters cascade 0 every frame + far cascades on the [1,2,3,6]
cadence ≈ 35% of R0's 1.44M clusters/frame ≈ ~10 ms+ of SW compute raster. The PCSS
SAMPLE in the resolve is a FIXED per-pixel cost (same static or moving) — NOT the
moving bottleneck. So the problem = re-rasterizing shadow geometry every frame as the
camera moves. Caching at per-cascade granularity (R1) is too coarse to fix it.

SCENE REALITY (USER CORRECTION 2026-06-14 — do NOT call this world "static"; that
framing is WRONG and was rejected): there are **162k trees**, and EVERY in-view
(near-camera) tree SWAYS in wind → its shadow-caster geometry DEFORMS every frame, in
the near field where shadow quality matters most. The wind-moved share is GROWING, not
shrinking: once N8's DAG cuts terrain tri counts, vegetation becomes the DOMINANT share
of shadow-casting geometry. So caching/baking helps ONLY the rigid remainder (terrain,
rock, and the far field beyond the ~380–480 m wind fade); the wind-animated casters
MUST re-rasterize every frame BY DEFINITION — no cache (R1 per-cascade, VSM per-page,
or bake) touches them. THE REAL PROBLEM is therefore NOT "exploit staticness" — it is
**cheaply shadowing a large, growing set of dynamically-deforming foliage at 100+ fps**.
The lever is the shadow REPRESENTATION + LOD of the DYNAMIC casters (crude per-tree
shadow proxies/capsules/billboards, shadow-LOD decoupled far below camera-LOD since
penumbra hides silhouette error, coarse moving primitives in an SDF/voxel field that
update by transform not per-vertex), NOT caching. R1's per-cascade cache was real but
addresses the wrong (rigid) half. Wind fade still bounds the dynamic set to <~480 m,
so a static-far / dynamic-near SPLIT is valid — but the dynamic-near half is BIG and is
the entire cost; do not design as if it were a small overlay.

RESEARCH FINDINGS (2026-06-13 — 4 parallel cited research threads: A production
foliage-shadow practice + caster representation/LOD, B Virtual Shadow Maps deep-dive,
C SDF/voxel/SW-raytraced shadows, D static/dynamic split + unified shadow+GI. Full
cited docs in session transcript; verdicts below. Recommendation PROVISIONAL pending
user sign-off, then promote to D-N29 + implement.)

CROSS-CUTTING CONCLUSIONS (all 4 threads independently converged):
 1. **Wind casters MUST re-raster every frame — NO cache/bake touches them.** Epic's
    own VSM docs, verbatim: WPO/skeletal geometry "always invalidates cached pages
    every frame." Confirmed by all threads. So VSM page-caching (its headline win) is
    DEAD WEIGHT for our dominant cost; R1's per-cascade cache likewise only ever helped
    the rigid half. This was the right read of SCENE REALITY.
 2. **The production answer is NOT a cleverer cache — it is a DISTANCE-REGIME SPLIT of
    the caster representation.** Near (0–~150 m): full sway, coarsen geometry MODESTLY
    (~2–4× via a coarser DAG cut; penumbra hides it). Mid (~150–480 m): coarsen HARD
    (~8–16× — UE ships a 300k-tri tree casting from a ~30k proxy) AND push the shadow-
    pass WPO-freeze distance MUCH closer than the view's wind fade → frozen = rigid =
    cacheable. Far (>480 m, our wind already faded by design): fully rigid, cached,
    coarsest DAG / impostor caster. (SpeedTree GPU Gems 3 ch.4 ships exactly this:
    cascade 0 every frame full geo; cascade 2 every 2nd frame drops fronds; cascade 3
    every 4th frame leaves-only — "shadows move realistically as the tree sways.")
 3. **Unified shadow+GI CANNOT be the primary sun shadow.** Decisive: UE5 Lumen (best-
    funded unified GI) deliberately keeps a SEPARATE Virtual Shadow Map for the direct
    sun. VXGI/SVOGI/SDFGI/DDGI all give SOFT low-frequency occlusion (CryEngine's own
    docs call SVO sun shadows "too soft … softness depends on voxel resolution"), are
    memory-hostile (512³ voxels ≈ 2.5 GB > our 1.5 GB), and need per-frame revoxelize
    for wind. Keep shadows and GI SEPARATE. The "shadow/LIGHTING" question is answered:
    don't merge. (Our existing probe-GI/contact already covers the no-black-shadows bar.)
 4. **The near-field wind cost is fundamentally "eat it, but eat less of it."** No
    technique makes a NEAR, leaf-dappled, swaying shadow cheap while KEEPING the dapple
    — everyone re-rasters real geometry there (SpeedTree full geo cascade 0; Guerrilla
    eats it + overlaps on async compute we don't have; UE re-rasters WPO every frame).
    Near levers are only: coarsen ~2–4×, CULL the count down (tighter shadow-relevant
    cull), and drop small/distant foliage to CONTACT SHADOWS (Epic ships grass-as-
    contact-shadows, NOT VSM). Capsule/SDF proxies make it cheap by THROWING AWAY the
    dapple → their real role is MID/FAR, not the near rescue.
 5. **SW ray-tracing deformed triangle clusters = NO** (per-frame BVH rebuild/refit
    over 162k swaying trees blows budget; M1 SW-RT ~10× off HW). Only viable against
    RIGID proxies (folds into the capsule idea, cheapest dynamic case).

PER-APPROACH VERDICT:
 - **VSM**: NO as a system — cache (its point) dies to wind, `StaticSeparate` is auto-
   disabled on Metal/M1, +256–512 MB pool + page-mgmt passes into a 1–2 ms budget.
   STEAL two wind-independent ideas: (a) screen-density resolution allocation (clipmap,
   ~1 texel/pixel) so near penumbra is crisp + far cheap; (b) shadow-visible-only
   cluster cull (a "needed-texel" mask, tighter than per-cascade frustum). Feasible w/o
   64-bit atomics — StratusGFX proves SVSM needs only 32-bit imageAtomicMin (our SW
   raster already IS the "software path").
 - **Capsule-proxy SDF soft shadows** (per-tree capsule trunk + ellipsoid crown,
   winds by TRANSFORM, splat into a near ~128³ clipmap SDF, sphere-trace w/ iq's
   min(res,k·h/t) 1-ray penumbra): the ONE representation where wind is ~free + a
   SHIPPED precedent (UE Capsule Shadows, chosen precisely because meshes deform). BUT
   blob/lollipop fidelity (no leaf dapple — worst exactly near-camera), UNPROVEN at
   162k, needs map() acceleration, full per-frame field rebuild ~3–10 ms on a 2080 Ti
   (~3× on M1) → 1–2 ms only via proxy-level splat + small clipmap + maybe 2–3-frame
   amortize. ESCALATION option for MID/FAR, not the near primary.
 - **Static/dynamic split**: DO IT, but it only caches the RIGID half (terrain/rock +
   trees beyond wind fade) — does NOT touch the 30-fps-moving cost (near wind foliage).
   Standard (DOOM Eternal, Decima, UE). Net win = static_cost − copy_cost; small when
   the dynamic set is the majority, so pair it with regime-split, don't bank on it alone.

DIRECTION — DECIDED (D-N29; user: "which gives BOTH best perf AND beauty? i dont care
how long it takes"). Build the LAYERED stack (see D-N29) as chunks S0–S5, MEASURE between
each (PERF LEDGER row + the static-sample-cost vs moving-raster-cost split). DAG-
independent wins FIRST; the coarse-caster-LOD half of the perf engine (S4) lands with the
N8 DAG. Order:
  S0 — RESOLVE-side sample cost (the static 115→70 gap = ~5 ms FIXED PCSS sample, NOT
       raster): eval shadowFactor at HALF-RES into a buffer + depth/normal-aware bilateral
       upsample in the resolve. Beauty-neutral on soft penumbra (contact band stays sharp
       via the bilateral). Helps static AND moving (sample is paid every frame). DAG-
       independent, self-contained (resolve-side only). ✅ DONE (log au): NaniteShadowHalf.ts
       — rg32f half-res (factor,camDist) + 4-tap camera-distance bilateral; SAMPLE roughly
       HALVED (Δ0.197 ms/17% @720p, Δ0.459/19% @1080p — ~linear in px, ~1.8 ms @4K), BEAUTY-
       NEUTRAL (shalfres on/off A/B identical), ?shalfres=0 restores full-res. (Overhead — the
       bilateral + 3×-reconstruct half-pass — caps it at ~2×, not the theoretical 4×.)
  S1 — Shadow-pass WPO-FREEZE + STATIC/DYNAMIC split: freeze trunk-wind in the shadow
       raster beyond a near distance (~80–150 m) → rigid → the R1 cadence caches it; cache
       rigid cascade depth, each frame atomicMin only the near-windy clusters on top
       (fixes the stale-wind-on-static-cache gap the R0 header flagged). DAG-independent. ← NOW.
  S2 — TIGHTER shadow cull (shadow-visible / needed-texel mask) + harder TEMPORAL STAGGER
       of far cascades + cast far cascades from a coarser DISCRETE LOD (interim, pre-DAG).
       Cuts the moving re-raster cluster count + frequency. DAG-independent.
  S3 — SCREEN-DENSITY RESOLUTION ALLOCATION (clipmap, not fixed 4-cascade splits): ~1
       shadow texel per screen pixel near = crisp penumbra + texel efficiency. The beauty
       centrepiece. DAG-independent (big).
  S4 — DAG-DECOUPLED caster LOD (GATED ON N8 DAG): cast each band from a progressively
       coarser DAG cut (near ~2× / mid ~8–16× / far coarsest). ~DONE/MINOR (log av): wired
       per-cascade shadow τ + minPx knobs, but shTotal moved only 0.3% (τ) / 12% (minPx=4) —
       the casters here are mostly NON-DAG (terrain clipmap + explicit veg; dagClusters ~6%),
       so LOD coarsening is second-order. The REAL lever surfaced: the shadow cascades carry
       38× the camera's clusters because they have NO OCCLUSION cull → S2-OCCL below.
  S2-OCCL (TRIED — WEAK, log aw): per-cascade light HZB + ortho sphereOccluded + two-phase
       cull, behind ?shadowoccl (OFF). Wiring proven (always-true → 99% cull) but the real test
       culls ~0: phase2 self-occludes (own phase-1 depth), CsmCached staleness starves phase1's
       prev-HZB, and a high sun has weak inter-caster occlusion (sees canopy tops). Kept as
       default-off WIP. The 38× disparity is COVERAGE×CASCADES×fine-geo, not occludable geo.
  S3 — SCREEN-DENSITY SHADOW CLIPMAP (PROMOTED to the real moving lever, log aw): collapse the 4
       overlapping cascades into ONE clipmap (~1 shadow texel/screen pixel) — cuts the wide-ortho
       × 4-cascade redundancy that dominates shTotal. Pair with far-caster PROXIES/impostors.
       Bigger architectural change + a far-shadow quality/perf tradeoff → USER-DIRECTION check-in. ← NEXT.
  S5 — ADDITIVE BEAUTY CEILING: capsule-SDF soft inter-tree occlusion (wind-free blobs for
       the large-scale soft band) + screen-space CONTACT shadows (fine sub-pixel band).
       Optional, once the sun term is fast + crisp.
REJECTED (killer citations in D-N29): full VSM (cache dies to wind; Metal disables
StaticSeparate); unified shadow+GI as primary sun shadow (Lumen keeps a separate VSM;
SVO "too soft"); SW-RT of deformed triangles (per-frame BVH rebuild over 162k blows it).
CONSTRAINTS (binding): WebGPU — NO 64-bit atomics, NO HW RT, ≤10 storage buffers/stage,
~1.5 GB UMA. Zero external assets. Deterministic seed. WIND shadows stay correct. Quality
floor: no black shadows, soft penumbra, no pop within 300 m.

The R2/R3/R4 chunks below are SUPERSEDED by S0–S5. STATUS: S0 DONE (log au — sample ~halved, real
win). S4 done but MINOR (log av — casters mostly non-DAG, ≤12%). S2-OCCL TRIED but WEAK (log aw —
occlusion ~0). **S3 BUILT (log ax) — the screen-density shadow CLIPMAP (NaniteShadowClip.ts) replaces
the 4 CSM cascades: own ortho VPs from sun+camera (CSM dropped for geometry), doubling texel-snapped
levels, gap-free HOLLOW ring cull (NaniteCull innerReject), shared vis. CORRECT + beauty ≥ cascades +
×0.32 the clusters + drops CSM. Default ON (?shadowclip=0 A/Bs to cascades).** BUT the honest finding:
fps is PARITY with the cascades (49.2 vs 49.9 moving @bm7/2592²) — the ×0.32 cluster win is offset by
per-level FIXED overhead (clear+copy+162k inst-cull × more maps; the cascade [1,2,3,6] cadence re-rasters
~1.8 maps/frame vs the clipmap's ~5). Uniform screen-density costs more maps than the cascades' coarse-far.
The NO-shadow ceiling here is 83.5 fps (100+ unreachable at this bookmark — the user's 115-120 was lighter),
so the target is "approach 83.5 / make shadows ~free". The clipmap is the BEAUTY win + the ×0.32 cluster
HEADROOM; the moving-fps WIN is the NEXT chunk on top of it: **S3-perf = (1) SHARE the instance cull across
levels (concentric + same-camera LOD ⇒ one inst-cull feeds all cluster culls, kills the ×N fixed cost);
(2) S4 DAG caster-coarsening on the clipmap's coarse far levels (minPx + DAG bite where the fine cascades
made them useless); (3) variable-T (sharp near / cheap far).** Coverage drops 3200 m→384 m (recorded
tradeoff, > the 300 m floor; a cached far-backstop level closes it). Also still open: S1 (WPO-freeze /
static-dynamic split — stale static-camera wind shadows).

---

N4 COMPLETE (2026-06-13; D-N16..D-N25 record the architecture). The material
übershader ports TERRAIN/ROCK/BARK/DEADWOOD (DEBRIS deferred to its N6 pool
migration). Chunks, each tsc-clean + committed:

1. ~~N4-C0~~ DONE (log entry p) — full-frame integration + the N3 depth-bias
   fix; all C0 gates measured, battery re-proven.
2. ~~N4-C1 — TERRAIN port~~ DONE (log p/q/s/t). CSM shadow-receive + manual
   PhysicalLightingModel BRDF (albedo/π) + probe-GI + caustics landed; micro-
   displacement already in fetchWorldVert. GATE RETIRED → ENERGY-CORRECT, not
   pixel-parity (D-N22, user directive): the old terrain double-adds an env-IBL
   skylight the nanite intentionally omits, so terrain lighting is judged on
   absolute quality, NOT diffed vs ?nanite=0. Old/shared path left untouched.
   Also: old geometry hard-disabled (D-N21, ?oldgeo=1 to restore for any A/B).
3. ~~N4-C2 — ROCK port~~ DONE (log u, commit 5525464). PORTED_CLASSES += rock;
   resolve gained per-vertex barycentric attribute interpolation (vdata 4×u8
   unpack + instance-rotated normals) + ported rockMaterial. F9 wall fixed
   (D-N23: heightfield read from heightTex texture, not the storage buffer).
   ?nandbg=cls debug added. Gate is the energy-correct/quality bar per D-N22
   (not a pixel diff). Per-instance slot-hash tint NOT yet wired (rock material
   uses vdata, not the B.w idF hash — revisit if clones show).
4. ~~N4-C3 — BARK + DEADWOOD + trunk wind~~ DONE (log v + w).
   - MATERIAL (v): texture-array (D-N24) + per-mesh matParam + UV/normal/vdata
     barycentric interp + TBN tangent normal map + hueShift×vdata + deadwood
     moss/rot. Mip = analytic isotropic LOD (hardware auto-mip dead in the
     non-uniform branch; .grad() NaNs near — ?nanbark=grad for the aniso pass).
     Infra: sampled-texture limit 16→24 (D-N25), storage-texture mip regen
     after compute (GOTCHA).
   - WIND (w): 'trunk' channel ported into the shared fetchWorldVert (raster +
     resolve bit-identical); profile in matParam high byte; gust fields
     precomputed per-instance in makeCtx; ?nanwind=0 A/B; ~2% perf. Deadwood
     rigid. A wind=1 shimmer/TRAA sanity probe over time is still worth a glance
     at N4-C4 (the static A/B is done).
5. ~~N4-C4 — close~~ DONE (log x). Battery green (registry/registry-gpu/
   nanitedbg/pan/parity/horizon-nanite — parity+horizon run `?nanwind=0` to
   match the rigid hwref; bit-identical to N3). Shadow-receive proven via NEW
   tools/probe-barkshadow.ts under `?oldgeo=1` (black slate has no casters —
   GOTCHA): coherent cast shadow on the trunk, bm7 50k shadowed-sunlit bark px,
   100% warm-albedo / 0% void (no black crush; deadwood covered by the shared
   isBD branch). Wind shimmer glance clean. Perf ledger row landed (cpu.submit
   →1.4 ms, draws →21). DEBRIS class branch arrives with its pool migration at
   N6 (the debris ring is not in the registry yet — N1 pool policy; phase-table
   wording predates it). Velocity output: see D-N16 (deferred, no consumer).

   **N4 COMPLETE.** USER CHECKPOINT (open in Chrome on the branch):
   `?scene=world&nanite=1&shot=N` (N∈1..9) = the black-slate beauty — terrain +
   rock + bark + deadwood all nanite-rendered, lit (sun×CSM-when-casters +
   probe-GI), wind-swayed trunks; HUD draws 21, cpu.submit ~1.4 ms. To see the
   nanite geometry receiving SHADOWS, add `&oldgeo=1` (restores casters until
   N5; tree camera draws stay hidden). A/B against the old pipeline =
   `?scene=world&nanite=0&oldgeo=1` (both flags — `?nanite=0` alone is an empty
   slate by D-N21). Per D-N22 materials are judged on quality, not pixel-diffed.
6. **N5 — cluster-driven CSM shadow casters** (design = D-N26; HW indirect
   draws into the cascades via per-cascade layers, mirroring the Forests caster
   path this phase deletes). Chunks, each tsc-clean + committed:
   - N5-C0 — PER-CASCADE CULL: a shadow cull per cascade = buildNaniteCull with
     the cascade NaniteCam (ortho frustum planes from csm.lights[c].shadow.camera,
     refreshed one frame stale like Forests.planesCsmU), sphereOccluded=null,
     cone disabled (new opt), camera-distance LOD. Produces a per-cascade
     visible-cluster qRaster + indirect args. HUD per-cascade counts; validate
     numerically (counts ≥ camera cull, grow with cascade index; off-screen
     casters present). Gate ?nanshadow2=1 (off by default until C2). NO visual
     change yet.
   - ~~N5-C1 — HW CASTER + INTEGRATION (the risky chunk)~~ DONE (log z, D-N27).
     Per-cascade vertex-pulling NodeMaterial Mesh (layer 2+c, castShadow,
     frustumCulled=false, identity matrixWorld) injecting world pos via
     `castShadowPositionNode` (the shadow pass IGNORES vertexNode — GOTCHA), depth
     into the cascade via three's CSM; non-indexed indirect DRAW (kCasterArgs:
     count·128·3) over cull.qRasterRO. GATE GREEN: black-slate nanite geometry
     self-shadows at ?nanshadow2=1 / ?oldgeo=0 (bm7 41% / bm3 71.8% px darker vs
     casters-off, coherent cast shadows). Root-cause eaten by the `map=null` trap
     (GOTCHA). ?nancasterdbg=1 main-pass caster viz kept. DEFERRED to C3 (not C1):
     the per-cascade re-cull is still every-frame (C0); three only DRAWS a cascade
     on its CsmCached refresh tick so the gate is met, but gating the CULL to the
     tick + killing the 128-stride over-draw (shadow.c0 18–36 ms) is the C3 perf
     pass. ?nanshadow2 still default-OFF (C2 flips it after parity).
   N5-C2/C3 (HW-caster retire + cadence) are SUPERSEDED by the R-CHUNK REWRITE
   (D-N28): the HW caster measured 14–31 ms/cascade (90→15 fps) — replaced by a
   depth-only compute SW raster into own r32 buffers, sampled by the resolve's own
   PCSS. User directive: PROPER nanite, ~1–2 ms shadow budget, zero quick measures.
   - R0 — DEPTH-ONLY SW SHADOW RASTER + OWN BUFFERS + RESOLVE SAMPLES THEM. Per
     cascade: r32uint depth buffer (atomicMin target). A depth-only raster (reuse
     rasterKernel('depth')'s scanline + SW/HW big-tri split; cam.vp = the cascade
     LIGHT VP from csm.lights[c].shadow.camera; NO payload pass) over the existing
     NaniteShadow C0 per-cascade qRaster, writing the cascade buffer. Resolve's
     shadow factor (replace nodeObject(world.csm).x, NaniteResolve.ts:499) = OWN
     PCSS over the 4 buffers: cascade-select by world pos, shadowCoord = the SAME
     cascade VP (lockstep), manual taps reading the r32 buffer. Re-raster EVERY
     frame (no cadence yet). GATE: shadows appear, ATTACHED (no crawl/peter-pan vs
     C1), correct under TRAA; measure cost. Keep C1 for A/B (?nanshadow3 vs
     ?nanshadow2). Validate cascade 0 FIRST (the one you can see), then 1–3.
   - ~~R1 — CADENCE~~ DONE (log ac). Per-cascade re-raster gated on exact light-VP
     change (`Matrix4.equals`; CsmCached freezes the pose → bit-identical VP when
     cached → no epsilon). depthTex persists across the skip; cascVP/cascParam left
     untouched in the same skip → lockstep automatic. New `rasteredMask()` →
     `nanite.shRaster`. GATE GREEN both halves @bm7: STATIC shRaster=0, nanRasterDepth
     35→1.77 ms (==payload ⇒ shadow raster zeroed), fps 22→63, shadows still
     attached; MOVING (tools/probe-shadowcadence.ts) clean [1,2,3,6] cadence, 1.83
     cascade-rasters/fr vs 4.00, cluster-weighted ~35% of R0. tsc clean.
   - R2 — COARSE LOD for far-cascade shadow casters (cascade 0 stays honest for
     PCSS) + a depthOnly buildNaniteRaster option (the per-cascade rasters build
     unused payload/HW-resolve kernels today, ~64 MB/cascade ≈ 200 MB total). GATE:
     cluster counts down, penumbra unaffected near; memory down.
   - R3 — STATIC/DYNAMIC SPLIT for wind: cache static cascade depth, every frame
     copy + atomicMin ONLY trunk-channel (wind, within 380–480 m fade) clusters on
     top. GATE: windy trunks cast MOVING shadows with a STILL camera; cost ~1–2 ms.
   - R4 — close. C1 HW caster mesh path = already deleted in R0; flag UNIFIED to one
     `?nanshadow` and FLIPPED DEFAULT-ON early per user directive (log ad); remnant
     comments cleaned. REMAINING: parity at bookmarks (incl. off-screen casters via
     pan probe), perf ledger row, USER CHECKPOINT, ⏸. (Old Forests/ShadowProxy
     casters retained as the ?oldgeo A/B ref — deleting them is N6/N9 scope.) Then
     N6 (migrate opaque pools). NOTE: R2/R3 (perf to 1–2 ms) now precede R4-close.

> Reconciled into 12-base-raster.md §Reconciliation (2026-07-02); this doc kept for detail. ⚠️ Known kills: "128 threads/WG" (actual 255); §3.5+L3 shadow-clipmap live cost & `?culloverlap` (forest passes csm:null ⇒ shadow system never built); f2b expectation reconciled to −0..−0.5.

# 03 — Base mesh pipe (trunks/bark/terrain SW raster + cull + HZB)

Reader 03. Sources: src/nanite/{NaniteRaster,NaniteCull,NaniteHzb,NaniteFrame,DagHierarchy,BuildDag,
BuildHeightGrid,NaniteShadowClip,NaniteCommon,Tsl}.ts read in full; measured numbers from the
2026-07-02 table and `fresh-noleaves-now.json` / `fresh-voxbocc.json` in the session scratchpad.
Premise audit (90-premise-audit.md) read first; targets eye ≤18 / oblique ≤21 / aerial ≤16.

## 1. TL;DR

- Base-sans-post is a **flat ~10.0/10.3/11.1 ms floor** (eye/oblique/aerial) while its triangle and
  cluster counts swing 3.7× — it is **pixel/fixed-cost bound, NOT triangle-emit bound**. The
  2026-06-26 "trunk DAG never coarsens, 46M tris" disease was **cured by 0f77edc** (bark cap-only →
  1-root collapse) + the default-on lodWarp; that chapter is closed pending one verification probe.
- Zero-quality-loss base headroom is ~2–4 ms, not the 10 ms the pool suggests: `?f2b=1` per-pixel
  early-out (implemented, default-off, loss-exact), resolve-pass cost (probe `?nores`), HW
  vertex-pull 3×-fetch waste.
- The live/moving story is where base really bites: the **shadow clipmap cull+raster runs ONLY when
  moving** (mask=0 static ⇒ every isolated probe measures shadows at zero), and the **single-phase
  prev-HZB cull drops disoccluded clusters for 1 frame (holes)**. Two-pass occlusion à la UE5 is
  buildable with buffers the code already reserved.
- Eye p95 spikes are NOT base (noleaves eye max 17.5, no spikes; spikes appear only with foliage).
  Oblique bimodality IS present without foliage (noleaves oblique alternates ~7–9.6 vs ~15–17.4 ms)
  ⇒ voxOccPyr is excluded as sole cause; GI-cycle/HZB-feedback/Dawn remain.

## 2. How it works today (mechanism walk)

### 2.1 Per-frame pass chain (static isolated pose, noleaves)

Frame order (NaniteFrame.ts:440–496):

1. **Camera cull, one submit** (`cull.runPhase1`, NaniteCull.ts:998–1010): `kClearHier` →
   `kSeedRoots` (one thread per instance: 400k noleaves / 600.8k default; frustum + lodDist far
   envelope + nearDist voxel handoff + instMinPx, NaniteCull.ts:760–805) → `(kArgs, kTraverse)` ×
   hierDepth ping-pong BFS passes → `kRasterArgs`. hierDepth = registry.maxDagDepth+2
   (NaniteFrame.ts:205) ≈ **14** (leaf DAGs reach 12 levels per the boot log in
   fresh-noleaves-now.json even in noleaves — leaf heads stay registered, they only drop at seed).
   Each kTraverse item projects `ownError` (NaniteCull.ts:829–835), applies **lodWarp** τ-banding
   (NaniteCull.ts:111–126, wired default-on — see §3.2), and on CUT emits with frustum + minPx +
   cone + **prev-frame HZB occlusion** (NaniteCull.ts:917–925), else enqueues DAG children.
2. `syncFullArgs` (kRasterArgs2: full-range + phase-2 append args, NaniteCull.ts:482–489) and, in
   the default config only, the voxel fanout (3 dispatches at voxf2b=0, NaniteCull.ts:1038–1048).
3. **world1, one submit + one render pass** (NaniteRaster.ts:1418–1424): `[kVisClear,
   kRasterWorld1, kHwArgs]` batched, then the HW vertex-pull render (`hwRender`), then (default
   only) the voxel scatter. kVisClear = 3.34M threads × 3 atomic stores (NaniteRaster.ts:389–408).
   The SW raster is one workgroup per emitted cluster, 128 threads = 1 tri each
   (NaniteRaster.ts:1015), fixed-point scanline with span-solving (837–874) and the 24-bit
   depth-key election `depthKey24<<8|id8` via guarded atomicMax into visPayloadV + visBV plain
   store (952–975). Triangles with bbox >16 px or near-crossing route to the HW queue (668–675,
   1001–1010); the HW pass's fragment stage writes the same election (1157–1184).
4. **HZB build** (NaniteFrame.ts:481): min...max-pooled classic-depth pyramid over
   `visPayloadV`'s top 16 bits (packed=true, NaniteFrame.ts:196–197; NaniteHzb.ts:63–68, 119–121),
   14 levels in one batched submit (NaniteHzb.ts:150–155). **Because it sources the election buffer
   AFTER the voxel scatter ran (in the default config), the pyramid contains canopy depth** — this
   is why the cull kills trunks under canopy (§2.3).
5. **Shadow clipmap** (NaniteShadowClip.ts): per-level snapped-VP equality gate
   (NaniteShadowClip.ts:314–315) ⇒ at a static pose **mask=0 and NOT ONE shadow dispatch runs**
   (NaniteShadowClip.ts:420–425). Moving: shared-cut BFS (another full hierDepth-pass cull) + per
   re-rastering level: filter + clearVis + depth raster + hwDepth + copy (382–392).
6. shadowHalf, resolve (two fullscreen passes), scene+post. Resolve *shading* is measured ~FREE
   (nandbg=flat ≈ baseline; nanshadow=0 ≈ baseline — 2026-07-02 table); the resolve *pass
   structure* has its own flag `?nores=1` (NaniteFrame.ts:283–288) and was never isolated.

### 2.2 The occlusion test, exactly

`sphereOccluded` (NaniteHzb.ts:158–205): nearest sphere point vs the 2×2 max-depth footprint at
the mip where the sphere fits one texel, evaluated with **cam.prevVp + cam.prevCamPos**
(NaniteCull.ts:921) against a pyramid holding **frame N−1's depth** (built at NaniteFrame.ts:481,
consumed by frame N's cull which runs earlier in the frame). This is a coherent pair — the
semantics are precisely *"cull what was occluded in the previous frame's view"*. Rejects are
**dropped, not recorded**; the world path has no re-test ("the two-phase record/re-test is a
follow-up — it needs a buffer-budget rework", NaniteCull.ts:917–918).

### 2.3 Evidence the cull side works very well statically

- Default oblique: visTris 1.27M < noleaves oblique 2.11M — canopy depth in the pyramid culls
  trunk clusters (fresh-voxbocc.json vs fresh-noleaves-now.json counters).
- Default aerial: the entire cut is **661 clusters, ALL voxel, 44.7k tris, hwTris 0**
  (fresh-voxbocc.json aerial counters) — terrain + trunks under the canopy are 100% occlusion-culled.
  The base pipe at aerial in the default frame is almost pure fixed/pixel cost.
- rejInst/rejClust are 0 everywhere (they are only written by the deleted two-phase path).

## 3. Waste map (each item: what it scales with, ms estimate, source)

### 3.1 The headline: base is FLAT — the per-triangle term is bounded ≲1–2 ms

| pose | base-sans-post (audit §2.4) | SW visTris (noleaves) | visClusters | hwTris |
|---|---|---|---|---|
| eye | ~10.0 | 2.093M | 11,375 | 234k |
| oblique | ~10.3 | 2.105M | 11,935 | 175k |
| aerial | ~11.1 | 0.558M | 3,180 | 38.5k |

Aerial has **3.7× fewer tris and clusters yet costs MORE** than eye. If the per-triangle emit cost
were ≥1 ms/Mtri, eye would exceed aerial by ≥1.5 ms; it is 1.1 ms *lower*. ⇒ the SW per-triangle
setup+emit term is ≲1–2 ms of the ~10 ms pool; the rest scales with **covered pixels** (terrain
fills the frame at aerial; the per-pixel election loop was already measured ≈90% of world1,
NaniteRaster.ts:264–271 rdbg finding) plus **fixed pass-chain cost**. This CONTRADICTS the standing
memory "base-raster-is-the-bottleneck / TRIANGLE-EMIT-bound" — that finding described the
pre-0f77edc geometry (see §3.2) and is superseded.

Cost model: `base(pose) ≈ F + kpx·coveredPx + ktri·visTris`, with F + kpx·3.34M ≈ 9–10 ms and
ktri·2.1M ≲ 1–2 ms. The split of F vs kpx needs probe P4 (noleaves dpr sweep); the known members of
F: cull BFS (~0.2 ms, refuted as bottleneck 2026-06-26), 14 HZB dispatch-barriers (~1.11M texels),
kVisClear (~40 MB ≈ 0.2–0.3 ms), submit/pass-switch overhead (cpu.submit med ~5.2 ms in noleaves —
fresh-noleaves-now.json cpuSubmit — but GPU-overlapped; dropped to ~1.4 in the voxbocc run).

### 3.2 The trunk far-field question — RECONCILED, chapter closing

The 2026-06-26 handoff (docs/perf-runs/2026-06-26-base-raster-review-handoff.md:26–28) measured
base at 28–30 ms with 8.6M(eye)–32.6M(aerial) trunk tris (under a constant CPU-load era, absolutes
inflated) and the memory claimed "trunk DAG never coarsens, 46M tris, 97% sub-pixel". What changed:

1. **Commit 0f77edc (2026-06-27, "connected-bark cap-only")**: every open bark base ring gets a
   buried cap ⇒ manifold ⇒ QEM collapses each tree to ~1 root (beech 98 roots/18,119 tris → 1
   root/92 tris). Measured in that commit: far-field visTris **31.6× lower aerial (8.86M → 280k)**,
   eye 2.95×; gpuWall −24%/−47%/−21% (eye/oblique/aerial). Today: noleaves eye 2.09M.
2. **lodWarp is default-ON in the frame path** (NaniteFrame.ts:172–177: simband=6, lodnear=4,
   lodpow=0.6; τ=3 (:157–158), minPx=2 (:165–166), instMinPx ≈ 0.075·min(fb) ≈ 110 px (:187–191)).
   τ_eff = 3·(1+((d−4)/6)^0.6): ≈15 px at 60 m, ≈27 px at 200 m. The far trunk field is ALREADY
   aggressively coarsened; note the lodWarp header still says "this knob never ships"
   (NaniteCull.ts:104–110) — stale comment, it shipped.

Residual sub-pixel-emit pool: 2.1M SW tris + 0.23M HW tris against ≤3.34M screen px (owned px
unknown) ⇒ ~0.6–1.0 tri/px — the 97%-sub-pixel disease is gone. ⚠️ The `?audit=1` covered-pixel
counter CANNOT verify this: kAudit keys on the visDepthV sentinel (NaniteRaster.ts:1046–1057) and
world1 never writes visDepthV (NaniteRaster.ts:967–971) — audit is dead in world1 mode. Probe P5
uses shot-based coverage instead. Any FURTHER τ/minPx/warp coarsening is Class R (red-list adjacent
— voxlodk≥0.7 and aggdist=60 lessons) and given ktri ≲1–2 ms it cannot pay anyway: **do not spend
quality budget here.**

### 3.3 Per-pixel election overdraw (world1) — the one pixel-side pool with a free lever

Every covered fragment pays z-interp + a relaxed load + (if it might win) atomicMax
(NaniteRaster.ts:892–975). Losing fragments behind the current winner still pay the z-interp. The
implemented-but-default-off `?f2b=1` gate (NaniteRaster.ts:286–289) computes the triangle's
nearest-possible key once per tri (:697–703) and skips the whole emitFrag for provably-losing
pixels (:983–989) — loss-exact by monotonicity. Scales with base overdraw (terrain-behind-trunks,
trunk-behind-trunk); unknown magnitude, bounded by the coverage-dominated ~8–9 ms. Estimate 0.5–2 ms
at eye/oblique; needs P2.

### 3.4 HW vertex-pull fetches 3 corners per vertex, uses 1

The HW vertex stage fetches ALL THREE corners with `fetchWorldVert` then selects by
`vertexIndex.mod(3)` (NaniteRaster.ts:1123–1128) ⇒ 3× the vertex work: at default eye hwTris 1.10M
(fresh-voxbocc.json) that is 3.3M vertex invocations × 3 world-vert fetches (incl. wind/displace
math) = 9.9M vs the 3.3M needed. `fetchWorldVertByIndex` already exists for the indexed path
(NaniteFetch.ts:124–130); the window-grid HF branch needs the (localTri, corner) form kept.
Estimate: eye ~0.3–1 ms (hwTris-heavy: near leaf needles — technically foliage load but the
mechanism is this file), oblique/aerial ~0.1. Quality-IDENTICAL (same vertices, same math).

### 3.5 Live-only base costs invisible to every isolated number in the table

- **Shadow clipmap**: static ⇒ mask=0 ⇒ zero dispatches (NaniteShadowClip.ts:314–315, 420–425); the
  `nanshadow=0 ≈ baseline` "shadows FREE" row was measured at static poses and says NOTHING about
  moving cost. Moving ⇒ level-0 re-rasters nearly every frame: a SECOND full hierDepth-pass BFS cull
  (the shared cut) + per-level clear/depth-raster/hwDepth/copy. This is a base-pipe live cost with
  its own counter (`nanite.shRaster` mask, NaniteFrame.ts:507).
- **Stale-HZB disocclusion**: single-phase prev-HZB cull (§2.2) drops newly-revealed clusters for
  exactly 1 frame (hole → far depth in next pyramid → self-heals). Under continuous motion this is a
  persistent crawling artifact currently smeared by TAA, and the 2026-06-26 review attributed
  ~9–11 ms of moving-vs-static gap to stale-HZB cut behavior. The pose-arrival ramp (aerial first
  2–3 frames at 6.5–11.5 ms then settle ~17) is the teleport-limit of the same mechanism: the
  pyramid+prevVp still describe the OLD pose ⇒ mass over-cull ⇒ cheap wrong frames.
- `?culloverlap=1` (implemented, default OFF, NaniteFrame.ts:255–261, 453–461) folds the shadow
  shared-cut into the camera-cull submit on re-raster frames — a live-only, quality-identical
  overlap lever that has never been measured.

### 3.6 Non-pools (checked, bounded, not worth levers)

- kVisClear ~0.2–0.3 ms (bandwidth bound, 40 MB).
- Voxel clusters riding qRaster through world1: ~9.2k workgroups bail at the matClass guard
  (NaniteRaster.ts:539–555) after the wgcache barrier — ≤0.05 ms at the measured <2.5 ns/wg launch.
- Empty-tail BFS passes (hierDepth 14 with an early-converging frontier): PERF LEDGER already
  measured BFS/submit batching marginal; don't retry.
- The HW pass's dead full-res rgba8 store (clear already skipped, NaniteRaster.ts:290–304): ~13 MB
  write ≈ 0.03 ms.
- kSeedRoots 600k threads: trivial ALU, ~0.1 ms.
- NaniteHwRef: debug view only (NaniteView.ts:43), not in the production frame.

## 4. Levers (ranked)

### L1 — Measure + default-on `?f2b=1` (per-pixel early-out in world1)
- **Mechanism**: skip z-interp+election for fragments whose triangle-nearest key can't beat the
  pixel's current winner; loss-exact (atomicMax monotone). Already implemented behind the flag.
- **Files**: NaniteRaster.ts:286–289 (flip default), nothing else.
- **Expected**: eye −0.5..−1.5, oblique −0.5..−2, aerial −0.3..−1 (scales with base overdraw;
  honest unknown until P2). Live: proportional.
- **Quality**: IDENTICAL (byte-equal claim; verify shotdiff maxDiff=0, election tie-order argument
  does not even arise — the skip only fires when cand provably ≤ winner).
- **Gate**: P2 interleaved A/B ≥1 ms at any pose + shotdiff. **Effort S.**
- **UE5**: SW rasterizer relies on tight cluster-level occlusion + HW early-Z on the HW path; a
  per-pixel pre-load gate is the SW-scatter equivalent.

### L2 — Two-pass occlusion (record rejects → re-test vs fresh HZB → append raster)
- **Mechanism**: UE5's CULLING_PASS_OCCLUSION_MAIN/POST
  (docs/perf-runs/Nanite-UE5-shaders/NaniteCulling.ush:9–12): pass 1 = prev-HZB visible set; pass 2
  re-tests pass-1 REJECTS against the freshly built HZB and rasters the delta. Ours: kTraverse's
  occlusion reject (NaniteCull.ts:919–925) appends (instId,ci) to a reject list instead of dropping;
  after world1 + hzb.build (already in this order, NaniteFrame.ts:475–481) a small kReTest re-tests
  the list with **current vp/camPos** vs the fresh pyramid (`sphereOccluded(cam.vp, cam.camPos)` —
  the API was built for exactly this, NaniteHzb.ts:46–50), appends survivors to qRaster past
  phase2Base, `kRasterArgs2` sizes rasterDispatch2 (both already exist, NaniteCull.ts:482–489), one
  SW+HW raster over the append range.
- **Buffer budget (Metal 10-buffer cliff, premise audit §3.9)**: kTraverse binds exactly 10
  (counters, inV, outV, instances, meshes, clusters, dag, dagLinks, qRaster, hzb) — an 11th reject
  buffer would silently kill it. Fold the reject records into the TOP END of the qRaster buffer
  (records grow down from QRASTER_CAP; count in the free counters slot 3 — reserved for precisely
  this per NaniteCull.ts:742–744; scar-fold precedent NaniteRaster.ts:339–345). Zero new bindings.
- **Expected**: isolated static ≈ +0.1..+0.3 (converged reject set is small; one extra small submit).
  Live moving: removes the 1-frame disocclusion holes and the pose-arrival over-cull ramp; direct ms
  effect on the moving gap unproven (pass-1 keeps are unchanged) — P3 sizes the inflation first.
- **Quality**: IMPROVING (motion correctness; static bit-identical). **Gate**: static shotdiff = 0 +
  a moving-capture A/B showing hole pixels eliminated; live slot histogram no worse. **Effort M.**

### L3 — Measure + default-on `?culloverlap=1` (camera‖shadow cull one submit)
- **Mechanism**: concatenate the two disjoint cull batches so Dawn overlaps them on shadow-re-raster
  frames (moving only). Implemented; correctness argument already in code
  (NaniteCull.ts:192–198, NaniteFrame.ts:453–461).
- **Expected**: isolated 0 (static: no shadow work); live −0.5..−1.5 on moving frames.
- **Quality**: IDENTICAL. **Gate**: P3 live A/B slot histogram. **Effort S** (flag flip after probe).

### L4 — Resolve-pass cost (probe, then conditional fusion/scissor)
- **Mechanism**: `?nores=1` (NaniteFrame.ts:283–288) skips both fullscreen resolve passes — the only
  unmeasured always-on fullscreen work in the base pool (shading math measured free; the pass
  structure — wp reconstruction, fetch chain, two passes for the Metal buffer split
  (NaniteFrame.ts:289–294) — is not). If ≥2 ms: design a fusion that respects the 10-buffer split
  (e.g. vox-winner stencil/scissor pass, or moving the vox decode behind a per-pixel branch with a
  reduced binding set). Probe first; do not design blind.
- **Expected**: unknown; pool bound ~1–2.5 ms. **Quality**: IDENTICAL required. **Effort M** (after P1).

### L5 — HW vertex-pull single-corner fetch
- **Mechanism/files**: §3.4; NaniteRaster.ts:1111–1136 + NaniteFetch fetchWorldVertByIndex.
- **Expected**: eye −0.3..−1 (1.1M hwTris default eye), oblique/aerial ~−0.1.
- **Quality**: IDENTICAL (same selected vertex, same math). **Gate**: shotdiff + A/B. **Effort S/M**
  (HF window-grid branch kept on the 3-fetch form or given its own indexed path).

Sum of realistic base-slice contributions at oblique: ~1.5–3.5 ms isolated (L1+L4+L5) + live-only
L2/L3 — the master plan should budget base at **−2 to −3 oblique**, and NOT the full 10.3 pool.

## 5. Refuted / rejected paths for this stage (do not retry)

- **"Base is TRIANGLE-EMIT-bound / trunk DAG never coarsens (46M tris, 97% sub-pixel)"** — the
  memory `base-raster-is-the-bottleneck` framing is SUPERSEDED: cured by 0f77edc (cap-only bark →
  1-root collapse; commit message numbers: aerial visTris 8.86M→280k) + default-on lodWarp
  (NaniteFrame.ts:172–177). Today's flat 10–11 ms across a 3.7× tri swing (§3.1) is the disproof.
- SW per-pixel ALU micro-opt (rcp hoist, FMA depth) — implemented + REFUTED 2026-06-26 (handoff
  :32–34; Tint already does it).
- Cull as the bottleneck — refuted (~0.2 ms, 2026-06-26 review).
- Triangle-granular re-dispatch of the raster grid — bounded-marginal (rdbg LEVER #1 finding,
  NaniteRaster.ts:264–271).
- Tiled/binned raster — refuted + REMOVED (memory: capture-as-ground-truth; +11.7/+18.1 ms).
- Submit/BFS batching beyond what shipped — measured marginal (PERF LEDGER, handoff :74–75).
- Further τ/minPx/lodWarp coarsening of trunks — Class R with ktri ≲1–2 ms: cannot pay for its
  quality risk; red-list adjacent.

## 6. Open questions + serial GPU probes wanted (decision rules)

- **Q-A (spikes/bimodality, base evidence)**: eye p95 spikes need foliage — fresh-voxbocc eye has
  35.4/26.6/26.5/24.x over a 17–19 base; fresh-noleaves-now eye max is 17.5 with LOW outliers only
  ⇒ base exonerated for task #14; route to foliage/bimodality readers. Oblique bimodality exists
  WITHOUT foliage: fresh-noleaves-now oblique alternates runs ~7–9.6 vs ~15–17.4 ms while
  voxActive=false (registry.brickCount=0 ⇒ no vox raster, no voxOccPyr — NaniteFrame.ts:85) ⇒
  **voxOccPyr/HZB-vox interaction is excluded as the sole bimodality cause**; refines premise-audit
  §6.6 suspects to GI probe cycle / HZB-cull feedback / Dawn-Metal pipelining. The ~8 ms lows need
  an engagement counter (are they over-culled frames? visClusters per frame) before any conclusion.
- **P1 — resolve-pass cost**: one session, 3 poses, interleaved default vs `?nores=1` and noleaves
  vs `noleaves&nores=1`. Decision: delta ≥2 ms at oblique ⇒ open L4 design; <1 ms ⇒ close L4.
- **P2 — f2b A/B**: default config, 3 poses, interleaved `?f2b=1` vs off, plus shotdiff maxDiff=0
  at the 3 canonical + 2 stress poses. Decision: ≥1 ms anywhere ⇒ default-on (Class I);
  <1 ms everywhere ⇒ close, record engagement (needs a skipped-fragment counter or accept the null
  only with the overdraw number from P5).
- **P3 — live base counters (the live-vs-isolated divergence probe this slice owns)**: live moving
  run logging per tick: `nanite.shRaster` mask, `nanite.visTris`, `nanite.visClusters`, cpu.submit;
  then the same route with `?culloverlap=1`; then a static-hold segment. Decision: (a) shadow
  re-raster frames cost ≥2 ms over mask=0 frames ⇒ L3 default-on + shadow budget becomes a named
  live lever; (b) moving visTris/static visTris >1.3 ⇒ stale-HZB cut inflation is real ms, L2
  priority up; ≤1.1 ⇒ L2 is quality-only.
- **P4 — noleaves dpr sweep**: noleaves at dpr 1.0 and 1.5, 3 poses (one session). Fits
  `base = F + kpx·Mpx`. Decision: kpx·3.34 ≥ 6 ms ⇒ pixel-side levers only (L1/L4); F ≥ 5 ms ⇒ a
  pass-chain/fixed-cost hunt is justified (would contradict the marginal-batching ledger — re-audit
  premise first).
- **P5 — base coverage/tri ratio**: noleaves shots at 3 poses → non-sky pixel count vs visTris
  (audit counter is dead in world1 mode, §3.2). Decision: tris/owned-px ≤1.5 ⇒ permanently close
  "trunk sub-pixel emit" (update the memory); >3 ⇒ reopen as Class R for explicit user decision.

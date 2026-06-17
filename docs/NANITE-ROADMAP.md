# NANITE WORKSTREAM — ROADMAP (the task DAG · READ FIRST on every compact)

> The live plan + "you are here". **Read this FIRST** on rehydration — it orients you,
> then points into `NANITE-SPEC.md` (durable design + D-N* decisions; read fully) and
> `NANITE-LOG.md` (dated journal; read recent-first). Update statuses + append tasks
> here EVERY session. Never re-plan from scratch; never re-derive a D-N* — challenge it
> explicitly if wrong. The built-in Claude Code task tool is a DISPOSABLE mirror of the
> 🔵/⬜-next slice only (not in git, doesn't survive compact) — THIS file is the truth.
>
> Status key: ✅ done · 🔵 active · ⬜ pending · 🚫 blocked. `blockedBy` = task ids that
> must finish first. `spec` = the `## header` in NANITE-SPEC.md (+ D-N* / file refs).

## YOU ARE HERE — 2026-06-17  →  **READ LOG `ca` (UE5-gap: scanline win) + `bz` (capture method + banded-τ) FIRST.**
**ALL COMMITTED this session — working tree clean. `19eb834` SCANLINE x-span (UE5-gap #1, ~2-4ms, bit-identical);
`b490d03` UE5-gap report; `152c467` banded-τ defaults; `72fb8a7` submit lever marginal; `60430c8` HZB batch + meter
fix; `6fb9ca1` capture-vs-intent analysis.** PRIOR: `41d2dd5` PERF-VB4 single-pass, `547bbfc` forest full-pipe testbed.
**SHIPPED PERF THIS SESSION:** banded-τ defaults (loderr=3/nanitemin=2/instminpx≈128, ~1.55× whole-frame, quality-gated)
+ the SCANLINE raster win (per-row covered x-span vs full-bbox walk; bit-identical; vista frame 20.5→16.3ms). The
UE5-gap hunt's other wins: #2 depth-DDA = measured non-win (atomic-bound loop) REVERTED; #4 terrain `If(isT)` gate
DONE (bit-identical, ~0.15ms — gates buildTerrainShading in NaniteResolve so non-terrain pixels skip the terrain
prelude; world+forest validated); #3 same-frame Hi-Z (~0.5ms, capped) + #5 HW makeCtx cache (~0.04ms) low-priority, NOT done.
**NEXT (post-compact, user-directed): N8-HIC cross-instance MERGE — the real ~2× vs UE5** (the cluster-count/overdraw
floor; UE5 covers many trees with few far clusters at ~1 tri/px, we stay above it). UE5-GAP-REPORT §3+§4: biggest
structural lever BUT NOT no-quality-loss as-is — merging far bark/rock/deadwood across instances loses the procedural
per-instance tint/wind (`slotHash`) unless baked into per-super-cluster vertex attributes (a quality approximation + a
cross-instance super-cluster BUILDER that doesn't exist). Needs an explicit quality budget. SPEC D-N43 +
`BuildAggregateDag.ts` (closest existing machinery). SEPARATION PRINCIPLE: nanite self-contained in src/nanite. Stage it
(builder → wire → judge shots), compact between stages.
**THE PERF METHOD that worked = WebGPU-Inspector capture as GROUND TRUTH, not the statistical harness** (LOG bz; the
harness dead-ended at "world1 ~90% per-pixel, nothing left" + the multi-agent RASTER-FORGE runs burned ~6M tokens
without a coherent result — user: "dogshit"). `slice.py` slices a capture → per-resource-type intent-divergence analysis
→ `docs/perf-runs/2026-06-17-webgpu-inspector/DIVERGENCE-REPORT.md`. **MEASURED CONCLUSION: the frame is genuinely
raster-bound (`nanRasterWorld1` ~16.7 ms = 75%); the submit-overhead lever is MARGINAL (measured, not assumed — don't
refactor the cull for it); ~1.5 GB dead full-world residency in the forest testbed is VRAM hygiene, not frame-time.**
**LANDED THIS SESSION — banded-τ perf defaults (the real ~1.55× whole-frame win, user-chosen after browser A/B):**
`NaniteFrame` defaults now `loderr=3` (τ error-px cut — coarsens LOD, never drops geometry), `nanitemin=2` (sub-2px
cluster cull), `instminpx=round(0.075·min(fbW,fbH))≈128` (far instances→impostors, resolution-relative). Forest @1280×720:
clusters ~2.5× fewer, raster ~2× (23→11 ms), whole-frame 30→19 ms; world scene validated (terrain/shadows fine at τ=3).
`?loderr/nanitemin/instminpx` override live.
**NEXT real frame-time lever = the per-pixel raster itself** (the 16.7 ms is now the dominant remaining cost): N8-HIC
cross-instance MERGE (#48, cuts the visible-cluster count at no quality loss) or same-frame cluster Hi-Z (overdraw).
Use the forest testbed (`?scene=forest&nanite=1`) + a WebGPU Inspector re-capture to validate structural changes.
**NOTE:** the old RASTER-FORGE multi-agent perf workflow + per-pass-timestamp harness are DEPRECATED for diagnosis
(LOG bz) — prefer capture-as-ground-truth + a single whole-frame `__laas.measureFrames` reading. The per-pass GPU
timestamps overcount (umbrella `compute`/`render` groups double-count vs leaf passes); trust gpuWall + cpuSubmit.

PRIOR (committed earlier): TERRAIN-RW (`ff0511a`) + PERF-VB3 camera (`6123c60`): HIER is the SOLE world cull, `nanitedag=all`
default. SHADOW-HIER + S3-perf + BRUTE DELETION (`4daf005`, LOG bw). The hierarchical DAG-BFS is now
the ONLY cull path in the engine — camera AND both shadow paths (clipmap + cascades). The legacy two-phase brute cull is
DELETED (NaniteCull 975→643 lines; chunk/reject buffers gone; `?shadowhier`/`?shadowoccl`/NaniteView `?hier`/`?phase2`
retired).** Highlights:
- **S3-perf SHARED CROSS-LEVEL clipmap cull (`NaniteClipCull.ts`):** the cut is identical across clipmap levels, so ONE
  hier traverse (over the largest re-rastering level's box; gated on the R1 cadence) + cheap per-level frustum+hollow FILTERS
  replace N full traversals. PARITY EXACT, tied-or-FASTER than brute (bm7 −0.4 noise, vista +0.3; static identical), ~3.6×
  lighter memory. Caveat: assumes one τ across levels (a future per-level-τ S4-on-clipmap breaks the shared cut).
- **First-cut per-level hier** (the simple migration) was −2..−8 fps vs brute (equal cut size ⇒ no draw-envelope win, only
  added overhead); S3-perf was the user-chosen fix before deletion. `probe-shadowhier`/`probe-hierdepth` are the gates.
- **Validated post-deletion:** world clipmap 119.9/70 fps @ 517,850 cl (== pre-deletion), cascades 120/71.8 @ 1.43M, Nanite
  view boots — zero regression. `HIER_MAX_DEPTH` default 18 (BFS converges at 9 in bm7 but the safe bound is the leaf-DAG
  ~14-16; `?hierdepth` knob).

Remaining:
1. **`PERF-VB4` — WORLD raster → SINGLE-PASS ✅ DONE + SHIPPED as default; the 2-pass world path is DELETED (LOG bx, NOT
   yet committed). user: "make this the default and drop the 2 pass version completely".** `world1` mode = a 24-bit depth
   `atomicMax` election (`visPayloadV` = `depthKey24<<8|id8`) whose winner stores the full 25-bit id into the side buffer
   `visBV`; the resolve/HZB/shadowHalf decode depth from the key (no exact depthV). **~1.85× on the raster** (2.8–3.5 vs
   ~5.3 ms). KEY FINDING that overturned the D-N45 plan: keeping ANY exact-depth write in the pass (depthV = a 3rd atomic
   storage buffer) is a hard **3× cliff + breaks the kernel's writes** — so depth MUST ride the election key; 16-bit banded
   (user-caught) → 24-bit fixes it free. Residual: <0.1% wrong-cluster speckle very close to objects (user-accepted; the
   zero-speckle fix needs native 64-bit atomics, not in browsers). frameMs is CPU-bound here ⇒ the win is GPU headroom, not
   fps. See D-N45.
2. **TERRAIN-RW tail:** skirt depth → error-sized (∝ measured edge error, not the fixed `24+12·level`).
3. **N9 cross-instance MERGE (#48)** — the proper far-field bound (render distant forest as merged super-clusters instead
   of dropping it at the envelope) ⇒ removes the per-instance floor so instMinPx isn't a density/fps trade.
4. **CLUSTER FLOOR / impostor far-field** — the established big perf lever (cut on-screen triangle count).

(Prior frontier, still open under N9: foliage-as-geometry. N9-C0 landed OKAY — see SPEC `### Foliage (N9)` + LOG bl.)
- **N9-C0 DONE (LOG bl):** the hero `foliageMesh` renders through the nanite path as MATERIAL_CLASS.leaf — lit (isL
  resolve: tint+hue+AO+warm backlight, OPAQUE, double-sided via geometry-dup), per-leaf flutter on the FULL
  vegWindOffset synced to the trunk (shared world-pos wind key), `?naniteleaf=1` / `?naniteleafdensity=N` (default 4000),
  same look with/without `?nanitedag`. 3 bugs fixed (wgcache `flutBase` slot = boot crash; HW_CAP 262k→2.1M = the needle
  HW-queue overflow behind "dag/terrain vanish"; wind instId→world-pos sync). The crown is LESS FLUFFY than non-nanite
  (DEFERRED → `N9-C0b`, a GENERATION rethink, AFTER core nanite — old hero leaned on D-N3-banned alpha CARDS).
- **NEXT — USER-DIRECTED (2026-06-15): DRIVE AUTONOMOUSLY + FAST. Do NOT ask for already-decided things; do NOT
  over-investigate/guess; make the call and proceed. The user flagged the pace as "shockingly slow."** Run this DAG
  in order without checkpoint-pausing:
  1. **N9-C1 ✅ DONE (LOG bm)** — the aggregate DAG builder: net-new `BuildAggregateDag.ts` + shared `DagCommon.ts`,
     node-validated (`probe-aggregate`: crack-free M/C/E/O/A + AREA 1.000× LOD0 at all distances = no balding,
     deterministic). The hard part is behind us. → **C2 IS NOW THE FRONTIER.**
  2. **N9-C2 — cut WIRED + VALIDATED (LOG bn); the flood FORCED N8-HIC (now active, below).** Aggregate attached to
     each leaf crown, envelope R0_FAR 26 m → TREE_GEO_FAR 496 m; `probe-leafzoom` τ-monotonic (2.8k→3.76M) + smooth +
     no errors. The two-sided raster **N9-C2-2s ✅ DONE (LOG bp** — geometry-dup dropped, leaf tris/clusters halved).
     PENDING after N8-HIC: the Worker build (15.5 s sync @ 4000 → off-thread), perf ledger + close.
  3. **N9-C3** — impostor retirement: DO the judge shots + present them at the close, don't block per-ring on approval.
  4. **N9-C4** — close (perf ledger, battery, two-frame-vs-main gate).
  5. **N8-HIC ⬅ FRONTIER — REDEFINED by D-N43 (deep research + measurement, LOG bo).** Root-caused: the flood is
     primitive OVER-EMISSION (≥1 cluster per visible instance × ~340k visible crowns ⇒ ~16 tris/px vs the ~1 of a
     correct Nanite; τ-sweep proves frame ∝ visible-CLUSTER count). The reference-is-10×-faster puzzle is SOLVED (its
     "billions" = a marketing denominator; structurally-easy scene). The fix = cross-instance AGGREGATION (not culling),
     STAGED with a compact between each: **(0) two-sided raster fix ✅ LANDED (LOG bp)** = free lossless 2× (`N9-C2-2s`):
     leaf registry tris 13.171M→6.585M (2×), leaf clusters 131.8k→66.3k, forest vis-cluster overflow→1.499M, no holes,
     tsc clean → **(0.5) ✓ DONE (LOG bq)** — perf SIM (banded-τ `simBandD` + `visTris`/`dagTris` counters + probe-simband)
     + 3-agent integration study. THE FINDING: the flood is per-CLUSTER OVERHEAD, not triangle density — LEAF floor =
     **0.147 tris/px (sub-pixel)** at 1,667 clusters (569k→1.7k, 341× LOD range); SW raster ∝ visClusters (≈55 ns/cl). ⇒
     cross-instance MERGE alone likely SUFFICES; **VOXELS DEFERRED** to the far tail (updates the bo "couple them"). Win
     bounded ~4-5× (42→~8-10 ms). Integration: super-clusters = synthetic identity-instance meshes + attachDag on the
     EXISTING buffers ⇒ ZERO new buffers/bindings/cull-path; build reuses DagWorkerPool + TerrainStreamer pacing →
     **(1) Stage 1 = MULTI-LEVEL cross-instance MERGE builder, MERGE-FIRST (voxels deferred) (NEXT)** — recursive merge
     bands (cell→region→…), which the existing arbitrary-depth DAG cut already selects for free; design the region records
     for N levels from the outset (user 2026-06-15). BINDING: the SEPARATION PRINCIPLE (nanite self-contained in
     `src/nanite/`, CALLED BY world/scene code, no creep out). Stage-1 runtime de-risked by Stage 0.5 = "more meshes +
     more DAG parent links."
  6. **N8-2B4** — always-resident coarse terrain base (teleport no-hole backstop).
  **The user explicitly DEPRIORITISED polish (shadow S4, N6/N7, C0b fluffiness) BELOW the core DAG-culling pieces —
  do NOT pivot to polish until N8-HIC + N8-2B4 are done.** un-black-slating / two-frame-vs-main gate re-applies at C4.
- **N8-D1e LEFT AT ITS CHECKPOINT (validated + measured, LOG bj / D-N41) — 3 pending USER decisions, NOT blocking N9:**
  (1) default-on **rock+deadwood DAG** (free no-pop — recommended); (2) **bark DAG** stays opt-in until **N8-HIC** (the
  hierarchical instance cull — N9-C2 will tell us if it's needed); (3) explicit-DAG **Worker build** only when a class
  goes default. I did NOT flip defaults (visible-everywhere = user-present rule). bark DAG = ~1.7× raster in dense forest
  (the per-instance floor, τ/minPx can't fix); rock+deadwood DAG = free.

**JUST CLOSED earlier this session (LOG bd–bi) — durable, don't re-derive:**
- **PERF-4 (post chain) ✅** — THE finding (LOG bg, high-res GPU-bound ablation ×2): **AO ≈ 100% of the real post
  cost** (removing AO ≈ removing all post: frameMs 33.4→25.0 vs →24.9, 3888×2520); **bloom + TAA + aerial + clouds
  ≈ 0.1 ms combined** — their per-pass timestamps OVERCOUNT ~7× (the passes overlap on the GPU). AO optimized +
  SHIPPED (early-out + packed-view-z bilateral + samples 6, ~1.5 ms direct, UNCONDITIONAL — A/B flags + slow path
  deleted); bloom = mirage (reverted), TAA fork = non-win (removed). Methodology in SPEC `## PERF METHODOLOGY`.
- **AUDIT-1 ✅ (LOG bh)** — impl FAITHFUL to the original Fable 5 spec (two-phase occlusion, Option C full-f32
  vis-buffer, fixed-point edges, near→HW, registerMesh/bindInstances all verified); every deviation D-N*-justified;
  one drift (per-instance TINT) found + FIXED (AUDIT-1a, LOG bi — `slotHash(instId,17/91)` restored on bark/deadwood).
- **PERF-3 CLOSED** — win #1 makeCtx cache LIVE (default on); win #2 vertex cache off-by-default (non-win, D-N40).
- The default `?scene=world&nanite=1` carries ALL shipped wins (no flags needed).
**OTHER FRONTIERS (deferred):** shadow S-stack (mostly polish; S4 the real perf is gated on N9), N6 (opaque migration
— but the missing density is foliage/grass = N9/N10), N8-D2 Stage 2b-4 (teleport no-hole coarse base).

**PERF-1 (LOG `ay`): per-pass measurement is now TRUSTWORTHY, the PURE nanite
renderer is ISOLATED (`?pure`), and the user's WORST view is decomposed with cool numbers.**
The first real dump exposed two integrity defects + one methodology error, all fixed:
- the harness was LYING — dead-CSM garbage NEGATIVE timestamps poisoned the `render` total
  (−97 ms) so `--gpusample` returned 0 samples; `GpuProfiler` now rejects non-finite/negative.
- `?pure` (master ablation = postmin + nanshadow=0 + nandbg=flat; strips beauty, KEEPS
  geometry) fixes the user's "?pure = zero terrain" (it was never wired) and isolates the floor.
- I'd been measuring bm7 = forest INTERIOR (cheapest view). `probe-worstpos.ts` boots the
  user's worst pos (cam −4.2,303.1,−1.4 @T11) and yaw-sweeps to the "long alley".

**THE FINDING (cool, trustworthy):** worst view = 82k visClusters / 130k hwTris. PURE nanite
SW raster = depth **2.82** + payload **2.95** = **5.77 ms** (+ HW 2.62 + flat resolve 2.10);
**fps 35→95 just by stripping post.** AND the post chain THERMALLY THROTTLES the GPU **~2.1×**
(the SAME raster reads 5.96 ms hot). ⇒ (a) the SW depth+payload raster is the #1 nanite lever;
(b) post throttles nanite on top of its own cost.

**PERF-3 ANALYSIS DONE (LOG `az`):** `nanRasterDepth` per-TRIANGLE bound — makeCtx 0.46 (16%), 3×
fetchWorldVert 1.11 (39%), edge 0.13 (5%), per-pixel loop 1.12 (40%); atomic NOT it (depth≈payload).
**WIN #1 LANDED (LOG `ba`): per-cluster makeCtx CACHE in workgroup shared memory** — 1 workgroup == 1
cluster, so compute makeCtx once (thread 0) + broadcast via `workgroupArray`/`workgroupBarrier`. **−0.59 ms
(−11%)** on the camera SW raster (alternated, bit-identical), default ON (`?wgcache=0` A/Bs); also speeds the
6 shadow rasters. First workgroup-shared-mem use in the codebase.
**→ ACTIVE: PERF-3 win #2 — per-cluster VERTEX-TRANSFORM CACHE via build-time COMPACTION** (user-confirmed
"full compaction straight away"). `?vrange` data: explicit redund 4.13× (95% range ≤128), HF-DAG 3.76× (40%
range >1024) ⇒ runtime range-cache FAILS terrain; workgroup atomics unsupported ⇒ only compaction generalizes
(cache sized by vertCount ≤~190 fits 16 KB shared mem for ALL geo; race-free strided transform, no atomics;
bonus local-index memory win ~−100 MB). DESIGN: PARALLEL `gpu.vcompact` buffer (2 u32/cluster: vertBase,
vertCount; 0 ⇒ per-thread fallback) — NOT a CLUSTER_WORDS change (would shift every ci·8 offset). Gated
`?vcompact`, compacted PER PACK PATH so the kernel handles compacted-or-not per cluster (incremental, always
self-consistent). STAGES: **(1) ✅ DONE (committed):** explicit-path `[vMin,count]` cache — `gpu.vcompact`
buffer + `populateVCompact()` (range over each boot cluster's indices, ≤VCACHE_VERTS=192 → store, else
0=fallback; window-grid + streamed terrain stay 0). Explicit uses the EXISTING tight ranges ⇒ NO duplication
(terrain = the later true-compaction stage). Nothing reads it yet ⇒ render bit-identical, boots clean (87.9k
visCl). **(2) ⬜ NEXT = THE WIN — kernel cooperative transform, gated `?vcompact`. (2a)** refactor NaniteFetch:
extract `fetchWorldVertByIndex(ctx, vi)` + a shared `hfWorld(ctx,sx,sz,skirtDrop)` from `fetchWorldVert`
(window-grid stays inline). BEHAVIOR-PRESERVING — `fetchWorldVert` is shared by resolve/shadow/hzb/raster, so
A/B the DEFAULT path bit-identical (screenshot) before/after the refactor. **(2b)** in NaniteRaster, read
`gpu.vcompact[ci]`→(vMin,count); count>0 ⇒ cooperatively transform [vMin,vMin+count) via
`fetchWorldVertByIndex` into `workgroupArray('vec3', VCACHE_VERTS)` (thread t: vMin+t,+128,…), barrier, tris
read `shVerts[vi−vMin]`; else fall back. Bind `gpu.vcompact` in the RASTER ONLY (conditional — avoids the
resolve 10-buffer ceiling). Validate bit-identical + measure the NET (ctx-cache net was BELOW gross — same
barrier+shared-read overhead, so confirm the win is real). → **(3)** terrain true compaction (duplication +
pool-cap growth) → **(4)** narrow local indices (memory). Owed: COOLED absolute-ms batch.

## Phases (coarse status — see SPEC `## Phase plan`)
N0 scaffold ✅ · N1 clusterize ✅ · N2 cull ✅ · N3 vis-buffer ✅ · N4 materials ✅ ·
**N5 shadows 🔵** (clipmap built, perf parity) · N6 opaque pools ⬜ · N7 hybrid close ⬜ ·
**N8 DAG 🔵** (explicit + terrain done; close pending) · N9 foliage DAG ⬜

## A. MEASUREMENT + CORE RASTER — immediate priority (user 2026-06-14)
| id | task | status | blockedBy | spec | scope (one line) |
|----|------|--------|-----------|------|------|
| `PERF-1` | Trustworthy per-pass measurement + `?pure` | ✅ | — | LOG `ay`; GpuProfiler/main.ts | DONE 1f2fdbc — hardened GpuProfiler vs garbage −timestamps (harness was lying: render=−97ms→0 samples); `?pure` master (postmin+nanshadow=0+nandbg=flat, keeps geometry, fixes "?pure=zero terrain"); probe-worstpos.ts. KEY: post chain THERMALLY THROTTLES nanite ~2.1×. |
| `PERF-2` | Profile pure-nanite floor + worst-view decomp | ✅ | `PERF-1` | LOG `ay`,`az`; PERF LEDGER | DONE (folded) — worst view 82k visCl: SW raster depth 2.82 + payload 2.95 = 5.77ms, HW 2.62, flat resolve 2.10 (cool). SW depth+payload = the #1 nanite cost. |
| `PERF-3` | Depth-rasterizer optimization (shared-mem caches) | ✅ | `PERF-2` | LOG `az`,`ba`,`bc`; D-N40; NaniteRaster/VertexCache | CLOSED. WIN #1 LANDED (`?wgcache` default ON): per-cluster makeCtx cache → **−0.59 ms (−11%)** camera SW raster (bit-identical, alternated) + the 6 shadow rasters. WIN #2 (vertex cache) BUILT + MEASURED = marginal/conditional non-win (R≈4.7 vs makeCtx R=128; far-terrain transform texture-cache-absorbed) → kept OFF-by-default, isolated to `NaniteVertexCache.ts`. In-kernel raster wins exhausted. |
| `PERF-4` | Post-chain optimization (the 2.1× thermal throttle) | ✅ CLOSED | `PERF-3` | SPEC `## PERF METHODOLOGY`; LOG bd–bg; PostStack/Gtao | **CLOSED.** DEFINITIVE finding (LOG bg, high-res ablation ×2): **AO ≈ 100% of real post cost; bloom/TAA/aerial/clouds ≈ 0.1 ms combined** (per-pass spans overcount ~7×, the passes overlap). AO ✅ SHIPPED + PERMANENT (`1db0bfd`): early-out + packed-view-z bilateral + samples 6 = ~1.5 ms direct; A/B flags + slower path deleted (bg). BLOOM ✅ = not optimizable (drain, reverted). TAA ✅ = measured non-win (~0.45 ms, ALU-bound), fork removed (bg). Quarter-res AO declined by user. Further post = beauty-trading (declined). |
| `PERF-4-TAA` | TAA resolve fork — built, measured non-win, REMOVED | ✅→deleted | `PERF-4` | LOG bf/bg | **DONE — removed.** Built `LeanTraa.ts` (user-sanctioned fork: subclass + faithful resolve copy, neighborhoods shrunk). Measured ~0.45 ms native (NOT ~3 ms — `TRAANode.resolve` is ALU+drain-bound, not fetch-bound; cutting fetches saves ~0). Deleted in cleanup (bg) — not worth a vendored ~240-line library fork for sub-ms. |
| `AUDIT-1` | Deviation audit vs original Fable 5 spec | ✅ DONE | — | LOG bh; `reference/fable5-original-NANITE.md` | **FAITHFUL.** Core technical contract honored (two-phase occlusion, Option C full-f32 vis-buffer, fixed-point edges, near→HW, HW writes same buffer, registerMesh/bindInstances, wind-phase variation). All deviations D-N*-justified (shadows D-N28/29, black-slate D-N21, terrain-lighting D-N22, velocity D-N16, flat-cut D-N31, terrain-DAG D-N32+). Gaps = unreached phases (N6 partial, N7 deferred by black-slate, N9–N11 pending). ONE drift → `AUDIT-1a`. META: two-frame-vs-main gate re-applies at N7/N10. |
| `AUDIT-1a` | Per-instance TINT drift — ratify or restore (USER CALL) | ⬜ | — | LOG bh; NaniteResolve/NaniteFetch | Orig variation law needs BOTH `tint=slotHash(slot,17/91)` + `windPhase=slotHash(slot,211)` "or migration clones trees (banned)". Impl reproduces the wind phase but NOT the tint — bark hue is per-VERTEX `vdata.x` (shared across a mesh's ~4k instances). Trees vary by pose+wind, not colour. RESTORE = add `slotHash(instId,17/91)` to the bark/deadwood albedo (~few lines), or RATIFY if pose+wind+per-vertex hue reads varied enough. |
| `PERF-VB4` | WORLD raster → SINGLE-PASS (drop the 2nd raster pass) | ✅ | LOG bx | **D-N45** | SHIPPED as default; 2-pass world DELETED (mode 'payload', `kRasterDepth2`, HW-payload, `?vb`/`?vbdepth`/`?nanhw`, world `?audit`). `world1` = a 24-bit depth `atomicMax` election (`visPayloadV` = `depthKey24<<8\|id8`) → winner `atomicStore`s the full 25-bit id into `visBV`; resolve/HZB/shadowHalf decode depth from the key (`cz = 1−(key>>8)/16777215`). **~1.85× raster** (2.8–3.5 vs ~5.3 ms). The D-N45 plan (recompute exact depth in the resolve) was OVERTURNED: any exact-depth write = depthV as a 3rd atomic storage buffer = a hard **3× cliff + broken kernel writes** (three.js/Metal), so depth rides the election key. 16-bit banded (user-caught grazing terracing) → 24-bit = sub-pixel, free. Residual <0.1% wrong-cluster speckle very close to objects (user-accepted; zero-speckle needs native 64-bit atomics). frameMs CPU-bound ⇒ GPU headroom, not fps. KEPT: mode 'depth'+depth1+hwDepth (shadows), mode 'combined'+audit (NaniteView debug). |

## B. DAG (N8) — active workstream (SPEC `### DAG (N8)`)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `N8-D1e` | Full-world DAG wiring + ledger + CHECKPOINT | 🔵 | — | D-N41; LOG bj | VALIDATED (bark/deadwood/rock no-pop gate green, bark under wind) + MEASURED (rock+deadwood DAG free; bark ~1.7× raster + 3 s boot = the per-instance forest floor, τ/minPx don't help) + ledger row. AT the USER CHECKPOINT: (1) default-on rock+deadwood DAG? (free, rec) (2) bark stays opt-in until N8-HIC? Defaults NOT flipped (user-present rule). |
| `N8-HIC` | Cross-instance AGGREGATION + opaque voxel far-field (THE flood fix) | 🔵 | — | **D-N43**; LOG bo, bn | **REDEFINED by D-N43 (research + measurement): "culling" was the wrong word — the fix is cross-instance AGGREGATION, not culling.** Root cause = primitive OVER-EMISSION (~16 tris/px vs ~1; per-mesh DAG floors at ≥1 cluster per visible instance; τ-sweep proves frame ∝ visible-CLUSTER count, per-cluster raster overhead). Reference-is-fast puzzle SOLVED (its "billions" = marketing denominator; easy scene). STAGED: **(0)** two-sided raster fix = free 2× (`N9-C2-2s`) **✅ LANDED LOG bp** (leaf tris/clusters halved, no holes) → compact → **(0.5) ✅ DONE LOG bq** perf SIM (banded-τ) + 3-agent integration study: flood = per-CLUSTER overhead NOT tri-density (LEAF floor **0.147 tris/px sub-pixel @ 1,667 cl**, 569k→1.7k = 341× range; SW raster ∝ visClusters), ⇒ merge alone suffices, **VOXELS DEFERRED** (updates bo "couple them"); super-clusters = synthetic identity meshes + attachDag on EXISTING buffers = ZERO new GPU plumbing; build reuses DagWorkerPool + TerrainStreamer pacing → compact → **(1) NEXT** MULTI-LEVEL cross-instance MERGE builder, **MERGE-FIRST** (voxels deferred to the far tail; recursive merge bands cell→region→… per user 2026-06-15; the arbitrary-depth cut selects them free). SEPARATION PRINCIPLE binding (nanite stays self-contained, called by others). Effort = hours of LLM grind/stage, not weeks. Stage-1 runtime de-risked by Stage 0.5 = "more meshes + more DAG parent links". |
| `N8-2b4` | Always-resident coarse terrain base | ⬜ | — | DAG (N8) | teleport no-hole backstop ring |

## C. POOLS / HYBRID / FOLIAGE (SPEC `## Phase plan`)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `N6` | Migrate remaining opaque pools (debris) | ⬜ | — | Phase plan N6 | register debris pool → DAG applies on registration |
| `N7` | Hybrid close | ⬜ | `N6` | Phase plan N7 | finish the HW/SW hybrid envelope |
| `N9` | **Foliage as REAL geometry** (SCOPED — D-N42, SPEC `### Foliage (N9)`) | 🔵 | — | D-N42; `### Foliage (N9)` | surface the existing `foliageMesh` + the aggregate DAG; the DOMINANT shadow casters; unlocks `S4`. Chunks ↓ |
| `N9-C0` | Leaf PLUMBING (material class + 'leaf' channel + hero-ring reg) | ✅ | — | `### Foliage (N9)`; LOG bl | LANDED, OKAY state (user-accepted). Real crowns ≤26 m: isL resolve (tint+hue+AO+backlight, OPAQUE, double-sided), full-vegWindOffset 'leaf' channel synced to trunk via shared world-pos key, `?naniteleaf=1`/`?naniteleafdensity=N`. Bugs fixed: wgcache `flutBase` slot (boot crash), HW_CAP 262k→2.1M (needle HW-queue overflow = the "dag/terrain vanish"). Same w/wo dag. |
| `N9-C0b` | Leaf GENERATION rethink — fluffiness (DEFERRED, user, post-core) | ⬜ | — | `### Foliage (N9)` N9-C0 LANDED note | nanite crown LESS FLUFFY than non-nanite (old hero leaned on D-N3-banned alpha CARDS; conifer spray distribution spruce≠pine). GENERATION question (denser/bushier sprays) OR the aggregate fills it. NOT plumbing. Do AFTER core nanite. |
| `N9-C2-2s` | Two-sided raster (per-mesh flag + back-face vert-swap) | ✅ | `N9-C0` | `### Foliage (N9)`; **LOG bp**, D-N43 Stage 0 | DONE 2026-06-15. General `MESH_FLAG_TWO_SIDED` bit + `orientForRaster` (re-wind back-face to CCW in the SW core) + HW `DoubleSide`; geometry-dup dropped. MEASURED: leaf registry tris **13.171M→6.585M (2×)**, leaf clusters **131.8k→66.3k (1.99×)**, forest vis-cluster overflow→1.499M; visual A/B identical (no holes), `leaf OFF` byte-unchanged, tsc clean. Shadow raster reuses the core ⇒ leaf shadows two-sided free. |
| `N9-C1` | AGGREGATE DAG builder (area-preserving leaf removal) | ✅ | `N9-C0` | D-N3; LOG bm; `BuildAggregateDag.ts` | DONE — net-new `BuildAggregateDag.ts` (+ shared `DagCommon.ts`; `probe-dag` still green). Per level: global connected-component islands → seed-det area-removal → grow survivors `g=√(total/kept)` (area EXACT) → per-group reclusterize w/ bit-exact sibling pairs. `probe-aggregate`: M/C/E/O/A crack-free + AREA 1.000× LOD0 at ALL distances (no balding) + 50%/level + deterministic + `?seed`-varied. Boot ~0.34 Mtri/s → C2 needs the Worker/time-slice path. |
| `N9-C2` | Wire aggregate → GPU (continuous leaf LOD, full distance) | 🔵 | `N9-C1` | `### Foliage (N9)`; LOG bn | CUT DONE — `buildAggregateDag` per crown, `attachDag`, envelope→TREE_GEO_FAR; `probe-leafzoom` τ-monotonic (2.8k→3.76M) + smooth + no errors. RE-MEASURED the floor → **the flood is real → N8-HIC FORCED** (now active). PENDING after HIC: Worker build (15.5 s sync @ 4000), two-sided raster (N9-C2-2s), perf ledger, close. |
| `N9-C3` | Impostor retirement (ring-by-ring, **USER JUDGE SHOTS**) | ⬜ | `N9-C2` | `### Foliage (N9)` | A/B real crowns vs cards+CanopyShell at vistas; retire where user signs off; CanopyShell dies after vista shots |
| `N9-C4` | Close — perf ledger + battery + two-frame-vs-main + CHECKPOINT | ⬜ | `N9-C3` | `### Foliage (N9)` | un-black-slating starts; two-frame gate re-applies (AUDIT-1 META); gallery A/B per species |

## D. SHADOWS (S-stack) — clipmap banked, perf deferred below core raster (SPEC D-N29)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `S3` | Screen-density shadow clipmap | ✅ | — | D-N29(1) | DONE 6154604 — `NaniteShadowClip.ts`, `?shadowclip` default on |
| `SHADOW-HIER` | Shadow culls → hier BFS + delete brute | ✅ | — | LOG bw | **DONE (uncommitted).** Both shadow culls run the hier BFS; the brute cull path is fully DELETED (NaniteCull 975→643, chunk/reject buffers gone, `?shadowhier`/`?shadowoccl`/NaniteView `?hier` retired). PARITY EXACT, visual A/B identical, validated post-deletion (zero regression). `probe-shadowhier`/`probe-hierdepth`. |
| `S3-perf` | Shared cross-level clipmap cull (`NaniteClipCull.ts`) | ✅ | — | D-N29; LOG bw | **DONE (uncommitted) — the brute-deletion unlock.** ONE hier traverse (largest re-rastering level's box, R1-gated) + cheap per-level frustum+hollow FILTERS replace N full traversals. Tied-or-FASTER than brute (bm7 −0.4 noise, vista +0.3, static identical), ~3.6× lighter. Caveat: assumes one τ/level (per-level-τ S4 would break the shared cut). REMAINING: variable-T per level (a fill-cost lever, separate). |
| `S1` | WPO-freeze / static-dynamic split | ⬜ | — | D-N29(2) | fixes stale static-camera wind shadows |
| `S4` | DAG-decoupled caster coarsening | 🚫 | `N9` | D-N29(2) | full value needs foliage DAG; minPx+DAG on the clipmap's coarse far levels |
| `S5` | Capsule-SDF + contact shadows | ⬜ | — | D-N29(5) | beauty ceiling, optional |
| `S-cloud` | Sever CSM fully (re-source cloud gate) | ⬜ | — | D-N29 | drop three CSM from the nanite path; `world.csm` is only the cloud-gate carrier today |
| `S-cov` | Far-backstop shadow level (>384 m) | ⬜ | `S3-perf` | D-N29 | cheap cached coarse level for distant vistas (clipmap covers 384 m vs cascades' 3200 m) |
| `S-test` | Broad clipmap validation | ⬜ | — | D-N29 | all bookmarks + walk-mode + low-sun (so far only bm3/bm7 static+moving) |

## Recently completed (newest first — detail in LOG)
- `S3` clipmap (ax, 6154604) · Shadow S0 half-res sample (au) · S4 caster-LOD knobs (av,
  minor) · S2-OCCL occlusion (aw, weak/off) · N8-D2 terrain DAG + streamer (aj–ar) ·
  N8-D1a–e explicit-mesh DAG (ag–ah) · N8-D0 QEM build (af) · N4 materials complete (p–x).

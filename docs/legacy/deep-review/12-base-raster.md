# base-raster deep review (2026-07-02)

Area: base mesh SW raster (trunks/bark/branches + near leaf mesh) — election inner loop,
per-tri setup, sub-pixel emit share, trunk DAG structure, bark caps.
Files: `src/nanite/NaniteRaster.ts`, `src/nanite/BuildDag.ts`, `src/nanite/Clusterize.ts`,
`src/nanite/DagCommon.ts`, `src/nanite/NaniteCull.ts` (the cut), `src/debug/ForestScene.ts`
(the scene wiring that bounds what "base" even is).

TL;DR: **The two headline premises of this focus are stale.** (1) The trunk DAG *does*
coarsen now — every bark DAG collapses to 1 root (connected-bark cap-only, `0f77edc`,
2026-06-27); the "97% sub-pixel emit / 46M tris" era is dead. (2) In the DEFAULT config
bark does NOT render "at all distances" — `ForestScene.ts:388` clamps every bark head to
`maxDist = aggdist = 140 m` when fartiles is on (default). Consequently the noleaves
run **overstates** the default frame's base share at eye/oblique, and the mesh raster is
**not** where the oblique −11..13 ms lives. Remaining in-area quality-identical levers
are small (≤ ~1–2 ms total, eye-skewed); the area's real deliverable is the corrected
attribution + probes to pin the residual split.

## Premise audit

1. **"Trunk far field never coarsens" — FIXED since 2026-06-27, before this review.**
   The 2026-06-26 handoff (`docs/perf-runs/2026-06-26-base-raster-review-handoff.md`)
   measured 8.6–32.6M trunk tris because bark tube meshes had open base rings → QEM
   boundary-lock left 98 roots/18,119 tris per beech. Commit `0f77edc` closes every ring
   with a buried cap; boot logs in this session's probe JSONs confirm **every bark DAG is
   1 root** (rarely 2): `[forest] c0: bark lod0 263→root 1 (550cl/9lvl)` …
   `c2: bark lod0 818→root 1 (1714cl/10lvl)` (from `fresh-voxbocc.json` consoleLines).
   Far-field visTris dropped 31.6× at aerial in that commit's own A/B. Any plan that
   still lists "build trunk far-field DAG coarsening" as an open big rock is planning
   against a fixed bug.

2. **"Trunks/bark render through the mesh DAG path at all distances" (fact pack) — WRONG
   for the default config.** `ForestScene.ts:386-390`: with fartiles on (default),
   `for (const m of meshes) reg.setMaxDistance(m.bark, aggDist)` → bark ends at 140 m;
   FarTiles trunk *columns* (splatted voxels) own trunks beyond. Only `?noleaves`
   (which disables fartiles, `ForestScene.ts:312`) extends bark to its 2000 m envelope —
   and there the per-instance `instMinPx` cull (default `0.075·min(dim)` ≈ 110 px,
   `NaniteFrame.ts:187-191`) pops trees at roughly 200–250 m (projK = cot(27.5°)·1473/2
   ≈ 1415; sizePx = 2·1415·r_eff/d with r_eff ≈ 8–10 m incl. swayPad).

3. **Therefore "pure base ~11/10/11" (noleaves minus post) is an UPPER BOUND that does
   not transfer to the default frame at eye/oblique.** noleaves both *adds* bark
   (140→~230 m ring) and *un-occludes* terrain+trunks (no canopy). Counters prove the
   default frame's mesh emit is small at the money pose: default-oblique visClusters
   11,614 of which **9,185 are voxel clusters** (`fresh-voxbocc.json`) → mesh clusters
   ≈ 2,430; visTris 1.27M of which ~0.62M are voxel *pseudo-tris* (a voxel cluster's
   word-7 low byte is brickCount reused as triCount, `VoxelBrick.ts:79`; cross-check:
   default-aerial visTris 44.7k / 661 all-vox clusters = 67.6 "tris"/cluster = bricks)
   → **real mesh tris at default-oblique ≈ 0.65M** (vs noleaves-oblique 2.11M).
   At default-aerial the mesh path emits **zero** (661 clusters, all voxel — canopy
   HZB-kills terrain, bark ends at 140 m < 150 m altitude).

4. **Metric caveat on `nanite.visTris`:** it sums cluster `triCount` over ALL emitted
   clusters — terrain + bark + leaf + voxel(=bricks). Nothing in the counter set splits
   by matClass. Every "sub-pixel share" claim below is therefore an estimate; the exact
   split needs the per-class counter probe (§Open questions).

5. **Cluster cap is 255, not 128.** `ForestScene.ts:70` and `WorldRegistry.ts:306` call
   `setClusterTriCap(…|| 256)` → `MAX_CLUSTER_TRIS = 255` (`GeometryRegistry.ts:123-133`).
   Observed avg 184 tris/emitted cluster (2.09M/11,375 noleaves-eye) is consistent.
   One SW workgroup = 255 threads = one cluster.

6. **Prior verdicts honored:** atomic contention REFUTED (and re-confirmed here:
   `fresh-base-ng.json` `?noguard=1` ≈ neutral vs `fresh-base-bl.json`); per-pixel ALU
   micro-opts incl. the ≤4px-rect fast path REFUTED (zero delta, 2026-06-26); tiled
   raster REFUTED+REMOVED. The `fresh-base-vc*.json` "5.4 ms" runs are the vcompact
   **empty-scene artifact** (11th storage buffer kills the pipeline; hwTris=0 in those
   JSONs is the tell) — not data.

## How it works today

**Cut → queue.** `NaniteCull.ts` `kSeedRoots` (:760-806) runs per instance (600,812 in
default forest: 3×200k heads + 812 tiles): frustum + draw-envelope (`lodDist`, :776-779)
+ near-envelope (:786-787) + `instMinPx` size cull (:791-795), then appends the mesh's
DAG roots to frontier A. `makeTraverse` (:817-948) BFS-descends: `pOwn =
projK·A.w·ownError/√(d²−r²)` (:835) vs `τ_eff = lodWarp(τ=3, d)` (:836-842) where
`lodWarp` (:111-125) = `τ·(1+((d−4)/6)^0.6)` — τ_eff ≈ 12.5 px @45 m, 18.8 @100 m,
22.5 @140 m (defaults `NaniteFrame.ts:157-177`). Cut-resolved clusters get frustum,
sub-`minPx`(2px), shadow-hollow, backface-cone, and prev-frame-HZB sphere tests
(:861-925), then emit `(instId, ci)` into qRaster; else children enqueue. Voxel(7)
clusters are emitted into the SAME qRaster and later fanned out (indirect-tight,
`NaniteCull.ts:1024-1037`) — they also STAY in qRaster.

**SW raster (world1).** `NaniteRaster.ts` `rasterKernel('world1')` (:417-1015), one
255-thread WG per qRaster item, dispatched indirect at the tight size (:1031). Thread 0
runs `makeCtx` once and broadcasts via workgroup shared memory (wgcache, :439-527).
Voxel clusters bail after the broadcast (:550-555). Per live thread (= per tri):
3× `fetchWorldVert` + 3× VP transform (:634-640), near-plane → HW queue (:668-675),
back/two-sided orient (:688), 1/256-px fixed-point snap (:710-715), bbox (:719-737),
>16 px bbox → HW queue (:729-731, `MAX_RASTER_SIZE=16`), sample-miss cull (:748-761),
edge setup (:764-813), then the scanline: per row a 3-divide x-span solve (:848-873),
per pixel 3 sign tests, and for covered pixels the election: z-interp (:896-901),
relaxed-load guard + `atomicMax(depthKey24<<8|id8)` + winner `atomicStore` of the full
id into visBV (:952-975). `?f2b=1` (default OFF) would hoist a per-pixel prevE read
before the z-interp (:983-989).

**HW pass.** Near-plane crossers + >16 px tris go to hwQueue; one indirect non-indexed
draw, vertex-pulling material (:1104-1199). NOTE the vertex stage fetches ALL THREE
corners then selects one (:1123-1128) — 3× redundant vertex work — and re-runs full
`makeCtx` per VERTEX (no wgcache in the render pipeline).

**Frame order** (`NaniteFrame.ts:445-495`): cull BFS (one batched submit, camera+shadow
cut overlapped) → vox fan-out → world1 (clear+SW+hwArgs in one submit, then HW render,
then vox raster) → HZB build → shadows (cached at static poses) → shadowHalf → resolve+post.

**DAG build.** `BuildDag.ts`: spatial-median groups (`DagCommon.ts:39-78`, ≤24 clusters),
boundary-locked QEM halving per level (`simplifyGroup` :312-911), sibling-pair
(error,sphere) equality for the crack-free cut, stuck-group → roots (:1083-1104).
Bark meshes: plain QEM (`ForestScene.ts:214-218`); leaves: `buildAggregateDag`.
With capped tubes the bark chain reaches 1 root in 6–10 levels (boot logs above);
root ≈ 92 tris for beech.

## Work model

Let P = covered pixels (≈ screen mesh coverage · overdraw), T = SW mesh tris emitted,
C = emitted clusters, H = HW tris.

- **world1 SW ≈ a·C(launch+ctx) + b·T(fetch+setup) + c·P(coverage loop+election).**
  Measured 2026-06-16 (rdbg stage-split): a·C + b·T floor < 8.3 ms even at 3.3M
  clusters/2× res ⇒ at today's C ≈ 11–23k, a·C ≲ 0.1 ms; per-pixel loop ≈ 90% of world1.
  Election atomics ≈ free (noguard neutral: 16.6 vs 15.8 eye, within thermal noise).
- **HW pass ≈ 3H vertex invocations × (makeCtx + 3 fetchWorldVert)** + fixed-function
  raster ≈ 9H vertex fetches. Eye default H = 1.10M (leaf needles) → ~9.9M fetch+
  transforms; oblique 83k; aerial 0.
- **Resolve/post/GI ride on P** but belong to other areas; lighting/shadows/leaf-decode
  measured ≈ FREE (attribution doc §2).

What the numbers explain:
- noleaves eye 16.8 / obl 15.4 / aerial 11.1 with visTris 2.09/2.11/0.56M and ~3.34 Mpx
  coverage at every pose ⇒ pure base (−post 6/5/0) ≈ 11/10/11 tracks **coverage, not
  tris** (aerial has 3.8× fewer tris, same base cost) — consistent with c·P dominance +
  a fixed infra floor (clears 3×3.34M u32 stores, HZB 14-level chain, cull, submits).
- default vs noleaves at eye: +2.3M visTris (near leaf band, H +0.87M) yet only
  +2.1 ms (18.9 vs 16.8) — the leaf band largely *replaces* trunk/terrain coverage
  (P roughly constant), confirming c·P dominance again.
- default-oblique mesh work: T ≈ 0.65M, C_mesh ≈ 2.4k, H = 83k, and canopy owns most
  pixels ⇒ **base mesh raster share at the money pose ≈ 2–4 ms** (bounded above by
  noleaves-oblique pure base 10 ms, which has ~3× the mesh tris and full terrain
  coverage; scaled by T and by P share).

## Waste inventory

Best-effort, default config, per pose (eye/oblique/aerial):

| Waste class | Mechanism | Size today | Note |
|---|---|---|---|
| Sub-pixel/zero-coverage tris | tris covering no sample still pay setup | **small** | sample-miss cull already in the kernel and measured ~neutral when added (NaniteRaster.ts:748-752 comment) ⇒ few zero-sample tris or cheap setup. The 97%-sub-pixel era is over (premise §1). |
| Occluded frags the HZB missed | frags behind the current front pay z-interp + guard-load, lose | ~0.1–0.5 ms | election guard already skips the RMW; only z-interp (~6 ALU) is saveable (`?f2b=1` exists, unmeasured at this operating point) |
| Dead voxel WGs in world1 | 5,935 (eye) / 9,185 (oblique, **79% of qRaster!**) / 661 (aerial) WGs launch, broadcast ctx, and bail (:550-555) | ≲0.1 ms | bounded by the measured launch+ctx floor (a·C ≲ 0.1 ms at 23k clusters); ugly but cheap |
| HW vertex 3×-fetch + per-vertex makeCtx | :1123-1128; 9.9M redundant fetches at eye | ~0.3–1.0 ms eye, ~0 elsewhere | the one real in-area arithmetic waste left |
| Intra-cluster vertex refetch (SW) | 3·184 corner fetches vs ~110 unique verts per cluster (~4.7× redundant) | ~0.5–1 ms conditional | vcache exists, measured marginal (barrier tax vs cheap verts), currently bitrot at the 11-buffer cliff (`NaniteVertexCache.ts:59-66`) |
| Idle WG lanes | avg 184 live of 255 threads (28% idle) | ~0 | triangle-granular re-dispatch measured bounded-marginal 2026-06-16 |
| Stale-HZB over-emit (moving) | prev-frame HZB at cluster grain | live-only, ~1–3 ms transient | pose-teleport aerial ramp confirms class; isolated static poses ≈ unaffected |

**Answer to "how much of remaining eye base is waste":** of eye-default 18.9 ms, base
mesh raster + infra ≈ 10–11 ms; of that, addressable quality-identical waste in THIS
area ≈ **1–2 ms** (HW-vertex fetch + vcache + f2b). The rest is coverage/infra floor
(election on ~3.3Mpx, clears, HZB, cull, resolve decode — resolve is another area).

## Levers

All quality-identical per the ABSOLUTE constraint unless noted. Ordered by expected value.

### L1 — HW vertex stage: fetch 1 corner, hoist nothing else (S)
Mechanism: `buildHwMaterial` vertexNode fetches w0,w1,w2 then selects by `corner`
(NaniteRaster.ts:1123-1128). Fetch only `fetchWorldVert(ctx, localTri, corner)` (index
math is already dynamic per-vertex). Bit-identical output (same selected value).
Expected: eye −0.3..−1.0 ms (H=1.10M ⇒ 6.6M fetch+transforms removed), oblique −0..0.1,
aerial 0. Discriminator: A/B flag `?hw1fetch=1`, eye pose, matched-session order.
Risks: TSL codegen for dynamic corner (should be fine — the fetch already takes a
computed vertex slot); if `fetchWorldVert` specializes on the literal corner, minor
refactor in `NaniteFetch.ts:451`.

### L2 — vcache re-measure after freeing the 11th binding (M)
Mechanism: cooperative per-cluster vertex-transform cache (`NaniteVertexCache.ts`),
bit-identical by construction; currently `?vcompact=1` = empty scene (11th storage
buffer > Metal 10 cliff, :59-66). Fix: fold `gpu.vcompact` (2 u32/cluster) into spare
cluster words or the hwQueue tail (scar precedent, NaniteRaster.ts:339-372). Prior
verdict: marginal/conditional (−1 quantum on wind-trunk forest, neutral on terrain) —
but that was pre-cap-collapse, pre-wind-hoist era; worth ONE re-measure, not more.
Expected: eye −0..−1.0, oblique −0..−0.5, aerial ~0. Discriminator: fixed
`?vcompact=1` vs `0`, noleaves AND default, eye; screenshot-gate (the empty-scene trap).
Risks: net-negative barrier tax on cheap clusters (the prior finding); binding surgery
touches the raster's buffer set (regression risk — validate with `tools/vcdebug.mjs`).

### L3 — `?f2b=1` default flip if it measures free (S)
Mechanism: per-pixel prevE read before z-interp skips the depth work for provably-losing
fragments (loss-exact, NaniteRaster.ts:983-989). Zero quality delta by construction.
Expected: −0..−0.5 across poses (guard already kills the RMW; only ~6 ALU/losing-frag
saved; winners pay a second load). Discriminator: `EXTRA=f2b=1` A/B at all three poses.
Risk: measurably net-negative (extra load per covered px) → keep off; cheap to test.

### L4 — voxel clusters out of qRaster (S-M) — **expected ≈ 0, listed to kill it**
Mechanism: traverse emits matClass-7 straight to qVoxRaster so world1 never launches
their WGs (79% of oblique qRaster items are dead vox launches). Bounded by the measured
launch+ctx floor: ≲0.1 ms even at oblique. Not worth the queue-index surgery (payload
itemIdx must stay consistent for the resolve). Do not build unless a probe shows the
floor grew.

### L5 — trunk far-field DAG coarsening — **ALREADY LANDED (de facto); residual ≈ 0.3–0.5 ms**
The named lever of this focus. Cap-only collapse (shipped) + fartiles bark clamp at
140 m (shipped) already deliver it: default eye bark ≈ trees ≤140 m, far ring emits
~1-root/92-tri cuts; default oblique/aerial bark ≈ HZB-killed/absent. What remains is
sub-root coarsening in the 60–140 m ring where root tris project ~2–10 px — NOT
sub-pixel, so coarsening there is visible in an A/B flip ⇒ fails the quality bar.
Verdict: closed. (Fund the mid-band DETAIL instead, per user directive #2.)

### REJECTED-BY-POLICY (quality-trading; listed for completeness only)
- `?loderr>3` / stronger lodWarp (coarser mesh cut) — visible tri-shape change.
- `instminpx` raise / bark `aggdist<140` — pops/deletes visible trees.
- Sub-root bark impostor/billboard ring (60–140 m) — visible silhouette change.
- `MAX_RASTER_SIZE` tuning to shift needles SW/HW — changes routing only in theory,
  but any coverage change at 16 px boundary is measurable → treat as A/B-gated
  neutral-or-reject; not worth it.

## What UE5/prior art does here

- UE5 Nanite rasterizes with a 64-bit atomic (depth|payload single word); we already
  match the effective behavior with the 32-bit election + side store, and measured
  atomics ≈ free — no gap worth chasing (F1 "single-word experiment" is below noise).
- UE5's SW/HW split is the same idea (small tris SW, big/clipped HW); UE5's HW path
  also pulls vertices but fetches ONE corner per vertex — L1 closes exactly that gap.
- UE5 cluster culling is two-phase (prev-frame HZB accept + current-frame re-test of
  rejects) — ours is single-phase prev-frame (`NaniteCull.ts:917-925`); the delta is
  disocclusion-hole correctness on fast motion, not steady-state perf. Scaffolding
  exists (rejClust buffer). Worth doing for correctness someday, not for the gap.
- UE5's per-cluster programmable raster keeps a vertex cache in groupshared — L2 is our
  equivalent; UE5 wins more because their material eval per-vertex is heavier (higher C
  in the R×C model), which is exactly the prior "conditional win" finding.
- Boundary-locked QEM DAG with group-shared (error,sphere) pairs — we match the
  reference construction (jglrxavpok/zeux #750); the cap-only fix was our missing
  "manifold input" precondition, now met.

## Open questions + proposed serial probes

All `tools/probe-fresh-stutter.ts`, TICKS=0, COOLDOWN_S=45, thermal-ordered
(baseline first). **Do not run in this workflow — GPU is serialized outside.**

1. **Per-matClass emit/frag split (instrumentation, S):** generalize the scar counters
   (already folded into hwQueue tail, zero new bindings) to 4 classes
   (terrain/bark/leaf/vox): frags + final-owned pixels each. Then one run:
   `CONFIG=default LABEL=classsplit EXTRA=scar=1 TREES=200000` → exact base-vs-foliage
   pixel ownership + overdraw per pose. This retires every estimate in §Waste.
2. **Terrain-only floor:** `CONFIG=noleaves LABEL=terrainonly EXTRA=instminpx=9999`
   (drops all tree instances; terrain spheres survive) → pins terrain+infra floor per
   pose; bark share = noleaves − this.
3. **noleaves pixel law:** `CONFIG=noleaves DPR=0.75 / 1.0 / 1.5` → fixed-vs-slope for
   the BASE path alone (the default-config law 13+6.0/Mpx is foliage-contaminated).
4. **L1:** implement `?hw1fetch` → `CONFIG=default LABEL=hw1f EXTRA=hw1fetch=1` vs
   control, eye pose is the money number.
5. **L3:** `CONFIG=default LABEL=f2b EXTRA=f2b=1` (+ noleaves variant), all poses.
6. **L2:** after binding surgery: `CONFIG=noleaves LABEL=vc-fixed EXTRA=vcompact=1`
   vs control; MUST screenshot-gate + check hwTris>0 (the 5.4 ms empty-scene trap).
7. **rdbg split refresh (only if 1-3 leave a residual unexplained):** `CONFIG=noleaves
   EXTRA=rdbg=1|2|3,occl=0` — occl=0 makes geometry identical across variants (kills
   the HZB-feedback confound the 2026-06-26 handoff flagged); compare ratios only.

**Bottom line for the master plan:** the oblique −11..13 ms cannot come from this area;
at most ~2 ms of quality-identical waste exists here (L1+L2+L3), eye-skewed. The gap
lives in voxel coverage + the per-pixel infra floor. The valuable outputs here are the
premise corrections (§Premise audit 1-3) and probe #1, which gives the whole review
exact class-level attribution for one instrumented run.

## Reconciliation & verification (2026-07-02, post-limit continuation)

Scope: this doc (canon) + sibling `03-base-mesh-pipe.md` (fleet A). Every load-bearing
file:line below was re-read from the working tree (HEAD 4ca9cd9); all counter numbers
re-derived from `fresh-voxbocc.json` / `fresh-noleaves-now.json` / `fresh-base-{bl,ng,vc,vc2}.json`.

### Corrections (both docs)

1. **Forest scene has NO shadow/GI system at all — doc 10's flag is CONFIRMED and it is
   stronger than either doc assumed.** `ForestScene.ts:457-459` passes `gi: null,
   canopyTex: null, csm: null` into `buildNaniteFrame`; `NaniteFrame.ts:244` gates
   `shadowOn` on `world.csm !== null` ⇒ `shadow = null`, `shadowHalf = null`. The shadow
   clipmap **never builds in the forest scene** — not "cached at static poses" (this doc's
   frame-order line), not "runs only when moving" (doc 03 §3.5). Every canonical number,
   live or isolated, contains ZERO shadow work; "shadows ≈ free" attribution rows are
   vacuous. Master-plan consequence: the whole budget is provisional against the mission's
   zero-quality-sacrifice law — when shadows/GI are actually wired into the forest, the
   frame gains a currently-unmeasured cost class that no lever in this area offsets.
2. **SW workgroup = 255 threads, not 128** (re-confirming Premise §5 against doc 03 §2.1):
   `ForestScene.ts:70` / `WorldRegistry.ts:306` call `setClusterTriCap(…|| 256)` →
   `MAX_CLUSTER_TRIS = 255` (`GeometryRegistry.ts:123-133`); the kernel dispatches
   `[MAX_CLUSTER_TRIS]` (`NaniteRaster.ts:1015`). Doc 03's "128 threads = 1 tri each" (and
   its rdbg quotes) echo the STALE comments at `NaniteRaster.ts:273-275` / `GeometryRegistry.ts:106`.
3. **Vox fan-out default path cite fixed:** default is `voxf2b` OFF (`NaniteCull.ts:307`,
   `?? '0'`), so the production fan-out is the 3-dispatch unordered path
   `NaniteCull.ts:1042-1047` (kVoxFanoutArgs → kVoxFanout indirect → kVoxRasterArgs) — this
   doc's `:1024-1037` cite pointed at the non-default F2B batch. Doc 03 had it right.
4. **HZB is 12 levels at 2268×1473, not 14** (`NaniteHzb.ts:71-83`: halving chain from
   half-res 1134×737 → 1×1 = 12 levels; ~1.11M texels total stands). `hierDepth ≈ 14`
   (BFS passes, `NaniteFrame.ts:205` = maxDagDepth 12 + 2) is a DIFFERENT number and stands.
5. **`voxTauCap` default is 12 px** (`NaniteCull.ts:317-318`); the "DEFAULT 8 px" in the
   traverse comment (`NaniteCull.ts:843`) is stale. Matches doc 14. Mesh/bark clusters are
   NOT capped (voxel-matClass-only, `NaniteCull.ts:852-859`) — this doc's bark τ_eff numbers stand.
6. **L1 risk is real:** `fetchWorldVert`'s corner is a BUILD-TIME literal
   (`NaniteFetch.ts:124` signature `v: 0|1|2`; def `:451`) — a dynamic-corner call is not
   possible as written. The single-fetch refactor goes through `fetchWorldVertByIndex`
   (`:130`, exists) with the window-grid HF branch kept on the 3-fetch form (no index
   buffer). L1 effort S → S/M. Doc 03 §3.4 already said exactly this; adopted.
7. **kAudit confirmed dead in world1 mode** (doc 03 §3.2): world1's SW election writes only
   visPayloadV/visBV (`NaniteRaster.ts:952-975`), the HW world1 frag likewise
   (`:1177-1183`); visDepthV holds only the kVisClear sentinel (`:391`) ⇒ kAudit
   (`:1046-1057`) can never count. Coverage probes must be shot-based (P5) or scar-based (probe #1).
8. **HW world1 fragment election already carries the relaxed-load guard**
   (`NaniteRaster.ts:1177-1183`) — no missing-guard waste on the HW path; the HW waste is
   solely the 3×-fetch + per-vertex makeCtx (`:1121-1128`).

### Contradictions resolved (canon vs doc 03)

- **"Base is a flat pixel-bound ~10-11 ms pool" (03) vs "base mesh raster ≈ 2-4 ms at the
  money pose" (canon).** Both verified, different objects: the noleaves floor
  (F + kpx·3.34Mpx ≈ 10-11 ms) is coverage+infra and includes full-frame terrain coverage
  + the 140→~230 m bark ring that the DEFAULT config does not render
  (`ForestScene.ts:312,388`); in the default frame those pixels are mostly owned by
  foliage/vox (their coverage cost is foliage-area budget). Transferable to the default
  frame: the infra floor (clears/HZB/cull/submits ≈ small) + mesh emit ≈ 2-4 ms oblique.
  Doc 03's own §3.1 tri-swing disproof (3.7× fewer tris, +1.1 ms) supports c·P dominance;
  numbers re-verified against the JSONs (eye 2.093M/11,375/234k; obl 2.105M/11,935/175k;
  aerial 0.558M/3,180/38.5k; default-obl 1.27M visTris of which ~0.62M vox pseudo-tris,
  9,185/11,614 vox clusters; default-aerial 661 all-vox/44.7k/hwTris 0).
- **f2b expected magnitude: −0.5..−2 (03) vs −0..−0.5 (canon).** Canon stands. The
  election guard (`NaniteRaster.ts:963-974`) already skips the RMW for losing fragments;
  f2b (`:697-703`, `:983-989`) additionally saves only the z-interp + range check
  (~6-10 ALU) per provably-losing fragment and ADDS a redundant load for surviving ones.
  Loss-exact quality claim confirmed in code (atomicMax monotone, `cand ≤ nearKey`).
  Probe stays (it's one flag), expectation is the canon's.
- **Two-pass occlusion (03 L2):** mechanism + buffer budget verified — kTraverse is at 10
  bindings; counters slot 3 is genuinely free (`NaniteCull.ts:742-744`; the "slots 2/3"
  comment at `:738-739` is stale — code uses FA=0/FB=4); reject records CAN fold into the
  qRaster top end (scar-fold precedent `NaniteRaster.ts:339-345`). Kept as a
  quality-IMPROVING lever (motion correctness), NOT a perf lever: isolated cost
  +0.1..+0.3 ms; live ms effect unproven until P3's moving-vs-static visTris ratio.
- **noleaves oblique bimodality (03 Q-A) re-verified from the JSON:** oblique alternates
  7-9.6 vs 15-17.4 ms with voxActive=false (brickCount=0 ⇒ no vox raster/voxOccPyr,
  `NaniteFrame.ts:85`) ⇒ voxOccPyr excluded as sole bimodality cause — stands, routed to
  the bimodality owner. noleaves eye max 17.5 (low outliers only) ⇒ eye p95 spikes need
  foliage — stands.

### New code-confirmed defect (cull-side, cross-cited from doc 11)

**Perspective `sphereOccluded` has no on-screen gate:** `NaniteHzb.ts:158-205` clamps the
footprint to edge texels (`:189-192`) and returns without any `|ndc|<1` test (`:200-204`),
while the ortho variant HAS the gate (`:246-247`). A frustum-SURVIVING cluster whose
sphere straddles the screen edge (center off-screen, `centerClip.w>0`) tests against an
arbitrary edge texel's depth → over-cull → pan/pose-arrival pop-in. This is a
CONSERVATIVE-CULL violation, i.e. a quality bug in the base cull, not a perf lever.
Fix is one line (mirror `:246`); costs only extra survivors. Gate: shotdiff at pan poses
+ visClusters delta. Pairs with (and is cheaper than) the two-pass occlusion lever.

### Surviving lever table (merged, this area)

| # | Lever | Mechanism (1-liner) | eye / obl / aerial (ms) | Quality | Probe (serial queue) | Effort | Conf |
|---|---|---|---|---|---|---|---|
| B1 | HW 1-corner fetch (canon L1 = 03 L5) | HW vertex stage fetches 3 corners, uses 1 (`NaniteRaster.ts:1123-1128`); refactor via `fetchWorldVertByIndex`, keep HF-window branch | −0.3..−1.0 / −0..−0.1 / 0 | IDENTICAL (same selected vertex) | `?hw1fetch=1` A/B eye + shotdiff | S/M | med |
| B2 | `?f2b=1` default flip (canon L3 = 03 L1) | skip z-interp for provably-losing frags (`:983-989`); guard already kills their RMW | −0..−0.5 each pose | IDENTICAL (loss-exact) | `EXTRA=f2b=1` 3 poses + noleaves variant | S | low that it ≥0.5 |
| B3 | vcache re-measure post-binding-fix (canon L2) | cooperative vertex-transform cache; currently 11th-buffer bitrot (`NaniteVertexCache.ts:56-64`) | −0..−1.0 / −0..−0.5 / ~0 | IDENTICAL by construction | fixed `?vcompact=1` vs 0, screenshot-gate + hwTris>0 | M | low |
| B4 | Two-pass occlusion (03 L2) | record HZB rejects → re-test vs fresh pyramid → append raster; slot-3 counter + qRaster-tail records = zero new bindings | +0.1..+0.3 isolated; live: removes 1-frame disocclusion holes | IMPROVING | P3 moving/static visTris ratio first; static shotdiff=0 gate | M | med |
| B5 | `sphereOccluded` on-screen gate (new, from doc 11 verify) | mirror ortho's `onScreen` (`NaniteHzb.ts:246`) into the perspective test | ~0 / ~0 / ~0 (may cost a hair: more survivors) | IMPROVING (kills edge-pop) | pan-pose shotdiff + visClusters delta | S | high (bug), med (visibility) |

Probe #1 (per-matClass emit/frag split via scar-style counters) remains the area's
highest-value deliverable — unchanged, see §Open questions.

**Master-plan budget from this area (reconciled):** quality-identical headroom ≈ 1-2 ms,
eye-skewed (B1+B2+B3); oblique contribution realistically −0..−1 ms. Doc 03's "−2 to −3
oblique" included the resolve-pass probe (`?nores`) which belongs to the resolve area and
an overstated f2b. The oblique −11..13 ms gap is NOT in this area (unchanged conclusion).

### Killed claims (one line each)

- **[03 §2.1] "SW raster = 128 threads/WG"** — 255 (`setClusterTriCap(256)→255`, dispatch `[MAX_CLUSTER_TRIS]` at `NaniteRaster.ts:1015`); stale comments echoed.
- **[03 §3.5 + L3] "shadow clipmap is a live/moving base cost" + `?culloverlap` lever (−0.5..−1.5 live)** — forest passes `csm:null` (`ForestScene.ts:459`) ⇒ shadow system never built (`NaniteFrame.ts:244`); `cullOverlap` requires `shadow?.cullPrepass` (`:260`) ⇒ can never fire. Dead in every canonical config; revisit only if shadows get wired into the forest.
- **[03 L1] "f2b −0.5..−2 ms"** — guard already skips losers' RMW; only z-interp saved ⇒ −0..−0.5 (see resolution above).
- **[03 P3] shRaster live counter half of the probe** — counter never set (shadow null, `NaniteFrame.ts:507`); keep only the visTris/culloverlap-free parts of P3.
- **[canon L4] voxel clusters out of qRaster** — confirmed dead: bail after broadcast (`NaniteRaster.ts:550-555`) bounded by the <2.5 ns/wg launch floor ⇒ ≲0.1 ms at 9.2k WGs; do not build.
- **[both] "HZB 14-level chain"** — 12 levels at the canonical res (`NaniteHzb.ts:71-83`).
- **[canon frame-order] "shadows (cached at static poses) → shadowHalf"** — those passes do not exist in the forest scene (correction #1).
- **[carried from earlier memory, re-killed] "base is TRIANGLE-EMIT-bound / 46M tris / 97% sub-pixel"** — bark 1-root collapse re-confirmed in boot logs (`[forest] c0: bark lod0 263→root 1 (550cl/9lvl)`, fresh-voxbocc consoleLines); tri-swing disproof re-verified.
- **[both] `fresh-base-vc*.json` 5.4-5.7 ms runs** — vcompact empty-scene artifact re-confirmed (hwTris=0 in both files); not data.

### Verify stamp (post-limit continuation, independent second pass)

Every load-bearing file:line cite and every JSON-derived number in this section was
independently re-verified against HEAD `cd14cc2` (code unchanged since `4ca9cd9` — the
delta is docs-only): `ForestScene.ts:70/:312/:388/:457-459`;
`NaniteFrame.ts:85/:205/:244/:260/:507` (cullOverlap kill confirmed: `:260` requires
`shadow?.cullPrepass`, forest `csm:null` ⇒ `shadow=null`);
`NaniteCull.ts:307/:317-318/:843/:852-859/:917-925/:741-744/:1044-1046`;
`NaniteHzb.ts` 12-level chain (1134×737 halving → 1×1 = 12) + perspective
`sphereOccluded` confirmed WITHOUT the `|ndc|<1` gate (returns `dist>2r ∧ nearClip.w>0 ∧
centerClip.w>0 ∧ nearestZ>maxZ`, edge-texel clamp only) while ortho HAS `onScreen`
(`:246-247`); `NaniteRaster.ts:343/:391/:550-555/:697-703/:963-989/:1015/:1046-1057/`
`:1123-1128/:1177-1183`; `NaniteFetch.ts:124/:130/:451` (corner is a build-time literal;
`fetchWorldVertByIndex` exists; HF window-grid has no index buffer);
`GeometryRegistry.ts:123-133`; `WorldRegistry.ts:306`; `NaniteVertexCache.ts:56-64`;
`VoxelBrick.ts:79-80` (word-7 low byte = brickCount). Counters re-extracted from the
JSONs and all match: voxbocc eye 22,897 cl / 5,935 vox / 4.428M tris / 1.099M hw;
oblique 11,614 / 9,185 (79.1%) / 1.271M / 82.9k; aerial 661 all-vox / 44,693 / hw 0
(44,693/661 = 67.6 bricks/cluster); noleaves 2.093M/11,375/234k · 2.105M/11,935/175k ·
0.558M/3,180/38.5k; vc/vc2 hwTris=0 (artifact confirmed); bark `lod0 263→root 1` boot
lines present. Derived arithmetic checked: oblique vox pseudo-tris 9,185×67.6 ≈ 0.62M ⇒
real mesh tris ≈ 0.65M; eye default−noleaves = +2.33M visTris / +0.865M hwTris.
**Zero corrections needed on the second pass; the section stands as written.**

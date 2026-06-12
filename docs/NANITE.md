# NANITE WORKSTREAM — operating manual (source of truth on branch `nanite-raster`)

> **Rehydration protocol (branch)**: read THIS FILE fully → `STATUS.md` (main-project
> context; do NOT update it for nanite work) → `docs/THREE-NOTES.md` (API gotchas) →
> `reference/three.js webgpu - compute rasterizer lighting.html` (the proven TSL
> reference implementation — re-read the kernels before touching raster code).
> Continue from **NEXT ACTIONS**. Never re-plan from scratch. Update PROGRESS LOG +
> NEXT ACTIONS every session; commit per milestone with measured numbers in the message.

## Mission

Replace per-system hardware rendering with a Nanite-style GPU-driven pipeline —
cluster hierarchy, HZB occlusion, compute/hardware hybrid visibility-buffer raster,
screen-space material resolve, continuous LOD DAG — staged through a hybrid that is
shippable at every phase boundary. Then raise ground vegetation/stone/debris density
≥ 5×. End state (user mandate, binding): **ONE standardized geometry path** — new
generated meshes are *registered* (mesh + material class + instance stream) and the
engine clusterizes/LODs/culls/rasters/shades them with **zero new render code**.
"An engine underneath that we barely need to touch."

Why (measured, from STATUS): cpu.submit 11–15 ms (draw-count-driven) and r.scene
~12 ms are the binding constraints at the user viewport; the 120 fps directive needs
both collapsed. 5× density is unreachable on the alpha-overdraw hardware path.

## Binding constraints (inherited from PROJECT_LAAS_v2 + session law)

- Zero external assets; TypeScript strict, zero `any`; deterministic `?seed=N`.
- Quality floors stand: ≥5M tris hero / ≥3M vista post-culling; no pop within 300 m;
  no black shadows; the final two-frame test MUST NOT regress vs `main`.
- Verify three.js APIs against `node_modules/three` source, never memory; record new
  gotchas in THREE-NOTES (renderer internals) or here (nanite-specific).
- MEASUREMENT METHODOLOGY (binding): M1 Max thermal drift → cooled ABAB pairs or
  in-session 24-sample averages only; per-pass GPU timestamps are encoder wall spans
  (rank with them, verify with wall fps + ablation); pixel-equivalence diffs need
  `--framealign N --wind 0 --lockexp 1` (deterministic floor ≤0.2%; water animates on
  wall-clock — exclude or accept); headless fps ≠ wall when CPU-bound.
- `?nanite=0` must boot the UNTOUCHED old pipeline until N7 closes (A/B + rollback);
  every debug/dbg view added here must be URL-gated and zero-cost when off.
- The Playwright tooling (shoot/compare/probe-*) is the verification surface — keep
  it working on this branch at all times.

## The content contract (end-state API — design against this from N1)

```ts
// the ONLY way geometry enters the renderer by N10:
registerMesh(geo: ClusterSource, mat: MaterialClassId, opts: {
  windChannel?: 'rigid' | 'trunk' | 'leaf' | 'grass';   // transform-stage animation
  aggregate?: boolean;        // foliage-style DAG collapse (area-preserving)
  castShadows?: boolean;
}): MeshHandle;
bindInstances(h: MeshHandle, stream: InstanceStream): void; // storage-buffer transforms
```
- `ClusterSource` = positions/normals/uvs/indices (+ per-vertex wind params where the
  channel needs them). Clusterization, BVH/DAG build, LOD, culling, raster, resolve,
  shadows: all generic downstream. A new species/rock/debris kit = generate mesh,
  pick material class, bind instances. NOTHING ELSE.
- MaterialClass = small closed set evaluated in the resolve übershader: TERRAIN,
  ROCK, BARK, DEADWOOD, LEAF, GRASS, DEBRIS (revisit count at N4; keep < 16).
- Screen-space systems stay outside the contract by design: water surface, sky,
  clouds, froxels, particles (camera-box quads), post. They are not geometry-path
  citizens; everything that IS solid geometry goes through the one path.
- The five bespoke systems this contract DELETES by the end: Forests.ts ring/pool
  draws, GroundRing clipmap draws, VegPrepass twins, CanopyShell, per-pool shadow
  proxy casters. (Their GPU scatter/placement compute SURVIVES — it feeds
  InstanceStreams.)

## Architecture (target)

```
boot:   procedural meshes → greedy clusterizer (~96–128 tri meshlets, bounds+cone)
        → [N8+] boundary-locked QEM simplification → cluster-group DAG w/ errors
frame:  instance cull (frustum + HZB sphere)               [compute]
        → cluster/DAG cut select (screen-space error) + cluster cull (frustum/cone/HZB)
        → visible-cluster compaction → atomic work queues  [compute, indirect chain]
        → transform stage (wind channels, prev-frame xforms for velocity)
        → raster: SW edge-function scanline for small tris (packed-u32 atomicMax
          vis-buffer) + HW indirect draw queue for big tris (same packing in fragment)
        → HZB build for next frame (+ second-phase re-test, see N2)
        → RESOLVE fullscreen pass: unpack → fetch cluster/tri → barycentrics →
          attributes → material übershader (probe GI, CSM sampling, canopy, wind-
          consistent normals) → writes beauty + depth + velocity for the post stack
post:   unchanged (TRAA gets REAL velocity from the vis-buffer for migrated geometry)
shadow: visible-cluster indirect draws into the existing CSM cascades (CsmCached
        cadence preserved); VSM explicitly out of scope
```

Reference implementation: `reference/three.js webgpu - compute rasterizer lighting.html`
(mrdoob). Proven there in OUR stack: HZB pyramid in TSL, work queue + indirect
dispatch, dual-u32 packed vis-buffer via single atomicMax (`depth17|tri15` +
`depth15|inst17`, fourth-root depth encode), SW/HW split by bbox size, fullscreen
resolve with manual interpolation + edge-derived tangents, prev-matrix velocity.
Known gaps vs real Nanite (verify in REVIEW): single-phase occlusion (prev-frame HZB
only — Nanite re-tests false-negatives in a second phase), discrete LODs (no DAG),
one material, no shadows, room-scale depth precision.

## Phase plan

Hybrid stage = N0–N7 (opaque dense geometry migrates; cards/grass stay hardware).
Full stage = N8–N11 (DAG, real-geometry foliage replacing cards, grass migration,
single-path consolidation). Every phase ends: tsc clean + gate measured + USER
CHECKPOINT + commit + PROGRESS LOG entry. "⏸ shippable" = safe pause point.

| # | Deliverable | Gate (measured) | Est |
|---|---|---|---|
| N0 | Baseline ledger (below) + feasibility spike: one rock pool + terrain tile through cull→queue→SW/HW raster→flat resolve in a dedicated scene | GO/NO-GO: ≥2× r.scene on that content, spike cpu.submit ≈ dispatch overhead | 8–14 h |
| N1 | Generic boot clusterizer (greedy ~96–128 tri, bounds+cone+error placeholder) + `registerMesh` skeleton + cluster tables for ALL opaque pools; existing ring LODs become discrete cluster sets per LOD | all opaque pools clusterized < +2 s gen; cluster stats printed (count/avg tris/bounds health) | 8–14 h |
| N2 | Culling chain: instance HZB+frustum → cluster cone/frustum/HZB → compaction → indirect dispatch; **two-phase occlusion** (prev-HZB pass + re-test pass, Nanite-style) from the start; `?cullfreeze=1` | visible counts match old path ±LOD policy at 4 bookmarks; zero disocclusion flicker on a hard pan (probe) | 8–12 h |
| N3 | Vis-buffer raster (SW+HW paths) for terrain+rocks, flat-lit resolve; DEPTH PRECISION DECISION (options below) | pixel-correct silhouettes vs HW reference; no z-artifacts at 4 km grazing (the horizon probe re-used); raster perf ledger entry | 10–18 h |
| N4 | Material resolve übershader: port TERRAIN, ROCK, BARK, DEADWOOD, DEBRIS classes (probe GI, CSM sampling, canopy attenuation, baked-noise splats); velocity from prev transforms | per-material frame-aligned equivalence vs `?nanite=0` (lockexp/wind0; floor ≤0.2% where geometry-identical); shadow-color + no-black-shadows pass | 16–28 h ⏸ |
| N5 | Shadows: visible-cluster indirect draws into CSM cascades; retire migrated pools' caster draws + proxies | shadow parity at all bookmarks; caster draw count ledger | 6–10 h ⏸ |
| N6 | Migrate remaining opaque bespoke draws (deadfall, stones, debris ring meshes, trunk rings); wind 'trunk' channel in transform stage | bm1/bm3/bm4/bm7 perf ledger vs baseline; draws + cpu.submit collapse documented | 8–12 h ⏸ |
| N7 | Hybrid close: full verification battery + two-frame check vs main; fold main fixes in; decide N8 go with data | battery green; ledger published; ⏸ MAJOR | 4–8 h |
| N8 | LOD DAG: boundary-locked QEM simplify, cluster groups (8–32), monotonic error propagation, runtime DAG cut (parentError > τ ≥ ownError), crossfade-free | continuous-zoom probe: no cracks, no pop, stable tri counts; heatmap dbg view | 18–30 h |
| N9 | Foliage as geometry: leaf/needle cluster meshes per species ring (sources exist — cards are baked FROM them), `aggregate` DAG collapse (area-preserving decimation), LEAF material class, wind 'leaf' channel; impostors retired ring-by-ring WITH judge shots | gallery A/B per species; forest-interior + vista framings ≥ current quality (user judges); perf ledger | 16–28 h |
| N10 | Grass migration (blade geometry through the path, 'grass' wind channel) + single-path consolidation: delete Forests draw path, GroundRing draws, VegPrepass, CanopyShell, proxies; `?nanite=0` demoted to a doc note | one geometry path remains; registerMesh is the only entry; LOC deleted ledger | 10–16 h |
| N11 | 5× density (understory/stones/debris ≥5×, judged distribution), memory budget pass, capacity re-tunes, FULL battery + two-frame test + final perf ledger vs main baseline | floors ≥5×; fps ≥ main baseline at all bookmarks (target: well above); two-frame test no regression | 8–12 h |

Total: hybrid 60–110 h, full +50–85 h. Estimates are agent wall-clock including
measurement discipline; boots ~2–3 min, cooled ABAB rounds 15–30 min each.

## USER CHECKPOINTS (visual, per phase — open in Chrome on the branch)

- N0: `?scene=rasterspike&sw=1` vs `&sw=0` — same content, spike vs hardware; HUD fps.
- N1: `?clusterdbg=1` — meshlet-colored world (hash colors per cluster, old pipeline).
- N2: `?cullfreeze=1` — freeze visibility then fly: culled geometry visibly missing
  behind you, none missing in view; `?clusterdbg=hzb` pyramid view.
- N3: `?nanite=1` (terrain+rocks migrated) vs `?nanite=0` — should look identical at
  flat-lit dbg (`?clusterdbg=flat`); `?clusterdbg=tri|depth|over` views.
- N4–N6: `?nanite=1` vs `?nanite=0` at bookmarks 1–9 — full shading; report anything
  that looks different (materials are gated on pixel-equivalence, your eyes are the
  second gate). HUD shows draws/cpu.submit collapsing.
- N8: continuous zoom on a hero rock/tree: no pops, no cracks; `?clusterdbg=lod`
  heatmap; `?loderr=N` threshold slider (1 = sub-pixel error, default).
- N9: `?scene=gallery` species pedestals A/B; forest interior bm7; vista bm3 —
  crowns at distance are THE judgement call (impostors retired only where you agree).
- N10: nothing visible changes (that is the point) — HUD draw count ~single digits.
- N11: meadow/forest framings: 5× ground detail; fps chip.

## Technical design notes (so fresh-context-me doesn't re-derive)

### Cluster build (N1)
- Greedy spatial clustering over index buffer: seed tri → grow by shared-edge
  adjacency picking min bounding-sphere growth, cap 128 tris (pad to fixed-size
  records). Per cluster: sphere (xyz,r), normal cone (axis, cosAngle, for backface
  cull), triOffset/triCount, vertexOffset window, materialClass, flags.
- Data layout (SoA storage buffers, all pools concatenated — "mega-buffers" like the
  example): positions vec4, normals vec4 (w bits spare: wind params), uv vec2,
  indices u32 (cluster-local u8 triples packed later if memory demands), cluster
  records, per-LOD cluster ranges. Instances: existing scatter buffers reused —
  InstanceStream = {transform mat (or pos+scale+yaw packed as today), prevTransform,
  poolId} — keep the CURRENT compact encodings, expand in transform stage.
- Boot cost budget: clusterization is O(tris); all pools ≈ 10–20M source tris →
  target < 2 s added (TS first; move to compute kernel only if measured slow).

### Culling (N2)
- Two-phase occlusion is NON-NEGOTIABLE (the example is single-phase; Nanite phase 1
  tests against prev-frame HZB reprojected, renders survivors, builds fresh HZB,
  phase 2 re-tests phase-1 rejects, renders late survivors, final HZB). Without it,
  fast pans show one-frame disocclusion holes — instantly visible to the user.
- HZB: max-depth pyramid (classic depth here, sky=1 — VERIFY conventions per pass;
  see THREE-NOTES depth notes), from the OPAQUE vis-buffer depth. Jitter: build from
  UNJITTERED matrices (TRAA clears view offset between frames — copy timing matters,
  see THREE-NOTES TRAA handshake).
- Frustum planes, sphere tests, cone backface: as in the example. Cluster cull
  appends to work queue (64-tri work items); `atomicAdd` counter → indirect dispatch
  args kernel (the example's `Compute HW Args` pattern, also for the SW dispatch).

### Vis-buffer + depth precision (N3 — DECISION REQUIRED, take with data)
- Payload need: visibleClusterIdx (compacted per frame; cap 64k? count at N2 with
  5× density headroom — if >64k, payload = clusterRecordIdx 20b + tri 7b = 27b and
  Option A is DEAD) + triIdx (7b for ≤128 tris).
- Option A (example-proven): dual u32 atomicMax — bufA `depth17|visClusterLo15`,
  bufB `depth15|payloadHi17`, fourth-root depth encode. Risk: 17-bit depth over 4 km
  → far-field co-planar fighting (terrain vs trunk bases at 2 km). Probe: the
  horizon-grazing framings from the GTAO saga.
- Option B: 64-bit emulation via atomicCompareExchange retry loop on a u32 pair
  (hi=floatBitsToUint(reversed depth), lo=payload). Correct depth everywhere;
  measure the contention cost on dense foliage before committing.
- Option C: depth-only atomicMin pass (full f32-as-u32) → second pass re-rasterizes
  visible clusters, writes payload where depth equals (non-atomic store, ties are
  benign same-surface). 2× raster ALU but no precision compromise and no CAS loop.
- Decide at N3 with: grazing-horizon artifact probe + raster-time ledger on bm4/bm7.
- SW/HW split: bbox ≤ MAX_RASTER_SIZE px → SW scanline (incremental edge functions,
  the example's inner loop verbatim as starting point); else HW queue. HW path
  renders the SAME packing from a fragment shader (three RawShaderMaterial-ish
  NodeMaterial with `isOutputStructNode` care — see THREE-NOTES MRT trap).
- Top-left fill rule from day one (the example uses bias terms — keep) or SW/HW seam
  double-hits will show as sparkle on the split boundary.

### Transform stage + wind channels (N3/N4/N9)
- Per visible cluster: fetch instance transform, apply wind channel in compute,
  write transformed positions to a transient cache buffer (visibleClusters × 128 ×
  vec4 ≈ 64 MB at 32k clusters — acceptable on UMA; ALTERNATIVE: recompute in
  resolve, costs ALU twice; MEASURE at N4 and pick).
- Channels: rigid (none), trunk (existing cantilever lean + sway — port the exact
  Wind.ts math), leaf (sway + flutter via vdata, far-fade 380–480 m contract),
  grass (tip² cantilever). Prev-frame wind phase needed for correct velocity —
  store prev gust-sample params per frame (worldTime-driven, freeze-deterministic).

### Resolve / materials (N4)
- Unpack pixel → cluster/tri → fetch 3 verts (transformed cache or recompute) →
  Pineda barycentrics at pixel center → attributes; derivatives: analytic per-tri
  ddx/ddy of UV from edge equations (the example derives tangents from edges —
  same machinery) → textureSampleGrad equivalents (TSL `.grad()` — VERIFY exists in
  0.184, else compute LOD manually with textureSampleLevel).
- Übershader with `Switch(materialClass)`: port order TERRAIN → ROCK → BARK →
  DEADWOOD → DEBRIS. Each port gates on frame-aligned equivalence vs `?nanite=0`.
  Probe-GI/canopy/contact inputs are world-space — they port mechanically. CSM is
  SAMPLED here (receiving); casting handled at N5.
- Outputs: beauty (rgba16f), device depth (write real depth for the post stack /
  water / froxels which all reconstruct from it), velocity rg16f (TRAA gets true
  motion vectors for migrated geometry; sky/hardware-path keep the analytic
  reprojection seam as today — see STATUS cloud-lag entry).
- Alpha-tested anything is BANNED from the SW raster path permanently. LEAF/GRASS
  enter only as real geometry (N9/N10). Masked-in-raster is the known perf trap.

### DAG (N8)
- Per mesh: level 0 = clusterized source. Loop: group 8–32 adjacent clusters
  (graph partition by shared-boundary length), LOCK group boundary vertices, QEM-
  simplify interior to ~50% tris, re-cluster the simplified set → level k+1; store
  parent/child links + per-group error (max QEM error, PROPAGATED: parent ≥ child —
  monotonicity is what makes the runtime cut crack-free).
- Runtime cut: render cluster iff ownError·screenFactor ≤ τ < parentError·
  screenFactor (one comparison, no global sort — same screen-space error projection
  the example uses for discrete LODs).
- Aggregates (N9): leaf clusters simplify by stochastic leaf REMOVAL with area
  redistribution onto survivors (preserve silhouette mass), not QEM on leaf quads
  (QEM on disconnected quads degenerates). Grass likewise. This is the Fortnite
  foliage approach; crown look at distance is the quality risk — judge with the
  user at N9 before retiring any impostor ring.
- 4-km far field: DAG bottoms out at coarse blobs; impostors stay until N9 judges
  each ring. CanopyShell deletion only after vista shots pass.

### Memory budget (track in ledger from N1)
- Mega-buffers (verts/indices/clusters, all pools + DAG levels): estimate at N1,
  budget ≤ 1.5 GB. Vis buffers: 2×u32×4.34 Mpx ≈ 35 MB. HZB ≈ 6 MB. Transform
  cache ≤ 64 MB transient. Instance streams at 5×: ~1M instances × 32 B ≈ 32 MB ×2
  (prev). Cluster visibility/queues: ≤ 16 MB. Adapter maxBufferSize is 1 GB (see
  Diagnostics requiredLimits) — mega-buffers may need splitting per attribute.

## Tracking protocol

- THIS file is the only planning/tracking authority on the branch. STATUS.md is
  read-only context here (main's bugs/methodology); merge main → branch when main
  lands fixes (water foam fix etc.) and note it in PROGRESS LOG.
- Per session: update PROGRESS LOG (dated, what landed, numbers), NEXT ACTIONS
  (always current), DECISIONS (append-only), GOTCHAS (append-only). Per phase gate:
  PERF LEDGER row + user checkpoint note.
- Commits: per milestone, message carries the measured numbers (the main project's
  discipline). tsc clean before every commit. New tools live in tools/ with the
  same launch.ts infra; remember the esbuild `__name` string-evaluate trap and the
  unique-filename rule for parallel probes (STATUS gotchas).

## PERF LEDGER (extend per phase; all at user viewport 2592×1676 unless noted)

| Point | bm1 | bm3 | bm4 | bm7 | cpu.submit | draws | notes |
|---|---|---|---|---|---|---|---|
| main baseline (cooled, 2026-06-13, STATUS pass-3) | 29.1 ms | 25.3 | 42.8 | 38.0 | 11.4–14.2 ms | ~548–905 | capture FRESH at N0 on this branch |

## BASELINE CAPTURE (N0 first task — exact commands)

```
npx tsx tools/shoot.ts --scene world --shot N --w 2592 --h 1676 --gpusample 24 \
  --stats shots/nanite/base-bmN.json --out shots/nanite/base-bmN.png   # N ∈ {1,3,4,7}
```
Cooled batches (idle ≥3 min between), record wall fps + cpu.submit + r.scene +
draws + tris per bookmark into the ledger. Also 1280×720 row (CI-speed checks).

## REVIEW CHECKLIST (for fresh-context review against Nanite literature)

Verify this plan against the SIGGRAPH 2021 "Nanite — A Deep Dive" (Karis et al.)
and current WebGPU-nanite community implementations; specifically re-derive:
1. Two-phase occlusion details (what exactly re-tests in phase 2 — instances,
   clusters, or both; ordering vs HZB rebuilds).
2. DAG cut correctness: the parent/child error comparison form that guarantees a
   crack-free cut with LOCKED group boundaries (and why boundaries must alternate
   lock sets between levels — verify my N8 sketch handles this; suspect it needs
   the group-merge-THEN-split-differently trick).
3. SW raster correctness details: fill convention, sub-pixel snapping precision
   (Nanite uses fixed-point 16.8 — the example uses float edges; decide), max
   raster size threshold value.
4. Vis-buffer payload budgets in shipping Nanite (depth bits vs cluster+tri bits)
   and what they do about depth precision at range.
5. Masked/foliage handling: confirm "opaque leaf geometry + aggregate simplify"
   matches Fortnite's shipped approach; what they do for grass specifically.
6. Material resolve: per-material tile binning vs übershader branching — at what
   material count does binning win.
7. WebGPU specifics: subgroup ops availability in Chrome stable (would accelerate
   the SW raster inner loop); 64-bit atomics status; timestamp granularity.
Record findings as amendments in DECISIONS with citations (URLs) before N0 code.

## RISK REGISTER

| Risk | Signal | Mitigation |
|---|---|---|
| 17-bit depth artifacts at km range | far co-planar sparkle at grazing probes | Options B/C ready (N3 decision with data) |
| visibleClusters > payload budget at 5× | N2 counts | payload re-split / per-tile cluster remap |
| Resolve slower than today's forward shading | N4 ledger | we are not fragment-bound today (r.scene is raster/submit); übershader → tile binning fallback |
| Crown look at distance (aggregates) | N9 judge shots | impostors retained per-ring until user signs off |
| cpu.submit floor from three.js per-frame overhead | N0 spike | known: renderObject pipeline still runs for the few remaining draws; acceptable if ≤3 ms |
| WGSL compile times / pipeline permutations | boot time creep | übershader (1 resolve pipeline), fixed kernel set |
| Branch drift vs main fixes | merge pain | merge main weekly; STATUS read-only here |

## DECISIONS (append-only)

- D-N1 (2026-06-12): Full route staged through hybrid; phases N0–N11 as above.
- D-N2 (2026-06-12): Single geometry path is the end-state acceptance criterion
  (user mandate) — bespoke render/cull systems get deleted, not wrapped.
- D-N3 (2026-06-12): Alpha-tested geometry permanently banned from the SW raster;
  foliage/grass enter as real geometry with aggregate DAG collapse (Fortnite model).
- D-N4 (2026-06-12): VSM out of scope; CSM retained with cluster-driven casters.

## GOTCHAS (append-only, nanite-specific)

- (seed) The reference example's `.toVar()` placements around chunk bounds are
  load-bearing ("store as var to prevent inlining") — WGSL codegen inlines
  re-reads otherwise; keep the pattern in ported kernels.

## PROGRESS LOG (append-only, newest first)

- 2026-06-12: Branch + this plan created. No implementation yet. Next: REVIEW
  CHECKLIST pass (fresh context, online sources), then N0.

## NEXT ACTIONS

1. REVIEW CHECKLIST pass against Nanite literature; amend DECISIONS with findings.
2. N0: baseline capture (commands above) → spike scene → GO/NO-GO numbers.

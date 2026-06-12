# NANITE WORKSTREAM — operating manual (source of truth on branch `nanite-raster`)

> **Rehydration protocol (branch)**: read THIS FILE fully → `STATUS.md` (main-project
> context; do NOT update it for nanite work) → `docs/THREE-NOTES.md` (API gotchas) →
> `reference/three.js webgpu - compute rasterizer lighting.html` (the proven TSL
> reference implementation — re-read the kernels before touching raster code).
> Continue from **NEXT ACTIONS**. Never re-plan from scratch. Update PROGRESS LOG +
> NEXT ACTIONS every session; commit per milestone with measured numbers in the message.
>
> REVIEWED 2026-06-12 (fresh-context adversarial pass against the Nanite literature +
> the example source + probed adapter limits). Findings F1–F19 below; design sections
> already incorporate them. Verdict: GO for N0 with amendments (see REVIEW FINDINGS).

## Mission

Replace per-system hardware rendering with a Nanite-style GPU-driven pipeline —
cluster hierarchy, HZB occlusion, compute/hardware hybrid visibility-buffer raster,
screen-space material resolve, continuous LOD DAG — staged through a hybrid that is
shippable at every phase boundary. Then raise ground vegetation/stone/debris density
≥ 5×. End state (user mandate, binding): **ONE standardized geometry path** — new
generated meshes are *registered* (mesh + material class + instance stream) and the
engine clusterizes/LODs/culls/rasters/shades them with **zero new render code**.
"An engine underneath that we barely need to touch." The pipeline must be truly
under-the-hood: one well-defined convention for everything that is solid geometry,
not five separately-culled, separately-rendered paths (see PATH UNIFICATION AUDIT —
every existing path has an explicit disposition).

Why (measured, from STATUS): cpu.submit 11–15 ms (draw-count-driven) and r.scene
~12 ms are the binding constraints at the user viewport; the 120 fps directive needs
both collapsed. 5× density is unreachable on the alpha-overdraw hardware path.
HONESTY NOTE (F11): the post chain floors at ~15 ms at the user viewport (STATUS
pass-3: TRAA 4.4 + megaquad 3.9 + GTAO 2.4 + clouds 2.5 + bloom + screen). A perfect
geometry path lands bm4 around ~20–25 ms wall, not 8.3 ms — the remaining post-chain
work (R11G11B10 RTs, leaner TRAA resolve, f16 math) is a SEPARATE workstream this
branch does not subsume. N11's fps gate is "≥ main baseline, target well above",
not 120.

## Binding constraints (inherited from PROJECT_LAAS_v2 + session law)

- Zero external assets; TypeScript strict, zero `any`; deterministic `?seed=N`.
  (Also no external geometry LIBRARIES for the DAG build — no meshoptimizer/METIS
  WASM blobs; the clusterizer/QEM/partitioner are hand-rolled TS.)
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

## WebGPU facts of record (probed on THIS machine 2026-06-12, Chrome stable headless, apple/metal-3)

- `maxStorageBuffersPerShaderStage` = **10** — HARD design constraint for every
  kernel and for the resolve fragment stage (F9). Mega-buffers must be packed/
  interleaved so no stage needs > 10 storage bindings (sampled textures are separate
  and plentiful; storage textures limit 8).
- `maxBufferSize` = `maxStorageBufferBindingSize` = **4294967292 (4 GiB−4)**.
  The old "adapter maxBufferSize is 1 GB" claim was wrong — 1 GiB is OUR requested
  clamp in `Diagnostics.buildRequiredLimits`; raise the request when mega-buffers
  need it (still budget ≤1.5 GB total — UMA pressure is real).
- Features PRESENT: `subgroups` (stable since Chrome 134, 2025-02 — available for
  the SW raster inner loop / compaction if wanted, not required),
  `shader-f16`, `timestamp-query`, `indirect-first-instance`, `primitive-index`
  (fragment `@builtin(primitive_index)` — candidate for the HW vis path, verify
  WGSL surface before relying on it), `clip-distances`, `dual-source-blending`.
- Features ABSENT: any 64-bit atomics (only a gpuweb PROPOSAL, issue #5071,
  explicitly motivated by Nanite vis-buffers; M1-class Metal may never support it).
  WGSL atomics are 32-bit only; `atomicCompareExchangeWeak` operates on ONE u32 —
  there is NO sound multi-word atomic emulation (F1).
- `maxComputeWorkgroupsPerDimension` 65535 (three auto-splits 1D dispatches and
  `instanceIndex` stays linear — verified on main, STATUS gotcha; pad-guard kernels),
  `maxComputeInvocationsPerWorkgroup` 1024, workgroup storage 32 KiB,
  maxColorAttachments 8.
- Citations: subgroups ship — developer.chrome.com/blog/new-in-webgpu-134,
  chromestatus.com/feature/5126409856221184; 64-bit atomics proposal —
  github.com/gpuweb/gpuweb/issues/5071.

## The content contract (end-state API — design against this from N1)

```ts
// the ONLY way geometry enters the renderer by N10:
registerMesh(geo: ClusterSource, mat: MaterialClassId, opts: {
  transformChannel?: 'rigid' | 'trunk' | 'leaf' | 'grass' | 'terrain';
  aggregate?: boolean;        // foliage-style DAG collapse (area-preserving)
  castShadows?: boolean;
}): MeshHandle;               // callable at boot AND later (hero trees are
                              // background-generated → late registration is law)
bindInstances(h: MeshHandle, stream: InstanceStream): void; // storage-buffer transforms
```
- `ClusterSource` is ONE of:
  (a) explicit arrays — positions/normals/uvs/indices (+ per-vertex wind params
      `vdata` where the channel needs them); or
  (b) a heightfield window — terrain enters PROCEDURALLY (F4): cluster records
      reference grid windows; positions/normals are reconstructed in the transform
      stage from the resident height texture ('terrain' channel = height fetch +
      the existing micro-displacement port). No 33M-tri terrain mega-buffer.
  Clusterization, DAG build, LOD, culling, raster, resolve, shadows: all generic
  downstream. A new species/rock/debris kit = generate mesh, pick material class,
  bind instances. NOTHING ELSE.
- `InstanceStream` = today's scatter encoding VERBATIM (F8): two vec4 records —
  A = (x, y, z, scale), B = (yaw, leanX, leanZ, idF). The transform stage applies
  scale → yaw rotation → LEAN SHEAR (B.yz · localY, base stays planted) → wind →
  translate, and yaw-rotates normals (VegInstance.ts is the reference math).
  Per-instance VARIATION LAW carries over: tint = slotHash(slot, 17/91), wind
  phase = slotHash(slot, 211) on the PERSISTENT scatter slot (not the compacted
  index) — the resolve must reproduce these or migration clones trees (banned).
  prevTransform mirrors A/B for velocity.
- MaterialClass = small closed set evaluated in the resolve übershader: TERRAIN,
  ROCK, BARK, DEADWOOD, LEAF, GRASS, DEBRIS (revisit count at N4; keep < 16).
- Screen-space systems stay outside the contract by design: water surface, sky,
  clouds, froxels, particles (camera-box quads), post. They are not geometry-path
  citizens; everything that IS solid geometry goes through the one path.
- The five bespoke systems this contract DELETES by the end: Forests.ts ring/pool
  draws, GroundRing clipmap draws, VegPrepass twins, CanopyShell, per-pool shadow
  proxy casters. (Their GPU scatter/placement compute SURVIVES — it feeds
  InstanceStreams.)

## PATH UNIFICATION AUDIT (user mandate: one convention, not five messy paths)

Every CURRENT draw/cull path and its disposition. "Migrates" = becomes registerMesh
+ bindInstances clusters; nothing migrates implicitly.

| Current path | Cull today | Disposition |
|---|---|---|
| Terrain CDLOD tiles + skirts + far shell | quadtree split | MIGRATES N3 as heightfield ClusterSource (F4); skirts die (crack-free cut); far shell folds into coarse DAG levels; micro-displacement → 'terrain' channel |
| Forests.ts tree rings R1/R2 + hero meshes | frustum+terrain-march+ring classify | trunks/bark N6; foliage cards → REAL leaf geometry N9; terrain-march occlusion dies (HZB subsumes) |
| Octahedral impostors (alpha quads) | ring classify | RETIRED ring-by-ring at N9 with user judge shots; until then the ONE sanctioned alpha-quad HW path; end state: gone (coarse DAG crowns ≈ few tris/tree at 2 km — that is the Nanite far field) |
| Understory/stone/extra pools | same | N6 (opaque), N9 (leafy) |
| GroundRing grass g1/g2/g3 clipmap | toroidal grid | N10 as 'grass' channel geometry |
| GroundRing debris ring | toroidal grid | N6 |
| Deadfall/logs/stumps | pool cull | N6 |
| CanopyShell far aggregate | none | dies N9/N10 after vista judge shots |
| VegPrepass depth twins | mirrors pools | die with their pools (vis-buffer IS the depth prepass) |
| ShadowProxy terrain grid + crown proxies + impostor-band casters | per-cascade | replaced N5 by per-cascade cluster culls; crown proxies die at N9 with real foliage |
| Water clipmap surface | clipmap | STAYS bespoke (translucent screen-space citizen) |
| Particles / clouds / sky / froxels / post | n/a | STAY bespoke by design |
| Scatter/probe/caustics/wind computes | n/a | SURVIVE as producers (InstanceStreams, GI, etc.) |

## Architecture (target)

```
boot:   procedural meshes → greedy clusterizer (~96–128 tri meshlets, bounds+cone)
        → [N8+] boundary-locked QEM simplification → cluster-group DAG w/ errors
frame:  PHASE 1: instance cull (frustum + prev-HZB) → cluster/DAG cut select
          (screen-space error) + cluster cull (frustum/cone/prev-HZB) → record
          occlusion-rejects → compaction → atomic work queues [compute, indirect]
        → transform stage (channels, prev-frame xforms for velocity)
        → raster phase 1: SW depth pass + payload pass (Option C, below) + HW
          big-tri queue (same vis-buffer writes from fragment stage)
        → HZB build from phase-1 depth
        → PHASE 2: re-test phase-1 occlusion-rejects (instances AND clusters)
          against the fresh HZB → raster late survivors → final HZB for next frame
        → RESOLVE fullscreen pass: unpack → fetch cluster/tri → barycentrics →
          attributes → material übershader (probe GI, CSM sampling, canopy, wind-
          consistent normals) → writes beauty + REAL f32 depth + velocity
post:   unchanged (TRAA gets REAL velocity from the vis-buffer for migrated geometry)
shadow: PER-CASCADE cluster re-cull (light frustum; camera visibility is NEVER
        reused for light views — F5) → indirect cluster draws into the existing
        CSM cascades (CsmCached cadence preserved); VSM explicitly out of scope
```

Reference implementation: `reference/three.js webgpu - compute rasterizer lighting.html`
(mrdoob). Proven there in OUR stack: HZB pyramid in TSL (storage-buffer mip chain,
level 0 at HALF res, max=farthest, classic depth), work queue + indirect dispatch
(64-tri chunk items, 2D-split dispatch), screen-space projected-error LOD select,
dual-u32 atomicMax vis-buffer, SW/HW split by per-triangle bbox (MAX_RASTER_SIZE
16 px), fullscreen resolve with manual perspective-correct barycentrics + analytic
UV/normal derivatives + edge-derived tangents, `.toVar()` anti-inlining.
CORRECTED claims after re-reading the source (F10):
- The example has NO velocity output. instancePrevWorld feeds the OCCLUSION test
  only. Vis-buffer velocity is OUR design, not example-proven.
- The example's HW big-tri path does NOT write the vis-buffer: it forward-renders
  a second MeshStandardNodeMaterial (a duplicated shading path — exactly what we
  must NOT ship). Our HW path writes the SAME vis-buffer from the fragment stage —
  a deliberate departure that needs its own spike (N0/N3, options below).
- The example DROPS any triangle with a vertex at w ≤ 0 BEFORE the HW-queue split —
  a verbatim port means near-plane holes underfoot in walk mode. Near-crossing
  triangles must be ROUTED TO HW (hardware clipping handles them), never dropped.
- The example's "meshlets" (126-tri debug coloring) ≠ its cull chunks (64 tri) —
  cosmetic only.
- Known example gaps vs real Nanite (confirmed vs literature): single-phase
  occlusion (prev-frame HZB only), discrete LODs (no DAG), one material, no
  shadows, 17-bit quantized depth (its resolve RECONSTRUCTS depth from 17 bits —
  do not copy; we write full f32).

## Phase plan

Hybrid stage = N0–N7 (opaque dense geometry migrates; cards/grass stay hardware).
Full stage = N8–N11 (DAG, real-geometry foliage replacing cards, grass migration,
single-path consolidation). Every phase ends: tsc clean + gate measured + USER
CHECKPOINT + commit + PROGRESS LOG entry. "⏸ shippable" = safe pause point.

| # | Deliverable | Gate (measured) | Est |
|---|---|---|---|
| N0 | Baseline ledger (below) + feasibility spike: one rock pool + ONE TERRAIN TILE (heightfield ClusterSource) through cull→queue→SW depth+payload raster (Option C)→flat resolve in a dedicated scene; verify fragment-stage storage atomics (or MRT fallback) for the HW path | GO/NO-GO: ≥2× r.scene on that content, spike cpu.submit ≈ dispatch overhead; HW-path write mechanism verified | 10–16 h |
| N1 | Generic boot clusterizer (greedy ~96–128 tri, bounds+cone+error placeholder) + `registerMesh` skeleton + cluster tables for ALL opaque pools (+ heightfield cluster records); existing ring LODs become discrete cluster sets per LOD; mega-buffer PACKED layout (≤10 storage bindings per stage — F9) | all opaque pools clusterized < +2 s gen; cluster stats printed (count/avg tris/bounds health); visible-cluster HUD counter | 8–14 h |
| N2 | Culling chain: instance prev-HZB+frustum → cluster cone/frustum/prev-HZB → reject recording → compaction → indirect dispatch; **two-phase occlusion** (phase 2 re-tests BOTH instance- and cluster-level rejects vs fresh HZB) from the start; `?cullfreeze=1`; `tools/probe-pan.ts` (scripted hard pan, frame-sequence hole detection — F13) | visible counts match old path ±LOD policy at 4 bookmarks; zero disocclusion holes on probe-pan; visible-cluster counts recorded at 4 bookmarks + 5× synthetic stress (payload-bit gate, F3) | 10–14 h |
| N3 | Vis-buffer raster (SW Option C + HW same-packing paths) for terrain+rocks, flat-lit resolve; fixed-point integer edge functions (≥8 subpixel bits, top-left rule); near-crossing tris → HW; DEPTH DECISION CONFIRMED with grazing-horizon probes | silhouette diff vs HW reference ≤0.05% with no structural breaks (F12); no z-artifacts at 4 km grazing (horizon probe re-used); raster perf ledger entry | 12–22 h |
| N4 | Material resolve übershader: port TERRAIN, ROCK, BARK, DEADWOOD, DEBRIS classes (probe GI, CSM sampling, canopy attenuation, baked-noise splats); velocity from prev transforms + prev wind phase | per-material frame-aligned equivalence vs `?nanite=0` (lockexp/wind0; floor ≤0.2% where geometry-identical); shadow-color + no-black-shadows pass | 20–32 h ⏸ |
| N5 | Shadows: PER-CASCADE cluster re-cull (light frustum per cascade on its CsmCached tick) → indirect draws into CSM; SW shadow raster uses single-u32 depth-only atomicMin (no payload needed); retire migrated pools' caster draws + proxies | shadow parity at all bookmarks (incl. off-screen casters — pan probe); caster draw count ledger | 8–12 h ⏸ |
| N6 | Migrate remaining opaque bespoke draws (deadfall, stones, debris ring meshes, trunk rings); wind 'trunk' channel in transform stage | bm1/bm3/bm4/bm7 perf ledger vs baseline; draws + cpu.submit collapse documented | 8–12 h ⏸ |
| N7 | Hybrid close: full verification battery + two-frame check vs main; fold main fixes in; decide N8 go with data | battery green; ledger published; ⏸ MAJOR | 4–8 h |
| N8 | LOD DAG (implementation-ready spec below): boundary-locked QEM simplify, groups 8–32 → split 4–16, per-cluster (own,parent) error+sphere pairs, containment + max-monotonicity, group-shared parent pair, hierarchical cut traversal via work queue, stuck-simplification fallback; DAG BUILD COST budgeted (time-sliced/Worker if >2 s — F15) | continuous-zoom probe (`tools/probe-zoom.ts`): no cracks, no pop, stable tri counts; heatmap dbg view; boot-time ledger | 22–36 h |
| N9 | Foliage as geometry: leaf/needle cluster meshes per species ring (sources exist — cards are baked FROM them), `aggregate` DAG collapse (area-preserving leaf removal, Epic "Preserve Area" precedent), LEAF material class, wind 'leaf' channel; impostors retired ring-by-ring WITH judge shots | gallery A/B per species; forest-interior + vista framings ≥ current quality (user judges); perf ledger | 16–28 h |
| N10 | Grass migration (blade geometry through the path, 'grass' channel — Fortnite precedent: opaque real-geometry blades) + single-path consolidation: delete Forests draw path, GroundRing draws, VegPrepass, CanopyShell, proxies; `?nanite=0` demoted to a doc note | one geometry path remains; registerMesh is the only entry; LOC deleted ledger | 10–16 h |
| N11 | 5× density (understory/stones/debris ≥5×, judged distribution), memory budget pass, capacity re-tunes, FULL battery + two-frame test + final perf ledger vs main baseline | floors ≥5×; fps ≥ main baseline at all bookmarks (target: well above; 120 fps needs the separate post-chain workstream — F11); two-frame test no regression | 8–12 h |

Total: hybrid 72–122 h, full +56–92 h. Estimates are agent wall-clock including
measurement discipline; shot cycles ~2–3 min, cooled ABAB rounds 15–30 min each.

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
- Data layout: PACKED mega-buffers (F9: ≤10 storage bindings per stage forces
  interleaving): one u32 blob per concern with manual decode — e.g. vertex blob
  (position 3×f32 + normal oct-u32 + uv 2×f16 + vdata u32 ≈ 24 B/vert), index blob
  (cluster-local u8 triples, 3 B/tri), cluster-record blob, instance blob (A/B
  vec4 pairs as today). Quantized-position variant (16-bit grid-relative, Nanite
  does this) is the fallback if memory pressure demands. Heightfield clusters
  store NO vertices — grid window + LOD stride only (F4).
- Instances: existing scatter buffers reused — InstanceStream as defined in the
  contract; keep the CURRENT compact encodings, expand in transform stage.
- Boot cost budget: clusterization is O(tris); all pools ≈ 10–20M source tris →
  target < 2 s added (TS first; move to compute kernel only if measured slow).

### Culling (N2) — two-phase occlusion (NON-NEGOTIABLE, details verified vs literature)
- Phase 1: test instances, then clusters of surviving instances, against the
  PREVIOUS frame's HZB using the PREVIOUS frame's transforms/VP (no reprojection of
  bounds — you project current bounds with prev matrices; for our static world,
  prev VP + current positions). Survivors raster. Anything REJECTED BY OCCLUSION
  ONLY (not frustum/cone) is RECORDED — both instance-level and cluster-level
  rejects (Karis deep dive; thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html).
- Build fresh HZB from phase-1 depth. Phase 2: re-test the recorded rejects against
  it (current matrices); newly-visible raster; final HZB for next frame builds
  after phase 2. Frame 0 / resize: no valid prev HZB → treat phase-1 occlusion as
  pass-through (everything visible), phases converge by frame 2.
- Wind-swayed geometry: cluster/instance bounds PADDED by the channel's max sway
  amplitude (else phase-1 prev-pose tests flicker foliage at gust onsets — F6).
- HZB: max-depth (= farthest, classic depth, sky=1) pyramid, level 0 at half res,
  storage-buffer mip chain exactly like the example (its sphereOccluded level-pick +
  2×2 footprint is correct — port verbatim). Source = the Option C depth buffer
  (f32 bits in u32), NOT a hardware depth texture. Jitter: TRAA jitters the raster
  proj matrix; HZB texels are ≥2 px so jitter is sub-texel noise — store and test
  with the SAME (jittered) VP used to raster that frame's depth; never mix.
- Frustum planes, sphere tests, cone backface: as in the example. Cluster cull
  appends 64-tri work items to the queue via atomicAdd → indirect dispatch args
  kernel (the example's `Compute HW Args` pattern, also for the SW dispatch).
  ALL queues get explicit capacity + overflow behavior: clamp, set HUD flag,
  never wrap (F14).

### Vis-buffer + depth precision (N3 — DECIDED at review, confirm with probes)
- Payload reality check (F3): hero framings draw 19.5M tris TODAY (STATUS Phase-5
  gate); post-DAG drawn tris are pixel-bound (~2–4/px × 4.34 Mpx ≈ 9–17M) →
  visible clusters ≈ 90k–170k at ~100 tris avg. ANY ≤16-bit visCluster budget is
  dead. Payload needs ≥18b visCluster + 7b tri = 25 bits.
- **PRIMARY: Option C (two-pass SW raster)** — pass 1: atomicMin of depth as
  f32-bits-in-u32 (positive floats order-preserve as uints; classic depth, min =
  nearest; early-skip via atomicLoad pre-check). pass 2: re-walk the same work
  items, store FULL 32-bit payload (visCluster | tri) with a plain (non-atomic)
  write wherever ownDepthBits == stored depth. Exact-equal co-planar ties race
  benignly (same surface, near-identical shading; far rarer than prefix ties).
  Full f32 depth everywhere — kills the 17-bit risk outright. Cost: ~2× SW raster
  ALU + one u32 clear; pay it, measure at N3 vs the spike's Option A numbers.
- Option A (example dual-u32 atomicMax) is DEMOTED to spike/debug comparison only
  (F2): two independently-atomicMax'd words can DISAGREE at depth-prefix ties
  (bufA 17-bit vs bufB 15-bit depth) → frankensteined cluster/tri payload →
  fetches arbitrary geometry, precisely at near-coplanar grazing surfaces (our
  documented horizon failure zone), and the winner is frame-nondeterministic
  (TRAA shimmer + diff-floor pollution). The example survives because its payload
  halves (tri | instance) are each independently valid for its single mesh. Ours
  are not. Shipping Nanite spends 30–32 bits on depth + 25b cluster + 7b tri in
  ONE 64-bit atomic (elopezr.com/a-macro-view-of-nanite; thecandidstartup) —
  the dual-u32 trick does not approximate that safely.
- Option B (64-bit emulation via CAS on a u32 pair) is STRUCK — unsound (F1):
  WGSL `atomicCompareExchangeWeak` is single-u32; there is no atomic pairwise
  update, so depth-hi/payload-lo tears under contention. The only WebGPU-shipped
  Nanite (Scthe/nanite-webgpu) confirms the wall: they fell back to 16-bit depth
  in one u32 and report "tons of artifacts like z-fighting or leaks"
  (github.com/Scthe/nanite-webgpu README). gpuweb #5071 exists precisely because
  this cannot be emulated.
- SHADOW SW raster: single-u32 depth-only atomicMin — no payload, no second pass.
  Perfect fit per cascade (F5/N5).
- HW big-tri path: fragment shader writes the SAME Option C buffers (pass-1
  atomicMin depth, pass-2 equality store), depthWrite OFF, no depth attachment —
  one resolve, one convention (departure from the example, F10). VERIFY at N0:
  three NodeMaterial fragment-stage storage atomics (storage().toAtomic() outside
  compute). Fallbacks if blocked: (a) `primitive-index` feature + rg32uint MRT +
  hardware depth, merged into the resolve by depth compare; (b) raise
  MAX_RASTER_SIZE so SW covers more (costs the O(n²) big-tri hazard). Pick with
  spike data.
- Near plane: per-vertex w ≤ ε ⇒ route the triangle to the HW queue (hardware
  clips); never drop (F10c). The SW path keeps the all-w>0 fast path.
- SW/HW split: per-TRIANGLE bbox ≤ MAX_RASTER_SIZE px → SW scanline (the example's
  incremental edge-function loop as starting point; its 16 px vs Nanite's ~32 px
  per-cluster threshold — tune with the N3 ledger).
- Fill convention (F12): snap vertices to FIXED-POINT (≥8 subpixel bits, i.e.
  16.8-style) and evaluate INTEGER edge functions with the top-left rule — exact
  watertightness, matches HW convention (D3D mandates 8 subpixel bits), removes
  the example's scale-dependent float `-1e-5` bias hack. i64 edge math not needed:
  with 13-bit screen coords + 8 subpixel bits, edge terms fit i32 if deltas are
  clamped to the bbox-guarded SW size; verify ranges in the spike.
- TRAA: SW raster uses the JITTERED proj (same as HW) or SW/HW seams crawl under
  jitter.

### Transform stage + channels (N3/N4/N9)
- Per visible cluster: fetch instance A/B, apply channel in compute, write
  transformed positions to a transient cache buffer. Budget HONESTLY at the F3
  cluster counts: 128k clusters × 128 verts × 16 B ≈ 262 MB — TOO BIG; so either
  (a) cache vec4-packed f16 positions (≈131 MB — still heavy), or (b) RECOMPUTE
  in resolve (ALU twice, zero cache). MEASURE at N4 and pick; default = recompute
  (the resolve already fetches 3 verts/px; transform math is cheap vs memory).
- Channels: rigid (none), trunk (existing cantilever lean + sway — port the exact
  Wind.ts math), leaf (sway + flutter via vdata, far-fade 380–480 m contract),
  grass (tip² cantilever), terrain (heightfield fetch + micro-displacement port).
  Per-instance lean shear + yaw + slot-hash phases per the contract (F8).
- Velocity: prev-frame transform AND prev-frame wind-phase params (worldTime-
  driven, freeze-deterministic) + prev camera VP → rg16f velocity in resolve.
  NOT example-proven (F10a) — three's VelocityNode is unusable for displaced
  geometry (THREE-NOTES); TRAA accepts a duck-typed velocity seam (THREE-NOTES).

### Resolve / materials (N4)
- Unpack pixel → cluster/tri → fetch 3 verts (recompute transform) → fixed-point-
  consistent barycentrics at pixel center → perspective-correct attributes;
  analytic UV/normal derivatives from edge equations (example lines ~1216–1263 —
  port verbatim) → `texture(...).grad(dUvDx, dUvDy)` — VERIFIED present in
  0.184 (TextureNode.grad, node_modules/three/src/nodes/accessors/TextureNode.js).
- Übershader with `Switch(materialClass)` (TSL `Switch` verified in 0.184):
  port order TERRAIN → ROCK → BARK → DEADWOOD → DEBRIS. Each port gates on
  frame-aligned equivalence vs `?nanite=0`. Probe-GI/canopy/contact inputs are
  world-space — they port mechanically. CSM is SAMPLED here (receiving); casting
  handled at N5. Per-instance tint reproduced from slot hashes (F8).
- ÜBERSHADER vs tile binning DECIDED (checklist #6): Nanite's material-depth +
  tile-grid passes / UE 5.4 shading bins exist to serve 16,384 artist materials
  (14-bit material IDs — elopezr; GDC 2024 "Nanite GPU Driven Materials"). At our
  CLOSED set of <16 engine classes a single Switch übershader is strictly simpler
  and avoids N full-screen passes; binning only becomes interesting if material
  count grows 10×. Decision stands (D-N10).
- Outputs: beauty (rgba16f), REAL f32 device depth via depthNode (post stack /
  water / froxels reconstruct from it — do NOT copy the example's 17-bit
  reconstruction, F18), velocity rg16f (sky/hardware-path keep the analytic
  reprojection as today).
- Alpha-tested anything is BANNED from the SW raster path permanently. LEAF/GRASS
  enter only as real geometry (N9/N10). Masked-in-raster is the known perf trap —
  UE 5.1+ supports masked Nanite but Epic's own Fortnite work moved foliage to
  REAL opaque geometry (leaves AND grass blades) for exactly this reason
  (unrealengine.com/en-US/tech-blog/bringing-nanite-to-fortnite-battle-royale-in-chapter-4).

### DAG (N8) — implementation-ready spec (corrected at review, F7)
- Build loop per mesh (level k → k+1):
  1. Group 8–32 adjacent clusters (Karis: 8–32; split target 4–16 new clusters)
     by graph partition minimizing shared-boundary edge count. Hand-rolled
     recursive bisection is an acceptable METIS substitute (zeux,
     github.com/zeux/meshoptimizer/discussions/750): recursively split the
     cluster adjacency graph in half (greedy boundary-min), stop at target size.
  2. Merge the group's triangles into one soup; WELD positions ignoring
     non-critical attributes first (un-welded seams are the #1 "simplification
     stuck" cause — zeux; Scthe).
  3. LOCK vertices on the GROUP boundary (shared with other groups); QEM-simplify
     interior to ~50% tris with unbounded target error (the runtime cut decides).
  4. Re-split the simplified soup into new clusters (4–16). These are the
     PARENTS of all the group's input clusters (DAG: a parent has many children;
     regrouping at k+1 mixes parents from different k-groups — that is what
     re-simplifies the previously locked boundaries; NO explicit alternating
     lock-set bookkeeping is needed, it falls out of re-grouping on the NEW
     adjacency — thecandidstartup; jglrxavpok LOD-generation post).
  5. ERRORS + BOUNDS (the crack-free machinery — get this exactly right):
     - groupError(k→k+1) = max(QEM error of this simplification,
       max over input clusters of their ownError)  → monotonic by construction.
     - groupSphere = the UNION sphere CONTAINING all input clusters' own spheres
       (containment is REQUIRED for projection monotonicity — jglrxavpok found
       non-monotonic cuts without it; zeux: union, not distance heuristics).
     - Every INPUT (child) cluster stores (parentError, parentSphere) :=
       (groupError, groupSphere) — IDENTICAL for all siblings.
     - Every OUTPUT (parent) cluster stores (ownError, ownSphere) :=
       (groupError, groupSphere) — the SAME values, so a parent's own pair
       exactly equals its children's parent pair (zeux: they must agree exactly).
     - LOD0 clusters: ownError = 0. Final roots: parentError = +∞.
- Runtime cut: render cluster C iff
  `project(C.ownError, C.ownSphere) ≤ τ  AND  project(C.parentError, C.parentSphere) > τ`
  with `project(e, sphere) = (screenH/2) · cot(fovY/2) · e / sqrt(d² − r²)`
  (d = camera→sphere-center distance, r = sphere radius; clamp d>r → ∞ when
  inside). Because siblings share the parent pair bit-for-bit, the cut boundary
  always falls BETWEEN groups, exactly where vertices were locked → crack-free
  (jglrxavpok runtime-LOD-selection post: clusterError ≤ τ AND parentError > τ,
  roots ∞). τ = 1 px default; `?loderr=N`.
- Cut traversal MUST NOT be a flat all-clusters test at our scale (162k+ instances
  × hundreds of DAG clusters — Scthe's flat list is their stated scalability
  wall): reuse the work-queue infra hierarchically — per visible instance push
  its root group; pop → project; if cut here emit clusters, else push child
  groups. Same MPMC queue pattern as the raster work items.
- Stuck-simplification fallback (Scthe's Jinx lesson): if a level reduces < ~15%
  tris, STOP that mesh's DAG there (multiple roots are legal — roots get
  parentError = ∞); never force-degenerate. Expect tubes/trunks to simplify well
  and disconnected leaf quads to refuse — that is what `aggregate` is for.
- Aggregates (N9): leaf clusters simplify by stochastic leaf REMOVAL with area
  redistribution onto survivors (preserve silhouette mass) — Epic shipped exactly
  this as the "Preserve Area" Nanite builder option for Fortnite trees (leaves
  thinning/going bare at distance was their symptom too; their mechanism dilates
  open boundary edges of remaining geometry). NOT QEM on disconnected quads
  (degenerates). Grass likewise. Crown look at distance is the quality risk —
  judge with the user at N9 before retiring any impostor ring.
- DAG BUILD COST is a first-class budget (F15): QEM over 10–20M source tris in TS
  will not be free; D6 law caps world gen ≈ 15 s. Measure at N8 start; if > ~2 s,
  time-slice the build (per-pool background like hero trees, progressive DAG
  enablement per pool) and/or move the QEM inner loop to a compute kernel.
  Deterministic by seed either way.
- 4-km far field: DAG bottoms out at coarse blobs; impostors stay until N9 judges
  each ring. CanopyShell deletion only after vista shots pass.

### Memory budget (track in ledger from N1; probed limits above)
- Per-stage binding ceiling: ≤10 storage buffers (F9) — the PACKED layout in
  "Cluster build" exists to satisfy this; count bindings per kernel in code review.
- Mega-buffers (verts/indices/clusters, all pools + DAG levels ≈ 2× leaf level):
  estimate at N1 with the packed layout (~24 B/vert, 3 B/tri index); budget
  ≤ 1.5 GB total; heightfield terrain contributes cluster records ONLY (F4).
  Adapter allows 4 GiB buffers/bindings — raise our requiredLimits clamp at N1;
  budget pressure is UMA, not API.
- Vis buffers (Option C): depth u32 + payload u32 = 2 × 4.34 Mpx × 4 B ≈ 35 MB.
  HZB ≈ 6 MB. Work queues: explicit caps (start 2M items × 16 B = 32 MB) +
  overflow clamp + HUD flag (F14). Visible-cluster list: cap 256k × 8 B = 2 MB.
- Transform cache: DEFAULT IS RECOMPUTE (zero); if N4 measurement flips the
  decision, budget at REAL counts (128k clusters → ≈131–262 MB — that cost is
  why recompute is the default).
- Instance streams at 5×: ~3M instances × 32 B (A+B) ≈ 96 MB ×2 (prev) — fine.

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
| main baseline (cooled, 2026-06-13, STATUS pass-3) | 29.1 ms | 25.3 | 42.8 | 38.0 | 11.4–14.2 ms | ~548–905 | reference |
| branch baseline (2026-06-12, this session, shots/nanite/base-bm*.json) | 33.6 | 58.3 ⚠ | 41.7 | 41.9 | 10.5–14.8 ms | 548–722 | bm3 = outlier (system-state spike, single run) — ABAB it before any bm3 conclusion; bm1/4/7 within thermal envelope of main |

N0 SPIKE LEDGER (2592×1676, gpusample-24 medians, back-to-back in-session;
content: 10.04M instanced tris, 1144 source clusters, 1937 instances,
55,568 visible work items, 22,784 HW-queued tris):

| Path | GPU total | breakdown |
|---|---|---|
| HW reference (5 instanced draws) | 2.8 ms | rt 2.16 + screen 0.52; cpu.submit 0.4 ms |
| SW Option C (2-pass) | 6.0 ms | depth 2.03 + payload 1.18 + cull 0.59 + hwPass 0.52 + clear 0.33 + resolve 0.33; cpu.submit 0.2 ms |
| SW Option A (single-pass dual-max) | 5.0 ms | rasterA 2.03 + cull 0.66 + hwPass 0.46 + clear 0.33 + resolve 0.66 |

## BASELINE CAPTURE (N0 first task — exact commands)

```
npx tsx tools/shoot.ts --scene world --shot N --w 2592 --h 1676 --gpusample 24 \
  --stats shots/nanite/base-bmN.json --out shots/nanite/base-bmN.png   # N ∈ {1,3,4,7}
```
Cooled batches (idle ≥3 min between), record wall fps + cpu.submit + r.scene +
draws + tris per bookmark into the ledger. Also 1280×720 row (CI-speed checks).

## REVIEW CHECKLIST — ANSWERED (2026-06-12 fresh-context pass; citations inline)

1. Two-phase occlusion: phase 1 = instances THEN clusters vs PREV-frame HZB with
   prev transforms; record occlusion-rejects at BOTH levels; raster; build HZB;
   phase 2 re-tests the rejects vs fresh HZB; raster late survivors; final HZB.
   (Karis deep dive via thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html,
   cs418.cs.illinois.edu/website/text/nanite.html.) → N2 section.
2. DAG cut: per-cluster (own,parent) error+sphere pairs; group-shared parent pair;
   containment + max-monotonicity; cut = own ≤ τ < parent; alternation falls out
   of re-grouping (NO explicit alternating lock sets — the old sketch's suspicion
   is resolved). (jglrxavpok LOD-generation + runtime-LOD-selection posts;
   zeux meshoptimizer discussion #750.) → N8 section.
3. SW raster: top-left fill rule with FIXED-POINT integer edges (≥8 subpixel
   bits); Nanite SW-rasters clusters with triangles ≲32 px (~3× faster than HW
   there); the example uses per-tri 16 px bbox + float edges with a -1e-5 bias
   (NOT watertight at our coord scale — replaced). Exact "16.8" figure is from
   the deep-dive slides (not independently re-fetched; the binding requirement is
   integer edges + ≥8 subpixel bits, matching the D3D convention).
4. Shipping Nanite vis-buffer: ONE 64-bit atomicMax per pixel — depth in the high
   30–32 bits, ~25-bit visible-cluster + 7-bit triangle below (elopezr macro view:
   R32G32_UINT, cluster 25 + tri 7 + depth 32; thecandidstartup: 30/27/7). WebGPU
   has no 64-bit atomics (gpuweb #5071 proposal only) → Option C two-pass is our
   answer; Scthe/nanite-webgpu's single-u32 16-bit-depth compromise is the
   documented artifact-ridden alternative. → N3 section.
5. Foliage: CONFIRMED — Fortnite Ch4 ships leaves AND grass as real opaque
   geometry through Nanite; "Preserve Area" builder option redistributes removed-
   leaf area by dilating open boundary edges; wind via baked WPO texture.
   (unrealengine.com tech blog "Bringing Nanite to Fortnite Battle Royale".)
   D-N3 stands, mechanism noted in N8 aggregates.
6. Material resolve: Nanite material-depth + tile-grid full-screen passes / 5.4
   shading bins serve 16k artist materials (14-bit IDs). At <16 closed classes the
   Switch übershader wins on simplicity and pass count. (elopezr; GDC 2024 "Nanite
   GPU Driven Materials" slides; sctheblog.com notes.) → D-N10.
7. WebGPU: subgroups STABLE (Chrome 134+, present on this adapter — probed);
   shader-f16 present; NO 64-bit atomics (proposal gpuweb#5071); timestamp-query
   present (already used); `primitive-index` + `indirect-first-instance` present
   (HW-path options). maxStorageBuffersPerShaderStage=10 HERE (hard);
   maxBufferSize/BindingSize 4 GiB−4 HERE. → facts-of-record section.

## REVIEW FINDINGS (2026-06-12, adversarial pass — each fix is already folded into the sections above)

- F1 BLOCKER: Option B (CAS-pair 64-bit emulation) impossible in WGSL — single-u32
  CAS only; pairs tear. STRUCK. (WGSL spec; gpuweb#5071.)
- F2 BLOCKER: Option A dual-u32 payload split is inconsistent at depth-prefix ties
  (cross-buffer frankenstein payloads, nondeterministic at grazing co-planar
  surfaces — our horizon zone). Demoted to spike comparison.
- F3 BLOCKER: payload math — hero draws 19.5M tris today; DAG-cut visible clusters
  ≈ 90k–170k ⇒ ≥18-bit visCluster + 7-bit tri; kills every ≤16-bit payload plan.
  Option C's full-u32 payload absorbs it.
- F4 BLOCKER: terrain as explicit mesh ≈ 33.5M tris L0 → memory-infeasible; now a
  heightfield-procedural ClusterSource ('terrain' channel, implicit verts, analytic
  errors from the existing height-range mip pyramid).
- F5 MAJOR: N5 culled shadow casters by CAMERA visibility — off-screen casters
  vanish. Now per-cascade light-frustum cluster re-cull on the CsmCached tick.
- F6 MAJOR: two-phase spec lacked reject recording at both levels, first-frame
  handling, and wind-sway bounds padding. Specified.
- F7 MAJOR: DAG sketch under-specified (shared parent PAIR incl. sphere,
  containment, exact own/parent equality, root/leaf conventions, hierarchical cut
  traversal, stuck fallback, welding). Rewritten implementation-ready.
- F8 MAJOR: content contract missed today's per-instance law (lean shear, slot-hash
  tint + wind phase, late hero registration). InstanceStream redefined as A/B
  vec4 pair + hash conventions.
- F9 MAJOR: maxStorageBuffersPerShaderStage = 10 (probed) forces packed mega-
  buffers; old doc claimed adapter maxBufferSize 1 GB — actually 4 GiB−4 (1 GiB is
  our requiredLimits clamp; raise at N1).
- F10 MAJOR: example misreadings corrected — (a) NO velocity in the example;
  (b) example HW path forward-shades (second material graph), does not write the
  vis-buffer — our unified HW write needs its own N0 spike (fragment storage
  atomics, primitive-index/MRT fallback); (c) example DROPS near-plane-crossing
  tris — must route to HW; (d) its resolve reconstructs 17-bit depth — we write
  real f32.
- F11 MAJOR (honesty): post chain ~15 ms is outside nanite scope; 120 fps is NOT
  this branch's deliverable. Mission + N11 gate updated.
- F12 MINOR: "pixel-correct vs HW" gate impossible literally → ≤0.05% + no
  structural breaks; fixed-point integer edges replace the float bias hack.
- F13 MINOR: N2/N8 gates referenced probes that don't exist → tools/probe-pan.ts,
  tools/probe-zoom.ts named as deliverables.
- F14 MINOR: queue budgets (16 MB) undersized vs example's own 45 MB; explicit
  caps + overflow clamp + HUD flag; transform cache re-budgeted at real cluster
  counts → recompute is the default.
- F15 MINOR: estimates widened (N0 10–16, N2 10–14, N3 12–22, N4 20–32, N5 8–12,
  N8 22–36; totals 72–122 / +56–92) + DAG build-time budget added.
- F16 MINOR: "cap 64k?" guess replaced by measured-counts gate at N2 (HUD counter
  + 5× synthetic stress) before payload bits lock at N3.
- F17 MINOR: `.grad()` and `Switch` VERIFIED in installed 0.184 — open questions
  closed.
- F18 NIT: never copy the example's 17-bit depth reconstruction into the resolve.
- F19 NIT: HZB source = Option C depth buffer (one source for SW+HW since HW
  writes the same buffers), not a hardware depth texture.

## RISK REGISTER

| Risk | Signal | Mitigation |
|---|---|---|
| Option C 2-pass raster cost too high | N0/N3 raster ledger | measure vs Option A spike; pass-2 early-skips; subgroups available if needed |
| Fragment-stage storage atomics blocked in three NodeMaterial | N0 spike | primitive-index + rg32uint MRT + depth-compare merge fallback; or larger MAX_RASTER_SIZE |
| visibleClusters > 256k cap at 5× | N2 counts + HUD flag | raise cap (u32 payload has headroom); hierarchical cut keeps the list cut-sized |
| Resolve slower than today's forward shading | N4 ledger | we are not fragment-bound today (r.scene is raster/submit); übershader → tile binning fallback |
| Crown look at distance (aggregates) | N9 judge shots | impostors retained per-ring until user signs off |
| DAG build blows boot budget (D6 ~15 s law) | N8 build-time ledger | time-sliced/Worker background build, progressive enablement, compute-kernel QEM |
| Simplification refuses to reduce (seams/aggregates) | N8 stats per level | weld-first, stuck fallback (multiple roots), aggregate path for leafy geometry |
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
- D-N5 (2026-06-12, review): Vis-buffer = Option C (atomicMin f32-bits depth pass +
  equality payload pass, full-u32 payload); Option A demoted to spike comparison;
  Option B struck as unsound. Shadow SW raster = depth-only single-u32 atomicMin.
  Cites: gpuweb#5071; Scthe/nanite-webgpu README (16-bit-depth artifacts);
  elopezr.com/a-macro-view-of-nanite (shipping 64-bit layout).
- D-N6 (2026-06-12, review): Terrain enters via heightfield-procedural
  ClusterSource ('terrain' transform channel, implicit vertices, analytic errors
  from the height-range mip pyramid); skirts/far-shell retire into the DAG cut.
- D-N7 (2026-06-12, review): transformChannel closed set {rigid, trunk, leaf,
  grass, terrain}; InstanceStream = today's A/B vec4 pair + slot-hash variation
  law; registerMesh supports post-boot registration (hero trees).
- D-N8 (2026-06-12, review): Shadow casting uses PER-CASCADE cluster re-culls
  (light frustum, CsmCached cadence); camera visibility is never reused for light
  views. Cite: Nanite re-culls per shadow view/page (thecandidstartup, VSM section).
- D-N9 (2026-06-12, review): WebGPU facts of record probed on this machine —
  see the dedicated section (subgroups present; no 64-bit atomics; 10 storage
  buffers/stage; 4 GiB buffer/binding max). Re-probe if Chrome major-updates.
- D-N10 (2026-06-12, review): Resolve = single Switch übershader at <16 material
  classes; tile binning/shading bins only if the class count grows 10× (Nanite's
  binning serves 16k artist materials — elopezr; GDC 2024 Nanite GPU Driven
  Materials).

- D-N11 (2026-06-12, N0): Fragment-stage vis-buffer writes ship via the
  opt-in ThreePatches.installFragmentStorageWrites + markFragmentWritable —
  three 0.184 hardcodes ReadOnlyStorage for non-compute storage bindings in
  BOTH WGSLNodeBuilder.getNodeAccess and WebGPUBindingUtils.createBindingsLayout;
  the patch honors node access for marked attributes only (re-verify on any
  three upgrade). The primitive-index/MRT fallback was not needed.
- D-N12 (2026-06-12, N0): Option C confirmed primary with measured cost:
  +1.0 ms (+20%) over Option A at 2592×1676 on 10M tris — the price of full
  f32 depth + full 32-bit payload. Spike A/B stays available via ?packing.

## GOTCHAS (append-only, nanite-specific)

- (seed) The reference example's `.toVar()` placements around chunk bounds are
  load-bearing ("store as var to prevent inlining") — WGSL codegen inlines
  re-reads otherwise; keep the pattern in ported kernels.
- (review) The example drops near-plane-crossing triangles entirely (w≤0 check
  wraps BOTH the SW loop and the HW enqueue) — fine for floating helmets, holes
  underfoot for us. Route near-crossers to HW.
- (review) Float edge functions with a constant -1e-5 top-left bias are NOT
  watertight at 2592-px coordinates (bias competes with f32 ulp at edge-term
  scale ~10⁶) — use fixed-point integer edges from day one.
- (review) atomicMax winners are deterministic per-buffer but NOT consistent
  ACROSS two buffers — never split one logical payload across two atomics.
- (N0) Work-queue overflow is INVISIBLE except as missing geometry: the
  counter keeps climbing past the cap while writes are skipped — whichever
  instances cull LAST lose their clusters (the terrain instance vanished
  wholesale). ALWAYS surface queueCount vs cap in the HUD and warn on
  overflow (F14 made law; it fired on day one — TWICE: sizing content to a
  pose-dependent frustum fraction is never a fix; caps must be memory-bound
  with dispatch 2D-splitting, never dispatch-bound).
- (N0) One dispatch must bind ONE view of a buffer: atomic + read-only views
  of the same attribute in a single kernel = WebGPU "writable usage and
  another usage in the same synchronization scope" validation error.
- (N0) A bare JS `return` inside If(() => {...}) silently builds NOTHING —
  use TSL Return() for an early WGSL return.
- (N0) Depth-equality across two RENDER pipelines misses by a few ulp (FMA
  fusion differs per pipeline — same mechanism as the VegPrepass @invariant
  trap). Compute-pass pairs compiled from identical TSL are exact in
  practice; the HW payload pass needs a small ulp tolerance until
  fixed-point depth (N3).
- (N0) TSL If/.toVar()/.assign() in MATERIAL node graphs need an Fn() stack
  exactly like compute — build vertexNode/fragmentNode as Fn(() => ...)()
  and pass varyings via varyingProperty assigned inside the vertex Fn (the
  example's hwPosition pattern).
- (N0) @types/three 0.184 TSL gaps needing casts (consolidate into typed
  helpers at N1): scalar/uvec storage type strings ('uvec2'/'uvec4'),
  ranged Loop objects with custom names, uvec2() ctor with uint nodes,
  min/max on uint nodes, float(uintNode) → use .toFloat().

## PROGRESS LOG (append-only, newest first)

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

1. N1-C3: src/nanite/GeometryRegistry.ts — registerMesh(ClusterSource,
   materialClass, opts)/bindInstances per the content contract; PACKED
   mega-buffers (vertex blob 24 B/vert: pos 3×f32 + oct-normal u32 + uv 2×f16
   + vdata u32, ONE u32 blob buffer; index blob u32 for now; cluster records
   8×u32: sphere 4×f32-bits, cone oct-axis u32 + cos f32-bits, triStart,
   triCount u16|matClass u8|flags u8; mesh table with lodGroup linkage);
   TSL decode helpers (readVertex/readCluster) in the registry module —
   designed against the 10-binding ceiling; counters-in-queue rule applies.
   Late registration supported (hero trees). HUD counters nanite.* + build().
2. N1-C4: boot wiring behind ?nanite=1 (world scene builds the registry from
   ALL opaque pools — rock/stone variants, deadfall, debris meshes, tree-ring
   BARK/opaque sub-geometries where separable (mixed card geoms deferred to
   N6 with a note), hero meshes late, terrain = implicit heightfield records
   over the REAL 4096² field; ?nanite=0/absent boots untouched). Print + gate:
   all pools clusterized, measured ms (<2 s), cluster stats table, registry
   memory MB. Commit with numbers → then N2.
3. N2 per the table (two-phase occlusion; replaces the spike's per-instance
   serial cluster loops with the real two-level culling chain).

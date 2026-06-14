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
- N1: meshlet-colored world — LIVE since N2-C1 as
  `?scene=world&nanite=1&nanitedbg=cluster` (world-side param is nanitedbg;
  `clusterdbg` is the SPIKE scene's param — user hit this trap 2026-06-12).
  Black sky/far field expected: only migrated geometry exists in the view.
- N2: `?cullfreeze=1` — freeze visibility then fly: culled geometry visibly missing
  behind you, none missing in view; `?nanitedbg=hzb` pyramid view; `?occl=0`
  kills occlusion for A/B.
- N3: `?scene=world&nanite=1&nanitedbg=flat` vs `&nanitedbg=hwref` — the
  nanite raster vs a hardware render of the same content: should be
  indistinguishable (machine gate: probe-parity, 0 px silhouette diff).
  `&shade=0` shows the gate's pure-class-color mode.
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
  records). Per cluster (AS BUILT, C3): sphere 4×f32-bits, cone oct-axis
  snorm2x16 + cos f32-bits, triStart (heightfield: gx|gz<<16), triCount u8 |
  flags u8 | **meshId u16** — matClass lives in the MESH record (D-N13): kernels
  need cluster→mesh anyway (hf params/channel), and indices are GLOBAL vertex
  ids so no vertexOffset is needed.
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
| old path re-ref (2026-06-13, uncooled singles, n2-old-bm*.json) | 32.9 | 66.8 ⚠ | 49.6 | 34.0 | 11–17.5 | 548–722 | bm3 slow AGAIN (58.3 then 66.8) — likely real on this content, not a spike; tris 12.6–16.5M |
| **N2 close — flat dbg view** (2026-06-13, n2-nan-bm*.json) | 7.9 | 8.5 | 16.8* | 8.3 | **0.6–0.9** | ~4 | NOT beauty-comparable (no materials/post/cards/grass); the deliverable: cpu.submit COLLAPSED 11–15→<1 ms on real content; nanite GPU (2-phase cull + 2×SW raster + HW + resolve, 4.34 Mpx) ≈ 2.5 ms bm1 → 7.4 ms bm3 (depth 2.5 + payload 3.1 + instCull 0.7 + hw 1.1); old-path compute hooks still tick underneath (+1–2 ms pollution: grassRingCull, vegCull, probeGather); *bm4 frameMs P95-polluted, fps 103 |
| **N3 close — fixed-point raster** (2026-06-13, bm3 2592×1676 single) | — | 8.1 | — | — | 0.6 | ~5 | integer scanline FASTER than the float core: nanRasterDepth 2.29 ms (was 2.5), nanRasterPayload 2.69 (was 3.1); hwTris 79k (unclamped-extent routing); gates: watertight ✓ silhouette parity 0–102 px = 0.000–0.011% vs HW at 5 framings ✓ 4-km grazing shimmer/holes/orphans 0 ✓ near-field F10c ✓ |
| **N4-C4 close — full beauty, 4 classes** (2026-06-13, c4-perf-bm*.json, gpusample-24 in-session, uncooled back-to-back) | 8.3 ms | 17.5 | 15.9 | 17.5 | **1.4–1.5 ms** | 21 | FIRST full-beauty nanite row (terrain+rock+bark+deadwood resolved + CSM-receive + post). NOT beauty-comparable to main (black slate per D-N21: no grass/cards/water/leaf/impostors — main draws all that). The deliverable signal: cpu.submit 11–15→**1.4 ms**, draws 548–905→**21**, on real content with shading. Nanite GPU: cull+SW-raster+HZB compute 2.95(bm1)/8.7(bm3)/8.6(bm4)/5.8(bm7) + HW pass 0.8–1.1; rasterDepth+payload 0.5(bm1)/5.9(bm3)/5.8(bm4)/3.5(bm7); visClusters 3.8k/81k/80k/53k, hwTris 18k/79k/55k/40k. Frame floor is the POST chain (bloom 6.5–7.5 + TRAA 6.5–7.4 + half.mrt 4.8–5.4 + scene 3.0–3.5) + the CSM cascade renders the resolve now solely drives (F11: post is a separate workstream). Thermal: in-session medians, bm7 ran warmest; not cooled ABAB. |

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
- D-N13 (2026-06-12, C3): Cluster record word 7 = triCount u8 | flags u8 |
  meshId u16 (the NEXT-ACTIONS sketch's matClass byte moved to the mesh
  record): raster/resolve need cluster→mesh for heightfield params + channel
  + matClass anyway, triCount fits u8 at the 128 cap, and global vertex ids
  kill the vertexOffset field. Mesh record = 12×u32 (cluster/instance ranges,
  lodNext/lodDist chain, channel|matClass|flags|winQuads, hf origin/cell/grid,
  swayPad). Instance blob = interleaved A/B vec4 pairs + a parallel u32
  instanceMesh buffer (cull-side instance→mesh without touching B.w idF).
  AMENDED N2-C1: mesh record = 16×u32 — words 12–15 = mesh-local bounding
  sphere (instance-level cull; heightfield spheres are world-space).
- D-N15 (2026-06-13, N3-C2): PARITY GATE DECOMPOSITION — the F12 gate is
  measured as TWO numbers: (1) SILHOUETTE diff (pixels where exactly one
  side shows background) ≤ 0.05% HARD — the literal F12 target; (2)
  interior matClass ownership flips (both sides lit, different class)
  ≤ 0.2% backstop. Flips are depth tie-breaks where surfaces INTERSECT
  (stones/trunks sunk into terrain — both rasters draw both surfaces
  correctly; the per-pixel winner at near-coplanar contact differs by
  sub-ulp interpolation convention). No raster convention can align them
  across two rasterizers; the backstop still catches real regressions.
  Shading is excluded from the machine gate (?shade=0 pure-class colors):
  HW derivative normals are garbage at silhouette pixels and average
  across sub-pixel triangles while the resolve fetches the pixel's exact
  triangle — a shading-MODEL difference, not a raster defect; lambert
  shots remain for human judgement, and N4 gates real shading with the
  lockexp pixel-equivalence methodology.

- D-N16 (2026-06-13, N4 recon): VELOCITY OUTPUT DEFERRED — the current
  TRAA consumes ANALYTIC camera reprojection from depth for ALL geometry
  (PostStack velReproject; the velocity MRT is skyveldbg-only). A
  vis-buffer velocity output would have NO consumer, so it cannot be a
  parity requirement; the resolve's real depth feeds the existing
  reprojection exactly like old-path geometry. Re-opened only if TRAA
  ever upgrades to true per-object velocity (then: prev transforms +
  prev wind phase as the phase table sketched).
- D-N17 (2026-06-13, N4 recon): RESOLVE LIGHTS MANUALLY — three's node
  lighting cannot receive per-fragment reconstructed positions:
  positionWorld is a vertex varying (Position.js:58), and the
  vertexNode-branch positionView reconstruction
  (cameraProjectionMatrixInverse × clipSpace varying, Position.js:84) is
  a NEAR-PLANE point — right ray direction, wrong magnitude; the
  reference example gets away with it because its scene is IBL-only (no
  shadows/fog/position-dependent terms). Patching the accessor
  singletons is brittle vs upgrades. So the übershader mirrors the
  lighting term-by-term: sun BRDF per three's PhysicalLightingModel +
  CSMShadowNode cascade select + OUR pcssFilter (ShadowSetup.ts — same
  code, portable) × cloud gate, probe-GI via the lightmap-slot
  convention (IrradianceNode contribution), aoNode = indirect-only.
  Fog/aerial/GTAO/contact/clouds need NO port: they are post-space from
  beauty+depth (PostStack).
- D-N18 (2026-06-13, N4 recon): IN-SCENE RESOLVE + JITTER MIRROR — the
  resolve is a fullscreen-triangle Mesh in engine.scene (renderOrder
  −1000, depthNode = vis-buffer f32 verbatim: classic depth, three's
  reversedDepthBuffer is opt-in and unset), so every later HW draw
  (grass, cards, water, sky) depth-tests against nanite geometry inside
  the ONE scene pass and the whole post chain applies unchanged. The
  nanite compute runs in an engine.post wrapper BEFORE postStack.render;
  NaniteCam mirrors TRAA's jitter (read _jitterIndex pre-render, halton
  (i+1,2)/(i+1,3) − 0.5 via scratch-camera setViewOffset) so SW/HW/
  resolve share the scene pass's jittered VP (else: sub-pixel offset +
  crawl vs hardware content under TRAA).
- D-N19 (2026-06-13, N4 recon): INCREMENTAL CLASS MIGRATION — registry
  gets a material-class filter (?naniteclasses=, default = the ported
  set so far; nanitedbg views default to ALL classes for pipeline
  probes). Only filtered classes raster + suppress their old camera
  draws (terrain tiles castShadow=false already — ShadowProxy is the
  caster; Forests camera meshes are separate from its per-cascade caster
  meshes, so hiding camera draws never touches shadows). Un-ported
  classes keep the old path — no double draw, no z-fight, per-material
  gates run on real frames.

- D-N20 (2026-06-13, N4-C1 reset): RESOLVE = CLIP-SPACE FULLSCREEN TRIANGLE +
  PLAIN NodeMaterial + fragmentNode + MANUAL lighting — re-aligning to
  D-N17/D-N18 after Fable's C1 deviation broke. Fable built the resolve as a
  camera-glued NEAR-PLANE triangle + MeshPhysicalNodeMaterial (to ride three's
  built-in sun/CSM/IBL lighting). That material FAILED to compile its lighting
  (THREE.TSL "Vertex attribute normal not found") and three SILENTLY FELL BACK
  to a material that ignored every node we set (colorNode/normalNode/
  emissiveNode/maskNode/setupLightMap) — so the resolve rendered nothing
  usable and near-camera terrain showed the sky behind it (user: "transparent,
  dithered out"). The fallback also swallowed every debug paint, hiding the
  failure through a long bisect. LESSON (GOTCHA): a NodeMaterial that emits a
  "vertex attribute not found" warning is in a fallback/again-state — STOP and
  fix the warning, do not trust any node you set on it. The C0 path (the
  ?nanitedbg=flat resolve in NaniteRaster) always worked because it is exactly
  this: clip-space fullscreen triangle + plain NodeMaterial + fragmentNode.
  CONSEQUENCE: shading is computed manually in the fragment (D-N17 as written,
  not three's pipeline). Specular view-dir is not needed (terrain is matte);
  the open question D-N17 flagged (matching three's IBL/CSM exactly for the
  ≤0.2% gate) is now the live N4-C1 work — CSM-receive + ambient/IBL parity
  are NOT done yet (terrain renders correct-coloured but unshadowed, ~39% vs
  ?nanite=0 at the walk spawn, almost all of it the missing canopy shadow).

- D-N14 (2026-06-12, N2-C1): HYBRID DRAW ENVELOPE — the LOD chain tail's
  lodDist (lodNext = NONE) is the instance cull-beyond distance, set by
  WorldRegistry to the old path's real-geometry edge (trees R2_FAR+BAND2 =
  496 m; other pools lib.clsMaxDist; terrain 0 = unlimited). Beyond it the
  old pipeline shows impostors — the sanctioned far field until the N8 DAG
  bottoms out (impostor retirement is N9's judged call). Without it the
  vista pushed 18.6M cluster items (every r2 ring to 4 km).

- D-N21 (2026-06-13, USER DIRECTIVE — supersedes the "?nanite=0 boots the
  untouched old pipeline" constraint for the build's duration): the OLD
  (non-nanite) SOLID-GEOMETRY render paths are HARD-DISABLED so the only thing
  that can appear in the world is the nanite output — no fallback, no
  hybrid-render confusion. Implemented as one switch `DISABLE_OLD_GEOMETRY =
  true` at the top of buildTerrainScene (TerrainScene.ts), gating every
  camera-pass add: TerrainTiles mesh/farShell + terrain ShadowProxy, Forests
  group, GroundRing grass+debris, CanopyShell, WaterSurface, Particles. What
  STAYS: the producers the registry is built from (Heightfield, Scatter,
  VegLibrary, ProbeGI, canopy map) and the screen-space environment
  (SunSky/atmosphere, Clouds, Froxels, CSM rig, PostStack, sun uniforms) so
  there is a lit frame to judge against. RATIONALE (user, verbatim intent):
  the hybrid kept drawing old-path trees/grass/water + their wind+lighting,
  which read as "nanite isn't active" every time — so until the nanite side is
  feature-complete we build against a BLACK-SLATE world where every visible
  surface is provably nanite. VERIFIED at the walk spawn 1280×720: draws
  724→21, triangles 12,034,968→2,002 (the resolve fullscreen tri + env quads),
  nanite.visClusters ~30.6k still rastering; the frame shows ONLY the nanite
  terrain + sky. REVERSIBLE: flip the constant to false to restore the full
  old pipeline (the N7 A/B path). The misleading "+ terrain tiles/far shell"
  suppression log is now moot (tilesRef is never created) — left as-is; the
  real witness is the draw/tri collapse, not the log.

- D-N22 (2026-06-13, USER DIRECTIVE — terrain lighting = ENERGY-CORRECT, NOT
  pixel-parity): the nanite terrain lights with sun×CSM + probe-GI ambient +
  caustics, all through three's exact BRDF (albedo/π on direct AND indirect —
  see NaniteResolve energy block). It is DELIBERATELY dimmer than the old
  terrain and we STOP chasing equivalence. Why the old is brighter (measured,
  not guessed): the old MeshPhysicalNodeMaterial gets a FULL env-IBL skylight
  term — scene.environment = sky PMREM at environmentIntensity 1.0
  (SunSky.ts:128-129) — ON TOP of the probe field; dimAmbientForGI (117-120)
  only dims the hemisphere, never the env. Whether that env term is a legit
  unoccluded-skylight or a double-count of the probe is genuinely ambiguous
  (disabling it globally drove golden hour near-black: old luma 28→11, so it is
  load-bearing, NOT a pure redundancy — the earlier "double-count bug" call was
  too strong). The user's ruling: the nanite uses the clean probe-only ambient
  (energy-correct), we DO NOT replicate the env term in the resolve, and we DO
  NOT disable it on the shared/old path (no reason to — we're not matching, and
  it only darkens code the black slate never shows). CONSEQUENCE: the
  `--framealign --lockexp` ≤0.2% terrain-lighting parity gate (N4-C1 NEXT
  ACTIONS item 2) is RETIRED — terrain lighting is judged on absolute quality
  (no-black-shadows floor, looks-right), not diffed against ?nanite=0. The
  measurement scaffold stays usable (?oldgeo=1) but is no longer a gate.
  Brightness, if wanted, is tuned via probe-GI strength / exposure / ToD as its
  own task — never by re-introducing the env double-count.

- D-N23 (2026-06-13, N4-C2): F9 BINDING BUDGET — the resolve übershader is the
  binding-heaviest stage (it does geometry fetch AND material shading in one
  fragment). Adding ROCK pushed it to 11 storage buffers (cap 10). FIX, straight
  from F9 ("sampled textures are separate and plentiful"): material-SAMPLING
  data reads from its TEXTURE, not its storage buffer. Concretely the heightfield
  height — used by the probe-GI ground-height lookup — is sampled from heightTex
  (already bound) instead of the height storage buffer (GI.irradiance gained an
  optional `groundY` the resolve supplies). This is NOT per-class buffers: ALL
  geometry shares the one registry mega-buffer set (verts/indices/clusters/
  meshes/instances) per D-N13 — the budget pressure is the COUNT of distinct
  shared buffers in one stage, not duplication. Resolve now ≤10. Headroom rule
  for the rest of N4: new material inputs must be TEXTURES (bark texA/texB are —
  fine); if a future class needs another storage buffer, pack two mega-buffers
  (F9) rather than split the übershader (D-N10 stands).

- D-N24 (2026-06-13, N4-C3): BARK/DEADWOOD texture-ARRAY + per-mesh matParam.
  Per-species bark (6 BARK_TABLE layers) can't be 12 separate bound 2D textures
  (the 16-sampled-texture budget), so the resolve samples ONE `texture_2d_array`
  per map (texA albedo+cavity, texB normal+rough+height), slice == bark layer.
  The per-mesh layer index rides in the registry mesh-record's word 7 — which
  holds hfOriginX for HEIGHTFIELD meshes ONLY, so explicit (bark/deadwood/rock)
  meshes reuse it free as a generic `matParam` (RegisterOpts.matParam; LODs
  inherit the head's). Deadwood shares slice 5 (snag) + adds moss/rot/dim; its
  per-pool dim is approximated by one logDim constant (energy-correct, not
  per-pool — D-N22 latitude). The resolve interpolates uv/normal/vdata by the
  same barycentric path as rock (D-N16) + a world-space TBN from triangle edges
  for the tangent normal map. MIP SELECTION is an analytic isotropic LOD (world
  texel footprint vs screen-pixel footprint) — NOT hardware auto-mip (uv is
  computed in the non-uniform If(isBD) branch → undefined derivatives) and NOT
  the anisotropic ray-plane .grad() (it NaNs on very near trunks → black; kept
  behind ?nanbark=grad for a later aniso pass).

- D-N25 (2026-06-13, N4-C3): maxSampledTexturesPerShaderStage RAISED 16→24
  (Diagnostics.buildRequiredLimits, clamped to adapterMax). The resolve samples
  17 (terrain maps + probe GI + 4 CSM cascades + canopy + caustics + bark texA/
  texB), over the spec-DEFAULT 16. This adapter reports 48, so the raise is free
  capacity, NOT a deviation: it makes D-N23's "sampled textures are separate and
  plentiful" literally true and follows the existing storage-buffer clamp (8→16)
  precedent. Storage buffers remain the real hard cap (adapter 10 = F9); sampled
  textures are not. Lower hardware degrades to its own ceiling (clamp), where the
  array would need merging to ≤16 — a future portability concern, not now.

- D-N28 (2026-06-14, N5 PIVOT — SUPERSEDES D-N26/D-N27's HW caster; research-
  grounded, see PROGRESS LOG (aa) + RESEARCH below): NANITE SHADOWS = DEPTH-ONLY
  COMPUTE SW RASTER into OWN r32 per-cascade depth buffers, sampled by the
  resolve's OWN PCSS. NOT three's CSM shadow map, NOT HW vertex-pulling. The C1
  HW caster (D-N26/D-N27) is a measured dead-end: 14–31 ms PER cascade, every
  cascade every frame (90→15 fps). Four cross-verified sources + a code trace say
  do what Nanite actually does:
  • REUSE THE DEPTH PASS, DEPTH-ONLY. rasterKernel('depth') (NaniteRaster.ts
    191–419) already projects tris through cam.vp and atomicMin(bcF2U(cz)) into a
    u32/px buffer. For a cascade: point cam.vp at the LIGHT VP, write a per-cascade
    r32 buffer, SKIP pass-2 (payload) + the material resolve. Shadows are < half
    the main-view cost. (elopezr "same process… only depth output"; CS418 "shadow
    views only end stage 1"; the SW/HW big-tri split is reused as-is.)
  • 64-BIT ATOMICS NOT NEEDED for shadows — depth-only = one 32-bit atomicMin,
    which we already do (bcF2U). (ktstephano Sparse VSM: 32-bit imageAtomicMin on
    f32-bits; Tellusim/Scthe restrict SW raster to depth-only ON METAL for this.)
  • OWN r32 BUFFERS, not three's shadow.map — a WebGPU compute shader CANNOT write
    a DepthTexture (depth formats aren't storage; spec + gpuweb#1043). So "fill
    three's depth texture" is impossible; that is WHY C1 had to vertex-pull into
    the shadow override. Allocate r32uint storage (atomicMin target) per cascade,
    like makeVisBuffers.
  • RESOLVE SAMPLES OUR BUFFERS. The pcssFilter already does manual taps
    (ShadowSetup.ts:71 raw .x read, :92 .compare()) and three 0.184 .compare()
    SOFTWARE-FALLBACKS on a non-DepthTexture (TextureNode.js:408–416, gated
    WGSLNodeBuilder:714) — so an r32 buffer/texture works with no sampler fork.
    Replace the resolve's nodeObject(world.csm).x (NaniteResolve.ts:499) with our
    own cascade-select + PCSS over the 4 buffers.
  • LOCKSTEP (the #1 correctness item): the raster VP MUST equal the sample VP.
    Use the SAME csm.lights[c].shadow.camera (projectionMatrix·matrixWorldInverse)
    for both — NaniteShadow.update already reads exactly these. Do NOT recompute
    ortho frusta (CSM texel-snap is non-obvious; mis-repro = crawling shadows).
  • CACHING IS THE SPEED. Keep CachedCsmShadowNode purely as the cascade-fit +
    [1,2,3,6] cadence bookkeeper; gate the SW raster on it — re-raster a cascade
    ONLY when its VP changes (auto-captures freeze + sun/drift force-refresh).
    Static camera ⇒ ~0 shadow cost. (All 4 reports: cadence = the poor-man's VSM
    page cache; UE "pages cached between frames unless invalidated by moving
    objects or light".)
  • WIND (the user's catch): caching freezes wind shadows. PROPER fix = STATIC/
    DYNAMIC dual-depth split (UE VSM "two copies of depth") — cache static
    (terrain/rock/rigid trunks = the bulk), and every frame copy + atomicMin ONLY
    the dynamic (trunk-channel, within the 380–480 m wind fade) clusters on top.
    Far cascades are wind-rigid ⇒ fully cached. (UE: WPO/vertex-anim invalidates
    its page every frame — grass-WPO shadow 8.2→3.1 ms when disabled; keep wind in
    the dynamic set / cascade 0.) Deferred to R3; R0–R2 raster every frame so wind
    is correct meanwhile.
  • COARSE LOD for far-cascade shadow casters (penumbra hides silhouette error) —
    keep cascade 0 honest (world-metric PCSS blocker search lives there).
  • DO NOT BUILD: the 16K virtual page table/clipmap (overkill at 1 sun/4
    cascades); light-space HZB caster occlusion (defer; sphereOccluded stays null —
    off-screen casters must cast, F5). camera-HZB on casters is the classic bug.
  R-CHUNK PLAN replaces N5-C2/C3 (NEXT ACTIONS). The C1 cull half (NaniteShadow C0)
  is KEPT; the C1 caster-mesh half is deleted at R4.

- D-N26 (2026-06-13, N5 design — SUPERSEDED by D-N28; HW caster measured too slow):
  NANITE SHADOW CASTERS = HW INDIRECT DRAWS INTO
  THE CSM CASCADES VIA PER-CASCADE LAYERS — mirror the Forests caster mechanism
  (the very path N5 deletes), NOT a compute SW raster into an owned depth texture
  (deferred). RATIONALE: three's CSM samples a per-cascade DEPTH TEXTURE through
  a compare sampler (ShadowSetup.pcssFilter — blocker search + world-metric
  penumbra + Vogel PCF). Forests already feeds those textures by (a) the cull
  kernel testing instances against each cascade's ortho frustum (6 planes ×
  CASCADES, refreshed one frame stale from `csm.lights[c].shadow.camera`, slack
  hidden in lightMargin), and (b) per-cascade sibling caster meshes on layer
  `2+c` with `castShadow=true`; `csm.lights[c].shadow.camera.layers.enable(2+c)`
  makes ONLY cascade c render layer `2+c`, while visible meshes set
  `castShadow=false` (Forests.ts:940-991, 391-431). Reusing this gets the proven
  pcssFilter + cascade split (CsmCached) + cadence FOR FREE — a SW raster would
  have to own the depth texture and reimplement the compare/penumbra, which is
  the integration wall. The SW depth-only atomicMin path (D-N5) stays the
  deferred perf option if HW caster raster measures too slow (it likely won't:
  nanite clusters are far fewer tris than the alpha cards Forests rastered, and
  shadow passes are depth-only). NANITE CASTER SPECIFICS vs the camera cull:
  shadow cull = cascade ortho frustum + camera-distance LOD (shadow LOD must
  match the visible geometry's LOD or peter-pan) + NO occlusion (an off-screen/
  ridge-hidden caster still casts — F5) + NO cone backface (a cluster facing away
  from the CAMERA still casts toward the light; cone uses the camera axis, wrong
  for light views). So the shadow cull is buildNaniteCull with the cascade
  NaniteCam, sphereOccluded=null, and cone disabled (new opt). The caster MESH
  is a vertex-pulling NodeMaterial (indirect draw over the cascade's visible-
  cluster list; vertexNode = fetchWorldVert by gl_VertexIndex→cluster/tri/corner,
  transformed by cameraView/Projection which during cascade c's shadow render
  ARE the cascade light VP — bit-identical positions incl. trunk wind to the
  camera path); depth-only. Old casters (ShadowProxy + Forests siblings) retire
  when nanite shadows own the cascades (gate like D-N21's DISABLE_OLD_GEOMETRY).

- D-N27 (2026-06-14, N5-C1): NANITE SHADOW CASTER = vertex-pulling NodeMaterial
  Mesh per cascade on layer 2+c, world position injected via
  `castShadowPositionNode` (NOT vertexNode — the shadow pass ignores it),
  identity matrixWorld, side=DoubleSide, **`map = null`** (the base-NodeMaterial
  `!== null` trap — GOTCHA). Indirect non-indexed DRAW over the cascade cull's
  qRasterRO, count = visClusters·128·3 via a per-cascade kCasterArgs kernel
  (padding tris collapse to vec3(0)). makeFetch built with the SAME (heightTex,
  disp, wind) as the camera raster ⇒ bit-identical geometry (no peter-pan). This
  realises D-N26's "HW indirect draws into the cascades via layers" — the SW
  depth-only atomicMin alternative stays the C3 perf fallback (the HW caster's
  128-stride over-draw measured heavy: shadow.c0 18–36 ms; cascade 0 is PERIOD 1
  so it redraws every frame and cadence can't amortise it).

- D-N29 (2026-06-13, N5 SHADOW RETHINK — user directive "which gives BOTH best perf
  AND beauty? i dont care how long it takes"; research = 4 cited threads, PROGRESS LOG
  ae): the sun shadow STAYS a SW-raster term. REJECTED: VSM (its page-cache — the whole
  point — dies to wind; Epic docs verbatim "WPO/skeletal always invalidates cached pages
  every frame"; + Metal auto-disables StaticSeparate); unified shadow+GI as the primary
  sun shadow (UE Lumen deliberately keeps a SEPARATE VSM; CryEngine SVO sun shadows "too
  soft … depends on voxel resolution"; 512³ voxels ≈ 2.5 GB > 1.5 GB); SW-RT of deformed
  triangles (per-frame BVH rebuild over 162k swaying trees, M1 SW-RT ~10× off HW).
  TARGET = a LAYERED stack, each frequency band by its cheapest-best tool:
   (1) SHARP SUN TERM = SW-raster, but RESOLUTION REALLOCATED by screen-pixel density
       (clipmap, not fixed 4-cascade splits) → ~1 shadow texel per screen pixel near =
       crisp penumbra (BEAUTY) + texels only where pixels are (PERF). VSM's one genuinely
       transferable win, taken WITHOUT its page-cache (dead to wind anyway). Feasible w/o
       64-bit atomics — StratusGFX proves SVSM needs only 32-bit imageAtomicMin; our SW
       raster already IS that "software path".
   (2) PERF ENGINE = DAG-decoupled caster LOD + distance-regime split: NEAR real-geometry,
       full sway, coarsen only ~2× (dapple survives, penumbra hides it) / MID coarsen
       ~8–16× + shadow-pass WPO-FREEZE → rigid → cached / FAR rigid coarsest-or-impostor,
       cached + temporally staggered (SpeedTree per-cascade scheme: cascade 0 every frame,
       far cascades every 2nd/4th, drop geometry classes).
   (3) SHADOW-VISIBLE-ONLY cluster cull (VSM's "needed-texel" mask — only raster clusters
       that shadow visible pixels; tighter than per-cascade frustum). Pure perf.
   (4) CONTACT-HARDENING soft penumbra (SMRT/PCSS: sharp at contact, soft away) — BEAUTY.
   (5) ADDITIVE BEAUTY CEILING = capsule-SDF soft inter-tree occlusion (wind-FREE — a
       capsule rotates by transform; blobs are CORRECT for the large-scale soft band) +
       screen-space CONTACT shadows (fine sub-pixel band). The bands a sun map can't do
       cheaply, each at its cheapest.
  PLUS a RESOLVE-side fix for the fixed ~5 ms PCSS-SAMPLE cost (the static 115→70 gap,
  NOT raster): half-res shadow eval + depth/normal-aware upsample (beauty-neutral on soft
  penumbra; contact band kept sharp by the bilateral). Supersedes R2/R3/R4. Chunks S0–S5
  in NEXT ACTIONS; the coarse-caster-LOD half of (2) is GATED ON the N8 DAG — interim uses
  discrete-LOD bias + tighter cull. HONEST RISK (Thread A): the near real-geometry raster
  is an irreducible per-frame floor (everyone re-rasters near swaying casters — SpeedTree,
  Guerrilla, UE); we minimise the COUNT (tight cull + only-near + small foliage→contact
  shadows) and coarsen modestly, but KEEP it real geometry because that is the near beauty.
  Cites: PROGRESS LOG ae (4 thread docs); Epic VSM docs; SpeedTree GPU Gems 3 ch.4;
  StratusGFX SVSM; UE Lumen/Capsule-Shadow docs; iq soft-shadow.

- D-N30 (2026-06-13, N8-D0 — F15 build-cost MITIGATION): the hand-rolled QEM DAG
  build (BuildDag.ts) runs at **0.16 Mtri/s in pure TS** (measured, probe-dag) →
  the registry's 1.52M explicit tris ≈ **9.3 s**, OVER the ~2 s F15 soft target.
  RULING: the DAG build runs **off the boot critical path — background / time-
  sliced per pool** (reuse the late-registration infra the content contract
  already mandates for hero trees). Boot stays at the current ~0.5 s clusterize
  budget; each pool keeps its DISCRETE LOD chain (D-N14 envelope) until its DAG
  is ready, then swaps to the continuous cut. This is the spec's pre-planned F15
  mitigation ("time-slice the build … progressive DAG enablement per pool") and
  is directly endorsed by the user directive that set up N8 ("i dont care how
  long it takes or how long it wont be visible before finished"). PREFERENCE: a
  Web Worker (buildDag takes plain typed arrays — no three — so geometry transfers
  in, cluster data transfers back; zero main-thread hitch) over main-thread time-
  slicing. FALLBACK FLOOR (documented, only if a background Worker still starves):
  move the QEM inner loop to a compute kernel (F15) — NOT attempted yet; the
  builder also has ~2× of un-exploited TS headroom (typed-array SoA heap, indexed
  decrease-key) if a cheaper win suffices. This is a D1 WIRING decision (the cost
  is paid only when the registry boot invokes the build); D0 delivers the
  validated builder + the measured budget. The pure-TS builder stays the
  reference/probe path regardless (node-runnable, deterministic).
  RATIFIED 2026-06-13 (user "stick to worker" after weighing the fork). The compute-
  kernel QEM is EXPLICITLY DEFERRED — the downsides (recorded so it is not re-
  litigated): (1) edge-collapse QEM is serial-greedy (pop global-min cost → collapse →
  recompute neighbours); the GPU-parallel variants (maximal-independent-set collapse,
  grid-snap) are LOWER QUALITY on the exact thing that matters (LOD fidelity + crack-
  freeness), and clusterize() is ALSO serial-greedy — the whole pipeline is; (2) WebGPU
  has NO atomic float-add (quadric scatter-add → CAS spin-loops), no 64-bit atomics,
  10 storage buffers/stage (F9), and the build is variable-size dynamic output across
  levels → bump-allocators + the F14 silent-overflow class; (3) GPU atomic ordering +
  FMA non-determinism breaks `?seed=N` AND threatens the bit-exact sibling-pair equality
  (E) the crack-free cut depends on — the CPU build is deterministic for free; (4) debug
  = storage-buffer readback forensics vs the node probe's exact-number headless check;
  (5) it steals the SCARCE resource (GPU/UMA, the whole reason for this branch) while a
  Worker uses an IDLE CPU core. ESCALATION LADDER if a Worker ever starves a real
  interactive case (e.g. late hero-tree registration latency): Worker → ~2× TS opt (SoA
  heap, indexed decrease-key) → frame-slice the CPU build → compute kernel (last resort).
- D-N31 (2026-06-13, N8-D1 — runtime cut shape): the spec sketched a HIERARCHICAL
  work-queue traversal (push root group → pop → emit-or-refine-children). D1 ships the
  semantically-equivalent **FLAT per-cluster cut** instead: every DAG cluster carries its
  own (ownErr,ownSphere)+(parentErr,parentSphere) in a parallel buffer, and kClusterCull
  tests `project(own)≤τ AND project(parent)>τ` independently per cluster. WHY: the
  predicate is the SEMANTICS; the traversal is only a pruning OPTIMISATION. The flat form
  is (a) provably crack-free with NO diamond/dedup problem (each cluster tested exactly
  once → emitted 0/1×, no per-instance group "visited" state — which on a non-tree DAG is
  the hard part the hierarchical form must solve), (b) a ~3-line addition to the EXISTING
  instance→chunk→cluster cull (attachDag repoints word0/1 at the full DAG range so
  kInstCull is untouched), (c) lets us VALIDATE the cut on real GPU projection before
  investing in traversal machinery. COST: tests every DAG cluster of every visible
  instance (≈2× LOD0 cluster count) vs the discrete chain's one ring — bounded to explicit
  classes (terrain stays heightfield-mip). Measured FINE for rock (gate green, no overflow).
  REVISIT (D1b-perf) only if a heavier class (bark = 162k trees) makes the cull-dispatch
  volume the bottleneck — THEN add the hierarchical traversal as a pure pruning layer on
  top (same predicate, fewer tests). The flat cut stays the correctness reference.
- D-N32 (2026-06-13, TERRAIN-DAG approach — user directive + my correction of record):
  terrain gets its OWN heightfield-native adaptive builder, SEPARATE from BuildDag, both
  emitting the SAME cut metadata (own/parent error+sphere → the parallel buffer →
  kClusterCull). CORRECTION OF RECORD: I first claimed terrain QEM would take "minutes" —
  WRONG, that figure was BuildDag's ITERATIVE Garland-Heckbert edge-collapse (global heap,
  recompute neighbour quadrics), the right tool for an IRREGULAR mesh but the wrong one for
  a grid. A heightmap is regular: connectivity known, vertex error = pure VERTICAL deviation
  → a **right-triangulated quadtree (RTIN, Mapbox `martini`-class)** builds a per-node error
  pyramid BOTTOM-UP in O(n) (each coarse node err = max(children err, own interp err)),
  sub-second for 4096², one-time + seed-deterministic + Worker-able. It is ADAPTIVE FOR
  FREE — the per-node error gates refinement, so a flat plain collapses to a couple of big
  right-triangles while a cliff/erosion-channel stays dense (user MANDATE 2026-06-13:
  "non-naive optimizer … plains have SIGNIFICANTLY less tris"). RTIN selects a SUBSET of
  grid vertices (never off-grid) → positions still reconstruct from the heights buffer (F4
  procedural preserved; store compact connectivity per cluster, NOT baked floats — my
  "+200 MB / loses F4" claim was also overstated; general off-grid QEM would lose it for a
  marginal error-per-tri gain not worth it on a heightmap). Crack-free via the restricted-
  quadtree balance rule (forced-split neighbours) — the analogue of BuildDag's locked
  boundaries. So "QEM adaptive terrain" = the FAST path when done grid-native; the earlier
  "cheap-uniform vs slow-adaptive" fork was a false dichotomy. SEQUENCING: explicit Worker
  rollout (D1d) continues first (shared infra); terrain RTIN builder is N8-D2. OPEN (settle
  while building): RTIN-patch → ≤128-tri cluster mapping; error metric (vertical RMS vs max,
  flatness weight); cross-tile crack-freeness for the 4 km field; LOD0 leaf-cluster count.

- D-N33 (2026-06-14, N8-D1 — DAG draw envelope + cut error scale, from the user re-test):
  a DAG'd mesh inherits the chain's MAX draw distance (the setMaxDistance value on the tail:
  trees TREE_GEO_FAR 496 m, rocks/deadwood clsMaxDist) — NOT the head's chain-SWITCH distance
  (the bug: attachDag retired the chain but left the switch as the envelope, so the cull rule
  `lodNext==NONE && lodDist>0 && dist>lodDist` dropped the WHOLE instance at 26 m / 120 m),
  and NOT unbounded. WHY NOT unbounded (lodDist=0): tried + MEASURED — 3.70M clusters / 92 ms
  (~11 fps) at an open vista even with occlusion ON, because the cut PINS the root
  (parentErr=1e30, never cut) so every sub-pixel far object still draws ≥1 cluster; HZB can't
  save an open plain. Unbounded IS the intended N8 end state (retire the impostor far-field
  with real geometry) but is GATED on a MIN-SCREEN-SIZE cull (drop an instance/cluster whose
  whole projected extent < ~1 px) — a D1e prerequisite, not a free lunch. Finite-intended-
  envelope ships now: fixes the vanish, matches pre-DAG nanite tuning, 16 ms. SECOND fix
  (same trace): the cut's screen error must scale the LOCAL-metre own/parentError by the
  instance scale A.w (the spheres already do), else non-unit-scale instances pick the wrong
  LOD band. Regression locked: probe-dagpack 2-LOD-chain envelope assertion (red/green) +
  NEW tools/probe-envelope.ts (−300 m far-pose, no collapse) + probe-envperf.ts (perf).

- D-N34 (2026-06-14, N8-D2a — terrain DAG construction + the D-N32 OPEN questions SETTLED):
  the terrain builder is martini's O(n) vertical-error PYRAMID feeding BuildDag's locked-
  boundary cluster-DAG scaffolding (NOT pure martini getMesh). REASON (decisive, supersedes
  any "pure RTIN" reading of D-N32): pure RTIN/martini is crack-free ONLY under a per-FRAME
  bintree traversal (ROAM-style); D-N31 fixed the runtime as a FLAT per-cluster cut (terrain
  must use the same, no per-frame traversal), and a flat cut over independent per-error-band
  martini meshes T-junctions at the frontier where adjacent regions at different distances
  pick different bands. The ONLY construction that is both crack-free under the flat cut AND
  adaptive is martini-error-metric × BuildDag-locked-cluster-DAG. Settled OPEN items:
  • (error metric) MAX vertical deviation in METRES (martini), not RMS, not a flatness weight
    — zero vertical error IS flatness; metres projects directly through the existing cut.
  • (RTIN→cluster mapping) clusters are clusterize()'d spatial patches of each error band;
    group = the BuildDag group, so own/parent (error,sphere) share bit-for-bit across siblings
    exactly like rock → the flat cut is crack-free by the SAME proof.
  • (decimation) on-grid half-edge collapse to the lower-martini-error ENDPOINT (F4: survivors
    stay on the grid; store packed `gx|gz<<16`, GPU fetches height — no baked floats), ERROR-
    BOUNDED by doubling bands e₀·2^ℓ (flat plains collapse at the lowest band, cliffs DEFER
    upward → smooth ±1 cut). LOD0 = full grid (ownError 0 ⇒ no up-close holes).
  • (crack-freeness) inherited from BuildDag PLUS three manifold guards that regular-grid
    endpoint-collapse needs (QEM-optimal on irregular rock never trips them): link-condition,
    near-collinear-degeneracy, all-3-verts-locked seam-triangle. Proven 0 cracks on an
    adversarial flat|ramp|ridge field (probe-heightdag W check, canonical-grid-id keyed).
  • (cross-tile 4 km) DEFERRED: the 2560 m field is ONE RTIN domain (no cross-tile cracks);
    the far shell's multi-tile stitch is D3 (far field), not D2.
  • (LOD0 leaf count) falls out of e₀ × terrain roughness, MEASURED in D2b on the real field.
  All flag-guarded by `gridEndpoint` (default false) ⇒ rock/bark/deadwood path byte-identical
  (probe-dag re-green). OPEN for D2b: build SPEED (0.01 Mtri/s — iterative QEM heap + guards;
  needs a martini-DIRECT removal and/or the D-N30 Worker for 4096²); the GPU grid-coord-indexed
  vertex-decode path (one isHF-indexed branch in NaniteFetch); the 2^k+1 grid reconciliation.

- D-N35 (2026-06-14, N8-D1e — the min-screen-size cull is NECESSARY-but-INSUFFICIENT for the
  unbounded envelope; D-N33's "gated on min-screen cull" was optimistic). BUILT the primitive
  (gated `?nanitemin=<px>`, default 0 = exact pre-D1e path): a per-CLUSTER cull dropping any
  cluster whose projected sphere radius < minPx (crack-safe — the gap is sub-pixel), plus a
  prototyped per-INSTANCE unbounded envelope (persist until the whole instance is sub-pixel).
  MEASURED + REVERTED the envelope half: at minPx=1, removing the finite cutoff yields 3.75M
  clusters / 100 ms WITH occlusion ON (reproduces D-N33's 3.70M) — because the count is
  dominated by the sheer NUMBER of scattered instances inside a kilometre-scale envelope (a 1 m
  rock survives to ~660 m, a 5 m tree to ~3.3 km at 1 px), and the pinned root means each costs
  ≥1 cluster. A per-instance/cluster size test CANNOT fix an O(instances) blow-up. CONCLUSION:
  the unbounded far-field ("retire impostors with real geometry", D-N33 end state) requires
  HIERARCHICAL instance culling (cull spatial GROUPS of distant instances at once → O(regions)),
  NOT a size flag — a real architecture task (own milestone), so the impostor far-field STAYS for
  now. What ships: the per-cluster min-screen primitive (gated, default-off, inert within the
  finite envelope — drops ~150 clusters — but the correct foundation). Validated: probe-minpx.ts
  (A/Bs minPx 0 vs 1, occl ON, far-700 m); probe-envelope.ts default path byte-identical
  (183679/175085). The finite-intended envelope (D-N33) remains the shipping default.

- D-N36 (2026-06-14, N8-D2b — terrain DAG GPU WIRING landed + headlessly validated). The
  D2a builder (BuildHeightDag) is now wired to the GPU so terrain renders through the SAME
  flat kClusterCull cut as rock/bark. Three pieces, all gated behind `?nanitedterrain=<gridN>`
  (default 0 = the discrete window path, byte-identical):
  • DECODE (NaniteFetch): a THIRD heightfield branch keyed on `isDAG` (CLUSTER_FLAG_DAG, read
    from the cluster flags byte alongside isHF). Adaptive terrain has EXPLICIT topology (an
    index buffer like rock) but each vertex's word0 holds a packed TEXEL coord (gx|gz<<16); the
    branch reads it by vertex index → `texLoadR(heightTex,gx,gz)` for height + `gx*cell+oX` for
    world XZ, then shares the EXACT window-path micro-disp. (isHF && !isDAG stays the implicit
    window grid; identity terrain instance makes the DAG cut's instTransformPoint a no-op, so
    HF+DAG rides the cut unchanged.)
  • PACK (GeometryRegistry): a lean `registerHeightDag` (entry with hf origin/cell + the
    HEIGHTFIELD flag but ZERO clusters — avoids the ~342k orphaned window clusters a
    register-then-repoint would waste) + `attachHeightDag` (mirrors attachDag's cluster +
    10-float DAG records, but packs grid coords into vertex word0 and sets
    CLUSTER_FLAG_HEIGHTFIELD|CLUSTER_FLAG_DAG). Words 1-5 of each vertex are UNUSED (height
    from heightTex, normal from normalTex) — a known 5/6 memory waste; a stride-1 terrain
    vertex buffer is a later opt (matters only at full res).
  • RECONCILE (WorldRegistry glue): build on a gridN² power-of-two SUBSAMPLE of the field
    (stride = res/gridN), then REMAP the build's grid coords ×stride → texel coords CLAMPED to
    res-1 (texLoadR does NOT clamp — out-of-range would read garbage). cell/origin passed to
    the mesh = the TEXEL cell/origin (= the window path's exactly), so the decode lands on the
    same world points as the placed objects. Far+ edge gets a 1-texel degenerate skirt from the
    clamp (zero-area, collapses — harmless 2 km out).
  VALIDATED headlessly (probe-dterrain.ts, gridN 256 & 512): (1) DECODE — DAG near ≈ window
  near, terrain seated correctly under the scattered trees; (2) CRACK-FREE — elevated vista has
  no sky holes / T-junction gaps; (3) ADAPTIVE — `?nandbg=cluster` tint shows cluster sizes
  VARYING with detail+distance (coarse on plains/far, fine on cliffs) vs the window path's
  uniform fine grid everywhere; (4) CUT live — dagClusters 320→245 @256² / 1058→857 @512²
  near→vista; ~8 ms, no boot error. STILL gridN-subsampled (validation res): full-res 4096²
  (1 m cells, near-camera parity with the window grid) is ~5 min sync ⇒ blocked on the D1d
  Worker — the remaining gate before this can be the DEFAULT terrain path.

## GOTCHAS (append-only, nanite-specific)

- (N5-C1) A SHADOW-CASTER NodeMaterial MUST SET `map = null`. three's shadow
  override builder (Renderer._getShadowNodes, three.webgpu.js:61050) gates the
  cast-alpha texture path on `material.map !== null` — NOT truthiness. A plain
  `new NodeMaterial()` leaves `map` UNDEFINED, and `undefined !== null` is TRUE, so
  three runs `reference('map','texture',material)` on a MISSING texture →
  "texture() expects a valid instance of THREE.Texture()" thrown at
  TextureNode.setup → the override fails to build → that caster writes no depth →
  no shadows. MeshStandardNodeMaterial sets `this.map = null` so the old Forests
  caster never hit it; a hand-rolled base NodeMaterial caster must do the same.
  (colorNode/castShadowNode/maskNode use truthy `&& .isNode` checks — only `map`
  is the `!== null` trap.) Symptom that misleads: shadow.c0 burns 18+ ms (pure
  vertex over-draw runs fine) so it LOOKS like the caster renders — but the depth
  attachment stays empty.
- (N5-C1) THE SHADOW PASS IGNORES A MESH's `vertexNode`. three swaps in a shared
  depth OVERRIDE material per light and reads ONLY colorNode / depthNode /
  castShadowPositionNode (or positionNode) off the source material
  (_getShadowNodes). A vertex-pulled caster MUST inject world position via
  `material.castShadowPositionNode` (LOCAL space → modelViewMatrix·it), and the
  mesh MUST have identity matrixWorld (matrixAutoUpdate=false +
  matrixWorldAutoUpdate=false + matrixWorld.identity()) so LOCAL==WORLD and the
  cascade light VP (the active camera during the shadow render) projects it
  correctly. Setting `vertexNode` instead silently renders the geometry's raw
  position attribute. side=DoubleSide (NOT FrontSide) or the default front→back
  shadow-side flip culls terrain's single up-faces ⇒ terrain casts no shadow.
- (tooling) Vite SERVES three from `build/three.webgpu.js` and PRE-BUNDLES it into
  `.vite/deps/chunk-*.js` — editing `node_modules/three/src/**` does NOTHING
  (src is not the served path), and editing the deps chunk is overwritten on
  re-optimize. To instrument three: patch `build/three.webgpu.js`, then
  `npm run dev -- --force` (re-optimizes from the build file). The dep optimizer
  needs `optimizeDeps.esbuildOptions.target='esnext'` in vite.config or it rejects
  three's top-level-await capabilities file on a cold cache.

- (N4-C4) BLACK SLATE HAS NO SHADOWS — the CSM map is EMPTY in the default
  black-slate build, so any shadow-RECEIVE check must run `?oldgeo=1`. Both
  caster sources — the terrain ShadowProxy AND the Forests per-cascade caster
  siblings — are gated behind `!DISABLE_OLD_GEOMETRY` (TerrainScene), and nanite
  cluster shadow casting is N5 (not yet). So with the default slate the resolve
  SAMPLES a cleared (all-far) shadow map ⇒ shadow factor = 1 everywhere ⇒
  `?nanshadow=0` vs on is a visual NO-OP. A naive A/B there shows a large
  "difference" that is PURELY cross-boot TRAA jitter on high-frequency bark
  texture (10.9% of px at bm7, salt-and-pepper riding the fissures) — it looks
  like shadows in the aggregate number but a red-overlay diff reveals it is NOT
  coherent shadow shapes. LESSON (re-confirms the framealign law): to test
  receive, run casters on (`?oldgeo=1`) AND frame-align (`--framealign N --wind 0
  --lockexp 1`) so the beauty−noshadow diff is the pure shadow term; verify the
  diff is COHERENT (a cast shadow with edges), not speckle, before trusting any
  shadow %. The migrated tree camera draws stay hidden under oldgeo (suppress-
  Migrated) so nanite bark still owns its pixels and receives the old casters.
- (N4-C4) NO-BLACK-SHADOWS is not an absolute luma floor — it is ALBEDO
  RETENTION. A fixed luma-floor gate on shadowed bark fights two NON-bugs: the
  tonemap toe (deep forest shadow is correctly dark, D-N22 energy-correct) and
  the bark's own cavity-AO fissure crevices (deep crevices in low light go
  near-black BY DESIGN — a zoomed crop read 21% pure-black px but 57% brown =
  correct detailed dark bark). The real failure mode (a zero-ambient code bug)
  zeroes albedo → flat GREY-BLACK with no chroma. So gate the warm-albedo
  fraction of the shadowed subset (chroma retained ⇒ albedo×ambient ≠ 0), not
  its min/p1 luma. bm7 shadowed-sunlit bark = 100% warm-albedo, 0% void.

- (N4-C3) STORAGE-TEXTURE MIPS DON'T REGENERATE AFTER A COMPUTE WRITE. three
  auto-generates a texture's mip chain ONCE (when first bound) — for a
  StorageTexture/StorageArrayTexture that's the COMPUTE storage bind, BEFORE the
  kernel fills level 0, so every mip > 0 is EMPTY. Sampling lands on an empty
  mip ⇒ pure black (the resolve's distant/grazing bark). The OLD 2D bark hid it
  by only ever sampling near mip 0. FIX: call `renderer.backend.generateMipmaps(
  tex)` AFTER the bake compute (BarkSynth.bakeBarkArray) — it downsamples the
  now-filled level 0 and self-submits. Symptom signature: forced .level(0) is
  correct, .level(2+) is black.
- (N4-C3) Hardware auto-mip is UNUSABLE in the resolve übershader: the per-pixel
  uv is computed inside `If(matClass==bark)` (non-uniform control flow), so
  WGSL `textureSample`'s implicit ddx/ddy are UNDEFINED → garbage mip → black on
  the whole class, not just silhouettes. Must pass an explicit level/grad.

- (N4-C0) THE SCANLINE DEPTH WAS BIASED-NEAR ON SUB-PIXEL TRIANGLES from
  N3a until N4-C0: the integer cz interpolated with the TOP-LEFT-BIASED
  edge weights (the −1 fill-rule biases) but divided by the UNBIASED
  area2 — a RELATIVE error of ~Σbias/area2, ulp-level on big triangles
  (the N3 comment "≤1/area2 ≈ 2⁻²⁵" silently assumed area2~2²⁵) but
  ~5e-4 of FULL DEPTH RANGE on far sub-pixel slivers (area2 ~10³
  units²), always NEGATIVE ⇒ far terrain depth landed hundreds of
  meters NEAR. Every internal gate was BLIND by construction: depth and
  payload share the formula (audit exact-match ✓), the resolve never
  compared depth to truth, hwref parity diffs silhouettes (coverage,
  not z), and the HW path only overlaps SW where the error is sub-ulp.
  The FIRST external depth consumer (water depth-testing the resolve's
  frag_depth at N4-C0) made the lake vanish wholesale. Found by
  exact-number GPU forensics (storage-buffer probe kernels — PNG-based
  number probes are POISONED by tone-map/sRGB; never decode quantities
  through the canvas). Lesson made law: a value is only verified when a
  consumer DIFFERENT from its producer checks it against independent
  truth — self-consistent pairs prove consistency, not correctness.
- (N4-C0) Slot collisions between stacked one-off GPU diagnostics
  (kSelfTest vs the in-kernel dump both writing audit[2..3]) produced
  two phantom "findings" (z>1 corners, 1e-3 corner spread) that cost a
  bisect cycle each — single-writer discipline for debug slots, or
  partition the buffer per probe.

- (N3) The page background #06080a SUMS TO EXACTLY 24 — every probe
  black/hole threshold written `r+g+b < 24` silently classified background
  as geometry, making probe-pan's N2 hole gate VACUOUS (it measured
  nothing; phase-2 evidence stood on p2-append counters, and a re-run with
  `<= 24` re-verified 0 holes honestly). Lesson made law: NEGATIVE-CONTROL
  every new pixel gate — force a known hole/diff and watch the gate fire —
  before trusting its first PASS.
- (N3) Debug views that freeze content on "camera stopped" must UNFREEZE
  on motion: hwref's stable-latch survived a probe TELEPORT and rendered
  spawn-frozen content from the horizon pose (wedge-of-nothing symptom).
- (N3) The #boot overlay fade and the always-on #hud-fps DOM chip pollute
  Playwright pixel gates (a low-alpha full-screen fade shifts EVERY pixel
  past tolerance) — hide both elements before screenshotting.

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
  ?nanitedag=all.) (d) move buildDag to a background Worker (D-N30 — three-free,
  typed arrays in/out; per-pool progressive: discrete LOD until each pool's DAG lands, then
  swap; off the boot critical path per F15). (e) bark is the heavy class, 162k trees → WATCH
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
       independent, self-contained (resolve-side only). ← NOW.
  S1 — Shadow-pass WPO-FREEZE + STATIC/DYNAMIC split: freeze trunk-wind in the shadow
       raster beyond a near distance (~80–150 m) → rigid → the R1 cadence caches it; cache
       rigid cascade depth, each frame atomicMin only the near-windy clusters on top
       (fixes the stale-wind-on-static-cache gap the R0 header flagged). DAG-independent.
  S2 — TIGHTER shadow cull (shadow-visible / needed-texel mask) + harder TEMPORAL STAGGER
       of far cascades + cast far cascades from a coarser DISCRETE LOD (interim, pre-DAG).
       Cuts the moving re-raster cluster count + frequency. DAG-independent.
  S3 — SCREEN-DENSITY RESOLUTION ALLOCATION (clipmap, not fixed 4-cascade splits): ~1
       shadow texel per screen pixel near = crisp penumbra + texel efficiency. The beauty
       centrepiece. DAG-independent (big).
  S4 — DAG-DECOUPLED caster LOD (GATED ON N8 DAG): cast each band from a progressively
       coarser DAG cut (near ~2× / mid ~8–16× / far coarsest). The clean perf engine.
  S5 — ADDITIVE BEAUTY CEILING: capsule-SDF soft inter-tree occlusion (wind-free blobs for
       the large-scale soft band) + screen-space CONTACT shadows (fine sub-pixel band).
       Optional, once the sun term is fast + crisp.
REJECTED (killer citations in D-N29): full VSM (cache dies to wind; Metal disables
StaticSeparate); unified shadow+GI as primary sun shadow (Lumen keeps a separate VSM;
SVO "too soft"); SW-RT of deformed triangles (per-frame BVH rebuild over 162k blows it).
CONSTRAINTS (binding): WebGPU — NO 64-bit atomics, NO HW RT, ≤10 storage buffers/stage,
~1.5 GB UMA. Zero external assets. Deterministic seed. WIND shadows stay correct. Quality
floor: no black shadows, soft penumbra, no pop within 300 m.

The R2/R3/R4 chunks below are SUPERSEDED by S0–S5. NOW BUILDING: S0 (half-res shadow
eval + bilateral upsample) — measure the bm7 static shadow-sample cost first, build, re-measure.

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

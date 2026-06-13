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

## GOTCHAS (append-only, nanite-specific)

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

N4 in flight (recon done 2026-06-13; D-N16..D-N19 record the architecture).
Chunks, each tsc-clean + committed:

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
4. **N4-C3 — BARK + DEADWOOD + trunk wind channel**: barkTextured/deadwood
   need texA/texB PER POOL → texture array or atlas decision (16 sampled
   textures/stage budget); uv f16 + analytic grad derivatives (example
   verbatim); hueShift×vdata; Wind.ts 'trunk' channel into fetchWorldVert
   (cull already swayPad-padded, F6) — gate runs --wind 0; separate living-
   wind eyeball + a wind=1 shimmer sanity probe. DEADWOOD shares the bark
   texture path + moss/rot terms.
5. **N4-C4 — close**: shadow-receive verification (shadow-color +
   no-black-shadows pass), full battery (probe-nanitedbg/pan/parity/
   horizon-nanite + registry probes), perf ledger row at 2592×1676,
   USER CHECKPOINT note (?nanite=1 vs ?nanite=0 bookmarks), PROGRESS LOG.
   DEBRIS class branch arrives with its pool migration at N6 (the debris
   ring is not in the registry yet — N1 pool policy; phase-table wording
   predates it). Velocity output: see D-N16 (deferred, no consumer).
6. Then N5 (per-cascade cluster shadow re-culls) per the table.

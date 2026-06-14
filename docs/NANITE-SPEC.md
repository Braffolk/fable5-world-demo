# NANITE WORKSTREAM — SPEC (durable design + decisions, branch `nanite-raster`)

> **THREE-DOC STRUCTURE** (split 2026-06-14 from the old monolithic NANITE.md, which
> grew to 3k lines and let durable DECISIONS get buried in journal noise → re-derivation
> loops). Each doc has a different volatility + a different read rule:
> - **`NANITE-ROADMAP.md`** — the task DAG + "you are here". Volatile. **Read FIRST on
>   every compact** — it is the anti-circle entry point and points into this SPEC.
> - **`NANITE-SPEC.md`** (this file) — the durable design contract + DECISIONS (D-N*) +
>   GOTCHAS. Stable. **Read FULLY on every compact.** Never re-plan from scratch; never
>   re-derive a D-N* decision — if one looks wrong, challenge it explicitly, don't silently
>   re-litigate.
> - **`NANITE-LOG.md`** — dated PROGRESS LOG + PERF LEDGER + in-flight research. A growing
>   journal: **read recent entries**, skim older. (The old prose `NEXT ACTIONS` lives there,
>   SUPERSEDED by the ROADMAP.)
>
> PROVENANCE: this SPEC descends from the ORIGINAL Claude Fable 5 spec, preserved
> unmodified at `reference/fable5-original-NANITE.md` (commit 8ac94518). The D-N* below
> are MY (Opus) additions/overrides on top of it. Some overrides are deliberate +
> measured (e.g. D-N29 drops the original's CSM-cascade shadows for the screen-density
> clipmap — see that decision); others may be unjustified drift. The deviation audit vs
> the original is a tracked ROADMAP task — when in doubt, the original's reasoning stands
> unless a D-N* gives an explicit counter-argument.
>
> Also read on rehydration: `docs/THREE-NOTES.md` (three.js API gotchas) + `reference/
> three.js webgpu - compute rasterizer lighting.html` (the proven TSL reference — re-read
> the kernels before touching raster code). `STATUS.md` = main-project context (do NOT
> update it for nanite work). Per session: append to the LOG, update the ROADMAP DAG,
> append D-N* / GOTCHAS here when a decision/trap is durable; commit per milestone with
> measured numbers. The built-in task tool is a disposable live MIRROR of the ROADMAP's
> active slice — NEVER the source of truth (not in git, doesn't survive compact).
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

| **PERF-1 — pure-nanite isolation + WORST-view decomp** (2026-06-14, probe-worstpos, cam −4.2/303.1/−1.4 yaw 67.5° T11, 2592×1676, freeze off, NOT cooled) | — | — | — | — | — | 6 (pure) | THE measurement reset. `?pure` (= postmin + nanshadow=0 + nandbg=flat) isolates pure nanite; GpuProfiler hardened vs GARBAGE negative timestamps (render total was −97 ms ⇒ --gpusample dropped all samples). Worst view (bare-tree hillside vista) = **82k visClusters / 130k hwTris**. **PURE (cool, 95 fps):** SW raster depth **2.82** + payload **2.95** = **5.77 ms**, HW **2.62**, flat resolve **2.10**. **FULL BEAUTY (hot, 35 fps):** the SAME raster reads depth **5.96** ms ≈ **2.1× thermal throttle** from the ~60 ms post chain (bloom 16 / TRAA 15.7 / rt#16 aerial-composite 14.5 / half.mrt 12 / resolve 8.3). ⇒ SW depth+payload raster = the #1 nanite lever (PERF-3); the post chain throttles the nanite side ~2× on top of its own cost. Absolute ms sit in a HOT envelope (no idle-between) — relative decomp is sound, cooled-ABAB pending. |

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

- D-N37 (2026-06-14, N8-D1d — terrain DAG is the ONLY terrain in DAG mode; build is AWAITED +
  CACHED; NO window/default fallback EVER. USER DIRECTIVE, verbatim intent: "it must immediately
  boot only to dag. there will be no fucking fallback to default rendering behavior"). A first
  attempt at off-boot building used a WINDOW PLACEHOLDER that rendered while the DAG built
  off-thread, then swapped in after boot — REJECTED outright + reverted (the user will not accept
  the window terrain shown even transiently). The correct shape:
  • DAG mode (?nanitedterrain>0) registers ONLY the lean DAG (registerHeightDag — no window
    clusters); the build is AWAITED before build(), so the FIRST terrain frame IS the DAG. The
    window path (registerMesh heightfield) survives solely as the SEPARATE DAG-OFF mode
    (?nanitedterrain=0), never a fallback within DAG mode. A worker failure falls back to a SYNC
    build (still a DAG, briefly blocks the main thread) — NEVER to window rendering.
  • The boot-cost problem (a ~5 min full-res build can't block every boot) is solved by a
    PERSISTENT CACHE, not a placeholder. DagCache.ts (IndexedDB), key = seed + gridN +
    DAG_CACHE_VERSION (the heights are deterministic in ?seed=N; bump the version on any worldgen/
    build change). MISS → subsample + build off-thread (three-free worker) + persist; HIT → load
    instantly, skipping the subsample AND the build. So the FIRST ever boot for a (seed,gridN)
    builds once; every boot after renders the DAG in ~tens of ms. Cluster records pack into one
    Float64Array (20 fields — the subset attachHeightDag reads; parentError ±Inf survives the
    round-trip); gridVerts/indices ride as their own typed arrays; all best-effort (any IndexedDB
    error → rebuild, never a crash). VALIDATED (probe-dagcache, two boots in one context): boot1
    [worker] 911 ms → boot2 [cache] 21 ms, byte-identical cut (320 cl). The window-swap path is
    gone; cache miss = build-and-wait, cache hit = instant, both DAG-only.
  REMAINING for full-res DEFAULT: (i) gridN=4096 needs a STRIDE-1 terrain vertex buffer (the
  6-word vert × millions of verts = GBs) + confirming the 2^k+1 clamp skirt at the true field
  res; (ii) flip the default on once (i) lands and the one-time ~5 min first-build UX (a loading
  state — NOT window terrain) is acceptable.

- D-N38 (2026-06-14, N8-D2 — TILED terrain DAG for full-res scale; the seam problem is solved
  FOR FREE. USER DIRECTIVE: tiled streaming DAG → true 1 m, "maximum quality+perf endpoint").
  MEASURED the single-DAG wall (probe-heightdag-scale, synthetic but the LOD0 floor is exact):
  a single 4096² DAG = ~76 M tris / ~640 k clusters → ~1.1 GB stored (stride-1) AND ~640 k
  per-frame cull invocations — over WebGPU buffer limits AND a cull-dispatch wall (gridN=1024 =
  46 k cl renders at 8.3 ms; 14× that does not). So full-res is NOT one DAG; it must TILE +
  STREAM (also the right shape for a 4 km world — cull/stream distant terrain). gridN=1024 (4 m)
  validated viable as a single DAG NOW (8.3 ms). THE KEY FINDING that de-risks tiling: crack-free
  seams need NO new builder code. buildDag already LOCKS any vertex on a mesh-boundary edge (used
  by ≠2 soup triangles, BuildDag.ts:460-486; locked verts never collapse, :557). A tile's outer
  PERIMETER is a mesh boundary ⇒ its edge verts auto-lock at full res ⇒ two adjacent tiles, both
  sampling the SAME shared texel column at the SAME stride, get bit-identical locked edges ⇒
  crack-free. STAGE 1 DONE + validated (probe-dterrain 256 ×4): T×T independent tile DAGs
  (?nanitedterrain=<gridN>&nanitedtiles=T), each gridN² over its texel sub-region (tileTexels =
  res/T; per-tile build positions in the tile's world origin, then gridVerts remap to GLOBAL
  texel coords; one registerHeightDag+attachHeightDag per tile; per-tile cache key suffix). 4×4
  @256 = 16 tiles, 48 840 cl, 5.85 M tris, offGrid 0, lit vista CRACK-FREE (no sky gaps), tint
  shows the 16 tile regions yet a seamless surface, 6.6–9.8 ms. REMAINING for the endgame:
  STAGE 2 = distance STREAMING (load near tiles / evict far → bounded memory + cull; needs
  registry eviction + per-frame tile selection) + STAGE 3 = stride-1 terrain vertex buffer +
  flip the full-res default. Tiles build SEQUENTIALLY now (16×~700 ms) — parallelise (N workers)
  or rely on the cache for full-res.

- D-N39 (2026-06-14, N8-D2 STAGE 2 architecture — the streaming design, grounded in a re-read of
  the cull pipeline. SUPERSEDES the D-N38 Stage-2 sketch's framing.) THE KEY FINDING that reshapes
  it: **the cull is INSTANCE-DRIVEN, not a flat global dispatch.** NaniteCull pipeline = kInstCull
  (one thread per INSTANCE, instanceCount threads) → frustum+occlusion cull the instance's world
  sphere → only VISIBLE instances call lodSelectAndPush → enqueue that mesh's [clusterBase,count)
  range into the chunk queue → kClusterCull (one 64-wide workgroup per queued chunk) runs the flat
  per-cluster cut. CONSEQUENCES: (1) an off-frustum / occluded tile enqueues ZERO clusters — it
  costs one inst-cull thread, nothing more. So STAGE-1 TILING ALREADY SOLVED THE CULL WALL FOR A
  GROUND CAMERA (frustum culls most tiles); the D-N38 "640 k cull invocations" wall was the SINGLE
  4096² DAG = ONE instance whose whole range enqueues whenever on-screen. (2) What streaming
  ACTUALLY buys is **(a) MEMORY** — every attached tile's verts/idx/clusters/dag stay resident in
  the mega-buffers regardless of visibility (~1.1 GB for all of 4096² stride-1) — **and (b) the
  VISTA cull** — when many fine tiles are simultaneously visible (camera lifted), they collectively
  re-enqueue the whole 640 k; evicting far detail → a coarse base bounds it. So streaming bounds
  MEMORY + the vista; it is NOT needed for the ground-camera cull (tiling already did that).
  EVICTION is therefore SIMPLE: a tile = (mesh entry + identity instance + contiguous slot). To
  evict, set the tile mesh's bounding sphere OFF-WORLD → kInstCull frustum-culls it → it enqueues
  nothing; AND free its slot for reuse. NO cluster tombstoning is needed for cull-correctness (no
  instance ⇒ no enqueue; mesh records + clusterBase/Count are mutable post-build + re-uploadable,
  exactly as attachDag already repoints them). To avoid unbounded mesh-handle growth over a session
  of loads/evicts, use a FIXED POOL of S (slot, meshHandle, instance) triples allocated once; each
  slot is a fixed-capacity byte block (clusterBase constant per slot; only clusterCount+sphere+the
  slot's buffer CONTENTS change per load) — the Nanite "page" model, O(1), no fragmentation.
  ARCHITECTURE = two-layer "coarse base + streamed detail" (clipmap-with-backstop): a single
  always-resident COARSE global terrain DAG (T=1, gridN≈512 = 8 m) covers the whole field → NO
  HOLES ever incl. vista, small mem, cheap (its own cut sheds it by distance); + the field split
  T×T into regions (T=16 → 256 m regions, 256² stride-1 = TRUE 1 m), only regions within radius R
  of the camera RESIDENT (the S-slot pool), a per-frame streamer diffing desired-vs-resident →
  load near (cache→attachTile / worker build) / evict far. The base is a DAG, not the window
  fallback — no-fallback rule (D-N37) HOLDS (both layers are DAGs; window only ever under
  ?nanitedterrain=0). INCREMENTS (each tsc-clean + committed + probe-validated): **2a** registry
  tile-slot POOL + evict (reserveTilePool / allocTileSlot / attachHeightDagTile reusable write into
  a fixed slot / evictTileSlot off-world+free) — foundation, NEEDED BY EITHER 2b policy, headless
  probe (load/evict/reload, no corruption, bounded). **2b** base DAG + TerrainStreamer (per-frame
  residency from camera XZ; engine.onUpdate hook). **2c** base SUPPRESSION where detail covers it
  (else z-fight): per-cluster region test in kClusterCull vs a T×T resident-bitmask uniform → skip
  base clusters in detail-covered regions. **2d** crack-free base↔detail seam: SKIRTS on the
  detail-window perimeter (base backstops the T-junction; standard clipmap). **2e** (was Stage 3)
  stride-1 terrain vertex buffer (6×→1× vert mem) + flip the full-res default on.
  FORK RESOLVED → **GEOMETRY CLIPMAP** (subsumes both candidates; cleaner than either): L levels,
  each a fixed M×M grid of SAME-gridN tiles at DOUBLING stride (level k cell = baseCell·2^k, tile =
  gridN·2^k m), each level a HOLLOW ring centered on the camera (its inner block is covered by the
  finer level k−1; the coarsest FULL level is the always-resident backstop spanning the field). WHY
  it beats the two candidates: (1) same gridN every level ⇒ UNIFORM slot cap ⇒ 2a's pool holds all
  resident tiles directly (no per-tier pools); (2) hollow rings ⇒ levels DON'T overlap ⇒ NO z-fight
  ⇒ **2c suppression is NOT NEEDED** (deleted from the plan — no hot-kernel edit, no region
  bitmask); (3) the coarsest ring = the "base" (no separate layer); (4) clipmap = correct quality to
  the horizon (concentric resolution rings), not a single 1 m→8 m jump. Inter-LEVEL seams (finer
  ring's outer edge meets coarser ring's inner edge, different stride) are hidden by SKIRTS (2d) —
  no neighbour-aware build, tiles stay independent (cache + perimeter-lock intact). Memory/cull
  bounded by Σ(per-level ring tile counts), independent of field size. REVISED 2b/2c/2d: **2b-1**
  route the existing Stage-1 tiles through the 2a pool (GPU-render parity with Stage 1 — proves
  pool→GPU in isolation); **2b-2** static clipmap at boot (build/load the spawn-centered ring set
  via the pool — proves the clipmap geometry + bounded + no-holes, sync/cache-backed, fixed camera);
  **2b-3** per-frame STREAMING (camera moves → re-center rings → async DagWorker loads / evict at
  ring edges; the already-resident coarser ring covers a fine tile WHILE it loads ⇒ graceful, never
  a hole); **2c REMOVED** (clipmap needs no suppression); **2d** skirts for the inter-level seams.
  2b-1 + 2b-2 (math + GPU) DONE + committed (the clipmap RENDERS, bounded, true 1 m at the field
  center, ?nanitedclip=1). 2b-3 FACTORING (worked out during 2b-2, build it this way): move
  buildTileGlobal INTO a new TerrainStreamer (ctor takes reg + heights + res/cell/origin/gridN +
  seed + a PERSISTENT DagWorker — clip mode must NOT dispose the worker after boot). WorldRegistry
  in clip mode: reserveTilePool sized for clipmapMaxTiles + in-flight headroom and the GLOBAL-worst
  tile cap (not just the boot set — a slot reloads arbitrary tiles; size generously + catch-skip on
  attachHeightDagTile overflow, the coarser ring backstops), construct the streamer, AWAIT
  streamer.loadInitial(centerTexel) (frame-1 terrain, no fallback), return it in the result.
  TerrainScene hooks engine.onUpdate(() => streamer.update(camWorldX,camWorldZ)). update(): camTexel
  → clipmapTiles → diff vs resident Map(tileKey→slot): EVICT departed immediately (cheap), LOAD
  arrived async (cache hit → attach next frame; miss → DagWorker.buildHeight, attach on resolve, a
  few attaches/frame to avoid hitches; if the tile left `desired` before the build returns, free the
  slot + discard). Cache key is per (gridN, stride, tx0, tz0) ⇒ revisits are instant. No per-frame
  reg.flush needed (attachHeightDagTile's pushRange marks needsUpdate → three auto-uploads).
  **2b-3 DONE + committed** (log an) — built ≈ the factoring, with two refinements found in build:
  (i) build-BEFORE-alloc + a cap-PRE-check (read reg.tilePoolCap) so a slot is never held across an
  await nor leaked when attach would overflow — over-cap tiles are SKIPPED (coarser ring backstops),
  not thrown; (ii) a `skipped` Set so a permanently-over-cap tile is built ONCE per residency, not
  rebuilt every frame (pruned when it leaves `desired`). update() is single-flight (busy-guard +
  pendingCam coalesce to the latest camera; one batch/frame, the next frame re-drives the drain).
  Headless probe-stream + GPU probe-streammove green: resident≡desired + bounded + no-leak at every
  pose, re-centers EXACTLY on return. **FINDING that reshapes the no-hole story:** the clipmap is
  hole-free at STEADY STATE and under CONTINUOUS motion (the just-departed backstop ring stays
  resident), but a FAST TELEPORT into never-cached terrain has a TRANSIENT void — because EVERY
  level re-centers, the coarse backstop ALSO churns, and its cache-miss rebuilds (~170 ms/tile,
  serialized) lag the jump. So the "coarsest ring = always-resident backstop" claim only holds while
  the camera stays put / moves slowly. THE FIX (new increment, before 2d): a genuinely
  always-resident, NON-churning coarse base spanning the field — either (a) pin level L-1 to a FIXED
  field grid (camera-independent) and REVIVE a minimal suppression so the camera-centered finer
  levels hide it where they cover (the deleted 2c, now justified — overlap is back once the base is
  fixed), or (b) a depth-underlay base (but the coarsest maxErr ≈ 194 m makes a flat Y-bias
  unviable). (a) is the path. Until then: realistic motion is hole-free; teleports/bookmarks show a
  brief void on first visit (instant on cached revisit).

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


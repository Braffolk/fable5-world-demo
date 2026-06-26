# Voxel LOD + raster perf — session state (2026-06-26, written for compaction/rehydration)

Read this FIRST to rehydrate. Branch of work = **`nanite-voxdaglod`** (worktree
`/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod`), off production `nanite-raster`.

## North star
Solid 60fps (p0.05 ≤16.6ms) in the FOREST under a MOVING camera, ZERO *visible* quality loss (methods free,
perceived image must match). The voxel-foliage path is the current frontier.

## The arc this session (newest = most important)

### 1. Structural research — `docs/perf-runs/2026-06-25-nanite-voxel-vs-ours.md` (23-agent workflow)
The ~10x voxel gap vs UE5 is STRUCTURAL (work emitted), not params/HW (HW only ~2-3x; UE voxelizes/rasters on the
GPU). Two portable causes: (Rank-1) no distance-LOD — degenerate always-cut DAG; (Rank-2) solid-AABB
over-election ignoring the baked occupancy mask. Memory: `nanite-voxel-vs-ue5-structural-gap`.

### 2. ROOT CAUSE of every prior DAG-LOD botch (premise-audit win)
The "locked" metric `ownError = cellSize·0.5·2^L` (errorK=0.5) pins the WHOLE LOD ladder into the NEAR field
(finest ~2m, coarsest-of-4 ~17m), BELOW the 35m mesh→voxel handoff → the entire [35,2000]m band rendered at ONE
coarsest level (the "semi-2-levels, no true dropoff" the user kept seeing). Shape (2^L) right, ANCHOR wrong.

### 3. The implement build (committed on `nanite-voxdaglod`, final sha 42f3308; tsc-clean)
Replaced the metric with a BAND-ANCHORED ladder `ownError(L)=anchorL0·errorK·2^L`, `anchorL0=transitionDist·tau/
projK`, 7 levels spanning [35,2000]m, + Rank-2 occupancy MASK gate (not literal DDA — per-pixel DDA fights the
per-brick depth-key + f2b ordering). Default-on. Flags added:
- `?voxlod=0` — old single-level degenerate DAG (Rank-1 off). `?voxocc=0` — old solid-AABB (Rank-2 off).
- `?voxlodk=` — anchor MULTIPLIER, **bigger = finer** (default 1). **User: `voxlodk=8` looks good on the single tree.**
- `?voxlodlevels=` (default 7), `?voxlodsparse=`/`?voxlodshell=` (default off, refuted-theory levers), `?voxdither=1`
  (opaque is default), `?voxao=0` (NEW — flatten the per-voxel N·L sun shading; NOT a perf lever, just look).

### 4. RESULT of eyeballing the build (user)
- ✅ It COARSENS now — true far-field dropoff works for the first time (the anchor fix landed).
- ❌ At voxlodk=1 it was WAY too coarse (big flat rectangles). `voxlodk=8` fixes the look on the single tree.
- ⚠️ Flat solid rectangles = Rank-2 occupancy carve self-disables on dense bricks (OCC_MASK_FULL=48) + OR-merge
  densifies fill 0.17→0.78 (far reads denser/darker). Not yet addressed.

### 5. CPU 600% on voxlod=1 — RESOLVED, red herring
Single tree voxlod=1 briefly hit ~600% renderer CPU. Investigations (agent a78dc0a + workflow wdsm7aysw) proved:
the voxel build is ONCE at boot (no workers in the forest scene; DagWorker is terrain-only), per-frame JS is
byte-identical to baseline. The 600% was a ONE-TIME shader/pipeline compile (bigger voxlod=1 shader) that "stopped"
after compiling. NOT a per-frame cost. Ignore it.

### 6b. UPDATE 2026-06-26 pm — atomic theory REFUTED, premise overturned, cost UNLOCALIZED (workflows wlbc8kgla → wcrjath77)
- `wlbc8kgla` ranked #1 = a single-global-address debug atomicAdd (NaniteVoxelRaster.ts:962, the `voxBrickWrites`
  counter). I gated it behind `?voxwrites` (default OFF; build-time TSL conditional ⇒ byte-identical removal).
  **USER MEASURED: zero delta.** REFUTED. (User's point: atomics have been ~free for a decade; not a 25ms cost.)
- `wcrjath77` (localize workflow) overturned the standing premise: **"the 33ms is in the SW raster loop" was never
  once per-pass measured.** Arithmetic ceiling: ≤~77M iterations (caps + strided loop verified to codegen) ⇒ ~3-5ms
  at any sane per-iter cost. **Work-COUNT cannot be 33ms.** Surviving hypotheses: (a) raster IS home but
  **occupancy-collapse / memory-stall** bound (≤19-36 WGs = ≤~4,600 threads close-up, latency unhidden), NOT work
  count; (b) **cull over-emission** (brick/cluster count secretly jumps close-up); (c) **a different pass**.
- RED-TEAM caught a measurement bug: `voxf2b` defaults ON ⇒ symptom URL dispatches `nanVoxScatterB0..B15` (16 depth
  buckets), NOT bare `nanVoxScatter`. Must sum the scatter FAMILY (`/^c\.nanVoxScatter/`). New suspect: the F2B
  machinery itself (16 serial dispatches + barriers + `nanVoxClearBins` + `nanVoxOccPyr*` occlusion pyramid).
- **PENDING: user runs `window.__laas.measureFrames({frames:21,warmup:10,cooldownMs:50})` FAR vs CLOSE** (snippet
  sums scatter family + occpyr + compute/render totals + voxClusters). Decision table: scatter-family grows +
  voxClusters flat ⇒ Lever 1a (occupancy point-splat, UE5 ScatterVoxels: iterate 64 occ bits not the rectangle,
  splat voxel centers into the existing election @:962-966; footprint-independent; also the forest's ~10x lever).
  voxClusters jumps ⇒ cull. render_total grows ⇒ resolve. compute−family/occpyr grows ⇒ F2B machinery.
  **Do NOT implement any fix until the table localizes the cost.** measureFrames (drain-isolated) NOT the live HUD
  (vsync-ghosts at FAR). Workflow scripts: voxel-33ms-localize (wf_e52609e9-39f).

### 6c. UPDATE 2026-06-26 pm2 — MEASURED: the cliff is r.scene SHADING, not the raster (workflow wm00cqjqk)
USER ran `window.__laas.measureFrames` FAR vs CLOSE (drain-isolated). GROUND TRUTH:
- INSIDE CROWN (~30fps): gpuWall 42.2ms, **r.scene 37.49ms** (render pass), compute_total **0.52ms**,
  *SCATTER_FAMILY **0**, voxClusters 32.
- FAR (~100fps): gpuWall 11.2ms, **r.scene 7.93ms**, compute_total 0.72ms, voxClusters 32.
- ⇒ **The voxel RASTER (all compute) is ≤0.72ms — NEGLIGIBLE. The cliff is `r.scene` = NaniteResolve deferred
  SHADING, 7.93→37.49ms (+29.6ms, ~95% of the delta).** voxClusters=32 at both poses ⇒ NOT cull, NOT more geom;
  only COVERAGE changes. ~30ms to shade one tree's pixels = ~15-30x a normal deferred shade ⇒ a per-pixel
  PATHOLOGY in the voxel resolve.
- **THIS FALSIFIES the 5-month "raster is the bottleneck" framing for this case** (solid-AABB over-election,
  occupancy splat, atomic contention, F2B, occ-pyramid, one-WG-per-cluster, LOD coarsening — ALL ≤0.7ms here).
  The structural-gap memory targeted the RASTER; the single-tree blocker is in SHADING. (Forest may differ —
  needs the same measureFrames read on the real 200k forest; do that before trusting raster levers for it.)
- The resolve = `NaniteResolve.ts` (r.scene). Per-voxel AO/N·L term `voxao` at :241 (prime suspect). Other
  per-pixel suspects: GI/SH eval, sun shadow sample, material/normal decode, a divergent storage refetch of
  brick/voxel data, or a per-pixel loop/ray-march. NOT yet sub-localized.
- **PENDING: A/B at the close pose — toggle voxao/GI/shadow flags one at a time, re-read r.scene** (workflow
  wm00cqjqk = voxel-shading-localize, wf_0c37ebd7-bb1, produces the exact flag set + fix). Quick head-start
  probe: re-run snippet close with `&voxao=0`. Do NOT implement a fix until the A/B pins the contributor.
- measureFrames snippet (sums scatter family + occpyr, drain-isolated, drops capSuspect ghost frames) is in
  the convo / the red-team result of wcrjath77.

### 6d. UPDATE 2026-06-26 pm3 — r.scene cliff SUB-LOCALIZED + free fix shipped (workflow wm00cqjqk)
MECHANISM (proven, red-team-corrected): the +29.6ms r.scene cost = **per-covered-pixel voxel DECODE** — a
~5-6-deep SERIAL dependent storage-read chain (visBV→qVoxRasterRO→clusters.meshId→meshes.matClass→instances/
brickBase→voxelBricks.normal→tint→depthNode re-read; all device-memory `StorageBufferNode<uint>`, no texture
cache, NaniteResolve.ts:357-707) that LOW OCCUPANCY can't hide because the whole terrain shading subgraph was
compiled into the vox pass. Scales with COVERAGE (pixel count), not overdraw (single-winner vis-buffer, no
resolve overdraw). REFUTED by red-team: NOT divergence (close-up a cluster's pixels share voxId→coherent),
NOT demote-fullscreen-doubling (far=7.93 proves non-vox waves terminate), NOT shadow/GI (build-dead in forest:
ForestScene.ts:354-356 gi/csm/canopy=null ⇒ nanshadow/ablate=gi are NO-OPS). r.scene also includes the sky
atmosphere background (SunSky.ts:63) — common-mode, cancels in same-pose A/B.
- **SHIPPED (free, bit-identical): terrain guard** at NaniteResolve.ts:388 — `If(isT,…)` → `if (pass==='tri')
  If(isT,…)` (matches existing ROCK/BARK/LEAF guards :424). Strips the vox shader's only demote-forcing
  texture samples + ~14-sample graph ⇒ occupancy. = RUN 2.
- **PENDING A/B (close pose, guard live):** (1) base [vs orig 37.5 = terrain-guard win]; (2) base+`&voxao=0`
  [vs 1 = the voxao 3-deepest-read sub-chain]. Read snippet logs r.scene/gpuWall.
- **STRUCTURAL ENDGAME (the real fix, gated on A/B): raster-side G-buffer.** Write brick mean-normal(oct rg16)
  + tint(rgba8) at raster time (NaniteVoxelRaster.ts winner-store ~:966; RED-TEAM: brick record NOT in scope
  there — needs a Phase-A stash mirroring wgDensBits :633-634; inherits the atomicStore race, benign). Replace
  NaniteResolve.ts:685-706 chain with 2 coherent textureLoads. ~0.5-1 day. ALSO cuts the 200k forest r.scene.
  UE ref: docs/perf-runs/Nanite-UE5-shaders/extended-all-related-shaders/shading/NaniteExportGBuffer.usf +
  NaniteShadeBinning.usf (decode-once→shade-coherent; we re-decode per pixel = the generating flaw).
- Follow-on: collapse the 2 fullscreen resolve passes (tri −1000 / vox −999) into a coverage-binned compute.
- Workflow: voxel-shading-localize (wf_0c37ebd7-bb1).

### 6e. UPDATE 2026-06-26 pm4 — single-tree cliff SOLVED + 200k forest MEASURED (workflows wfg5a16gz, wlw4h11oh)
SINGLE TREE (wfg5a16gz, unbiased measure-first, NO fed hypothesis): the close-up 42→25fps cliff was the
voxel-raster front-to-back path `voxf2b` (default-ON) splitting the scatter into K=16 BARRIER-serialized
bucket dispatches (~2 clusters/bucket → catastrophic occupancy, ~16ms serialization floor scaling with K not
coverage; its occlusion-pyramid payoff was structurally dead — built once, never rebuilt between buckets).
FIX = voxf2b default 1→0 (single unordered scatter; atomicMax is order-independent ⇒ image-identical). Worst
pose 25.4→10.5ms gpuWall, **42→121fps, all angles at the 120Hz cap.** SHIPPED (commit 1e80a1c, on nanite-raster).
⚠️ ALL the prior r.scene/decode-chain/terrain-guard theories were REFUTED: r.scene is a PER-PASS PROFILER
ARTIFACT for this pipeline (resflat=3 made the shader free, r.scene 22→0.07ms, ZERO live-frame change; the
indirect-dispatched voxel raster never appears in `passes` at all). **TRUST gpuWall + live-fps, NEVER `passes`.**

200k FOREST @ forcevox=all (wlw4h11oh, unbiased, UE5-cross-ref): canonical config ~20-25fps. Clean gpuWall
A/Bs (premise-audit FORCED the decisive ?nores=1 test, refuting a 3-way converged "overdraw→occlusion-cull"
story). MEASURED:
- Frame is **COMPUTE/FILL-BOUND in the voxel-scatter Phase-B per-footprint-pixel loop** (NaniteVoxelRaster.ts
  ~:973). NOT the resolve (?nores=1 skip both fullscreen passes = −2.2ms only; `render:63ms` was the artifact),
  NOT shading/post/shadow (pure=1 ≈ baseline), NOT tree count (50k≡200k byte-identical — cull saturates <50k),
  NOT primitive count (aerial_high 25.5M tris=26ms vs ground_canopy 8.6M=40ms). Cost = covered-pixels × overdraw.
- **voxoccl (the default-on per-block occlusion cull) is NET-NEGATIVE at 200k** (−4.4ms when DISABLED, counters
  byte-identical) — gappy canopy can't be conservatively occluded (min-key 0 → never culls). It pays its own
  GPU cost for ~zero benefit at forest poses. **OPEN DECISION: flip voxoccl default OFF? — needs a single-tree
  close-up regression check first (it was tuned for that case).**
- voxlod coarsening IS load-bearing + working: voxlod=0 → voxClusters 9.7k→240k (25×) → 80ms.
- FIX SHIPPED: `voxrecip` (commit a927a1f, default-on, ?voxrecip=0 disables) — per-brick float reciprocal
  replaces the per-fragment int div/mod (Apple has no HW int-divide). Bit-identical. Worst pose 49.1→40.3ms
  (−8.8ms/−18%), eye-walk 35.5→32.7ms (30→33fps). Added default-off ablations ?nores=1, ?voxrdbg=2.
- **STILL ~24ms from 16.6 at worst pose.** Residual = overdraw-bound scattered global-mem traffic (one uncached
  visPayloadV[px] relaxed-load per footprint fragment) — ALU fixes can't touch it. **THE path to 60fps =
  reduce FRAGMENT COUNT: shell-only bricks (carve interior, paint only silhouette) + finer/occupancy-tightened
  footprints. Quality-sensitive structural change (= the Rank-2 lever in `nanite-voxel-vs-ue5-structural-gap`),
  surfaced for the user's call.** This is ALSO entangled with the user's "far trees → single SQUARE" coarseness-
  limit complaint (over-coarsening = solid blobs); the right LOD is both visually fine AND low-fragment.
- Method note: BOOT IS EXPENSIVE at 200k → harness boots once/config + teleports poses; thermal throttling
  inflates gpuWall absolutes ~30-60% (track refreshMs: cool~33ms, hot 50-73; counters + within-run ratios are
  thermal-invariant). Harnesses: forest-sweep.mjs, ab.mjs, sweep-multi.mjs, shotdiff.mjs (in worktree, untracked).
- Impostor bake now skipped under ?forcevox / ?noimpostors (committed 1e80a1c) — impostors ARE the live >424m
  far-field LOD in the NORMAL path, so default unchanged; only the voxel-superseded path skips the bake.

### 6f. UPDATE 2026-06-26 pm5 — occlusion audit + UE5-port design (workflows whpx1vxrn, wz5abjqsv, CODE-ONLY)
AUDIT (whpx1vxrn): we have ~8 empty/hidden skip mechanisms (not 2); the two runtime carvers BOTH fail at
forest/forcevox — voxoccl net-negative (near-plane gate + any see-through gap pins min-pool key 0 → never
culls), voxocc self-disables on the costly bricks (arms only dagLevel>0 ∧ area≥16 ∧ occCount≤48/64). Root =
upstream solid [center±half] brick-cube footprint projected with occupancy IGNORED at L0.
UE5-PORT DESIGN (wz5abjqsv, skeptical) — KEY NEGATIVE FINDINGS (save future-us from chasing these):
- ⚠️ **Occupancy-tightened BOUNDS (UE BlockBounds) ≈0 win on the dominant near-DENSE bricks.** Fill = screen-rect
  area `bbW·bbH` (NaniteVoxelRaster.ts:969), DEPTH-INDEPENDENT; a front-facing dense canopy brick fills the
  screen plane (all 4 cells on both screen axes), its empty cells are behind in DEPTH → tightening the AABB buys
  ~0 screen pixels. Also a silhouette-thinning/popping quality risk. SHELVE bounds-tightening.
- ⚠️ **Per-fragment DDA discard (UE RayCastBrick) is NET-NEGATIVE for us** — no ROP/HiZ to make a discarded
  fragment free, so a missed fragment still pays its march; carves ≈0 on dense. DO NOT implement.
- Per-cell scatter (UE ScatterBricks): ≈0 perf on dense; only a possible voxocc SIMPLIFICATION (removes the mask
  buffer), adopt iff measured ≥ neutral.
- THE residual is CROSS-BRICK OVERDRAW (class C): the unconditional per-fragment `aLoadU(visPayloadV[px])`
  :994 (the `If(candL>prevE)` :995 gates only the store, never the load). NO occupancy lever touches it.
- DROP-LIST: **voxoccl → DROP** (dead weight; caveat: confirm net-neg outside forcevox before flipping the
  production default). **voxocc → KEEP** (still earns on the coarse/far sparse band). per-pixel earlyZ = the
  election, nothing to drop.
- The ONE small UE-derived portable lever worth a MEASURED shot: **1a silhouette-corner reject** (`?voxsil`) —
  reject the ~20-35% of footprint-rect fragments outside the projected cube's hexagon silhouette; occupancy-
  INDEPENDENT so it DOES hit dense bricks; ~1.3× ceiling. Cheap octagon form: in Phase A accumulate min/max of
  (x+y) and (x−y) over the already-projected 8 corners (:701-705), reject in Phase B before :994 (~8 ALU, no
  matrix/rcp, no new buffer). ADOPT IFF measured net-positive (must beat the partially-cached load it dodges).
- **PREMISE-AUDIT (both design agents independently): none of these is "the fix."** The real levers are one
  level up and OUT of the occupancy charge: **(1) near-LOD brick-COUNT reduction** (fewer L0 bricks = the
  multiplier on everything) and **(2) per-pixel FRONT-TO-BACK early-out** (UE TileBricks-style; kills cross-brick
  overdraw). Both need leaving brick-parallel for tile/pixel-parallel or changing the DAG cut. The whole-pipeline
  review (wxdbsv124, running) is expected to rank LOD/F2B ABOVE anything occupancy.
- NEXT: await wxdbsv124's whole-frame cost map + ranked levers; implement in priority order (likely F2B-overdraw
  / near-LOD, + 1a as a small measured add-on, + drop voxoccl). Nothing to implement until GPU frees + the review
  ranks. Workflow scripts: voxel-occlusion-systems-audit (wf_1f79696a), ue5-voxel-occupancy-port-design (wf_05675d3c).

### 6g. UPDATE 2026-06-26 pm6 — "could we BUILD a software early-Z?" answered (workflow wuqn7vfvw, CODE-ONLY)
The "DDA discard net-negative bc no ROP" conclusion was challenged (user: when you say "can't bc we lack Y",
first ask "could we HAVE Y?"). Costed, not hand-waved. VERDICT:
- ✅ YES, software early-Z is BUILDABLE — form (c) tile-F2B IS one, built TWICE; "no ROP" is REFUTED by
  measurement (the tile raster's per-pixel early-out half WON ~16%, 8.85 vs 10.55ms).
- ❌ BUT it does NOT net-positively kill the residual on OUR structure, for a deeper reason than "no ROP":
  UE's early-Z pays off because it gates a heavy per-fragment DDA occupancy ray-march. WE resolve occupancy at
  Phase A (baked mask), so our fragment's only cost is addressing + one global load — and **the load IS the test**
  (must read nearest-depth to reject). Nothing expensive left to gate ⇒ the early-Z prize collapses to a single
  per-fragment LOAD CONSTANT, cuts ZERO fragments.
- Buildable forms COSTED + REFUTED: (a) depth-only prepass = net-NEGATIVE (runs all F to build depth, then all F
  to read ⇒ DOUBLES the load; fused variant hits the hard 3-atomic-storage-buffer Metal cliff 15-17ms + broken
  writes, D-N45). (c) tile-F2B = MEASURED net-neg, removed TWICE (?voxraster=bin +4.8..+14.1ms; ?tileproto
  +11.7/+18.1ms — frame is COVERAGE-bound not submit-bound; bin floor taxes the brick-count multiplier). (b)
  software Hi-Z = dominated, min-pool gap-pins to 0 on canopy.
- SURVIVOR (one-probe follow-on): **depth-mirror** — swap the :994 atomic load for a plain cached load from a
  +1 plain `depthMirror` buffer seeded by f2b winners (visPayloadV atomicMax stays source of truth ⇒ provably
  pixel-identical), gated in electHere :993-998. RETIRES voxoccl. BUT benefit = atomic_load − plain_load only,
  which on Apple/Metal/Dawn may be ~0 (same cache; miss cost identical). Speculative constant-factor, not a kill.
- ⭐ **PRIORITY CORRECTION (premise-audit, the consolidated conclusion across ALL voxel design workflows):**
  overdraw residual = brick_count × footprint × depth-complexity-D. Early-Z attacks only the occluded FRACTION
  (D−1)/D. **NEAR-LOD brick-count reduction (rank-1 distance-LOD) attacks the MULTIPLIER directly** — cuts F
  itself (L0 brick count = the over-emission, the portable ~10× gap) AND deflates the D that bounds every
  early-Z's ceiling. Early-Z is downstream of + largely SUBSUMED by near-LOD. **ORDER FIXED: near-LOD FIRST;
  depth-mirror only after, on the smaller surviving F, only if its A/B pays.**
- DECISIVE PROBES (when GPU frees): (1) near-LOD brick-count reduction = the prime lever, prototype + measure at
  canopy/200k. (2) depth-mirror A/B = is an atomic-relaxed load materially costlier than a plain load on Metal at
  canopy? (settles the mirror's whole sign + retires voxoccl). Workflow: software-early-z-design (wf_f61f1533).
- DESIGN PHASE IS CONVERGED — next step is GPU IMPLEMENTATION (near-LOD), gated on the big pipeline review
  (wxdbsv124) freeing the GPU + ranking near-LOD whole-frame. No more design workflows needed for the voxel path.

### 6h. UPDATE 2026-06-26 pm7 — FULL-PIPELINE REVIEW reframes everything (workflow wxdbsv124, 10 agents, GPU-measured)
THE BIG MEASURED REFRAMING (supersedes the voxel-only framing for prioritisation):
- **FOREST IS COMPUTE-BOUND.** Matched-refreshMs nores pair: removing BOTH resolve passes did NOT reduce gpuWall
  (render collapsed 99→6.9ms, compute rose 60→69ms) ⇒ resolve/post/shadow OVERLAP and HIDE under the geometry
  compute critical path. **In the forest, post/resolve/shadow contribute ~0; the whole lever is compute (cull +
  SW triangle raster + voxel scatter).** (General world is the OPPOSITE: resolve ~22%/6ms, post ~11%, shadow ~9%.)
- ⭐ **PREMISE CATCH #1 (stands): the 8.5–32.6M triangles dominating forest compute are tree TRUNK/BARK, NOT
  foliage.** Under forcevox=all, ForestScene.ts:158 keeps bark maxDist=2000 (trunks raster as TRIANGLES to 2000m)
  while :290 suppresses the leaf head (leaves→voxel). visClusters:voxClusters ≈ 15:1 = bark-trunk : voxel-crown.
  The "voxel forest" cost is mostly the woody-skeleton triangle raster, by design.
- CATCH #2 (REFUTED by measurement — good news): forcevox=all is NOT a cost-inflating debug mode. At aerial,
  forcevox vs no-forcevox geometry is IDENTICAL (impostors don't engage there); at ground it's LIGHTER than
  shipping. **Our 200k forcevox benchmark is legitimate.**
- CATCH #3 (stands): the "per-pixel loop ≈90% of world1" claim is UNVERIFIED — the rdbg/voxrdbg stage-split flags
  are CONFOUNDED by HZB feedback (gutting a raster stage changes next frame's emitted geom 3–10× via the
  occlusion cull reading last-frame depth). Clean stage isolation needs geometry-neutral flags, not rdbg.
- CONFIRMED LEVERS (measured, thermal-invariant ratios):
  1. **Lever 3 — voxel per-pixel fill: CONFIRMED DOMINANT at the representative ground_canopy pose** (the real
     ~33fps target; voxrecip A/B ground/far ratio 1.75→2.55 when only the fill math changes). Forest-only,
     QUALITY-SENSITIVE (DDA-discard → foliage see-through) → USER'S CALL. Safe first step: build-time L0
     occupancy-tighten (?voxtight, no per-pixel cost). ⚠️ NOTE the wz5abjqsv design said bounds-tighten ≈0 on
     DENSE bricks while this review cites ~0.2 density (75–84% empty rect) → the two DISAGREE on brick density;
     ?voxrdbg=2 + ?voxtight + voxBrickWrites must settle it before building.
  2. **Lever 4 — resolve per-material specialization: CONFIRMED ~5–8ms/25–37% of the GENERAL frame** (~0 in
     forest). Quality-neutral, shippable. **FIX SHIPPED THIS RUN (uncommitted in worktree): `?reskeep=0`** drops
     a redundant full-screen CSM `keep` sample (three's cascade maps are EMPTY in the black slate ⇒ keep≡1);
     measured −3.9ms/~19% on depth-diverse general vistas, quality-neutral (shotdiff ≈ TAA floor); cascade fit
     preserved via a [0,0]-corner reference. NaniteResolve.ts:245,256,261,783-799,939 + NaniteFrame.ts:362.
     Recommended default-on after a fast-motion spot-check. ~0 forest benefit.
  3. **Lever 1 — SW trunk-tri raster ALU (rcp-hoist + 2-FMA depth plane + rect-small for ≤4px tris):** real &
     large (8.6M ground / 32.5M aerial tris), QUALITY-NEUTRAL, also speeds general + 6 shadow cascades. PRIMARY
     at aerial, SECONDARY at ground. Per-pixel SHARE UNVERIFIED (confounded flags) → implement behind
     ?rcphoist/?rectsmall to BOTH ship the win AND finally measure the split. UE: NaniteRasterizer.ush:238.
- KILLED: HW/SW reclassify (hwTris <3%), persistent-threads (non-portable Dawn/Metal), R64 election, F2B/tile
  voxel march (refuted prior), shade-binning (no wave ops), WPO velocity MRT.
- ⚠️ **THERMAL BLOCKER:** shared/busy machine this session → cross-boot gpuWall absolutes useless (refreshMs
  17→150; 200k boot pins it 58–75). Forest perf MUST be measured via within-boot ratios + geometry-neutral
  load-preserving A/Bs, or with a heavier cooldown / freeze-render idle hook. This gates precise forest lever ms.
- WHAT UE DOES BETTER (user's Q): sparse voxel ray-march+discard+tight block-bounds (vs our dense ~80%-empty-rect
  election); per-triangle rcp-hoist + 2-FMA depth + adaptive Rect raster (vs per-row div + per-pixel 9-ALU bary);
  one-material-per-dispatch shade (vs uber-shader). Workflow: pipeline-shader-review-vs-ue5 (wf_eba1bf81).
- OPEN DECISION (user's call): next lever = Lever 1 (trunk raster, quality-neutral, resolves the split,
  generalizes) vs Lever 3 (voxel fill, dominant at gameplay, quality-sensitive); + commit reskeep? + solve the
  thermal-measurement problem first.

### 6i. UPDATE 2026-06-26 pm8 — BOTH levers implemented + MEASURED → DROP; the real lever found (workflow wzzwsum7x)
"Do both" → implemented + validated behind flags, matched-condition A/B (machine now has CONSTANT 2-core load
8h → deltas/ratios reliable). Both DROP for shipping. Honest negative result that redirects.
- **LEVER 1 (trunk-tri raster ALU: rcphoist/rectsmall) — DROP, premise REFUTED.** Matched cross-boot A/B: 0.0
  ground / +1.8 aerial (rcphoist), ≤0.4 (rectsmall). visTris identical (geometry-neutral), quality-neutral.
  WGSL confirmed changed (not DCE'd) yet gpuWall flat ⇒ **Tint→Metal ALREADY hoists the rcp / FMA-contracts the
  depth plane** — our "optimization" is what the compiler does. AND the trunk raster is NOT frame-dominant: the
  heavy compute is **nanTraverseAB (cull) + nanVoxScatter (voxel fill)**. (Also caught a thermal confound — a hot
  baseline faked a −8/−11ms "win"; matched comparison erased it.) Revert the flags.
- **LEVER 3 (voxel crown fill: voxtight/voxdda) — DROP default-on, SURFACE.** Density disagreement SETTLED via
  ?voxdenslog: L0 bricks ~24% mean occupancy, 67% have ≤16/64 cells ⇒ bricks ARE SPARSE (the review was right,
  the "dense" camp wrong). So a real empty-rect fill ceiling exists — BUT:
  - Pure-fill A/B (occl DECOUPLED, occl=0, 24/24 kept): tight = −1.4ms ground / −0.9ms aerial = **REAL but SMALL
    (~3%)**, not the "big ceiling."
  - Shipping config (occl=1): **the saving EVAPORATES.** ⭐ ROOT CAUSE (go-up-a-level): the solid-AABB crowns
    DOUBLE AS the depth occluders feeding our voxel occlusion cull (voxoccl). Tightening/DDA-carving makes them
    see-through → the cull sees THROUGH them → more occluded foliage survives (voxClusters +14%, visTris +17%) →
    the revealed raster cost ≈ the fill saved. **FILL AND OCCLUSION ARE COUPLED.** UE's brick win doesn't port
    because UE ray-marches bricks AND its HZB occluder is SEPARATE geometry; ours uses the solid bricks
    themselves as the occlusion frontier, so carving them defeats our own cull.
  - QUALITY: voxtight is NOT quality-neutral (changes 56% of ground px — the "empty margin" was load-bearing
    phantom fill); voxdda is see-through (18% px). Both visible, default-off, surfaced.
- **⭐ THE REAL STRUCTURAL LEVER (surfaced, user's call):** DECOUPLE the occluder from the shaded fill — keep the
  solid AABB as the cull occluder but DDA/ray-march only the SHADED pixels (what UE effectively does). Larger
  redesign. OR attack the cull (nanTraverseAB) directly — newly visible as a heavy compute item, never
  investigated. Distance to 60fps UNCHANGED: ground_canopy ~29.6ms floor (~13ms over), aerial ~37.2ms (~21ms over).
- CODE STATE: worktree has uncommitted experiments (rcphoist/rectsmall=DROP/revert; voxtight=refuted; voxdda+
  voxdenslog=keep-as-knob/diagnostic if pursuing decouple) + the ready-to-commit reskeep general-scene win
  (NaniteResolve/NaniteFrame, −3.9ms vista, quality-neutral) + throwaway .mjs harnesses. Untangle on direction.
- Workflow: implement-levers-1-and-3 (wf_7d20765e). Harnesses: l3v-final.mjs, tight-ab.mjs, l1-ab.mjs (worktree root).

### 6j. UPDATE 2026-06-26 pm9 — decouple REFUTED at gate; DEFINITIVE convergence on near-LOD (workflow wvhr07zzj)
The occluder/fill decouple (user-chosen) was REFUTED at the go/no-go gate (no GPU burned). Sound + buildable +
occlusion-preservable, but: (1) ceiling is the ~1.4ms isolated carve (a thin slice; frame mass is trunk-tris
15:1 + cull nanTraverseAB + scatter); (2) **my "decouple unlocks near-LOD" premise was FALSE** — near-LOD
coarsens far crowns into FEWER/BIGGER/STILL-SOLID bricks, which STRENGTHENS the solid-AABB occluder and cuts the
multiplier directly; the cull only fights per-pixel CARVING, never COUNT reduction, so near-LOD needs no
decouple; (3) quality-negative (solid occluder + carved fill → gaps resolve to SKY, 56% px change).
⭐ **DEFINITIVE CONVERGENCE (all ~10 workflows): the ONE real lever is NEAR-LOD BRICK-COUNT REDUCTION** (UE
`Level = floor(log2(Distance·factor))` MIP) — attacks the ~2-5x STRUCTURAL multiplier (work emitted), cuts the
cull + scatter + raster simultaneously, strengthens the occluder. EVERY voxel micro-lever is DEAD/subsumed:
occupancy-bounds, DDA-discard, software-early-Z, decouple, raster-ALU (compiler already does it). #2 lever = the
never-profiled CULL (nanTraverseAB), the actual compute bottleneck. The frame is also 15:1 trunk-TRIANGLES whose
LOD-τ coarsening was NEVER measured (big-review Lever 2) → trunk count reduction cuts the same cull+raster.
⭐ **near-LOD IS the user's original "far trees → single SQUARE" coarsening concern** — the perf lever and the
quality limit are the SAME knob. The fix = a LOD curve fine-near, aggressively-coarse-far, CAPPED so it never
reads as rectangles. voxlodk=8 (the user's calibration) is FINER (more bricks, slower) precisely to dodge
squares — so there's a real perf↔quality tradeoff to get right. QUALITY-SENSITIVE ⇒ USER MUST EYEBALL (the
'does far coarsen without squares' observable, memory verify-user-observable-output).
SHELVED (cleared design, do NOT delete): ?voxoccdecouple (half-res per-brick atomicMin-splat occluder buffer
voxOccDepth + kVoxOccSplat between scatter & hzb.build NaniteFrame.ts:481 + HZB L0 merge NaniteHzb.ts:97-124) —
only relevant AFTER near-LOD lands AND if a residual per-pixel carve still pays; even then UE's per-pixel
occupancy hit-depth (one buffer, empty→discard, gaps show REAL depth) beats the parallel solid occluder.
CONSOLIDATED CODE STATE: reskeep SHIPPED (nanite-raster e46ec9c, default-off ?reskeep=0, general-scene −3.9ms
vista, quality-neutral, default-on pending fast-motion eyeball). All refuted experiments REVERTED (rcphoist/
rectsmall/voxtight/voxdda/voxdenslog). Worktree clean. Distance to 60fps: ground_canopy ~13ms over, aerial ~21ms.
NEXT: near-LOD brick-count reduction — understand current band-anchor/voxlodk curve → design aggressive capped
far-coarsening → implement behind flag → USER eyeballs coarseness + measure brick-count/gpuWall. Then cull
(nanTraverseAB). Workflow: voxel-decouple-occluder-from-fill (wf_a470218f).

### 6. THE EARLIEST FRAMING (workflow `wlbc8kgla`) — SUPERSEDED, see 6b/6c/6e/6f/6g/6h/6i then 6j
Single tree, camera ~2-3m (inside crown), `forcevox=all`: close-up fps 120→30 (~33ms) at **BOTH voxlod=0 AND
voxlod=1**. So it's the **BASE SW voxel raster**, PRE-EXISTING on nanite-raster, NOT the new pyramid/occupancy.
User: it's a code pathology (hundreds of squares should be ~free), it's FUNDAMENTAL, and probably a main thing
tanking the 200k forest. Do NOT hand-wave "big squares close = costly" or "non-issue because close".

VERIFIED base-raster code facts (NaniteVoxelRaster.ts, worktree):
- SW raster compute kernel; ONE workgroup per voxel CLUSTER; WG_RASTER=MAX_BRICKS_PER_CLUSTER=128 lanes.
- Phase A (1 lane/brick): projects brick → footprint AABB clamped to BRICK_MAX_EXT=64 (≤128×128 px/brick), :712-745.
- Phase B (128 lanes): SERIAL loop over the cluster's bricks (:925), each strided across lanes (:942). Per
  footprint pixel: relaxed GLOBAL LOAD `prevE=visPayloadV[px]` for EVERY brick-pixel even on loss (:956); on win
  atomicMax + atomicStore + **debug `atomicAdd(WRITE_CTR)` to ONE global addr (:962)**. `cand` = FLAT per-brick
  depth (:932,955) → billboards, solid AABB (empty corners painted).
- NO hierarchical/tile occlusion reject before the per-pixel loop. Only early-out = the per-pixel relaxed load
  (gates the ATOMIC, not the LOAD). f2b near→far buckets exist (`?voxf2b`).
- Magnitude puzzle: footprints CAPPED at ≤128×128 ⇒ naive fillrate shouldn't be 33ms for ~hundreds of bricks ⇒
  dominant cost UNCERTAIN: (a) low workgroup OCCUPANCY (single tree = few clusters = few WGs, GPU idle), (b) serial
  per-brick loop, (c) per-brick-pixel load fillrate, (d) atomic contention on hot center px, (e) debug atomicAdd,
  (f) far more bricks than "couple hundred". Workflow wlbc8kgla ranks these + gives the decisive measurement + fix.

## OPEN DECISIONS / PENDING (surface, don't park)
1. **band-anchor vs locked cellSize metric**: implementer OVERRODE the locked `cellSize·0.5·2^L`; 2 reviewers
   contest (band-anchor bakes projK at build → resolution drift; global const → large crowns may blob). The
   "too coarse at k=1" result is evidence the band-anchor absolute scale is off. Resolve empirically (voxlodk tune
   vs switch to cellSize metric which self-clamps to ~a-few-px). USER'S CALL.
2. Bake voxlodk=8 as default once the FOREST (not just single tree) is confirmed.
3. OR-densification → far darker/denser (visible quality); Rank-2 self-disable (OCC_MASK_FULL) + ~512 proj/brick
   build cost (maybe net-negative — the ?voxocc=0 A/B). LOD popping unmitigated (no cross-fade).
4. The base-raster fillrate problem (#6) — the active blocker.

## How to test (dev server runs from the WORKTREE, not the main checkout)
```
cd /Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod && npm run dev   # vite :5173
```
- Single tree (see dropoff, fly toward/away): `http://localhost:5173/?scene=forest&trees=1&nanite=1&naniteleaf=1&forcevox=all&voxdither=0&voxlodk=8`
- Forest (real target): `http://localhost:5173/?scene=forest&trees=200000&nanite=1&dpr=1.5&voxdither=0&voxlodk=8`
- A/B flags: `&voxlod=0` (no pyramid), `&voxocc=0` (no occupancy gate), `&voxao=0` (flat shading).
- Counters: `window.__laas.stats.counters['nanite.voxClusters']` (workgroup count), `['nanite.voxBrickWrites']` (election wins).
- Canonical forest perf config: scene=forest trees=200000, retina 2268×1473 (?dpr=1.5), voxdither=0 (opaque).

## Process / standing rules (from CLAUDE.md + memory — agents do NOT inherit these)
- Research workflows: ≥5-6 rigorous stages, NEVER fed a focus area beyond the goal; premise-audit + "go up a level"
  embedded verbatim. Implement workflows: review panel + A/B + keep/drop, premise-audit gate.
- NEVER run heavy GPU/probes myself (thermal + context). User runs measurements; agents do code-only.
- Worktrees: NEVER `git add -A` (symlink clobber); stage specific files. Branch work lives on nanite-voxdaglod.
- Gate on the USER-OBSERVABLE (does far visibly coarsen / does fps recover), not "machinery correct".
- Surface decisions, never park. Premise-audit on every negative result (the anchor bug was found this way).

## Tasks (this session)
- RUNNING: workflow `wlbc8kgla` — base-raster fillrate diagnosis (the active problem).
- DONE: research `wf_9c47edb8` (→ 2026-06-25 doc); implement `w2jgplmxm` (→ 42f3308 on nanite-voxdaglod);
  CPU diagnosis `wdsm7aysw` + agent `a78dc0a` (CPU = one-time compile, resolved); agent `acbba4e1` (voxao flag).

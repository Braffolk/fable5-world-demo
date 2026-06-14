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

## YOU ARE HERE — 2026-06-15
**FRONTIER: PERF-4 = the POST CHAIN (user pick, after compaction). PERF-3 (depth rasterizer) is CLOSED:
win #1 (makeCtx shared-mem cache) is LIVE (default ON, −0.59 ms / −11%); win #2 (the vertex cache) was BUILT,
MEASURED, and found a MARGINAL / CONDITIONAL non-win (R≈4.7 vs makeCtx's R=128; far-terrain transform too cheap
— texture-cache-absorbed), so it's kept OFF-by-default, isolated to `NaniteVertexCache.ts` (see LOG `bc`, D-N40
MEASURED OUTCOME). The in-kernel raster wins are exhausted (fetchWorldVert irreducible by caching; 2-pass forced
by no 64-bit atomics).**
**→ BEFORE PERF-4: read the new `## PERF METHODOLOGY — the bar for a real win` in SPEC. Post (~60 ms, ~10× the
raster budget) has MORE headroom and the gains should be LARGER, so the bar is HIGHER — target multi-ms wins, not
the sub-quantum margins of the raster kernel; and post cuts COMPOUND via the 2.1× thermal throttle. Profile the
post chain first (`r.UnrealBloomPass.*`, `r.TRAANode.resolve`, `r.half.mrt`, `r.rt#16` aerial) before touching it.**

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
| `PERF-4` | Post-chain optimization (the 2.1× thermal throttle) | 🔵 | `PERF-3` | SPEC `## PERF METHODOLOGY`; PostStack | **FRONTIER (user pick).** Post (bloom/TRAA/aerial/half.mrt ~60 ms) heats the GPU ~2.1×, inflating nanite raster itself (2.82→5.96 ms). ~10× the raster budget ⇒ more headroom, LARGER expected gains, HIGHER bar (multi-ms, not sub-quantum); cuts COMPOUND via the throttle. STEP 1 = profile `r.UnrealBloomPass.*` / `r.TRAANode.resolve` / `r.half.mrt` / `r.rt#16` cool. Hold the beauty bar (A/B the look). |
| `AUDIT-1` | Deviation audit vs original Fable 5 spec | ⬜ | — | PROVENANCE; `reference/fable5-original-NANITE.md` | diff current state/impl vs the 937-line original; flag unjustified drops from my D-N* edits (shadows = D-N29, justified) |

## B. DAG (N8) — active workstream (SPEC `### DAG (N8)`)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `N8-D1e` | Full-world DAG wiring + ledger + CHECKPOINT | ⬜ | — | DAG (N8) | close N8-D1: DAG across all migrated explicit meshes world-wide; perf ledger row; USER CHECKPOINT |
| `N8-2b4` | Always-resident coarse terrain base | ⬜ | — | DAG (N8) | teleport no-hole backstop ring |

## C. POOLS / HYBRID / FOLIAGE (SPEC `## Phase plan`)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `N6` | Migrate remaining opaque pools (debris) | ⬜ | — | Phase plan N6 | register debris pool → DAG applies on registration |
| `N7` | Hybrid close | ⬜ | `N6` | Phase plan N7 | finish the HW/SW hybrid envelope |
| `N9` | Foliage aggregate DAG (leaf-removal, area-preserving) | ⬜ | `N8-D1e` | DAG (N8) / Phase N9 | the DOMINANT shadow casters; unlocks `S4`'s full value |

## D. SHADOWS (S-stack) — clipmap banked, perf deferred below core raster (SPEC D-N29)
| id | task | status | blockedBy | spec | scope |
|----|------|--------|-----------|------|------|
| `S3` | Screen-density shadow clipmap | ✅ | — | D-N29(1) | DONE 6154604 — `NaniteShadowClip.ts`, `?shadowclip` default on |
| `S3-perf` | Shared inst-cull across levels + variable-T | ⬜ | — | D-N29 | DAG-independent clipmap perf (~1ms each); BELOW `PERF-3` in priority |
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

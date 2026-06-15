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
**FRONTIER: PERF-4 = the POST CHAIN, running effect-by-effect (user directive: one effect at a time, they review
when back). PERF-3 CLOSED (win #1 makeCtx cache LIVE −0.59 ms; win #2 vertex cache off-by-default, LOG bc/D-N40).**
**PERF-4 progress (LOG bd, be):**
- **AO ✅ SHIPPED** (`1db0bfd`): far-fade early-out (march+upsample) + packed-view-z bilateral + samples 8→6 =
  **~1.5 ms direct / ~4 ms render** (thermal un-throttle compounding); ALL beauty-validated 0.00% (`?aocheap`
  master toggle, `?aodbg` AO-term view, `?aosamples`/`?aofar`). AO is a near-flat ~0.8 cue here (sparse black-slate).
- **BLOOM ✅ INVESTIGATED — NOT optimizable.** Its ~6.5 ms is DRAIN-ABSORPTION, not pixel work: resolution-scaling
  is FLAT at native AND GPU-bound high-res (the `bright` pass absorbs upstream TRAA/scene drain into its
  wall-span). Reverted — scaling only changes the glow for zero perf. Bloom's cost falls when TAA is cut.
- **TAA 🔵 = the real remaining prize** (`TRAANode.resolve` fetch-bound ~6.8 ms, ~18 fetches/px: 9 depth + 8
  variance + history). Hitting 2–3 ms means cutting the variance ghosting-guard and/or the parallax-edge depth
  neighborhood — **MOTION-quality properties unvalidatable by static screenshots**, needing a ~300-line TRAANode
  resolve fork. **FLAGGED for the user: needs a sanctioned fork + their in-motion review; NOT forked solo** (the
  methodology forbids shipping unvalidated beauty changes).
- **AERIAL (`rt#16`) / CLOUDS (`half.mrt`):** thin safe headroom (clouds 1.4 ms half-res; rest is beauty-critical
  froxel fog + Hillaire haze). AO's share already cut.
**KEY MEASUREMENT REALITY (durable):** per-pass post timestamps LIE (drain-absorption — bloom reads 14 ms but is
6.5 ms); headless `frameMs` is rAF-capped at 16.7 ms (use high-res 3888×2520 to go GPU-bound for real `Δframe`, or
bracketed `Δrender`); big-effect marginals are thermally inflated. ABLATION is the truth. Tool: `probe-postablate.ts`.
**→ NEXT: the actionable post win is the TAA fork — the user's call. Otherwise PERF-4's safe shippable wins (AO)
are banked; bloom is a mirage; aerial/clouds are thin.**

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
| `PERF-4` | Post-chain optimization (the 2.1× thermal throttle) | 🔵 | `PERF-3` | SPEC `## PERF METHODOLOGY`; LOG bd/be; PostStack/Gtao | **IN PROGRESS, effect-by-effect (user).** AO ✅ `1db0bfd`: early-out + packed-view-z bilateral + samples 8→6 = ~1.5 ms direct / ~4 ms render, beauty 0.00% (`?aocheap`/`?aodbg`). BLOOM ✅ = NOT optimizable (drain-absorption, resolution-FLAT at native + GPU-bound high-res; reverted). TAA 🔵 = the real prize (fetch-bound ~6.8 ms, ~18 fetches/px); 2–3 ms cut needs a TRAANode resolve fork + USER in-motion review (ghosting unvalidatable headless) — FLAGGED, not forked solo. Aerial/clouds = thin (clouds 1.4 ms; rest beauty-critical haze). |
| `PERF-4-TAA` | TAA resolve fork: reduce neighborhood fetches 9-depth/8-variance → ~half | ⬜ | `PERF-4` | LOG be; TRAANode.js | **USER CALL** — projected ~3–3.5 ms (`TRAANode.resolve` 6.8→~3.5). Needs: (1) a sanctioned ~300-line resolve fork (or custom lean resolve), (2) in-MOTION quality review (variance taps = ghosting guard; depth neighborhood = parallax-edge velocity dilation; both motion-only, unvalidatable by static shots). Decide approach (fork three vs custom vs accept cost) before building. |
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

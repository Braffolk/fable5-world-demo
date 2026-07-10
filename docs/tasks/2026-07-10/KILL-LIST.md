# KILL-LIST — cleanup arc 2026-07-10 (merged from 3 inventory agents)

Totals: **~15,000 LOC deletable** (~6,600 src + ~8,400 tools) + a dual-shader-compile
build-matrix shrink + a boot-time win (unconsumed impostor bake).
Gates per slice: `tsc --noEmit` + worst-pose headless smoke
(`?cam=-582.1,302.4,1006.1,2.5692,-0.0077`) + census counter parity
(visTris/mid/clhw/hwTris/splat byte-match vs pre-slice). One commit per slice.

Headline finding: **`nanite=1` is not a code default anywhere** — bare URL boots an
EMPTY world (sky/water/clouds only) since the 06-13 oldgeo hard-disable. S0 fixes that
and is the prerequisite for the big S3 harvest.

---

## S0 — make nanite unconditional (prerequisite)
Hardcode `naniteOn=true` in TerrainScene.ts:48/80 + ForestScene.ts:525; delete the
`?nanite` param. Bare product URL boots the real renderer.

## S1 — fold defaults (behavior-identical for canonical config)
| item | payload | notes |
|---|---|---|
| **`?preset` removal (USER-DECIDED)** | param + 'ultra' arm | quality low→qualityConfig('low'), med/high→'high'; verify grid cfg in BootCache key or bump CACHE_REV |
| ~~`ksplit` → always 1~~ REFUTED (S1 premise-audit) | — | ksplit=1 is a MEASUREMENT scaffold: doubled whole-queue launches ≈ +1ms until the Stage-2 cull-side queue partition lands; `variant:'both'` in makeFetch is LIVE (depth/combined/ctx/vcache users), not the dead side. Param stays until Stage 2. |
| `fp16w` → always 1 — **NEEDS USER CALL** | f32 TSL wind path | fp16w defaults OFF: f32 is the SHIPPED default; fp16 path requires device shader-f16 with NO fallback (Engine gates `enable f16;` on feature presence) and is not bit-identical. Folding = making shader-f16 a hard hw requirement. |
| `ctxsm` → always 1 | register-hoist ctx variant (~10 ln) | kills a dual compile |
| `clhwmax` → const 32 | numeric fold | FIX the 3-site independently-defaulted skew (NaniteCull/NaniteRaster/NaniteResolve) in same commit |
| `hw1fetch` → always 1 | 3-corner fetch path raster/Hw.ts:292-299 (~10 ln + plumbing) | measured winner |
| `culloverlap` → always on | `=0` escape | shipped 07-09, smoked since |
| `grassbomb/grassshear/grasstilt` | 3 disable-escapes (~30 ln, 3 dual compiles) | long-shipped |
| disable-only escape batch (gisleep/coalesce/occg/voxocc*/sh*/aorg/traapp/…) | ~20 params, small dual paths each | fold in 2-3 batches after S1 core soaks |
| shadow/grass tuning constants (shnb/shdb/shslope/…/grassbak*) | fold to constants | NOT the bootcache-key ones |
| bootcache-key knobs (voxlod*/voxgrid/ftcell/clustertris/…) | KEEP as knobs OR fold + one CACHE_REV bump | ⚠️ silent cache reuse hazard |
| NaniteView.ts:61-69 duplicate LOD param set (different defaults!) | collapse to NaniteFrame's | latent skew bug |

## S2 — dead experiment arms (param + compiled branch)
| item | LOC | notes |
|---|---|---|
| `shadowclip` fold → **delete src/nanite/NaniteShadow.ts** | 466 | legacy 4-cascade nanite shadow + shadowtau/shadowminpx |
| `swcoop` + `coopv` | ~250 | cooperative/two-bin pixel loop, measured non-wins |
| `voxf2b/voxf2bk/voxtaucap` depth-slab mode | ~150 | ⚠️ KEEP the K-bucket plumbing — `voxprev` (live, default-on) rides it at K=2 |
| `trihzb` | ~120 | W2 per-tri occlusion experiment + Queues tail-fold |
| `vcompact` → **delete src/nanite/NaniteVertexCache.ts** | 106 | "bitrotted"; ⚠️ KEEP gpu.vcompact records (Project.ts consumer) |
| `grasslean` [JUDGMENT J7] | ~100 | refuted 07-04 but deliberately kept "may win on tap-bound GPUs" |
| `rdbg` 1-5 sinks [JUDGMENT J6] | ~100 | it's the raster measurement harness — recommend KEEP |
| `scar/scarnear/scarfar` | ~80 | superseded by ?census |
| `middz` incDepth branch (Scanline.ts:46-125) | ~60 | measured regression 07-09 |
| `grasspatch` | ~60 | demoted geometry-grass remnant |
| `pyrfuse` | ~40 | fused-pyramid experiment (verify no doc claims live) |
| `shvox` arm [JUDGMENT J8] | ~45 | "never worked"; recommend delete, resurrect from git if wanted |
| `nospl/nomid/nohw` | ~15 | self-declared temporary; blink fixed |
| `dvclear/visclear/hwrt/voxraster` | ~21 | A/B restores of removed clears + defensive no-op |
| KEEP: `hwproj` (flagship re-land base), `crownlod0`, `leaflodk` (active arc) | — | |

## S3 — legacy render path (unlocked by S0) — ~4,500 LOC
| item | LOC | judgment gates |
|---|---|---|
| Forests.ts + VegInstance + ImpostorRuntime + Impostors cascade | 1,787 | |
| GroundRing.ts (pre-nanite grass) | 993 | |
| TerrainTiles.ts + ShadowProxy.ts | 565 | J4: `view=split` erosion A/B also uses TerrainTiles |
| ShadowSetup.ts + CsmCached.ts | 517 | J3: `?shadowclip=0` CSM fallback — delete? |
| CanopyShell.ts | 135 | |
| impostor+atlas bake in VegLibrary (+ boot cost: 6sp×64views×3passes, unconsumed!) | ~80 | |
| VegPrepass depthPrepassTwin/PrepassNodes, oldgeo branches in NaniteResolve/Frame, suppressMigrated | ~200 | J2: `?oldgeo` parity harness still wanted? |
| Particles.ts (snow/pollen/leaves) | 274 | **J1: gated behind oldgeo — nanite world has NO ambient particles. Re-gate (recommended) or delete?** |
| FoliageCards.ts + card geometry | 322 | J10: needs deeper trace (does WorldRegistry clusterize card geo?) — verify during execution |
| F3 relic counters rejInst/rejClust (NaniteCull.ts:1524-1542) | ~20 | never filled in hier mode |

## S4 — dead files / scenes / exports
| item | LOC | notes |
|---|---|---|
| RasterSpikeScene + SpikeRaster + SpikeContent (`rasterspike`) | 1,348 | N0 pre-vis-buffer spike — recommend DELETE |
| GalleryScene + Dressing (`gallery`) [JUDGMENT J5] | 820 | Phase-4 review surface — user call |
| VoxelDebugScene (`voxdbg`) | 193 | header: "DELIBERATELY THROWAWAY" |
| SanityScene (`sanity`) | 180 | Phase-0 stack test |
| ShadowTestScene (`shadowtest`) | 139 | |
| src/nanite/raster/index.ts | 45 | zero importers (barrel) |
| 16 fully-dead exports | ~60 | barkMaterial, BIOME_NAMES, warp2, … |
| export-keyword prune (132 symbols) | 0 | optional, low value — skip unless trivial |

## S5 — tools/ (~8,400 LOC)
- **DELETE: 64 probe/one-off scripts** (full table in tools-audit output; all leaf files
  importing only launch/measure — zero blast radius). Includes 3 statically BROKEN
  (probe-voxlod-ab dead worktree path; treeconnect ×2 hardcoded port 5180).
- KEEP-CORE: launch.ts, measure.ts, shoot.ts, compare.ts, diff.ts, tools/profile/ (13),
  tools/perf/ (interleaved_ab.mjs + README).
- KEEP-NICHE (16): the 9 node-only validators (probe-dag/dagpack/heightgrid/aggregate/
  clusterize/clipmap/manifold-check/voxlod-occgate + herotris/vegtris) = de-facto unit
  tests, all resolve vs live src; probe-forestfull (the probe TEMPLATE); probe-motion
  (canonical measurement); probe-fresh-stutter; probe-worstpos; find-water; vcdebug.mjs.
- JUDGMENT J9: bench-tree.ts + gen-tree-geo.ts + tools/geo/ (closed geometry experiment
  — recommend delete); probe-profileswap.ts (one-off ?profile smoke — recommend delete).
- Untracked artifacts to `rm` (no commit): .DS_Store ×2, tools/.cache/, tools/geo/*.json
  (0-byte) + shots/, __pycache__/, last_ab.json regenerates.

## S4b — DEEP dead-code review (user 07-10: "import-graph sweep is bullshit... there
## may be dead crap thats imported and unused. or imported but gated under some
## non-existing vars now")
After S1-S4 land (tree is smallest + tsc-clean), a real symbol-level pass:
1. Deterministic tooling first: knip / ts-prune (unused exports, unused files,
   unused deps) + tsc noUnusedLocals sweep — machine truth, not LLM grep.
2. Dead-gate hunt: conditions that can never be true anymore — reads of URL params
   that no longer exist anywhere, window globals nothing sets, config fields with no
   writer, `if (x)` where x is a constant false / never-assigned option field.
3. Within-file dead symbols: exported-and-imported but caller-side dead (imported for
   a branch deleted in S1-S3), private methods with no call site.
4. Opus verification agents on every candidate before deletion (false-positive check:
   dynamic access, TSL/node-material reflection, worker `new URL` refs, ?raw SRC_HASH
   imports — this repo has all four patterns).
Findings → same gates, one commit.

## S6 — restructure src/nanite/ into hierarchical layout (user 07-10)
Pure moves + import rewrites after all deletions land. Same gates.

## S8 — after S7 (user 07-10): delete Impostors.ts + FoliageCards.ts
User: "after the scene standardisation we can get rid of impostors and foliage cards
(we simply remove the impostor render stuff from gallery)". S3 verified card geometry
is NOT in the live nanite pools (WorldRegistry defers card/leaf parts, registers only
opaque parts[0]+LODs), so this is gallery-review-surface surgery only: strip the
impostor demo row + card/fern/vine preview usage from GalleryScene, then delete
Impostors.ts (~326) + FoliageCards.ts (~322) + card hooks in TreeBuilder/Understory
+ captureFoliageAtlas→lib.atlases IF nothing live samples them (verify: bark/leaf
texture arrays for resolve are separate). ShadowSetup.ts + CsmCached.ts die earlier,
in S4 with ShadowTest/VoxelDebug scenes (their only remaining importers + gallery
already stripped).

## S7 — after S6 (user 07-10): scene unification
Go over GalleryScene + ForestScene: strip weird quirky scene-specific behavior and
share as much as possible with the world scene (TerrainScene) — one common
boot/frame/registry path, scenes differ only in content + camera, not in plumbing.
CRITICAL (user): sharing means EXTRACT into common functions/classes that all three
scenes call — NOT copying the world scene's code into the others. Zero duplication;
if two scenes need the same behavior, it moves to one shared implementation.
Watch for known drift: ForestScene defaults were NOT in bootcache SRC_HASH
(beautification-arc lesson); forest csm:null quirk; gallery's own knob reads.

---

## JUDGMENT CALLS — RESOLVED (user 07-10 approval round)
- **J1 Particles**: KEEP — re-gate to the nanite path + CRITICAL comment in the file
  that it needs rework (user verbatim: "keep particles with critical comment in the
  file that it needs rework").
- **J2 `?oldgeo=1`**: DELETE (user: "get rid of. its so old no one remembers wtf that is").
- **J3 `?shadowclip=0` CSM fallback + NaniteShadow.ts**: DELETE — verified unrelated to
  current shadows (NaniteShadowClip imports only the `NaniteShadow` interface TYPE →
  move the interface into NaniteShadowClip.ts or a types module; NaniteShadowHalf.ts +
  NaniteResolve.ts import the type too).
- **J4 `view=split`**: DELETE with TerrainTiles (default recommendation, no override).
- **J5 `gallery` (+Dressing)**: KEEP (user-confirmed).
- **J6 `rdbg` sinks**: KEEP generally (measurement harness), but DROP the unused/random
  individual sinks (user: "generally keep, but the unused random ones can be dropped") —
  audit each rdbg mode 1-5 during S2: keep the stage-stop ladder that the perf workflow
  actually uses, delete one-off sinks that no longer correspond to a real stage.
- **J7 `grasslean`**: DELETE (resurrectable from git).
- **J8 `shvox`**: DELETE.
- **J9 tools experiments** (bench-tree, gen-tree-geo, tools/geo/, probe-profileswap): DELETE.
- **J10 FoliageCards**: DELETE (user explicit: "foliage card atlas pipeline delete") —
  still verify during S3 that WorldRegistry doesn't clusterize card geometry into pools.

## Working-tree note (07-10)
Uncommitted docs reorg found (NOT mine, do not touch/stage): docs/NANITE-*.md +
deep-review/ + DELTA/DEVIATIONS moved → docs/legacy/; mobile-gpu-perf/ → docs/deep-research/.

---
# ARC CLOSED 2026-07-10 ~03:40 — all slices executed
S0 f66c8b0 · S1 15296a2 · S2 be9eb44 · S3 ede1282 · S4 c67bf19 · S4b 8babfe5 ·
S5 32b6c13 · S6 5abc47f · S7 374a2a0 · S8 fa0f7af (+ docs 7dda3ce/a2f200d).
Net ≈ −18.6k LOC (≈21k deletions). Every slice gated: tsc + worst-pose smoke +
counter parity + visual; per-slice commits for surgical revert.
OPEN ITEMS carried out of the arc:
- fp16w fold = user call pending (f32 stays the shipped default until then).
- TODO(missing-leaves, CRITICAL) sites from S8: shrubs bark-only, ferns gone,
  vine stems leafless — need real MESH leaf crowns (LeafMesh) in a content pass.
- Particles re-enabled with CRITICAL needs-rework comment (perf+visual audit due).
- ksplit stays until Stage-2 cull-side queue partition lands.
NEXT: the user's "biggest refactor demand yet" — brief not yet given.

# PLAN — Vis-buffer projection pre-pass (nanite SW raster rewrite)

**Task #76 (Apple/mobile-GPU arc). Status: LOCKED, not yet built. Date: 2026-07-07.**
**Supersedes `splat-election-register-cut.md` — that plan's splat election is now just ONE consumer
(`Splat.ts`) in this pipeline; its inline-projection premise (world1 keeps projecting) is what this
rewrite removes.**

---

## 0. Decision (one sentence)

Rebuild the world SW raster from ONE inline megakernel (`nanRasterWorld1` = project + classify +
rasterize, per tri, 80 regs) into a **vis-buffer pipeline**: **project each vertex once → classify/route
each tri → deferred per-class rasterization reading the projected verts by index.**

---

## 1. Why — the chain that got here

- Splat election (the old plan §5) is DONE: `nanSplatElect` = **24 regs**. But per its own R2, it did
  **not** lower `nanRasterWorld1` — world1 still projects every tri.
- The de-über (this session) split the **scanline** out (`nanMidRaster`), still **not** the projection.
  So world1 sits at ~76 (its 64-reg projection floor + the ≤1px edge-setup), and the mid record is a fat
  **10 u32** (world1's projection *output*, copied per-tri so the consumer needn't re-project).
- Near-forest, the mid band is the **bulk** (~16–32M tris; sub-pixel is a couple %, not 97% — that was a
  far/aerial figure). So the mid queue is **~1.28 GB**, and bigger caps don't fix the root.
- **Both walls share one root: projection is done inline, per-tri, and its result is either carried
  (memory) or recomputed (registers).** HW proves projection is *disposable* if it can be redone cheaply
  (HW redoes it in fixed-function, queues only an ID). A compute consumer can't redo it cheaply (that's
  the 64-reg floor landing in the consumer), so it stores it → the 10 u32.
- The fix: **do projection once, per vertex, in its own pass**, into a shared buffer. Then world1 becomes
  a lean classifier and the consumers read projected verts by index. Verts are shared ~6× (each used by
  ~6 tris), so storing them once collapses that redundancy.

---

## 2. Granularity — the tri-vs-cluster audit (the point of the rewrite)

| work | today | correct level | in this plan |
|---|---|---|---|
| **vertex projection** (transform+wind+clip→screen) | per-**tri-corner** (128 tris × 3 = 384/cluster, ~3× redundant vs ~128 unique verts) | **per-VERTEX** | `ProjectVerts`: one thread per vertex, projected once |
| makeCtx (transform mat, wind params, clhw class) | per-**cluster** (`ClusterCtx` pre-pass) | per-cluster | unchanged — already isolated |
| winding / bbox / size-class / near-plane | per-tri | per-tri | `Classify`, per-tri (reads verts) |
| coverage scanline + election | per-tri | per-tri | `Mid`/`Splat`, per-tri |

The per-tri-corner projection **is** the 64-reg floor at the wrong granularity. No "cluster calcs redone
per tri" and no "thread-0 does the cluster, 127 lanes idle" — `ProjectVerts` is a clean flat per-vertex
grid (proper data-parallel). `ClusterCtx` already killed the old thread-0-makeCtx-broadcast hack.

---

## 3. The pipeline

```
cull → visible clusters (qRaster)                                            [existing]
  ↓
ClusterCtx    per CLUSTER: makeCtx → transform mat + wind params + clhw class  [existing pre-pass]
  ↓
ProjectVerts  per VERTEX:  fetch + transform + wind + project → projVertBuf   [NEW]  ~58 regs, high occ
  ↓
Classify      per TRI:     read 3 projVerts → bbox → winding → ROUTE          [world1 becomes this] ~30 regs
                            ≤1px → splatQ | 2..swmax → midQ | big/near → hwQ   (bare tri-id records)
  ↓  (deferred; each reads projVerts by index)
Splat   sub-pixel point election                          → visbuffer
Mid     2..swmax scanline (shared Scanline) + election    → visbuffer
Hw      big/near — HW re-projects in its vertex stage     → visbuffer
  ↓
Resolve  deferred shade (unchanged)                                          [existing]
```

Key: `Classify`, `Mid`, `Splat` all **read the same `projVertBuf`** (indexed by the tri's 3 vertex ids
from the existing index buffer). Nobody re-projects except `Hw` (for free, in hardware).

---

## 4. `clhw=1` is the new default; `clhw=0` is DELETED

`clhw` (route trunks/near-terrain clusters to HW) is always on. Every `clhw===0` / `!clhw` branch is
**removed** as the relevant file is migrated — not left as a dead path. (One less axis of branching in
already-branchy code.)

---

## 5. Clean file layout — `src/nanite/raster/`

One pass/path per file; names say what they do; no `Raster2`/synonyms; disable-only debug flags OK,
no `?param` A/B *branches*; target ≤ a few hundred lines each. `NaniteRaster.ts` is gutted into these.

```
src/nanite/raster/
  ClusterCtx.ts   per-cluster makeCtx pre-pass                         (migrate kClusterCtx)
  Project.ts      projection math + ProjectVerts pass + projVertBuf    [NEW]
  Classify.ts     per-tri: read verts → bbox → winding → route         (world1's classify+route half)
  Scanline.ts     the ONE shared coverage loop                         (migrate swScanline)
  Splat.ts        sub-pixel point election                             (migrate kSplat*)
  Mid.ts          2..swmax scanline consumer                           (migrate kMid*)
  Hw.ts           HW raster path (materials, draws, hwQueue)           (migrate hw* )
  VisBuffer.ts    visbuffer buffers + depthKey packing + election + clear
  Queues.ts       splat/mid/hw queue buffers + 2-D dispatch args
  index.ts        assembles the per-frame pipeline (the only orchestrator)
```

Depth/combined (shadow + debug-view) raster reuse `Scanline.ts` — no second copy of the coverage loop.

---

## 6. `projVertBuf` — the one real design decision (sizing)

Projected vert = screen xy (1/256 fixed) + depth. ~2–3 u32/vert. Full-screen range (not bbox-relative),
so no small-range packing on xy.

| scheme | size @ dense forest | notes |
|---|---|---|
| **compacted** (atomicAdd global vert slot; project only used verts) | ~16M × 3 = **~190 MB** + IDs 128 MB ≈ **320 MB** | ≈4× smaller than 1.28 GB; needs a dedup/compaction step |
| **cluster-indexed** (`clusterSlot·MAX_VERTS + local`) | visClusters × 128 × 3 ≈ **~770 MB** | simple, no compaction; wasteful per-cluster reservation |
| ⚠️ **naive** (`QRASTER_CAP·128`) | 1M × 128 × 3 ≈ **1.5 GB** | **worse than today** — do NOT size this way |

Decide by **measuring the visible-vert peak** first (needs the high-water readback — see §10, now a
prerequisite). Start cluster-indexed (simple, ~770 MB, still a win), move to compacted only if memory
demands it.

---

## 7. Targets — TO MEASURE, not assumed (my register estimates have been wrong all session)

- `Classify` (world1): 64 → **~30** regs (Xcode capture, gate).
- `ProjectVerts`: **~58** regs — the projection floor **relocated**, not eliminated; runs at high occ,
  ~2–6× fewer projections than per-tri-corner.
- Mid record: **10 u32 → 1 u32**; `projVertBuf` per §6.
- world1 occupancy up because Classify + consumers are lean; net frame-time is the real gate (deferred
  until registers are good, per user law).

---

## 8. Negatives / risks (honest — read before building)

1. **Memory win is sizing-dependent** (§6) — naive sizing is *worse* than today.
2. **Projection floor relocates, not eliminated** — still a ~58-reg pass; we've only stopped it pinning
   world1 + the consumers.
3. **Big restructure** — this is UE5-Nanite's vis-buffer SW rasterizer shape. New pass, cull changes to
   surface the vertex set, world1 rewritten, consumers read verts by index. High cost/risk.
4. **Vertex indirection** — Classify + consumers read verts through the index buffer (scattered), vs the
   inline record today. Cache-friendly *within* a cluster, but real bandwidth ("parse more").
5. **Extra pass + serial dependency** — project → classify → raster is 3 stages with barriers, vs 1.
6. **Contradicts old plan §3.2** ("a pre-pass doesn't help") — believed a flawed premise, but that makes
   this an *unvalidated bet*; prove the 30/58 numbers with a capture before trusting them.

---

## 9. Build order (incremental; get each `tsc`-green + visual/register-checked before the next)

1. **Scaffold** `src/nanite/raster/` + move the *already-clean* pieces (`Scanline`, `Splat`, `Mid`,
   `VisBuffer`, `Queues`, `ClusterCtx`) out of `NaniteRaster.ts` verbatim — pure extraction, no behavior
   change, delete `clhw=0` on the way. Verify parity.
2. **`Project.ts`** — `ProjectVerts` pass + `projVertBuf` (cluster-indexed to start). Measure its regs.
3. **`Classify.ts`** — rewrite world1 as: read 3 projVerts → bbox → winding → route (tri-id records).
   world1 stops projecting. Measure world1/Classify regs (gate: ~30).
4. **Point `Mid`/`Splat` at `projVertBuf`** (read verts by index; drop the 10-u32 record → 1-u32 id).
   Verify visual parity + the mid blink stays gone. Measure memory.
5. **`Hw.ts`** unchanged in behavior (already re-projects) — just migrated + `clhw=0` removed.
6. Only then: frame-time A/B (the ship gate), and revisit `projVertBuf` compaction (§6) if memory bites.

---

## 10. Carried over from the earlier plans — still in scope (don't lose these)

Unfinished or shipped-as-keeper before; they remain part of THIS plan. Some become *more* important.

**Verification gates (splat plan §7 — now apply to the whole pipeline):**
- **Visual parity at `rdbg=0`** — every build step (§9) renders identically to pre-change.
- **Register captures** — `ProjectVerts`, `Classify`, `Mid`, `Splat`, `Hw`. The whole point; measure, don't assume (my estimates have been wrong all session).
- **Frame-time A/B** (`gpuWall`, ablated) — THE ship gate, *after* registers are good (user law: no frame-time work before then).
- **Append-contention check** — the splat/mid/hw queue counters are single `atomicAdd`s (R4).

**Now a PREREQUISITE, not optional:**
- **Overflow / high-water accounting** — §6's vert-buffer sizing needs the *measured* visible-vert peak (and the mid/splat counts). Wire the cull's pattern (`registerQueueHw` / `readBuffer` / `window.__qHW`, persistent `atomicMax`) for the queues + `projVertBuf`. This is the readback I kept deferring; the rewrite can't be sized without it.

**Keepers that migrate into the new files:**
- `?nospl` / `?nomid` / `?nohw` disable-flags → into `Classify` (disable a route). User-approved (they *disable*, don't *branch*).
- `swScanline` shared coverage core → `Scanline.ts` (already extracted this session).
- 2-D dispatch (`split2D` + `wgLinear`) → `Queues.ts` / `Mid.ts` (already built for mid; apply to splat if its count grows past 4.19M).
- `?rdbg` stage-return probes + `gputrace.sh NOROBUST=1` → diagnostics for measuring the rewrite's stages + the robustness floor.
- **≤1px CENTROID point** (`cz` = mean of the 3 projected-vert depths, edge-setup-free) → folds into `Classify`'s splat route. This was the unfinished "step 2b"; it's now *intrinsic* because `Classify` reads verts.

**Risks that carry (splat plan §6):**
- **R4** append contention — 3 queue counters now.
- **R7** pass ordering — MORE passes (project → classify → 3 consumers → HZB); all consumers must finish before the HZB build / resolve reads the visbuffer.
- **R9** three-way election (splat + mid + hw all `atomicMax` into the visbuffer) — order-independent, but all before resolve; depth24-tie speckle unchanged.

**Subsumed (no longer a separate task):**
- Old §8 Phase-2 "precompute the splat point / lean the projection" — this rewrite *is* that, generalized from the sub-pixel centroid to all verts.

---

## 11. Premise-audit standing note (CLAUDE.md)

A disappointing result (regs not ~30, memory not < today, frame flat) is the trigger to interrogate the
setup — is the vert buffer sized right? is Classify actually reading verts and not re-deriving? is the
projection genuinely gone from world1? — NOT to thrash. Measure the specific claim. We reached this plan
by refuting a chain of "one level up" premises (loop=peak, projection=peak, pre-pass-can't-help,
sub-pixel-is-97%); keep doing that.

---

## 12. STATUS LOG (2026-07-08) — rewrite essentially DONE

**Built via subagents, on branch `nanite-raster`. `tsc` green throughout.**

- **CP1 COMMITTED `31b46cc`** — `NaniteRaster.ts` 3283→2586 extracted into `src/nanite/raster/`
  (VisBuffer/Queues/ClusterCtx/Scanline/Splat/Mid/Hw/index). Behaviour-preserving; user-verified parity.
- **CP2 — Project + Classify (DONE):** `Project.ts` = per-vertex projection pass writing `projVertBuf`;
  world1's `rasterKernel('world1')` rewritten to READ projected verts (stops projecting). **world1 80→56
  regs (user Xcode capture)** — user: "good enough, proceed, no fallback"; 56→50 is a *later* polish.
  Projection extracted verbatim ⇒ bit-identical corners.
- **Queue shrink (DONE):** mid record 10-u32 → 1-u32 tri-id; `nanMidRaster` re-reads the 3 corners from
  `projVertBuf` via `projRecBase`. (`Splat` already stored a point — untouched.)
- **DEDUP (DONE, the hard part):** the blocker was the QEM leaf DAG — leaf clusters have ≤494 unique verts
  *scattered* across a 1.46M-vert range ⇒ no dedup key, so a flat buffer was pinned at stride 765 (2.4 GB,
  OOM). FIX = **`BuildDag.meshletizeDag`** (main-thread, idempotent, cache-safe, +0.5% verts): reorder each
  leaf cluster's verts contiguous + rebase indices local, so `canonLocal = vi − vBase`
  (`vBase = indices[triStart*3]`) — no vcompact/VCACHE bump. Terrain-DAG stays per-corner (≤384<512).
  Only new binding = `gpu.indices` on Classify+Mid (world1 at **8** storage bufs — the "10 ceiling"
  comment was stale). Stride **765→512**, `PROJ_CLUSTER_CAP=192K` (covers the real ~150K visible-cluster
  peak — 128K was silently dropping ~22K/frame). **`projVertBuf = 1.13 GB`** (saved ~600 MB vs undeduped).
- **BLINK FIXED:** `MID_CAP` 4.19M→40M (cheap now at 1-u32/record, ~160 MB). User-confirmed blink gone.
- **Memory end state ≈ 1.4 GB** (projVertBuf 1.13 GB + midQueue 160 MB + splat/hw) — ~neutral vs today's
  ~1.33 GB non-blinking, but viable (no OOM, no blink) + carries the register win. Compacted §6
  (variable-stride) → ~700 MB is the deferred further win.

**Corrections banked (were wrong):** visclusters max ~150K not 477K; sub-pixel is a couple % near-forest
(not 97% — that's aerial) so MID is the bulk; single-buffer OOM ceiling ~1–2 GB on the user's box.

**Register re-capture PENDING:** dedup added +1 binding + 2 index-reads/corner to Classify — confirm 56
held before committing the rewrite.

**IN PROGRESS:** clhw=1 as the permanent default + delete every `clhw=0` path (the user's "last task" —
it's a CULL change: the cull only builds the HW-partition buffers under `?clhw=1`; keep `?clhwmax`).

**AFTER:** frame-time A/B (a datapoint, NOT a go/no-go gate — user law: architecture is decided);
56→50 register polish; compacted §6 memory; shadow/depth could get the same treatment if a cost.

**Superseded:** `splat-election-register-cut.md` (splat = one consumer now) + the whole per-`?splat`/de-über
saga — the rewrite absorbed all of it.

---

## 13. POST-BUILD AUDIT (2026-07-08) — done / deviated / missing / deferred

**`tsc` exit 0. Rewrite + clhw committed. Frame profile: `nanProjectVerts` 30% (the whale, looks
mem-bound), `nanMidRaster` 14.5%.**

**DONE:** the pipeline (Project→Classify→Splat/Mid/Hw→Resolve); meshletize (`BuildDag.meshletizeDag`) +
storage dedup (`vi−vBase`, stride 765→512, projVertBuf 1.13 GB @192K cap); mid record 10→1 u32; blink fix
(MID_CAP 40M); **clhw=1 the permanent default + `?clhw` flag deleted** (cull always builds+dispatches the
HW partition; `?clhwmax` kept); `?nospl/?nomid/?nohw` + `?rdbg` + `NOROBUST` preserved; batch order R7
(`ProjectVerts` before Classify, splat/mid before the HZB tail); world1 **56 regs** (user capture).

**DEVIATED (accepted):**
- **`Classify.ts` never extracted** — the classifier stayed inside `NaniteRaster.ts`'s `rasterKernel`
  (the subagent avoided parity risk on a kernel shared with depth/combined). §5's layout is otherwise met.
  → either finish the extraction (cleanliness) or amend §5. OPEN.
- **≤1px stayed EXACT** (reads corners, does rw/bias coverage) instead of the planned centroid — *better*
  (parity-exact; centroid wasn't needed to reach 56). §10's "centroid intrinsic" is moot.
- **clhw isn't literally a bare `true`** — one presence-check boolean survives in the raster
  (`!!cull.qHwRasterRO && !!cull.hwClusterDrawAttr`) because the **shadow-clip raster** (`NaniteShadowClip`,
  ortho, no camPos, different queue) legitimately has no HW partition. Byte-identical to `?clhw=1` for the
  camera path; not a `?param` branch.

**MISSING (small):** the §10 **high-water readback** was never wired — the vert peak was measured one-off
(instrumentation reverted). The `minU(canonLocal, 511)` clamp is the only backstop if a frame's visclusters
exceed the 192K `PROJ_CLUSTER_CAP` (→ dropped clusters). Wire a persistent readback OR raise the cap with
margin. OPEN.

**DEFERRED (→ 2026-07-08 folder, not lost):** per-unique-vert projection dispatch (halve the 30% whale —
projection is still per-CORNER, dedup was storage-only); compacted §6 (don't project cull-doomed verts);
frame A/B; append-contention; 56→50; per-kernel register captures (ProjectVerts/Mid/Splat).

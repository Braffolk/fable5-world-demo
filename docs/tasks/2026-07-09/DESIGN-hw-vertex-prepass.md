# DESIGN — kill the per-vertex re-decode in `vertex_nanRaster_world1_cl` (HW clhw draw)

Date: 2026-07-09 · Branch `nanite-raster` · Read-only design (no source edits, no commit).
Target profile: `docs/tasks/2026-07-09/PROFILE-POST-ARC.md`
(pose `?cam=-582.1,302.4,1006.1,2.5692,-0.0077`, world, grass=0, nanshadow=0, dpr2).

---

## 0. Premise-audit (fired BEFORE designing — CLAUDE standing rule)

Before treating "make the vertex shader cheaper" as the task, I put the context one level up on trial:

- **Is the METRIC right?** Yes. Fresh 2026-07-09 Xcode export, 2 s at the census worst pose.
  `vertex_nanRaster_world1_cl` = 15.66 % of sampled GPU, 96 regs, **80-byte spill**, 868 instr,
  **Sync-Wait-Memory 45.14 %**, Mem-Load 15.60 %. The #1 shader. Not a bucketing artifact — it is a
  single named vertex stage.

- **Is the STRUCTURE one level up the flaw?** Two upstream facts matter:
  1. **The HW `_cl` draw always dispatches `MAX_CLUSTER_TRIS*3 = 384` vertex invocations per
     instance** (`NaniteCull.ts:798` sets `vertexCount/instance = MAX_CLUSTER_TRIS*3`), regardless of
     the cluster's real tri count. `makeCtx` + `fetchWorld` run for **all 384** and the partial-tail
     collapse (`localTri < triCount → clip=(0,0,2,1)`, `Hw.ts:221`) happens *after*. So a 10-tri
     cluster still pays 384 full re-decodes. → magnifies the disease on partial clusters.
  2. **Why is 6.66 M tris on HW at all?** `clusterHwClass` (`NaniteHwClass.ts:60`) routes a whole
     cluster HW when `projTri = projK·(radius/√triCount)/nearestDepth > swmaxCl` (?clhwmax, 32 in the
     test URL). SW world1 is only **2.8 %** of frame for its tri load; HW is **15.66 %**. The gap is
     NOT that HW carries more tris — it is that HW pays the per-vertex re-decode SW already cured
     (ClusterCtx/Project pre-passes). So the honest fix is **make the HW path pay the SAME cured
     cost**, not re-route. Once HW is cured, the routing threshold could even be *relaxed* (more
     SW→HW) for better big-tri raster quality. Routing right-sizing is a real but *complementary*
     lever (Option E), not the primary one.

- **Can the generating context be changed cheaply?** Yes — and it already was, for the twin path.
  The SW compute raster had this exact disease pre-rewrite and it was dissolved by two pre-passes
  that **already run every frame before the HW draw** and **already contain the HW clusters' data**.
  The lever is to let the vertex shader *read* what the pre-passes already computed. This is a
  context change (feed the shader decoded data), not a method tweak inside the shader.

Premise cleared: metric sound, structure-up-a-level IS the flaw, and it is changeable at low cost.
Proceed.

---

## 1. The verified disease (file:line)

The `_cl` instanced material is `buildHwMaterial('world1', instanced=true)` in
`src/nanite/raster/Hw.ts:147-306`. Its `vertexNode` (Hw.ts:161-245), per **every one of the ~20–26 M
vertex invocations** (see §2 arithmetic), does:

```
tid   = qHwRaster[instanceIndex]           // Hw.ts:187   1 load
item  = qRaster[tid+1] → instId, ci        // Hw.ts:191   1 load (2 words)
ctx   = makeCtx(instId, ci)                // Hw.ts:196   ← THE WHALE
world = fetchWorld(ctx, localTri)          // Hw.ts:196   fetch + transform + wind + disp
clip  = cam.vp · (world,1)                 // Hw.ts:197
{ D-seam snap to 1/256 px }                // Hw.ts:205-218
{ partial-tail collapse }                  // Hw.ts:221-227
```

`makeCtx` (`NaniteFetch.ts:260-432`) re-derives, **per vertex**:
- **Metadata decode** — `gpu.clusters` (2 loads, `NaniteFetch.ts:262-263`), `gpu.instances` A+B (2
  loads, `269-270`), `gpu.meshes` w6/matParam/quadsX/oX/oZ/cell (~6 loads, `272-284`). Shifts/masks
  for isHF/isDAG/meshId/channel/twoSided/gx/gz/qxw.
- **Trunk/leaf wind PRECOMPUTE** (`NaniteFetch.ts:293-411`) — gated on channel, runs
  `windExposure` + `gustAt` + `gustLagAt` (**3–4 gust TEXTURE samples**) + ~15 live float temps
  (`e,g,gL,dist,eks,leanBase,swayABase,natW,ph,branchBase,farAtten,brAtten,instPhase,fJit,posKey`) +
  2 hoisted `sin()` (`398-399`).

`fetchWorld` (`fetchWorldVert`/`fetchWorldVertDyn`, variant `'both'` — the default `nfetch` at
`NaniteRaster.ts:591`, passed to `buildHw` at `2300`) then compiles the **branch UNION** of both
arms (`NaniteFetch.ts:664-670`):
- explicit-mesh arm → `gpu.indices` (1) + `gpu.verts` pos+vdata (~6 loads) + `instTransformPoint` +
  per-vertex `windOffset`.
- heightfield arm → `gpu.hfVerts`/`gpu.verts` (1) + `heightTex` (1 sample) + **`terrainDispAt`
  (6 texture samples + fbm mixing, `NaniteFetch.ts:99-141`)**.

**Which loads plausibly cause the 45 % Sync-Wait-Memory:**
- The instruction mix is 45 Mem-Load instr (15.6 %) + 11 Sample instr (0.88 %) but **45 % of GPU is
  spent *waiting* on those ~56 memory ops.** Samples are only 11 instr, so texture latency is NOT
  the whale — the **scattered strided buffer loads** (`clusters` stride-8, `instances` stride-2×instId,
  `meshes` stride-`MESH_WORDS`, `verts` stride-`VERT_WORDS`) are, plus **the 80-byte register spill**:
  96 live regs force stack spill/reload traffic that ALSO shows up as Mem-Load + Sync-Wait. The
  wind-precompute + fetch-union temps are what push register count to 96.

**Diagnosis = the exact pre-rewrite SW disease, in a vertex stage with no workgroup-shared memory.**
Each vertex invocation re-derives cluster-uniform data. A compute kernel fixed this with wgcache +
`ClusterCtx`/`Project` pre-passes; a vertex shader cannot share, so the cure is to **read** pre-passed
data instead of re-deriving it.

### Key enabling facts (verified)

1. **`ClusterCtx` already decodes HW clusters.** `nanClusterCtxPrepass`
   (`ClusterCtx.ts:72-149`) iterates **every** visible qRaster cluster (`tid < count`), writes the
   35-word ctx, and even writes **slot 11 = `clusterHwClass`** (`114-119`). It does **NOT** skip HW
   clusters — the SW world1 raster reads slot 11 and skips them, but the record exists. So
   `clusterCtxV[tid·35 …]` is a fully-decoded ctx for HW clusters, keyed by `tid` — exactly what the
   `_cl` draw carries (`tid = qHwRaster[instanceIndex]`, Hw.ts:187). **Wind scalars are already in it**
   (slots 11–20, `ClusterCtx.ts:131-143`) — the gust texture samples ran ONCE per cluster in the
   pre-pass.

2. **`Project` already RESERVES HW cluster slots but skips filling them.** `projVertBuf` is indexed
   by `itemIdx` (= `tid`) up to `PROJ_CLUSTER_CAP = 128Ki` (`Project.ts:144`), covering ALL qRaster
   clusters including HW. But `nanProjectVerts` bails on HW clusters at `Project.ts:319`
   (`returnIf(rU(11).equal(uint(1)))`), leaving those reserved slots **unwritten / wasted**. The
   dispatch (`rasterDispatchFull`, one workgroup/cluster) already launches a workgroup for them.

3. **Render order is safe** (`NaniteRaster.ts:2539-2587`): inside `world1()`, `kClusterCtx` (2555)
   and `kProjectVerts` (2560) dispatch, THEN the soup `hwRender` (2582), THEN `hwRenderCluster`
   (2587). Both source buffers are fully populated for the frame before the `_cl` draw reads them.

4. **The SW pipeline interpolates depth SCREEN-LINEARLY** (`Scanline.ts:197-205`:
   `cz = (uw0·dz0 + uw1·dz1 + uw2·dz2)·rcpArea`, barycentric on the fixed-point *screen* edge
   functions — NOT perspective-correct). The current HW `_cl` path emits a REAL perspective clip
   (`clip.w` = true w) ⇒ its depth interp is perspective-correct ⇒ **already inconsistent with SW**.
   This fact is the crux of Option D (below).

---

## 2. Sizing arithmetic (measured peaks)

Constants (`GeometryRegistry.ts:108,124`, `ClusterCtx.ts:35`, `Project.ts:90,144`):
`MAX_CLUSTER_TRIS = 128` ⇒ 384 verts/instance · `CTX_STRIDE = 35` · `MAX_CLUSTER_VERTS = 512` ·
`PROJ_VERT_STRIDE = 3` · `PROJ_CLUSTER_CAP = 128Ki = 131072` · `QRASTER_CAP(world) = 1 048 576`.

Census (`census-post-ladder.json`, worst pose): `clhwTris = 6.66 M`, `visClusters = 66 827`.

- **HW cluster count** = 6.66 M tris ÷ 128 tris/cluster ≈ **52 031** (full-cluster lower bound;
  upper bound = visClusters cap 66 827; partial clusters push toward the upper end).
- **Vertex invocations** = HW-clusters × 384 = 52 031 × 384 ≈ **19.98 M** … 66 827 × 384 ≈ **25.66 M**.
  Each currently runs the full `makeCtx`+`fetchWorld`. **This ~20–26 M-wide re-decode is the 15.66 %.**
- Partial-cluster waste: verts beyond `triCount·3` collapse to a clipped point, but the collapse is
  *after* `makeCtx`+`fetchWorld` (Hw.ts:196 → 221), so their decode is pure waste.

Existing buffers (both already allocated, reused at 0 MB by A/D):
- `clusterCtxV` = 1 048 576 × 35 × 4 B = **140 MB** (has HW records).
- `projVertBuf` = 131 072 × 512 × 3 × 4 B = **805 MB** (HW slots reserved, currently unwritten).

Frame framing: 15.66 % ≈ **~1.6 ms** at the worst pose (per the profile coordinator's read).

---

## 3. Option table

| Opt | What's precomputed / read | New VRAM | Removes from vertex shader | Est. regs / spill | Est. shader share | Added pre-pass cost | Parity risk |
|----|----|--:|----|----|--:|----|----|
| **A** — read existing `clusterCtx` | 35-word ctx (incl. wind scalars) via `clusterCtxV[tid·35]` | **0 MB** (exists) | `makeCtx` entirely: metadata loads + wind precompute + 3–4 gust samples + ~15 wind temps | 96→**~72–80**, spill **shrinks/gone** | 15.66→**~9–11 %** | 0 (ctx already runs; `fetchWorld` keeps REAL clip) | **None** (byte-identical world + real clip) |
| **A2** — A + class-split the `_cl` draw | + route HW clusters into two instanced draws (terrain / mesh) each with a specialized `nfetch` variant (`'terrain'`/`'explicit'`) | 0 MB (2× 4-word draw-arg + a partition bit) | A, plus the fetch branch-UNION: terrain draw sheds wind, mesh draw sheds `terrainDispAt`'s 6 samples+fbm | **~55–65**, spill **gone** | **~7–9 %** | tiny (kHwPartition writes 2 counters) | None |
| **B** — full pre-transformed WORLD buffer | one compute pre-pass writes world xyz/unique-vert; shader = `vp·world`+snap | **320 MB** compacted (52 031×512×12 B) / 805 MB uncompacted — **OVER 150 MB budget** | `makeCtx`+`fetchWorld` (keeps `vp·` transform + snap) | ~45–55 | ~6–8 % | a whole new fetch pre-pass + (for compaction) a prefix-sum | Low (real clip kept) — but **rejected on memory** |
| **D** — read existing `projVertBuf` (fill the skipped HW slots) | snapped `xi/yi/dz` per corner via `canonVertSlot`; shader reconstructs clip **w=1** | **0 MB** (slots already reserved) | `makeCtx`+`fetchWorld`+`vp·`transform+snap ENTIRELY | **~30–40**, **no spill** | **~4–6 %** | `nanProjectVerts` also projects HW clusters (remove the `:319` skip) — **moves** work off the whale into the cured compute pre-pass; ~+0.2 ms there | **Real, gated** — see §4 |
| **E** — upstream: right-size HW routing | (complementary) tune `clusterHwClass` so fewer/only-genuinely-big clusters ride HW | 0 MB | — (fewer invocations, not cheaper ones) | — | reduces the 20–26 M count | none | perf-only by design (`NaniteHwClass.ts:8-12`) |

Estimated ms recovered (of ~1.6 ms):
A ≈ 0.6–0.8 · A2 ≈ 0.8–1.0 · D ≈ 1.0–1.2 gross (−~0.2 ms added projection ⇒ ~0.8–1.0 net) · B ≈ dominated by D.

Rank by **(ms recovered)/(complexity + memory)**:
1. **A** — best ratio: substantial relief, ~trivial plumbing, 0 MB, **0 parity risk**.
2. **A2** — incremental relief over A, low complexity, 0 MB, 0 parity risk. Natural bundle with A.
3. **D** — highest raw relief, 0 MB, but **two hard gates** (§4) ⇒ complexity dominates ⇒ Phase 2.
4. **B** — dominated by D (D reuses the existing buffer; B needs 320 MB+). **Reject.**
5. **E** — orthogonal; pairs with any of the above (the QUEUED tri-census/cap-audit task).

Note: Option **C** from the brief ("A + per-INSTANCE wind precompute") is **subsumed by A** — the
wind gust samples are ALREADY precomputed per-cluster into `clusterCtx` slots 11–20
(`ClusterCtx.ts:131-143`), so reading ctx IS the per-instance wind precompute. No separate option.

---

## 4. Why D is Phase 2, not Phase 1 (two gates)

D deletes the whole per-vertex geometry+transform by reading `projVertBuf`'s snapped `xi/yi/dz` and
emitting `clip = (ndc_from_xi, ndc_from_yi, dz, 1)`. Two problems that A/A2 do not have:

- **Gate 1 — screen-linear depth on BIG triangles.** With `clip.w = 1` the HW rasterizer interpolates
  `dz` (= ndc.z) linearly in screen space. That MATCHES the SW convention (§1 fact 4, `Scanline.ts:200`)
  — it is actually *more* consistent with SW than today's perspective-w HW path. **But** the HW `_cl`
  path carries the *big / near* triangles (trunks, near-terrain) precisely because they are large;
  screen-linear ndc.z bows across a large near triangle vs true perspective depth. For the depth-key
  election this only matters where big HW tris self-overlap or fight near-terrain. This has never been
  screen-linear-rasterized before (big tris never went SW). ⇒ **must be eyeball-verified** (trunks /
  near-terrain occlusion + z-fight at the worst pose) before shipping.

- **Gate 2 — near-plane-crossing `_cl` clusters can't be pre-projected.** `clusterHwClass` routes a
  near-crossing cluster HW because `nearestDepth→0 ⇒ projTri→∞` (`NaniteHwClass.ts:14-16,52-57`) — so
  the `_cl` draw DOES contain near-crossing clusters. `nanProjectVerts` stamps such corners with
  `NEAR_SENTINEL` and discards their `xi/yi` (`Project.ts:128-130,380-385`). A near-crossing corner
  therefore has NO valid pre-projected position. If the `_cl` vertex shader must keep a real-clip
  fallback for `dz == SENTINEL`, the full `makeCtx`+`fetchWorld` path **stays compiled** ⇒ the
  register/spill win evaporates (occupancy is set by the worst compiled path). To realize D's register
  win, near-crossing `_cl` clusters must be handled *elsewhere* (e.g. a per-cluster near-cross bit in
  `clusterHwClass` routing them to the per-tri soup draw, which already exists and is only 0.99 %), so
  the `_cl` shader compiles the pre-projected path ONLY. That split is the bulk of D's complexity.

Neither gate touches A/A2 (they keep `fetchWorld` + real perspective clip ⇒ byte-identical geometry
and depth to today).

---

## 5. Recommendation + phased sketch

**Phase 1 — ship A (bundled with A2).** Guaranteed structural win, 0 MB, 0 parity risk, mirrors the
SW cure. Then re-profile (fresh Xcode export per the standing rule) before deciding on Phase 2.

- **A implementation:**
  1. Plumb `clusterCtxV` (already in `buildNaniteRaster` scope, `NaniteRaster.ts:635`) into `buildHw`
     (`raster/Hw.ts` param list, call site `NaniteRaster.ts:2296-2313`). Non-null only on the world1
     `singlePass` build, which is the only build that renders the `_cl` world1 material — the depth/
     combined `_cl` materials (if built) keep the current path (guard on `clusterCtxV != null`).
  2. In `buildHwMaterial(..., instanced=true)`, `pass==='world1'`, replace `makeCtx(instId, ci)`
     (Hw.ts:196) with a flat read of `clusterCtxV[tid·CTX_STRIDE … +35]` decoded into a `VertCtx`
     exactly as `Project.ts:307-357` and `NaniteRaster.ts:833-843` already do (same 35-word layout,
     `bcU2F` for the f32 slots). `item.x/instId` is then unused (ctx already carries A/B); keep `tid`
     for the payload and `ci` is not needed.
  3. `fetchWorld(ctx, localTri)` is unchanged (Hw.ts:173-181 / `hw1fetch`), so world reconstruction &
     the real perspective clip & the D-seam snap stay byte-identical. Only `makeCtx` is deleted.
  - **Bit-identity check:** the decoded ctx equals `makeCtx`'s output by construction (the pre-pass
    ran `makeCtx` and stored its fields) — same guarantee `Project`/world1 already rely on.

- **A2 implementation (bundle):** in `kHwPartition` (`NaniteCull.ts:753-801`) append HW clusters into
  **two** lists partitioned by ctx slot-0 `isHF` (terrain vs mesh) with two draw-arg sets; build two
  `_cl` instanced draws, one with `makeFetch(..., 'terrain')` and one with `makeFetch(..., 'explicit')`
  (the variant machinery already exists, `NaniteFetch.ts:244`, used by ksplit at `NaniteRaster.ts:744`).
  Each draw compiles only its arm ⇒ sheds the other's registers. Reuses A's ctx read.

**Phase 2 — evaluate D (only if A+A2 leaves the shader on top).** Requires: (a) remove the
`Project.ts:319` HW skip so `nanProjectVerts` fills HW slots; (b) route near-crossing `_cl` clusters to
the soup draw (add a near-cross bit to `clusterHwClass` / kHwPartition) so the `_cl` shader can compile
the pre-projected-only path; (c) the `_cl` world1 vertex shader reads `xi/yi/dz` via `canonVertSlot`
(`Project.ts:107-127`, same as `NaniteRaster.ts:1297-1317`) and emits `clip.w = 1`; (d) **verify
Gate 1** by eyeball at the worst pose. 0 MB.

**E (any time, complementary):** the QUEUED tri-count census / cap-audit — confirm 6.66 M HW tris is
legit vs `clusterHwClass` over-election; once the HW path is cured, the routing threshold can be
re-tuned for quality rather than perf.

---

## 6. Open questions for the user (surface, don't silently decide)

1. **Phase 2 depth convention (Gate 1).** Are you comfortable putting big HW trunk/near-terrain
   triangles on **screen-linear** ndc.z depth (consistent with the whole SW pipeline, but new for big
   tris)? This is the one user-taste/quality call that gates D. If not, D is off the table and A+A2 is
   the ceiling of this arc (still ~40–50 % of the whale) unless we spend +268 MB to store `clip.w`
   (over the 150 MB budget).
2. **near-cross routing (Gate 2).** OK to route near-plane-crossing `_cl` clusters to the existing
   per-tri soup draw (tiny, 0.99 %) so the `_cl` shader can shed the real-clip fallback? Small
   correctness surface (soup already handles near-crossers), but it moves a class of geometry between
   paths.
3. **A2 draw-count.** Splitting `_cl` into terrain+mesh doubles the instanced draw calls (2 vs 1).
   Acceptable, or prefer A-only to keep the single draw and a smaller register win?
4. **E scope.** Do you want the HW-routing census run in this arc, or keep it as the separate QUEUED
   task? (It could change how much of the 6.66 M even needs HW, shifting the whole target.)

---

## 7. What I did NOT verify (honest gaps)

- Exact HW-cluster count / partial-cluster fill ratio (used the 6.66 M ÷ 128 bound). A one-frame
  readback of `hwClusterDraw[1]` (instanceCount) would pin it.
- The register/spill estimates are structural reasoning (which temps die), not a re-compile — the
  Phase-1 re-profile is the witness, per the standing rule.
- Whether any non-`singlePass` build actually renders a `_cl` world1 material (I believe `_cl` world1
  is world1-only; the `clusterCtxV != null` guard makes A safe either way).

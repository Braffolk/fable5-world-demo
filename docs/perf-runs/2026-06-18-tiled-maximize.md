# Tiled raster — maximize + 60fps-p0.05 forest research (2026-06-18)

5-stage dynamic workflow (research → analyze → ideate → synthesize → adversarial-review; 18 agents,
~1.9M tokens). Full result: the task output for run `wf_b5b2efce-f9d`. This is the durable digest +
the implementation tracker.

## The two confounds (RESOLVED — the user's "param-dependent bindings that exclude one another")
1. **vcompact slowed the tiled (20→25.6 ms setup)** — NOT "tiling is slower". The cluster-tri cap is
   **256** (`setClusterTriCap`, WorldRegistry.ts:289) ⇒ MAX_CLUSTER_TRIS=255, and kSetup is
   `.compute(…, [MAX_CLUSTER_TRIS])` = **255-thread** workgroups. `vcache.prime` (NaniteVertexCache.ts:
   63-80) does an **unconditional `workgroupBarrier`** over a ~6.1 KB shared array that all 255 lanes
   hit per cluster. vcompact's 1.8× was measured at the **128** cap; at 255 threads the barrier/occupancy
   tax roughly doubles and dwarfs the ~4.7× vertex-reuse → slower. Off-spec workgroup width, not tiling.
2. **world1 measurement broke (29/30 cap-suspect)** — kSetup is packed to **exactly 10 storage buffers**
   (NaniteTileRaster.ts:240, deliberate); `?vcompact=1` adds `gpu.vcompact` as the **11th** → driver
   spill/rebind → unstable timestamp resolve → mass rejection. **HW-queue routing and vcompact are
   mutually-exclusive-by-budget.**

⇒ The "tiled 43 vs world1 33" was measured with **vcompact OFF on both sides** (so the vcompact confound
doesn't apply to it), but the prior "structural / 2.8× slower" verdict is REJECTED — the optimization
levers below were never tried. **De-confounded baseline (occl=0 dense canopy, 256 cap, vcompact off):
tiled ~44 ms (setup ~21, raster ~13.6) vs world1 ~33 ms** — tiling ~1.34× slower as-is, setup-dominated.

## Cost map (forest p0.05)
SW raster dominant (~60-75%), scales ~linearly in visClusters (~50-62 ns/cluster). Tiled setup splits
into a **FUNDAMENTAL** half (the transform, shared with world1 + the sort-middle xtri store/reread that
scatter avoids) and a **FIXABLE** half (per-fragment global election round-trip, the K-pass re-walk, the
static 20-wave/all-tiles dispatch). makeCtx wind taps are ALREADY deduped once-per-cluster (the briefs'
"48 taps/tri" is stale). Cluster COUNT (over-emission, HZB ~0% on holey foliage) is the linear-cost root.

## Ranked portfolio (adversarially verified: 6/8 top hold; quality skeptic + feasibility skeptic + perf skeptic)
| # | win | target | est | conf | status |
|---|---|---|---|---|---|
| 1 | Clean re-measure (256 cap, vcompact off both; vcompact=1 runs excluded) | gate | 0 | high | ✅ done (44 vs 33) |
| 2 | **On-chip per-tile election** (`var<workgroup> atomic<u32>`, atomicMax-MERGE flush) | tiled-raster | −1…2.5 ms | med (prototype) | ✅ done: raster 13.6→**12.3** (−1.3ms), render clean, cut tracks baseline (commit) |
| 3 | **Count-sort the K-pass → one ordered pass** (8-bin histogram+scan+scatter) | tiled-raster | −1…2.5 ms | med | 🔄 next |
| 5 | **Indirect dispatch off live cut** (skip ~8 idle waves; arg already exists) | tiled-setup | −0.7…2 ms | high | ⬜ |
| 6 | Dedup HW vertex (fetch only selected corner) | hw | −0.3…0.8 ms | high | ⬜ |
| 4ʹ | HW **self**-Z only (NO SW pre-seed) | hw | −0.3…1 ms | med | ⬜ |
| 7b | Pack `gpu.vcompact` into a dataBuf base-offset (free the 11th slot) | bindings | 0 (unblock) | high | ⬜ |
| 8 | vcompact cost-aware GATE (uniform barrier only for high-cost clusters), re-measure @128 | tiled-setup | −1…2 ms @128 | low | ⬜ |
| 10 | **Far-field crown impostors** as the election word — cuts cluster COUNT (linear-cost root) | cull | −3…6 ms | low | ⬜ big bet |

**KILLED:** full HW early-Z with SW pre-seed (depth reject unrecoverable by atomicMax → popping/holes;
quantized-vs-float mismatch — holds ONLY as HW-internal self-Z); #7a "drop gpu.verts" (factual error,
breaks vcompact — use 7b); resolve zero-discard (#9, already shipped as the `If(isT)` gate).

**Tiled verdict:** maximizable — NOT proven to beat scatter yet, but the dismissal was invalid + the
experiment (on-chip election + count-sort + indirect dispatch + impostors) was never run. The transform/
xtri-store half is the irreducible sort-middle floor; whether the fixable half + cluster-count cuts flip
the gap is the open question the implementation will answer.

## Implementation order (workflows, each GPU-measured after)
#2 on-chip election → #3 count-sort → #5 indirect dispatch → #6/#4ʹ HW → #7b/#8 vcompact → #10 impostors.
Measure at the production **256 cap** (128 doubles cluster count → blows the 640k cut cap). Honest
methodology: GPU-bound `measureActiveGpu` (not vsync-capped live gpuPasses); occl=0 deterministic A/B.

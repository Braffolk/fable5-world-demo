# Forest raster — diagnosis run (2026-06-16)

Run of the `perf-review` skeleton on the forest debug-raster perf problem. (Run id `ww8rwwimi`;
19 agents, ~2M tokens.) This is the **run output** — the generic workflow lives in
`.claude/workflows/perf-review.js`; the standing project/task brief is in `docs/NANITE-PERF-WORKFLOW.md` §B.

## Integrity of this run
- **Guardrail held:** of 19 agents, exactly 1 (the harvester) ran GPU probes; all analysts/ideators/
  rankers/synth/critic ran zero (`PARALLEL_RULE` worked — fixed the prior 5-way concurrent-probe pile-up).
- **Caveat:** the adversarial **Confirm** agent died on an API "Overloaded" error, so the cost map is
  harvester-corpus + analyst-derived, **not independently re-confirmed**. The red-team critic backfilled
  and flagged the holes a confirmer would have. Treat shares as **preliminary**.

## ⚠ CORRECTION (post measure-first on lever #1) — the 65/35 split was WRONG
The measure-first gate for #1 (ported `rdbg` into the world1 kernel, `src/nanite/NaniteRaster.ts`)
overturned the cost map AND exposed a metric artifact:
- **Real split ≈ ~90% per-pixel coverage / atomic-election + ≤~10% fixed-per-cluster** (NOT 65/35).
  Proof: at 2× resolution full world1 = 69.6 ms (GPU-bound), but the everything-but-the-per-pixel-loop
  variant is still sub-vsync (<8.3 ms) ⇒ per-pixel ≥ ~88–90%.
- **#1 is DEAD — and #3 (vcompact) with it.** The launch/dispatch-shape slice #1 could reclaim is
  **<~1.2 ms (<7%)**, below the gate. Idle lanes (`localTri ≥ triCount`) fall straight through the
  `If(localTri < triCount)` gate without running makeCtx / vertex-fetch / the per-pixel loop, so they
  cost ~nothing. The original "~4.3 ms fixed term" was a conflation of the confounded resolution sweep
  + the timestamp artifact below.
- **METHODOLOGY BOMBSHELL — per-pass GPU timestamps LIE when sub-vsync.** `c.nanRasterWorld1` reads a
  BOGUS ~constant (~15.2 ms) whenever the frame sits at the 120 fps rAF cap (frameMs ≈ 8.3) — it's a
  cross-frame pipelining span, not active GPU time. Proven: the "floor" is pinned at ~15.2 ms across
  242k → 3.3M clusters and at 2× resolution (slope ~0; real work would scale). **Only GPU-bound readings
  (frameMs ≫ 8.3) are real.** To measure honestly, force GPU-bound: 2× internal resolution / denser scene
  / worst pose. Deeper version of the "frameMs is vsync-capped" lesson — the per-pass GPU timestamp is
  ALSO cap-contaminated. → SHOULD propagate to NANITE-SPEC `## PERF METHODOLOGY`.
- **Durable artifact:** the `rdbg` stage-split now lives in world1 (`?rdbg=1|2|3|4`, default 0 =
  byte-identical), `src/nanite/NaniteRaster.ts` — the trustworthy world1 attribution tool going forward.
- **Implication:** bit-identical big wins are exhausted. The ~90% per-pixel half is the real target
  (genuine open problem), reachable bit-identically only via OVERDRAW reduction (#5 same-frame cluster
  Hi-Z), or non-bit-identically via the quality-gated count cut (#4 banded-τ — measured, needs judge-shots).
- **Kernel-scope note:** this measured `world1` (full forest pipe, `c.nanRasterWorld1`). The lean debug
  view `?nanitedbg=flat` runs a *different* kernel (`combined` / `c.nanRasterCombined`); they share the
  per-triangle/per-pixel core so the ~90%-per-pixel conclusion transfers, but confirm WHICH kernel is
  "the debug render" target before optimizing.

## What's actually hitting perf (measured — SUPERSEDED by the correction above for the split)
- **GPU-bound** (cpuSubmit 0.6–1.2 ms vs compute 4.6–32 ms). CPU-submit ruled out.
- **Tree count is a NON-AXIS:** 200k ≈ 40k (raster 13.6 vs 12.2 ms) — the visible-cluster set is
  **screen-bounded by the LOD-cut + resolution, not tree-bounded**. ⇒ *"make 200k fast" = "make the
  on-screen cluster+pixel budget fast."*
- **Far-field cross-instance MERGE confirmed useless** at `instminpx=128` (far already impostored).
- **Cost split @ 1280×720 (PRELIMINARY, see confound): ~65% per-pixel coverage / atomic depth-election
  (~7.9 ms) + ~35% fixed-per-cluster overhead (~4.3 ms).**
- **Ruled out by ablation:** makeCtx / vertex-fetch (wgcache=0 forcing 128× recompute = only +5.4%),
  atomic contention (depth≈payload election), CPU-submit.
- **Two honesty caveats (critic):**
  1. The fixed-cluster resolution sweep that produced 65/35 is **confounded** — it varied *both* loderr
     and instminpx to hold cluster-count constant, changing *which* clusters draw. Qualitative
     "per-pixel is the majority" likely holds; exact split needs a clean re-measure.
  2. The 4.3 ms fixed term is **ambiguous**: killable launch+barrier vs irreducible per-tri edge-setup —
     this gates the #1 lever's magnitude.

## Idea portfolio (ranked)
1. **Triangle-granular dispatch re-shaping** — prefix-sum `qRaster.triCount` → flat ~`visTris` grid;
   each 128-lane workgroup drains *real* triangles from many tiny clusters, killing the ~117/128 idle
   lanes on ~11-tri leaf clusters. **Bit-identical, zero quality risk.** Attacks the 35% fixed half.
   Magnitude gated by the launch-vs-edge-setup split (see #2). *(known-but-ignored; UE5 Nanite does this.)*
2. **MEASURE FIRST — rebuild `rdbg` stage-split ON world1** (existing one is gated to the deleted depth
   kernel). Resolves launch-vs-setup; sizes #1 (could kill it). Zero-risk debug gate.
3. **`vcompact` vertex cache ON, stacked on #1** — the dismissed win re-justified by restored occupancy.
   Bit-identical. Standalone marginal; multiplicative with #1.
4. **Banded-τ / cluster-floor count cut** — the *fastest big win* (τ=2: 336k→166k clusters, −33%
   12.3→8.3 ms; τ=4: 3×) but **QUALITY-AFFECTING** → mandatory user judge-shots. User: willing to
   consider, but *after* #1.
5. Two-level same-frame cluster Hi-Z occlusion (forest-interior; zero-risk if conservative).
6. Shrink the HW thin-needle path (secondary; i32-edge correctness risk).
7–8. **KILLED** (tile-bin/coherence rewrite; early-Z/splat) — attack non-bottlenecks / unconfirmed terms.

## The Amdahl ceiling (the hard truth)
The only **bit-identical** levers (#1, #3) attack *only* the ~35% fixed half (~4.3 ms of 12.2 ms). So
**>60 fps during motion at 1280×720 with zero quality loss is likely unreachable by bit-identical levers
alone** — it needs a quality-affecting count cut (#4) and/or cracking the per-pixel coverage half.

## Biggest gaps (critic) — to close in the next measurement run
- **MOTION never measured.** The literal target ("avg fps > 60 *during movement*") was never measured —
  every number is a static settled pose. Need a fixed-replay motion glide sampling `c.nanRasterWorld1`.
- **Single pose.** Only canonical spawn; worst-pose (alley / horizon-grazing = max overdraw) never run
  (`probe-worstpos.ts` is in the working tree).
- **Overdraw not isolated.** world1 does up to 2 atomics/pixel; the per-pixel bucket lumps coverage +
  election. Need a cliff-free overdraw counter (via the audit pass, **not** a 3rd atomic).
- **HW-resolve coupling** (`renderMs` ~6 ms at canonical) under-quantified; simband shuttles work
  SW↔HW, so the joint objective is `world1 + renderMs`.

## Decision pending (user)
Given the Amdahl ceiling: **banded-τ LOD-band coarsening (#4) is on the table, pending judge-shots** —
but **#1 first** (bit-identical). Per-pixel-half attack (the larger 65%) remains the open hard problem.

## Next action (post #1-STOP)
#1/#3 (bit-identical, fixed-half) are exhausted. The ~90% per-pixel half is the target. Disciplined next
step = MEASURE the per-pixel half's reducibility before committing a lever, and finally measure the real
target, all GPU-bound (≫ vsync) so the timestamps are honest:
1. **Overdraw counter** — cliff-free covered-pixel-fragment count via the audit pass (NOT a 3rd atomic).
   Tells us how much of the per-pixel ~90% is redundant (occluded/overdrawn) ⇒ whether #5 (same-frame
   cluster Hi-Z, bit-identical) has headroom.
2. **Worst-pose + a fixed-replay motion glide** (`probe-worstpos.ts`) sampling `c.nanRasterWorld1` —
   the literal "during movement" target, never yet measured.
3. Re-verify the **vsync-cap timestamp artifact** independently (it's load-bearing for all the above).
Then decide: #5 (overdraw/occlusion, bit-identical) vs #4 (banded-τ count cut, quality → judge-shots).

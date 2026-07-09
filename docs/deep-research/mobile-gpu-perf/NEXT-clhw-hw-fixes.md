# NEXT ACTIONS — clhw HW-path fix bundle (post-compaction handoff, 2026-07-05)

## PROGRESS 2026-07-05b — E + F + D SHIPPED (behind flag / always-on), A′ SURFACED FOR DECISION
- **E (yawSc broadcast) + F (matClass broadcast): DONE, typecheck-clean, DEFAULT SW path, bit-identical.**
  shU renumbered: slot 10 = matClass (unconditional), slot 11 = clhw classify (clhw+world1); size
  `?12:11`. shF 21→23: slots 21,22 = yawSc.cy/.sy (makeCtx already computed them ⇒ zero recompute).
  `instYaw` import dropped (last use was the reconstruction). Slot map audited coherent (every write
  index has a matching read; wind + clhw conditionals gate write+read symmetrically). matClassBroadcast
  escapes the wgcache block via a `let` + `??` fallback so `?wgcache=0` still does the per-thread reload.
- **D (clhw seam snap): DONE, typecheck-clean, clhw-only** (instanced VS at ~:2142). Exact inverse of the
  SW `(ndc+1)·0.5·(W,H)` screen transform with a 1/256 quantize between; Y-flip preserves the grid.
- **A′ NOT built — premise-audit fired (see below). Awaiting user call on measure-first vs build-A′.**

### ⚠️ A′ FINDING that changes the calculus (surfaced to user 2026-07-05b)
1. **A′'s precompute buffer ≈130 MB.** QRASTER_CAP(world)=1,048,576; per-cluster ctx pack ≈33 floats ⇒
   ~130MB. Can't safely shrink: capping the HW-cluster count desyncs the *stateless* SW-skip
   (clusterHwClass, no queue read) from the HW-append ⇒ holes. It's clhw-only (default boot untouched),
   but it's a real cost the greenlit plan didn't price.
2. **A′ only fixes VS COST (makeCtx-per-vertex). The doc's own "big lever" is B (native depth + HSR).**
   If the clhw HW path is overdraw/no-HSR-bound rather than VS-bound, A′ polishes the wrong thing.
3. **clhw is UNMEASURED.** With E+F+D in, a clhw build is measurable NOW. Recommendation = MEASURE FIRST
   (gpuWall clhw on/off at clhwmax 16/32/64 + `?scar` for HW fragment-share/overdraw): the result says
   whether the per-cluster idea wins at all AND whether the bottleneck is VS-cost (→A′ worth 130MB) or
   overdraw (→skip A′, go to B). Cheaper + more informative than building A′ blind.
   Recipe: `http://localhost:5173/?scene=world&nanite=1&dpr=2&clhw=1` (+`&scar=1`, +`&nandbg=clhw`,
   sweep `&clhwmax=16/32/64`); A/B vs no `clhw`.

## State right now
- **`?clhw=1` per-cluster SW/HW split is BUILT** (behind flag, default byte-identical, typecheck-clean).
  User tested: **"looks the same" (correct).** The classify is hoisted to the wgcache thread-0
  broadcast (slot 10) — 1×/cluster not 128×. Debug tint `?nandbg=clhw` (RED=HW/GREEN=SW) built.
  Old `?nanitedbg=hwref` DELETED. See memory `sw-hw-per-cluster-raster` + `SW-HW-CLUSTER-AUDIT.md`.
- **Two cold reviews done** (fresh general-purpose subagents, opus):
  - *Granularity + contamination:* **clhw A/B is CLEAN** — clhw-off ≡ pre-feature (every buffer/kernel/
    scene/dispatch gated; shU collapses to 10; projK dead node; swmax-clamp-removal only bites explicit
    `?swmax>16`). Always-on deltas (grass widenT, hwref delete) hit both arms equally → don't confound.
  - *HW optimality:* HW path **correct** but ~50-70% off optimal for the clhw direction. Findings below.
- ⚠️ **A FORK FAILED to build this bundle** (0 tool uses, echoed a plan — fork-mode continues the chat
  instead of executing). **Use a FRESH general-purpose subagent (opus) OR do it yourself. DO NOT FORK
  for implementation.** The two review subagents (fresh, not forks) executed perfectly.

## THE IMMEDIATE NEXT BUILD — bundle A′ + D + E + F (user said GO). NO NEW PARAMS.
User rule (verbatim intent): "for obvious wins no more params please. our code is ugly enough with all
these params and variants." So: A′/D ride the EXISTING `?clhw`; E/F are UNCONDITIONAL (always-on SW wins).
Code-only, typecheck until clean, commit nothing. Verify every anchor with Read/grep first.

### A′ — kill makeCtx-per-vertex in the ?clhw HW draw (KEYSTONE; the "bad" thing the user flagged)
Today the ?clhw instanced HW VS (`buildHwMaterial` `instanced` branch, ~NaniteRaster.ts:2085) calls
`makeCtx(instId,ci)` PER VERTEX (384×/cluster incl. wind gust TEXTURE samples) + fetches all 3 corners.
1. NEW compute pass `kHwCtxPrecompute` (gated under existing `clhw`; put it where makeCtx is available —
   NaniteRaster has it via makeFetch; sequence it AFTER kHwPartition, BEFORE the HW cluster draw). ONE
   THREAD PER HW CLUSTER (count = hwPartCount; add a tiny dispatch-args kernel or reuse a count). Thread i:
   qHwRaster[i]=tid → qRaster[tid+1]=(instId,ci) → `makeCtx` ONCE → write the ~30 decoded fields to NEW
   `hwCtxBuf` (clhw-only) at stride×i, using the SAME pack layout the wgcache uses (NaniteRaster ~759-830:
   isHF,isDAG,triStart,triCount,meshId,channel,gx,gz,qxw,twoSided, A.xyzw, B.xyzw, oX,oZ,cell, + wind
   h0/dirX/dirY/leanBase/swayABase/swayPhase/ph/branchBase/flutBase/swayXPhase when wind on).
2. Rewrite the ?clhw instanced VS to READ hwCtxBuf[instanceIndex] + reconstruct ctx (mirror the wgcache
   post-barrier reconstruction) INSTEAD of makeCtx; then `fetchWorldVertDyn(ctx,localTri,corner)`
   single-corner (drop the 3-corner select; do NOT touch the ?hw1fetch param). Payload + degenerate-tail
   clip unchanged.
3. Binding budget (Metal 10 storage/stage): the new VS binds FEWER (hwCtxBuf+qHwRaster+indices+verts+
   visPayloadV+visBV) since instances/meshes/clusters move to the precompute. VERIFY both stages ≤10.

### D — seam fix: snap the ?clhw HW clip xy to the 1/256 grid (SW uses it at ~NaniteRaster.ts:1150-1165). clhw only.

### E — broadcast yawSc (DEFAULT SW path — MUST be BIT-IDENTICAL). wgcache post-barrier `yawSc: instYaw(cB)`
recomputes cos+sin on all 128 threads. Thread 0 writes cy=cos(B.x), sy=sin(B.x) to TWO new shF slots
(21,22); reconstruction reads them. shF 21→23.

### F — broadcast matClass (DEFAULT SW path — bit-identical). Voxel-skip (~NaniteRaster.ts:887-899) reloads
mesh word6 per-thread. Thread 0 extracts matClass=(word6>>8)&0xff → NEW shU slot; per-thread skip reads it.
**SLOT LAYOUT (do coherently — a mismatch corrupts the SHIPPED SW path and typecheck won't catch it):** shU
today = 10 (slots 0-9) + slot 10 = clhw classify (conditional). Make matClass UNCONDITIONAL slot 10; MOVE
clhw classify to slot 11 (conditional). shU size = `clhw&&mode==='world1' ? 12 : 11`. Update EVERY
setU/getU index + size ternary + the `?wgcache=0` fallback. Document the final slot map in a comment.

### ⚠️ REVIEW GATE (orchestrator does this after the build): line-by-line check the E/F shU/shF slot map
(write index == every read index) — this is the one place NOT to trust the builder; it's the shipped path.
Also verify the two binding counts. Then hand user the test recipe.

### Test after build: default (NO flags) still renders right [E/F shipped-path check] + `?clhw=1` still
"looks the same" + `?nandbg=clhw` tint. Recipe: `http://localhost:5173/?scene=world&nanite=1&dpr=2&clhw=1`.

## DEFERRED — next, MEASUREMENT-GATED stage (do NOT build until measured)
- **Measure first:** `?scar` (HW-routed fragment share + overdraw, NaniteRaster ~:2181), A/B gpuWall clhw
  on/off, eyeball `?nandbg=clhw`. The old "SW≈HW parity" is UNTRUSTWORTHY (botched soup + [2,16] swmax clamp).
- **B — native depth + R32Uint id-MRT + per-pixel merge** (HW-review F1, THE big lever): HW draws into a
  real depth attachment + id MRT (Apple HSR/early-Z rejects occluded frags BEFORE shading, ZERO per-frag
  atomics) → a per-pixel merge folds (depth,id) into the unified visPayloadV/visBV via the same depthKey24
  election. hwRT today = `RenderTarget(w,h,{depthBuffer:false})` (NaniteRaster:2248), depthTest/Write=false,
  side=DoubleSide (2218) → HSR fully defeated. The unified vis-buffer IS load-bearing (voxel/grass pre-seed
  reads it, NaniteVoxelRaster) — but that requires the per-PIXEL winner, NOT per-fragment atomics; separable.
- **C — split cull** (CullBack solids / DoubleSide leaves via twoSided flag) — fold into B (needs 2 draws;
  compounds with early-Z). **Full indexed/vertex-shared draw** — also fold into B (B rebuilds the draw).

## Standing constraints (user)
No new params; keep code CLEAN (reduce variants, don't add). Opus only, no fable agents. Implementation
subagents CODE-ONLY (no GPU/probe/bench/dev-server). Commit only when asked. rscale/DRS banned. Crowns
POSTPONED (this SW/HW arc is the detour). FORMATTING debt: NaniteRaster.ts was prettier-reflowed by the
user's format-on-save (quotes flipped back via scratchpad/flipquotes.mjs; line-reflow remains) — a
verified-clean reconstruction (revert HEAD + reapply, whitespace-stripped==tested) is deferred to commit
time; format-on-save now disabled project-local (.zed/, git-excluded).

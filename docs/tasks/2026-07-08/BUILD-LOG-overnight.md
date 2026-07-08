# Overnight build log — 2026-07-08 · task #76 easy-mid perf levers

> **UPDATE (post-eyeball) — Lever B (compaction) was REVERTED and is NOT shipped.** It dropped whole
> clusters at some views. Root cause: the projVertBuf is only "empty" on the *average* view; at a
> **crown-canopy peak** it is ~full — leaf-crown clusters are **mesh (already deduped)** and genuinely
> hold ~450–494 of their 512 slots, and ~192K are visible, so the drop-free size ≈ the flat 98.3M (no
> memory win; sizing tight = drops). B is memory-only, so with no win it's pure downside → reverted
> (flat 1.18 GB restored). The real projVertBuf memory lever is **fewer/coarser crown clusters**
> (far-field DAG coarsening or voxelized distant crowns), a separate arc — NOT a buffer-layout trick.
> **SHIPPED (this commit): Lever 1 + Lever A + the Mid levers + Lever C.** The §3 "Proj B+C" section
> below is superseded (B reverted; C kept).

All non-merge, easy-mid levers from the two proposals (`PROPOSAL-nanProjectVerts.md`,
`PROPOSAL-nanMidRaster.md`) implemented on top of the uncommitted Lever 1.
**tsc green. NOTHING committed** (your standing rule). No refusion (per your decree).
⚠️ All perf figures are **PROJECTED** from the proposals — not yet re-profiled or eyeballed.
Every changed block is marked `// LEVER A/B/C` for locatability in the uncommitted tree.

---

## DO THIS FIRST — eyeball checklist
Test URL: `http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhwmax=32&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1`
1. **Renders at all?** EMPTY scene ⇒ the compaction pushed Classify over the 10-storage-buffer ceiling (Risk #1) → revert compaction (below).
2. **Scrambled / cross-cluster geometry?** ⇒ a compaction base/clamp desync (Risk #2) → revert compaction.
3. **Holes/corruption otherwise?** ⇒ look at Lever A / Mid.
4. `&middz=1` ⇒ exercises the incremental-depth mid path (quality-risk; the default *without* it is bit-identical).
5. **Register counts:** world1 should still be **56** (untouched); `nanProjectVerts` changed (Lever A).

---

## Implemented (tsc-green, uncommitted, bit-identical unless noted)

### 1 · Projection Lever A — cooperative shared ctx  — ✅ verified, LOW risk
`Project.ts`. Broadcast-read fix: the 255 threads/workgroup re-reading the same 35-word ctx (the measured
buffer-read-limiter 92% / LLC 96% wall) now load it **once/workgroup into threadgroup shared memory + one barrier**,
decode from shared. Bit-identical records, zero VRAM. Barrier uniform-by-construction (verified). Projected ~2–2.7×.

### 2 · Mid levers  — ✅ verified, LOW risk (A.2 gated)
`Mid.ts`, `Scanline.ts`, `NaniteCommon.ts` (+ uW→uint ripple: `NaniteCull/Hzb/Resolve/Raster`).
- **Bit-identical (no flag):** A.1 area2-by-negate, A.3 pack-bias, A.4 winding-crest, `MID_STRIDE *1u` fold,
  and `uW/uH → uint` uniform (kills the per-scanline-row `f32→u32(width)`). Verified the uW→uint ripple is exact
  (integer pixel dims ⇒ `float(uint(w)) == w`). world1 kept its **own** `makeScanline` instance ⇒ untouched, still 56.
- **A.2 incremental depth — behind `?middz` (default OFF).** OFF = bit-identical recompute; ON = faster, MEDIUM
  parity risk (float accumulation can differ in the `depthKey24` LSB). Clean build-time branch, delete one side trivially.

### 3 · Projection Lever B+C — compaction (−~900 MB)  — ⚠️ HIGHEST EYEBALL RISK
New `raster/ProjBase.ts`; edits in `Project.ts`, `NaniteRaster.ts`, `Mid.ts`, `GeometryRegistry.ts`, `VisBuffer.ts`.
- **C:** `VCACHE_VERTS` 382→512 so every mesh cluster dedups (no tooWide fallback). Consumers verified safe.
- **B:** `projVertBuf` **1.18 GB → 288 MB** via **atomic-bump** per-cluster allocation (new `nanProjBase` pre-pass,
  spliced kVisClear→kClusterCtx→**kProjBase**→kProjectVerts→Classify→Mid) + a **single interleaved** `nanClusterReserve`
  buffer (`[2i]=base, [2i+1]=vertCount`) read identically at all 3 sites. Bit-identical (only each cluster's base offset
  moves; per-cluster clamp contains any rogue within the cluster). Overflow is graceful (SENTINEL → cluster renders nothing).
- **My spec had 2 errors the agent's premise-audit caught + fixed:** (i) "two buffers" would breach the 10-buffer
  ceiling → interleaved to one (+1 binding, Classify 8→9); (ii) "replace the cap guard" → OOB/corruption → **kept BOTH**
  the `PROJ_CLUSTER_CAP` guard AND the new SENTINEL guard (identical in Project & Classify).
- **Residual risks (only your GPU can settle):**
  1. **Binding ceiling** — Classify 8→9 storage buffers (safe ≤10, but "8" is from a code comment, not a live run).
     *Empty scene = this.* Fix if hit: fold `nanClusterReserve` into an existing buffer's tail (the scar/trihzb pattern).
  2. **Base/clamp desync** → scrambled geometry. (Logic verified sound; only a render confirms.)
  3. **`PROJ_VERT_CAP = 24M`** (23% headroom over the ~19.5M measured peak) → an extreme frame drops late clusters =
     far-field holes (not scramble). One-line bump.
- **REVERT PATH if broken:** delete `src/nanite/raster/ProjBase.ts` + revert the `// LEVER B COMPACTION` / `// LEVER C`
  hunks in the 5 listed files → falls back to the flat 1.18 GB layout + the verified-safe A + Mid levers.

---

## Skipped (with reasons — surfaced, not parked)
- **Refusion** (fuse mid into classifier / eliminate projVertBuf) — your decree; kills world1 occupancy.
- **Mid dead baked-grid guard removal** — needs a change to shared `Tsl.setIndirectDispatch` (many kernels) → too risky unsupervised.
- **Lever E** (drop Tint robustness `min()` clamps) — Dawn robustness is unstrippable on public web (our own findings).
- **Lever A tightening** (21-word ctx subset / `vec4<u32>` loads) — free but marginal; A already dropped ctx *below* the
  vertex-gather limiter floor, so it no longer moves the needle. Documented, not built.

## Research verdict — `ANALYSIS-projection-broadcast-read.md`
Complete elimination of the ctx broadcast is **NOT** achievable without refusion (the one true HW broadcast — Metal
`constant`/uniform buffer — is blocked by the 64 KB uniform limit vs 21 MB ctx + per-dispatch-not-per-workgroup binding;
subgroups issue 8× more loads for ~0 net gain). **Lever A is the refusion-free floor, and a good one.**

## Parked for you (morning, as agreed)
- **Classifier extraction** into `raster/Classify.ts` — you chose the shared-`TriSetup` split (needs shadow + NaniteView eyeball).
- **The param cull** — relect 0/2, swcoop, coopv, ctxsm, ksplit, scar (dead experimental flags).

## Next real lever (NOT done — needs your call)
- Mid full 3× needs either the fuse (refused) or `?trihzb` occlusion (temporal quality risk); the bit-identical crest
  cuts alone are ~1.6–1.9×.
- **Re-profile after eyeball** to replace the projected numbers with measured ones (fresh capture → Xcode export → `run_all.sh`).

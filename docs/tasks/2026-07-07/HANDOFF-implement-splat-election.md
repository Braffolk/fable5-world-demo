# HANDOFF → implement the splat-election pipeline (post-compaction, cold start)

> **STATUS 2026-07-07 (post-compaction): IMPLEMENTED, `tsc` green, awaiting the user's GPU capture.**
> All 6 pieces are in `src/nanite/NaniteRaster.ts`, gated behind **`?splat=1`** (default OFF ⇒ byte-identical
> to today — the ≤1px branch still calls `doElection` inline). Decisions taken during the build:
> - **INDIRECT dispatch** (`kSplatArgs`→`nanSplatElect`), NOT the direct over-dispatch — the proven cull-BFS
>   `kArgs`→`kTraverse` in-batch pattern (auto-synced in-pass; `Tsl.ts:332-343` + `NaniteCull.ts:1300-1343`).
>   Direct-grid was rejected because splat's N is a *tiny* fraction of the cap ⇒ a fixed grid wastes ~60k idle
>   workgroups/frame; indirect launches only `ceil(N/64)`.
> - `SPLAT_CAP = 4_194_240` (= 65535·64, the 1-D grid ceiling) ⇒ ~50 MB. `splatQueue[0]` is the **free overflow
>   gauge**: it is `atomicAdd`'d past the cap, so `count − CAP` = dropped sub-pixel fragments. The append is
>   **overdraw-scaled** (one record per covered sub-pixel fragment, not per triangle) — this counter is how we
>   learn the true N and whether 50 MB is enough. Read it in the same capture.
> - `nanSplatElect` inlines the `relect===1` election verbatim (closure can't cross `Fn`s); binds only
>   `visPayloadV`/`visBV`/`splatQueue` = 3 storage buffers.
> - Batch order: `… kRasterWorld1, grass, kHwArgs, kSplatArgs, kSplatElect` — after the raster (RAW on
>   splatQueue), before the HZB-tail submit reads `visPayloadV`.
>
> **THE capture (hand to the user):**
> `URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhw=1&clhwmax=32&profile=1&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1&splat=1' CAPTURE=1 tools/profile/gputrace.sh`
> → read **`nanSplatElect`** register count (gate 1, target ≤~30), visual parity at this vs `&splat=0` (gate 2),
> frame-time A/B `splat=1` vs `splat=0` ablated (gate 3, the ship decision), and `splatQueue[0]` overflow.
>
> **#1 risk to watch:** a black frame / validation error with `?splat=1` ⇒ world1 hit **11 storage buffers**
> (the ctx-prepass was supposed to leave room for exactly 1). If so, fold the splat records into the `hwQueue`
> tail the way `scar`/`trihzb` were (plan §5 Piece 1 ⚠, `NaniteRaster.ts:542-568`). **SMOKE RESULT
> 2026-07-07: CLEARED** — user eyeballed the live page at `?splat=1`: renders fine, no black frame ⇒ world1
> sits at 10 storage buffers (valid), and the appended sub-pixel fragments elect correctly (visual parity
> preliminarily holds). Only the register/frame/overflow numbers remain — those need the Xcode capture.
>
> The rest of this note is the original recipe (now executed).

**Read `docs/tasks/2026-07-07/splat-election-register-cut.md` FIRST — it is the design + all reasoning + drawbacks. This note is the execution recipe.**

You (future me) are implementing the splat-election visbuffer for `nanRasterWorld1`. The design is validated by measurement; do NOT re-derive it. Your job: write the 6 pieces, get `tsc` green, then hand the register/visual/frame-time measurement to the user (you cannot run the GPU capture — it needs their Xcode).

---

## Where you are right now (working-tree state, branch `nanite-raster`, NOT committed)

Uncommitted, and these are **KEEPERS — do not revert**:
- `src/nanite/NaniteRaster.ts` (+82 lines): (a) the **≤1px point-sample branch** and (b) the **`rdbg=5` probe**.
- `tools/profile/gputrace.sh`: `NOROBUST=1` support (diagnostic; never ship).

Also modified but **NOT yours / do NOT commit**:
- `docs/METAL-PROFILING.md` (stale 2-line tweak — leave it, don't stage it).
- `docs/todo-human-only.md` — **the user's personal file, NEVER commit it. Standing rule.**

New untracked: `docs/tasks/2026-07-07/` (this note + the plan). Fine to keep.

`tsc` is green (`npx tsc --noEmit` exit 0). The throwaways (1-point splat, synthetic-election, ≤1px-only) are already reverted.

**Commit/push policy: only when the user explicitly asks.** You're on a feature branch already.

---

## The one-paragraph WHY (so you don't second-guess mid-build)

`nanRasterWorld1` is 80 registers / ~40% occupancy; target 40. Measured bisection (plan §2) proved: the loop isn't the peak (−4), the projection isn't the peak (it's a 64 floor *below* the emit), a per-vertex pre-pass can't help a triangle kernel (peak is downstream), and 16 registers are Dawn robustness that **public web cannot strip**. The only thing that reaches 40 is to **stop rasterizing the sub-pixel majority as triangles and splat them as points**: measured election floor with a precomputed point = **26 registers**. So we move the sub-pixel election out of the megakernel into its own ~28-register `nanSplatElect` kernel, fed by a queue `world1` appends to. Same visbuffer, same resolve.

---

## Execute in this order (each step: make the edit, then `npx tsc --noEmit`)

Code for every piece is in **plan doc §5** — copy it, adjust to live line numbers. Locate sites by grep (line numbers drift):

1. **Confirm the binding budget is safe.** `world1` is at the WebGPU 10-storage-buffer ceiling; `ctxsm=1` sheds to 9. Verify before adding a buffer:
   - `grep -n "10-storage-buffer\|storage buffer\|SCAR_BASE" src/nanite/NaniteRaster.ts` and re-read ~L548-584.
   - If `world1` truly has room for 1 more with the ctx pre-pass on → proceed. If not → fold the splat counter into `hwQueue` spare space like `scar` was (plan §5 Piece 1 ⚠️).

2. **Piece 1 — `splatQueue` buffer.** Anchor: `grep -n "hwQueueAttr.name = 'nanHwQueue'"` (~L561). Add `splatQueueAttr`/`splatQueueV` + `splatDrawAttr`/`splatDrawBuf` right after the `hwQueue`/`hwDraw` block, mirroring it exactly. `SPLAT_CAP = 4_194_304` to start.

3. **Piece 2 — clear in `kVisClear`.** Anchor: `grep -n "atomicStore(hwQueueV.atomic.element(0), uint(0))"` → **two** hits (~L682 main clear, ~L706 scoped/shadow clear). Add `atomicStore(splatQueueV.atomic.element(0), uint(0));` next to the **main** one (L682 path). Check whether the scoped clear (L706) also needs it — only if world1 runs after a scoped clear; safest to clear in both.

4. **Piece 3 — append instead of elect (the ≤1px branch).** Anchor: `grep -n "task #76 PERF: a single-pixel-bbox tri"` (~L1776). Inside that `If(cz in range)` block it calls `doElection(px, cand, payload)`. **Replace that single call** with the atomic-append snippet (plan §5 Piece 3). `px`, `cand`, `payload` are all already in scope there. Leave the rest of the branch (the coverage test, `uw`, `cz`) untouched.

5. **Piece 4+5 — `kSplatArgs` + `nanSplatElect`.** Anchor: `grep -n "const kHwArgs = Fn"` (~L2248). Add both kernels right after `kHwArgs`. **CRITICAL: `doElection` is a closure *inside* `world1`'s `Fn` — you cannot call it from `nanSplatElect`. Inline the `relect===1` election body verbatim** (it's at `grep -n "const doElection" ` ~L772, the `else` branch L783-793). `nanSplatElect` binds only `visPayloadV`, `visBV`, `splatQueue`. Use `setIndirectDispatch(kSplatElect, splatDrawAttr)`.
   - ⚠️ `splatDrawBuf.element(0)` is a **workgroup count** for a compute indirect dispatch = `ceil(count / wgSize)`, NOT the record count. With `[64]` workgroups, `kSplatArgs` must write `(n + 63) / 64` into `element(0)`. Double-check against how `setIndirectDispatch` + the existing indirect kernels interpret the 3 dispatch words (read `grep -n "setIndirectDispatch" src/nanite/NaniteRaster.ts` and one existing user, e.g. `rasterDispatchFullAttr`).

6. **Piece 6 — batch wiring.** Anchor: `grep -n "dispatchBatchMixed(renderer, \["` (~L2825). Insert `kSplatArgs` then `kSplatElect` into the world1 batch **after** the raster kernel and **before** `kHwArgs`/resolve. Also confirm nothing reads the visbuffer between the raster and `nanSplatElect` (HZB build, resolve) — `nanSplatElect` must finish before any visbuffer read (plan §6 R7).

After all 6: `npx tsc --noEmit` must be exit 0.

---

## Decisions already locked (do not re-litigate)

- **Single-counter append first** (not workgroup-compaction). Measure contention before optimizing (plan §6 R4). The `hwQueue` single-counter pattern is the template.
- **Inline the election** in `nanSplatElect` (closure can't be shared).
- **`world1` keeps projecting** — its register count does NOT drop; only the sub-pixel election leaves. The win is `nanSplatElect` at 28 + high occupancy (plan §6 R2).
- **Reuse the existing ≤1px point-sample branch** as the append site (it already computes `px/cand/payload`).
- **`cand = depthKey24(cz) << 8 | (payload & 0xff)`; side-store `payload`.** Do not drift the packing (plan §5 reference).

---

## Verification (what you can do vs what needs the user)

- **You:** `npx tsc --noEmit` after each piece. Read-check the packing/ordering.
- **User (GPU capture — you cannot):** ask them to run
  `URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhw=1&clhwmax=32&profile=1&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1' CAPTURE=1 tools/profile/gputrace.sh`
  then open in Xcode and read **`nanSplatElect`**'s register count.
- **Acceptance gates (plan §7):** (1) `nanSplatElect` ≤ ~30 regs. (2) visual parity at `rdbg=0`. (3) frame-time A/B neutral-or-better (ablated, not summed — Apple render‖compute overlap). Gate 3 decides ship.

---

## If it disappoints — premise-audit before concluding (repo `CLAUDE.md` rule, and you won't have it in context)

A negative result triggers a premise-audit, NOT method-thrashing. Check the specific claim:
- `nanSplatElect` > 40? → is robustness on the 3 reads the cost? is the indirect workgroup-count wrong? Measure the isolated claim.
- Registers hit 40 but frame time flat? → the election was NOT the bottleneck (plan §6 R1). Then `world1`'s projection at 40% occ is the cost → plan §8 Phase 2 (lean the projection; needs per-instance wind precompute), and only pivot to "SW-triangle can't hit 40 on public web → push geometry to HW / redirect to the 34% divergence" after §8's kill criteria genuinely fire.
- Append hot in a capture? → workgroup-compaction (plan §6 R4).

Do not return "not-doable" until the premise-audit has cleared params/metric/structure. We were wrong four times this session (loop=peak, projection=peak, pre-pass helps, robustness strippable) — each refuted by ONE measurement. Measure the specific claim before building on it.

---

## Quick orientation greps (paste these first thing)

```
grep -n "task #76 PERF: a single-pixel-bbox tri" src/nanite/NaniteRaster.ts   # Piece 3 site (append here)
grep -n "hwQueueAttr.name = 'nanHwQueue'"        src/nanite/NaniteRaster.ts   # Piece 1 site
grep -n "atomicStore(hwQueueV.atomic.element(0), uint(0))" src/nanite/NaniteRaster.ts  # Piece 2 sites
grep -n "const kHwArgs = Fn"                     src/nanite/NaniteRaster.ts   # Piece 4/5 site
grep -n "const doElection ="                     src/nanite/NaniteRaster.ts   # copy the relect===1 body from here
grep -n "dispatchBatchMixed(renderer, \\["       src/nanite/NaniteRaster.ts   # Piece 6 batch
grep -n "setIndirectDispatch"                    src/nanite/NaniteRaster.ts   # indirect dispatch API + workgroup-count semantics
```

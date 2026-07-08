# PROPOSAL — make `nanMidRaster` ~3× faster, ZERO-VRAM (≤150 MB hard ceiling)

Task #76 follow-up. **HARD CONSTRAINT (coordinator, non-negotiable): ≤150 MB additional VRAM,
full stop.** The current mid queue is ~160 MB (1 u32 × 40 M records); 150 MB spread over the real
~36 M mid count is **≈1 extra u32 per record** — nowhere near the 12-16 u32 an edge-setup record
needs. **Record growth is DEAD (fat/medium/"small" all off the table). The entire 3× must come from
zero-VRAM levers.** (The earlier fat-record proposal is retracted for this reason.)

Fresh capture: `/private/tmp/laas_trace-2026-07-08T02-03-38-c000(-perf).gputrace`. Analysis:
`profile-results-20260708-021537/`. **No source modified; this is a written proposal only.**

TL;DR: On the fresh trace `nanMidRaster` is the **#1 GPU consumer (25.8 %)** and **45.5 % of its
own runtime is line-0 register-spill/prologue** — with **0 spilled bytes** (agc), so that mass is
**occupancy-stall + redundant prologue, not memory spilling.** It is integer-ALU bound at **~44 %
occupancy, register-capped (40 agc-temp / ≈64 Xcode-allocated)**. The lever order is therefore:
**(A) cut the register crest to raise occupancy and drain the 45 % stall; (B) strength-reduce the
integer-ALU + delete the redundant prologue; (C) fuse the raster into the classifier — which
*removes the queue entirely (−160 MB)* — gated on a world1 register check; (D) if that stack is
short, cut the *number* of mid triangles with per-tri HZB occlusion (off by default, zero extra
VRAM).** All four spend ≤0 bytes; two of them *free* memory.

---

## 1. REFRESHED DIAGNOSIS (fresh trace, re-confirmed) & PREMISE RE-AUDIT

### 1a. The numbers the zero-VRAM levers target (fresh trace)
| metric | old trace | **fresh trace** | source |
|---|---|---|---|
| nanMidRaster share of GPU | 12.4 % (#2) | **25.8 % (#1)** | `runtime/001_bin_1b2b.txt` |
| **line-0 (spill/prologue/stall)** | 41.9 % | **45.5 %** | same |
| inner scanline loop (lines 338/341) | ~19 % | **~21 %** | same |
| election atomics (255/549/374) | ~14 % | **~15 %** | same |
| agc temp registers / spilled bytes | 40 / **0** | 40 / **0** | `static/_ranking.txt` (raster family) |
| Xcode allocated regs / occupancy | 64 / 44 % | 64 / 44 % | Xcode timeline (coordinator) |
| ALU limiter / ALU utilization | 91 % / 64 % | 91 % / 64 % | Xcode counters |
| memory read/write util | 10-35 % / idle | 10-35 % / idle | Xcode counters |

The identification is unchanged and unambiguous: `bin_1b2b` does the depth-keyed election
(`atomic_fetch_max`, line 549) with **no `atomic_fetch_add`** (a consumer, not the classifier) and
the identical hot-line fingerprint (341/338/255/549/374/333/399/197). Its MSL is `store0` **block
42** (`min(count,41943040u)` mid-guard, `canonVertSlot` `…min(…,511u))*3u` ×3, the double-`while`
scanline). **The capture moment shifted (mid 12→26 %) but the *structure* is byte-identical** — same
block 42, same crest, same 45 % line-0. The shift only makes this the more urgent target.

### 1b. Premise re-audit — going up a level, correctly this time
My first pass defaulted to "make mid's setup available in VRAM" — that stayed inside the problem and
is now structurally impossible (≤150 MB). Putting the real premise on trial:

- **The metric is right, and it points *away* from VRAM.** 45.5 % of the shader is line-0 with **0
  spilled bytes**. So this is **not** a bandwidth or a spill-to-memory problem — it is an
  **occupancy** problem: at 44 % occupancy the long dependent integer setup chain
  (`corner-read → area2raw → flip → wind → area2 → rcpArea → rw`) stalls with too few resident warps
  to hide its latency, and the PC sampler bins those stalls to the prologue (line 0). **Occupancy is
  the biggest bucket and costs zero bytes.**
- **Part of that 45 % is *redundant prologue*, not irreducible stall.** The MSL prologue carries a
  **dead baked-grid guard + a second thread-linearization** (block 154-160), a **`* 1u`**
  (`MID_STRIDE`, block 163), **37 Tint `min(idx, tint_array_length-1)` bounds-checks** (many on the
  12 corner re-reads), and a per-row **`f32_to_u32(width)`** (block 328). These are strength-
  reducible (B) or removable-by-fusion (C) with zero VRAM.
- **The generating structure is still the flaw — but the zero-VRAM fix is to move the RASTER to the
  setup, not the setup to VRAM.** The classifier already holds the full edge-setup live at the
  append site; fusing the scanline there (C) removes the re-derivation *and* the 160 MB queue.
- **If A+B+C are short of 3×, the deeper premise is the mid-triangle COUNT** (~36 M, inflated by the
  un-coarsened far-field DAG). Per-tri HZB occlusion (D) culls occluded mid tris for zero extra
  VRAM. This is the audit's fallback and a proportional multiplier.

Cleared. The answer is a zero-VRAM stack, not "needs VRAM" and not "already optimal."

### 1c. Where the register crest comes from (Lever-A evidence, block 42)
The scanline forces the entire edge-setup to stay **live across the loop** — this is why mid (which
runs the loop) sits above world1 (which appends and returns). Loop-carried live set (block 310-378):

`rw0/1/2` (→`cw`, 3) · `sx0/1/2` (3) · `sy0/1/2` (3) · `bias0/1/2` (3) · `dz0/1/2` (3) · `rcpArea`
(1) · `startX/Y,endX/Y` (4) · `payload` (1) ≈ **21 values held for the whole loop**, plus the
per-pixel working temps (`cw` copies, `cz`, `uw0/1/2`, `px`, `cand`, `x`, `y`, two `tint_loop_idx`
guards). That ~21-value floor + the setup crest (raw 3 corners **and** the 6 wound temps coexisting
during the winding swap, block 198-235) is the 40-temp / 64-alloc / 44 %-occupancy cap.

**Naïveté found (mid does *more* than world1 here):** block 236 **recomputes `area2` as a fresh
cross-product from the wound corners** (`nodeVar37 = …nodeVar32 − …`), then `rcpArea = 1/area2`.
world1 instead does `area2 = flip ? −area2raw : area2raw` (`NaniteRaster.ts:1417`) — a negate, no
second cross-product. Mid pays 2 extra integer muls **on the dependent chain** the stall is waiting
on. (Mid *did* inherit world1's R3 lazy-edge-deltas — `ex0..ey2` are un-`toVar`'d, block 244-309 —
so that crest trick is present; the R2 streaming doesn't apply to the projVertBuf-read path, which
must hold all 3 corners to wind.)

---

## 2. THE ZERO-VRAM LEVERS (ranked; VRAM delta ≤ 0 for every one)

### ★ LEVER A — Register-crest reduction → occupancy (LEAD; hits the 45 % line-0 bucket, 0 bytes)
**Mechanism.** Shrink the peak simultaneous live-set so the register allocation drops below an Apple
occupancy step, lifting occupancy from 44 % → higher, which (i) hides the dependent-chain latency
that is currently binned as the 45 % line-0 stall, and (ii) finally feeds the pegged-but-starved ALU
(91 % limiter vs 64 % util). Concrete crest cuts, in impact order:
1. **`area2 = flip ? −area2raw : area2raw`** instead of the block-236 recompute — removes 2 muls
   from the critical dependency chain and a temp. *(Also a Lever-B item; it belongs to both.)*
2. **Incremental depth interpolation.** Replace the 4 loop-carried floats `dz0/dz1/dz2,rcpArea` +
   the per-pixel `uw=cw−bias; cz=(uw0·dz0+uw1·dz1+uw2·dz2)·rcpArea` (3 sub + 3 mul + 2 add + 1 mul)
   with a held `{z, dzdx, dzdy}` (3 floats) stepping `z += dzdx` per pixel / `+= dzdy` per row.
   `cz` is already an affine function of the incrementally-stepped `cw`, so `dzdx =
   (sx0·dz0+sx1·dz1+sx2·dz2)·rcpArea` is constant — mathematically identical, computed once at setup.
   **Drops 1 held register AND ~6 ALU/pixel.** *(Parity: float accumulation vs per-pixel recompute
   can differ in the depthKey24 LSB → medium risk; gate on the eyeball render + a depth A/B. If it
   fails parity, keep it as an opt-in and fall back to recompute.)*
3. **Pack `bias0/1/2`** (each 0/−1) into one register (or the spare bits of a start/end word):
   −2 held registers; costs a cheap unpack that folds into the existing `cw−bias`.
4. **Minimise the winding-swap crest** (block 198-235): read → sign → assign wound corners so the
   raw corners die immediately (avoid the `.toVar()` chain that pins raw + wound together).

- **Measured cost removed:** the 45.5 % line-0 (occupancy stall) — the single biggest bucket — plus
  per-pixel loop ALU (item 2).
- **VRAM delta: 0.**  **world1 register risk: 0** (mid-only).  **Parity risk: LOW** except item 2
  (medium — gated).  **Expected gain: ~1.4-1.7×** (register drop 40→~30-32 temp → occupancy
  ~44 %→~55-60 % → reclaim ~20-30 of the 45 points; item 2 additionally speeds the loop). Occupancy
  *alone* caps near ~1.8× because the useful work (~55 %) is fixed — so A must stack with B/C/D.
- **Impl sketch:** `raster/Mid.ts` — swap the block-236 `area2` recompute for `flip.select`; refactor
  the emit/scanline depth path to incremental `z` (touches `raster/Scanline.ts` `emit`/`cz`, or a
  mid-specific scanline variant to keep world1's inline path unchanged); pack bias. Gate: `tsc`,
  eyeball world1 fresh+cache, no mid-field blink, depth A/B for item 2.

### ☆ LEVER B — Strength-reduce the integer-ALU + delete redundant prologue (0 bytes, stacks on A)
**Mechanism.** Cut real integer work and the redundant prologue that inflates line-0:
- **Delete the dead baked-grid guard + second linearization** (block 154-160): the
  `if (instanceIndex >= nodeUniform7) return;` is dead (the grid is overridden by
  `setIndirectDispatch`); the real guard at block 160 uses a *separate* linearization. One
  linearization + one compare per thread are pure waste. *(Origin: the `.compute(MID_CAP,[64])`
  baked grid — investigate suppressing the baked-count guard when an indirect dispatchSize is
  attached, in `Tsl.setIndirectDispatch`/`Mid.ts`.)*
- **Fold `MID_STRIDE`'s `* 1u`** (block 163).
- **Make `cam.uW` a `uint` uniform** (it is `UniformF` today — `NaniteCommon.ts:144`), killing the
  **per-row `f32_to_u32(width)`** at block 328 (inside the hot loop, every scanline row).
- **`±area2raw`** (shared with A.1) — removes 2 muls/tri.
- Fewer buffer reads (see C) also delete Tint bounds-checks (37 today).

- **Measured cost removed:** a slice of the 45 % prologue (dead guard/linearization/bounds-checks) +
  per-row width conversion + 2 muls/tri.  **VRAM delta: 0.  world1 risk: 0** (uW is shared — making
  it uint touches all rasters; verify the depth/shadow inline paths still typecheck, trivial).
  **Parity risk: NONE** (all bit-preserving).  **Expected gain: ~1.1-1.2×**, near-free, stacks on A.
- **Impl sketch:** `Mid.ts` (guard/`*1u`/`area2`), `NaniteCommon.ts` (`uW/uH` → `uniformU`, update
  the ~2 float uses), `Scanline.ts` (`rowBase` uses the uint width).

### ☆ LEVER C — Fuse the mid raster into the classifier (0 bytes, and FREES ~160 MB)
**Mechanism.** The classifier (world1) already computes the full edge-setup and **holds it live at
the mid-append site** (`NaniteRaster.ts:1551-1611`), where today it stores only the payload
(`:1796-1804`). Instead of appending, **call `swScanline` right there.** This is the *structural
inverse* of the rejected fat-record: rather than moving the setup to VRAM for a second pass, move the
one-pass raster to where the setup already lives. It deletes: the entire 25.8 % `nanMidRaster` pass,
its second dispatch, the **12 corner re-reads + their bounds-checks**, the ~150-instruction
re-derivation, **and the ~160 MB mid queue** (−VRAM — this *helps* the memory arc).

- **Measured cost removed:** effectively all of mid's 25.8 % except the scanline+election, which
  moves into the classifier and runs at the classifier's occupancy.
- **VRAM delta: −160 MB.  Parity risk: LOW** (same setup, same election; the append order changes to
  direct raster — watch the mid-field blink test).
- **world1 register risk — the honest number, this is where it lives or dies:** the scanline's
  loop-carried set *is* the setup values world1 already holds, so `cw` reuses `rw`'s registers; the
  genuinely NEW live values are the loop iterators + row-reset copies + `rowBase` + the two
  `tint_loop_idx` guards ≈ **+4-6 registers**. world1 is **at the 56-register mandate**, so this
  projects **~60-62 — a real breach of ~+6.** Whether that matters depends on whether 56→62 crosses
  an Apple occupancy step (banked, stepwise — not linear). **It cannot be settled by reading; it
  needs a one-flag prototype + a register/occupancy read.** *The frame-time case is strong even on a
  breach:* deleting a 25.8 % pass + freeing 160 MB dwarfs a modest world1 occupancy dip — the "56"
  is a proxy for frame time, and the net frame time is what to gate on. **Recommendation: prototype C
  behind a flag, measure world1's registers and the net frame delta; if world1 stays ≤ ~60 with no
  occupancy-step cliff, C is the single biggest lever and it *removes* the shader rather than tuning
  it.** Mid divergence (only mid-lanes rasterize) is marginal — world1 already pays it for the ≤1 px
  splat point-sample.
- **Impl sketch:** `NaniteRaster.ts` mid `.Else()` (~1787-1805): replace the queue append with an
  inline `swScanline(rw…, emit)` reusing the splat path's election; delete `raster/Mid.ts`,
  `buildMid`, and the mid queue/args/clear (`Queues.ts`, `VisBuffer.ts` clear tail). Keep `?nomid`
  as a compile guard.

### ☆ LEVER D — Cut the NUMBER of mid triangles: per-tri HZB occlusion (premise-audit lever, 0 bytes)
**Mechanism.** `?trihzb` (per-triangle occlusion vs the prev-frame HZB) is **implemented but OFF by
default** (`NaniteRaster.ts:329`), and its HZB mirror **folds into the existing hwQueue tail — zero
extra buffer** (`Queues.ts:100-111`). A mid tri whose nearest z is behind the prev-frame HZB farthest
over its bbox never enters the queue. In a dense forest (heavy depth complexity), a large fraction of
the ~36 M mid tris are occluded behind nearer crowns. Culling them cuts `nanMidRaster` **proportion-
ally** — a multiplier on top of A/B (or on the fused classifier in C).

- **Measured cost removed:** a proportional slice of the whole 25.8 % (however many tris are
  occluded).  **VRAM delta: 0** (tail-folded).  **world1 risk: small** (adds the triVis test to the
  classifier — a few HZB loads + compares, already coded).  **Parity/quality risk: MEDIUM** — it is
  *temporal* (prev-frame HZB), so fast camera motion can transiently reject disoccluded tris (1-frame
  holes). This is a *quality* risk, not bit-identity; the memory notes flag it "EXPERIMENT."
- **Gate/measurement (do this before sizing D):** turn `?trihzb=1` on, read the mid-queue count (its
  `[0]` counter) and the `?scar` covered-fragment counters vs off, to measure the actual cull
  fraction in the canonical dense-forest view. If it culls ≥ ~30-40 %, D is a top-2 lever; if small,
  drop it. Also A/B a fast pan for the temporal-hole artifact before shipping default-on.
- **Impl sketch:** flip the `triLvls` default (or a new default-on flag) once the cull fraction +
  artifact check pass; the kernel path already exists (`:1481-1541`, `kTriHzbCopy` `:2168`).

---

## 3. PROJECTING 3× — honest combination math

Occupancy alone caps ~1.8× (fixed useful work). So the 3× is a **stack**, led by A per the biggest
bucket:

| path | levers | projected | notes |
|---|---|---|---|
| conservative | A × B | ~1.6-1.9× | zero risk, no world1 change, no quality change |
| **+ occlusion cull** | A × B × D | **~2.2-2.9×** | reaches 3× iff D culls ≳40 % (dense forest likely) — **measure D first** |
| **fusion** | C (+ B) | **~2.5-3×** *and −160 MB* | highest ceiling; removes the pass; **gated on world1 ≤ ~60 regs / net frame** |

**Recommended sequence:** land **B** (free, no risk) and **A.1/A.3** (free, no parity risk)
immediately; **measure D's cull fraction** (one flag flip) — if ≥40 %, A×B×D clears 3× with no
world1 change; **in parallel prototype C** (one flag) and read world1's register count — if it holds
≤ ~60 with no occupancy cliff, C is the cleanest 3× *and* returns 160 MB to the memory arc. A.2
(incremental depth) is the reserve for the last stretch, behind its parity gate.

**No lever grows VRAM.** Two (C, D) spend ≤0; C returns 160 MB. This is the opposite of the retracted
record-growth approach.

---

## 4. DATA PROVENANCE

Ran `tools/profile/run_all.sh` on the **new** trace pair
(`…T02-03-38-c000.gputrace` raw + `…-perf.gputrace` embedded-perf export) →
`profile-results-20260708-021537/`. **Not blocked on the manual Xcode export — the perf data was
already embedded.** nanMidRaster's MSL cited throughout is `store0` **block 42** (extracted via
`_srcmap.load_store0_text` + `build_index`, keyed on `41943040`), because the `--src` pass declines
to auto-attach source for same-family raster siblings ("block alignment unverified"). Register/
occupancy figures (64 alloc / 44 %) are the Xcode-timeline counters relayed by the coordinator,
cross-checked against agc static (40 temp / **0 spill**) and the per-line runtime (45.5 % line-0) —
all agree: **register-capped occupancy, not memory spilling.** Per the standing rule, re-profile
after implementing any lever (one fresh capture → Xcode re-export with "Embed performance data" →
`run_all.sh`) to confirm the line-0 tax drops and occupancy rises.

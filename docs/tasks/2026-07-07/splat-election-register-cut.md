# Splat-Election Visbuffer — cutting `nanRasterWorld1` to the 40-register target

> **⚠️ SUPERSEDED 2026-07-07 by `PLAN-visbuffer-rewrite.md`.** The splat election here IS built and
> measured (`nanSplatElect` = 24 regs) and remains as ONE consumer (`Splat.ts`) of the new pipeline.
> But this plan's core premise — world1 keeps projecting inline (R2) — is exactly what the rewrite
> removes: projection moves to a per-vertex pre-pass, world1 becomes a lean classifier. Read the new
> plan for the current direction; keep this for the splat-election details + the register bisection.

**Task #76 (Apple/mobile-GPU arc). Status: DESIGN — validated by measurement, not yet built.**
**Date: 2026-07-07. Author context: a long M1-Max register-bisection session; this doc is the durable output.**

---

## 0. TL;DR / the decision

The SW cluster raster `nanRasterWorld1` is **occupancy-bound at 80 registers / ~40% occupancy** on M1 Max. Target is **40 registers (45 max)** — a near-halving.

We bisected the 80 registers empirically (see §2). The conclusion is structural, not a tuning problem:

- The register **peak is the fragment emit sitting on the edge-setup**, *downstream* of the vertex projection. Micro-cuts (planar depth, serpentine, killing the loop, a per-vertex pre-pass) each move ≤12 and **cannot** reach 40.
- **~16 registers are Dawn robustness** (`min(idx,len-1)` bounds clamps). This is a **public website** → the WebGPU spec forces robustness on → **that 16 is a permanent floor we cannot strip.** So the *logical* core must go 64 → 24.
- You cannot shrink a per-thread triangle rasterizer's core by 62%. **You have to stop rasterizing triangles for the sub-pixel majority.**

**The validated answer: a splat visbuffer.** 97% of the tris are sub-pixel = effectively points. Splat them. Measured floor of a precomputed-point splat election = **26 registers** — it clears 40 with headroom. The build moves the sub-pixel election into its own ~28-register kernel (§4–5).

**Honest caveat up front (§6):** a low register count is *necessary* but not *sufficient*. The frame-time win depends on the sub-pixel election being latency-bound (so higher occupancy hides it). That must be A/B-measured after building; if it isn't the bottleneck, this is a lateral move and we escalate (§8).

---

## 1. Goal & constraints

- **Goal:** `nanRasterWorld1` (the world single-pass SW cluster rasterizer) to ≤40 registers (≤45 absolute) on M1 Max, to roughly double occupancy from ~40%.
- **Ship target: public website.** ⇒ no control over Chrome/Dawn flags ⇒ **robustness cannot be disabled** in production. `--enable-dawn-features=disable_robustness` is a *diagnostic-only* lever (wired behind `NOROBUST=1` in `tools/profile/gputrace.sh`), NOT shippable.
- **User law (still in force):** no systemic knobs (rscale/DRS banned); perf buys DETAIL, no noise tricks; verify user-observable output; a different visbuffer scheme is explicitly **acceptable** if it reaches the target.
- **What renders where (important):** default world foliage is **voxels** (`NaniteVoxelRaster`, `naniteleaf` is opt-in). `?clhw=1` routes trunks + near-terrain to the **HW** rasterizer. So `world1` SW mesh handles the *residual* mesh (terrain + non-voxel mesh + naniteleaf crowns if enabled) — much of it sub-pixel. This optimization targets `world1` specifically; it does not touch the voxel foliage kernel.

---

## 2. The measurement journey (the empirical bisection)

All numbers are the Xcode-reported register count of `nanRasterWorld1`, captured via `tools/profile/gputrace.sh` on the canonical URL
(`?scene=world&nanite=1&dpr=2&nanodisp=1&clhw=1&clhwmax=32&profile=1&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1`).
Register count is a **static compile property** — capture duration (1s vs 3s) does not affect it.

| build / probe | what is isolated | registers | what it proved |
|---|---|---|---|
| base | full kernel | **80** | the wall |
| `?nanwind=0` | wind nodes removed (compile-time) | **80** | wind is **not** at the peak |
| `?relect=0` | entire election removed (compile-time) | **80** | the election is **not** at the peak |
| `?rdbg=1` | after makeCtx/broadcast (before vert fetch) | **96** | ⚠️ **SINK ARTIFACT** — its sink forces the whole broadcast ctx live at once, a moment that never happens in the real kernel. Ignore for peak analysis; only tells us the ctx live-set is large. |
| `?rdbg=5` (added this session) | after the 3-corner projection, before winding/edge-setup | **64** | the projection **floor** |
| `?rdbg=2` | after winding + edge-setup, before the scanline loop | **76** | edge-setup adds **+12** over projection |
| point-sample throwaway (loop deleted, `sx/sy` DCE'd) | emit with no loop at all | **80** | the loop/incremental scheme is **not** the peak — it overlaps the setup; deleting it changes nothing |
| `NOROBUST=1` (disable_robustness) | robustness clamps stripped | **64** | **robustness = 16 registers** (80−64). Public web ⇒ **forced floor.** |
| 1-point splat throwaway (project 1 vertex → elect, no triangle setup) | full triangle pipeline replaced by a single-point projection | **58** | a lean splat is *still* 58 ⇒ the mass is **`corner()` itself** = per-vertex fetch + **transform + wind** (`nanWindF16`, 9 params + instance-transform vecs), ~32 of the 58 |
| synthetic-election throwaway (no geometry, no `corner()`, no `vp` — px & depth are cheap functions of ids) | pure election + robustness floor | **26** | **the election is cheap.** A *precomputed*-point splat kernel (read point → elect) lands ~26–30 ⇒ **clears 40.** |

### The register profile through the kernel

```
launch + ctx-read  →  projection  →  edge-setup  →  emit (PEAK)
   (heavy, but        rises then      64 → 76        76 → 80
    consumed           falls to 64     (+12)          (+4)
    incrementally)     (rdbg=5)        (rdbg=2)       (base)
```

- The **peak (80) is at the emit**, on top of the edge-setup (76).
- The **projection (64) is a floor BELOW the peak.** By the emit, projection machinery (ctx, clip vec4, ndc, wind, instance transforms) is **dead** — only its 9-coord output (`xi/yi/dz`) survives into the peak.

---

## 3. Key structural insights (why the obvious levers fail)

1. **Killing the loop does nothing for registers (−0..−4).** `rdbg=2` (before loop) = 76, base = 80; a full point-sample throwaway = 80. The loop's `cw/rw/sx/sy` overlap the setup live-set and never rise above it. *(It IS a real ALU + warp-divergence win, though — see §9, that branch shipped.)*

2. **A per-vertex projection pre-pass does NOT cut the triangle-raster peak.** The peak (emit) lives *downstream* of projection and holds only the 9 projected coords, which are live whether they come from inline math or a buffer. Move projection out and the raster kernel still peaks at ~80. (Verified by the profile shape: projection floor 64 < emit peak 80.)

3. **Robustness is 16 and forced.** Public web ⇒ the `min(idx,len-1)` + length-load on every buffer fetch cannot be removed. Core must reach 24 for a 40 total.

4. **Why the splat scheme escapes all of the above:** it changes what is *downstream*. In a splat, the downstream is a tiny election (26 incl. robustness), not a heavy triangle emit. So — and only so — precomputing the point *does* help: the transform/wind (`corner()`, ~32) moves out of the hot kernel, and the hot kernel becomes read-point + elect ≈ 28.

5. **The election was never the problem; the transform/wind was.** `relect=0` = 80 (removing election off-peak), synthetic election = 26 (election is cheap), 1-point splat = 58 (transform/wind is the mass). The 58 − 26 = ~32 is `corner()`.

---

## 4. The validated structure — splat-election pipeline

Split the sub-pixel **election** out of the megakernel into its own low-register dispatch, fed by a queue that `world1` fills.

```
                    ┌─────────────────────────────────────────────┐
   world1 raster →  │ per (cluster,tri):                          │
   (still projects, │   ≤1px  → APPEND (px, cand, payload)        │→ splatQueue
    ~76-80 regs)    │           to splatQueue   (no inline elect)  │
                    │   2-16px → existing scanline loop (elects)   │
                    │   big / near-plane → HW queue (existing)     │
                    └─────────────────────────────────────────────┘
                                        │
   kSplatArgs (n = min(count, CAP))     │  indirect dispatch args
                                        ▼
   nanSplatElect  →  per record: read (px,cand,payload) → ELECT into visPayloadV/visBV
   (~28 regs, HIGH OCCUPANCY — the sub-pixel foliage election runs here)
                                        │
                                        ▼
   resolve  (UNCHANGED — reads the same visbuffer depth+id)
```

- **The hot kernel is `nanSplatElect` at ~28 registers.** The foliage-dominant sub-pixel election runs there at high occupancy; the atomic-election latency is hidden by more resident warps.
- **`world1`'s register count does NOT drop** (it still projects every tri and runs the loop for 2–16px). Its *workload* drops (it no longer elects sub-pixel fragments, only appends). See §6 for why this is still expected to be a net win and how to falsify it.
- **The visbuffer and resolve are unchanged** — same `(depth<<8|id)` election into `visPayloadV`, same full-id side store into `visBV`. This is purely a *where the election runs* change, so `id` still points at the triangle and the deferred resolve re-derives attributes exactly as today.

### Why "election in a separate kernel" and not "just a lower-register world1"

Register count = max live-set over the *whole* kernel. As long as the 2–16px loop path lives in the same kernel as the sub-pixel path, the kernel's count = the loop path's ~76–80 regardless of how lean the sub-pixel path is. The only way to give the sub-pixel election a low register count is to compile it as its **own dispatch** that never sees the loop/projection code. That is what `nanSplatElect` is.

---

## 5. Build plan (concrete, mirrors existing infra)

All five pieces mirror the **`hwQueue`** pattern already in `src/nanite/NaniteRaster.ts`. Line numbers are approximate (they will have drifted).

### Piece 1 — `splatQueue` buffer (mirror `hwQueue`, ~L558-568)
```ts
const SPLAT_CAP = 4_194_304;                       // records; tune vs memory (§6)
const splatQueueInit = new Uint32Array(1 + SPLAT_CAP * 3);   // [0]=count, then (px,cand,payload)*
const splatQueueAttr = new StorageBufferAttribute(splatQueueInit, 1);
splatQueueAttr.name = 'nanSplatQueue';
const splatQueueV = sU32Views(splatQueueAttr, 1 + SPLAT_CAP * 3);
const splatDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
splatDrawAttr.name = 'nanSplatDraw';
const splatDrawBuf = sU32Views(splatDrawAttr as unknown as StorageBufferAttribute, 4).rw;
```
⚠️ **Binding budget (§6):** `world1` is at the WebGPU **10-storage-buffer ceiling**. With `ctxsm=1` (the ctx pre-pass, which the canonical path runs) it sheds to **9**, so `splatQueue` fits as the 10th. **This build REQUIRES the ctx pre-pass path.** Confirm the live buffer count before adding; if already at 10, fold the splat counter into `hwQueue` spare space the way `scar` was folded (see the `SCAR_BASE` note ~L548-584).

### Piece 2 — clear in `kVisClear` (mirror the `hwQueue[0]` clear, ~L682/706)
```ts
atomicStore(splatQueueV.atomic.element(0), uint(0));
```

### Piece 3 — `world1` ≤1px branch: append instead of elect (~L1746, the point-sample branch already exists there)
The `mode==='world1' && !scar`, `coopMode===0`, `If(endX<=startX && endY<=startY)` branch already computes `px`, `cand`, `payload` and calls `doElection(px,cand,payload)`. **Replace that call** with an append:
```ts
const slot = atomicAdd(splatQueueV.atomic.element(0), uint(1)) as unknown as NU;
If(slot.lessThan(uint(SPLAT_CAP)), () => {
  const base = uint(1).add(slot.mul(uint(3)));
  atomicStore(splatQueueV.atomic.element(base),           px);
  atomicStore(splatQueueV.atomic.element(base.add(uint(1))), cand);
  atomicStore(splatQueueV.atomic.element(base.add(uint(2))), payload);
});
```
Note: `world1` still projects/derives `px/cand/payload` — that is fine; the register win is entirely in `nanSplatElect`, not in `world1`.

### Piece 4 — `kSplatArgs` indirect args (mirror `kHwArgs`, ~L2247)
```ts
const kSplatArgs = Fn(() => {
  const n = minU(aLoadU(splatQueueV.atomic.element(0)), uint(SPLAT_CAP));
  splatDrawBuf.element(0).assign(n);        // one thread per record (adjust for wg size, see below)
  splatDrawBuf.element(1).assign(uint(1));
  splatDrawBuf.element(2).assign(uint(0));
  splatDrawBuf.element(3).assign(uint(0));
})().compute(1, [1]);
(kSplatArgs as unknown as ComputeKernel).setName('nanSplatArgs');
```
⚠️ `splatDrawBuf.element(0)` is a **workgroup count**, not a thread count — divide `n` by the workgroup size (round up), OR dispatch one workgroup per record group. Match how `kHwArgs` feeds `hwRender` (`n.mul(3)` there is instance-count for a draw; here it is a compute dispatch, so it's `ceil(n / wgSize)`).

### Piece 5 — `nanSplatElect` kernel (inline the election — `doElection` is a closure *inside* `world1`'s `Fn`, so it cannot be shared; copy the `relect===1` body verbatim)
```ts
const kSplatElect = Fn(() => {
  const i = instanceIndex;                                  // 0..dispatch-1
  If(i.lessThan(minU(aLoadU(splatQueueV.atomic.element(0)), uint(SPLAT_CAP))), () => {
    const base = uint(1).add(i.mul(uint(3)));
    const px   = aLoadU(splatQueueV.atomic.element(base));
    const cand = aLoadU(splatQueueV.atomic.element(base.add(uint(1))));
    const pay  = aLoadU(splatQueueV.atomic.element(base.add(uint(2))));
    // verbatim relect===1 election (NaniteRaster.ts:784-793):
    const prevE = aLoadU(visPayloadV.atomic.element(px));
    If(cand.greaterThan(prevE), () => {
      const wonE = atomicMax(visPayloadV.atomic.element(px), cand) as unknown as NU;
      If(cand.greaterThan(wonE), () => {
        atomicStore(visBV.atomic.element(px), pay);
      });
    });
  });
})().compute(SPLAT_CAP, [64]);                              // real size comes from setIndirectDispatch
(kSplatElect as unknown as ComputeKernel).setName('nanSplatElect');
setIndirectDispatch(kSplatElect, splatDrawAttr);
```
Binds only `visPayloadV`, `visBV`, `splatQueue` = **3 storage buffers** — miles under the ceiling; that is why it can be lean.

### Piece 6 — batch wiring
Insert `kSplatArgs` then `kSplatElect` into the `world1` batch **after** the raster kernel and **before** resolve. Find the batch array (`batch: readonly unknown[]`, `dispatchBatchMixed`). Ordering rule: `kVisClear → (kClusterCtx) → raster → kSplatArgs → kSplatElect → kHwArgs → …`. The splat elections and `world1`'s 2–16px elections both `atomicMax` into `visPayloadV`; order between them is irrelevant (nearest wins by construction). But **`nanSplatElect` must complete before anything reads the visbuffer** (HZB build for next frame, resolve).

### Reference: the exact packing (do not drift)
- `payload = itemIdx << CLUSTER_TRI_BITS | localTri` (`CLUSTER_TRI_BITS = log2(MAX_CLUSTER_TRIS) = 7 @128`).
- `cand = depthKey24(cz) << 8 | (payload & 0xff)` — depth in the high 24 bits, low 8 id bits for the packed key; the full 25-bit id is the side-store `pay`.
- A depth24 tie ⇒ valid-but-maybe-wrong cluster (sparse wrong-material speckle), never torn — same race as today, unchanged.

---

## 6. Drawbacks, risks & open questions (read before building)

**R1 — Register win ≠ frame-time win (THE big one).** A 28-register `nanSplatElect` only helps if the sub-pixel election is **latency-bound** (atomic contention on `visPayloadV`), so more resident warps hide the stall. This is an assumption. **Falsify it first**: after building, A/B the frame time (`gpuWall`), not just the register count. ⚠️ Apple render‖compute overlap means per-pass timestamps are NOT additive — ablate, don't sum (memory `mobile-gpu-apple-arc`). If frame time doesn't improve, the election was not the bottleneck and this is a lateral move → §8.

**R2 — `world1` register count does not drop.** It still projects every tri and runs the 2–16px loop (~76–80 regs, ~40% occ). We only removed the sub-pixel *election* from it. If the **projection** of sub-pixel tris (at 40% occ) is the real bottleneck rather than the election, splitting the election buys little. Phase 2 (§8) leans the projection, but note the pre-pass caveat (§3.2) — it does not lower a *triangle* kernel; it only helps once the kernel is already a splat.

**R3 — Added work.** We ADD an atomic append (piece 3) and a kernel launch (piece 5). Net win only if `high-occ election` > `append + 40%-occ election`. The append is the risk:

**R4 — Append contention.** Millions of sub-pixel tris incrementing a **single** counter (`splatQueue[0]`) can serialize badly. `hwQueue` uses the same single-counter pattern but only for the *few* near/big tris. Mitigation if it contends: **workgroup-local compaction** — mirror swcoop's `list` + `count` shared arrays (a thread-0 scan compacts the workgroup's sub-pixel points, then **one** atomic bump per cluster instead of per tri, cutting contention ~128×). Start with the simple single-counter version and *measure* contention before adding this.

**R5 — Memory.** `splatQueue = 1 + SPLAT_CAP*3` u32. At `SPLAT_CAP = 4M` that's ~48 MB; at 8M, ~96 MB. Relevant to the MEM arc (task #65). A *dense* per-(cluster,tri) buffer would be `QRASTER_CAP(1M) × 128 × ...` ≈ 1 GB — **rejected**, must be a compacted queue. Size `SPLAT_CAP` to the realistic visible sub-pixel count; the `If(slot < SPLAT_CAP)` guard drops overflow (log it — silent truncation reads as "covered everything").

**R6 — Coverage / correctness of the ≤1px gate.** The branch fires only for **single-pixel-bbox** tris (`endX<=startX && endY<=startY`). A <1px tri straddling a pixel boundary (`endX==startX+1`) has a 2-pixel bbox → it goes to the **loop**, not the splat — correct, no holes. The point-sample of a single-pixel tri is **bit-identical** to the loop's single iteration (`rw` is already the edge value at that pixel-center; `uw = rw − bias`, same `cz`, same `px`, same election). Verify visually anyway (user law).

**R7 — Pass ordering / HZB.** Confirm nothing reads `visPayloadV`/`visBV` between `world1` and `nanSplatElect` that would observe un-elected sub-pixel pixels (HZB build for occlusion, resolve). If the HZB is built from this frame's visbuffer, `nanSplatElect` must precede it.

**R8 — Scope.** Only helps `world1` (mesh SW). Default foliage is voxels (different kernel). This is correct for the task (#76 targets `world1`) but means the visible dense-foliage win depends on whether the deployment uses `naniteleaf` or the residual mesh is the cost. Re-confirm `world1` is actually a frame-time cost worth this before/after.

**R9 — Two elections into one visbuffer.** `world1` (2–16px) and `nanSplatElect` (≤1px) both `atomicMax` `visPayloadV[px]`. Disjoint pixels usually; where they collide, `atomicMax` picks nearest regardless of order. Fine. But both must run before resolve.

---

## 7. Acceptance criteria (what to measure after building)

1. **`nanSplatElect` register count ≤ ~30** (Xcode). If > 40, the scheme failed its own premise — investigate (robustness on the 3 reads? the election dance? the workgroup-count math?).
2. **Visual parity** at `rdbg=0` default: the scene renders identically to pre-change (sub-pixel tris elect via the queue instead of inline). No holes, no wrong-material speckle beyond today's baseline.
3. **Frame-time A/B** (the real test): `gpuWall` eye/oblique/aerial, splat-path ON vs OFF, ablated (not summed). Must be **neutral-or-better**. This is the gate that decides whether the scheme ships.
4. **Append contention check**: if `gpuWall` regressed and a capture shows the append/counter hot → apply R4 workgroup-compaction.
5. **Overflow log**: how many sub-pixel tris exceeded `SPLAT_CAP` (should be ~0 at a sane cap).

---

## 8. Phase 2 & kill criteria

**If Phase 1 clears 40 registers but frame time is flat** (election wasn't the bottleneck): the honest read is that `world1`'s **projection at 40% occ** is the cost. Phase 2 = make the projection lean — but per §3.2 you cannot just pre-pass a *triangle* kernel. The move is: for the sub-pixel class, precompute the **splat point** (transform+wind+project the centroid) in a lean pass and have `world1` skip projection for those. This resurrects the memory-heavy point buffer and an atomic append at projection time — weigh carefully. The per-INSTANCE wind precompute (memory `sw-hw-per-cluster-raster`, "A′") is a prerequisite to make the projection itself cheap.

**Kill criteria (be honest, don't grind):** if `nanSplatElect` does not clear 40, OR frame time does not improve after Phase 1+2, OR append contention dominates irreducibly — then the SW-triangle structure **cannot** hit 40 on public web (robustness floor + irreducible core), and the correct pivot is: push more of `world1`'s geometry to the **HW rasterizer** (register-free), or accept the ~70–80 register ceiling and redirect the arc to the **34% warp inefficiency** / work-reduction, which the shipped point-sample branch (§9) already helps.

---

## 9. Current code state (what is already in `main`-branch working tree)

**Keepers (shipped this session, still in the tree):**
- **Per-size point-sample branch** — `world1`, `coopMode===0`, `mode==='world1' && !scar`, `If(≤1px)` point-samples (no loop), `.Else` runs `oldScanline()`. Bit-identical to the 1-iteration loop. **No register change** (peak is the setup) — it's a pure **ALU + warp-divergence** win for the sub-pixel majority. *This is the branch Piece 3 repurposes into an append.*
- **`?rdbg=5` probe** — returns right after the 3-corner projection (before winding/edge-setup); the diagnostic that isolated the projection floor (64). Toggle-able, harmless at `rdbg=0`.
- **`tools/profile/gputrace.sh` `NOROBUST=1`** — appends Dawn `disable_robustness` to measure the robustness floor. **Diagnostic only — never ship.**

**Reverted (throwaways, gone):** the ≤1px-only render throwaway, the 1-point splat throwaway, the synthetic-election throwaway.

---

## 10. Premise-audit note for whoever implements this

*(This repo's `CLAUDE.md` mandates it, and an implementer working only from this doc will not have inherited it.)*

Before concluding "this doesn't work" or "40 is unreachable," **go up a level and put the setup on trial**: is the metric right (register count of the *hot* kernel, not `world1`)? Is the frame-time actually gated by the election (R1)? Is the append the flaw rather than the scheme (R4)? A negative/disappointing result is the trigger to interrogate the premise, NOT to start varying the method. Only after the premise-audit clears the setup may you return "not-doable" — and then §8's kill criteria apply, honestly.

The whole reason this doc exists: we repeatedly entered with a flawed premise one level up (the loop is the peak; the projection is the peak; the pre-pass will help; robustness is strippable) and each was refuted by one measurement. **Measure the specific claim before building on it.**

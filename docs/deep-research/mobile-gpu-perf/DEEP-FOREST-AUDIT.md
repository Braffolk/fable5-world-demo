# DEEP-FOREST FULL-RENDERER AUDIT — Apple Silicon / mobile TBDR

**Consolidated action plan, 2026-07-05.** Target = the DEEP DENSE FOREST case (surrounded by
trees/leaves/voxel-crowns), M1 Max @ dpr2. This is **not** a grass report — grass is ~5ms and off in
the base numbers below. Source of truth = `deep-forest-breakdown.md` (honest gpuWall ablation), NOT the
per-pass timestamps (they OVERLAP — Apple runs render‖compute concurrently, so they are clues, not slabs).

---

## 0. Premise correction up front (the task brief's own framing was one level off)

The brief pointed at **"20.8ms base"** and flagged `c.nanVisClear ~11`, `r.half.mrt ~10.5`, `r.scene ~9.6`
as the whales. The ablation says otherwise:

- **The base is 17.4ms gpuWall, not 20.8ms.** The 20.8 is the `render` *pass timestamp*
  (`deep-forest-breakdown.md` render column = 20.84), which overlaps the concurrent 8.72ms compute stream.
  It is not additive and it is not the wall.
- **`c.nanVisClear ~11ms` (7.73 here) is NOT clear cost.** Memset math caps a 2×3.3M×4B zero-fill at
  <1ms; the timestamp is a concurrency artifact. **Proof in the ablation itself:** `nanVisClear` drops
  7.73 → **1.44** when the leaf mesh is removed (`naniteleaf=0`) and → 3.21 with `crownlod0=0`. The clear's
  timestamp window is swallowing the concurrently-running **leaf/crown SW-raster** work. The clear is a
  red herring; the leaf-mesh raster is the fish.
- **`r.half.mrt ~10.5ms` is POST, not shadows** (cloud march + GTAO + bounce, `src/render/HalfResMrt.ts:70`).
  The real shadow eval is `c.nanShadowHalf = 0.2ms`. Whole shadow system ablates to **0.5ms** static.

So the map moved. What actually generates the deep-forest frame is the **near/mid leaf-crown geometry**,
split across the compute (SW raster) and render (resolve) streams.

---

## 1. Deep-forest cost model (17.4ms gpuWall base, grass OFF)

Marginal costs are ablation deltas (`base − ablated`); shares overlap because Apple runs the compute and
render streams concurrently, so they do **not** sum to 17.4.

| Slab | Marginal (ablation) | Stream | What it is | Confidence |
|---|---|---|---|---|
| **Leaf mesh (`naniteleaf`)** | **4.1ms** (17.4→13.3) | compute (SW raster) | mid-band real-leaf triangle head | CONFIRMED |
| **Crown LOD0 mesh (`crownlod0`)** | **3.9ms** (17.4→13.5) | compute (SW raster) | near crowns rendered as full-detail LOD0 leaf tris | CONFIRMED |
| **↳ combined leaf-crown SW raster** | **≈8ms** (dominates the 8.72 compute stream) | compute | the whale — see §3 | CONFIRMED |
| Voxel scatter (`kVoxScatter`) | **net −54ms** vs mesh fallback; own fill ~3–6ms | compute | far/mid crown bricks | fill=SPECULATIVE, net=CONFIRMED |
| Deferred resolve (`r.scene`) | ~5.6 timestamp, overlaps; isolated est ~4–5 | render (fragment) | per-covered-pixel shade | SPECULATIVE (unablated) |
| Post (`r.half.mrt`) | 6.29 timestamp, overlaps; est ~5 rotated to contact/AO | render (fragment) | GTAO+contact+bounce+clouds+froxel | SPECULATIVE (unablated at this pose) |
| Cull BFS + HZB build | ~2–3ms | compute | not in any flagged pass ⇒ each <4ms | SPECULATIVE |
| Shadows (static) | **0.5ms** | compute | toroidal clip + half-res PCSS | CONFIRMED |
| Vis-buffer clear (real) | **<1ms** | compute | 7.73 timestamp is overlap, not clear | CONFIRMED |
| Orchestration (timestamp/readback) | ~0.5–1ms fixed | CPU/GPU | trackTimestamp ON in prod, meterRead | SPECULATIVE |
| Grass (when ON, not in base) | +~5ms | compute | secondary, see §7 | CONFIRMED |
| Shadows (MOVING, not in static base) | ~8ms p50 spike | compute | strip re-raster + shvox2 — the STUTTER | CONFIRMED |

**Reading of the model.** The compute stream (8.72ms) is almost entirely the **leaf-crown SW-raster**
(`crownlod0` removes 4.46 from compute, `naniteleaf` removes 5.57 — they overlap, so jointly they ≈ the
whole compute stream). The render stream (resolve + post) runs concurrently and is the other long pole.
gpuWall 17.4 < render-timestamp 20.84 because the streams overlap but do not fully hide each other. **Two
poles: leaf-mesh raster (compute) and resolve+post (render). Everything else is <3ms.**

---

## 2. THE SINGLE BIGGEST OPPORTUNITY

**The near/mid leaf-crown MESH-triangle SW raster — ~8ms, the whole compute stream.**
(`crownlod0` 3.9 + `naniteleaf` 4.1, `deep-forest-breakdown.md` lines 5–6.)

Why it is the whale, and the premise-audit one level up: the near/mid crowns render as **full-density
LOD0 leaf triangle meshes** (`NaniteCull.ts:1074`, user mandate: crowns are LOD0-or-voxel). The DAG does
**not** coarsen this band, so millions of sub-pixel/sliver leaf triangles pile overdraw onto the same
pixels — walked-pixels = overdraw, and the intrinsic per-walked-pixel loop is the cost
(`NaniteRaster.ts:1146-1286`). This is the *identical* disease documented for voxels
(memory `voxel-lod-only-two-levels-root-cause`) and base bark (`base-raster-is-the-bottleneck`): **a DAG
that never coarsens the near field.** Every *in-loop* raster lever (row-solve, election, coop, occupancy,
swmax) was already measured dead (`62d607e`) precisely because the generator is upstream in the DAG, not
in the pixel loop. `?rscale` (internal-res) would dissolve it but is **BANNED** by the quality bar.

The lever that survives the ban: **reduce leaf-crown walked-pixels without touching resolution** — either
(a) aggressively coarsen the near/mid leaf-crown DAG (shotdiff-gated), or (b) push the voxel near-band
(`voxnear`) inward so fewer crowns need the LOD0 mesh at all (voxels net **−54ms** — they are strictly
cheaper than the mesh they replace, the whole reason removing them explodes the frame to 71.4ms). The
tension is the beautification finding that voxels looked bad up close (offset bricks, one-square collapse),
fixed in `6a93dfb` — so a fresh shotdiff A/B of `voxnear` pushed in is now warranted and is the highest-
leverage single experiment in this arc. Halving this slab moves the frame ~4ms — more than any other lever.

---

## 3. TOP LEVERS, RANKED BY EXPECTED DEEP-FOREST IMPACT

Ranked by expected wall-time impact at the deep-forest pose, not by ease. Each is the change + file:line +
source + WebGPU-feasibility + expected impact + confidence + first measurement.

### L1 — Push the voxel near-band inward / coarsen the leaf-crown DAG (attack the ~8ms whale)
- **Change:** shotdiff-gated A/B of `voxnear` pushed inward so near crowns become voxel bricks (−54ms/crown-
  class cheaper) instead of LOD0 leaf mesh; and/or a real near/mid coarsening level on the leaf-crown
  aggregate DAG so sub-pixel leaf tris shed before they flood the pixel loop.
- **Where:** `NaniteCull.ts:1074` (crownlod0 LOD0-or-descend), `NaniteFrame.ts:212` (instMinPx),
  `WorldRegistry.ts:596` (crown aggregate = LOD0-only); raster loop `NaniteRaster.ts:1146-1286`.
- **Source:** Karis SIGGRAPH-2021 (overdraw from overlapping leaf aggregates is why Nanite underperforms on
  foliage); memory `voxel-lod-only-two-levels-root-cause`, `base-raster-is-the-bottleneck`.
- **WebGPU:** yes (cull/DAG params + shotdiff gate).
- **Impact:** HIGH (up to ~4ms). **Confidence:** CONFIRMED the slab is real & the biggest; SPECULATIVE that
  coarsening holds the quality bar (the crux — must be shotdiff-gated).
- **First measure:** `crownlod0` shotdiff at shipped density vs `voxnear` pushed in; gpuWall + shotdiff at
  the deep pose, `freeze=1`.

### L2 — Flip the forest resolve to TWO-PASS (`respass=0` default in foliage frames)
- **Change:** in forest (`qVoxRasterRO && !gi && csm===null`) the resolve defaults to single-pass `'both'`
  (`NaniteResolve.ts:1574-1591`), which compiles terrain-derivative + rock + bark + leaf + grass subgraphs
  **into the voxel shader**. TBDR occupancy is set by the worst-case register footprint across *all*
  branches, so the >90%-of-pixels voxel path runs at collapsed occupancy and the long dependent voxel-decode
  fetch chain has no second wavefront to hide behind. Two-pass strips those subgraphs from the vox shader.
- **Where:** `NaniteResolve.ts:1574-1591`; the code itself flags the terrain-derivative merge as *"the
  dominant driver of the close-up voxel r.scene cliff (37.5ms inside a crown)"* (`:556-561`).
- **Source:** occupancy-interplayoflight.md:21-27 (VGPR↔occupancy; WGSL allocates for the whole function);
  CROSS-CHECK F3 (Andersson/DICE übershader worst-case VGPR).
- **WebGPU:** yes — it is a **default flip of an existing flag**, identity already proven.
- **Impact:** HIGH (resolve is a ~5ms pole; occupancy recovery on the dominant tier). **Confidence:**
  CONFIDENT (mechanism + in-code cliff note + cheap to test).
- **First measure:** `?respass=0` vs default at the deep pose, dpr2, grass off, honest gpuWall + shotdiff.

### L3 — Ping-pong (double-buffer) the vis buffers to restore render‖compute overlap
- **Change:** the compute clear+raster of frame N+1 write the same `payloadV`/`visBV` that frame N's
  fragment resolve still reads → cross-queue WAR hazard forces Apple to serialize, destroying the
  render‖compute overlap the arch is built for. This is *why* `nanVisClear`'s timestamp balloons and
  tracks the leaf-raster cost. Allocate 2× vis sets, index by frame parity.
- **Where:** `NaniteRaster.ts:203-218` (`makeVisBuffers`, single instance), `NaniteFrame.ts:219`,
  `NaniteResolve.ts:424-425` (resolve reads next frame's clear target).
- **Source:** apple-tbdr-blakecrosley.md:10,31 (overlap render+compute on one timeline; honoring it needs
  no shared-buffer cross-queue hazard).
- **WebGPU:** yes (+~26MB, trivial vs the 6GB heap).
- **Impact:** HIGH-if-real (could recover a chunk of the serialization gap between the 8.72 compute and the
  render stream). **Confidence:** SPECULATIVE — the ablation confirms the *artifact* exists (nanVisClear
  tracks leaf raster) but not that ping-pong recovers wall time.
- **First measure:** two-step — (a) build-time no-op the two hot clears, A/B gpuWall (tiny delta ⇒ clear is
  cheap, confirms artifact); (b) implement `?visping`, A/B gpuWall (large drop ⇒ WAR stall was real).

### L4 — Split the voxel megakernel: a LEAN near-scatter kernel for the dense L0 crowns
- **Change:** `kVoxScatter` is one `Fn` compiling 5 default-on paths (occl pyramid, per-brick occl, 512-
  projection mask build, ray+DDA, wind). In deep forest near crowns take the cheap flat path but every lane
  still allocates registers for the ray/mask/pyramid code (WGSL allocates for the union of branches), so
  Apple **spills** (its documented policy: spill rather than drop ALU utilization). Emit a near-kernel with
  the ray/mask closures build-time gated OFF; keep the ray kernel for far coarse bricks.
- **Where:** `NaniteVoxelRaster.ts:571-1564` (megakernel), ray basis `:1230-1268`, mask build `:1091-1191`.
- **Source:** occupancy-interplayoflight.md:21-27; metal-benchmarks-README.md:208 (Apple spills to device
  rather than cut ALU utilization); the resolve already fixed this exact disease (`NaniteResolve.ts:556-561`).
- **WebGPU:** yes (two `Fn` instances, same buffers).
- **Impact:** HIGH-if-register-bound. **Confidence:** SPECULATIVE — gated on a Metal capture showing
  spilled-bytes>0 / low occupancy (needs the user's Xcode).
- **First measure:** `?voxrdbg=2` (`NaniteVoxelRaster.ts:246,1561`) → `base − voxrdbg2` gpuWall = the pure
  voxel fill share; then Xcode GPU capture of `kVoxScatter` occupancy% + spilled bytes. **Ship nothing
  register-related before the capture.**

### L5 — Post: move contact shadows to half-res + enable temporal GTAO reuse
- **Change:** contact shadows march full-res 12 steps on every pixel with a trunk/leaf <240m — i.e. **every
  pixel** in deep forest (nothing skips), the single post cost that hits its worst case exactly here. Move
  into the existing half-res MRT + bilateral-upsample (reuse `aoFaded`'s depth guide). Separately, GTAO
  temporal reuse is OFF (`Gtao.ts:11-12`); in forest all geometry is inside the 90m full-AO band so the
  far-fade skip saves nothing and the full 6-sample horizon march runs everywhere — rotate the slice angle
  per frame + reproject via the existing `velReproject` history to halve samples (6→3) at equal quality.
- **Where:** contact `PostStack.ts:451-504` (`SSCS_STEPS=12`); GTAO `Gtao.ts:11-12,153`, `PostStack.ts:224-234`.
- **Source:** cloudscapes-toft2016.md:43-52 (half-res + reproject; accumulate few-sample over frames);
  occupancy-interplayoflight.md (right frequency).
- **WebGPU:** yes (contact half-res); partial (temporal AO — reprojection exists, needs history + clamp).
- **Impact:** MEDIUM-HIGH (post rotates toward contact+AO in forest; contact quarters its march invocations).
  **Confidence:** CONFIDENT (contact half-res); SPECULATIVE (AO temporal — disocclusion clamp needed).
- **First measure:** `ablate=contact` / `ablate=ao` one-at-a-time A/B/A at the deep pose, gpuWall delta —
  confirm they dominate the post share before building. **Post has NOT been ablated at the forest pose.**

### L6 — Shadows: tame the MOVING strip re-raster + `shvox2` crown caster (the STUTTER)
- **Change:** static shadows are 0.5ms — do **not** touch the PCSS path. The real cost is motion-gated:
  strip re-raster ≈ **8ms p50 walking** vs 0.8ms still, and `shvox2` re-scatters crown bricks into every
  re-rastered level (seeded at ALL distances). Temporally stagger far-level re-raster (1 coarse level/frame),
  cap per-frame re-raster count, and restrict `seedVoxAllDist` to far levels (near crowns already cast via
  leaf/tri depth). This attacks the documented **<30fps + stutter** symptom directly.
- **Where:** `NaniteShadowClip.ts:1203` (rasterLevels), `:1257` (run gating), `:649` (shVox2 default-ON;
  the `:648` "DEFAULT OFF" comment is STALE), `:493` (seedVoxAllDist).
- **Source:** `docs/perf-runs/2026-07-04-90fps-arc.md:68`; shadow-arc.md:156; Karis/Epic foliage-shadow
  guidance (small casters → cheaper cadence).
- **WebGPU:** yes (dispatch cadence gating in `run()`).
- **Impact:** MEDIUM-HIGH on p95/stutter (near-zero on static avg). **Confidence:** CONFIRMED the moving
  cost is real; SPECULATIVE which cadence lever pays most.
- **First measure:** `?nanshadow=0` and `?shvox2=0` at a **MOVING** deep pose (static ablation cannot see
  this) — split re-raster vs voxel-caster.

### L7 — Shadow half-res upsample: pack the rg32f target to rg16f/packed
- **Change:** the bilateral upsample does 4 `textureLoad` taps on an rg32f (FP32) target per covered pixel,
  in the resolve. Shadow needs ~8 bits, camDist fits fp16-range; pack to rg16f/unorm16 → halves the per-tap
  format cost (Mali FP32 = 2×) and the copy bandwidth.
- **Where:** `NaniteShadowHalf.ts:83-85` (FloatType+RGFormat), `:168-181` (4 taps), consumed at
  `NaniteResolve.ts:1305-1306`; shadow depth `NaniteShadowClip.ts:321` (r32float, z_g already ∈[0,1]).
- **Source:** arm-mali-guide.md:90 (FP32 texture = 2× cost; use packed 32-bit formats).
- **WebGPU:** yes (StorageTexture format is a TS choice, not shader-f16).
- **Impact:** LOW-MEDIUM. **Confidence:** SPECULATIVE (verify bilateral tolerance after quantizing camDist).
- **First measure:** A/B gpuWall at the deep pose after the format swap.

### L8 — Gate `trackTimestamp` behind a profiling flag (free prod win)
- **Change:** `trackTimestamp:true` is hardcoded ON in production (`Engine.ts:86`) and timestamps are
  resolved **every frame** (`Engine.ts:231-251`) — ~12–18 query-pair writes + 2 resolve/copy/mapAsync per
  frame, paid where nobody reads them. Gate behind `?gpuprof` and resolve on a cadence.
- **Where:** `Engine.ts:86,231-251`; live `meterRead` `NaniteFrame.ts:667-694` (~7 readbacks every 15th
  frame — a p95/stutter candidate; stagger to 1/frame or gate on HUD visibility).
- **Source:** threejsroadmap-profiling-webgpu-shaders.md (trackTimestamp is opt-in with a cost;
  readback jitter); reac2023 Aaltonen (no per-frame staging round-trips).
- **WebGPU:** yes (constructor flag + resolve cadence, no node_modules edit).
- **Impact:** LOW-MEDIUM (~0.5–1ms fixed + p95 relief). **Confidence:** SPECULATIVE (unmeasured).
- **First measure:** boot with trackTimestamp forced off vs on, diff gpuWall; A/B live `meterRead` off,
  watch p95 at the deep pose.

---

## 4. WHAT WE ALREADY DO RIGHT — do not waste effort here

- **32-bit split depth-key election + relaxed-load pre-gate + no 3rd atomic buffer** (`NaniteRaster.ts:442,595-607`;
  `NaniteVoxelRaster.ts:1343-1361`). WGSL has no atomic<u64>; M1 has no 64-bit min/max. This is the *only*
  correct Apple port; a 3rd hot atomic was measured to triple cost. **Guard-rail: never add one.**
- **SW compute raster + two-sided leaf re-wind** halves leaf triangles (`NaniteRaster.ts:954-956`). Tellusim:
  M1 compute-raster 2.30B tri/s > HW 1.37B. Keep it; do not move mesh raster to HW.
- **Micro-poly / sub-pixel setup is FREE** (`rdbg2−rdbg1≈0`). Do NOT chase tighter primitive culling to cut
  the pixel loop — setup is not the cost, walked-pixels are.
- **Voxel min-pooled occlusion pyramid + per-block/brick cull** (`NaniteVoxelRaster.ts:434-557,730-789`;
  `voxbocc` measured 36.1→18.9ms) — this is *why* removing voxels regresses +54ms. **Do not strip.**
- **Per-brick reciprocal for the int div+mod** (`NaniteVoxelRaster.ts:1334-1337`, −8.8ms). Done.
- **`depthV` clear already skipped in the world path** (`NaniteRaster.ts:524-529`). 2-buffer hot set. Keep.
- **Single-phase prev-HZB occlusion at emit + coalesced cull/HZB submit** (`NaniteCull.ts:1142-1150`;
  `NaniteFrame.ts:582-636`; `NaniteHzb.ts:229-234`). Do not port Nanite's two-phase re-cull — the second
  traverse+raster costs more than the overdraw it saves on TBDR.
- **Resolve stays a fullscreen FRAGMENT pass** (`NaniteResolve.ts:383-385`) — keeps DCC/tile-memory
  compression (Apple-optimal). Any material-binning (§5) MUST stay fragment-side. Do NOT move to compute.
- **Forest `gi=null` elides the whole probe subgraph**; **`reskeep` corner-only CSM sample**; **half-res
  shadow eval + manual compare (no HW-compare 2×) + NearestFilter** — static shadow is 0.2ms, at its floor.
- **Hot scalars/matrices in UBOs, bulk in storage** (`Tsl.ts:357-383`) — the correct mobile split.

---

## 5. BLOCKED BY WEBGPU (and the second-best reachable substitute)

- **fp16 in the resolve / voxel / bark ALU chains** — would shrink the worst-case register footprint →
  raise occupancy → hide the voxel-decode fetch latency (the same axis as L2/L4). **BLOCKED:** three r184
  TSL has no half node and the device lacks `shader-f16`. Second-best (reachable, same occupancy goal):
  **structurally split the shader** (L2 two-pass, L4 kernel split) to shrink the live-register set instead
  of numerically. This is the actionable form.
- **Nanite material-ID-as-depth binning** (shade each material class in its own tight fragment pass with a
  depth-EQUALS reject; 64×64 NaN-tile cull) — the full form of L2. Reachable via material-ID→depth +
  `depthCompare:'equal'` + a small tile-classification compute pass. The **wave-intrinsic 32-bit-mask** path
  is NOT reachable (no subgroup ops at our layer) — use the portable 64×64 tile path. Second-best after L2.
- **`clearBuffer` DMA zero-fill** instead of the atomic compute-clear — three r184 WebGPUBackend does not
  cleanly expose the raw `GPUCommandEncoder` mid-frame. Low priority anyway (the clear is <1ms; L3 ping-pong
  is the real clear-adjacent lever). Second-best: leave the compute clear, fix the WAR hazard via L3.
- **Imageblocks / tile shaders / programmable blending / memoryless / ROV** — Metal-only, not in WGSL.
  Our vis-buffer + deferred-fragment resolve is the correct WebGPU substitute and already captures the DCC
  win these would provide.
- **Cache-line-spaced / L2-amortised atomics** (Mali trick) — target is a screen pixel (4B apart), cannot
  pre-combine. Un-actionable AND measured minor (`?relect`). Do not re-stage threadgroup-atomic election.

---

## 6. INSTRUMENT-FIRST SEQUENCE

The static-base ablation is done (`deep-forest-breakdown.md`). Remaining measurements, in order — all at the
deep pose (x=-852,z=228,y=+2,yaw0.9,pitch-0.08), dpr2, grass off, honest gpuWall, same-session A/B/A
(cross-boot contamination law):

1. **L2 first — cheapest high-confidence:** `?respass=0` vs default. One flag, identity proven. If it wins,
   flip the forest default. (No Xcode needed.)
2. **L3 clear/ping-pong disambiguation:** build-time no-op the two hot clears (A/B), then `?visping` (A/B).
   Separates the artifact from a real WAR stall.
3. **Post forest ablation (never run at this pose):** `ablate=contact` / `ablate=ao` / `ablate=bounce` /
   `ablate=clouds` / `ablate=froxels` one-at-a-time. Confirms the post share rotates to contact+AO (predicted)
   before building L5. Expect clouds ≈ free in forest (tree hits are nearer than the 1250m slab).
4. **Voxel fill isolation:** `?voxrdbg=2` → `base − voxrdbg2` = the pure kVoxScatter fill share.
5. **MOVING pose leg for L6:** `?nanshadow=0` and `?shvox2=0` while walking — the static ablation cannot see
   the 8ms strip re-raster.
6. **Metal capture (needs the user's Xcode — gates L4 and any register work):** Xcode "Capture GPU Frame" or
   Instruments on the Dawn/Chrome GPU process → `kVoxScatter` and the `'both'` resolve fragment shader
   occupancy% + spilled-bytes + top limiter. **Decision rule:** spilling / low occupancy ⇒ ship the kernel
   split (L4) and confirm L2; healthy occupancy but slow ⇒ latency-bound ⇒ double down on occlusion culls
   and cut the voxel-decode dependent-fetch chain instead.
7. **L1 (the whale) is last to *build* but the experiment is cheap:** `voxnear` pushed in + `crownlod0`
   shotdiff at shipped density. It is the biggest slab; it is also the most quality-fragile, so it rides on a
   shotdiff gate, not a gpuWall gate alone.

---

## 7. SECONDARY: grass (~5ms, real but not the whale)

Grass is off in the base above; folding the `MASTER-AUDIT.md` levers as a secondary track (the near-band
procedural raymarch S7 is the shipped lane). Apply the *same* premise: grass cost is the per-step raymarch
field eval, not overdraw. It is a ~5ms lever worth ~1–2ms after the leaf-mesh whale, resolve, and post are
addressed. Do not let it reorder the deep-forest priority — the 8ms leaf-crown mesh + the resolve/post poles
come first.

---

*Every lever above is grounded in BOTH a source AND our code (file:line) and tagged CONFIRMED/SPECULATIVE.
The frame is 17.4ms gpuWall, not 20.8ms; the whale is the leaf-crown MESH raster, not the vis clear.*

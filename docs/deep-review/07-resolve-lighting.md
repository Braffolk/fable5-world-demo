# 07 — Resolve + lighting (NaniteResolve.ts): measured ~FREE, and why

Scope: the two fullscreen resolve passes (`src/nanite/NaniteResolve.ts`, 1108 lines), the fetch
helpers they use (`src/nanite/NaniteFetch.ts`, `src/nanite/Tsl.ts`, `src/nanite/VoxelBrick.ts`),
and the manual lighting chain (sun + wrap + PCSS shadow + probe GI + ambient floor + backlight).
Premise audit read first (docs/deep-review/90-premise-audit.md); every number below is from the
2026-07-02 table or docs/perf-runs/2026-07-02-attribution-and-waves.md.

## 1. TL;DR

- Resolve shading + lighting + shadows + GI are measured ~FREE (each ablation ≤ apparatus floor):
  `nandbg=flat` 29.9/39.3/17.6 vs 31.2/38.8/14.6; `nanshadow=0` 32.6/37.0/15.6; `leafcheap=all`
  35.2/42.4/16.3 vs 36.1/43.2/16.8. There is NO meaningful ms pool in this slice. Do not rebuild
  shade-binning.
- One Class-I lever exists and is already built: `?reskeep=0` (corner-only CSM keep sample),
  −3.9ms on general vistas, expected ~0–1ms forest — flip default after a within-boot
  `setKeepFull` A/B (probe R1).
- Free Class-Q polish is available (bead-v2 crown anchor / per-species tilt / far-tile bead) —
  quality UP at ~0ms.
- Main value of this doc: the freeness mechanism (§3), the budget rule for future resolve
  branches (§5), and the landmine list (§6) so water/grass don't silently end the free era.

## 2. How it works today

**Two fullscreen passes, Discard-partitioned.** `buildMat(pass)` (NaniteResolve.ts:320) builds the
same fragment graph twice: the `tri` mesh (renderOrder −1000, NaniteResolve.ts:1085–1088) shades
triangle pixels and Discards vox winners; the `vox` mesh (renderOrder −999, built only when the
voxel queue exists, NaniteResolve.ts:1097–1105) shades only bit31 voxel winners
(NaniteResolve.ts:379–391). The split exists for the Metal 10-storage-buffer cliff: tri binds 8
buffers, vox binds 7 (NaniteResolve.ts:305–319). Both passes: read `visPayloadV` + `visBV` at the
pixel, Discard uncovered (elect==0, NaniteResolve.ts:359–363), reconstruct world position from the
24-bit election key (`cz = 1 − (key>>8)/2^24`, NaniteResolve.ts:365–371 per the PERF-VB4 comment at
292–295), and write that same reconstructed depth via `depthNode` (NaniteResolve.ts:1071–1078) so
the sky/scene composite works.

**Material mux (tri pass).** matClass from mesh word 6 (NaniteResolve.ts:414–421); branches are
build-time-guarded with `if (pass === 'tri')` so their buffers never enter the vox binding set:
terrain (`buildTerrainShading` + caustics, NaniteResolve.ts:446–472), rock (triangle re-fetch +
barycentric vdata, 478–508), bark/deadwood (UV + tangent frame + analytic mip LOD at 574–578, moss
and normal-map distance-gated by `?resfar=60` at 634–660, per-instance tint via `slotHash` at
669–676), leaf (full path = `makeCtx` + 3-vertex fetch, 703–736; cheap far path beyond
`resfar·0.6 ≈ 36m` = single-vertex quad normal + species tint, 737–754). `makeCtx`
(NaniteFetch.ts:147–290) does the per-instance wind precompute — 4 gust/exposure texture reads
(NaniteFetch.ts:206–208, 236–238) + hoisted sway sines (256–257) — per PIXEL in the resolve's
full-leaf branch.

**Vox branch (vox pass).** Winning brick index from payload bits 21–27 (`?voxbn`,
NaniteResolve.ts:788–791), baked brick mean normal (word 2, VoxelBrick.ts:68) decoded + yaw-rotated
+ camera-flipped (793–833), bead-v2 crown field (crownDir = horizontal radial from tree origin
`A.xyz` + fixed 0.55 up-tilt, 75/25 blended with the per-pixel bead, 803–830), per-brick baked
albedo (word 4, 838–848), world-anchored value jitter `?voxjit` (860–868).

**Lighting (shared tail).** Sun lambert; vox pixels take wrapped N·L (·0.5+0.5, floor 0.25·0.9,
NaniteResolve.ts:908–919); shadow factor from half-res PCSS + bilateral upsample
(`world.shadowHalf.upsample`, 933–935); the three-CSM `keep` factor (≡1, empty maps) multiplied in
per pixel when `keepFullU=1` — the default (286–291, 946–954); probe-GI irradiance + canopy
darkening (973–984); ambient floor with vox up-bias 0.6 (992–1001); leaf/vox translucent backlight
(1006–1020). `receivedShadowPositionNode` does a SECOND per-pixel wp reconstruction purely to feed
three's CSM cascade select (330–350).

## 3. Why it is free (so nobody rebuilds shade-binning)

1. **Exactly one shading invocation per covered pixel, zero geometry scaling.** The vis-buffer
   resolve has no overdraw and never touches per-triangle work — 4.4M visTris at eye cost nothing
   here; the pass is ~3.34M fragment invocations regardless of scene. All geometry cost was paid
   upstream in election (the actual bottleneck, other readers' slices).
2. **The most expensive branch is measured <1ms in aggregate.** The full-leaf path is the
   worst-case decode in the file: per-pixel `makeCtx` (4 texture reads + ~20 storage words), 3
   world-vertex fetches WITH wind, barycentrics, per-vertex vdata. `leafcheap=all` replaces it
   with the cheapest candidate look on EVERY mesh-leaf pixel and moved −0.9/−0.8/−0.5
   (35.2/42.4/16.3 vs 36.1/43.2/16.8). That is the measured ceiling of ANY leaf-decode
   optimization, shade-binning included.
3. **The whole lighting+GI expression chain is inside thermal noise** (`nandbg=flat` −1.3 eye,
   +0.5 oblique) and **the whole shadow system is 0–2ms** (`nanshadow=0`). Note flat does NOT
   ablate the shadow statements (the `sf` If/assign block at 946–954 is a stack statement,
   emitted regardless of the early return at 1023) — the two flags together cover the tail.
4. **Coherence by construction.** The vox/tri material split is two PASSES, so there is zero
   intra-wave divergence between tiers at band boundaries — the only cost of the partition is
   both passes running a 2-load prologue on every pixel (~4 u32 loads/px ≈ 53MB ≈ ~0.15ms at
   ~400GB/s; bounded above by the measured nulls). Within the tri pass, matClass regions are
   contiguous blobs (crowns, trunks, ground); divergent waves exist only along silhouettes, and
   the measured <1ms ceiling on the entire full-vs-cheap leaf delta bounds any divergence saving
   far below the 2ms apparatus floor (premise audit §3.7).
5. **What UE5 does and why it doesn't transfer:** UE5 bins pixels by shading bin and dispatches
   one compute per material (NaniteShadeBinning.usf) because its materials are arbitrary
   user-authored graphs, hundreds of instructions, incoherent material IDs. Here there are six
   fixed cheap materials with blob coherence and a measured-free tail. Binning solves a problem
   this renderer does not have; the 2026-06-26 review already killed it ("shade-binning (no wave
   ops)", docs/perf-runs/2026-06-26-voxel-lod-session-state.md §6h KILLED list).

## 4. Waste map (small; each item bounded by measurement)

| # | waste | mechanism | cost model | bound |
|---|---|---|---|---|
| W1 | CSM `keep` sample ≡1 multiplied per pixel (default `keepFullU=1`) | NaniteResolve.ts:286–291, 946–954 | full-res cascade-select + PCF taps × covered px | −3.9ms on depth-diverse general vistas, "~0 forest" (2026-06-26-voxel-lod-session-state.md:235–239); ≤ the 0–2ms `nanshadow=0` delta |
| W2 | second wp reconstruction for `receivedShadowPositionNode` | NaniteResolve.ts:330–350 | ~10 ALU + 1 storage load × px | dies automatically with W1 (comment at 280–282) |
| W3 | two-pass prologue duplication (both passes load elect+visBV on all px) | NaniteResolve.ts:359–391 | ~4 u32 loads × 3.34Mpx | ~0.15ms; it IS the storage-cliff solution — keep |
| W4 | per-pixel `makeCtx` wind precompute on near full-leaf px | NaniteFetch.ts:202–257 via NaniteResolve.ts:704 | 4 tex + ~20 storage words × near-leaf px | <1ms total (leafcheap ceiling); already gated to <36m |
| W5 | terrain subgraph (~14 tex samples) on terrain px | NaniteResolve.ts:446–472 | part of the base pool (~10.3 oblique, premise audit §2.4) | base reader's slice, not this doc's |

Sum of what this slice can recover at zero quality loss: roughly W1+W2 ≈ 0–1ms forest. That is the
honest whole pool.

## 5. Budget: when does the resolve STOP being free? (water / grass / future branches)

Rule set for anyone adding a matClass branch:

- **Per-pixel budget.** Today's decode sits at ≲0.3ms/Mpx (leafcheap bound). A new branch covering
  fraction f of 3.34Mpx with cost c ms/Mpx adds f·c·3.34ms. Stay ≤ the leaf-full envelope (~10
  storage/texture reads + ~200 ALU per pixel) and it stays under the 2ms measurement floor. Ship a
  `leafcheap`-style ceiling flag WITH the feature so its ceiling is measurable on day one; >2ms
  measured = the branch is not free, gets its own pool line in the master plan.
- **Storage-buffer headroom.** tri pass 8/10, vox pass 7/10 (NaniteResolve.ts:305–319). Water/grass
  buffers must fit the headroom or become a THIRD Discard-partitioned fullscreen pass (proven
  pattern, ~0.15ms prologue) — never the 11th buffer (silent pipeline kill, premise audit §3.9).
- **Occupancy landmine (the one real cliff in this file's history).** Including
  `buildTerrainShading` in the vox pass collapsed occupancy — its implicit-derivative `texture()`
  samples are demote-forcing and the subgraph's register pressure broke latency hiding: measured
  37.5ms r.scene inside a crown until build-time-guarded (NaniteResolve.ts:439–445). Rules: fat
  subgraphs behind `if (pass === ...)` build-time guards (a runtime `If()` still BINDS and still
  inflates the shader); explicit `.level()/.grad()` sampling only in the vox pass.
- **Partition gating.** In the vox pass, fall-through must gate on `isV` (bit31), never on
  matClass — a garbage visBV id decoding matClass≠7 on a real vox pixel is what produced the gray
  slabs (NaniteResolve.ts:770–778, 872–878). New classes must keep the tri/vox Discard partition
  exhaustive and mutually exclusive.
- **Bit-identity coupling.** The resolve's `makeFetch` must keep flag parity with the raster
  (`nanwind` read by both, NaniteResolve.ts:221–227) — a future branch that re-fetches positions
  with mismatched wind gets barycentrics off the rasterized surface. And resolve depth is the
  24-bit election key, not HW depth (NaniteResolve.ts:1071–1078); depth-reading features (water
  refraction) must reconstruct identically.

## 6. Levers (ranked)

**L1 — flip `?reskeep=0` to default (corner-only keep sample; W1+W2).**
Mechanism: `keepFullU=0` samples three's CSM keep only at pixel [0,0] (keeps the cascade-fit node
alive for NaniteShadow.run); every real pixel skips a full-res PCSS/cascade-select whose result is
≡1 (NaniteResolve.ts:939–954). Files: src/nanite/NaniteResolve.ts:286 (default), possibly
src/nanite/NaniteFrame.ts:361–365 (A/B hook already exists). Expected ms: eye ~0–0.5, oblique
~0–1, aerial ~0, live: helps most in depth-diverse general vistas (−3.9ms measured 2026-06-26) —
i.e. it is a LIVE/general win more than a canonical-pose win. Quality: **IDENTICAL** (keep≡1 ⇒
`sf·1.0` bit-exact; supporting evidence: prior shotdiff ≈ TAA floor, commit 63fc29b — note that
shotdiff had jitter UNPINNED and a nonzero floor, so it does NOT by itself certify Class I).
Gate (must conform to premise audit §4 Class I): probe R1 (within-boot `setKeepFull` A/B) + a
fresh shotdiff at **maxDiff=0** with TAA jitter index pinned, same seed/time-of-day, at the 3
canonical poses + 2 stress poses (grazing tree-line silhouette, dense mid-ring) + the fast-motion
spot-check the 2026-06-26 doc left pending (cascade fit is driven explicitly per
NaniteResolve.ts:283–285; identity under motion rests on that explicit drive matching). Effort: S.
UE5 n/a (no vestigial second shadow system to skip).

**L2 — bead-v2 crown-field polish (quality UP, ~0ms).** Class **IMPROVING**, effort S each,
gate = side-by-side crops + user sign-off (premise audit §4 Class Q):
  a) *Crown-height-aware anchor:* crownDir anchors at tree origin `A.xyz` and strips Y
     (NaniteResolve.ts:819–824), so vertical rounding is only the fixed 0.55 tilt — crown top and
     underside shade identically. Anchor at the vox cluster's AABB center (cluster words 0–3;
     `gpu.clusters` is already bound) → real top-lit/under-dark gradient. ~3 extra loads on vox px.
  b) *Per-species up-tilt:* the 0.55 constant (NaniteResolve.ts:823) could read species matParam
     (`meshWord(meshId,7)`, 1 load — already decoded in the non-voxbn path at 852) — conifer vs
     broadleaf crown shapes.
  c) *Far-tile bead:* FarTiles bricks ride identity instances ⇒ crownDir is ~constant per 64m tile
     (comment at 810–811) — the rounding vanishes exactly in the far field where the user reports
     blockiness. Cheap fix: detect identity instance and raise the `beadPix` weight (825–826).
Expected ms all: ~0 (a few ALU/loads on vox pixels; the entire vox shade tail is inside the
measured-free envelope).

**L3 — (guardrail, not a ms lever) adopt §5 as the acceptance rule for any new resolve branch.**
Prevents the free pool from silently becoming a 5ms pool when water/grass land.

## 7. Refuted / rejected for this stage (do not retry)

- **Shade-binning / per-material dispatch:** premise measured half-wrong 2026-07-02 (lighting ~free;
  docs/perf-runs/2026-07-02-attribution-and-waves.md §2 conclusion) and killed 2026-06-26 (no wave
  ops; docs/perf-runs/2026-06-26-voxel-lod-session-state.md §6h). The decode ceiling with a real
  candidate look is <1ms (`leafcheap=all`).
- **Cheaper leaf/vox decode as a perf lever:** bounded <1ms by the same measurement.
- **`?voxao=0` as perf:** the brick normal is baked; disabling is a visual A/B, not a win
  (NaniteResolve.ts:238–247 honest-perf note).
- **Resolve lighting as bimodality suspect:** refuted — bimodality persists under `nandbg=flat`
  (2026-07-02-attribution-and-waves.md §1).
- **Merging the two resolve passes back into one:** busts the Metal storage ceiling ⇒ silent
  pipeline kill (NaniteResolve.ts:305–319); the ~0.15ms prologue duplication is the cheapest
  possible rent.

## 8. Open questions + probes requested

- **R1 (the one probe this slice needs): within-boot keep-sample A/B at the canonical poses.**
  Boot default forest 200k; isolated eye+oblique with `window.__laasNanite.setKeepFull(1)` vs `(0)`
  alternated ×3 inside ONE boot (thermal-invariant by design, NaniteFrame.ts:361–365). Decision:
  med delta ≥0.5ms in ≥2/3 alternations at any pose ⇒ flip `reskeep` default to 0 (Class I,
  gated per L1: maxDiff=0 jitter-pinned shotdiff at 3+2 poses + fast-motion spot-check); else
  close L1 for canonical poses but STILL consider default-flip on the general-vista evidence
  (it is bit-identical and strictly less work) — the default-flip requires the same conforming
  Class-I shotdiff either way; the sub-2ms detection here is sanctioned only because the A/B is
  interleaved within one boot (premise audit §3.7).
- **R2 (honesty re-pin, piggyback on any probe session): `nandbg=flat` and `nanshadow=0` singles
  against the post-voxbocc baseline (18.9/37.2/16.5).** The FREE verdicts were measured pre-voxbocc
  (baseline 36.1/43.2/16.8); voxbocc changed pipelining (cpu.submit 5.3→1.4). Decision: any delta
  >2ms ⇒ reopen this doc; else the FREE verdict carries over to the new baseline and this slice is
  closed for the rest of the review.
- Open question (no probe yet): when water lands, does its resolve branch fit the tri pass's
  2-buffer headroom or does it need the third-pass pattern? Decide at design time against §5.

# spec-vox-kernel-microcuts — voxel scatter kernel micro-cut bundle (4 items)

Status: SPEC (buildable, not yet implemented). Branch: `nanite-raster`.
Target file: `src/nanite/NaniteVoxelRaster.ts` (all four items live in the one scatter kernel).
Expected bundle win: **~2–5 ms oblique** (point estimate ~3.5), ~0.9 eye, ~0.25 aerial.
**[REVIEW-FIX]** was "~3–6 (point 4.0)": doc 13's 3–6 bottom line INCLUDES L2 (F2B K=2,
+1.0 obl), which is NOT in this bundle; the doc-13 per-item bounds for A–D are 1–3 + 1–2 +
~0.7 + 0–0.3 and the spec itself declares the savings sub-additive (A/B overlap on dead-brick
cost, D shrinks the population A/C act on). 4.0 is the non-overlapping straight sum = the
optimistic edge, not the point estimate.
Quality class: every item is bit-identical, provably-dead-value-elided, or conservative-cull —
argued per item in §A–§D. Any visible pixel change at the gate poses = the stage is rejected.

Per-item flags (each independently revertible, so gates can isolate regressions):

| item | flag | default at land | default after gate pass |
|---|---|---|---|
| A — skip dead occ-mask builds | `?voxmaskray` | `0` (legacy) | `1` |
| B — Phase-B group-per-brick + compaction | `?voxsgb` | `0` (legacy) | `1` or `2` (probe decides) |
| C — linear-clip corner projection | `?voxlinproj` | `0` (legacy) | `1` |
| D — early sphere pre-bocc | `?voxsphocc` | `0` (legacy) | `1` only if it nets ≥0.2 ms |

---

## 0. Context: how the kernel works today (self-contained; no corpus needed)

All line numbers are against `src/nanite/NaniteVoxelRaster.ts` at nanite-raster HEAD
`d02a9f4` (1500 lines) unless prefixed. **[REVIEW-FIX]** was "`6a93dfb`" — wrong: the cited
code (?voxbocc :842-884, ?voxcell, ?fartiles) landed AFTER 6a93dfb (cc73e88/d636353/…;
+404 lines); every line number was re-verified against d02a9f4 and matches THAT tree. Do not
check out 6a93dfb. Read the cited ranges before editing — the file is dense with load-bearing
comments.

**Dispatch shape.** `buildNaniteVoxelRaster` (:175) builds ONE compute kernel per scatter
variant via the `makeVoxScatter` factory (:488). One WORKGROUP of `WG_RASTER = 128` lanes
(:480, = `MAX_BRICKS_PER_CLUSTER`, `src/nanite/VoxelBrick.ts:81`) per qVoxRaster work item
(= one ≤128-brick voxel cluster/BLOCK), indirect-dispatched over the culled count
(`.compute(DISPATCH_ROW*WG_RASTER,[WG_RASTER])` :1419; real grid from
`voxRasterDispatchAttr` :1490). Default path = one unordered whole-list dispatch (:1424-1428,
:1487-1491); `?voxf2b=1` builds K per-bucket instances (:1437-1452) — not touched here.

**Per-cluster prologue** (:550-586): fetch item, instance A/B words, cluster words 0-7,
block sphere; `payload`/`voxId` (:585-586). Thread 0 runs the per-BLOCK occlusion test
against `voxOccPyr` (:614-665) and broadcasts via the `wgVisible` workgroup flag + barrier.

**voxOccPyr** (:399-474): a min-pooled pyramid over THIS frame's `visPayloadV` (mesh SW+HW
winners; built before the scatter, :1461). Election keys pack `depthKey24<<8|id8`,
NEARER = LARGER; the min over a window is the farthest/most-see-through pixel, so
`cull iff frontSlabKey ≤ pooledMin` is conservative and an empty texel (key 0) always keeps.
`pyrTable` (:416-422) gives per-level (offset,w,h).

**Phase A — 1 lane = 1 brick** (:675-1063): seed empty records (:679-689); gate on
`brickActive = brickLocal < brickCount && wgVisible==1` (:690); decode brick words
(POS_XYZ/HALF, `VoxelBrick.ts:73-77`) (:695-700); world center via `instTransformPoint`
(`NaniteCommon.ts:141-148`) and world-AABB half-extent `brWR` via `instSphereRadius`
(`NaniteCommon.ts:164-167`) (:701-702); project the **8 corners of the world-axis-aligned
cube [brWCenter ± brWR]** — per corner 3 adds + one `cam.vp` mat4·vec4 + divide, with
near-plane/NDC-explode straddle classification (:737-772); non-straddler → raw clamped bbox
+ span cap (:803-812), straddler → BRICK_MAX_EXT centre box (:813-836); **?voxbocc**
(default ON) per-brick front-slab test vs voxOccPyr, straddlers exempt (:842-884); survivor
stores its record: bbox + precomputed election key `cand` (:885-902), ?voxcell ray record
(local AABB + occ words + non-straddle flag, :908-920), and — for a coarse (`dagLevel>0`),
armed (`area ≥ 16`), sparse (`popcount ≤ 48`) brick — the **4×4 screen occupancy-mask
build**: loop 64 cells × project 8 corners each, ≤512 projections (:921-1060).

**Phase B — all 128 lanes, cooperative** (:1064-1413): barrier (:1064), then a serial loop
over the cluster's bricks (:1140) executed by EVERY lane; per brick ~13-15 workgroup reads +
2 fp reciprocals + eligibility logic (:1142-1181); then the brick's footprint `[0, bbW·bbH)`
strides across the 128 lanes (`loopU(brickLocal, area, …, WG_RASTER)` :1183, :1412).
Per pixel: reciprocal addressing (:1187-1193); **ray path** for eligible bricks
(`cellElig` = non-straddler AND `area > voxCellMinArea`(64), :1167-1171): relaxed-load
early-out (:1266-1267), linear-in-NDC local ray from per-cluster bases (:1098-1136), slab
test (:1288-1298), ≤6-step DDA (:1332-1381), exact-depth guarded election (:1394-1403);
**flat path** otherwise: optional 4×4 occ-mask bucket test (:1240-1252) then the guarded
election `electHere` (:1196-1215).

**Measured motivation** (docs/deep-review/13-vox-raster.md, 05-voxel-raster-runtime.md;
raw JSONs `fresh-*.json` in the scratchpad): isolated gpuWall med at 200k trees, dpr 1.5
(2268×1473): eye 18.9 / **oblique 37.2** / aerial 16.5. The kernel is **count-bound, not
fill-bound**: cost fits ~1.6–2.2 µs per emitted vox cluster; aerial covers ~100% of the
screen with 659 clusters for 5.4 ms of foliage while oblique covers ~90% with 9188 clusters
for ~20 ms — i.e. ~75-80% of the oblique voxel cost is per-brick/per-cluster machinery, not
per-pixel work. This bundle attacks exactly that machinery. It cannot close the whole
−11 ms oblique gap (the cluster COUNT is generated upstream by the 45–140 m per-tree ring);
its honest range is ~2–5 ms (**[REVIEW-FIX]** was "~3–6"; see the header correction — 6 is
the no-overlap optimistic edge of the doc-13 waste bounds, not a ceiling this bundle's four
items support jointly).

**Environment constraints that shape every edit below:**
- **TSL r184 hoist hazard**: any value consumed by an atomic (or inside a deep conditional)
  must have its FIRST node build in the SAME conditional subtree, else the builder hoists it
  above the branch (see the in-file fix notes :1194-1201, :1391-1397). Rule of thumb:
  `.toVar()` per-scope, build-inside-consumer for election keys, build-outside-consumer only
  for uniform per-cluster bases (the proven pattern :1098-1136).
- **Metal 10-storage-buffer cliff**: kVoxScatter binds ~7-8 storage buffers today
  (qVoxRaster, instances, clusters, voxelBricks, visPayloadV, visBV, voxOccPyr [+ bucket
  range under f2b]). **None of the four items adds a storage buffer** — items add only
  workgroup memory / registers.
- **No 64-bit atomics** (WebGPU): the election stays `guard → atomicMax(u32 key) → winner
  atomicStore(visBV)`. Do not touch the idiom.
- **No workgroup-scope atomics in three r184 TSL** (verified:
  `node_modules/three/src/nodes/gpgpu/WorkgroupInfoNode.js` declares plain-typed workgroup
  arrays only; there is no `atomic<u32>` workgroup path). Item B's compaction therefore uses
  a lane-0 serial scan, NOT a workgroup atomic counter.
- **No HW integer divide on Apple** — keep the `?voxrecip` reciprocal addressing (:1187-1193)
  untouched; new index math below uses shifts/ands only.
- **Barriers must sit in workgroup-uniform control flow.** The whole body is inside
  `If(guard)` (:550) where `guard` derives from `workgroupId` only (:1424-1427) — uniform,
  which is why the existing barriers (:665, :672, :1064) compile. New barriers must sit in
  that same scope, never inside a lane-divergent `If`.

---

## A. `?voxmaskray` — skip the occupancy-mask build for ray-eligible bricks

### Problem + measured motivation
The Phase-A occ-mask build (:921-1060) costs up to 64 cells × 8 corner projections
(≤512 `instTransformPoint` + mat4·vec4 chains) on ONE lane, and one armed lane serializes
its whole 32-lane SIMD group (divergence amplification). But the mask is **consumed only by
the flat path** (:1240-1252). A brick that Phase B routes to the ray path never reads it.
Verified consumer set: `wgOccMask` is written at :687 (seed) and :1057 (build), read ONCE at
:1153 into `occMask`, which feeds `gateActive` (:1157) — consumed exclusively inside
`flatPath` (:1240-1252). The ray branch (`If(cellElig, …)` :1261-1407) never touches it, and
`cellElig` is per-brick-uniform (:1167-1171), so an eligible brick's pixels ALL take the ray
branch. Today's arm gate (:938) is `area ≥ 16 AND dagLevel > 0` with **no upper bound** —
the τcap≈12 px far field projects ~144 px² bricks, all armed, all dead builds.
Doc 13 bounds this waste at **~1–3 ms oblique** (per-brick, divergence-amplified).

### Design — exact edit
One condition change at :934-938, plus the flag parse next to `voxCell` (:295-302).

Ray-eligibility at :1167-1171 is exactly:
`wgCellOk[b]==1 AND area > voxCellMinArea`, compiled only when `voxCell` (i.e.
`?voxcell!=0 && !voxDither`). `wgCellOk` stores `1 - straddles` (:919) inside the same
`If(validBB && bVis)` block that hosts the mask build, and Phase B's `area` = `bbW·bbH`
of the same stored record = `gateArea` (:935). So the *skip* predicate can be evaluated
in Phase A with locally available values, exactly:

```ts
// flag parse (near :302):
const voxMaskRay = new URLSearchParams(window.location.search).get('voxmaskray') === '1';
// (flip to !== '0' when the gate passes)

// at :934-938 — extend the arm gate; EVERYTHING inside the If is unchanged:
if (voxOccGate && wgOccMask && dagLevel) {
  const gateArea = bbW.mul(bbH).toVar();
  let arm = gateArea.greaterThanEqual(uint(OCC_GATE_MIN_AREA)).and(dagLevel.greaterThan(uint(0)));
  if (voxMaskRay && voxCell) {
    // ray-eligible ⇒ mask provably unread (dead value): non-straddler AND area > voxcellmin.
    // MUST mirror :1167-1171 exactly: strict greaterThan, same voxCellMinArea, straddle sense.
    const rayEligible = straddles.equal(uint(0)).and(gateArea.greaterThan(uint(voxCellMinArea)));
    arm = arm.and(rayEligible.not());
  }
  If(arm, () => { /* :939-1059 byte-identical */ });
}
```

Compile-time gating: when `voxCell` is false (`?voxcell=0` or `?voxdither=1`) every brick is
flat-path ⇒ the extra term is not emitted ⇒ byte-identical legacy. The predicate holds for
ANY `?voxcellmin` value because both sites read the same `voxCellMinArea` (:301-302).

**Fold-in micro (same diff, bit-identical):** dedupe the double `occLo/occHi` fetch — the
voxcell record loads words BRICK_OCC_LO/HI at :913-914 and the mask build re-loads them at
:939-940. Hoist the two `elemU` loads into `.toVar()`s just above :908 and consume in both
places. Same buffer words, same values; pure load elimination. **[REVIEW-FIX]** the hoist
MUST be compile-gated `if (voxCell)`: when voxCell is ON, :913-914 already load these words
for EVERY stored brick, so the hoist adds zero loads. When voxCell is OFF
(`?voxcell=0`/`?voxdither=1`) and voxOccGate is ON, legacy loads them ONLY inside the armed
`If` (:938→:939-940); an unguarded hoist would add two loads per stored brick in that
config — a pessimization, not a dedupe. `if (voxCell)` → hoisted vars consumed at both
sites; `else` → keep :939-940 exactly as today.

### Quality-equivalence argument
**Bit-identical, provable statically.** The skipped mask's only runtime reader is the flat
path of the very brick that built it; for every skipped brick the flat path is unreachable
(its `cellElig` is true for all footprint pixels, per-brick-uniform). The seed value 0xffff
(:687) remains in place but is equally unread. No election input changes; no other brick
reads another brick's mask. The dedupe micro reads the same u32 words once instead of twice.

### Expected
0.2 eye / **1.5 oblique** / 0.1 aerial (doc 13 L1; share unmeasured — the gate decides).

### Risks + fallbacks
- Predicate drift: if someone later changes `cellElig` (:1167-1171) without mirroring the
  Phase-A skip, bricks could flat-path with a wrongly-skipped mask ⇒ they'd paint SOLID
  (0xffff seed) — a fuller footprint, never a hole; visible-change class though, so add a
  comment at BOTH sites pointing at each other. Fallback: `?voxmaskray=0` restores legacy.
- None else — S effort, single-condition change.

---

## B. `?voxsgb` — Phase-B group-per-brick distribution + live-brick compaction + block-cull gate

### Problem + measured motivation (claim verified)
Phase B's outer loop (:1140) is executed by ALL 128 lanes. A 128-lane workgroup on Apple M1
runs as 4×32-wide SIMD groups; within one SIMD group the per-brick setup (~13-15 workgroup
reads: bbox 5 + occMask 1 + voxcell 7, plus 2 fp divides :1161-1162 and the eligibility
logic :1163-1181) issues once, but the four SIMD groups each issue it independently ⇒
**per-brick setup is issued 4× today** — the claim from doc 13 waste #3, confirmed
structurally (there is no `if lane<N` guard; the loop body is unconditional per lane).
Worse, **zero-area records still pay full setup**: `area` is computed at :1158 and only
zero-trips the inner loop; every bocc-culled brick (the eye-pose majority) still costs
4×(reads+divides+eligibility). And a **block-culled cluster** (`wgVisible=0`) never stores
records, yet Phase B still iterates `nBricks` × setup ×4 groups — `wgVisible` is only
consulted in Phase A (:690). Doc 13 bounds the bundle at ~1–2 ms oblique + eye-side gains.

### Design
`?voxsgb=1`: virtual-group distribution + early empty-skip + block-cull gate.
`?voxsgb=2`: additionally compact live bricks so dead records aren't even visited.
No subgroup intrinsics are used (three r184 exposes them, but they'd require the `subgroups`
device feature + carry non-portable subgroup sizes) — the "group" is a **virtual 32-lane
slice of the workgroup derived from `localId.x` arithmetic only**, so correctness is
scheduling-independent and no feature/extension is needed.

Edit sites: everything between the Phase-A barrier (:1064) and the Phase-B loop head
(:1140), the loop head itself, and the inner-stride call (:1183, :1412). The per-brick setup
body (:1142-1181) and the entire per-pixel body (:1183-1411) are **moved, not modified**.
**[REVIEW-FIX]** ordering: `nBricks` is currently defined at :1076 (AFTER the :1064 barrier,
before the voxcell bases) — the sgb=2 compaction scan below consumes it, so move the :1076
declaration ABOVE the compaction block (it depends only on `brickCount`, defined :562;
trivially legal). Landing the sketch verbatim without the move is a use-before-def.

```ts
const VOX_GROUPS = 4;                       // WG_RASTER / 32 — virtual groups, not HW subgroups
const gid  = brickLocal.shiftRight(uint(5)).toVar();   // 0..3
const lane = brickLocal.bitAnd(uint(31)).toVar();      // 0..31

// ── ?voxsgb=2 live-brick compaction. NO workgroup atomics exist in r184 TSL
// (WorkgroupInfoNode declares plain types only), so lane 0 builds the list serially
// post-barrier: 128 iterations of 1 wg-read + branch — trivially cheaper than the
// nBricks×4-group setup it deletes. wgLive/wgLiveN = +129 u32 = 516 B workgroup memory
// (current usage ≈ 6.2 KB of the 32 KB budget — fine).
const wgLive  = voxSgb >= 2 ? workgroupArray('uint', WG_RASTER) : null;
const wgLiveN = voxSgb >= 2 ? workgroupArray('uint', 1) : null;
workgroupBarrier();                          // the EXISTING :1064 barrier — unchanged
if (voxSgb >= 2) {
  If(brickLocal.equal(uint(0)), () => {
    const n = uint(0).toVar();
    loopU(uint(0), nBricks, (i) => {
      If((wgBbW.element(i) as unknown as NU).notEqual(uint(0)), () => {
        wgSet(wgLive!, n, i);
        n.assign(n.add(uint(1)));
      });
    });
    wgSet(wgLiveN!, uint(0), n);
  });
  workgroupBarrier();                        // NEW barrier — same uniform scope as :1064
}

// ── block-cull gate (both sgb levels): a culled block stored no records; skip Phase B.
// wgVisible is per-workgroup-uniform; no barrier lives inside the If ⇒ legal flow.
const blockVis = (wgVisible.element(uint(0)) as unknown as NU).equal(uint(1));
If(blockVis, () => {
  const outerEnd = voxSgb >= 2 ? (wgLiveN!.element(uint(0)) as unknown as NU).toVar() : nBricks;
  loopU(gid, outerEnd, (slot) => {
    const b = voxSgb >= 2 ? (wgLive!.element(slot) as unknown as NU).toVar() : slot;
    const bbW = (wgBbW.element(b) as unknown as NU).toVar();
    If(bbW.notEqual(uint(0)), () => {        // sgb=1 empty-skip; sgb=2: always true, keep for safety
      /* :1142-1181 setup body UNCHANGED (reads keyed on b) */
      /* CRITICAL: voxIdB (:1150) keeps b — the TRUE brick index — in bits 21-27,
         never `slot`; the resolve decodes brick normal/albedo from those bits. */
      loopU(lane, area, (localPx) => {
        /* :1184-1411 per-pixel body UNCHANGED */
      }, 32);                                // stride 32, was WG_RASTER (:1412)
    });
  }, VOX_GROUPS);                            // outer stride 4, was a serial 0..nBricks loop
});
```

`loopU` supports NU start/end + constant numeric step (`src/nanite/Tsl.ts:122-130`; the
existing inner loop already passes `WG_RASTER` the same way). Legacy path (`?voxsgb=0`)
compiles today's exact loop — keep both code paths build-time selected, like `voxRecip`.

### Quality-equivalence argument
The set of (pixel, key, id) election candidates is **unchanged** — only which lane visits
which (brick, pixel) pair changes. The election is an unordered `atomicMax` on the packed
key with a strictly-greater guard on both the pre-load and the winner store (:1201-1206,
:1394-1403): the final key per pixel is the max over the same candidate set ⇒ for every
pixel whose winning KEY is unique, the output is bit-identical.
**[REVIEW-FIX] Tie case, corrected — the original "introduces no new nondeterminism class"
claim was too strong.** A tie = two bricks of one cluster with the identical 32-bit key
(same quantized front-slab depth, same id8) but different `voxIdB` bits 21-27 → a tie flip
changes which brick's baked normal/albedo shades that pixel (a real, if subtle, pixel
change). Ties are realistic: adjacent same-cluster bricks in a screen-parallel crown face
share the quantized depth, and their inclusive floor/ceil bboxes overlap by ~1 px at edges.
Today the winner is scheduler-order-dependent ONLY when the two candidates' visiting lanes
sit in DIFFERENT SIMD groups; when both lanes land in the SAME SIMD group the serial `b`
loop resolves the tie deterministically (lower brick index first ⇒ its store is the one the
strictly-greater guard lets through... note the SECOND candidate loses the guard, so the
FIRST stays) — and stably so, frame after frame. Item B can move such a pair cross-group
(or reorder it), flipping a today-stable tie winner PERSISTENTLY. Consequence: the §Gates
shot protocol is **load-bearing for item B**, not belt-and-braces — the expected diff is a
sparse set of brick-edge pixels whose depth is identical but whose brick shading flips; it
must land within the base-vs-base TRAA noise floor or the stage is rejected
(`?voxsgb=0` reverts; nothing else in the bundle depends on B).
Depth-distinct winners are bit-identical by the max argument. The block-cull gate skips
only clusters whose records are all empty seeds (Phase A stores records only under
`brickActive`, which includes `wgVisible==1`, :690) — skipping a no-op.

### Expected
0.3 eye / **1.5 oblique** / 0.1 aerial. Watch eye specifically: bocc-culled bricks (the eye
majority) currently pay 4× setup for zero pixels; sgb may over-deliver there.

### Risks + fallbacks
- **Barrier placement** is the one real hazard: the new sgb=2 barrier MUST be a sibling of
  :1064 (directly inside `If(guard)`, outside every lane-divergent If). Tint/naga reject it
  otherwise — a compile error, not a silent wrong.
- Register pressure: `gid/lane/b` add ~3 registers; the moved setup body is unchanged. If
  occupancy drops (visible as a REGRESSION at aerial where blocks are huge), fall back
  `?voxsgb=0` — nothing else in the bundle depends on B.
- Ragged tails: a brick's footprint now spreads over 32 lanes not 128 (`ceil(area/32)`
  rounds), but 4 bricks run concurrently — net utilization ≥ today on mixed sizes; the probe
  decides between sgb=1 and sgb=2 (the serial scan may cost more than dead-record visits
  save on brick-light clusters).
- **[REVIEW-FIX]** Perf-expectation caveat: the "setup issued 4× → 1×" claim assumes Apple's
  SIMD width is 32 for this kernel. Metal drops thread_execution_width to 16 under high
  register pressure — and this monolith is exactly the register-pressure suspect (doc 13
  open Q5). At width 16 a 32-lane virtual group spans 2 HW SIMD groups ⇒ dedup is 2×, not
  4× — still a win, correctness unaffected, but halve the expectation if the probe
  under-delivers before blaming the design.
- Effort: M. The moved body must be moved verbatim — diff review should show pure
  indentation/loop-head changes plus the two new prologue blocks.

---

## C. `?voxlinproj` — linear-clip corner projection (Phase A walk + mask-cell corners)

### Problem + measured motivation
Two projection hot spots in Phase A:
1. **Brick 8-corner walk** (:737-772): per corner, 3 adds for the world corner (:743-747)
   then a full `cam.vp` mat4·vec4 (:748) ≈ ~34 ALU ⇒ ~270 ALU/brick, paid by EVERY live
   brick (9k clusters × ~dozens of bricks at oblique) — including bricks voxbocc then culls.
   NOTE a doc-13 inaccuracy, corrected here: the brick walk does NOT call
   `instTransformPoint` per corner (the cube is world-axis-aligned already); only the
   mat4·vec4 is on the table for site 1.
2. **Mask-build cell corners** (:997-1019): per occupied cell, 8 × (`instTransformPoint`
   ≈ 12 ALU + mat4·vec4 ≈ 28 ALU) ≈ ~320 ALU/cell, ≤64 cells/armed brick. Item A removes
   most builds; survivors (small flat bricks, 16 < area ≤ 64) still pay it.

Clip transforms are linear, so both collapse to basis-add form. **[REVIEW-FIX]** attribution
corrected: doc 13 (L4) bounds the item at ~0.7 oblique; doc 05 (its L4, "shared clip-basis
Phase A projection") estimates −1 to −2.5 oblique. Plan on 0.7 (the conservative number used
in the bundle estimate); treat doc 05's range as upside the gate may or may not confirm —
do NOT promise it.

### Design
**Site 1** — per-cluster, before Phase A (uniform scope, after :586):

```ts
// clip-space images of the world axes under vp (per-cluster uniforms; the compiler
// folds vp·e_i to column loads):
const vpCx = (cam.vp.mul(vec4(1, 0, 0, 0)) as unknown as NV4).toVar();
const vpCy = (cam.vp.mul(vec4(0, 1, 0, 0)) as unknown as NV4).toVar();
const vpCz = (cam.vp.mul(vec4(0, 0, 1, 0)) as unknown as NV4).toVar();
```

Per brick (replacing :743-748 inside the existing sz3/sy3/sx3 loops — the loops, the
w/NDC-explode classification (:749-769) and min/max accumulation stay byte-identical):

```ts
// before the corner loops (per brick):
const clipC = (cam.vp.mul(vec4(brWCenter, 1)) as unknown as NV4).toVar();
const dX = (vpCx.mul(brWR) as unknown as NV4).toVar();
const dY = (vpCy.mul(brWR) as unknown as NV4).toVar();
const dZ = (vpCz.mul(brWR) as unknown as NV4).toVar();
// inside the loops (sx,sy,sz = ±1 exactly as today):
const p = clipC.add(dX.mul(sx)).add(dY.mul(sy)).add(dZ.mul(sz)).toVar();
```

**[REVIEW-FIX]** ~34→~24 ALU per corner as written (3 vec4 scaled-adds = 12 mul + 12 add);
the "~12" figure holds only for the ±sign add-tree variant (sx/sy/sz = ±1 folded to
add/sub — 14 vec4 adds for all 8 corners), which is optional and the same quality class.
Cost added per brick: 1 mat4·vec4 (clipC) + 12 muls (dX/dY/dZ), amortized over 8 corners.

**Site 2** — per-cluster local-axis basis (only when `voxOccGate`; place next to site 1):
`instTransformPoint(p) = L·p + A.xyz` with `L = scale∘yaw∘shear` (`NaniteCommon.ts:141-148`),
so `clip(p) = M·p + Mb` where `Mx = vp·(L·ex,0)`, etc. From the transform definition:
`L·ex = (A.w·cy, 0, −A.w·sy)`, `L·ey = (B.y·A.w, A.w, B.z·A.w)`, `L·ez = (A.w·sy, 0, A.w·cy)`
(cy/sy from `instYaw`, `NaniteCommon.ts:132-135`), and `Mb = vp·(A.xyz,1)`. Per cell centre:
`clipCell = Mb + Mx·clx + My·cly + Mz·clz` (12 FMA, replacing :1005-1006's transform+mat4);
per cell the 8 corners = `clipCell ± Mx·hc ± My·hc ± Mz·hc` with `Mx·hc` etc. hoisted per
BRICK (hc is per-brick, :991). ~320→~70 ALU per cell. The w-test/degenerate handling
(:1007, :996-1019) stays byte-identical.

### Quality-equivalence argument — and a correction to doc 13
This is the ONE item that is **not provably bit-exact**: fp addition is non-associative, so
the linearized clip coords can differ from today's by a few ulps. Screen coords are O(10³)
px; an fp32 ulp there is ~2⁻¹² px, so a floor/ceil (:804-812) flips only when the exact
coordinate sits within ~0.00025 px of an integer — expected zero flips in practice, but it
must be GATED, not assumed (shot protocol in §Gates; class "identical-with-gate").
**Do NOT apply doc 13's "+1 px conservative expansion"** — it is wrong for this codebase:
the bbox IS the painted footprint on the flat path (a 1 px bigger rect = visible change),
and on the ray path a wider bbox can ADMIT true-silhouette pixels today's bbox clips (the
`brWR` bound ignores yaw, see :702 + `NaniteCommon.ts:164-167` — under-coverage up to ~29%
linear at yaw≈45°, doc 13 §nit 9), which is also a visible change. Exact linearized values +
zero-diff gate is the only admissible form. If the gate shows ANY flip: reject the stage
(fallback `?voxlinproj=0`); there is no provably-safe partial variant (the ray-only variant
has the same bbox-admits-new-silhouette-pixels hazard).

### TSL codegen notes
Bases are per-cluster uniform values built OUTSIDE the per-brick `If` and consumed inside —
the same direction as the proven voxcell ray bases (:1098-1136); no hoist hazard (the hazard
is the reverse direction: first-build inside a conditional, reuse outside/in a sibling).
All bases `.toVar()`'d once. Site-2 bases compile only under `voxOccGate` (no dead registers
when `?voxlod=0`).

### Expected
0.2 eye / **0.7 oblique** / 0.05 aerial.

### Risks + fallbacks
- Register pressure: +7 vec4 per-cluster live ranges across Phase A (the known monolith
  occupancy question, doc 13 open Q5). If any pose regresses, try site 1 only (site 2 is
  mostly dead after item A anyway); else `?voxlinproj=0`.
- Shot-gate flakiness from TRAA jitter is handled by the noise-floor protocol (§Gates) —
  do not eyeball single diffs.
- Effort: M.

---

## D. `?voxsphocc` — early sphere pre-bocc (cull before the 8-corner walk)

### Problem + measured motivation
The per-brick voxbocc test (:842-884) fires AFTER the corner walk — it needs the bbox. So
every occluded brick (the eye-pose majority; measured −17.2 ms eye when voxbocc landed,
:277-281) still pays the walk (~270 ALU + classification) before dying. A sphere test needs
only the already-computed `brWCenter`/`brWR` (:701-702) and 4 pyramid loads.

### Design
Insert directly after :702, wrapping the REST of Phase A (the dither stash :708-715, corner
walk, bocc, record store) in `If(preVis)`. Compile only when `voxSphOcc && voxBocc &&
voxOccl` (it is a refinement of the same cull; `?voxbocc=0`/`?voxoccl=0` A/B controls keep
their meaning).

```ts
const preVis = uint(1).toVar();
const bp = (cam.vp.mul(vec4(brWCenter, 1)) as unknown as NV4).toVar(); // shared with C's clipC
const rW = brWR.mul(float(1.7320508)).toVar();  // √3·halfExtent = circumradius of [c±brWR]
const wSafe = bp.w.sub(rW).toVar();
If(wSafe.greaterThan(float(NEAR_EPS)), () => {
  // straddle-safety guard, divide-free: every cube corner q satisfies
  // |clip.x(q)| ≤ |bp.x| + rW·cot  and  w(q) ≥ wSafe, so corners cannot trip the
  // w ≤ NEAR_EPS branch (:767-769) nor the NDC_EXPLODE branch (:751-756) when:
  const cot = cam.cotHalfFov as unknown as NF;
  const safeX = bp.x.abs().add(rW.mul(cot)).lessThan(float(NDC_EXPLODE).mul(wSafe));
  const safeY = bp.y.abs().add(rW.mul(cot)).lessThan(float(NDC_EXPLODE).mul(wSafe));
  If(safeX.and(safeY), () => {
    const bndc = bp.xyz.div(bp.w).toVar();
    const nzS = bndc.z.sub(rW.div(bp.w)).clamp(0, 1);            // block-test idiom :620
    const kS = depthKey24(nzS as unknown as NF).shiftLeft(uint(8)).bitOr(uint(0xff)).toVar();
    // rigorous over-bound of the projected radius: r/√(d²−r²) ≤ r/(d−r) = rW/wSafe
    const rPx = rW.mul(cot).mul(H).div(wSafe.mul(2)).toVar();
    /* level pick + 2×2 min-window EXACTLY as the block test :636-659 (pyrTable, minU) */
    If(kS.lessThanEqual(occK), () => { preVis.assign(uint(0)); });
  });
});
If(preVis.equal(uint(1)), () => { /* :704-1062 unchanged */ });
```

### Quality-equivalence argument
**Conservative cull, identical output** — the same class as shipped voxbocc (which gated
shot-identical at eye+oblique, :277-281):
1. *Coverage*: the brick's painted pixels ⊆ bbox(8 projected corners) (flat path paints the
   rect; ray path paints a subset). Corners lie on the sphere (center, √3·brWR) ⊇ the world
   cube; projection of a convex set in front of the camera ⊆ projection of a containing
   sphere ⇒ painted pixels ⊆ the sphere's screen extent, which the picked level's 2×2
   window covers (the `rW/wSafe` radius over-bounds the exact `r/√(d²−r²)` projected radius
   since `(d−r)² ≤ d²−r²` for `r ≤ d`, guaranteed by the `wSafe > 0` guard).
2. *Key bound*: `kS` (sphere front-slab, `|0xff` tiebreak) ≥ any key the brick can elect —
   the identical front-slab idiom the shipped block test uses for everything inside a sphere
   (:600-612); ray-path keys are ≤ `cand` ≤ `kS` (hits at/behind the front slab :1262-1264).
3. Therefore `kS ≤ pooledMin` ⇒ every footprint pixel's current winner beats every candidate
   the brick could offer ⇒ the brick changes nothing; dropping it is invisible. Empty texels
   (key 0) force KEEP, straddle-suspect bricks (guard fails) proceed unculled ⇒ the
   straddler exemption (:844) is preserved by construction — a brick that could classify as
   a straddler can never pass `wSafe > NEAR_EPS` + the explode-margin guard.
4. Assumptions stated: `cam.vp = P·V` with V rigid (true: `NaniteCommon.ts:112`,
   perspective camera, no scale) so `w(p)` is view depth with unit gradient, and the clip-x/y
   gradients are ≤ `cot` (x row = cot/aspect ≤ cot at 2268×1473).

### Expected
0.4 eye / **0.3 oblique** / 0.0 aerial — eye-leaning (that's where bricks are mesh-hidden).
Surviving bricks pay +4 pyramid loads + ~20 ALU; oblique could net ~0. The gate decides;
if oblique < +0.2 net at every pose, leave default OFF and keep the flag for the eye-heavy
live path (live p95 is eye-dominated).

### Risks + fallbacks
- Double pyramid reads for visible bricks — measured, not argued.
- The level pick + window MUST be copied from the block test verbatim (:636-659) with only
  `rPx` sourced as above; a hand-rolled variant risks the under-coarse/hole direction.
- **[REVIEW-FIX]** honesty note on the coverage argument (§1 above): the
  `rPx = r·cot·H/(2·w)`-family level pick treats the sphere's screen footprint as the
  isotropic centered-sphere radius; a sphere near the screen EDGE projects slightly wider
  (off-center anisotropy, up to ~1 missing level in the worst corner). This is INHERITED
  from the shipped block test (:631-635) — D's variant is strictly safer than it (wSafe
  denominator ⊃ r/(d−r) over-bound, √3 circumradius) — and the shipped idiom gated
  shot-identical at all poses, with under-pick further absorbed by the ceil, the 2×2 (not
  1×1) window, and min-pool-keeps-on-any-empty-texel. So the static argument is
  "conservative modulo the same approximation the production block cull already ships";
  the §Gates shot protocol is the arbiter of the residual, exactly as it was for voxbocc.
- Effort: S/M. Fallback `?voxsphocc=0` (and it never compiles unless voxbocc+voxoccl are on).

---

## Staged landing plan

One stage = one commit = one flag, in this order (independent code paths; order chosen by
expected value ÷ risk). Each lands with the flag default-LEGACY, is measured (gates below),
then a one-line default flip ships in the gate-pass commit. `npx tsc --noEmit` clean per
stage.

1. **Stage A** `?voxmaskray` (S) — condition-only change + occLo/Hi dedupe.
2. **Stage B** `?voxsgb` (M) — land 0/1/2 tri-state; probe picks 1 vs 2.
3. **Stage C** `?voxlinproj` (M) — both sites; strict shot gate.
4. **Stage D** `?voxsphocc` (S/M) — reuses C's `clipC` if C landed (share the `bp` var).

After each default flip, re-run the composed default once (label `mc-cum-N`) so the running
baseline stays honest; items are independent but their savings overlap sub-additively (A and
B both shrink dead-brick cost; D shrinks the population A/C act on). Report the bundle as
the final composed delta, not the sum of stage deltas.

## Measurement gates

**Apparatus** (no GPU work in this spec — these are the commands the implementing engineer
runs): vite dev server on :5173, then `tools/probe-fresh-stutter.ts` (env: CONFIG/EXTRA/
LABEL/TICKS/COOLDOWN_S/TREES/DPR; defaults TREES=200000 DPR=1.5 = canonical 2268×1473). It
writes `fresh-<LABEL>.json` (per-pose `gpu[]` = gpuWall samples, `counters`) + per-pose PNGs
to the scratchpad, and prints per-pose med/p95.

**Protocol per stage — interleaved A/B/A, same session, no other GPU load:**

```bash
CONFIG=default                      LABEL=mc-base-1  TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default EXTRA=<flag>=<on>    LABEL=mc-<item>  TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default                      LABEL=mc-base-2  TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
```

(`<flag>=<on>`: `voxmaskray=1`; `voxsgb=1` then `voxsgb=2`; `voxlinproj=1`; `voxsphocc=1`.)

**Perf decision rule** (per pose, gpuWall median): noise floor `N_pose = |med(base-1) −
med(base-2)|` (oblique is bimodal, historically ±2 ms — never trust a single unbracketed
median). SHIP (flip default) iff
`med(item) ≤ min(base medians) − max(floor_item, N_pose)` at the item's target pose AND no
pose regresses by more than `max(0.3, N_pose)`. Floors: A 0.8, B 0.8 (pick the better of
sgb=1/2; if within noise of each other prefer 1 — less code), C 0.4, D 0.2 (D additionally:
if it only helps eye, flip default only after checking live p95 with a TICKS=2200 run).

**Sanity counters:** `nanite.voxClusters` (HUD counter, `src/nanite/NaniteFrame.ts:524`)
must be equal (±0) between base and item at each pose — none of these items may change the
emitted cluster set; a drift means a bug, not a win.

**Quality gate (every stage, all 3 poses — eye/oblique/aerial as defined in the probe):**
compare the probe's own screenshots with `tools/diff.ts`:

```bash
npx tsx tools/diff.ts --a $SCRATCH/shots/mc-base-1-<pose>.png --b $SCRATCH/shots/mc-<item>-<pose>.png --out /tmp/d-<pose>.png
npx tsx tools/diff.ts --a $SCRATCH/shots/mc-base-1-<pose>.png --b $SCRATCH/shots/mc-base-2-<pose>.png --out /tmp/n-<pose>.png
```

PASS iff, per pose, item-vs-base `changed%` AND `mean max-channel delta` are ≤ the
base-vs-base values (the TRAA/temporal noise floor). For items A/D this is belt-and-braces
on top of the static arguments; for item B it is **load-bearing for the tie-flip class**
(**[REVIEW-FIX]** see §B — same-SIMD-group ties are stable today and sgb can flip them
persistently); for item C it IS the gate — any excess diff rejects the
stage outright (no +1 px "conservative" rescue; see §C). If a stage fails quality, revert
the default, keep the flag for diagnosis, and file the failure against the stage — do not
re-tune within the stage without re-running the full triple.

**Bundle exit criterion:** composed default (all passed flags on) vs the pre-bundle
baseline, same triple protocol: oblique gpuWall med improvement ≥ 3.0 ms ⇒ bundle SHIPPED as
scoped; 1.5–3.0 ⇒ shipped-partial (report which items carried); < 1.5 ⇒ premise-audit the
count-bound model before touching the kernel further (the upstream ring-cluster count is
then confirmed as the only remaining lever — see docs/deep-review/13-vox-raster.md §3).

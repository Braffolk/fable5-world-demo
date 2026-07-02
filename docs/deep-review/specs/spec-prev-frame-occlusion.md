# SPEC: UE5-style two-pass / prev-frame occlusion for the vox path

**Lever id:** `prev-frame-occlusion` · **Target:** oblique −2..5 ms (net, isolated gpuWall med),
plus a live-metric correctness fix · **Quality class:** byte-identical output (stage 2) +
strictly-more-conservative bug fix (stage 1) · **Status:** spec, nothing built.

Baselines this spec is written against (isolated gpuWall med, 200k trees @ 2268×1473, dpr 1.5):
eye 18.9 / oblique 37.2 / aerial 16.5; live p50 16.7 / p95 25.1. Mission gap: oblique −11 ms,
zero quality sacrifice (user law: **any visible pixel change = rejected**).

All `file:line` cites are at `nanite-raster` HEAD d02a9f4 (**[REVIEW-FIX]** was written
against "6a93dfb + uncommitted voxbocc work"; that work is now committed — 8357d99 — and
every cite below was re-verified line-exact against the d02a9f4 working tree during
review). Every cited file was read in full for this spec.

---

## 1. Problem + measured motivation

### 1.1 The two occlusion structures, and what each can and cannot see

The renderer has TWO hierarchical occlusion pyramids with different content and different
consumers (this is the single most important fact for this lever):

| structure | content at test time | consumer | granularity |
|---|---|---|---|
| main HZB (`src/nanite/NaniteHzb.ts`) | **FULL** (mesh+vox+terrain) but **one frame stale** — built at `NaniteFrame.ts:481`, *after* vox scatter, consumed by *next* frame's DAG traverse | cluster emit test `NaniteCull.ts:919-925` (prev VP + prev camPos) | cluster sphere, 2×2 max window |
| `voxOccPyr` (`NaniteVoxelRaster.ts:399-474`) | **same-frame** but **mesh-only** — min-pooled over `visPayloadV` right after `world1`+`hwRender`, *before* any vox election (`NaniteVoxelRaster.ts:1461`, order at `NaniteRaster.ts:1418-1424`) | per-BLOCK cull (`NaniteVoxelRaster.ts:613-673`) + per-BRICK `?voxbocc` (`NaniteVoxelRaster.ts:843-884`) | block sphere / brick bbox, 2×2 min window |

Consequences, all measured:

- **Cluster-granularity occlusion is structurally useless at oblique.** `sphereOccluded`
  picks the mip where the whole cluster sphere fits one texel (`NaniteHzb.ts:175-176`) and
  max-pools a 2×2 window; a ~105-brick crown cluster's window spans ≥4× its footprint and any
  sky/canopy-gap texel (empty ⇒ depth 1.0) forces KEEP. So even though the prev-frame HZB
  *does* contain vox content, the cluster-level test kills almost nothing at oblique.
- **Brick granularity vs the mesh-only pyramid was decisive:** `?voxbocc` (per-brick test,
  default ON) measured **−17.2 eye / −6.0 oblique / ~0 aerial** at byte-identical output.
- **Brick granularity vs VOX content has never been measured.** The `voxOccPyr` is mesh-only
  when the scatter runs, so a brick fully behind *nearer vox* is invisible to every existing
  cull. Block-level waves (`?voxwaves`, pre-voxbocc) measured −0/−0.6/−4.7 (eye/obl/aerial)
  — block granularity fails against the porous canopy exactly as it failed against mesh.
  **Brick-vs-vox at oblique is the open cell of the 2×2 (granularity × occluder-content)
  matrix, and the only cull-side path with multi-ms oblique potential.**
- The prize bound: oblique foliage = 37.2 − 15.4 (noleaves) ≈ 21.8 ms; the 45-140 m per-tree
  ring alone holds ~10 ms (aggdist=60 diagnostic); oblique stacks crowns 3-8 deep, and the
  kernel is COUNT-bound, not fill-bound (aerial paints the same coverage at 659 clusters for
  5.4 ms; oblique needs 9188 clusters for ~20 ms). Killing buried bricks/blocks *before their
  Phase A/B work* is the mechanism voxbocc already proved; it just needs vox occluders in the
  pyramid. Honest expectation: −2..6 ms oblique gross (doc-11 estimate), overhead ~0.3-0.6 ms.

### 1.2 The refuted vehicle, and what survived it

`?voxf2b=1` (K=16 depth buckets, barrier-serialized near→far) measured **+5/+6/+16 ms**
(eye/oblique/aerial) — refuted as a vehicle. The post-mortem (doc 10, Mission 2) attributes
the cost to *occupancy collapse under serialization* (16 under-filled barrier-separated
stages; ~41 workgroups/stage at aerial), NOT to barriers themselves: `?voxwaves=4` added 3
full pyramid rebuilds + 3 submits over an F2B control for **+0.1 ms** at eye. Law:
**barriers are cheap; serialized under-filled stages are not.** Therefore anything built here
uses at most **2** sub-dispatches (K=2), each thousands of workgroups at oblique, and pays
~one pyramid rebuild (measured ≲0.1-0.3 ms) — never the K=16 chain.

### 1.3 The correctness bug this lever must also fix (doc-11 premise §4)

The perspective `sphereOccluded` (`NaniteHzb.ts:158-205`) clamps its prev-NDC footprint to
edge texels (`NaniteHzb.ts:189-192`) with **no on-screen guard**, while its own ortho variant
explicitly refuses to occlusion-cull off-map casters for exactly this reason
(`NaniteHzb.ts:246-247`). A cluster inside the CURRENT frustum whose prev-frame projection
falls off-screen (screen-edge entry during pans; any teleport) is tested against clamped edge
texels of a stale pyramid ⇒ **spurious cull ⇒ 1-frame pop-in at screen edges** and the
measured aerial pose-arrival over-cull ramp (first 1-3 frames 6.5-11.5 ms, then ~17). This is
a live correctness gap in the shipped single-phase scheme and it contaminates every
teleport-adjacent measurement. It is fixed as stage 1 below.

---

## 2. Design

### 2.1 Architecture choice: partition + defer, not a reprojected-seed pyramid

Two candidate architectures were on the table:

**(A) Prev-frame-seeded voxOccPyr** — warp last frame's full-content depth into the current
VP and seed the pyramid before the scatter. **Rejected.** The pyramid is in *election-key*
space (current-frame `depthKey24<<8|id8`); seeding it from prev-frame content requires
per-texel reprojection, which is either (i) non-conservative (forward-warp holes / stretched
disocclusions can present a phantom occluder over a pixel that is actually visible ⇒ a
dropped brick ⇒ a HOLE — a direct quality-law violation, same class as doc-11's
REJECTED-BY-POLICY "prev-frame single-phase brick cull"), or (ii) made conservative by
dilating with camera-motion bounds + pushing depth by a motion-dependent slack, which guts
the occluder exactly where the win lives (the whole far carpet moves several texels/frame at
oblique pan speeds). Any variant that lets prev-frame data *drop* work needs a re-test pass
to be safe, i.e. it degenerates into (B) with extra buffers.

**(B) Two-pass partition + defer (CHOSEN — UE5's two-pass occlusion mapped onto the vox
stack, with "deferral" replacing "re-test"):** in the vox fanout, classify each voxel
cluster against the **prev-frame full-content HZB** (which already exists and already
contains vox) and partition `qVoxRaster` into:

> **[REVIEW-FIX] BLOCKER FIXED — the partition test must NOT be the emit test.** The
> original draft partitioned with the *same conservative* `sphereOccluded` closure (same
> `instWorldSphere`+swayPad sphere, same `prevVp`/`prevCamPos` uniforms, same pyramid
> bytes) that the DAG traverse already applies at emit (`NaniteCull.ts:919-925`). But
> every cluster in `qRaster` — and therefore every `qVoxRaster` entry the fanout sees —
> has, THIS same frame, already tested NOT-occluded under exactly that closure against
> exactly those pyramid bytes (the HZB's only writer runs later, `NaniteFrame.ts:481`;
> `prevVp`/`prevCamPos` are written once per frame, `NaniteCommon.ts:108-115`). A
> deterministic recompute returns the same verdict, so **bucket 1 would be exactly
> empty**: pass B degenerates to a no-op and the whole lever is pure overhead. (Doc-11's
> L2 sketch carries the same latent flaw — its "bad partition = no gain" risk row
> understates this: the partition as sketched is not "bad", it is provably degenerate.)
> The fix is licensed by Q4: the partition drops nothing, so the classifier needs NO
> conservatism at all. Stage 2 therefore uses a **liberal variant**
> `sphereProbablyOccluded` (§2.4.0): identical math to `sphereOccluded` except the 2×2
> max-window is read `?voxprevlvl` (default 2, min 1) mip levels FINER than the
> diameter-fits-one-texel pick — it asks "is the CENTRAL region of the footprint behind
> prev content?" instead of "is the whole footprint conservatively occluded?", the
> latter being precisely the test every queue member already failed. At Δ=0 the
> classifier provably degenerates (empty bucket 1); the stage-2 gate measures bucket
> occupancy and sweeps Δ (§6, R9).

- **pass A** = "probably visible last frame" (bucket-1 test false) — scattered first,
  against the mesh-only `voxOccPyr` exactly as today;
- rebuild `voxOccPyr` (now **vox-inclusive**: it min-pools `visPayloadV`, which now holds
  pass A's elections);
- **pass B** = "probably occluded last frame" — scattered second; its per-BLOCK and
  per-BRICK culls now fire against real same-frame vox occluders.

**Nothing is ever dropped by the prev-frame test — it only picks the pass.** Disocclusion
needs no motion bounds, no dilation, no epsilon: a newly-disoccluded cluster is merely
mis-routed to pass B, where the same-frame conservative culls keep every brick with any
visible pixel. Correctness is *independent* of the prev-frame test's accuracy (§3); the test
only targets the occlusion win. This is why (B) needs no new correctness machinery at all.

UE5 correspondence: UE renders last frame's visible set, builds a fresh HZB from it, then
tests the remainder against that fresh HZB. Ours is the same dataflow with "last frame's
visible set" approximated by the §2.4.0 liberal prev-HZB centre test (UE keeps a persisted
per-cluster visibility bit; we have no per-(instance,cluster) persistent key space, so a
prev-HZB classifier is the S-effort stand-in) and "test the remainder" implemented as the
existing (proven-exact-conservative) block+brick culls against the rebuilt pyramid.

### 2.2 Reuse: the whole runtime already exists as the F2B/waves machinery at K=2

The F2B plumbing (`NaniteCull.ts:390-415, 545-710`) already implements: per-bucket count →
prefix → scatter into contiguous `qVoxRaster` slices, per-bucket indirect args, K baked
scatter instances (`NaniteVoxelRaster.ts:1437-1452`), and — under `?voxwaves` — pyramid
rebuilds *between* bucket chunks (`NaniteVoxelRaster.ts:1465-1486`). The two-pass design is
exactly this machinery with **K=2** and **bucket = prev-frame-visibility bit instead of
depth slab**. **[REVIEW-FIX]** The build touches: one ~25-line liberal-test closure in
`NaniteHzb` + three one-line wirings (§2.4.0), one flag, one bucket function, one
batch-list tweak, and one waves-default override. No new buffers, no new kernels, no
frame-graph reorder.

### 2.3 Stage 1 — the `sphereOccluded` off-screen guard (exact edit)

File `src/nanite/NaniteHzb.ts`:

1. At the top of `buildNaniteHzb` (after line 69's opening, next to the existing level-layout
   code at :71), add a build-time flag:

   ```ts
   // ?occg=0 reverts — off-screen guard for the PERSPECTIVE occlusion test (the ortho
   // variant already has one, :246-247). A cluster whose PREV-frame footprint is not
   // fully inside the prev screen has unknown occluders there ⇒ never cull it.
   const occGuard = new URLSearchParams(window.location.search).get('occg') !== '0';
   ```

2. Inside `sphereOccluded`, after `px`/`py` are computed (`NaniteHzb.ts:187-188`), add:

   ```ts
   // The level pick (:176) guarantees the sphere's footprint radius ≤ 0.5 texel at the
   // chosen level, so the footprint ⊆ [px−0.5, px+0.5]×[py−0.5, py+0.5]. Fully-inside ⇔
   // the 2×2 window below is unclamped ⇔ the whole sphere projected on-screen last frame.
   // [REVIEW-FIX] caveat: when levelF CLAMPS at levelCount−1 (giant spheres; the top level
   // is 1×1 at the canonical 2268×1473 — the layout loop :75-81 runs to 1×1 in 12 levels,
   // under MAX_LEVELS=16) the ≤0.5-texel guarantee lapses. There lw=lh=1 makes inPrev
   // require px==0.5 exactly ⇒ it ~always fails ⇒ KEEP — the conservative direction, and
   // those near-screen-sized spheres were effectively uncullable anyway (R6).
   const inPrev = px.greaterThanEqual(0.5)
     .and(px.lessThanEqual(toF(lw).sub(0.5)))
     .and(py.greaterThanEqual(0.5))
     .and(py.lessThanEqual(toF(lh).sub(0.5)));
   ```

3. Extend the return conjunction (`NaniteHzb.ts:200-204`): build-time compose —

   ```ts
   let result = dist.greaterThan(radius.mul(2))
     .and(nearClip.w.greaterThan(0))
     .and(centerClip.w.greaterThan(0))
     .and(nearestZ.greaterThan(maxZ));
   if (occGuard) result = result.and(inPrev);
   return result as unknown as NB;
   ```

Notes: guard direction is **keep-more** (can only ADD content); it is strictly stronger than
the ortho variant's centre-only guard (footprint-inside, needed because the prev pyramid has
no information for the off-screen part of the footprint). Perf cost: 4 compares + 3 ands per
emit-tested cluster — noise. Perf effect: near-zero at static poses (steady-state footprints
are interior); during pans/teleports it keeps entering clusters (the CORRECT image). It may
slightly raise the aerial median by removing spurious arrival-ramp culls — that is the bug
being fixed, not a regression (those cheap frames were missing-content frames). This guard
applies to every perspective caller of `sphereOccluded` — the camera DAG traverse
(`NaniteCull.ts:919-925`) and the stage-2 partition test — one fix, both sites.

### 2.4 Stage 2 — `?voxprev=1`: visibility-partitioned two-pass vox scatter (exact edits)

#### 2.4.0 `src/nanite/NaniteHzb.ts` — the liberal classifier `sphereProbablyOccluded` **[REVIEW-FIX]**

New closure, placed directly under `sphereOccluded` (after `NaniteHzb.ts:205`), added to
the `NaniteHzb` interface (`:43-59`, next to `sphereOccluded` at `:51`) and to the return
object (`:287`). It reads the SAME buffers (`hzbF.ro` + the `table`/`levelCountU`
uniforms) with the same 4 loads, so the §2.6 storage-buffer budget is unchanged — the
consuming kernels bind one HZB storage buffer either way.

```ts
// LIBERAL prev-frame visibility classifier for the ?voxprev two-pass vox partition.
// Deliberately NON-conservative: its output only ROUTES work between two passes that
// both end at the same exact-conservative same-frame culls (spec Q4), so false
// positives AND false negatives are perf-only. Identical math to sphereOccluded EXCEPT:
//   • the 2×2 window is read `prevFiner` mips FINER than the diameter-fits-one-texel
//     pick, so it tests the CENTRAL ~2^(1-Δ)·diameter of the footprint, not the whole
//     sphere. Δ=0 is degenerate BY CONSTRUCTION: every queued voxel cluster already
//     passed the conservative whole-footprint test at emit (NaniteCull.ts:919-925)
//     against these same pyramid bytes, so the Δ=0 verdict is false for all of them.
//   • off-screen/behind-camera prev projections return FALSE (route to pass A) via the
//     centre on-screen guard (the ortho variant's idiom, :246-247) — routing needs no
//     conservatism, and pass A is the behaves-like-today default.
const prevLvlRaw = Number(new URLSearchParams(window.location.search).get('voxprevlvl') ?? '2');
const prevFiner = Number.isFinite(prevLvlRaw) ? Math.max(1, Math.min(8, prevLvlRaw)) : 2;
const sphereProbablyOccluded = (
  center: NV3, radius: NF, vp: UniformMat4, camPos: UniformV3,
): NB => {
  const toCamera = camPos.sub(center).toVar();
  const dist = toCamera.length().toVar();
  const nearPoint = center.add(toCamera.div(dist).mul(radius));
  const nearClip = vp.mul(vec4(nearPoint, 1)).toVar();
  const centerClip = vp.mul(vec4(center, 1)).toVar();
  const nearestZ = nearClip.z.div(nearClip.w);
  const ndc = centerClip.xy.div(centerClip.w);
  const radiusTexels = radius.mul(cam.cotHalfFov).mul(cam.uH).div(4).div(dist);
  const levelF = radiusTexels.mul(2).max(1).log2().ceil()
    .sub(float(prevFiner))                    // ← the ONLY math change vs :176
    .clamp(0, levelCountU.sub(1));
  const info = table.element(uint(levelF));
  const lw = uint(info.y).toVar();
  const lh = uint(info.z).toVar();
  const lo = uint(info.x).toVar();
  const px = ndc.x.mul(0.5).add(0.5).mul(toF(lw));    // NO Y flip — buffer law (:182-186)
  const py = ndc.y.mul(0.5).add(0.5).mul(toF(lh));
  const x0 = uint(px.sub(0.5).clamp(0, toF(lw.sub(uint(1))))).toVar();
  const y0 = uint(py.sub(0.5).clamp(0, toF(lh.sub(uint(1))))).toVar();
  const x1 = minU(x0.add(uint(1)), lw.sub(uint(1)));
  const y1 = minU(y0.add(uint(1)), lh.sub(uint(1)));
  const z00 = hzbF.ro.element(lo.add(y0.mul(lw)).add(x0));
  const z01 = hzbF.ro.element(lo.add(y0.mul(lw)).add(x1));
  const z10 = hzbF.ro.element(lo.add(y1.mul(lw)).add(x0));
  const z11 = hzbF.ro.element(lo.add(y1.mul(lw)).add(x1));
  const maxZ = z00.max(z01).max(z10.max(z11));
  const onScreen = ndc.x.abs().lessThan(1).and(ndc.y.abs().lessThan(1));
  return onScreen
    .and(nearClip.w.greaterThan(0))
    .and(centerClip.w.greaterThan(0))
    .and(nearestZ.greaterThan(maxZ)) as unknown as NB;
};
```

Routing semantics (why this populates bucket 1 where the emit test cannot): the emit
test's window spans ≥2× the footprint diameter and max-pools, so ANY canopy-gap / sky
texel (empty ⇒ depth 1.0) anywhere in that oversized window forces "visible" — that is
doc-11's W2 bound-looseness, and it is why 100% of queue survivors fail Δ=0. At Δ=2 the
window covers only the central ~half-diameter: bucket 1 ⇔ the footprint CENTRE region sat
fully behind nearer prev-frame content. A front-carpet cluster's centre shows its OWN
prev-frame surface (own painted depth ≥ own nearestZ ⇒ maxZ ≥ nearestZ ⇒ bucket 0 —
correctly routed to pass A, where it becomes an occluder in pyramid #2); a buried
cluster's centre shows the nearer crown (that crown's depth < nearestZ ⇒ bucket 1 — pass
B, where the same-frame vox-inclusive culls harvest it). Larger Δ → closer to a centre
point-sample → more liberal → bigger pass B. `?voxprevlvl` is the stage-2 sweep knob (R9).

Wiring (one line each):
- `NaniteCull.ts` opts (`:219-242`): add `voxPrevTest?: SphereOccludedFn | null;`.
- `NaniteFrame.ts:206-214`: add `voxPrevTest: occl ? hzb.sphereProbablyOccluded : null`
  to the opts object at `:214`.
- `NaniteView.ts:72-78`: the same one-liner at `:77` (keeps the isolated harness in
  parity). Shadow culls (`NaniteShadow.ts:210`, `NaniteClipCull.ts:133`) pass no opts
  field ⇒ `voxPrevTest` undefined ⇒ untouched.

#### 2.4.1 `src/nanite/NaniteCull.ts`

**(a) Flag + K override** — at the existing param block (`NaniteCull.ts:292-328`):

```ts
// ?voxprev=1 — TWO-PASS vox scatter: partition qVoxRaster by the LIBERAL prev-frame
// visibility classifier (§2.4.0; bucket 0 = probably-visible, bucket 1 = probably-
// occluded), scattered as 2 waves with a voxOccPyr rebuild between. Rides the F2B
// plumbing at K=2. [REVIEW-FIX] Gated on opts.voxPrevTest, NOT on the emit-test
// sphereOccluded param — the emit test already filtered qRaster with that exact
// closure, so partitioning by it yields an exactly-empty bucket 1 (§2.1). Under
// ?occl=0 voxPrevTest is null ⇒ voxPrev false ⇒ voxprev is fully INERT (the plain
// unordered fanout + single scatter dispatch run; zero overhead).
const voxPrevTest = opts?.voxPrevTest ?? null;
const voxPrev = voxParams.get('voxprev') === '1' && voxPrevTest !== null;
const voxf2b = voxPrev || (voxParams.get('voxf2b') ?? '0') !== '0';   // was :307
...
const VOX_F2B_K = voxPrev ? 2
  : Math.min(32, Math.max(1, Number.isFinite(voxF2bKraw) ? voxF2bKraw : 16)); // was :328
```

**[REVIEW-FIX]** `voxPrevTest` is a NEW nullable opts field (`SphereOccludedFn` shape,
declared with the other opts at `NaniteCull.ts:219-242`) carrying the §2.4.0 liberal
closure. The emit-test parameter (`NaniteCull.ts:214`) stays exactly as-is (stage 1's
`?occg` guard applies to it at :919-925); the partition closure carries its own
centre-on-screen guard instead.

**(b) The partition function** — next to `voxClusterDepth` (`NaniteCull.ts:561-569`):

```ts
// TWO-PASS partition bit: 1 ⇔ the cluster's world sphere classifies PROBABLY-OCCLUDED
// against the PREV-frame full-content HZB via the §2.4.0 LIBERAL centre test
// (prevVp/prevCamPos — the same uniform pair the emit test at :919-925 uses; the
// classifier is deliberately NOT the emit closure, see §2.1). Sphere math mirrors the
// emit site exactly (instWorldSphere + swayPad, :864-868) so the partition classifies
// the same bound the raster will paint. [REVIEW-FIX] Self-contained: fetches the A/B
// instance words itself, mirroring voxClusterDepth (:562-563) — it is only ever called
// INSIDE the If(matClass==7) branch, so every node it builds (instance loads included)
// stays in the conditional subtree (the §2.7 hoist discipline; hoisting A/B above the
// If, as the draft suggested, would pay 2 dead instance loads per non-voxel entry AND
// contradict that discipline).
const voxPrevBucket = (instId: NU, ci: NU): NU => {
  const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
  const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
  const c = readCluster(gpu.clusters, ci);
  const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0));
  const swayPad = bcU2F(elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))));
  const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);
  const occ = voxPrevTest!(s.center, s.radius, cam.prevVp, cam.prevCamPos);
  return (occ as unknown as { select(a: NU, b: NU): NU }).select(uint(1), uint(0));
};
```

**(c) Bucket selection in the two histogram kernels** — build-time ternary, *inside* the
`If(matClass.equal(uint(VOXEL_MATCLASS)))` branch (TSL hoist hygiene — keep every node of the
occlusion test in the conditional subtree that consumes it, the same discipline the scatter's
election uses at `NaniteVoxelRaster.ts:1196-1215`):

- `kVoxCount` (`NaniteCull.ts:634-637`):

  ```ts
  const bIdx = voxPrev
    ? voxPrevBucket(instId, ci)   // [REVIEW-FIX] self-contained; builds inside the If
    : voxDepthBucket(voxClusterDepth(instId, ci));
  atomicAdd(voxBucketCount.element(bIdx), uint(1));
  ```

- `kVoxScatterFan` (`NaniteCull.ts:697-698`): the identical ternary for its `bIdx`.

**(d) Batch list** — `voxF2bBatch` (`NaniteCull.ts:1030-1037`): drop the depth-range pass
when partitioning by visibility (it only feeds `voxDepthBucket`):

```ts
const voxF2bBatch: unknown[] = [
  kVoxRangeArgs,                    // still needed: clears bucket counters + sizes the grid
  ...(voxPrev ? [] : [kVoxRange]),
  kVoxCount, kVoxPrefix, ...kVoxBucketArgs, kVoxScatterFan,
];
```

`kVoxRangeArgs`'s `voxRange` clears become dead writes under voxprev — harmless (2 words).
Everything else (`kVoxPrefix` at K=2, the 2 `kVoxBucketArgs`, the indirect tags at
`NaniteCull.ts:1024-1026`) is untouched.

#### 2.4.2 `src/nanite/NaniteVoxelRaster.ts`

**(e) Force the 2-wave dispatch** — at the `voxWaves` parse (`NaniteVoxelRaster.ts:269-270`):

```ts
const voxPrevR = new URLSearchParams(window.location.search).get('voxprev') === '1';
const voxWaves = voxPrevR ? 2
  : (Number.isFinite(voxWavesRaw) ? Math.min(16, Math.max(0, voxWavesRaw)) : 0);
```

Nothing else changes: `deps.voxF2bEnabled` arrives true (propagated from the cull via
`NaniteRaster.ts:1388-1392`), so the 2 per-bucket scatter instances are built
(`NaniteVoxelRaster.ts:1437-1452`), and `dispatchVoxel`'s existing waves path
(`NaniteVoxelRaster.ts:1465-1480`) with `kVoxScatterB.length=2, voxWaves=2 ⇒ per=1` executes
exactly the intended sequence:

```
dispatchBatch(voxPyrKernels)                    // pyramid #1: mesh-only (unchanged, :1461)
dispatchBatchMixed([kClearBins, kVoxScatterB0]) // pass A: probably-visible
dispatchBatch(voxPyrKernels)                    // pyramid #2: mesh + pass-A vox
dispatchBatchMixed([kVoxScatterB1])             // pass B: block+brick culls see vox
```

**Flag precedence** (document in both flag comments): `voxprev=1` forces `voxf2b` on, `K=2`,
`waves=2`; explicit `?voxf2bk`/`?voxwaves`/`?voxf2b` values are ignored while it is set.
**[REVIEW-FIX]** `voxprev` + `?occl=0` is fully INERT (`voxPrevTest` null ⇒ `voxPrev` false ⇒
plain unordered fanout + single scatter dispatch; the raster-side `voxWaves=2` override is
dead because `voxF2bEnabled` arrives false at `NaniteVoxelRaster.ts:1465`) — zero overhead,
not "all-bucket-0 with dead partition cost" as the draft said. `voxprevlvl` (§2.4.0) tunes
the classifier; it is read in `NaniteHzb` only. `voxprev=0` (default until stage 3) leaves
every existing path byte-for-byte identical — all edits are build-time gated.

### 2.5 What pass B actually saves (mechanism, for the reviewer)

For a pass-B cluster fully behind pass-A vox: the thread-0 block test
(`NaniteVoxelRaster.ts:613-673`) kills the whole 128-lane workgroup after ~4 pyramid loads
(Phase A per-brick work is gated by `wgVisible` at :690; Phase B's per-brick loop runs but
every record is empty ⇒ zero-trip inner loops). For partially-visible pass-B clusters, the
per-brick `voxbocc` test (:843-884) kills buried bricks at ~4 loads/brick, zeroing their
Phase-B footprint rounds. Both tests are the *shipped, proven-exact-conservative* code —
this lever changes only what the pyramid *contains* when they run.

### 2.6 Buffer budget (Metal ≤10-storage-buffer cliff)

Storage-buffer bindings after the edit (uniforms — `cam.*`, HZB level table — don't count):

- `kVoxCount`: qRaster.ro, counters, clusters, meshes, voxBucketCount **+ instances + hzb
  pyramid** = **7** ✓ (voxRange is NOT bound under voxprev — `voxDepthBucket` is compiled
  out by the build-time ternary; the depth path binds 6 today. The §2.4.0 classifier reads
  the SAME single hzb storage buffer `sphereOccluded` reads — `table`/`levelCountU`/
  `cam.prevVp`/`cam.prevCamPos` are uniforms and don't count. **[REVIEW-FIX]** counts
  re-verified against the closure swap: unchanged.)
- `kVoxScatterFan`: qRaster.ro, counters, clusters, meshes, voxBucketRange.ro, voxCursor,
  qVoxRaster.rw **+ instances + hzb pyramid** = **9** ✓ (tight — do not add anything else to
  this kernel; if a future need arises, move the bucket bit to a packed per-entry bitmask
  written by kVoxCount, which *frees* instances+hzb here at the cost of a 1 MB buffer)
- `kTraverse`, `kVoxScatter*`, pyramid kernels: untouched.

### 2.7 r184 / TSL codegen notes

- The batched fanout keeps the `setIndirectDispatch` tags (`NaniteCull.ts:1024-1026`,
  mechanism `Tsl.ts:283-289`) — removing `kVoxRange` from the array does not disturb the
  per-node `dispatchSize` attach; array order remains execution order
  (`Tsl.ts:296-306`).
- Build the occlusion-test nodes inside the `If(matClass==7)` branch (hoist hazard: a
  `.toVar()` first built outside the conditional subtree that consumes it gets hoisted and
  Metal-compiles dead loads for every non-voxel entry; same pathology class as the election's
  `candL` note at `NaniteVoxelRaster.ts:1196-1201`).
- `sphereOccluded` internally `.toVar()`s — fine inside the branch; it is already
  instantiated per-call-site in `kTraverse`.
- No 64-bit atomics anywhere in this design; the election idiom is untouched.

---

## 3. Correctness + quality-equivalence argument

The user law is absolute; here is the full argument, piece by piece.

**(Q1) The partition cannot change the work-item SET.** Each voxel qRaster entry is assigned
to exactly one bucket and appended exactly once (count==scatter through the prefix,
`NaniteCull.ts:649-710`) — the same invariant the shipped F2B path holds. The bucket function
is deterministic within the frame: its inputs are uniforms (`prevVp`, `prevCamPos` — written
once per frame at `NaniteCommon.ts:108-115`), static geometry buffers, and the HZB buffer,
which is written ONLY by `hzb.build` at `NaniteFrame.ts:481` — *after* `runVoxFanout`
(`NaniteFrame.ts:468`) in the frame graph, so both histogram kernels read identical pyramid
bytes. Inherited caveat (identical in shipped F2B): `kVoxCount` and `kVoxScatterFan`
recompute the bucket independently; a cross-kernel FP divergence on a borderline value would
trip the `off < cnt` drop-guard (`NaniteCull.ts:704`). The expression trees are identical and
this exact pattern is shipped + shot-verified for the depth bucket; the stage-2 shotdiff gate
re-verifies it for the visibility bucket. Hardening fallback if it ever flips: single-source
the bit (kVoxCount writes a packed per-entry bitmask; scatter reads it — see §2.6).

**(Q2) Reordering cannot change any election winner.** The election is
`atomicMax(visPayloadV, depthKey24<<8|id8)` + winner-guarded `visBV` store
(`NaniteVoxelRaster.ts:1196-1215`, ray path :1394-1403) — max over a fixed multiset of
(pixel, key) pairs is order-invariant; the relaxed-load early-outs (:1203, :1266-1267) only
skip provably-losing work (`cand` is an upper bound on every per-pixel key of its brick).
This is the shipped F2B loss-exactness argument, unchanged.

**(Q3) Pass-B culls remain exact-conservative against a mid-frame pyramid.** The pyramid
min-pools election keys; keys at any pixel only *grow* over the frame (atomicMax). Pyramid #2
is a snapshot ⇒ its pooled window-min `occK_snap ≤ occK_final`. A block/brick is dropped iff
`bNearKey ≤ occK` where `bNearKey` (front-slab key `|0xff` keep-on-tie) upper-bounds every
key the item could elect at any covered pixel, and the min-pool lower-bounds the current
winner at every covered pixel (empty texel ⇒ 0 ⇒ KEEP; the 2×2-at-covering-level window is a
coverage superset). So a dropped item would have lost the strict-greater election at every
footprint pixel against content that is already ≥ `occK_snap` and only grows ⇒ removing it
leaves every final (payload, visB) word bit-identical. This is precisely the shipped voxbocc
proof (`NaniteVoxelRaster.ts:588-612` comment block); it is content-agnostic — adding vox
occluders to the pyramid does not touch it.

**(Q4) The prev-frame data cannot cause a drop, ever.** It routes items between two passes
that both reach the same conservative machinery. A 100%-wrong partition (teleport, first
frame, `?occl=0` fallback) degrades to today's behaviour + overhead — never to a pixel
change. No camera-motion bounds are needed anywhere; disocclusion is handled structurally.

**(Q5) Tie-break identity.** `id8` = low 8 bits of the qVoxRaster item index
(`NaniteVoxelRaster.ts:891-894`), and the partition re-orders item indices, so a pixel where
two DIFFERENT bricks tie on the full 24-bit depth key could elect the other brick vs the
default path (same class of reordering as shipped F2B, whose A/B was verified
byte-identical). The stage-2 gate shot-diffs all three poses at threshold 0; if a diff ever
appears it must be verified to be exact-tie pixels only and then surfaced for the user's
call (never silently accepted) — with the stable-tiebreak rework (id8 from a per-cluster
hash) as the ready fix.

**(Q6) Stage 1 is strictly conservative.** The guard only converts "occluded" verdicts to
"kept" — it can only add geometry that the current frustum says is present but the stale
clamped-edge sample wrongly culled. **[REVIEW-FIX]** Static isolated poses: verdicts are
unchanged for every cluster whose prev footprint is interior — but clusters STRADDLING the
screen edge exist at settled poses too (prev=current there), and for those the guard flips
today's clamped-window verdict to KEEP. The honest static expectation is therefore:
bit-identical everywhere EXCEPT possibly ADDED content hugging the screen border — content
the `?occl=0` render shows and today's clamped-edge test wrongly culls. The stage-1 gate
(§6) is written to that expectation (additions-only, border-only, matches occl=0 ground
truth), not to a blanket zero-diff that would mislabel the fix as a failure. Moving: it
removes an existing 1-frame pop-in artifact class — a quality *improvement* mandated by
the quality law, not a change to argue about.

---

## 4. Risks + fallbacks

| # | risk | likelihood/impact | mitigation / fallback |
|---|------|-------------------|-----------------------|
| R1 | The oblique carpet is mostly genuinely visible at brick granularity ⇒ occlusion gain caps ≈ −1 ms | the main unknown; this is why stage 0 is measure-first with zero code | stage-0 decision rule (§6); if refuted, the lever is abandoned CHEAPLY (no code written) and the oblique budget shifts to the ring/aggregation lever (doc 13 §premise 3) |
| R2 | 2-sub-dispatch serialization + extra pyramid rebuild (+1 net dispatchVoxel submit, −2 fanout submits — net −1/frame **[REVIEW-FIX]**) eat the win at eye/aerial | bounded by measurement: rebuild ≲0.1-0.3 ms, K=2 chain tax extrapolates +0.6/+0.75/+2 (eye/obl/aer) from K16 — but the aerial +2 extrapolation assumed depth-slab occupancy collapse; the visibility split at aerial puts ~everything in pass A (top-down ⇒ prev-visible ≈ visible), so pass B is near-empty and the tax ≈ rebuild + barrier only | stage-2 gate requires eye/aerial regression < 0.5 ms; if aerial regresses more, add a build-time `voxprevmin` (skip wave-2 rebuild when bucket-1 count < N via a tiny indirect-args guard kernel) — only if needed |
| R3 | Partition quality poor under fast motion (everything lands "visible") | perf-only (Q4); live metric already dominated by other motion costs | acceptable; two-pass cluster re-test (stage 4) is the structural fix for motion |
| R4 | fp divergence between kVoxCount/kVoxScatterFan bucket recompute (Q1 caveat) | very low (shipped F2B precedent), impact = dropped cluster = pixels | shotdiff gate at 0-threshold; hardening fallback = packed bitmask single-source (§2.6) |
| R5 | Tie-flip pixels vs default path (Q5) | low; F2B precedent measured byte-identical | shotdiff; stable-tiebreak fallback; surface any residual to the user |
| R6 | Stage-1 guard removes real (legitimate) culls of huge clusters whose coarse-level footprint fails the fully-inside test | perf-only, tiny: those clusters were effectively uncullable anyway (2×2 max window over near-whole-screen ⇒ sky texel forces keep, doc-11 premise §1) | measured by the stage-1 gate; `?occg=0` reverts |
| R7 | 10-buffer cliff on kVoxScatterFan (at 9) | build-time explosion if someone adds a binding later | budget documented in §2.6 + comment at the kernel; bitmask variant frees 2 |
| R8 | `?voxprev` interacts with a user-set `?voxf2b/voxf2bk/voxwaves` | confusion only | precedence documented in flag comments (§2.4.2); voxprev wins |
| R9 | **[REVIEW-FIX]** partition classifier degenerate or imbalanced: bucket 1 ≈ empty ⇒ pass B a no-op ⇒ pure overhead (GUARANTEED at Δ=0 — the draft's blocker, fixed in §2.1/§2.4.0); or bucket 0 ≈ empty ⇒ pyramid #2 has ~no vox occluders ⇒ no gain | the central mechanism risk after the fix; never a quality risk (Q4) | stage-2 gate makes the per-bucket `nanite.voxB0/voxB1` counters MANDATORY (§6) and sweeps `?voxprevlvl` 1→4; ship only a Δ whose oblique bucket-1 share is material (≥20%) AND whose perf gate passes; a perf null with B≈0 is a classifier failure, NOT evidence about occlusion |

**PREMISE-AUDIT rule (mandatory before any "refuted/disappointing" verdict at any stage):**
go up one level before varying the method. Specifically: (a) a null stage-0 result does NOT
refute brick-vs-vox occlusion — it refutes the *depth-slab proxy* of it (the near/far split
can straddle the canopy surface so wave-1 never contains the occluders; the visibility
partition is exactly the fix for that flaw). Before abandoning after a null stage-0, check
the per-bucket cluster counts (add `nanite.voxB0/voxB1` to the meter via the existing
`readVoxCount` idiom — CPU-only, 20 lines) and only conclude "no occlusion to harvest" if
wave-1 actually contained the front carpet. (b) A disappointing stage-2 result must first
re-check the metric (same-session ordered A/B, oblique bimodality ±2 ms ⇒ medians over 32
frames + per-frame arrays, meter readbacks contaminate 2/32 frames), the params (is `?occl`
on? did `voxprev` actually force K=2/waves=2 — verify via the submit count in a
WebGPU-inspector capture or the `[probe] url` line), and the upstream structure (is the
partition balanced? counters above) before blaming the mechanism. (c) Surface any
park/drop decision explicitly — never silently.

---

## 5. Staged landing plan

Each stage is independently gated, flag-gated, and revertible. Order is deliberate: stage 0
is free; stage 1 is a correctness fix that also de-noises the gates for everything after.

| stage | what | flag | code size |
|---|---|---|---|
| **0** | discriminating probe: does brick-vs-vox occlusion exist at oblique? | existing `voxf2b=1,voxf2bk=2[,voxwaves=2]` | **zero** |
| **1** | `sphereOccluded` off-screen guard (§2.3) | `occg` (default ON, `=0` reverts) | ~10 lines |
| **2** | visibility-partitioned two-pass vox scatter (§2.4, incl. the §2.4.0 liberal classifier, its `?voxprevlvl` knob, and the mandatory voxB0/voxB1 counters) | `voxprev=1` (default OFF) | ~90 lines **[REVIEW-FIX]** |
| **3** | default flip after gates | `voxprev` default ON, `=0` reverts | 1 line |
| **4** | *(optional, separate follow-up)* cluster-level record+re-test vs fresh HZB — the mesh-cluster half of UE two-pass; scaffolding exists (`rejClust` caps `NaniteCull.ts:98-99`, comment :743-744, `rasterDispatch2`/`p2Appends` plumbing :419-425, :482-488). Isolated ~0; targets the live-motion stale-HZB inflation (~9-11 ms measured 2026-06-26, pre-voxbocc). Spec it only after stages 1-3 land and a live-moving `occl=0` A/B re-quantifies the prize. | `clust2p=1` | M effort |

---

## 6. Measurement gates

Discipline (memory `nanite-perf-canonical-config-and-baseline` + thermal rules): 200k trees,
dpr 1.5, isolated poses via `tools/probe-fresh-stutter.ts`, SERIAL runs, baselines FIRST in a
session (thermal bias lands against the candidate), medians over 32 frames, same-session
deltas only. Oblique bimodality makes single medians ±2 ms soft — quote deltas vs the
same-session control and eyeball the per-frame arrays in the JSON.

### Stage 0 — zero-code discriminator (run before writing ANY code)

```
CONFIG=default LABEL=pfo0-base TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default EXTRA=voxf2b=1,voxf2bk=2 LABEL=pfo0-k2ctl TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=pfo0-k2w2 TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
```

- `k2ctl − base` = the K=2 partition/serialization tax (predicted ≈ +0.6/+0.75/+2).
- `k2ctl − k2w2` (per pose) = **the money number**: brick-granular vox-behind-vox occlusion
  gain net of one pyramid rebuild, under a *depth* partition.
- **Decision rule:** oblique gain ≥ 1.5 ms ⇒ build stages 1-2 (visibility partition should
  meet or beat it with a smaller tax). Gain 0.5-1.5 ⇒ build anyway (the depth split
  under-measures the mechanism — §4 premise-audit (a)), expectation lowered. Gain ≤ 0.5 ⇒
  run the premise-audit checks (§4), then surface to the user with the per-bucket counters
  before abandoning.

### Stage 1 gate

```
CONFIG=default LABEL=pfo1-ctl TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default EXTRA=occg=1 LABEL=pfo1-on TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts   # (occg default ON once landed — ctl is `occg=0`)
npx tsx tools/diff.ts --a <scratch>/shots/pfo1-ctl-eye.png --b <scratch>/shots/pfo1-on-eye.png --out /tmp/d-eye.png --thr 1
# … same for oblique/aerial shots
```

- PASS (perf): isolated medians within ±0.5 ms/pose; the aerial per-frame array LOSES the
  frame-0..2 arrival dip (6-12 ms cheap frames) — that dip disappearing is the observable
  fix. Optional visual: teleport-adjacent screenshots (frame 1-2 after `setPose`) no longer
  show edge pop-in.
- PASS (quality) **[REVIEW-FIX]**: shotdiff 0 ⇒ pass outright. A NONZERO static diff is
  EXPECTED to be possible (Q6: settled edge-straddlers flip cull→keep) and is acceptable
  only if ALL THREE hold: (i) additions only (the occg=1 image is a superset of occg=0),
  (ii) confined to a screen-border band, (iii) the added pixels match a `?occl=0` render at
  the same pose (they are content today's clamped-edge test wrongly culls — restored, not
  invented). That is the bug being FIXED; surface it to the user WITH the diff images
  (never silently accept — memory `surface-decisions-never-park-silently`). Any interior or
  subtractive diff = FAIL (guard mis-wired).
- FAIL any pose > +1 ms: investigate which culls were lost (R6) before shipping; the guard
  is correctness-mandated, so a real cost here is surfaced to the user, not silently traded.

### Stage 2 gate (same session, this order)

```
CONFIG=default LABEL=pfo2-ctl TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
CONFIG=default EXTRA=voxprev=1 LABEL=pfo2-on TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts
npx tsx tools/diff.ts --a <scratch>/shots/pfo2-ctl-oblique.png --b <scratch>/shots/pfo2-on-oblique.png --out /tmp/d-obl.png --thr 1   # + eye, aerial
```

- **Partition sanity (hard, read BEFORE the perf numbers)** **[REVIEW-FIX]**: the stage-2
  build must add per-bucket meter counters `nanite.voxB0/voxB1` (read `voxBucketRangeAttr`
  slots via the existing `readVoxCount`/`readBuffer` idiom, `NaniteCull.ts:1050-1053` —
  CPU-only, ~20 lines). At oblique, bucket 1 must be a material share (≥20% of vox
  clusters). If B1 ≈ 0: the classifier is degenerate (R9) — first CHECK the wiring (is
  `voxPrevTest` the §2.4.0 closure and not the emit-test closure? Δ ≥ 1?), then sweep
  `?voxprevlvl=2,3,4`. Do NOT interpret any perf number while B1 ≈ 0.
- **Quality gate (hard):** all three pose shotdiffs pixel-identical (Q5 caveat: any nonzero
  diff ⇒ verify tie-pixels-only, then STOP and surface; do not ship on "close enough").
  Run at the stage-1-landed baseline on BOTH sides (occg default-on in ctl and candidate)
  so stage-1's border additions can't masquerade as a voxprev diff. **[REVIEW-FIX]**
- **Perf gate:** oblique med ≤ ctl − 1.5 ms AND eye med ≥ ctl − 0.5 ms is not required
  (wins welcome) but eye/aerial must not regress > 0.5 ms. Compare per-frame arrays, not
  just medians (bimodality). Sweep `?voxprevlvl` (2/3/4) at oblique only, then confirm the
  winner on the full trio.
- **Live sanity (before stage 3):**
  `CONFIG=default EXTRA=voxprev=1 LABEL=pfo2-live TICKS=600 COOLDOWN_S=45 TREES=200000 …`
  vs a ctl live run — live p50/p95 must not regress. **[REVIEW-FIX]** Submit-count is
  actually NET −1 vs today's default (fanout: 3 submits → 1 batched `voxF2bBatch`;
  dispatchVoxel: 3 → 4 — pyramid, clear+B0, pyramid, B1); the real added cost is the
  second voxOccPyr rebuild (GPU ≲0.1-0.3 ms, doc-11 measured) + one wave barrier, not "2
  extra submits/frame" as the draft claimed.

### Stage 3 gate

Re-run the canonical trio (`pfo3-default`) with the flipped default + one full-canonical
confirm at the next rested session; update the baseline docs; keep `voxprev=0` as the
permanent A/B control.

---

## 7. Expected outcome (headline)

- **Oblique:** −2..5 ms net (gross brick-vs-vox occlusion 2-6 ms on the ~21.8 ms foliage
  share with 3-8-deep crown stacking; minus ~0.3-0.6 ms partition+rebuild tax).
  Point estimate: **−3 ms** (37.2 → ~34.2). **[REVIEW-FIX]** Conditional on the §2.4.0
  classifier routing a material bucket-1 share at oblique (R9 — the draft's classifier
  routed exactly 0%); the estimate assumes brick-vs-vox behaves like the measured
  brick-vs-mesh (voxbocc −6.0 oblique) on the buried fraction. Stage 0's depth-split
  probe bounds the mechanism before any code; the voxB0/voxB1 counters bound the
  classifier while building.
- **Eye:** 0..−1 ms (eye vox already mesh-hidden and voxbocc'd; pass B small).
- **Aerial:** −0..2 ms (vertical stacking exists, but pass B is small top-down; tax ~0.3).
- **Live p95:** stage 1 removes a 1-frame-hole artifact class + arrival-ramp over-cull;
  stage 4 (optional) is the real live-motion lever.
- Not the whole −11: the remaining oblique gap is generated one level up (the 45-140 m
  per-tree ring's cluster count — aggregation-area lever) plus the post stack; this lever is
  the cull-side share of the portfolio.

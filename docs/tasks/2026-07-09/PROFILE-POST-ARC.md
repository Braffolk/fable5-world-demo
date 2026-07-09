# Post-arc profile — 2026-07-09, 2s capture at the census worst pose

Trace: `/tmp/laas_trace_postarc-2026-07-09T17-38-53-c000.gputrace` (raw) + `…-c000-perf.gputrace`
(user Xcode export, embedded perf data). Pose: `?cam=-582.1,302.4,1006.1,2.5692,-0.0077`
(census-post-ladder worst: dense-forest look-across). World config, grass=0, nanshadow=0, dpr2.
Numbers below transcribed from the user's Xcode cost list + per-shader breakdowns (screenshots);
`run_all.sh` per-line results land in the sibling `profile-results-*` folder.

## Cost ranking (per-shader % of sampled GPU, Xcode)

| # | shader | cost% | regs | type | note |
|---|---|--:|--:|---|---|
| 1 | vertex_nanRaster_world1_cl | 15.66 | 96 | Vertex | ⚠️ 80-BYTE SPILL (user-observed); the clhw HW-draw vertex-pull |
| 2 | compute_nanMidRaster | 14.02 | 64 | Compute | |
| 3 | fragment_nanRaster_world1_cl | 7.86 | 20 | Fragment | HW-draw frag (election) |
| 4 | compute_probeGather | 7.62 | 136 | Compute | lighting/GI probes (non-nanite) |
| 5 | compute_nanProjectVerts | 7.38 | 70 | Compute | **was 23-30% pre-arc — the whale, slain** |
| 6 | fragment_HalfResMRT | 4.43 | 141 | Fragment | post stack |
| 7 | compute_nanVoxScatterB1 | 4.31 | 145 | Compute | |
| 8 | fragment_naniteResolve_tri | 3.97 | 208 | Fragment | ⚠️ 208 regs |
| 9 | fragment_PMREM_ggx | 2.91 | 60 | Fragment | ⚠️ env prefilter AT RUNTIME? suspicious |
| 10 | compute_nanTraverseAB | 2.89 | 56 | Compute | cull |
| 11 | compute_nanRasterWorld1 | 2.80 | 56 | Compute | the classifier — de-über vindicated |
| 12 | compute_nanVoxScatterB0 | 2.13 | 145 | Compute | |
| 13 | compute_nanTraverseBA | 2.06 | 56 | Compute | |
| 14 | fragment_TRAA.resolve | 1.69 | 92 | Fragment | |
| 15 | fragment_RTT | 1.66 | 75 | Fragment | |
| 16 | vertex_nanRaster_world1 | 0.99 | 90 | Vertex | non-cl HW vertex |
| 17 | fragment_nanRaster_world1 | 0.93 | 20 | Fragment | |
| 18 | fragment_RenderPipeline | 0.74 | 46 | Fragment | |

## Per-shader instruction breakdowns (Xcode, % of shader cost)

### 1. vertex_nanRaster_world1_cl — 868 instr, 96 regs, 80 B spill
Sync Wait Memory **45.14%** (40) · Mem Load 15.60% (45) · ALU F 12.70% (422) · ALU I 9.04% (152) ·
Complex 2.62% (27) · Other 1.93% (52) · Shift 1.69% (30) · Bool 0.89% (20) · Sample 0.88% (11) ·
Branch 2.97% (23). **Memory-latency-bound + spilling: the per-vertex full ctx/wind/displacement
re-decode disease, in a VERTEX shader (no wgcache/shared-mem tools there).**

### 2. compute_nanMidRaster — 328 instr, 64 regs
ALU Integer **38.27%** (172) · Branch 9.78% (13) · ALU F 8.31% (12) · Sync Wait 5.77% (14) ·
Load 4.51% (35) · Other 4.41% (21) · Bool 2.75% (10) · Atomic 2.52% (2). Still the per-tri
setup-recompute integer whale IN-shader (172 int-instr ≈ the old 176 — the f0cc6a6 crest cuts were
value-preserving, small ALU); its FRAME share fell 25.8→14% because crown-LOD removed most mid tris.

### 3. fragment_nanRaster_world1_cl — 96 instr, 20 regs
ALU F 21.28% · Sync Wait 10.28% · **Sync Atomic 9.60% (2)** · ALU I 12.43% · Complex 7.56% (1) ·
Load 3.38%. The per-pixel election in the HW frag.

### 4. compute_probeGather — 1816 instr, 136 regs
Sync Wait Memory **50.63%** (15) · Memory Sample **29.11%** (41) · ALU F 12.98% (1255).
Texture-sample-latency-bound. Non-nanite (GI/probe stack).

### 5. compute_nanProjectVerts — 739 instr, 70 regs
Mem Load **36.36%** (54) · Sync Wait **35.70%** (29) · Barrier 2.15% (2) · ALU I 2.48% (150) ·
ALU F 2.39% (330). Honest vertex-gather-bound now; the broadcast-read wall is GONE (Lever A).

### 6. fragment_HalfResMRT — 6127 instr (!), 141 regs
ALU F **32.39%** (3800) · Sync Wait 21.54% (93) · ALU I 10.85% · Other 8.98% · Load 8.46% ·
Complex 4.89% (142). Enormous instruction count; post-stack arc (#63) territory.

### 7. compute_nanVoxScatterB1 — 419 instr, 145 regs
ALU F **48.14%** (270) · Complex **15.78%** (31) · Half 3.33% · Sync Wait 13.56% · Sample 3.97%.
Float-ALU-bound (the slab/ray math + projection); 145 regs.

### 8. fragment_naniteResolve_tri — 621 instr, **208 regs**
Sync Wait 36.23% (23) · ALU I 22.28% (122) · Load 17.36% (49) · Store 4.72% · ALU F 1.45% (213).
Register monster (occupancy floor) but only 3.97% of frame.

### 11. compute_nanRasterWorld1 (classifier) — 451 instr, 56 regs
Mem Load **31.03%** (28) · Sync Wait 27.23% (21) · ALU I 17.52% (194) · Branch 2.05% ·
Atomic 1.08% (10). 2.80% of frame at 56 regs — the vis-buffer rewrite's end-state, vindicated.

## Read of the ranking (coordinator)
- **Wins booked:** projection 23-30% → 7.38% (Lever 1 + Lever A + crown-LOD); classifier 2.8% @56;
  mid frame-share 25.8 → 14.02% (fewer tris via crown-LOD; in-shader recompute unchanged by design —
  fat-record VETOED for VRAM, fusion VETOED for world1 occupancy).
- **NEW #1: the clhw HW-draw VERTEX shader (15.66%, 96 regs, 80 B spill, 45% mem-wait).** Census
  corroborates: clhwTris peak 6.66M (+23%) — the HW-cluster path carries trunks/near-terrain and is
  now the top whale. Disease = per-VERTEX full cluster-ctx/wind/disp re-decode (the same pathology
  world1-compute had pre-rewrite), but vertex shaders lack workgroup-shared tools ⇒ the lever is
  structural: precompute per-cluster/per-instance decoded data (a tiny compute pre-pass writing a
  decoded-vertex or decoded-ctx buffer the vertex shader reads flat), and/or kill the 80 B spill
  (96 regs). The old parked idea "A′ = per-INSTANCE wind precompute" is exactly this family.
- **The post/lighting stack aggregates ~15%** (probeGather 7.62 + HalfResMRT 4.43 + PMREM_ggx 2.91):
  task #63's arc. ⚠️ PMREM_ggx at runtime in a 2 s steady-state capture is SUSPICIOUS — env
  prefiltering should be boot/ToD-change-only, not per-frame; possible free ~3%.
- naniteResolve_tri's 208 regs is an occupancy affront but only ~4% — park.
- Remaining mid levers: ?middz A/B (built, off), ?trihzb occlusion (quality risk) — both gated.

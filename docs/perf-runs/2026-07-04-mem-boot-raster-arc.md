# 2026-07-04 — MEM + BOOT + RASTER arc (user: "3 massive issues")

User mandate: (1) startup mem up to 8GB → cut 5-10×; (2) buffer/vertex memory
too high, suspected too-many-unique-trees → cap 16-32 + instancing; (3) SW
raster ~40% of frame vs Nanite's ~10% = massive smell. (4, added) startup
extremely slow + Chrome frozen-tab dialogs.

## Diagnoses (3 parallel audit agents)

- **Unique-trees hypothesis DEAD**: geometry already template-shared — 20
  unique crowns / ~178k instances, ~100 B per instance. The "per-location
  uniqueness" that costs is FARTILES (3417 tiles, 8.0M bricks).
- **8GB =** fartile packed slab (866MB f64 CPU + 297MB GPU + mirror) +
  BrickCPU JS-object grids (~1.5GB, empty bricks emitted as objects) +
  voxelizer dense scratch churn (~492MB/crown × 20) + worst-case cull queues
  (403MB GPU + 403MB mirrors) + permanent CPU mirrors of all storage buffers
  (~1GB) + VegLib pools (100-300MB).
- **Boot =** deterministic derived data rebuilt EVERY boot on the MAIN thread,
  cached only as intermediate object graphs. Cold: 36-48s crown-voxelize +
  clusterize slab, 15s DAG slab, VegLibrary sync builds. Warm frozen-tab: the
  zero-yield build/append slab (reg.build→appends→flush). The dags IDB store
  was 36s of mid-boot structured-clone churn.
- **Raster =** loop walks every bbox row (~100 ALU + 3 fp div/row), ±1px span
  pads, no divergence handling; vox splat kernel already had the cooperative
  pattern.

## Shipped (commits a5c7bbb, 0955d2d)

- **Queues −786MB** (403 GPU + 403 mirrors → 19 total): high-waters measured
  at 4 poses + walk + boot = ~90k worst vs 8.4M caps; new caps ~2-3× worst,
  world-only; ?qrcap/?qfrontier/?qvcap/?qshcap/?qshfrontier; appends were
  already guarded; graceful-degrade verified; kSplat latent clamp bug fixed;
  __qHW instrumentation kept.
- **Boot: cold 103→~65s, warm ~29s, frozen-tab GONE** (worst task 2.5s):
  crowns → 'crown' jobs on DagWorker pool via three-free VoxelBrickCore.ts;
  veg DAGs → 'mesh'/'aggregate' jobs; VegLibrary+prep overlapped with GPU
  phases; 250ms yieldIfDue everywhere; world-veg placement-independent cache
  key; cache stores post-first-frame; BootTrace.ts phase table + stall
  heartbeat (window.__bootTrace).
- **Mem: peak cold 7351→~6180MB + 1145MB mirrors released** (by construction;
  web-API metric can't see ArrayBuffer stores): BrickCPU graph killed (packed
  9×u32 straight into GPU array, empties = 0 bytes); brick 108→36B; fartile
  pack 866→288MB; CACHE_REV 2; persistent per-worker voxelizer scratch;
  releaseImmutableMirrors (verts 861 + voxelBricks 284MB, ?noreleasemirrors
  escape; mutable idx/clusters/dag/hfVerts/inst keep mirrors). Accepted
  trade: 243-brick 0.003% far-field splat drift (below visual floor;
  dial-back = f32 lanes 56B).

## Raster: two hypotheses REFUTED honestly (uncommitted ?swcoop in tree)

- Cooperative Phase A/B (the audit's #1 fix): **+17/+11ms WORSE** — barriers +
  ~7KB extra shared mem (16.5KB overflow fixed via i16 packing; thread-0
  compaction, no TSL workgroup atomics) cost more than divergence; cluster
  tris are size-coherent so the 16×16-stalls-32-lanes case is rare.
- Row-solve removal (two-bin, small≤4px full walk): **neutral ±0.2ms** — the
  ~100-ALU row machinery was never the wall.
- Identity: scar fragment counters BIT-EQUAL across modes (oblique
  6,871,171); shots at/below ctl floor. ?swcoop 0/1/2 kept (default → 0 at
  commit), ?rdbg forces 0 (uniformity).
- **OPEN: ?relect ablation running** — splits loop cost into walk/math vs
  guard-load vs atomic-RMW vs occupancy (?wgcache=0); premise check: if
  nothing recovers 7.8/18.4, the rdbg2-derived "loop is the cost" number is
  itself suspect (rdbg legs kill HZB feedback + shift vox load).

## Open / next

- Fartile residency ladder/streaming (all levels of all 3417 tiles resident
  forever; only ~hundreds in the usable band; terrain tile pool = the model;
  ~−220MB GPU + boot peak; unblocks ftcell 0.6). NOTE: interacts with mirror
  release (streaming bricks needs a write path; resident pool + per-tile IDB
  records).
- Re-profile settled heap post-landing (web-API pinned at 3887MB regardless —
  metric quantized; need CDP snapshot with ArrayBuffer attribution) before
  claiming the 5-10× target; VegLib pool release + DagBuild release are the
  next known slabs.
- VegLibrary is now the cold critical path (~12-17s) and is deterministic →
  cacheable (next boot win). ForestScene still builds sync (can adopt worker
  kinds).
- Remaining >1s boot tasks: hero buildTree 2.1s, reg.build+flush 2.5s, grass
  ray bake 1.6-2.5s, first-frame shader compile.
- Machine-contention hazard all session: 3+ agents benching/booting
  concurrently — perf A/Bs need quiet-machine re-verification before any
  number is treated as canonical.

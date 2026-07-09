/**
 * ?scene=forest — a CONTROLLED renderer testbed: 200k full-detail trees, our
 * nanite cull→raster→resolve, and NOTHING else (no terrain, water, sky, GI, CSM,
 * post). Built to step-by-step close the gap to the reference compute-rasterizer
 * (reference/ref-tree.html): same kind of workload (a dense forest of instanced
 * trees), isolated so the only cost measured is OUR cull + raster.
 *
 * The trees are real VegLibrary crowns (full leaf density), shared per species and
 * INSTANCED across a jittered grid — uniqueness from per-instance transform, like
 * the reference's helmets. Renders through buildNaniteView (flat/cluster resolve,
 * no PBR/GI/CSM) so the measurement isolates the geometry pipeline.
 *
 * Knobs:  ?trees=N (default 200000) · ?spacing=M (default 4 m) · ?leafdensity=N
 *         (per-crown anchors, default 4000 = full) · ?nanitedbg=cluster|flat
 *         (default cluster — per-meshlet colours show the flood) · ?dag=0 (no LOD,
 *         raw full geometry) · ?occl=0 · ?loderr=N (the screen-error cut τ).
 * HUD:    nanite.visClusters / chunks / hwTris (the numbers we're driving down).
 */

import { FloatType, StorageTexture } from 'three/webgpu';
import type { WorldContext } from './Scenes';
import { buildVegLibrary, type VegPool } from '../vegetation/VegLibrary';
import {
  GeometryRegistry,
  DAG_VERT_STRIDE,
  MAX_CLUSTER_TRIS,
  explicitToDagVerts,
  setClusterTriCap,
} from '../nanite/GeometryRegistry';
import { type DagBuild, buildDag, meshletizeDag } from '../nanite/BuildDag';
import {
  BootCache,
  type PackedPreparedCrown,
  type PackedFarTile,
  packPreparedCrown,
  unpackPreparedCrown,
  packFarTiles,
  unpackFarTiles,
} from '../nanite/BootCache';
import { buildAggregateDag, setAggLodErrorK } from '../nanite/BuildAggregateDag';
import {
  appendFarTiles,
  buildFarTilesAsync,
  DEFAULT_AGG_DIST,
  DEFAULT_FT_CELL,
  FT_TILE_SIZE,
  type FarTileSpecies,
} from '../nanite/FarTiles';
import type { BrickCPU } from '../nanite/VoxelBrick';
import { setClusterFill } from '../nanite/Clusterize';
import { DEFAULT_TRANSITION_DIST, geometryToSource } from '../nanite/WorldRegistry';
import {
  appendVoxelCrown,
  DEFAULT_VOXEL_GRID_DIM,
  type PreparedVoxelCrown,
  computeVoxlodAnchorL0,
  prepareVoxelCrown,
  releaseVoxelizerScratch,
  setVoxlodConfig,
  setVoxOccThreshold,
  voxOccThreshold,
  voxlodLevels,
} from '../nanite/VoxelizeCrown';
import { Vector2 } from 'three';
import { internalSize } from '../render/RenderScale';
import { buildNaniteView } from '../nanite/NaniteView';
import { Heightfield } from '../world/Heightfield';
import { SunSky } from '../sky/SunSky';
import { PostStack } from '../render/PostStack';
import { updateSunUniforms } from '../render/VegMaterials';
import { setWindContext, windU } from '../render/Wind';

/** per-species leaf tint → matParam (linear RGB low 3 bytes + hueVar high byte) */
function packLeafTint(c: { r: number; g: number; b: number; hueVar: number }): number {
  const u8 = (x: number): number => Math.max(0, Math.min(255, Math.round(x * 255)));
  return (u8(c.r) | (u8(c.g) << 8) | (u8(c.b) << 16) | (u8(c.hueVar) << 24)) >>> 0;
}

export async function buildForestScene(ctx: WorldContext): Promise<void> {
  const { engine, seed } = ctx;
  const q = new URLSearchParams(window.location.search);
  // boot stage map ([forest][boot] lines) — feeds the boot-cache design (which stages
  // are worth caching) and any future boot-regression triage.
  const tBoot0 = performance.now();
  let tBootPrev = tBoot0;
  const bootStage = (label: string): void => {
    const now = performance.now();
    // eslint-disable-next-line no-console
    console.log(`[forest][boot] ${label} +${(now - tBootPrev).toFixed(0)}ms (cum ${((now - tBoot0) / 1000).toFixed(1)}s)`);
    tBootPrev = now;
  };
  const nTrees = Math.max(1, Math.floor(Number(q.get('trees') ?? '200000')));
  const spacing = Number(q.get('spacing') ?? '4');
  const leafDensity = Math.max(1, Math.floor(Number(q.get('leafdensity') ?? '4000')));
  const wantDag = q.get('dag') !== '0';
  const mode = (q.get('nanitedbg') as 'flat' | 'cluster' | 'lod' | null) ?? 'cluster';
  // A/B knobs — MUST be set before the DAG builds + the nanite shaders build (all read the
  // cap live). ForestScene has its OWN build path (not buildWorldRegistry), so it wires these
  // itself; without this `?clustertris`/`?clusterfill` are silently ignored here.
  setClusterTriCap(Number(q.get('clustertris')) || 256);
  setClusterFill(Number(q.get('clusterfill')) || 0.95);

  // voxel-foliage (spec §3 / Stage 3a): the canonical perf config is ?scene=forest, so the
  // mesh→voxel transition is wired HERE (ForestScene has its OWN build path). Voxelize each
  // leaf crown OFFLINE, register a voxel:7 sibling head over the same instances, and hand
  // the leaf head off to the voxel head at transitionDist. ON by default at scene=forest so
  // the NET-WIN is measurable at the canonical config; ?voxreg=0 = pure-triangle A/B.
  //   ?voxreg=0   — disable voxels (pure-triangle baseline)
  //   ?forcevox   — DEBUG: voxel EVERYWHERE (suppress the leaf head; nearDist=0)
  //   ?voxnear=M  — mesh→voxel handoff distance (m), default DEFAULT_TRANSITION_DIST (~35)
  //   ?voxgrid=N  — voxel DETAIL (cell edge count / crown), default DEFAULT_VOXEL_GRID_DIM
  const forceVox = q.get('forcevox') !== null;
  const voxOn = q.get('voxreg') !== '0' || forceVox;
  // ?noleaves — DEBUG ablation: render TRUNKS/BARK ONLY (no leaf crown at all — neither the
  // triangle leaf head nor its voxel sibling). For isolating how much of the forest frame the
  // foliage (voxel crowns) actually costs vs the woody-skeleton triangles + the cull.
  const noLeaves = q.get('noleaves') !== null;
  // DEFAULT_VOXEL_GRID_DIM = 256 (2026-07-03: the beautification pick is the SHARED
  // engine default now — world scene rides the same value; see VoxelizeCrown.ts).
  const voxGridDim = Number(q.get('voxgrid') ?? DEFAULT_VOXEL_GRID_DIM) || DEFAULT_VOXEL_GRID_DIM;
  // DEFAULT_TRANSITION_DIST = 60 (2026-07-03: the beautification pick is the SHARED
  // engine default now — history of the 35→90→45→60 sweeps lives on the constant in
  // WorldRegistry.ts + docs/perf-runs/2026-07-02-beautification.md).
  const transitionDist = Number(q.get('voxnear') ?? DEFAULT_TRANSITION_DIST) || DEFAULT_TRANSITION_DIST;
  // ?voxlod (G1, DEFAULT ON): voxel MIP pyramid + a REAL multi-level DAG (UE5-style: far coarsens
  // the SAME crown through a band-anchored octave ladder, near refines, picked by the screen-error
  // cut). ?voxlod=0 forces the old single-level degenerate always-cut DAG (the A/B baseline).
  const voxLod = q.get('voxlod') !== '0';
  // ANCHOR the ladder to the band (correction 1): ownError(L0) = transitionDist*tau/projK so the
  // FINEST level's cut lands at the mesh→voxel handoff and each octave of distance descends one
  // level (spans [35,2000] m). projK mirrors the cull (cot(fovY/2)*renderHeight*0.5).
  {
    const anchorL0 = computeVoxlodAnchorL0(
      transitionDist,
      internalSize(engine.renderer, new Vector2()).y, // ?rscale: τ anchor follows render res
      engine.camera.fov,
    );
    // ?voxlodk= (anchor multiplier) / ?voxlodlevels= / ?voxlodsparse= / ?voxlodshell= sweep the
    // ladder for A/B; unset = the band-anchored defaults (7 levels, K=1, sparse off, shell off).
    const kRaw = q.get('voxlodk');
    const lRaw = q.get('voxlodlevels');
    const spRaw = q.get('voxlodsparse');
    const shRaw = q.get('voxlodshell');
    setVoxlodConfig({
      anchorL0,
      errorK: kRaw !== null ? Number(kRaw) : undefined,
      levels: lRaw !== null ? Number(lRaw) : undefined,
      sparseK: spRaw !== null ? Number(spRaw) : undefined,
      shell: shRaw !== null ? Number(shRaw) : undefined,
    });
    // ?voxocc= — occupancy coverage threshold (crown slimming; see VoxelizeCrown).
    const occRaw = q.get('voxocc');
    if (occRaw !== null) setVoxOccThreshold(Number(occRaw));
  }
  // ?fartiles=1 (wave 3, EXPERIMENTAL): cross-instance far-field aggregation — beyond
  // ?aggdist (default 140 m) whole 64 m tiles of trees render as ONE merged voxel head
  // (crowns + trunk columns splatted at boot; see FarTiles.ts). Collapses far cluster/brick
  // counts by orders of magnitude AND extends the forest past the instMinPx (~300 m) edge.
  // DEFAULT ON (2026-07-02 wave 3): 200k A/B eye 40.2→27.4 / oblique 38.5→31.3 / aerial
  // 26.0→19.5 ms, aerial whole-frame clusters 30k→290, forest extends past the old ~300 m
  // instMinPx pop-out to the horizon. ?fartiles=0 reverts to per-tree-only.
  const farTilesOn = q.get('fartiles') !== '0';
  const aggDist = Number(q.get('aggdist') ?? DEFAULT_AGG_DIST) || DEFAULT_AGG_DIST;
  // ?leaflodk= — aggregate LEAF ladder error scale (see BuildAggregateDag AGG_LOD_CFG). The
  // 2026-07-01 cost-map found the leaf-mesh band (<35 m) renders LOD0 everywhere (~10.3M of
  // 12.4M eye visTris) because the ladder's L1 cut lands beyond the voxel handoff; K<1 pulls
  // coarsening in-band (0.25 ≈ L1 at ~14 m). Baked at DAG build; set before buildAggregateDag.
  {
    // DEFAULT 0.4 lives in BuildAggregateDag AGG_LOD_CFG (SHARED forest+world since
    // 2026-07-03); ?leaflodk= is the per-boot override.
    const lk = q.get('leaflodk');
    if (lk !== null) setAggLodErrorK(Number(lk));
  }

  // ── tree geometry (real crowns, full leaf density) ────────────────────────
  ctx.progress(0.1, 'forest: building veg library');
  const lib = await buildVegLibrary(
    engine.renderer,
    seed,
    (p, m) => ctx.progress(0.1 + p * 0.4, m),
    // impostors:false — ForestScene never builds an ImpostorRuntime, so the octahedral bake
    // (6 species × 192 GPU renders + readbacks) was pure wasted boot time (~confirmed 2026-07-01).
    { leafAnchorTarget: leafDensity, impostors: false },
  );
  // canopy species with a bark trunk + a real leaf crown (cls 0–4)
  const pools = lib.pools.filter((p): p is VegPool => p.cls <= 4 && !!p.r0?.[0] && !!p.leaf);
  if (pools.length === 0) throw new Error('forest: no canopy tree pools with leaf crowns');

  // ── register bark + leaf per species ──────────────────────────────────────
  bootStage('veg library (trees gen + bark textures)');
  ctx.progress(0.5, 'forest: registering tree meshes');
  // ── boot cache (DDC): crown voxelizations / DAG builds / fartiles splat ────
  // key = builder-source hash + every param feeding those builds (BootCache.ts).
  // anchorL0 folds in viewport height + fov (voxlod ladder anchor); seed covers
  // the veg geometry; leaflodk/caps cover the DAG shapes.
  const bootCache = new BootCache({
    params: {
      seed: seed.seed,
      nTrees,
      spacing,
      leafDensity,
      wantDag,
      noLeaves,
      voxOn,
      forceVox,
      voxGridDim,
      voxLod,
      transitionDist,
      farTilesOn,
      aggDist,
      // RESOLVED ftcell, not the raw query param: a raw-null knob made a code-default
      // change (0.75→0.6, 2026-07-02) silently HIT the stale cache entry.
      ftCell: Number(q.get('ftcell') ?? DEFAULT_FT_CELL) || DEFAULT_FT_CELL,
      voxOcc: voxOccThreshold(), // resolved (post-setVoxOccThreshold) for the same reason
      anchorH: internalSize(engine.renderer, new Vector2()).y,
      fov: engine.camera.fov,
      knobs: ['voxlodk', 'voxlodlevels', 'voxlodsparse', 'voxlodshell', 'leaflodk', 'clustertris', 'clusterfill'].map(
        (k) => q.get(k),
      ),
    },
  });
  const cachedCrowns = voxOn && !noLeaves ? await bootCache.get<(PackedPreparedCrown | null)[]>('crowns') : null;
  const crownPacks: (PackedPreparedCrown | null)[] = [];
  const reg = new GeometryRegistry();
  const dagJobs: { handle: number; build: () => DagBuild }[] = [];
  // voxel-foliage (§5.2/§5.3): per-species voxelization collected here so the brick budget
  // can be reserved BEFORE build() (addLate freezes caps) and the voxel:7 sibling heads
  // appended AFTER. The leaf head index lets the transition hand it off post-build.
  const toVoxel: { poolIdx: number; leafHead: number; prep: PreparedVoxelCrown; src: ReturnType<typeof geometryToSource>; matParam: number }[] = [];
  const meshes = pools.map((pool, poolIdx) => {
    const barkPart = pool.r0?.[0];
    if (!barkPart) throw new Error('forest: bark part missing');
    const barkSrc = geometryToSource(barkPart.geo);
    const bark = reg.registerMesh(barkSrc, 'bark', {
      transformChannel: 'trunk',
      castShadows: false,
      swayPad: 3.8,
      matParam: pool.barkLayer ?? 0,
      label: `c${pool.cls}/bark`,
    });
    const leafSrc = geometryToSource(pool.leaf!.geo);
    const leafTint = packLeafTint(pool.leaf!.color);
    const leaf = reg.registerMesh(leafSrc, 'leaf', {
      transformChannel: 'leaf',
      castShadows: false,
      twoSided: true,
      swayPad: 3.8,
      matParam: leafTint,
      aggregate: true,
      label: `c${pool.cls}/leaf`,
    });
    reg.setMaxDistance(bark, 2000);
    // Stage-3a: the LEAF head culls beyond transitionDist when this crown is voxelized
    // (its voxel sibling owns mid/far); pure-triangle (voxreg=0) keeps the full envelope.
    reg.setMaxDistance(leaf, noLeaves ? 0.001 : voxOn ? transitionDist : 2000);
    if (voxOn && !noLeaves) {
      const cached = cachedCrowns?.[poolIdx];
      const prep = cached ? unpackPreparedCrown(cached) : prepareVoxelCrown(leafSrc, pool.leaf!.color, voxGridDim, voxLod);
      if (!cached) crownPacks[poolIdx] = packPreparedCrown(prep);
      // a real leaf crown always voxelizes to >0 bricks; guard a degenerate empty crown
      // (registerVoxelHead throws on 0 blocks) so the leaf keeps its full mesh envelope.
      if (prep.brickCount > 0) {
        toVoxel.push({ poolIdx, leafHead: leaf, prep, src: leafSrc, matParam: leafTint });
      } else {
        reg.setMaxDistance(leaf, 2000);
      }
    }
    if (wantDag) {
      dagJobs.push({
        handle: bark,
        build: () =>
          buildDag(explicitToDagVerts(barkSrc), DAG_VERT_STRIDE, barkSrc.indices, {
            normalOffset: 3,
            maxTris: MAX_CLUSTER_TRIS,
          }),
      });
      dagJobs.push({
        handle: leaf,
        build: () =>
          buildAggregateDag(explicitToDagVerts(leafSrc), DAG_VERT_STRIDE, leafSrc.indices, {
            seed: seed.seed,
            maxTris: MAX_CLUSTER_TRIS,
            // crown-LOD Phase 2 (2026-07-08): ?crownlod0's default flipped OFF (the
            // world near-crown mesh ladder is now default-on). ForestScene still uses
            // the OLD grow-based aggregate, whose coarse levels are the spiky/merged
            // crowns crownlod0 used to hide — so pin it LOD0-only here to preserve
            // ForestScene's appearance. (A ladder bring-up for ?scene=forest is a
            // follow-up; the world scene is the Phase-2 target.)
            maxLevels: 1,
          }),
      });
    }
    return { bark, leaf };
  });

  // ── plant a jittered grid, species spatially mixed; per-mesh streams must be
  //    contiguous, so collect per-pool instance lists then bind each once ─────
  bootStage('mesh registration + crown voxelization prep');
  if (!cachedCrowns && voxOn && !noLeaves && crownPacks.length > 0) {
    void bootCache.put('crowns', pools.map((_, i) => crownPacks[i] ?? null));
  }
  ctx.progress(0.6, `forest: planting ${nTrees} trees`);
  const side = Math.ceil(Math.sqrt(nTrees));
  const half = (side * spacing) / 2;
  const rng = seed.rng('forest/plant');
  const lists = pools.map(() => ({ a: [] as number[], b: [] as number[] }));
  let planted = 0;
  for (let gz = 0; gz < side && planted < nTrees; gz++) {
    for (let gx = 0; gx < side && planted < nTrees; gx++) {
      const jx = (rng.float() - 0.5) * spacing * 0.85;
      const jz = (rng.float() - 0.5) * spacing * 0.85;
      const wx = gx * spacing - half + jx;
      const wz = gz * spacing - half + jz;
      const scale = 0.8 + rng.float() * 0.6;
      const yaw = rng.float() * Math.PI * 2;
      const p = ((gx * 73856093) ^ (gz * 19349663)) >>> 0;
      const list = lists[p % pools.length];
      if (!list) continue;
      list.a.push(wx, 0, wz, scale);
      list.b.push(yaw, 0, 0, 0);
      planted++;
    }
  }
  // bind the SAME instance stream to each species' bark + leaf (trunk + crown). Keep the
  // per-pool arrays so the voxel sibling head can re-bind the SAME stream post-build.
  const poolStreams: ({ a: Float32Array; b: Float32Array } | null)[] = pools.map(() => null);
  for (let i = 0; i < pools.length; i++) {
    const list = lists[i];
    const m = meshes[i];
    if (!list || !m || list.a.length === 0) continue;
    const a = new Float32Array(list.a);
    const b = new Float32Array(list.b);
    poolStreams[i] = { a, b };
    reg.bindInstances(m.bark, { a, b });
    reg.bindInstances(m.leaf, { a, b });
  }

  // ── DAG build (sync; ~0.8 s/crown @ 4000) → addLate → build → attach ──────
  bootStage('planting + instance streams');
  const builds: { handle: number; dag: DagBuild }[] = [];
  if (dagJobs.length > 0) {
    ctx.progress(0.75, `forest: building ${dagJobs.length} LOD DAGs`);
    // boot cache: DagBuild[] in dagJobs order (handles are re-derived from THIS boot's
    // registration, which always runs — only the expensive build() is skipped on hit).
    const cachedDags = await bootCache.getMany<DagBuild>('dags');
    const usable = cachedDags && cachedDags.length === dagJobs.length ? cachedDags : null;
    let lateV = 0;
    let lateT = 0;
    let lateC = 0;
    for (let ji = 0; ji < dagJobs.length; ji++) {
      const job = dagJobs[ji]!;
      // MESHLET-LOCAL indexing (task #76 projVertBuf dedup) — same as WorldRegistry: the
      // shared deduped raster keys mesh clusters on vi−vBase, which needs each cluster's
      // verts contiguous. Render-neutral reorder; idempotent (cache-safe).
      const dag = meshletizeDag(usable ? usable[ji]! : job.build());
      lateV += dag.verts.length / DAG_VERT_STRIDE;
      lateT += dag.indices.length / 3;
      lateC += dag.clusters.length;
      builds.push({ handle: job.handle, dag });
    }
    reg.addLate({ verts: lateV, tris: lateT, clusters: lateC });
    if (!usable) void bootCache.putMany('dags', builds.map((b) => b.dag));
  }
  // voxel-foliage (§5.3 HARD precondition): reserve the brick budget + voxel sibling heads
  // (1 mesh + 1 instance stream + ceil(bricks/128) clusters each) BEFORE build() freezes the
  // late caps. Σ occupied bricks across the per-species crowns. The append happens post-build.
  if (toVoxel.length > 0) {
    let lateBricks = 0;
    let lateVoxClusters = 0;
    let lateVoxInst = 0;
    for (const v of toVoxel) {
      lateBricks += v.prep.brickCount;
      lateVoxClusters += v.prep.clusterCount;
      lateVoxInst += poolStreams[v.poolIdx]?.a.length ? (poolStreams[v.poolIdx] as { a: Float32Array }).a.length / 4 : 0;
    }
    reg.addLate({ bricks: lateBricks, meshes: toVoxel.length, instances: lateVoxInst, clusters: lateVoxClusters });
    // eslint-disable-next-line no-console
    console.log(
      `[forest] voxel-foliage: reserving ${lateBricks} bricks across ${toVoxel.length} crowns ` +
        `(grid ${voxGridDim}${voxLod ? `, voxlod ${voxlodLevels()}L` : ''}) = ${((lateBricks * 5 * 4) / (1024 * 1024)).toFixed(2)} MB, ` +
        `+${lateVoxClusters} clusters / +${lateVoxInst} instances / +${toVoxel.length} voxel:7 heads, ` +
        `handoff ${transitionDist} m${forceVox ? ' (?forcevox: voxel-only)' : ''}`,
    );
  }
  // ?fartiles=1 (wave 3): build the far-tile aggregation pre-build (needs exact counts for
  // the reservation), append post-build. Each tile = 64 m of trees merged into one voxel
  // head; per-tree heads then END at aggDist (their maxDist is clamped in the append loop
  // below and setMaxDistance for bark).
  let farTiles: import('../nanite/FarTiles').FarTileBuild[] = [];
  if (farTilesOn && toVoxel.length > 0 && !noLeaves) {
    const tFt0 = performance.now();
    // 0.75 m cells (not 0.5): 200k extrapolates to ~10.7M bricks at 0.5 (386 MB — over the
    // 256 MB buffer cliff); 0.75 lands ~4-5M (~170 MB). ?voxcell renders CELLS, so visible
    // granularity at the 140 m handoff is ~0.75 m ≈ 8 px — close to the per-tree side.
    // DEFAULT_FT_CELL = 0.6 (2026-07-03: SHARED default, see FarTiles.ts for the
    // 0.75→0.6 rationale + the 0.5 cold-boot rejection).
    const TILE_CELL = Number(q.get('ftcell') ?? DEFAULT_FT_CELL) || DEFAULT_FT_CELL;
    const ftPools: { a: Float32Array; b: Float32Array; species: FarTileSpecies }[] = [];
    for (const v of toVoxel) {
      const s = poolStreams[v.poolIdx];
      const levels = v.prep.vox.levels;
      if (!s || !levels || levels.length === 0) continue;
      // pick the crown pyramid level whose brick size best matches the tile cell size
      let pick = 0;
      let bestD = Infinity;
      for (let L = 0; L < levels.length; L++) {
        const bw = (levels[L] as { cellSize: number }).cellSize * 4;
        const d = Math.abs(bw - TILE_CELL);
        if (d < bestD) {
          bestD = d;
          pick = L;
        }
      }
      const lvl = levels[pick] as { bricks: BrickCPU[]; occupied: number[] };
      const bricks = lvl.occupied.map((i) => lvl.bricks[i] as BrickCPU);
      let crownMinY = 2;
      for (const b of bricks) crownMinY = Math.min(crownMinY, b.center[1] - b.half);
      ftPools.push({
        a: s.a,
        b: s.b,
        species: { bricks, crownMinY: Math.max(0.5, crownMinY), bark: { r: 0.42, g: 0.33, b: 0.24 } },
      });
    }
    // wave 4: worker-pool splat (36-42 s single-threaded → ~core-count× less wall);
    // falls back to the sync build internally on any worker failure.
    const cachedFt = await bootCache.get<PackedFarTile[]>('fartiles');
    if (cachedFt) {
      farTiles = unpackFarTiles(cachedFt);
    } else {
      farTiles = await buildFarTilesAsync({ tileSize: FT_TILE_SIZE, cellSize: TILE_CELL, pools: ftPools });
      void bootCache.put('fartiles', packFarTiles(farTiles));
    }
    let ftBricks = 0;
    let ftClusters = 0;
    for (const t of farTiles) {
      ftBricks += t.prep.brickCount;
      ftClusters += t.prep.clusterCount;
    }
    reg.addLate({ bricks: ftBricks, meshes: farTiles.length, instances: farTiles.length, clusters: ftClusters });
    // eslint-disable-next-line no-console
    console.log(
      `[forest] fartiles: ${farTiles.length} tiles, ${ftBricks} bricks (${((ftBricks * 36) / 1048576).toFixed(1)} MB), ` +
        `${ftClusters} clusters, aggDist ${aggDist} m, built in ${(performance.now() - tFt0).toFixed(0)} ms`,
    );
  }
  bootStage('LOD DAG builds + fartiles splat');
  // C (memory arc): the inline (main-thread) crown voxelizations are done — drop the
  // persistent ~490 MB cell-accumulator scratch before it sits resident all session.
  releaseVoxelizerScratch();
  ctx.progress(0.9, 'forest: building registry');
  const report = reg.build(engine.renderer, engine.stats.counters);
  bootStage('registry build + GPU upload');
  for (const b of builds) reg.attachDag(b.handle, b.dag);
  // voxel-foliage (§5.2 / Stage 3a): append each crown's bricks + register a voxel:7 sibling
  // head over the SAME instances now that build() froze the caps, then hand off the leaf head
  // to the voxel head at transitionDist (the leaf maxDist was already lowered above). ?forcevox
  // suppresses the leaf entirely (voxel everywhere, nearDist=0). flush() uploads the late ranges.
  if (toVoxel.length > 0) {
    const tVox0 = performance.now();
    let appended = 0;
    for (const v of toVoxel) {
      const r = appendVoxelCrown(reg, v.prep, v.src, {
        matParam: v.matParam,
        swayPad: 3.8,
        // ?fartiles: the per-tree voxel crown ENDS at aggDist — the merged tile head owns
        // the far field beyond it (ranges overlap by the tile radius, see FarTiles.ts).
        maxDist: farTilesOn && farTiles.length > 0 ? aggDist : 2000,
        nearDist: forceVox ? 0 : transitionDist,
        label: `c${pools[v.poolIdx]?.cls}/voxel`,
      });
      appended += r.brickCount;
      const s = poolStreams[v.poolIdx];
      if (s) reg.bindInstances(r.head, { a: s.a, b: s.b });
      // ?forcevox DEBUG: the voxel head renders everywhere ⇒ suppress the leaf head.
      if (forceVox) reg.setMaxDistance(v.leafHead, 0.001);
    }
    // ?fartiles: clamp the per-tree BARK envelope to aggDist too, then append the tiles.
    if (farTilesOn && farTiles.length > 0) {
      for (const m of meshes) reg.setMaxDistance(m.bark, aggDist);
      const ftTint = toVoxel[0]?.matParam ?? 0;
      const ftBricks = appendFarTiles(reg, farTiles, { nearDist: Math.max(10, aggDist - 46), matParam: ftTint });
      // eslint-disable-next-line no-console
      console.log(`[forest] fartiles: appended ${ftBricks} bricks across ${farTiles.length} tile heads`);
      farTiles = []; // release the CPU-side pyramids
    }
    reg.flush(engine.renderer, engine.stats.counters);
    // eslint-disable-next-line no-console
    console.log(
      `[forest] voxel-foliage: appended ${appended} bricks (${reg.brickCount}/${reg.brickCapacity}) ` +
        `+ ${toVoxel.length} voxel:7 heads in ${(performance.now() - tVox0).toFixed(0)} ms`,
    );
  }
  // eslint-disable-next-line no-console
  console.log(
    `[forest] ${planted} trees · ${pools.length} species · ${report.meshes} meshes / ` +
      `${reg.instanceCount} instances / ${(report.tris / 1e6).toFixed(2)}M tris · ` +
      `DAG ${builds.length} · dbg=${mode} dag=${wantDag}`,
  );
  // per-mesh DAG shape: lod0 cluster count → root count (= the MINIMUM clusters a
  // tree of that mesh can emit when the cut reaches its coarsest level). If roots
  // ≫ 1, the aggregate isn't collapsing — a 2nd over-emission source beyond the
  // error-vs-size cut. (Leaf = the aggregate DAG; bark = QEM.)
  for (let i = 0; i < pools.length; i++) {
    const m = meshes[i];
    if (!m) continue;
    const bk = builds.find((b) => b.handle === m.bark)?.dag.stats;
    const lf = builds.find((b) => b.handle === m.leaf)?.dag.stats;
    if (bk || lf) {
      // eslint-disable-next-line no-console
      console.log(
        `[forest] c${pools[i]?.cls}: bark lod0 ${bk?.lod0Clusters ?? '-'}→root ${bk?.roots ?? '-'} ` +
          `(${bk?.totalClusters ?? '-'}cl/${bk?.levels ?? '-'}lvl) · ` +
          `leaf lod0 ${lf?.lod0Clusters ?? '-'}→root ${lf?.roots ?? '-'} ` +
          `(${lf?.totalClusters ?? '-'}cl/${lf?.levels ?? '-'}lvl)`,
      );
    }
  }

  // ── render path ───────────────────────────────────────────────────────────
  // DEFAULT (`?nanite=1` without `?nanitedbg`) = the STANDARD WHOLE PIPE on isolated
  // trees: the real NaniteFrame (world1 raster → NaniteResolve bark/leaf PBR shading →
  // PostStack), so tree render speed can be profiled through the actual pipeline, not
  // just the flat debug resolve. `?nanitedbg=cluster|flat|lod` keeps the lean NaniteView.
  const nanitedbg = q.get('nanitedbg');
  const fullFrame = q.get('nanite') === '1' && !nanitedbg && q.get('naniteframe') !== '0';

  if (fullFrame) {
    // The forest has NO terrain clusters, so the resolve's required hf maps are bound
    // but NEVER sampled (terrain shading is gated on terrain pixels). Generate a real
    // heightfield purely so those bindings are valid + correctly formatted; the trees
    // sit at y=0 over it (the terrain itself is not in the registry, so not rendered).
    // GI/CSM/canopy = null (no GI bounce, no shadows — forest trees are castShadows:false
    // anyway; a shadow profile is a separate follow-up). Bark textures come from VegLib.
    ctx.progress(0.93, 'forest: env for the full pipe (heightfield + sky + post)');
    const hf = await Heightfield.generate(engine.renderer, ctx.params, seed, (p, m) =>
      ctx.progress(0.93 + p * 0.05, m),
    );
    bootStage('vox append + heightfield');
    const bootTod = ctx.params.timeOfDay;
    const sunSky = new SunSky(engine, bootTod);
    await sunSky.init(engine.renderer);
    updateSunUniforms(sunSky.sun);
    // trunk/leaf sway reads a module-global wind context (set by the world scene) — the
    // raster + resolve both sample it, so it must exist before buildNaniteFrame (the
    // vegetation materials THROW without it — the context cannot be skipped).
    if (hf.noiseA) setWindContext({ noiseA: hf.noiseA, canopyTex: null });
    // ?wind=N strength override (same knob as TerrainScene). ?wind=0 = still air —
    // every sway term scales by strength, so 0 is exactly static. MEASUREMENT USE:
    // sway rides three's wall-clock `time` node, so two probe runs are never at the
    // same gust phase; screenshot-equivalence gates (spec-orchestration-submit-folds
    // §4) need a still scene to compare builds. Default unchanged.
    {
      const ws = Number(q.get('wind') ?? NaN);
      if (Number.isFinite(ws)) windU.strength.value = ws;
    }
    const post = new PostStack(engine, sunSky.atmosphere, bootTod);
    const { buildNaniteFrame } = await import('../nanite/NaniteFrame');
    const frame = buildNaniteFrame(engine, reg, hf, post, {
      gi: null,
      canopyTex: null,
      csm: null,
      barkTexA: lib.barkArray?.texA ?? null,
      barkTexB: lib.barkArray?.texB ?? null,
    });
    bootStage('sky + post + NaniteFrame build');
    engine.post = frame as unknown as typeof engine.post;
    // meter() is driven once/frame by Engine.renderStep (this.post.meter) — same as
    // the world scene. Do NOT also wire it via onUpdate or it dispatches autoExposure
    // (and the cull/raster/shadow readbacks) TWICE per frame.
    // eslint-disable-next-line no-console
    console.log('[forest] FULL-FRAME pipe (NaniteFrame resolve + post) — ?nanitedbg=cluster for the lean debug view');
  } else {
    // lean cull→raster→flat-resolve debug view (dummy 1×1 heightTex — never sampled).
    const heightTex = new StorageTexture(1, 1);
    heightTex.name = 'forestSceneHeight';
    heightTex.type = FloatType;
    const hf = { heightTex } as unknown as Heightfield;
    const view = buildNaniteView(engine, reg, hf, mode);
    engine.post = view as unknown as typeof engine.post;
    // metered once/frame by Engine.renderStep (this.post.meter) — see note above.
  }

  // camera inside the forest at eye height, looking horizontally
  ctx.hooks.initialPose = { p: [0, 2, 0], yaw: 0.6, pitch: -0.02 };
  ctx.hooks.initialPoseMode = 'fly';
  engine.camera.position.set(0, 2, 0);
  ctx.hooks.groundProbe = () => ({ ground: 0, water: -1e9 });
  ctx.progress(1, 'forest: ready');
}

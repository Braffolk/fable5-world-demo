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
import { type DagBuild, buildDag } from '../nanite/BuildDag';
import { buildAggregateDag } from '../nanite/BuildAggregateDag';
import { setClusterFill } from '../nanite/Clusterize';
import { DEFAULT_TRANSITION_DIST, geometryToSource } from '../nanite/WorldRegistry';
import {
  appendVoxelCrown,
  DEFAULT_VOXEL_GRID_DIM,
  type PreparedVoxelCrown,
  computeVoxlodAnchorL0,
  prepareVoxelCrown,
  setVoxlodConfig,
  voxlodLevels,
} from '../nanite/VoxelizeCrown';
import { Vector2 } from 'three';
import { buildNaniteView } from '../nanite/NaniteView';
import { Heightfield } from '../world/Heightfield';
import { SunSky } from '../sky/SunSky';
import { PostStack } from '../render/PostStack';
import { updateSunUniforms } from '../render/VegMaterials';
import { setWindContext } from '../render/Wind';

/** per-species leaf tint → matParam (linear RGB low 3 bytes + hueVar high byte) */
function packLeafTint(c: { r: number; g: number; b: number; hueVar: number }): number {
  const u8 = (x: number): number => Math.max(0, Math.min(255, Math.round(x * 255)));
  return (u8(c.r) | (u8(c.g) << 8) | (u8(c.b) << 16) | (u8(c.hueVar) << 24)) >>> 0;
}

export async function buildForestScene(ctx: WorldContext): Promise<void> {
  const { engine, seed } = ctx;
  const q = new URLSearchParams(window.location.search);
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
  //   ?voxdither=0|1 — voxel-band raster coverage (read in NaniteVoxelRaster): 0 = OPAQUE
  //                    bricks (DEFAULT, cheap — no coverHash, occlusion-collapse intact;
  //                    correct at sub-pixel brick size via finer ?voxgrid / farther ?voxnear);
  //                    1 = Stage-3b density DITHER (see-through sparse-foliage, no occlusion).
  const forceVox = q.get('forcevox') !== null;
  const voxOn = q.get('voxreg') !== '0' || forceVox;
  // ?noleaves — DEBUG ablation: render TRUNKS/BARK ONLY (no leaf crown at all — neither the
  // triangle leaf head nor its voxel sibling). For isolating how much of the forest frame the
  // foliage (voxel crowns) actually costs vs the woody-skeleton triangles + the cull.
  const noLeaves = q.get('noleaves') !== null;
  const voxGridDim = Number(q.get('voxgrid') ?? DEFAULT_VOXEL_GRID_DIM) || DEFAULT_VOXEL_GRID_DIM;
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
      engine.renderer.getDrawingBufferSize(new Vector2()).y,
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
  }

  // ── tree geometry (real crowns, full leaf density) ────────────────────────
  ctx.progress(0.1, 'forest: building veg library');
  const lib = await buildVegLibrary(
    engine.renderer,
    seed,
    (p, m) => ctx.progress(0.1 + p * 0.4, m),
    { leafAnchorTarget: leafDensity },
  );
  // canopy species with a bark trunk + a real leaf crown (cls 0–4)
  const pools = lib.pools.filter((p): p is VegPool => p.cls <= 4 && !!p.r0?.[0] && !!p.leaf);
  if (pools.length === 0) throw new Error('forest: no canopy tree pools with leaf crowns');

  // ── register bark + leaf per species ──────────────────────────────────────
  ctx.progress(0.5, 'forest: registering tree meshes');
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
      const prep = prepareVoxelCrown(leafSrc, pool.leaf!.color, voxGridDim, voxLod);
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
          }),
      });
    }
    return { bark, leaf };
  });

  // ── plant a jittered grid, species spatially mixed; per-mesh streams must be
  //    contiguous, so collect per-pool instance lists then bind each once ─────
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
  const builds: { handle: number; dag: DagBuild }[] = [];
  if (dagJobs.length > 0) {
    ctx.progress(0.75, `forest: building ${dagJobs.length} LOD DAGs`);
    let lateV = 0;
    let lateT = 0;
    let lateC = 0;
    for (const job of dagJobs) {
      const dag = job.build();
      lateV += dag.verts.length / DAG_VERT_STRIDE;
      lateT += dag.indices.length / 3;
      lateC += dag.clusters.length;
      builds.push({ handle: job.handle, dag });
    }
    reg.addLate({ verts: lateV, tris: lateT, clusters: lateC });
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
  ctx.progress(0.9, 'forest: building registry');
  const report = reg.build(engine.renderer, engine.stats.counters);
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
        maxDist: 2000,
        nearDist: forceVox ? 0 : transitionDist,
        label: `c${pools[v.poolIdx]?.cls}/voxel`,
      });
      appended += r.brickCount;
      const s = poolStreams[v.poolIdx];
      if (s) reg.bindInstances(r.head, { a: s.a, b: s.b });
      // ?forcevox DEBUG: the voxel head renders everywhere ⇒ suppress the leaf head.
      if (forceVox) reg.setMaxDistance(v.leafHead, 0.001);
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
    const bootTod = ctx.params.timeOfDay;
    const sunSky = new SunSky(engine, bootTod);
    await sunSky.init(engine.renderer);
    updateSunUniforms(sunSky.sun);
    // trunk/leaf sway reads a module-global wind context (set by the world scene) — the
    // raster + resolve both sample it, so it must exist before buildNaniteFrame.
    if (hf.noiseA) setWindContext({ noiseA: hf.noiseA, canopyTex: null });
    const post = new PostStack(engine, sunSky.atmosphere, bootTod);
    const { buildNaniteFrame } = await import('../nanite/NaniteFrame');
    const frame = buildNaniteFrame(engine, reg, hf, post, {
      gi: null,
      canopyTex: null,
      csm: null,
      barkTexA: lib.barkArray?.texA ?? null,
      barkTexB: lib.barkArray?.texB ?? null,
    });
    engine.post = frame as unknown as typeof engine.post;
    // meter() is driven once/frame by Engine.renderStep (this.post.meter) — same as
    // the world scene. Do NOT also wire it via onUpdate or it dispatches autoExposure
    // (and the cull/raster/shadow readbacks) TWICE per frame.
    // eslint-disable-next-line no-console
    console.log('[forest] FULL-FRAME pipe (NaniteFrame resolve + post) — ?nanitedbg=cluster for the lean debug view');
  } else {
    // lean cull→raster→flat-resolve debug view (dummy 1×1 heightTex — never sampled).
    const heightTex = new StorageTexture(1, 1);
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

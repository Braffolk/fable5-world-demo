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
import { GeometryRegistry, DAG_VERT_STRIDE, explicitToDagVerts } from '../nanite/GeometryRegistry';
import { type DagBuild, buildDag } from '../nanite/BuildDag';
import { buildAggregateDag } from '../nanite/BuildAggregateDag';
import { geometryToSource } from '../nanite/WorldRegistry';
import { buildNaniteView } from '../nanite/NaniteView';
import type { Heightfield } from '../world/Heightfield';

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
  const meshes = pools.map((pool) => {
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
    const leaf = reg.registerMesh(leafSrc, 'leaf', {
      transformChannel: 'leaf',
      castShadows: false,
      twoSided: true,
      swayPad: 3.8,
      matParam: packLeafTint(pool.leaf!.color),
      aggregate: true,
      label: `c${pool.cls}/leaf`,
    });
    reg.setMaxDistance(bark, 2000);
    reg.setMaxDistance(leaf, 2000);
    if (wantDag) {
      dagJobs.push({
        handle: bark,
        build: () => buildDag(explicitToDagVerts(barkSrc), DAG_VERT_STRIDE, barkSrc.indices, { normalOffset: 3 }),
      });
      dagJobs.push({
        handle: leaf,
        build: () => buildAggregateDag(explicitToDagVerts(leafSrc), DAG_VERT_STRIDE, leafSrc.indices, { seed: seed.seed }),
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
  // bind the SAME instance stream to each species' bark + leaf (trunk + crown)
  for (let i = 0; i < pools.length; i++) {
    const list = lists[i];
    const m = meshes[i];
    if (!list || !m || list.a.length === 0) continue;
    const a = new Float32Array(list.a);
    const b = new Float32Array(list.b);
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
  ctx.progress(0.9, 'forest: building registry');
  const report = reg.build(engine.renderer, engine.stats.counters);
  for (const b of builds) reg.attachDag(b.handle, b.dag);
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

  // ── render: the lean cull→raster→resolve view (a dummy heightTex — tree
  //    clusters never sample it; only terrain clusters do, and there are none) ─
  const heightTex = new StorageTexture(1, 1);
  heightTex.type = FloatType;
  const hf = { heightTex } as unknown as Heightfield;
  const view = buildNaniteView(engine, reg, hf, mode);
  engine.post = view as unknown as typeof engine.post;
  engine.onUpdate(() => view.meter(engine.renderer));

  // camera inside the forest at eye height, looking horizontally
  ctx.hooks.initialPose = { p: [0, 2, 0], yaw: 0.6, pitch: -0.02 };
  ctx.hooks.initialPoseMode = 'fly';
  engine.camera.position.set(0, 2, 0);
  ctx.hooks.groundProbe = () => ({ ground: 0, water: -1e9 });
  ctx.progress(1, 'forest: ready');
}

/**
 * Full-frame nanite integration (N4-C0, D-N18/D-N19) — `?nanite=1` without
 * `?nanitedbg`: the cull→raster compute runs BEFORE the post pipeline each
 * frame, and the resolve mesh (NaniteResolve) shades the world inside the main
 * scene pass, depth-composing with everything else the scene draws (water, sky,
 * particles). Sun-shadow casting is the nanite shadow system (N5/D-N28:
 * depth-only SW raster into our own r32 cascades, default-on).
 *
 * JITTER MIRROR (D-N18): TRAA applies a per-frame Halton view offset to the
 * scene camera inside the pipeline render (onBeforeRenderPipeline), AFTER our
 * compute would have read the matrices. The raster must project with that
 * same offset or nanite content sits a sub-pixel off the hardware content
 * every frame (crawl + a systematic diff vs ?nanite=0). We read the TRAA
 * node's _jitterIndex (it increments in onAfterRenderPipeline, so at compute
 * time it still holds THIS frame's index) and re-derive the offset with
 * three's own formula: halton(i+1, 2/3) − 0.5 texels via setViewOffset.
 * Verified against TRAANode.js 0.184; re-check on any three upgrade.
 *
 * `?cullfreeze=1`, `?occl=0`, `?phase2=0`, `?audit=1` work as in the debug
 * view. `?naniteframe=0` keeps registry build + old rendering (boot probes).
 */

import { PerspectiveCamera, Vector2, Vector4 } from 'three';
import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';
import { Fn, float, instanceIndex, storage, uint, vec2, vec4 } from 'three/tsl';
import type { NF } from '../../gpu/TSLTypes';
import type { Engine } from '../../core/Engine';
import type { PostStack } from '../../render/PostStack';
import { internalSize } from '../../render/RenderScale';
import type { Heightfield } from '../../world/Heightfield';
import type { TerrainField } from '../world/TerrainField';
import type { GeometryRegistry } from '../world/GeometryRegistry';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK } from '../world/GeometryRegistry';
import { deriveLodParams, makeNaniteCam } from '../NaniteCommon';
import { buildNaniteCull } from '../cull/NaniteCull';
import { buildNaniteHzb } from '../cull/NaniteHzb';
import { buildGrassField } from '../grass/NaniteGrass';
import { makeFetch } from '../raster/NaniteFetch';
import { buildNaniteRaster, makeVisBuffers } from '../raster/NaniteRaster';
import { buildNaniteResolve } from '../shade/NaniteResolve';
import { buildNaniteShadowClip, type NaniteShadow } from '../shade/NaniteShadowClip';
import { buildShadowHalf, type ShadowHalf } from '../shade/NaniteShadowHalf';
import { bcU2F, dispatch, dispatchBatchMixed, elemU, readBuffer, returnIf, texLoadR, toF, uniformArrV4 } from '../Tsl';

export interface NaniteFrameHandles {
  render(): void;
  meter(renderer: WebGPURenderer): void;
  /** measure-infra W5: the meter's counter readbacks, runnable OUTSIDE a timed window
   *  (MeasureHarness calls this on a drained queue between samples). */
  meterRead(renderer: WebGPURenderer): Promise<Record<string, number>>;
  /** W2 (water arc): composed sun visibility (clipmap PCSS × cloud × far-shadow) for
   *  non-resolve materials; undefined when the scene has no nanite sun shadows.
   *  pix = explicit IGN-noise coord — REQUIRED from compute (no fragCoord there). */
  sunVis?: (
    wp: import('../../gpu/TSLTypes').NV3,
    n: import('../../gpu/TSLTypes').NV3,
    pix?: import('../../gpu/TSLTypes').NV2,
  ) => NF;
}

/** halton(index, base) — TRAANode.js's exact sequence (verbatim formula) */
function halton(index: number, base: number): number {
  let fraction = 1;
  let result = 0;
  let i = index;
  while (i > 0) {
    fraction /= base;
    result += fraction * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}

export function buildNaniteFrame(
  engine: Engine,
  registry: GeometryRegistry,
  hf: Heightfield,
  /** S3b: the streamed terrain field — the hot shaders' terrain data source
   *  (height/fields/water planes); hf keeps feeding only the not-yet-ported
   *  pieces (noise ctx, resolve/grass/shadow legacy reads until their slices) */
  field: TerrainField,
  post: PostStack,
  world: {
    gi: import('../../gpu/passes/ProbeGI').ProbeGI | null;
    canopyTex: import('three/webgpu').StorageTexture | null;
    /** "scene has sun shadows" — drives the nanite screen-density shadow clipmap. */
    sunShadows?: boolean;
    /** world-space cloud sun-transmittance gate — the resolve multiplies it into the
     *  sun term directly. */
    cloudShadow?: ((wxz: import('../../gpu/TSLTypes').NV2) => NF) | null;
    /** P4: baked heightfield sun-visibility (FarShadow) — the beyond-clipmap term
     *  (mountains shade valleys at any distance). Multiplied like cloudShadow. */
    farShadow?: ((wxz: import('../../gpu/TSLTypes').NV2) => NF) | null;
    barkTexA: import('three').Texture | null;
    barkTexB: import('three').Texture | null;
    /** S6d PRECISION: the per-frame render anchor A (= StreamOrigin) for the
     *  streamed (Estonia) world — the camera VP / reconstruct / shadow chain is
     *  built RELATIVE to it so f32 stays sub-metre at ~311 km absolute coords.
     *  Omitted on the generated world ⇒ A=(0,0) ⇒ byte-identical absolute build. */
    streamAnchor?: () => { x: number; z: number };
  },
): NaniteFrameHandles {
  const renderer = engine.renderer;
  const size = internalSize(renderer, new Vector2()); // ?rscale: match the scene pass
  const params = new URLSearchParams(window.location.search);
  // voxel-foliage (spec §A1 / Stage 3a): the voxel subsystem is active iff the registry
  // actually holds bricks — i.e. crowns were voxelized + voxel:7 heads registered (the
  // automatic ?scene=forest transition, ?voxreg in the world scene, or ?forcevox). Keying
  // off registry.brickCount (NOT a query param) means the per-frame fan-out + voxel raster
  // turn on wherever the build path wired voxels; a pure-triangle build pays nothing.
  const voxActive = registry.brickCount > 0;
  // ?vrange=1 — PERF-3 diagnostic: per-cluster vertex-INDEX range distribution +
  // redundancy, to size/justify a vertex-transform cache. range = max−min global
  // index over a cluster's tri corners; a runtime [vMin,vMax] shared-mem cache of
  // size S hits when range ≤ S. redund = 3·tris / unique-verts (the fetchWorldVert
  // over-fetch the cache eliminates). Split explicit (gpu.verts) vs HF-DAG
  // (gpu.hfVerts); window-grid HF is skipped (no index buffer).
  if (params.get('vrange') === '1') {
    const dbg = registry.debug().arrays;
    const idx = dbg.indices;
    const cl = dbg.clusters;
    const N = registry.clusterCount;
    const CW = 8; // CLUSTER_WORDS
    const buckets = [64, 128, 256, 512, 1024, Number.POSITIVE_INFINITY];
    const tally = (label: string, pred: (flags: number) => boolean): void => {
      const hist = new Array<number>(buckets.length).fill(0);
      let nCl = 0;
      let sumRange = 0;
      let sumUnique = 0;
      let sumTris = 0;
      let maxRange = 0;
      const seen = new Set<number>();
      for (let c = 0; c < N; c++) {
        const triStart = cl[c * CW + 6] ?? 0;
        const w7 = cl[c * CW + 7] ?? 0;
        const triCount = w7 & 0xff;
        const flags = (w7 >>> 8) & 0xff;
        if (triCount === 0 || !pred(flags)) continue;
        let mn = 0xffffffff;
        let mx = 0;
        seen.clear();
        for (let t = 0; t < triCount; t++) {
          for (let v = 0; v < 3; v++) {
            const vi = idx[(triStart + t) * 3 + v] ?? 0;
            if (vi < mn) mn = vi;
            if (vi > mx) mx = vi;
            seen.add(vi);
          }
        }
        const range = mx - mn + 1;
        nCl++;
        sumRange += range;
        sumUnique += seen.size;
        sumTris += triCount;
        if (range > maxRange) maxRange = range;
        for (let b = 0; b < buckets.length; b++) {
          if (range <= (buckets[b] ?? 0)) {
            hist[b] = (hist[b] ?? 0) + 1;
            break;
          }
        }
      }
      if (nCl === 0) {
        console.log(`[laas][vrange] ${label}: (none)`);
        return;
      }
      const pct = (x: number): string => `${((x / nCl) * 100).toFixed(0)}%`;
      console.log(
        `[laas][vrange] ${label}: ${nCl} cl · avg range ${(sumRange / nCl).toFixed(0)} · avg unique ${(sumUnique / nCl).toFixed(0)} · avg tris ${(sumTris / nCl).toFixed(0)} · redund ${((3 * sumTris) / sumUnique).toFixed(2)}× · maxRange ${maxRange}`,
      );
      console.log(`[laas][vrange] ${label} range≤[64,128,256,512,1024,∞]: ${hist.map(pct).join(' ')}`);
    };
    tally('explicit', (f) => (f & 1) === 0);
    tally('HF-DAG', (f) => (f & 1) !== 0 && (f & 2) !== 0);
  }
  const occl = params.get('occl') !== '0';
  const frozenParam = params.get('cullfreeze') === '1';
  // LOD-warp cut params (τ / min-px culls / distance-banded falloff) — shared with
  // the ?nanitedbg debug view via deriveLodParams so the two stay in lockstep. The
  // live setters below (probe sweeps) mutate these uniforms in place.
  const { tau, minPx, simBandD, lodNear, lodPow, instMinPx } = deriveLodParams(params, size);

  const cam = makeNaniteCam(size.x, size.y);
  const vis = makeVisBuffers(size.x * size.y);
  // PERF-VB4 (D-N45): the WORLD is single-pass — the HZB reads the packed depth key from
  // the election anchor (visPayloadV high bits, packed=true), there is no exact depthV.
  const hzb = buildNaniteHzb(vis.payloadV.ro, cam, true);
  // item 6: the BFS pass count = the MEASURED deepest DAG anchor-chain + a small safety
  // margin (defends against a streaming tile attaching a slightly deeper chain after this
  // point — terrain tiles share one uniform grid depth, so +2 is ample). This replaces the
  // old constant 18 (the comment estimated the real depth ~13), shedding the empty-tail
  // passes paid TWICE/frame (camera + shadow shared-cut). ?hierdepth still overrides inside
  // buildNaniteCull. Clamped ≥ 1. Going too LOW under-traverses = holes, so we never go
  // below the measured value; the margin only ever ADDS passes.
  const measuredHierDepth = Math.max(1, registry.maxDagDepth + 2);
  // A14: the pass count is now BAKED into the cull pipelines — any later attach
  // growing the DAG depth past it throws at the attach instead of silently
  // never emitting its deep leaves.
  registry.freezeHierDepth(measuredHierDepth);
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
    // PERF-VB3: HIERARCHICAL DAG-BFS cull is the SOLE world cull (every mesh is
    // DAG'd — terrain via TERRAIN-RW, veg via the always-on nanitedag). Single-phase
    // BFS, NON-packed two-pass raster (depthV ⇒ HZB + exact-depth world resolve unchanged).
    {
      tau,
      minPx,
      simBandD,
      lodNear,
      lodPow,
      instMinPx,
      hierDepth: measuredHierDepth,
      // ?voxprev two-pass partition classifier — the LIBERAL centre test, NOT the
      // conservative emit test (spec-prev-frame-occlusion §2.1). null at ?occl=0 ⇒ inert.
      voxPrevTest: occl ? hzb.sphereProbablyOccluded : null,
      // crown-LOD Phase 2 (2026-07-08): the near-crown mesh LOD ladder is now the
      // DEFAULT — leaf clusters emit at coarser rungs inside 0..60 m (BuildCrownLodDag).
      // ?crownlod0=1 is the DISABLE-only compare flag: it force-descends every leaf
      // cluster with children back to full LOD0 (the pre-Phase-2 behavior), for A/B.
      // Camera path only (shadow culls omit it — casters already stay coarse).
      crownLod0: params.get('crownlod0') === '1',
    },
  );
  if (!hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteFrame: heightfield noise bakes missing (boot order)');
  }
  // ?nanodisp=1 — disable terrain micro-displacement (root-cause bisect for
  // near-camera transparency: the disp branch only runs within 85 m)
  const dispOff = params.get('nanodisp') === '1';
  // S6e: the render-anchor uniform for terrain FIELD sampling (NaniteFetch hfWorld /
  // terrainDispAt) — the S6d anchor-relative vert positions must be re-absoluted to
  // hit the world-anchored field planes. Streamed only; undefined ⇒ generated compiles
  // the verbatim absolute path (byte-identical shader).
  const fieldAnchor = world.streamAnchor ? cam.anchor : undefined;
  const disp = dispOff
    ? undefined
    : {
        field,
        noiseA: hf.noiseA,
        noiseB: hf.noiseB,
        camPos: cam.camPos,
        anchor: fieldAnchor,
      };
  // trunk wind (matches the resolve's makeFetch — both read ?nanwind so the
  // rastered geometry and the resolve's barycentric corners stay bit-identical)
  const windOn = params.get('nanwind') !== '0';
  const windOpt = windOn ? { camPos: cam.camPos } : undefined;
  // PROCEDURAL GRASS (NaniteGrass.ts; ledger docs/perf-runs/2026-07-03-grass-arc.md).
  // __laasNanite.setGrass(0|1) toggles within a boot.
  // DEFAULT ON (user call 2026-07-04, look accepted): the single ray lane
  // (Sannikov baked-raycast, NaniteGrass.ts). ?grass=0|off disables. The old
  // geo/hybrid/rayold lanes were deleted the same day — git history has them.
  const grassMode = params.get('grass');
  const grassOn = grassMode !== '0' && grassMode !== 'off';
  const grass = grassOn
    ? buildGrassField({ cam, vis, field, canopyTex: world.canopyTex, disp })
    : null;
  const raster = buildNaniteRaster(
    registry.gpu, field, cam, cull, vis, 'flat', true, disp, windOpt, false, true, voxActive,
    grass ? { batch: grass.batch, renderHw: grass.renderHw, enabled: grass.enabled } : undefined,
    fieldAnchor, // S6e: absolute field-sample coords for the anchor-relative terrain verts
  );

  // Nanite shadows (N5, D-N28): depth-only SW raster into own r32 cascade textures,
  // sampled by the resolve's own PCSS. R1 caches per cascade (re-raster only on a
  // VP change) → ~0 cost static, the [1,2,3,6] cadence moving. ON BY DEFAULT;
  // ?nanshadow=0 disables the whole system (producer here + receive in the resolve,
  // same flag). Built BEFORE the resolve so the resolve binds shadowFactor.
  const shadowOn =
    params.get('nanshadow') !== '0' && world.sunShadows === true;
  // S3 (D-N29): the SCREEN-DENSITY SHADOW CLIPMAP — a camera-centred clipmap fits
  // its own per-level light VPs (no CSM cascade cameras).
  const shadow: NaniteShadow | null = shadowOn
    ? buildNaniteShadowClip(
        registry.gpu,
        registry.instanceCount,
        field,
        disp,
        windOpt,
        measuredHierDepth,
        voxActive,
        world.streamAnchor, // S6d: fit the light frame in the StreamOrigin-relative space
        world.streamAnchor ? () => registry.tileEpoch : undefined, // S6d: dirty on tile stream
        fieldAnchor, // S6e: absolute field-sample coords in the shadow depth raster's fetch
      )
    : null;
  // CAMERA||SHADOW CULL OVERLAP: fold the (CLIP-path) shadow shared-cut cull into the SAME
  // submit as the camera cull so Dawn can overlap the two disjoint culls on frames where
  // shadows re-raster. Order-equivalent (disjoint buffers: the shadow cut writes its own
  // counters/queues, read-only on qRaster; appended last ⇒ same order as a separate submit).
  // The win fires when the camera moves enough to re-raster a shadow level. Requires the clip
  // shadow's cullPrepass — the cascade path has no shared cut, so it stays on the plain leg.
  const cullOverlap = typeof shadow?.cullPrepass === 'function';
  // SUBMIT-COALESCE (?coalesce=1, spec-orchestration-submit-folds §1): fold the frame's
  // 7 foldable submits into 2 — cull side (BFS + kRasterArgs2 + voxel fan-out [+ shadow
  // cut]) becomes ONE dispatchBatchMixed, and the raster side folds the HZB chain into
  // dispatchVoxel's submit (voxPyr + scatter + HZB, §1b/§1c). Same kernels, same order,
  // same indirect grids ⇒ bit-identical; only submit granularity changes.
  // DEFAULT ON (gated 2026-07-02: shot-diff at D0 band all poses, live p50 16.6/p95 17.3
  // 0×>33ms, gpu medians sub-noise — fresh-w0-*/fold-*-live2 JSONs). ?coalesce=0 = legacy.
  const coalesce = params.get('coalesce') !== '0';

  // S0 (D-N29): half-res PCSS eval + depth-aware bilateral upsample — quarters the
  // per-pixel shadow SAMPLE cost (paid every frame, static or moving). Built from
  // the MAIN vis + cam (it reconstructs wp exactly like the resolve) and shadow's
  // PCSS sampler. ?shalfres=0 falls back to the full-res per-pixel path for A/B.
  const halfResShadow = params.get('shalfres') !== '0';
  const shadowHalf: ShadowHalf | null =
    shadow && halfResShadow ? buildShadowHalf(vis, cam, shadow) : null;

  // W2 (water arc): the composed per-pixel sun-visibility factor — clipmap PCSS ×
  // cloud transmittance × baked far-shadow — for NON-resolve materials (water foam/
  // glint) and the grass texel light bake. Mirrors the resolve's own composition
  // (NaniteResolve.ts sun block) incl. the cloud NaN guard. shadowHalf is
  // deliberately NOT offered: its half-res eval sits at the OPAQUE depth (the
  // lakebed), wrong for a surface above it. (Defined here — before the resolve —
  // since the grass lean-light bake consumes it at build time.)
  const sunVis = shadow
    ? (
        wp: import('../../gpu/TSLTypes').NV3,
        n: import('../../gpu/TSLTypes').NV3,
        pix?: import('../../gpu/TSLTypes').NV2,
      ): NF => {
        // pure expression chain — this runs at MATERIAL BUILD time, outside any
        // Fn() stack, so toVar()/assign() are illegal here (TSL "no stack" spam)
        let sf = (shadow.shadowFactor(wp, n, pix) as unknown as { clamp(a: number, b: number): NF })
          .clamp(0, 1) as NF;
        if (world.cloudShadow) {
          const c = world.cloudShadow(wp.xz as unknown as import('../../gpu/TSLTypes').NV2);
          const safe = c.equal(c).select(c.clamp(0, 1), float(1)) as unknown as NF;
          sf = sf.mul(safe) as unknown as NF;
        }
        if (world.farShadow) {
          const fv = world
            .farShadow(wp.xz as unknown as import('../../gpu/TSLTypes').NV2)
            .clamp(0, 1) as unknown as NF;
          sf = sf.mul(fv) as unknown as NF;
        }
        return sf;
      }
    : undefined;

  // voxel-foliage (Stage 2 §7): give the resolve the voxel work-queue ONLY when active, so
  // a pure-triangle world's resolve never binds qVoxRaster/voxelBricks (stays at 8 buffers).
  const resolveCull = voxActive ? { qRasterRO: cull.qRasterRO, qVoxRasterRO: cull.qVoxRasterRO } : cull;
  const resolve = buildNaniteResolve(registry.gpu, field, cam, resolveCull, vis, {
    hf,
    field,
    // RP-1: registered matClass ids — the resolve strips absent classes' subgraphs
    presentClasses: registry.presentClasses,
    gi: world.gi,
    canopyTex: world.canopyTex,
    sunShadows: world.sunShadows,
    cloudShadow: world.cloudShadow,
    farShadow: world.farShadow,
    barkTexA: world.barkTexA,
    barkTexB: world.barkTexB,
    naniteShadow: shadow,
    shadowHalf,
    grassProc: grass
      ? { ray: grass.resolveRay }
      : null,
  });
  // ?nores=1 — MEASUREMENT ablation (default OFF): skip ALL fullscreen resolve passes (the
  // class-family split `mesh`/`meshMesh` + vox `voxMesh`). Decomposes the frame: (baseline −
  // nores) gpuWall = the cost of the per-pixel resolve passes, which survive ?pure=1 and have no
  // other isolating flag. Nothing shades when ON (the scene pass renders sky only) — perf probe.
  const noResolve = params.get('nores') === '1';
  if (!noResolve) engine.scene.add(resolve.mesh);
  // resolve P2 (class-family split): the 'mesh' pass — shades mesh families (matClass 1-5),
  // Discards terrain + voxel pixels. Present only in two-pass mode (undefined under 'both'
  // single-pass merge). renderOrder −999.5, right after the 'terr' `mesh`.
  if (resolve.meshMesh && !noResolve) engine.scene.add(resolve.meshMesh);
  // voxel-foliage two-pass resolve (spec §4.6): the SECOND fullscreen pass that shades only
  // voxel-winner pixels (present only when ?voxreg/?forcevox wired the voxel queue). Splitting
  // the resolve in two keeps BOTH materials ≤10 fragment storage buffers — the single-pass
  // design bound the tri-fetch set + voxelBricks + qVoxRasterRO together and busted the Metal
  // ceiling, invalidating the pipeline so NOTHING shaded.
  if (resolve.voxMesh && !noResolve) engine.scene.add(resolve.voxMesh);

  // ?nanprobe=1 — exact-number depth forensics: a compute kernel reads the
  // SCENE PASS depth texture and the vis buffer at up to 8 pixels into a
  // storage buffer (no PNG/tone-map/color-space layers in the way)
  const probeOn = params.get('nanprobe') === '1';
  let probeRun: ((renderer2: WebGPURenderer) => void) | null = null;
  let probeRead: (() => Promise<Float32Array>) | null = null;
  let probeSet: ((pix: number[][]) => void) | null = null;
  if (probeOn) {
    const fetchDbg = makeFetch(registry.gpu, field, undefined, undefined, true, 'both', fieldAnchor);
    const probeAttr = new StorageBufferAttribute(new Float32Array(32), 1);
    probeAttr.name = 'nanProbeReadback';
    const outBuf = storage(probeAttr, 'float', 32);
    const pixU = uniformArrV4(Array.from({ length: 8 }, () => new Vector4()));
    const kProbe = Fn(() => {
      returnIf(instanceIndex.greaterThanEqual(uint(8)));
      const p = pixU.element(instanceIndex);
      const x = uint(p.x);
      const yTop = uint(p.y);
      const depthNode = post.sceneDepthNode as unknown as {
        load(t: unknown): { x: NF };
      } | null;
      const sceneD = depthNode
        ? depthNode.load(vec2(toF(x), toF(yTop)).add(0.5)).x
        : (float(-1) as unknown as NF);
      const fy = uint(size.y - 1).sub(yTop);
      const visD = bcU2F(elemU(vis.depthV.ro, fy.mul(uint(size.x)).add(x)));
      void texLoadR;
      void sceneD;
      // payload decode + FULL refetch/reproject via the SAME makeFetch the
      // raster uses: recomputed cz must equal the stored depth (coherence),
      // and the vertex world position tells whether the triangle is where
      // its window says it should be
      const pay = elemU(vis.payloadV.ro, fy.mul(uint(size.x)).add(x)).toVar();
      const itemIdx = pay.shiftRight(uint(CLUSTER_TRI_BITS));
      const localTri = pay.bitAnd(uint(CLUSTER_TRI_MASK));
      const item = cull.qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const ctx = fetchDbg.makeCtx(instId, ci);
      const w0 = fetchDbg.fetchWorldVert(ctx, localTri, 0);
      void w0;
      // THE RESOLVE'S wp EXPRESSIONS, verbatim (pixel-center ndc + invVp):
      const ndcX = toF(x).add(0.5).div(float(size.x)).mul(2).sub(1);
      const ndcY = toF(fy).add(0.5).div(float(size.y)).mul(2).sub(1);
      const hpos = cam.invVp.mul(vec4(ndcX, ndcY, visD, 1));
      const wpx = hpos.x.div(hpos.w);
      const wpy = hpos.y.div(hpos.w);
      const wpz = hpos.z.div(hpos.w);
      outBuf.element(instanceIndex.mul(uint(4))).assign(wpx);
      outBuf.element(instanceIndex.mul(uint(4)).add(uint(1))).assign(wpy);
      outBuf.element(instanceIndex.mul(uint(4)).add(uint(2))).assign(wpz);
      outBuf.element(instanceIndex.mul(uint(4)).add(uint(3))).assign(visD);
    })().compute(8, [8]);
    (kProbe as unknown as { setName(n: string): void }).setName('nanProbe');
    probeSet = (pix) => {
      for (let i = 0; i < 8; i++) {
        const v = pixU.array[i];
        if (v) v.set(pix[i]?.[0] ?? 0, pix[i]?.[1] ?? 0, pix[i]?.[2] ?? 0, pix[i]?.[3] ?? 0);
      }
    };
    probeRun = (r) => dispatch(r, kProbe);
    probeRead = async () => new Float32Array(await readBuffer(renderer, probeAttr, 0, 128));
  }
  (window as unknown as { __laasNanite?: object }).__laasNanite = {
    setProbe: probeSet,
    readProbe: probeRead,
    vp: () => cam.vp.value.toArray(),
    /** N8-D1: live τ (screen-error px) for the continuous-zoom gate / A-B */
    setTau: (v: number) => {
      tau.value = v;
    },
    tau: () => tau.value,
    /** D-N43 Stage 0.5 SIM: live distance-band scale (m) for the region-collapse sim */
    setSimBand: (v: number) => {
      simBandD.value = v;
    },
    simBand: () => simBandD.value,
    /** procedural grass within-boot A/B (the water-visible-toggle idiom): flips the
     *  kernel enable uniform + the HW mesh — thermal-invariant perf attribution. */
    setGrass: (v: number) => {
      grass?.setEnabled(v !== 0);
    },
    grassOn: () => (grass ? grass.enabled() : false),
  };

  // jitter-mirrored projection: scratch camera = engine camera + TRAA's
  // current Halton view offset (null when TAA is ablated)
  const scratch = new PerspectiveCamera();
  let warnedJitter = false;
  const jitteredCamera = (): PerspectiveCamera => {
    engine.camera.updateMatrixWorld();
    scratch.copy(engine.camera);
    const node = post.traaNode as { _jitterIndex?: number } | null;
    if (node) {
      const idx = node._jitterIndex;
      if (typeof idx !== 'number') {
        if (!warnedJitter) {
          warnedJitter = true;
          // eslint-disable-next-line no-console
          console.warn('[nanite] TRAANode._jitterIndex missing — jitter mirror dead (three upgrade?)');
        }
      } else {
        // ?nanjitter — mirror-phase bisects: 0 off | neg | swap | +1/-1 index
        const jm = params.get('nanjitter');
        if (jm !== '0') {
          let i2 = idx;
          if (jm === 'p1') i2 = (idx + 1) % 31;
          if (jm === 'm1') i2 = (idx + 30) % 31;
          let jx = halton(i2 + 1, 2) - 0.5;
          let jy = halton(i2 + 1, 3) - 0.5;
          if (jm === 'neg') {
            jx = -jx;
            jy = -jy;
          }
          if (jm === 'swap') {
            const t = jx;
            jx = jy;
            jy = t;
          }
          scratch.setViewOffset(size.x, size.y, jx, jy, size.x, size.y);
        }
      }
    }
    return scratch;
  };

  let frame = 0;
  let reading = false;
  let warned = '';
  let frozen = false;

  const render = (): void => {
    const cur = internalSize(renderer, new Vector2());
    if ((cur.x !== cam.width || cur.y !== cam.height) && warned !== 'size') {
      warned = 'size';
      // eslint-disable-next-line no-console
      console.warn(
        `[nanite] internal size ${cur.x}×${cur.y} != frame ${cam.width}×${cam.height} — reload to rebuild`,
      );
    }
    const freezeNow = frozenParam && frame > 0;
    if (!frozen && freezeNow) {
      frozen = true;
      // eslint-disable-next-line no-console
      console.log('[nanite] cullfreeze: visibility frozen — fly to inspect');
    }
    const anchor = world.streamAnchor?.();
    cam.update(jitteredCamera(), anchor?.x ?? 0, anchor?.z ?? 0);
    // PERF-VB3 HIER (single-phase BFS, the SOLE world cull): seed roots (terrain + veg)
    // → BFS-descend the DAG (reads LAST frame's HZB for occlusion) → fills qRaster
    // directly. Then ONE depth+payload over the NON-packed two-pass raster (depthV
    // written ⇒ HZB + the exact-depth world resolve are unchanged). No depth2 / phase-2.
    if (!frozen) {
      // CAMERA||SHADOW CULL OVERLAP (item 4): when enabled, fold the shadow shared-cut
      // cull's batch into the SAME submit as the camera cull (they write DISJOINT buffers
      // — fresh counters/qRaster/qFrontier, sphereOccluded=null ⇒ no HZB dep), so Dawn may
      // overlap the two on re-raster frames. The shadow's later run() consumes the prepass
      // mask and SKIPS its own cut dispatch. Camera-cull half is dispatched FIRST in the
      // combined list (its order is internally self-consistent); the shadow half follows.
      if (coalesce) {
        // SUBMIT-COALESCE §1a: BFS + kRasterArgs2 + voxel fan-out [+ shadow cut] in ONE
        // submit. RAW audit (spec): kRasterArgs2 reads qRaster[0] written by kRasterArgs
        // one dispatch earlier; the fan-out chain reads counters[1]/qRaster and writes
        // its own args/queues; the shadow cut writes DISJOINT buffers (own counters/
        // queues), so appending it last is equivalent to today's order. When no shadow
        // level re-rasters this frame the cut is null ⇒ empty tail (shadow run() sees
        // prepassMask===0, same fall-through semantics as the plain leg).
        const shadowCut =
          cullOverlap && shadow?.cullPrepass
            ? (shadow.cullPrepass(renderer, engine.camera) ?? [])
            : [];
        dispatchBatchMixed(renderer, [
          ...cull.phase1Batch(), // [kClearHier, kSeedRoots, (args,traverse)×D, kRasterArgs]
          ...cull.fullArgsBatch(), // [kRasterArgs2]
          ...(voxActive ? cull.voxFanoutBatch() : []),
          // fill qHwRaster from the cut (read-only on qRaster ⇒ order-free vs the voxel
          // fanout) so world1's SW skip + instanced HW draw agree.
          ...cull.hwPartitionBatch(),
          ...shadowCut,
        ]);
      } else {
        let cameraCullDispatched = false;
        if (cullOverlap && shadow?.cullPrepass) {
          const shadowCutBatch = shadow.cullPrepass(renderer, engine.camera);
          if (shadowCutBatch && shadowCutBatch.length > 0) {
            dispatchBatchMixed(renderer, [...cull.phase1Batch(), ...shadowCutBatch]);
            cameraCullDispatched = true;
          }
          // shadowCutBatch null ⇒ no level re-rasters this frame; fall through to the plain
          // camera cull (the shadow run() will see prepassMask===0 and raster nothing).
        }
        if (!cameraCullDispatched) cull.runPhase1(renderer); // hier BFS → qRaster (+ kRasterArgs)
        cull.syncFullArgs(renderer); // full-range args for the payload pass
        // voxel-foliage (spec §4.6 / §A1): fan the emitted voxel(7) clusters out of
        // qRaster into qVoxRaster + publish the voxel-raster dispatch args. No-op-cheap
        // (one re-scan) when no voxel heads were registered; gated to ?voxreg/?forcevox
        // so a pure-triangle world pays nothing. The Stage-2 voxel raster consumes it.
        if (voxActive) cull.runVoxFanout(renderer);
        // fill qHwRaster from the cut (read-only on qRaster; see the batched path).
        dispatchBatchMixed(renderer, [...cull.hwPartitionBatch()]);
      }
    }
    // PERF-VB4 (D-N45): single SW + single HW pass — 24-bit depth election (visPayloadV)
    // + full-id side buffer (visBV). Replaced the old depth1 → hwDepth → payload 2-pass.
    // SUBMIT-COALESCE (item 3): world1() now OWNS the vis CLEAR (kVisClear is the first
    // dispatch in its batched submit), so the prior standalone raster.clearVis() call is
    // gone — folding the clear into the world1 submit removes one full queue.submit drain.
    // SUBMIT-COALESCE §1c: fold the HZB chain into the raster side's last compute submit
    // (dispatchVoxel's pyr+scatter batch — HZB still runs strictly AFTER kVoxScatter,
    // UAV-synced in-pass, so it pools the identical post-vox election). Excluded when
    // frozen (HZB must NOT rebuild — legacy line below) and under ?nanprobe=1 (keeps the
    // probe insertion points' semantics exact).
    const foldHzb = coalesce && !frozen && !probeOn;
    const hzbChain = foldHzb ? hzb.batch() : [];
    raster.world1(renderer, engine.camera, hzbChain);
    if (probeRun && params.get('nanprobeat') === 'payload') probeRun(renderer);
    if (!frozen && !foldHzb) {
      hzb.build(renderer); // this frame's depth → next frame's occluder
    }
    if (probeRun && params.get('nanprobeat') === 'hzb') probeRun(renderer);
    // Nanite shadows (R0+R1): per-level light-frustum cull → depth-only SW raster
    // into our own r32 cascade textures (R1 skips a level when its VP is unchanged
    // → cached). The clipmap fits its own per-level light VPs.
    if (shadow && !frozen) shadow.run(renderer, engine.camera);
    // S0: half-res shadow eval over the FINAL vis depth (after payload), before the
    // resolve samples it. Runs every frame the resolve does (incl. frozen — the vis
    // reprojects), reading cam.invVp set by cam.update above.
    if (shadowHalf) shadowHalf.run(renderer);
    post.render(); // scene pass (resolve mesh + old-path remainder) + post chain
    if (probeRun && !params.get('nanprobeat')) probeRun(renderer); // default: after scene
    frame++;
  };

  const meter = (r: WebGPURenderer): void => {
    post.meter(r);
    const traa = post.traaNode as { _jitterIndex?: number } | null;
    if (traa && typeof traa._jitterIndex === 'number') {
      engine.stats.counters['nanite.jitterIdx'] = traa._jitterIndex;
    }
    // R1: which cascades re-rastered this frame (bitmask). Static settled camera
    // → 0 (all served from cache); moving → bit0 (c0) every frame + far on cadence.
    if (shadow) engine.stats.counters['nanite.shRaster'] = shadow.rasteredMask();
    // frame 0: no dispatch has created the GPU buffers yet — readback throws.
    // measure-infra W5: engine.meterQuiet ⇒ MeasureHarness is timing isolated frames —
    // skip the async readbacks entirely (it reads them itself OUTSIDE the timed window).
    if (engine.meterQuiet || frame === 0 || frame % 15 !== 0 || reading) return;
    reading = true;
    void meterRead(r)
      .then((m) => Object.assign(engine.stats.counters, m))
      .finally(() => {
        reading = false;
      });
  };

  // measure-infra W5 (spec-orchestration-submit-folds §4b): the counter READBACKS
  // factored out of meter() — returns the would-be assignments instead of writing them,
  // so MeasureHarness can run them on a drained queue BETWEEN samples (per-frame
  // counters in every MeasuredFrame, zero perturbation of gpuWallMs).
  const meterRead = (r: WebGPURenderer): Promise<Record<string, number>> => {
    const out: Record<string, number> = {};
    return Promise.all([
      cull.readCounts(r),
      raster.readHwCount(r),
      raster.readSplatCount(r),
      raster.readMidCount(r),
      shadow ? shadow.readCounts(r) : Promise.resolve(null),
      grass ? grass.readCounts(r) : Promise.resolve(null),
      voxActive ? cull.readVoxCount(r) : Promise.resolve(null),
      voxActive ? raster.readVoxWrites(r) : Promise.resolve(null),
      // ?voxprev gate counters (spec §6 R9): per-bucket cluster counts. B1 ≈ 0 ⇒ the
      // partition classifier is degenerate — no perf verdict may be read while so.
      voxActive && cull.voxPrevEnabled ? cull.readVoxBuckets(r) : Promise.resolve(null),
    ])
      .then(([c, hw, splatFrags, midTris, sh, grassCounts, voxCount, voxWrites, voxBuckets]) => {
        if (grassCounts) {
          out['nanite.grassClumps'] = grassCounts.clumps;
          out['nanite.grassHwTris'] = grassCounts.hwTris;
        }
        // voxel-foliage (§A1): the fanned voxel-cluster count → HUD (the Verify agent
        // reads window.__laas.stats.counters). > 0 ⇒ the cull is emitting voxel clusters
        // into qVoxRaster and the Stage-2 bin/raster has work to consume.
        if (voxCount !== null) out['nanite.voxClusters'] = voxCount;
        // Stage-2 §A2: the per-pixel BRICK-WRITE count (occlusion-skip overlay number) —
        // the elections the scatter raster actually committed (FAR below the overlapping
        // triangle fragments if the occlusion cull works). > 0 ⇒ the voxel raster produced winners.
        if (voxWrites !== null && voxWrites !== undefined)
          out['nanite.voxBrickWrites'] = voxWrites;
        if (voxBuckets) {
          out['nanite.voxB0'] = voxBuckets[0] ?? 0; // pass A: probably-visible
          out['nanite.voxB1'] = voxBuckets[1] ?? 0; // pass B: probably-occluded
        }
        if (sh) {
          let shTotal = 0;
          for (let i = 0; i < sh.length; i++) {
            out[`nanite.shC${i}`] = sh[i] ?? 0;
            shTotal += sh[i] ?? 0;
          }
          out['nanite.shTotal'] = shTotal;
        }
        out['nanite.visClusters'] = c.visClusters;
        out['nanite.dagClusters'] = c.dagClusters;
        out['nanite.visTris'] = c.visTris;
        out['nanite.dagTris'] = c.dagTris;
        // F3 triangle-budget cluster-granular attribution (always-on; piggybacks the same
        // readCounts async batch as visTris — no new stall). voxRouted = voxel-matclass
        // clusters fanned to the voxel raster; clhw = clusters routed to the HW instanced
        // draw. Both are subsets of visTris the SW classifier skips (raster slots 10/11).
        out['nanite.voxRoutedTris'] = c.voxRoutedTris;
        out['nanite.clhwTris'] = c.clhwTris;
        out['nanite.hwTris'] = hw;
        // F3 triangle-budget per-layer append counts (RAW, pre-cap). Only present on the
        // single-pass world1 path (null otherwise → omitted so the HUD shows n/a).
        if (splatFrags !== null) out['nanite.splatFrags'] = splatFrags;
        if (midTris !== null) out['nanite.midTris'] = midTris;
        if (c.overflow && warned !== c.overflow) {
          warned = c.overflow;
          // eslint-disable-next-line no-console
          console.warn(`[nanite] QUEUE OVERFLOW (geometry dropped): ${c.overflow}`);
        }
        return out;
      });
  };

  return { render, meter, meterRead, sunVis };
}

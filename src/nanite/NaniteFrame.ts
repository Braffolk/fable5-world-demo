/**
 * Full-frame nanite integration (N4-C0, D-N18/D-N19) — `?nanite=1` without
 * `?nanitedbg`: the cull→raster compute runs BEFORE the post pipeline each
 * frame, and the resolve mesh (NaniteResolve) shades the migrated classes
 * inside the main scene pass, depth-composing with everything the old
 * pipeline still draws (grass, cards, water, sky). The migrated classes' old
 * CAMERA draws are suppressed by the scene (same predicate as the registry
 * filter); their SHADOW casting is now the nanite shadow system (N5/D-N28:
 * depth-only SW raster into our own r32 cascades, default-on). The old
 * ShadowProxy / per-pool caster meshes only run under ?oldgeo (the A/B ref).
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
import type { NF } from '../gpu/TSLTypes';
import type { Engine } from '../core/Engine';
import type { PostStack } from '../render/PostStack';
import type { Heightfield } from '../world/Heightfield';
import type { GeometryRegistry } from './GeometryRegistry';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK } from './GeometryRegistry';
import { makeNaniteCam } from './NaniteCommon';
import { buildNaniteCull } from './NaniteCull';
import { buildNaniteHzb } from './NaniteHzb';
import { makeFetch } from './NaniteFetch';
import { buildNaniteRaster, makeVisBuffers } from './NaniteRaster';
import { buildNaniteResolve } from './NaniteResolve';
import { buildNaniteShadow, type NaniteShadow } from './NaniteShadow';
import { buildNaniteShadowClip } from './NaniteShadowClip';
import { buildShadowHalf, type ShadowHalf } from './NaniteShadowHalf';
import { bcU2F, dispatch, elemU, readBuffer, returnIf, texLoadR, toF, uniformArrV4, uniformF } from './Tsl';

export interface NaniteFrameHandles {
  render(): void;
  meter(renderer: WebGPURenderer): void;
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
  post: PostStack,
  world: {
    gi: import('../gpu/passes/ProbeGI').ProbeGI | null;
    canopyTex: import('three/webgpu').StorageTexture | null;
    csm: import('three/addons/csm/CSMShadowNode.js').CSMShadowNode | null;
    barkTexA: import('three').Texture | null;
    barkTexB: import('three').Texture | null;
  },
): NaniteFrameHandles {
  const renderer = engine.renderer;
  const size = renderer.getDrawingBufferSize(new Vector2());
  const params = new URLSearchParams(window.location.search);
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
  // N8-D1 continuous-LOD cut threshold τ (screen-error px). The cull applies it per
  // DAG cluster (project(own)≤τ AND project(parent)>τ); pre-DAG / terrain pools ignore
  // it. DEFAULT 3 px (banded-τ perf landing 2026-06-17: ~2× fewer visible clusters vs
  // τ=1, near-invisible at eye level — judge shots in docs/perf-runs). ?loderr=N for
  // A/B (1 = finest/sub-pixel); the setter below lets a probe sweep it live.
  const loderrParam = Number(params.get('loderr') ?? '3');
  const tau = uniformF(Number.isFinite(loderrParam) && loderrParam > 0 ? loderrParam : 3);
  // N8-D1e min-screen-size cull (D-N33): drop any cluster whose projected sphere
  // radius < minPx, and for DAG'd meshes REPLACE the finite hybrid draw envelope
  // with this sub-pixel bound (Nanite-style "draw until it vanishes" — trees no
  // longer wink out at 496 m). Crack-safe: any gap left is sub-pixel by definition.
  // DEFAULT 2 px (banded-τ perf landing) — drops clusters whose error-sphere projects
  // sub-2px (safe: any gap is sub-pixel by definition). ?nanitemin=0 to disable.
  const minpxParam = Number(params.get('nanitemin') ?? '2');
  const minPx = uniformF(Number.isFinite(minpxParam) && minpxParam > 0 ? minpxParam : 0);
  // PERF-VB3: the LOD-WARP falloff — distance-banded τ (region collapse) + plateau +
  // power. THESE WERE ONLY WIRED IN THE DEBUG VIEW (NaniteView) before — now wired here
  // too. Defaults are the validated preset (NaniteView at lodnear=4/simband=6/lodpow=0.6/
  // instminpx=256): full detail to lodNear m, τ doubles every simBandD m past it, lodPow
  // <1 = detail drops fast near / slow far. ?simband/?lodnear/?lodpow override.
  const simbandParam = Number(params.get('simband') ?? '6');
  const simBandD = uniformF(Number.isFinite(simbandParam) && simbandParam > 0 ? simbandParam : 0);
  const lodnearParam = Number(params.get('lodnear') ?? '4');
  const lodNear = uniformF(Number.isFinite(lodnearParam) && lodnearParam > 0 ? lodnearParam : 0);
  const lodpowParam = Number(params.get('lodpow') ?? '0.6');
  const lodPow = uniformF(Number.isFinite(lodpowParam) && lodpowParam > 0 ? Math.max(0.05, lodpowParam) : 1);
  // PERF-VB3 / D-N33: per-INSTANCE min screen-SIZE cull (px diameter) — drops a whole
  // instance whose projected sphere is smaller than this. THE far-field bound for the
  // hier cull (hier has no brute draw-envelope; without a bound every visible instance
  // seeds ≥1 root). Was wired only in NaniteView; now here. DEFAULT 0 = drop nothing
  // (full forest; the lodWarp below still collapses far DETAIL to roots so the count
  // stays bounded without losing trees). DEFAULT = resolution-relative 0.075 × min(framebuffer
  // dim) (≈128 px at the dev res) so far trees hand off to impostors consistently across
  // resolutions. ?instminpx=N overrides as an absolute px diameter; ?instminpx=0 disables
  // (full forest). The N9 cross-instance MERGE will remove this per-instance floor properly.
  const instMinPxDefault = Math.round(0.075 * Math.min(size.x, size.y));
  const instminpxRaw = params.get('instminpx');
  const instminpxParam =
    instminpxRaw != null && Number.isFinite(Number(instminpxRaw)) ? Number(instminpxRaw) : instMinPxDefault;
  const instMinPx = uniformF(instminpxParam > 0 ? instminpxParam : 0);

  const cam = makeNaniteCam(size.x, size.y);
  const vis = makeVisBuffers(size.x * size.y);
  // PERF-VB4 (D-N45): the WORLD is single-pass — the HZB reads the packed depth key from
  // the election anchor (visPayloadV high bits, packed=true), there is no exact depthV.
  const hzb = buildNaniteHzb(vis.payloadV.ro, cam, true);
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
    // PERF-VB3: HIERARCHICAL DAG-BFS cull is the SOLE world cull (every mesh is
    // DAG'd — terrain via TERRAIN-RW, veg via the always-on nanitedag). Single-phase
    // BFS, NON-packed two-pass raster (depthV ⇒ HZB + exact-depth world resolve unchanged).
    { tau, minPx, simBandD, lodNear, lodPow, instMinPx },
  );
  if (!hf.biomeTex || !hf.fieldsTex || !hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteFrame: heightfield derived maps missing (boot order)');
  }
  // ?nanodisp=1 — disable terrain micro-displacement (root-cause bisect for
  // near-camera transparency: the disp branch only runs within 85 m)
  const dispOff = params.get('nanodisp') === '1';
  const disp = dispOff
    ? undefined
    : {
        normalTex: hf.normalTex,
        biomeTex: hf.biomeTex,
        fieldsTex: hf.fieldsTex,
        noiseA: hf.noiseA,
        noiseB: hf.noiseB,
        camPos: cam.camPos,
      };
  // trunk wind (matches the resolve's makeFetch — both read ?nanwind so the
  // rastered geometry and the resolve's barycentric corners stay bit-identical)
  const windOn = params.get('nanwind') !== '0';
  const windOpt = windOn ? { camPos: cam.camPos } : undefined;
  const raster = buildNaniteRaster(registry.gpu, hf.heightTex, cam, cull, vis, 'flat', true, disp, windOpt, false, true);

  // Nanite shadows (N5, D-N28): depth-only SW raster into own r32 cascade textures,
  // sampled by the resolve's own PCSS. R1 caches per cascade (re-raster only on a
  // VP change) → ~0 cost static, the [1,2,3,6] cadence moving. ON BY DEFAULT;
  // ?nanshadow=0 disables the whole system (producer here + receive in the resolve,
  // same flag). Built BEFORE the resolve so the resolve binds shadowFactor; needs
  // the CSM (its cascade ortho cameras provide the per-cascade light VPs).
  const shadowOn = params.get('nanshadow') !== '0' && world.csm !== null;
  // S3 (D-N29): the SCREEN-DENSITY SHADOW CLIPMAP replaces the 4 fixed CSM
  // cascades (the resolved sun-shadow rethink — CSM dropped for shadow geometry).
  // ?shadowclip=0 A/Bs back to the cascade path (NaniteShadow.ts). world.csm stays
  // alive only as the resolve's cloud-gate carrier until that gate is re-sourced.
  const useClip = params.get('shadowclip') !== '0';
  const shadow: NaniteShadow | null = shadowOn
    ? useClip
      ? buildNaniteShadowClip(registry.gpu, registry.instanceCount, hf.heightTex, disp, windOpt)
      : buildNaniteShadow(registry.gpu, registry.instanceCount, hf.heightTex, disp, windOpt)
    : null;

  // S0 (D-N29): half-res PCSS eval + depth-aware bilateral upsample — quarters the
  // per-pixel shadow SAMPLE cost (paid every frame, static or moving). Built from
  // the MAIN vis + cam (it reconstructs wp exactly like the resolve) and shadow's
  // PCSS sampler. ?shalfres=0 falls back to the full-res per-pixel path for A/B.
  const halfResShadow = params.get('shalfres') !== '0';
  const shadowHalf: ShadowHalf | null =
    shadow && halfResShadow ? buildShadowHalf(vis, cam, shadow) : null;

  const resolve = buildNaniteResolve(registry.gpu, hf.heightTex, cam, cull, vis, {
    hf,
    gi: world.gi,
    canopyTex: world.canopyTex,
    csm: world.csm,
    barkTexA: world.barkTexA,
    barkTexB: world.barkTexB,
    naniteShadow: shadow,
    shadowHalf,
  });
  engine.scene.add(resolve.mesh);

  // ?nanprobe=1 — exact-number depth forensics: a compute kernel reads the
  // SCENE PASS depth texture and the vis buffer at up to 8 pixels into a
  // storage buffer (no PNG/tone-map/color-space layers in the way)
  const probeOn = params.get('nanprobe') === '1';
  let probeRun: ((renderer2: WebGPURenderer) => void) | null = null;
  let probeRead: (() => Promise<Float32Array>) | null = null;
  let probeSet: ((pix: number[][]) => void) | null = null;
  if (probeOn) {
    const fetchDbg = makeFetch(registry.gpu, hf.heightTex);
    const probeAttr = new StorageBufferAttribute(new Float32Array(32), 1);
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
    const cur = renderer.getDrawingBufferSize(new Vector2());
    if ((cur.x !== cam.width || cur.y !== cam.height) && warned !== 'size') {
      warned = 'size';
      // eslint-disable-next-line no-console
      console.warn(
        `[nanite] drawing buffer ${cur.x}×${cur.y} != frame ${cam.width}×${cam.height} — reload to rebuild`,
      );
    }
    const freezeNow = frozenParam && frame > 0;
    if (!frozen && freezeNow) {
      frozen = true;
      // eslint-disable-next-line no-console
      console.log('[nanite] cullfreeze: visibility frozen — fly to inspect');
    }
    cam.update(jitteredCamera());
    // PERF-VB3 HIER (single-phase BFS, the SOLE world cull): seed roots (terrain + veg)
    // → BFS-descend the DAG (reads LAST frame's HZB for occlusion) → fills qRaster
    // directly. Then ONE depth+payload over the NON-packed two-pass raster (depthV
    // written ⇒ HZB + the exact-depth world resolve are unchanged). No depth2 / phase-2.
    if (!frozen) {
      cull.runPhase1(renderer); // hier BFS → qRaster (+ kRasterArgs)
      cull.syncFullArgs(renderer); // full-range args for the payload pass
    }
    raster.clearVis(renderer);
    // PERF-VB4 (D-N45): single SW + single HW pass — 24-bit depth election (visPayloadV)
    // + full-id side buffer (visBV). Replaced the old depth1 → hwDepth → payload 2-pass.
    raster.world1(renderer, engine.camera);
    // 0a SCAR (?scar=1): the per-pixel covered-pixel denominator post-pass over the
    // FINAL world1 winners. No-op unless ?scar=1. The per-fragment band/total counters
    // are already accumulated inside world1 itself.
    raster.scar(renderer);
    if (probeRun && params.get('nanprobeat') === 'payload') probeRun(renderer);
    if (!frozen) hzb.build(renderer); // this frame's depth → next frame's occluder
    if (probeRun && params.get('nanprobeat') === 'hzb') probeRun(renderer);
    // Nanite shadows (R0+R1): per-cascade light-frustum cull → depth-only SW
    // raster into our own r32 cascade textures (R1 skips a cascade when its VP is
    // unchanged → cached). The resolve KEEPS three's CSM node built (a ×1
    // keep-alive on the empty black-slate map — NaniteResolve), so three runs its
    // setup + per-frame cascade FIT; shadow.run then reads the fitted
    // csm.lights[c].shadow.camera VPs (one frame stale, absorbed by lightMargin).
    if (shadow && !frozen) shadow.run(renderer, world.csm, engine.camera);
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
    // frame 0: no dispatch has created the GPU buffers yet — readback throws
    if (frame === 0 || frame % 15 !== 0 || reading) return;
    reading = true;
    const scarOn = params.get('scar') === '1';
    void Promise.all([
      cull.readCounts(r),
      raster.readHwCount(r),
      shadow ? shadow.readCounts(r) : Promise.resolve(null),
      scarOn ? raster.readScar(r) : Promise.resolve(null),
    ])
      .then(([c, hw, sh, scar]) => {
        if (scar) {
          // 0a SCAR readouts → HUD / window.__laas.stats.counters (the Verify agent
          // reads these). overdraw = band fragments / band covered pixels; bandShare =
          // band fragments / all frame fragments. Counters scaled ×100 where fractional
          // (the HUD/stats are integers): scarOverdrawX100, scarBandShareX1000.
          const bandPx = scar.bandPx;
          const ovX100 = bandPx > 0 ? Math.round((scar.bandFrags / bandPx) * 100) : 0;
          const shareX1000 =
            scar.totalFrags > 0 ? Math.round((scar.bandFrags / scar.totalFrags) * 1000) : 0;
          engine.stats.counters['nanite.scarBandFrags'] = scar.bandFrags;
          engine.stats.counters['nanite.scarBandPx'] = bandPx;
          engine.stats.counters['nanite.scarTotalFrags'] = scar.totalFrags;
          engine.stats.counters['nanite.scarBandClusters'] = scar.bandClusters;
          engine.stats.counters['nanite.scarOverdrawX100'] = ovX100;
          engine.stats.counters['nanite.scarBandShareX1000'] = shareX1000;
        }
        if (sh) {
          let shTotal = 0;
          for (let i = 0; i < sh.length; i++) {
            engine.stats.counters[`nanite.shC${i}`] = sh[i] ?? 0;
            shTotal += sh[i] ?? 0;
          }
          engine.stats.counters['nanite.shTotal'] = shTotal;
        }
        engine.stats.counters['nanite.visClusters'] = c.visClusters;
        engine.stats.counters['nanite.dagClusters'] = c.dagClusters;
        engine.stats.counters['nanite.visTris'] = c.visTris;
        engine.stats.counters['nanite.dagTris'] = c.dagTris;
        engine.stats.counters['nanite.chunks'] = c.chunks;
        engine.stats.counters['nanite.rejInst'] = c.rejInst;
        engine.stats.counters['nanite.rejClust'] = c.rejClust;
        engine.stats.counters['nanite.p2'] = c.p2Appends;
        engine.stats.counters['nanite.hwTris'] = hw;
        if (c.overflow && warned !== c.overflow) {
          warned = c.overflow;
          // eslint-disable-next-line no-console
          console.warn(`[nanite] QUEUE OVERFLOW (geometry dropped): ${c.overflow}`);
        }
      })
      .finally(() => {
        reading = false;
      });
  };

  return { render, meter };
}

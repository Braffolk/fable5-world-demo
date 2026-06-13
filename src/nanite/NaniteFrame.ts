/**
 * Full-frame nanite integration (N4-C0, D-N18/D-N19) — `?nanite=1` without
 * `?nanitedbg`: the cull→raster compute runs BEFORE the post pipeline each
 * frame, and the resolve mesh (NaniteResolve) shades the migrated classes
 * inside the main scene pass, depth-composing with everything the old
 * pipeline still draws (grass, cards, water, sky). The migrated classes' old
 * CAMERA draws are suppressed by the scene (same predicate as the registry
 * filter); their shadow casting is untouched until N5 — terrain casts via
 * ShadowProxy and the pools via their separate per-cascade caster meshes.
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
import { makeNaniteCam } from './NaniteCommon';
import { buildNaniteCull } from './NaniteCull';
import { buildNaniteHzb } from './NaniteHzb';
import { makeFetch } from './NaniteFetch';
import { buildNaniteRaster, makeVisBuffers } from './NaniteRaster';
import { buildNaniteResolve } from './NaniteResolve';
import { bcU2F, dispatch, elemU, readBuffer, returnIf, texLoadR, toF, uniformArrV4 } from './Tsl';

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
  world: { gi: import('../gpu/passes/ProbeGI').ProbeGI | null; canopyTex: import('three/webgpu').StorageTexture | null },
): NaniteFrameHandles {
  const renderer = engine.renderer;
  const size = renderer.getDrawingBufferSize(new Vector2());
  const params = new URLSearchParams(window.location.search);
  const occl = params.get('occl') !== '0';
  const phase2 = params.get('phase2') !== '0';
  /** ?nanhw=0 — bisect: skip the HW big/near-tri passes (expect bbox-routed
   *  holes; isolates which raster path wrote a disputed depth) */
  const hwOn = params.get('nanhw') !== '0';
  const frozenParam = params.get('cullfreeze') === '1';
  const auditOn = params.get('audit') === '1';

  const cam = makeNaniteCam(size.x, size.y);
  const vis = makeVisBuffers(size.x * size.y);
  const hzb = buildNaniteHzb(vis.depthV.ro, cam);
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
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
  const raster = buildNaniteRaster(registry.gpu, hf.heightTex, cam, cull, vis, 'flat', true, disp);
  const resolve = buildNaniteResolve(registry.gpu, hf.heightTex, cam, cull, vis, {
    hf,
    gi: world.gi,
    canopyTex: world.canopyTex,
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
      const itemIdx = pay.shiftRight(uint(7));
      const localTri = pay.bitAnd(uint(127));
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
    if (!frozen) cull.runPhase1(renderer); // tests read LAST frame's HZB
    raster.clearVis(renderer);
    raster.depth1(renderer);
    if (hwOn) raster.hwDepth(renderer, engine.camera);
    if (!frozen) {
      hzb.build(renderer); // phase-1 depth → fresh occluder
      if (phase2) cull.runPhase2(renderer); // re-test rejects, current VP
      else cull.syncFullArgs(renderer);
    }
    raster.depth2(renderer); // appended range (0 workgroups when none)
    if (hwOn) raster.hwDepth(renderer, engine.camera); // late big/near tris
    raster.payload(renderer, engine.camera); // all items vs final depth
    if (probeRun && params.get('nanprobeat') === 'payload') probeRun(renderer);
    if (auditOn) raster.audit(renderer);
    if (!frozen) hzb.build(renderer); // final — next frame's occluder
    if (probeRun && params.get('nanprobeat') === 'hzb') probeRun(renderer);
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
    // frame 0: no dispatch has created the GPU buffers yet — readback throws
    if (frame === 0 || frame % 15 !== 0 || reading) return;
    reading = true;
    void Promise.all([
      cull.readCounts(r),
      raster.readHwCount(r),
      auditOn ? raster.readAudit(r) : Promise.resolve(null),
    ])
      .then(([c, hw, aud]) => {
        if (aud) {
          engine.stats.counters['nanite.orphans'] = aud.orphans;
          engine.stats.counters['nanite.covered'] = aud.covered;
        }
        engine.stats.counters['nanite.visClusters'] = c.visClusters;
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

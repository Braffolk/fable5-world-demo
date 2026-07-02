/**
 * `?nanitedbg=flat|cluster|hzb` — world-scene debug view of the N2 pipeline:
 * two-phase cull → Option C raster → flat resolve on the live registry,
 * REPLACING the frame render via the engine's post slot (the old pipeline
 * keeps booting and updating untouched; only its render call is bypassed —
 * zero code paths shared, zero cost when the param is absent).
 *
 * Frame sequence (NANITE-SPEC.md "Culling (N2)"):
 *   cull phase 1 (prev HZB, prev VP) → SW+HW depth → fresh HZB →
 *   cull phase 2 (re-test rejects, current VP) → late SW+HW depth →
 *   payload vs final depth → final HZB (next frame's occluder) → resolve
 *
 * `cluster` paints meshlet hash colors (the N1 checkpoint); `hzb` shows a
 * pyramid level (`&hzblevel=N`); `?occl=0` disables occlusion; `?cullfreeze=1`
 * freezes visibility (cull + HZB stop; fly to inspect what was culled).
 * HUD: nanite.visClusters / chunks / rejInst / rejClust / hwTris (+ overflow
 * warnings, F14).
 */

import { Vector2 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type { Engine } from '../core/Engine';
import type { Heightfield } from '../world/Heightfield';
import type { GeometryRegistry } from './GeometryRegistry';
import { makeNaniteCam } from './NaniteCommon';
import { buildNaniteCull } from './NaniteCull';
import { buildNaniteHwRef } from './NaniteHwRef';
import { buildNaniteHzb } from './NaniteHzb';
import { buildNaniteRaster, makeVisBuffers } from './NaniteRaster';
import { uniformF } from './Tsl';

export interface NaniteViewHandles {
  render(): void;
  meter(renderer: WebGPURenderer): void;
}

export function buildNaniteView(
  engine: Engine,
  registry: GeometryRegistry,
  hf: Heightfield,
  mode: 'flat' | 'cluster' | 'hzb' | 'hwref' | 'lod',
): NaniteViewHandles {
  if (mode === 'hwref') return buildNaniteHwRef(engine, registry, hf);
  const renderer = engine.renderer;
  const size = renderer.getDrawingBufferSize(new Vector2());
  const params = new URLSearchParams(window.location.search);
  const occl = params.get('occl') !== '0';
  const frozenParam = params.get('cullfreeze') === '1';
  const hzbLevel = Number(params.get('hzblevel') ?? '1');
  /** ?audit=1 — per-frame raster consistency count (orphans must be 0) */
  const auditOn = params.get('audit') === '1';
  /** ?shade=0 — pure matClass color (probe-parity's shading-free compare) */
  const shade = params.get('shade') !== '0';

  const cam = makeNaniteCam(size.x, size.y);
  // vis buffers first: the HZB views the depth buffer, the cull consumes the
  // HZB (prev frame's content), the raster fills it — no builder cycle
  const vis = makeVisBuffers(size.x * size.y);
  // packed (hier combined) path: the raster writes no depthV — the HZB reads visA's key
  const hzb = buildNaniteHzb(vis.payloadV.ro, cam, true);
  // N9-IMP test: ?instminpx=N drops instances smaller than N px on screen (the
  // imposter far-field). ?loderr=τ sets the screen-error cut. Both default off/1.
  const instMinPx = uniformF(Math.max(0, Number(params.get('instminpx') ?? '0')));
  const loderrVal = Number(params.get('loderr') ?? '1');
  const tau = uniformF(Number.isFinite(loderrVal) ? loderrVal : 1);
  // lodWarp falloff: ?simband=S (τ doubles S m past the plateau; 0=off, the linear
  // case), ?lodnear=N (full-detail plateau radius m), ?lodpow=P (P<1 = detail drops
  // FAST near / SLOW far — concentrate detail on the player; P=1 linear).
  const simBandD = uniformF(Math.max(0, Number(params.get('simband') ?? '0')));
  const lodNear = uniformF(Math.max(0, Number(params.get('lodnear') ?? '0')));
  const lodPow = uniformF(Math.max(0.05, Number(params.get('lodpow') ?? '1')));
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
    // voxPrevTest: keep the isolated harness in ?voxprev parity with NaniteFrame
    { instMinPx, tau, simBandD, lodNear, lodPow, voxPrevTest: occl ? hzb.sphereProbablyOccluded : null },
  );
  const raster = buildNaniteRaster(
    registry.gpu,
    hf.heightTex,
    cam,
    cull,
    vis,
    mode === 'hzb' ? 'flat' : mode,
    shade,
    undefined, // disp
    undefined, // wind
    true, // packed vis buffer — the race-free combined path (the sole cull path)
  );
  // testbed hook: sweep the imposter cull / cut without re-booting (probe-forest)
  (window as unknown as { __naniteView?: unknown }).__naniteView = {
    setInstMinPx: (v: number): void => {
      instMinPx.value = v;
    },
    setTau: (v: number): void => {
      tau.value = v;
    },
    // live lodWarp tuning (no reload): __naniteView.setLod(near, scale, pow)
    setLod: (near: number, scale: number, pow: number): void => {
      lodNear.value = Math.max(0, near);
      simBandD.value = Math.max(0, scale);
      lodPow.value = Math.max(0.05, pow);
    },
  };
  const viewScene = mode === 'hzb' ? hzb.makeViewer(hzbLevel) : raster.resolveScene;

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
        `[nanite] drawing buffer ${cur.x}×${cur.y} != view ${cam.width}×${cam.height} — debug view renders stretched (reload to rebuild)`,
      );
    }
    // freeze after the first full frame so the frozen state is a real one
    const freezeNow = frozenParam && frame > 0;
    if (!frozen && freezeNow) {
      frozen = true;
      // eslint-disable-next-line no-console
      console.log('[nanite] cullfreeze: visibility frozen — fly to inspect');
    }
    cam.update(engine.camera);
    // VIS-BUFFER path (single-phase BFS): cull (reads LAST frame's HZB) → set the
    // full raster args → ONE combined Z+payload pass → HZB (next frame's occluder).
    // No depth prepass, no payload re-raster — depth + the winning triangle id are
    // written together by a single atomicMin/speculative-claim pass.
    if (!frozen) {
      cull.runPhase1(renderer);
      cull.syncFullArgs(renderer); // qRaster[0]=(nTotal,0) + full-range dispatch args
    }
    raster.clearVis(renderer);
    raster.combined(renderer, engine.camera); // SW + HW, depth & payload in one
    if (auditOn) raster.audit(renderer);
    if (!frozen) hzb.build(renderer); // this frame's depth → next frame's occluder
    renderer.render(viewScene, engine.camera);
    frame++;
  };

  const meter = (r: WebGPURenderer): void => {
    // meter() runs BEFORE render() each frame — at frame 0 no dispatch has
    // created the GPU buffers yet, so a readback would throw
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

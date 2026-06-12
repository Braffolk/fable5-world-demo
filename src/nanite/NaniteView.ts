/**
 * `?nanitedbg=flat|cluster|hzb` — world-scene debug view of the N2 pipeline:
 * two-phase cull → Option C raster → flat resolve on the live registry,
 * REPLACING the frame render via the engine's post slot (the old pipeline
 * keeps booting and updating untouched; only its render call is bypassed —
 * zero code paths shared, zero cost when the param is absent).
 *
 * Frame sequence (NANITE.md "Culling (N2)"):
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
import { buildNaniteHzb } from './NaniteHzb';
import { buildNaniteRaster, makeVisBuffers } from './NaniteRaster';

export interface NaniteViewHandles {
  render(): void;
  meter(renderer: WebGPURenderer): void;
}

export function buildNaniteView(
  engine: Engine,
  registry: GeometryRegistry,
  hf: Heightfield,
  mode: 'flat' | 'cluster' | 'hzb',
): NaniteViewHandles {
  const renderer = engine.renderer;
  const size = renderer.getDrawingBufferSize(new Vector2());
  const params = new URLSearchParams(window.location.search);
  const occl = params.get('occl') !== '0';
  /** ?phase2=0 — single-phase occlusion (disocclusion-hole A/B; probe-pan's
   *  negative control proves the gate detects what phase 2 fixes) */
  const phase2 = params.get('phase2') !== '0';
  const frozenParam = params.get('cullfreeze') === '1';
  const hzbLevel = Number(params.get('hzblevel') ?? '1');
  /** ?audit=1 — per-frame raster consistency count (orphans must be 0) */
  const auditOn = params.get('audit') === '1';

  const cam = makeNaniteCam(size.x, size.y);
  // vis buffers first: the HZB views the depth buffer, the cull consumes the
  // HZB (prev frame's content), the raster fills it — no builder cycle
  const vis = makeVisBuffers(size.x * size.y);
  const hzb = buildNaniteHzb(vis.depthV.ro, cam);
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
  );
  const raster = buildNaniteRaster(
    registry.gpu,
    hf.heightTex,
    cam,
    cull,
    vis,
    mode === 'hzb' ? 'flat' : mode,
  );
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
    if (!frozen) cull.runPhase1(renderer); // tests read LAST frame's HZB
    raster.clearVis(renderer);
    raster.depth1(renderer);
    raster.hwDepth(renderer, engine.camera);
    if (!frozen) {
      hzb.build(renderer); // phase-1 depth → fresh occluder
      if (phase2) cull.runPhase2(renderer); // re-test rejects, current VP
      else cull.syncFullArgs(renderer);
    }
    raster.depth2(renderer); // appended range (0 workgroups when none)
    raster.hwDepth(renderer, engine.camera); // late big/near tris
    raster.payload(renderer, engine.camera); // all items vs final depth
    if (auditOn) raster.audit(renderer);
    if (!frozen) hzb.build(renderer); // final — next frame's occluder
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

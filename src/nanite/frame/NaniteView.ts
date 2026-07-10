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
 * HUD: nanite.visClusters / hwTris (+ overflow warnings, F14).
 */

import { Vector2 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type { Engine } from '../../core/Engine';
import type { TerrainField } from '../world/TerrainField';
import type { GeometryRegistry } from '../world/GeometryRegistry';
import { deriveLodParams, makeNaniteCam } from '../NaniteCommon';
import { buildNaniteCull } from '../cull/NaniteCull';
import { buildNaniteHzb } from '../cull/NaniteHzb';
import { buildNaniteRaster, makeVisBuffers } from '../raster/NaniteRaster';
import { internalSize } from '../../render/RenderScale';

export interface NaniteViewHandles {
  render(): void;
  meter(renderer: WebGPURenderer): void;
}

export function buildNaniteView(
  engine: Engine,
  registry: GeometryRegistry,
  /** terrain height source for the raster's terrain fetch — the scene's
   *  TerrainField planes (scenes without terrain clusters pass their
   *  single-level field; it binds but never samples) */
  heightSrc: TerrainField,
  mode: 'flat' | 'cluster' | 'hzb' | 'lod',
): NaniteViewHandles {
  const renderer = engine.renderer;
  const size = internalSize(renderer, new Vector2()); // ?rscale: match the scene pass
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
  // LOD-warp cut params — shared with the live frame (NaniteFrame) via deriveLodParams
  // so the debug view matches the shipped defaults. The setters below sweep them live.
  const { tau, simBandD, lodNear, lodPow, instMinPx } = deriveLodParams(params, size);
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
    heightSrc,
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
    const cur = internalSize(renderer, new Vector2());
    if ((cur.x !== cam.width || cur.y !== cam.height) && warned !== 'size') {
      warned = 'size';
      // eslint-disable-next-line no-console
      console.warn(
        `[nanite] internal size ${cur.x}×${cur.y} != view ${cam.width}×${cam.height} — debug view renders stretched (reload to rebuild)`,
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

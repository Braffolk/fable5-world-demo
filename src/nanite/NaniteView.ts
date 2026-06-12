/**
 * `?nanitedbg=flat|cluster` — world-scene debug view of the N2 pipeline:
 * cull → raster → flat resolve on the live registry, REPLACING the frame
 * render via the engine's post slot (the old pipeline keeps booting and
 * updating untouched; only its render call is bypassed — zero code paths
 * shared, zero cost when the param is absent).
 *
 * `cluster` mode paints meshlet hash colors on the real world = the deferred
 * N1 USER CHECKPOINT. HUD: nanite.visClusters / nanite.chunks / nanite.hwTris
 * (+ overflow warnings, F14).
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
  mode: 'flat' | 'cluster',
): NaniteViewHandles {
  const renderer = engine.renderer;
  const size = renderer.getDrawingBufferSize(new Vector2());
  const cam = makeNaniteCam(size.x, size.y);
  // vis buffers first: the HZB views the depth buffer, the cull consumes the
  // HZB (prev frame's content), the raster fills it — no builder cycle
  const vis = makeVisBuffers(size.x * size.y);
  const occl = new URLSearchParams(window.location.search).get('occl') !== '0';
  const hzb = buildNaniteHzb(vis.depthV.ro, cam);
  const cull = buildNaniteCull(
    registry.gpu,
    registry.instanceCount,
    cam,
    occl ? hzb.sphereOccluded : null,
  );
  const raster = buildNaniteRaster(registry.gpu, hf.heightTex, cam, cull, vis, mode);

  let frame = 0;
  let reading = false;
  let warned = '';

  const render = (): void => {
    const cur = renderer.getDrawingBufferSize(new Vector2());
    if (cur.x !== cam.width || cur.y !== cam.height) {
      if (warned !== 'size') {
        warned = 'size';
        // eslint-disable-next-line no-console
        console.warn(
          `[nanite] drawing buffer ${cur.x}×${cur.y} != view ${cam.width}×${cam.height} — debug view renders stretched (rebuild by reloading)`,
        );
      }
    }
    cam.update(engine.camera);
    cull.run(renderer); // occlusion tests read LAST frame's HZB content
    raster.update(renderer, engine.camera);
    hzb.build(renderer); // this frame's depth = next frame's occluder
    renderer.render(raster.resolveScene, engine.camera);
  };

  const meter = (r: WebGPURenderer): void => {
    frame++;
    if (frame % 15 !== 0 || reading) return;
    reading = true;
    void Promise.all([cull.readCounts(r), raster.readHwCount(r)])
      .then(([c, hw]) => {
        engine.stats.counters['nanite.visClusters'] = c.visClusters;
        engine.stats.counters['nanite.chunks'] = c.chunks;
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

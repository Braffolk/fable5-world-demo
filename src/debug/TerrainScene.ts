/**
 * ?scene=terrain — terrain inspection scene (also currently ?scene=world).
 * Real CDLOD tiles + far shell + PBR terrain material, temporary sun/sky
 * lighting (replaced by the Phase-2 atmosphere stack).
 *
 * Views: ?view=hydro paints hydrology diagnostics on a preview grid.
 * ?alt=N puts the camera N meters above ground (ground-clamped spawn).
 */

import { ProbeGI } from '../gpu/passes/ProbeGI';
import { buildCanopyMap, runScatter } from '../gpu/passes/Scatter';
import { addScatterDebug } from './ScatterDebug';
import { Forests } from '../vegetation/Forests';
import { GroundRing } from '../vegetation/GroundRing';
import { buildVegLibrary } from '../vegetation/VegLibrary';
import { updateSunUniforms } from '../render/VegMaterials';
import { Heightfield } from '../world/Heightfield';
import { buildTerrainShadowProxy } from '../world/ShadowProxy';
import { TerrainTiles } from '../world/TerrainTiles';
import { PostStack } from '../render/PostStack';
import { setupSunShadows } from '../render/ShadowSetup';
import { Clouds } from '../sky/Clouds';
import { SunSky } from '../sky/SunSky';
import type { WorldContext } from './Scenes';

export async function buildTerrainScene(ctx: WorldContext): Promise<void> {
  const { engine, params, seed } = ctx;

  const hf = await Heightfield.generate(
    engine.renderer,
    params,
    seed,
    (p, m) => ctx.progress(p * 0.92, m),
  );
  (engine as unknown as { heightfield?: Heightfield }).heightfield = hf;

  if (hf.cpuHeights) {
    let maxH = -Infinity;
    for (let i = 0; i < hf.cpuHeights.length; i += 7) {
      const v = hf.cpuHeights[i] as number;
      if (v > maxH) maxH = v;
    }
    engine.stats.counters['terrain.maxH'] = Math.round(maxH);
  }

  // physical sky first: probe gathering needs the atmosphere LUTs
  ctx.progress(0.93, 'sky: baking atmosphere LUTs');
  const sunSky = new SunSky(engine, params.timeOfDay);
  await sunSky.init(engine.renderer);
  (engine as unknown as { sunSky?: SunSky }).sunSky = sunSky;

  // irradiance probe field (Phase 3 GI)
  ctx.progress(0.945, 'gi: gathering irradiance probes');
  const gi = new ProbeGI(hf, sunSky.atmosphere);
  await gi.init(engine.renderer);
  sunSky.dimAmbientForGI();
  engine.onUpdate(() => gi.tick(engine.renderer));

  // vegetation/rock placement (Phase 5): GPU clustered-Poisson scatter +
  // canopy coverage map (lighting + density field) — BEFORE tiles, which
  // consume the canopy map for under-crown ambient
  ctx.progress(0.952, 'vegetation: scattering instances');
  const scatter = await runScatter(engine.renderer, hf, seed);
  const canopyTex = await buildCanopyMap(engine.renderer, scatter.trees);
  engine.stats.counters['veg.trees'] = scatter.trees.count;
  engine.stats.counters['veg.under'] = scatter.understory.count;
  engine.stats.counters['veg.extras'] = scatter.extras.count;

  ctx.progress(0.958, 'terrain: building tiles');
  const view = new URLSearchParams(window.location.search).get('view');
  if (view === 'scatter') addScatterDebug(engine.scene, scatter);
  if (view === 'split' && hf.preErosion) {
    // erosion before/after: pre-erosion clay on the left, eroded on the right
    const pre = new TerrainTiles(hf, null, {
      heightBuf: hf.preErosion,
      neutral: true,
      screenHalf: 'left',
    });
    const post = new TerrainTiles(hf, null, { neutral: true, screenHalf: 'right' });
    engine.scene.add(pre.mesh, post.mesh);
    engine.onUpdate(() => {
      pre.update(engine.camera);
      post.update(engine.camera);
    });
  } else {
    const tiles = new TerrainTiles(hf, view, { gi, canopyTex });
    engine.scene.add(tiles.mesh);
    engine.scene.add(tiles.farShell);
    engine.scene.add(buildTerrainShadowProxy(hf));
    engine.onUpdate(() => {
      tiles.update(engine.camera);
      engine.stats.counters['terrain.tiles'] = tiles.activeTiles;
    });
  }

  // Phase 5: variant pools + GPU cull → compacted indirect draws
  const ablate = new Set(
    (new URLSearchParams(window.location.search).get('ablate') ?? '').split(','),
  );
  if (view !== 'scatter' && !ablate.has('veg')) {
    const lib = await buildVegLibrary(engine.renderer, seed, (p, m) =>
      ctx.progress(0.963 + p * 0.006, m),
    );
    const forests = new Forests(
      hf,
      scatter,
      lib,
      ablate.has('gi') ? null : gi,
      canopyTex,
    );
    forests.init(engine.renderer);
    engine.scene.add(forests.group);
    updateSunUniforms(sunSky.sun);
    engine.onUpdate(() => {
      forests.update(engine.renderer, engine.camera);
      Object.assign(engine.stats.counters, forests.counterSnapshot());
    });

    // near-field carpets: 800k-blade grass ring + 80k debris ring
    if (!ablate.has('grass')) {
      const ring = new GroundRing(hf, canopyTex, seed);
      ring.init(lib.atlases.get('beech') ?? null);
      engine.scene.add(ring.group);
      engine.onUpdate(() => {
        ring.update(engine.renderer, engine.camera);
        Object.assign(engine.stats.counters, ring.counterSnapshot());
      });
    }
  }

  // volumetric clouds (noise bake + sun-shadow map)
  ctx.progress(0.97, 'sky: baking cloud noise');
  const clouds = new Clouds(sunSky.atmosphere);
  await clouds.init(engine.renderer);

  // 4-cascade CSM + PCSS contact hardening; cloud shadows gate the sun term
  setupSunShadows(sunSky.sun, engine.camera, (wxz) => clouds.shadowAt(wxz));

  // HDR post stack: aerial perspective, clouds, GTAO, TRAA, bloom, exposure, grade
  ctx.progress(0.98, 'post: building pipeline');
  const post = new PostStack(engine, sunSky.atmosphere, params.timeOfDay, clouds);
  engine.post = post;

  ctx.hooks.setTimeOfDay = (t: number) => {
    void (async () => {
      await sunSky.setTimeOfDay(t);
      await clouds.refreshShadow(engine.renderer);
      gi.invalidate();
      post.setTimeOfDay(t);
    })();
  };
  window.addEventListener('keydown', (e) => {
    if (e.code === 'BracketLeft' || e.code === 'BracketRight') {
      void clouds.refreshShadow(engine.renderer);
      post.setTimeOfDay(sunSky.timeOfDay);
    }
  });

  // camera: ground-clamped spawn (?alt=) or a default SE vista
  const q = new URLSearchParams(window.location.search);
  const alt = Number(q.get('alt') ?? NaN);
  if (params.cam === null) {
    if (Number.isFinite(alt)) {
      const x = Number(q.get('x') ?? 600);
      const z = Number(q.get('z') ?? 900);
      const yaw = Number(q.get('yaw') ?? 2.4); // rad; 0 = looking −z (north)
      const pitch = Number(q.get('pitch') ?? NaN); // rad; negative = down
      const y = hf.heightAtCpu(x, z) + alt;
      engine.camera.position.set(x, y, z);
      const dy = Number.isFinite(pitch) ? Math.sin(pitch) * 100 : -4;
      const dh = Number.isFinite(pitch) ? Math.cos(pitch) * 100 : 100;
      engine.camera.lookAt(x - Math.sin(yaw) * dh, y + dy, z - Math.cos(yaw) * dh);
    } else {
      engine.camera.position.set(1500, 1000, 1900);
      engine.camera.lookAt(0, 350, -300);
    }
  }
  // soft ground collision for fly camera
  engine.onUpdate(() => {
    const c = engine.camera.position;
    const ground = hf.heightAtCpu(c.x, c.z) + 1.4;
    if (c.y < ground) c.y = ground;
  });

  ctx.progress(1, 'terrain ready');
}

/**
 * ?scene=terrain — terrain inspection scene (also currently ?scene=world).
 * Real CDLOD tiles + far shell + PBR terrain material, temporary sun/sky
 * lighting (replaced by the Phase-2 atmosphere stack).
 *
 * Views: ?view=hydro paints hydrology diagnostics on a preview grid.
 * ?alt=N puts the camera N meters above ground (ground-clamped spawn).
 */

import { BOOKMARKS, installBookmarks } from './Bookmarks';
import { Froxels } from '../gpu/passes/Froxels';
import { PARTICLE_COUNT, Particles } from '../gpu/passes/Particles';
import { ProbeGI } from '../gpu/passes/ProbeGI';
import { buildCanopyMap, runScatter } from '../gpu/passes/Scatter';
import { addScatterDebug } from './ScatterDebug';
import { Forests } from '../vegetation/Forests';
import { GroundRing } from '../vegetation/GroundRing';
import { buildVegLibrary } from '../vegetation/VegLibrary';
import { CausticsBake, setCausticContext } from '../render/Caustics';
import { setWindContext, windU } from '../render/Wind';
import { sunU, updateSunUniforms } from '../render/VegMaterials';
import { buildCanopyShell } from '../world/CanopyShell';
import { Heightfield } from '../world/Heightfield';
import { buildTerrainShadowProxy } from '../world/ShadowProxy';
import { TerrainTiles } from '../world/TerrainTiles';
import { WaterSurface } from '../world/WaterSurface';
import { PostStack } from '../render/PostStack';
import { setupSunShadows } from '../render/ShadowSetup';
import { Clouds } from '../sky/Clouds';
import { SunSky } from '../sky/SunSky';
import type { WorldContext } from './Scenes';
import type { GeometryRegistry } from '../nanite/GeometryRegistry';

export async function buildTerrainScene(ctx: WorldContext): Promise<void> {
  const { engine, params, seed } = ctx;
  let naniteRegistry: GeometryRegistry | null = null;
  /** material classes the nanite full-frame mode owns (D-N19) — set when the
   *  registry builds; drives old-path camera-draw suppression */
  let naniteClasses: ReadonlySet<string> | null = null;
  /** bark texture-array (from VegLib) for the nanite resolve — hoisted out of
   *  the veg block so the full-frame build below can thread it */
  let naniteBark: { texA: import('three').Texture; texB: import('three').Texture } | null = null;
  const qNan = new URLSearchParams(window.location.search);
  /** `?nanite=1` without a debug view = full-frame mode (N4); `?naniteframe=0`
   *  keeps N1 build-only semantics (boot probes) */
  const naniteFrameMode =
    qNan.get('nanite') === '1' && !qNan.get('nanitedbg') && qNan.get('naniteframe') !== '0';

  // ── USER DIRECTIVE (2026-06-13): OLD GEOMETRY HARD-DISABLED ──────────────
  // Every default (non-nanite) SOLID-GEOMETRY render path is switched OFF so
  // the ONLY thing that can appear in the world is the nanite-rendered output.
  // No fallback: with this true, ?nanite=0 shows bare sky. The DATA those
  // systems produce (heightfield, scatter, VegLibrary, GI, canopy map) still
  // builds because the nanite registry is constructed from it — only the
  // camera-pass meshes are withheld from engine.scene. Environment systems
  // (sky/atmosphere/clouds/froxels/CSM/post) stay on so there is a frame to
  // look at. Flip to false to restore the full old pipeline (the N7 A/B path).
  // Overrides the NANITE-SPEC.md "?nanite=0 boots the untouched old pipeline"
  // constraint deliberately, for the duration of the nanite build.
  // DEFAULT = disabled; `?oldgeo=1` restores the full old world — used ONLY to
  // capture the parity reference for the N4 lighting gate (and to bring back
  // the CSM shadow casters the nanite terrain receives from until N5). The
  // default still shows bare nanite; this is a gate harness, not a fallback.
  const DISABLE_OLD_GEOMETRY = qNan.get('oldgeo') !== '1';

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

  // physical sky first: probe gathering needs the atmosphere LUTs.
  // ?shot=N boots straight into a composed bookmark — use ITS time of day
  const bootBm = params.shot !== null ? BOOKMARKS[params.shot - 1] : undefined;
  const bootTod = bootBm?.tod ?? params.timeOfDay;
  ctx.progress(0.93, 'sky: baking atmosphere LUTs');
  const sunSky = new SunSky(engine, bootTod);
  await sunSky.init(engine.renderer);
  (engine as unknown as { sunSky?: SunSky }).sunSky = sunSky;
  // tooling probe handle (tools/probe-state.ts) — light/scene state triage
  (window as unknown as { __laasDbg?: unknown }).__laasDbg = { engine, sunSky };

  // vegetation/rock placement (Phase 5): GPU clustered-Poisson scatter +
  // canopy coverage map — BEFORE the probe field (probes ray-march the bare
  // heightfield; the canopy map is their only knowledge of the forest) and
  // before tiles (under-crown ambient)
  ctx.progress(0.94, 'vegetation: scattering instances');
  const scatter = await runScatter(engine.renderer, hf, seed);
  const canopyTex = await buildCanopyMap(engine.renderer, scatter.trees);
  engine.stats.counters['veg.trees'] = scatter.trees.count;
  engine.stats.counters['veg.under'] = scatter.understory.count;
  engine.stats.counters['veg.extras'] = scatter.extras.count;
  engine.stats.counters['veg.stones'] = scatter.stones.count;

  const ablate = new Set(
    (new URLSearchParams(window.location.search).get('ablate') ?? '').split(','),
  );

  // irradiance probe field (Phase 3 GI; canopy-aware since Phase 5 —
  // ?ablate=canopygi rebuilds the bare-heightfield field for A/B)
  ctx.progress(0.95, 'gi: gathering irradiance probes');
  const gi = new ProbeGI(
    hf,
    sunSky.atmosphere,
    ablate.has('canopygi') ? null : canopyTex,
  );
  await gi.init(engine.renderer);
  sunSky.dimAmbientForGI();
  engine.onUpdate(() => gi.tick(engine.renderer));

  // Phase 6 caustics: per-frame analytic bake + module context — MUST be
  // set before any material factory runs (terrain tiles, rocks, debris all
  // self-apply at build time). ?ablate=caustics to A/B, ?caustk=N to tune.
  if (!ablate.has('caustics')) {
    const bake = new CausticsBake();
    const ck = Number(new URLSearchParams(window.location.search).get('caustk') ?? NaN);
    if (Number.isFinite(ck)) bake.focusK.value = ck;
    setCausticContext({ hf, bake, sunDir: sunU.dir });
    engine.onUpdate(() => bake.update(engine.renderer));
  }

  // Phase 6 wind: global gust field for all vegetation (?wind=N strength,
  // ?winddir=deg, ?ablate=wind to A/B) — context before veg materials build
  if (!ablate.has('wind') && hf.noiseA) {
    setWindContext({ noiseA: hf.noiseA, canopyTex });
    const q0 = new URLSearchParams(window.location.search);
    const ws = Number(q0.get('wind') ?? NaN);
    if (Number.isFinite(ws)) windU.strength.value = ws;
    const wdeg = Number(q0.get('winddir') ?? NaN);
    if (Number.isFinite(wdeg)) {
      windU.dir.value.set(Math.cos((wdeg * Math.PI) / 180), Math.sin((wdeg * Math.PI) / 180));
    }
  }

  ctx.progress(0.958, 'terrain: building tiles');
  let tilesRef: TerrainTiles | null = null;
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
  } else if (!DISABLE_OLD_GEOMETRY) {
    const tiles = new TerrainTiles(hf, view, { gi, canopyTex });
    tilesRef = tiles;
    engine.scene.add(tiles.mesh);
    engine.scene.add(tiles.farShell);
    // ?ablate=proxy — drop the terrain shadow caster (shadow-debug bisect)
    if (!ablate.has('proxy')) engine.scene.add(buildTerrainShadowProxy(hf));
    engine.onUpdate(() => {
      // suppressed by the nanite full-frame mode (D-N19): skip the CDLOD
      // walk too — the registry heightfield owns the camera-pass terrain
      if (!tiles.mesh.visible) return;
      tiles.update(engine.camera);
      engine.stats.counters['terrain.tiles'] = tiles.activeTiles;
    });
  }

  // Phase 6: stream/lake water clipmap (?ablate=water to A/B)
  if (view !== 'split' && !ablate.has('water') && !DISABLE_OLD_GEOMETRY) {
    const water = new WaterSurface(
      hf,
      sunSky.atmosphere,
      canopyTex,
      ablate.has('gi') ? null : gi,
    );
    engine.scene.add(water.group);
    engine.onUpdate(() => water.update(engine.camera));
  }

  // Phase 5: variant pools + GPU cull → compacted indirect draws
  let forestsRef: Forests | null = null;
  if (view !== 'scatter' && !ablate.has('veg')) {
    const lib = await buildVegLibrary(engine.renderer, seed, (p, m) =>
      ctx.progress(0.963 + p * 0.006, m),
    );
    // sun uniforms feed the nanite terrain shading too — keep them current
    // even when the old veg render is disabled
    updateSunUniforms(sunSky.sun);
    naniteBark = lib.barkArray; // resolve bark/deadwood sampled-array (N4-C3)
    if (!DISABLE_OLD_GEOMETRY) {
      const forests = new Forests(
        hf,
        scatter,
        lib,
        ablate.has('gi') ? null : gi,
        canopyTex,
      );
      forests.init(engine.renderer);
      forestsRef = forests;
      engine.scene.add(forests.group);
      engine.onUpdate(() => {
        forests.update(engine.renderer, engine.camera);
        Object.assign(engine.stats.counters, forests.counterSnapshot());
      });
    }

    // ?nanite=1 — N1-C4: build the GeometryRegistry from all opaque pools
    // (cluster tables + packed mega-buffers only; rendering unchanged until
    // N2/N3). ?nanite=0/absent: this block never runs.
    if (new URLSearchParams(window.location.search).get('nanite') === '1') {
      ctx.progress(0.985, 'nanite: clusterizing opaque pools');
      const { buildWorldRegistry, PORTED_CLASSES } = await import('../nanite/WorldRegistry');
      // D-N19 migration set: explicit ?naniteclasses=csv|all wins; full-frame
      // mode defaults to the ported set; dbg/build-only modes take everything
      type MatCls = 'terrain' | 'rock' | 'bark' | 'deadwood';
      const ALL: readonly MatCls[] = ['terrain', 'rock', 'bark', 'deadwood'];
      const clsParam = qNan.get('naniteclasses');
      let classes: Set<MatCls> | undefined;
      if (clsParam && clsParam !== 'all') {
        classes = new Set(
          clsParam.split(',').filter((c): c is MatCls => (ALL as readonly string[]).includes(c)),
        );
      } else if (!clsParam && naniteFrameMode) {
        classes = new Set(PORTED_CLASSES as readonly MatCls[]);
      }
      naniteClasses = classes ?? new Set(ALL);
      // N8-D1: ?nanitedag=rock|bark|deadwood|all → continuous-LOD DAG for those
      // explicit classes (built sync at boot for now; D1d moves it to a Worker).
      const dagParam = qNan.get('nanitedag');
      let dagClasses: Set<MatCls> | undefined;
      if (dagParam) {
        const want =
          dagParam === 'all'
            ? (['rock', 'bark', 'deadwood'] as MatCls[])
            : dagParam.split(',').filter((c): c is MatCls => (ALL as readonly string[]).includes(c));
        dagClasses = new Set(want.filter((c) => c !== 'terrain'));
      }
      // N8-D2 Stage 2e (D-N39) — the "boot only to dag" FLIP: terrain is the full-res
      // clip-STREAMED DAG by default, no window-grid fallback. `?nanitedterrain` absent ⇒
      // production default (gridN 128, clip on). `?nanitedterrain=0` is the explicit opt-out
      // to the legacy implicit window grid (tooling / A-B). An explicit `?nanitedterrain=<gridN>`
      // (>0) selects that grid and stays one-shot uniform unless `?nanitedclip=1` (preserves
      // the per-flag tool semantics — probe-dterrain etc.).
      const dterrainParam = qNan.get('nanitedterrain');
      const terrainDefault = dterrainParam == null;
      const dagTerrainGridN = terrainDefault ? 128 : Math.max(0, Math.floor(Number(dterrainParam)));
      // N8-D2 (D-N38): ?nanitedtiles=T → split the terrain DAG into T×T tiles.
      const dtilesParam = qNan.get('nanitedtiles');
      const dagTerrainTiles = dtilesParam ? Math.max(1, Math.floor(Number(dtilesParam))) : 1;
      // N8-D2 Stage 2b-1 (D-N39): ?nanitedpool=1 → route terrain tiles through the
      // streaming tile POOL (reserveTilePool/attachHeightDagTile) rather than the
      // per-tile registerHeightDag+attachHeightDag path. GPU-render parity proof.
      const dagTerrainPool = qNan.get('nanitedpool') === '1';
      // N8-D2 Stage 2b-2/2e (D-N39): the geometry CLIPMAP (concentric same-gridN rings,
      // true full-res at the center, coarse to the field edge, bounded). DEFAULT ON (the
      // 2e flip); `?nanitedclip=1` also forces it for an explicit gridN. Implies the pool.
      const dagTerrainClip = terrainDefault || qNan.get('nanitedclip') === '1';
      // N8-D2 Stage 2d: ?nanitedskirt=0 disables the inter-level seam skirts (A/B). Default ON.
      const dagTerrainSkirt = qNan.get('nanitedskirt') !== '0';
      const wr = await buildWorldRegistry({
        renderer: engine.renderer,
        hf,
        scatter,
        lib,
        counters: engine.stats.counters,
        seed: seed.seed,
        ...(classes ? { classes } : {}),
        ...(dagClasses && dagClasses.size > 0 ? { dag: dagClasses } : {}),
        ...(dagTerrainGridN > 0 ? { dagTerrainGridN } : {}),
        ...(dagTerrainTiles > 1 ? { dagTerrainTiles } : {}),
        ...(dagTerrainPool ? { dagTerrainPool: true } : {}),
        ...(dagTerrainClip ? { dagTerrainClip: true } : {}),
        ...(dagTerrainSkirt ? {} : { dagTerrainSkirt: false }),
      });
      (engine as unknown as { naniteRegistry?: unknown }).naniteRegistry = wr.registry;
      naniteRegistry = wr.registry;
      // eslint-disable-next-line no-console
      console.log(
        `[laas] nanite registry: total ${wr.totalMs.toFixed(0)} ms (readback ` +
          `${wr.readbackMs.toFixed(0)} + partition ${wr.partitionMs.toFixed(0)} + terrain minMax ` +
          `${wr.terrainMs.toFixed(0)} + build ${wr.buildMs.toFixed(0)}` +
          (wr.dagMeshes > 0
            ? ` + DAG ${wr.dagMeshes}m/${wr.dagBuildMs.toFixed(0)}ms/${(wr.dagTris / 1000).toFixed(0)}k tris`
            : '') +
          `); deferred instances ${wr.deferredInstances}\n${wr.report.table}\ndeferred: ${wr.deferred.join('; ')}`,
      );
      // N8-D2 Stage 2b-3 (D-N39): drive the clipmap streamer from the live camera
      // — re-center the 1 m detail rings each frame (evict departed / stream in
      // arrived). The coarser resident ring backstops in-flight loads ⇒ no holes.
      if (wr.terrainStreamer) {
        const streamer = wr.terrainStreamer;
        engine.onUpdate(() => {
          streamer.update(engine.camera.position.x, engine.camera.position.z);
          Object.assign(engine.stats.counters, streamer.counters());
        });
      }
    }

    // near-field carpets: 800k-blade grass ring + 80k debris ring
    if (!ablate.has('grass') && !DISABLE_OLD_GEOMETRY) {
      const ring = new GroundRing(hf, canopyTex, seed, ablate.has('gi') ? null : gi);
      ring.init(lib.atlases.get('beech') ?? null);
      engine.scene.add(ring.group);
      engine.onUpdate(() => {
        ring.update(engine.renderer, engine.camera);
        Object.assign(engine.stats.counters, ring.counterSnapshot());
      });
    }

    // far forests: aggregate canopy shell beyond the impostor mid-band
    if (!ablate.has('shell') && !DISABLE_OLD_GEOMETRY) {
      engine.scene.add(buildCanopyShell(hf, canopyTex));
    }
  }

  // volumetric clouds (noise bake + sun-shadow map)
  ctx.progress(0.97, 'sky: baking cloud noise');
  const clouds = new Clouds(sunSky.atmosphere);
  await clouds.init(engine.renderer);
  // weather motion (Pillar F): drift on WORLD time so ?freeze=1 shots stay
  // deterministic; the drifted shadow map re-bakes itself every ~2.5 s
  let lastWt = 0;
  engine.onUpdate((_dt, wt) => {
    clouds.tick(engine.renderer, wt - lastWt);
    lastWt = wt;
  });

  // 4-cascade CSM + PCSS contact hardening; cloud shadows gate the sun term
  const shadowRig = setupSunShadows(sunSky.sun, engine.camera, (wxz) =>
    clouds.shadowAt(wxz),
  );
  // cascade cameras drive the per-cascade caster cull in Forests
  forestsRef?.setCSM(shadowRig.csm ?? null);
  (window as unknown as { __laasDbg?: Record<string, unknown> }).__laasDbg = {
    engine,
    sunSky,
    shadowRig,
  };

  // GPU particles: snow/pollen/leaves riding the wind (?ablate=particles)
  if (view !== 'split' && !ablate.has('particles') && !DISABLE_OLD_GEOMETRY) {
    const parts = new Particles(hf, canopyTex, ablate.has('gi') ? null : gi);
    engine.scene.add(parts.mesh);
    engine.onUpdate((dt) => parts.update(engine.renderer, engine.camera, dt));
    engine.stats.counters['particles'] = PARTICLE_COUNT;
  }

  // froxel volumetrics: canopy shafts + valley fog (?ablate=froxels, ?fog=N)
  let froxels: Froxels | null = null;
  if (!ablate.has('froxels')) {
    froxels = new Froxels(hf, sunSky.atmosphere, canopyTex, clouds);
    const fq = Number(new URLSearchParams(window.location.search).get('fog') ?? NaN);
    if (Number.isFinite(fq)) froxels.fogK.value = fq;
    const fx = froxels;
    engine.onUpdate(() => fx.update(engine.renderer, engine.camera));
  }

  // HDR post stack: aerial perspective, clouds, GTAO, TRAA, bloom, exposure, grade
  ctx.progress(0.98, 'post: building pipeline');
  const post = new PostStack(engine, sunSky.atmosphere, bootTod, clouds, froxels);
  engine.post = post;

  // ?nanitedbg=flat|cluster (needs ?nanite=1) — N2 debug view: cull → raster
  // → flat resolve replaces the frame render via the post slot; the old
  // pipeline keeps booting/updating untouched. `cluster` = the deferred N1
  // checkpoint (meshlet colors on the real world); `hwref` = the N3 parity
  // reference (same content, hardware instanced draws).
  const nanitedbg = new URLSearchParams(window.location.search).get('nanitedbg');
  if (nanitedbg === 'flat' || nanitedbg === 'cluster' || nanitedbg === 'hzb' || nanitedbg === 'hwref') {
    if (naniteRegistry) {
      const { buildNaniteView } = await import('../nanite/NaniteView');
      engine.post = buildNaniteView(engine, naniteRegistry, hf, nanitedbg);
      // eslint-disable-next-line no-console
      console.log(`[laas] nanitedbg=${nanitedbg}: N2 debug view replacing the frame render`);
    } else {
      // eslint-disable-next-line no-console
      console.warn('[laas] ?nanitedbg needs ?nanite=1 with vegetation enabled — ignored');
    }
  } else if (naniteRegistry && naniteClasses && naniteFrameMode) {
    // N4 full-frame mode (D-N18/D-N19): nanite compute + in-scene resolve own
    // the migrated classes; their old camera draws hide (shadow casting stays
    // on the old path until N5 — ShadowProxy + per-cascade caster siblings)
    const { buildNaniteFrame } = await import('../nanite/NaniteFrame');
    const { migratedMatClass } = await import('../nanite/WorldRegistry');
    engine.post = buildNaniteFrame(engine, naniteRegistry, hf, post, {
      gi: ablate.has('gi') ? null : gi,
      canopyTex,
      csm: shadowRig.csm ?? null,
      barkTexA: naniteBark?.texA ?? null,
      barkTexB: naniteBark?.texB ?? null,
    });
    if (naniteClasses.has('terrain') && tilesRef) {
      tilesRef.mesh.visible = false;
      tilesRef.farShell.visible = false;
    }
    const hidden = forestsRef?.suppressMigrated(migratedMatClass, naniteClasses) ?? 0;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] nanite full-frame: classes [${[...naniteClasses].join(',')}]; suppressed ` +
        `${hidden} pool draws${naniteClasses.has('terrain') ? ' + terrain tiles/far shell' : ''}`,
    );
  }

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

  // terrain/water probe for the camera rig: walk-mode ground physics + the
  // fly-mode soft collision / underwater guard both live in FlyCamera now
  ctx.hooks.groundProbe = (x, z) => ({
    ground: hf.heightAtCpu(x, z),
    water: hf.waterYAtCpu(x, z),
  });

  // camera spawn: ground-clamped (?alt/x/z → fly) or the DEFAULT WALK SPAWN
  // at the map center — first dry, reasonably flat spot on a spiral out
  // from (0,0), eye at head height, facing the NE massif
  const q = new URLSearchParams(window.location.search);
  const alt = Number(q.get('alt') ?? NaN);
  if (params.cam === null) {
    if (Number.isFinite(alt)) {
      const x = Number(q.get('x') ?? 600);
      const z = Number(q.get('z') ?? 900);
      const yaw = Number(q.get('yaw') ?? 2.4); // rad; 0 = looking −z (north)
      const pitch = Number(q.get('pitch') ?? -0.04); // rad; negative = down
      const y = hf.heightAtCpu(x, z) + alt;
      // the fly camera doesn't exist yet — main applies this after rigging
      ctx.hooks.initialPose = { p: [x, y, z], yaw, pitch };
      ctx.hooks.initialPoseMode = 'fly';
      engine.camera.position.set(x, y, z);
    } else {
      const spawn = findWalkSpawn(hf);
      ctx.hooks.initialPose = {
        p: [spawn.x, hf.heightAtCpu(spawn.x, spawn.z) + 1.7, spawn.z],
        yaw: -0.78, // face NE — the serrated massif anchors the first frame
        pitch: -0.02,
      };
      ctx.hooks.initialPoseMode = 'walk';
      engine.camera.position.set(spawn.x, ctx.hooks.initialPose.p[1], spawn.z);
    }
  }

  // composed bookmarks (keys 1-9, ?shot=N) + 92 s flythrough (?fly=1 / F)
  installBookmarks(engine, hf, ctx.hooks, params);

  ctx.progress(1, 'terrain ready');
}

/**
 * Default walk spawn: first dry, reasonably flat spot on a coarse spiral
 * out from the map center (dry = waterY sits below the bed there; flat =
 * central-difference slope under ~19°).
 */
function findWalkSpawn(hf: Heightfield): { x: number; z: number } {
  for (let r = 0; r <= 240; r += 12) {
    const steps = Math.max(1, Math.round((2 * Math.PI * r) / 18));
    for (let k = 0; k < steps; k++) {
      const a = (k / steps) * Math.PI * 2;
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      const h = hf.heightAtCpu(x, z);
      if (hf.waterYAtCpu(x, z) > h - 0.05) continue; // wet or waterline
      const sx = hf.heightAtCpu(x + 6, z) - hf.heightAtCpu(x - 6, z);
      const sz = hf.heightAtCpu(x, z + 6) - hf.heightAtCpu(x, z - 6);
      if (Math.hypot(sx, sz) / 12 > 0.35) continue; // too steep
      return { x, z };
    }
  }
  return { x: 0, z: 0 };
}

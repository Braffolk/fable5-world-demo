/**
 * ?scene=terrain — terrain inspection scene (also currently ?scene=world).
 * Real CDLOD tiles + far shell + PBR terrain material, temporary sun/sky
 * lighting (replaced by the Phase-2 atmosphere stack).
 *
 * Views: ?view=hydro paints hydrology diagnostics on a preview grid.
 * ?alt=N puts the camera N meters above ground (ground-clamped spawn).
 */

import { BOOKMARKS, installBookmarks } from './Bookmarks';
import { BootTrace } from './BootTrace';
import { Froxels } from '../gpu/passes/Froxels';
import { PARTICLE_COUNT, Particles } from '../gpu/passes/Particles';
import { ProbeGI } from '../gpu/passes/ProbeGI';
import { CanopyWindow } from '../gpu/passes/CanopyWindow';
import { addScatterDebug } from './ScatterDebug';
import { buildVegLibrary } from '../vegetation/VegLibrary';
import { CausticsBake, setCausticContext } from '../render/Caustics';
import { setWindContext, windU } from '../render/Wind';
import { sunU, updateSunUniforms } from '../render/VegMaterials';
import { Heightfield } from '../world/Heightfield';
import { GeneratedWorldSource } from '../world/source/GeneratedWorldSource';
import { RemoteWorldSource } from '../world/source/RemoteWorldSource';
import type { WorldSource } from '../world/source/WorldSource';
import { buildChunkContentStreams, type ChunkContentStreams } from '../nanite/world/ChunkContent';
import { StreamBrainClient } from '../nanite/world/StreamBrainClient';
import { StreamOrigin } from '../nanite/world/StreamOrigin';
import { buildSpeciesMap } from '../nanite/world/SpeciesMap';
import { InstanceBand } from '../nanite/world/InstanceBand';
import { VegClass } from '../gpu/passes/Scatter';
import { chunkBox, coverageCenter } from '../nanite/world/PlaneFill';
import type { TerrainField } from '../nanite/world/TerrainField';
import { WaterSurface } from '../world/WaterSurface';
import { PostStack } from '../render/PostStack';
import { Clouds } from '../sky/Clouds';
import { SunSky } from '../sky/SunSky';
import type { WorldContext } from './Scenes';
import type { GeometryRegistry } from '../nanite/world/GeometryRegistry';

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
  /** nanite is THE renderer (unconditional since 2026-07-10); no debug view =
   *  full-frame mode (N4); `?naniteframe=0` keeps N1 build-only semantics (boot probes) */
  const naniteFrameMode = !qNan.get('nanitedbg') && qNan.get('naniteframe') !== '0';

  // ── COLD-BOOT OVERLAP (2026-07-04): the VegLibrary build and the nanite
  // crown/DAG worker prep depend only on (renderer, seed, lib) — kick them NOW
  // so their CPU work (yield-sliced; crowns/DAGs on a Worker pool) interleaves
  // with the GPU-await boot phases below (heightfield/erosion/sky/scatter/GI)
  // instead of serializing after them. Determinism: veg geometry is seeded per
  // label (seed.rng is stateless), the atlas/impostor captures use bespoke
  // materials (no wind/caustic context reads), and prep results are keyed by
  // idF / job index — completion order cannot reorder them.
  const ablate = new Set(
    (new URLSearchParams(window.location.search).get('ablate') ?? '').split(','),
  );
  const view = new URLSearchParams(window.location.search).get('view');
  const vegEnabled = view !== 'scatter' && !ablate.has('veg');
  // D-N19 migration set: explicit ?naniteclasses=csv|all wins; full-frame mode
  // defaults to the ported set; dbg/build-only modes take everything. Resolved
  // in ONE place — the early prep and the registry build must never drift.
  type MatCls = 'terrain' | 'rock' | 'bark' | 'deadwood';
  const NAN_ALL: readonly MatCls[] = ['terrain', 'rock', 'bark', 'deadwood'];
  const resolveNaniteSetup = (
    ported: readonly MatCls[],
  ): { classes: Set<MatCls> | undefined; dagClasses: Set<MatCls>; naniteLeaf: boolean } => {
    const clsParam = qNan.get('naniteclasses');
    let classes: Set<MatCls> | undefined;
    if (clsParam && clsParam !== 'all') {
      classes = new Set(
        clsParam.split(',').filter((c): c is MatCls => (NAN_ALL as readonly string[]).includes(c)),
      );
    } else if (!clsParam && naniteFrameMode) {
      classes = new Set(ported);
    }
    return {
      classes,
      // N8-D1 / PERF-VB3: continuous-LOD DAG for EVERY veg class — ALWAYS ON (the
      // old `?nanitedag` selector is retired; terrain rides its own DAG path).
      dagClasses: new Set(['rock', 'bark', 'deadwood'] as MatCls[]),
      // N9-C0/C2: real mesh-leaf crowns DEFAULT ON since 2026-07-03 (world-hookup
      // arc) — without them the nanite-only world renders LEAFLESS. ?naniteleaf=0
      // is the A/B opt-out.
      naniteLeaf: qNan.get('naniteleaf') !== '0',
    };
  };
  const leafDensityQ = Number(qNan.get('naniteleafdensity'));
  const worldRegistryModule = import('../nanite/world/WorldRegistry');
  let vegLibPromise: ReturnType<typeof buildVegLibrary> | null = null;
  let vegPrepPromise: Promise<import('../nanite/world/WorldRegistry').WorldVegPrep> | null = null;
  if (vegEnabled) {
    const endVegSpan = BootTrace.span('veg library (overlapped with GPU phases)');
    // no progress callback: the bar is owned by the serial phases; interleaved
    // updates would jump backwards
    vegLibPromise = buildVegLibrary(
      engine.renderer,
      seed,
      () => {},
      // N9-C0: ?naniteleafdensity=N caps the nanite leaf head's per-crown anchor
      // budget (real-leaf fullness vs memory/cluster cost). Default 2500.
      Number.isFinite(leafDensityQ) && leafDensityQ > 0 ? { leafAnchorTarget: leafDensityQ } : undefined,
    );
    vegLibPromise.then(
      () => endVegSpan(),
      () => endVegSpan(),
    );
    if (worldRegistryModule) {
      vegPrepPromise = (async () => {
        const [wr, lib] = await Promise.all([worldRegistryModule, vegLibPromise as NonNullable<typeof vegLibPromise>]);
        const setup = resolveNaniteSetup(wr.PORTED_CLASSES as readonly MatCls[]);
        return wr.prepareWorldVeg({
          renderer: engine.renderer,
          lib,
          seed: seed.seed,
          ...(setup.classes ? { classes: setup.classes } : {}),
          dag: setup.dagClasses,
          leaf: setup.naniteLeaf,
        });
      })();
      // boot failures elsewhere must not surface as an unhandled rejection here;
      // the real await (buildWorldRegistry) still sees the error
      void vegPrepPromise.catch(() => undefined);
    }
  }

  // WORLD SOURCE (S2): heightfield + scatter run ONCE inside the source; the scene
  // consumes chunk-keyed views for placement and the live hf/scatter handles for the
  // consumers whose windowed ports land at S3a/S4.
  // STREAMED WORLD (S6): ?src=estonia streams the cooked Estonia release through
  // RemoteWorldSource; default = the procedural GeneratedWorldSource. This is the
  // ONE source-construction site (law 3) — every subsystem downstream rides the
  // WorldSource/WorldManifest/TerrainField abstraction and never learns which
  // source feeds it.
  const streamed = params.src === 'estonia';
  // S7 streamed-instance pool + band sizing (§5, A4/A12). A tree consumes 2 slots
  // (trunk + crown head), a boulder 1; 8k/block × 24 blocks = 196k parked slots
  // (~6.3 MB f32 mirror + equal VRAM). This is ABOVE §6's 40k/2 MB estimate — that
  // predated (a) the per-head trunk+crown split and (b) whole-2 km-chunk residency
  // (records are per-chunk; we load the whole chunk though only ~300 m is visible).
  // S8: a tree now consumes 3 slots (trunk + leaf mesh + voxel crown), so the band's
  // wanted set grew ~1.5× — measured pilot 4 dense LOD0 chunks = ~190 k slots wanted,
  // which SATURATED the 24-block pool ("full of wanted chunks" drops). 40 blocks
  // (327 k slots) clears the wanted set with a motion-churn margin (band.* HUD reports
  // live usage). Mirror/GPU A-B cost = 40·8192·32 B ≈ 10.5 MB — trivial vs the arc's
  // −290 MB net. Streamed-only (generated keeps boot-bound instances) ⇒ no effect on
  // the generated determinism gate.
  const INST_BLOCK_SIZE = 8192;
  const INST_BLOCKS = 40;
  const INST_BAND_DIST = 300;
  BootTrace.phase(streamed ? 'world source (estonia stream)' : 'world source (heightfield + scatter)');
  const worldSource: WorldSource = streamed
    ? new RemoteWorldSource(params.dataUrl ?? undefined)
    : new GeneratedWorldSource(engine.renderer, seed);
  const worldManifest = await worldSource.open((p, m) => ctx.progress(p * 0.94, m));
  // The live boot heightfield — after S4 it feeds ONLY boot-time consumers
  // (scatter + classification inside source.open, the registry terrain build),
  // the still-live waterY/flow buffers (water material + caustics — S9 ports),
  // wind noise, and the ?profile=1 texture handoff. Every other runtime read
  // lives on the TerrainField planes + the S4 windows below; the boot-only GPU
  // set (incl. biome/fields textures since S4) is released right after the
  // render graph builds (releaseBootGpuSet).
  // The generated source owns a live boot Heightfield; the streamed world has no
  // procedural terrain — a stub carries ONLY the still-live boot handles the
  // subsystems read off `hf` (real procedural noise for wind/froxels/resolve;
  // placeholder dry water/flow for water+caustics until the S9 port). Terrain
  // DATA lives on the TerrainField planes for BOTH.
  const hf: Heightfield = streamed
    ? await Heightfield.forStreamedWorld(engine.renderer, seed)
    : (worldSource as GeneratedWorldSource).heightfield;
  (engine as unknown as { heightfield?: Heightfield }).heightfield = hf;

  if (!streamed && hf.cpuHeights) {
    let maxH = -Infinity;
    for (let i = 0; i < hf.cpuHeights.length; i += 7) {
      const v = hf.cpuHeights[i] as number;
      if (v > maxH) maxH = v;
    }
    engine.stats.counters['terrain.maxH'] = Math.round(maxH);
  }

  // N8-D2 Stage 2e (D-N39) — terrain mode resolution, hoisted above the stream
  // brain (its tile config needs gridN/skirt): default = the clip-STREAMED DAG
  // (gridN 128); `?nanitedterrain=0` = the legacy window grid; explicit
  // `?nanitedterrain=<gridN>` = one-shot uniform tiles unless `?nanitedclip=1`.
  const dterrainParam = qNan.get('nanitedterrain');
  const terrainDefault = dterrainParam == null;
  const dagTerrainGridN = terrainDefault ? 128 : Math.max(0, Math.floor(Number(dterrainParam)));
  const dtilesParam = qNan.get('nanitedtiles');
  const dagTerrainTiles = dtilesParam ? Math.max(1, Math.floor(Number(dtilesParam))) : 1;
  const dagTerrainPool = qNan.get('nanitedpool') === '1';
  const dagTerrainClip = terrainDefault || qNan.get('nanitedclip') === '1';
  const dagTerrainSkirt = qNan.get('nanitedskirt') !== '0';

  // TERRAIN FIELD + STREAM BRAIN (S5): the brain worker owns residency —
  // plane-window fills/scrolls, the terrain tile clipmap, fetch/decode/bake
  // orchestration; the scene allocates the TerrainField from the shared plan
  // and drains the brain's packet mailbox under ONE token bucket per frame.
  // The generated world's windows cover its whole span, so the boot fill pins
  // everything resident — steady-state-identical to the pre-S5 static fill.
  BootTrace.phase('terrain field (stream brain plane fills)');
  ctx.progress(0.942, 'terrain field: filling planes');
  const brain = new StreamBrainClient(worldSource, worldManifest, {
    gridN: dagTerrainGridN > 0 ? dagTerrainGridN : 128,
    tilesPerSide: 4,
    skirt: dagTerrainSkirt,
    seed: seed.seed,
  });
  const field = await brain.openField();
  const streamOrigin = new StreamOrigin(worldManifest.grid.chunkMeters);

  // FAR-VIEW EXTENT (S6): the camera far plane is tuned for a 4 km world (30 km
  // hardcoded); Estonia needs 100+ km. Derive it UNIVERSALLY from the height
  // layer's coarsest-lod world span — the generated world's ≤8 km span keeps it
  // below the 30 km floor (so today's value is untouched, bit-identical), while
  // Estonia's country-spanning L4 lifts it. Aerial haze fully obscures terrain
  // past ~50 km, so a 150 km cap covers the whole horizon with room to spare.
  {
    const hmeta = worldManifest.layers.height;
    let extentHalf = 0;
    for (const lod of hmeta?.lods ?? []) {
      const box = chunkBox(worldManifest.chunks('height', lod));
      if (!box) continue;
      const foot = worldManifest.grid.chunkMeters * worldManifest.grid.lodStep ** lod;
      extentHalf = Math.max(extentHalf, ((box.maxX - box.minX + 1) * foot) / 2, ((box.maxZ - box.minZ + 1) * foot) / 2);
    }
    const viewFar = Math.min(150000, extentHalf * 1.15);
    if (viewFar > engine.camera.far) {
      // eslint-disable-next-line no-console
      console.log(`[laas] camera far ${engine.camera.far}→${Math.round(viewFar)} m (world half-extent ${Math.round(extentHalf / 1000)} km)`);
      engine.camera.far = viewFar;
      engine.camera.updateProjectionMatrix();
    }
  }

  // physical sky first: probe gathering needs the atmosphere LUTs.
  // ?shot=N boots straight into a composed bookmark — use ITS time of day
  const bootBm = params.shot !== null ? BOOKMARKS[params.shot - 1] : undefined;
  const bootTod = bootBm?.tod ?? params.timeOfDay;
  BootTrace.phase('sky: atmosphere LUTs');
  ctx.progress(0.93, 'sky: baking atmosphere LUTs');
  const sunSky = new SunSky(engine, bootTod);
  await sunSky.init(engine.renderer);
  (engine as unknown as { sunSky?: SunSky }).sunSky = sunSky;
  // tooling probe handle (tools/probe-state.ts) — light/scene state triage
  (window as unknown as { __laasDbg?: unknown }).__laasDbg = { engine, sunSky };

  // canopy coverage window from the source's TREE RECORDS (S4 — the WorldSource
  // path both sources ride; pinned over the whole generated world) — BEFORE the
  // probe field (probes ray-march the bare heightfield; the canopy window is
  // their only knowledge of the forest) and before tiles
  BootTrace.phase('canopy window');
  ctx.progress(0.945, 'vegetation: canopy window');
  // streamed world (S6): no scatter yet — trees/understory are S7. The near
  // CanopyWindow (record-fed) is therefore empty; canopy=null is the fully
  // supported "source has no trees" state every downstream consumer accepts,
  // and the FOREST reads from the far material's canopy-plane term instead.
  const scatter = streamed ? null : (worldSource as GeneratedWorldSource).scatter;
  const canopy = streamed
    ? null
    : await CanopyWindow.build(
        engine.renderer,
        worldSource,
        worldManifest,
        field,
        engine.camera.position.x,
        engine.camera.position.z,
      );
  const canopyTex = canopy?.tex ?? null;
  engine.stats.counters['veg.trees'] = scatter?.trees.count ?? 0;
  engine.stats.counters['veg.under'] = scatter?.understory.count ?? 0;
  engine.stats.counters['veg.extras'] = scatter?.extras.count ?? 0;
  engine.stats.counters['veg.stones'] = scatter?.stones.count ?? 0;

  // (ablate hoisted to the top of the function — the overlap kick needs it)

  // irradiance probe field (Phase 3 GI; canopy-aware since Phase 5 —
  // ?ablate=canopygi rebuilds the bare-heightfield field for A/B)
  BootTrace.phase('probe GI');
  ctx.progress(0.95, 'gi: gathering irradiance probes');
  const gi = new ProbeGI(
    field,
    sunSky.atmosphere,
    ablate.has('canopygi') ? null : canopy,
  );
  await gi.init(engine.renderer);
  sunSky.dimAmbientForGI();
  engine.onUpdate(() => gi.tick(engine.renderer));

  // Phase 6 caustics: per-frame analytic bake + module context — MUST be
  // set before any material factory runs (terrain tiles, rocks, debris all
  // self-apply at build time). ?ablate=caustics to A/B, ?caustk=N to tune.
  if (!ablate.has('caustics')) {
    if (!hf.flow) throw new Error('caustic context without hydrology');
    const bake = new CausticsBake();
    const ck = Number(new URLSearchParams(window.location.search).get('caustk') ?? NaN);
    if (Number.isFinite(ck)) bake.focusK.value = ck;
    setCausticContext({ field, flow: hf.flow, simRes: hf.simRes, noiseA: hf.noiseA, bake, sunDir: sunU.dir });
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

  if (view === 'scatter' && scatter) addScatterDebug(engine.scene, scatter);

  // Phase 6: stream/lake water clipmap (?ablate=water to A/B) is CONSTRUCTED
  // AFTER the nanite frame below — W2 threads the frame's composed sun-visibility
  // (clipmap PCSS × cloud × far shadow) into the foam/glint lighting.

  // Phase 5: variant pools → nanite registry
  if (vegEnabled && vegLibPromise) {
    // kicked at the top of the function (cold-boot overlap) — by now most/all of
    // it ran interleaved with the GPU phases above; this await is just the tail
    BootTrace.phase('veg library (await tail)');
    ctx.progress(0.963, 'vegetation: variant pools');
    const lib = await vegLibPromise;
    // sun uniforms feed the nanite terrain shading
    updateSunUniforms(sunSky.sun);
    naniteBark = lib.barkArray; // resolve bark/deadwood sampled-array (N4-C3)

    // N1-C4: build the GeometryRegistry from all opaque pools
    // (cluster tables + packed mega-buffers only).
    BootTrace.phase('nanite: world registry');
    ctx.progress(0.985, 'nanite: clusterizing opaque pools');
    const { buildWorldRegistry, PORTED_CLASSES } = await worldRegistryModule;
    // class/DAG/leaf resolution shared with the early prep kick (top of function)
    const setup = resolveNaniteSetup(PORTED_CLASSES as readonly MatCls[]);
    const classes = setup.classes;
    naniteClasses = classes ?? new Set(NAN_ALL);
    const dagClasses = setup.dagClasses;
    // terrain mode consts resolved above the stream brain (scene scope)
    // N9-C0/C2: leaf heads — resolved in resolveNaniteSetup (see above)
    const naniteLeaf = setup.naniteLeaf;
    // S2: placement flows Source→ChunkContent→registry — the ONE placement path
    // (record chunks → per-species streams; the registry never sees scatter buffers)
    BootTrace.phase('nanite: chunk content streams');
    const tStreams0 = performance.now();
    // streamed world (S6): instance streams are EMPTY — Estonia trees/boulders
    // need the SpeciesMap resolver (S7) before they can bind to library pools;
    // S6 is terrain + far forests only. The registry builds its pools from an
    // empty stream set (ChunkContent's empty path).
    const streams: ChunkContentStreams = streamed
      ? { perId: new Map(), total: 0 }
      : await buildChunkContentStreams(worldSource, worldManifest);
    const streamsMs = performance.now() - tStreams0;
    if (qNan.get('s2gate') === '1' && !streamed) {
      // gate probe handle (scratchpad/s2gate.mjs) — own global; __laasDbg gets reassigned
      (window as unknown as { __laasS2Gate?: unknown }).__laasS2Gate = { rawLayers: (worldSource as GeneratedWorldSource).rawLayers, streams };
    }
    const wr = await buildWorldRegistry({
      renderer: engine.renderer,
      hf,
      streams,
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
      ...(naniteLeaf ? { leaf: true } : {}),
      // cold-boot overlap: crowns+DAGs already building since the top of boot
      ...(vegPrepPromise ? { pre: vegPrepPromise } : {}),
      // S5: clip terrain streams through the brain (boot tiles + runtime clipmap)
      brain,
      // S6f: boot the residency cut AT the expected spawn (Estonia's default fly
      // spawn orbits the coverage center; (0,0) IS the generated world's center),
      // so frame 1 is fine-where-you-stand. A ?cam elsewhere is a teleport the
      // tree absorbs at runtime (cancel + instant merges + re-refine).
      ...(streamed ? { bootPose: coverageCenter(worldManifest) } : {}),
      // S7: streamed worlds reserve an instance pool (§5, A4) for the tree/boulder
      // band; the generated world keeps its boot-bound instances (no pool).
      ...(streamed ? { instancePool: { blockSize: INST_BLOCK_SIZE, blocks: INST_BLOCKS } } : {}),
    });
    (engine as unknown as { naniteRegistry?: unknown }).naniteRegistry = wr.registry;
    naniteRegistry = wr.registry;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] nanite registry: total ${wr.totalMs.toFixed(0)} ms (streams ` +
        `${streamsMs.toFixed(0)} + terrain minMax ` +
        `${wr.terrainMs.toFixed(0)} + build ${wr.buildMs.toFixed(0)}` +
        (wr.dagMeshes > 0
          ? ` + DAG ${wr.dagMeshes}m/${wr.dagBuildMs.toFixed(0)}ms/${(wr.dagTris / 1000).toFixed(0)}k tris`
          : '') +
        `); deferred instances ${wr.deferredInstances}\n${wr.report.table}\ndeferred: ${wr.deferred.join('; ')}`,
    );

    // S7: arm the streamed tree/boulder instance band (Estonia only — the generated
    // world's instances are boot-bound). SpeciesMap folds the manifest species dict
    // onto the 6 tree pools; unmapped ids route to the KarstGnarl fallback pool
    // (logged loud) until the dedicated checker material lands. ETAK boulders ride
    // the SAME pool via EtakBoulders (§H), grounded per chunk on the height window.
    let speciesMap: ReturnType<typeof buildSpeciesMap> | null = null;
    if (streamed) {
      speciesMap = buildSpeciesMap(worldManifest.dictionaries, VegClass.KarstGnarl * 8);
      // eslint-disable-next-line no-console
      console.log(speciesMap.summary);
      const band = new InstanceBand({
        source: worldSource,
        manifest: worldManifest,
        idFOf: speciesMap.idFOf,
        boulderRadiusOf: (cls) => lib.clsRadius[cls] ?? 1,
        headsOf: (idF) => {
          const out: number[] = [];
          const bark = wr.heads.get(idF);
          if (bark !== undefined) out.push(bark);
          const leaf = wr.leafHeads.get(idF);
          if (leaf !== undefined) out.push(leaf); // co-located near crown (mesh, ≤ transitionDist)
          // S8: the voxel-crown sibling owns the mid/far crown band (transitionDist..
          // TREE_GEO_FAR). Without it, streamed trees go bare past the 60 m mesh handoff
          // while generated (boot-bound) trees keep a crown — the reported "bare streamed
          // trees" bug. Binding it on the SAME instance restores parity across the band.
          const vox = wr.voxHeads.get(idF);
          if (vox !== undefined) out.push(vox);
          return out;
        },
        reg: wr.registry,
        bandDist: INST_BAND_DIST,
      });
      brain.setInstanceBand(band);
      // eslint-disable-next-line no-console
      console.log(
        `[laas] instance band armed: bandDist ${INST_BAND_DIST} m, pool ` +
          `${wr.registry.instancePoolBlockCount}×${wr.registry.instancePoolBlockSize} = ${wr.registry.instancePoolCapacity} slots`,
      );
    }

    // S8: arm the runtime FAR-TILE band (BOTH sources — the generated world runs the
    // SAME streamed fartile path, law 3; the boot all-resident build is excised). The
    // brain owns per-cell residency + splat/emit; here we only ship the species→pool
    // resolution: Estonia via the SpeciesMap, generated by identity (species IS the
    // VegClass, idF = species·8 + variant).
    if (wr.fartileArm) {
      const arm = wr.fartileArm;
      let speciesToClass: Int32Array;
      if (streamed && speciesMap) {
        const dictSpecies = worldManifest.dictionaries.species;
        let maxId = 0;
        for (const id of dictSpecies.keys()) maxId = Math.max(maxId, id);
        speciesToClass = new Int32Array(maxId + 1).fill(-1);
        for (const id of dictSpecies.keys()) speciesToClass[id] = speciesMap.idFOf(id, 0) >>> 3;
      } else {
        speciesToClass = new Int32Array(256);
        for (let i = 0; i < 256; i++) speciesToClass[i] = i;
      }
      const info = wr.registry.fartilePoolInfo;
      brain.armFartiles({ kind: 'ftArm', ...arm, speciesToClass, slots: info.slots, clusterCap: info.clusterCap, granules: info.granules });
      // eslint-disable-next-line no-console
      console.log(`[laas] fartile band armed: ${arm.pools.length} species crown pools → ${info.slots} slots / ${info.granules} granules`);
    }
  }

  // S5: drive the STREAM BRAIN from the live camera — pose feed at ~10 Hz
  // (ring/window diffs, bakes and priorities run in the worker; teleports are
  // detected from the pose discontinuity) + the ONE token-bucket mailbox drain
  // (≤2 MB or ≤1.5 ms/frame across plane fills + tile attaches, strict FIFO =
  // the demote→scroll→promote transaction) + the StreamOrigin rebase check
  // (generated world: origin stays (0,0) forever — under the 8 km threshold).
  engine.onUpdate(() => {
    const c = engine.camera.position;
    brain.update(c.x, c.z);
    brain.drain(engine.renderer);
    // S6d KEYSTONE: the camera-relative raster hook has landed (NaniteCam builds
    // vp/invVp/camPos RELATIVE to StreamOrigin, the resolve reconstructs anchor-
    // relative, the shadow-clip levelVP fits in the same frame), so rebasing is
    // now LIVE on the streamed world too: the first frame snaps the origin near
    // Estonia's ~311 km spawn, dropping the whole project chain to sub-metre f32.
    // rebaseTilePoolOrigins shifts terrain-tile origin words; rebaseInstanceOrigins
    // shifts the fartile identity A-words; both keep the pooled frame coherent with
    // the new anchor. Generated: origin stays (0,0) forever (< 8 km) ⇒ a no-op.
    streamOrigin.maybeRebase(c.x, c.z, naniteRegistry);
    // S6c PRECISION: on the streamed world (Estonia, absolute coords ~311 km) pin
    // every GPU field sampler's coordinate frame to a near-camera snapped anchor so
    // gridCoords stays sub-metre (the terrain height/normal sampling terraced on
    // slopes otherwise). Snapped to 512 m to keep the uniform stable across frames;
    // the sampled VALUE is anchor-invariant, so the snap step is cosmetic. The
    // generated world leaves the anchor at (0,0) ⇒ every sampler is IEEE-identical.
    if (streamed) {
      field.setRenderAnchor(Math.round(c.x / 512) * 512, Math.round(c.z / 512) * 512);
    }
    Object.assign(engine.stats.counters, brain.counters(), streamOrigin.counters());
  });

  // volumetric clouds (noise bake + sun-shadow map)
  BootTrace.phase('clouds + shadows + post');
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

  // Sun shadows are the nanite screen-density shadow clipmap (built inside the
  // nanite frame below); the cloud sun-transmittance gate is applied directly by
  // the resolve via world.cloudShadow. There is no CSM rig in the world path.
  // P4 (shadow arc): baked heightfield sun-visibility — mountains shade valleys at
  // ANY distance (the clipmap reaches 384 m; this is the far-field term). Re-baked
  // on ToD edits below. ?ablate=farshadow drops it.
  const farSh =
    !ablate.has('shadows') && !ablate.has('farshadow')
      ? new (await import('../gpu/passes/FarShadow')).FarShadow(field, sunSky.atmosphere)
      : null;
  if (farSh) await farSh.init(engine.renderer);
  (window as unknown as { __laasDbg?: Record<string, unknown> }).__laasDbg = {
    engine,
    sunSky,
    naniteRegistry,
    streamOrigin,
  };

  // S4 camera-follow: the subsystem windows track the camera every frame (on
  // the generated world their coverage clamp pins them — these are no-ops; S5's
  // roaming camera drives real scrolls through the same three calls).
  engine.onUpdate(() => {
    const c = engine.camera.position;
    canopy?.update(engine.renderer, c.x, c.z);
    gi.followCamera(c.x, c.z);
    farSh?.follow(engine.renderer, c.x, c.z);
  });

  // GPU particles: snow/pollen/leaves riding the wind (?ablate=particles)
  if (!ablate.has('particles')) {
    const parts = new Particles(field, canopy, ablate.has('gi') ? null : gi);
    engine.scene.add(parts.mesh);
    engine.onUpdate((dt) => parts.update(engine.renderer, engine.camera, dt));
    engine.stats.counters['particles'] = PARTICLE_COUNT;
  }

  // froxel volumetrics: canopy shafts + valley fog (?ablate=froxels, ?fog=N)
  let froxels: Froxels | null = null;
  if (!ablate.has('froxels')) {
    froxels = new Froxels(field, { noiseA: hf.noiseA }, sunSky.atmosphere, canopy, clouds);
    const fq = Number(new URLSearchParams(window.location.search).get('fog') ?? NaN);
    if (Number.isFinite(fq)) froxels.fogK.value = fq;
    const fx = froxels;
    engine.onUpdate(() => fx.update(engine.renderer, engine.camera));
  }

  // The render graph below (PostStack + nanite frame + water) captures the renderer
  // at BUILD time — PostStack's post-processing pipeline and buildNaniteFrame's
  // `renderer` local both bind whatever engine.renderer is now. ?profile=1 defers
  // this whole block so core/ProfileBoot can build it ONCE, natively, on the
  // swapped-in 'laas-render' device (no rebuild). Normal mode runs it inline below.
  const buildRenderGraph = async (): Promise<void> => {
    // HDR post stack: aerial perspective, clouds, GTAO, TRAA, bloom, exposure, grade
    ctx.progress(0.98, 'post: building pipeline');
    const post = new PostStack(engine, sunSky.atmosphere, bootTod, clouds, froxels);
    engine.post = post;

    // ?nanitedbg=flat|cluster (needs ?nanite=1) — N2 debug view: cull → raster
    // → flat resolve replaces the frame render via the post slot; the old
    // pipeline keeps booting/updating untouched. `cluster` = the deferred N1
    // checkpoint (meshlet colors on the real world).
    const nanitedbg = new URLSearchParams(window.location.search).get('nanitedbg');
    let naniteSunVis: import('../nanite/frame/NaniteFrame').NaniteFrameHandles['sunVis'];
    if (
      nanitedbg === 'flat' ||
      nanitedbg === 'cluster' ||
      nanitedbg === 'lod' ||
      nanitedbg === 'hzb'
    ) {
      if (naniteRegistry) {
        const { buildNaniteView } = await import('../nanite/frame/NaniteView');
        engine.post = buildNaniteView(engine, naniteRegistry, field, nanitedbg);
        // eslint-disable-next-line no-console
        console.log(`[laas] nanitedbg=${nanitedbg}: N2 debug view replacing the frame render`);
      } else {
        // eslint-disable-next-line no-console
        console.warn('[laas] ?nanitedbg needs ?nanite=1 with vegetation enabled — ignored');
      }
    } else if (naniteRegistry && naniteClasses && naniteFrameMode) {
      // N4 full-frame mode (D-N18/D-N19): nanite compute + in-scene resolve own
      // the migrated classes.
      BootTrace.phase('nanite: frame build (raster/resolve/grass)');
      const { buildNaniteFrame } = await import('../nanite/frame/NaniteFrame');
      const nanFrame = buildNaniteFrame(engine, naniteRegistry, hf, field, post, {
        gi: ablate.has('gi') ? null : gi,
        canopyTex,
        // sunShadows carries the "scene has sun shadows" signal; the cloud gate is
        // applied directly by the resolve.
        sunShadows: !ablate.has('shadows'),
        cloudShadow:
          !ablate.has('cloudshadow')
            ? (wxz: import('../gpu/TSLTypes').NV2) => clouds.shadowAt(wxz)
            : null,
        farShadow: farSh ? (wxz: import('../gpu/TSLTypes').NV2) => farSh.visAt(wxz) : null,
        barkTexA: naniteBark?.texA ?? null,
        barkTexB: naniteBark?.texB ?? null,
        // S6d KEYSTONE: on the streamed world feed the camera/reconstruct/shadow
        // chain the live StreamOrigin as its render anchor (rebase-rare, 8 km-
        // snapped) so the whole project chain is small-coordinate. Generated ⇒
        // omitted ⇒ A=(0,0) ⇒ byte-identical absolute build.
        streamAnchor: streamed ? () => ({ x: streamOrigin.x, z: streamOrigin.z }) : undefined,
      });
      engine.post = nanFrame;
      naniteSunVis = nanFrame.sunVis;
      // eslint-disable-next-line no-console
      console.log(
        `[laas] nanite full-frame: classes [${[...naniteClasses].join(',')}]`,
      );
    }

    // Phase 6 water (moved after the nanite frame — W2 needs its sunVis): the
    // clipmap draws in the scene pass after the resolve meshes (transparent,
    // depthWrite) — the SLW-over-resolve seam. The foam/glint lighting is manual,
    // gated by nanite sun visibility.
    if (!ablate.has('water')) {
      const water = new WaterSurface(
        hf,
        field,
        sunSky.atmosphere,
        canopyTex,
        ablate.has('gi') ? null : gi,
        { sunVis: naniteSunVis },
      );
      engine.scene.add(water.group);
      engine.onUpdate(() => water.update(engine.camera));
      // runtime visible-toggle for within-session perf A/B
      (window as unknown as { __laasDbg: Record<string, unknown> }).__laasDbg.water = water;
    }

    ctx.hooks.setTimeOfDay = (t: number) => {
      void (async () => {
        await sunSky.setTimeOfDay(t);
        await clouds.refreshShadow(engine.renderer);
        farSh?.bake(engine.renderer); // P4: sun moved — re-march the far-shadow map
        gi.invalidate();
        post.setTimeOfDay(t);
      })();
    };
    window.addEventListener('keydown', (e) => {
      if (e.code === 'BracketLeft' || e.code === 'BracketRight') {
        void clouds.refreshShadow(engine.renderer);
        farSh?.bake(engine.renderer);
        post.setTimeOfDay(sunSky.timeOfDay);
      }
    });
  }; // end buildRenderGraph

  // ?profile=1: defer the render graph to after the device swap (ProfileBoot builds
  // it on 'laas-render'); otherwise build it now, preserving the normal path exactly.
  const profileMode = new URLSearchParams(window.location.search).get('profile') === '1';
  if (!profileMode) {
    await buildRenderGraph();
  } else {
    const bark = naniteBark;
    (
      window as unknown as { __laasProfile?: import('../core/ProfileBoot').ProfileHandoff }
    ).__laasProfile = {
      // GPU-only StorageTextures the game samples but that cannot auto-remigrate to
      // a fresh device (no CPU image data) — ProfileBoot reads these back before the
      // swap and writes them onto the render device afterward.
      textures: (
        [
          // heightTex is EXCISED and normalTex/height/biomeTex/fieldsTex are
          // RELEASED post-boot (S3b/S4 finales) — the render device never
          // references them, so they are deliberately NOT transferred across
          // the ?profile device swap.
          hf.noiseA && { tex: hf.noiseA },
          hf.noiseB && { tex: hf.noiseB },
          bark && { tex: bark.texA, mips: true },
          bark && { tex: bark.texB, mips: true },
          canopyTex && { tex: canopyTex },
        ] as ({ tex: import('three').Texture; mips?: boolean } | null | false | undefined)[]
      ).filter(Boolean) as { tex: import('three').Texture; mips?: boolean }[],
      // subsystems that own build-time GPU state or a captured renderer and expose a
      // clean re-init on a new renderer (atmosphere LUTs + IBL; GI field).
      reheal: async (r2) => {
        await sunSky.init(r2); // atmosphere LUTs + IBL cube
        await gi.init(r2); // GI probe field
        await clouds.init(r2); // bake-once 3D cloud noise (empty on the fresh device)
      },
      buildRenderGraph,
      timeOfDay: bootTod,
    };
    // eslint-disable-next-line no-console
    console.log('[profile] render graph deferred — ProfileBoot builds it on laas-render');
  }

  // S3b/S4 finale: every boot bake that read the hf GPU field set has run
  // (scatter/classification inside source.open; GI/far-shadow/canopy/froxels/
  // particles all live on the TerrainField planes + S4 windows) — free the
  // boot-only GPU set (height + hardness + erosion scratch buffers, normalTex,
  // and since S4 biomeTex + fieldsTex). waterY/flow stay (water material +
  // caustics read them live until their S9 ports). Under ?profile these are
  // loading-device resources outside the swap handoff, so releasing early is
  // safe (see Heightfield.releaseBootGpuSet).
  {
    const freedMb = hf.releaseBootGpuSet(engine.renderer);
    // eslint-disable-next-line no-console
    console.log(
      `[laas] heightfield boot GPU set released: ${freedMb.toFixed(1)} MB ` +
        `(height/hardness/erosion-scratch buffers + normalTex + biomeTex + fieldsTex; ` +
        `heightTex is excised — never allocated)`,
    );
  }

  // terrain/water probe for the camera rig: walk-mode ground physics + the
  // fly-mode soft collision / underwater guard both live in FlyCamera now
  ctx.hooks.groundProbe = (x, z) => ({
    ground: field.heightAt(x, z),
    water: field.waterAt(x, z),
  });

  // camera spawn: ground-clamped (?alt/x/z → fly) or the DEFAULT WALK SPAWN
  // at the map center — first dry, reasonably flat spot on a spiral out
  // from (0,0), eye at head height, facing the NE massif
  const q = new URLSearchParams(window.location.search);
  const alt = Number(q.get('alt') ?? NaN);
  if (params.cam === null) {
    if (streamed && !Number.isFinite(alt)) {
      // Estonia default (S6): an elevated scenic fly over the pilot valley so the
      // first frame shows the whole-country horizon + forested far shell. Walk
      // mode (ground probe on streamed heights) is reachable via M / ?alt.
      const c = coverageCenter(worldManifest);
      const spawn = findWalkSpawn(field, c.cx, c.cz);
      const y = field.heightAt(spawn.x, spawn.z) + 140;
      ctx.hooks.initialPose = { p: [spawn.x, y, spawn.z], yaw: 2.4, pitch: -0.1 };
      ctx.hooks.initialPoseMode = 'fly';
      engine.camera.position.set(spawn.x, y, spawn.z);
    } else if (Number.isFinite(alt)) {
      const c = streamed ? coverageCenter(worldManifest) : { cx: 600, cz: 900 };
      const x = Number(q.get('x') ?? c.cx);
      const z = Number(q.get('z') ?? c.cz);
      const yaw = Number(q.get('yaw') ?? 2.4); // rad; 0 = looking −z (north)
      const pitch = Number(q.get('pitch') ?? -0.04); // rad; negative = down
      const y = field.heightAt(x, z) + alt;
      // the fly camera doesn't exist yet — main applies this after rigging
      ctx.hooks.initialPose = { p: [x, y, z], yaw, pitch };
      ctx.hooks.initialPoseMode = 'fly';
      engine.camera.position.set(x, y, z);
    } else {
      const spawn = findWalkSpawn(field);
      ctx.hooks.initialPose = {
        p: [spawn.x, field.heightAt(spawn.x, spawn.z) + 1.7, spawn.z],
        yaw: -0.78, // face NE — the serrated massif anchors the first frame
        pitch: -0.02,
      };
      ctx.hooks.initialPoseMode = 'walk';
      engine.camera.position.set(spawn.x, ctx.hooks.initialPose.p[1], spawn.z);
    }
  }

  // composed bookmarks (keys 1-9, ?shot=N) + 92 s flythrough (?fly=1 / F)
  installBookmarks(engine, field, ctx.hooks, params);

  BootTrace.phase('first frames (compile + settle)');
  ctx.progress(1, 'terrain ready');
}

/**
 * Default walk spawn: first dry, reasonably flat spot on a coarse spiral
 * out from the map center (dry = waterY sits below the bed there; flat =
 * central-difference slope under ~19°).
 */
function findWalkSpawn(field: TerrainField, cx = 0, cz = 0): { x: number; z: number } {
  for (let r = 0; r <= 240; r += 12) {
    const steps = Math.max(1, Math.round((2 * Math.PI * r) / 18));
    for (let k = 0; k < steps; k++) {
      const a = (k / steps) * Math.PI * 2;
      const x = cx + Math.cos(a) * r;
      const z = cz + Math.sin(a) * r;
      const h = field.heightAt(x, z);
      if (field.waterAt(x, z) > h - 0.05) continue; // wet or waterline
      const sx = field.heightAt(x + 6, z) - field.heightAt(x - 6, z);
      const sz = field.heightAt(x, z + 6) - field.heightAt(x, z - 6);
      if (Math.hypot(sx, sz) / 12 > 0.35) continue; // too steep
      return { x, z };
    }
  }
  return { x: 0, z: 0 };
}

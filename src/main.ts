/** LAAS entry point — boot sequence with fail-loud diagnostics. */

import { BootUI } from './core/BootUI';
import { browserGate } from './core/BrowserGate';
import {
  describeDiagnostics,
  failLoud,
  installGlobalErrorHooks,
  probeWebGPU,
} from './core/Diagnostics';
import { Engine } from './core/Engine';
import { FlyCamera } from './core/FlyCamera';
import { initHooks } from './core/Hooks';
import { parseCamString, parseParams } from './core/Params';
import { WorldSeed } from './core/Seed';
import { Hud } from './debug/HUD';
import { buildGalleryScene } from './debug/GalleryScene';
import { buildRasterSpikeScene } from './debug/RasterSpikeScene';
import { buildSanityScene } from './debug/SanityScene';
import { buildShadowTestScene } from './debug/ShadowTestScene';
import { buildTerrainScene } from './debug/TerrainScene';
import { buildScene, registerScene, type WorldContext } from './debug/Scenes';

/**
 * ?pure=1 — MASTER ablation (PERF directive: isolate the PURE nanite renderer
 * from every beauty/effect so its true GPU floor is measurable, not guessed).
 * Composes the already-wired flags so each existing read-site just works with
 * ZERO per-site plumbing — rewrite the URL ONCE, before any subsystem reads it:
 *   nanite=1   → the pure nanite path is what we're isolating
 *   postmin=1  → strip the ENTIRE post chain (half.mrt / aerial+atmosphere /
 *                AO / bounce / clouds / froxels / contact / TRAA / bloom) and
 *                output the raw scene pass — the ~60 ms of post that buried the
 *                nanite cost (and the anonymous r.rt#16) and overheated the GPU
 *   nanshadow=0→ no shadow-clipmap raster, no shadow sample in the resolve
 *   nandbg=flat→ flat-albedo resolve (no lighting / IBL / GI) — r.scene becomes
 *                the pure materialize cost (unpack → fetch 3 verts → barycentric)
 * SURVIVORS = exactly the pure geometry pipeline: instance/cluster cull → SW
 * depth+payload raster → HW big-tri pass → flat resolve. Geometry (terrain /
 * rock / bark / deadwood) is KEPT — `veg` is deliberately NOT ablated because it
 * gates the whole nanite registration. Each implied flag is only set when ABSENT,
 * so explicit overrides survive (e.g. ?pure=1&nandbg=albedo to time a textured-
 * but-unlit resolve, or ?pure=1&nanshadow=1 to add just shadows back).
 */
function expandPureAblation(): void {
  const q = new URLSearchParams(window.location.search);
  if (q.get('pure') !== '1') return;
  const implied: Record<string, string> = {
    nanite: '1',
    postmin: '1',
    nanshadow: '0',
    nandbg: 'flat',
  };
  let changed = false;
  for (const [k, v] of Object.entries(implied)) {
    if (!q.has(k)) {
      q.set(k, v);
      changed = true;
    }
  }
  if (changed) {
    history.replaceState(null, '', `${window.location.pathname}?${q.toString()}`);
  }
}

async function boot(): Promise<void> {
  expandPureAblation();
  const hooks = initHooks();
  installGlobalErrorHooks();
  // environment gate BEFORE any loading: mobile / non-Chromium / missing
  // WebGPU each get a clear notice instead of a broken boot (?nogate=1 skips)
  if (!browserGate()) return;
  const params = parseParams();
  const bootUI = new BootUI(hooks);

  bootUI.set(0.02, 'probing WebGPU');
  const diag = await probeWebGPU();
  hooks.diag = diag;
  if (!diag.ok) {
    failLoud('WebGPU unavailable — LAAS has no fallback by design', [
      diag.reason ?? 'unknown reason',
      '',
      'Chrome exposes WebGPU here, but no usable GPU adapter came up. Check:',
      '  • chrome://gpu — WebGPU should read “Hardware accelerated”',
      '  • Settings → System → hardware acceleration ON, then relaunch',
      '  • update Chrome and the GPU driver',
    ]);
    return;
  }
  // eslint-disable-next-line no-console
  console.log('[laas] webgpu ok\n' + describeDiagnostics(diag).join('\n'));

  bootUI.set(0.08, 'creating renderer');
  const engine = await Engine.create(params, hooks);

  // FlyCamera's update MUST register before any scene system: updateFns run
  // in registration order, and subsystems copy camera state in their own
  // updates — the mover has to run first or every copy is one frame stale
  // during interactive motion (clouds/aerial visibly lagged the camera).
  const fly = new FlyCamera(engine.camera, engine.renderer.domElement);
  engine.onUpdate((dt) => fly.update(dt));
  // probe surface: scripted camera control (tools/probe-pan.ts drives yaw)
  (window as unknown as { __laasFly?: FlyCamera }).__laasFly = fly;

  const seed = new WorldSeed(params.seed);
  registerScene('sanity', buildSanityScene);
  registerScene('terrain', buildTerrainScene);
  registerScene('gallery', buildGalleryScene);
  registerScene('shadowtest', buildShadowTestScene);
  registerScene('rasterspike', buildRasterSpikeScene);
  // 'world' becomes the streamed open world once terrain tiles land.
  registerScene('world', buildTerrainScene);

  const ctx: WorldContext = {
    engine,
    params,
    seed,
    hooks,
    progress: (p, msg) => bootUI.set(0.1 + p * 0.85, msg),
  };
  await buildScene(params.scene, ctx);

  // terrain probe first — walk mode + fly soft-collision depend on it
  if (hooks.groundProbe) fly.groundProbe = hooks.groundProbe;
  if (params.cam !== null) {
    const pose = parseCamString(params.cam);
    if (pose) fly.setPose(pose); // explicit pose ⇒ fly semantics
  } else if (hooks.initialPose) {
    fly.setPose(hooks.initialPose);
    // grounded RPG exploration is the interactive default (V toggles fly);
    // ?walk=0 keeps tooling/legacy behavior
    const q = new URLSearchParams(window.location.search);
    if (hooks.initialPoseMode === 'walk' && q.get('walk') !== '0') {
      fly.setMode('walk');
    }
  }

  new Hud(engine, params);

  hooks.setPose = (p) => fly.setPose(p);
  hooks.getPose = () => fly.getPose();
  hooks.settle = (frames?: number) => engine.settle(frames ?? 8);
  hooks.flyCamEnabled = (on) => {
    fly.enabled = on;
  };

  engine.start();
  await engine.settle(6);
  bootUI.hide();
  hooks.ready = true;
  // eslint-disable-next-line no-console
  console.log('[laas] ready');
}

boot().catch((e: unknown) => {
  const msg = e instanceof Error ? `${e.message}\n\n${e.stack ?? ''}` : String(e);
  failLoud('Boot failed', [msg]);
});

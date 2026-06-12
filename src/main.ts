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

async function boot(): Promise<void> {
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

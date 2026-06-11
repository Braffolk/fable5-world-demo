/** LAAS entry point — boot sequence with fail-loud diagnostics. */

import { BootUI } from './core/BootUI';
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
import { buildSanityScene } from './debug/SanityScene';
import { buildShadowTestScene } from './debug/ShadowTestScene';
import { buildTerrainScene } from './debug/TerrainScene';
import { buildScene, registerScene, type WorldContext } from './debug/Scenes';

async function boot(): Promise<void> {
  const hooks = initHooks();
  installGlobalErrorHooks();
  const params = parseParams();
  const bootUI = new BootUI(hooks);

  bootUI.set(0.02, 'probing WebGPU');
  const diag = await probeWebGPU();
  hooks.diag = diag;
  if (!diag.ok) {
    failLoud('WebGPU unavailable — LAAS has no fallback by design', [
      diag.reason ?? 'unknown reason',
      '',
      'Requirements: Chrome/Edge 113+, Safari 26+, or equivalent with WebGPU enabled.',
    ]);
    return;
  }
  // eslint-disable-next-line no-console
  console.log('[laas] webgpu ok\n' + describeDiagnostics(diag).join('\n'));

  bootUI.set(0.08, 'creating renderer');
  const engine = await Engine.create(params, hooks);

  const seed = new WorldSeed(params.seed);
  registerScene('sanity', buildSanityScene);
  registerScene('terrain', buildTerrainScene);
  registerScene('gallery', buildGalleryScene);
  registerScene('shadowtest', buildShadowTestScene);
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

  const fly = new FlyCamera(engine.camera, engine.renderer.domElement);
  engine.onUpdate((dt) => fly.update(dt));
  if (params.cam !== null) {
    const pose = parseCamString(params.cam);
    if (pose) fly.setPose(pose);
  } else if (hooks.initialPose) {
    fly.setPose(hooks.initialPose);
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

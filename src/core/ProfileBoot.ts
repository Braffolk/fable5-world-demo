/**
 * ProfileBoot — the ?profile=1 two-device GPU-trace split.
 *
 * WHY: `DAWN_TRACE_DEVICE_FILTER=<label>` records a WebGPU device from creation
 * to destruction — there is no start/stop hook, so a single-device app always
 * captures the ~11 GB of boot GPU compute (heightfield/erosion/flow/bark bakes)
 * along with the game. The fix is structural: run all loading on a throwaway
 * device labelled `laas-loading`, then — after buildScene — swap to a FRESH
 * device labelled `laas-render` and run the game loop there. A capture filtered
 * to `laas-render` then contains only game frames, because that device only ever
 * existed during the game.
 *
 * HOW the swap survives:
 *  - CPU-backed resources (the whole scene graph + the nanite mega-buffers, which
 *    are StorageBufferAttributes over retained CPU arrays) re-upload lazily on the
 *    new renderer's first render — three keys GPU resources per-backend, so a fresh
 *    backend re-creates them from the retained arrays. `expandProfileMode()` forces
 *    `?noreleasemirrors=1` so those arrays are still alive at swap time.
 *  - GPU-only StorageTextures with no CPU copy (the ~500 MiB of terrain + bark +
 *    canopy bakes) CANNOT auto-remigrate — they are read back here before the swap
 *    and written onto the render device afterward (transfer, below).
 *  - The render graph (PostStack + nanite frame + water) captures the renderer at
 *    BUILD time, so TerrainScene DEFERS its construction under ?profile and we build
 *    it here, once, natively on the render device (handoff.buildRenderGraph).
 *  - Subsystems that own build-time GPU state / a captured renderer are re-inited on
 *    the new renderer (handoff.reheal → sky/atmosphere/IBL + GI; setTimeOfDay →
 *    cloud/far-shadow/grade re-bake).
 *
 * Capture (M1 Max):
 *   DAWN_TRACE_FILE_BASE=/tmp/laas_trace DAWN_TRACE_DEVICE_FILTER=laas-render \
 *   MTL_CAPTURE_ENABLED=1 <chrome …> 'http://localhost:5173/?scene=world&nanite=1&profile=1'
 *
 * This module + the guarded deferral in TerrainScene are the ONLY places the
 * two-device dance lives; the normal single-device path (no ?profile) is untouched.
 */

import type { Texture } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { Engine } from './Engine';

/**
 * Handed from TerrainScene (under ?profile=1) to runProfileSwap: the GPU-only
 * textures to transfer, the subsystem re-init, the current time-of-day, and the
 * deferred render-graph builder.
 */
export interface ProfileHandoff {
  /** GPU-only StorageTextures the game samples but that cannot auto-remigrate
   *  (no CPU image data). `mips` = regenerate mipmaps after writeback (bark). */
  textures: { tex: Texture; mips?: boolean }[];
  /** re-init subsystems bound to the old renderer (sky/atmosphere/IBL + GI). */
  reheal: (r2: WebGPURenderer) => Promise<void>;
  /** build the deferred render graph (PostStack + nanite frame + water) on r2. */
  buildRenderGraph: () => Promise<void>;
  /** boot time-of-day, for the post-swap shadow/grade re-bake. */
  timeOfDay: number;
}

/** GPU bytes-per-texel for the storage-texture formats the world bakes use. */
const BYTES_PER_TEXEL: Record<string, number> = {
  r8unorm: 1,
  r16float: 2,
  rg8unorm: 2,
  r32float: 4,
  rg16float: 4,
  rgba8unorm: 4,
  rgba8uint: 4,
  bgra8unorm: 4,
  rg32float: 8,
  rgba16float: 8,
  rgba32float: 16,
};

interface TexTransfer {
  tex: Texture;
  w: number;
  h: number;
  layers: number;
  bytesPerRow: number;
  mips: boolean;
  /** one padded typed array per array layer (256-aligned rows, as three copies). */
  layerData: ArrayBufferView<ArrayBuffer>[];
}

/** three's WebGPUBackend surface we reach for raw texture copy (no public API). */
interface RawBackend {
  device: GPUDevice;
  get(tex: Texture): { texture: GPUTexture; textureDescriptorGPU: { size: { width: number; height: number; depthOrArrayLayers?: number }; format: string } };
  copyTextureToBuffer(tex: Texture, x: number, y: number, w: number, h: number, faceIndex: number): Promise<ArrayBufferView<ArrayBuffer>>;
  createTexture(tex: Texture): void;
  generateMipmaps?(tex: Texture): void;
}

const backendOf = (r: WebGPURenderer): RawBackend => r.backend as unknown as RawBackend;

/** Read every array layer of a StorageTexture back to CPU (still on the old device). */
async function readbackTexture(r: WebGPURenderer, tex: Texture, mips: boolean): Promise<TexTransfer> {
  const b = backendOf(r);
  const d = b.get(tex).textureDescriptorGPU;
  const w = d.size.width;
  const h = d.size.height;
  const layers = d.size.depthOrArrayLayers ?? 1;
  const bpt = BYTES_PER_TEXEL[d.format];
  if (!bpt) throw new Error(`[profile] no bytes-per-texel for format '${d.format}' (${(tex.name || 'unnamed')})`);
  const bytesPerRow = Math.ceil((w * bpt) / 256) * 256;
  const layerData: ArrayBufferView<ArrayBuffer>[] = [];
  for (let l = 0; l < layers; l++) {
    layerData.push(await b.copyTextureToBuffer(tex, 0, 0, w, h, l));
  }
  return { tex, w, h, layers, bytesPerRow, mips, layerData };
}

/** Write the read-back layers into the (freshly re-created) texture on the render device. */
function writebackTexture(r: WebGPURenderer, device: GPUDevice, t: TexTransfer): void {
  const b = backendOf(r);
  // three re-creates the texture on the new backend during the prime render; if it
  // hasn't yet, force it so get().texture exists.
  let entry = b.get(t.tex);
  if (!entry.texture) {
    b.createTexture(t.tex);
    entry = b.get(t.tex);
  }
  const gpuTex = entry.texture;
  for (let l = 0; l < t.layers; l++) {
    device.queue.writeTexture(
      { texture: gpuTex, origin: { x: 0, y: 0, z: l } },
      t.layerData[l],
      { offset: 0, bytesPerRow: t.bytesPerRow, rowsPerImage: t.h },
      { width: t.w, height: t.h, depthOrArrayLayers: 1 },
    );
  }
  if (t.mips && b.generateMipmaps) b.generateMipmaps(t.tex);
}

/**
 * Run the loading→render device swap. Called by main.ts after buildScene and
 * before engine.start(), only when ?profile=1.
 */
export async function runProfileSwap(engine: Engine): Promise<void> {
  const t0 = performance.now();
  const handoff = (window as unknown as { __laasProfile?: ProfileHandoff }).__laasProfile;
  if (!handoff) {
    // eslint-disable-next-line no-console
    console.warn('[profile] no __laasProfile handoff — ?profile needs the world/terrain scene; running single-device.');
    return;
  }

  // 1. Flush the loading device, then read the GPU-only textures back to RAM
  //    while that device is still alive.
  const oldRenderer = engine.renderer;
  const loadingDevice = engine.device;
  if (loadingDevice) await loadingDevice.queue.onSubmittedWorkDone();
  const transfers: TexTransfer[] = [];
  for (const { tex, mips } of handoff.textures) {
    transfers.push(await readbackTexture(oldRenderer, tex, mips ?? false));
  }
  const mib = transfers.reduce((s, t) => s + t.layerData.reduce((a, d) => a + d.byteLength, 0), 0) / 1048576;

  // 2. Create the fresh render device + renderer (label 'laas-render' — what the
  //    DAWN_TRACE filter keys on) and swap it into the Engine. Its backend has an
  //    empty per-object cache → CPU-backed resources re-upload lazily.
  const { renderer: renderRenderer } = await Engine.createDeviceRenderer('laas-render', engine.hooks.diag ?? null);
  engine.swapRenderer(renderRenderer);
  const renderDevice = engine.device;
  if (!renderDevice) throw new Error('[profile] render device missing after swap');

  // 3. Re-init subsystems bound to the old renderer (sky/atmosphere/IBL + GI),
  //    then build the deferred render graph natively on the render device.
  await handoff.reheal(renderRenderer);
  await handoff.buildRenderGraph();

  // 4. Prime one frame so three creates every GPU resource on the render device
  //    (nanite mega-buffers re-upload; the HW-count buffer the meter path reads is
  //    created by the raster pass — priming BEFORE the loop avoids the first
  //    meter()-before-render() readback of a not-yet-created buffer). engine.post
  //    was just set by buildRenderGraph. Terrain/bark/canopy are BLANK here.
  engine.post?.render();
  await renderDevice.queue.onSubmittedWorkDone();

  // 5. Write the read-back textures onto the render device's (now-created) textures.
  for (const t of transfers) writebackTexture(renderRenderer, renderDevice, t);
  await renderDevice.queue.onSubmittedWorkDone();

  // 6. Re-bake sun-dependent state on the render device (cloud + far shadow, grade,
  //    GI wake) at the boot time-of-day. setTimeOfDay is fire-and-forget (void); its
  //    bakes run on the render device over the next frames.
  engine.hooks.setTimeOfDay?.(handoff.timeOfDay);

  // 7. Free the loading device (memory only — the trace is already game-only via the
  //    label). Its textures are in RAM (step 1) and the nanite CPU mirrors are kept,
  //    so nothing the render device needs lives on it.
  loadingDevice?.destroy();

  // eslint-disable-next-line no-console
  console.error(
    `[profile] swapped laas-loading → laas-render in ${(performance.now() - t0).toFixed(0)}ms` +
      ` (transferred ${mib.toFixed(0)} MiB of bake textures) —` +
      ` a DAWN_TRACE_DEVICE_FILTER=laas-render capture is now game-only`,
  );
}

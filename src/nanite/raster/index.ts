/**
 * raster/index.ts — the src/nanite/raster package barrel (task #76 vis-buffer rewrite,
 * Step 1).
 *
 * The public entry `buildNaniteRaster` stays in ../NaniteRaster — it still hosts the world1
 * project+classify+route megakernel (to become Classify.ts + Project.ts in a later step) +
 * the depth/combined variants + the deferred resolve, and it WIRES the builders exported
 * here (queues, vis buffers, cluster-ctx pre-pass, the shared scanline, and the splat / mid /
 * hw consumers). Existing callers keep importing buildNaniteRaster / makeVisBuffers from
 * '../NaniteRaster'; this barrel is the internal package surface for the raster/ modules.
 */

export { buildNaniteRaster } from '../NaniteRaster';
export type { NaniteRasterHandles } from '../NaniteRaster';

export {
  buildVisClear,
  depthKey16,
  depthKey24,
  makeElect,
  makeVisBuffers,
  type NaniteVisBuffers,
  type VisClearParams,
} from './VisBuffer';
export {
  HW_CAP,
  MID_CAP,
  MID_STRIDE,
  SPLAT_CAP,
  buildQueues,
  type Queues,
} from './Queues';
export {
  CTX_F,
  CTX_STRIDE,
  CTX_U,
  buildClusterCtx,
  type ClusterCtxBundle,
} from './ClusterCtx';
export { makeScanline, type SwScanline } from './Scanline';
export { buildSplat } from './Splat';
export { buildMid } from './Mid';
export { buildHw, type HwPath } from './Hw';

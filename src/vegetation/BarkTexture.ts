/**
 * Bark texture upload — turns the CPU-baked BarkField bytes into a single
 * 6-layer DataArrayTexture (slice == barkLayer). rgba8: RG = micro-grain
 * normal.xy, B = albedo tone, A = cavity AO. Replaces the ~535 MB of GPU-baked
 * 2048² StorageTextures (per-layer 2D pair + array pair) with ONE ≈ 8.4 MB
 * sampled array. Chunked + yielded so the cold bake never freezes the tab.
 */
import {
  DataArrayTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
} from 'three';
import { yieldIfDue } from '../debug/BootTrace';
import { BARK_FIELDS, BARK_TEX_RES, barkTexByteSize, fillBarkRows } from './BarkField';

export async function bakeBarkArray(res = BARK_TEX_RES): Promise<DataArrayTexture> {
  const t0 = performance.now();
  const depth = BARK_FIELDS.length;
  const data = new Uint8Array(barkTexByteSize(res));
  const CHUNK = 32; // rows per yield
  for (let layer = 0; layer < depth; layer++) {
    for (let y = 0; y < res; y += CHUNK) {
      fillBarkRows(data, layer, y, Math.min(res, y + CHUNK), res);
      await yieldIfDue();
    }
  }
  const tex = new DataArrayTexture(data, res, res, depth);
  tex.name = 'barkFieldArray';
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.generateMipmaps = true;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  // eslint-disable-next-line no-console
  console.log(
    `[bark] baked ${depth}×${res}² field array (${(barkTexByteSize(res) / 1e6).toFixed(1)} MB, ` +
      `${(performance.now() - t0).toFixed(0)} ms)`,
  );
  return tex;
}

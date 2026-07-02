/**
 * FarShadow — P4 of the shadow arc (2026-07-03): baked heightfield sun-visibility,
 * the "should a massive hill far away cast a shadow?" answer (YES).
 *
 * The camera-centred shadow clipmap reaches 384 m half-extent — a mountain beyond
 * the ring can never write shadow depth, and a hill 500 m away can never shade a
 * receiver 100 m away (the caster is outside every level's box even though the
 * receiver is inside). Extending the clipmap is the wrong tool (each doubling
 * re-rasters enormous rings for sub-pixel detail); the standard answer is a baked
 * horizon/sun-visibility term over the static heightfield:
 *
 *   one world-space r16f map (RES² over WORLD_SIZE, ~4 m texels), each texel =
 *   sun visibility at ground level, ray-marched toward the sun over the CPU-baked
 *   height texture with growing strides (reach ~1 km — mountain-scale). Re-baked
 *   ONLY on sun change (ToD edit; the march reads the live sun uniform). Sampled
 *   in the nanite resolve as ONE bilinear tap, multiplied into the sun term at
 *   ALL distances — mountains shade valleys at any receiver distance, sunset
 *   throws km-long terrain shadows, and within the clipmap band the overlap only
 *   deepens penumbra (both terms are ~binary there).
 *
 * Approximations (booked): receivers are assumed near ground level (a 30 m crown
 * top inside a soft boundary is slightly over-darkened — the 4 m texels + the
 * 1.5° smoothstep keep the boundary soft); canopy long-shadows (forest silhouette
 * at sunset) are NOT in the map yet — candidate follow-up: add canopy-slab height
 * (cov·CAN_TOP) to the occluder height with a transmittance floor.
 */

import { HalfFloatType, RedFormat } from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import { LinearFilter } from 'three';
import {
  Fn,
  If,
  clamp,
  float,
  instanceIndex,
  smoothstep,
  texture,
  textureStore,
  uint,
  uvec2,
  vec2,
  vec4,
} from 'three/tsl';
import type { NF, NV2 } from '../TSLTypes';
import type { Heightfield } from '../../world/Heightfield';
import type { Atmosphere } from '../../sky/Atmosphere';
import { WORLD_SIZE } from '../../world/WorldConst';
import { dispatch } from '../../nanite/Tsl';

const RES = 1024;
/** march sample distances (m) — dense near for local ridges, sparse far for
 *  mountains; reach ~1.1 km ≈ a 500 m peak at ~25° sun elevation. */
const STEPS = [6, 12, 22, 38, 62, 98, 150, 225, 330, 470, 660, 900, 1120];
/** receiver lift + per-metre slack (m) — absorbs heightfield filtering error so a
 *  texel never self-shadows on its own slope. */
const BASE_LIFT = 1.5;
const SLOPE_SLACK = 0.015;
/** occluder-excess (m) over which visibility fades 1→0 — the soft boundary. */
const SOFT_M = 6;

export class FarShadow {
  readonly tex: StorageTexture;
  private kBake: unknown = null;

  constructor(
    private hf: Heightfield,
    private atmosphere: Atmosphere,
  ) {
    const t = new StorageTexture(RES, RES);
    t.type = HalfFloatType;
    t.format = RedFormat;
    t.magFilter = LinearFilter;
    t.minFilter = LinearFilter;
    t.generateMipmaps = false;
    t.name = 'farShadowVis';
    this.tex = t;
  }

  async init(renderer: Renderer): Promise<void> {
    const hf = this.hf;
    const sunDir = this.atmosphere.sunDir;
    const tex = this.tex;
    const kBake = Fn(() => {
      const idx = instanceIndex;
      If(idx.lessThan(uint(RES * RES)), () => {
        const tx = idx.mod(uint(RES));
        const ty = idx.div(uint(RES));
        const wx = float(tx).add(0.5).div(RES).sub(0.5).mul(WORLD_SIZE);
        const wz = float(ty).add(0.5).div(RES).sub(0.5).mul(WORLD_SIZE);
        const p0 = vec2(wx, wz) as unknown as NV2;
        const h0 = hf.sampleHeight(p0).add(BASE_LIFT).toVar();
        const sunXZ = vec2(sunDir.x, sunDir.z) as unknown as NV2;
        const dy = (sunDir.y as unknown as NF).max(0.08);
        // worst occluder EXCESS (m above the sun ray) across the march
        const worst = float(0).toVar();
        for (const t of STEPS) {
          const sp = (p0 as unknown as { add(o: unknown): NV2 }).add(
            (sunXZ as unknown as { mul(o: number): NV2 }).mul(t),
          );
          const rayY = h0.add(dy.mul(t)).add(SLOPE_SLACK * t);
          const excess = hf.sampleHeight(sp).sub(rayY);
          worst.assign(worst.max(excess));
        }
        const vis = float(1).sub(smoothstep(float(0), float(SOFT_M), worst));
        textureStore(tex, uvec2(tx, ty), vec4(vis, 0, 0, 1)).toWriteOnly();
      });
    })().compute(RES * RES, [256]);
    (kBake as unknown as { setName(n: string): unknown }).setName('farShadowBake');
    this.kBake = kBake;
    this.bake(renderer);
    await Promise.resolve();
  }

  /** re-march the map (sun changed — ToD edit). ~13 height taps × 1M texels, once. */
  bake(renderer: Renderer): void {
    if (this.kBake) dispatch(renderer, this.kBake);
  }

  /** resolve-side sample: sun visibility at a world xz (bilinear, 1 tap). */
  visAt(wxz: NV2): NF {
    const uv = clamp(
      (wxz as unknown as { div(o: number): NV2 }).div(WORLD_SIZE).add(0.5),
      0,
      1,
    );
    return (texture(this.tex, uv, 0) as unknown as { x: NF }).x;
  }
}

/**
 * CanopyWindow (S4) — the camera-following canopy coverage window, THE canopy
 * field for both world sources (SPEC-STREAMING-WORLD §3 disposition row
 * "Canopy map"; laws 2-3).
 *
 * A 1024² rgba8 window at 4 m/texel (4096 m footprint) splatted on the GPU from
 * RESIDENT TREE RECORDS fetched through the WorldSource ('trees' record chunks —
 * the same universal path ChunkContent placement rides). The splat/blur/pack
 * math is the boot one-shot's exactly (crown radius/opacity per class, (1−d/r)^1.5
 * weights, 3×3 box blur, cov^0.75), so on the generated 4096 m world — where the
 * coverage clamp pins the window over the whole world — the output texture is
 * equivalent to the retired `buildCanopyMap(scatter.trees)` texel for texel.
 *
 * Sampling:
 *  - `covAt(wxz)` — the windowed tap (origin uniform): window coverage inside,
 *    blending to the TerrainField biome-plane `cover` channel beyond the rim
 *    (the far term — asset-gen canopy pyramid on Estonia, zero on generated,
 *    where the window covers everything). Fn-stack contexts only (the far
 *    fallback rides the biome level chain). Used by the S4 compute consumers
 *    (ProbeGI, Froxels, Particles).
 *  - `.tex` — consumers fenced this slice (nanite resolve/grass, water, wind)
 *    keep their world-UV `canopyAt(tex, wxz)` tap on this texture; identical
 *    while the window is pinned (generated world). When the window first MOVES
 *    (S5 Estonia roam) those call sites swap to `covAt` — see the S5 hook list
 *    in the S4 report.
 *
 * Scroll (S5-driven, no-op while pinned): `update()` re-centers the window on
 * 64 m snaps with a dead zone, re-resolves record-chunk residency via
 * source.fetch, and re-splats the whole window off-frame (single-flight; one
 * clear+splat+pack dispatch chain ≈ the boot cost, every ~64 m of travel).
 *
 * VRAM ceiling (stated): tex 4 MB + accum 4 MB + record pool 6.3 MB ≈ 14.3 MB
 * (the one-shot kept 8 MB); REC_CAP overflow throws loud (VRAM law).
 */

import { Vector2 } from 'three';
import { StorageTexture, type Renderer, type StorageBufferNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  clamp,
  float,
  instanceIndex,
  instancedArray,
  max,
  mix,
  smoothstep,
  texture,
  textureStore,
  uint,
  uniform,
  uvec2,
  vec2,
  vec4,
} from 'three/tsl';
import type { NF, NU, NV2, NV4 } from '../TSLTypes';
import type { TerrainField } from '../../nanite/world/TerrainField';
import type { ChunkKey, WorldManifest, WorldSource } from '../../world/source/WorldSource';
import { byBiome } from './Scatter';

const RES = 1024;
/** the crown-splat's native footprint scale — the ±3-texel kernel below is
 *  calibrated to it (crown radii clamp at 11 m ≈ 3 texels) */
const TEXEL = 4;
const WINDOW = RES * TEXEL;
/** window recenter snap + dead zone (m): move on 64 m half-stride crossings,
 *  only once the drift clears the snap (exit > enter, F-8 hysteresis) */
const SNAP = 64;
/** crowns reaching in from just outside the window still splat (3 texels) */
const MARGIN = 12;
/** resident tree records the splat pool holds (≈23k trees/km² over the window;
 *  generated world: 188.7k). Overflow = a data-density surprise, not a config. */
const REC_CAP = 393_216;

/** crown radius (m at scale 1) and skylight opacity per tree class */
const CROWN_R = [2.9, 2.7, 3.8, 2.7, 3.2, 0.9];
const OPACITY = [0.85, 0.7, 0.9, 0.65, 0.8, 0.12];

export class CanopyWindow {
  /** rgba8, cov in .x — the drop-in for every `canopyAt(tex, wxz)` consumer */
  readonly tex: StorageTexture;
  /** world coord of the window's min corner (texel (0,0) edge) */
  private readonly uMin = uniform(new Vector2());
  private readonly uCount = uniform(0);
  private readonly accum: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private readonly recs: StorageBufferNode<'vec4'>;
  private readonly clearK: Parameters<Renderer['compute']>[0];
  private readonly splatK: Parameters<Renderer['compute']>[0];
  private readonly packK: Parameters<Renderer['compute']>[0];
  private minX = 0;
  private minZ = 0;
  private refilling = false;
  private wantX: number | null = null;
  private wantZ: number | null = null;

  private constructor(
    private readonly source: WorldSource,
    private readonly manifest: WorldManifest,
    private readonly field: TerrainField,
  ) {
    const t = new StorageTexture(RES, RES);
    t.name = 'canopyWindow';
    t.generateMipmaps = false;
    this.tex = t;
    this.accum = instancedArray(RES * RES, 'uint').toAtomic();
    this.recs = instancedArray(new Float32Array(REC_CAP * 4), 'vec4');

    this.clearK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(RES * RES), () => {
        Return();
      });
      atomicStore(this.accum.element(i), uint(0));
    })().compute(RES * RES);
    (this.clearK as { setName(n: string): void }).setName('canopyClear');

    this.splatK = Fn(() => {
      const i = instanceIndex;
      If(float(i).greaterThanEqual(this.uCount as unknown as NF), () => {
        Return();
      });
      const rec = this.recs.element(i) as unknown as NV4;
      const cls = rec.w.toInt();
      const r = byBiome(cls, CROWN_R).mul(rec.z).clamp(1, 11);
      const op = byBiome(cls, OPACITY);
      const mn = vec2(this.uMin as unknown as NV2);
      const gx = rec.x.sub(mn.x).div(TEXEL);
      const gy = rec.y.sub(mn.y).div(TEXEL);
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const tx = gx.add(dx).floor();
          const ty = gy.add(dy).floor();
          const inB = tx.greaterThanEqual(0)
            .and(tx.lessThan(RES))
            .and(ty.greaterThanEqual(0))
            .and(ty.lessThan(RES));
          const d = vec2(tx.add(0.5).sub(gx), ty.add(0.5).sub(gy)).length().mul(TEXEL);
          const w = float(1).sub(d.div(r)).max(0).pow(1.5).mul(op).mul(255);
          If(inB.and(w.greaterThan(1)), () => {
            atomicAdd(
              this.accum.element(ty.toInt().mul(RES).add(tx.toInt())),
              w.toUint(),
            );
          });
        }
      }
    })().compute(REC_CAP);
    (this.splatK as { setName(n: string): void }).setName('canopySplat');

    this.packK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(RES * RES), () => {
        Return();
      });
      const x = i.mod(RES);
      const y = i.div(RES);
      // 3×3 box blur of the fixed-point accumulation → soft canopy field
      const sum = float(0).toVar();
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = float(x).add(dx).clamp(0, RES - 1).toInt();
          const yy = float(y).add(dy).clamp(0, RES - 1).toInt();
          sum.addAssign(
            float(atomicLoad(this.accum.element(yy.mul(RES).add(xx))) as unknown as NU),
          );
        }
      }
      const cov = sum.div(9 * 255).div(1.6).clamp(0, 1).pow(0.75);
      textureStore(this.tex, uvec2(x.toUint(), y.toUint()), vec4(cov, cov, cov, 1)).toWriteOnly();
    })().compute(RES * RES);
    (this.packK as { setName(n: string): void }).setName('canopyPack');
  }

  /** Build the window over the source's tree records, camera-centered but
   *  clamped into the coverage box (a source the window covers entirely —
   *  the generated world — pins it). `null` when the source has no trees. */
  static async build(
    renderer: Renderer,
    source: WorldSource,
    manifest: WorldManifest,
    field: TerrainField,
    camX: number,
    camZ: number,
  ): Promise<CanopyWindow | null> {
    if (!manifest.layers.trees) return null;
    const w = new CanopyWindow(source, manifest, field);
    const { x, z } = w.desiredBase(camX, camZ);
    await w.refill(renderer, x, z);
    return w;
  }

  /** windowed coverage tap (Fn-stack contexts): window inside, biome-plane
   *  `cover` channel beyond the rim (zero on the generated world — the pinned
   *  window covers everything there, so the far term never engages). */
  covAt(wxz: NV2): NF {
    const g = wxz.sub(vec2(this.uMin as unknown as NV2)).div(WINDOW);
    const tap = (texture(this.tex, clamp(g, 0, 1) as unknown as NV2, 0) as unknown as NV4).x;
    if (this.field.biomeLevels.length === 0) return tap;
    // per-axis meters outside the window → 64 m blend band into the far term
    const exc = g.sub(0.5).abs().sub(0.5).max(0).mul(WINDOW);
    const k = smoothstep(float(0), float(SNAP), max(exc.x, exc.y));
    const out = tap.toVar();
    If(k.greaterThan(0), () => {
      out.assign(mix(tap, (this.field.biomeAt(wxz) as unknown as { w: NF }).w, k));
    });
    return out as unknown as NF;
  }

  /** S5 camera-follow hook (called every frame; pinned windows no-op): re-center
   *  on 64 m snaps, re-resolve record residency, re-splat off-frame. */
  update(renderer: Renderer, camX: number, camZ: number): void {
    const { x, z } = this.desiredBase(camX, camZ);
    if (x === this.minX && z === this.minZ) return;
    if (this.refilling) {
      this.wantX = x;
      this.wantZ = z;
      return;
    }
    this.refilling = true;
    void this.refill(renderer, x, z)
      .catch((e: unknown) => {
        // eslint-disable-next-line no-console
        console.error('[laas] canopy window refill failed:', e);
      })
      .finally(() => {
        this.refilling = false;
        if (this.wantX !== null && this.wantZ !== null) {
          const nx = this.wantX;
          const nz = this.wantZ;
          this.wantX = null;
          this.wantZ = null;
          this.update(renderer, nx + WINDOW / 2, nz + WINDOW / 2);
        }
      });
  }

  /** camera-centered window base snapped to the recenter stride, clamped into
   *  the coverage box (span ≤ window ⇒ pinned over the coverage center) */
  private desiredBase(camX: number, camZ: number): { x: number; z: number } {
    const box = this.field.coverageBox;
    const axis = (cam: number, lo: number, hi: number): number => {
      if (hi - lo <= WINDOW) return (lo + hi) / 2 - WINDOW / 2;
      const want = Math.round((cam - WINDOW / 2) / SNAP) * SNAP;
      return Math.min(Math.max(want, lo), hi - WINDOW);
    };
    return { x: axis(camX, box.minX, box.maxX), z: axis(camZ, box.minZ, box.maxZ) };
  }

  /** re-resolve record residency for a window base + full re-splat */
  private async refill(renderer: Renderer, baseX: number, baseZ: number): Promise<void> {
    const grid = this.manifest.grid;
    const lo = (v: number, o: number): number => Math.floor((v - MARGIN - o) / grid.chunkMeters);
    const hi = (v: number, o: number): number => Math.floor((v + WINDOW + MARGIN - o) / grid.chunkMeters);
    const keys: ChunkKey[] = [];
    for (let cz = lo(baseZ, grid.originZ); cz <= hi(baseZ, grid.originZ); cz++) {
      for (let cx = lo(baseX, grid.originX); cx <= hi(baseX, grid.originX); cx++) {
        const key = { lod: 0, cx, cz };
        if (this.manifest.coverage('trees', key)) keys.push(key);
      }
    }
    const arr = (this.recs as unknown as { value: { array: Float32Array; needsUpdate: boolean } }).value;
    let n = 0;
    for (const key of keys) {
      const payload = await this.source.fetch('trees', key);
      if (!payload || payload.kind !== 'records') continue;
      const { count, cols } = payload;
      if (n + count > REC_CAP) {
        throw new Error(`CanopyWindow: ${n + count} records exceed the ${REC_CAP} pool ceiling`);
      }
      const cminX = grid.originX + key.cx * grid.chunkMeters;
      const cminZ = grid.originZ + key.cz * grid.chunkMeters;
      for (let i = 0; i < count; i++, n++) {
        const o = n * 4;
        // exactness columns when the source carries them (generated); else the
        // one shared derive rule (chunk-local + chunk origin)
        arr.array[o] = cols.xw ? (cols.xw[i] as number) : cminX + (cols.x[i] as number);
        arr.array[o + 1] = cols.zw ? (cols.zw[i] as number) : cminZ + (cols.z[i] as number);
        arr.array[o + 2] = cols.scale[i] as number;
        arr.array[o + 3] = cols.species[i] as number;
      }
    }
    arr.needsUpdate = true;
    this.uCount.value = n;
    this.minX = baseX;
    this.minZ = baseZ;
    this.uMin.value.set(baseX, baseZ);
    renderer.compute(this.clearK);
    renderer.compute(this.splatK);
    await renderer.computeAsync(this.packK);
  }
}

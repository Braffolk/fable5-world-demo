/**
 * FarShadow — P4 of the shadow arc (2026-07-03): baked heightfield sun-visibility,
 * the "should a massive hill far away cast a shadow?" answer (YES).
 *
 * The camera-centred shadow clipmap reaches 384 m half-extent — a mountain beyond
 * the ring can never write shadow depth, and a hill 500 m away can never shade a
 * receiver 100 m away (the caster is outside every level's box even though the
 * receiver is inside). Extending the clipmap is the wrong tool (each doubling
 * re-rasters enormous rings for sub-pixel detail); the standard answer is a baked
 * horizon/sun-visibility term over the terrain:
 *
 *   one r16f map (RES²) over a CAMERA WINDOW of min(16 km, coverage span) —
 *   S4: base-clamped into the TerrainField coverage box, so the generated 4096 m
 *   world pins it over the whole world (= the retired world-static map, 4 m
 *   texels) while Estonia gets a 16 km window at 16 m texels riding the same
 *   toroidal storage. Each texel = sun visibility at ground level, ray-marched
 *   toward the sun over the TerrainField height planes with growing strides
 *   (reach ~1 km — mountain-scale). Re-baked in full ONLY on sun change (ToD
 *   edit; the march reads the live sun uniform); window scrolls re-bake just the
 *   entering rows/cols under a per-frame texel budget (F-8). Sampled in the
 *   nanite resolve as ONE bilinear tap, multiplied into the sun term at ALL
 *   distances — mountains shade valleys at any receiver distance, sunset throws
 *   km-long terrain shadows, and within the clipmap band the overlap only
 *   deepens penumbra (both terms are ~binary there). Beyond the window the term
 *   fades to UNSHADOWED over 128 m — at 16 km a terrain shadow is sub-pixel
 *   (demand law, not a drop).
 *
 * Approximations (booked): receivers are assumed near ground level (a 30 m crown
 * top inside a soft boundary is slightly over-darkened — the coarse texels + the
 * 1.5° smoothstep keep the boundary soft); canopy long-shadows (forest silhouette
 * at sunset) are NOT in the map yet — candidate follow-up: add canopy-slab height
 * (cov·CAN_TOP) to the occluder height with a transmittance floor.
 */

import { HalfFloatType, RedFormat, RepeatWrapping, Vector2 } from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import { LinearFilter } from 'three';
import {
  Fn,
  If,
  clamp,
  float,
  instanceIndex,
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
import type { NF, NV2 } from '../TSLTypes';
import type { TerrainField } from '../../nanite/world/TerrainField';
import type { Atmosphere } from '../../sky/Atmosphere';
import { dispatch } from '../../nanite/Tsl';

const RES = 1024;
/** window footprint cap (m): beyond it terrain shadows are sub-pixel */
const WINDOW_CAP = 16384;
/** march sample distances (m) — dense near for local ridges, sparse far for
 *  mountains; reach ~1.1 km ≈ a 500 m peak at ~25° sun elevation. */
const STEPS = [6, 12, 22, 38, 62, 98, 150, 225, 330, 470, 660, 900, 1120];
/** receiver lift + per-metre slack (m) — absorbs heightfield filtering error so a
 *  texel never self-shadows on its own slope. */
const BASE_LIFT = 1.5;
const SLOPE_SLACK = 0.015;
/** occluder-excess (m) over which visibility fades 1→0 — the soft boundary. */
const SOFT_M = 6;
/** window recenter snap (texels): rows/cols re-bake in these strides; the snap
 *  doubles as the dead zone (exit > enter, F-8) */
const SNAP_TX = 32;
/** scroll re-bake budget (texels/frame): 64 rows ≈ 850k height taps, off-frame */
const RECT_CAP = RES * 64;
/** beyond-window fade band (m) → unshadowed */
const FADE_M = 128;

export class FarShadow {
  readonly tex: StorageTexture;
  private kBake: unknown = null;
  private kRect: unknown = null;
  /** window base (texel indices into the coverage lattice) */
  private readonly uBase = uniform(new Vector2(0, 0));
  /** pending scroll re-bake rects (world-texel space), FIFO */
  private readonly rects: { x0: number; z0: number; w: number; h: number }[] = [];
  private readonly uRect = uniform(new Vector2(0, 0));
  private readonly uRectSize = uniform(new Vector2(0, 0));
  /** world min corner of the coverage lattice + this window's texel size */
  private readonly latMinX: number;
  private readonly latMinZ: number;
  private readonly texel: number;
  /** coverage span in texels (window pinned when ≤ RES) */
  private readonly covTx: number;
  private readonly covTz: number;

  constructor(
    /** the TerrainField height planes — every (re-)bake marches them live */
    private field: TerrainField,
    private atmosphere: Atmosphere,
  ) {
    const box = field.coverageBox;
    const span = Math.max(box.maxX - box.minX, box.maxZ - box.minZ);
    this.texel = Math.min(span, WINDOW_CAP) / RES;
    this.latMinX = box.minX;
    this.latMinZ = box.minZ;
    this.covTx = Math.max(1, Math.round((box.maxX - box.minX) / this.texel));
    this.covTz = Math.max(1, Math.round((box.maxZ - box.minZ) / this.texel));
    const t = new StorageTexture(RES, RES);
    t.type = HalfFloatType;
    t.format = RedFormat;
    t.magFilter = LinearFilter;
    t.minFilter = LinearFilter;
    t.generateMipmaps = false;
    // toroidal window: world texel c lives at texture texel c mod RES
    t.wrapS = RepeatWrapping;
    t.wrapT = RepeatWrapping;
    t.name = 'farShadowVis';
    this.tex = t;
  }

  async init(renderer: Renderer): Promise<void> {
    const field = this.field;
    const sunDir = this.atmosphere.sunDir;
    const tex = this.tex;

    /** march the sun ray from a world-texel index pair, store at its wrapped texel */
    const bakeTexel = (wtx: NF, wtz: NF): void => {
      const wx = wtx.add(0.5).mul(this.texel).add(this.latMinX);
      const wz = wtz.add(0.5).mul(this.texel).add(this.latMinZ);
      const p0 = vec2(wx, wz) as unknown as NV2;
      const h0 = field.fieldHeightFinest(p0).add(BASE_LIFT).toVar();
      const sunXZ = vec2(sunDir.x, sunDir.z) as unknown as NV2;
      const dy = (sunDir.y as unknown as NF).max(0.08);
      // worst occluder EXCESS (m above the sun ray) across the march
      const worst = float(0).toVar();
      for (const t of STEPS) {
        const sp = (p0 as unknown as { add(o: unknown): NV2 }).add(
          (sunXZ as unknown as { mul(o: number): NV2 }).mul(t),
        );
        const rayY = h0.add(dy.mul(t)).add(SLOPE_SLACK * t);
        const excess = field.fieldHeightFinest(sp).sub(rayY);
        worst.assign(worst.max(excess));
      }
      const vis = float(1).sub(smoothstep(float(0), float(SOFT_M), worst));
      const px = wtx.mod(RES).add(RES).mod(RES);
      const pz = wtz.mod(RES).add(RES).mod(RES);
      textureStore(tex, uvec2(px.toUint(), pz.toUint()), vec4(vis, 0, 0, 1)).toWriteOnly();
    };

    const kBake = Fn(() => {
      const idx = instanceIndex;
      If(idx.lessThan(uint(RES * RES)), () => {
        const px = float(idx.mod(uint(RES)));
        const pz = float(idx.div(uint(RES)));
        // toroidal unwrap: world texel ≡ p (mod RES) inside [base, base+RES)
        const base = vec2(this.uBase as unknown as NV2);
        bakeTexel(
          base.x.add(px.sub(base.x).mod(RES).add(RES).mod(RES)),
          base.y.add(pz.sub(base.y).mod(RES).add(RES).mod(RES)),
        );
      });
    })().compute(RES * RES, [256]);
    (kBake as unknown as { setName(n: string): unknown }).setName('farShadowBake');
    this.kBake = kBake;

    // scroll re-bake: one queued rect of ENTERING world texels per dispatch
    const kRect = Fn(() => {
      const idx = instanceIndex;
      const size = vec2(this.uRectSize as unknown as NV2);
      If(float(idx).lessThan(size.x.mul(size.y)), () => {
        const w = size.x;
        const rect = vec2(this.uRect as unknown as NV2);
        bakeTexel(
          rect.x.add(float(idx).mod(w)),
          rect.y.add(float(idx).div(w).floor()),
        );
      });
    })().compute(RECT_CAP, [256]);
    (kRect as unknown as { setName(n: string): unknown }).setName('farShadowBakeRect');
    this.kRect = kRect;

    this.bake(renderer);
    await Promise.resolve();
  }

  /** re-march the whole window (sun changed — ToD edit). ~13 height taps × 1M
   *  texels, once; supersedes any queued scroll rects. */
  bake(renderer: Renderer): void {
    this.rects.length = 0;
    if (this.kBake) dispatch(renderer, this.kBake);
  }

  /**
   * S5 camera-follow hook (call every frame; pinned windows — the generated
   * world — no-op): re-center the window base in SNAP_TX strides, queue the
   * entering rows/cols, drain ONE rect per frame under RECT_CAP (F-8 FIFO).
   */
  follow(renderer: Renderer, camX: number, camZ: number): void {
    const desired = (cam: number, latMin: number, covT: number): number => {
      const want = Math.round((cam - latMin) / this.texel / SNAP_TX) * SNAP_TX - RES / 2;
      return Math.min(Math.max(want, 0), Math.max(0, covT - RES));
    };
    const bx = desired(camX, this.latMinX, this.covTx);
    const bz = desired(camZ, this.latMinZ, this.covTz);
    const cur = this.uBase.value;
    if (bx !== cur.x || bz !== cur.y) {
      const dx = bx - cur.x;
      const dz = bz - cur.y;
      if (Math.abs(dx) >= RES || Math.abs(dz) >= RES) {
        // teleport: everything enters — one full re-bake
        this.uBase.value.set(bx, bz);
        this.bake(renderer);
        return;
      }
      if (dx !== 0) {
        this.rects.push({
          x0: dx > 0 ? cur.x + RES : bx,
          z0: bz,
          w: Math.abs(dx),
          h: RES,
        });
      }
      if (dz !== 0) {
        this.rects.push({
          x0: bx,
          z0: dz > 0 ? cur.y + RES : bz,
          w: RES,
          h: Math.abs(dz),
        });
      }
      this.uBase.value.set(bx, bz);
    }
    const rect = this.rects.shift();
    if (rect && this.kRect) {
      // split oversized rects back onto the queue (budget law)
      const rows = Math.max(1, Math.floor(RECT_CAP / rect.w));
      if (rect.h > rows) {
        this.rects.unshift({ ...rect, z0: rect.z0 + rows, h: rect.h - rows });
        rect.h = rows;
      }
      this.uRect.value.set(rect.x0, rect.z0);
      this.uRectSize.value.set(rect.w, rect.h);
      dispatch(renderer, this.kRect);
    }
  }

  /** resolve-side sample: sun visibility at a world xz (bilinear, 1 tap).
   *  Continuous texel coords clamp to the window rim (≡ the old clamp-to-edge
   *  when the window is pinned over the world); beyond the rim the term fades
   *  to unshadowed over FADE_M (sub-pixel shadows at 16 km — demand law). */
  visAt(wxz: NV2): NF {
    const w = wxz as unknown as { x: NF; y: NF };
    const base = vec2(this.uBase as unknown as NV2);
    const cx = w.x.sub(this.latMinX).div(this.texel).sub(0.5);
    const cz = w.y.sub(this.latMinZ).div(this.texel).sub(0.5);
    const ux = clamp(cx, base.x, base.x.add(RES - 1));
    const uz = clamp(cz, base.y, base.y.add(RES - 1));
    const tap = (
      texture(this.tex, vec2(ux.add(0.5).div(RES), uz.add(0.5).div(RES)) as unknown as NV2, 0) as unknown as {
        x: NF;
      }
    ).x;
    const exc = max(cx.sub(ux).abs(), cz.sub(uz).abs()).mul(this.texel);
    return mix(tap, float(1), smoothstep(float(0), float(FADE_M), exc)) as unknown as NF;
  }
}

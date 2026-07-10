/**
 * Irradiance probe field (Phase 3 GI) — replaces the hemisphere-light ambient.
 *
 * Terrain-only world ⇒ probes gather by RAY-MARCHING THE HEIGHTFIELD in
 * compute (no per-probe scene renders):
 *  - 256×256 horizontal probes at 16 m spacing — since S4 a TOROIDAL
 *    CAMERA WINDOW (4096 m), base-clamped into the source coverage box, so a
 *    source the window covers entirely (the generated 4096 m world) pins it
 *    and the field is world-static exactly as the retired one-shot was.
 *    6 TERRAIN-RELATIVE layers (1.5…105 m above ground) — the slabs follow
 *    the surface.
 *  - Per probe, D jittered fibonacci directions; each ray either hits the
 *    heightfield (→ albedo proxy × sun N·L × horizon-test visibility, plus a
 *    small second-bounce fudge) or escapes to the sky-view LUT.
 *  - Radiance is projected to SH-L1 and EMA'd into storage buffers
 *    (time-sliced: PROBES_PER_FRAME per frame ⇒ full refresh < 1 s, so
 *    time-of-day changes wash through automatically).
 *  - A publish kernel copies updated probes into three rgba16f 3D textures
 *    (one per color channel, xyzw = SH c0,c1x,c1y,c1z) for hardware
 *    trilinear sampling in materials via `probeIrradiance()` — REPEAT wrap on
 *    xz makes the toroidal window seamless; samples clamp to the window rim.
 *  - Window scroll (S5 camera roam; no-op while pinned): half-stride base
 *    crossings queue the entering probe rows/cols; a queue-driven twin of the
 *    gather/publish pair re-bakes them at blend=1 under the same
 *    PROBES_PER_FRAME budget (F-8: budgeted, FIFO). A jump bigger than the
 *    whole field falls back to invalidate() (the teleport refill, spec §5).
 */

import { HalfFloatType, RepeatWrapping, Vector2 } from 'three';
import type { Renderer } from 'three/webgpu';
import { Storage3DTexture, type StorageBufferNode } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  Return,
  clamp,
  dot,
  exp,
  float,
  instanceIndex,
  instancedArray,
  log2,
  max,
  min,
  mix,
  texture3D,
  textureStore,
  uniform,
  uvec3,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { TerrainField } from '../../nanite/world/TerrainField';
import type { Atmosphere } from '../../sky/Atmosphere';
import { SUN_E } from '../../sky/Atmosphere';
import { hash12 } from '../noise/NoiseTSL';
import type { NF, NI, NV2, NV3 } from '../TSLTypes';
import type { CanopyWindow } from './CanopyWindow';

const PROBE_XZ = 256;
const PROBE_LAYERS = 6;
/** layer i sits LAYER_BASE·LAYER_RATIO^i meters above ground */
const LAYER_BASE = 1.5;
const LAYER_RATIO = 2.36;
/** probe spacing (m) — window footprint = PROBE_XZ·SPACING = 4096 m */
const SPACING = 16;
const TOTAL = PROBE_XZ * PROBE_XZ * PROBE_LAYERS;
const PROBES_PER_FRAME = 3072;
const DIRS = 16;
const MARCH_STEPS = 16;
/** window recenter dead zone (strides): adopt a new base only past 2 strides —
 *  wider than the half-stride crossing that computes it (F-8 exit > enter) */
const BASE_DEADZONE = 2;

/**
 * Canopy slab for probe rays: tree crowns approximated as a terrain-relative
 * Beer-Lambert slab (CAN_BOT…CAN_TOP m above ground) whose extinction scales
 * with the canopy-window coverage. This is what makes forest interiors DIM in
 * the probe field — without it probes see the naked heightfield, gather the
 * full sky dome + sun bounce under dense forest, and noon interiors wash out
 * to a flat ambient (no dapple contrast against the direct-sun pools).
 */
const CAN_BOT = 5;
const CAN_TOP = 23;
/**
 * Chromatic extinction per meter at coverage 1 (vertical crossing ⇒
 * T ≈ 0.10/0.21/0.09 rgb). Green passes ~2× more than red/blue — leaf
 * transmission — so everything under the canopy is bathed in the green
 * light of the references, instead of a neutral gray dim.
 */
const CAN_SIGMA = [0.128, 0.087, 0.134] as const;
const CAN_SIGMA_Y = 0.11;

export class ProbeGI {
  /** SH-L1 per color channel: xyzw = c0, c1x, c1y, c1z */
  readonly texR: Storage3DTexture;
  readonly texG: Storage3DTexture;
  readonly texB: Storage3DTexture;
  private shR: StorageBufferNode<'vec4'>;
  private shG: StorageBufferNode<'vec4'>;
  private shB: StorageBufferNode<'vec4'>;
  private gatherK: Parameters<Renderer['compute']>[0] | null = null;
  private publishK: Parameters<Renderer['compute']>[0] | null = null;
  private scrollGatherK: Parameters<Renderer['compute']>[0] | null = null;
  private scrollPublishK: Parameters<Renderer['compute']>[0] | null = null;
  private frameBase = uniform(0);
  private blend = uniform(0.22);
  private rot = uniform(0);
  /** window base (probe-grid indices into the coverage lattice) */
  private uBase = uniform(new Vector2(0, 0));
  /** world coord of coverage probe column/row 0 (its CENTER) */
  private readonly anchorX: number;
  private readonly anchorZ: number;
  /** probe columns/rows the coverage box spans (window pinned when ≤ 256) */
  private readonly covColsX: number;
  private readonly covColsZ: number;
  /** queued texture-probe ids awaiting a scroll re-bake (CPU side) */
  private pending: number[] = [];
  private queue = instancedArray(PROBES_PER_FRAME, 'uint');
  private uQueueN = uniform(0);
  /** frames of boosted blend after a ToD jump */
  private boost = 0;
  /** P6 gi-sleep (shadow arc 2026-07-03): ticks since the last invalidate() */
  private cleanFrames = 0;
  private sleepOn = new URLSearchParams(window.location.search).get('gisleep') !== '0';

  constructor(
    /** the TerrainField planes — height/normal/biome source for the gather
     *  (S3b/S4: the kernels re-dispatch per frame, so they must never read the
     *  released boot Heightfield GPU set) */
    private field: TerrainField,
    private atmosphere: Atmosphere,
    private canopy: CanopyWindow | null = null,
  ) {
    const box = field.coverageBox;
    this.anchorX = box.minX + SPACING / 2;
    this.anchorZ = box.minZ + SPACING / 2;
    this.covColsX = Math.max(1, Math.round((box.maxX - box.minX) / SPACING));
    this.covColsZ = Math.max(1, Math.round((box.maxZ - box.minZ) / SPACING));
    const mk = (name: string): Storage3DTexture => {
      const t = new Storage3DTexture(PROBE_XZ, PROBE_XZ, PROBE_LAYERS);
      t.type = HalfFloatType;
      t.generateMipmaps = false;
      t.name = name;
      // toroidal window: xz wraps (world col c lives at texel c mod 256)
      t.wrapS = RepeatWrapping;
      t.wrapT = RepeatWrapping;
      return t;
    };
    this.texR = mk('probeFieldR');
    this.texG = mk('probeFieldG');
    this.texB = mk('probeFieldB');
    this.shR = instancedArray(TOTAL, 'vec4');
    this.shG = instancedArray(TOTAL, 'vec4');
    this.shB = instancedArray(TOTAL, 'vec4');
  }

  async init(renderer: Renderer): Promise<void> {
    const field = this.field;

    const heightAt = (p: NV2): NF => field.fieldHeightFinest(p);

    // canopy coverage 0..1 at a world xz (0 when no canopy window is wired)
    const canopy = this.canopy;
    const covAt = (wxz: NV2): NF => (canopy ? canopy.covAt(wxz) : (float(0) as unknown as NF));

    /**
     * Path-integrated coverage (m of cov=1 equivalent) of the crown slab
     * along a ray (pure expression). Slab is terrain-relative at the RAY
     * ORIGIN's ground height — crowns follow the surface closely enough at
     * probe scales. Grazing paths cap at 90 m (crown fields are patchy; one
     * mid-point coverage sample).
     */
    const canopyPath = (origin: NV3, dir: NV3, ground: NF, tCap?: NF): NF => {
      if (!canopy) return float(0);
      const ady = dir.y.abs().max(0.025);
      const dy = dir.y.greaterThanEqual(0).select(ady, ady.negate());
      const h = origin.y.sub(ground);
      const t1 = float(CAN_BOT).sub(h).div(dy);
      const t2 = float(CAN_TOP).sub(h).div(dy);
      const tEnter = min(t1, t2).max(0);
      let tExit: NF = max(t1, t2).clamp(0, 90);
      if (tCap) tExit = tExit.min(tCap);
      const len = tExit.sub(tEnter).max(0);
      const mid = origin.xz.add(dir.xz.mul(tEnter.add(tExit).mul(0.5)));
      return covAt(mid).mul(len);
    };

    /** chromatic slab transmittance for a path-integrated coverage */
    const canopyT = (path: NF): NV3 =>
      vec3(
        exp(path.mul(-CAN_SIGMA[0])),
        exp(path.mul(-CAN_SIGMA[1])),
        exp(path.mul(-CAN_SIGMA[2])),
      ) as unknown as NV3;

    // ground-hit radiance proxy: biome palette × (sun + sky fudge)
    const sunDir = this.atmosphere.sunDir;
    const hitRadiance = (hp: NV3): NV3 => {
      // vegDensity from the biome plane, snow from the surface-fields plane
      // (S4: the boot biomeTex is released — these are the streamed planes)
      const veg = (field.biomeAt(hp.xz as unknown as NV2) as unknown as { y: NF }).y;
      const snowK = (field.fieldsAt(hp.xz as unknown as NV2) as unknown as { z: NF }).z;
      const nrm = field.fieldNormalSlope(hp.xz as unknown as NV2).xyz as unknown as NV3;
      const grass = vec3(0.16, 0.2, 0.09);
      const rock = vec3(0.3, 0.28, 0.25);
      const snow = vec3(0.8, 0.82, 0.88);
      const litter = vec3(0.1, 0.075, 0.045);
      let albedo: NV3 = mix(rock, grass, veg);
      albedo = mix(albedo, snow, snowK);
      const cov = covAt(hp.xz);
      // forest floor is leaf litter / moss, not meadow grass — and the
      // crowns overhead shadow both the sun and the sky on the bounce source
      albedo = mix(albedo, litter, cov.mul(0.75)) as NV3;
      // sun horizon test from the hit (short march)
      const sVis = float(1).toVar();
      const sunXZ = vec2(sunDir.x, sunDir.z);
      for (let s = 1; s <= 6; s++) {
        const t = 14 * s * s;
        const sp = hp.xz.add(sunXZ.mul(t));
        const sy = hp.y.add(sunDir.y.mul(t));
        sVis.mulAssign(heightAt(sp).lessThan(sy.add(1)).select(float(1), float(0.0)));
      }
      const sunSlab = canopyT(
        cov.mul(float(CAN_TOP - CAN_BOT).div(clamp(sunDir.y, 0.25, 1))),
      );
      const ndl = clamp(dot(nrm, sunDir), 0, 1);
      const sun = this.atmosphere
        .sampleTransmittance(float(6360.35), clamp(sunDir.y, -1, 1))
        .mul(SUN_E)
        .mul(ndl)
        .mul(sVis)
        .mul(sunSlab)
        .div(Math.PI);
      // skylight on the hit (cheap: zenith sky × upness) — second bounce fudge
      const skyUp = this.atmosphere
        .skyColor(vec3(0, 1, 0))
        .mul(0.25)
        .mul(canopyT(cov.mul(11)));
      return albedo.mul(sun.add(skyUp.mul(clamp(nrm.y, 0, 1))));
    };

    /** texture probe id → gather + EMA into the SH buffers. `blend` = 1 for
     *  scroll re-bakes (the previous SH is another world column's). */
    const gatherProbe = (pid: NI, blend: NF): void => {
      // ground-level sun color (transmittance × irradiance) for crown glow
      const sunCol = this.atmosphere
        .sampleTransmittance(float(6360.35), clamp(sunDir.y, -1, 1))
        .mul(SUN_E)
        .div(Math.PI);
      const lay = pid.div(PROBE_XZ * PROBE_XZ);
      const rem = pid.mod(PROBE_XZ * PROBE_XZ);
      const px = rem.mod(PROBE_XZ);
      const pz = rem.div(PROBE_XZ);
      // toroidal unwrap: world column ≡ px (mod 256) inside [base, base+256)
      const base = vec2(this.uBase as unknown as NV2);
      const wcx = base.x.add(float(px).sub(base.x).mod(PROBE_XZ).add(PROBE_XZ).mod(PROBE_XZ));
      const wcz = base.y.add(float(pz).sub(base.y).mod(PROBE_XZ).add(PROBE_XZ).mod(PROBE_XZ));
      const wx = wcx.mul(SPACING).add(this.anchorX);
      const wz = wcz.mul(SPACING).add(this.anchorZ);
      const ground = heightAt(vec2(wx, wz));
      const layerH = float(LAYER_BASE).mul(
        float(LAYER_RATIO).pow(float(lay)),
      );
      const ppos = vec3(wx, ground.add(layerH), wz).toVar();

      const c0R = float(0).toVar();
      const c0G = float(0).toVar();
      const c0B = float(0).toVar();
      const c1R = vec3(0).toVar();
      const c1G = vec3(0).toVar();
      const c1B = vec3(0).toVar();

      Loop(DIRS, ({ i: di }: { readonly i: NI }) => {
        // jittered fibonacci sphere (rotates per refresh pass)
        const fi = float(di).add(hash12(vec2(float(pid), float(this.rot))).mul(0.8));
        const phi = fi.mul(2.39996323).add(float(this.rot));
        const y = float(1).sub(fi.add(0.5).mul(2 / DIRS));
        const r = float(1).sub(y.mul(y)).max(0).sqrt();
        const dir = vec3(phi.cos().mul(r), y, phi.sin().mul(r)).toVar();

        // march the heightfield
        const hitT = float(-1).toVar();
        const t = float(6).toVar();
        Loop(MARCH_STEPS, () => {
          const sp = ppos.add(dir.mul(t));
          If(sp.y.lessThan(heightAt(sp.xz)).and(hitT.lessThan(0)), () => {
            hitT.assign(t);
          });
          t.mulAssign(1.6);
        });

        const L = vec3(0).toVar();
        const path = float(0).toVar();
        If(hitT.greaterThan(0), () => {
          // bounce seen THROUGH the slab (probe above canopy looking down,
          // or across a crown field) is extinguished like the sky is
          path.assign(canopyPath(ppos, dir, ground, hitT));
          L.assign(hitRadiance(ppos.add(dir.mul(hitT))).mul(canopyT(path)));
        }).Else(() => {
          path.assign(canopyPath(ppos, dir, ground));
          L.assign(this.atmosphere.skyColor(dir).mul(canopyT(path)));
        });
        // translucent crown glow: whatever the slab swallowed is re-emitted
        // as leaf-filtered green (the "lit canopy from below" of the refs) —
        // this is what keeps interiors COLORFUL instead of neutral-dark
        L.addAssign(
          sunCol
            .mul(vec3(0.04, 0.085, 0.022))
            .mul(float(1).sub(path.mul(CAN_SIGMA_Y).negate().exp())),
        );

        // SH-L1 projection (radiance)
        const w = 4 / DIRS; // Σ ≈ 4π·(1/D)·… folded into eval constants
        c0R.addAssign(L.x.mul(w));
        c0G.addAssign(L.y.mul(w));
        c0B.addAssign(L.z.mul(w));
        c1R.addAssign(dir.mul(L.x.mul(w)));
        c1G.addAssign(dir.mul(L.y.mul(w)));
        c1B.addAssign(dir.mul(L.z.mul(w)));
      });

      const prevR = this.shR.element(pid);
      const prevG = this.shG.element(pid);
      const prevB = this.shB.element(pid);
      prevR.assign(mix(prevR, vec4(c0R, c1R.x, c1R.y, c1R.z), blend));
      prevG.assign(mix(prevG, vec4(c0G, c1G.x, c1G.y, c1G.z), blend));
      prevB.assign(mix(prevB, vec4(c0B, c1B.x, c1B.y, c1B.z), blend));
    };

    /** texture probe id → publish its SH into the 3D textures */
    const publishProbe = (pid: NI): void => {
      const lay = pid.div(PROBE_XZ * PROBE_XZ);
      const rem = pid.mod(PROBE_XZ * PROBE_XZ);
      const px = rem.mod(PROBE_XZ);
      const pz = rem.div(PROBE_XZ);
      const xyz = uvec3(px.toUint(), pz.toUint(), lay.toUint());
      textureStore(this.texR, xyz, this.shR.element(pid)).toWriteOnly();
      textureStore(this.texG, xyz, this.shG.element(pid)).toWriteOnly();
      textureStore(this.texB, xyz, this.shB.element(pid)).toWriteOnly();
    };

    this.gatherK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(PROBES_PER_FRAME), () => {
        Return();
      });
      const pid = float(this.frameBase).add(float(i)).mod(TOTAL).toInt();
      gatherProbe(pid, float(this.blend) as unknown as NF);
    })().compute(PROBES_PER_FRAME);
    this.gatherK.setName('probeGather');

    this.publishK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(PROBES_PER_FRAME), () => {
        Return();
      });
      const pid = float(this.frameBase).add(float(i)).mod(TOTAL).toInt();
      publishProbe(pid);
    })().compute(PROBES_PER_FRAME);
    this.publishK.setName('probePublish');

    // queue-driven twins: entering rows/cols after a window base move (blend=1
    // — the previous SH at those texels belongs to the world's opposite side)
    this.scrollGatherK = Fn(() => {
      const i = instanceIndex;
      If(float(i).greaterThanEqual(this.uQueueN as unknown as NF), () => {
        Return();
      });
      const pid = (this.queue.element(i) as unknown as { toInt(): NI }).toInt();
      gatherProbe(pid, float(1) as unknown as NF);
    })().compute(PROBES_PER_FRAME);
    this.scrollGatherK.setName('probeScrollGather');

    this.scrollPublishK = Fn(() => {
      const i = instanceIndex;
      If(float(i).greaterThanEqual(this.uQueueN as unknown as NF), () => {
        Return();
      });
      const pid = (this.queue.element(i) as unknown as { toInt(): NI }).toInt();
      publishProbe(pid);
    })().compute(PROBES_PER_FRAME);
    this.scrollPublishK.setName('probeScrollPublish');

    // warm the whole field once (batched: uniform updates must land per
    // dispatch, so submit pairs, awaiting only every 16 batches)
    this.blend.value = 1;
    const batches = Math.ceil(TOTAL / PROBES_PER_FRAME);
    for (let n = 0; n < batches; n++) {
      const wait = n % 16 === 15 || n === batches - 1;
      if (wait) {
        const warmGroup = [this.gatherK, this.publishK];
        (warmGroup as { id?: string }).id = 'probeWarm';
        await renderer.computeAsync(warmGroup);
      }
      else {
        renderer.compute(this.gatherK);
        renderer.compute(this.publishK);
      }
      this.frameBase.value = (this.frameBase.value + PROBES_PER_FRAME) % TOTAL;
    }
    this.blend.value = 0.22;
  }

  /**
   * S5 camera-follow hook (call every frame; a pinned window — the generated
   * world — computes the same clamped base forever and no-ops). Base moves on
   * half-stride crossings past the dead zone; entering rows/cols queue for the
   * budgeted scroll re-bake in tick().
   */
  followCamera(camX: number, camZ: number): void {
    const desired = (cam: number, anchor: number, covCols: number): number => {
      const want = Math.round((cam - anchor) / SPACING) - PROBE_XZ / 2;
      return Math.min(Math.max(want, 0), Math.max(0, covCols - PROBE_XZ));
    };
    const bx = desired(camX, this.anchorX, this.covColsX);
    const bz = desired(camZ, this.anchorZ, this.covColsZ);
    const cur = this.uBase.value;
    if (Math.abs(bx - cur.x) < BASE_DEADZONE && Math.abs(bz - cur.y) < BASE_DEADZONE) return;
    const dx = bx - cur.x;
    const dz = bz - cur.y;
    // a jump wider than the field = teleport → full refill, not a queue flood
    if (Math.abs(dx) >= PROBE_XZ || Math.abs(dz) >= PROBE_XZ) {
      this.uBase.value.set(bx, bz);
      this.pending.length = 0;
      this.invalidate();
      return;
    }
    // entering world columns (x) and rows (z) → their texture texels
    const pushCol = (wc: number): void => {
      const px = ((wc % PROBE_XZ) + PROBE_XZ) % PROBE_XZ;
      for (let lay = 0; lay < PROBE_LAYERS; lay++) {
        for (let pz = 0; pz < PROBE_XZ; pz++) {
          this.pending.push(lay * PROBE_XZ * PROBE_XZ + pz * PROBE_XZ + px);
        }
      }
    };
    const pushRow = (wr: number): void => {
      const pz = ((wr % PROBE_XZ) + PROBE_XZ) % PROBE_XZ;
      for (let lay = 0; lay < PROBE_LAYERS; lay++) {
        for (let px = 0; px < PROBE_XZ; px++) {
          this.pending.push(lay * PROBE_XZ * PROBE_XZ + pz * PROBE_XZ + px);
        }
      }
    };
    for (let c = 0; c < Math.abs(dx); c++) {
      pushCol(dx > 0 ? cur.x + PROBE_XZ + c : cur.x - 1 - c);
    }
    for (let r = 0; r < Math.abs(dz); r++) {
      pushRow(dz > 0 ? cur.y + PROBE_XZ + r : cur.y - 1 - r);
    }
    this.uBase.value.set(bx, bz);
    this.cleanFrames = 0; // wake the sleeping field — the window moved
    if (this.pending.length > TOTAL) {
      this.pending.length = 0;
      this.invalidate();
    }
  }

  /** one time slice per frame (sync submit, no readback) */
  tick(renderer: Renderer): void {
    if (!this.gatherK || !this.publishK) return;
    // scroll re-bakes preempt the sequential sweep under the SAME budget
    if (this.pending.length > 0 && this.scrollGatherK && this.scrollPublishK) {
      const n = Math.min(this.pending.length, PROBES_PER_FRAME);
      const attr = (this.queue as unknown as { value: { array: Uint32Array; needsUpdate: boolean } }).value;
      for (let i = 0; i < n; i++) attr.array[i] = this.pending[i] as number;
      attr.needsUpdate = true;
      this.pending.splice(0, n);
      this.uQueueN.value = n;
      renderer.compute(this.scrollGatherK);
      renderer.compute(this.scrollPublishK);
      return;
    }
    // P6 gi-sleep-when-converged (shadow arc 2026-07-03): with a static sun the
    // field is fully converged after two complete refresh cycles past any boost;
    // every later gather re-estimates the SAME integral with a rotated jitter
    // (EMA 0.22 ⇒ sub-noise wobble). Sleep until invalidate() (ToD edit) wakes it.
    // ?gisleep=0 restores the legacy always-churn.
    if (this.sleepOn && this.cleanFrames > 2 * Math.ceil(TOTAL / PROBES_PER_FRAME) + 4) {
      return;
    }
    this.cleanFrames++;
    renderer.compute(this.gatherK);
    renderer.compute(this.publishK);
    this.frameBase.value = (this.frameBase.value + PROBES_PER_FRAME) % TOTAL;
    this.rot.value = (this.rot.value + 1.61803) % 6.2831;
    if (this.boost > 0) {
      this.boost--;
      if (this.boost === 0) this.blend.value = 0.22;
    }
  }

  /** call after a time-of-day jump: converge faster for a full cycle */
  invalidate(): void {
    this.blend.value = 0.6;
    this.boost = Math.ceil(TOTAL / PROBES_PER_FRAME) + 2;
    this.cleanFrames = 0; // P6: wake the sleeping field
  }

  /**
   * Irradiance at a world position/normal (SH-L1 cosine-lobe evaluation).
   * Sample point is pushed up by `lift` meters (normal offset of the caller).
   */
  /** `groundY`: optional pre-sampled ground height. The vis-buffer resolve
   *  (F9, ≤10 storage buffers/stage) passes its own height-plane tap; other
   *  callers omit it and sample the TerrainField height planes here. */
  irradiance(wp: NV3, n: NV3, lift = 2.0, groundY?: NF): NV3 {
    const gY = groundY ?? this.field.fieldHeightFinest(wp.xz);
    const hAbove = max(wp.y.sub(gY).add(lift), 0.0);
    // invert layerH = BASE·RATIO^i  →  i = log2(h/BASE)/log2(RATIO)
    const li = clamp(
      log2(hAbove.div(LAYER_BASE).max(1)).div(Math.log2(LAYER_RATIO)),
      0,
      PROBE_LAYERS - 1,
    );
    // continuous probe-column coords, clamped to the window rim (≡ the old
    // clamp-to-edge on the world-static field), sampled with REPEAT wrap
    // through the toroidal texture
    const base = vec2(this.uBase as unknown as NV2);
    const cx = clamp(wp.x.sub(this.anchorX).div(SPACING), base.x, base.x.add(PROBE_XZ - 1));
    const cz = clamp(wp.z.sub(this.anchorZ).div(SPACING), base.y, base.y.add(PROBE_XZ - 1));
    const uvw = vec3(
      cx.add(0.5).div(PROBE_XZ),
      cz.add(0.5).div(PROBE_XZ),
      li.add(0.5).div(PROBE_LAYERS),
    );
    const R = texture3D(this.texR, uvw, 0);
    const G = texture3D(this.texG, uvw, 0);
    const B = texture3D(this.texB, uvw, 0);
    // L1 irradiance: E(n) ≈ c0·a0 + (c1·n)·a1   (constants folded/eye-calibrated)
    const a0 = 0.6;
    const a1 = 0.7;
    const e = vec3(
      R.x.mul(a0).add(dot(R.yzw, n).mul(a1)),
      G.x.mul(a0).add(dot(G.yzw, n).mul(a1)),
      B.x.mul(a0).add(dot(B.yzw, n).mul(a1)),
    );
    return max(e, vec3(0));
  }
}

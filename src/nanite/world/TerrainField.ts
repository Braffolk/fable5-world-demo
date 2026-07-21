/**
 * TerrainField — the streamed terrain field set (SPEC-STREAMING-WORLD §3, S5).
 * The scene handle threaded where the one-shot Heightfield went: a camera-window
 * plane pyramid whose texels arrive EXCLUSIVELY as StreamBrain fill packets
 * through the main-thread mailbox (the universal path both the generated world
 * and Estonia ride) — this class allocates the planes from a placement PLAN
 * (PlaneFill.planField — the same pure math the brain plans against) and
 * applies fills/origin commits; it never fetches.
 *
 *  - HEIGHT plane pyramid — r32float, level k texel = base·4^k, res per level
 *    by the cover-or-window rule. A level whose window covers the layer span is
 *    PINNED (plan.wraps=false — every generated level): its samplers compile
 *    the exact pre-S5 form, bit-identical shaders. A wrapping level (Estonia's
 *    near rungs) samples toroidally via its phase uniform and scrolls by brain
 *    packets: fills land first, the origin commit re-points uOrigin/uPhase
 *    after (promote-after-fill is packet order — F-8).
 *  - BIOME/CANOPY + SURFACE-FIELDS planes — rgba8 (filtered), same rule.
 *  - WATER planes — waterY (r32float) + ×8 min-reduced far level.
 *  - CPU mirrors — the DataTexture backing stores double as the mirrors (zero
 *    extra RAM) and stay coherent under partial fills, so the ?profile=1
 *    device swap re-uploads correct state and heightAt/waterAt serve walk
 *    probe, spawn and bookmarks at any moment.
 *
 * Lattice honesty (A11) lives in PlaneFill.placeLevel: plane texel p IS source
 * lattice sample n0+p — L0 fills are bit-identical to the source raster.
 */

import { DataTexture, FloatType, LinearFilter, NearestFilter, RGBAFormat, RedFormat, UnsignedByteType, Vector2 } from 'three';
import { If, clamp, float, floor, fract, mix, texture, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { texLoadR } from '../Tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import type { CoverageBox, FieldPlan, PlanePlan } from './PlaneFill';
import type { LevelGridEdit } from './PartitionTree';
import type { PlaneKind } from './StreamProtocol';
import { availabilityMorphWeight, cameraMorphWeight, MICRO_MORPH_BANDS } from './TerrainMorph';

export type { CoverageBox } from './PlaneFill';

/** Format-2 Estonia with the two near-only physical height rungs plans 195.6 MB;
 *  keep finite fail-loud headroom above that measured set. The generated world
 *  remains ~130 MB, and an accidental unbounded coverage plan still trips. */
export const TERRAIN_FIELD_VRAM_CEILING_MB = 256;

export interface FieldLevel {
  /** data LOD — texel = layer base texel · lodStep^lod */
  lod: number;
  res: number;
  texel: number;
  /** world coord of texel (0,0)'s SAMPLE point (texel-centered source lattice) */
  originX: number;
  originZ: number;
  /** origin as a uniform — scroll commits re-point it without shader rebuilds */
  uOrigin: { value: Vector2 };
  /** S6c PRECISION: origin relative to the render anchor (= uOrigin − uAnchor),
   *  recomputed CPU-side (f64) whenever the origin or anchor moves. gridCoords
   *  subtracts uAnchor from the (absolute) sample coord and uOriginRel here so the
   *  texel-index math is small−small on Estonia (~311 km absolute ⇒ f32 ULP 3 cm
   *  ⇒ the terrain height/normal sampling STAIRCASED into slope terraces). Anchor
   *  stays (0,0) on the generated world ⇒ uOriginRel ≡ uOrigin and the result is
   *  IEEE-identical to `wxz − uOrigin` (x−0 == x). */
  uOriginRel: { value: Vector2 };
  /** the shared render anchor (near-camera snapped world XZ; (0,0) = disabled) */
  uAnchor: { value: Vector2 };
  /** toroidal phase (n0 mod res) — only wrapping levels compile reads of it */
  uPhase: { value: Vector2 };
  /** Published sample bounds expressed in the current logical window grid. */
  uCoverageMin: { value: Vector2 };
  uCoverageMax: { value: Vector2 };
  /** Immutable published sample bounds in world coordinates (CPU sampler). */
  coverageMinX: number;
  coverageMinZ: number;
  coverageMaxX: number;
  coverageMaxZ: number;
  /** frozen at plan time: window < coverage ⇒ the level scrolls toroidally.
   *  Pinned levels (all generated ones) compile the exact pre-S5 samplers. */
  wraps: boolean;
  tex: DataTexture;
}

interface HeightLevel extends FieldLevel {
  data: Float32Array;
}

export class TerrainField {
  /** True only for format-2 fields carrying packed negative height levels. */
  readonly cookedMicroHeight: boolean;
  /** finest → coarsest; index = the `level` arg of fieldHeight */
  readonly heightLevels: readonly HeightLevel[];
  /** rgba8 [classId, vegDensity, canopyHeight, cover] */
  readonly biomeLevels: readonly FieldLevel[];
  /** rgba8 [moisture, flow, snow, rockExposure] */
  readonly fieldsLevels: readonly FieldLevel[];
  /** waterY at ~sim-res texel (r32float) — null when the source has no water */
  readonly water: HeightLevel | null;
  /** ×8 min-reduced far waterY (conservative: channels vanish, lakes survive) */
  readonly waterFar: HeightLevel | null;
  /** #114 anti-aliased water coverage α (rgba8, α×255 in channel 0) — a camera
   *  window on water's 2 m lattice. null when the source has no watercover layer
   *  (the generated world), which drives hasWaterCoverage and the WaterMaterial gate. */
  readonly waterCover: FieldLevel | null;
  /** #115 ×8 mean-reduced far coverage α (rgba8, α in channel 0) — the FAR water
   *  levels gate their shore on this instead of the min-reduced bed dive. null with
   *  waterCover (⇒ the far coverage path stays a NO-OP on the generated world). */
  readonly waterCoverFar: FieldLevel | null;
  /** true iff a watercover plane exists — the flag/absence that keeps the shoreline
   *  coverage path a NO-OP on the generated world (the #108 biomeCarriesCanopy pattern). */
  readonly hasWaterCoverage: boolean;
  /** #116 soil pedology (rgba8 [texCore, stoniness, boniteet, texSkeleton]) — a pilot-
   *  only (LOD0) camera window on the 2 m soil lattice. null when the source has no soil
   *  layer (the generated world), which drives hasSoil and the TerrainMaterial gate. */
  readonly soil: FieldLevel | null;
  /** true iff a soil plane exists — the flag/absence that keeps the soil-modulation path
   *  a NO-OP on the generated world (the #108 biomeCarriesCanopy / hasWaterCoverage pattern). */
  readonly hasSoil: boolean;
  /** Optional categorical geology [bedrock, surficial, process, coverage flags]. */
  readonly geology: FieldLevel | null;
  readonly hasGeology: boolean;
  /** Cook-side ground-cover control. A is categorical
   *  [typeA,typeB,clumpLo,clumpHi], B continuous
   *  [blend,vigor,moisture,canopyProximity]. */
  readonly groundCoverA: FieldLevel | null;
  readonly groundCoverB: FieldLevel | null;
  readonly groundCoverC: FieldLevel | null;
  readonly hasGroundCover: boolean;
  readonly hasGroundCoverClosure: boolean;
  readonly coverageBox: CoverageBox;
  /** biome plane channels 2/3 carry the merged far-forest canopy (heightM, cover)
   *  — true iff the source has a canopy layer. The generated world packs snow/
   *  rockExposure there instead, so its resolve canopy tint stays off (bit-identical). */
  readonly biomeCarriesCanopy: boolean;
  private morphCenterX = 0;
  private morphCenterZ = 0;
  /** Morph centre relative to the same precision anchor used by field levels. */
  private readonly uMorphCenterRel = uniform(new Vector2(0, 0));

  private constructor(
    heightLevels: HeightLevel[],
    biomeLevels: FieldLevel[],
    fieldsLevels: FieldLevel[],
    water: HeightLevel | null,
    waterFar: HeightLevel | null,
    waterCover: FieldLevel | null,
    waterCoverFar: FieldLevel | null,
    soil: FieldLevel | null,
    geology: FieldLevel | null,
    groundCoverA: FieldLevel | null,
    groundCoverB: FieldLevel | null,
    groundCoverC: FieldLevel | null,
    coverageBox: CoverageBox,
    biomeCarriesCanopy: boolean,
    cookedMicroHeight: boolean,
  ) {
    if (heightLevels.length === 0) throw new Error('TerrainField: needs at least one height level');
    this.heightLevels = heightLevels;
    this.biomeLevels = biomeLevels;
    this.fieldsLevels = fieldsLevels;
    this.water = water;
    this.waterFar = waterFar;
    this.waterCover = waterCover;
    this.waterCoverFar = waterCoverFar;
    this.hasWaterCoverage = waterCover !== null;
    this.soil = soil;
    this.hasSoil = soil !== null;
    this.geology = geology;
    this.hasGeology = geology !== null;
    this.groundCoverA = groundCoverA;
    this.groundCoverB = groundCoverB;
    this.groundCoverC = groundCoverC;
    this.hasGroundCover = groundCoverA !== null && groundCoverB !== null;
    this.hasGroundCoverClosure = this.hasGroundCover && groundCoverC !== null;
    this.coverageBox = coverageBox;
    this.biomeCarriesCanopy = biomeCarriesCanopy;
    this.cookedMicroHeight = cookedMicroHeight;
    const finest = heightLevels[0] as HeightLevel;
    this.setSurfaceCenter(
      finest.originX + ((finest.res - 1) * finest.texel) / 2,
      finest.originZ + ((finest.res - 1) * finest.texel) / 2,
    );
    const mb = this.vramBytes() / 2 ** 20;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] terrain field: height [${heightLevels.map((l) => `${l.res}²@${l.texel}m${l.wraps ? '~' : ''}`).join(' ')}] r32f + ` +
        `biome [${biomeLevels.map((l) => `${l.res}²`).join(' ')}] + fields [${fieldsLevels.map((l) => `${l.res}²`).join(' ')}] rgba8 + ` +
        `water ${water ? `${water.res}² r32f (+far ${waterFar?.res ?? 0}²)` : 'none'} + ` +
        `watercover ${waterCover ? `${waterCover.res}² rgba8${waterCover.wraps ? '~' : ''} (+far ${waterCoverFar?.res ?? 0}²)` : 'none'} + ` +
        `soil ${soil ? `${soil.res}² rgba8${soil.wraps ? '~' : ''}` : 'none'} + ` +
        `geology ${geology ? `${geology.res}² rgba8${geology.wraps ? '~' : ''}` : 'none'} + ` +
        `groundcover ${groundCoverA && groundCoverB ? `${groundCoverC ? 3 : 2}×${groundCoverA.res}² rgba8${groundCoverA.wraps ? '~' : ''}` : 'none'} = ` +
        `${mb.toFixed(1)} MB VRAM (CPU mirrors share the backing; ~ = camera-window level)`,
    );
    if (mb > TERRAIN_FIELD_VRAM_CEILING_MB) {
      throw new Error(`TerrainField: ${mb.toFixed(1)} MB exceeds the ${TERRAIN_FIELD_VRAM_CEILING_MB} MB ceiling`);
    }
  }

  /** Allocate every plane from the placement plan (PlaneFill.planField) —
   *  texels arrive as StreamBrain fill packets (the S5 streaming path). */
  static fromPlan(plan: FieldPlan): TerrainField {
    const heightLevels = plan.height.map((p) => makeHeightLevel(`terrainFieldHeightL${p.lod}`, p));
    const biomeLevels = plan.biome.map((p) => makeU8Level(`terrainFieldBiomeL${p.lod}`, p));
    const fieldsLevels = plan.fields.map((p) => makeU8Level(`terrainFieldFieldsL${p.lod}`, p));
    const water = plan.water ? makeHeightLevel('terrainFieldWaterY', plan.water) : null;
    const waterFar = plan.waterFar ? makeHeightLevel('terrainFieldWaterYFar', plan.waterFar) : null;
    const waterCover = plan.waterCover ? makeU8Level('terrainFieldWaterCover', plan.waterCover) : null;
    const waterCoverFar = plan.waterCoverFar ? makeU8Level('terrainFieldWaterCoverFar', plan.waterCoverFar) : null;
    const soil = plan.soil ? makeU8Level('terrainFieldSoil', plan.soil) : null;
    const geology = plan.geology ? makeU8Level('terrainFieldGeology', plan.geology) : null;
    const groundCoverA = plan.groundCoverA ? makeU8Level('terrainFieldGroundCoverA', plan.groundCoverA) : null;
    const groundCoverB = plan.groundCoverB ? makeU8Level('terrainFieldGroundCoverB', plan.groundCoverB) : null;
    const groundCoverC = plan.groundCoverC ? makeU8Level('terrainFieldGroundCoverC', plan.groundCoverC) : null;
    return new TerrainField(
      heightLevels,
      biomeLevels,
      fieldsLevels,
      water,
      waterFar,
      waterCover,
      waterCoverFar,
      soil,
      geology,
      groundCoverA,
      groundCoverB,
      groundCoverC,
      plan.coverageBox,
      plan.biomeHasCanopy,
      plan.cookedMicroHeight,
    );
  }

  /** One-level field over a flat/explicit height array — forest/gallery-class
   *  scenes whose world is not source-streamed (F11: the fake-hf hack's grave). */
  static singleLevel(opts: {
    res: number;
    texel: number;
    /** world min corner of the covered square */
    worldMinX: number;
    worldMinZ: number;
    /** res² heights; omit for a flat-0 world */
    heights?: Float32Array;
  }): TerrainField {
    const data = opts.heights ?? new Float32Array(opts.res * opts.res);
    if (data.length !== opts.res * opts.res) throw new Error('TerrainField.singleLevel: heights length != res²');
    const plan: PlanePlan = {
      lod: 0,
      res: opts.res,
      wraps: false,
      originX: opts.worldMinX + opts.texel * 0.5,
      originZ: opts.worldMinZ + opts.texel * 0.5,
      texel: opts.texel,
      stride: 1,
      n0x: 0,
      n0z: 0,
      nMinX: 0,
      nMaxX: opts.res - 1,
      nMinZ: 0,
      nMaxZ: opts.res - 1,
    };
    const lvl = makeHeightLevel('terrainFieldHeightL0', plan, data);
    return new TerrainField(
      [lvl],
      [],
      [],
      null,
      null,
      null,
      null,
      null, // no soil plane on a single-level field
      null, // no geology plane on a single-level field
      null, // no cook-side ground-cover plane on a single-level field
      null, // no second ground-cover carrier
      null, // no v2 closure/profile carrier
      {
        minX: opts.worldMinX,
        minZ: opts.worldMinZ,
        maxX: opts.worldMinX + opts.res * opts.texel,
        maxZ: opts.worldMinZ + opts.res * opts.texel,
      },
      false, // no biome/canopy planes on a single-level field
      false, // no packed negative height levels
    );
  }

  // ---- stream mailbox application (the ONLY mutation surface) ----------------------

  /** the level record a fill/origin packet targets. */
  levelFor(plane: PlaneKind, level: number): FieldLevel {
    const lvl =
      plane === 'height'
        ? this.heightLevels[level]
        : plane === 'biome'
          ? this.biomeLevels[level]
          : plane === 'fields'
            ? this.fieldsLevels[level]
            : plane === 'water'
              ? this.water
              : plane === 'waterFar'
                ? this.waterFar
                : plane === 'watercover'
                  ? this.waterCover
                  : plane === 'waterCoverFar'
                  ? this.waterCoverFar
                    : plane === 'soil'
                      ? this.soil
                      : plane === 'geology'
                        ? this.geology
                        : plane === 'groundcoverA'
                          ? this.groundCoverA
                          : plane === 'groundcoverB'
                            ? this.groundCoverB
                            : this.groundCoverC;
    if (!lvl) throw new Error(`TerrainField: no ${plane} level ${level}`);
    return lvl;
  }

  /** write a physical-rect fill into the CPU backing (the mirror). The GPU
   *  upload is the mailbox's job (writeTexture, or a full needsUpdate flip at
   *  boot) — backing-first keeps the ?profile device swap coherent. */
  applyFill(plane: PlaneKind, level: number, x: number, y: number, w: number, h: number, data: Float32Array | Uint8Array): void {
    const lvl = this.levelFor(plane, level);
    const backing = (lvl.tex.image as { data: Float32Array | Uint8Array }).data;
    const ch = backing instanceof Float32Array ? 1 : 4;
    if (data.length !== w * h * ch) throw new Error(`TerrainField: fill ${plane}L${level} ${w}×${h} data length mismatch`);
    for (let row = 0; row < h; row++) {
      backing.set(data.subarray(row * w * ch, (row + 1) * w * ch), ((y + row) * lvl.res + x) * ch);
    }
  }

  /** S6f SURFACE AUTHORITY (§7): the fringe-level grid, one entry per coarse root
   *  cell, updated ATOMICALLY with the tile refine/merge that changed it — so a
   *  height consumer clamped to it can never disagree with which LOD renders. Kept
   *  coherent here (CPU store, keyed by packed root-cell coord); the GPU grass
   *  sampler that reads it is the remaining wire-up (see the S6f report). */
  private readonly fringeGrid = new Map<number, number>();
  applyLevelGrid(edits: readonly LevelGridEdit[]): void {
    for (const e of edits) this.fringeGrid.set(((e.cellX & 0xffff) << 16) | (e.cellZ & 0xffff), e.level);
  }
  /** the fringe level published for a root cell (levels-1 = coarsest, if unset). */
  fringeLevelForCell(cellX: number, cellZ: number, fallback: number): number {
    return this.fringeGrid.get(((cellX & 0xffff) << 16) | (cellZ & 0xffff)) ?? fallback;
  }

  /** re-point a wrapping level after its fills landed (F-8 packet order). */
  commitOrigin(plane: PlaneKind, level: number, originX: number, originZ: number, phaseX: number, phaseY: number): void {
    const lvl = this.levelFor(plane, level);
    if (!lvl.wraps) throw new Error(`TerrainField: origin commit on pinned plane ${plane}L${level}`);
    lvl.originX = originX;
    lvl.originZ = originZ;
    lvl.uOrigin.value.set(originX, originZ);
    // S6c: keep the anchor-relative origin coherent with the new window origin.
    lvl.uOriginRel.value.set(originX - lvl.uAnchor.value.x, originZ - lvl.uAnchor.value.y);
    lvl.uCoverageMin.value.set(
      (lvl.coverageMinX - originX) / lvl.texel,
      (lvl.coverageMinZ - originZ) / lvl.texel,
    );
    lvl.uCoverageMax.value.set(
      (lvl.coverageMaxX - originX) / lvl.texel,
      (lvl.coverageMaxZ - originZ) / lvl.texel,
    );
    lvl.uPhase.value.set(phaseX, phaseY);
  }

  /** S6c PRECISION — set the shared render anchor (near-camera snapped world XZ)
   *  used by every GPU field sampler to keep gridCoords sub-metre on Estonia's
   *  ~311 km absolute coords. Call per frame with the (snapped) camera XZ on the
   *  streamed world; leave at (0,0) on the generated world (< 3 km ⇒ f32 exact,
   *  and (0,0) makes every sampler IEEE-identical to the pre-S6c path). Only the
   *  precision changes — gridCoords' value is anchor-invariant — so the anchor may
   *  move any amount between frames with zero visual discontinuity. */
  setRenderAnchor(ax: number, az: number): void {
    for (const lvl of this.allLevels()) {
      lvl.uAnchor.value.set(ax, az);
      lvl.uOriginRel.value.set(lvl.originX - ax, lvl.originZ - az);
    }
    this.uMorphCenterRel.value.set(this.morphCenterX - ax, this.morphCenterZ - az);
  }

  /** Continuous camera centre for packed-level geomorphing. This is deliberately
   *  independent of the 512 m precision-anchor snap and plane scroll cadence. */
  setSurfaceCenter(x: number, z: number): void {
    this.morphCenterX = x;
    this.morphCenterZ = z;
    const anchor = this.heightLevels[0]?.uAnchor.value ?? new Vector2(0, 0);
    this.uMorphCenterRel.value.set(x - anchor.x, z - anchor.y);
  }

  /** flip every plane's full backing to the GPU (boot: after the unbudgeted
   *  mailbox drain filled the mirrors). */
  markAllDirty(): void {
    for (const lvl of this.allLevels()) lvl.tex.needsUpdate = true;
  }

  private allLevels(): FieldLevel[] {
    return [
      ...this.heightLevels,
      ...this.biomeLevels,
      ...this.fieldsLevels,
      ...(this.water ? [this.water] : []),
      ...(this.waterFar ? [this.waterFar] : []),
      ...(this.waterCover ? [this.waterCover] : []),
      ...(this.waterCoverFar ? [this.waterCoverFar] : []),
      ...(this.soil ? [this.soil] : []),
      ...(this.geology ? [this.geology] : []),
      ...(this.groundCoverA ? [this.groundCoverA] : []),
      ...(this.groundCoverB ? [this.groundCoverB] : []),
      ...(this.groundCoverC ? [this.groundCoverC] : []),
    ];
  }

  // ---- CPU sampling (walk probe, spawn, bookmarks) --------------------------------

  /** bilinear height (m) — finest level whose window contains the point */
  heightAt(x: number, z: number): number {
    if (this.cookedMicroHeight) return this.microHeightAtCpu(x, z);
    const L = this.heightLevels;
    for (let i = 0; i < L.length - 1; i++) {
      const lvl = L[i] as HeightLevel;
      const gx = (x - lvl.originX) / lvl.texel;
      const gz = (z - lvl.originZ) / lvl.texel;
      // 1-texel margin: the plane rim is clamp-extended fill (or the scroll
      // rim), the coarser level holds the real slope there
      if (gx >= 1 && gx <= lvl.res - 2 && gz >= 1 && gz <= lvl.res - 2) {
        return bilerpCpu(lvl, gx, gz);
      }
    }
    const lvl = L[L.length - 1] as HeightLevel;
    return bilerpCpu(lvl, (x - lvl.originX) / lvl.texel, (z - lvl.originZ) / lvl.texel);
  }

  /** bilinear waterY (m); dry cells sit well below the bed (sentinel/bed−2), so
   *  max(ground, waterAt + ε) stays a safe camera floor. −1e4 without water. */
  waterAt(x: number, z: number): number {
    const w = this.water;
    if (!w) return -1e4;
    return bilerpCpu(w, (x - w.originX) / w.texel, (z - w.originZ) / w.texel);
  }

  // ---- TSL sampling ----------------------------------------------------------------

  /** bilinear height at a KNOWN level — the hoisted form for draw classes whose
   *  band is known per draw/tile (hot S3b consumers hoist the level per tile) */
  fieldHeight(wxz: NV2, level = 0): NF {
    return planeBilerp(this.heightLevels[level] as HeightLevel, wxz);
  }

  /** nearest-texel height at a known level (cost-insensitive probe taps) */
  fieldHeightNearest(wxz: NV2, level = 0): NF {
    return planeNearest(this.heightLevels[level] as HeightLevel, wxz);
  }

  /** Bilinear height, finest resident level selected per sample. This is the
   *  cold/expression form: it is legal while constructing a material node
   *  outside a TSL Fn stack. Format-2 eagerly builds each possible texture tap
   *  and combines them with select/mix expressions; hot kernels should use
   *  fieldHeightFinestHot so inactive packed levels are not sampled. */
  fieldHeightFinest(wxz: NV2): NF {
    if (this.cookedMicroHeight) return this.microHeightSelect(wxz, false);
    return this.finestSelect(wxz, planeBilerp);
  }

  /** Cold/expression counterpart of fieldHeightFinestNearestHot. */
  fieldHeightFinestNearest(wxz: NV2): NF {
    if (this.cookedMicroHeight) return this.microHeightSelect(wxz, true);
    return this.finestSelect(wxz, planeNearest);
  }

  /** Fn-stack-only bilinear sampler. Exactly one containing authority level is
   *  read, and packed child reads stay behind their morph branches. */
  fieldHeightFinestHot(wxz: NV2): NF {
    if (this.cookedMicroHeight) return this.microHeightHot(wxz, false);
    // Preserve the exact generated/format-1 expression tree. Only format-2
    // needs the branch form to avoid eagerly sampling its packed child rungs.
    return this.finestSelect(wxz, planeBilerp);
  }

  /** Fn-stack-only nearest sampler. Packed parent/authority reads remain
   *  bilinear while morphing so the stored parent-child continuity is kept. */
  fieldHeightFinestNearestHot(wxz: NV2): NF {
    if (this.cookedMicroHeight) return this.microHeightHot(wxz, true);
    return this.finestSelect(wxz, planeNearest);
  }

  /** bilinear waterY (near consumers: caustics, water gates) */
  fieldWaterY(wxz: NV2): NF {
    const w = this.water;
    if (!w) throw new Error('TerrainField: no water plane');
    return planeBilerp(w, wxz);
  }

  /** bilinear waterY from the min-reduced far level (distant clipmap rings) */
  fieldWaterYFar(wxz: NV2): NF {
    const w = this.waterFar;
    if (!w) throw new Error('TerrainField: no far water plane');
    return planeBilerp(w, wxz);
  }

  /** #GAP wet-preferring bilinear waterY for the NEAR Estonia surface — the near
   *  analog of PlaneFill.maxReduce (the #115 far surface). Estonia's dry cells hold
   *  the −1e4 sentinel; a plain bilinear near a shore blends the ~40 m water level
   *  with −1e4 and PLUNGES the surface into a pit (the shore GAP). This masks the
   *  sentinel corners out of the 2×2 and re-normalizes over the WET corners, so the
   *  surface stays FLAT at the true water level right up to the last wet texel
   *  (dilating the wet surface outward, like maxReduce) instead of diving. Falls
   *  back to the plain bilerp (≈ sentinel) only where all four corners are dry — the
   *  caller clamps that degenerate vertex to the bed. Only compiled behind
   *  field.hasWaterCoverage (the generated world never builds it). */
  fieldWaterYWet(wxz: NV2): NF {
    const w = this.water;
    if (!w) throw new Error('TerrainField: no water plane');
    return planeBilerpWet(w, wxz);
  }

  /** #GAP bilinear WET fraction ∈ [0,1] of the 2 m water plane (bilerp of the
   *  binary above-sentinel mask): 1 inside the wet lattice, a continuous one-texel
   *  ramp to 0 across the dilation band where fieldWaterYWet extends the flat pool
   *  surface — the fineShore analog of fieldWaterCoverage's α feather. NaniteResolve
   *  fades the waterline fringe with it so the wet-preferring surface's all-dry
   *  sentinel cut never shows as 2 m stair-steps on the banks. */
  fieldWaterWetFrac(wxz: NV2): NF {
    const w = this.water;
    if (!w) throw new Error('TerrainField: no water plane');
    return planeWetFrac(w, wxz);
  }

  /** #114 bilinear water coverage α ∈ [0,1] (channel 0 of the watercover rgba8
   *  plane). One filtered tap of the ANTI-ALIASED fraction — the sub-texel signal
   *  a bilinear of the BINARY water mask could never resolve, so the shore tracks
   *  α's 0.5 iso-contour instead of quantizing to 2 m grid squares. Only compiled
   *  behind field.hasWaterCoverage (WaterMaterial's near-level path). */
  fieldWaterCoverage(wxz: NV2): NF {
    const wc = this.waterCover;
    if (!wc) throw new Error('TerrainField: no water coverage plane');
    return planeLinear(wc, wxz).x as unknown as NF;
  }

  /** #115 bilinear far water coverage α ∈ [0,1] (channel 0 of the ×8 mean-reduced
   *  waterCoverFar plane) — the FAR-level shore gate. One filtered tap of the far-
   *  scale coverage fraction; its 0.5 iso-contour tracks the shoreline at the 16 m
   *  far texel scale, so the far shore stops quantizing to blocky squares. Only
   *  compiled behind field.hasWaterCoverage (WaterMaterial's far-level path). */
  fieldWaterCoverageFar(wxz: NV2): NF {
    const wc = this.waterCoverFar;
    if (!wc) throw new Error('TerrainField: no far water coverage plane');
    return planeLinear(wc, wxz).x as unknown as NF;
  }

  // ---- HOT samplers (S3b) — Fn-stack only (If/ElseIf level chains: exactly ONE
  // arm executes, and the containment conditions are near-uniform per cluster).
  // The cold material-build-time samplers above use select() chains instead.

  /** height for hot per-vertex paths: finest containing level via a BRANCH
   *  chain. Level 0 reads the exact texel — terrain verts sit ON the finest
   *  lattice, so this is bit-identical to the retired global heightTex tap —
   *  and coarser levels bilerp (their lattice is coarser than the verts'). */
  fieldHeightHot(wxz: NV2): NF {
    if (this.cookedMicroHeight) return this.microHeightHot(wxz, true);
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl, finest) => {
      out.assign(finest ? planeNearest(lvl, wxz) : planeBilerp(lvl, wxz));
    });
    return out as unknown as NF;
  }

  /** One-level hoisted slope, or the legacy alias of the Fn-stack-only hot
   *  selector. Every unhoisted caller is explicitly migrated to fieldSlopeHot. */
  fieldSlope(wxz: NV2, level?: number): NF {
    if (level !== undefined) return slope4(this.heightLevels[level] as HeightLevel, wxz);
    return this.fieldSlopeHot(wxz);
  }

  /** Fn-stack-only central-difference slope (rise/run) from the height planes — the in-shader
   *  replacement for the retired normalTex.w: the SAME ±1-texel stencil the old
   *  bake ran (Heightfield derived-maps kernel), evaluated at the nearest texel
   *  of the finest containing level. */
  fieldSlopeHot(wxz: NV2): NF {
    if (this.cookedMicroHeight) {
      const hg = this.microHeightGradientHot(wxz);
      return vec2(hg.y, hg.z).length() as unknown as NF;
    }
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => {
      out.assign(slope4(lvl, wxz));
    });
    return out as unknown as NF;
  }

  /** Legacy alias; normal/slope selection is Fn-stack-only because preserving
   *  exact parent derivatives at a zero morph endpoint requires branching. */
  fieldNormalSlope(wxz: NV2): NV4 {
    return this.fieldNormalSlopeHot(wxz);
  }

  /** Fn-stack-only central-difference world normal (xyz) + slope (w) — the retired normalTex's
   *  EXACT bake stencil (n = normalize(hl−hr, 2·texel, hd−hu); slope = |∇h|/2texel)
   *  evaluated in-shader at the finest containing level (S3b resolve). */
  fieldNormalSlopeHot(wxz: NV2): NV4 {
    if (this.cookedMicroHeight) {
      const hg = this.microHeightGradientHot(wxz);
      const slope = vec2(hg.y, hg.z).length();
      return vec4(vec3(hg.y.negate(), 1, hg.z.negate()).normalize(), slope) as unknown as NV4;
    }
    const out = vec4(0, 1, 0, 0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => {
      out.assign(normalSlope4(lvl, wxz));
    });
    return out as unknown as NV4;
  }

  /** Fn-stack-only CLASS slope (rise/run) — the material-SELECTION slope for
   *  fine-lattice cooks. A micro height source (texel < 0.5 m) carries
   *  centimetre texel-to-texel relief whose nearest-texel CD slope is
   *  piecewise-CONSTANT per texel and statistical noise against the material
   *  class windows (measured 2026-07-16: p50 adjacent-texel slope jump 0.27 ≈
   *  the entire grassW window; 1 cm height quanta step slope by 0.08) — so the
   *  material BLEND field updated in texel-sized blocks over full-resolution
   *  carriers. Classes read the first ≥0.5 m level through the C0 smooth
   *  gradient instead: selection becomes continuous at the scale its windows
   *  were tuned on. */
  fieldClassSlopeHot(wxz: NV2): NF {
    const first = this.heightLevels.findIndex((l) => l.texel >= 0.5);
    const levels = first < 0 ? this.heightLevels.slice(-1) : this.heightLevels.slice(first);
    const out = float(0).toVar();
    hotLevelChain(levels, wxz, (lvl) => {
      const g = planeGradientSmooth(lvl as HeightLevel, wxz);
      out.assign(g.length());
    });
    return out as unknown as NF;
  }

  /** Fn-stack-only fine-relief slope: C0-continuous gradient magnitude at the
   *  finest level with texel ≥ 0.2 m (the 4×4-reduced rung — measured adjacent
   *  jump p50 0.067 vs 0.27 at the raw micro texel, so the field varies at
   *  ~25 cm+ form scale instead of echoing per-texel synth noise as 6–12 cm
   *  blobs). Drives sub-metre material exposure that traces actual micro-forms
   *  without printing any lattice. */
  fieldReliefSlopeHot(wxz: NV2): NF {
    const first = this.heightLevels.findIndex((l) => l.texel >= 0.2);
    const levels = first < 0 ? this.heightLevels.slice(-1) : this.heightLevels.slice(first);
    const out = float(0).toVar();
    hotLevelChain(levels, wxz, (lvl) => {
      const g = planeGradientSmooth(lvl as HeightLevel, wxz);
      out.assign(g.length());
    });
    return out as unknown as NF;
  }

  /** Fn-stack-only C0 world normal (xyz) + slope (w) from planeGradientSmooth at
   *  the finest containing level — the smooth counterpart of fieldNormalSlopeHot
   *  for fine-lattice cooks, where the nearest-texel CD stencil's per-texel
   *  CONSTANT normal renders as faint 6–12 cm facet tiles under the sun term. */
  fieldNormalSlopeSmoothHot(wxz: NV2): NV4 {
    const out = vec4(0, 1, 0, 0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => {
      const g = planeGradientSmooth(lvl as HeightLevel, wxz);
      out.assign(
        vec4(vec3(g.x.negate(), 1, g.y.negate()).normalize(), g.length()),
      );
    });
    return out as unknown as NV4;
  }

  /** surface-fields sample [moisture, flowStrength, snow, rockExposure] — one
   *  hardware-filtered rgba8 tap at the finest containing level. vec4(0) when
   *  the source has no fields layer (forest/gallery single-level fields). */
  fieldsAt(wxz: NV2): NV4 {
    if (this.fieldsLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.fieldsLevels, wxz, (lvl) => {
      out.assign(planeLinear(lvl, wxz));
    });
    return out as unknown as NV4;
  }

  /** biome/canopy sample [classId (raw id byte ⇒ ×255 to decode), vegDensity,
   *  canopyHeight, cover] — one filtered rgba8 tap, finest containing level.
   *  vec4(0) when the source has no biome layer. */
  biomeAt(wxz: NV2): NV4 {
    if (this.biomeLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.biomeLevels, wxz, (lvl) => {
      out.assign(planeLinear(lvl, wxz));
    });
    return out as unknown as NV4;
  }

  /** Discrete biome/land-cover class at the nearest source texel. The remaining
   *  biome channels stay linearly filtered through biomeAt(); class ids must not
   *  interpolate through unrelated numeric categories at polygon boundaries. */
  biomeClassAt(wxz: NV2): NF {
    if (this.biomeLevels.length === 0) return float(0) as unknown as NF;
    const out = float(0).toVar();
    hotLevelChain(this.biomeLevels, wxz, (lvl) => {
      out.assign(planeNearest(lvl, wxz));
    });
    return out as unknown as NF;
  }

  /** #116 soil sample [texCore, stoniness, boniteet, texSkeleton] (raw byte /255 — the
   *  material decodes per channel: ×255 for the id/score channels). ONE filtered rgba8
   *  tap of the single soil level (LOD0 only — soil is pilot-near); the level wraps on
   *  Estonia so this takes planeLinear's toroidal 4-tap path. vec4(0) when the source
   *  has no soil layer (the generated world — the caller compile-gates on hasSoil, so
   *  this branch is never constructed there). */
  soilAt(wxz: NV2): NV4 {
    if (!this.soil) return vec4(0) as unknown as NV4;
    return planeLinear(this.soil, wxz) as unknown as NV4;
  }

  /** Nearest categorical geology sample, normalized as raw bytes / 255.
   *  The caller decodes each channel with round(value * 255). A single explicit
   *  nearest tap prevents interpolation between unrelated polygon categories. */
  geologyAt(wxz: NV2): NV4 {
    if (!this.geology) return vec4(0) as unknown as NV4;
    return planeNearest4(this.geology, wxz);
  }

  /** Bilinear tap of the geology plane used ONLY as a boundary-confidence
   *  signal: |linear − nearest| per channel ≈ proximity to a category edge.
   *  Category IDENTITY always comes from geologyAt's nearest tap — this tap's
   *  mixed values are never decoded as ids (the biome classInterior idiom). */
  geologyLinearAt(wxz: NV2): NV4 {
    if (!this.geology) return vec4(0) as unknown as NV4;
    return planeLinear(this.geology, wxz) as unknown as NV4;
  }

  /** Nearest sample of ground-cover carrier A:
   *  [typeA, typeB, clumpLo, clumpHi]. All four channels are categorical. */
  groundCoverAt(wxz: NV2): NV4 {
    if (!this.groundCoverA) return vec4(0) as unknown as NV4;
    return planeNearest4(this.groundCoverA, wxz);
  }

  /** Filtered sample of ground-cover carrier B:
   *  [blend, vigor, moisture, canopyProximity]. */
  groundCoverLinearAt(wxz: NV2): NV4 {
    if (!this.groundCoverB) return vec4(0) as unknown as NV4;
    return planeLinear(this.groundCoverB, wxz) as unknown as NV4;
  }

  /** v2 categorical closure/profile carrier:
   * [candidateMaskLo, candidateMaskHi, profileA, profileB]. */
  groundCoverProfilesAt(wxz: NV2): NV4 {
    if (!this.groundCoverC) return vec4(0) as unknown as NV4;
    return planeNearest4(this.groundCoverC, wxz);
  }

  /** nearest-texel waterY — hot gates (raster riverDepth, grass water gate).
   *  −1e4 (the dry sentinel, far below any bed) when the source has no water. */
  fieldWaterYNearest(wxz: NV2): NF {
    const w = this.water;
    if (!w) return float(-1e4) as unknown as NF;
    return planeNearest(w, wxz);
  }

  /** a height level's plane texture — raster/debug paths that bind the texture
   *  directly (their in-shader reads migrate at S3b) */
  heightPlane(level = 0): DataTexture {
    return (this.heightLevels[level] as HeightLevel).tex;
  }

  vramBytes(): number {
    let b = 0;
    for (const l of this.allLevels()) b += l.res * l.res * 4;
    return b;
  }

  private finestSelect(wxz: NV2, sample: (lvl: HeightLevel, wxz: NV2) => NF): NF {
    const L = this.heightLevels;
    let h = sample(L[L.length - 1] as HeightLevel, wxz);
    for (let i = L.length - 2; i >= 0; i--) {
      const lvl = L[i] as HeightLevel;
      h = insideLevel(lvl, wxz).select(sample(lvl, wxz), h) as NF;
    }
    return h;
  }

  private microHeightAtCpu(x: number, z: number): number {
    const fine = this.heightLevels.find((level) => level.lod === -2);
    if (!fine) return this.microParentHeightAtCpu(x, z);
    const weight = this.microMorphWeightCpu(fine, x, z);
    if (weight <= 0) return this.microParentHeightAtCpu(x, z);
    const child = this.levelHeightAtCpu(fine, x, z);
    if (weight >= 1) return child;
    const parent = this.microParentHeightAtCpu(x, z);
    return parent + (child - parent) * weight;
  }

  private microParentHeightAtCpu(x: number, z: number): number {
    const parent = this.heightLevels.find((level) => level.lod === -1);
    if (!parent) return this.authorityHeightAtCpu(x, z);
    const weight = this.microMorphWeightCpu(parent, x, z);
    if (weight <= 0) return this.authorityHeightAtCpu(x, z);
    const child = this.levelHeightAtCpu(parent, x, z);
    if (weight >= 1) return child;
    const authority = this.authorityHeightAtCpu(x, z);
    return authority + (child - authority) * weight;
  }

  private authorityHeightAtCpu(x: number, z: number): number {
    const authority = this.heightLevels.findIndex((level) => level.lod >= 0);
    if (authority < 0) throw new Error('TerrainField: packed micro field lacks authority level');
    return this.heightAtCpuFrom(authority, x, z);
  }

  private levelHeightAtCpu(level: HeightLevel, x: number, z: number): number {
    return bilerpCpu(level, (x - level.originX) / level.texel, (z - level.originZ) / level.texel);
  }

  private microMorphWeightCpu(level: HeightLevel, x: number, z: number): number {
    const lod = level.lod as -2 | -1;
    return cameraMorphWeight(lod, x, z, this.morphCenterX, this.morphCenterZ)
      * availabilityMorphWeight(
        lod,
        x,
        z,
        level.originX,
        level.originZ,
        level.texel,
        level.res,
        level.coverageMinX,
        level.coverageMinZ,
        level.coverageMaxX,
        level.coverageMaxZ,
      );
  }

  private heightAtCpuFrom(first: number, x: number, z: number): number {
    const levels = this.heightLevels;
    for (let i = first; i < levels.length - 1; i++) {
      const level = levels[i] as HeightLevel;
      const gx = (x - level.originX) / level.texel;
      const gz = (z - level.originZ) / level.texel;
      if (gx >= 1 && gx <= level.res - 2 && gz >= 1 && gz <= level.res - 2) {
        return bilerpCpu(level, gx, gz);
      }
    }
    const level = levels[levels.length - 1] as HeightLevel;
    return bilerpCpu(level, (x - level.originX) / level.texel, (z - level.originZ) / level.texel);
  }

  /** Pure-expression packed geomorph. Safe outside a TSL Fn stack; unlike the
   *  hot twin this necessarily constructs taps for both sides of each mix. */
  private microHeightSelect(wxz: NV2, exactFineLattice: boolean): NF {
    const parent = this.microParentHeightSelect(wxz);
    const fine = this.heightLevels.find((level) => level.lod === -2);
    if (!fine) return parent;
    const child = exactFineLattice ? planeNearest(fine, wxz) : planeBilerp(fine, wxz);
    return mix(parent, child, this.microMorphWeightGpu(fine, wxz)) as unknown as NF;
  }

  private microParentHeightSelect(wxz: NV2): NF {
    const authority = this.authorityHeightSelect(wxz);
    const parent = this.heightLevels.find((level) => level.lod === -1);
    if (!parent) return authority;
    return mix(authority, planeBilerp(parent, wxz), this.microMorphWeightGpu(parent, wxz)) as unknown as NF;
  }

  private authorityHeightSelect(wxz: NV2): NF {
    const authority = this.heightLevels.findIndex((level) => level.lod >= 0);
    if (authority < 0) throw new Error('TerrainField: packed micro field lacks authority level');
    const levels = this.heightLevels.slice(authority);
    let h = planeBilerp(levels[levels.length - 1] as HeightLevel, wxz);
    for (let i = levels.length - 2; i >= 0; i--) {
      const level = levels[i] as HeightLevel;
      h = insideLevel(level, wxz).select(planeBilerp(level, wxz), h) as NF;
    }
    return h;
  }

  /** Statement/branch packed geomorph. Fn-stack only: inactive child levels
   *  are not sampled outside their camera/availability band. */
  private microHeightHot(wxz: NV2, exactFineLattice: boolean): NF {
    const fine = this.heightLevels.find((level) => level.lod === -2);
    if (!fine) return this.microParentHeightHot(wxz);
    const weight = this.microMorphWeightGpu(fine, wxz);
    const out = float(0).toVar();
    const branch = If(weight.greaterThanEqual(1), () => {
      out.assign(exactFineLattice ? planeNearest(fine, wxz) : planeBilerp(fine, wxz));
    });
    branch.ElseIf(weight.greaterThan(0), () => {
      const parent = this.microParentHeightHot(wxz);
      const child = exactFineLattice ? planeNearest(fine, wxz) : planeBilerp(fine, wxz);
      out.assign(mix(parent, child, weight));
    }).Else(() => {
      out.assign(this.microParentHeightHot(wxz));
    });
    return out as unknown as NF;
  }

  private microParentHeightHot(wxz: NV2): NF {
    const parent = this.heightLevels.find((level) => level.lod === -1);
    if (!parent) return this.authorityHeightHot(wxz);
    const weight = this.microMorphWeightGpu(parent, wxz);
    const out = float(0).toVar();
    const branch = If(weight.greaterThanEqual(1), () => {
      out.assign(planeBilerp(parent, wxz));
    });
    branch.ElseIf(weight.greaterThan(0), () => {
      out.assign(mix(this.authorityHeightHot(wxz), planeBilerp(parent, wxz), weight));
    }).Else(() => {
      out.assign(this.authorityHeightHot(wxz));
    });
    return out as unknown as NF;
  }

  private authorityHeightHot(wxz: NV2): NF {
    const authority = this.heightLevels.findIndex((level) => level.lod >= 0);
    if (authority < 0) throw new Error('TerrainField: packed micro field lacks authority level');
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels.slice(authority), wxz, (level) => {
      out.assign(planeBilerp(level as HeightLevel, wxz));
    });
    return out as unknown as NF;
  }

  /** vec3(height, dh/dx, dh/dz), including the derivative of both morph weights. */
  private microHeightGradientHot(wxz: NV2): NV3 {
    const fine = this.heightLevels.find((level) => level.lod === -2);
    if (!fine) return this.microParentHeightGradientHot(wxz);
    const weight = this.microMorphWeightGpu(fine, wxz);
    const out = vec3(0).toVar();
    const branch = If(weight.greaterThanEqual(1), () => {
      out.assign(planeHeightGradient(fine, wxz));
    });
    branch.ElseIf(weight.greaterThan(0), () => {
      out.assign(this.blendHeightGradient(
        this.microParentHeightGradientHot(wxz),
        planeHeightGradient(fine, wxz),
        fine,
        wxz,
        weight,
      ));
    }).Else(() => {
      out.assign(this.microParentHeightGradientHot(wxz));
    });
    return out as unknown as NV3;
  }

  private microParentHeightGradientHot(wxz: NV2): NV3 {
    const parent = this.heightLevels.find((level) => level.lod === -1);
    if (!parent) return this.authorityHeightGradientHot(wxz);
    const weight = this.microMorphWeightGpu(parent, wxz);
    const out = vec3(0).toVar();
    const branch = If(weight.greaterThanEqual(1), () => {
      out.assign(planeHeightGradient(parent, wxz));
    });
    branch.ElseIf(weight.greaterThan(0), () => {
      out.assign(this.blendHeightGradient(
        this.authorityHeightGradientHot(wxz),
        planeHeightGradient(parent, wxz),
        parent,
        wxz,
        weight,
      ));
    }).Else(() => {
      out.assign(this.authorityHeightGradientHot(wxz));
    });
    return out as unknown as NV3;
  }

  private authorityHeightGradientHot(wxz: NV2): NV3 {
    const authority = this.heightLevels.findIndex((level) => level.lod >= 0);
    if (authority < 0) throw new Error('TerrainField: packed micro field lacks authority level');
    const out = vec3(0).toVar();
    hotLevelChain(this.heightLevels.slice(authority), wxz, (level) => {
      out.assign(planeHeightGradient(level as HeightLevel, wxz));
    });
    return out as unknown as NV3;
  }

  private blendHeightGradient(
    parent: NV3,
    child: NV3,
    level: HeightLevel,
    wxz: NV2,
    weight: NF,
  ): NV3 {
    const eps = level.texel;
    const dwdx = this.microMorphWeightGpu(level, wxz.add(vec2(eps, 0)) as unknown as NV2)
      .sub(this.microMorphWeightGpu(level, wxz.sub(vec2(eps, 0)) as unknown as NV2))
      .div(2 * eps);
    const dwdz = this.microMorphWeightGpu(level, wxz.add(vec2(0, eps)) as unknown as NV2)
      .sub(this.microMorphWeightGpu(level, wxz.sub(vec2(0, eps)) as unknown as NV2))
      .div(2 * eps);
    const delta = child.x.sub(parent.x);
    return vec3(
      mix(parent.x, child.x, weight),
      mix(parent.y, child.y, weight).add(delta.mul(dwdx)),
      mix(parent.z, child.z, weight).add(delta.mul(dwdz)),
    ) as unknown as NV3;
  }

  private microMorphWeightGpu(level: HeightLevel, wxz: NV2): NF {
    const lod = level.lod as -2 | -1;
    const band = MICRO_MORPH_BANDS[lod];
    const local = wxz
      .sub(vec2(level.uAnchor as unknown as NV2))
      .sub(vec2(this.uMorphCenterRel as unknown as NV2));
    const radius = local.x.abs().max(local.y.abs());
    const camera = smootherStep01Tsl(radius.sub(band.innerM).div(band.outerM - band.innerM)).oneMinus();
    const available = this.microAvailabilityGpu(level, wxz);
    return camera.mul(available) as unknown as NF;
  }

  /** Coverage-edge availability factor of {@link microMorphWeightGpu} WITHOUT the
   *  camera term — the camera-INDEPENDENT "is a packed rung actually cooked here"
   *  signal (smootherstep to 1 inside the level's coverage box + res interior, 0
   *  at/outside its edge). Uniforms + ALU only, no texture taps. Extracted so both
   *  the shipping morph weight and the debug availability accessor share ONE exact
   *  expression — the default morph path is node-for-node unchanged. */
  private microAvailabilityGpu(level: HeightLevel, wxz: NV2): NF {
    const lod = level.lod as -2 | -1;
    const band = MICRO_MORPH_BANDS[lod];
    const grid = gridCoords(level, wxz);
    const coverageMin = vec2(level.uCoverageMin as unknown as NV2);
    const coverageMax = vec2(level.uCoverageMax as unknown as NV2);
    const edgeSamples = grid.x
      .sub(1)
      .min(grid.y.sub(1))
      .min(float(level.res - 2).sub(grid.x))
      .min(float(level.res - 2).sub(grid.y))
      .min(grid.x.sub(coverageMin.x).sub(1))
      .min(grid.y.sub(coverageMin.y).sub(1))
      .min(coverageMax.x.sub(grid.x).sub(1))
      .min(coverageMax.y.sub(grid.y).sub(1));
    const edgeM = edgeSamples.mul(level.texel);
    return smootherStep01Tsl(edgeM.div(band.availabilityM)) as unknown as NF;
  }

  /** DEBUG-only (`?nandbg=lod`): the packed-rung morph weights the height sampler
   *  ACTUALLY applies at a world point — camera×availability, so the pair IS the
   *  LOD the surface resolves to right now. `fineW`→lod-2 (0.0625 m), `parentW`→
   *  lod-1 (0.25 m); both ~0 ⇒ only the LOD0 (1 m) base is sampled. Uniforms+ALU
   *  only (no texture/storage reads ⇒ no new bindings). Returns null when the
   *  field carries no packed negative levels (nothing but LOD0 base anywhere). */
  lodDebugSampleWeights(wxz: NV2): { fineW: NF; parentW: NF } | null {
    if (!this.cookedMicroHeight) return null;
    const fine = this.heightLevels.find((level) => level.lod === -2);
    const parent = this.heightLevels.find((level) => level.lod === -1);
    return {
      fineW: fine ? this.microMorphWeightGpu(fine, wxz) : (float(0) as unknown as NF),
      parentW: parent ? this.microMorphWeightGpu(parent, wxz) : (float(0) as unknown as NF),
    };
  }

  /** DEBUG-only (`?nandbg=finelod` clip): camera-INDEPENDENT availability of the
   *  packed fine rungs — where the cook PACKED negative-LOD data (coverage extent),
   *  as opposed to where the camera-gated geomorph currently blends it in. `fineA`→
   *  lod-2 present, `parentA`→lod-1 present (each >0 inside its coverage). Uniforms+
   *  ALU only. Returns null when the field carries no packed negative levels. */
  lodDebugAvailability(wxz: NV2): { fineA: NF; parentA: NF } | null {
    if (!this.cookedMicroHeight) return null;
    const fine = this.heightLevels.find((level) => level.lod === -2);
    const parent = this.heightLevels.find((level) => level.lod === -1);
    return {
      fineA: fine ? this.microAvailabilityGpu(fine, wxz) : (float(0) as unknown as NF),
      parentA: parent ? this.microAvailabilityGpu(parent, wxz) : (float(0) as unknown as NF),
    };
  }
}

function smootherStep01Tsl(value: NF): NF {
  const t = value.clamp(0, 1);
  return t.mul(t).mul(t).mul(t.mul(t.mul(6).sub(15)).add(10)) as unknown as NF;
}

// ---- plane sampling helpers (TSL) ----------------------------------------------------
// planeBilerp / planeNearest / insideLevel are exported for the S3b hot-shader
// fetch (NaniteFetch composes its own level chain over field.heightLevels).
// A PINNED level (wraps=false — every generated level) compiles the exact
// pre-S5 addressing; a wrapping level adds the phase-mod on its texel indices.

/** world → continuous sample-grid coords (integer = exact source sample),
 *  window-relative on wrapping levels. */
export function gridCoords(lvl: FieldLevel, wxz: NV2): NV2 {
  // S6c: (wxz − anchor) − (uOrigin − anchor), each side small on Estonia ⇒ the
  // f32 subtraction no longer cancels two ~311 km values into texel-quantized
  // noise. anchor (0,0) on generated ⇒ IEEE-identical to `wxz − uOrigin`.
  return wxz
    .sub(vec2(lvl.uAnchor as unknown as NV2))
    .sub(vec2(lvl.uOriginRel as unknown as NV2))
    .div(lvl.texel);
}

/** logical texel index → physical (toroidal) index on wrapping levels. `i` is
 *  a float in [0, res-1]; the phase add stays < 2·res so one mod suffices. */
function physF(lvl: FieldLevel, i: NF, axis: 0 | 1): NF {
  if (!lvl.wraps) return i;
  const p = vec2(lvl.uPhase as unknown as NV2);
  const ph = (axis === 0 ? p.x : p.y) as unknown as NF;
  return i.add(ph).mod(lvl.res) as unknown as NF;
}

function texelU(lvl: FieldLevel, ix: NF, iy: NF): { x: NU; y: NU } {
  return {
    x: floor(physF(lvl, ix, 0)).toUint() as NU,
    y: floor(physF(lvl, iy, 1)).toUint() as NU,
  };
}

/** finest-containing-level branch chain: runs `arm` for exactly one level (the
 *  last level is the unconditional backstop — always resident, spec §3). */
export function hotLevelChain(
  levels: readonly FieldLevel[],
  wxz: NV2,
  arm: (lvl: FieldLevel, finest: boolean) => void,
): void {
  const n = levels.length;
  if (n === 0) throw new Error('TerrainField: hotLevelChain over zero levels');
  if (n === 1) {
    arm(levels[0] as FieldLevel, true);
    return;
  }
  let chain = If(insideLevel(levels[0] as FieldLevel, wxz), () => {
    arm(levels[0] as FieldLevel, true);
  });
  for (let i = 1; i < n - 1; i++) {
    const lvl = levels[i] as FieldLevel;
    chain = chain.ElseIf(insideLevel(lvl, wxz), () => {
      arm(lvl, false);
    });
  }
  chain.Else(() => {
    arm(levels[n - 1] as FieldLevel, false);
  });
}

/** one hardware-filtered tap (LinearFilter rgba8 planes) at a level. Wrapping
 *  levels take the 4 texel loads instead (a filtered tap would blend across
 *  the toroidal seam). */
function planeLinear(lvl: FieldLevel, wxz: NV2): NV4 {
  if (!lvl.wraps) {
    const uv = gridCoords(lvl, wxz).add(0.5).div(lvl.res);
    return texture(lvl.tex, uv as unknown as NV2, 0) as unknown as NV4;
  }
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0 = physF(lvl, i0.x as unknown as NF, 0);
  const y0 = physF(lvl, i0.y as unknown as NF, 1);
  const x1 = physF(lvl, clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF, 0);
  const y1 = physF(lvl, clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF, 1);
  const uvOf = (x: NF, y: NF): NV2 => vec2(x.add(0.5), y.add(0.5)).div(lvl.res) as unknown as NV2;
  const s00 = texture(lvl.tex, uvOf(x0, y0), 0) as unknown as NV4;
  const s10 = texture(lvl.tex, uvOf(x1, y0), 0) as unknown as NV4;
  const s01 = texture(lvl.tex, uvOf(x0, y1), 0) as unknown as NV4;
  const s11 = texture(lvl.tex, uvOf(x1, y1), 0) as unknown as NV4;
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y) as unknown as NV4;
}

/** ±1-texel central differences at the nearest texel — the normalTex bake's
 *  exact stencil (see Heightfield rebuildDerivedMaps history) */
function cdTaps(lvl: FieldLevel, wxz: NV2): { dx: NF; dz: NF } {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 1, lvl.res - 2);
  const gx = floor(g.x) as unknown as NF;
  const gy = floor(g.y) as unknown as NF;
  const tl = (ix: NF, iy: NF): NF => {
    const t = texelU(lvl, ix, iy);
    return texLoadR(lvl.tex, t.x, t.y);
  };
  const hl = tl(gx.sub(1) as unknown as NF, gy);
  const hr = tl(gx.add(1) as unknown as NF, gy);
  const hd = tl(gx, gy.sub(1) as unknown as NF);
  const hu = tl(gx, gy.add(1) as unknown as NF);
  return { dx: hl.sub(hr) as unknown as NF, dz: hd.sub(hu) as unknown as NF };
}

function slope4(lvl: FieldLevel, wxz: NV2): NF {
  const { dx, dz } = cdTaps(lvl, wxz);
  return vec2(dx, dz).length().div(lvl.texel * 2) as unknown as NF;
}

/** vec4(world normal, slope) — the retired normalTex texel, derived live:
 *  n = normalize(hl−hr, 2·texel, hd−hu); slope = |(hl−hr, hd−hu)| / 2·texel */
function normalSlope4(lvl: FieldLevel, wxz: NV2): NV4 {
  const { dx, dz } = cdTaps(lvl, wxz);
  const n = vec3(dx, lvl.texel * 2, dz).normalize();
  return vec4(n, vec2(dx, dz).length().div(lvl.texel * 2)) as unknown as NV4;
}

export function planeBilerp(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0i = i0.x as unknown as NF;
  const y0i = i0.y as unknown as NF;
  const x1i = clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF;
  const y1i = clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF;
  const t00 = texelU(lvl, x0i, y0i);
  const t10 = texelU(lvl, x1i, y0i);
  const t01 = texelU(lvl, x0i, y1i);
  const t11 = texelU(lvl, x1i, y1i);
  const s00 = texLoadR(lvl.tex, t00.x, t00.y);
  const s10 = texLoadR(lvl.tex, t10.x, t10.y);
  const s01 = texLoadR(lvl.tex, t01.x, t01.y);
  const s11 = texLoadR(lvl.tex, t11.x, t11.y);
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

/** Bilinear value and its exact within-cell world gradient from the same 4 taps. */
function planeHeightGradient(lvl: HeightLevel, wxz: NV2): NV3 {
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0i = i0.x as unknown as NF;
  const y0i = i0.y as unknown as NF;
  const x1i = clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF;
  const y1i = clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF;
  const t00 = texelU(lvl, x0i, y0i);
  const t10 = texelU(lvl, x1i, y0i);
  const t01 = texelU(lvl, x0i, y1i);
  const t11 = texelU(lvl, x1i, y1i);
  const s00 = texLoadR(lvl.tex, t00.x, t00.y);
  const s10 = texLoadR(lvl.tex, t10.x, t10.y);
  const s01 = texLoadR(lvl.tex, t01.x, t01.y);
  const s11 = texLoadR(lvl.tex, t11.x, t11.y);
  const height = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
  const dx = mix(s10.sub(s00), s11.sub(s01), f.y).div(lvl.texel);
  const dz = mix(s01.sub(s00), s11.sub(s10), f.x).div(lvl.texel);
  return vec3(height, dx, dz) as unknown as NV3;
}

/** C0-continuous height gradient (m/m): the 2×2 surrounding CELL mean-gradients
 *  (each cell's average bilinear-surface gradient) bilinearly interpolated at the
 *  continuous sample point — the dual grid, 3×3 stencil. cdTaps snaps to the
 *  nearest texel (piecewise-CONSTANT slope per texel) and planeHeightGradient's
 *  components are piecewise-constant along their own axis, so any threshold
 *  downstream prints the texel lattice as axis-aligned blocks; this field is
 *  continuous everywhere, so thresholds trace the relief forms instead. */
function planeGradientSmooth(lvl: HeightLevel, wxz: NV2): NV2 {
  const gc = clamp(gridCoords(lvl, wxz).sub(0.5), 0, lvl.res - 2);
  const i0 = floor(gc);
  const f = fract(gc);
  const xi = [
    i0.x as unknown as NF,
    clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF,
    clamp(i0.x.add(2), 0, lvl.res - 1) as unknown as NF,
  ];
  const yi = [
    i0.y as unknown as NF,
    clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF,
    clamp(i0.y.add(2), 0, lvl.res - 1) as unknown as NF,
  ];
  const h: NF[][] = xi.map((x) =>
    yi.map((y) => {
      const t = texelU(lvl, x, y);
      return texLoadR(lvl.tex, t.x, t.y);
    }),
  );
  const H = (a: number, b: number): NF => (h[a] as NF[])[b] as NF;
  // cell (a,b) ∈ {0,1}²: mean gradient over its 2×2 sample corners
  const cdx = (a: number, b: number): NF =>
    H(a + 1, b).sub(H(a, b)).add(H(a + 1, b + 1)).sub(H(a, b + 1)) as unknown as NF;
  const cdz = (a: number, b: number): NF =>
    H(a, b + 1).sub(H(a, b)).add(H(a + 1, b + 1)).sub(H(a + 1, b)) as unknown as NF;
  const dx = mix(mix(cdx(0, 0), cdx(1, 0), f.x), mix(cdx(0, 1), cdx(1, 1), f.x), f.y);
  const dz = mix(mix(cdz(0, 0), cdz(1, 0), f.x), mix(cdz(0, 1), cdz(1, 1), f.x), f.y);
  return vec2(dx, dz).div(lvl.texel * 2) as unknown as NV2;
}

export function planeNearest(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 0, lvl.res - 1);
  const t = texelU(lvl, floor(g.x) as unknown as NF, floor(g.y) as unknown as NF);
  return texLoadR(lvl.tex, t.x, t.y);
}

function planeNearest4(lvl: FieldLevel, wxz: NV2): NV4 {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 0, lvl.res - 1);
  const x = physF(lvl, floor(g.x) as unknown as NF, 0);
  const y = physF(lvl, floor(g.y) as unknown as NF, 1);
  const uv = vec2(x.add(0.5), y.add(0.5)).div(lvl.res);
  return texture(lvl.tex, uv as unknown as NV2, 0) as unknown as NV4;
}

/** the 2×2 tap set around wxz shared by the wet-masked samplers below:
 *  corner values s, bilinear corner weights w (sum 1), wet mask flags. */
function wetTaps(lvl: HeightLevel, wxz: NV2): { s: NF[]; w: NF[]; wet: NF[]; f: NV2 } {
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0i = i0.x as unknown as NF;
  const y0i = i0.y as unknown as NF;
  const x1i = clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF;
  const y1i = clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF;
  const t00 = texelU(lvl, x0i, y0i);
  const t10 = texelU(lvl, x1i, y0i);
  const t01 = texelU(lvl, x0i, y1i);
  const t11 = texelU(lvl, x1i, y1i);
  const s = [
    texLoadR(lvl.tex, t00.x, t00.y),
    texLoadR(lvl.tex, t10.x, t10.y),
    texLoadR(lvl.tex, t01.x, t01.y),
    texLoadR(lvl.tex, t11.x, t11.y),
  ];
  const w = [
    f.x.oneMinus().mul(f.y.oneMinus()) as unknown as NF,
    f.x.mul(f.y.oneMinus()) as unknown as NF,
    f.x.oneMinus().mul(f.y) as unknown as NF,
    f.x.mul(f.y) as unknown as NF,
  ];
  // wet = above the −1e4 dry sentinel (real Estonia water levels are ≫ −1000)
  const wet = s.map(
    (v) => v.greaterThan(-1000).select(float(1), float(0)) as unknown as NF,
  );
  return { s, w, wet, f: f as unknown as NV2 };
}

/** #GAP wet-masked bilinear: planeBilerp with the 2×2 weights gated by a wetness
 *  mask (texel above the dry sentinel) and re-normalized, so the −1e4 sentinel
 *  corners never drag the interpolated surface down near a shore. All-dry ⇒ the
 *  plain average (the sentinel), which the caller clamps to the bed. */
function planeBilerpWet(lvl: HeightLevel, wxz: NV2): NF {
  const { s, w, wet, f } = wetTaps(lvl, wxz);
  const a = w.map((wi, i) => wi.mul(wet[i] as NF) as unknown as NF);
  const wsum = (a[0] as NF).add(a[1] as NF).add(a[2] as NF).add(a[3] as NF);
  const vsum = (a[0] as NF)
    .mul(s[0] as NF)
    .add((a[1] as NF).mul(s[1] as NF))
    .add((a[2] as NF).mul(s[2] as NF))
    .add((a[3] as NF).mul(s[3] as NF));
  const plain = mix(
    mix(s[0] as NF, s[1] as NF, f.x),
    mix(s[2] as NF, s[3] as NF, f.x),
    f.y,
  );
  return wsum.greaterThan(1e-4).select(vsum.div(wsum.max(1e-4)), plain) as unknown as NF;
}

/** #GAP bilinear of the binary wet mask (= Σ wᵢ·wetᵢ, weights sum 1): the smooth
 *  0→1 one-texel ramp across the wet-dilation band — see fieldWaterWetFrac. */
function planeWetFrac(lvl: HeightLevel, wxz: NV2): NF {
  const { w, wet } = wetTaps(lvl, wxz);
  return (w[0] as NF)
    .mul(wet[0] as NF)
    .add((w[1] as NF).mul(wet[1] as NF))
    .add((w[2] as NF).mul(wet[2] as NF))
    .add((w[3] as NF).mul(wet[3] as NF)) as unknown as NF;
}

/** true where the point sits ≥1 texel inside the level's window (the rim texel is
 *  clamp-extended fill or the scroll seam — the next-coarser level owns it) */
export function insideLevel(lvl: FieldLevel, wxz: NV2): NB {
  const g = gridCoords(lvl, wxz);
  return g.x
    .greaterThanEqual(1)
    .and(g.y.greaterThanEqual(1))
    .and(g.x.lessThanEqual(lvl.res - 2))
    .and(g.y.lessThanEqual(lvl.res - 2)) as NB;
}

// ---- CPU bilinear ---------------------------------------------------------------------

function bilerpCpu(lvl: HeightLevel, gx: number, gz: number): number {
  const res = lvl.res;
  const cx = Math.min(Math.max(gx, 0), res - 1.001);
  const cz = Math.min(Math.max(gz, 0), res - 1.001);
  const x0 = Math.floor(cx);
  const z0 = Math.floor(cz);
  const fx = cx - x0;
  const fz = cz - z0;
  const px = lvl.wraps ? lvl.uPhase.value.x : 0;
  const pz = lvl.wraps ? lvl.uPhase.value.y : 0;
  const at = (x: number, z: number): number => {
    const ix = (Math.min(x, res - 1) + px) % res;
    const iz = (Math.min(z, res - 1) + pz) % res;
    return lvl.data[iz * res + ix] ?? 0;
  };
  const a = at(x0, z0) * (1 - fx) + at(x0 + 1, z0) * fx;
  const b = at(x0, z0 + 1) * (1 - fx) + at(x0 + 1, z0 + 1) * fx;
  return a * (1 - fz) + b * fz;
}

// ---- level construction -----------------------------------------------------------------

function makeHeightLevel(name: string, plan: PlanePlan, data?: Float32Array): HeightLevel {
  const backing = data ?? new Float32Array(plan.res * plan.res);
  const tex = new DataTexture(backing, plan.res, plan.res, RedFormat, FloatType);
  configurePlane(tex, name);
  return { ...levelCommon(plan), tex, data: backing };
}

function makeU8Level(name: string, plan: PlanePlan): FieldLevel {
  const tex = new DataTexture(new Uint8Array(plan.res * plan.res * 4), plan.res, plan.res, RGBAFormat, UnsignedByteType);
  // rgba8 IS filterable — hot consumers (raster disp, grass density, resolve
  // shading) take ONE hardware-filtered tap instead of a manual 4-tap bilerp
  configurePlane(tex, name, LinearFilter);
  return { ...levelCommon(plan), tex };
}

function levelCommon(plan: PlanePlan): Omit<FieldLevel, 'tex'> {
  const coverageMinX = plan.originX + (plan.nMinX - plan.n0x) * plan.texel;
  const coverageMinZ = plan.originZ + (plan.nMinZ - plan.n0z) * plan.texel;
  const coverageMaxX = plan.originX + (plan.nMaxX - plan.n0x) * plan.texel;
  const coverageMaxZ = plan.originZ + (plan.nMaxZ - plan.n0z) * plan.texel;
  return {
    lod: plan.lod,
    res: plan.res,
    texel: plan.texel,
    originX: plan.originX,
    originZ: plan.originZ,
    uOrigin: uniform(new Vector2(plan.originX, plan.originZ)) as unknown as FieldLevel['uOrigin'],
    // anchor (0,0) initially ⇒ uOriginRel ≡ uOrigin (generated-world identity)
    uOriginRel: uniform(new Vector2(plan.originX, plan.originZ)) as unknown as FieldLevel['uOriginRel'],
    uAnchor: uniform(new Vector2(0, 0)) as unknown as FieldLevel['uAnchor'],
    uPhase: uniform(new Vector2(0, 0)) as unknown as FieldLevel['uPhase'],
    uCoverageMin: uniform(new Vector2(
      (coverageMinX - plan.originX) / plan.texel,
      (coverageMinZ - plan.originZ) / plan.texel,
    )) as unknown as FieldLevel['uCoverageMin'],
    uCoverageMax: uniform(new Vector2(
      (coverageMaxX - plan.originX) / plan.texel,
      (coverageMaxZ - plan.originZ) / plan.texel,
    )) as unknown as FieldLevel['uCoverageMax'],
    coverageMinX,
    coverageMinZ,
    coverageMaxX,
    coverageMaxZ,
    wraps: plan.wraps,
  };
}

function configurePlane(tex: DataTexture, name: string, filter: typeof NearestFilter | typeof LinearFilter = NearestFilter): void {
  tex.name = name;
  tex.magFilter = filter;
  tex.minFilter = filter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
}

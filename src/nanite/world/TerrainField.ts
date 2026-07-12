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
import type { NB, NF, NU, NV2, NV4 } from '../../gpu/TSLTypes';
import type { CoverageBox, FieldPlan, PlanePlan } from './PlaneFill';
import type { LevelGridEdit } from './PartitionTree';
import type { PlaneKind } from './StreamProtocol';

export type { CoverageBox } from './PlaneFill';

/** full 5-level Estonia set ≈ 84 (height) + 17 (biome) + ~40 (fields+water) MB;
 *  the generated full-coverage set ≈ 130 MB — anything past this is a leak, not
 *  a config (throw-loud VRAM law). */
const VRAM_CEILING_MB = 160;

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
  /** frozen at plan time: window < coverage ⇒ the level scrolls toroidally.
   *  Pinned levels (all generated ones) compile the exact pre-S5 samplers. */
  wraps: boolean;
  tex: DataTexture;
}

interface HeightLevel extends FieldLevel {
  data: Float32Array;
}

export class TerrainField {
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
  readonly coverageBox: CoverageBox;
  /** biome plane channels 2/3 carry the merged far-forest canopy (heightM, cover)
   *  — true iff the source has a canopy layer. The generated world packs snow/
   *  rockExposure there instead, so its resolve canopy tint stays off (bit-identical). */
  readonly biomeCarriesCanopy: boolean;

  private constructor(
    heightLevels: HeightLevel[],
    biomeLevels: FieldLevel[],
    fieldsLevels: FieldLevel[],
    water: HeightLevel | null,
    waterFar: HeightLevel | null,
    waterCover: FieldLevel | null,
    waterCoverFar: FieldLevel | null,
    soil: FieldLevel | null,
    coverageBox: CoverageBox,
    biomeCarriesCanopy: boolean,
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
    this.coverageBox = coverageBox;
    this.biomeCarriesCanopy = biomeCarriesCanopy;
    const mb = this.vramBytes() / 2 ** 20;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] terrain field: height [${heightLevels.map((l) => `${l.res}²@${l.texel}m${l.wraps ? '~' : ''}`).join(' ')}] r32f + ` +
        `biome [${biomeLevels.map((l) => `${l.res}²`).join(' ')}] + fields [${fieldsLevels.map((l) => `${l.res}²`).join(' ')}] rgba8 + ` +
        `water ${water ? `${water.res}² r32f (+far ${waterFar?.res ?? 0}²)` : 'none'} + ` +
        `watercover ${waterCover ? `${waterCover.res}² rgba8${waterCover.wraps ? '~' : ''} (+far ${waterCoverFar?.res ?? 0}²)` : 'none'} + ` +
        `soil ${soil ? `${soil.res}² rgba8${soil.wraps ? '~' : ''}` : 'none'} = ` +
        `${mb.toFixed(1)} MB VRAM (CPU mirrors share the backing; ~ = camera-window level)`,
    );
    if (mb > VRAM_CEILING_MB) {
      throw new Error(`TerrainField: ${mb.toFixed(1)} MB exceeds the ${VRAM_CEILING_MB} MB ceiling`);
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
    return new TerrainField(heightLevels, biomeLevels, fieldsLevels, water, waterFar, waterCover, waterCoverFar, soil, plan.coverageBox, plan.biomeHasCanopy);
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
      {
        minX: opts.worldMinX,
        minZ: opts.worldMinZ,
        maxX: opts.worldMinX + opts.res * opts.texel,
        maxZ: opts.worldMinZ + opts.res * opts.texel,
      },
      false, // no biome/canopy planes on a single-level field
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
                    : this.soil;
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
    ];
  }

  // ---- CPU sampling (walk probe, spawn, bookmarks) --------------------------------

  /** bilinear height (m) — finest level whose window contains the point */
  heightAt(x: number, z: number): number {
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

  /** bilinear height, finest resident level selected per sample */
  fieldHeightFinest(wxz: NV2): NF {
    return this.finestSelect(wxz, planeBilerp);
  }

  /** nearest-texel height, finest resident level selected per sample */
  fieldHeightFinestNearest(wxz: NV2): NF {
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
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl, finest) => {
      out.assign(finest ? planeNearest(lvl, wxz) : planeBilerp(lvl, wxz));
    });
    return out as unknown as NF;
  }

  /** central-difference slope (rise/run) from the height planes — the in-shader
   *  replacement for the retired normalTex.w: the SAME ±1-texel stencil the old
   *  bake ran (Heightfield derived-maps kernel), evaluated at the nearest texel
   *  of the finest containing level. `level` HOISTS the select for consumers
   *  whose window containment is guaranteed (grass guide ring ≪ the L0 window). */
  fieldSlope(wxz: NV2, level?: number): NF {
    if (level !== undefined) return slope4(this.heightLevels[level] as HeightLevel, wxz);
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => out.assign(slope4(lvl, wxz)));
    return out as unknown as NF;
  }

  /** central-difference world normal (xyz) + slope (w) — the retired normalTex's
   *  EXACT bake stencil (n = normalize(hl−hr, 2·texel, hd−hu); slope = |∇h|/2texel)
   *  evaluated in-shader at the finest containing level (S3b resolve). */
  fieldNormalSlope(wxz: NV2): NV4 {
    const out = vec4(0, 1, 0, 0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => out.assign(normalSlope4(lvl, wxz)));
    return out as unknown as NV4;
  }

  /** surface-fields sample [moisture, flowStrength, snow, rockExposure] — one
   *  hardware-filtered rgba8 tap at the finest containing level. vec4(0) when
   *  the source has no fields layer (forest/gallery single-level fields). */
  fieldsAt(wxz: NV2): NV4 {
    if (this.fieldsLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.fieldsLevels, wxz, (lvl) => out.assign(planeLinear(lvl, wxz)));
    return out as unknown as NV4;
  }

  /** biome/canopy sample [classId (raw id byte ⇒ ×255 to decode), vegDensity,
   *  canopyHeight, cover] — one filtered rgba8 tap, finest containing level.
   *  vec4(0) when the source has no biome layer. */
  biomeAt(wxz: NV2): NV4 {
    if (this.biomeLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.biomeLevels, wxz, (lvl) => out.assign(planeLinear(lvl, wxz)));
    return out as unknown as NV4;
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
  let chain = If(insideLevel(levels[0] as FieldLevel, wxz), () => arm(levels[0] as FieldLevel, true));
  for (let i = 1; i < n - 1; i++) {
    const lvl = levels[i] as FieldLevel;
    chain = chain.ElseIf(insideLevel(lvl, wxz), () => arm(lvl, false));
  }
  chain.Else(() => arm(levels[n - 1] as FieldLevel, false));
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

export function planeNearest(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 0, lvl.res - 1);
  const t = texelU(lvl, floor(g.x) as unknown as NF, floor(g.y) as unknown as NF);
  return texLoadR(lvl.tex, t.x, t.y);
}

/** #GAP wet-masked bilinear: planeBilerp with the 2×2 weights gated by a wetness
 *  mask (texel above the dry sentinel) and re-normalized, so the −1e4 sentinel
 *  corners never drag the interpolated surface down near a shore. All-dry ⇒ the
 *  plain average (the sentinel), which the caller clamps to the bed. */
function planeBilerpWet(lvl: HeightLevel, wxz: NV2): NF {
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
  const w00 = f.x.oneMinus().mul(f.y.oneMinus());
  const w10 = f.x.mul(f.y.oneMinus());
  const w01 = f.x.oneMinus().mul(f.y);
  const w11 = f.x.mul(f.y);
  // wet = above the −1e4 dry sentinel (real Estonia water levels are ≫ −1000)
  const wet = (s: NF): NF => s.greaterThan(-1000).select(float(1), float(0)) as unknown as NF;
  const a00 = w00.mul(wet(s00));
  const a10 = w10.mul(wet(s10));
  const a01 = w01.mul(wet(s01));
  const a11 = w11.mul(wet(s11));
  const wsum = a00.add(a10).add(a01).add(a11);
  const vsum = a00.mul(s00).add(a10.mul(s10)).add(a01.mul(s01)).add(a11.mul(s11));
  const plain = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
  return wsum.greaterThan(1e-4).select(vsum.div(wsum.max(1e-4)), plain) as unknown as NF;
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

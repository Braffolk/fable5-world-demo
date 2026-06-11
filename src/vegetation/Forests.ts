/**
 * Forests — GPU-driven rendering of the scattered world (spec §3.6 core).
 *
 * Per frame, compute passes run before render:
 *   clear counters → cull each scatter layer → write indirect args.
 * The cull kernel does: per-class distance bound → frustum sphere test (6
 * planes) → terrain-occlusion march (heightfield ray test camera→crown-top,
 * the "Hi-Z" of a heightfield world) → LOD ring classification with overlap
 * bands → atomic append of the instance slot into per-(pool,ring) compact
 * regions. Draw instance counts go straight into an indirect buffer
 * (geometry.setIndirect) — instance data and counts never touch the CPU.
 * Cull granularity = instance (tree/shrub/rock), not 64-tri meshlets — the
 * deviation and rationale are documented in DEVIATIONS D-5.
 *
 * LOD rings (dithered crossfades in the materials):
 *   trees:  R1 full cards ≤110 m → R2 branch-cards ≤340 m → octahedral
 *           impostors beyond (4-tile view blend, relit — D-4 runtime)
 *   understory: single ring with per-class max distance
 *   extras: boulders/slabs swap to low-detail rock at 120 m, live to 700 m
 */

import { Group, Mesh, Vector3, Vector4 } from 'three';
import type { PerspectiveCamera } from 'three';
import { Frustum, Matrix4 } from 'three';
import {
  IndirectStorageBufferAttribute,
  StorageBufferAttribute,
  type MeshStandardNodeMaterial,
  type Renderer,
  type StorageBufferNode,
  type StorageTexture,
} from 'three/webgpu';
import { IrradianceNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  float,
  instanceIndex,
  instancedArray,
  int,
  normalWorld,
  positionWorld,
  storage,
  uint,
  uniform,
  uniformArray,
  vec2,
  vec3,
} from 'three/tsl';
import type { Heightfield } from '../world/Heightfield';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import { canopyAt, type ScatterLayer, type ScatterResult } from '../gpu/passes/Scatter';
import { impostorQuad, impostorRuntimeMaterial } from '../render/ImpostorRuntime';
import { instanceVeg, type RingFade } from '../render/VegInstance';
import type { NF, NI, NU, NV3, NV4 } from '../gpu/TSLTypes';
import type { VegLib } from './VegLibrary';

// ring distances (m) + dither bands (user feedback: transitions read too
// close — full-card trees hold to 150 m, impostors start at 460 m)
const R1_FAR = 150;
const BAND1 = 14;
const R2_FAR = 460;
const BAND2 = 36;
const EX_R1_FAR = 120;
const EX_BAND = 15;

// per-group compact-region capacities
const CAP_TREE_R1 = 6144;
const CAP_TREE_R2 = 8192;
const CAP_IMPOSTOR = 49152;
const CAP_UNDER = 4096;
const CAP_EX_R1 = 1024;
const CAP_EX_R2 = 2048;

const GROUPS = 146;

function groupOf(cls: number, variant: number, ring: 1 | 2 | 3): number {
  if (cls < 6) {
    if (ring === 3) return 48 + cls;
    return (cls * 4 + variant) * 2 + (ring - 1);
  }
  if (cls < 15) return 54 + (cls - 8) * 4 + variant;
  const pe = (cls - 16) * 4 + variant;
  return 82 + pe * 2 + (ring - 1);
}

function capOf(g: number): number {
  if (g < 48) return g % 2 === 0 ? CAP_TREE_R1 : CAP_TREE_R2;
  if (g < 54) return CAP_IMPOSTOR;
  if (g < 82) return CAP_UNDER;
  if (g < 114) return (g - 82) % 2 === 0 ? CAP_EX_R1 : CAP_EX_R2;
  // size-stratified stones/branches (cls 20–23)
  const cls = 16 + ((g - 82) >> 3);
  const isR1 = (g - 82) % 2 === 0;
  if (cls === 20) return isR1 ? 4096 : 24576; // StoneL → 900 m
  if (cls === 21) return isR1 ? 8192 : 16384; // StoneM → 280 m
  if (cls === 22) return isR1 ? 24576 : 64; // StoneS — single ring
  return 8192; // Branch
}

export class Forests {
  readonly group = new Group();

  private compact!: StorageBufferNode<'uint'>;
  private counters!: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private kernels: object[] = [];
  private camU = uniform(new Vector3());
  private planesU = uniformArray(
    Array.from({ length: 6 }, () => new Vector4()),
  );
  private frustum = new Frustum();
  private projView = new Matrix4();
  private indirectAttr!: IndirectStorageBufferAttribute;
  private groupTris = new Float32Array(GROUPS);
  private groupCaps = new Uint32Array(GROUPS);
  private reading = false;
  private frame = 0;
  private hud: Record<string, number> = {};

  constructor(
    private hf: Heightfield,
    private scatter: ScatterResult,
    private lib: VegLib,
    private gi: ProbeGI | null,
    private canopyTex: StorageTexture | null = null,
  ) {}

  private patchGI(mat: MeshStandardNodeMaterial): void {
    const gi = this.gi;
    if (!gi) return;
    let irr = gi.irradiance(positionWorld as unknown as NV3, normalWorld as unknown as NV3);
    if (this.canopyTex) {
      // probes don't see trees — canopy coverage pulls ambient down inside
      // the forest (veg gets a lighter clamp than ground: crowns curve up
      // into open sky)
      irr = irr.mul(
        canopyAt(this.canopyTex, (positionWorld as unknown as NV3).xz)
          .mul(0.4)
          .oneMinus(),
      ) as typeof irr;
    }
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }

  init(renderer: Renderer): void {
    void renderer;
    const lib = this.lib;

    // ---- compact regions / group tables ------------------------------------
    const offsets = new Uint32Array(GROUPS);
    let off = 0;
    for (let g = 0; g < GROUPS; g++) {
      offsets[g] = off;
      this.groupCaps[g] = capOf(g);
      off += capOf(g);
    }
    this.compact = instancedArray(off, 'uint');
    this.counters = instancedArray(GROUPS, 'uint').toAtomic();
    const offBuf = storage(new StorageBufferAttribute(offsets, 1), 'uint', GROUPS);
    const capBuf = storage(
      new StorageBufferAttribute(this.groupCaps.slice(), 1),
      'uint',
      GROUPS,
    );

    // per-class cull info: (height, radius, maxDist, hasR2)
    const clsInfo = new Float32Array(24 * 4);
    for (let c = 0; c < 24; c++) {
      clsInfo[c * 4 + 0] = this.lib.clsHeight[c] ?? 1;
      clsInfo[c * 4 + 1] = this.lib.clsRadius[c] ?? 1;
      clsInfo[c * 4 + 2] = this.lib.clsMaxDist[c] ?? 150;
      const hasR2 = c < 6 || c === 18 || c === 19 || c === 20 || c === 21 || c === 23;
      clsInfo[c * 4 + 3] = hasR2 ? 1 : 0;
    }
    const clsBuf = storage(new StorageBufferAttribute(clsInfo, 4), 'vec4', 24);

    // ---- draws ---------------------------------------------------------------
    interface DrawSpec {
      group: number;
      indexCount: number;
    }
    const draws: DrawSpec[] = [];
    const meshes: Mesh[] = [];

    const addDraw = (
      geo: import('three').BufferGeometry,
      mat: MeshStandardNodeMaterial,
      g: number,
      tris: number,
      castShadow: boolean,
    ): void => {
      const indexCount = geo.index ? geo.index.count : geo.attributes.position?.count ?? 0;
      draws.push({ group: g, indexCount });
      this.groupTris[g] += tris;
      const mesh = new Mesh(geo, mat);
      mesh.frustumCulled = false;
      mesh.castShadow = castShadow;
      mesh.receiveShadow = true;
      meshes.push(mesh);
      this.group.add(mesh);
    };

    const layerOf = (cls: number): ScatterLayer =>
      cls < 6
        ? this.scatter.trees
        : cls < 15
          ? this.scatter.understory
          : cls < 20
            ? this.scatter.extras
            : this.scatter.stones;

    const fadeFor = (cls: number, ring: 1 | 2 | 3): RingFade => {
      if (cls < 6) {
        if (ring === 1) return { fadeOutAt: R1_FAR, band: BAND1 };
        if (ring === 2)
          return { fadeInAt: R1_FAR, fadeOutAt: R2_FAR, band: BAND1 };
        return { fadeInAt: R2_FAR, band: BAND2 };
      }
      const maxD = this.lib.clsMaxDist[cls] ?? 150;
      if (cls < 15) return { fadeOutAt: maxD - 15, band: 15 };
      const hasR2 = cls === 18 || cls === 19 || cls === 20 || cls === 21 || cls === 23;
      if (ring === 1)
        return hasR2
          ? { fadeOutAt: EX_R1_FAR, band: EX_BAND }
          : { fadeOutAt: maxD - 20, band: 20 };
      return { fadeInAt: EX_R1_FAR, fadeOutAt: maxD - 20, band: EX_BAND };
    };

    for (const pool of lib.pools) {
      const layer = layerOf(pool.cls);
      const rings: { ring: 1 | 2; parts: typeof pool.r1 }[] = [];
      if (pool.r1) rings.push({ ring: 1, parts: pool.r1 });
      if (pool.r2) rings.push({ ring: 2, parts: pool.r2 });
      for (const { ring, parts } of rings) {
        if (!parts) continue;
        const g = groupOf(pool.cls, pool.variant, ring);
        // shadow budget: tree rings 1+2 cast (≤370 m — cascades 0–2 reach;
        // impostors beyond don't); understory is grounded by contact
        // shadows + AO instead
        const ringCasts = pool.cls < 6 ? true : pool.cls < 15 ? false : true;
        for (const part of parts) {
          const mat = part.make();
          instanceVeg(mat, {
            bufA: layer.bufA,
            bufB: layer.bufB,
            compact: this.compact,
            groupBase: offsets[g] ?? 0,
            fade: fadeFor(pool.cls, ring),
          });
          this.patchGI(mat);
          addDraw(part.geo, mat, g, part.tris, part.castShadow && ringCasts);
        }
      }
    }

    // tree impostors: one billboard draw per species
    for (const [cls, atlas] of lib.impostors) {
      const g = groupOf(cls, 0, 3);
      const mat = impostorRuntimeMaterial(atlas, {
        bufA: this.scatter.trees.bufA,
        bufB: this.scatter.trees.bufB,
        compact: this.compact,
        groupBase: offsets[g] ?? 0,
        fade: fadeFor(cls, 3),
      });
      this.patchGI(mat);
      addDraw(impostorQuad(), mat, g, 2, false);
    }

    // ---- indirect buffer -------------------------------------------------------
    const D = draws.length;
    const indirectData = new Uint32Array(D * 5);
    const drawGroups = new Uint32Array(D);
    for (let d = 0; d < D; d++) {
      const spec = draws[d] as DrawSpec;
      indirectData[d * 5] = spec.indexCount;
      drawGroups[d] = spec.group;
    }
    this.indirectAttr = new IndirectStorageBufferAttribute(indirectData, 5);
    for (let d = 0; d < D; d++) {
      (meshes[d] as Mesh).geometry.setIndirect(this.indirectAttr, d * 20);
    }
    const indirectStore = storage(this.indirectAttr, 'uint', D * 5);
    const drawGroupBuf = storage(new StorageBufferAttribute(drawGroups, 1), 'uint', D);

    // ---- kernels ---------------------------------------------------------------
    const counters = this.counters;
    const compact = this.compact;
    const camU = this.camU;
    const planesU = this.planesU;
    const hf = this.hf;

    const clearK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(GROUPS), () => {
        Return();
      });
      atomicStore(counters.element(i), uint(0));
    })().compute(GROUPS);
    clearK.setName('vegClear');

    const inFrustum = (center: NV3, rad: NF): NF => {
      // product of per-plane step(−r ≤ dist) — 1 inside, 0 outside
      let inside: NF = float(1);
      for (let p = 0; p < 6; p++) {
        const pl = planesU.element(int(p)) as unknown as NV4;
        const d = pl.xyz.dot(center).add(pl.w);
        inside = inside.mul(d.greaterThan(rad.negate()).select(float(1), float(0)));
      }
      return inside;
    };

    const appendTo = (g: NI | NU, slot: NU): void => {
      const idx = atomicAdd(counters.element(g), uint(1)) as unknown as NU;
      If(idx.lessThan(capBuf.element(g) as unknown as NU), () => {
        compact
          .element((offBuf.element(g) as unknown as NU).add(idx))
          .assign(slot);
      });
    };

    const makeCull = (
      layer: ScatterLayer,
      kind: 'trees' | 'under' | 'extras',
    ): object => {
      const N = layer.count;
      const k = Fn(() => {
        const i = instanceIndex;
        If(i.greaterThanEqual(uint(Math.max(N, 1))), () => {
          Return();
        });
        const A = layer.bufA.element(i) as unknown as NV4;
        const B = layer.bufB.element(i) as unknown as NV4;
        const idF = B.w;
        const cls = idF.div(8).floor();
        const variant = idF.sub(cls.mul(8));
        const info = clsBuf.element(cls.toInt()) as unknown as NV4;
        const scl = A.w;
        const hgt = info.x.mul(scl);
        const rad = info.y.mul(scl);
        const center = A.xyz.add(vec3(0, 1, 0).mul(hgt.mul(0.5)));
        const dist = A.xyz.sub(camU).length();

        if (kind !== 'trees') {
          If(dist.greaterThanEqual(info.z), () => {
            Return();
          });
        }
        If(inFrustum(center, rad).lessThan(0.5), () => {
          Return();
        });

        // terrain occlusion: march the sight line to the crown top
        if (kind !== 'under') {
          If(dist.greaterThan(140), () => {
            const top = vec3(A.x, A.y.add(hgt), A.z);
            const occ = float(0).toVar();
            for (let st = 1; st <= 7; st++) {
              const t = st / 8;
              const sp = camU.mul(1 - t).add(top.mul(t)) as unknown as NV3;
              const th = hf.sampleHeightNearest(vec2(sp.x, sp.z));
              occ.assign(occ.max(th.sub(sp.y)));
            }
            If(occ.greaterThan(4), () => {
              Return();
            });
          });
        }

        if (kind === 'trees') {
          const pool = cls.mul(4).add(variant).toInt();
          If(dist.lessThan(R1_FAR + BAND1), () => {
            appendTo(pool.mul(2) as unknown as NI, i as unknown as NU);
          });
          If(
            dist.greaterThanEqual(R1_FAR - BAND1).and(dist.lessThan(R2_FAR + BAND2)),
            () => {
              appendTo(pool.mul(2).add(1) as unknown as NI, i as unknown as NU);
            },
          );
          If(dist.greaterThanEqual(R2_FAR - BAND2), () => {
            appendTo(cls.add(48).toInt() as unknown as NI, i as unknown as NU);
          });
        } else if (kind === 'under') {
          const g = cls.sub(8).mul(4).add(variant).add(54).toInt();
          appendTo(g as unknown as NI, i as unknown as NU);
        } else {
          const pe = cls.sub(16).mul(4).add(variant);
          const hasR2 = info.w.greaterThan(0.5);
          If(hasR2, () => {
            If(dist.lessThan(EX_R1_FAR + EX_BAND), () => {
              appendTo(pe.mul(2).add(82).toInt() as unknown as NI, i as unknown as NU);
            });
            If(dist.greaterThanEqual(EX_R1_FAR - EX_BAND), () => {
              appendTo(pe.mul(2).add(83).toInt() as unknown as NI, i as unknown as NU);
            });
          }).Else(() => {
            appendTo(pe.mul(2).add(82).toInt() as unknown as NI, i as unknown as NU);
          });
        }
      })().compute(Math.max(N, 1));
      k.setName(`vegCull_${kind}`);
      return k;
    };

    const indirectK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(D), () => {
        Return();
      });
      const g = drawGroupBuf.element(i) as unknown as NU;
      const raw = atomicLoad(counters.element(g)) as unknown as NU;
      const cap = capBuf.element(g) as unknown as NU;
      const n = raw.greaterThan(cap).select(cap, raw);
      indirectStore.element(i.mul(5).add(1)).assign(n);
    })().compute(D);
    indirectK.setName('vegIndirect');

    this.kernels = [
      clearK,
      makeCull(this.scatter.trees, 'trees'),
      makeCull(this.scatter.understory, 'under'),
      makeCull(this.scatter.extras, 'extras'),
      makeCull(this.scatter.stones, 'extras'),
      indirectK,
    ];
  }

  /** per-frame: update frustum/camera uniforms, run cull+indirect computes */
  update(renderer: Renderer, camera: PerspectiveCamera): void {
    this.camU.value.copy(camera.position);
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const arr = this.planesU.array as Vector4[];
    for (let p = 0; p < 6; p++) {
      const pl = this.frustum.planes[p];
      if (!pl) continue;
      (arr[p] as Vector4).set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
    }
    for (const k of this.kernels) {
      renderer.compute(k as Parameters<Renderer['compute']>[0]);
    }
    this.frame++;
    if (this.frame % 90 === 0 && !this.reading) {
      this.reading = true;
      void this.readStats(renderer);
    }
  }

  /** HUD stats (throttled async readback of the group counters) */
  counterSnapshot(): Record<string, number> {
    return this.hud;
  }

  private async readStats(renderer: Renderer): Promise<void> {
    try {
      const attr = (this.counters as unknown as { value: unknown }).value;
      const ab = await renderer.getArrayBufferAsync(
        attr as Parameters<Renderer['getArrayBufferAsync']>[0],
      );
      const counts = new Uint32Array(ab);
      let r1 = 0;
      let r2 = 0;
      let imp = 0;
      let under = 0;
      let extras = 0;
      let tris = 0;
      for (let g = 0; g < GROUPS; g++) {
        const n = Math.min(counts[g] ?? 0, this.groupCaps[g] ?? 0);
        tris += n * (this.groupTris[g] ?? 0);
        if (g < 48) {
          if (g % 2 === 0) r1 += n;
          else r2 += n;
        } else if (g < 54) imp += n;
        else if (g < 82) under += n;
        else extras += n;
      }
      this.hud = {
        'veg.r1': r1,
        'veg.r2': r2,
        'veg.imp': imp,
        'veg.underDrawn': under,
        'veg.extraDrawn': extras,
        'veg.tris': Math.round(tris),
      };
    } finally {
      this.reading = false;
    }
  }
}

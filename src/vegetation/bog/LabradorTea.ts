/**
 * Labrador tea — Rhododendron tomentosum (Ledum palustre). An erect, open
 * evergreen dwarf shrub (grows to ~0.5 m; here ~0.4 m). Diagnostic reads:
 *   • LEATHERY, narrow-elliptic leaves 12–50 mm long, 2–12 mm broad, with strongly
 *     INROLLED (revolute) margins, dark green above; undersides + young twigs
 *     densely RUSTY-woolly (orange-brown tomentum). Leaves CLUSTERED toward the
 *     branch tips. (Wikipedia; iNaturalist; MDPI Plants 2024 review.)
 *   • Terminal rounded CLUSTERS (corymbs 3–5 cm) of many small WHITE 5-lobed star
 *     flowers. Habit more upright/open than heather.
 * Reference: en.wikipedia.org/wiki/Rhododendron_tomentosum, iNaturalist 516757,
 * Alaska Wildflowers (lwpetersen.com).
 *
 * Routes as SHRUB → { bark, crown }. Crown = broad elliptic leaves (vdata.x=0)
 * merged with white corymb florets (vdata.x=1). The RUSTY tomentum reads on the
 * bark tint (rusty-brown) — distinct from heather's plain brown wiry stems.
 */

import { Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { growStem, walkStem, leafBlade, starFloret, mergeGeo, perpFrame, type StemSample } from './EricaceousKit';

// ---- recommended integration params ----------------------------------------
export const LABTEA_HEIGHT: [number, number] = [0.3, 0.55];
export const LABTEA_FOLIAGE = { r: 0.13, g: 0.22, b: 0.09, hueVar: 0.16 };
// white corymbs — a small fraction of the crown, at the tips
export const LABTEA_BLOSSOM = { r: 0.9, g: 0.9, b: 0.86, frac: 0.22 };
export const LABTEA_CLS_MAX_DIST = 130;
export const LABTEA_ROUTING = 'SHRUB' as const;
/** rusty tomentum accent for young twigs/undersides (integration note). */
export const LABTEA_RUSTY = { r: 0.42, g: 0.24, b: 0.12 };

interface Parts {
  bark: MeshGrower;
  leaf: MeshGrower;
  flower: MeshGrower;
}

/** clothe a stem with elliptic leathery leaves, densest toward the tip. */
function dressStem(parts: Parts, samples: StemSample[], rng: Rng): void {
  const u = new Vector3();
  const v = new Vector3();
  let node = 0;
  // spiral leaves; density ramps up over the upper stem (leaves cluster at tips)
  walkStem(samples, 0.01, 0.18, (p, dir, t) => {
    perpFrame(dir, u, v);
    // thin out lower leaves, crowd the top
    if (t < 0.5 && rng.float() > 0.55) { node++; return; }
    const golden = node * 2.399963; // spiral phyllotaxis
    node++;
    for (const jitter of [0]) {
      const a = golden + jitter;
      const outN = new Vector3().copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
      // leaf axis: mostly along the stem, tilted out; leaves stand semi-erect
      const axis = new Vector3().copy(dir).multiplyScalar(0.72).addScaledVector(outN, 0.62).normalize();
      const side = new Vector3().crossVectors(axis, outN).normalize();
      const len = 0.022 + rng.float() * 0.013 + t * 0.006;
      const hue = (rng.float() - 0.5) * 0.5;
      // narrow-elliptic: narrow base, widest mid, tapered tip; strong revolute margins
      leafBlade(parts.leaf, p, axis, side, len, len * 0.12, len * 0.3, len * 0.07, 0.28, 0.4, hue, 0, 0.5 + 0.4 * t, Math.min(1, 0.8 + 0.4 * t), 3);
    }
  });
}

/** terminal corymb: a full rounded pom of small white 5-petal florets at a tip.
 *  Florets sit on a slightly-flattened ball (spherical cap wrapping past the
 *  equator) and face radially outward, so the cluster reads as a soft fluffy
 *  dome from every angle rather than a sparse flat disc. */
function corymb(parts: Parts, tip: Vector3, up: Vector3, rng: Rng, swayPhase: number): void {
  const u = new Vector3();
  const v = new Vector3();
  perpFrame(up, u, v);
  const florets = 24 + rng.int(13);
  const R = 0.021 + rng.float() * 0.009;
  // ball centre sits just above the stem tip so the underside wraps the twig
  const center = new Vector3().copy(tip).addScaledVector(up, R * 0.45);
  const radial = new Vector3();
  for (let i = 0; i < florets; i++) {
    const a = rng.float() * Math.PI * 2;
    // cos(elevation) from 1 (top) down past the equator to ~-0.3 (skirt)
    const ce = 1 - rng.float() * 1.3;
    const se = Math.sqrt(Math.max(0, 1 - ce * ce));
    radial
      .copy(u).multiplyScalar(Math.cos(a) * se)
      .addScaledVector(v, Math.sin(a) * se)
      .addScaledVector(up, ce);
    // slightly flattened ball: full radius sideways, 0.8 R vertically
    const c = new Vector3()
      .copy(center)
      .addScaledVector(u, radial.dot(u) * R)
      .addScaledVector(v, radial.dot(v) * R)
      .addScaledVector(up, radial.dot(up) * R * 0.8);
    // florets face outward from the ball centre, biased a touch upward
    const faceN = new Vector3().copy(radial).addScaledVector(up, 0.25).normalize();
    starFloret(parts.flower, c, faceN, 0.0062 + rng.float() * 0.003, swayPhase);
  }
}

export function buildLabradorTeaParts(rng: Rng): Parts {
  const bark = new MeshGrower();
  const leaf = new MeshGrower();
  const flower = new MeshGrower();
  const parts: Parts = { bark, leaf, flower };
  const stems = 3 + rng.int(3);
  const H = LABTEA_HEIGHT[0] + rng.float() * (LABTEA_HEIGHT[1] - LABTEA_HEIGHT[0]);
  for (let i = 0; i < stems; i++) {
    const az = (i / stems) * Math.PI * 2 + rng.float() * 0.8;
    const h = H * (0.8 + rng.float() * 0.24);
    const samples = growStem(
      bark,
      {
        origin: new Vector3(Math.cos(az) * 0.02 * rng.float(), 0, Math.sin(az) * 0.02 * rng.float()),
        azimuth: az,
        lean: 0.06 + rng.float() * 0.16, // upright/open habit
        height: h,
        baseR: 0.0032,
        tipR: 0.0009,
        segs: 6,
        wander: 0.12,
        ascend: 0.6,
        recurve: 0.12,
        swayPhase: rng.float() * Math.PI * 2,
        hue: rng.float() * 2 - 1,
      },
      rng.fork(`stem${i}`),
    );
    dressStem(parts, samples, rng.fork(`dress${i}`));
    const tip = samples[samples.length - 1] as StemSample;
    if (rng.chance(0.85)) corymb(parts, tip.p.clone(), tip.dir.clone(), rng.fork(`cor${i}`), rng.float() * Math.PI * 2);
    // one upper side branch (open, few-branched habit)
    if (rng.chance(0.6)) {
      const si = Math.max(2, samples.length - 3 + rng.int(2));
      const anchor = samples[Math.min(samples.length - 1, si)] as StemSample;
      const br = growStem(
        bark,
        {
          origin: anchor.p.clone(),
          azimuth: az + (rng.float() - 0.5) * 1.6,
          lean: 0.3 + rng.float() * 0.2,
          height: h * (0.3 + rng.float() * 0.2),
          baseR: 0.0016,
          tipR: 0.0007,
          segs: 4,
          wander: 0.14,
          ascend: 0.55,
          recurve: 0.15,
          swayPhase: rng.float() * Math.PI * 2,
          hue: rng.float() * 2 - 1,
        },
        rng.fork(`br${i}`),
      );
      dressStem(parts, br, rng.fork(`bdress${i}`));
      const bt = br[br.length - 1] as StemSample;
      if (rng.chance(0.7)) corymb(parts, bt.p.clone(), bt.dir.clone(), rng.fork(`bcor${i}`), rng.float() * Math.PI * 2);
    }
  }
  return parts;
}

export function buildLabradorTea(rng: Rng): { bark: BufferGeometry; crown: BufferGeometry; barkTris: number; crownTris: number } {
  const parts = buildLabradorTeaParts(rng);
  const bark = parts.bark.build();
  const crown = mergeGeo([parts.leaf.build(), parts.flower.build()]);
  return { bark, crown, barkTris: parts.bark.triCount, crownTris: parts.leaf.triCount + parts.flower.triCount };
}

export function buildPreview(rng: Rng): Object3D {
  const parts = buildLabradorTeaParts(rng);
  const g = new Group();
  const barkMat = new MeshStandardMaterial({ color: 0x7c512e, roughness: 0.95, metalness: 0 }); // rusty tomentose twigs
  const leafMat = new MeshStandardMaterial({ color: 0x38562b, roughness: 0.55, metalness: 0, side: DoubleSide }); // leathery mid-green above
  const flowerMat = new MeshStandardMaterial({ color: 0xf3f1e8, roughness: 0.7, metalness: 0, side: DoubleSide, emissive: 0x3a3a34, emissiveIntensity: 0.25 });
  g.add(new Mesh(parts.bark.build(), barkMat));
  g.add(new Mesh(parts.leaf.build(), leafMat));
  g.add(new Mesh(parts.flower.build(), flowerMat));
  return g;
}

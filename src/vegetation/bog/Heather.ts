/**
 * Heather — Calluna vulgaris. A low, densely-branched wiry evergreen dwarf shrub
 * (here 0.22–0.34 m), bushy mounded habit. The two diagnostic reads:
 *   • TINY (~1–2 mm) SCALE-LEAVES in opposite pairs packed densely along wiry
 *     stems — NOT broad leaves (US Forest Service FEIS; NCSU Extension). Foliage
 *     dark-to-mid green.
 *   • A late-summer PURPLE HAZE: dense terminal racemes of many tiny urn/bell
 *     magenta-lilac florets along the upper stems (RHS; FEIS "terminal racemes,
 *     rosy to purplish-pink"). The signature is MANY small florets, not a few
 *     big flowers.
 * Reference: US Forest Service FEIS (research.fs.usda.gov/feis/species-reviews/calvul),
 * RHS Calluna vulgaris, NCSU Extension Gardener Plant Toolbox.
 *
 * Routes as SHRUB → { bark, crown }. The crown merges scale-leaves (vdata.x=0)
 * with purple florets (vdata.x=1); the integrator colours the crown with a
 * flower-style material (foliageColor → blossom by vdata.x).
 */

import { Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { growStem, walkStem, scaleLeaf, urnBell, mergeGeo, perpFrame, stemFlexAt, type StemSample } from './EricaceousKit';

// ---- recommended integration params ----------------------------------------
export const HEATHER_HEIGHT: [number, number] = [0.22, 0.36];
export const HEATHER_FOLIAGE = { r: 0.09, g: 0.16, b: 0.06, hueVar: 0.22 };
export const HEATHER_BLOSSOM = { r: 0.66, g: 0.44, b: 0.58, frac: 0.34 };
export const HEATHER_CLS_MAX_DIST = 130;
export const HEATHER_ROUTING = 'SHRUB' as const;

interface Parts {
  bark: MeshGrower;
  leaf: MeshGrower;
  flower: MeshGrower;
}

/** decorate one wiry stem with dense whorled scale-leaves + (upper) florets.
 *  Heather foliage must MASS to hide the stems — 4 appressed scale-leaves per
 *  node in a decussate cross, at a short (~5 mm) internode. */
function dressStem(parts: Parts, samples: StemSample[], rng: Rng, floreting: boolean): void {
  const u = new Vector3();
  const v = new Vector3();
  const leafPhase = rng.float() * Math.PI * 2;
  walkStem(samples, 0.007, 0.02, (p, dir, t, k) => {
    perpFrame(dir, u, v);
    const attachFlex = stemFlexAt(t);
    const roll = (k % 2) * (Math.PI / 4); // decussate: quarter-turn each node
    for (let q = 0; q < 4; q++) {
      const a = roll + (q / 4) * Math.PI * 2;
      const outN = new Vector3().copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
      const axis = new Vector3().copy(dir).multiplyScalar(0.45).addScaledVector(outN, 0.9).normalize();
      const side = new Vector3().crossVectors(axis, outN).normalize();
      const hue = (rng.float() - 0.5) * 0.6;
      const len = 0.006 + rng.float() * 0.003;
      scaleLeaf(parts.leaf, p, axis, side, outN, len, len * 0.34, hue, attachFlex, 0.5 + 0.45 * t);
    }
  });
  if (!floreting) return;
  // terminal raceme: many tiny magenta florets down the upper stem, pendent
  walkStem(samples, 0.0085, 0.55, (p, dir, t) => {
    perpFrame(dir, u, v);
    const a = rng.float() * Math.PI * 2;
    // floret hangs slightly out + down from the stem
    const hang = new Vector3()
      .copy(u).multiplyScalar(Math.cos(a) * 0.6)
      .addScaledVector(v, Math.sin(a) * 0.6)
      .addScaledVector(new Vector3(0, -1, 0), 0.8)
      .normalize();
    const size = 0.0042 + rng.float() * 0.0024;
    void rng.float();
    urnBell(parts.flower, p, hang, size, 5, leafPhase, stemFlexAt(t));
  });
}

export function buildHeatherParts(rng: Rng): Parts {
  const bark = new MeshGrower();
  const leaf = new MeshGrower();
  const flower = new MeshGrower();
  const parts: Parts = { bark, leaf, flower };
  // a compact bushy MOUND: many ascending stems packed close, gentle outward
  // splay, curving up — foliage fills the dome and hides the wood.
  const stems = 8 + rng.int(3);
  const H = HEATHER_HEIGHT[0] + rng.float() * (HEATHER_HEIGHT[1] - HEATHER_HEIGHT[0]);
  for (let i = 0; i < stems; i++) {
    const az = (i / stems) * Math.PI * 2 + rng.float() * 0.9;
    const h = H * (0.7 + rng.float() * 0.34);
    // inner stems stand tall, outer stems shorter + lean out → dome envelope
    const ring = rng.float();
    const samples = growStem(
      bark,
      {
        origin: new Vector3(Math.cos(az) * 0.02 * ring, 0, Math.sin(az) * 0.02 * ring),
        azimuth: az,
        lean: 0.08 + ring * 0.26,
        height: h * (1 - ring * 0.3),
        baseR: 0.0017,
        tipR: 0.0004,
        segs: 6,
        wander: 0.2,
        ascend: 0.55,
        recurve: 0.22,
        swayPhase: rng.float() * Math.PI * 2,
        hue: rng.float() * 2 - 1,
      },
      rng.fork(`stem${i}`),
    );
    dressStem(parts, samples, rng.fork(`dress${i}`), true);
    // short secondary twigs off the mid/upper stem for mounded density
    const twigs = 1 + rng.int(2);
    for (let s = 0; s < twigs; s++) {
      const si = 2 + rng.int(Math.max(1, samples.length - 3));
      const anchor = samples[Math.min(samples.length - 1, si)] as StemSample;
      const taz = az + (rng.float() - 0.5) * 2.4;
      const tw = growStem(
        bark,
        {
          origin: anchor.p.clone(),
          azimuth: taz,
          lean: 0.3 + rng.float() * 0.3,
          height: h * (0.26 + rng.float() * 0.24),
          baseR: 0.001,
          tipR: 0.0003,
          segs: 4,
          wander: 0.26,
          ascend: 0.55,
          recurve: 0.3,
          swayPhase: rng.float() * Math.PI * 2,
          hue: rng.float() * 2 - 1,
        },
        rng.fork(`twig${i}_${s}`),
      );
      dressStem(parts, tw, rng.fork(`tdress${i}_${s}`), rng.chance(0.7));
    }
  }
  return parts;
}

/** Integration builder: { bark, crown } with tri counts (mirrors buildShrub). */
export function buildHeather(rng: Rng): { bark: BufferGeometry; crown: BufferGeometry; barkTris: number; crownTris: number } {
  const parts = buildHeatherParts(rng);
  const bark = parts.bark.build();
  const leafGeo = parts.leaf.build();
  const flowerGeo = parts.flower.build();
  const crown = mergeGeo([leafGeo, flowerGeo]);
  return {
    bark,
    crown,
    barkTris: parts.bark.triCount,
    crownTris: parts.leaf.triCount + parts.flower.triCount,
  };
}

export function buildPreview(rng: Rng): Object3D {
  const parts = buildHeatherParts(rng);
  const g = new Group();
  const barkMat = new MeshStandardMaterial({ color: 0x4a3524, roughness: 0.9, metalness: 0 });
  const leafMat = new MeshStandardMaterial({ color: 0x2f4a1c, roughness: 0.72, metalness: 0, side: DoubleSide });
  const flowerMat = new MeshStandardMaterial({ color: 0xbb84a4, roughness: 0.66, metalness: 0, side: DoubleSide, emissive: 0x241621, emissiveIntensity: 0.25 });
  g.add(new Mesh(parts.bark.build(), barkMat));
  g.add(new Mesh(parts.leaf.build(), leafMat));
  g.add(new Mesh(parts.flower.build(), flowerMat));
  return g;
}

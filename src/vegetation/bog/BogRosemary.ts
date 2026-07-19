/**
 * Bog rosemary — Andromeda polifolia. A slender, low, SPARSE evergreen dwarf
 * shrub (15–30 cm), wiry and few-branched — more delicate/open than heather or
 * Labrador tea. Diagnostic reads:
 *   • Narrow, LINEAR-LANCEOLATE leathery leaves, GLAUCOUS blue-green above with a
 *     whitish waxy underside, margins strongly rolled under (revolute), held ±erect
 *     and SPARSE along thin stems (Minnesota Wildflowers; Go Botany; Wikipedia).
 *   • The signature: small, NODDING, urn-shaped (globular bell) PINK flowers hanging
 *     in a small terminal cluster of 2–6 on thin pink pedicels (NCSU Extension:
 *     "1–4 nodding urn-shaped pinkish flowers").
 * Reference: minnesotawildflowers.info/shrub/bog-rosemary, en.wikipedia.org/wiki/
 * Andromeda_polifolia, plants.ces.ncsu.edu/plants/andromeda-polifolia.
 *
 * Routes as SHRUB → { bark, crown }. Crown = sparse glaucous linear leaves
 * (vdata.x=0) + nodding pink urn-bells & pink pedicels (vdata.x=1). Foliage tint
 * is PALER/BLUER than the other two — the glaucous read.
 */

import { Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { growStem, walkStem, leafBlade, urnBell, pedicel, mergeGeo, perpFrame, type StemSample } from './EricaceousKit';

// ---- recommended integration params ----------------------------------------
export const BOGROSEMARY_HEIGHT: [number, number] = [0.15, 0.3];
export const BOGROSEMARY_FOLIAGE = { r: 0.16, g: 0.26, b: 0.2, hueVar: 0.14 }; // glaucous blue-green
export const BOGROSEMARY_BLOSSOM = { r: 0.85, g: 0.55, b: 0.62, frac: 0.18 }; // soft pink urn-bells
export const BOGROSEMARY_CLS_MAX_DIST = 120;
export const BOGROSEMARY_ROUTING = 'SHRUB' as const;

interface Parts {
  bark: MeshGrower;
  leaf: MeshGrower;
  flower: MeshGrower;
}

/** sparse, ±erect, strongly revolute linear leaves along a thin stem. */
function dressStem(parts: Parts, samples: StemSample[], rng: Rng): void {
  const u = new Vector3();
  const v = new Vector3();
  let node = 0;
  walkStem(samples, 0.018, 0.12, (p, dir, t) => {
    perpFrame(dir, u, v);
    const a = node * 2.399963 + rng.float() * 0.3; // sparse spiral
    node++;
    const outN = new Vector3().copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
    // held ±erect: axis leans strongly along the stem
    const axis = new Vector3().copy(dir).multiplyScalar(0.85).addScaledVector(outN, 0.45).normalize();
    const side = new Vector3().crossVectors(axis, outN).normalize();
    const len = 0.016 + rng.float() * 0.014;
    const hue = (rng.float() - 0.5) * 0.4;
    // linear: near-constant narrow width, strong keel + revolute → almost needle-like
    leafBlade(parts.leaf, p, axis, side, len, len * 0.07, len * 0.09, len * 0.03, 0.34, 0.6, hue, 0, 0.55 + 0.35 * t, 0.95, 3);
    void t;
  });
}

/** nodding terminal cluster of 2–6 pink urn-bells on thin pink pedicels. */
function bellCluster(parts: Parts, tip: Vector3, up: Vector3, rng: Rng, swayPhase: number): void {
  const u = new Vector3();
  const v = new Vector3();
  perpFrame(up, u, v);
  const bells = 2 + rng.int(5);
  const down = new Vector3(0, -1, 0);
  for (let i = 0; i < bells; i++) {
    const a = (i / bells) * Math.PI * 2 + rng.float() * 0.5;
    const reach = 0.006 + rng.float() * 0.006;
    // pedicel arcs up-and-out from the tip, then the bell nods down from its end
    const stalkEnd = new Vector3()
      .copy(tip)
      .addScaledVector(u, Math.cos(a) * reach)
      .addScaledVector(v, Math.sin(a) * reach)
      .addScaledVector(up, 0.004 + rng.float() * 0.004);
    pedicel(parts.flower, tip, stalkEnd, 0.0006, swayPhase);
    const size = 0.006 + rng.float() * 0.0035;
    // hang direction: mostly down, slightly outward (nodding)
    const hang = new Vector3()
      .copy(down).multiplyScalar(0.85)
      .addScaledVector(u, Math.cos(a) * 0.25)
      .addScaledVector(v, Math.sin(a) * 0.25)
      .normalize();
    void rng.float();
    urnBell(parts.flower, stalkEnd, hang, size, 6, swayPhase);
  }
}

export function buildBogRosemaryParts(rng: Rng): Parts {
  const bark = new MeshGrower();
  const leaf = new MeshGrower();
  const flower = new MeshGrower();
  const parts: Parts = { bark, leaf, flower };
  const stems = 2 + rng.int(3);
  const H = BOGROSEMARY_HEIGHT[0] + rng.float() * (BOGROSEMARY_HEIGHT[1] - BOGROSEMARY_HEIGHT[0]);
  for (let i = 0; i < stems; i++) {
    const az = (i / stems) * Math.PI * 2 + rng.float() * 1.1;
    const h = H * (0.78 + rng.float() * 0.3);
    const samples = growStem(
      bark,
      {
        origin: new Vector3(Math.cos(az) * 0.03 * rng.float(), 0, Math.sin(az) * 0.03 * rng.float()),
        azimuth: az,
        lean: 0.28 + rng.float() * 0.3, // decumbent base, ascending tip
        height: h,
        baseR: 0.0018,
        tipR: 0.0006,
        segs: 6,
        wander: 0.16,
        ascend: 0.5,
        recurve: 0.1,
        swayPhase: rng.float() * Math.PI * 2,
        hue: rng.float() * 2 - 1,
      },
      rng.fork(`stem${i}`),
    );
    dressStem(parts, samples, rng.fork(`dress${i}`));
    const tip = samples[samples.length - 1] as StemSample;
    if (rng.chance(0.8)) bellCluster(parts, tip.p.clone(), tip.dir.clone(), rng.fork(`bells${i}`), rng.float() * Math.PI * 2);
  }
  return parts;
}

export function buildBogRosemary(rng: Rng): { bark: BufferGeometry; crown: BufferGeometry; barkTris: number; crownTris: number } {
  const parts = buildBogRosemaryParts(rng);
  const bark = parts.bark.build();
  const crown = mergeGeo([parts.leaf.build(), parts.flower.build()]);
  return { bark, crown, barkTris: parts.bark.triCount, crownTris: parts.leaf.triCount + parts.flower.triCount };
}

export function buildPreview(rng: Rng): Object3D {
  const parts = buildBogRosemaryParts(rng);
  const g = new Group();
  const barkMat = new MeshStandardMaterial({ color: 0x5a3320, roughness: 0.92, metalness: 0 });
  const leafMat = new MeshStandardMaterial({ color: 0x5a7a63, roughness: 0.6, metalness: 0, side: DoubleSide }); // glaucous blue-green
  const flowerMat = new MeshStandardMaterial({ color: 0xe39aa6, roughness: 0.62, metalness: 0, side: DoubleSide, emissive: 0x3a1820, emissiveIntensity: 0.3 });
  g.add(new Mesh(parts.bark.build(), barkMat));
  g.add(new Mesh(parts.leaf.build(), leafMat));
  g.add(new Mesh(parts.flower.build(), flowerMat));
  return g;
}

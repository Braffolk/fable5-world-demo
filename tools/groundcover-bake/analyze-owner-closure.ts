/**
 * Lower-bound feasibility probe for a periodic GCRP/v4 owner closure.
 *
 * This intentionally does not claim certification: it measures only the union
 * of owners already visible at the 16 corners of each canonical
 * phase-x/phase-z/azimuth/elevation cell. A live-only owner can occur strictly
 * inside the cell, and v4 stores no predicate-filtered successor events.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  derivePeriodicProfileLattice,
  parsePeriodicProfile,
} from '../../src/nanite/groundcover/GroundCoverProfiles';

const source = resolve(process.argv[2] ?? 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const profile = parsePeriodicProfile(readFileSync(source));
if (profile.version !== 4 || !profile.ownerTexels) {
  throw new Error('owner-closure analysis requires GCRP/v4 owner texels');
}

const width = profile.interiorTileWidth;
const height = profile.interiorTileHeight;
const lattice = derivePeriodicProfileLattice(profile);
const azimuths = lattice.azimuthCount;
const elevations = lattice.elevationCount;
const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
const owners = profile.ownerTexels;
const MISS = 0xffff_ffff;

const sliceIndex = (azimuth: number, elevation: number): number =>
  lattice.order === 'azimuth-major'
    ? azimuth * elevations + elevation
    : elevation * azimuths + azimuth;

const ownerAt = (
  phaseX: number,
  phaseZ: number,
  azimuth: number,
  elevation: number,
): number => {
  const slice = sliceIndex(azimuth % azimuths, elevation);
  const tileX = slice % profile.atlasColumns;
  const tileY = Math.floor(slice / profile.atlasColumns);
  const x = tileX * profile.storedTileWidth + profile.gutter + (phaseX % width);
  const y = tileY * profile.storedTileHeight + profile.gutter + (phaseZ % height);
  return owners[y * atlasWidth + x]!;
};

const histogram = new Uint32Array(17);
const unique = new Uint32Array(16);
let cellCount = 0;
let ownerCornerReferences = 0;
let missOnlyCells = 0;

for (let elevation = 0; elevation < elevations - 1; elevation++) {
  for (let azimuth = 0; azimuth < azimuths; azimuth++) {
    for (let phaseZ = 0; phaseZ < height; phaseZ++) {
      for (let phaseX = 0; phaseX < width; phaseX++) {
        let count = 0;
        for (let de = 0; de < 2; de++) {
          for (let da = 0; da < 2; da++) {
            for (let dz = 0; dz < 2; dz++) {
              for (let dx = 0; dx < 2; dx++) {
                const owner = ownerAt(
                  phaseX + dx,
                  phaseZ + dz,
                  azimuth + da,
                  elevation + de,
                );
                if (owner === MISS) continue;
                ownerCornerReferences++;
                let seen = false;
                for (let i = 0; i < count; i++) {
                  if (unique[i] === owner) {
                    seen = true;
                    break;
                  }
                }
                if (!seen) unique[count++] = owner;
              }
            }
          }
        }
        histogram[count]++;
        if (count === 0) missOnlyCells++;
        cellCount++;
      }
    }
  }
}

const quantile = (fraction: number): number => {
  const target = Math.ceil(cellCount * fraction);
  let cumulative = 0;
  for (let count = 0; count < histogram.length; count++) {
    cumulative += histogram[count]!;
    if (cumulative >= target) return count;
  }
  return histogram.length - 1;
};

const cellsAbove = (bound: number): number => {
  let total = 0;
  for (let count = bound + 1; count < histogram.length; count++) total += histogram[count]!;
  return total;
};

let maximum = 0;
for (let count = histogram.length - 1; count >= 0; count--) {
  if (histogram[count] !== 0) {
    maximum = count;
    break;
  }
}

console.log(JSON.stringify({
  source,
  domain: {
    phase: [width, height],
    azimuths,
    elevations: lattice.elevations.map((value) => value * 180 / Math.PI),
    canonical4dCells: cellCount,
  },
  cornerOwnerUnionLowerBound: {
    histogram: Array.from(histogram),
    p50: quantile(0.5),
    p95: quantile(0.95),
    p99: quantile(0.99),
    maximum,
    cellsAbove4: cellsAbove(4),
    cellsAbove8: cellsAbove(8),
    cellsAbove12: cellsAbove(12),
    missOnlyCells,
    ownerCornerReferences,
  },
  certificationLimits: [
    'corner union is only a lower bound; owners may win strictly inside a 4D cell',
    'the current elevation domain begins at 15 degrees and cannot measure the approved 5-degree row',
    'v4 stores only the fully populated first owner, not ordered or predicate-filtered successors',
    'copy/root ownership and finite-horizon eligibility require the successor bake before K can be certified',
  ],
}, null, 2));

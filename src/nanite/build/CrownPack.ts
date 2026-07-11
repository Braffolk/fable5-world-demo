/**
 * CrownPack — the CrownVoxelization compact pack codec, THREE-FREE (imports only
 * VoxelBrickCore + VoxelizeCrown types) so it runs in module Workers: the DagWorker
 * pool packs crown voxelizations today, and the StreamBrain worker packs runtime
 * fartile bakes (S8). Moved VERBATIM out of BootCache.ts (whose ?raw source imports
 * made it a poor worker import) — BootCache stores these packed values opaquely.
 *
 * BrickCPU → BRICK_WORDS (9) × u32 GPU words per occupied brick (the SAME record the
 * append path copies straight into gpu.voxelBricks — writeBrick codec). Only OCCUPIED
 * bricks are stored; unpack leaves empty grid slots as holes (append paths never read
 * them). 36 B/brick vs the old 108 B (2×u32 + 12×f64) — cuts the fartile pack from
 * ~866 MB to ~288 MB and lets appendPackedCrown skip per-brick BrickCPU objects.
 * "Lossless enough": the words ARE what the GPU renders (byte-identical to a fresh
 * build's writeBrick output); only build-time intermediates (fartile splat INPUT)
 * see oct-normal / rgba8-albedo quantization — a cold-only, >280 m far-field effect.
 */

import type { CrownVoxelization, PreparedVoxelCrown, VoxelBlock, VoxelLevel } from './VoxelizeCrown';
import { BRICK_WORDS, readBrick, writeBrick } from '../voxel/VoxelBrickCore';

export interface PackedGrid {
  occupied: Uint32Array;
  words: Uint32Array; // BRICK_WORDS per brick — gpu.voxelBricks record, occupied[] order
  totalBricks: number;
}

export interface PackedLevel extends PackedGrid {
  level: number;
  brickGrid: { x: number; y: number; z: number };
  cellSize: number;
  geomError: number;
  blocks: VoxelBlock[]; // plain data (Sphere = plain xyzr) — structured clone handles it
}

interface PackedVox {
  grid: PackedGrid;
  brickGrid: CrownVoxelization['brickGrid'];
  cellGrid: CrownVoxelization['cellGrid'];
  origin: CrownVoxelization['origin'];
  cellSize: number;
  stats: CrownVoxelization['stats'];
  levels?: PackedLevel[];
}

export interface PackedPreparedCrown {
  vox: PackedVox;
  brickCount: number;
  clusterCount: number;
  dagLinkCount: number;
}

type BrickList = CrownVoxelization['bricks'];

function packGrid(bricks: BrickList, occupied: number[]): PackedGrid {
  const n = occupied.length;
  const occ = new Uint32Array(occupied);
  const words = new Uint32Array(n * BRICK_WORDS);
  for (let i = 0; i < n; i++) {
    const b = bricks[occ[i] as number];
    if (!b) throw new Error('crownpack: occupied index out of range');
    writeBrick(words, i, b);
  }
  return { occupied: occ, words, totalBricks: bricks.length };
}

/** reconstruct a SPARSE BrickCPU grid from the packed words (holes for empty slots).
 *  Compat path — the world append reads `words` directly (appendPackedCrown); this is
 *  for ForestScene's BrickCPU appendVoxelCrown + probes. Values are word-precision
 *  (oct-normal / rgba8-albedo) = exactly what a fresh build's writeBrick emits. */
function unpackGrid(p: PackedGrid): { bricks: BrickList; occupied: number[] } {
  const bricks: BrickList = new Array(p.totalBricks);
  const occupied: number[] = new Array(p.occupied.length);
  for (let i = 0; i < p.occupied.length; i++) {
    const bi = p.occupied[i] as number;
    occupied[i] = bi;
    bricks[bi] = readBrick(p.words, i);
  }
  return { bricks, occupied };
}

function packVox(vox: CrownVoxelization): PackedVox {
  return {
    grid: packGrid(vox.bricks, vox.occupied),
    brickGrid: vox.brickGrid,
    cellGrid: vox.cellGrid,
    origin: vox.origin,
    cellSize: vox.cellSize,
    stats: vox.stats,
    levels: vox.levels?.map((l) => ({
      ...packGrid(l.bricks, l.occupied),
      level: l.level,
      brickGrid: l.brickGrid,
      cellSize: l.cellSize,
      geomError: l.geomError,
      blocks: l.blocks,
    })),
  };
}

function unpackVox(p: PackedVox): CrownVoxelization {
  const g = unpackGrid(p.grid);
  const levels: VoxelLevel[] | undefined = p.levels?.map((l) => {
    const lg = unpackGrid(l);
    return {
      level: l.level,
      bricks: lg.bricks,
      occupied: lg.occupied,
      brickGrid: l.brickGrid,
      cellSize: l.cellSize,
      geomError: l.geomError,
      blocks: l.blocks,
    };
  });
  const out: CrownVoxelization = {
    bricks: g.bricks,
    occupied: g.occupied,
    brickGrid: p.brickGrid,
    cellGrid: p.cellGrid,
    origin: p.origin,
    cellSize: p.cellSize,
    stats: p.stats,
  };
  if (levels) out.levels = levels;
  return out;
}

export function packPreparedCrown(prep: PreparedVoxelCrown): PackedPreparedCrown {
  return {
    vox: packVox(prep.vox),
    brickCount: prep.brickCount,
    clusterCount: prep.clusterCount,
    dagLinkCount: prep.dagLinkCount,
  };
}

export function unpackPreparedCrown(p: PackedPreparedCrown): PreparedVoxelCrown {
  return {
    vox: unpackVox(p.vox),
    brickCount: p.brickCount,
    clusterCount: p.clusterCount,
    dagLinkCount: p.dagLinkCount,
  };
}

import { ClampToEdgeWrapping, Data3DTexture, LinearFilter, RepeatWrapping, RGBAFormat } from 'three';
import { bakeGrassRayTile, packGroundCoverRayAtlas } from '../build/GrassRayBake';
import { GroundCoverId } from '../groundcover/GroundCoverTypes';
import {
  GRASS_BAKE_ANGLES,
  GRASS_BAKE_RESOLUTION,
  GRASS_BAKE_SHIFT,
  GRASS_BAKE_THICKNESS,
  GRASS_BLADE_TABLE,
  GRASS_CELL_METRES,
  GRASS_DENSITY_TIERS,
  GRASS_RAY_ATLAS_MAX_BYTES,
  GRASS_TIER_KEEP_SALT,
  GUIDE_SUB,
  grassNumberParam,
} from './LegacyPeriodicGroundCoverPolicy';

export interface LegacyPeriodicRayAtlas {
  readonly textures: readonly Data3DTexture[];
  readonly maximumTileDistance: number;
  readonly angleStride: number;
  readonly depth: number;
}

/** @deprecated Boot-time integration fixture for the legacy functional fallback. */
export function createLegacyPeriodicRayAtlas(): LegacyPeriodicRayAtlas {
  const profiles = [
    {
      id: GroundCoverId.Grass,
      label: 'grass',
      section: 'rectangle' as const,
      halfW: grassNumberParam('grassbakw', 0.0055, 0.001, 0.05),
      halfT: grassNumberParam('grassbakt', 0.003, 0.001, 0.02),
      fibers: Math.round(grassNumberParam('grassbakn', 6, 2, 16)),
      spread: 1.3,
      arcK: grassNumberParam('grassarck', 0, 0, 3),
    },
    { id: GroundCoverId.Moss, label: 'moss', section: 'ellipse' as const, halfW: 0.052, halfT: 0.044, fibers: 4, spread: 1, arcK: 0 },
    { id: GroundCoverId.Sedge, label: 'sedge', section: 'rectangle' as const, halfW: 0.0038, halfT: 0.0022, fibers: 8, spread: 1.25, arcK: 0 },
    { id: GroundCoverId.Lichen, label: 'lichen', section: 'ellipse' as const, halfW: 0.041, halfT: 0.026, fibers: 3, spread: 0.9, arcK: 0 },
    { id: GroundCoverId.Forb, label: 'forb', section: 'ellipse' as const, halfW: 0.039, halfT: 0.009, fibers: 5, spread: 1.05, arcK: 0 },
    { id: GroundCoverId.DwarfShrub, label: 'dwarf-shrub', section: 'ellipse' as const, halfW: 0.024, halfT: 0.018, fibers: 5, spread: 1.1, arcK: 0 },
  ];
  profiles.forEach((profile, profileId) => {
    if (profile.id !== profileId) throw new Error('legacy ground-cover atlas ids must be contiguous');
  });
  const atlas = packGroundCoverRayAtlas(profiles.map((profile) => bakeGrassRayTile({
    res: GRASS_BAKE_RESOLUTION,
    angles: GRASS_BAKE_ANGLES,
    blades: GRASS_BLADE_TABLE,
    sub: GUIDE_SUB,
    cellM: GRASS_CELL_METRES,
    shiftK: profile.id === GroundCoverId.Grass ? GRASS_BAKE_SHIFT : 0,
    thickK: profile.id === GroundCoverId.Grass ? GRASS_BAKE_THICKNESS : 0,
    halfW: profile.halfW,
    halfT: profile.halfT,
    fibers: profile.fibers,
    section: profile.section,
    shape: 'extruded',
    spread: profile.spread,
    tiers: [...GRASS_DENSITY_TIERS],
    keepSalt: GRASS_TIER_KEEP_SALT,
    arcK: profile.arcK,
    label: profile.label,
  })));
  const atlasBytes = atlas.data.reduce((sum, volume) => sum + volume.byteLength, 0);
  if (atlasBytes > GRASS_RAY_ATLAS_MAX_BYTES) {
    throw new Error(`legacy ground-cover atlas ${Math.ceil(atlasBytes / 1048576)} MiB exceeds budget`);
  }
  return Object.freeze({
    textures: Object.freeze(atlas.data.map((data, index) => {
      const texture = new Data3DTexture(data, atlas.res, atlas.res, atlas.depth);
      texture.format = RGBAFormat;
      texture.minFilter = LinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = RepeatWrapping;
      texture.wrapT = RepeatWrapping;
      texture.wrapR = ClampToEdgeWrapping;
      texture.generateMipmaps = false;
      texture.needsUpdate = true;
      texture.name = `legacyGroundCoverRayAtlas${index}`;
      return texture;
    })),
    maximumTileDistance: atlas.dMaxTile,
    angleStride: atlas.angleStride,
    depth: atlas.depth,
  });
}

import type { PerspectiveCamera } from 'three';
import type { StorageTexture, Renderer } from 'three/webgpu';
import type { NU, NV4 } from '../../gpu/TSLTypes';
import type { NaniteCam } from '../NaniteCommon';
import type { LoadedPeriodicProfile } from '../groundcover/GroundCoverProfiles';
import type { NaniteVisBuffers } from '../raster/NaniteRaster';
import type { TerrainField } from '../world/TerrainField';

/** @deprecated Compatibility renderer only. New ground cover uses the separate
 * exterior boundary-transfer builder and must never enter this option bag. */
export interface LegacyPeriodicGroundCoverBuildOpts {
  cam: NaniteCam;
  vis: NaniteVisBuffers;
  field: TerrainField;
  canopyTex: StorageTexture | null;
  periodicProfiles: readonly LoadedPeriodicProfile[];
  /** Build-time diagnostic isolation. This does not select another renderer;
   * it restricts the canonical GCAR union to one authored profile. */
  forcedProfileId?: number | null;
}

/** @deprecated Remove with the legacy periodic renderer after boundary transfer
 * passes visual and performance acceptance. */
export interface LegacyPeriodicGroundCoverField {
  batch: readonly unknown[];
  renderHw(renderer: Renderer, camera: PerspectiveCamera): void;
  resolveRay(px: NU): NV4;
  readonly authoredPackedColor: boolean;
  setEnabled(value: boolean): void;
  enabled(): boolean;
  readCounts(renderer: Renderer): Promise<{ clumps: number; hwTris: number }>;
}

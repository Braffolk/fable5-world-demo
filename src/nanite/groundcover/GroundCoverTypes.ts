/** Stable 6-bit ids carried by procedural ground-cover visibility records.
 *
 * These are functional cover classes, not species claims. Native species get
 * their own authored atlas entries after reference verification. GRASS=0 is a
 * migration invariant: legacy procedural grass bodies already have zero low bits.
 */
export const GroundCoverId = {
  Grass: 0,
  Moss: 1,
  Sedge: 2,
  Lichen: 3,
  Forb: 4,
  DwarfShrub: 5,
  Bare: 63,
} as const;

export type GroundCoverIdValue = (typeof GroundCoverId)[keyof typeof GroundCoverId];

export const GROUND_COVER_ID_MASK = 0x3f;

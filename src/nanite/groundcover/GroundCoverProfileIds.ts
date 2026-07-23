import { GroundCoverId } from './GroundCoverTypes';

/** Exact native-profile ids. These are deliberately independent from the
 * six functional GroundCoverId classes used for wind and broad materials. */
export const GroundCoverProfileId = {
  AgrostisCapillaris: 0,
  AvenellaFlexuosa: 1,
  CalamagrostisCanescens: 2,
  CarexCespitosa: 3,
  EriophorumVaginatum: 4,
  SphagnumCapillifolium: 5,
  PleuroziumSchreberi: 6,
  CladoniaRangiferina: 7,
  OxalisAcetosella: 8,
  MaianthemumBifolium: 9,
  VacciniumMyrtillus: 10,
  CallunaVulgaris: 11,
} as const;

/** Canonical GCAR layer order. The array loader verifies this exact closure. */
export const GROUND_COVER_PROFILE_IDS = [
  GroundCoverProfileId.AgrostisCapillaris,
  GroundCoverProfileId.AvenellaFlexuosa,
  GroundCoverProfileId.CalamagrostisCanescens,
  GroundCoverProfileId.CarexCespitosa,
  GroundCoverProfileId.EriophorumVaginatum,
  GroundCoverProfileId.SphagnumCapillifolium,
  GroundCoverProfileId.PleuroziumSchreberi,
  GroundCoverProfileId.CladoniaRangiferina,
  GroundCoverProfileId.OxalisAcetosella,
  GroundCoverProfileId.MaianthemumBifolium,
  GroundCoverProfileId.VacciniumMyrtillus,
  GroundCoverProfileId.CallunaVulgaris,
] as const;

export const GROUND_COVER_PROFILE_COUNT = GROUND_COVER_PROFILE_IDS.length;

/** Exact profile -> functional response. Array order is the stable profile id. */
export const GROUND_COVER_PROFILE_FUNCTIONAL_IDS = [
  GroundCoverId.Grass,
  GroundCoverId.Grass,
  GroundCoverId.Grass,
  GroundCoverId.Sedge,
  GroundCoverId.Sedge,
  GroundCoverId.Moss,
  GroundCoverId.Moss,
  GroundCoverId.Lichen,
  GroundCoverId.Forb,
  GroundCoverId.Forb,
  GroundCoverId.DwarfShrub,
  GroundCoverId.DwarfShrub,
] as const;

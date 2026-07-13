import { QRASTER_CAP } from '../NaniteCommon';

/** Shared finite domain for the world1 ctx/projection/classifier pre-pass chain. */
export const RASTER_PREPASS_CLUSTER_CAP = Math.min(QRASTER_CAP, 96 * 1024);

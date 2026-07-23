import {
  CALAMAGROSTIS_CARRIER_CLOSURE_URL,
  loadGroundCoverCarrierClosure,
  type LoadedGroundCoverCarrierClosure,
} from '../groundcover/GroundCoverCarrierClosure';
import {
  CALAMAGROSTIS_GBR4_V4_REFERENCE_URL,
  loadGbr4V4Profile,
  type LoadedGbr4V4Profile,
} from '../groundcover/GroundCoverGbr4V4';
import {
  GROUND_COVER_PROFILE_ARRAY_URL,
  GROUND_COVER_PROFILE_IDS,
  CALAMAGROSTIS_ACCEPTANCE_PROFILE_URL,
  GroundCoverProfileId,
  loadPeriodicProfile,
  loadPeriodicProfileArray,
  type LoadedPeriodicProfile,
} from '../groundcover/GroundCoverProfiles';

/**
 * The frame may select exactly one ground-cover renderer.  Keeping this as a
 * discriminated union prevents the periodic GCAR renderer and the boundary
 * transfer renderer from sharing one option bag or silently falling through to
 * one another.
 */
export type GroundCoverFrameBinding =
  | { readonly kind: 'off' }
  | {
      readonly kind: 'periodic-multispecies';
      readonly profiles: readonly LoadedPeriodicProfile[];
    }
  | {
      /** Single-species visual recovery lane.  This is intentionally distinct
       * from the deprecated multi-species composition and may be deleted when
       * the GREEN boundary codec replaces it. */
      readonly kind: 'calamagrostis-preview';
      readonly profiles: readonly LoadedPeriodicProfile[];
      readonly profileId: typeof GroundCoverProfileId.CalamagrostisCanescens;
    }
  | {
      readonly kind: 'boundary-blocked';
      readonly transfer: LoadedGbr4V4Profile;
      readonly carrier: LoadedGroundCoverCarrierClosure;
    };

function validateBoundaryPair(
  transfer: LoadedGbr4V4Profile,
  carrier: LoadedGroundCoverCarrierClosure,
): void {
  if (
    carrier.profileId !== transfer.profileId
    || carrier.sourceSha256 !== transfer.sourceSha256
    || Math.abs(carrier.horizonMetres - transfer.horizonMetres) > 1e-5
  ) {
    throw new Error(
      'ground-cover boundary transfer and carrier closure do not share one source/horizon',
    );
  }
}

export async function loadGroundCoverFrameBinding(
  enabled: boolean,
  hasGroundCoverClosure: boolean,
  params: URLSearchParams,
): Promise<GroundCoverFrameBinding> {
  if (!enabled || !hasGroundCoverClosure) return { kind: 'off' };

  const isolatedCalamagrostis = params.get('grassprofile') === String(
    GroundCoverProfileId.CalamagrostisCanescens,
  );
  if (!isolatedCalamagrostis) {
    return {
      kind: 'periodic-multispecies',
      profiles: (await loadPeriodicProfileArray(
        GROUND_COVER_PROFILE_ARRAY_URL,
        GROUND_COVER_PROFILE_IDS,
      )).profiles,
    };
  }

  const [transfer, carrier, profile] = await Promise.all([
    loadGbr4V4Profile(
      CALAMAGROSTIS_GBR4_V4_REFERENCE_URL,
      GroundCoverProfileId.CalamagrostisCanescens,
    ),
    loadGroundCoverCarrierClosure(CALAMAGROSTIS_CARRIER_CLOSURE_URL),
    loadPeriodicProfile(
      CALAMAGROSTIS_ACCEPTANCE_PROFILE_URL,
      GroundCoverProfileId.CalamagrostisCanescens,
      { authoredColor: true },
    ),
  ]);
  validateBoundaryPair(transfer, carrier);

  if (transfer.runtimeBindAllowed) {
    // A GREEN asset must enter through the standalone boundary renderer and a
    // concrete fixed-chart adapter.  It must never be handed to the periodic
    // renderer merely because its publication bit changed.
    throw new Error('ground-cover GREEN boundary asset requires its fixed-chart runtime adapter');
  }

  // eslint-disable-next-line no-console
  console.warn(
    `[ground-cover] isolated Calamagrostis boundary codec is ${transfer.publicationStatus}; using the isolated periodic visual recovery lane`,
  );
  return {
    kind: 'calamagrostis-preview',
    profiles: [profile],
    profileId: GroundCoverProfileId.CalamagrostisCanescens,
  };
}

"""Raised-bog v6: hydrology-conditioned two-field plurigaussian microtopography.

The spec-authorized route (docs/specs/terrain/MICROTOPOGRAPHY.md 9.10.1 v2). Supersedes
the terminated marked-form route (v1/v5) and the reaction-diffusion Turing route (v3/v4).

Two stationary Gaussian random fields (a fine isotropic microform field 0.5-3 m and a
coarse anisotropic ridge-hollow patterning field 10-100 m, elongated perpendicular to
mire flow) are combined into one latent. An ordered threshold on that latent guarantees
the Moore transiogram ordering (hollow -> lawn -> hummock, hollow never abutting hummock).
Height is a SMOOTH continuous Gaussian-anamorphosis of the latent calibrated to Moore's
pooled per-class relative-height quantiles and the reconciled 0.20-0.40 m amplitude, NOT a
per-class constant step. Real ETAK pools stay relief-free and hollows grade into them.

All stochastic fields draw from world-PRF (BLAKE2b laas-micro-prf1, keyed on integer world
lattice coordinates) so the field is identical regardless of chunking, worker count, batch
order, or crop window. The whole core + halo is solved once; the storage core is cropped
last. Deterministic float64.

Produces the relief float in the interface the existing packer/verifier consume
(network_preview.py / network_preview_verify.py: key ``core_relief_00625m``, 2048x2048,
0.0625 m). The surface is synthesized NATIVELY at the LOD -2 finest pitch (0.0625 m) so the
packer places it 1:1 into the fine core -- no 4x nearest-neighbour upsample, which had stamped
flat 0.25 m terraces into the 6 cm rung.
"""

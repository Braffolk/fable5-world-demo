# Ground-cover format and loader status

The files at this level support the live GCAR/GCRP compatibility profiles or validation of
historical carrier/GBR4-v4 records. `GroundCoverGbr4V4.ts` and `GroundCoverCarrierClosure.ts` are
validation loaders only; they do not bind a boundary-transfer renderer.

Disconnected boundary-transfer, GBC2, and non-v4 GBR4 runtime prototypes are preserved under
`rejected-inactive/` and excluded from production TypeScript checks.

# Grass runtime status

There are two live compatibility lanes, neither accepted as the final generic ground-cover system:

- `NaniteGrass.ts` plus `LegacyPeriodic*` — deprecated but still-default multi-species periodic
  renderer.
- `CalamagrostisPrecomputedRay*` — isolated `grassprofile=2` Calamagrostis preview.

`PeriodicGroundCoverGuideField.ts`, `GroundCoverRayPolicy.ts`, and
`GroundCoverResolveShade.ts` are shared live support. Zero-import historical shims are under
`rejected-inactive/`; nothing there is runtime reachable.
